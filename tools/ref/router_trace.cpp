// router_trace — which experts ik_llama.cpp's MoE router picks, per layer, over a token stream.
//
// The placement design keeps a fixed number of each layer's experts in VRAM. If routing is
// skewed, putting each layer's hottest experts there serves more than n/n_expert of the expert
// reads from the card; this tool measures the skew. It runs the ids through llama_decode as a
// prefill: the router's choice at a position is a function of the tokens up to it, so a prefill
// makes a decode's choices at prefill speed, apart from near-tie flips where batched and
// single-token kernels round differently (the schedule paragraph below says how far one carries).
//
// What it captures: every node named `ffn_moe_topk-<layer>`, the tensor llm_build_moe_ffn names
// after the expert selection. On a routed layer it is a VIEW [n_expert_used, n_tokens] of the
// ARGSORT of the biased router probabilities, whose rows are n_expert long, so it is read through
// its own nb strides — the flat memory at its data pointer is the argsort's head, not each
// token's top-k. On a hash-routed layer the ids come from GET_ROWS of `ffn_gate_tid2eid` under the
// same name. A same-named node of any other op is not a selection; it is counted in the manifest
// and not captured. Every capture must be I32, n_expert_used x the decode call's token count, ids
// in [0, n_expert) and distinct within a token, and every MoE layer must be captured exactly once
// per decode call — anything else stops the run, because a count built from a misread tensor
// looks like a real skew.
//
// The schedule: by default the eval callback asks for every node, so the scheduler computes one
// node at a time. That is the dumper's schedule, the one the oracle set was produced under: no
// lookahead fusion of ik's CPU backend (SUM_ROWS+DIV, CONT+SUM_ROWS+TRANSPOSE, ...) can fire when
// the graph view holds one node, and a trace under it reproduces the oracle's ids exactly.
// `--every-node` names that default. `--top-k-only` asks for the top-k nodes alone: the graph runs in segments that each end at one,
// and the fusions inside a segment fire as they do in serving. The fused path rounds differently,
// and through the top-k a last-bit difference becomes a different expert, so per-token choices
// drift apart with depth while the hot-expert counts do not move; its ids cannot be checked
// against the oracle set.
//
// Tokens: `--ids <file>`, one decimal token id per line (the format engram-corpus.sh writes), the
// first `--max-tokens` of them. They go through the model in independent contexts of `--chunk`
// tokens (the KV cache and V4's compressed-attention state are cleared between chunks and
// positions restart at 0), each chunk in decode calls of at most n_ubatch tokens, so the stream
// is not limited by n_ctx.
//
// Output, into `--out` — a directory under $BLOOMERY_DATA/router/ — staged as `<out>.staging` and
// renamed into place only after the manifest's trailer is written; a run that dies leaves the
// staging directory without a trailer and the previous set untouched:
//   topk-<layer>.u16   tokens x n_expert_used expert ids, little-endian u16, token-major, in the
//                      order the router ranked them
//   counts.tsv         `layer expert count` for every (layer, expert) pair, zeros included
//   MANIFEST.tsv       provenance header, one `layer` row per traced layer, one `call` row per
//                      decode call (its wall ms and the page faults it took), `# complete` last
//
// This binary's side effect is a data set, so it refuses to run without BLOOMERY_ROUTER_WRITE=1
// (tools/ref/router-trace.sh sets it, with the lease and the witnesses around the run), refuses an
// output directory outside $BLOOMERY_DATA/router/, and never replaces a directory that does not
// hold a router set.
//
// Build: tools/ref/build-router-trace.sh   Run: tools/ref/router-trace.sh

#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace fs = std::filesystem;

static const char * const MANIFEST_TITLE = "# router_trace";

// How a captured `ffn_moe_topk-<layer>` node was produced.
enum topk_source { SRC_NONE, SRC_SORT_VIEW, SRC_GROUPED, SRC_HASH };

static const char * source_name(topk_source s) {
    switch (s) {
        case SRC_SORT_VIEW: return "VIEW-of-ARGSORT";
        case SRC_GROUPED:   return "GROUPED_TOPK";
        case SRC_HASH:      return "GET_ROWS";
        default:            return "none";
    }
}

static topk_source classify(const ggml_tensor * t) {
    if (t->type != GGML_TYPE_I32) return SRC_NONE;
    if (t->op == GGML_OP_VIEW && t->view_src &&
        (t->view_src->op == GGML_OP_ARGSORT || t->view_src->op == GGML_OP_ARGSORT_THRESH)) {
        return SRC_SORT_VIEW;
    }
    if (t->op == GGML_OP_GROUPED_TOPK) return SRC_GROUPED;
    if (t->op == GGML_OP_GET_ROWS) return SRC_HASH;
    return SRC_NONE;
}

// `ffn_moe_topk-<layer>` exactly; -1 for any other name.
static int topk_layer(const char * name) {
    static const char prefix[] = "ffn_moe_topk-";
    const size_t n = sizeof(prefix) - 1;
    if (strncmp(name, prefix, n) != 0 || name[n] == '\0') return -1;
    int layer = 0;
    for (const char * p = name + n; *p; ++p) {
        if (*p < '0' || *p > '9' || layer > 100000) return -1;
        layer = layer * 10 + (*p - '0');
    }
    return layer;
}

struct layer_state {
    topk_source           source = SRC_NONE;
    std::string           src_name;          // the producing tensor, for the manifest
    bool                  got = false;       // captured in the current decode call
    int                   ignored = 0;       // same-named nodes that were not a selection
    std::vector<uint16_t> rows;              // the current decode call's ids, token-major
    std::vector<uint64_t> counts;            // per expert, over the whole run
    uint64_t              tokens = 0;
    uint64_t              id_sum = 0;
    FILE *                file = nullptr;
};

struct trace_ctx {
    int                      n_layer = 0;
    int                      n_expert = 0;
    int                      n_used = 0;
    int64_t                  call_tokens = 0;   // tokens in the decode call now running
    bool                     every_node = true;  // the oracle's schedule; --top-k-only clears it
    bool                     failed = false;
    std::string              error;
    std::vector<layer_state> layers;
    std::vector<uint8_t>     staging;           // device tensors land here before the read
};

static void fail(trace_ctx * tr, const std::string & msg) {
    if (!tr->failed) {
        tr->failed = true;
        tr->error  = msg;
    }
}

static void capture(trace_ctx * tr, const ggml_tensor * t, int layer) {
    const topk_source source = classify(t);
    if (layer >= tr->n_layer) {
        fail(tr, std::string(t->name) + ": layer index past n_layer");
        return;
    }
    layer_state & L = tr->layers[layer];
    if (source == SRC_NONE) {
        L.ignored++;
        return;
    }
    if (L.got) {
        fail(tr, std::string(t->name) + ": a second selection node of this name in one graph");
        return;
    }
    if (t->ne[0] != tr->n_used || t->ne[1] != tr->call_tokens || t->ne[2] != 1 || t->ne[3] != 1) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s: shape [%lld, %lld, %lld, %lld], expected [%d, %lld, 1, 1]", t->name,
                 (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                 tr->n_used, (long long) tr->call_tokens);
        fail(tr, buf);
        return;
    }
    if (L.source != SRC_NONE && L.source != source) {
        fail(tr, std::string(t->name) + ": produced by " + source_name(source) + " here and by " +
                 source_name(L.source) + " in an earlier call");
        return;
    }
    const ggml_tensor * producer = source == SRC_SORT_VIEW ? t->view_src : t->src[0];
    if (L.source == SRC_NONE) {
        L.source   = source;
        L.src_name = producer && producer->name[0] ? producer->name : "-";
    }
    if (!t->buffer || !t->data) {
        fail(tr, std::string(t->name) + ": not allocated");
        return;
    }

    // ggml_nbytes covers the furthest strided element, so the walk below stays inside it.
    const size_t    nbytes = ggml_nbytes(t);
    const uint8_t * src;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        src = (const uint8_t *) t->data;
    } else {
        tr->staging.resize(nbytes);
        ggml_backend_tensor_get(t, tr->staging.data(), 0, nbytes);
        src = tr->staging.data();
    }

    L.rows.resize((size_t) t->ne[0] * t->ne[1]);
    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
        uint16_t * row = L.rows.data() + (size_t) i1 * t->ne[0];
        for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
            int32_t id;
            memcpy(&id, src + (size_t) i0 * t->nb[0] + (size_t) i1 * t->nb[1], sizeof id);
            if (id < 0 || id >= tr->n_expert) {
                fail(tr, std::string(t->name) + ": expert id " + std::to_string(id) + " out of range");
                return;
            }
            for (int64_t j = 0; j < i0; ++j) {
                if (row[j] == id) {
                    fail(tr, std::string(t->name) + ": expert " + std::to_string(id) +
                             " twice in one token's selection");
                    return;
                }
            }
            row[i0] = (uint16_t) id;
        }
    }
    L.got = true;
}

static int on_tensor(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * tr = (trace_ctx *) user_data;
    const int layer = topk_layer(t->name);
    if (ask) {
        // Asked for a node after a failure: yes, so its callback stops the graph.
        return tr->failed || tr->every_node || layer >= 0 ? 1 : 0;
    }
    if (tr->failed) return 0;
    if (layer >= 0) capture(tr, t, layer);
    return tr->failed ? 0 : 1;  // stop the graph once anything is wrong
}

static bool read_ids(const std::string & path, long long max_tokens, std::vector<llama_token> & ids,
                     long long & lines) {
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "router_trace: cannot read %s\n", path.c_str());
        return false;
    }
    std::string line;
    lines = 0;
    while (std::getline(in, line)) {
        ++lines;
        if (max_tokens >= 0 && (long long) ids.size() >= max_tokens) continue;  // count the rest
        char * end = nullptr;
        errno = 0;
        const long v = strtol(line.c_str(), &end, 10);
        if (line.empty() || *end != '\0' || errno != 0 || v < 0 || v > INT32_MAX) {
            fprintf(stderr, "router_trace: %s:%lld is not a token id: '%s'\n", path.c_str(), lines, line.c_str());
            return false;
        }
        ids.push_back((llama_token) v);
    }
    return true;
}

static bool is_router_set(const fs::path & dir) {
    std::ifstream in(dir / "MANIFEST.tsv");
    std::string first;
    return in && std::getline(in, first) && first.rfind(MANIFEST_TITLE, 0) == 0;
}

static long meta_int(const llama_model * model, const std::string & key) {
    char buf[64];
    if (llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf)) < 0) return -1;
    return strtol(buf, nullptr, 10);
}

struct faults {
    long maj = 0;
    long min = 0;
};

static faults faults_now() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return { ru.ru_majflt, ru.ru_minflt };
}

static double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Our own flags; everything else on the command line is gpt_params.
struct options {
    std::string ids_path, out_arg, expect_arch;
    long long   max_tokens = -1;  // -1: the whole file
    long long   chunk      = 0;
    bool        every_node = true;
    std::string flags;            // the whole command line, for the manifest
};

static bool parse_options(int argc, char ** argv, options & o, std::vector<char *> & passthrough) {
    for (int i = 1; i < argc; ++i) o.flags += (i > 1 ? " " : "") + std::string(argv[i]);
    passthrough.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const bool has_value = i + 1 < argc;
        if (strcmp(argv[i], "--ids") == 0 && has_value) {
            o.ids_path = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && has_value) {
            o.out_arg = argv[++i];
        } else if (strcmp(argv[i], "--expect-arch") == 0 && has_value) {
            o.expect_arch = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && has_value) {
            o.max_tokens = std::max(0LL, atoll(argv[++i]));  // 0 is refused below
        } else if (strcmp(argv[i], "--chunk") == 0 && has_value) {
            o.chunk = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--top-k-only") == 0) {
            o.every_node = false;
        } else if (strcmp(argv[i], "--every-node") == 0) {
            o.every_node = true;  // the default, named so a set's recorded flags line runs again
        } else {
            passthrough.push_back(argv[i]);
        }
    }
    if (o.ids_path.empty() || o.out_arg.empty() || o.expect_arch.empty() || o.chunk <= 0 || o.max_tokens == 0) {
        fprintf(stderr, "usage: router_trace -m <gguf> --expect-arch <arch> --ids <file> --out <dir> --chunk C\n"
                        "                    [--max-tokens N] [--top-k-only | --every-node] [gpt_params...]\n");
        return false;
    }
    return true;
}

struct set_paths {
    fs::path out, stage, old_set;
};

// Where the set may go: a directory directly or deeper under $BLOOMERY_DATA/router, never the root
// itself, and never over a directory that is not a router set (router/bin is one such).
static bool resolve_set_paths(const std::string & out_arg, set_paths & sp) {
    const char * data_env = getenv("BLOOMERY_DATA");
    const fs::path data_dir = data_env && *data_env ? data_env : "/root/bloomery-data";
    std::error_code ec;
    const fs::path root = fs::canonical(data_dir / "router", ec);
    if (ec) {
        fprintf(stderr, "router_trace: %s/router does not exist (the runner creates it)\n", data_dir.c_str());
        return false;
    }
    std::string out_str = out_arg;
    while (out_str.size() > 1 && out_str.back() == '/') out_str.pop_back();
    const fs::path out_abs  = fs::absolute(out_str);
    const std::string leaf  = out_abs.filename().string();
    const fs::path parent   = fs::canonical(out_abs.parent_path(), ec);
    const std::string p_str = parent.string();
    const std::string r_str = root.string();
    const bool under_root = !ec && (p_str == r_str || p_str.rfind(r_str + "/", 0) == 0);
    if (!under_root || leaf.empty() || leaf == "." || leaf == ".." ||
        leaf.find(".staging") != std::string::npos || leaf.find(".old") != std::string::npos) {
        fprintf(stderr, "router_trace: --out %s is not a set name under %s — not writing there\n",
                out_arg.c_str(), r_str.c_str());
        return false;
    }
    sp.out     = parent / leaf;
    sp.stage   = parent / (leaf + ".staging");
    sp.old_set = parent / (leaf + ".old");
    if (fs::exists(sp.out) && !is_router_set(sp.out)) {
        fprintf(stderr, "router_trace: %s exists and holds no router set — not replacing it\n", sp.out.c_str());
        return false;
    }
    return true;
}

// A file of another architecture is refused from its header alone, before a weight is paged in.
static bool check_arch(const std::string & model_path, const std::string & expect_arch) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(model_path.c_str(), gp);
    if (!g) {
        fprintf(stderr, "router_trace: cannot read the GGUF header of %s\n", model_path.c_str());
        return false;
    }
    const int         k         = gguf_find_key(g, "general.architecture");
    const std::string file_arch = k >= 0 ? gguf_get_val_str(g, k) : "unknown";
    gguf_free(g);
    if (file_arch != expect_arch) {
        fprintf(stderr, "router_trace: %s is a %s model and this trace is for %s -- not loading it\n",
                model_path.c_str(), file_arch.c_str(), expect_arch.c_str());
        return false;
    }
    return true;
}

struct run_result {
    std::vector<int> moe_layers;  // fixed by the first decode call
    std::string      call_rows;   // the manifest's `call` rows
    double           wall_s = 0;
    long             majflt = 0;
};

// Opens topk-<layer>.u16 in the staging directory for every layer the first decode call captured.
static bool open_layer_files(trace_ctx & tr, const fs::path & stage, std::vector<int> & moe_layers) {
    for (int l = 0; l < tr.n_layer; ++l) {
        if (!tr.layers[l].got) continue;
        moe_layers.push_back(l);
        const fs::path f = stage / ("topk-" + std::to_string(l) + ".u16");
        tr.layers[l].file = fopen(f.c_str(), "wb");
        if (!tr.layers[l].file) {
            fprintf(stderr, "router_trace: cannot write %s\n", f.c_str());
            return false;
        }
    }
    if (moe_layers.empty()) {
        fprintf(stderr, "router_trace: the first decode call produced no ffn_moe_topk node\n");
        return false;
    }
    return true;
}

// One decode call's captures: every MoE layer exactly once, appended to its file and its counts.
static bool commit_call(trace_ctx & tr, int n, long long p0) {
    for (int l = 0; l < tr.n_layer; ++l) {
        layer_state & L = tr.layers[l];
        if (!L.got) {
            if (L.file) {
                fprintf(stderr, "router_trace: layer %d has no selection in the call at token %lld\n", l, p0);
                return false;
            }
            continue;
        }
        if (!L.file) {
            fprintf(stderr, "router_trace: layer %d has a selection at token %lld but not in the first call\n", l, p0);
            return false;
        }
        for (uint16_t id : L.rows) {
            L.counts[id]++;
            L.id_sum += id;
        }
        L.tokens += (uint64_t) n;
        if (fwrite(L.rows.data(), sizeof(uint16_t), L.rows.size(), L.file) != L.rows.size() || fflush(L.file) != 0) {
            fprintf(stderr, "router_trace: short write on topk-%d.u16\n", l);
            return false;
        }
    }
    return true;
}

// The trace: independent contexts of `chunk` tokens, each in decode calls of at most n_ubatch.
static bool run_trace(llama_context * ctx, trace_ctx & tr, std::vector<llama_token> & ids, long long chunk,
                      const fs::path & stage, run_result & r) {
    const long long n_ubatch = llama_n_ubatch(ctx);
    const long long n_tokens = (long long) ids.size();
    const long long n_chunks = (n_tokens + chunk - 1) / chunk;
    const faults    f_run0   = faults_now();
    const auto      t_run0   = std::chrono::steady_clock::now();
    long long       done     = 0;
    int             n_call   = 0;
    bool            ok       = true;
    for (long long c = 0; c < n_chunks && ok; ++c) {
        llama_kv_cache_clear(ctx);
        const long long c0 = c * chunk;
        const long long c1 = std::min(n_tokens, c0 + chunk);
        for (long long p0 = c0; p0 < c1 && ok; p0 += n_ubatch) {
            const int n = (int) std::min(n_ubatch, c1 - p0);
            tr.call_tokens = n;
            for (auto & L : tr.layers) L.got = false;

            const faults f0 = faults_now();
            const auto   t0 = std::chrono::steady_clock::now();
            const int rc = llama_decode(ctx, llama_batch_get_one(ids.data() + p0, n, (llama_pos) (p0 - c0), 0));
            const double ms = seconds_since(t0) * 1e3;
            const faults f1 = faults_now();

            if (rc != 0) {
                fprintf(stderr, "router_trace: llama_decode returned %d at token %lld\n", rc, p0);
                ok = false;
            } else if (tr.failed) {
                // The callback stops the graph on a bad capture, but the decode still returns 0.
                fprintf(stderr, "router_trace: %s\n", tr.error.c_str());
                ok = false;
            } else {
                ok = (!r.moe_layers.empty() || open_layer_files(tr, stage, r.moe_layers)) && commit_call(tr, n, p0);
            }
            if (!ok) break;

            done += n;
            const double run_s = seconds_since(t_run0);
            char row[160];
            snprintf(row, sizeof(row), "call\t%d\t%lld\t%lld\t%d\t%.1f\t%ld\t%ld\n", n_call, c, p0 - c0, n, ms,
                     f1.maj - f0.maj, f1.min - f0.min);
            r.call_rows += row;
            fprintf(stderr, "router_trace: call %d chunk %lld/%lld pos %lld +%d  %lld/%lld tokens  %.1f ms  "
                            "majflt +%ld  run %.1f s (%.1f tok/s wall)\n",
                    n_call, c + 1, n_chunks, p0 - c0, n, done, n_tokens, ms, f1.maj - f0.maj, run_s, done / run_s);
            ++n_call;
        }
    }
    r.wall_s = seconds_since(t_run0);
    r.majflt = faults_now().maj - f_run0.maj;
    for (int l : r.moe_layers) {
        if (tr.layers[l].file && fclose(tr.layers[l].file) != 0) {
            fprintf(stderr, "router_trace: cannot flush topk-%d.u16\n", l);
            ok = false;
        }
        tr.layers[l].file = nullptr;
    }
    return ok;
}

// counts.tsv: every (layer, expert) pair, so a reader never infers a zero from a missing row.
static bool write_counts(const fs::path & stage, const trace_ctx & tr, const std::vector<int> & moe_layers) {
    const fs::path f = stage / "counts.tsv";
    FILE * out = fopen(f.c_str(), "w");
    if (!out) {
        fprintf(stderr, "router_trace: cannot write %s\n", f.c_str());
        return false;
    }
    fprintf(out, "# layer\texpert\tcount\n");
    for (int l : moe_layers) {
        for (int e = 0; e < tr.n_expert; ++e) {
            fprintf(out, "%d\t%d\t%" PRIu64 "\n", l, e, tr.layers[l].counts[e]);
        }
    }
    if (fclose(out) != 0) {
        fprintf(stderr, "router_trace: cannot flush %s\n", f.c_str());
        return false;
    }
    return true;
}

// The manifest is written last, and its trailer is its last line: a set without the trailer is from
// a run that died, whatever else the directory holds.
static bool write_manifest(const fs::path & stage, const options & o, const gpt_params & params,
                           llama_context * ctx, const char * arch, const trace_ctx & tr, long long n_tokens,
                           long long ids_lines, double load_s, const run_result & r) {
    const fs::path f = stage / "MANIFEST.tsv";
    FILE * m = fopen(f.c_str(), "w");
    if (!m) {
        fprintf(stderr, "router_trace: cannot write %s\n", f.c_str());
        return false;
    }
    const char * slash = strrchr(params.model.c_str(), '/');
    const char * build = getenv("BLOOMERY_REF_BUILD");
    const char * md5   = getenv("BLOOMERY_ROUTER_IDS_MD5");
    fprintf(m, "%s — ik_llama.cpp MoE router selections (ffn_moe_topk) by prefill, one row per token\n",
            MANIFEST_TITLE);
    fprintf(m, "# model\t%s\n", params.model.c_str());
    fprintf(m, "# model_file\t%s\n", slash ? slash + 1 : params.model.c_str());
    fprintf(m, "# arch\t%s\n", arch);
    fprintf(m, "# build\t%s\n", build && *build ? build : "unknown");
    fprintf(m, "# ids\t%s\n", o.ids_path.c_str());
    fprintf(m, "# ids_md5\t%s\n", md5 && *md5 ? md5 : "unknown");
    fprintf(m, "# ids_lines\t%lld\n", ids_lines);
    fprintf(m, "# tokens\t%lld\n", n_tokens);
    fprintf(m, "# chunk\t%lld\n", o.chunk);
    fprintf(m, "# chunks\t%lld\n", (n_tokens + o.chunk - 1) / o.chunk);
    fprintf(m, "# n_ctx\t%u\n", llama_n_ctx(ctx));
    fprintf(m, "# n_batch\t%u\n", llama_n_batch(ctx));
    fprintf(m, "# n_ubatch\t%u\n", llama_n_ubatch(ctx));
    fprintf(m, "# n_threads\t%u\t%u\n", llama_n_threads(ctx), llama_n_threads_batch(ctx));
    fprintf(m, "# n_expert\t%d\n", tr.n_expert);
    fprintf(m, "# n_expert_used\t%d\n", tr.n_used);
    fprintf(m, "# schedule\t%s\n", o.every_node ? "every-node" : "top-k-only");
    fprintf(m, "# flags\t%s\n", o.flags.c_str());
    fprintf(m, "# load_s\t%.1f\n", load_s);
    fprintf(m, "# wall_s\t%.1f\n", r.wall_s);
    fprintf(m, "# run_tok_per_s\t%.2f\t(this run's own wall rate over the decode calls, not an engine speed)\n",
            n_tokens / r.wall_s);
    fprintf(m, "# majflt\t%ld\n", r.majflt);
    fprintf(m, "# layer\tlayer\tsource\tproducer\ttokens\tid_sum\tignored\tfile\n");
    for (int l : r.moe_layers) {
        const layer_state & L = tr.layers[l];
        fprintf(m, "layer\t%d\t%s\t%s\t%" PRIu64 "\t%" PRIu64 "\t%d\ttopk-%d.u16\n", l, source_name(L.source),
                L.src_name.c_str(), L.tokens, L.id_sum, L.ignored, l);
    }
    fprintf(m, "# call\tindex\tchunk\tpos0\ttokens\tms\tmajflt\tminflt\n");
    fputs(r.call_rows.c_str(), m);
    fprintf(m, "# complete\t%lld\t%zu\n", n_tokens, r.moe_layers.size());
    if (fclose(m) != 0) {
        fprintf(stderr, "router_trace: cannot flush %s\n", f.c_str());
        return false;
    }
    return true;
}

// Install: the previous set (a router set, checked again here) steps aside, the staged one takes its
// name, the old one goes. A failure between the two renames leaves both on disk, named.
static bool install_set(const set_paths & sp) {
    std::error_code ec;
    if (fs::exists(sp.out) && !is_router_set(sp.out)) {
        fprintf(stderr, "router_trace: %s changed into a non-router directory during the run — not replacing it\n",
                sp.out.c_str());
        return false;
    }
    fs::remove_all(sp.old_set, ec);
    if (fs::exists(sp.out)) {
        fs::rename(sp.out, sp.old_set, ec);
        if (ec) {
            fprintf(stderr, "router_trace: cannot move %s aside: %s\n", sp.out.c_str(), ec.message().c_str());
            return false;
        }
    }
    fs::rename(sp.stage, sp.out, ec);
    if (ec) {
        fprintf(stderr, "router_trace: cannot install %s: %s\n", sp.out.c_str(), ec.message().c_str());
        return false;
    }
    fs::remove_all(sp.old_set, ec);
    return true;
}

int main(int argc, char ** argv) {
    const auto t_start = std::chrono::steady_clock::now();

    options o;
    std::vector<char *> passthrough;
    if (!parse_options(argc, argv, o, passthrough)) return 2;

    if (!getenv("BLOOMERY_ROUTER_WRITE")) {
        fprintf(stderr,
                "router_trace: refusing to run -- this tool writes a router trace set under\n"
                "              $BLOOMERY_DATA/router/ and pages in most of the model file set.\n"
                "              tools/ref/router-trace.sh runs it under the machine-wide CPU lease\n"
                "              with witnesses around it and sets BLOOMERY_ROUTER_WRITE=1.\n");
        return 3;
    }

    set_paths sp;
    if (!resolve_set_paths(o.out_arg, sp)) return 2;

    std::vector<llama_token> ids;
    long long ids_lines = 0;
    if (!read_ids(o.ids_path, o.max_tokens, ids, ids_lines)) return 2;
    if (ids.empty()) {
        fprintf(stderr, "router_trace: %s holds no token ids\n", o.ids_path.c_str());
        return 2;
    }

    gpt_params params;
    if (!gpt_params_parse((int) passthrough.size(), passthrough.data(), params)) {
        fprintf(stderr, "router_trace: bad arguments\n");
        return 2;
    }
    if (!check_arch(params.model, o.expect_arch)) return 2;

    std::error_code ec;
    fs::remove_all(sp.stage, ec);
    if (!fs::create_directory(sp.stage, ec)) {
        fprintf(stderr, "router_trace: cannot create %s\n", sp.stage.c_str());
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    trace_ctx tr;
    tr.every_node            = o.every_node;
    params.cb_eval           = on_tensor;
    params.cb_eval_user_data = &tr;
    // A warmup decode routes to every expert (n_expert_used = n_expert) and would land in the counts.
    params.warmup            = false;

    llama_init_result init = llama_init_from_gpt_params(params);
    if (!init.model || !init.context) {
        fprintf(stderr, "router_trace: failed to load the model\n");
        return 1;
    }
    llama_model *   model = init.model;
    llama_context * ctx   = init.context;
    const double load_s = seconds_since(t_start);

    char arch[128];
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0) {
        snprintf(arch, sizeof(arch), "unknown");
    }
    tr.n_layer  = llama_n_layer(model);
    tr.n_expert = (int) meta_int(model, std::string(arch) + ".expert_count");
    tr.n_used   = (int) meta_int(model, std::string(arch) + ".expert_used_count");
    if (tr.n_layer <= 0 || tr.n_expert <= 0 || tr.n_expert > 65536 || tr.n_used <= 0 || tr.n_used > tr.n_expert) {
        fprintf(stderr, "router_trace: %s reports n_layer %d, expert_count %d, expert_used_count %d — not a MoE "
                        "model this tool can trace\n", arch, tr.n_layer, tr.n_expert, tr.n_used);
        return 1;
    }
    tr.layers.resize(tr.n_layer);
    for (auto & L : tr.layers) L.counts.assign(tr.n_expert, 0);

    const int n_vocab = llama_n_vocab(model);
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] >= n_vocab) {
            fprintf(stderr, "router_trace: token %zu of %s is id %d, past n_vocab %d\n", i, o.ids_path.c_str(), ids[i],
                    n_vocab);
            return 2;
        }
    }
    if (o.chunk > (long long) llama_n_ctx(ctx)) {
        fprintf(stderr, "router_trace: --chunk %lld is larger than n_ctx %u\n", o.chunk, llama_n_ctx(ctx));
        return 2;
    }

    const long long n_tokens = (long long) ids.size();
    fprintf(stderr, "router_trace: %lld tokens of %s in %lld chunk(s) of %lld, decode calls of <= %u, %s schedule; "
                    "loaded in %.1f s\n", n_tokens, o.ids_path.c_str(), (n_tokens + o.chunk - 1) / o.chunk, o.chunk,
            llama_n_ubatch(ctx), o.every_node ? "every-node" : "top-k-only", load_s);

    run_result r;
    if (!run_trace(ctx, tr, ids, o.chunk, sp.stage, r)) {
        fprintf(stderr, "router_trace: no trailer — %s is not a complete set and is not installed\n",
                sp.stage.c_str());
        return 1;
    }
    if (!write_counts(sp.stage, tr, r.moe_layers)) return 1;
    if (!write_manifest(sp.stage, o, params, ctx, arch, tr, n_tokens, ids_lines, load_s, r)) return 1;
    if (!install_set(sp)) return 1;

    printf("router_trace: %lld tokens x %zu layers in %.1f s (load %.1f s) into %s\n", n_tokens, r.moe_layers.size(),
           r.wall_s, load_s, sp.out.c_str());
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
