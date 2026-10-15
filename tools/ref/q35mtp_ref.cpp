// q35mtp_ref — llama.cpp mainline's MTP draft step, tap by tap, on a teacher-forced prompt.
//
// What it runs. The target's trunk takes the prompt's first <n> ids in one llama_decode with the nextn
// embeddings on and every position an output: row t of the trunk's `h_nextn` (the post-output_norm hidden state
// the MTP head is seeded with) and the trunk's logits of every position are kept. A second context of type
// LLAMA_CONTEXT_TYPE_MTP, over the same model loaded with load_mtp, then takes one token a step, in order,
// i = 0 .. n-2: token ids[i+1], position i, state h[i]. The MTP context's K/V therefore covers the prompt as a
// draft that walked it does. Each step's own argmax is kept beside the trunk's argmax at position i+1 (the token
// the target itself takes after ids[i+1]).
//
// What it dumps. At the steps named by --rows only, the graph nodes named by --tap (repeatable; the name is the
// graph's own, `cb(tensor, name, il)`, without the `-<il>` suffix the context adds), f32 or i32 as the node is,
// one file a (tap, row): `<tap>.r<row>.<f32|i32>`; the prompt ids (`prompt.ids`); the trunk's h rows
// (`trunk_h.f32`, row t at t*n_embd); and MANIFEST.tsv. The taps are read by the scheduler's eval callback, which
// ends a scheduler split at each tapped node: an op the backend would have fused across a tapped node runs
// unfused. A second MTP context with no callback runs every step first, and the tapped run's logits are held to
// it step by step (`# untapped`): the distance is what tapping costs the oracle.
//
// The manifest's header lines the reader checks: `# model` (the path as given), `# build`
// (BLOOMERY_REF_BUILD, the mainline commit the runner names), `# arch` (the file's general.architecture, which
// --arch must equal) and `# mtp <n_layer_nextn> <steps run>`, written only when the model has a next-token layer
// and the MTP context ran every step. `# complete` is written last, by renaming the manifest.
//
// The hidden state rides on the token's entry as llama_batch_ext_set_embd_token does for
// common/speculative.cpp's draft (llama_batch_ext_set_embd_state is a stub in this tree).
//
//   q35mtp_ref -m <gguf> --arch <name> --tokens-file <ids> --tokens-count <n> --rows <i,i,...> --tap <name> ...
//              --out <dir> [-ngl <n>] [-t <threads>] [-c <ctx>]
//
// Build: tools/ref/build-q35mtp-ref.sh   Run: tools/ref/dump-q35mtp.sh (just dump-ref-q35mtp)

#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "ref_io.h"

static int fail(const std::string & what, const std::string & detail = "") {
    fprintf(stderr, "q35mtp_ref: %s%s%s\n", what.c_str(), detail.empty() ? "" : ": ", detail.c_str());
    return 1;
}

static bool write_ids(const std::string & path, const std::vector<llama_token> & ids) {
    FILE * f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "q35mtp_ref: cannot write %s\n", path.c_str());
        return false;
    }
    for (llama_token id : ids) {
        if (fprintf(f, "%d\n", (int) id) < 0) {
            fclose(f);
            fprintf(stderr, "q35mtp_ref: short write on %s\n", path.c_str());
            return false;
        }
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "q35mtp_ref: short write on %s\n", path.c_str());
        return false;
    }
    return true;
}

static bool parse_rows(const char * spec, std::vector<int> & rows) {
    const char * p = spec;
    while (*p) {
        char * end = nullptr;
        errno = 0;
        const long v = strtol(p, &end, 10);
        if (end == p || errno != 0 || v < 0 || v > 1000000) return false;
        rows.push_back((int) v);
        if (*end == ',') ++end;
        else if (*end != '\0') return false;
        p = end;
    }
    return !rows.empty();
}

// The first index of the largest value: the argmax a tie gives to the lower id. A non-finite logit ends the run
// with 1 naming its id: no comparison orders a NaN, so no token or distance is written from such a row.
static int first_argmax(const float * x, int n) {
    int best = 0;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(x[i])) {
            fprintf(stderr, "q35mtp_ref: non-finite logit %g at id %d\n", (double) x[i], i);
            std::exit(1);
        }
        if (x[i] > x[best]) best = i;
    }
    return best;
}

// The graph's own name of a node, without the `-<il>` the context appends to a layered one.
static std::string base_name(const char * name) {
    std::string s(name);
    const size_t dash = s.rfind('-');
    if (dash == std::string::npos || dash + 1 >= s.size()) return s;
    for (size_t i = dash + 1; i < s.size(); ++i) {
        if (s[i] < '0' || s[i] > '9') return s;
    }
    return s.substr(0, dash);
}

struct Capture {
    int type = 0;        // GGML_TYPE_F32 or GGML_TYPE_I32
    int64_t count = 0;   // elements
    std::vector<char> bytes;
};

struct TapState {
    std::set<std::string> want;
    bool armed = false;
    std::map<std::string, Capture> got;
    std::string failure;   // the first refusal, by name
};

// Read tensor `t` row by row so a view (a top-k's first columns of an argsort) comes out dense.
static bool fetch(const ggml_tensor * t, Capture & out, std::string & why) {
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) {
        why = std::string(t->name) + " is " + ggml_type_name(t->type) + ", want f32 or i32";
        return false;
    }
    const size_t es = ggml_type_size(t->type);
    if (t->nb[0] != es) {
        why = std::string(t->name) + " has a strided first dimension";
        return false;
    }
    out.type = t->type;
    out.count = ggml_nelements(t);
    out.bytes.resize((size_t) out.count * es);
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, out.bytes.data(), 0, out.bytes.size());
        return true;
    }
    char * dst = out.bytes.data();
    const size_t row = (size_t) t->ne[0] * es;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                ggml_backend_tensor_get(t, dst, (size_t) i3 * t->nb[3] + (size_t) i2 * t->nb[2] + (size_t) i1 * t->nb[1], row);
                dst += row;
            }
        }
    }
    return true;
}

// Ask phase: observe the nodes the caller named while a tapped row runs. Data phase: copy each out.
// Returns true always: false would cancel the graph.
static bool on_node(struct ggml_tensor * t, bool ask, void * user) {
    auto * s = static_cast<TapState *>(user);
    const std::string base = base_name(t->name);
    if (ask) return s->armed && s->failure.empty() && s->want.count(base) != 0;
    if (s->got.count(base)) {
        s->failure = "tap " + base + " is named by two nodes of one step";
        return true;
    }
    Capture c;
    std::string why;
    if (!fetch(t, c, why)) {
        s->failure = why;
        return true;
    }
    s->got[base] = std::move(c);
    return true;
}

struct Mtp {
    llama_context * ctx = nullptr;
    llama_batch_ext * batch = nullptr;
};

// One teacher-forced step: token `id` at position `pos`, state `h` (n_embd floats).
static bool mtp_step(Mtp & m, llama_token id, int pos, const float * h, int n_embd) {
    llama_batch_ext_clear(m.batch);
    const int32_t idx = llama_batch_ext_add_token(m.batch, 0, id);
    if (idx < 0) {
        fprintf(stderr, "q35mtp_ref: llama_batch_ext_add_token rc %d\n", idx);
        return false;
    }
    const llama_pos p[GGML_MROPE_SECTIONS] = { pos, 0, 0, 0 };
    if (!llama_batch_ext_set_pos(m.batch, idx, p)) {
        fprintf(stderr, "q35mtp_ref: llama_batch_ext_set_pos refused position %d\n", pos);
        return false;
    }
    const llama_embd state = { h, 1, (size_t) n_embd };
    if (!llama_batch_ext_set_embd_token(m.batch, idx, state)) {
        fprintf(stderr, "q35mtp_ref: llama_batch_ext_set_embd_token refused the state at position %d\n", pos);
        return false;
    }
    if (!llama_batch_ext_set_output_logits(m.batch, idx, true)) {
        fprintf(stderr, "q35mtp_ref: llama_batch_ext_set_output_logits refused position %d\n", pos);
        return false;
    }
    const int32_t rc = llama_process(m.ctx, LLAMA_PROCESS_TYPE_DECODE, m.batch);
    if (rc != 0) {
        fprintf(stderr, "q35mtp_ref: llama_process rc %d at position %d\n", rc, pos);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, out, arch_want, tokens_file;
    int n_ctx = 0, ngl = 999, threads = 0, n = 0;
    std::vector<int> rows;
    std::vector<std::string> taps;
    for (int i = 1; i < argc; ++i) {
        const bool more = i + 1 < argc;
        if (strcmp(argv[i], "-m") == 0 && more) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && more) {
            out = argv[++i];
        } else if (strcmp(argv[i], "--arch") == 0 && more) {
            arch_want = argv[++i];
        } else if (strcmp(argv[i], "--tokens-file") == 0 && more) {
            tokens_file = argv[++i];
        } else if (strcmp(argv[i], "--tokens-count") == 0 && more) {
            if (!parse_i32("q35mtp_ref", "--tokens-count", argv[++i], 2, n)) return 2;
        } else if (strcmp(argv[i], "--rows") == 0 && more) {
            if (!parse_rows(argv[++i], rows)) return fail("--rows wants i,i,...", argv[i]);
        } else if (strcmp(argv[i], "--tap") == 0 && more) {
            taps.push_back(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && more) {
            if (!parse_i32("q35mtp_ref", "-c", argv[++i], 0, n_ctx)) return 2;
        } else if (strcmp(argv[i], "-ngl") == 0 && more) {
            if (!parse_i32("q35mtp_ref", "-ngl", argv[++i], 0, ngl)) return 2;
        } else if (strcmp(argv[i], "-t") == 0 && more) {
            if (!parse_i32("q35mtp_ref", "-t", argv[++i], 0, threads)) return 2;
        } else {
            return fail("unknown or incomplete argument", argv[i]);
        }
    }
    const char * build = getenv("BLOOMERY_REF_BUILD");
    const char * sha = getenv("BLOOMERY_REF_TOKENS_SHA256");
    if (model_path.empty() || out.empty() || arch_want.empty() || tokens_file.empty() || n < 2 || rows.empty() ||
        taps.empty() || !build || !*build) {
        fprintf(stderr, "usage: q35mtp_ref -m <gguf> --arch <name> --tokens-file <ids> --tokens-count <n> "
                        "--rows <i,i,...> --tap <name> ... --out <dir> [-ngl <n>] [-t <threads>] [-c <ctx>]\n"
                        "BLOOMERY_REF_BUILD must name the mainline build; --tokens-count is at least 2\n");
        return 2;
    }
    std::sort(rows.begin(), rows.end());
    if (std::adjacent_find(rows.begin(), rows.end()) != rows.end()) return fail("--rows names a row twice");
    if (rows.back() > n - 2) {
        return fail("a row feeds ids[row+1]: the last row is at most tokens-count - 2", std::to_string(rows.back()));
    }
    {
        std::set<std::string> uniq(taps.begin(), taps.end());
        if (uniq.size() != taps.size()) return fail("--tap names a tap twice");
    }
    std::vector<llama_token> ids;
    if (!read_token_file(tokens_file, n, ids)) return 2;
    if (n_ctx == 0) n_ctx = n;
    if (n_ctx < n) return fail("the prompt does not fit in -c", std::to_string(n_ctx));

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    mp.load_mtp = true;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) return fail("failed to load", model_path);
    char arch[128] = {};
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0 || arch_want != arch) {
        return fail("general.architecture is not the --arch", std::string(arch) + " vs " + arch_want);
    }
    const int n_nextn = llama_model_n_layer_nextn(model);
    if (n_nextn <= 0) return fail("the model has no next-token layer (llama_model_n_layer_nextn)", model_path);
    const int n_embd = llama_model_n_embd_out(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (n_embd <= 0 || n_vocab <= 0) return fail("the model names no width or vocabulary");
    for (llama_token id : ids) {
        if (id >= n_vocab) return fail("a prompt id is past the vocabulary", std::to_string(id));
    }
    std::string devices;
    for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(d);
        if (!devices.empty()) devices += "; ";
        devices += std::string(ggml_backend_dev_name(dev)) + " " + ggml_backend_dev_description(dev);
    }

    // The trunk: nextn rows for every position, logits for every position.
    llama_context_params tp = llama_context_default_params();
    tp.n_ctx = (uint32_t) n_ctx;
    tp.n_batch = (uint32_t) n;
    tp.n_ubatch = (uint32_t) n;
    tp.n_seq_max = 1;
    if (threads > 0) {
        tp.n_threads = threads;
        tp.n_threads_batch = threads;
    }
    llama_context * trunk = llama_init_from_model(model, tp);
    if (!trunk) return fail("failed to make the trunk context");
    llama_set_embeddings_nextn(trunk, true, /*masked*/ false);
    llama_batch tb = llama_batch_init(n, 0, 1);
    for (int t = 0; t < n; ++t) {
        tb.token[t] = ids[t];
        tb.pos[t] = t;
        tb.n_seq_id[t] = 1;
        tb.seq_id[t][0] = 0;
        tb.logits[t] = 1;
    }
    tb.n_tokens = n;
    if (llama_decode(trunk, tb) != 0) return fail("the trunk decode failed");
    llama_batch_free(tb);
    std::vector<float> h((size_t) n * n_embd);
    std::vector<int> target_next(n);
    for (int t = 0; t < n; ++t) {
        const float * row = llama_get_embeddings_nextn_ith(trunk, t);
        if (!row) return fail("no nextn row for position", std::to_string(t));
        for (int j = 0; j < n_embd; ++j) {
            if (!std::isfinite(row[j])) return fail("a trunk h_nextn value is not finite", "position " + std::to_string(t));
        }
        memcpy(h.data() + (size_t) t * n_embd, row, (size_t) n_embd * sizeof(float));
        const float * lg = llama_get_logits_ith(trunk, t);
        if (!lg) return fail("no trunk logits for position", std::to_string(t));
        target_next[t] = first_argmax(lg, n_vocab);
    }
    llama_free(trunk);

    // The MTP contexts, as common/speculative.cpp makes its draft's.
    auto make_mtp = [&](Mtp & m, TapState * taps_state) -> bool {
        llama_context_params cp = llama_context_default_params();
        cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cp.n_ctx = (uint32_t) n_ctx;
        cp.n_batch = (uint32_t) n;
        cp.n_ubatch = (uint32_t) n;
        cp.n_seq_max = 1;
        cp.n_rs_seq = 0;
        if (threads > 0) {
            cp.n_threads = threads;
            cp.n_threads_batch = threads;
        }
        if (taps_state) {
            cp.cb_eval = on_node;
            cp.cb_eval_user_data = taps_state;
        }
        m.ctx = llama_init_from_model(model, cp);
        if (!m.ctx) return false;
        llama_set_embeddings_nextn(m.ctx, true, /*masked*/ true);
        m.batch = llama_batch_ext_init(m.ctx);
        return m.batch != nullptr;
    };

    const int steps = n - 1;
    // The untapped run first: every step's logits.
    std::vector<float> plain((size_t) steps * n_vocab);
    {
        Mtp m;
        if (!make_mtp(m, nullptr)) return fail("failed to make the MTP context (llama_model_n_layer_nextn > 0, ctx_type MTP)");
        for (int i = 0; i < steps; ++i) {
            if (!mtp_step(m, ids[i + 1], i, h.data() + (size_t) i * n_embd, n_embd)) return fail("the untapped MTP step failed");
            const float * lg = llama_get_logits_ith(m.ctx, 0);
            if (!lg) return fail("no MTP logits at step", std::to_string(i));
            memcpy(plain.data() + (size_t) i * n_vocab, lg, (size_t) n_vocab * sizeof(float));
        }
        llama_batch_ext_free(m.batch);
        llama_free(m.ctx);
    }

    // The tapped run.
    TapState st;
    st.want.insert(taps.begin(), taps.end());
    Mtp m;
    if (!make_mtp(m, &st)) return fail("failed to make the tapped MTP context");
    if (mkdir(out.c_str(), 0755) != 0 && errno != EEXIST) return fail("cannot create", out);
    const std::set<int> want_rows(rows.begin(), rows.end());
    std::vector<int> mtp_argmax(steps);
    std::vector<std::string> tap_rows;   // manifest lines
    double worst_all = 0.0, worst_rows = 0.0;
    int same_all = 0, same_rows = 0;
    for (int i = 0; i < steps; ++i) {
        const bool tapped = want_rows.count(i) != 0;
        st.armed = tapped;
        st.got.clear();
        if (!mtp_step(m, ids[i + 1], i, h.data() + (size_t) i * n_embd, n_embd)) return fail("the MTP step failed");
        if (!st.failure.empty()) return fail("a tap is refused", st.failure);
        const float * lg = llama_get_logits_ith(m.ctx, 0);
        if (!lg) return fail("no MTP logits at step", std::to_string(i));
        for (int j = 0; j < n_vocab; ++j) {
            if (!std::isfinite(lg[j])) return fail("an MTP logit is not finite", "step " + std::to_string(i));
        }
        mtp_argmax[i] = first_argmax(lg, n_vocab);
        const float * pl = plain.data() + (size_t) i * n_vocab;
        double worst = 0.0;
        for (int j = 0; j < n_vocab; ++j) worst = std::max(worst, (double) std::fabs(lg[j] - pl[j]));
        worst_all = std::max(worst_all, worst);
        same_all += first_argmax(pl, n_vocab) == mtp_argmax[i];
        if (tapped) {
            worst_rows = std::max(worst_rows, worst);
            same_rows += first_argmax(pl, n_vocab) == mtp_argmax[i];
            for (const std::string & name : taps) {
                auto it = st.got.find(name);
                if (it == st.got.end()) return fail("tap " + name + " was not seen at row", std::to_string(i));
                const Capture & c = it->second;
                const char * ext = c.type == GGML_TYPE_F32 ? "f32" : "i32";
                const std::string file = name + ".r" + std::to_string(i) + "." + ext;
                if (!write_bytes(out + "/" + file, c.bytes.data(), c.bytes.size())) return 1;
                if (c.type == GGML_TYPE_F32) {
                    const float * v = (const float *) c.bytes.data();
                    for (int64_t j = 0; j < c.count; ++j) {
                        if (!std::isfinite(v[j])) return fail("tap " + name + " holds a value that is not finite", "row " + std::to_string(i));
                    }
                }
                tap_rows.push_back("tap\t" + name + "\t" + std::to_string(i) + "\t" + ext + "\t" + std::to_string(c.count) + "\t" + file);
            }
        }
    }
    llama_batch_ext_free(m.batch);
    llama_free(m.ctx);

    if (!write_ids(out + "/prompt.ids", ids)) return 1;
    if (!write_raw(out + "/trunk_h.f32", h.data(), h.size())) return 1;

    // The sanity line: the draft's token against the target's own next token after the fed id, as the gate's
    // acceptance reads it (the last 16 steps, and every step).
    const int last = std::min(16, steps);
    int hits_last = 0, hits_all = 0;
    for (int i = 0; i < steps; ++i) {
        const bool hit = mtp_argmax[i] == target_next[i + 1];
        hits_all += hit;
        if (i >= steps - last) hits_last += hit;
    }
    for (int r : rows) {
        fprintf(stderr, "q35mtp_ref: row %d token %d pos %d: mtp argmax %d, target next %d%s\n", r, (int) ids[r + 1], r,
                mtp_argmax[r], target_next[r + 1], mtp_argmax[r] == target_next[r + 1] ? "" : " (differs)");
    }
    fprintf(stderr, "q35mtp_ref: mtp argmax = target next at %d/%d of the last steps, %d/%d of every step\n", hits_last, last,
            hits_all, steps);
    fprintf(stderr, "q35mtp_ref: tapped vs untapped logits: worst |d| %.6g over every step, %.6g over the tapped rows; "
                    "argmax equal at %d/%d steps, %d/%zu tapped rows\n",
            worst_all, worst_rows, same_all, steps, same_rows, rows.size());

    const std::string final_path = out + "/MANIFEST.tsv";
    const std::string partial_path = final_path + ".partial";
    FILE * mf = fopen(partial_path.c_str(), "w");
    if (!mf) return fail("cannot write", partial_path);
    const char * slash = strrchr(model_path.c_str(), '/');
    std::string flags;
    for (int i = 1; i < argc; ++i) flags += (i > 1 ? " " : "") + std::string(argv[i]);
    fprintf(mf, "# q35mtp_ref — llama.cpp mainline's MTP draft step, taps at the named rows, raw little-endian\n");
    fprintf(mf, "# model\t%s\n", model_path.c_str());
    fprintf(mf, "# build\t%s\n", build);
    fprintf(mf, "# arch\t%s\n", arch);
    fprintf(mf, "# model_file\t%s\n", slash ? slash + 1 : model_path.c_str());
    fprintf(mf, "# flags\t%s\n", flags.c_str());
    fprintf(mf, "# tokens_file\t%s\n", tokens_file.c_str());
    fprintf(mf, "# tokens_file_sha256\t%s\n", sha && *sha ? sha : "unknown");
    fprintf(mf, "# tokens_count\t%d\n", n);
    fprintf(mf, "# mtp\t%d\t%d\n", n_nextn, steps);
    fprintf(mf, "# backend\tngl=%d\t%s\n", ngl, devices.c_str());
    fprintf(mf, "# n_embd\t%d\n", n_embd);
    fprintf(mf, "# n_vocab\t%d\n", n_vocab);
    fprintf(mf, "# rows");
    for (int r : rows) fprintf(mf, "\t%d", r);
    fprintf(mf, "\n");
    fprintf(mf, "# feed\tstep i: token ids[i+1], position i, state trunk h_nextn row i; steps 0..%d in order, taps at the rows\n",
            steps - 1);
    fprintf(mf, "# untapped\t%.9g\t%.9g\t%d\t%d\n", worst_all, worst_rows, same_all, same_rows);
    fprintf(mf, "# trunk columns\tname\ttype\trows\twidth\tfile\n");
    fprintf(mf, "trunk\th_nextn\tf32\t%d\t%d\ttrunk_h.f32\n", n, n_embd);
    fprintf(mf, "# step columns\trow\ttoken\tpos\tmtp_argmax\ttarget_next\n");
    for (int i = 0; i < steps; ++i) {
        fprintf(mf, "step\t%d\t%d\t%d\t%d\t%d\n", i, (int) ids[i + 1], i, mtp_argmax[i], target_next[i + 1]);
    }
    fprintf(mf, "# tap columns\tname\trow\ttype\tn\tfile\n");
    for (const std::string & line : tap_rows) fprintf(mf, "%s\n", line.c_str());
    fprintf(mf, "# complete\t%zu\t0\n", tap_rows.size());
    if (fclose(mf) != 0 || rename(partial_path.c_str(), final_path.c_str()) != 0) {
        return fail("cannot install", final_path);
    }
    fprintf(stderr, "q35mtp_ref: %zu tap rows (%zu taps at %zu rows), %d steps, into %s\n", tap_rows.size(), taps.size(),
            rows.size(), steps, out.c_str());
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
