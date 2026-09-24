// dump_draft — write ik_llama.cpp's DSpark draft tensors to disk as raw f32, one set per decode.
//
// The oracle of the V4.1 draft port. The target decodes a fixed prompt greedily with ik's DSpark
// speculative loop (the loop llama-spec-bench runs: prompt warmup, then common_speculative_run_round
// per step); the eval callback is on the DRAFT context only, so every node of every graph the draft
// computes — the feature-to-KV graph (dflash_kv_*) and the block pass (dsv4_dflash_*, the
// hyper-connection mixes, dflash_base_result_output, result_output, draft_argmax) — is written, and
// no target node is. The target runs with no callback, fused, as ik serves it.
//
// The file format and the discipline are dump_ref's (tools/ref/dump_ref.cpp's header is the
// long form): raw little-endian f32 per tensor, a logical twin for views and non-contiguous
// tensors, lossless integer twins (.i32/.i64) with `int` rows, graph inputs as `input` rows at their
// first reader, persistent leaves (the draft's KV ring, the target-feature window) as `input` rows
// and graph scratch as `skip-input` rows, BLOOMERY_REF_WRITE=1 or nothing is written, a
// `.partial` manifest renamed only after the decode returns, `# build` in the header, and the
// `# complete <written> <skipped>` trailer only when every file was written.
//
// What differs, because a draft set is many small graphs instead of one:
//
//  - Blocks. Block b is everything the draft context computes during the b-th speculative round;
//    the prompt warmup is block -1. A block runs up to two graphs: the feature-to-KV graph (graph `kv`,
//    the committed target rows into the draft's KV ring) and the block pass (graph `block`). Occurrence
//    counters start at zero in every block and every graph, and every file name starts with both:
//    `b<b>.[kv.]<name>.<occurrence>[.input|.logical].<type>` (`w.` for the warmup). Every manifest row
//    of a block ends with four columns — `block`, `row` (`-`: a tensor row covers the whole block),
//    `accepted` (the verify outcome of that block, `-` for the warmup) and `graph` (kv or block) — so a
//    block's rows are buffered and written after its verify. Each graph writes an input at its own first
//    reader: the ring is a kv input as the append found it and a block input as the block pass reads it.
//  - `draft` rows: one per row of the block's last draft_argmax, `draft block row token`.
//  - `verify` rows: one per round, `verify block pos id_last carry drafted accepted target`: the
//    target position the round started at, the token the block starts from (ik's sampled_before)
//    and whether it came from the previous round's carry, the tokens ik proposed and accepted
//    (its own metrics, summed over stages), and the target tokens the round committed.
//  - `plain` rows: a step decoded without the draft (ik's loop drafts only with 3 or more tokens of
//    budget left), `plain pos token`.
//
// The feature-to-KV graph runs on a scheduler of its own that never gets the context's callback;
// this binary interposes ggml_backend_sched_graph_compute_async and hooks any graph holding
// dflash_kv_fused_target, split only after that node (on_kv_tensor says why): of that graph the set holds its
// inputs, the ring before the append, and dflash_kv_fused_target. Host inputs the draft reads on the card
// reach the callback only as scheduler copies (`CUDA0#inp_pos#0`); those are written as input rows under
// the copy's name.
//
// The callback asks for every node of the block pass, so it runs under the dumped schedule: node by
// node, no CUDA graph, no fusion. Its arithmetic is the dumped schedule's and the proposals and
// accept counts in this set are that schedule's; the target's are ik's serving path.
//
// Build: tools/ref/build-dump-draft.sh   Run: tools/ref/dump-draft.sh (never by hand)

#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "sampling.h"
#include "speculative.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <sys/stat.h>
#include <vector>

#include <dlfcn.h>

struct dump_ctx {
    std::string                   dir;
    FILE *                        manifest = nullptr;
    std::vector<std::pair<std::string, bool>> pending;  // this block's rows and whether each is the kv graph's
    int                           block = -1;   // -1: the prompt warmup
    bool                          kv = false;   // the node being handled is the feature-to-KV graph's
    // Per block and per graph (index: kv): node name -> times emitted, the same for inputs and state, and
    // the inputs already written — each graph writes an input at its own first reader.
    std::map<std::string, int>    seen_by[2];
    std::map<std::string, int>    seen_input_by[2];
    std::set<const ggml_tensor *> inputs_done_by[2];
    std::set<ggml_backend_buffer_t> node_bufs;  // buffers the draft's own nodes live in
    std::map<std::string, int>    unhandled;    // type name -> tensors skipped for it
    std::vector<uint8_t>          staging;      // device tensors land here before the write
    std::vector<int32_t>          argmax;       // the block's last draft_argmax
    bool                          failed  = false;
    int                           written = 0;
    int                           skipped = 0;
    int                           inputs  = 0;
    int                           twins   = 0;
    int                           state   = 0;
    int                           scratch = 0;
    int                           copies  = 0;  // scheduler copies of host inputs, written as input rows
    int                           kv_graphs = 0;  // feature-to-KV graphs the interposer hooked
};

static std::string safe_name(const char * name) {
    std::string s(name);
    for (char & c : s) {
        if (c == '/' || c == '\\' || c == ' ') c = '_';
    }
    return s;
}

static void row(dump_ctx * d, const char * fmt, ...) {
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    d->pending.emplace_back(buf, d->kv);
}

static void begin_block(dump_ctx * d, int block) {
    d->block = block;
    for (int g = 0; g < 2; ++g) {
        d->seen_by[g].clear();
        d->seen_input_by[g].clear();
        d->inputs_done_by[g].clear();
    }
    d->argmax.clear();
}

// Writes the block's buffered rows with its three trailing columns, then its draft rows.
static void flush_block(dump_ctx * d, const std::string & accepted) {
    for (const auto & [r, kv] : d->pending) {
        fprintf(d->manifest, "%s\t%d\t-\t%s\t%s\n", r.c_str(), d->block, accepted.c_str(), kv ? "kv" : "block");
    }
    d->pending.clear();
    for (size_t j = 0; j < d->argmax.size(); ++j) {
        fprintf(d->manifest, "draft\t%d\t%zu\t%d\n", d->block, j, d->argmax[j]);
    }
}

enum elem_kind { ELEM_UNHANDLED, ELEM_FLOAT, ELEM_INT };

static elem_kind read_elem(ggml_type type, const uint8_t * p, float & f, int64_t & i) {
    switch (type) {
        case GGML_TYPE_F32:  { float       x; memcpy(&x, p, sizeof x); f = x;                    return ELEM_FLOAT; }
        case GGML_TYPE_F16:  { ggml_fp16_t x; memcpy(&x, p, sizeof x); f = ggml_fp16_to_fp32(x); return ELEM_FLOAT; }
        case GGML_TYPE_BF16: { ggml_bf16_t x; memcpy(&x, p, sizeof x); f = ggml_bf16_to_fp32(x); return ELEM_FLOAT; }
        case GGML_TYPE_I8:   { int8_t      x; memcpy(&x, p, sizeof x); i = x; break; }
        case GGML_TYPE_I16:  { int16_t     x; memcpy(&x, p, sizeof x); i = x; break; }
        case GGML_TYPE_I32:  { int32_t     x; memcpy(&x, p, sizeof x); i = x; break; }
        case GGML_TYPE_I64:  { int64_t     x; memcpy(&x, p, sizeof x); i = x; break; }
        default:             return ELEM_UNHANDLED;
    }
    f = (float) i;
    return ELEM_INT;
}

static void gather(const ggml_tensor * t, const uint8_t * src, bool logical,
                   std::vector<float> & f, std::vector<int64_t> & ints) {
    const size_t esize = ggml_type_size(t->type);
    const bool   is_int = !ints.empty();
    int64_t v_int = 0;
    size_t  li    = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
        const size_t off = logical
            ? (size_t) i0 * t->nb[0] + (size_t) i1 * t->nb[1] + (size_t) i2 * t->nb[2] + (size_t) i3 * t->nb[3]
            : li * esize;
        read_elem(t->type, src + off, f[li], v_int);
        if (is_int) ints[li] = v_int;
        ++li;
    }
}

static bool write_raw(const std::string & path, const void * data, size_t size, size_t count) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "dump_draft: cannot write %s\n", path.c_str());
        return false;
    }
    if (fwrite(data, size, count, f) != count) {
        fprintf(stderr, "dump_draft: short write on %s\n", path.c_str());
        fclose(f);
        return false;
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "dump_draft: cannot flush %s\n", path.c_str());
        return false;
    }
    return true;
}

static bool write_twin(dump_ctx * d, const ggml_tensor * t, const char * name, int occurrence, bool input,
                       const std::string & stem, bool logical, const std::vector<int64_t> & ints) {
    const bool   wide = t->type == GGML_TYPE_I64;
    const char * twin = wide ? "i64" : "i32";
    const std::string file = stem + (logical ? ".logical." : ".") + twin;
    uint64_t sum = 0;
    uint64_t absmax = 0;
    for (int64_t v : ints) {
        sum += (uint64_t) v;
        const uint64_t mag = v < 0 ? 0 - (uint64_t) v : (uint64_t) v;
        if (mag > absmax) absmax = mag;
    }
    bool ok;
    if (wide) {
        ok = write_raw(d->dir + "/" + file, ints.data(), sizeof(int64_t), ints.size());
    } else {
        const std::vector<int32_t> narrow(ints.begin(), ints.end());
        ok = write_raw(d->dir + "/" + file, narrow.data(), sizeof(int32_t), narrow.size());
    }
    if (!ok) return false;
    row(d, "int\t%s\t%d\t%s\t%s\t%s\t%s\t%zu\t%zu\t%" PRId64 "\t%" PRIu64 "\t%s",
        name, occurrence, input ? "input" : "tensor", ggml_type_name(t->type), twin,
        logical ? "logical" : "flat", ints.size(), ints.size() * (wide ? 8 : 4),
        (int64_t) sum, absmax, file.c_str());
    d->twins++;
    return true;
}

static bool dump_one(dump_ctx * d, const ggml_tensor * t, bool input) {
    const char * name       = t->name[0] ? t->name : "(unnamed)";
    const int    occurrence = (input ? d->seen_input_by[d->kv] : d->seen_by[d->kv])[name]++;
    const char * skip_kind  = input ? "skip-input" : "skip";

    if (ggml_is_quantized(t->type)) {
        row(d, "%s\t%s\t%d\t%s\tquantized", skip_kind, name, occurrence, ggml_type_name(t->type));
        if (!input) d->skipped++;
        return true;
    }
    const uint8_t zero[8] = {};
    float   f0;
    int64_t i0;
    const elem_kind kind = read_elem(t->type, zero, f0, i0);
    if (kind == ELEM_UNHANDLED) {
        row(d, "%s\t%s\t%d\t%s\tunhandled", skip_kind, name, occurrence, ggml_type_name(t->type));
        d->unhandled[ggml_type_name(t->type)]++;
        if (!input) d->skipped++;
        return true;
    }
    if (!t->buffer || !t->data) {
        row(d, "%s\t%s\t%d\t%s\tunallocated", skip_kind, name, occurrence, ggml_type_name(t->type));
        if (!input) d->skipped++;
        return true;
    }

    const size_t nbytes = ggml_nbytes(t);
    const uint8_t * src;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        src = (const uint8_t *) t->data;
    } else {
        d->staging.resize(nbytes);
        ggml_backend_tensor_get(t, d->staging.data(), 0, nbytes);
        src = d->staging.data();
    }

    const size_t n = (size_t) ggml_nelements(t);
    std::vector<float>   out(n);
    std::vector<int64_t> ints(kind == ELEM_INT ? n : 0);
    gather(t, src, false, out, ints);
    double sum = 0.0;
    for (float v : out) sum += v;

    const std::string prefix = (d->block < 0 ? std::string("w.") : "b" + std::to_string(d->block) + ".") + (d->kv ? "kv." : "");
    const std::string stem   = prefix + safe_name(name) + "." + std::to_string(occurrence) + (input ? ".input" : "");
    if (!write_raw(d->dir + "/" + stem + ".f32", out.data(), sizeof(float), n)) return false;

    const bool contig  = ggml_is_contiguous(t);
    const bool is_view = t->view_src != nullptr;
    const bool logical = !contig || is_view;
    std::vector<float>   logical_out;
    std::vector<int64_t> logical_ints;
    if (logical) {
        logical_out.resize(n);
        logical_ints.resize(ints.size());
        gather(t, src, true, logical_out, logical_ints);
        if (!write_raw(d->dir + "/" + stem + ".logical.f32", logical_out.data(), sizeof(float), n)) return false;
    }

    auto src_name = [](const struct ggml_tensor * s) -> const char * {
        return s ? (s->name[0] ? s->name : "(unnamed)") : "-";
    };
    row(d, "%s\t%s\t%d\t%s\t%lld\t%lld\t%lld\t%lld\t%zu\t%.6f\t%s\t%d\t%d\t%s\t%s",
        input ? "input" : "tensor", name, occurrence, ggml_type_name(t->type),
        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
        n * sizeof(float), sum, ggml_op_desc(t),
        contig ? 1 : 0, logical ? 1 : 0, src_name(t->src[0]), src_name(t->src[1]));
    if (input) {
        d->inputs++;
    } else {
        d->written++;
    }

    if (kind == ELEM_INT) {
        if (!write_twin(d, t, name, occurrence, input, stem, false, ints)) return false;
        if (logical && !write_twin(d, t, name, occurrence, input, stem, true, logical_ints)) return false;
    }
    if (!input && strcmp(name, "draft_argmax") == 0 && t->type == GGML_TYPE_I32 && contig) {
        d->argmax.assign(ints.begin(), ints.end());
    }
    return true;
}

// A leaf the host does not fill that is not a weight: the draft's persistent state (its KV ring, the
// target-feature window) or graph scratch, told apart by dump_state as in dump_ref.
static bool is_state_leaf(const ggml_tensor * s) {
    return s && s->op == GGML_OP_NONE && !(s->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) &&
           s->buffer && ggml_backend_buffer_get_usage(s->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static bool dump_state(dump_ctx * d, const ggml_tensor * s) {
    const char * name = s->name[0] ? s->name : "(unnamed)";
    if (d->node_bufs.empty()) {
        fprintf(stderr, "dump_draft: leaf %s is read before any graph-allocated node — cannot tell state from scratch\n",
                name);
        return false;
    }
    // A scheduler copy (`<backend>#<source>#<n>`) is the only form in which a host input the draft reads on
    // the card (positions, the mask, the ring rows to write) reaches this callback: the copy runs when its
    // split starts, before any of the split's nodes, so its bytes are its source's. It is written as an input
    // row under its own name.
    const bool sched_copy = strchr(name, '#') != nullptr;
    if (d->node_bufs.count(s->buffer) && !sched_copy) {
        row(d, "skip-input\t%s\t%d\t%s\tgraph-scratch", name, d->seen_input_by[d->kv][name]++, ggml_type_name(s->type));
        d->scratch++;
        return true;
    }
    const int inputs = d->inputs;
    if (!dump_one(d, s, true)) return false;
    (sched_copy ? d->copies : d->state) += d->inputs - inputs;
    return true;
}

// The draft's feature-to-KV graph (dflash_kv_*) runs on a scheduler of its own that ik creates and computes
// in the same call and never gives the context's eval callback. This binary interposes the one entry point
// both schedulers are computed through: a graph that holds dflash_kv_fused_target gets the draft callback on
// its scheduler before it runs. Everything else passes through untouched.
static dump_ctx * g_dump = nullptr;
static void write_sources(dump_ctx * d, const ggml_tensor * t);

// The feature-to-KV graph is not computed node by node: split per node, ik's append rewrites ring rows the
// unsplit graph leaves alone, so a set taken that way is not ik's answer. Every node's sources are
// still written at their ask (before the scheduler computes the range), and one node is asked for:
// dflash_kv_fused_target, the draft's input features after fc and hidden_norm. The ring the graph writes is
// read again by the block pass, as that graph's input.
static int on_kv_tensor(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (dump_ctx *) user_data;
    d->kv = true;
    if (d->failed) {
        return 0;
    }
    const bool want = strcmp(t->name, "dflash_kv_fused_target") == 0;
    if (ask) {
        write_sources(d, t);
        return want ? 1 : 0;
    }
    if (want && !dump_one(d, t, false)) {
        d->failed = true;
        return 0;
    }
    return 1;
}

extern "C" enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    using fn_t = enum ggml_status (*)(ggml_backend_sched_t, struct ggml_cgraph *);
    static fn_t real = (fn_t) dlsym(RTLD_NEXT, "ggml_backend_sched_graph_compute_async");
    if (!real) {
        fprintf(stderr, "dump_draft: cannot find ggml_backend_sched_graph_compute_async in libggml\n");
        abort();
    }
    if (g_dump && ggml_graph_get_tensor(graph, "dflash_kv_fused_target")) {
        ggml_backend_sched_set_eval_callback(sched, on_kv_tensor, g_dump);
        g_dump->kv_graphs++;
    }
    return real(sched, graph);
}

// The ask half shared by both graphs: the node's inputs and state leaves, each at its first reader in
// this graph of this block, read before the scheduler computes the range the node ends.
static void write_sources(dump_ctx * d, const ggml_tensor * t) {
    if (!t->view_src && t->buffer) d->node_bufs.insert(t->buffer);
    auto & done = d->inputs_done_by[d->kv];
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        const ggml_tensor * s = t->src[j];
        if (s && s->op == GGML_OP_NONE && (s->flags & GGML_TENSOR_FLAG_INPUT) &&
            !(s->flags & GGML_TENSOR_FLAG_OUTPUT) && done.insert(s).second && !dump_one(d, s, true)) {
            d->failed = true;
            return;
        }
        if (is_state_leaf(s) && done.insert(s).second && !dump_state(d, s)) {
            d->failed = true;
            return;
        }
    }
}

// The block pass: every node asked for and written.
static int on_tensor(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (dump_ctx *) user_data;
    d->kv = false;
    if (d->failed) {
        return ask ? 1 : 0;
    }
    if (ask) {
        write_sources(d, t);
        return 1;
    }
    if (!dump_one(d, t, false)) {
        d->failed = true;
        return 0;
    }
    return 1;
}

static void report_unhandled(const dump_ctx & d) {
    int total = 0;
    std::string types;
    for (const auto & it : d.unhandled) {
        total += it.second;
        types += (types.empty() ? "" : ", ") + it.first + " x" + std::to_string(it.second);
    }
    fprintf(stderr, "dump_draft: %d tensors skipped as unhandled: %s\n", total, types.empty() ? "none" : types.c_str());
}

static bool read_token_file(const std::string & path, long long count, std::vector<llama_token> & ids) {
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "dump_draft: cannot read %s\n", path.c_str());
        return false;
    }
    std::string line;
    long long   lines = 0;
    while ((long long) ids.size() < count && std::getline(in, line)) {
        ++lines;
        char * end = nullptr;
        errno = 0;
        const long v = strtol(line.c_str(), &end, 10);
        if (line.empty() || *end != '\0' || errno != 0 || v < 0 || v > INT32_MAX) {
            fprintf(stderr, "dump_draft: %s:%lld is not a token id: '%s'\n", path.c_str(), lines, line.c_str());
            return false;
        }
        ids.push_back((llama_token) v);
    }
    if ((long long) ids.size() < count) {
        fprintf(stderr, "dump_draft: %s holds %zu token ids and --tokens-count asks for %lld\n", path.c_str(),
                ids.size(), count);
        return false;
    }
    return true;
}

static std::string file_arch(const std::string & path) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return "";
    const int k = gguf_find_key(g, "general.architecture");
    const std::string arch = k >= 0 ? gguf_get_val_str(g, k) : "unknown";
    gguf_free(g);
    return arch;
}

static const char * basename_of(const std::string & path) {
    const char * slash = strrchr(path.c_str(), '/');
    return slash ? slash + 1 : path.c_str();
}

static std::string join(const llama_tokens & ids) {
    std::string s;
    for (size_t i = 0; i < ids.size(); ++i) s += (i ? "," : "") + std::to_string(ids[i]);
    return s.empty() ? "-" : s;
}

struct spec_counts {
    uint64_t drafted  = 0;
    uint64_t accepted = 0;
    std::vector<uint64_t> drafted_by_position;
    std::vector<uint64_t> accepted_by_position;
};

static spec_counts counts_of(const common_speculative * spec) {
    spec_counts c;
    for (const auto & st : common_speculative_get_metrics_snapshot(spec).stages) {
        c.drafted  += st.n_gen_tokens;
        c.accepted += st.n_acc_tokens;
        c.drafted_by_position.resize(std::max(c.drafted_by_position.size(), st.drafted_by_position.size()));
        c.accepted_by_position.resize(std::max(c.accepted_by_position.size(), st.accepted_by_position.size()));
        for (size_t i = 0; i < st.drafted_by_position.size(); ++i) c.drafted_by_position[i] += st.drafted_by_position[i];
        for (size_t i = 0; i < st.accepted_by_position.size(); ++i) c.accepted_by_position[i] += st.accepted_by_position[i];
    }
    return c;
}

static bool decode_tokens(llama_context * ctx, const llama_tokens & toks, int n_batch, int & n_past,
                          common_speculative * spec, bool warmup) {
    for (int i = 0; i < (int) toks.size(); i += n_batch) {
        const int n_eval = std::min(n_batch, (int) toks.size() - i);
        llama_batch batch = llama_batch_init(n_eval, 0, 1);
        for (int k = 0; k < n_eval; ++k) {
            // Every row an output, as llama-spec-bench does: DSpark's feature capture reads them.
            common_batch_add(batch, toks[i + k], n_past + k, { 0 }, true);
        }
        const int rc = llama_decode(ctx, batch);
        const bool ok = rc == 0 && (!warmup || common_speculative_on_target_seq_batch(spec, ctx, batch, 0, true) == 0);
        llama_batch_free(batch);
        if (!ok) {
            fprintf(stderr, "dump_draft: %s decode failed at position %d (llama_decode %d)\n",
                    warmup ? "prompt" : "step", n_past, rc);
            return false;
        }
        n_past += n_eval;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string flags;
    for (int i = 1; i < argc; ++i) flags += (i > 1 ? " " : "") + std::string(argv[i]);

    // Ours: --tokens-file, --tokens-count, --expect-arch, --expect-draft-arch. The rest is gpt_params.
    std::string tokens_file;
    long long   tokens_count = 0;
    std::string expect_arch;
    std::string expect_draft_arch;
    std::vector<char *> passthrough;
    passthrough.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokens-file") == 0 && i + 1 < argc) {
            tokens_file = argv[++i];
        } else if (strcmp(argv[i], "--tokens-count") == 0 && i + 1 < argc) {
            tokens_count = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--expect-arch") == 0 && i + 1 < argc) {
            expect_arch = argv[++i];
        } else if (strcmp(argv[i], "--expect-draft-arch") == 0 && i + 1 < argc) {
            expect_draft_arch = argv[++i];
        } else {
            passthrough.push_back(argv[i]);
        }
    }
    if (tokens_file.empty() || tokens_count <= 0) {
        fprintf(stderr, "dump_draft: --tokens-file <ids> --tokens-count <n> are required; this tool does not tokenize\n");
        return 2;
    }
    if (expect_arch.empty() || expect_draft_arch.empty()) {
        fprintf(stderr, "dump_draft: --expect-arch and --expect-draft-arch are required\n");
        return 2;
    }
    std::vector<llama_token> prompt;
    if (!read_token_file(tokens_file, tokens_count, prompt)) return 2;

    gpt_params params;
    if (!gpt_params_parse((int) passthrough.size(), passthrough.data(), params)) {
        fprintf(stderr, "dump_draft: bad arguments\n");
        return 2;
    }
    if (params.speculative.model.empty() || !params.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_DSPARK)) {
        fprintf(stderr, "dump_draft: needs -md <draft> and --spec-type dspark[:...]\n");
        return 2;
    }
    const int n_predict = params.n_predict;
    // The past-window mask (q_pos - k_pos < 128) hides the oldest 1 + j ring rows from block row j,
    // where the reference shows every row the last 128; the two rules agree while every query
    // position stays below 128 minus the 5-row block. A set past that is confounded by the defect.
    if (n_predict <= 0 || (long long) prompt.size() + n_predict >= 123) {
        fprintf(stderr, "dump_draft: -n %d after %zu prompt tokens reaches depth 123 or more (or is not positive)\n",
                n_predict, prompt.size());
        return 2;
    }

    // This binary's side effect is an oracle set: without BLOOMERY_REF_WRITE=1 it writes nothing
    // (dump_ref's rule, for the reason its header gives), and it never picks its own directory.
    const char * ref_dir = getenv("BLOOMERY_REF_DIR");
    if (!getenv("BLOOMERY_REF_WRITE") || !ref_dir || !*ref_dir) {
        fprintf(stderr,
                "dump_draft: refusing to run -- this tool writes the draft oracle set. It is not a general ik\n"
                "            harness; tools/ref/dump-draft.sh sets BLOOMERY_REF_WRITE=1 and BLOOMERY_REF_DIR to a\n"
                "            staging directory. For an interactive draft run use llama-spec-bench.\n");
        return 3;
    }

    for (const auto & [path, want, what] : { std::make_tuple(params.model, expect_arch, "target"),
                                             std::make_tuple(params.speculative.model, expect_draft_arch, "draft") }) {
        const std::string got = file_arch(path);
        if (got != want) {
            fprintf(stderr, "dump_draft: the %s %s is a '%s' model, expected '%s' -- not loading it\n", what,
                    path.c_str(), got.c_str(), want.c_str());
            return 2;
        }
    }

    dump_ctx d;
    d.dir = ref_dir;
    mkdir(d.dir.c_str(), 0755);
    const std::string manifest_final   = d.dir + "/MANIFEST.tsv";
    const std::string manifest_partial = manifest_final + ".partial";
    d.manifest = fopen(manifest_partial.c_str(), "w");
    if (!d.manifest) {
        fprintf(stderr, "dump_draft: cannot write %s\n", manifest_partial.c_str());
        return 1;
    }

    // Greedy target, fixed seed: the set is one decode, not a sample.
    params.sparams.temp = 0.0f;
    params.warmup       = false;
    params.cb_eval      = nullptr;  // the target is not dumped; its feature capture chains to this
    common_speculative_prepare_startup(params);

    llama_backend_init();
    llama_numa_init(params.numa);
    llama_init_result init = llama_init_from_gpt_params(params);
    llama_model *   model = init.model;
    llama_context * ctx   = init.context;
    if (!model || !ctx) {
        fprintf(stderr, "dump_draft: failed to load the target\n");
        return 1;
    }
    if (!common_speculative_finalize_startup(params, model) || !params.speculative.model_dft) {
        fprintf(stderr, "dump_draft: failed to load the draft\n");
        return 1;
    }
    // The draft context is created from cparams_dft by try_init: the callback goes there and nowhere else.
    params.speculative.cparams_dft.cb_eval           = on_tensor;
    params.speculative.cparams_dft.cb_eval_user_data = &d;
    g_dump = &d;
    common_speculative * spec = nullptr;
    if (!common_speculative_is_compat(ctx) ||
        common_speculative_try_init(params.speculative, ctx, &spec) != COMMON_SPECULATIVE_INIT_READY || !spec) {
        fprintf(stderr, "dump_draft: the DSpark speculative context did not initialize\n");
        return 1;
    }
    common_sampler * sampler = common_sampler_init(model, params.sparams);
    if (!sampler) {
        fprintf(stderr, "dump_draft: failed to initialize the sampler\n");
        return 1;
    }
    const int n_ctx   = (int) llama_n_ctx(ctx);
    const int n_batch = params.n_batch;
    if ((int) prompt.size() + n_predict >= n_ctx - 2) {
        fprintf(stderr, "dump_draft: %zu + %d tokens do not fit in n_ctx %d\n", prompt.size(), n_predict, n_ctx);
        return 2;
    }

    char arch[128];
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0) snprintf(arch, sizeof arch, "unknown");
    fprintf(d.manifest, "# dump_draft — ik_llama.cpp DSpark draft tensors, raw f32, little-endian\n");
    fprintf(d.manifest, "# model\t%s\n", params.model.c_str());
    if (const char * b = getenv("BLOOMERY_REF_BUILD")) fprintf(d.manifest, "# build\t%s\n", b);
    fprintf(d.manifest, "# arch\t%s\n", arch);
    fprintf(d.manifest, "# model_file\t%s\n", basename_of(params.model));
    fprintf(d.manifest, "# draft_model\t%s\n", params.speculative.model.c_str());
    fprintf(d.manifest, "# draft_model_file\t%s\n", basename_of(params.speculative.model));
    fprintf(d.manifest, "# draft_arch\t%s\n", expect_draft_arch.c_str());
    fprintf(d.manifest, "# tokens\t%s\n", join(prompt).c_str());
    fprintf(d.manifest, "# tokens_file\t%s\n", tokens_file.c_str());
    const char * sha = getenv("BLOOMERY_REF_TOKENS_SHA256");
    fprintf(d.manifest, "# tokens_file_sha256\t%s\n", sha && *sha ? sha : "unknown");
    fprintf(d.manifest, "# tokens_count\t%zu\n", prompt.size());
    fprintf(d.manifest, "# n_predict\t%d\n", n_predict);
    fprintf(d.manifest, "# spec\t%s\n", common_speculative_stage_chain_to_str(params.speculative).c_str());
    fprintf(d.manifest, "# flags\t%s\n", flags.c_str());
    fprintf(d.manifest, "# schedule\tdraft: every node asked for and computed alone (no CUDA graph, no fusion); "
                        "target: no callback, ik's serving schedule\n");
    fprintf(d.manifest, "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1\tblock\trow\taccepted\tgraph\n");
    fprintf(d.manifest, "# int\tname\toccurrence\tof\ttype\ttwin\tlayout\tcount\tbytes\tsum\tabsmax\tfile\tblock\trow\taccepted\tgraph\n");
    fprintf(d.manifest, "# input\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1\tblock\trow\taccepted\tgraph\n");
    fprintf(d.manifest, "# draft\tblock\trow\ttoken\n");
    fprintf(d.manifest, "# verify\tblock\tpos\tid_last\tcarry\tdrafted\taccepted\ttarget\n");
    fprintf(d.manifest, "# plain\tpos\ttoken\n");

    // The loop is llama-spec-bench's (spec_bench_run_attempt), with the prompt given as ids.
    common_sampler_reset(sampler);
    common_speculative_clear_sequence_kv(spec, ctx, 0);
    for (llama_token t : prompt) common_sampler_accept(sampler, ctx, t, false);
    int n_past = 0;
    begin_block(&d, -1);
    bool ok = decode_tokens(ctx, prompt, n_batch, n_past, spec, true);
    llama_tokens history = prompt;
    llama_tokens generated;
    if (ok) common_speculative_begin(spec, history);
    flush_block(&d, "-");

    int  n_remain   = n_predict;
    bool have_carry = false;
    llama_token carry = LLAMA_TOKEN_NULL;
    int  blocks     = 0;
    while (ok && n_remain > 0 && !d.failed) {
        llama_tokens next;
        bool used = false;
        bool have_fallback = false;
        llama_token fallback = LLAMA_TOKEN_NULL;
        if (n_remain >= 3) {
            const spec_counts before = counts_of(spec);
            const int pos = n_past;
            begin_block(&d, blocks);
            auto round = common_speculative_run_round(spec, model, ctx, sampler, nullptr, params.speculative,
                                                      params.sparams, 0, n_past, n_remain, have_carry, history, carry);
            if (round.failed) {
                fprintf(stderr, "dump_draft: round %d failed: %s\n", blocks, round.error.c_str());
                ok = false;
                break;
            }
            const spec_counts after = counts_of(spec);
            const uint64_t drafted  = after.drafted - before.drafted;
            const uint64_t accepted = after.accepted - before.accepted;
            if (round.attempted || round.used_speculative || !d.pending.empty()) {
                llama_tokens target;
                if (round.used_speculative && !round.sampled_before_from_carry) target.push_back(round.sampled_before);
                if (round.used_speculative) target.insert(target.end(), round.ids.begin(), round.ids.end());
                flush_block(&d, std::to_string(accepted));
                fprintf(d.manifest, "verify\t%d\t%d\t%d\t%d\t%" PRIu64 "\t%" PRIu64 "\t%s\n", blocks, pos,
                        round.sampled_before, round.sampled_before_from_carry ? 1 : 0, drafted, accepted,
                        join(target).c_str());
                ++blocks;
            }
            if (round.sampled_before_ready && !round.used_speculative) {
                have_fallback = true;
                fallback = round.sampled_before;
            }
            if (round.used_speculative) {
                if (!round.sampled_before_from_carry) {
                    generated.push_back(round.sampled_before);
                    n_remain -= 1;
                }
                generated.insert(generated.end(), round.ids.begin(), round.ids.end());
                n_remain -= (int) round.ids.size();
                n_past += (int) round.ids.size();
                carry = round.ids.back();
                have_carry = !llama_token_is_eog(model, carry);
                if (!have_carry) n_remain = 0;
                history.push_back(round.sampled_before);
                if (round.ids.size() > 1) history.insert(history.end(), round.ids.begin(), round.ids.end() - 1);
                used = true;
            }
        }
        if (!used && have_carry) {
            next.push_back(carry);
            have_carry = false;
            used = true;
        }
        if (!used) {
            const llama_token id = have_fallback ? fallback : common_sampler_sample_legacy(sampler, ctx, nullptr);
            if (!have_fallback) common_sampler_accept(sampler, ctx, id, true);
            generated.push_back(id);
            next.push_back(id);
            n_remain -= 1;
            fprintf(d.manifest, "plain\t%d\t%d\n", n_past, id);
        }
        if (!generated.empty() && llama_token_is_eog(model, generated.back())) break;
        begin_block(&d, blocks);  // a plain step computes nothing on the draft; anything it did is kept
        ok = decode_tokens(ctx, next, n_batch, n_past, spec, false);
        history.insert(history.end(), next.begin(), next.end());
        if (!d.pending.empty()) {
            fprintf(stderr, "dump_draft: the draft computed outside a round at position %d\n", n_past);
            ok = false;
        }
        if (n_past >= n_ctx - 2) break;
    }
    report_unhandled(d);
    if (!ok) {
        fprintf(stderr, "dump_draft: decode failed — no trailer, the set is not complete\n");
        return 1;
    }
    if (d.failed) {
        fprintf(stderr, "dump_draft: a file could not be written — no trailer, the set is not complete\n");
        return 1;
    }
    // A set without the feature-to-KV graph lacks the draft's input features: the interposer did not bind.
    if (d.kv_graphs == 0) {
        fprintf(stderr, "dump_draft: no feature-to-KV graph reached the callback — no trailer, the set is not complete\n");
        return 1;
    }
    const spec_counts total = counts_of(spec);
    std::string by_pos;
    for (size_t i = 0; i < total.drafted_by_position.size(); ++i) {
        by_pos += (i ? " " : "") + std::to_string(i + 1) + ":" +
                  std::to_string(i < total.accepted_by_position.size() ? total.accepted_by_position[i] : 0) + "/" +
                  std::to_string(total.drafted_by_position[i]);
    }
    fprintf(d.manifest, "# generated\t%s\n", join(generated).c_str());
    fprintf(d.manifest, "# blocks\t%d\tdrafted\t%" PRIu64 "\taccepted\t%" PRIu64 "\tby_position\t%s\n", blocks,
            total.drafted, total.accepted, by_pos.c_str());
    fprintf(d.manifest, "# kv_graphs\t%d\n", d.kv_graphs);
    fprintf(d.manifest, "# complete\t%d\t%d\n", d.written, d.skipped);
    fclose(d.manifest);
    d.manifest = nullptr;
    if (rename(manifest_partial.c_str(), manifest_final.c_str()) != 0) {
        fprintf(stderr, "dump_draft: cannot install %s\n", manifest_final.c_str());
        return 1;
    }
    printf("dump_draft: %d blocks, drafted %" PRIu64 ", accepted %" PRIu64 " (accepted/drafted by position %s), "
           "%zu target tokens\n", blocks, total.drafted, total.accepted, by_pos.c_str(), generated.size());
    printf("dump_draft: wrote %d tensors, skipped %d, %d graph inputs (%d persistent state, %d scheduler copies, "
           "%d scratch skipped), %d integer twins, %d feature-to-KV graphs, into %s\n", d.written, d.skipped, d.inputs,
           d.state, d.copies, d.scratch, d.twins, d.kv_graphs, d.dir.c_str());
    common_sampler_free(sampler);
    common_speculative_free(spec);
    params.speculative.clear_dft();
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
