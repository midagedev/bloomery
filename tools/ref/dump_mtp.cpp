// dump_mtp — write ik_llama.cpp's MTP (NextN) draft tensors to disk as raw f32, one set per decode.
//
// The oracle of the GLM-5.3-Flash MTP port. The target decodes a fixed prompt greedily with ik's MTP
// speculative stage at one draft token (the loop this tree's llama-spec-bench runs for an MTP stage:
// prompt warmup, then common_speculative_run_round per step). ik computes the NextN block (the file's last block) on a
// context of its own over the target's model, ctx_mtp; the eval callback is on that context only, so
// every node of every graph it computes — mtp_fused, the block's attention and FFN, result_output — is
// written, and no target node is. The target runs with no callback, as ik serves it.
//
// The file format and the discipline are dump_ref's and dump_draft's (tools/ref/dump_ref.cpp's header
// is the long form): raw little-endian f32 per tensor, a logical twin for views and non-contiguous
// tensors, lossless integer twins (.i32/.i64) with `int` rows, graph inputs as `input` rows at their
// first reader, persistent leaves (the NextN block's KV cache, the indexer's key pool) as `input` rows
// and graph scratch as `skip-input` rows, BLOOMERY_REF_WRITE=1 or nothing is written, a `.partial`
// manifest renamed only after the decode returns, `# build` in the header, and the
// `# complete <written> <skipped>` trailer only when every file was written.
//
// What is its own:
//
//  - Blocks and graphs. Block b is everything ctx_mtp computes during the b-th speculative round; the
//    prompt warmup is block -1. A graph's label is the MTP op ik set on ctx_mtp for it
//    (llama_set_mtp_op_type, interposed below): `warmup` (the prompt's rows), `gen` (a draft decoded
//    from the stored hidden state, when no cached proposal exists) and `update` (the committed rows
//    after a verify; its last row's argmax is the next round's proposal). A block holds each label at
//    most once. Occurrence counters start at zero in every graph, and every file name starts with the
//    block and the label: `b<b>.<graph>.<name>.<occurrence>[.input|.logical].<type>` (`w.` for the
//    warmup). Every manifest row of a block ends with four columns — `block`, `row` (`-`), `accepted`
//    (the verify outcome of that block, `-` for the warmup) and `graph` — so a block's rows are
//    buffered and written after its verify.
//  - `draft` rows: the block's proposal, `draft block 0 token`: the argmax of the last row of
//    result_output in the last graph computed before the block's update graph (its own gen graph, or
//    the previous block's update graph). One draft token a block: the set is taken at n_max = 1.
//  - `verify` rows: one per round, `verify block pos id_last carry drafted accepted target`, as in a
//    draft set: the target position the round started at, the token the block starts from and whether
//    it came from the previous round's carry, ik's own drafted and accepted counts, and the target
//    tokens the round committed.
//  - `plain` rows: a step decoded without the draft, `plain pos token`. A plain step feeds nothing to
//    ctx_mtp, so a round after one would draft from a stale hidden state: the dump refuses it.
//
// Two checks make the host-side argmax a proposal: after every round, ik's own draft sampler
// (common_sampler_sample_speculative) over ctx_mtp's logits must name the update graph's argmax, and in
// every block the proposal must equal the target's token at the draft row exactly when ik accepted it.
// Either failing leaves the set without its trailer.
//
// The callback asks for every node of ctx_mtp's graphs, so they run under the dumped schedule: node by
// node, no fusion. Their arithmetic is the dumped schedule's and the proposals and accept counts in this
// set are that schedule's; the target's are ik's serving path.
//
// Build: tools/ref/build-dump-mtp.sh   Run: tools/ref/dump-mtp.sh (never by hand)

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

struct pending_row {
    std::string text;
    std::string graph;
};

struct dump_ctx {
    std::string                   dir;
    FILE *                        manifest = nullptr;
    std::vector<pending_row>      pending;      // this block's rows and the graph each belongs to
    int                           block = -1;   // -1: the prompt warmup
    bool                          armed = false;  // the prompt's block has begun
    std::string                   graph;        // the label of the graph being computed, "" before any
    std::set<std::string>         graphs_in_block;
    // Per graph: node name -> times emitted, the same for inputs and state, and the inputs already written.
    std::map<std::string, int>    seen;
    std::map<std::string, int>    seen_input;
    std::set<const ggml_tensor *> inputs_done;
    std::set<ggml_backend_buffer_t> node_bufs;  // buffers ctx_mtp's own nodes live in
    std::map<std::string, int>    unhandled;    // type name -> tensors skipped for it
    std::vector<uint8_t>          staging;      // device tensors land here before the write
    int                           graph_argmax = -1;  // the current graph's last result_output row
    int                           last_argmax  = -1;  // the last graph's, carried across blocks
    int                           proposal     = -1;  // this block's, taken when its update graph begins
    std::map<std::string, int>    graph_counts;
    int                           op_sets = 0;  // llama_set_mtp_op_type calls seen
    bool                          failed  = false;
    int                           written = 0;
    int                           skipped = 0;
    int                           inputs  = 0;
    int                           twins   = 0;
    int                           state   = 0;
    int                           scratch = 0;
    int                           copies  = 0;  // scheduler copies of host inputs, written as input rows
};

static std::string safe_name(const char * name) {
    std::string s(name);
    for (char & c : s) {
        if (c == '/' || c == '\\' || c == ' ') c = '_';
    }
    return s;
}

static void fail(dump_ctx * d, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "dump_mtp: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    d->failed = true;
}

static void row(dump_ctx * d, const char * fmt, ...) {
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    d->pending.push_back({ buf, d->graph });
}

static void begin_block(dump_ctx * d, int block) {
    d->block = block;
    d->graph.clear();
    d->graphs_in_block.clear();
    d->proposal = -1;
}

// ctx_mtp is about to compute a graph labelled `label`: its occurrence counters and first-reader inputs
// start over, and an update graph takes the block's proposal from the graph before it.
static void begin_graph(dump_ctx * d, const std::string & label) {
    if (!d->armed) {
        fail(d, "ctx_mtp computed a %s graph before the prompt's block began", label.c_str());
        return;
    }
    if (!d->graphs_in_block.insert(label).second) {
        fail(d, "block %d computed a second %s graph", d->block, label.c_str());
        return;
    }
    if (label == "update") {
        if (d->last_argmax < 0) {
            fail(d, "block %d's update graph has no proposal before it", d->block);
            return;
        }
        d->proposal = d->last_argmax;
    }
    d->graph = label;
    d->seen.clear();
    d->seen_input.clear();
    d->inputs_done.clear();
    d->graph_argmax = -1;
    d->graph_counts[label]++;
}

// Writes the block's buffered rows with their four trailing columns, then its draft row.
static void flush_block(dump_ctx * d, const std::string & accepted) {
    for (const auto & r : d->pending) {
        fprintf(d->manifest, "%s\t%d\t-\t%s\t%s\n", r.text.c_str(), d->block, accepted.c_str(), r.graph.c_str());
    }
    d->pending.clear();
    if (d->block >= 0 && d->proposal >= 0) {
        fprintf(d->manifest, "draft\t%d\t0\t%d\n", d->block, d->proposal);
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
        fprintf(stderr, "dump_mtp: cannot write %s\n", path.c_str());
        return false;
    }
    if (fwrite(data, size, count, f) != count) {
        fprintf(stderr, "dump_mtp: short write on %s\n", path.c_str());
        fclose(f);
        return false;
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "dump_mtp: cannot flush %s\n", path.c_str());
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

// The first index of the largest value in the last row of a contiguous [n_vocab, n_rows] logits tensor:
// the rule of ik's scalar draft sampler, which the round check holds the AVX2 one to.
static int last_row_argmax(const ggml_tensor * t, const std::vector<float> & v) {
    const int64_t n = t->ne[0];
    const int64_t r = ggml_nrows(t) - 1;
    const float * p = v.data() + r * n;
    int64_t best = 0;
    for (int64_t i = 1; i < n; ++i) {
        if (p[i] > p[best]) best = i;
    }
    return (int) best;
}

static bool dump_one(dump_ctx * d, const ggml_tensor * t, bool input) {
    const char * name       = t->name[0] ? t->name : "(unnamed)";
    const int    occurrence = (input ? d->seen_input : d->seen)[name]++;
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

    const std::string prefix = (d->block < 0 ? std::string("w.") : "b" + std::to_string(d->block) + ".") + d->graph + ".";
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
    if (!input && strcmp(name, "result_output") == 0) {
        if (t->type != GGML_TYPE_F32 || !contig) {
            fail(d, "block %d %s: result_output is %s%s, not contiguous f32 logits", d->block, d->graph.c_str(),
                 ggml_type_name(t->type), contig ? "" : " (non-contiguous)");
            return true;
        }
        d->graph_argmax = last_row_argmax(t, out);
        d->last_argmax  = d->graph_argmax;
    }
    return true;
}

// A leaf the host does not fill that is not a weight: ctx_mtp's persistent state (the NextN block's KV
// cache, the indexer's key pool) or graph scratch, told apart as in dump_ref.
static bool is_state_leaf(const ggml_tensor * s) {
    return s && s->op == GGML_OP_NONE && !(s->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) &&
           s->buffer && ggml_backend_buffer_get_usage(s->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static bool dump_state(dump_ctx * d, const ggml_tensor * s) {
    const char * name = s->name[0] ? s->name : "(unnamed)";
    if (d->node_bufs.empty()) {
        fprintf(stderr, "dump_mtp: leaf %s is read before any graph-allocated node — cannot tell state from scratch\n",
                name);
        return false;
    }
    // A scheduler copy (`<backend>#<source>#<n>`) of a host input read on another backend is written as an
    // input row under its own name; the copy runs when its split starts, so its bytes are its source's.
    const bool sched_copy = strchr(name, '#') != nullptr;
    if (d->node_bufs.count(s->buffer) && !sched_copy) {
        row(d, "skip-input\t%s\t%d\t%s\tgraph-scratch", name, d->seen_input[name]++, ggml_type_name(s->type));
        d->scratch++;
        return true;
    }
    const int inputs = d->inputs;
    if (!dump_one(d, s, true)) return false;
    (sched_copy ? d->copies : d->state) += d->inputs - inputs;
    return true;
}

static dump_ctx * g_dump = nullptr;
static enum llama_mtp_op_type g_op = MTP_OP_NONE;

// The label of each ctx_mtp graph is the op ik sets on the context before its decode. libcommon (linked
// into this binary) calls this definition; it records the op and forwards to libllama's.
extern "C" void llama_set_mtp_op_type(struct llama_context * ctx, enum llama_mtp_op_type mtp_op_type) {
    using fn_t = void (*)(struct llama_context *, enum llama_mtp_op_type);
    static fn_t real = (fn_t) dlsym(RTLD_NEXT, "llama_set_mtp_op_type");
    if (!real) {
        fprintf(stderr, "dump_mtp: cannot find llama_set_mtp_op_type in libllama\n");
        abort();
    }
    g_op = mtp_op_type;
    if (g_dump) g_dump->op_sets++;
    real(ctx, mtp_op_type);
}

static const char * op_label(enum llama_mtp_op_type op) {
    switch (op) {
        case MTP_OP_WARMUP:          return "warmup";
        case MTP_OP_UPDATE_ACCEPTED: return "update";
        case MTP_OP_DRAFT_GEN:       return "gen";
        default:                     return nullptr;
    }
}

// Every graph llama_decode computes passes here (libllama calls it; -rdynamic binds the call to this
// definition). A graph that holds inp_mtp_states is ctx_mtp's NextN graph: it begins under the op last
// set. Everything else passes through untouched.
extern "C" enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    using fn_t = enum ggml_status (*)(ggml_backend_sched_t, struct ggml_cgraph *);
    static fn_t real = (fn_t) dlsym(RTLD_NEXT, "ggml_backend_sched_graph_compute_async");
    if (!real) {
        fprintf(stderr, "dump_mtp: cannot find ggml_backend_sched_graph_compute_async in libggml\n");
        abort();
    }
    if (g_dump && !g_dump->failed && ggml_graph_get_tensor(graph, "inp_mtp_states")) {
        const char * label = op_label(g_op);
        if (!label) {
            fail(g_dump, "an MTP graph in block %d computed under op %d, not warmup, update or gen", g_dump->block,
                 (int) g_op);
        } else {
            begin_graph(g_dump, label);
        }
    }
    return real(sched, graph);
}

// The ask half: the node's inputs and state leaves, each at its first reader in this graph, read before
// the scheduler computes the range the node ends.
static void write_sources(dump_ctx * d, const ggml_tensor * t) {
    if (!t->view_src && t->buffer) d->node_bufs.insert(t->buffer);
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        const ggml_tensor * s = t->src[j];
        if (s && s->op == GGML_OP_NONE && (s->flags & GGML_TENSOR_FLAG_INPUT) &&
            !(s->flags & GGML_TENSOR_FLAG_OUTPUT) && d->inputs_done.insert(s).second && !dump_one(d, s, true)) {
            d->failed = true;
            return;
        }
        if (is_state_leaf(s) && d->inputs_done.insert(s).second && !dump_state(d, s)) {
            d->failed = true;
            return;
        }
    }
}

// ctx_mtp's graphs: every node asked for and written.
static int on_tensor(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (dump_ctx *) user_data;
    if (d->failed) {
        return ask ? 1 : 0;
    }
    if (d->graph.empty()) {
        fail(d, "ctx_mtp computed %s outside any labelled MTP graph (block %d)", t->name, d->block);
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
    fprintf(stderr, "dump_mtp: %d tensors skipped as unhandled: %s\n", total, types.empty() ? "none" : types.c_str());
}

static bool read_token_file(const std::string & path, long long count, std::vector<llama_token> & ids) {
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "dump_mtp: cannot read %s\n", path.c_str());
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
            fprintf(stderr, "dump_mtp: %s:%lld is not a token id: '%s'\n", path.c_str(), lines, line.c_str());
            return false;
        }
        ids.push_back((llama_token) v);
    }
    if ((long long) ids.size() < count) {
        fprintf(stderr, "dump_mtp: %s holds %zu token ids and --tokens-count asks for %lld\n", path.c_str(),
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
            // Every row an output, as llama-spec-bench does: the MTP warmup reads every row's hidden state.
            common_batch_add(batch, toks[i + k], n_past + k, { 0 }, true);
        }
        const int rc = llama_decode(ctx, batch);
        const bool ok = rc == 0 && (!warmup || common_speculative_on_target_seq_batch(spec, ctx, batch, 0, true) == 0);
        llama_batch_free(batch);
        if (!ok) {
            fprintf(stderr, "dump_mtp: %s decode failed at position %d (llama_decode %d)\n",
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

    // Ours: --tokens-file, --tokens-count, --expect-arch. The rest is gpt_params.
    std::string tokens_file;
    long long   tokens_count = 0;
    std::string expect_arch;
    std::vector<char *> passthrough;
    passthrough.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokens-file") == 0 && i + 1 < argc) {
            tokens_file = argv[++i];
        } else if (strcmp(argv[i], "--tokens-count") == 0 && i + 1 < argc) {
            tokens_count = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--expect-arch") == 0 && i + 1 < argc) {
            expect_arch = argv[++i];
        } else {
            passthrough.push_back(argv[i]);
        }
    }
    if (tokens_file.empty() || tokens_count <= 0) {
        fprintf(stderr, "dump_mtp: --tokens-file <ids> --tokens-count <n> are required; this tool does not tokenize\n");
        return 2;
    }
    if (expect_arch.empty()) {
        fprintf(stderr, "dump_mtp: --expect-arch is required\n");
        return 2;
    }
    std::vector<llama_token> prompt;
    if (!read_token_file(tokens_file, tokens_count, prompt)) return 2;

    gpt_params params;
    if (!gpt_params_parse((int) passthrough.size(), passthrough.data(), params)) {
        fprintf(stderr, "dump_mtp: bad arguments\n");
        return 2;
    }
    // The NextN block is the target file's own: no -md, an MTP stage, one draft token a round (the
    // draft rows and the accept check are a one-token rule).
    if (!params.speculative.model.empty() || !params.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP) ||
        params.speculative.get_max_stage_n_max() != 1) {
        fprintf(stderr, "dump_mtp: needs --spec-type mtp:n_max=1 and no -md (the target file carries the NextN block)\n");
        return 2;
    }
    const int n_predict = params.n_predict;
    if (n_predict <= 0) {
        fprintf(stderr, "dump_mtp: -n %d is not positive\n", n_predict);
        return 2;
    }

    // This binary's side effect is an oracle set: without BLOOMERY_REF_WRITE=1 it writes nothing
    // (dump_ref's rule), and it never picks its own directory. Exactly 1 writes, unset or 0 is the
    // refusal below, and any other value is refused by name.
    const char * ref_dir   = getenv("BLOOMERY_REF_DIR");
    const char * write_env = getenv("BLOOMERY_REF_WRITE");
    if (write_env && strcmp(write_env, "0") != 0 && strcmp(write_env, "1") != 0) {
        fprintf(stderr, "dump_mtp: BLOOMERY_REF_WRITE is 1 (write the MTP oracle set) or 0 or unset (refuse), got '%s'\n",
                write_env);
        return 64;
    }
    if (!write_env || strcmp(write_env, "1") != 0 || !ref_dir || !*ref_dir) {
        fprintf(stderr,
                "dump_mtp: refusing to run -- this tool writes the MTP oracle set. It is not a general ik\n"
                "          harness; tools/ref/dump-mtp.sh sets BLOOMERY_REF_WRITE=1 and BLOOMERY_REF_DIR to a\n"
                "          staging directory.\n");
        return 3;
    }
    // IK_PREGATE adds a router read per layer to the target's graph (the tree's instrumentation).
    if (getenv("IK_PREGATE")) {
        fprintf(stderr, "dump_mtp: IK_PREGATE is set; it changes the target's graph -- unset it\n");
        return 2;
    }

    const std::string got_arch = file_arch(params.model);
    if (got_arch != expect_arch) {
        fprintf(stderr, "dump_mtp: the target %s is a '%s' model, expected '%s' -- not loading it\n",
                params.model.c_str(), got_arch.c_str(), expect_arch.c_str());
        return 2;
    }

    dump_ctx d;
    d.dir = ref_dir;
    mkdir(d.dir.c_str(), 0755);
    const std::string manifest_final   = d.dir + "/MANIFEST.tsv";
    const std::string manifest_partial = manifest_final + ".partial";
    d.manifest = fopen(manifest_partial.c_str(), "w");
    if (!d.manifest) {
        fprintf(stderr, "dump_mtp: cannot write %s\n", manifest_partial.c_str());
        return 1;
    }

    // Greedy target, fixed seed: the set is one decode, not a sample.
    params.sparams.temp = 0.0f;
    params.warmup       = false;
    params.cb_eval      = nullptr;  // the target is not dumped
    common_speculative_prepare_startup(params);

    llama_backend_init();
    llama_numa_init(params.numa);
    llama_init_result init = llama_init_from_gpt_params(params);
    llama_model *   model = init.model;
    llama_context * ctx   = init.context;
    if (!model || !ctx) {
        fprintf(stderr, "dump_mtp: failed to load the target\n");
        return 1;
    }
    if (!common_speculative_finalize_startup(params, model) || !params.has_mtp) {
        fprintf(stderr, "dump_mtp: the MTP stage did not prepare (does the file carry its NextN block?)\n");
        return 1;
    }
    // ctx_mtp is created from cparams_dft by try_init: the callback goes there and nowhere else.
    params.speculative.cparams_dft.cb_eval           = on_tensor;
    params.speculative.cparams_dft.cb_eval_user_data = &d;
    g_dump = &d;
    common_speculative * spec = nullptr;
    if (!common_speculative_is_compat(ctx) ||
        common_speculative_try_init(params.speculative, ctx, &spec) != COMMON_SPECULATIVE_INIT_READY || !spec) {
        fprintf(stderr, "dump_mtp: the MTP speculative context did not initialize\n");
        return 1;
    }
    llama_context * ctx_mtp = common_speculative_get_companion_ctx(spec);
    if (!ctx_mtp) {
        fprintf(stderr, "dump_mtp: the speculative stage holds no MTP context\n");
        return 1;
    }
    common_sampler * sampler = common_sampler_init(model, params.sparams);
    if (!sampler) {
        fprintf(stderr, "dump_mtp: failed to initialize the sampler\n");
        return 1;
    }
    const int n_ctx   = (int) llama_n_ctx(ctx);
    const int n_batch = params.n_batch;
    if ((int) prompt.size() + n_predict >= n_ctx - 2) {
        fprintf(stderr, "dump_mtp: %zu + %d tokens do not fit in n_ctx %d\n", prompt.size(), n_predict, n_ctx);
        return 2;
    }
    if ((int) prompt.size() > n_batch) {
        // One warmup graph: a prompt in two batches would be two warmup graphs in block -1.
        fprintf(stderr, "dump_mtp: %zu prompt tokens exceed -b %d\n", prompt.size(), n_batch);
        return 2;
    }

    char arch[128];
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0) snprintf(arch, sizeof arch, "unknown");
    fprintf(d.manifest, "# dump_mtp — ik_llama.cpp MTP (NextN) draft tensors, raw f32, little-endian\n");
    fprintf(d.manifest, "# model\t%s\n", params.model.c_str());
    if (const char * b = getenv("BLOOMERY_REF_BUILD")) fprintf(d.manifest, "# build\t%s\n", b);
    fprintf(d.manifest, "# arch\t%s\n", arch);
    fprintf(d.manifest, "# model_file\t%s\n", basename_of(params.model));
    fprintf(d.manifest, "# tokens\t%s\n", join(prompt).c_str());
    fprintf(d.manifest, "# tokens_file\t%s\n", tokens_file.c_str());
    const char * sha = getenv("BLOOMERY_REF_TOKENS_SHA256");
    fprintf(d.manifest, "# tokens_file_sha256\t%s\n", sha && *sha ? sha : "unknown");
    fprintf(d.manifest, "# tokens_count\t%zu\n", prompt.size());
    fprintf(d.manifest, "# n_predict\t%d\n", n_predict);
    fprintf(d.manifest, "# spec\t%s\n", common_speculative_stage_chain_to_str(params.speculative).c_str());
    fprintf(d.manifest, "# flags\t%s\n", flags.c_str());
    fprintf(d.manifest, "# schedule\tctx_mtp: every node asked for and computed alone (no fusion); "
                        "target: no callback, ik's serving schedule\n");
    fprintf(d.manifest, "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1\tblock\trow\taccepted\tgraph\n");
    fprintf(d.manifest, "# int\tname\toccurrence\tof\ttype\ttwin\tlayout\tcount\tbytes\tsum\tabsmax\tfile\tblock\trow\taccepted\tgraph\n");
    fprintf(d.manifest, "# input\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1\tblock\trow\taccepted\tgraph\n");
    fprintf(d.manifest, "# draft\tblock\trow\ttoken\n");
    fprintf(d.manifest, "# verify\tblock\tpos\tid_last\tcarry\tdrafted\taccepted\ttarget\n");
    fprintf(d.manifest, "# plain\tpos\ttoken\n");

    // The loop is this tree's llama-spec-bench's under an MTP stage (spec_bench_run_attempt), with the
    // prompt given as ids: the target's embeddings on for the prompt, the stage begun with no history,
    // the last prompt row's hidden state captured, the embeddings off again; rounds draft from no history.
    common_sampler_reset(sampler);
    common_speculative_clear_sequence_kv(spec, ctx, 0);
    llama_set_embeddings(ctx, true);
    for (llama_token t : prompt) common_sampler_accept(sampler, ctx, t, false);
    int n_past = 0;
    begin_block(&d, -1);
    d.armed = true;
    bool ok = decode_tokens(ctx, prompt, n_batch, n_past, spec, true);
    const llama_tokens no_history;
    llama_tokens generated;
    if (ok) {
        common_speculative_begin(spec, no_history);
        // One prompt batch (checked above), so its last row is output index P - 1.
        if (!common_speculative_capture_output_hidden(spec, ctx, (int32_t) prompt.size() - 1, 0,
                                                      (llama_pos) prompt.size() - 1)) {
            fprintf(stderr, "dump_mtp: failed to capture the last prompt row's hidden state\n");
            ok = false;
        }
    }
    llama_set_embeddings(ctx, false);
    if (ok && d.graph_counts["warmup"] != 1) {
        fail(&d, "the prompt ran %d warmup graphs on ctx_mtp, not 1", d.graph_counts["warmup"]);
    }
    flush_block(&d, "-");

    int  n_remain   = n_predict;
    bool have_carry = false;
    llama_token carry = LLAMA_TOKEN_NULL;
    int  blocks     = 0;
    int  plain_at   = -1;  // the position of a plain step's token: ctx_mtp never saw its hidden state
    while (ok && n_remain > 0 && !d.failed) {
        llama_tokens next;
        bool used = false;
        bool have_fallback = false;
        llama_token fallback = LLAMA_TOKEN_NULL;
        if (n_remain >= 3) {
            if (plain_at >= 0) {
                fail(&d, "a round after the plain step at position %d: ctx_mtp holds no hidden state for it", plain_at);
                break;
            }
            const spec_counts before = counts_of(spec);
            const int pos = n_past;
            begin_block(&d, blocks);
            auto round = common_speculative_run_round(spec, model, ctx, sampler, nullptr, params.speculative,
                                                      params.sparams, 0, n_past, n_remain, have_carry, no_history, carry);
            if (round.failed) {
                fprintf(stderr, "dump_mtp: round %d failed: %s\n", blocks, round.error.c_str());
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
                if (round.used_speculative) {
                    // The target's token at the draft row: the first committed after the round's start token.
                    const size_t skip = round.sampled_before_from_carry ? 0 : 1;
                    if (!d.graphs_in_block.count("update") || d.proposal < 0) {
                        fail(&d, "block %d committed without an MTP update graph", blocks);
                    } else if (drafted != 1 || accepted > 1 || target.size() <= skip) {
                        fail(&d, "block %d: drafted %" PRIu64 ", accepted %" PRIu64 ", %zu target tokens -- not a "
                             "one-token round", blocks, drafted, accepted, target.size());
                    } else if ((d.proposal == target[skip]) != (accepted == 1)) {
                        fail(&d, "block %d: proposal %d, target token %d, ik accepted %" PRIu64
                             " -- the dumped argmax is not ik's proposal", blocks, d.proposal, target[skip], accepted);
                    } else {
                        const llama_token sampled = common_sampler_sample_speculative(nullptr, ctx_mtp, -1, nullptr);
                        if (sampled != d.graph_argmax) {
                            fail(&d, "block %d: ik's draft sampler names %d, the update graph's result_output argmax %d",
                                 blocks, sampled, d.graph_argmax);
                        }
                    }
                } else if (!d.pending.empty()) {
                    fail(&d, "block %d computed on ctx_mtp and committed nothing", blocks);
                }
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
        begin_block(&d, blocks);  // a plain step computes nothing on ctx_mtp; anything it did is kept
        if (!next.empty() && plain_at < 0) plain_at = n_past;
        ok = decode_tokens(ctx, next, n_batch, n_past, spec, false);
        if (!d.pending.empty()) {
            fprintf(stderr, "dump_mtp: ctx_mtp computed outside a round at position %d\n", n_past);
            ok = false;
        }
        if (n_past >= n_ctx - 2) break;
    }
    report_unhandled(d);
    if (!ok) {
        fprintf(stderr, "dump_mtp: decode failed — no trailer, the set is not complete\n");
        return 1;
    }
    if (d.failed) {
        fprintf(stderr, "dump_mtp: the set failed a check above or a file could not be written — no trailer, "
                        "the set is not complete\n");
        return 1;
    }
    // A set whose graphs were never labelled did not bind the interposers.
    if (d.op_sets == 0 || d.graph_counts["update"] == 0) {
        fprintf(stderr, "dump_mtp: %d MTP op sets and %d update graphs reached this binary — no trailer, the set "
                        "is not complete\n", d.op_sets, d.graph_counts["update"]);
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
    fprintf(d.manifest, "# graphs\twarmup\t%d\tgen\t%d\tupdate\t%d\n", d.graph_counts["warmup"], d.graph_counts["gen"],
            d.graph_counts["update"]);
    fprintf(d.manifest, "# complete\t%d\t%d\n", d.written, d.skipped);
    fclose(d.manifest);
    d.manifest = nullptr;
    if (rename(manifest_partial.c_str(), manifest_final.c_str()) != 0) {
        fprintf(stderr, "dump_mtp: cannot install %s\n", manifest_final.c_str());
        return 1;
    }
    printf("dump_mtp: %d blocks, drafted %" PRIu64 ", accepted %" PRIu64 " (accepted/drafted by position %s), "
           "%zu target tokens\n", blocks, total.drafted, total.accepted, by_pos.c_str(), generated.size());
    printf("dump_mtp: wrote %d tensors, skipped %d, %d graph inputs (%d persistent state, %d scheduler copies, "
           "%d scratch skipped), %d integer twins, graphs warmup %d gen %d update %d, into %s\n", d.written, d.skipped,
           d.inputs, d.state, d.copies, d.scratch, d.twins, d.graph_counts["warmup"], d.graph_counts["gen"],
           d.graph_counts["update"], d.dir.c_str());
    common_sampler_free(sampler);
    common_speculative_free(spec);
    params.speculative.clear_dft();
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
