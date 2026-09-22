// dump_ref — write ik_llama.cpp's intermediate forward-pass tensors to disk as raw f32.
//
// This is the oracle boundary for stage 1. Every round from 1-2 on compares its own
// forward pass against files this tool writes; the gate reads $BLOOMERY_DATA/ref/.
//
// Why this exists when llama-eval-callback already prints tensors: it prints a TEXT
// SUMMARY — three values, an ellipsis, three values, and a sum (measured on the box
// 2026-09-19). That cannot support a 1e-3 elementwise gate. This writes every element.
//
// Two decisions that the gate depends on, made here on purpose:
//
//  1. Tokens come in as integers on the command line, never as text. If this tool
//     tokenized a prompt and the Rust side tokenized the same prompt, a tokenizer
//     difference would show up as a numeric difference in every downstream tensor and
//     look like a kernel bug. The tokenizer is not what stage 1 is testing.
//  2. A tensor name can appear more than once in one graph. Writing by name alone would
//     silently keep only the last one, and the gate would compare the wrong tensor while
//     staying green. Every file carries its occurrence index, and the manifest records it.
//
// Quantized tensors are skipped (there is no f32 to write); the manifest records the skip
// so a missing file is never mistaken for a tensor that did not run. A type the conversion
// below has no arm for is skipped the same way, as `unhandled`, and the run ends with one
// stderr line counting those skips by type — a new model's graph shows at once whether it
// carries a type this tool drops. A node the graph builder never named is written under the
// name `(unnamed)`: an empty name would make its file `.<occurrence>.f32`, a hidden file.
//
// A VIEW / non-contiguous tensor's main file is the FLAT memory at the tensor's own data
// pointer, nelements values read contiguously — not the logical tensor (a top-6 ids view
// of an argsort holds the parent's head, not each token's top-6). Every such tensor ALSO
// gets `<name>.<occurrence>.logical.f32`: the logical elements in ggml index order (ne0
// fastest), gathered through the tensor's own nb strides from its own data pointer, same
// per-type f32 conversion as the main file. Manifest rows carry four trailing columns:
// `contig` (ggml_is_contiguous), `logical` (1 = a logical twin exists), `src0`/`src1`
// (names of the producing tensors, `-` if none) so chains are readable from the manifest
// instead of probes. Sets written before these columns exist stay readable: the loader
// accepts both row widths.
//
// An integer tensor (i8/i16/i32/i64) is exact in its f32 file only up to 2^24 in magnitude,
// and a row id or an index above that cannot back an exact-match gate from there. So every
// integer tensor also gets a lossless twin per file: `<name>.<occurrence>.i32` (i8/i16/i32,
// widened) or `.i64`, and `.logical.i32`/`.logical.i64` beside a logical twin — raw
// little-endian, the same element order as the f32 file it twins. Each twin file has an
// `int` manifest row: the twinned tensor's name and occurrence, whether it is a `tensor` or
// an `input`, the ggml type, the twin type, `flat` or `logical`, the element count, the byte
// count, the exact sum (wrapping 64-bit, printed signed), the largest magnitude, the file.
//
// Graph inputs — the leaves the host fills before the graph runs: token ids, positions,
// masks, engram row ids — are not nodes, so the scheduler never hands them to this callback.
// Each is written once, when the first node that reads it is about to run (the ask call
// comes before that node is computed, and nothing can have overwritten an input before its
// first reader). Its files are `<name>.<occurrence>.input.f32` (+ twins), its manifest row is
// an `input` row with the tensor columns, and its occurrence index is a namespace of its own.
// Scheduler copies also carry the INPUT flag, but OUTPUT as well; they are other tensors' data
// and are left alone.
//
// The tensor readers parse `tensor` rows only, so `int`, `input` and `skip-input` rows are
// invisible to them, and the `# complete <written> <skipped>` trailer counts nodes only.
//
// Build: tools/ref/build-dump.sh   Run: $BLOOMERY_DATA/bin/dump_ref -m <gguf> --tokens 1,2,3

#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <sys/stat.h>
#include <vector>

struct dump_ctx {
    std::string                   dir;
    FILE *                        manifest = nullptr;
    std::map<std::string, int>    seen;         // node name -> times emitted, for the occurrence index
    std::map<std::string, int>    seen_input;   // the same for graph inputs, a namespace of their own
    std::set<const ggml_tensor *> inputs_done;  // an input is written once, at its first reader
    std::map<std::string, int>    unhandled;    // type name -> tensors skipped for it
    std::vector<uint8_t>          staging;      // device tensors land here before the write
    bool                          failed  = false;  // a file could not be written: no trailer
    int                           written = 0;
    int                           skipped = 0;
    int                           inputs  = 0;
    int                           twins   = 0;
};

// Tensor names carry '/' and '.' in some graphs; keep the file name a single path element.
static std::string safe_name(const char * name) {
    std::string s(name);
    for (char & c : s) {
        if (c == '/' || c == '\\' || c == ' ') c = '_';
    }
    return s;
}

enum elem_kind { ELEM_UNHANDLED, ELEM_FLOAT, ELEM_INT };

// The one per-type conversion every pass reads through — flat file, logical twin, integer
// twins, inputs — so a type handled in one pass cannot fall through to a silent zero or a
// missing twin in another. `p` is one element. f16/bf16 widen to f32 exactly; an integer comes
// back exact in `i` and cast in `f`, which is exact up to 2^24 in magnitude.
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

// One layout of a tensor's elements through read_elem: flat (nelements values read
// contiguously from byte 0 — the main file) or logical (ggml index order, ne0 fastest,
// through the tensor's own nb strides). `src` maps byte 0 to t->data (host pointer, or the
// staging copy the backend filled from t->data), and ggml_nbytes covers the furthest strided
// element, so both walks stay inside that buffer. `ints` is filled for an integer type only.
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
        fprintf(stderr, "dump_ref: cannot write %s\n", path.c_str());
        return false;
    }
    if (fwrite(data, size, count, f) != count) {
        fprintf(stderr, "dump_ref: short write on %s\n", path.c_str());
        fclose(f);
        return false;
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "dump_ref: cannot flush %s\n", path.c_str());
        return false;
    }
    return true;
}

// An integer tensor's lossless twin for one layout, and its `int` row. i64 stays 64-bit;
// every narrower integer widens to i32, which is exact.
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
    fprintf(d->manifest, "int\t%s\t%d\t%s\t%s\t%s\t%s\t%zu\t%zu\t%" PRId64 "\t%" PRIu64 "\t%s\n",
            name, occurrence, input ? "input" : "tensor", ggml_type_name(t->type), twin,
            logical ? "logical" : "flat", ints.size(), ints.size() * (wide ? 8 : 4),
            (int64_t) sum, absmax, file.c_str());
    d->twins++;
    return true;
}

// Write one node (`input` false) or one graph input: the f32 file, the logical twin of a view
// or non-contiguous tensor, the integer twins, and the manifest rows. False only when a file
// cannot be written.
static bool dump_one(dump_ctx * d, const ggml_tensor * t, bool input) {
    const char * name       = t->name[0] ? t->name : "(unnamed)";
    const int    occurrence = (input ? d->seen_input : d->seen)[name]++;
    const char * skip_kind  = input ? "skip-input" : "skip";

    if (ggml_is_quantized(t->type)) {
        fprintf(d->manifest, "%s\t%s\t%d\t%s\tquantized\n", skip_kind, name, occurrence, ggml_type_name(t->type));
        if (!input) d->skipped++;
        return true;
    }
    // Classify through the same switch every element goes through, before reading anything.
    const uint8_t zero[8] = {};
    float   f0;
    int64_t i0;
    const elem_kind kind = read_elem(t->type, zero, f0, i0);
    if (kind == ELEM_UNHANDLED) {
        fprintf(d->manifest, "%s\t%s\t%d\t%s\tunhandled\n", skip_kind, name, occurrence, ggml_type_name(t->type));
        d->unhandled[ggml_type_name(t->type)]++;
        if (!input) d->skipped++;
        return true;
    }
    if (!t->buffer || !t->data) {
        fprintf(d->manifest, "%s\t%s\t%d\t%s\tunallocated\n", skip_kind, name, occurrence, ggml_type_name(t->type));
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

    // Convert to f32 so the consumer has exactly one format to read. f16/bf16 intermediates
    // exist in some graphs and a gate that had to branch on dtype would be a second thing
    // to get wrong.
    const size_t n = (size_t) ggml_nelements(t);
    std::vector<float>   out(n);
    std::vector<int64_t> ints(kind == ELEM_INT ? n : 0);
    gather(t, src, false, out, ints);
    double sum = 0.0;
    for (float v : out) sum += v;

    const std::string stem = safe_name(name) + "." + std::to_string(occurrence) + (input ? ".input" : "");
    if (!write_raw(d->dir + "/" + stem + ".f32", out.data(), sizeof(float), n)) return false;

    // The logical twin: for every view or non-contiguous tensor, the elements in ggml index
    // order, gathered through this tensor's own nb strides from its own data pointer.
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

    // src names for the chain columns: the producer's name (`(unnamed)`, as its own row is
    // named, when the builder gave it none), `-` when there is no producer.
    auto src_name = [](const struct ggml_tensor * s) -> const char * {
        return s ? (s->name[0] ? s->name : "(unnamed)") : "-";
    };

    // ne[] in ggml order, verbatim — bloomery's tensors carry the same order by decision
    // (docs/plan.md), so a shape mismatch in the gate is a real mismatch, not a convention.
    // The four trailing columns are v2 additions; the first eleven fields are byte-for-byte
    // what earlier sets carry (the flat `sum` included: it is the flat read's sum).
    fprintf(d->manifest, "%s\t%s\t%d\t%s\t%lld\t%lld\t%lld\t%lld\t%zu\t%.6f\t%s\t%d\t%d\t%s\t%s\n",
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
    return true;
}

static int on_tensor(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (dump_ctx *) user_data;
    if (d->failed) {
        // Asked: yes, so the scheduler computes this node and calls back, and that call
        // stops the graph. The set is not installed either way: main writes no trailer.
        return ask ? 1 : 0;
    }
    if (ask) {
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * s = t->src[j];
            if (s && s->op == GGML_OP_NONE && (s->flags & GGML_TENSOR_FLAG_INPUT) &&
                !(s->flags & GGML_TENSOR_FLAG_OUTPUT) && d->inputs_done.insert(s).second &&
                !dump_one(d, s, true)) {
                d->failed = true;
                break;
            }
        }
        return 1;  // yes, we want every node
    }
    if (!dump_one(d, t, false)) {
        d->failed = true;
        return 0;  // stop the graph: a partial reference set is worse than none
    }
    return 1;
}

// `N tensors skipped as unhandled: <type> x<count>, ...` on stderr, printed even for N = 0 so
// that the absence of the line is never read as "nothing was dropped".
static void report_unhandled(const dump_ctx & d) {
    int total = 0;
    std::string types;
    for (const auto & it : d.unhandled) {
        total += it.second;
        types += (types.empty() ? "" : ", ") + it.first + " x" + std::to_string(it.second);
    }
    fprintf(stderr, "dump_ref: %d tensors skipped as unhandled: %s\n", total, types.empty() ? "none" : types.c_str());
}

int main(int argc, char ** argv) {
    // --tokens and --expect-arch are ours; everything else is gpt_params. Pull them out
    // before the parser sees them.
    std::vector<llama_token> tokens;
    std::string              expect_arch;
    std::vector<char *>      passthrough;
    passthrough.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            char * list = argv[++i];
            for (char * p = strtok(list, ","); p; p = strtok(nullptr, ",")) {
                tokens.push_back((llama_token) atoi(p));
            }
        } else if (strcmp(argv[i], "--expect-arch") == 0 && i + 1 < argc) {
            expect_arch = argv[++i];
        } else {
            passthrough.push_back(argv[i]);
        }
    }
    if (tokens.empty()) {
        fprintf(stderr, "dump_ref: --tokens <id,id,...> is required; this tool does not tokenize\n");
        return 2;
    }
    if (expect_arch.empty()) {
        fprintf(stderr, "dump_ref: --expect-arch <general.architecture> is required; a set is one architecture's\n");
        return 2;
    }

    gpt_params params;
    if (!gpt_params_parse((int) passthrough.size(), passthrough.data(), params)) {
        fprintf(stderr, "dump_ref: bad arguments\n");
        return 2;
    }

    // This binary's side effect IS the oracle. Running it for any other reason -- under a
    // debugger to breakpoint into an ik kernel, say -- overwrites 1155 files that every
    // gate in this repo reads. That happened on 2026-09-19: a round ran it about ten times
    // under gdb to inspect the flash-attention path, each run was killed at a breakpoint,
    // and what survived was a half-written set with a zero-byte manifest. The spec said
    // "never run dump_ref yourself" and the spec was not the right place for that rule --
    // the round was not reaching for the dumper, it was reaching for the one binary that
    // links ik and takes --tokens. So the rule lives here instead, where it cannot be
    // missed: without BLOOMERY_REF_WRITE=1 this tool refuses to write anything.
    if (!getenv("BLOOMERY_REF_WRITE")) {
        fprintf(stderr,
                "dump_ref: refusing to run -- this tool OVERWRITES the oracle reference set,\n"
                "          which every gate in this repo reads. It is not a general ik harness.\n"
                "          The lead regenerates the set with `just dump-ref`, which sets\n"
                "          BLOOMERY_REF_WRITE=1 and stages the output so a failed run keeps the\n"
                "          old set. If you want to inspect ik's kernels under a debugger, use\n"
                "          llama-cli or llama-eval-callback -- not this.\n");
        return 3;
    }

    // The caller's profile names the set, and it also decides whether the dump runs under the
    // machine lease. A file of another architecture is refused here, from its header alone,
    // before the loader pages in a single weight.
    {
        gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        gguf_context * g = gguf_init_from_file(params.model.c_str(), gp);
        if (!g) {
            fprintf(stderr, "dump_ref: cannot read the GGUF header of %s\n", params.model.c_str());
            return 2;
        }
        const int         k         = gguf_find_key(g, "general.architecture");
        const std::string file_arch = k >= 0 ? gguf_get_val_str(g, k) : "unknown";
        gguf_free(g);
        if (file_arch != expect_arch) {
            fprintf(stderr, "dump_ref: %s is a %s model and this dump is for %s -- not loading it\n",
                    params.model.c_str(), file_arch.c_str(), expect_arch.c_str());
            return 2;
        }
    }

    const char * data_dir = getenv("BLOOMERY_DATA");
    const char * ref_dir  = getenv("BLOOMERY_REF_DIR");
    dump_ctx d;
    d.dir = ref_dir ? std::string(ref_dir)
                    : std::string(data_dir ? data_dir : "/root/bloomery-data") + "/ref";
    mkdir(d.dir.c_str(), 0755);

    // The manifest is written to a .partial name and renamed only after the decode
    // returns. An interrupted run therefore leaves the previous MANIFEST.tsv alone
    // instead of truncating it to zero bytes, which is what turned a redundant re-run
    // into "the oracle is gone".
    const std::string manifest_final   = d.dir + "/MANIFEST.tsv";
    const std::string manifest_partial = manifest_final + ".partial";
    d.manifest = fopen(manifest_partial.c_str(), "w");
    if (!d.manifest) {
        fprintf(stderr, "dump_ref: cannot write %s\n", manifest_partial.c_str());
        return 1;
    }
    fprintf(d.manifest, "# dump_ref — ik_llama.cpp intermediate tensors, raw f32, little-endian\n");
    fprintf(d.manifest, "# model\t%s\n", params.model.c_str());
    if (const char * b = getenv("BLOOMERY_REF_BUILD")) {
        // Which ik build produced this set. The reference IS that build's output, so a
        // set whose build is unknown cannot be reasoned about after the fact.
        fprintf(d.manifest, "# build\t%s\n", b);
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = on_tensor;
    params.cb_eval_user_data = &d;
    params.warmup            = false;

    llama_init_result init = llama_init_from_gpt_params(params);
    if (!init.model || !init.context) {
        fprintf(stderr, "dump_ref: failed to load the model\n");
        return 1;
    }

    // Which model this is, from the file rather than from the path: the architecture the
    // loader dispatched on, and the file's own name, which survives a moved directory.
    char arch[128];
    if (llama_model_meta_val_str(init.model, "general.architecture", arch, sizeof(arch)) < 0) {
        snprintf(arch, sizeof(arch), "unknown");
    }
    const char * slash = strrchr(params.model.c_str(), '/');
    fprintf(d.manifest, "# arch\t%s\n", arch);
    fprintf(d.manifest, "# model_file\t%s\n", slash ? slash + 1 : params.model.c_str());
    fprintf(d.manifest, "# tokens\t");
    for (size_t i = 0; i < tokens.size(); ++i) fprintf(d.manifest, "%s%d", i ? "," : "", tokens[i]);
    fprintf(d.manifest, "\n# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1\n");
    fprintf(d.manifest, "# int\tname\toccurrence\tof\ttype\ttwin\tlayout\tcount\tbytes\tsum\tabsmax\tfile\n");
    fprintf(d.manifest, "# input\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1\n");

    const int rc = llama_decode(init.context, llama_batch_get_one(tokens.data(), (int) tokens.size(), 0, 0));
    report_unhandled(d);
    if (rc) {
        fprintf(stderr, "dump_ref: decode failed\n");
        return 1;
    }
    // A write that failed stopped the graph from the callback, but the decode still returns
    // success; without this check the trailer would certify a set missing every later tensor.
    if (d.failed) {
        fprintf(stderr, "dump_ref: a file could not be written — no trailer, the set is not complete\n");
        return 1;
    }

    // The trailer is the completion proof: a reader that does not find it is looking at a
    // set from a run that died, whatever the file count says.
    fprintf(d.manifest, "# complete\t%d\t%d\n", d.written, d.skipped);
    fclose(d.manifest);
    d.manifest = nullptr;
    if (rename(manifest_partial.c_str(), manifest_final.c_str()) != 0) {
        fprintf(stderr, "dump_ref: cannot install %s\n", manifest_final.c_str());
        return 1;
    }
    printf("dump_ref: wrote %d tensors, skipped %d, %d graph inputs, %d integer twins, into %s\n",
           d.written, d.skipped, d.inputs, d.twins, d.dir.c_str());
    llama_free(init.context);
    llama_free_model(init.model);
    llama_backend_free();
    return 0;
}
