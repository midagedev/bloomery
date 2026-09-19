// dump_ref — write ik_llama.cpp's intermediate forward-pass tensors to disk as raw f32.
//
// This is the oracle boundary for stage 1. Every round from 1-2 on compares its own
// forward pass against files this tool writes; the gate reads $MULLE_DATA/ref/.
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
// so a missing file is never mistaken for a tensor that did not run.
//
// Build: tools/ref/build-dump.sh   Run: $MULLE_DATA/bin/dump_ref -m <gguf> --tokens 1,2,3

#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <sys/stat.h>
#include <vector>

struct dump_ctx {
    std::string                  dir;
    FILE *                       manifest = nullptr;
    std::map<std::string, int>   seen;      // name -> times emitted, for the occurrence index
    std::vector<uint8_t>         staging;   // device tensors land here before the write
    int                          written = 0;
    int                          skipped = 0;
};

// Tensor names carry '/' and '.' in some graphs; keep the file name a single path element.
static std::string safe_name(const char * name) {
    std::string s(name);
    for (char & c : s) {
        if (c == '/' || c == '\\' || c == ' ') c = '_';
    }
    return s;
}

static int on_tensor(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (dump_ctx *) user_data;
    if (ask) {
        return 1;  // yes, we want every node
    }

    const int occurrence = d->seen[t->name]++;

    if (ggml_is_quantized(t->type)) {
        fprintf(d->manifest, "skip\t%s\t%d\t%s\tquantized\n", t->name, occurrence, ggml_type_name(t->type));
        d->skipped++;
        return 1;
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
    const int64_t n = ggml_nelements(t);
    std::vector<float> out((size_t) n);
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        float v;
        switch (t->type) {
            case GGML_TYPE_F32:  v = ((const float *)    src)[i]; break;
            case GGML_TYPE_F16:  v = ggml_fp16_to_fp32(((const ggml_fp16_t *) src)[i]); break;
            case GGML_TYPE_I32:  v = (float) ((const int32_t *) src)[i]; break;
            case GGML_TYPE_I16:  v = (float) ((const int16_t *) src)[i]; break;
            case GGML_TYPE_I8:   v = (float) ((const int8_t  *) src)[i]; break;
            default:
                fprintf(d->manifest, "skip\t%s\t%d\t%s\tunhandled\n", t->name, occurrence, ggml_type_name(t->type));
                d->skipped++;
                return 1;
        }
        out[(size_t) i] = v;
        sum += v;
    }

    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.%d.f32", d->dir.c_str(), safe_name(t->name).c_str(), occurrence);
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "dump_ref: cannot write %s\n", path);
        return 0;  // stop the graph: a partial reference set is worse than none
    }
    const size_t want = (size_t) n;
    if (fwrite(out.data(), sizeof(float), want, f) != want) {
        fprintf(stderr, "dump_ref: short write on %s\n", path);
        fclose(f);
        return 0;
    }
    fclose(f);

    // ne[] in ggml order, verbatim — mulle's tensors carry the same order by decision
    // (docs/plan.md), so a shape mismatch in the gate is a real mismatch, not a convention.
    fprintf(d->manifest, "tensor\t%s\t%d\t%s\t%lld\t%lld\t%lld\t%lld\t%zu\t%.6f\t%s\n",
            t->name, occurrence, ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            want * sizeof(float), sum, ggml_op_desc(t));
    d->written++;
    return 1;
}

int main(int argc, char ** argv) {
    // --tokens is ours; everything else is gpt_params. Pull it out before the parser sees it.
    std::vector<llama_token> tokens;
    std::vector<char *>      passthrough;
    passthrough.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            char * list = argv[++i];
            for (char * p = strtok(list, ","); p; p = strtok(nullptr, ",")) {
                tokens.push_back((llama_token) atoi(p));
            }
        } else {
            passthrough.push_back(argv[i]);
        }
    }
    if (tokens.empty()) {
        fprintf(stderr, "dump_ref: --tokens <id,id,...> is required; this tool does not tokenize\n");
        return 2;
    }

    gpt_params params;
    if (!gpt_params_parse((int) passthrough.size(), passthrough.data(), params)) {
        fprintf(stderr, "dump_ref: bad arguments\n");
        return 2;
    }

    const char * data_dir = getenv("MULLE_DATA");
    dump_ctx d;
    d.dir = std::string(data_dir ? data_dir : "/root/mulle-data") + "/ref";
    mkdir(d.dir.c_str(), 0755);

    const std::string manifest_path = d.dir + "/MANIFEST.tsv";
    d.manifest = fopen(manifest_path.c_str(), "w");
    if (!d.manifest) {
        fprintf(stderr, "dump_ref: cannot write %s\n", manifest_path.c_str());
        return 1;
    }
    fprintf(d.manifest, "# dump_ref — ik_llama.cpp intermediate tensors, raw f32, little-endian\n");
    fprintf(d.manifest, "# model\t%s\n", params.model.c_str());
    fprintf(d.manifest, "# tokens\t");
    for (size_t i = 0; i < tokens.size(); ++i) fprintf(d.manifest, "%s%d", i ? "," : "", tokens[i]);
    fprintf(d.manifest, "\n# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\n");

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

    if (llama_decode(init.context, llama_batch_get_one(tokens.data(), (int) tokens.size(), 0, 0))) {
        fprintf(stderr, "dump_ref: decode failed\n");
        return 1;
    }

    fclose(d.manifest);
    printf("dump_ref: wrote %d tensors, skipped %d, into %s\n", d.written, d.skipped, d.dir.c_str());
    llama_free(init.context);
    llama_free_model(init.model);
    llama_backend_free();
    return 0;
}
