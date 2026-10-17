// Diagnostic forced-prefix logits from mainline llama.cpp, one f32 row per requested continuation id.
// Row 0 follows the whole prompt; row i > 0 follows forced[i-1]. The final forced id is not fed.
// The Python runner verifies identities and installs the complete diagnostic bundle. This writer is
// not a registered reference family or a numerical gate. --validate-inputs opens no model or device.

#include "llama.h"
#include "gguf.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

static int integer(const std::string & text, const std::string & what, int minimum) {
    int result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size() || result < minimum) {
        throw std::runtime_error(what + ": invalid integer '" + text + "'");
    }
    return result;
}

static std::vector<llama_token> ids(const std::string & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read ids: " + path);
    std::vector<llama_token> result;
    std::string line;
    while (std::getline(input, line)) {
        result.push_back(integer(line, path + ":" + std::to_string(result.size() + 1), 0));
        if (result.size() > 65536) throw std::runtime_error(path + ": more than 65536 ids");
    }
    if (input.bad() || result.empty()) throw std::runtime_error(path + ": unreadable or empty ids");
    return result;
}

struct Options {
    std::string model, prompt, forced, out, prefill = "batch";
    int ctx = 4352, ubatch = 512, gpu_layers = 99, threads = 16;
    bool validate = false;
};

static Options options(int argc, char ** argv) {
    Options o;
    std::vector<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (std::find(seen.begin(), seen.end(), key) != seen.end()) {
            throw std::runtime_error("duplicate option " + key);
        }
        seen.push_back(key);
        if (key == "--validate-inputs") { o.validate = true; continue; }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--model") o.model = value;
        else if (key == "--prompt-file") o.prompt = value;
        else if (key == "--forced-file") o.forced = value;
        else if (key == "--out") o.out = value;
        else if (key == "--prefill") o.prefill = value;
        else if (key == "--ctx") o.ctx = integer(value, key, 1);
        else if (key == "--ubatch") o.ubatch = integer(value, key, 1);
        else if (key == "--gpu-layers") o.gpu_layers = integer(value, key, 0);
        else if (key == "--threads") o.threads = integer(value, key, 1);
        else throw std::runtime_error("unknown option " + key);
    }
    if (o.model.empty() || o.prompt.empty() || o.forced.empty() || o.out.empty()) {
        throw std::runtime_error("requires --model, --prompt-file, --forced-file and --out");
    }
    if (o.prefill != "step" && o.prefill != "batch") throw std::runtime_error("--prefill is step or batch");
    return o;
}

struct Batch {
    llama_batch value;
    explicit Batch(int size) : value(llama_batch_init(size, 0, 1)) {}
    ~Batch() { llama_batch_free(value); }
};

static void decode(llama_context * ctx, Batch & batch, const llama_token * tokens, int n, int position) {
    batch.value.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        batch.value.token[i] = tokens[i];
        batch.value.pos[i] = position + i;
        batch.value.n_seq_id[i] = 1;
        batch.value.seq_id[i][0] = 0;
        batch.value.logits[i] = i == n - 1;
    }
    const int rc = llama_decode(ctx, batch.value);
    if (rc != 0) throw std::runtime_error("llama_decode at position " + std::to_string(position) + " returned " + std::to_string(rc));
}

static void put_ids(std::ostream & output, const char * key, const std::vector<llama_token> & tokens) {
    output << key << '\t';
    for (size_t i = 0; i < tokens.size(); ++i) output << (i ? "," : "") << tokens[i];
    output << '\n';
}

static int run(int argc, char ** argv) {
    const Options o = options(argc, argv);
    const auto prompt = ids(o.prompt), forced = ids(o.forced);
    if (forced.size() > 4096 || prompt.size() + forced.size() > size_t(o.ctx)) {
        throw std::runtime_error("prompt plus continuation exceeds context, or continuation exceeds 4096 rows");
    }
    if (o.validate) {
        printf("inputs valid: prompt=%zu forced=%zu ctx=%d; no model or device opened\n", prompt.size(), forced.size(), o.ctx);
        return 0;
    }
    const char * write = std::getenv("BLOOMERY_REF_WRITE");
    if (!write || std::strcmp(write, "1") != 0) throw std::runtime_error("use dense-logits.py; BLOOMERY_REF_WRITE=1 is required");
    {
        const gguf_init_params gp = {true, nullptr};
        std::unique_ptr<gguf_context, decltype(&gguf_free)> header(gguf_init_from_file(o.model.c_str(), gp), gguf_free);
        if (!header) throw std::runtime_error("cannot read GGUF header");
        const int64_t arch = gguf_find_key(header.get(), "general.architecture");
        if (arch < 0 || gguf_get_kv_type(header.get(), arch) != GGUF_TYPE_STRING ||
            std::strcmp(gguf_get_val_str(header.get(), arch), "qwen35") != 0) {
            throw std::runtime_error("expected dense qwen35 before model load");
        }
        if (gguf_find_key(header.get(), "qwen35.decision.type") >= 0) throw std::runtime_error("decision model is not a dense generation diagnostic");
    }
    if (std::ifstream(o.out) || std::ifstream(o.out + ".meta")) throw std::runtime_error("output already exists");
    llama_backend_init();
    struct Backend { ~Backend() { llama_backend_free(); } } backend;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = o.gpu_layers;
    ggml_backend_dev_t no_devices[] = {nullptr};
    if (o.gpu_layers == 0) mp.devices = no_devices;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(o.model.c_str(), mp), llama_model_free);
    if (!model) throw std::runtime_error("model load failed");
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    if (vocab <= 0) throw std::runtime_error("model has no vocabulary");
    for (const auto * values : {&prompt, &forced}) {
        for (const auto token : *values) if (token >= vocab) throw std::runtime_error("token id past vocabulary");
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = uint32_t(o.ctx);
    cp.n_batch = uint32_t(prompt.size());
    cp.n_ubatch = uint32_t(std::min(o.ubatch, int(prompt.size())));
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = o.threads;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (o.gpu_layers == 0) {
        cp.offload_kqv = false;
        cp.op_offload = false;
    }
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) throw std::runtime_error("context creation failed");
    Batch batch(int(prompt.size()));
    const int prefix = int(prompt.size()) - 1;
    if (o.prefill == "step") {
        for (int i = 0; i < prefix; ++i) decode(ctx.get(), batch, &prompt[size_t(i)], 1, i);
    } else if (prefix) {
        decode(ctx.get(), batch, prompt.data(), prefix, 0);
    }
    decode(ctx.get(), batch, &prompt.back(), 1, prefix);
    std::unique_ptr<FILE, decltype(&std::fclose)> raw(std::fopen(o.out.c_str(), "wbx"), std::fclose);
    if (!raw) throw std::runtime_error("cannot create raw output");
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "requires IEEE754 f32");
    std::vector<unsigned char> bytes(size_t(vocab) * 4);
    for (size_t i = 0; i < forced.size(); ++i) {
        if (i) decode(ctx.get(), batch, &forced[i - 1], 1, int(prompt.size() + i - 1));
        const float * row = llama_get_logits_ith(ctx.get(), -1);
        if (!row) throw std::runtime_error("missing logit row " + std::to_string(i));
        for (int j = 0; j < vocab; ++j) {
            if (!std::isfinite(row[j])) throw std::runtime_error("nonfinite logit at row " + std::to_string(i) + " token " + std::to_string(j));
            uint32_t bits = 0;
            std::memcpy(&bits, row + j, sizeof(bits));
            for (int b = 0; b < 4; ++b) bytes[size_t(j) * 4 + size_t(b)] = static_cast<unsigned char>(bits >> (8 * b));
        }
        if (std::fwrite(bytes.data(), 1, bytes.size(), raw.get()) != bytes.size()) throw std::runtime_error("short write of logit row");
    }
    if (std::fclose(raw.release()) != 0) throw std::runtime_error("raw logit close failed");
    const auto temporary = o.out + ".meta.partial";
    std::ostringstream meta;
    meta << "format\tbloomery-forced-logits-v1\ndtype\tfloat32-le\nrows\t" << forced.size()
         << "\nvocab\t" << vocab << "\nmode\treference\nprefill\t" << o.prefill << "\nlast_step\ttrue\n";
    put_ids(meta, "prompt_ids", prompt);
    put_ids(meta, "forced_ids", forced);
    meta << "complete\ttrue\n";
    const auto content = meta.str();
    std::unique_ptr<FILE, decltype(&std::fclose)> metadata(std::fopen(temporary.c_str(), "wx"), std::fclose);
    if (!metadata || std::fwrite(content.data(), 1, content.size(), metadata.get()) != content.size()) throw std::runtime_error("cannot write metadata");
    if (std::fclose(metadata.release()) != 0 || ::link(temporary.c_str(), (o.out + ".meta").c_str()) != 0) throw std::runtime_error("cannot complete metadata without replacement");
    if (::unlink(temporary.c_str()) != 0) throw std::runtime_error("cannot remove metadata staging name");
    printf("dense_logits_ref: rows=%zu vocab=%d raw=%s; diagnostic only\n", forced.size(), vocab, o.out.c_str());
    return 0;
}

int main(int argc, char ** argv) {
    try { return run(argc, argv); }
    catch (const std::exception & e) { std::fprintf(stderr, "dense_logits_ref: %s\n", e.what()); return 1; }
}
