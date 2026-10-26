// Forced-prefix logits from mainline llama.cpp, one f32 row per requested continuation id.
// Row 0 follows the whole prompt; row i > 0 follows forced[i-1]. The final forced id is not fed.
// tools/ref/dense-logits.py verifies the identities and installs the complete set: with its manifest this is the
// CUDA writer of the refset families logits-qwen38d and logits-nanbeige43 (crates/refset/src/arch/qwen35/dense.rs,
// crates/refset/src/arch/llama/dense.rs); HIP captures are lab diagnostics, not those families' sets. The raw rows and
// sidecar are not a set until the runner writes the manifest. --arch names the general.architecture the model file must
// state (the writer passes its target's). --validate-inputs opens no model or device.

#include "llama.h"
#include "gguf.h"
#include "ggml-backend.h"

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
    std::string model, prompt, forced, out, prefill = "batch", arch;
    int ctx = 4352, ubatch = 512, gpu_layers = 99, threads = 16;
    std::string backend;
    int reset_repeats = 1;
    size_t hip_max_model_bytes = 16 * 1024 * 1024, hip_reserve_bytes = 0;
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
        else if (key == "--arch") o.arch = value;
        else if (key == "--backend") o.backend = value;
        else if (key == "--reset-repeats") o.reset_repeats = integer(value, key, 1);
        else if (key == "--hip-max-model-bytes") o.hip_max_model_bytes = size_t(integer(value, key, 1));
        else if (key == "--hip-reserve-bytes") o.hip_reserve_bytes = size_t(integer(value, key, 1));
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
    if (o.backend != "cpu" && o.backend != "cuda" && o.backend != "hip") throw std::runtime_error("requires explicit --backend cpu/cuda/hip");
    if (o.reset_repeats > 2) throw std::runtime_error("--reset-repeats is 1 or 2");
    if ((o.backend == "cpu") != (o.gpu_layers == 0)) throw std::runtime_error("backend and gpu-layers disagree");
    if (o.backend == "hip" && (o.hip_reserve_bytes == 0 || o.gpu_layers != 99)) throw std::runtime_error("HIP requires reserve and gpu-layers 99");
    return o;
}

// A named deleter: gcc rejects (-Werror=ignored-attributes) decltype(&std::fclose) as a template argument.
struct Closer { void operator()(FILE * f) const { if (f) std::fclose(f); } };
using File = std::unique_ptr<FILE, Closer>;

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

// The pinned scheduler asks before compute and calls again after synchronization.
// Never unwind through its C callback; retain failure state for the decode caller.
struct Placement {
    ggml_backend_dev_t device = nullptr;
    std::vector<bool> layers;
    size_t hip_math = 0, cpu_math = 0, cpu_input = 0;
    bool output = false, failed = false;
    static bool observe(ggml_tensor * t, bool ask, void * data) noexcept {
        auto & p = *static_cast<Placement *>(data);
        if (ask) return true;
        if (p.failed) return false;
        try {
            const ggml_tensor * allocation = t;
            while (allocation->view_src) allocation = allocation->view_src;
            const auto buffer = allocation->buffer;
            const auto buft = buffer ? ggml_backend_buffer_get_type(buffer) : nullptr;
            const auto dev = buft ? ggml_backend_buft_get_device(buft) : nullptr;
            const bool math = t->op != GGML_OP_NONE && t->op != GGML_OP_VIEW &&
                t->op != GGML_OP_RESHAPE && t->op != GGML_OP_PERMUTE &&
                t->op != GGML_OP_TRANSPOSE && t->op != GGML_OP_CPY &&
                t->op != GGML_OP_CONT && t->op != GGML_OP_DUP && !ggml_is_empty(t);
            const bool on_hip = dev == p.device && buft && !ggml_backend_buft_is_host(buft);
            if (math) {
                if (on_hip) ++p.hip_math;
                else if (dev && (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU ||
                                 ggml_backend_buft_is_host(buft))) {
                    ++p.cpu_math;
                    // Only the standard token-embedding gather may execute CPU math.
                    // Other CPU trunk math (including GDN/attention) is a failed capture.
                    if (t->op == GGML_OP_GET_ROWS && std::strcmp(t->name, "model.input_embed") == 0)
                        ++p.cpu_input;
                    else p.failed = true;
                } else p.failed = true;
            }
            if (std::strcmp(t->name, "result_output") == 0) {
                if (!on_hip || !math) p.failed = true;
                else p.output = true;
            }
            constexpr char prefix[] = "ffn_out-";
            if (std::strncmp(t->name, prefix, sizeof(prefix) - 1) == 0) {
                const char * begin = t->name + sizeof(prefix) - 1;
                int layer = -1;
                const auto parsed = std::from_chars(begin, begin + std::strlen(begin), layer);
                if (parsed.ec != std::errc() || *parsed.ptr || layer < 0 ||
                    size_t(layer) >= p.layers.size() || !on_hip || !math) p.failed = true;
                else p.layers[size_t(layer)] = true;
            }
        } catch (...) { p.failed = true; }
        return !p.failed;
    }
    void require() const {
        if (failed) throw std::runtime_error("HIP evaluation callback observed invalid placement or failed");
        if (!output || !hip_math || std::find(layers.begin(), layers.end(), false) != layers.end())
            throw std::runtime_error("HIP trunk layers and output-head compute were not all observed");
    }
};

static size_t add_bytes(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) throw std::runtime_error("memory estimate overflow");
    return a + b;
}

static size_t multiply_bytes(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a) throw std::runtime_error("memory estimate overflow");
    return a * b;
}

static size_t free_memory(ggml_backend_dev_t dev, size_t required, const char * stage) {
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(dev, &free, &total);
    if (!total || free > total || free < required)
        throw std::runtime_error(std::string("HIP selected-device memory insufficient/unknown at ") + stage +
            ": free=" + std::to_string(free) + " required=" + std::to_string(required));
    return free;
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
    size_t tensor_bytes = 0, context_allowance = 0;
    {
        const gguf_init_params gp = {true, nullptr};
        std::unique_ptr<gguf_context, decltype(&gguf_free)> header(gguf_init_from_file(o.model.c_str(), gp), gguf_free);
        if (!header) throw std::runtime_error("cannot read GGUF header");
        if (o.arch.empty()) throw std::runtime_error("requires --arch, the general.architecture of the model file");
        const int64_t key_arch = gguf_find_key(header.get(), "general.architecture");
        if (key_arch < 0 || gguf_get_kv_type(header.get(), key_arch) != GGUF_TYPE_STRING ||
            o.arch != gguf_get_val_str(header.get(), key_arch)) {
            throw std::runtime_error("expected dense " + o.arch + " before model load");
        }
        if (gguf_find_key(header.get(), (o.arch + ".decision.type").c_str()) >= 0) throw std::runtime_error("decision model is not a dense generation diagnostic");
        if (o.backend == "hip") {
            for (int64_t i = 0; i < gguf_get_n_tensors(header.get()); ++i)
                tensor_bytes = add_bytes(tensor_bytes, gguf_get_tensor_size(header.get(), i));
            if (tensor_bytes > o.hip_max_model_bytes) throw std::runtime_error("GGUF tensor payload exceeds HIP model limit before allocation");
            const auto dimension = [&](const char * name) -> size_t {
                const int64_t key = gguf_find_key(header.get(), name);
                if (key < 0 || gguf_get_kv_type(header.get(), key) != GGUF_TYPE_UINT32)
                    throw std::runtime_error(std::string("unsupported HIP estimate geometry: ") + name);
                const auto value = gguf_get_val_u32(header.get(), key);
                if (!value) throw std::runtime_error("zero HIP estimate geometry");
                return value;
            };
            // Conservative planning allowance, not an allocator cap or measured peak:
            // baseline scratch + expanded weights + context activations/state + FFN scratch.
            const size_t layers = dimension((o.arch + ".block_count").c_str());
            const size_t embedding = dimension((o.arch + ".embedding_length").c_str());
            const size_t ffn = dimension((o.arch + ".feed_forward_length").c_str());
            context_allowance = add_bytes(32 * 1024 * 1024, multiply_bytes(tensor_bytes, 4));
            context_allowance = add_bytes(context_allowance, multiply_bytes(multiply_bytes(multiply_bytes(size_t(o.ctx), embedding), layers), 32));
            context_allowance = add_bytes(context_allowance, multiply_bytes(multiply_bytes(size_t(o.ubatch), ffn), 32));
        }
    }
    const std::string repeat_path = o.out + ".repeat-1.f32";
    if (std::ifstream(o.out) || std::ifstream(o.out + ".meta") || std::ifstream(repeat_path)) throw std::runtime_error("output already exists");
    llama_backend_init();
    struct Backend { ~Backend() { llama_backend_free(); } } backend;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = o.gpu_layers;
    ggml_backend_dev_t no_devices[] = {nullptr};
    if (o.gpu_layers == 0) mp.devices = no_devices;
    Placement placement;
    size_t free_before = 0, free_load = 0, free_context = 0, free_run = 0;
    ggml_backend_dev_t selected_devices[] = {nullptr, nullptr};
    if (o.backend == "hip") {
        // ROCm is the registry name in the pinned public ggml-cuda.h HIP branch.
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            const auto dev = ggml_backend_dev_get(i);
            if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "ROCm") != 0) continue;
            const auto type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                placement.device = dev;
                break;
            }
        }
        if (!placement.device) throw std::runtime_error("no visible ROCm registry GPU; refusing CPU fallback");
        selected_devices[0] = placement.device;
        mp.devices = selected_devices;
        mp.main_gpu = 0;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        free_before = free_memory(placement.device, add_bytes(o.hip_reserve_bytes, add_bytes(tensor_bytes, context_allowance)), "before_model");
    }
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(o.model.c_str(), mp), llama_model_free);
    if (!model) throw std::runtime_error("model load failed");
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    if (vocab <= 0) throw std::runtime_error("model has no vocabulary");
    for (const auto * values : {&prompt, &forced}) {
        for (const auto token : *values) if (token >= vocab) throw std::runtime_error("token id past vocabulary");
    }
    if (placement.device) {
        free_load = free_memory(placement.device, add_bytes(o.hip_reserve_bytes, context_allowance), "after_load");
        const int layers = llama_model_n_layer(model.get());
        if (layers < 1) throw std::runtime_error("model has no trunk layers");
        placement.layers.resize(size_t(layers), false);
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
    if (placement.device) {
        cp.cb_eval = Placement::observe;
        cp.cb_eval_user_data = &placement;
    }
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) throw std::runtime_error("context creation failed");
    if (placement.device) free_context = free_memory(placement.device, o.hip_reserve_bytes, "after_context");
    Batch batch(int(prompt.size()));
    const auto step = [&](const llama_token * tokens, int n, int position) {
        if (placement.device) free_memory(placement.device, o.hip_reserve_bytes, "before_decode");
        decode(ctx.get(), batch, tokens, n, position);
        if (placement.failed) throw std::runtime_error("HIP evaluation callback failed");
        if (placement.device) free_memory(placement.device, o.hip_reserve_bytes, "after_decode");
    };
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "requires IEEE754 f32");
    std::vector<unsigned char> bytes(size_t(vocab) * 4);
    std::vector<unsigned char> first_bytes(o.reset_repeats == 2 ? size_t(vocab) * 4 : 0);
    for (int repeat = 0; repeat < o.reset_repeats; ++repeat) {
        if (repeat) {
            const auto memory = llama_get_memory(ctx.get());
            if (!memory) throw std::runtime_error("context has no resettable memory");
            llama_memory_clear(memory, true);
            std::fill(placement.layers.begin(), placement.layers.end(), false);
            placement.output = false;
        }
        const int prefix = int(prompt.size()) - 1;
        if (o.prefill == "step") {
            for (int i = 0; i < prefix; ++i) step(&prompt[size_t(i)], 1, i);
        } else if (prefix) step(prompt.data(), prefix, 0);
        step(&prompt.back(), 1, prefix);
        const std::string path = repeat ? repeat_path : o.out;
        File raw(std::fopen(path.c_str(), "wbx"));
        if (!raw) throw std::runtime_error("cannot create raw output");
        std::ifstream first;
        if (repeat) {
            first.open(o.out, std::ios::binary);
            if (!first) throw std::runtime_error("cannot reopen first raw output");
        }
        bool equal = true;
        for (size_t i = 0; i < forced.size(); ++i) {
            if (i) step(&forced[i - 1], 1, int(prompt.size() + i - 1));
            const float * row = llama_get_logits_ith(ctx.get(), -1);
            if (!row) throw std::runtime_error("missing logit row " + std::to_string(i));
            for (int j = 0; j < vocab; ++j) {
                if (!std::isfinite(row[j])) throw std::runtime_error("nonfinite logit at row " + std::to_string(i) + " token " + std::to_string(j));
                uint32_t bits = 0;
                std::memcpy(&bits, row + j, sizeof(bits));
                for (int b = 0; b < 4; ++b) bytes[size_t(j) * 4 + size_t(b)] = static_cast<unsigned char>(bits >> (8 * b));
            }
            if (std::fwrite(bytes.data(), 1, bytes.size(), raw.get()) != bytes.size()) throw std::runtime_error("short write of logit row");
            if (repeat) {
                if (!first.read(reinterpret_cast<char *>(first_bytes.data()), std::streamsize(first_bytes.size())))
                    throw std::runtime_error("truncated first raw output");
                equal = equal && bytes == first_bytes;
            }
        }
        if (std::fclose(raw.release()) != 0) throw std::runtime_error("raw logit close failed");
        if (repeat && (first.peek() != std::char_traits<char>::eof() || !equal))
            throw std::runtime_error("reset repeat raw logits differ bitwise");
        if (placement.device) placement.require();
    }
    if (placement.device) free_run = free_memory(placement.device, o.hip_reserve_bytes, "after_run");
    const auto temporary = o.out + ".meta.partial";
    std::ostringstream meta;
    meta << "format\tbloomery-forced-logits-v1\ndtype\tfloat32-le\nrows\t" << forced.size()
         << "\nvocab\t" << vocab << "\nmode\treference\nprefill\t" << o.prefill << "\nlast_step\ttrue\n";
    put_ids(meta, "prompt_ids", prompt);
    put_ids(meta, "forced_ids", forced);
    meta << "backend\t" << o.backend << "\nreset_repeats\t" << o.reset_repeats << '\n';
    if (o.reset_repeats == 2) {
        const auto slash = repeat_path.find_last_of('/');
        meta << "reset_repeat_bitwise_equal\ttrue\nrepeat_file\t" << repeat_path.substr(slash == std::string::npos ? 0 : slash + 1) << '\n';
    }
    if (placement.device) {
        meta << "hip_device\t" << ggml_backend_dev_name(placement.device)
             << "\nhip_description\t" << ggml_backend_dev_description(placement.device)
             << "\nhip_registry\t" << ggml_backend_reg_name(ggml_backend_dev_backend_reg(placement.device))
             << "\nhip_trunk_layers\t" << std::count(placement.layers.begin(), placement.layers.end(), true)
             << "\nhip_model_layers\t" << placement.layers.size() << "\nhip_output\ttrue"
             << "\nhip_math_nodes\t" << placement.hip_math << "\nhip_cpu_math_nodes\t" << placement.cpu_math
             << "\nhip_cpu_input_nodes\t" << placement.cpu_input
             << "\nhip_reserve_bytes\t" << o.hip_reserve_bytes << "\nhip_max_model_bytes\t" << o.hip_max_model_bytes
             << "\nmodel_tensor_bytes\t" << tensor_bytes << "\nhip_context_allowance_bytes\t" << context_allowance
             << "\nhip_memory_guard\tstage_preflight_not_allocation_cap"
             << "\nhip_estimate_exclusions\tdriver_external_allocations_allocator_fragmentation_unmodeled_backend_workspace"
             << "\nhip_free_before_model\t" << free_before << "\nhip_free_after_load\t" << free_load
             << "\nhip_free_after_context\t" << free_context << "\nhip_free_after_run\t" << free_run << '\n';
    }
    meta << "complete\ttrue\n";
    const auto content = meta.str();
    File metadata(std::fopen(temporary.c_str(), "wx"));
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
