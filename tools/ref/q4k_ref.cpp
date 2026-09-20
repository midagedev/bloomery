// MUL-27 ground truth: run ik's OWN ggml_vec_dot_q4_K_q8_K on the first
// aligned Q4_K tensor's first rows, with ik's own quantize_row_q8_K coding
// the same activation column the Rust gate quantizes (attn_norm-0 first k
// floats). Prints %a floats for exact comparison against the Rust
// kernel/mirror. Extraction pattern copied from q3k_cpu_ref.cpp.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-quants.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const char *kGgufPath = "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf";

static std::vector<uint8_t> read_range(const std::string &p, int64_t off, size_t n) {
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", p.c_str()); exit(1); }
    std::vector<uint8_t> b(n);
    if (fseek(f, off, SEEK_SET) || fread(b.data(), 1, n, f) != n) { fprintf(stderr, "read failed\n"); exit(1); }
    fclose(f);
    return b;
}

int main() {
    struct ggml_context *gctx = nullptr;
    struct gguf_init_params gp = {true, &gctx};
    struct gguf_context *gguf = gguf_init_from_file(kGgufPath, gp);
    if (!gguf || !gctx) { fprintf(stderr, "gguf open failed\n"); return 1; }
    const size_t data_off = gguf_get_data_offset(gguf);

    // First Q4_K tensor with k % 256 == 0 — the same scan the Rust gate does.
    int pick = -1;
    for (int i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        const char *nm = gguf_get_tensor_name(gguf, i);
        struct ggml_tensor *c = ggml_get_tensor(gctx, nm);
        if (c && c->type == GGML_TYPE_Q4_K && c->ne[0] % 256 == 0) { pick = i; break; }
    }
    if (pick < 0) { fprintf(stderr, "no Q4_K tensor\n"); return 1; }
    // Copy the name BEFORE freeing: gguf_get_tensor_name points into the
    // context, and the first dump wrote freed-memory garbage into the header.
    std::string tensor_name = gguf_get_tensor_name(gguf, pick);
    struct ggml_tensor *t = ggml_get_tensor(gctx, tensor_name.c_str());
    const int k = (int)t->ne[0];
    const size_t rs = ggml_row_size(t->type, k);
    fprintf(stderr, "tensor %s k=%d rowsz=%zu\n", tensor_name.c_str(), k, rs);
    std::vector<uint8_t> w = read_range(kGgufPath, (int64_t)(data_off + gguf_get_tensor_offset(gguf, pick)), rs * 64);
    gguf_free(gguf);
    ggml_free(gctx);
    const char *nm = tensor_name.c_str();

    FILE *f = fopen("/root/bloomery-data/ref/attn_norm-0.0.f32", "rb");
    if (!f) { fprintf(stderr, "no oracle dump\n"); return 1; }
    std::vector<float> x(k);
    if (fread(x.data(), 4, k, f) != (size_t)k) { fprintf(stderr, "short dump\n"); return 1; }
    fclose(f);

    std::vector<uint8_t> y(296 * (k / 256));
    quantize_row_q8_K(x.data(), y.data(), k);

    ggml_type_traits_t tr = ggml_internal_get_type_traits(GGML_TYPE_Q4_K);
    FILE *out = fopen("/root/bloomery-data/ref/q4k-ik-dot.txt", "w");
    fprintf(out, "tensor %s k %d\n", nm, k);
    // ik's own q8_K coding of the same column, hex bytes, so the Rust gate
    // can feed the EXACT same activations to its kernel and compare
    // kernels — not kernels-plus-encoders.
    for (size_t i = 0; i < y.size(); ++i) fprintf(out, "%02x", y[i]);
    fprintf(out, "\n");
    for (int r = 0; r < 64; ++r) {
        float s = 0;
        tr.vec_dot(k, &s, 0, w.data() + (size_t)r * rs, 0, y.data(), 0, 1);
        uint32_t bits; memcpy(&bits, &s, 4);
        fprintf(out, "row %d %08x\n", r, bits);
    }
    fclose(out);
    printf("dumped 64 rows of %s\n", nm);
    return 0;
}
