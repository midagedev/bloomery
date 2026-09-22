// MUL-27 continuation: run ik's OWN x4 kernel — the pairing the oracle
// actually dispatches (iqk_gemm_kquants.cpp:2751/2768:
// mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2>, expected_type_B =
// GGML_TYPE_Q8_2_X4) — on the first aligned Q4_K tensor's first 64 rows,
// with ik's quantize_row_q8_2_x4 coding the same attn_norm-0 column the
// Rust gate quantizes. Output file format matches q4k-ik-dot.txt (tensor
// header, one long hex line of ik's activation bytes, `row R %08x` lines)
// so the Rust side parses both dumps the same way. Extraction pattern
// copied from q4k_ref.cpp.
#include "ggml.h"
#include "ggml-backend.h"

#define IQK_IMPLEMENT
#include "iqk_gemm_kquants.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// The data directory the gates read (tools/box.sh exports BLOOMERY_DATA; a parallel
// track moves both this harness's input and its dump by setting it).
static const char *kDataDir = getenv("BLOOMERY_DATA") ? getenv("BLOOMERY_DATA") : "/root/bloomery-data";

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
    // Copy the name BEFORE freeing (q4k_ref.cpp lesson: the pointer dies
    // with the context).
    std::string tensor_name = gguf_get_tensor_name(gguf, pick);
    struct ggml_tensor *t = ggml_get_tensor(gctx, tensor_name.c_str());
    const int k = (int)t->ne[0];
    const size_t rs = ggml_row_size(t->type, k);
    fprintf(stderr, "tensor %s k=%d rowsz=%zu\n", tensor_name.c_str(), k, rs);
    std::vector<uint8_t> w = read_range(kGgufPath, (int64_t)(data_off + gguf_get_tensor_offset(gguf, pick)), rs * 64);
    gguf_free(gguf);
    ggml_free(gctx);
    const char *nm = tensor_name.c_str();

    FILE *f = fopen((std::string(kDataDir) + "/ref/attn_norm-0.0.f32").c_str(), "rb");
    if (!f) { fprintf(stderr, "no oracle dump\n"); return 1; }
    std::vector<float> x(k);
    if (fread(x.data(), 4, k, f) != (size_t)k) { fprintf(stderr, "short dump\n"); return 1; }
    fclose(f);

    // ik's own q8_2_x4 coding of the same column: 144 bytes per 128 values
    // (x4 groups of four 32-value sub-blocks; k % 256 == 0 means no tail).
    std::vector<uint8_t> y(144 * (k / 128));
    quantize_row_q8_2_x4(x.data(), y.data(), k);

    // The x4 kernel through ik's own dispatch table: kernels[0] is the
    // nrc_y = 1 instantiation of mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2>.
    std::array<mul_mat_t, IQK_MAX_NY> kernels{};
    mul_mat_t func16 = nullptr;
    if (!iqk_set_kernels_kquants(k, GGML_TYPE_Q4_K, GGML_TYPE_Q8_2_X4, kernels, func16) || !kernels[0]) {
        fprintf(stderr, "ik declined the Q4_K x Q8_2_X4 pairing (k=%d)\n", k);
        return 1;
    }
    std::vector<float> dst(64);
    DataInfo info;
    info.s = dst.data();
    info.cy = (const char *)y.data();
    info.bs = 64;          // one dst row of 64 outputs
    info.by = y.size();    // one activation row
    info.cur_y = 0;
    info.ne11 = 1;
    info.row_mapping = nullptr;
    info.bs2 = 0;
    // bx is the ROW STRIDE, not an offset — BaseDequantizer::new_row walks
    // `vx + bx*ix` (iqk_common.h:368), and the real caller passes the row
    // size (iqk_mul_mat.cpp:96). Zero here computed row 0 sixty-four times.
    kernels[0](k, w.data(), rs, info, 64);

    FILE *out = fopen((std::string(kDataDir) + "/ref/q4k-x4-ik-dot.txt").c_str(), "w");
    fprintf(out, "tensor %s k %d\n", nm, k);
    for (size_t i = 0; i < y.size(); ++i) fprintf(out, "%02x", y[i]);
    fprintf(out, "\n");
    for (int r = 0; r < 64; ++r) {
        uint32_t bits; memcpy(&bits, &dst[r], 4);
        fprintf(out, "row %d %08x\n", r, bits);
    }
    fclose(out);
    printf("dumped 64 rows of %s (x4 pairing)\n", nm);
    return 0;
}
