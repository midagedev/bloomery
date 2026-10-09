// Run ik's OWN IQ3_S kernel — the pairing the oracle dispatches on this CPU — on synthetic rows,
// with ik's Q8_K coding of the column the Rust gate reads back. Same dump format as
// iq4xs_ref.cpp: tensor header, ik's activation bytes hex, `row R <bits>` results, `w R <hex>`
// weight rows.
//
// The pairing: ggml.c:1142 — [GGML_TYPE_IQ3_S].vec_dot_type = GGML_TYPE_Q8_K;
// iqk_gemm_iquants.cpp:2753 — iqk_set_kernels_iquants (:2696) accepts typeB = Q8_K only, and
// :2772 picks set_functions<DequantizerIQ3S>, whose kernels[0] is
// mul_mat_qX_K_q8_K_IQ<DequantizerIQ3S, 1>, i.e. mul_mat_qX_K_q8_K_IQ_N<..., 1> on a build
// without HAVE_FANCY_SIMD (:1021, the box's znver3): make_scales over the nibble scales,
// IndexHelperIQ3S's SIMD nine-bit index build, SignHelper::sign_4_values, the -16 bsums fold.
// The activation is coded by type_traits[Q8_K].from_float, the call the matmul makes: 296-byte
// block_q8_K {d, sum, qs[256], bsums[16]}, the layout the Rust side reads.
//
// There is no IQ3_S file on the box, so the rows are synthetic and the dump says so: the tensor
// name is synthetic-iq3_s. kQuantRows rows are seeded random values quantized by ggml itself
// (quantize_iq3_s through ggml_quantize_chunk, with a synthetic importance matrix), as the
// community files are made; kRandomRows rows carry random code bytes under finite f16 scales,
// so every grid index (qs byte and qh bit) and every sign byte is hit. The shape is k = 4096,
// GLM-5.3-Flash UD-IQ4_XS's gate/up row. The column is the first k f32 of the attn_norm-0
// dump the other harnesses read: the real value distribution.
#include "ggml.h"
#include "ggml-backend.h"

#define IQK_IMPLEMENT
#include "iqk_gemm_iquants.h"

#include <cerrno>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "ref_paths.h"

static const char *kDataDir = ref_data_dir();
static const int kK = 4096;
static const int kQuantRows = 32;
static const int kRandomRows = 32;
static const int kDotRows = kQuantRows + kRandomRows;
static const size_t kBlock = 110;  // sizeof(block_iq3_s)
static const uint64_t kSeed = 0x1a2b3c4d5e6f7788ull ^ 21;

struct Lcg {
    uint64_t s;
    uint32_t next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (uint32_t)(s >> 32);
    }
    float unit() { return (next() >> 8) * (1.0f / 16777216.0f); }  // [0, 1)
};

// Write to <path>.tmp.<pid> and rename, as q5k_x4_ref.cpp does: a gate on another track
// reading the shared $BLOOMERY_DATA never sees a half-written dump.
static bool write_atomic(const std::string &path, const void *data, size_t n) {
    const std::string tmp = path + ".tmp." + std::to_string(getpid());
    FILE *out = fopen(tmp.c_str(), "wb");
    if (!out) {
        fprintf(stderr, "iq3s_ref: cannot open %s for writing: %s\n", tmp.c_str(), strerror(errno));
        return false;
    }
    const bool wrote = fwrite(data, 1, n, out) == n;
    if (fclose(out) != 0 || !wrote) {
        fprintf(stderr, "iq3s_ref: cannot write %s: %s\n", tmp.c_str(), strerror(errno));
        return false;
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        fprintf(stderr, "iq3s_ref: cannot rename %s to %s: %s\n", tmp.c_str(), path.c_str(), strerror(errno));
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    (void)argv;
    if (argc != 1) {
        fprintf(stderr, "usage: iq3s_ref  (no model: the rows are synthetic)\n");
        return 64;
    }
    // ggml_init fills the f16 -> f32 table GGML_FP16_TO_FP32 reads on this build.
    struct ggml_init_params ip = {1024, nullptr, true};
    struct ggml_context *init_ctx = ggml_init(ip);
    if (!init_ctx) { fprintf(stderr, "iq3s_ref: ggml_init failed\n"); return 1; }

    const size_t rs = ggml_row_size(GGML_TYPE_IQ3_S, kK);
    if (rs != kBlock * (kK / 256)) { fprintf(stderr, "iq3s_ref: row size %zu, not %zu blocks of %zu\n", rs, (size_t)(kK / 256), kBlock); return 1; }
    Lcg rng{kSeed};

    // Roughly Gaussian values (sum of four uniforms) with a per-row scale over two decades and
    // one outlier per 256, and an importance matrix of positive weights, as dequant_ref.cpp.
    std::vector<float> x((size_t)kQuantRows * kK);
    for (int r = 0; r < kQuantRows; ++r) {
        const float scale = 0.01f * std::pow(100.0f, rng.unit());
        for (int j = 0; j < kK; ++j) {
            float g = rng.unit() + rng.unit() + rng.unit() + rng.unit() - 2.0f;
            if (j % 256 == 17) g *= 6.0f;
            x[(size_t)r * kK + j] = scale * g;
        }
    }
    std::vector<float> imatrix((size_t)kK);
    for (float &w : imatrix) w = 0.25f + rng.unit();

    std::vector<uint8_t> w((size_t)kDotRows * rs);
    ggml_quantize_init(GGML_TYPE_IQ3_S);
    const size_t wrote = ggml_quantize_chunk(GGML_TYPE_IQ3_S, x.data(), w.data(), 0, kQuantRows, kK,
                                             imatrix.data(), nullptr);
    if (wrote != (size_t)kQuantRows * rs) { fprintf(stderr, "iq3s_ref: ggml_quantize_chunk wrote %zu bytes, not %zu\n", wrote, (size_t)kQuantRows * rs); return 1; }
    ggml_quantize_free();

    // Random-code rows: every byte random, then each block's f16 d replaced by a finite
    // positive half in [2^-10, 2^-2).
    for (size_t i = (size_t)kQuantRows * rs; i < w.size(); ++i) w[i] = (uint8_t)rng.next();
    for (size_t b = (size_t)kQuantRows * rs; b < w.size(); b += kBlock) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(std::ldexp(1.0f + rng.unit(), -10 + (int)(rng.next() % 8)));
        memcpy(&w[b], &h, sizeof h);
    }

    FILE *f = fopen((std::string(kDataDir) + "/ref/attn_norm-0.0.f32").c_str(), "rb");
    if (!f) { fprintf(stderr, "iq3s_ref: no oracle dump\n"); return 1; }
    std::vector<float> col(kK);
    if (fread(col.data(), 4, kK, f) != (size_t)kK) { fprintf(stderr, "iq3s_ref: short dump\n"); return 1; }
    fclose(f);

    ggml_type_traits_t q8k = ggml_internal_get_type_traits(GGML_TYPE_Q8_K);
    if (!q8k.from_float) { fprintf(stderr, "iq3s_ref: no from_float for q8_K\n"); return 1; }
    std::vector<uint8_t> y(ggml_row_size(GGML_TYPE_Q8_K, kK));
    q8k.from_float(col.data(), y.data(), kK);

    std::array<mul_mat_t, IQK_MAX_NY> kernels{};
    mul_mat_t func16 = nullptr;
    if (!iqk_set_kernels_iquants(kK, GGML_TYPE_IQ3_S, GGML_TYPE_Q8_K, kernels, func16) || !kernels[0]) {
        fprintf(stderr, "iq3s_ref: ik declined the IQ3_S x Q8_K pairing (k=%d)\n", kK);
        return 1;
    }
    std::vector<float> dst(kDotRows);
    DataInfo info;
    info.s = dst.data();
    info.cy = (const char *)y.data();
    info.bs = kDotRows;    // one dst row of kDotRows outputs
    info.by = y.size();    // one activation row
    info.cur_y = 0;
    info.ne11 = 1;
    info.row_mapping = nullptr;
    info.bs2 = 0;
    // bx is the ROW STRIDE (BaseDequantizer::new_row walks vx + bx*ix).
    kernels[0](kK, w.data(), rs, info, kDotRows);

    std::string dot = "tensor synthetic-iq3_s k " + std::to_string(kK) + "\n";
    char line[32];
    for (size_t i = 0; i < y.size(); ++i) {
        snprintf(line, sizeof line, "%02x", y[i]);
        dot += line;
    }
    dot += "\n";
    for (int r = 0; r < kDotRows; ++r) {
        uint32_t bits; memcpy(&bits, &dst[r], 4);
        snprintf(line, sizeof line, "row %d %08x\n", r, bits);
        dot += line;
    }
    for (int r = 0; r < kDotRows; ++r) {
        dot += "w " + std::to_string(r) + " ";
        for (size_t i = 0; i < rs; ++i) {
            snprintf(line, sizeof line, "%02x", w[(size_t)r * rs + i]);
            dot += line;
        }
        dot += "\n";
    }
    if (!write_atomic(std::string(kDataDir) + "/ref/iq3s-ik-dot.txt", dot.data(), dot.size())) return 1;
    printf("dumped %d synthetic rows (%d quantized + %d random-code) of iq3_s k=%d (q8_K pairing)\n",
           kDotRows, kQuantRows, kRandomRows, kK);
    ggml_free(init_ctx);
    return 0;
}
