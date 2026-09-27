// Run ik's OWN Q8_0 kernel — the pairing the oracle dispatches on this CPU — on a synthetic
// block set: 64 rows of k = 2144 (67 blocks: 16 whole x4 groups and THREE tail blocks, the
// most a q8_2_x4 column carries), with ik's quantize_row_q8_2_x4 coding one synthetic column.
//
// The pairing: ggml.c:819 — [GGML_TYPE_Q8_0].vec_dot_type = GGML_TYPE_Q8_2_X4 under __AVX2__
// with GGML_USE_IQK_MULMAT; iqk_mul_mat.cpp:936/946 routes Q8_0 to
// iqk_set_kernels_legacy_quants (expected_typeB Q8_2_X4, :2481), whose case Q8_0 (:2503) picks
// set_functions<Q8_0_Unpacker> on a build without HAVE_FANCY_SIMD (AVX512F+VNNI+VL+BW+DQ; this
// CPU has none), i.e. mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, nrc_y, block_q8_2> (:404, :2438):
// ScaleHelperQ_0 x ScaleHelperQ8_2S, Sum4TypeQ82S (the sign trick), MinusType0.
//
// Synthetic, not a model's row: every input the kernel can meet at once, deterministic, and no
// model file read. Weight blocks: xorshift codes (-128 among them), f16 scales normal of both
// signs, zero on every 23rd block, subnormal on every 29th, every code -128 on every 17th.
// Column: blocks of magnitudes 1e-4 .. 1e4, block 5 zero (both zero signs), and block 9 built
// so its bf16 scale is the subnormal 2^-127 and its most negative value codes to -128 — the
// one code where the sign trick wraps.
//
// Output (q8f0-ik-dot.txt): `tensor synthetic-q8_0 k K`, then one tagged hex line each —
// `x` the column's f32 bits, `a` ik's q8_2_x4 bytes of it, `w R` row R's bytes — then
// `row R %08x` lines of ik's result. Every line carries a tag, so the Rust gate reads the
// inputs from this file and the generator lives here only.
#include "ggml.h"
#include "ggml-backend.h"

#define IQK_IMPLEMENT
#include "iqk_gemm_legacy_quants.h"

#include <array>
#include <cerrno>
#include <cmath>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "ref_paths.h"

static const char *kDataDir = ref_data_dir();

static uint64_t g_state = 0x8f0d0c5eedull;
static uint64_t next_u64() {
    g_state ^= g_state << 13;
    g_state ^= g_state >> 7;
    g_state ^= g_state << 17;
    return g_state;
}
// Uniform in [-1, 1).
static float unit() { return (float)((double)(next_u64() >> 11) / 4503599627370496.0 - 1.0); }

static void hex(FILE *out, const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; ++i) fprintf(out, "%02x", b[i]);
}

int main() {
    // ik's tail path converts the weight scale with GGML_FP16_TO_FP32, on an F16C build the
    // lookup in ggml_table_f32_f16, which ggml_init fills; no gguf is opened here to do it.
    ggml_free(ggml_init({0, nullptr, true}));
    const int k = 2144;
    const int nb = k / 32;
    const int rows = 64;
    const size_t rs = 34 * (size_t)nb;

    std::vector<uint8_t> w(rs * rows);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            uint8_t *blk = w.data() + r * rs + 34 * b;
            const int n = r * nb + b;
            uint16_t d;
            if (n % 23 == 0) {
                d = 0;
            } else if (n % 29 == 0) {
                d = (uint16_t)(1 + next_u64() % 0x3ff);           // subnormal f16
            } else {
                const uint16_t e = (uint16_t)(3 + next_u64() % 12); // 2^-12 .. 2^-1
                d = (uint16_t)((e << 10) | (next_u64() & 0x3ff) | ((next_u64() & 1) << 15));
            }
            memcpy(blk, &d, 2);
            for (int m = 0; m < 32; ++m) blk[2 + m] = (n % 17 == 3) ? 0x80 : (uint8_t)next_u64();
        }
    }
    {
        // A table ggml_init did not fill reads 0 for every scale: refuse by name before the
        // kernel runs, instead of dumping rows of plausible zeros.
        uint16_t d_tail;
        memcpy(&d_tail, w.data() + 34 * (nb - 1), 2);
        if ((d_tail & 0x7fff) != 0 && GGML_FP16_TO_FP32(d_tail) == 0.0f) {
            fprintf(stderr, "q8f0_ref: the f16 table reads 0 for %04x — ggml_init did not fill it\n", d_tail);
            return 1;
        }
    }

    std::vector<float> x(k);
    for (int b = 0; b < nb; ++b) {
        const float mag = std::pow(10.0f, (float)(b % 9) - 4.0f);
        for (int m = 0; m < 32; ++m) x[32 * b + m] = unit() * mag;
    }
    for (int m = 0; m < 32; ++m) x[32 * 5 + m] = (m & 1) ? -0.0f : 0.0f;
    {
        // amax/127 = 2^-127 (1 + 0.99 * 2^-7): its bf16 rounds down to the subnormal 2^-127, so
        // v / d = -127 (1 + 0.99 * 2^-7) ~ -127.98 codes to -128.
        const float amax = 127.0f * std::ldexp(1.0f + 0.99f * std::ldexp(1.0f, -7), -127);
        for (int m = 0; m < 32; ++m) x[32 * 9 + m] = unit() * amax * 0.5f;
        x[32 * 9] = -amax;
    }

    const size_t ysz = 144 * (k / 128) + 36 * ((k % 128) / 32);
    std::vector<uint8_t> y(ysz);
    quantize_row_q8_2_x4(x.data(), y.data(), k);

    // kernels[0] is the nrc_y = 1 instantiation of mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 1, block_q8_2>.
    std::array<mul_mat_t, IQK_MAX_NY> kernels{};
    mul_mat_t func16 = nullptr;
    if (!iqk_set_kernels_legacy_quants(k, GGML_TYPE_Q8_0, GGML_TYPE_Q8_2_X4, kernels, func16) || !kernels[0]) {
        fprintf(stderr, "ik declined the Q8_0 x Q8_2_X4 pairing (k=%d)\n", k);
        return 1;
    }
    std::vector<float> dst(rows);
    DataInfo info;
    info.s = dst.data();
    info.cy = (const char *)y.data();
    info.bs = rows;
    info.by = y.size();
    info.cur_y = 0;
    info.ne11 = 1;
    info.row_mapping = nullptr;
    info.bs2 = 0;
    // bx is the row stride (iqk_common.h:368).
    kernels[0](k, w.data(), rs, info, rows);

    // <path>.tmp.<pid> then rename: a reader on another track never sees a half-written dump.
    const std::string out_p = std::string(kDataDir) + "/ref/q8f0-ik-dot.txt";
    const std::string tmp_p = out_p + ".tmp." + std::to_string(getpid());
    FILE *out = fopen(tmp_p.c_str(), "w");
    if (!out) {
        fprintf(stderr, "q8f0_ref: cannot open %s for writing: %s\n", tmp_p.c_str(), strerror(errno));
        return 1;
    }
    fprintf(out, "tensor synthetic-q8_0 k %d\n", k);
    fprintf(out, "x ");
    hex(out, x.data(), 4 * (size_t)k);
    fprintf(out, "\na ");
    hex(out, y.data(), y.size());
    fprintf(out, "\n");
    for (int r = 0; r < rows; ++r) {
        fprintf(out, "w %d ", r);
        hex(out, w.data() + r * rs, rs);
        fprintf(out, "\n");
    }
    for (int r = 0; r < rows; ++r) {
        uint32_t bits;
        memcpy(&bits, &dst[r], 4);
        fprintf(out, "row %d %08x\n", r, bits);
    }
    if (fclose(out) != 0) {
        fprintf(stderr, "q8f0_ref: cannot write %s: %s\n", tmp_p.c_str(), strerror(errno));
        return 1;
    }
    if (rename(tmp_p.c_str(), out_p.c_str()) != 0) {
        fprintf(stderr, "q8f0_ref: cannot rename %s to %s: %s\n", tmp_p.c_str(), out_p.c_str(), strerror(errno));
        return 1;
    }
    // The table the tail read, on the last block of row 0: bits and value, never 0 for a
    // nonzero scale.
    uint16_t d_tail;
    memcpy(&d_tail, w.data() + 34 * (nb - 1), 2);
    printf("tail scale row 0 block %d: f16 %04x = %.9g\n", nb - 1, d_tail, GGML_FP16_TO_FP32(d_tail));
    printf("dumped %d synthetic Q8_0 rows (k=%d, q8_2_x4 pairing, %zu-byte column with %d tail blocks)\n",
           rows, k, ysz, (k % 128) / 32);
    return 0;
}
