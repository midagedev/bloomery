// Run ik's OWN IQ4_NL kernel — the pairing the oracle dispatches on this CPU — on the
// first aligned IQ4_NL tensor's first rows, with ik's quantize_row_q8_2_x4 coding the
// column the Rust gate quantizes. Output format extends q5f1_ref.cpp's dump by one line
// kind: a tensor header, one long hex line of ik's activation bytes, then `w R <hex>` row
// bytes beside the `row R <bits>` results — the Rust gates read the rows from the dump, so
// they need no model file and stay correct on tensors with fewer than 64 rows.
//
// The pairing: ggml.c:1286-1288 — [GGML_TYPE_IQ4_NL].vec_dot_type = GGML_TYPE_Q8_2_X4
// under __AVX2__ + GGML_USE_IQK_MULMAT (no-AVX2 builds say Q8_0_X4 and are not this
// oracle); iqk_gemm_legacy_quants.cpp:2634-2637 — the entry requires ne00 % 32 == 0 and
// typeB = Q8_2_X4, and case IQ4_NL picks IQ4_NL_UnpackerS on a non-HAVE_FANCY_SIMD build
// (:2665-2667), which set_functions routes (:2596-2597) to mul_mat_qX_0_q8_0_T<IQ4_NL_
// UnpackerS, nrc_y, block_q8_2> (:407-413) — AccumT<MinusType0> + ScaleHelperQ8_2S +
// Sum4TypeQ82S: the signed iq4k_values table, the sign fold into the activation, no min.
//
// The GGUF is argv[1]: build-qdot-ref.sh passes the Qwen3.8 UD-Q4_K_XL second shard
// (BLOOMERY_QWEN_Q4_MODEL), whose first IQ4_NL tensor is per_layer_token_embd.weight.
// The column is the first k f32 of the attn_norm-0 dump the other harnesses read: the
// real value distribution, not a token-aligned column.
#include "ggml.h"
#include "ggml-backend.h"

#define IQK_IMPLEMENT
#include "iqk_gemm_legacy_quants.h"

#include <cerrno>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "ref_paths.h"

static const char *kDataDir = ref_data_dir();
static const int kDotRows = 64;

static std::vector<uint8_t> read_range(const char *p, int64_t off, size_t n) {
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "iq4nl_ref: open %s failed\n", p); exit(1); }
    std::vector<uint8_t> b(n);
    if (fseeko(f, (off_t)off, SEEK_SET) || fread(b.data(), 1, n, f) != n) { fprintf(stderr, "iq4nl_ref: read failed\n"); exit(1); }
    fclose(f);
    return b;
}

// Write to <path>.tmp.<pid> and rename, as q5k_x4_ref.cpp does: a gate on another track
// reading the shared $BLOOMERY_DATA never sees a half-written dump.
static bool write_atomic(const std::string &path, const void *data, size_t n) {
    const std::string tmp = path + ".tmp." + std::to_string(getpid());
    FILE *out = fopen(tmp.c_str(), "wb");
    if (!out) {
        fprintf(stderr, "iq4nl_ref: cannot open %s for writing: %s\n", tmp.c_str(), strerror(errno));
        return false;
    }
    const bool wrote = fwrite(data, 1, n, out) == n;
    if (fclose(out) != 0 || !wrote) {
        fprintf(stderr, "iq4nl_ref: cannot write %s: %s\n", tmp.c_str(), strerror(errno));
        return false;
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        fprintf(stderr, "iq4nl_ref: cannot rename %s to %s: %s\n", tmp.c_str(), path.c_str(), strerror(errno));
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: iq4nl_ref <model.gguf>  (a file with an IQ4_NL tensor)\n");
        return 64;
    }
    const char *path = argv[1];

    struct ggml_context *gctx = nullptr;
    struct gguf_init_params gp = {true, &gctx};
    struct gguf_context *gguf = gguf_init_from_file(path, gp);
    if (!gguf || !gctx) { fprintf(stderr, "iq4nl_ref: gguf open failed: %s\n", path); return 1; }
    const size_t data_off = gguf_get_data_offset(gguf);

    // First IQ4_NL tensor with k % 32 == 0 and enough rows — the same scan the Rust
    // gate's dump checks pin afterwards.
    int pick = -1;
    for (int i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        struct ggml_tensor *c = ggml_get_tensor(gctx, gguf_get_tensor_name(gguf, i));
        if (c && c->type == GGML_TYPE_IQ4_NL && c->ne[0] % 32 == 0
                && c->ne[1] * c->ne[2] * c->ne[3] >= 16) { pick = i; break; }
    }
    if (pick < 0) { fprintf(stderr, "iq4nl_ref: no aligned IQ4_NL tensor in %s\n", path); return 1; }
    // Copy the name BEFORE freeing: it points into the context.
    const std::string tensor_name = gguf_get_tensor_name(gguf, pick);
    struct ggml_tensor *t = ggml_get_tensor(gctx, tensor_name.c_str());
    const int k = (int)t->ne[0];
    const size_t rs = ggml_row_size(t->type, k);
    const int nrows = (int)(t->ne[1] * t->ne[2] * t->ne[3]);
    const int dot_rows = nrows < kDotRows ? nrows : kDotRows;
    fprintf(stderr, "tensor %s k=%d rows=%d rowsz=%zu\n", tensor_name.c_str(), k, nrows, rs);
    std::vector<uint8_t> w = read_range(path, (int64_t)(data_off + gguf_get_tensor_offset(gguf, pick)), rs * dot_rows);
    gguf_free(gguf);
    ggml_free(gctx);

    FILE *f = fopen((std::string(kDataDir) + "/ref/attn_norm-0.0.f32").c_str(), "rb");
    if (!f) { fprintf(stderr, "iq4nl_ref: no oracle dump\n"); return 1; }
    std::vector<float> x(k);
    if (fread(x.data(), 4, k, f) != (size_t)k) { fprintf(stderr, "iq4nl_ref: short dump\n"); return 1; }
    fclose(f);

    // ik's own q8_2_x4 coding: 144 bytes per 128 values, 36-byte q8_2 tails past the groups.
    const size_t ysz = 144 * (k / 128) + 36 * ((k % 128) / 32);
    std::vector<uint8_t> y(ysz);
    quantize_row_q8_2_x4(x.data(), y.data(), k);

    std::array<mul_mat_t, IQK_MAX_NY> kernels{};
    mul_mat_t func16 = nullptr;
    if (!iqk_set_kernels_legacy_quants(k, GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_2_X4, kernels, func16) || !kernels[0]) {
        fprintf(stderr, "iq4nl_ref: ik declined the IQ4_NL x Q8_2_X4 pairing (k=%d)\n", k);
        return 1;
    }
    std::vector<float> dst(dot_rows);
    DataInfo info;
    info.s = dst.data();
    info.cy = (const char *)y.data();
    info.bs = dot_rows;    // one dst row of dot_rows outputs
    info.by = y.size();    // one activation row
    info.cur_y = 0;
    info.ne11 = 1;
    info.row_mapping = nullptr;
    info.bs2 = 0;
    // bx is the ROW STRIDE (Q_Unpacker::set_row walks cx_0 + ix*bx).
    kernels[0](k, w.data(), rs, info, dot_rows);

    std::string dot = "tensor " + tensor_name + " k " + std::to_string(k) + "\n";
    char line[32];
    for (size_t i = 0; i < y.size(); ++i) {
        snprintf(line, sizeof line, "%02x", y[i]);
        dot += line;
    }
    dot += "\n";
    for (int r = 0; r < dot_rows; ++r) {
        uint32_t bits; memcpy(&bits, &dst[r], 4);
        snprintf(line, sizeof line, "row %d %08x\n", r, bits);
        dot += line;
    }
    // The weight rows ride along: the Rust gates dot these exact bytes, so a rescan on
    // the Rust side cannot land on another tensor.
    std::string rows_hex;
    for (int r = 0; r < dot_rows; ++r) {
        rows_hex += "w " + std::to_string(r) + " ";
        for (size_t i = 0; i < rs; ++i) {
            snprintf(line, sizeof line, "%02x", w[(size_t)r * rs + i]);
            rows_hex += line;
        }
        rows_hex += "\n";
    }
    dot += rows_hex;
    if (!write_atomic(std::string(kDataDir) + "/ref/iq4nl-ik-dot.txt", dot.data(), dot.size())) return 1;
    printf("dumped %d rows of %s (q8_2_x4 pairing, %zu-byte column with %d tail blocks)\n",
           dot_rows, tensor_name.c_str(), ysz, (k % 128) / 32);
    return 0;
}
