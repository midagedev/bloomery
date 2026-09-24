// Run ik's OWN MXFP4 kernel — the pairing the oracle dispatches on this CPU — on the first
// aligned MXFP4 tensor's first 64 rows, with ik's quantize_row_q8_2_x4 coding the column the
// Rust gate quantizes. Same shape and output format as q5f0_ref.cpp: a tensor header, one
// long hex line of ik's activation bytes, then `row R %08x` lines. mxfp4_ref.cpp is the
// dequant twin (ggml's to_float of the DSpark draft's experts, for gate-dspark-read); this
// one is the dot.
//
// The pairing: ggml.c:1316 — [GGML_TYPE_MXFP4].vec_dot_type = GGML_TYPE_Q8_2_X4 under
// __AVX2__; iqk_mul_mat.cpp:945 routes MXFP4 to iqk_set_kernels_legacy_quants, whose
// expected_typeB is Q8_2_X4 (iqk_gemm_legacy_quants.cpp:2483) and whose case MXFP4 (:2517)
// picks set_functions<MXFP4_Unpacker>, i.e. mul_mat_qX_1_q8_2_T<MXFP4_Unpacker, nrc_y>
// (:2451-2454) — the Q5_0 template with ScaleHelperQ_0_1_MXFP4<12> and the unsigned code
// table.
//
// The GGUF is argv[1]: build-qdot-ref.sh passes the V4-Flash first data shard, whose first
// MXFP4 tensor is blk.0.ffn_down_exps.weight (k = 2048, 16 whole x4 groups, no tail). The
// column is the first k = 2048 f32 of the attn_norm-0 dump — token 0.
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
    if (!f) { fprintf(stderr, "mxfp4_x4_ref: open %s failed\n", p); exit(1); }
    std::vector<uint8_t> b(n);
    if (fseeko(f, (off_t)off, SEEK_SET) || fread(b.data(), 1, n, f) != n) { fprintf(stderr, "mxfp4_x4_ref: read failed\n"); exit(1); }
    fclose(f);
    return b;
}

// Write to <path>.tmp.<pid> and rename, as q5k_x4_ref.cpp does: a gate on another track
// reading the shared $BLOOMERY_DATA never sees a half-written dump.
static bool write_atomic(const std::string &path, const void *data, size_t n) {
    const std::string tmp = path + ".tmp." + std::to_string(getpid());
    FILE *out = fopen(tmp.c_str(), "wb");
    if (!out) {
        fprintf(stderr, "mxfp4_x4_ref: cannot open %s for writing: %s\n", tmp.c_str(), strerror(errno));
        return false;
    }
    const bool wrote = fwrite(data, 1, n, out) == n;
    if (fclose(out) != 0 || !wrote) {
        fprintf(stderr, "mxfp4_x4_ref: cannot write %s: %s\n", tmp.c_str(), strerror(errno));
        return false;
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        fprintf(stderr, "mxfp4_x4_ref: cannot rename %s to %s: %s\n", tmp.c_str(), path.c_str(), strerror(errno));
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: mxfp4_x4_ref <model.gguf>  (a file with an MXFP4 tensor)\n");
        return 64;
    }
    const char *path = argv[1];

    struct ggml_context *gctx = nullptr;
    struct gguf_init_params gp = {true, &gctx};
    struct gguf_context *gguf = gguf_init_from_file(path, gp);
    if (!gguf || !gctx) { fprintf(stderr, "mxfp4_x4_ref: gguf open failed: %s\n", path); return 1; }
    const size_t data_off = gguf_get_data_offset(gguf);

    // First MXFP4 tensor with k % 32 == 0 — the same scan the Rust gate does.
    int pick = -1;
    for (int i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        struct ggml_tensor *c = ggml_get_tensor(gctx, gguf_get_tensor_name(gguf, i));
        if (c && c->type == GGML_TYPE_MXFP4 && c->ne[0] % 32 == 0) { pick = i; break; }
    }
    if (pick < 0) { fprintf(stderr, "mxfp4_x4_ref: no MXFP4 tensor in %s\n", path); return 1; }
    // Copy the name BEFORE freeing: it points into the context.
    const std::string tensor_name = gguf_get_tensor_name(gguf, pick);
    struct ggml_tensor *t = ggml_get_tensor(gctx, tensor_name.c_str());
    const int k = (int)t->ne[0];
    const size_t rs = ggml_row_size(t->type, k);
    if (t->ne[1] * t->ne[2] * t->ne[3] < kDotRows) { fprintf(stderr, "mxfp4_x4_ref: %s has fewer than %d rows\n", tensor_name.c_str(), kDotRows); return 1; }
    fprintf(stderr, "tensor %s k=%d rowsz=%zu\n", tensor_name.c_str(), k, rs);
    std::vector<uint8_t> w = read_range(path, (int64_t)(data_off + gguf_get_tensor_offset(gguf, pick)), rs * kDotRows);
    gguf_free(gguf);
    ggml_free(gctx);

    FILE *f = fopen((std::string(kDataDir) + "/ref/attn_norm-0.0.f32").c_str(), "rb");
    if (!f) { fprintf(stderr, "mxfp4_x4_ref: no oracle dump\n"); return 1; }
    std::vector<float> x(k);
    if (fread(x.data(), 4, k, f) != (size_t)k) { fprintf(stderr, "mxfp4_x4_ref: short dump\n"); return 1; }
    fclose(f);

    // ik's own q8_2_x4 coding: 144 bytes per 128 values, 36-byte q8_2 tails past the groups.
    std::vector<uint8_t> y(144 * (k / 128) + 36 * ((k % 128) / 32));
    quantize_row_q8_2_x4(x.data(), y.data(), k);

    std::array<mul_mat_t, IQK_MAX_NY> kernels{};
    mul_mat_t func16 = nullptr;
    if (!iqk_set_kernels_legacy_quants(k, GGML_TYPE_MXFP4, GGML_TYPE_Q8_2_X4, kernels, func16) || !kernels[0]) {
        fprintf(stderr, "mxfp4_x4_ref: ik declined the MXFP4 x Q8_2_X4 pairing (k=%d)\n", k);
        return 1;
    }
    std::vector<float> dst(kDotRows);
    DataInfo info;
    info.s = dst.data();
    info.cy = (const char *)y.data();
    info.bs = kDotRows;    // one dst row of 64 outputs
    info.by = y.size();    // one activation row
    info.cur_y = 0;
    info.ne11 = 1;
    info.row_mapping = nullptr;
    info.bs2 = 0;
    // bx is the ROW STRIDE (Q_Unpacker::set_row walks cx_0 + ix*bx).
    kernels[0](k, w.data(), rs, info, kDotRows);

    std::string dot = "tensor " + tensor_name + " k " + std::to_string(k) + "\n";
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
    if (!write_atomic(std::string(kDataDir) + "/ref/mxfp4-x4-ik-dot.txt", dot.data(), dot.size())) return 1;
    printf("dumped %d rows of %s (q8_2_x4 pairing)\n", kDotRows, tensor_name.c_str());
    return 0;
}
