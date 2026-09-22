// Run ik's OWN Q5_K x4 kernel — the pairing the oracle dispatches on this CPU
// (iqk_gemm_kquants.cpp:2752/2777: mul_mat_qX_K_q8_2_X4_T<DequantizerQ5K_AVX2>,
// expected_type_B = GGML_TYPE_Q8_2_X4) — on the first aligned Q5_K tensor's
// first 64 rows, with ik's quantize_row_q8_2_x4 coding the same column the
// Rust gate quantizes. Same shape and output format as q4k_x4_ref.cpp: a tensor
// header, one long hex line of ik's activation bytes, then `row R %08x` lines.
//
// The GGUF is argv[1], not the reference model: V2-Lite carries no Q5_K
// tensor. build-qdot-ref.sh passes the V4.1 first shard, whose first Q5_K
// tensor is blk.0.ffn_down_exps.weight (k = 2304).
//
// The activation column is the first k = 2304 f32 of the attn_norm-0 dump the
// other x4 harnesses read. That dump is 2048 values per token, so the column
// is token 0 followed by the first 256 values of token 1: a column with the
// real value distribution, not a token-aligned one.
//
// It also dumps ggml's own to_float of the same tensor's first 4 rows
// (ggml_internal_get_type_traits(GGML_TYPE_Q5_K).to_float, as dequant_ref.cpp
// does) to q5k-v41-dequant.raw, with a .meta in dequant_ref.cpp's format. The
// q5_K.raw and manifest.txt names belong to dequant_ref.cpp and are not
// written here.
#include "ggml.h"
#include "ggml-backend.h"

#define IQK_IMPLEMENT
#include "iqk_gemm_kquants.h"

#include <cerrno>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "ref_paths.h"

// The data directory the gates read (tools/box.sh exports BLOOMERY_DATA; a parallel
// track moves both this harness's input and its dumps by setting it).
static const char *kDataDir = ref_data_dir();

// Rows ik's kernel is run on (gate B), and rows ggml dequantizes (the dequant gate,
// dequant_ref.cpp's count).
static const int kDotRows = 64;
static const int kDequantRows = 4;

static std::vector<uint8_t> read_range(const char *p, int64_t off, size_t n) {
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", p); exit(1); }
    std::vector<uint8_t> b(n);
    if (fseeko(f, (off_t)off, SEEK_SET) || fread(b.data(), 1, n, f) != n) { fprintf(stderr, "read failed\n"); exit(1); }
    fclose(f);
    return b;
}

// Write to <path>.tmp.<pid> and rename: a reader (gate-qdot on another track, same
// shared $BLOOMERY_DATA) must never see a half-written dump. rename(2) within one
// directory is atomic, so the final path is either the old dump or the new one; the pid
// suffix keeps two tracks running build-ref at once from renaming each other's half file.
static bool write_atomic(const std::string &path, const void *data, size_t n) {
    const std::string tmp = path + ".tmp." + std::to_string(getpid());
    FILE *out = fopen(tmp.c_str(), "wb");
    if (!out) {
        fprintf(stderr, "q5k_x4_ref: cannot open %s for writing: %s\n", tmp.c_str(), strerror(errno));
        return false;
    }
    const bool wrote = fwrite(data, 1, n, out) == n;
    if (fclose(out) != 0 || !wrote) {
        fprintf(stderr, "q5k_x4_ref: cannot write %s: %s\n", tmp.c_str(), strerror(errno));
        return false;
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        fprintf(stderr, "q5k_x4_ref: cannot rename %s to %s: %s\n", tmp.c_str(), path.c_str(), strerror(errno));
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: q5k_x4_ref <model.gguf>  (a file with a Q5_K tensor; V2-Lite has none)\n");
        return 64;
    }
    const char *path = argv[1];

    struct ggml_context *gctx = nullptr;
    struct gguf_init_params gp = {true, &gctx};
    struct gguf_context *gguf = gguf_init_from_file(path, gp);
    if (!gguf || !gctx) { fprintf(stderr, "gguf open failed: %s\n", path); return 1; }
    const size_t data_off = gguf_get_data_offset(gguf);

    // First Q5_K tensor with k % 256 == 0 — the same scan the Rust gate does.
    int pick = -1;
    for (int i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        const char *nm = gguf_get_tensor_name(gguf, i);
        struct ggml_tensor *c = ggml_get_tensor(gctx, nm);
        if (c && c->type == GGML_TYPE_Q5_K && c->ne[0] % 256 == 0) { pick = i; break; }
    }
    if (pick < 0) { fprintf(stderr, "no Q5_K tensor in %s\n", path); return 1; }
    // Copy the name BEFORE freeing: gguf_get_tensor_name points into the
    // context, so a name read after gguf_free is freed memory.
    std::string tensor_name = gguf_get_tensor_name(gguf, pick);
    struct ggml_tensor *t = ggml_get_tensor(gctx, tensor_name.c_str());
    const int k = (int)t->ne[0];
    const size_t rs = ggml_row_size(t->type, k);
    const int n_dims = ggml_n_dims(t);
    const int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    if (ne[1] * ne[2] * ne[3] < kDotRows) { fprintf(stderr, "%s has fewer than %d rows\n", tensor_name.c_str(), kDotRows); return 1; }
    fprintf(stderr, "tensor %s k=%d rowsz=%zu\n", tensor_name.c_str(), k, rs);
    std::vector<uint8_t> w = read_range(path, (int64_t)(data_off + gguf_get_tensor_offset(gguf, pick)), rs * kDotRows);
    gguf_free(gguf);
    ggml_free(gctx);
    const char *nm = tensor_name.c_str();

    FILE *f = fopen((std::string(kDataDir) + "/ref/attn_norm-0.0.f32").c_str(), "rb");
    if (!f) { fprintf(stderr, "no oracle dump\n"); return 1; }
    std::vector<float> x(k);
    if (fread(x.data(), 4, k, f) != (size_t)k) { fprintf(stderr, "short dump\n"); return 1; }
    fclose(f);

    // ik's own q8_2_x4 coding of the column: 144 bytes per 128 values
    // (x4 groups of four 32-value sub-blocks; k % 256 == 0 means no tail).
    std::vector<uint8_t> y(144 * (k / 128));
    quantize_row_q8_2_x4(x.data(), y.data(), k);

    // The x4 kernel through ik's own dispatch table: kernels[0] is the
    // nrc_y = 1 instantiation of mul_mat_qX_K_q8_2_X4_T<DequantizerQ5K_AVX2>.
    std::array<mul_mat_t, IQK_MAX_NY> kernels{};
    mul_mat_t func16 = nullptr;
    if (!iqk_set_kernels_kquants(k, GGML_TYPE_Q5_K, GGML_TYPE_Q8_2_X4, kernels, func16) || !kernels[0]) {
        fprintf(stderr, "ik declined the Q5_K x Q8_2_X4 pairing (k=%d)\n", k);
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
    // bx is the ROW STRIDE, not an offset — BaseDequantizer::new_row walks
    // `vx + bx*ix` (iqk_common.h:368), and the real caller passes the row
    // size (iqk_mul_mat.cpp:96).
    kernels[0](k, w.data(), rs, info, kDotRows);

    // ggml's own dequantization of the first rows, row by row as dequant_ref.cpp does.
    ggml_type_traits_t traits = ggml_internal_get_type_traits(GGML_TYPE_Q5_K);
    if (!traits.to_float) { fprintf(stderr, "no to_float for q5_K\n"); return 1; }
    std::vector<float> deq((size_t)kDequantRows * k);
    for (int r = 0; r < kDequantRows; ++r) {
        traits.to_float(w.data() + (size_t)r * rs, deq.data() + (size_t)r * k, k);
    }

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

    std::string meta = "tensor=" + tensor_name + "\n";
    meta += "type=" + std::to_string((int)GGML_TYPE_Q5_K) + " " + ggml_type_name(GGML_TYPE_Q5_K) + "\n";
    meta += "dims=" + std::to_string(n_dims);
    for (int d = 0; d < 4; ++d) meta += " " + std::to_string((long long)ne[d]);
    meta += "\n";
    meta += "rows=" + std::to_string(kDequantRows) + "\n";
    meta += "rowlen=" + std::to_string(k) + "\n";

    const std::string ref = std::string(kDataDir) + "/ref/";
    if (!write_atomic(ref + "q5k-x4-ik-dot.txt", dot.data(), dot.size()) ||
        !write_atomic(ref + "q5k-v41-dequant.raw", deq.data(), deq.size() * sizeof(float)) ||
        !write_atomic(ref + "q5k-v41-dequant.meta", meta.data(), meta.size())) {
        return 1;
    }
    printf("dumped %d rows of %s (x4 pairing) and ggml's to_float of its first %d rows\n", kDotRows, nm, kDequantRows);
    return 0;
}
