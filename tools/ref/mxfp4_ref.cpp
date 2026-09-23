// mxfp4_ref.cpp — ggml's own MXFP4 dequantization of the DSpark draft's experts, the reference
// gate-dspark-read compares our decode with.
//
// Opens the draft GGUF (argv[1]), finds one MXFP4 expert tensor (argv[2], default
// blk.0.ffn_gate_exps.weight) and dumps ggml's to_float of its first kRows rows
// (ggml_internal_get_type_traits(GGML_TYPE_MXFP4).to_float, which is dequantize_row_mxfp4 —
// ggml.c:1311), as dequant_ref.cpp does per type:
//   $BLOOMERY_DATA/ref/mxfp4-dspark-dequant.raw   f32 rows, row-major
//   $BLOOMERY_DATA/ref/mxfp4-dspark-dequant.meta  tensor, type, dims, rows, rowlen, ggml build
// The first rows of an expert tensor are its first expert's first rows. kRows is sized so the
// rows use all 16 codes (the gate checks that, so a wrong table entry cannot hide in an unused
// code).
//
// Build and run: bash tools/ref/build-qdot-ref.sh (just build-ref). REF_GGML_BUILD names the ik
// tree and commit the harness links; the build script defines it.

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#include "ggml.h"
#include "ref_paths.h"

#ifndef REF_GGML_BUILD
#error "REF_GGML_BUILD must name the ggml build (tools/ref/build-qdot-ref.sh defines it)"
#endif

static const char *kDataDir = ref_data_dir();
static const int64_t kRows = 64;

static std::vector<uint8_t> read_range(const char *p, int64_t off, size_t n) {
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "mxfp4_ref: open %s failed\n", p); exit(1); }
    std::vector<uint8_t> b(n);
    if (fseeko(f, (off_t)off, SEEK_SET) || fread(b.data(), 1, n, f) != n) {
        fprintf(stderr, "mxfp4_ref: read of %s failed\n", p);
        exit(1);
    }
    fclose(f);
    return b;
}

// Write to <path>.tmp.<pid> and rename, as q5k_x4_ref.cpp does: a gate on another track reading
// the shared $BLOOMERY_DATA never sees a half-written dump.
static bool write_atomic(const std::string &path, const void *data, size_t n) {
    const std::string tmp = path + ".tmp." + std::to_string(getpid());
    FILE *out = fopen(tmp.c_str(), "wb");
    if (!out) {
        fprintf(stderr, "mxfp4_ref: cannot open %s for writing: %s\n", tmp.c_str(), strerror(errno));
        return false;
    }
    const bool wrote = fwrite(data, 1, n, out) == n;
    if (fclose(out) != 0 || !wrote) {
        fprintf(stderr, "mxfp4_ref: cannot write %s: %s\n", tmp.c_str(), strerror(errno));
        return false;
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        fprintf(stderr, "mxfp4_ref: cannot rename %s to %s: %s\n", tmp.c_str(), path.c_str(), strerror(errno));
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: mxfp4_ref <draft.gguf> [tensor]\n");
        return 64;
    }
    const char *path = argv[1];
    const std::string tensor_name = argc > 2 ? argv[2] : "blk.0.ffn_gate_exps.weight";

    struct ggml_context *gctx = nullptr;
    struct gguf_init_params gp = {true, &gctx};
    struct gguf_context *gguf = gguf_init_from_file(path, gp);
    if (!gguf || !gctx) { fprintf(stderr, "mxfp4_ref: gguf open failed: %s\n", path); return 1; }
    const int tidx = gguf_find_tensor(gguf, tensor_name.c_str());
    if (tidx < 0) { fprintf(stderr, "mxfp4_ref: no %s in %s\n", tensor_name.c_str(), path); return 1; }
    struct ggml_tensor *t = ggml_get_tensor(gctx, tensor_name.c_str());
    if (!t || t->type != GGML_TYPE_MXFP4) {
        fprintf(stderr, "mxfp4_ref: %s is not an MXFP4 tensor\n", tensor_name.c_str());
        return 1;
    }
    const int64_t k = t->ne[0];
    const int64_t rows_total = t->ne[1] * t->ne[2] * t->ne[3];
    if (rows_total < kRows) { fprintf(stderr, "mxfp4_ref: %s has fewer than %lld rows\n", tensor_name.c_str(), (long long)kRows); return 1; }
    const size_t rs = ggml_row_size(t->type, k);
    const int n_dims = ggml_n_dims(t);
    const int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    const size_t off = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, tidx);
    std::vector<uint8_t> w = read_range(path, (int64_t)off, rs * kRows);
    gguf_free(gguf);
    ggml_free(gctx);

    ggml_type_traits_t traits = ggml_internal_get_type_traits(GGML_TYPE_MXFP4);
    if (!traits.to_float) { fprintf(stderr, "mxfp4_ref: no to_float for mxfp4\n"); return 1; }
    std::vector<float> deq((size_t)kRows * k);
    for (int64_t r = 0; r < kRows; ++r) {
        traits.to_float(w.data() + (size_t)r * rs, deq.data() + (size_t)r * k, k);
    }

    std::string meta = "tensor=" + tensor_name + "\n";
    meta += "type=" + std::to_string((int)GGML_TYPE_MXFP4) + " " + ggml_type_name(GGML_TYPE_MXFP4) + "\n";
    meta += "dims=" + std::to_string(n_dims);
    for (int d = 0; d < 4; ++d) meta += " " + std::to_string((long long)ne[d]);
    meta += "\n";
    meta += "rows=" + std::to_string((long long)kRows) + "\n";
    meta += "rowlen=" + std::to_string((long long)k) + "\n";
    meta += std::string("ggml=") + REF_GGML_BUILD + "\n";

    const std::string ref = std::string(kDataDir) + "/ref/";
    if (!write_atomic(ref + "mxfp4-dspark-dequant.raw", deq.data(), deq.size() * sizeof(float)) ||
        !write_atomic(ref + "mxfp4-dspark-dequant.meta", meta.data(), meta.size())) {
        return 1;
    }
    printf("dumped ggml's to_float of the first %lld rows of %s (k = %lld, %s)\n",
           (long long)kRows, tensor_name.c_str(), (long long)k, REF_GGML_BUILD);
    return 0;
}
