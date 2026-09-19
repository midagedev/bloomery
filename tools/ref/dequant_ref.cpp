// dequant_ref.cpp — ggml dequantization oracle for mulle stage 1, round 1-1.
//
// Opens a GGUF file, finds every distinct ggml type among its tensors, and
// for each type dumps the dequantization of the first rows of one tensor of
// that type, using ggml's own to_float (ggml_internal_get_type_traits):
//   $MULLE_DATA/ref/<type_name>.raw   — f32 reference rows, row-major
//   $MULLE_DATA/ref/<type_name>.meta  — tensor name, type, dims, rows, rowlen
//   $MULLE_DATA/ref/manifest.txt      — "<type_num> <type_name> <count>" per type
// The Rust gate (crates/gguf/tests/oracle.rs) re-derives the same rows from
// its own loader and demands max |diff| <= 1e-6 per type.
//
// Build: bash tools/ref/build-dequant.sh   (on the box, IK=/home/user/ik_llama.cpp)
// Run:   $MULLE_DATA/bin/dequant_ref [model.gguf]

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "ggml.h"

static const char * kGgufPath = "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf";
static const char * kDataDir = getenv("MULLE_DATA") ? getenv("MULLE_DATA") : "/root/mulle-data";
static const int64_t kRows = 4; // first N rows of the chosen tensor, per type

static void fail(const std::string & msg) {
    std::fprintf(stderr, "dequant_ref FATAL: %s\n", msg.c_str());
    std::exit(1);
}

static void write_file(const std::string & path, const void * data, size_t n) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) fail("cannot open for write: " + path);
    if (std::fwrite(data, 1, n, f) != n) fail("short write: " + path);
    std::fclose(f);
}

static std::vector<uint8_t> read_range(const char * path, int64_t off, size_t n) {
    FILE * f = std::fopen(path, "rb");
    if (!f) fail(std::string("cannot open: ") + path);
    if (::fseeko(f, off, SEEK_SET) != 0) fail("seek failed");
    std::vector<uint8_t> out(n);
    if (std::fread(out.data(), 1, n, f) != n) fail("short read");
    std::fclose(f);
    return out;
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : kGgufPath;

    // ---- open GGUF, tensor metadata only (no_alloc), as q3k_ref.cpp does ----
    struct ggml_context * gctx = nullptr;
    struct gguf_init_params gp = {true, &gctx};
    struct gguf_context * gguf = gguf_init_from_file(path, gp);
    if (!gguf || !gctx) fail("gguf_init_from_file failed");
    const int n_tensors = gguf_get_n_tensors(gguf);
    const size_t data_off = gguf_get_data_offset(gguf);

    // ---- one representative tensor + per-type counts, file order ----
    std::vector<int> first_of_type;      // tidx of first tensor per seen type
    std::vector<std::pair<int,int>> counts; // (type, count), first-seen order
    for (int i = 0; i < n_tensors; ++i) {
        const enum ggml_type t = gguf_get_tensor_type(gguf, i);
        auto it = std::find_if(counts.begin(), counts.end(),
                               [t](const std::pair<int,int> & c) { return c.first == (int)t; });
        if (it == counts.end()) {
            counts.emplace_back((int)t, 1);
            first_of_type.push_back(i);
        } else {
            it->second++;
        }
    }

    std::filesystem::create_directories(std::string(kDataDir) + "/ref");
    std::string manifest_path = std::string(kDataDir) + "/ref/manifest.txt";
    FILE * mf = std::fopen(manifest_path.c_str(), "wb");
    if (!mf) fail("cannot open for write: " + manifest_path);

    for (size_t k = 0; k < counts.size(); ++k) {
        const enum ggml_type type = (enum ggml_type)counts[k].first;
        const int tidx = first_of_type[k];
        const char * name = gguf_get_tensor_name(gguf, tidx); // borrowed, not freed
        struct ggml_tensor * info = ggml_get_tensor(gctx, name);
        if (!info) fail(std::string("ggml_get_tensor failed for ") + name);
        if (info->type != type) fail("type disagreement on " + std::string(name));

        const int64_t ne0 = info->ne[0];
        const int64_t rows_total = info->ne[1] * info->ne[2] * info->ne[3];
        const int64_t rows = std::min(kRows, rows_total);
        const size_t row_bytes = ggml_row_size(type, ne0);
        const size_t n_values = (size_t)rows * (size_t)ne0;

        // raw quantized bytes of the first `rows` rows, via the GGUF offsets
        const size_t tensor_off = gguf_get_tensor_offset(gguf, tidx);
        std::vector<uint8_t> raw =
            read_range(path, (int64_t)(data_off + tensor_off), rows * row_bytes);

        // dequantize with ggml itself. F32 has no to_float in the type
        // traits of this fork (ggml.c:657 leaves it NULL) — f32 rows are
        // their own reference, so copy the bytes.
        std::vector<float> out(n_values);
        ggml_type_traits_t traits = ggml_internal_get_type_traits(type);
        for (int64_t r = 0; r < rows; ++r) {
            const uint8_t * row = raw.data() + (size_t)r * row_bytes;
            if (type == GGML_TYPE_F32) {
                std::memcpy(out.data() + (size_t)r * ne0, row, (size_t)ne0 * sizeof(float));
            } else {
                if (!traits.to_float) fail("no to_float for this type");
                traits.to_float(row, out.data() + (size_t)r * ne0, ne0);
            }
        }

        const char * tname = ggml_type_name(type);
        write_file(std::string(kDataDir) + "/ref/" + tname + ".raw",
                   out.data(), out.size() * sizeof(float));
        FILE * meta = std::fopen((std::string(kDataDir) + "/ref/" + tname + ".meta").c_str(), "wb");
        if (!meta) fail("cannot open meta for write");
        std::fprintf(meta, "tensor=%s\n", name);
        std::fprintf(meta, "type=%d %s\n", (int)type, tname);
        std::fprintf(meta, "dims=%d %lld %lld %lld %lld\n", ggml_n_dims(info),
                     (long long)info->ne[0], (long long)info->ne[1],
                     (long long)info->ne[2], (long long)info->ne[3]);
        std::fprintf(meta, "rows=%lld\n", (long long)rows);
        std::fprintf(meta, "rowlen=%lld\n", (long long)ne0);
        std::fclose(meta);
        std::fprintf(mf, "%d %s %d\n", (int)type, tname, counts[k].second);
        std::printf("%-6s tensor=%s rows=%lld rowlen=%lld row_bytes=%zu\n",
                    tname, name, (long long)rows, (long long)ne0, row_bytes);
    }
    std::fclose(mf);
    gguf_free(gguf);
    ggml_free(gctx);
    std::printf("wrote %zu types to %s/ref\n", counts.size(), kDataDir);
    return 0;
}
