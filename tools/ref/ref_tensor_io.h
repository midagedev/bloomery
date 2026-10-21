#pragma once
// The element reads every node dumper shares (dump_ref, dump_mtp, dump_draft, cb_dump): a tensor's
// file-name stem, the one per-type conversion and the two layouts of its elements. One owner, so a
// type handled in one dumper cannot fall through to a silent zero in another. Each dumper's build
// and run scripts hash this header with the dumper's source.
#include "ggml.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Tensor names carry '/' and '.' in some graphs; keep the file name a single path element.
inline std::string safe_name(const char * name) {
    std::string s(name);
    for (char & c : s) {
        if (c == '/' || c == '\\' || c == ' ') c = '_';
    }
    return s;
}

enum elem_kind { ELEM_UNHANDLED, ELEM_FLOAT, ELEM_INT };

// The one per-type conversion every pass reads through — flat file, logical twin, integer
// twins, inputs — so a type handled in one pass cannot fall through to a silent zero or a
// missing twin in another. `p` is one element. f16/bf16 widen to f32 exactly; an integer comes
// back exact in `i` and cast in `f`, which is exact up to 2^24 in magnitude.
inline elem_kind read_elem(ggml_type type, const uint8_t * p, float & f, int64_t & i) {
    switch (type) {
        case GGML_TYPE_F32:  { float       x; memcpy(&x, p, sizeof x); f = x;                    return ELEM_FLOAT; }
        case GGML_TYPE_F16:  { ggml_fp16_t x; memcpy(&x, p, sizeof x); f = ggml_fp16_to_fp32(x); return ELEM_FLOAT; }
        case GGML_TYPE_BF16: { ggml_bf16_t x; memcpy(&x, p, sizeof x); f = ggml_bf16_to_fp32(x); return ELEM_FLOAT; }
        case GGML_TYPE_I8:   { int8_t      x; memcpy(&x, p, sizeof x); i = x; break; }
        case GGML_TYPE_I16:  { int16_t     x; memcpy(&x, p, sizeof x); i = x; break; }
        case GGML_TYPE_I32:  { int32_t     x; memcpy(&x, p, sizeof x); i = x; break; }
        case GGML_TYPE_I64:  { int64_t     x; memcpy(&x, p, sizeof x); i = x; break; }
        default:             return ELEM_UNHANDLED;
    }
    f = (float) i;
    return ELEM_INT;
}

// One layout of a tensor's elements through read_elem: flat (nelements values read
// contiguously from byte 0 — the main file) or logical (ggml index order, ne0 fastest,
// through the tensor's own nb strides). `src` maps byte 0 to t->data (host pointer, or the
// staging copy the backend filled from t->data), and ggml_nbytes covers the furthest strided
// element, so both walks stay inside that buffer. `ints` is filled for an integer type only.
inline void gather(const ggml_tensor * t, const uint8_t * src, bool logical,
                   std::vector<float> & f, std::vector<int64_t> & ints) {
    const size_t esize = ggml_type_size(t->type);
    const bool   is_int = !ints.empty();
    int64_t v_int = 0;
    size_t  li    = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
        const size_t off = logical
            ? (size_t) i0 * t->nb[0] + (size_t) i1 * t->nb[1] + (size_t) i2 * t->nb[2] + (size_t) i3 * t->nb[3]
            : li * esize;
        read_elem(t->type, src + off, f[li], v_int);
        if (is_int) ints[li] = v_int;
        ++li;
    }
}
