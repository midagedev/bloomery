//! The engine's one Q8_0 quantizer: the 32-value block rule the reference's
//! weight requant runs (`quantize_row_q8_0`, ggml-quants.c's x86 branch),
//! shared by the deepseek2 requant that owned it first and the qwen3
//! family's q8_0 cache seed.

use gguf::quant::{Q8Block, f32_to_f16_bits};

/// The weight requant the reference's `wk_b` cast runs: `quantize_row_q8_0`,
/// x86 branch (ggml-quants.c:938+) — `d = amax/127` stored f16, `id =
/// 127/amax` (a different f32 than `1/d`), `_mm256_round_ps
/// (_MM_ROUND_NEAREST)` codes; the ref variant (`id = 1/d`, `roundf`) is NOT
/// what runs here.
///
/// One 32-value block at a time: a shorter slice would leave trailing codes
/// at 0 silently.
#[must_use]
pub fn quantize_q8_0(x: &[f32]) -> Q8Block {
    assert_eq!(x.len(), 32, "quantize_q8_0: one 32-value block at a time");
    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }
    let d = amax / 127.0;
    let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
    let mut q = [0i8; 32];
    for (j, &v) in x.iter().enumerate() {
        q[j] = qdot::nearest_int(v * id).clamp(-128, 127) as i8;
    }
    Q8Block {
        d: f32_to_f16_bits(d),
        q,
    }
}
