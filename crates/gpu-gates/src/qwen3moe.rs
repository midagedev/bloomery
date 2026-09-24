//! What the qwen3moe kernel gates share: the oracle sets they walk, the
//! reader for ik's f16 cache views, the host transcriptions of our per-head
//! norm and NEOX turn (`bloomery_gpu::rope_neox`'s numeric contract), and a
//! Q4_K row's scale and min terms, which the expert gates' rounding bounds
//! are taken on. Host code only.

use crate::oracle::{self, Set};
use crate::{GateError, RefManifest, RefRow, widened_f16_rows_in};
use model::arch::Arch;
use std::path::Path;

/// Values per head, the NEOX turn's width.
pub const HEAD: usize = 128;

/// Every qwen3moe oracle set: the 5-token prefill (`Set::Cpu`) first, then
/// the decode steps by depth.
pub fn sets() -> Result<Vec<(&'static str, RefManifest)>, GateError> {
    let o = oracle::for_arch(Arch::Qwen3moe)?;
    let mut sets = vec![(o.set_name(Set::Cpu)?, o.open(Set::Cpu)?)];
    for &name in o.step_sets {
        sets.push((name, o.open_named(name)?));
    }
    Ok(sets)
}

/// The decode-step sets alone, by depth.
pub fn step_sets() -> Result<Vec<(&'static str, RefManifest)>, GateError> {
    let o = oracle::for_arch(Arch::Qwen3moe)?;
    o.step_sets
        .iter()
        .map(|&name| Ok((name, o.open_named(name)?)))
        .collect()
}

/// A Q4_K row's values split as `w = d1·q − m1`: the scale term `d1·q`
/// and the min term `m1` of each value (`dequantize_row_q4_K`'s factors,
/// every product exact in f32), into `dq` and `mn`.
pub fn q4k_parts(row: &[u8], dq: &mut Vec<f32>, mn: &mut Vec<f32>) {
    use gguf::quant::half_to_f32;
    dq.clear();
    mn.clear();
    for blk in row.as_chunks::<144>().0 {
        let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = half_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let sc = &blk[4..16];
        let scale_min = |j: usize| -> (f32, f32) {
            let (s, m) = if j < 4 {
                (sc[j] & 63, sc[j + 4] & 63)
            } else {
                (
                    (sc[j + 4] & 0x0f) | ((sc[j - 4] >> 6) << 4),
                    (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4),
                )
            };
            (d * f32::from(s), dmin * f32::from(m))
        };
        for j in 0..4 {
            let q = &blk[16 + 32 * j..16 + 32 * j + 32];
            for (half, shift) in [(0usize, 0u8), (1, 4)] {
                let (d1, m1) = scale_min(2 * j + half);
                for &b in q {
                    dq.push(f32::from((b >> shift) & 0x0f) * d1);
                    mn.push(m1);
                }
            }
        }
    }
}

/// The f16 bits of f16 row `row` of the set at `dir` in its LOGICAL order:
/// the `.logical.f32` twin for a view (each value a widened half, rounded
/// back with the oracle's own `f32_to_f16_bits` and refused unless it widens
/// back to itself), else the plain rows ([`widened_f16_rows_in`]).
pub fn f16_logical_bits(dir: &Path, row: &RefRow) -> Result<Vec<u16>, GateError> {
    use gguf::quant::{f32_to_f16_bits, half_to_f32};

    if row.ty != "f16" {
        return Err(format!("{}/{} is {}, want f16", row.name, row.occurrence, row.ty).into());
    }
    if row.logical != Some(1) {
        return widened_f16_rows_in(dir, row);
    }
    let path = dir.join(row.logical_file_name());
    let raw = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    if raw.len() as u64 != 4 * row.count() {
        return Err(format!(
            "{} is {} bytes, want 4 x {} values",
            path.display(),
            raw.len(),
            row.count()
        )
        .into());
    }
    raw.as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let v = f32::from_le_bytes(*c);
            let h = f32_to_f16_bits(v);
            if half_to_f32(h).to_bits() == v.to_bits() {
                Ok(h)
            } else {
                Err(format!("{} holds {v} at {i}, not a widened f16", path.display()).into())
            }
        })
        .collect()
}

/// Our norm of one head on the host, the kernel's order op for op: thread
/// `t` adds the f32 squares of values `t` and `t + 64` in f64, each warp's
/// 32 lanes by the xor butterfly (16, 8, 4, 2, 1), warp 0's sum plus warp
/// 1's, the mean `(sum / 128) as f32`, `1 / sqrt(mean + eps)`, then
/// `(scale · gain) · x`.
pub fn head_norm(x: &[f32], gain: &[f32], eps: f32) -> Vec<f32> {
    let half = HEAD / 2;
    let lanes: Vec<f64> = (0..half)
        .map(|t| f64::from(x[t] * x[t]) + f64::from(x[t + half] * x[t + half]))
        .collect();
    let warp = |w: usize| {
        let mut v: [f64; 32] = lanes[32 * w..32 * w + 32]
            .try_into()
            .expect("a warp is 32 lanes");
        for off in [16, 8, 4, 2, 1] {
            let prev = v;
            for (l, s) in v.iter_mut().enumerate() {
                *s = prev[l] + prev[l ^ off];
            }
        }
        v[0]
    };
    let sum = warp(0) + warp(1);
    let mean = (sum / HEAD as f64) as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    x.iter().zip(gain).map(|(&v, &g)| (scale * g) * v).collect()
}

/// The NEOX turn of heads of [`HEAD`] values, `n_vec` per token: pair `i`
/// is `(x[i], x[i + 64])`, turned by table pair `i` of its token's `cs`
/// (`cs[t·HEAD + 2i ..]`) as `y0 = fma(x0, c, −(x1·s))`,
/// `y1 = fma(x0, s, x1·c)` — the kernel's `neox_pair` and the form ik's CPU
/// build compiles ggml's NEOX loop to alike.
pub fn neox_rotate(x: &[f32], cs: &[f32], n_vec: usize) -> Vec<f32> {
    let half = HEAD / 2;
    let mut y = x.to_vec();
    for (r, head) in y.chunks_mut(HEAD).enumerate() {
        let t = r / n_vec;
        for i in 0..half {
            let (c, s) = (cs[t * HEAD + 2 * i], cs[t * HEAD + 2 * i + 1]);
            let (x0, x1) = (head[i], head[i + half]);
            head[i] = x0.mul_add(c, -(x1 * s));
            head[i + half] = x0.mul_add(s, x1 * c);
        }
    }
    y
}

/// One layer's attention inputs and ik's rows around the norm, the turn and
/// the append, read from a set: `Qcur-L (reshaped)`, `Kcur-L (reshaped)`
/// and `Vcur-L` in, `Qcur_normed-L`/`Kcur_normed-L` (FUSED_RMS_NORM, gain
/// in `src1`), `Qcur_roped-L`/`Kcur_roped-L` (ROPE, positions in `src1`)
/// and the two cache writes (`… (copy of Kcur_roped-L)`, `… (copy of
/// Vcur-L)`) as f16 bits, token after token.
pub struct AttnRows {
    pub layer: usize,
    pub m: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub pos: Vec<u32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub q_normed: Vec<f32>,
    pub k_normed: Vec<f32>,
    pub q_roped: Vec<f32>,
    pub k_roped: Vec<f32>,
    pub k_rows: Vec<u16>,
    pub v_rows: Vec<u16>,
    pub gq_name: String,
    pub gk_name: String,
}

impl AttnRows {
    /// Layer `layer`'s rows of `man`, or `None` when the set holds no
    /// `Qcur_normed-L`.
    pub fn read(man: &RefManifest, layer: usize) -> Result<Option<AttnRows>, GateError> {
        use crate::{Layout, RowKind, find_int_row, ref_ints_of_in, ref_tensor_logical_in};

        let Ok(qn) = man.tensor(&format!("Qcur_normed-{layer}"), 0) else {
            return Ok(None);
        };
        let kn = man.tensor(&format!("Kcur_normed-{layer}"), 0)?;
        let qr = man.tensor(&format!("Qcur_roped-{layer}"), 0)?;
        let kr = man.tensor(&format!("Kcur_roped-{layer}"), 0)?;
        let vcur = man.tensor(&format!("Vcur-{layer}"), 0)?;
        let kw = man.tensor(
            &format!("cache_k_l{layer} (view) (copy of Kcur_roped-{layer})"),
            0,
        )?;
        let vw = man.tensor(&format!("v_cache_view-{layer} (copy of Vcur-{layer})"), 0)?;
        let src = |row: &RefRow| -> Result<&RefRow, GateError> {
            let name = row.src0.as_deref().ok_or("a norm row has no src0")?;
            man.tensor(name, 0)
        };
        let (n_head, n_kv, m) = (qn.ne[1] as usize, kn.ne[1] as usize, qn.ne[2] as usize);
        if qn.ne[0] as usize != HEAD || kn.ne[0] as usize != HEAD || kn.ne[2] as usize != m {
            return Err(format!(
                "layer {layer}: Qcur_normed {:?}, Kcur_normed {:?}",
                qn.ne, kn.ne
            )
            .into());
        }
        let pos_name = qr.src1.as_deref().ok_or("the rope row has no src1")?;
        let pos_row = find_int_row(man, pos_name, 0, RowKind::Input, Layout::Flat)?;
        let pos = ref_ints_of_in(&man.dir, pos_row)?
            .into_iter()
            .map(|v| u32::try_from(v).map_err(|_| GateError::from(format!("position {v}"))))
            .collect::<Result<Vec<_>, _>>()?;
        if pos.len() != m {
            return Err(format!("layer {layer}: {} positions for {m} tokens", pos.len()).into());
        }
        let rd = |row: &RefRow| ref_tensor_logical_in(&man.dir, row);
        Ok(Some(AttnRows {
            layer,
            m,
            n_head,
            n_kv,
            pos,
            q: rd(src(qn)?)?,
            k: rd(src(kn)?)?,
            v: rd(vcur)?,
            q_normed: rd(qn)?,
            k_normed: rd(kn)?,
            q_roped: rd(qr)?,
            k_roped: rd(kr)?,
            k_rows: f16_logical_bits(&man.dir, kw)?,
            v_rows: f16_logical_bits(&man.dir, vw)?,
            gq_name: qn.src1.clone().ok_or("the q norm has no gain column")?,
            gk_name: kn.src1.clone().ok_or("the k norm has no gain column")?,
        }))
    }
}

/// The device half: one launch of `head_norm_neox_append` over a layer's
/// rows.
#[cfg(feature = "gpu")]
pub mod dev {
    use super::{AttnRows, HEAD};
    use crate::GateError;
    use bloomery_gpu::Gpu;
    use bloomery_gpu::rope_neox::{NeoxArgs, RopeNeoxKernels};
    use cuda_core::{CudaStream, DeviceBuffer};

    /// What the gate fills the planes with before an append: an f16 NaN,
    /// which `f32_to_f16_bits` never writes, so a slot still holding it was
    /// not written.
    pub const SENTINEL: u16 = 0xffff;

    /// One launch's results, read back: the query and key heads after the
    /// norm and the turn, and both whole planes.
    pub struct NeoxOut {
        pub q: Vec<f32>,
        pub k: Vec<f32>,
        pub cache_k: Vec<u16>,
        pub cache_v: Vec<u16>,
    }

    /// Run the kernel on `rows` with gains `gq`/`gk`, tables `cs` (`m ·
    /// HEAD`), into planes of `ctx` rows per key head filled with
    /// [`SENTINEL`].
    #[allow(clippy::too_many_arguments, reason = "one launch's inputs, each named")]
    pub fn run(
        k: &RopeNeoxKernels,
        stream: &CudaStream,
        rows: &AttnRows,
        gq: &[f32],
        gk: &[f32],
        cs: &[f32],
        eps: f32,
        ctx: usize,
    ) -> Result<NeoxOut, GateError> {
        Ok(launch(k, stream, None, rows, gq, gk, cs, eps, ctx)?.0)
    }

    /// [`run`] with the launch captured on `gpu`'s stream as a graph and
    /// replayed once; also returns the graph's node count.
    #[allow(clippy::too_many_arguments, reason = "one launch's inputs, each named")]
    pub fn run_graph(
        gpu: &Gpu,
        k: &RopeNeoxKernels,
        rows: &AttnRows,
        gq: &[f32],
        gk: &[f32],
        cs: &[f32],
        eps: f32,
        ctx: usize,
    ) -> Result<(NeoxOut, usize), GateError> {
        launch(k, gpu.stream(), Some(gpu), rows, gq, gk, cs, eps, ctx)
    }

    #[allow(clippy::too_many_arguments, reason = "one launch's inputs, each named")]
    fn launch(
        k: &RopeNeoxKernels,
        stream: &CudaStream,
        graph: Option<&Gpu>,
        rows: &AttnRows,
        gq: &[f32],
        gk: &[f32],
        cs: &[f32],
        eps: f32,
        ctx: usize,
    ) -> Result<(NeoxOut, usize), GateError> {
        let mut q = DeviceBuffer::from_host(stream, &rows.q)?;
        let mut kb = DeviceBuffer::from_host(stream, &rows.k)?;
        let v = DeviceBuffer::from_host(stream, &rows.v)?;
        let gq = DeviceBuffer::from_host(stream, gq)?;
        let gk = DeviceBuffer::from_host(stream, gk)?;
        let cs = DeviceBuffer::from_host(stream, cs)?;
        let pos = DeviceBuffer::from_host(stream, &rows.pos)?;
        let plane = vec![SENTINEL; rows.n_kv * ctx * HEAD];
        let mut cache_k = DeviceBuffer::from_host(stream, &plane)?;
        let mut cache_v = DeviceBuffer::from_host(stream, &plane)?;
        let mut enqueue = |s: &CudaStream| {
            k.enqueue_head_norm_neox_append(
                s,
                NeoxArgs {
                    q: &mut q,
                    k: &mut kb,
                    v: &v,
                    gq: &gq,
                    gk: &gk,
                    cs: &cs,
                    pos: &pos,
                    eps,
                    n_head: rows.n_head,
                    n_kv: rows.n_kv,
                    ctx,
                    m: rows.m,
                    cache_k: &mut cache_k,
                    cache_v: &mut cache_v,
                },
            )
        };
        let nodes = match graph {
            None => {
                enqueue(stream)?;
                0
            }
            Some(gpu) => {
                let g = gpu.capture(&mut enqueue)?;
                g.launch(stream)?;
                g.node_count()
            }
        };
        stream.synchronize()?;
        Ok((
            NeoxOut {
                q: q.to_host_vec(stream)?,
                k: kb.to_host_vec(stream)?,
                cache_k: cache_k.to_host_vec(stream)?,
                cache_v: cache_v.to_host_vec(stream)?,
            },
            nodes,
        ))
    }
}
