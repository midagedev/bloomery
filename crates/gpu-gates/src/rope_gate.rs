//! The NeoX gates' host inputs, expected outputs, device launch and readback.

use crate::{GateError, activations, bits_equal, verdict};
use bloomery_gpu::fault::{Fault, FaultSink, FaultSite};
use bloomery_gpu::rope_neox::PartialNeoxArgs;
use bloomery_gpu::rope_table::{Direction, RopeSpec, RopeTable};
use bloomery_gpu::{Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};

/// The bits an untouched f16 cache cell keeps.
pub const SENTINEL: u16 = 0x5a5a;

/// The rope launch's query, in-place key and cache planes read back.
pub struct RopeOut {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub ck: Vec<u16>,
    pub cv: Vec<u16>,
}

/// Host rows, gains and position table of a rope launch.
#[derive(Clone)]
pub struct RopeIn {
    pub qg: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub gq: Vec<f32>,
    pub gk: Vec<f32>,
    pub table: Vec<f32>,
}

/// The seeded inputs of a model-less head launch at `head` values a head, `rot` of them turned at
/// base `theta`: `m` rows at the cache's last `m` positions of `ctx`, the queries and the keys (×4)
/// of `n_head` and `n_kv` heads, the values, and the two gains around 1.25 and 1.3. Every gate
/// that checks a head kernel against the host rule draws the same rows, so a seed lives here once.
pub fn seeded_inputs(
    head: usize,
    rot: usize,
    theta: f32,
    (ctx, m): (usize, usize),
    (n_head, n_kv): (usize, usize),
) -> Result<(RopeIn, Vec<u32>), GateError> {
    let table_rt = RopeTable::new(&RopeSpec::window(theta, rot))?;
    let mut table = Vec::with_capacity(ctx * rot);
    for p in 0..ctx {
        table_rt.push(u32::try_from(p)?, Direction::Forward, &mut table);
    }
    let gain = |seed: u32, base: f32| -> Vec<f32> {
        activations(head, 1, seed)
            .iter()
            .map(|v| base + 0.5 * v)
            .collect()
    };
    let inp = RopeIn {
        qg: activations(head, m * n_head, 3),
        k: activations(head, m * n_kv, 6)
            .iter()
            .map(|v| 4.0 * v)
            .collect(),
        v: activations(head, m * n_kv, 7),
        gq: gain(4, 1.25),
        gk: gain(5, 1.3),
        table,
    };
    let pos: Vec<u32> = (ctx - m..ctx)
        .map(u32::try_from)
        .collect::<Result<_, _>>()?;
    Ok((inp, pos))
}

/// The head and cache geometry the launch reads, and its norm epsilon.
#[derive(Clone, Copy)]
pub struct RopeShape {
    pub head: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub ctx: usize,
    pub eps: f32,
}

/// Expected heads and cache planes under the host NeoX rule.
#[must_use]
pub fn expected<const H: usize, const R: usize, const Q: usize>(
    inp: &RopeIn,
    pos: &[u32],
    s: RopeShape,
) -> RopeOut {
    assert_eq!(s.head, H);
    let (nh, nk, ctx) = (s.n_head, s.n_kv, s.ctx);
    let mut out = RopeOut {
        q: vec![0.0; pos.len() * nh * H],
        k: vec![0.0; pos.len() * nk * H],
        ck: vec![SENTINEL; nk * ctx * H],
        cv: vec![SENTINEL; nk * ctx * H],
    };
    for (t, &p) in pos.iter().enumerate() {
        let cs = &inp.table[p as usize * R..(p as usize + 1) * R];
        for h in 0..nh {
            let src = (t * nh + h) * Q;
            let dst = (t * nh + h) * H;
            out.q[dst..dst + H].copy_from_slice(&crate::rounding::neox_head::<H, R>(
                &inp.qg[src..src + H],
                &inp.gq,
                cs,
                s.eps,
            ));
        }
        for h in 0..nk {
            let at = (t * nk + h) * H;
            let y = crate::rounding::neox_head::<H, R>(&inp.k[at..at + H], &inp.gk, cs, s.eps);
            out.k[at..at + H].copy_from_slice(&y);
            let row = (h * ctx + p as usize) * H;
            for (d, &yd) in y.iter().enumerate().take(H) {
                out.ck[row + d] = gguf::quant::f32_to_f16_bits(yd);
                out.cv[row + d] = gguf::quant::f32_to_f16_bits(inp.v[at + d]);
            }
        }
    }
    out
}

/// Check exact NeoX output, reruns and isolated cache-position faults.
pub fn check<const H: usize, const R: usize, const Q: usize>(
    gpu: &Gpu,
    inp: &RopeIn,
    pos: &[u32],
    s: RopeShape,
    layer: usize,
    run_at: impl Fn(&[u32], FaultSink) -> Result<RopeOut, GateError>,
) -> Result<bool, GateError> {
    let (m, p0, ctx) = (pos.len(), pos[0], s.ctx);
    let unl = gpu.unlabelled_sink();
    let a = run_at(pos, unl)?;
    let b = run_at(pos, unl)?;
    let want = expected::<H, R, Q>(inp, pos, s);
    let q_ok = bits_equal(&a.q, &want.q);
    let k_ok = bits_equal(&a.k, &want.k);
    let planes_ok = a.ck == want.ck && a.cv == want.cv;
    let rerun = bits_equal(&a.q, &b.q) && bits_equal(&a.k, &b.k) && a.ck == b.ck && a.cv == b.cv;
    let pass = q_ok && k_ok && planes_ok && rerun;
    println!(
        "rope m={m} positions {p0}.. ctx={ctx} head {H} rot {R}: q (read at \
         head stride {Q}) bit_exact_host={q_ok} k in place bit_exact_host={k_ok} \
         planes_exact={planes_ok} rerun={rerun} {}",
        verdict(pass)
    );

    // The last token's position at the cache's height.
    let bad_t = m - 1;
    let mut bad = pos.to_vec();
    bad[bad_t] = u32::try_from(ctx)?;
    let before = gpu.fault()?;
    let sink = gpu.layer_sink(layer)?;
    let r = run_at(&bad, sink)?;
    let raised = gpu.take_fault()?;
    let want_fault = Some(Fault::at(u32::try_from(layer)?, FaultSite::CachePos));
    let (qw, kw) = (s.n_head * H, s.n_kv * H);
    let others = bits_equal(&r.q[..bad_t * qw], &a.q[..bad_t * qw])
        && bits_equal(&r.k[..bad_t * kw], &a.k[..bad_t * kw]);
    let nan = r.q[bad_t * qw..].iter().all(|v| v.is_nan())
        && r.k[bad_t * kw..].iter().all(|v| v.is_nan());
    // The clean run minus the bad token's rows: those keep the sentinel.
    let mut ck_want = a.ck.clone();
    let mut cv_want = a.cv.clone();
    for j in 0..s.n_kv {
        let row = (j * ctx + pos[bad_t] as usize) * H;
        ck_want[row..row + H].fill(SENTINEL);
        cv_want[row..row + H].fill(SENTINEL);
    }
    let not_appended = r.ck == ck_want && r.cv == cv_want;
    let again = run_at(pos, unl)?;
    let clean_after = gpu.fault()?.is_none() && bits_equal(&again.q, &a.q);
    let fault_ok =
        before.is_none() && raised == want_fault && others && nan && not_appended && clean_after;
    println!(
        "rope fault: token {bad_t} at position {ctx} (the cache's height), layer {layer}: \
         word {raised:?} (want {want_fault:?}), its heads NaN {nan}, nothing appended \
         {not_appended}, other tokens bit-identical {others}, clean rerun and word clean \
         {clean_after} {}",
        verdict(fault_ok)
    );
    Ok(pass && fault_ok)
}

/// A head of values whose squares overflow the f32 products — query head 0
/// and key head 0 of token 0, every value `1e20` — has no norm behind it: the
/// kernel raises the named `F32Product` fault, that head's outputs and its
/// key row in the cache plane are NaN (never the zeros an infinite sum scales
/// it to), and every other head, token and plane word is the clean run's.
pub fn check_overflow<const H: usize>(
    gpu: &Gpu,
    inp: &RopeIn,
    pos: &[u32],
    s: RopeShape,
    layer: usize,
    run_at: impl Fn(&RopeIn, &[u32], FaultSink) -> Result<RopeOut, GateError>,
) -> Result<bool, GateError> {
    const BIG: f32 = 1.0e20;
    let (nh, nk, ctx) = (s.n_head, s.n_kv, s.ctx);
    let unl = gpu.unlabelled_sink();
    let clean = run_at(inp, pos, unl)?;
    let mut big = inp.clone();
    big.qg[..H].fill(BIG);
    big.k[..H].fill(BIG);
    let before = gpu.fault()?;
    let r = run_at(&big, pos, gpu.layer_sink(layer)?)?;
    let word = gpu.take_fault()?;
    let want = Some(Fault::at(u32::try_from(layer)?, FaultSite::F32Product));
    let heads_nan = r.q[..H].iter().all(|v| v.is_nan()) && r.k[..H].iter().all(|v| v.is_nan());
    let others = bits_equal(&r.q[H..], &clean.q[H..]) && bits_equal(&r.k[H..], &clean.k[H..]);
    let row = pos[0] as usize * H;
    let key_row_nan = r.ck[row..row + H]
        .iter()
        .all(|&h| gguf::quant::half_to_f32(h).is_nan());
    let mut ck_want = clean.ck.clone();
    ck_want[row..row + H].copy_from_slice(&r.ck[row..row + H]);
    let planes = r.ck == ck_want && r.cv == clean.cv;
    let again = run_at(inp, pos, unl)?;
    let clean_after = gpu.fault()?.is_none() && bits_equal(&again.q, &clean.q);
    let pass = before.is_none()
        && word == want
        && heads_nan
        && others
        && key_row_nan
        && planes
        && clean_after;
    println!(
        "rope overflow: query head 0 and key head 0 of token 0 at {BIG:e}, head {H} ({nh} query, \
         {nk} key heads, ctx {ctx}), layer {layer}: word {word:?} (want {want:?}), the heads NaN \
         {heads_nan}, every other head bit-identical {others}, its key row NaN {key_row_nan}, \
         the rest of the planes the clean run's {planes}, clean rerun and word clean \
         {clean_after} {}",
        verdict(pass)
    );
    Ok(pass)
}

/// Refuse two cache-valid positions that write the same row of a plane or ring.
pub fn check_cache_positions(pos: &[u32], ctx: usize, rows: usize) -> Result<(), GpuError> {
    let what = "rope gate positions";
    if rows == 0 {
        return Err(GpuError::shape(what, "cache rows must be positive"));
    }
    if pos.iter().enumerate().any(|(i, &p)| {
        (p as usize) < ctx
            && pos[..i]
                .iter()
                .any(|&q| (q as usize) < ctx && p as usize % rows == q as usize % rows)
    }) {
        return Err(GpuError::shape(
            what,
            "cache-valid positions must map to distinct rows",
        ));
    }
    Ok(())
}

/// Check distinct positions, enqueue the rope and read its outputs.
pub fn run(
    stream: &CudaStream,
    inp: &RopeIn,
    pos: &[u32],
    shape: RopeShape,
    fault: FaultSink,
    enqueue: impl FnOnce(&CudaStream, PartialNeoxArgs<'_>) -> Result<(), GpuError>,
) -> Result<RopeOut, GateError> {
    check_cache_positions(pos, shape.ctx, shape.ctx)?;
    let RopeShape {
        head,
        n_head,
        n_kv,
        ctx,
        eps,
    } = shape;
    let m = pos.len();
    let qg = DeviceBuffer::from_host(stream, &inp.qg)?;
    let mut q = DeviceBuffer::from_host(stream, &vec![0.0f32; m * n_head * head])?;
    let mut k = DeviceBuffer::from_host(stream, &inp.k)?;
    let v = DeviceBuffer::from_host(stream, &inp.v)?;
    let gq = DeviceBuffer::from_host(stream, &inp.gq)?;
    let gk = DeviceBuffer::from_host(stream, &inp.gk)?;
    let table = DeviceBuffer::from_host(stream, &inp.table)?;
    let posd = DeviceBuffer::from_host(stream, pos)?;
    let plane = vec![SENTINEL; n_kv * ctx * head];
    let mut ck = DeviceBuffer::from_host(stream, &plane)?;
    let mut cv = DeviceBuffer::from_host(stream, &plane)?;
    enqueue(
        stream,
        PartialNeoxArgs {
            qg: &qg,
            q: &mut q,
            k: &mut k,
            v: &v,
            gq: &gq,
            gk: &gk,
            table: &table,
            pos: &posd,
            eps,
            n_head,
            n_kv,
            ctx,
            m,
            fault,
            cache_k: &mut ck,
            cache_v: &mut cv,
        },
    )?;
    stream.synchronize()?;
    Ok(RopeOut {
        q: q.to_host_vec(stream)?,
        k: k.to_host_vec(stream)?,
        ck: ck.to_host_vec(stream)?,
        cv: cv.to_host_vec(stream)?,
    })
}

#[cfg(test)]
mod tests {
    use super::check_cache_positions;

    #[test]
    fn cache_writers_are_distinct() {
        assert!(check_cache_positions(&[0, 0], 2, 2).is_err());
        assert!(check_cache_positions(&[0, 2], 4, 2).is_err());
        assert!(check_cache_positions(&[0, 1], 2, 2).is_ok());
        assert!(check_cache_positions(&[1, 2], 4, 2).is_ok());
        assert!(check_cache_positions(&[2, 2], 2, 2).is_ok());
        assert!(check_cache_positions(&[0], 2, 0).is_err());
    }

    #[test]
    #[ignore = "needs a CUDA card"]
    fn hw_rope_rejects_duplicate_writers() -> Result<(), crate::GateError> {
        let gpu = bloomery_gpu::Gpu::new()?;
        let (inp, _) = super::seeded_inputs(128, 128, 10000.0, (2, 2), (1, 1))?;
        let result = super::run(
            gpu.stream(),
            &inp,
            &[0, 0],
            super::RopeShape {
                head: 128,
                n_head: 1,
                n_kv: 1,
                ctx: 2,
                eps: 1e-5,
            },
            gpu.unlabelled_sink(),
            |_, _| panic!("duplicate writers reached enqueue"),
        );
        assert!(result.is_err_and(|e| e.to_string().contains("distinct rows")));
        Ok(())
    }
}
