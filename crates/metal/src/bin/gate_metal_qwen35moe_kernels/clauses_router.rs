//! The router's clauses.

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, DeviceError, Elems};
use bloomery_metal::buffer::MetalBuffer;
use bloomery_metal::geom::RouterGeom;
use bloomery_metal::kernels::router::RouterParams;
use gguf::Gguf;
use gguf::quant::{GgmlType, dequant_row};

use super::common::{self, BadArg, Edit, LAYER};
use super::fixture::Fixture;
use super::oracle;
use super::oracle_router::{self, Ranking, Step};
use super::{Harness, SENTINEL_BITS};

/// The ids and weights a launch writes: the routed picks, then the shared
/// expert's when the router has its row.
fn written(r: &RouterGeom) -> usize {
    r.top_k_max + r.shared
}
/// Words past the ids and the weights a launch must leave as they were.
const ID_GUARD: usize = 4;

const ROUTED_TENSOR: &str = "blk.0.ffn_gate_inp.weight";
const SHARED_TENSOR: &str = "blk.0.ffn_gate_inp_shexp.weight";
/// The pre-FFN norm's gain under the names the families give it.
const GAIN_TENSORS: [&str; 2] = ["blk.0.post_attention_norm.weight", "blk.0.ffn_norm.weight"];

/// Layer 0's pre-FFN norm gain, whichever of its names the fixture holds.
pub(super) fn gain_for(r: &RouterGeom, fixture: &Fixture) -> Result<Vec<f32>, String> {
    let file = Gguf::open(&fixture.path).map_err(|e| format!("the fixture did not reopen: {e}"))?;
    match GAIN_TENSORS.map(|name| file.find(name).is_some()) {
        [true, false] => fixture.f32s(GAIN_TENSORS[0], r.k),
        [false, true] => fixture.f32s(GAIN_TENSORS[1], r.k),
        held => Err(format!(
            "the fixture holds {held:?} of {GAIN_TENSORS:?}; the router reads exactly one"
        )),
    }
}

struct Weights {
    routed: Vec<f32>,
    shared: Vec<f32>,
    gain: Vec<f32>,
    eps: f32,
}

pub(super) struct Out {
    pub(super) logits: Vec<f32>,
    pub(super) ids: Vec<u32>,
    pub(super) weights: Vec<f32>,
}

fn load_weights(r: &RouterGeom, fixture: &Fixture) -> Result<Weights, String> {
    let file = Gguf::open(&fixture.path).map_err(|e| format!("{ROUTED_TENSOR}: {e}"))?;
    let t = file
        .find(ROUTED_TENSOR)
        .ok_or_else(|| format!("the fixture has no tensor {ROUTED_TENSOR}"))?;
    if t.ty != GgmlType::F32 || t.dims != [r.k as u64, r.experts as u64] {
        return Err(format!(
            "{ROUTED_TENSOR} is {:?} {:?}, the router reads {} rows of {} f32s",
            t.ty, t.dims, r.experts, r.k
        ));
    }
    let bytes = file.data(t).map_err(|e| format!("{ROUTED_TENSOR}: {e}"))?;
    let mut routed = vec![0f32; r.experts * r.k];
    dequant_row(GgmlType::F32, bytes, &mut routed).map_err(|e| format!("{ROUTED_TENSOR}: {e}"))?;
    Ok(Weights {
        routed,
        shared: if r.shared == 1 {
            fixture.f32s(SHARED_TENSOR, r.k)?
        } else {
            Vec::new()
        },
        gain: gain_for(r, fixture)?,
        eps: fixture.rms_eps()?,
    })
}

/// One nonzero column, so each logit is that column's scaled product and the
/// 8th-vs-9th gap is the gap between the targets.
fn constructed(r: &RouterGeom, eps: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut x = vec![0f32; r.k];
    x[0] = 1.0;
    let gain = vec![1.0f32; r.k];
    let xhat = oracle::rms_norm(&x, &gain, eps);
    let x0 = xhat[0];
    let mut routed = vec![0f32; r.experts * r.k];
    for i in 0..r.experts {
        let target = if i < r.top_k_max {
            4.0 - 0.5 * f64::from(u32::try_from(i).unwrap_or(0))
        } else {
            -2.0 - 0.01 * f64::from(u32::try_from(i - r.top_k_max).unwrap_or(0))
        };
        routed[i * r.k] = (target / x0) as f32;
    }
    let mut shared = vec![0f32; r.shared * r.k];
    if let Some(first) = shared.first_mut() {
        *first = (0.5 / x0) as f32;
    }
    (routed, shared, x, gain)
}

/// The tie construction: rows `LOW` and `HIGH` carry the same weights, so
/// their logits are the same bits, and the tie sits at the 8th and 9th
/// places of the ranking.
fn constructed_tie(r: &RouterGeom, eps: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    const ABOVE: usize = 7;
    const TIE: f64 = 0.75;
    let mut x = vec![0f32; r.k];
    x[0] = 1.0;
    let gain = vec![1.0f32; r.k];
    let xhat = oracle::rms_norm(&x, &gain, eps);
    let x0 = xhat[0];
    let mut routed = vec![0f32; r.experts * r.k];
    for i in 0..r.experts {
        let target = if i < ABOVE {
            4.0 - 0.5 * f64::from(u32::try_from(i).unwrap_or(0))
        } else if i == ABOVE || i == ABOVE + 1 {
            TIE
        } else {
            -2.0 - 0.01 * f64::from(u32::try_from(i - ABOVE - 2).unwrap_or(0))
        };
        routed[i * r.k] = (target / x0) as f32;
    }
    let mut shared = vec![0f32; r.shared * r.k];
    if let Some(first) = shared.first_mut() {
        *first = (0.5 / x0) as f32;
    }
    (routed, shared, x, gain)
}

pub(super) fn run(
    h: &Harness<'_>,
    routed: &[f32],
    shared: &[f32],
    x: &[f32],
    gain: &[f32],
    eps: f32,
) -> Result<Out, String> {
    let r = h.geom().router;
    let (device, stream) = (h.device, h.stream);
    let routed_b = device.upload(stream, routed).map_err(why)?;
    let shared_b = (r.shared == 1)
        .then(|| device.upload(stream, shared))
        .transpose()
        .map_err(why)?;
    let x_b = device.upload(stream, x).map_err(why)?;
    let gain_b = device.upload(stream, gain).map_err(why)?;
    let mut partials = device
        .upload(stream, &vec![0f32; r.partials])
        .map_err(why)?;
    let mut ticket = device.upload(stream, &[0u32]).map_err(why)?;
    let mut logits = device.upload(stream, &vec![0f32; r.rows]).map_err(why)?;
    let mut ids = device
        .upload(stream, &vec![0xFFFFu32; written(&r) + ID_GUARD])
        .map_err(why)?;
    let mut weights = device
        .upload(
            stream,
            &vec![f32::from_bits(SENTINEL_BITS); written(&r) + ID_GUARD],
        )
        .map_err(why)?;
    h.encode(|encoder| {
        h.kernels.router(
            encoder,
            RouterParams {
                rows: r.rows,
                k: r.k,
                top_k: r.top_k_max,
                routed: routed_b.window(Elems(0), Elems(routed.len()))?,
                shared: shared_b
                    .as_ref()
                    .map(|b| b.window(Elems(0), Elems(shared.len())))
                    .transpose()?,
                x: x_b.window(Elems(0), Elems(x.len()))?,
                gain: gain_b.window(Elems(0), Elems(gain.len()))?,
                partials: partials.window_mut(Elems(0), Elems(r.partials))?,
                ticket: ticket.window_mut(Elems(0), Elems(1))?,
                logits: logits.window_mut(Elems(0), Elems(r.rows))?,
                ids: ids.window_mut(Elems(0), Elems(written(&r)))?,
                weights: weights.window_mut(Elems(0), Elems(written(&r)))?,
                eps,
                fault: device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let mut logits_h = vec![0f32; r.rows];
    let mut ids_h = vec![0u32; written(&r) + ID_GUARD];
    let mut weights_h = vec![0f32; written(&r) + ID_GUARD];
    device
        .read(stream, logits.whole().map_err(why)?, &mut logits_h)
        .map_err(why)?;
    device
        .read(stream, ids.whole().map_err(why)?, &mut ids_h)
        .map_err(why)?;
    device
        .read(stream, weights.whole().map_err(why)?, &mut weights_h)
        .map_err(why)?;
    if ids_h[written(&r)..].iter().any(|&id| id != 0xFFFF)
        || weights_h[written(&r)..]
            .iter()
            .any(|w| w.to_bits() != SENTINEL_BITS)
    {
        return Err(format!(
            "the router wrote past its {} ids or weights: ids {:?}, weights {:?}",
            written(&r),
            &ids_h[written(&r)..],
            &weights_h[written(&r)..]
        ));
    }
    ids_h.truncate(written(&r));
    weights_h.truncate(written(&r));
    Ok(Out {
        logits: logits_h,
        ids: ids_h,
        weights: weights_h,
    })
}

fn match_ids(r: &RouterGeom, name: &str, out: &Out, ranking: &Ranking) -> Result<(), String> {
    for (s, &want) in ranking.ids.iter().enumerate() {
        let got = out.ids[s] as usize;
        if got != want {
            return Err(format!(
                "router-ids: step {name} slot {s} is {got}, the f64 ranking has {want}"
            ));
        }
    }
    shared_id(r, name, out)
}

/// A gated router writes the shared expert's id after the picks.
fn shared_id(r: &RouterGeom, name: &str, out: &Out) -> Result<(), String> {
    if r.shared == 0 {
        return Ok(());
    }
    let shared = out.ids[r.top_k_max];
    if shared != r.experts as u32 {
        return Err(format!(
            "router-ids: step {name} shared id is {shared}, the shared expert is {}",
            r.experts
        ));
    }
    Ok(())
}

fn check_ids(r: &RouterGeom, name: &str, out: &Out, step: &Step) -> Result<&'static str, String> {
    let ranking = oracle_router::rank(r, step);
    if !ranking.cut_stable {
        println!(
            "router-ids: step {name} 8th-vs-9th margin {:.3e} is inside the band {:.3e}",
            ranking.margin, ranking.margin_band
        );
        return Ok("printed");
    }
    if !ranking.sequence_stable {
        println!(
            "router-ids: step {name} an internal margin is inside the band; the id set is checked"
        );
        let mut got: Vec<usize> = out.ids[..r.top_k_max]
            .iter()
            .map(|id| *id as usize)
            .collect();
        let mut want = ranking.ids;
        got.sort_unstable();
        want.sort_unstable();
        if got != want {
            return Err(format!(
                "router-ids: step {name} id set {got:?}, the f64 ranking has {want:?}"
            ));
        }
        shared_id(r, name, out)?;
        return Ok("set");
    }
    match_ids(r, name, out, &ranking)?;
    Ok("sequence")
}

fn check_weights(r: &RouterGeom, name: &str, out: &Out, step: &Step) -> Result<f64, String> {
    let ranking = oracle_router::rank(r, step);
    if ranking.sequence_stable {
        match_ids(r, name, out, &ranking)?;
    }
    let ids: Vec<usize> = out.ids[..r.top_k_max]
        .iter()
        .map(|id| *id as usize)
        .collect();
    let oracle = oracle_router::weights(r, step, &ids)?;
    let mut worst = 0.0f64;
    for (s, (&got, (&want, &band))) in out
        .weights
        .iter()
        .zip(oracle.want.iter().zip(&oracle.band))
        .enumerate()
    {
        let err = (f64::from(got) - want).abs();
        if !got.is_finite() || err > band {
            return Err(format!(
                "router-weights: step {name} slot {s} got {got:e}, want {want:e}, error {err:e} over the band {band:e}"
            ));
        }
        worst = worst.max(err / band);
    }
    Ok(worst)
}

pub fn router_logit(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let r = h.geom().router;
    h.clear_fault()?;
    let weights = load_weights(&r, fixture)?;
    let mut worst = 0.0f64;
    let mut notes = Vec::new();
    for (name, seed) in [("seed-1", 1u64), ("seed-2", 2u64)] {
        let x = oracle::activations(r.k, seed, 0.5);
        let out = run(
            h,
            &weights.routed,
            &weights.shared,
            &x,
            &weights.gain,
            weights.eps,
        )?;
        common::expect_clean(h, name)?;
        let step = oracle_router::step(
            name,
            &r,
            &weights.routed,
            &weights.shared,
            &x,
            &weights.gain,
            weights.eps,
        )?;
        let ratio = check_logits_named(name, &out.logits, &step)?;
        worst = worst.max(ratio);
        notes.push(format!("{name} {ratio:.3}"));
    }
    let eps = 1e-5;
    let (routed, shared, x, gain) = constructed(&r, eps);
    let out = run(h, &routed, &shared, &x, &gain, eps)?;
    common::expect_clean(h, "constructed")?;
    let step = oracle_router::step("constructed", &r, &routed, &shared, &x, &gain, eps)?;
    let ratio = check_logits_named("constructed", &out.logits, &step)?;
    worst = worst.max(ratio);
    notes.push(format!("constructed {ratio:.3}"));
    Ok(Verdict::Noted(format!(
        "worst {worst:.3}; {}",
        notes.join(", ")
    )))
}

fn check_logits_named(name: &str, got: &[f32], step: &Step) -> Result<f64, String> {
    oracle_router::check_logits(got, step).map_err(|e| format!("{name}: {e}"))
}

pub fn router_ids(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let r = h.geom().router;
    h.clear_fault()?;
    let weights = load_weights(&r, fixture)?;
    let mut notes = Vec::new();
    for (name, seed) in [("seed-1", 1u64), ("seed-2", 2u64)] {
        let x = oracle::activations(r.k, seed, 0.5);
        let out = run(
            h,
            &weights.routed,
            &weights.shared,
            &x,
            &weights.gain,
            weights.eps,
        )?;
        common::expect_clean(h, name)?;
        let step = oracle_router::step(
            name,
            &r,
            &weights.routed,
            &weights.shared,
            &x,
            &weights.gain,
            weights.eps,
        )?;
        notes.push(format!("{name} {}", check_ids(&r, name, &out, &step)?));
    }
    let eps = 1e-5;
    let (routed, shared, x, gain) = constructed(&r, eps);
    let out = run(h, &routed, &shared, &x, &gain, eps)?;
    common::expect_clean(h, "constructed")?;
    let step = oracle_router::step("constructed", &r, &routed, &shared, &x, &gain, eps)?;
    let ranking = oracle_router::rank(&r, &step);
    if !ranking.sequence_stable {
        return Err(format!(
            "router-ids: the constructed step is not stable (margin {:.3e}, band {:.3e})",
            ranking.margin, ranking.margin_band
        ));
    }
    match_ids(&r, "constructed", &out, &ranking)?;
    Ok(Verdict::Noted(format!(
        "constructed margin {:.3e} over {:.3e}; {}",
        ranking.margin,
        ranking.margin_band,
        notes.join(", ")
    )))
}

/// The 8th and 9th logits tie in bits; the smaller id must take the 8th
/// slot. The construction carries no fixture weights, so the clause reads
/// none.
pub fn router_tie(h: &Harness<'_>) -> Clause {
    let r = h.geom().router;
    const LOW: usize = 7;
    const HIGH: usize = 8;
    const ABOVE: usize = 7;
    let eps = 1e-5;
    h.clear_fault()?;
    let (routed, shared, x, gain) = constructed_tie(&r, eps);
    let out = run(h, &routed, &shared, &x, &gain, eps)?;
    common::expect_clean(h, "constructed")?;
    if out.logits[LOW].to_bits() != out.logits[HIGH].to_bits() {
        return Err(format!(
            "rows {LOW} and {HIGH} are {:e} and {:e}, not the same bits",
            out.logits[LOW], out.logits[HIGH]
        ));
    }
    let tied = out.logits[LOW];
    if !tied.is_finite() {
        return Err(format!("the tied logit {tied:e} is not finite"));
    }
    for (i, &v) in out.logits.iter().enumerate() {
        if i == LOW || i == HIGH || i == r.experts {
            continue;
        }
        let above = if i < ABOVE { v > tied } else { v < tied };
        if !above {
            return Err(format!(
                "row {i} is {v:e}, not on its side of the tie {tied:e}"
            ));
        }
    }
    shared_id(&r, "tie", &out)?;
    for (s, &got) in out.ids[..r.top_k_max].iter().enumerate() {
        let want = u32::try_from(s).map_err(|_| "a slot id does not fit u32")?;
        if got != want {
            return Err(format!(
                "slot {s} is {got}, the tied ranking selects {want} (the tie is {LOW} and {HIGH})"
            ));
        }
    }
    if out.weights[..r.top_k_max].iter().any(|w| !w.is_finite()) {
        return Err("a routed weight is not finite".to_string());
    }
    Ok(Verdict::Noted(format!(
        "slots 0..{} are 0..{LOW}; the tie {LOW}/{HIGH} takes {LOW}",
        r.top_k_max - 1
    )))
}

pub fn router_weights(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let r = h.geom().router;
    h.clear_fault()?;
    let weights = load_weights(&r, fixture)?;
    let mut worst = 0.0f64;
    let mut notes = Vec::new();
    for (name, seed) in [("seed-1", 1u64), ("seed-2", 2u64)] {
        let x = oracle::activations(r.k, seed, 0.5);
        let out = run(
            h,
            &weights.routed,
            &weights.shared,
            &x,
            &weights.gain,
            weights.eps,
        )?;
        common::expect_clean(h, name)?;
        let step = oracle_router::step(
            name,
            &r,
            &weights.routed,
            &weights.shared,
            &x,
            &weights.gain,
            weights.eps,
        )?;
        let ratio = check_weights(&r, name, &out, &step)?;
        worst = worst.max(ratio);
        notes.push(format!("{name} {ratio:.3}"));
    }
    let eps = 1e-5;
    let (routed, shared, x, gain) = constructed(&r, eps);
    let out = run(h, &routed, &shared, &x, &gain, eps)?;
    common::expect_clean(h, "constructed")?;
    let step = oracle_router::step("constructed", &r, &routed, &shared, &x, &gain, eps)?;
    let ranking = oracle_router::rank(&r, &step);
    if !ranking.sequence_stable {
        return Err(format!(
            "router-weights: the constructed step is not stable (margin {:.3e}, band {:.3e})",
            ranking.margin, ranking.margin_band
        ));
    }
    let ratio = check_weights(&r, "constructed", &out, &step)?;
    worst = worst.max(ratio);
    Ok(Verdict::Noted(format!(
        "worst {worst:.3}; constructed {ratio:.3}; {}",
        notes.join(", ")
    )))
}

pub fn router_ticket(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    const LAUNCHES: usize = 10_000;
    const BATCH: usize = 1_000;
    let r = h.geom().router;
    h.clear_fault()?;
    let weights = load_weights(&r, fixture)?;
    let (device, stream) = (h.device, h.stream);
    let mut xs = vec![0f32; LAUNCHES * r.k];
    let mut rng = oracle::Rng::new(0x715E_0000);
    for v in &mut xs {
        *v = 0.25 * rng.signed();
    }
    let id_len = written(&r);
    let routed = device.upload(stream, &weights.routed).map_err(why)?;
    // A router with no shared row binds none: the buffer only stands in.
    let shared = device
        .upload(
            stream,
            &[weights.shared.clone(), vec![0f32; (1 - r.shared) * r.k]].concat(),
        )
        .map_err(why)?;
    let x = device.upload(stream, &xs).map_err(why)?;
    let gain = device.upload(stream, &weights.gain).map_err(why)?;
    let mut partials = device
        .upload(stream, &vec![0f32; r.partials])
        .map_err(why)?;
    let mut ticket = device.upload(stream, &[0u32]).map_err(why)?;
    let mut logits = device
        .upload(stream, &vec![0f32; LAUNCHES * r.rows])
        .map_err(why)?;
    let mut ids = device
        .upload(stream, &vec![0xFFFFu32; LAUNCHES * id_len])
        .map_err(why)?;
    let mut out_w = device
        .upload(stream, &vec![0f32; LAUNCHES * id_len])
        .map_err(why)?;
    let mut bufs = Buffers {
        routed: &routed,
        shared: &shared,
        x: &x,
        gain: &gain,
        partials: &mut partials,
        ticket: &mut ticket,
        logits: &mut logits,
        ids: &mut ids,
        weights: &mut out_w,
    };
    for batch in (0..LAUNCHES).step_by(BATCH) {
        let end = (batch + BATCH).min(LAUNCHES);
        dispatch_ticket(h, batch, end, &mut bufs, weights.eps)?;
        let mut word = [1u32];
        device
            .read(
                stream,
                bufs.ticket.window(Elems(0), Elems(1)).map_err(why)?,
                &mut word,
            )
            .map_err(why)?;
        if word[0] != 0 {
            return Err(format!(
                "after {end} launches the ticket counter is {}, not 0",
                word[0]
            ));
        }
    }
    let mut logits_h = vec![0f32; LAUNCHES * r.rows];
    let mut ids_h = vec![0u32; LAUNCHES * id_len];
    device
        .read(stream, bufs.logits.whole().map_err(why)?, &mut logits_h)
        .map_err(why)?;
    device
        .read(stream, bufs.ids.whole().map_err(why)?, &mut ids_h)
        .map_err(why)?;
    common::expect_clean(h, "router-ticket")?;
    for launch in 0..LAUNCHES {
        let row = launch * r.rows;
        let logits_l = &logits_h[row..row + r.rows];
        if logits_l.iter().any(|v| !v.is_finite()) {
            return Err(format!("launch {launch} wrote a non-finite logit"));
        }
        let want = oracle_router::select_f32(&r, logits_l, r.top_k_max);
        let got = &ids_h[launch * id_len..(launch + 1) * id_len];
        if got != want {
            return Err(format!(
                "launch {launch} ids {got:?}, the logits select {want:?}"
            ));
        }
    }
    Ok(Verdict::Noted(format!(
        "{LAUNCHES} launches, ids match the logits, ticket is 0"
    )))
}

/// The buffers of one router launch. One struct, so a batch encodes without a
/// long argument list.
struct Buffers<'a> {
    routed: &'a MetalBuffer<f32>,
    shared: &'a MetalBuffer<f32>,
    x: &'a MetalBuffer<f32>,
    gain: &'a MetalBuffer<f32>,
    partials: &'a mut MetalBuffer<f32>,
    ticket: &'a mut MetalBuffer<u32>,
    logits: &'a mut MetalBuffer<f32>,
    ids: &'a mut MetalBuffer<u32>,
    weights: &'a mut MetalBuffer<f32>,
}

fn dispatch_ticket(
    h: &Harness<'_>,
    batch: usize,
    end: usize,
    bufs: &mut Buffers<'_>,
    eps: f32,
) -> Result<(), String> {
    let r = h.geom().router;
    let id_len = written(&r);
    let fault = h.device.fault_word();
    h.encode(|encoder| {
        for launch in batch..end {
            h.kernels.router(
                encoder,
                RouterParams {
                    rows: r.rows,
                    k: r.k,
                    top_k: r.top_k_max,
                    routed: bufs.routed.window(Elems(0), Elems(r.experts * r.k))?,
                    shared: (r.shared == 1)
                        .then(|| bufs.shared.window(Elems(0), Elems(r.k)))
                        .transpose()?,
                    x: bufs.x.window(Elems(launch * r.k), Elems(r.k))?,
                    gain: bufs.gain.window(Elems(0), Elems(r.k))?,
                    partials: bufs.partials.window_mut(Elems(0), Elems(r.partials))?,
                    ticket: bufs.ticket.window_mut(Elems(0), Elems(1))?,
                    logits: bufs
                        .logits
                        .window_mut(Elems(launch * r.rows), Elems(r.rows))?,
                    ids: bufs.ids.window_mut(Elems(launch * id_len), Elems(id_len))?,
                    weights: bufs
                        .weights
                        .window_mut(Elems(launch * id_len), Elems(id_len))?,
                    eps,
                    fault,
                    layer: LAYER,
                },
            )?;
        }
        Ok(())
    })
}

pub fn router_nan(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let r = h.geom().router;
    h.clear_fault()?;
    let mut weights = load_weights(&r, fixture)?;
    weights.routed[0] = f32::NAN;
    let x = oracle::activations(r.k, 1, 0.5);
    let out = run(
        h,
        &weights.routed,
        &weights.shared,
        &x,
        &weights.gain,
        weights.eps,
    )?;
    common::expect_fault(h, "a non-finite weight", FaultSite::Router)?;
    for (s, &id) in out.ids.iter().enumerate() {
        if s < r.top_k_max {
            if id >= r.experts as u32 {
                return Err(format!("slot {s} id {id} is past the routed experts"));
            }
        } else if id != r.experts as u32 {
            return Err(format!(
                "shared id {id}, the shared expert is {}",
                r.experts
            ));
        }
    }
    if let Some(s) = out.weights.iter().position(|w| w.is_finite()) {
        return Err(format!("slot {s} weight {} is finite", out.weights[s]));
    }
    Ok(Verdict::Noted(
        "router site, ids in range, weights non-finite".to_string(),
    ))
}

/// The input the norm cannot take: every threadgroup's rows write NaN
/// partials, the tail's router site joins the norm site in the fault word,
/// and the outputs are the non-finite set. The norm is the threadgroup's, so
/// the clause pins the whole threadgroup's rows going NaN together.
pub fn router_norm_nan(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let r = h.geom().router;
    h.clear_fault()?;
    let weights = load_weights(&r, fixture)?;
    let mut x = oracle::activations(r.k, 1, 0.5);
    x[17] = f32::INFINITY;
    let out = run(
        h,
        &weights.routed,
        &weights.shared,
        &x,
        &weights.gain,
        weights.eps,
    )?;
    let both = (1u32 << (FaultSite::NormQuant as u32)) | (1u32 << (FaultSite::Router as u32));
    match h.fault()? {
        Some(f) if f.layer == LAYER && f.sites == both => {}
        Some(f) => {
            return Err(format!(
                "a non-finite input raised {f}, not the norm and router sites in layer {LAYER}"
            ));
        }
        None => {
            return Err("a non-finite input raised nothing".to_string());
        }
    }
    if let Some(r) = out.logits.iter().position(|v| v.is_finite()) {
        return Err(format!("logit {r} is {:?}, not NaN", out.logits[r]));
    }
    for (s, &id) in out.ids.iter().enumerate() {
        if s < r.top_k_max {
            if id >= r.experts as u32 {
                return Err(format!("slot {s} id {id} is past the routed experts"));
            }
        } else if id != r.experts as u32 {
            return Err(format!(
                "shared id {id}, the shared expert is {}",
                r.experts
            ));
        }
    }
    if let Some(s) = out.weights.iter().position(|w| w.is_finite()) {
        return Err(format!("slot {s} weight {} is finite", out.weights[s]));
    }
    Ok(Verdict::Noted(
        "norm and router sites, every logit NaN, ids in range, weights non-finite".to_string(),
    ))
}

/// The launch of one case: a shape, an epsilon or a window, one of them bad.
#[derive(Clone, Copy)]
struct Args {
    k: usize,
    rows: usize,
    top_k: usize,
    eps: f32,
    routed: usize,
    shared: usize,
    x_len: usize,
    x_at: usize,
    gain: usize,
    partials: usize,
    ticket: usize,
    logits: usize,
    ids: usize,
    weights: usize,
}

impl Args {
    fn whole(r: &RouterGeom) -> Self {
        Self {
            k: r.k,
            rows: r.rows,
            top_k: r.top_k_max,
            eps: 1e-5,
            routed: r.experts * r.k,
            shared: r.shared * r.k,
            x_len: r.k,
            x_at: 0,
            gain: r.k,
            partials: r.partials,
            ticket: 1,
            logits: r.rows,
            ids: written(r),
            weights: written(r),
        }
    }
}

fn refuse(
    h: &Harness<'_>,
    bufs: &mut Buffers<'_>,
    a: &Args,
    encoder: &mut bloomery_metal::encode::ComputeEncoder<'_>,
) -> Result<(), DeviceError> {
    h.kernels.router(
        encoder,
        RouterParams {
            rows: a.rows,
            k: a.k,
            top_k: a.top_k,
            routed: bufs.routed.window(Elems(0), Elems(a.routed))?,
            shared: (a.shared > 0)
                .then(|| bufs.shared.window(Elems(0), Elems(a.shared)))
                .transpose()?,
            x: bufs.x.window(Elems(a.x_at), Elems(a.x_len))?,
            gain: bufs.gain.window(Elems(0), Elems(a.gain))?,
            partials: bufs.partials.window_mut(Elems(0), Elems(a.partials))?,
            ticket: bufs.ticket.window_mut(Elems(0), Elems(a.ticket))?,
            logits: bufs.logits.window_mut(Elems(0), Elems(a.logits))?,
            ids: bufs.ids.window_mut(Elems(0), Elems(a.ids))?,
            weights: bufs.weights.window_mut(Elems(0), Elems(a.weights))?,
            eps: a.eps,
            fault: h.device.fault_word(),
            layer: LAYER,
        },
    )
}

fn bad<'a>(
    what: &'static str,
    needle: &'static str,
    edit: impl FnOnce(&mut Args) + 'a,
    h: &'a Harness<'_>,
) -> BadArg<'a, Buffers<'a>> {
    let whole = Args::whole(&h.geom().router);
    BadArg::new(
        what,
        needle,
        Box::new(move |bufs, encoder| {
            let mut a = whole;
            edit(&mut a);
            refuse(h, bufs, &a, encoder)
        }),
    )
}

pub fn router_bounds(h: &Harness<'_>) -> Clause {
    let r = h.geom().router;
    let (device, stream) = (h.device, h.stream);
    let routed = device
        .upload(stream, &vec![0f32; r.experts * r.k])
        .map_err(why)?;
    let shared = device.upload(stream, &vec![0f32; r.k]).map_err(why)?;
    let x = device.upload(stream, &vec![0f32; r.k + 8]).map_err(why)?;
    let gain = device.upload(stream, &vec![1f32; r.k]).map_err(why)?;
    let mut partials = device
        .upload(stream, &vec![0f32; r.partials])
        .map_err(why)?;
    let mut ticket = device.upload(stream, &[0u32, 0]).map_err(why)?;
    let mut logits = device
        .upload(stream, &vec![f32::from_bits(SENTINEL_BITS); r.rows])
        .map_err(why)?;
    let mut ids = device
        .upload(stream, &vec![SENTINEL_BITS; written(&r)])
        .map_err(why)?;
    let mut weights = device
        .upload(stream, &vec![f32::from_bits(SENTINEL_BITS); written(&r)])
        .map_err(why)?;
    let mut bufs = Buffers {
        routed: &routed,
        shared: &shared,
        x: &x,
        gain: &gain,
        partials: &mut partials,
        ticket: &mut ticket,
        logits: &mut logits,
        ids: &mut ids,
        weights: &mut weights,
    };
    let cases = vec![
        bad("K of 2047", "2048", move |a| a.k = r.k - 1, h),
        bad("K of 4096", "2048", move |a| a.k = r.k * 2, h),
        bad(
            "one row short",
            "row count",
            move |a| a.rows = r.rows - 1,
            h,
        ),
        bad("one row over", "row count", move |a| a.rows = r.rows + 1, h),
        bad("top-k of 0", "top-k", |a| a.top_k = 0, h),
        bad("top-k of 9", "top-k", move |a| a.top_k = r.top_k_max + 1, h),
        bad("top-k of 32", "top-k", |a| a.top_k = 32, h),
        bad("a non-finite eps", "eps", |a| a.eps = f32::NAN, h),
        bad("an infinite eps", "eps", |a| a.eps = f32::INFINITY, h),
        bad(
            "a short routed window",
            "routed",
            move |a| a.routed = r.experts * r.k - 1,
            h,
        ),
        if r.shared == 1 {
            bad(
                "a short shared window",
                "shared",
                move |a| a.shared = r.k - 1,
                h,
            )
        } else {
            bad(
                "a shared row the router has none of",
                "shared expert's row",
                move |a| a.shared = r.k,
                h,
            )
        },
        bad(
            "a short x window",
            "x window",
            move |a| a.x_len = r.k - 1,
            h,
        ),
        bad("a short gain window", "gain", move |a| a.gain = r.k - 1, h),
        bad(
            "a short partials window",
            "partials",
            move |a| a.partials = r.partials - 1,
            h,
        ),
        bad("an empty ticket window", "ticket", |a| a.ticket = 0, h),
        bad(
            "a short logits window",
            "logits",
            move |a| a.logits = r.rows - 1,
            h,
        ),
        bad(
            "a short ids window",
            "ids",
            move |a| a.ids = written(&r) - 1,
            h,
        ),
        bad(
            "a short weights window",
            "weights",
            move |a| a.weights = written(&r) - 1,
            h,
        ),
        bad(
            "an x window that is not 16-byte aligned",
            "16-byte aligned",
            |a| a.x_at = 1,
            h,
        ),
    ];
    let unchanged = |h: &Harness<'_>, b: &Buffers<'_>| {
        common::still(h, b.logits, |v| v.to_bits() == SENTINEL_BITS)?;
        common::still(h, b.ids, |v| *v == SENTINEL_BITS)?;
        common::still(h, b.weights, |v| v.to_bits() == SENTINEL_BITS)
    };
    let count = common::refuse_launches(h, "router", &mut bufs, cases, unchanged)?;
    let good: Edit<'_, Buffers<'_>> = Box::new(|bufs, encoder| {
        let a = Args::whole(&r);
        refuse(h, bufs, &a, encoder)
    });
    common::accepted_writes(h, &mut bufs, good, |h, b| {
        common::moved(h, b.logits, |v| v.to_bits() != SENTINEL_BITS)
    })?;
    Ok(Verdict::Noted(format!(
        "{count} refusals left the outputs unchanged, and an accepted launch wrote"
    )))
}

/// The router clauses, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, super::ClauseRun<'a>)> {
    vec![
        ("router_logit", Box::new(|| ctx.with(router_logit))),
        ("router_ids", Box::new(|| ctx.with(router_ids))),
        ("router_tie", Box::new(|| router_tie(ctx.harness))),
        ("router_weights", Box::new(|| ctx.with(router_weights))),
        ("router_ticket", Box::new(|| ctx.with(router_ticket))),
        ("router_nan", Box::new(|| ctx.with(router_nan))),
        ("router_norm_nan", Box::new(|| ctx.with(router_norm_nan))),
        ("router_bounds", Box::new(|| router_bounds(ctx.harness))),
    ]
}
