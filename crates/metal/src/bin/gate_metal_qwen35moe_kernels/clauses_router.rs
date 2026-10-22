//! The router's clauses.

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, DeviceError, Elems};
use bloomery_metal::buffer::MetalBuffer;
use bloomery_metal::kernels::router::{EXPERTS, K, PARTIALS, ROWS, RouterParams, TOP_K_MAX};
use gguf::Gguf;
use gguf::quant::{GgmlType, dequant_row};

use super::fixture::Fixture;
use super::oracle;
use super::oracle_router::{self, Ranking, Step};
use super::{Harness, SENTINEL_BITS};

/// The layer the clauses launch in.
const LAYER: u32 = 5;

/// The routed count the numeric clauses select.
const TOP_K: usize = TOP_K_MAX;

const ROUTED_TENSOR: &str = "blk.0.ffn_gate_inp.weight";
const SHARED_TENSOR: &str = "blk.0.ffn_gate_inp_shexp.weight";
const GAIN_TENSOR: &str = "blk.0.post_attention_norm.weight";

struct Weights {
    routed: Vec<f32>,
    shared: Vec<f32>,
    gain: Vec<f32>,
    eps: f32,
}

struct Out {
    logits: Vec<f32>,
    ids: Vec<u32>,
    weights: Vec<f32>,
}

fn load_weights(fixture: &Fixture) -> Result<Weights, String> {
    let file = Gguf::open(&fixture.path).map_err(|e| format!("{ROUTED_TENSOR}: {e}"))?;
    let t = file
        .find(ROUTED_TENSOR)
        .ok_or_else(|| format!("the fixture has no tensor {ROUTED_TENSOR}"))?;
    if t.ty != GgmlType::F32 || t.dims != [K as u64, EXPERTS as u64] {
        return Err(format!(
            "{ROUTED_TENSOR} is {:?} {:?}, the router reads {EXPERTS} rows of {K} f32s",
            t.ty, t.dims
        ));
    }
    let bytes = file.data(t).map_err(|e| format!("{ROUTED_TENSOR}: {e}"))?;
    let mut routed = vec![0f32; EXPERTS * K];
    dequant_row(GgmlType::F32, bytes, &mut routed).map_err(|e| format!("{ROUTED_TENSOR}: {e}"))?;
    Ok(Weights {
        routed,
        shared: fixture.f32s(SHARED_TENSOR, K)?,
        gain: fixture.f32s(GAIN_TENSOR, K)?,
        eps: fixture.rms_eps()?,
    })
}

/// One nonzero column, so each logit is that column's scaled product and the
/// 8th-vs-9th gap is the gap between the targets.
fn constructed(eps: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut x = vec![0f32; K];
    x[0] = 1.0;
    let gain = vec![1.0f32; K];
    let xhat = oracle::rms_norm(&x, &gain, eps);
    let x0 = xhat[0];
    let mut routed = vec![0f32; EXPERTS * K];
    for i in 0..EXPERTS {
        let target = if i < TOP_K {
            4.0 - 0.5 * f64::from(u32::try_from(i).unwrap_or(0))
        } else {
            -2.0 - 0.01 * f64::from(u32::try_from(i - TOP_K).unwrap_or(0))
        };
        routed[i * K] = (target / x0) as f32;
    }
    let mut shared = vec![0f32; K];
    shared[0] = (0.5 / x0) as f32;
    (routed, shared, x, gain)
}

/// The tie construction: rows `LOW` and `HIGH` carry the same weights, so
/// their logits are the same bits, and the tie sits at the 8th and 9th
/// places of the ranking.
fn constructed_tie(eps: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    const ABOVE: usize = 7;
    const TIE: f64 = 0.75;
    let mut x = vec![0f32; K];
    x[0] = 1.0;
    let gain = vec![1.0f32; K];
    let xhat = oracle::rms_norm(&x, &gain, eps);
    let x0 = xhat[0];
    let mut routed = vec![0f32; EXPERTS * K];
    for i in 0..EXPERTS {
        let target = if i < ABOVE {
            4.0 - 0.5 * f64::from(u32::try_from(i).unwrap_or(0))
        } else if i == ABOVE || i == ABOVE + 1 {
            TIE
        } else {
            -2.0 - 0.01 * f64::from(u32::try_from(i - ABOVE - 2).unwrap_or(0))
        };
        routed[i * K] = (target / x0) as f32;
    }
    let mut shared = vec![0f32; K];
    shared[0] = (0.5 / x0) as f32;
    (routed, shared, x, gain)
}

fn run(
    h: &Harness<'_>,
    routed: &[f32],
    shared: &[f32],
    x: &[f32],
    gain: &[f32],
    eps: f32,
) -> Result<Out, String> {
    let (device, stream) = (h.device, h.stream);
    let routed_b = device.upload(stream, routed).map_err(why)?;
    let shared_b = device.upload(stream, shared).map_err(why)?;
    let x_b = device.upload(stream, x).map_err(why)?;
    let gain_b = device.upload(stream, gain).map_err(why)?;
    let mut partials = device.upload(stream, &vec![0f32; PARTIALS]).map_err(why)?;
    let mut ticket = device.upload(stream, &[0u32]).map_err(why)?;
    let mut logits = device.upload(stream, &vec![0f32; ROWS]).map_err(why)?;
    let mut ids = device
        .upload(stream, &[0xFFFFu32; TOP_K + 1])
        .map_err(why)?;
    let mut weights = device
        .upload(stream, &[f32::from_bits(SENTINEL_BITS); TOP_K + 1])
        .map_err(why)?;
    h.encode(|encoder| {
        h.kernels.router(
            encoder,
            RouterParams {
                rows: ROWS,
                k: K,
                top_k: TOP_K,
                routed: routed_b.window(Elems(0), Elems(routed.len()))?,
                shared: shared_b.window(Elems(0), Elems(shared.len()))?,
                x: x_b.window(Elems(0), Elems(x.len()))?,
                gain: gain_b.window(Elems(0), Elems(gain.len()))?,
                partials: partials.window_mut(Elems(0), Elems(PARTIALS))?,
                ticket: ticket.window_mut(Elems(0), Elems(1))?,
                logits: logits.window_mut(Elems(0), Elems(ROWS))?,
                ids: ids.window_mut(Elems(0), Elems(TOP_K + 1))?,
                weights: weights.window_mut(Elems(0), Elems(TOP_K + 1))?,
                eps,
                fault: device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let mut logits_h = vec![0f32; ROWS];
    let mut ids_h = vec![0u32; TOP_K + 1];
    let mut weights_h = vec![0f32; TOP_K + 1];
    device
        .read(stream, logits.whole().map_err(why)?, &mut logits_h)
        .map_err(why)?;
    device
        .read(stream, ids.whole().map_err(why)?, &mut ids_h)
        .map_err(why)?;
    device
        .read(stream, weights.whole().map_err(why)?, &mut weights_h)
        .map_err(why)?;
    Ok(Out {
        logits: logits_h,
        ids: ids_h,
        weights: weights_h,
    })
}

fn clean(h: &Harness<'_>, name: &str) -> Result<(), String> {
    match h.fault()? {
        None => Ok(()),
        Some(f) => Err(format!("{name} raised {f}")),
    }
}

fn match_ids(name: &str, out: &Out, ranking: &Ranking) -> Result<(), String> {
    for (s, &want) in ranking.ids.iter().enumerate() {
        let got = out.ids[s] as usize;
        if got != want {
            return Err(format!(
                "router-ids: step {name} slot {s} is {got}, the f64 ranking has {want}"
            ));
        }
    }
    let shared = out.ids[TOP_K];
    if shared != EXPERTS as u32 {
        return Err(format!(
            "router-ids: step {name} shared id is {shared}, the shared expert is {EXPERTS}"
        ));
    }
    Ok(())
}

fn check_ids(name: &str, out: &Out, step: &Step) -> Result<&'static str, String> {
    let ranking = oracle_router::rank(step);
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
        let mut got: Vec<usize> = out.ids[..TOP_K].iter().map(|id| *id as usize).collect();
        let mut want = ranking.ids.clone();
        got.sort_unstable();
        want.sort_unstable();
        if got != want {
            return Err(format!(
                "router-ids: step {name} id set {got:?}, the f64 ranking has {want:?}"
            ));
        }
        let shared = out.ids[TOP_K];
        if shared != EXPERTS as u32 {
            return Err(format!(
                "router-ids: step {name} shared id is {shared}, the shared expert is {EXPERTS}"
            ));
        }
        return Ok("set");
    }
    match_ids(name, out, &ranking)?;
    Ok("sequence")
}

fn check_weights(name: &str, out: &Out, step: &Step) -> Result<f64, String> {
    let ranking = oracle_router::rank(step);
    if ranking.sequence_stable {
        match_ids(name, out, &ranking)?;
    }
    let ids: Vec<usize> = out.ids[..TOP_K].iter().map(|id| *id as usize).collect();
    let oracle = oracle_router::weights(step, &ids)?;
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
    h.clear_fault()?;
    let weights = load_weights(fixture)?;
    let mut worst = 0.0f64;
    let mut notes = Vec::new();
    for (name, seed) in [("seed-1", 1u64), ("seed-2", 2u64)] {
        let x = oracle::activations(K, seed, 0.5);
        let out = run(
            h,
            &weights.routed,
            &weights.shared,
            &x,
            &weights.gain,
            weights.eps,
        )?;
        clean(h, name)?;
        let step = oracle_router::step(
            name,
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
    let (routed, shared, x, gain) = constructed(eps);
    let out = run(h, &routed, &shared, &x, &gain, eps)?;
    clean(h, "constructed")?;
    let step = oracle_router::step("constructed", &routed, &shared, &x, &gain, eps)?;
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
    h.clear_fault()?;
    let weights = load_weights(fixture)?;
    let mut notes = Vec::new();
    for (name, seed) in [("seed-1", 1u64), ("seed-2", 2u64)] {
        let x = oracle::activations(K, seed, 0.5);
        let out = run(
            h,
            &weights.routed,
            &weights.shared,
            &x,
            &weights.gain,
            weights.eps,
        )?;
        clean(h, name)?;
        let step = oracle_router::step(
            name,
            &weights.routed,
            &weights.shared,
            &x,
            &weights.gain,
            weights.eps,
        )?;
        notes.push(format!("{name} {}", check_ids(name, &out, &step)?));
    }
    let eps = 1e-5;
    let (routed, shared, x, gain) = constructed(eps);
    let out = run(h, &routed, &shared, &x, &gain, eps)?;
    clean(h, "constructed")?;
    let step = oracle_router::step("constructed", &routed, &shared, &x, &gain, eps)?;
    let ranking = oracle_router::rank(&step);
    if !ranking.sequence_stable {
        return Err(format!(
            "router-ids: the constructed step is not stable (margin {:.3e}, band {:.3e})",
            ranking.margin, ranking.margin_band
        ));
    }
    match_ids("constructed", &out, &ranking)?;
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
    const LOW: usize = 7;
    const HIGH: usize = 8;
    const ABOVE: usize = 7;
    let eps = 1e-5;
    h.clear_fault()?;
    let (routed, shared, x, gain) = constructed_tie(eps);
    let out = run(h, &routed, &shared, &x, &gain, eps)?;
    clean(h, "constructed")?;
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
        if i == LOW || i == HIGH || i == EXPERTS {
            continue;
        }
        let above = if i < ABOVE { v > tied } else { v < tied };
        if !above {
            return Err(format!(
                "row {i} is {v:e}, not on its side of the tie {tied:e}"
            ));
        }
    }
    let shared_id = out.ids[TOP_K];
    if shared_id != EXPERTS as u32 {
        return Err(format!(
            "shared id is {shared_id}, the shared expert is {EXPERTS}"
        ));
    }
    for (s, &got) in out.ids[..TOP_K].iter().enumerate() {
        let want = u32::try_from(s).map_err(|_| "a slot id does not fit u32")?;
        if got != want {
            return Err(format!(
                "slot {s} is {got}, the tied ranking selects {want} (the tie is {LOW} and {HIGH})"
            ));
        }
    }
    if out.weights[..TOP_K].iter().any(|w| !w.is_finite()) {
        return Err("a routed weight is not finite".to_string());
    }
    Ok(Verdict::Noted(format!(
        "slots 0..{} are 0..{LOW}; the tie {LOW}/{HIGH} takes {LOW}",
        TOP_K - 1
    )))
}

pub fn router_weights(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    h.clear_fault()?;
    let weights = load_weights(fixture)?;
    let mut worst = 0.0f64;
    let mut notes = Vec::new();
    for (name, seed) in [("seed-1", 1u64), ("seed-2", 2u64)] {
        let x = oracle::activations(K, seed, 0.5);
        let out = run(
            h,
            &weights.routed,
            &weights.shared,
            &x,
            &weights.gain,
            weights.eps,
        )?;
        clean(h, name)?;
        let step = oracle_router::step(
            name,
            &weights.routed,
            &weights.shared,
            &x,
            &weights.gain,
            weights.eps,
        )?;
        let ratio = check_weights(name, &out, &step)?;
        worst = worst.max(ratio);
        notes.push(format!("{name} {ratio:.3}"));
    }
    let eps = 1e-5;
    let (routed, shared, x, gain) = constructed(eps);
    let out = run(h, &routed, &shared, &x, &gain, eps)?;
    clean(h, "constructed")?;
    let step = oracle_router::step("constructed", &routed, &shared, &x, &gain, eps)?;
    let ranking = oracle_router::rank(&step);
    if !ranking.sequence_stable {
        return Err(format!(
            "router-weights: the constructed step is not stable (margin {:.3e}, band {:.3e})",
            ranking.margin, ranking.margin_band
        ));
    }
    let ratio = check_weights("constructed", &out, &step)?;
    worst = worst.max(ratio);
    Ok(Verdict::Noted(format!(
        "worst {worst:.3}; constructed {ratio:.3}; {}",
        notes.join(", ")
    )))
}

pub fn router_ticket(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    const LAUNCHES: usize = 10_000;
    const BATCH: usize = 1_000;
    h.clear_fault()?;
    let weights = load_weights(fixture)?;
    let (device, stream) = (h.device, h.stream);
    let mut xs = vec![0f32; LAUNCHES * K];
    let mut rng = oracle::Rng::new(0x715E_0000);
    for v in &mut xs {
        *v = 0.25 * rng.signed();
    }
    let id_len = TOP_K + 1;
    let routed = device.upload(stream, &weights.routed).map_err(why)?;
    let shared = device.upload(stream, &weights.shared).map_err(why)?;
    let x = device.upload(stream, &xs).map_err(why)?;
    let gain = device.upload(stream, &weights.gain).map_err(why)?;
    let mut partials = device.upload(stream, &vec![0f32; PARTIALS]).map_err(why)?;
    let mut ticket = device.upload(stream, &[0u32]).map_err(why)?;
    let mut logits = device
        .upload(stream, &vec![0f32; LAUNCHES * ROWS])
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
    let mut logits_h = vec![0f32; LAUNCHES * ROWS];
    let mut ids_h = vec![0u32; LAUNCHES * id_len];
    device
        .read(stream, bufs.logits.whole().map_err(why)?, &mut logits_h)
        .map_err(why)?;
    device
        .read(stream, bufs.ids.whole().map_err(why)?, &mut ids_h)
        .map_err(why)?;
    clean(h, "router-ticket")?;
    for launch in 0..LAUNCHES {
        let row = launch * ROWS;
        let logits_l = &logits_h[row..row + ROWS];
        if logits_l.iter().any(|v| !v.is_finite()) {
            return Err(format!("launch {launch} wrote a non-finite logit"));
        }
        let want = oracle_router::select_f32(logits_l, TOP_K);
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
    let id_len = TOP_K + 1;
    let fault = h.device.fault_word();
    h.encode(|encoder| {
        for launch in batch..end {
            h.kernels.router(
                encoder,
                RouterParams {
                    rows: ROWS,
                    k: K,
                    top_k: TOP_K,
                    routed: bufs.routed.window(Elems(0), Elems(EXPERTS * K))?,
                    shared: bufs.shared.window(Elems(0), Elems(K))?,
                    x: bufs.x.window(Elems(launch * K), Elems(K))?,
                    gain: bufs.gain.window(Elems(0), Elems(K))?,
                    partials: bufs.partials.window_mut(Elems(0), Elems(PARTIALS))?,
                    ticket: bufs.ticket.window_mut(Elems(0), Elems(1))?,
                    logits: bufs.logits.window_mut(Elems(launch * ROWS), Elems(ROWS))?,
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
    h.clear_fault()?;
    let mut weights = load_weights(fixture)?;
    weights.routed[0] = f32::NAN;
    let x = oracle::activations(K, 1, 0.5);
    let out = run(
        h,
        &weights.routed,
        &weights.shared,
        &x,
        &weights.gain,
        weights.eps,
    )?;
    match h.fault()? {
        Some(f)
            if f.layer == LAYER
                && f.code == FaultSite::Router as u32
                && f.sites == 1 << (FaultSite::Router as u32) => {}
        Some(f) => {
            return Err(format!(
                "a non-finite weight raised {f}, not {} in layer {LAYER}",
                FaultSite::Router.name()
            ));
        }
        None => return Err("a non-finite weight raised nothing".to_string()),
    }
    for (s, &id) in out.ids.iter().enumerate() {
        if s < TOP_K {
            if id >= EXPERTS as u32 {
                return Err(format!("slot {s} id {id} is past the routed experts"));
            }
        } else if id != EXPERTS as u32 {
            return Err(format!("shared id {id}, the shared expert is {EXPERTS}"));
        }
    }
    if let Some(s) = out.weights.iter().position(|w| w.is_finite()) {
        return Err(format!("slot {s} weight {} is finite", out.weights[s]));
    }
    Ok(Verdict::Noted(
        "router site, ids in range, weights non-finite".to_string(),
    ))
}

#[derive(Clone, Copy)]
struct Refusal {
    what: &'static str,
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
    needle: &'static str,
}

pub fn router_bounds(h: &Harness<'_>) -> Clause {
    let (device, stream) = (h.device, h.stream);
    let routed = device
        .upload(stream, &vec![0f32; EXPERTS * K])
        .map_err(why)?;
    let shared = device.upload(stream, &vec![0f32; K]).map_err(why)?;
    let x = device.upload(stream, &vec![0f32; K + 8]).map_err(why)?;
    let gain = device.upload(stream, &vec![1f32; K]).map_err(why)?;
    let mut partials = device.upload(stream, &vec![0f32; PARTIALS]).map_err(why)?;
    let mut ticket = device.upload(stream, &[0u32, 0]).map_err(why)?;
    let mut logits = device
        .upload(stream, &vec![f32::from_bits(SENTINEL_BITS); ROWS])
        .map_err(why)?;
    let mut ids = device
        .upload(stream, &[SENTINEL_BITS; TOP_K + 1])
        .map_err(why)?;
    let mut weights = device
        .upload(stream, &[f32::from_bits(SENTINEL_BITS); TOP_K + 1])
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
    let whole = Refusal {
        what: "",
        k: K,
        rows: ROWS,
        top_k: TOP_K,
        eps: 1e-5,
        routed: EXPERTS * K,
        shared: K,
        x_len: K,
        x_at: 0,
        gain: K,
        partials: PARTIALS,
        ticket: 1,
        logits: ROWS,
        ids: TOP_K + 1,
        weights: TOP_K + 1,
        needle: "",
    };
    let cases = refusal_cases(whole);
    for case in cases {
        let refused = h.encode(|encoder| refuse_launch(h, encoder, &case, &mut bufs));
        match refused {
            Ok(()) => return Err(format!("{} was accepted", case.what)),
            Err(text) if text.contains(case.needle) => {}
            Err(text) => {
                return Err(format!(
                    "{} was refused without naming {}: {text}",
                    case.what, case.needle
                ));
            }
        }
        still_sentinel(h, bufs.logits, bufs.ids, bufs.weights)
            .map_err(|e| format!("{}: {e}", case.what))?;
    }
    let accepted = h.encode(|encoder| refuse_launch(h, encoder, &whole, &mut bufs));
    accepted?;
    let mut back = vec![0f32; ROWS];
    device
        .read(stream, bufs.logits.whole().map_err(why)?, &mut back)
        .map_err(why)?;
    if back.iter().all(|v| v.to_bits() == SENTINEL_BITS) {
        return Err("an accepted launch wrote nothing".to_string());
    }
    Ok(Verdict::Noted(format!(
        "{} refusals left the outputs unchanged, and an accepted launch wrote",
        cases.len()
    )))
}

fn refusal_cases(whole: Refusal) -> [Refusal; 19] {
    [
        Refusal {
            what: "K of 2047",
            k: 2047,
            needle: "2048",
            ..whole
        },
        Refusal {
            what: "K of 4096",
            k: 4096,
            needle: "2048",
            ..whole
        },
        Refusal {
            what: "256 rows",
            rows: 256,
            needle: "257",
            ..whole
        },
        Refusal {
            what: "258 rows",
            rows: 258,
            needle: "257",
            ..whole
        },
        Refusal {
            what: "top-k of 0",
            top_k: 0,
            needle: "top-k",
            ..whole
        },
        Refusal {
            what: "top-k of 9",
            top_k: 9,
            needle: "top-k",
            ..whole
        },
        Refusal {
            what: "top-k of 32",
            top_k: 32,
            needle: "top-k",
            ..whole
        },
        Refusal {
            what: "a non-finite eps",
            eps: f32::NAN,
            needle: "eps",
            ..whole
        },
        Refusal {
            what: "an infinite eps",
            eps: f32::INFINITY,
            needle: "eps",
            ..whole
        },
        Refusal {
            what: "a short routed window",
            routed: EXPERTS * K - 1,
            needle: "routed",
            ..whole
        },
        Refusal {
            what: "a short shared window",
            shared: K - 1,
            needle: "shared",
            ..whole
        },
        Refusal {
            what: "a short x window",
            x_len: K - 1,
            needle: "x window",
            ..whole
        },
        Refusal {
            what: "a short gain window",
            gain: K - 1,
            needle: "gain",
            ..whole
        },
        Refusal {
            what: "a short partials window",
            partials: PARTIALS - 1,
            needle: "partials",
            ..whole
        },
        Refusal {
            what: "an empty ticket window",
            ticket: 0,
            needle: "ticket",
            ..whole
        },
        Refusal {
            what: "a short logits window",
            logits: ROWS - 1,
            needle: "logits",
            ..whole
        },
        Refusal {
            what: "a short ids window",
            ids: TOP_K,
            needle: "ids",
            ..whole
        },
        Refusal {
            what: "a short weights window",
            weights: TOP_K,
            needle: "weights",
            ..whole
        },
        Refusal {
            what: "an x window that is not 16-byte aligned",
            x_at: 1,
            needle: "16-byte aligned",
            ..whole
        },
    ]
}

fn refuse_launch(
    h: &Harness<'_>,
    encoder: &mut bloomery_metal::encode::ComputeEncoder<'_>,
    case: &Refusal,
    bufs: &mut Buffers<'_>,
) -> Result<(), DeviceError> {
    h.kernels.router(
        encoder,
        RouterParams {
            rows: case.rows,
            k: case.k,
            top_k: case.top_k,
            routed: bufs.routed.window(Elems(0), Elems(case.routed))?,
            shared: bufs.shared.window(Elems(0), Elems(case.shared))?,
            x: bufs.x.window(Elems(case.x_at), Elems(case.x_len))?,
            gain: bufs.gain.window(Elems(0), Elems(case.gain))?,
            partials: bufs.partials.window_mut(Elems(0), Elems(case.partials))?,
            ticket: bufs.ticket.window_mut(Elems(0), Elems(case.ticket))?,
            logits: bufs.logits.window_mut(Elems(0), Elems(case.logits))?,
            ids: bufs.ids.window_mut(Elems(0), Elems(case.ids))?,
            weights: bufs.weights.window_mut(Elems(0), Elems(case.weights))?,
            eps: case.eps,
            fault: h.device.fault_word(),
            layer: LAYER,
        },
    )
}

fn still_sentinel(
    h: &Harness<'_>,
    logits: &MetalBuffer<f32>,
    ids: &MetalBuffer<u32>,
    weights: &MetalBuffer<f32>,
) -> Result<(), String> {
    let (device, stream) = (h.device, h.stream);
    let mut logits_h = vec![0f32; ROWS];
    let mut ids_h = vec![0u32; TOP_K + 1];
    let mut weights_h = vec![0f32; TOP_K + 1];
    device
        .read(stream, logits.whole().map_err(why)?, &mut logits_h)
        .map_err(why)?;
    device
        .read(stream, ids.whole().map_err(why)?, &mut ids_h)
        .map_err(why)?;
    device
        .read(stream, weights.whole().map_err(why)?, &mut weights_h)
        .map_err(why)?;
    if logits_h.iter().any(|v| v.to_bits() != SENTINEL_BITS)
        || ids_h.iter().any(|v| *v != SENTINEL_BITS)
        || weights_h.iter().any(|v| v.to_bits() != SENTINEL_BITS)
    {
        return Err("a refused launch wrote".to_string());
    }
    Ok(())
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
        ("router_bounds", Box::new(|| router_bounds(ctx.harness))),
    ]
}
