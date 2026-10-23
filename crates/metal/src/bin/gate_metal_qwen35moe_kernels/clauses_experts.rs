//! The routed pair and the folded down. The pair norms its input as it loads
//! it, with a synthetic gain and the fixture's epsilon.

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, Elems};
use bloomery_metal::MetalBuffer;
use bloomery_metal::kernels::{
    BLOCK_VALUES, EXPERT_K, EXPERT_SLOTS, FoldParams, PairParams, Quant, ROUTED_IDS, SHARED_SLOT,
    SharedPlan,
};
use gguf::quant::GgmlType;

use super::common::{self, BadArg, Edit, LAYER};
use super::fixture::Fixture;
use super::oracle;
use super::oracle_experts;
use super::{
    Harness, OUT_GUARD, SENTINEL_BITS, VALUE_PAD, WEIGHT_PAD, WEIGHT_TAIL, check_guards, padded,
};
const N_EXPERTS: usize = 4;
const PAIR_COLS: usize = 2048;
const FOLD_ROWS: usize = 2048;
const IDS: [i32; ROUTED_IDS] = [0, 1, 2, 3, 1, 2, 3, 0];
const ROUTE: [f32; EXPERT_SLOTS] = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8];

/// The norm a pair launch applies to its input.
struct PairNorm {
    gain: Vec<f32>,
    eps: f32,
}

/// A positive synthetic gain, not the fixture's FFN gain: that one is all
/// ones, under which a dropped gain is invisible. The epsilon is the
/// fixture's.
fn ffn_norm(fix: &Fixture) -> Result<PairNorm, String> {
    Ok(PairNorm {
        gain: oracle::activations(PAIR_COLS, 0x6C0, 0.5)
            .iter()
            .map(|g| 1.0 + g.abs())
            .collect(),
        eps: fix.rms_eps()?,
    })
}

fn row_bytes(quant: Quant, cols: usize) -> usize {
    cols / BLOCK_VALUES * quant.block_bytes()
}

fn tag(quant: Quant) -> u64 {
    match quant {
        Quant::Q4K => 0,
        Quant::Q6K => 1,
    }
}

struct PairWeights {
    gate: Vec<u8>,
    up: Vec<u8>,
    shared_gate: Vec<u8>,
    shared_up: Vec<u8>,
}

fn stack(
    fix: &Fixture,
    name: &str,
    ty: GgmlType,
    cols: usize,
    rows: usize,
    from: usize,
    count: usize,
) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for expert in from..from + count {
        out.extend_from_slice(fix.expert(name, ty, cols, rows, expert)?);
    }
    Ok(out)
}

fn plane(
    fix: &Fixture,
    name: &str,
    ty: GgmlType,
    cols: usize,
    rows: usize,
) -> Result<Vec<u8>, String> {
    Ok(fix.rows(name, ty, cols, rows)?.to_vec())
}

/// Q4 reads the gate and up experts. Q6 has no gate or up experts in the
/// fixture, so each expert is one Q6_K down expert's blocks, read as 512 by
/// 2048: the byte length is the same.
fn pair_weights(fix: &Fixture, quant: Quant) -> Result<PairWeights, String> {
    let ty = quant.ggml();
    let weights = match quant {
        Quant::Q4K => PairWeights {
            gate: stack(
                fix,
                "blk.0.ffn_gate_exps.weight",
                ty,
                PAIR_COLS,
                EXPERT_K,
                0,
                N_EXPERTS,
            )?,
            up: stack(
                fix,
                "blk.0.ffn_up_exps.weight",
                ty,
                PAIR_COLS,
                EXPERT_K,
                0,
                N_EXPERTS,
            )?,
            shared_gate: plane(fix, "blk.0.ffn_gate_shexp.weight", ty, PAIR_COLS, EXPERT_K)?,
            shared_up: plane(fix, "blk.0.ffn_up_shexp.weight", ty, PAIR_COLS, EXPERT_K)?,
        },
        Quant::Q6K => PairWeights {
            gate: stack(
                fix,
                "blk.2.ffn_down_exps.weight",
                ty,
                EXPERT_K,
                PAIR_COLS,
                0,
                N_EXPERTS,
            )?,
            up: stack(
                fix,
                "blk.2.ffn_down_exps.weight",
                ty,
                EXPERT_K,
                PAIR_COLS,
                8,
                N_EXPERTS,
            )?,
            shared_gate: plane(fix, "blk.2.ffn_down_shexp.weight", ty, EXPERT_K, PAIR_COLS)?,
            shared_up: fix
                .expert("blk.2.ffn_down_exps.weight", ty, EXPERT_K, PAIR_COLS, 16)?
                .to_vec(),
        },
    };
    let one = weights.gate.len() / N_EXPERTS;
    if weights.up.len() != weights.gate.len()
        || weights.shared_gate.len() != one
        || weights.shared_up.len() != one
    {
        return Err(format!(
            "{} pair weights are not expert-sized",
            quant.name()
        ));
    }
    if weights.shared_gate == weights.gate[..one] || weights.shared_up == weights.up[..one] {
        return Err(format!(
            "{} shared expert matches expert 0, so the slot select cannot be seen",
            quant.name()
        ));
    }
    Ok(weights)
}

struct FoldWeights {
    down: Vec<u8>,
    shared: Vec<u8>,
}

fn fold_weights(fix: &Fixture, quant: Quant) -> Result<FoldWeights, String> {
    let ty = quant.ggml();
    let (name, shared) = match quant {
        Quant::Q4K => ("blk.0.ffn_down_exps.weight", "blk.0.ffn_down_shexp.weight"),
        Quant::Q6K => ("blk.2.ffn_down_exps.weight", "blk.2.ffn_down_shexp.weight"),
    };
    let weights = FoldWeights {
        down: stack(fix, name, ty, EXPERT_K, FOLD_ROWS, 0, N_EXPERTS)?,
        shared: plane(fix, shared, ty, EXPERT_K, FOLD_ROWS)?,
    };
    let one = weights.down.len() / N_EXPERTS;
    if weights.shared.len() != one {
        return Err(format!("{} shared down is not one expert", quant.name()));
    }
    if weights.shared == weights.down[..one] {
        return Err(format!(
            "{} shared down matches expert 0, so the slot select cannot be seen",
            quant.name()
        ));
    }
    Ok(weights)
}

struct PairIn<'a> {
    quant: Quant,
    n_experts: usize,
    ids: &'a [i32],
    gate: &'a [u8],
    up: &'a [u8],
    shared_gate: &'a [u8],
    shared_up: &'a [u8],
    x: &'a [f32],
    norm: &'a PairNorm,
    y0: &'a [f32],
}

fn read_interior(
    h: &Harness<'_>,
    y: &impl Buffer<f32, Device = bloomery_metal::device::MetalDevice>,
    rows: usize,
) -> Result<Vec<f32>, String> {
    let mut back = vec![0f32; rows + 2 * OUT_GUARD];
    h.device
        .read(h.stream, y.whole().map_err(why)?, &mut back)
        .map_err(why)?;
    check_guards(&back, rows)?;
    Ok(back[OUT_GUARD..OUT_GUARD + rows].to_vec())
}

fn launch_pair(h: &Harness<'_>, launch: PairIn<'_>) -> Result<Vec<f32>, String> {
    let (device, stream) = (h.device, h.stream);
    let y_len = EXPERT_SLOTS * EXPERT_K;
    let gate = padded(0xFFu8, WEIGHT_PAD, launch.gate, WEIGHT_TAIL);
    let up = padded(0xFFu8, WEIGHT_PAD, launch.up, WEIGHT_TAIL);
    let shared_gate = padded(0xFFu8, WEIGHT_PAD, launch.shared_gate, WEIGHT_TAIL);
    let shared_up = padded(0xFFu8, WEIGHT_PAD, launch.shared_up, WEIGHT_TAIL);
    let x = padded(f32::from_bits(SENTINEL_BITS), VALUE_PAD, launch.x, 1);
    let gain = padded(
        f32::from_bits(SENTINEL_BITS),
        VALUE_PAD,
        &launch.norm.gain,
        1,
    );
    let ids = padded(-1, VALUE_PAD, launch.ids, 1);
    let y_words = padded(
        f32::from_bits(SENTINEL_BITS),
        OUT_GUARD,
        launch.y0,
        OUT_GUARD,
    );
    let gate = device.upload(stream, &gate).map_err(why)?;
    let up = device.upload(stream, &up).map_err(why)?;
    let shared_gate = device.upload(stream, &shared_gate).map_err(why)?;
    let shared_up = device.upload(stream, &shared_up).map_err(why)?;
    let x = device.upload(stream, &x).map_err(why)?;
    let gain = device.upload(stream, &gain).map_err(why)?;
    let ids = device.upload(stream, &ids).map_err(why)?;
    let mut y = device.upload(stream, &y_words).map_err(why)?;
    let n_experts = launch.n_experts;
    let quant = launch.quant;
    h.encode(|encoder| {
        h.kernels.pair(
            encoder,
            PairParams {
                quant,
                slots: EXPERT_SLOTS,
                k: EXPERT_K,
                cols: PAIR_COLS,
                n_experts,
                ids: ids.window(Elems(VALUE_PAD), Elems(launch.ids.len()))?,
                gate: gate.window(Elems(WEIGHT_PAD), Elems(launch.gate.len()))?,
                up: up.window(Elems(WEIGHT_PAD), Elems(launch.up.len()))?,
                shared_gate: shared_gate
                    .window(Elems(WEIGHT_PAD), Elems(launch.shared_gate.len()))?,
                shared_up: shared_up.window(Elems(WEIGHT_PAD), Elems(launch.shared_up.len()))?,
                x: x.window(Elems(VALUE_PAD), Elems(launch.x.len()))?,
                gain: gain.window(Elems(VALUE_PAD), Elems(launch.norm.gain.len()))?,
                eps: launch.norm.eps,
                y: y.window_mut(Elems(OUT_GUARD), Elems(y_len))?,
                out_stride: EXPERT_K,
                fault: device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    read_interior(h, &y, y_len)
}

struct FoldIn<'a> {
    quant: Quant,
    n_experts: usize,
    ids: &'a [i32],
    down: &'a [u8],
    shared: &'a [u8],
    h: &'a [f32],
    w: &'a [f32],
    y0: &'a [f32],
}

fn launch_fold(h: &Harness<'_>, launch: FoldIn<'_>) -> Result<Vec<f32>, String> {
    let (device, stream) = (h.device, h.stream);
    let down = padded(0xFFu8, WEIGHT_PAD, launch.down, WEIGHT_TAIL);
    let shared = padded(0xFFu8, WEIGHT_PAD, launch.shared, WEIGHT_TAIL);
    let hidden = padded(f32::from_bits(SENTINEL_BITS), VALUE_PAD, launch.h, 1);
    let route = padded(0.0, VALUE_PAD, launch.w, 1);
    let ids = padded(-1, VALUE_PAD, launch.ids, 1);
    let y_words = padded(
        f32::from_bits(SENTINEL_BITS),
        OUT_GUARD,
        launch.y0,
        OUT_GUARD,
    );
    let down = device.upload(stream, &down).map_err(why)?;
    let shared = device.upload(stream, &shared).map_err(why)?;
    let hidden = device.upload(stream, &hidden).map_err(why)?;
    let route = device.upload(stream, &route).map_err(why)?;
    let ids = device.upload(stream, &ids).map_err(why)?;
    let mut y = device.upload(stream, &y_words).map_err(why)?;
    let quant = launch.quant;
    let n_experts = launch.n_experts;
    let rows = launch.y0.len();
    h.encode(|encoder| {
        let shared = shared.window(Elems(WEIGHT_PAD), Elems(launch.shared.len()))?;
        let out = y.window_mut(Elems(OUT_GUARD), Elems(rows))?;
        h.kernels.fold(
            encoder,
            FoldParams {
                quant,
                shared: SharedPlan::Same(shared),
                rows,
                slots: EXPERT_SLOTS,
                k: EXPERT_K,
                n_experts,
                ids: ids.window(Elems(VALUE_PAD), Elems(launch.ids.len()))?,
                down: down.window(Elems(WEIGHT_PAD), Elems(launch.down.len()))?,
                h: hidden.window(Elems(VALUE_PAD), Elems(launch.h.len()))?,
                w: route.window(Elems(VALUE_PAD), Elems(launch.w.len()))?,
                y: &out,
                fault: device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    read_interior(h, &y, rows)
}

/// The words `got` holds, against a solo launch of the same rows.
fn expect_bits(got: &[f32], want: &[f32], what: &str) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!(
            "{what}: {} values against {}",
            got.len(),
            want.len()
        ));
    }
    if let Some((i, g, w)) = common::first_diff(got, want) {
        return Err(format!("{what} value {i}: got {g:?}, the solo read {w:?}"));
    }
    Ok(())
}

fn check_pair_quant(h: &Harness<'_>, fix: &Fixture, quant: Quant) -> Result<String, String> {
    let weights = pair_weights(fix, quant)?;
    let norm = ffn_norm(fix)?;
    let x = oracle::activations(PAIR_COLS, 0xA11 + tag(quant), 1.0);
    let y0 = vec![f32::from_bits(SENTINEL_BITS); EXPERT_SLOTS * EXPERT_K];
    h.clear_fault()?;
    let got = launch_pair(
        h,
        PairIn {
            quant,
            n_experts: N_EXPERTS,
            ids: &IDS,
            gate: &weights.gate,
            up: &weights.up,
            shared_gate: &weights.shared_gate,
            shared_up: &weights.shared_up,
            x: &x,
            norm: &norm,
            y0: &y0,
        },
    )?;
    if let Some(f) = h.fault()? {
        return Err(format!("{} pair raised {f}", quant.name()));
    }
    let x64 = oracle::rms_norm(&x, &norm.gain, norm.eps);
    let nblocks = PAIR_COLS / BLOCK_VALUES;
    let rb = row_bytes(quant, PAIR_COLS);
    let mut worst = 0.0f64;
    for slot in 0..EXPERT_SLOTS {
        for row in 0..EXPERT_K {
            let (gate_row, up_row) = if slot == SHARED_SLOT {
                (
                    &weights.shared_gate[row * rb..(row + 1) * rb],
                    &weights.shared_up[row * rb..(row + 1) * rb],
                )
            } else {
                let expert = IDS[slot] as usize;
                let at = (expert * EXPERT_K + row) * rb;
                (&weights.gate[at..at + rb], &weights.up[at..at + rb])
            };
            let g = oracle::reference(quant, gate_row, 1, PAIR_COLS, &x64)?;
            let u = oracle::reference(quant, up_row, 1, PAIR_COLS, &x64)?;
            let (want, width) = oracle_experts::pair_width(
                quant,
                nblocks,
                g.y[0],
                g.magnitude[0],
                u.y[0],
                u.magnitude[0],
            )?;
            let value = got[slot * EXPERT_K + row];
            let err = (f64::from(value) - want).abs();
            if !value.is_finite() || err > width {
                let ratio = if width > 0.0 {
                    err / width
                } else {
                    f64::INFINITY
                };
                return Err(format!(
                    "{} slot {slot} row {row}: got {value:e}, want {want:e}, error {err:e} over the band {width:e} (ratio {ratio:.3})",
                    quant.name()
                ));
            }
            if width > 0.0 {
                worst = worst.max(err / width);
            }
        }
    }
    Ok(format!("{} {worst:.3}", quant.name()))
}

pub fn pair(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let q4 = check_pair_quant(h, fix, Quant::Q4K)?;
    let q6 = check_pair_quant(h, fix, Quant::Q6K)?;
    Ok(Verdict::Noted(format!(
        "worst error over its band: {q4}, {q6}"
    )))
}

/// An id past the stack raises [`FaultSite::ExpertId`] and that slot's rows
/// are NaN. The other slots stay bit-equal to a clean launch.
///
/// `row_tail_reads` re-runs only the row-tail child under shader validation, so
/// it cannot see this launch and cannot prove the bad slot loaded no weight.
/// The fault word and the NaN rows are the pin. The output fill is finite, so
/// a row the kernel leaves unwritten is not a NaN.
pub fn pair_id(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let quant = Quant::Q4K;
    let weights = pair_weights(fix, quant)?;
    let norm = ffn_norm(fix)?;
    let x = oracle::activations(PAIR_COLS, 0xB1D, 1.0);
    let y_len = EXPERT_SLOTS * EXPERT_K;
    h.clear_fault()?;
    let clean = launch_pair(
        h,
        PairIn {
            quant,
            n_experts: N_EXPERTS,
            ids: &IDS,
            gate: &weights.gate,
            up: &weights.up,
            shared_gate: &weights.shared_gate,
            shared_up: &weights.shared_up,
            x: &x,
            norm: &norm,
            y0: &vec![f32::from_bits(SENTINEL_BITS); y_len],
        },
    )?;
    if let Some(f) = h.fault()? {
        return Err(format!("the clean pair raised {f}"));
    }
    let bad_ids = [-1, i32::MIN, N_EXPERTS as i32];
    for bad in bad_ids {
        let mut ids = IDS;
        ids[3] = bad;
        h.clear_fault()?;
        let got = launch_pair(
            h,
            PairIn {
                quant,
                n_experts: N_EXPERTS,
                ids: &ids,
                gate: &weights.gate,
                up: &weights.up,
                shared_gate: &weights.shared_gate,
                shared_up: &weights.shared_up,
                x: &x,
                norm: &norm,
                y0: &vec![1.0; y_len],
            },
        )?;
        common::expect_fault(h, &format!("id {bad}"), FaultSite::ExpertId)?;
        for slot in 0..EXPERT_SLOTS {
            let row = &got[slot * EXPERT_K..(slot + 1) * EXPERT_K];
            let clean_row = &clean[slot * EXPERT_K..(slot + 1) * EXPERT_K];
            if slot == 3 {
                if let Some(k) = row.iter().position(|v| !v.is_nan()) {
                    return Err(format!("id {bad} slot 3 row {k} is {:?}, not NaN", row[k]));
                }
            } else if let Some(k) = row
                .iter()
                .zip(clean_row)
                .position(|(g, c)| g.to_bits() != c.to_bits())
            {
                return Err(format!(
                    "id {bad} slot {slot} row {k} changed to {:?} (clean {:?})",
                    row[k], clean_row[k]
                ));
            }
        }
    }
    Ok(Verdict::Noted(
        "ids -1, i32::MIN and the stack length raised expert_id, NaN in that slot only; shader validation cannot see this launch".to_string(),
    ))
}

/// Input the pair's norm cannot take raises [`FaultSite::NormQuant`] in the
/// layer, and only that site, and every slot's rows are NaN, the shared
/// slot's too: a NaN input, an infinite one, and a zero input with a zero
/// epsilon, both quants. The output fill is finite, so a row the kernel
/// leaves unwritten is not a NaN.
pub fn pair_norm(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let norm = ffn_norm(fix)?;
    let y_len = EXPERT_SLOTS * EXPERT_K;
    for quant in [Quant::Q4K, Quant::Q6K] {
        let weights = pair_weights(fix, quant)?;
        for (what, mut x, eps) in [
            (
                "a NaN input",
                oracle::activations(PAIR_COLS, 0xA0 + tag(quant), 1.0),
                norm.eps,
            ),
            (
                "an infinite input",
                oracle::activations(PAIR_COLS, 0xA1 + tag(quant), 1.0),
                norm.eps,
            ),
            (
                "a zero input and a zero epsilon",
                vec![0f32; PAIR_COLS],
                0.0,
            ),
        ] {
            match what {
                "a NaN input" => x[17] = f32::NAN,
                "an infinite input" => x[PAIR_COLS - 1] = f32::INFINITY,
                _ => {}
            }
            let what = format!("{} {what}", quant.name());
            h.clear_fault()?;
            let got = launch_pair(
                h,
                PairIn {
                    quant,
                    n_experts: N_EXPERTS,
                    ids: &IDS,
                    gate: &weights.gate,
                    up: &weights.up,
                    shared_gate: &weights.shared_gate,
                    shared_up: &weights.shared_up,
                    x: &x,
                    norm: &PairNorm {
                        gain: norm.gain.clone(),
                        eps,
                    },
                    y0: &vec![1.0; y_len],
                },
            )?;
            common::expect_fault(h, &what, FaultSite::NormQuant)?;
            if let Some(k) = got.iter().position(|v| !v.is_nan()) {
                return Err(format!(
                    "{what}: slot {} row {} is {:?}, not NaN",
                    k / EXPERT_K,
                    k % EXPERT_K,
                    got[k]
                ));
            }
        }
    }
    h.clear_fault()?;
    Ok(Verdict::Noted(
        "a NaN input, an infinite input and a zero input with a zero epsilon raised norm_quant only, NaN in all nine slots' rows, both quants".to_string(),
    ))
}

/// Slot 0 of a pair over a one-expert stack, `rows` being `[gate, up, shared
/// gate, shared up]`.
fn solo_slot(
    h: &Harness<'_>,
    quant: Quant,
    rows: [&[u8]; 4],
    x: &[f32],
    norm: &PairNorm,
) -> Result<Vec<f32>, String> {
    let [gate, up, shared_gate, shared_up] = rows;
    let ids = [0; ROUTED_IDS];
    let y_len = EXPERT_SLOTS * EXPERT_K;
    let got = launch_pair(
        h,
        PairIn {
            quant,
            n_experts: 1,
            ids: &ids,
            gate,
            up,
            shared_gate,
            shared_up,
            x,
            norm,
            y0: &vec![f32::from_bits(SENTINEL_BITS); y_len],
        },
    )?;
    if let Some(f) = h.fault()? {
        return Err(format!("a solo pair raised {f}"));
    }
    Ok(got[..EXPERT_K].to_vec())
}

fn check_pair_shared(h: &Harness<'_>, fix: &Fixture, quant: Quant) -> Result<(), String> {
    let weights = pair_weights(fix, quant)?;
    let norm = ffn_norm(fix)?;
    let x = oracle::activations(PAIR_COLS, 0x5E1 + tag(quant), 1.0);
    let y_len = EXPERT_SLOTS * EXPERT_K;
    h.clear_fault()?;
    let full = launch_pair(
        h,
        PairIn {
            quant,
            n_experts: N_EXPERTS,
            ids: &IDS,
            gate: &weights.gate,
            up: &weights.up,
            shared_gate: &weights.shared_gate,
            shared_up: &weights.shared_up,
            x: &x,
            norm: &norm,
            y0: &vec![f32::from_bits(SENTINEL_BITS); y_len],
        },
    )?;
    if let Some(f) = h.fault()? {
        return Err(format!("{} pair raised {f}", quant.name()));
    }
    let one = weights.gate.len() / N_EXPERTS;
    for slot in [0usize, 1, SHARED_SLOT] {
        let (gate, up, shared_gate, shared_up) = if slot == SHARED_SLOT {
            (
                weights.shared_gate.as_slice(),
                weights.shared_up.as_slice(),
                &weights.gate[..one],
                &weights.up[..one],
            )
        } else {
            let expert = IDS[slot] as usize;
            (
                &weights.gate[expert * one..(expert + 1) * one],
                &weights.up[expert * one..(expert + 1) * one],
                weights.shared_gate.as_slice(),
                weights.shared_up.as_slice(),
            )
        };
        h.clear_fault()?;
        let solo = solo_slot(h, quant, [gate, up, shared_gate, shared_up], &x, &norm)?;
        let got = &full[slot * EXPERT_K..(slot + 1) * EXPERT_K];
        expect_bits(got, &solo, &format!("{} slot {slot}", quant.name()))?;
    }
    Ok(())
}

pub fn pair_shared(h: &Harness<'_>, fix: &Fixture) -> Clause {
    check_pair_shared(h, fix, Quant::Q4K)?;
    check_pair_shared(h, fix, Quant::Q6K)?;
    Ok(Verdict::Noted(
        "slots 0, 1 and 8 bit-equal to a solo launch of the same rows, both quants".to_string(),
    ))
}

fn hidden_states(quant: Quant) -> Vec<f32> {
    let mut out = Vec::with_capacity(EXPERT_SLOTS * EXPERT_K);
    for slot in 0..EXPERT_SLOTS {
        out.extend(oracle::activations(
            EXPERT_K,
            0xB00 + slot as u64 + tag(quant),
            1.0,
        ));
    }
    out
}

fn check_fold_quant(h: &Harness<'_>, fix: &Fixture, quant: Quant) -> Result<String, String> {
    let weights = fold_weights(fix, quant)?;
    let hidden = hidden_states(quant);
    let resid = oracle::activations(FOLD_ROWS, 0xF01D + tag(quant), 2.0);
    h.clear_fault()?;
    let got = launch_fold(
        h,
        FoldIn {
            quant,
            n_experts: N_EXPERTS,
            ids: &IDS,
            down: &weights.down,
            shared: &weights.shared,
            h: &hidden,
            w: &ROUTE,
            y0: &resid,
        },
    )?;
    if let Some(f) = h.fault()? {
        return Err(format!("{} fold raised {f}", quant.name()));
    }
    let rb = row_bytes(quant, EXPERT_K);
    let mut worst = 0.0f64;
    for row in 0..FOLD_ROWS {
        let mut want = f64::from(resid[row]);
        let mut magnitude = 0.0;
        for slot in 0..EXPERT_SLOTS {
            let bytes = if slot == SHARED_SLOT {
                &weights.shared[row * rb..(row + 1) * rb]
            } else {
                let expert = IDS[slot] as usize;
                let at = (expert * FOLD_ROWS + row) * rb;
                &weights.down[at..at + rb]
            };
            let x: Vec<f64> = hidden[slot * EXPERT_K..(slot + 1) * EXPERT_K]
                .iter()
                .map(|&v| f64::from(ROUTE[slot]) * f64::from(v))
                .collect();
            let reference = oracle::reference(quant, bytes, 1, EXPERT_K, &x)?;
            want += reference.y[0];
            magnitude += reference.magnitude[0];
        }
        let width = oracle_experts::fold_width(quant, magnitude, want);
        let err = (f64::from(got[row]) - want).abs();
        if !got[row].is_finite() || err > width {
            let ratio = if width > 0.0 {
                err / width
            } else {
                f64::INFINITY
            };
            return Err(format!(
                "{} row {row}: got {:e}, want {want:e}, error {err:e} over the band {width:e} (ratio {ratio:.3})",
                quant.name(),
                got[row]
            ));
        }
        if width > 0.0 {
            worst = worst.max(err / width);
        }
    }
    Ok(format!("{} {worst:.3}", quant.name()))
}

fn fold_bad_id(h: &Harness<'_>, weights: &FoldWeights, quant: Quant) -> Result<(), String> {
    let hidden = hidden_states(quant);
    for bad in [-1, N_EXPERTS as i32] {
        let mut ids = IDS;
        ids[0] = bad;
        h.clear_fault()?;
        let got = launch_fold(
            h,
            FoldIn {
                quant,
                n_experts: N_EXPERTS,
                ids: &ids,
                down: &weights.down,
                shared: &weights.shared,
                h: &hidden,
                w: &ROUTE,
                y0: &vec![1.0; FOLD_ROWS],
            },
        )?;
        common::expect_fault(
            h,
            &format!("{} id {bad}", quant.name()),
            FaultSite::ExpertId,
        )?;
        if let Some(row) = got.iter().position(|v| !v.is_nan()) {
            return Err(format!(
                "{} id {bad} row {row} is {:?}, not NaN",
                quant.name(),
                got[row]
            ));
        }
    }
    Ok(())
}

pub fn fold(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let q4 = check_fold_quant(h, fix, Quant::Q4K)?;
    let q6 = check_fold_quant(h, fix, Quant::Q6K)?;
    fold_bad_id(h, &fold_weights(fix, Quant::Q4K)?, Quant::Q4K)?;
    Ok(Verdict::Noted(format!(
        "worst error over its band: {q4}, {q6}; a bad id writes NaN and raises expert_id"
    )))
}

/// The buffers of the two refusal tables: one weight buffer big enough for a
/// well-formed pair of one expert, and the windows the tables share.
struct Refusals {
    marker: MetalBuffer<f32>,
    bytes: MetalBuffer<u8>,
    values: MetalBuffer<f32>,
    route: MetalBuffer<f32>,
    ids: MetalBuffer<i32>,
}

/// One pair case: a slot count, a width, an id list, a window or an output
/// stride, one of them bad.
#[derive(Clone, Copy)]
struct PairArgs {
    slots: usize,
    k: usize,
    n_experts: usize,
    id_len: usize,
    gate_len: usize,
    stride: usize,
    x: usize,
    y_len: usize,
}

impl PairArgs {
    /// The table's arguments: every window short of its good length, so a
    /// case's own check refuses before any window check can.
    const fn whole(gate_len: usize) -> Self {
        Self {
            slots: EXPERT_SLOTS,
            k: EXPERT_K,
            n_experts: 1,
            id_len: ROUTED_IDS,
            gate_len,
            stride: EXPERT_K,
            x: 16,
            y_len: 64,
        }
    }

    /// The well-formed launch of the same buffers.
    const fn good(gate_len: usize) -> Self {
        Self {
            x: PAIR_COLS,
            y_len: EXPERT_SLOTS * EXPERT_K,
            ..Self::whole(gate_len)
        }
    }
}

fn refuse_pair(
    h: &Harness<'_>,
    b: &mut Refusals,
    a: &PairArgs,
    encoder: &mut bloomery_metal::encode::ComputeEncoder<'_>,
) -> Result<(), bloomery_device::DeviceError> {
    h.kernels.pair(
        encoder,
        PairParams {
            quant: Quant::Q4K,
            slots: a.slots,
            k: a.k,
            cols: PAIR_COLS,
            n_experts: a.n_experts,
            ids: b.ids.window(Elems(0), Elems(a.id_len))?,
            gate: b.bytes.window(Elems(0), Elems(a.gate_len))?,
            up: b.bytes.window(Elems(0), Elems(a.gate_len))?,
            shared_gate: b.bytes.window(Elems(0), Elems(a.gate_len))?,
            shared_up: b.bytes.window(Elems(0), Elems(a.gate_len))?,
            x: b.values.window(Elems(0), Elems(a.x))?,
            gain: b.values.window(Elems(0), Elems(a.x))?,
            eps: 1e-6,
            y: b.marker.window_mut(Elems(0), Elems(a.y_len))?,
            out_stride: a.stride,
            fault: h.device.fault_word(),
            layer: LAYER,
        },
    )
}

/// One fold case: a slot count, a width, an id list, a window or the rows,
/// one of them bad.
#[derive(Clone, Copy)]
struct FoldArgs {
    slots: usize,
    k: usize,
    n_experts: usize,
    id_len: usize,
    rows: usize,
    down_len: usize,
    h_len: usize,
    y_len: usize,
}

impl FoldArgs {
    /// The table's arguments: every window short of its good length, so a
    /// case's own check refuses before any window check can.
    const fn whole() -> Self {
        Self {
            slots: EXPERT_SLOTS,
            k: EXPERT_K,
            n_experts: 1,
            id_len: ROUTED_IDS,
            rows: 8,
            down_len: 16,
            h_len: 16,
            y_len: 64,
        }
    }
}

fn refuse_fold(
    h: &Harness<'_>,
    b: &mut Refusals,
    a: &FoldArgs,
    encoder: &mut bloomery_metal::encode::ComputeEncoder<'_>,
) -> Result<(), bloomery_device::DeviceError> {
    h.kernels.fold(
        encoder,
        FoldParams {
            quant: Quant::Q4K,
            shared: SharedPlan::Same(b.bytes.window(Elems(0), Elems(a.down_len))?),
            rows: a.rows,
            slots: a.slots,
            k: a.k,
            n_experts: a.n_experts,
            ids: b.ids.window(Elems(0), Elems(a.id_len))?,
            down: b.bytes.window(Elems(0), Elems(a.down_len))?,
            h: b.values.window(Elems(0), Elems(a.h_len))?,
            w: b.route.window(Elems(0), Elems(EXPERT_SLOTS))?,
            y: &b.marker.window_mut(Elems(0), Elems(a.y_len))?,
            fault: h.device.fault_word(),
            layer: LAYER,
        },
    )
}

pub fn experts_bounds(h: &Harness<'_>, _fix: &Fixture) -> Clause {
    let (device, stream) = (h.device, h.stream);
    let gate_len = EXPERT_K * row_bytes(Quant::Q4K, PAIR_COLS);
    let mut bufs = Refusals {
        marker: device
            .upload(stream, &vec![1.0f32; EXPERT_SLOTS * EXPERT_K])
            .map_err(why)?,
        bytes: device.upload(stream, &vec![0u8; gate_len]).map_err(why)?,
        values: device
            .upload(stream, &vec![0f32; EXPERT_SLOTS * EXPERT_K])
            .map_err(why)?,
        route: device
            .upload(stream, &[1.0f32; EXPERT_SLOTS])
            .map_err(why)?,
        ids: device.upload(stream, &[0i32; ROUTED_IDS]).map_err(why)?,
    };
    let pair_whole = PairArgs::whole(gate_len);
    let pair_good = PairArgs::good(gate_len);
    let bad_pair = |what: &'static str,
                    needle: &'static str,
                    also: &'static str,
                    mask: &'static str,
                    edit: Box<dyn FnOnce(&mut PairArgs)>| {
        BadArg::new(
            what,
            needle,
            Box::new(move |b: &mut Refusals, encoder| {
                let mut a = pair_whole;
                edit(&mut a);
                refuse_pair(h, b, &a, encoder)
            }),
        )
        .naming(also)
        .masking(mask)
    };
    let pair_cases = vec![
        bad_pair(
            "a pair of 3 slots",
            "9",
            "",
            "window",
            Box::new(|a| a.slots = 3),
        ),
        bad_pair(
            "a pair of width 256",
            "512",
            "",
            "window",
            Box::new(|a| a.k = 256),
        ),
        bad_pair(
            "a short gate window",
            "window",
            "gate",
            "",
            Box::new(|a| a.gate_len = 16),
        ),
        bad_pair(
            "an empty pair id list",
            "empty",
            "id list",
            "window",
            Box::new(|a| a.id_len = 0),
        ),
        bad_pair(
            "overlapping pair outputs",
            "overlap",
            "",
            "window",
            Box::new(|a| a.stride = 1),
        ),
    ];
    let fold_whole = FoldArgs::whole();
    let bad_fold = |what: &'static str,
                    needle: &'static str,
                    also: &'static str,
                    mask: &'static str,
                    edit: Box<dyn FnOnce(&mut FoldArgs)>| {
        BadArg::new(
            what,
            needle,
            Box::new(move |b: &mut Refusals, encoder| {
                let mut a = fold_whole;
                edit(&mut a);
                refuse_fold(h, b, &a, encoder)
            }),
        )
        .naming(also)
        .masking(mask)
    };
    let fold_cases = vec![
        bad_fold(
            "a fold of 3 slots",
            "9",
            "",
            "window",
            Box::new(|a| a.slots = 3),
        ),
        bad_fold(
            "a fold of width 256",
            "512",
            "",
            "window",
            Box::new(|a| a.k = 256),
        ),
        bad_fold(
            "a short down window",
            "window",
            "down",
            "",
            Box::new(|a| a.down_len = 16),
        ),
        bad_fold(
            "an empty fold id list",
            "empty",
            "id list",
            "window",
            Box::new(|a| a.id_len = 0),
        ),
        {
            // The hidden states window of the overlap case is the output
            // window, so the launch is built by hand.
            BadArg::new(
                "a fold whose output overlaps the hidden states",
                "overlap",
                Box::new(
                    |b: &mut Refusals, encoder: &mut bloomery_metal::encode::ComputeEncoder<'_>| {
                        let y = b.marker.window_mut(Elems(0), Elems(FOLD_ROWS))?;
                        let same = y.as_read();
                        h.kernels.fold(
                            encoder,
                            FoldParams {
                                quant: Quant::Q4K,
                                shared: SharedPlan::Same(b.bytes.window(Elems(0), Elems(16))?),
                                rows: FOLD_ROWS,
                                slots: EXPERT_SLOTS,
                                k: EXPERT_K,
                                n_experts: 1,
                                ids: b.ids.window(Elems(0), Elems(ROUTED_IDS))?,
                                down: b.bytes.window(Elems(0), Elems(16))?,
                                h: same,
                                w: b.route.window(Elems(0), Elems(EXPERT_SLOTS))?,
                                y: &y,
                                fault: h.device.fault_word(),
                                layer: LAYER,
                            },
                        )
                    },
                ),
            )
            .masking("window")
        },
    ];
    let unchanged = |h: &Harness<'_>, b: &Refusals| common::still(h, &b.marker, |v| *v == 1.0);
    let pairs = common::refuse_launches(h, "pair", &mut bufs, pair_cases, unchanged)?;
    let folds = common::refuse_launches(h, "fold", &mut bufs, fold_cases, unchanged)?;
    let good_pair: Edit<'_, Refusals> = Box::new(|b, encoder| {
        let a = pair_good;
        refuse_pair(h, b, &a, encoder)
    });
    common::accepted_writes(h, &mut bufs, good_pair, |h, b| {
        common::moved(h, &b.marker, |v| *v != 1.0)
    })?;
    let good_fold: Edit<'_, Refusals> = Box::new(|b, encoder| {
        let mut a = fold_whole;
        a.down_len = a.rows * row_bytes(Quant::Q4K, EXPERT_K);
        a.h_len = EXPERT_SLOTS * EXPERT_K;
        a.y_len = a.rows;
        refuse_fold(h, b, &a, encoder)
    });
    common::accepted_writes(h, &mut bufs, good_fold, |h, b| {
        common::moved(h, &b.marker, |v| *v != 1.0)
    })?;
    Ok(Verdict::Noted(format!(
        "{} refusals, each by name, nothing written",
        pairs + folds
    )))
}

pub fn fold_type(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let (device, stream) = (h.device, h.stream);
    let down_len = 4 * row_bytes(Quant::Q4K, EXPERT_K);
    let down = fix.rows("blk.0.ffn_down_exps.weight", GgmlType::Q4_K, EXPERT_K, 4)?;
    struct Bufs {
        marker: MetalBuffer<f32>,
        down: MetalBuffer<u8>,
        hidden: MetalBuffer<f32>,
        route: MetalBuffer<f32>,
        ids: MetalBuffer<i32>,
    }
    let mut bufs = Bufs {
        marker: device.upload(stream, &[1.0f32; 8]).map_err(why)?,
        down: device.upload(stream, down).map_err(why)?,
        hidden: device
            .upload(stream, &vec![1.0f32; EXPERT_SLOTS * EXPERT_K])
            .map_err(why)?,
        route: device
            .upload(stream, &[1.0f32; EXPERT_SLOTS])
            .map_err(why)?,
        ids: device.upload(stream, &[0i32; ROUTED_IDS]).map_err(why)?,
    };
    let cases = [
        ("Apart", EXPERT_SLOTS, EXPERT_K, 1usize, ROUTED_IDS),
        ("Apart with a bad slot count", 1, EXPERT_K, 1, ROUTED_IDS),
        ("Apart with a bad width", EXPERT_SLOTS, 256, 1, ROUTED_IDS),
        (
            "Apart with an empty stack",
            EXPERT_SLOTS,
            EXPERT_K,
            0,
            ROUTED_IDS,
        ),
        ("Apart with an empty id list", EXPERT_SLOTS, EXPERT_K, 1, 0),
    ]
    .into_iter()
    .map(|(what, slots, k, n_experts, id_len)| {
        BadArg::new(
            what,
            "SharedPlan::Apart",
            Box::new(move |b: &mut Bufs, encoder| {
                let y = b.marker.window_mut(Elems(0), Elems(8))?;
                h.kernels.fold(
                    encoder,
                    FoldParams {
                        quant: Quant::Q4K,
                        shared: SharedPlan::Apart,
                        rows: 4,
                        slots,
                        k,
                        n_experts,
                        ids: b.ids.window(Elems(0), Elems(id_len))?,
                        down: b.down.window(Elems(0), Elems(16))?,
                        h: b.hidden.window(Elems(0), Elems(16))?,
                        w: b.route.window(Elems(0), Elems(EXPERT_SLOTS))?,
                        y: &y,
                        fault: h.device.fault_word(),
                        layer: LAYER,
                    },
                )
            }),
        )
        .masking("window")
    })
    .collect::<Vec<_>>();
    let unchanged = |h: &Harness<'_>, b: &Bufs| common::still(h, &b.marker, |v| *v == 1.0);
    common::refuse_launches(h, "fold", &mut bufs, cases, unchanged)?;
    let good: Edit<'_, Bufs> = Box::new(|b, encoder| {
        let y = b.marker.window_mut(Elems(0), Elems(4))?;
        h.kernels.fold(
            encoder,
            FoldParams {
                quant: Quant::Q4K,
                shared: SharedPlan::Same(b.down.window(Elems(0), Elems(down_len))?),
                rows: 4,
                slots: EXPERT_SLOTS,
                k: EXPERT_K,
                n_experts: 1,
                ids: b.ids.window(Elems(0), Elems(ROUTED_IDS))?,
                down: b.down.window(Elems(0), Elems(down_len))?,
                h: b.hidden.window(Elems(0), Elems(EXPERT_SLOTS * EXPERT_K))?,
                w: b.route.window(Elems(0), Elems(EXPERT_SLOTS))?,
                y: &y,
                fault: h.device.fault_word(),
                layer: LAYER,
            },
        )
    });
    common::accepted_writes(h, &mut bufs, good, |h, b| {
        common::moved(h, &b.marker, |v| *v != 1.0)
    })?;
    Ok(Verdict::Noted(
        "Apart is refused by name before the slot count, the width, an empty stack and an empty id list".to_string(),
    ))
}

/// The expert pair and fold clauses, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, super::ClauseRun<'a>)> {
    vec![
        ("pair", Box::new(|| ctx.with(pair))),
        ("pair_id", Box::new(|| ctx.with(pair_id))),
        ("pair_norm", Box::new(|| ctx.with(pair_norm))),
        ("pair_shared", Box::new(|| ctx.with(pair_shared))),
        ("fold", Box::new(|| ctx.with(fold))),
        ("fold_type", Box::new(|| ctx.with(fold_type))),
        ("experts_bounds", Box::new(|| ctx.with(experts_bounds))),
    ]
}
