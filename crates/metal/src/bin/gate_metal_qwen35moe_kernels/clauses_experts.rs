//! The routed pair and the folded down.

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, Elems};
use bloomery_metal::MetalBuffer;
use bloomery_metal::kernels::{
    BLOCK_VALUES, EXPERT_K, EXPERT_SLOTS, FoldParams, PairParams, Quant, ROUTED_IDS, SHARED_SLOT,
    SharedPlan,
};
use gguf::quant::GgmlType;

use super::fixture::Fixture;
use super::oracle;
use super::oracle_experts;
use super::{
    Harness, OUT_GUARD, SENTINEL_BITS, VALUE_PAD, WEIGHT_PAD, WEIGHT_TAIL, check_guards, padded,
};

/// The layer these clauses launch in.
const LAYER: u32 = 5;
const N_EXPERTS: usize = 4;
const PAIR_COLS: usize = 2048;
const FOLD_ROWS: usize = 2048;
const IDS: [i32; ROUTED_IDS] = [0, 1, 2, 3, 1, 2, 3, 0];
const ROUTE: [f32; EXPERT_SLOTS] = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8];

fn ggml(quant: Quant) -> GgmlType {
    match quant {
        Quant::Q4K => GgmlType::Q4_K,
        Quant::Q6K => GgmlType::Q6_K,
    }
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
    let ty = ggml(quant);
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
    let ty = ggml(quant);
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

fn expert_id(fault: Option<bloomery_device::Fault>, what: &str) -> Result<(), String> {
    match fault {
        Some(f)
            if f.layer == LAYER
                && f.code == FaultSite::ExpertId as u32
                && f.sites == 1 << (FaultSite::ExpertId as u32) =>
        {
            Ok(())
        }
        Some(f) => Err(format!(
            "{what} raised {f}, not {} in layer {LAYER}",
            FaultSite::ExpertId.name()
        )),
        None => Err(format!("{what} raised nothing")),
    }
}

fn bits_eq(got: &[f32], want: &[f32], what: &str) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!(
            "{what}: {} values against {}",
            got.len(),
            want.len()
        ));
    }
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g.to_bits() != w.to_bits() {
            return Err(format!("{what} value {i}: got {g:?}, the solo read {w:?}"));
        }
    }
    Ok(())
}

fn check_pair_quant(h: &Harness<'_>, fix: &Fixture, quant: Quant) -> Result<String, String> {
    let weights = pair_weights(fix, quant)?;
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
            y0: &y0,
        },
    )?;
    if let Some(f) = h.fault()? {
        return Err(format!("{} pair raised {f}", quant.name()));
    }
    let x64: Vec<f64> = x.iter().copied().map(f64::from).collect();
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
                y0: &vec![1.0; y_len],
            },
        )?;
        expert_id(h.fault()?, &format!("id {bad}"))?;
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

fn solo_slot(
    h: &Harness<'_>,
    quant: Quant,
    gate: &[u8],
    up: &[u8],
    shared_gate: &[u8],
    shared_up: &[u8],
    x: &[f32],
) -> Result<Vec<f32>, String> {
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
        let solo = solo_slot(h, quant, gate, up, shared_gate, shared_up, &x)?;
        let got = &full[slot * EXPERT_K..(slot + 1) * EXPERT_K];
        bits_eq(got, &solo, &format!("{} slot {slot}", quant.name()))?;
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
        expert_id(h.fault()?, &format!("{} id {bad}", quant.name()))?;
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

fn expect_named(
    result: Result<(), String>,
    what: &str,
    needle: &str,
    also: &str,
) -> Result<(), String> {
    match result {
        Ok(()) => Err(format!("{what} was accepted")),
        Err(text) => {
            if !text.contains(needle) {
                return Err(format!(
                    "{what} was refused without naming {needle}: {text}"
                ));
            }
            if !also.is_empty() && !text.contains(also) {
                return Err(format!("{what} was refused without naming {also}: {text}"));
            }
            if needle != "window" && text.contains("window") {
                return Err(format!("{what} named a window: {text}"));
            }
            Ok(())
        }
    }
}

fn pair_bounds(
    h: &Harness<'_>,
    marker: &mut MetalBuffer<f32>,
    bytes: &MetalBuffer<u8>,
    values: &MetalBuffer<f32>,
    ids: &MetalBuffer<i32>,
) -> Result<(), String> {
    let device = h.device;
    let mut run = |slots, k, n_experts, id_len, gate_len, stride| {
        h.encode(|encoder| {
            h.kernels.pair(
                encoder,
                PairParams {
                    quant: Quant::Q4K,
                    slots,
                    k,
                    cols: PAIR_COLS,
                    n_experts,
                    ids: ids.window(Elems(0), Elems(id_len))?,
                    gate: bytes.window(Elems(0), Elems(gate_len))?,
                    up: bytes.window(Elems(0), Elems(gate_len))?,
                    shared_gate: bytes.window(Elems(0), Elems(gate_len))?,
                    shared_up: bytes.window(Elems(0), Elems(gate_len))?,
                    x: values.window(Elems(0), Elems(16))?,
                    y: marker.window_mut(Elems(0), Elems(64))?,
                    out_stride: stride,
                    fault: device.fault_word(),
                    layer: LAYER,
                },
            )
        })
    };
    expect_named(
        run(3, EXPERT_K, 1, ROUTED_IDS, 16, EXPERT_K),
        "a pair of 3 slots",
        "9",
        "",
    )?;
    expect_named(
        run(EXPERT_SLOTS, 256, 1, ROUTED_IDS, 16, EXPERT_K),
        "a pair of width 256",
        "512",
        "",
    )?;
    expect_named(
        run(EXPERT_SLOTS, EXPERT_K, 1, ROUTED_IDS, 16, EXPERT_K),
        "a short gate window",
        "window",
        "gate",
    )?;
    expect_named(
        run(EXPERT_SLOTS, EXPERT_K, 1, 0, 16, EXPERT_K),
        "an empty pair id list",
        "empty",
        "id list",
    )?;
    expect_named(
        run(EXPERT_SLOTS, EXPERT_K, 1, ROUTED_IDS, 16, 1),
        "overlapping pair outputs",
        "overlap",
        "",
    )?;
    Ok(())
}

fn fold_bounds(
    h: &Harness<'_>,
    marker: &mut MetalBuffer<f32>,
    bytes: &MetalBuffer<u8>,
    hidden: &MetalBuffer<f32>,
    route: &MetalBuffer<f32>,
    ids: &MetalBuffer<i32>,
) -> Result<(), String> {
    let device = h.device;
    let mut run = |slots, k, n_experts, id_len, rows, down_len| {
        h.encode(|encoder| {
            let y = marker.window_mut(Elems(0), Elems(64))?;
            h.kernels.fold(
                encoder,
                FoldParams {
                    quant: Quant::Q4K,
                    shared: SharedPlan::Same(bytes.window(Elems(0), Elems(down_len))?),
                    rows,
                    slots,
                    k,
                    n_experts,
                    ids: ids.window(Elems(0), Elems(id_len))?,
                    down: bytes.window(Elems(0), Elems(down_len))?,
                    h: hidden.window(Elems(0), Elems(16))?,
                    w: route.window(Elems(0), Elems(EXPERT_SLOTS))?,
                    y: &y,
                    fault: device.fault_word(),
                    layer: LAYER,
                },
            )
        })
    };
    expect_named(
        run(3, EXPERT_K, 1, ROUTED_IDS, 8, 16),
        "a fold of 3 slots",
        "9",
        "",
    )?;
    expect_named(
        run(EXPERT_SLOTS, 256, 1, ROUTED_IDS, 8, 16),
        "a fold of width 256",
        "512",
        "",
    )?;
    expect_named(
        run(EXPERT_SLOTS, EXPERT_K, 1, ROUTED_IDS, 8, 16),
        "a short down window",
        "window",
        "down",
    )?;
    expect_named(
        run(EXPERT_SLOTS, EXPERT_K, 1, 0, 8, 16),
        "an empty fold id list",
        "empty",
        "id list",
    )?;
    let overlap = h.encode(|encoder| {
        let y = marker.window_mut(Elems(0), Elems(FOLD_ROWS))?;
        let same = y.as_read();
        h.kernels.fold(
            encoder,
            FoldParams {
                quant: Quant::Q4K,
                shared: SharedPlan::Same(bytes.window(Elems(0), Elems(16))?),
                rows: FOLD_ROWS,
                slots: EXPERT_SLOTS,
                k: EXPERT_K,
                n_experts: 1,
                ids: ids.window(Elems(0), Elems(ROUTED_IDS))?,
                down: bytes.window(Elems(0), Elems(16))?,
                h: same,
                w: route.window(Elems(0), Elems(EXPERT_SLOTS))?,
                y: &y,
                fault: device.fault_word(),
                layer: LAYER,
            },
        )
    });
    expect_named(
        overlap,
        "a fold whose output overlaps the hidden states",
        "overlap",
        "",
    )?;
    Ok(())
}

pub fn experts_bounds(h: &Harness<'_>, _fix: &Fixture) -> Clause {
    let (device, stream) = (h.device, h.stream);
    let mut marker = device.upload(stream, &[1.0f32; FOLD_ROWS]).map_err(why)?;
    let bytes = device.upload(stream, &[0u8; 64]).map_err(why)?;
    let hidden = device
        .upload(stream, &[0f32; EXPERT_SLOTS * EXPERT_K])
        .map_err(why)?;
    let route = device
        .upload(stream, &[1.0f32; EXPERT_SLOTS])
        .map_err(why)?;
    let ids = device.upload(stream, &[0i32; ROUTED_IDS]).map_err(why)?;
    pair_bounds(h, &mut marker, &bytes, &hidden, &ids)?;
    fold_bounds(h, &mut marker, &bytes, &hidden, &route, &ids)?;
    let mut back = vec![0.0; FOLD_ROWS];
    device
        .read(stream, marker.whole().map_err(why)?, &mut back)
        .map_err(why)?;
    if back.iter().any(|&v| v != 1.0) {
        return Err("a refused launch wrote".to_string());
    }
    Ok(Verdict::Noted(
        "10 refusals, each by name, nothing written".to_string(),
    ))
}

pub fn fold_type(h: &Harness<'_>, _fix: &Fixture) -> Clause {
    let (device, stream) = (h.device, h.stream);
    let mut marker = device.upload(stream, &[1.0f32; 8]).map_err(why)?;
    let bytes = device.upload(stream, &[0u8; 64]).map_err(why)?;
    let hidden = device.upload(stream, &[0f32; 16]).map_err(why)?;
    let route = device
        .upload(stream, &[1.0f32; EXPERT_SLOTS])
        .map_err(why)?;
    let ids = device.upload(stream, &[0i32; ROUTED_IDS]).map_err(why)?;
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
    ];
    for (what, slots, k, n_experts, id_len) in cases {
        let refused = h.encode(|encoder| {
            let y = marker.window_mut(Elems(0), Elems(8))?;
            h.kernels.fold(
                encoder,
                FoldParams {
                    quant: Quant::Q4K,
                    shared: SharedPlan::Apart,
                    rows: 4,
                    slots,
                    k,
                    n_experts,
                    ids: ids.window(Elems(0), Elems(id_len))?,
                    down: bytes.window(Elems(0), Elems(16))?,
                    h: hidden.window(Elems(0), Elems(16))?,
                    w: route.window(Elems(0), Elems(EXPERT_SLOTS))?,
                    y: &y,
                    fault: device.fault_word(),
                    layer: LAYER,
                },
            )
        });
        expect_named(refused, what, "SharedPlan::Apart", "")?;
    }
    let mut back = vec![0.0; 8];
    device
        .read(stream, marker.whole().map_err(why)?, &mut back)
        .map_err(why)?;
    if back.iter().any(|&v| v != 1.0) {
        return Err("an Apart launch wrote".to_string());
    }
    Ok(Verdict::Noted(
        "Apart is refused by name before the slot count, the width, an empty stack and an empty id list".to_string(),
    ))
}

/// The expert pair and fold clauses, in gate order.
pub fn table<'a>(ctx: &'a super::ClauseCtx<'a>) -> Vec<(&'static str, super::ClauseRun<'a>)> {
    vec![
        ("pair", Box::new(|| ctx.with(pair))),
        ("pair_id", Box::new(|| ctx.with(pair_id))),
        ("pair_shared", Box::new(|| ctx.with(pair_shared))),
        ("fold", Box::new(|| ctx.with(fold))),
        ("fold_type", Box::new(|| ctx.with(fold_type))),
        ("experts_bounds", Box::new(|| ctx.with(experts_bounds))),
    ]
}
