//! The routed pair and the folded down. The pair norms its input as it loads
//! it, with a synthetic gain and the fixture's epsilon. The slots are the
//! row's instance's: its routed picks, then the shared expert's when it has
//! one.

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, Elems};
use bloomery_metal::MetalBuffer;
use bloomery_metal::geom::{ExpertsGeom, Geometry};
use bloomery_metal::instance::REGISTERED;
use bloomery_metal::kernels::{
    APART_DOWN, APART_GATE_UP, FoldParams, LoadNorm, PairParams, Quant, ROWS_PER_SIMD_GROUP,
    SharedPlan,
};
use gguf::quant::GgmlType;

use super::common::{self, BadArg, Edit, LAYER};
use super::fixture::Fixture;
use super::oracle;
use super::oracle_experts;
use super::synthetic;
use super::{
    ACT_PAD, Harness, OUT_GUARD, SENTINEL_BITS, VALUE_PAD, WEIGHT_TAIL, check_guards, padded,
    weight_pad,
};
const N_EXPERTS: usize = 4;
const PAIR_COLS: usize = 2048;
pub const FOLD_ROWS: usize = 2048;
/// The ids of a launch: one per pick, over the stack's four experts.
const IDS: [i32; 8] = [0, 1, 2, 3, 1, 2, 3, 0];
/// The route weight of each slot, the shared expert's last.
const ROUTE_ALL: [f32; 9] = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8];

/// What `IDS` and `ROUTE_ALL` are written for, held for every registered
/// row: eight picks, and no more slots than route weights.
const _: () = {
    let mut i = 0;
    while i < REGISTERED.len() {
        let e = Geometry::of(&REGISTERED[i]).experts;
        assert!(e.routed_ids == IDS.len());
        assert!(e.slots <= ROUTE_ALL.len());
        i += 1;
    }
};

/// The route weight of each of the instance's slots.
fn route_weights(e: &ExpertsGeom) -> Vec<f32> {
    ROUTE_ALL[..e.slots].to_vec()
}

/// The fixture's Q6_K down stack and its shared expert's: the first layer
/// the fixture keeps at Q6_K.
fn q6_down(fix: &Fixture) -> Result<[String; 2], String> {
    let l = fix.first_layer_of("ffn_down_exps.weight", GgmlType::Q6_K)?;
    Ok([
        format!("blk.{l}.ffn_down_exps.weight"),
        format!("blk.{l}.ffn_down_shexp.weight"),
    ])
}

/// The shared expert's bytes when the instance's experts have its slot, and
/// none otherwise.
fn if_shared(
    e: &ExpertsGeom,
    read: impl FnOnce() -> Result<Vec<u8>, String>,
) -> Result<Vec<u8>, String> {
    if e.has_shared { read() } else { Ok(Vec::new()) }
}

/// The fold's shared plan: the window when the instance's experts have a
/// shared slot, [`SharedPlan::None`] otherwise.
fn shared_plan<'w>(
    e: &ExpertsGeom,
    window: bloomery_device::Window<'w, u8, bloomery_metal::MetalDevice>,
) -> SharedPlan<'w> {
    if e.has_shared {
        SharedPlan::Same(window)
    } else {
        SharedPlan::None
    }
}

/// A count a refusal names, as the text it is named by.
fn named(count: usize) -> &'static str {
    count.to_string().leak()
}

/// The norm a pair launch applies to its input.
pub struct PairNorm {
    pub gain: Vec<f32>,
    pub eps: f32,
}

/// A positive synthetic gain, not the fixture's FFN gain: that one is all
/// ones, under which a dropped gain is invisible. The epsilon is the
/// fixture's.
pub fn ffn_norm(fix: &Fixture) -> Result<PairNorm, String> {
    Ok(synthetic_norm(fix.rms_eps()?))
}

/// The synthetic gain [`ffn_norm`] reads, at epsilon `eps`.
pub fn synthetic_norm(eps: f32) -> PairNorm {
    PairNorm {
        gain: oracle::activations(PAIR_COLS, 0x6C0, 0.5)
            .iter()
            .map(|g| 1.0 + g.abs())
            .collect(),
        eps,
    }
}

pub fn row_bytes(quant: Quant, cols: usize) -> usize {
    cols / quant.values_per_block() * quant.block_bytes()
}

fn tag(quant: Quant) -> u64 {
    match quant {
        Quant::Q4K => 0,
        Quant::Q6K => 1,
        Quant::A4 => 2,
        Quant::A8 => 6,
        Quant::Q8_0 => 3,
        Quant::Q5K => 4,
        Quant::F32 => 5,
    }
}

pub struct PairWeights {
    pub gate: Vec<u8>,
    pub up: Vec<u8>,
    pub shared_gate: Vec<u8>,
    pub shared_up: Vec<u8>,
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

/// A stack of experts quantized at 8 bits ([`Fixture::expert_as_a8`]): the
/// MLX split holds its experts at MLX_A4.
fn stack_as_a8(
    fix: &Fixture,
    name: &str,
    cols: usize,
    rows: usize,
    from: usize,
    count: usize,
) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for expert in from..from + count {
        out.extend_from_slice(&fix.expert_as_a8(name, cols, rows, expert)?);
    }
    Ok(out)
}

/// A plane quantized at 8 bits ([`Fixture::rows_as_a8`]).
fn plane_as_a8(fix: &Fixture, name: &str, cols: usize, rows: usize) -> Result<Vec<u8>, String> {
    fix.rows_as_a8(name, cols, rows)
}

/// Q4 reads the gate and up experts. Q6 has no gate or up experts in the
/// fixture, so each expert is one Q6_K down expert's blocks, read as 512 by
/// 2048: the byte length is the same. A4 reads its own gate and up stacks,
/// which an MLX-format fixture holds; A8 reads the same stacks quantized at
/// 8 bits. The fixtures hold no Q8_0 or Q5_K experts: those are refused by
/// name (their clauses build synthetic rows).
pub fn pair_weights_for(
    e: &ExpertsGeom,
    fix: &Fixture,
    quant: Quant,
) -> Result<PairWeights, String> {
    let ty = quant.ggml();
    let weights = match quant {
        Quant::Q8_0 | Quant::Q5K | Quant::F32 => {
            return Err(format!("the fixture has no {} experts", quant.name()));
        }
        Quant::Q4K => PairWeights {
            gate: stack(
                fix,
                "blk.0.ffn_gate_exps.weight",
                ty,
                PAIR_COLS,
                e.k,
                0,
                N_EXPERTS,
            )?,
            up: stack(
                fix,
                "blk.0.ffn_up_exps.weight",
                ty,
                PAIR_COLS,
                e.k,
                0,
                N_EXPERTS,
            )?,
            shared_gate: if_shared(e, || {
                plane(fix, "blk.0.ffn_gate_shexp.weight", ty, PAIR_COLS, e.k)
            })?,
            shared_up: if_shared(e, || {
                plane(fix, "blk.0.ffn_up_shexp.weight", ty, PAIR_COLS, e.k)
            })?,
        },
        Quant::Q6K => {
            let [down, shared] = q6_down(fix)?;
            PairWeights {
                gate: stack(fix, &down, ty, e.k, PAIR_COLS, 0, N_EXPERTS)?,
                up: stack(fix, &down, ty, e.k, PAIR_COLS, 8, N_EXPERTS)?,
                shared_gate: if_shared(e, || plane(fix, &shared, ty, e.k, PAIR_COLS))?,
                shared_up: if_shared(e, || {
                    Ok(fix.expert(&down, ty, e.k, PAIR_COLS, 16)?.to_vec())
                })?,
            }
        }
        Quant::A4 => PairWeights {
            gate: stack(
                fix,
                "blk.0.ffn_gate_exps.weight",
                ty,
                PAIR_COLS,
                e.k,
                0,
                N_EXPERTS,
            )?,
            up: stack(
                fix,
                "blk.0.ffn_up_exps.weight",
                ty,
                PAIR_COLS,
                e.k,
                0,
                N_EXPERTS,
            )?,
            shared_gate: if_shared(e, || {
                plane(fix, "blk.0.ffn_gate_shexp.weight", ty, PAIR_COLS, e.k)
            })?,
            shared_up: if_shared(e, || {
                plane(fix, "blk.0.ffn_up_shexp.weight", ty, PAIR_COLS, e.k)
            })?,
        },
        // The MLX split holds its experts at MLX_A4; the A8 weights are the
        // same experts quantized at 8 bits.
        Quant::A8 => PairWeights {
            gate: stack_as_a8(
                fix,
                "blk.0.ffn_gate_exps.weight",
                PAIR_COLS,
                e.k,
                0,
                N_EXPERTS,
            )?,
            up: stack_as_a8(
                fix,
                "blk.0.ffn_up_exps.weight",
                PAIR_COLS,
                e.k,
                0,
                N_EXPERTS,
            )?,
            shared_gate: if_shared(e, || {
                plane_as_a8(fix, "blk.0.ffn_gate_shexp.weight", PAIR_COLS, e.k)
            })?,
            shared_up: if_shared(e, || {
                plane_as_a8(fix, "blk.0.ffn_up_shexp.weight", PAIR_COLS, e.k)
            })?,
        },
    };
    let one = weights.gate.len() / N_EXPERTS;
    let shared_len = if e.has_shared { one } else { 0 };
    if weights.up.len() != weights.gate.len()
        || weights.shared_gate.len() != shared_len
        || weights.shared_up.len() != shared_len
    {
        return Err(format!(
            "{} pair weights are not expert-sized",
            quant.name()
        ));
    }
    if e.has_shared
        && (weights.shared_gate == weights.gate[..one] || weights.shared_up == weights.up[..one])
    {
        return Err(format!(
            "{} shared expert matches expert 0, so the slot select cannot be seen",
            quant.name()
        ));
    }
    Ok(weights)
}

pub struct FoldWeights {
    pub down: Vec<u8>,
    pub shared: Vec<u8>,
}

/// A pair's weights of synthetic rows ([`synthetic::matrix`]), for a quant
/// no fixture holds: the four experts' gate and up stacks and the shared
/// pair, each from its own seed.
pub fn synthetic_pair_for(e: &ExpertsGeom, quant: Quant, seed: u64) -> PairWeights {
    let stack = N_EXPERTS * e.k;
    PairWeights {
        gate: synthetic::matrix(quant, stack, PAIR_COLS, seed),
        up: synthetic::matrix(quant, stack, PAIR_COLS, seed + 1),
        shared_gate: if e.has_shared {
            synthetic::matrix(quant, e.k, PAIR_COLS, seed + 2)
        } else {
            Vec::new()
        },
        shared_up: if e.has_shared {
            synthetic::matrix(quant, e.k, PAIR_COLS, seed + 3)
        } else {
            Vec::new()
        },
    }
}

/// A fold's weights of synthetic rows at `rows` rows an expert: the four
/// experts' down stack and the shared down.
pub fn synthetic_fold_for(e: &ExpertsGeom, quant: Quant, rows: usize, seed: u64) -> FoldWeights {
    FoldWeights {
        down: synthetic::matrix(quant, N_EXPERTS * rows, e.k, seed),
        shared: if e.has_shared {
            synthetic::matrix(quant, rows, e.k, seed + 1)
        } else {
            Vec::new()
        },
    }
}

fn fold_weights(e: &ExpertsGeom, fix: &Fixture, quant: Quant) -> Result<FoldWeights, String> {
    fold_weights_for(e, fix, quant, FOLD_ROWS)
}

/// The fold's weights at `rows` rows an expert: the engine's whole rows, or a
/// count a threadgroup does not divide, for the row tail. The reader hands
/// out an expert's whole rows, so a partial count is the stack repacked at
/// the count's own stride, the stride the kernel's row offsets read.
pub fn fold_weights_for(
    e: &ExpertsGeom,
    fix: &Fixture,
    quant: Quant,
    rows: usize,
) -> Result<FoldWeights, String> {
    let ty = quant.ggml();
    let [name, shared] = match quant {
        Quant::Q8_0 | Quant::Q5K | Quant::F32 => {
            return Err(format!("the fixture has no {} experts", quant.name()));
        }
        Quant::Q4K | Quant::A4 | Quant::A8 => [
            "blk.0.ffn_down_exps.weight".to_owned(),
            "blk.0.ffn_down_shexp.weight".to_owned(),
        ],
        Quant::Q6K => q6_down(fix)?,
    };
    let rb = row_bytes(quant, e.k);
    let mut down = Vec::with_capacity(N_EXPERTS * rows * rb);
    for expert in 0..N_EXPERTS {
        // The MLX split holds its down experts at MLX_A4; the A8 weights are
        // the same experts quantized at 8 bits.
        let whole_a8;
        let whole: &[u8] = if quant == Quant::A8 {
            whole_a8 = fix.expert_as_a8(&name, e.k, FOLD_ROWS, expert)?;
            &whole_a8
        } else {
            fix.expert(&name, ty, e.k, FOLD_ROWS, expert)?
        };
        down.extend_from_slice(&whole[..rows * rb]);
    }
    let weights = FoldWeights {
        down,
        shared: if_shared(e, || {
            if quant == Quant::A8 {
                fix.rows_as_a8(&shared, e.k, rows)
            } else {
                Ok(fix.rows(&shared, ty, e.k, rows)?.to_vec())
            }
        })?,
    };
    let one = weights.down.len() / N_EXPERTS;
    if weights.shared.len() != if e.has_shared { one } else { 0 } {
        return Err(format!("{} shared down is not one expert", quant.name()));
    }
    if e.has_shared && weights.shared == weights.down[..one] {
        return Err(format!(
            "{} shared down matches expert 0, so the slot select cannot be seen",
            quant.name()
        ));
    }
    Ok(weights)
}

pub struct PairIn<'a> {
    pub quant: Quant,
    /// The type of the shared gate and up: `quant`, or the second of a pair
    /// of the gate-up table.
    pub shared_quant: Quant,
    pub n_experts: usize,
    pub ids: &'a [i32],
    pub gate: &'a [u8],
    pub up: &'a [u8],
    pub shared_gate: &'a [u8],
    pub shared_up: &'a [u8],
    pub x: &'a [f32],
    pub norm: &'a PairNorm,
    pub y0: &'a [f32],
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

pub fn launch_pair(h: &Harness<'_>, launch: PairIn<'_>) -> Result<Vec<f32>, String> {
    let e = h.geom().experts;
    let (device, stream) = (h.device, h.stream);
    let y_len = e.slots * e.k;
    let gate = padded(0xFFu8, weight_pad(launch.quant), launch.gate, WEIGHT_TAIL);
    let up = padded(0xFFu8, weight_pad(launch.quant), launch.up, WEIGHT_TAIL);
    let shared_gate = padded(
        0xFFu8,
        weight_pad(launch.shared_quant),
        launch.shared_gate,
        WEIGHT_TAIL,
    );
    let shared_up = padded(
        0xFFu8,
        weight_pad(launch.shared_quant),
        launch.shared_up,
        WEIGHT_TAIL,
    );
    let x = padded(f32::from_bits(SENTINEL_BITS), ACT_PAD, launch.x, 1);
    let gain = padded(f32::from_bits(SENTINEL_BITS), ACT_PAD, &launch.norm.gain, 1);
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
                shared_quant: launch.shared_quant,
                slots: e.slots,
                k: e.k,
                cols: PAIR_COLS,
                n_experts,
                ids: ids.window(Elems(VALUE_PAD), Elems(launch.ids.len()))?,
                gate: gate.window(Elems(weight_pad(launch.quant)), Elems(launch.gate.len()))?,
                up: up.window(Elems(weight_pad(launch.quant)), Elems(launch.up.len()))?,
                shared_gate: e
                    .has_shared
                    .then(|| {
                        shared_gate.window(
                            Elems(weight_pad(launch.shared_quant)),
                            Elems(launch.shared_gate.len()),
                        )
                    })
                    .transpose()?,
                shared_up: e
                    .has_shared
                    .then(|| {
                        shared_up.window(
                            Elems(weight_pad(launch.shared_quant)),
                            Elems(launch.shared_up.len()),
                        )
                    })
                    .transpose()?,
                x: x.window(Elems(ACT_PAD), Elems(launch.x.len()))?,
                norm: LoadNorm::Gain {
                    gain: gain.window(Elems(ACT_PAD), Elems(launch.norm.gain.len()))?,
                    eps: launch.norm.eps,
                },
                y: y.window_mut(Elems(OUT_GUARD), Elems(y_len))?,
                out_stride: e.k,
                fault: device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    read_interior(h, &y, y_len)
}

pub struct FoldIn<'a> {
    pub quant: Quant,
    /// The type of the shared down: `quant`, or the second of a pair of the
    /// down table.
    pub shared_quant: Quant,
    pub n_experts: usize,
    pub ids: &'a [i32],
    pub down: &'a [u8],
    pub shared: &'a [u8],
    pub h: &'a [f32],
    pub w: &'a [f32],
    pub y0: &'a [f32],
}

pub fn launch_fold(h: &Harness<'_>, launch: FoldIn<'_>) -> Result<Vec<f32>, String> {
    let e = h.geom().experts;
    let (device, stream) = (h.device, h.stream);
    let down = padded(0xFFu8, weight_pad(launch.quant), launch.down, WEIGHT_TAIL);
    let shared = padded(
        0xFFu8,
        weight_pad(launch.shared_quant),
        launch.shared,
        WEIGHT_TAIL,
    );
    let hidden = padded(f32::from_bits(SENTINEL_BITS), ACT_PAD, launch.h, 1);
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
        let shared = shared.window(
            Elems(weight_pad(launch.shared_quant)),
            Elems(launch.shared.len()),
        )?;
        let out = y.window_mut(Elems(OUT_GUARD), Elems(rows))?;
        let plan = if launch.shared_quant == quant {
            shared_plan(&e, shared)
        } else {
            SharedPlan::Apart {
                quant: launch.shared_quant,
                weights: shared,
            }
        };
        h.kernels.fold(
            encoder,
            FoldParams {
                quant,
                shared: plan,
                rows,
                slots: e.slots,
                k: e.k,
                n_experts,
                ids: ids.window(Elems(VALUE_PAD), Elems(launch.ids.len()))?,
                down: down.window(Elems(weight_pad(launch.quant)), Elems(launch.down.len()))?,
                h: hidden.window(Elems(ACT_PAD), Elems(launch.h.len()))?,
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
pub fn expect_bits(got: &[f32], want: &[f32], what: &str) -> Result<(), String> {
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

pub fn check_pair_quant(h: &Harness<'_>, fix: &Fixture, quant: Quant) -> Result<String, String> {
    check_pair_weights(
        h,
        quant,
        &pair_weights_for(&h.geom().experts, fix, quant)?,
        &ffn_norm(fix)?,
    )
}

/// The pair over `weights` against the oracle band, every slot's rows, the
/// shared slot's too: the worst error over the band.
pub fn check_pair_weights(
    h: &Harness<'_>,
    quant: Quant,
    weights: &PairWeights,
    norm: &PairNorm,
) -> Result<String, String> {
    check_pair_mixed(h, quant, quant, weights, norm)
}

/// [`check_pair_weights`] for stacks of `quant` and a shared expert of
/// `shared_quant`: the shared slot's rows are held to the shared type's band.
pub fn check_pair_mixed(
    h: &Harness<'_>,
    quant: Quant,
    shared_quant: Quant,
    weights: &PairWeights,
    norm: &PairNorm,
) -> Result<String, String> {
    let e = h.geom().experts;
    let x = oracle::activations(PAIR_COLS, 0xA11 + tag(quant), 1.0);
    let y0 = vec![f32::from_bits(SENTINEL_BITS); e.slots * e.k];
    h.clear_fault()?;
    let got = launch_pair(
        h,
        PairIn {
            quant,
            shared_quant,
            n_experts: N_EXPERTS,
            ids: &IDS,
            gate: &weights.gate,
            up: &weights.up,
            shared_gate: &weights.shared_gate,
            shared_up: &weights.shared_up,
            x: &x,
            norm,
            y0: &y0,
        },
    )?;
    if let Some(f) = h.fault()? {
        return Err(format!("{} pair raised {f}", quant.name()));
    }
    let x64 = oracle::rms_norm(&x, &norm.gain, norm.eps);
    let mut worst = 0.0f64;
    for slot in 0..e.slots {
        // The shared slot is read in the shared type, the rest in the stacks'.
        let slot_quant = if slot == e.shared_slot {
            shared_quant
        } else {
            quant
        };
        let nblocks = PAIR_COLS / slot_quant.values_per_block();
        let rb = row_bytes(slot_quant, PAIR_COLS);
        for row in 0..e.k {
            let (gate_row, up_row) = if slot == e.shared_slot {
                (
                    &weights.shared_gate[row * rb..(row + 1) * rb],
                    &weights.shared_up[row * rb..(row + 1) * rb],
                )
            } else {
                let expert = IDS[slot] as usize;
                let at = (expert * e.k + row) * rb;
                (&weights.gate[at..at + rb], &weights.up[at..at + rb])
            };
            let g = oracle::reference(slot_quant, gate_row, 1, PAIR_COLS, &x64)?;
            let u = oracle::reference(slot_quant, up_row, 1, PAIR_COLS, &x64)?;
            let (want, width) = oracle_experts::pair_width(
                slot_quant,
                nblocks,
                g.y[0],
                g.magnitude[0],
                u.y[0],
                u.magnitude[0],
            )?;
            let value = got[slot * e.k + row];
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
    let e = h.geom().experts;
    let quant = Quant::Q4K;
    let weights = pair_weights_for(&e, fix, quant)?;
    let norm = ffn_norm(fix)?;
    let x = oracle::activations(PAIR_COLS, 0xB1D, 1.0);
    let y_len = e.slots * e.k;
    h.clear_fault()?;
    let clean = launch_pair(
        h,
        PairIn {
            quant,
            shared_quant: quant,
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
                shared_quant: quant,
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
        for slot in 0..e.slots {
            let row = &got[slot * e.k..(slot + 1) * e.k];
            let clean_row = &clean[slot * e.k..(slot + 1) * e.k];
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
    let e = h.geom().experts;
    let norm = ffn_norm(fix)?;
    let y_len = e.slots * e.k;
    for quant in [Quant::Q4K, Quant::Q6K] {
        let weights = pair_weights_for(&e, fix, quant)?;
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
                    shared_quant: quant,
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
                    k / e.k,
                    k % e.k,
                    got[k]
                ));
            }
        }
    }
    h.clear_fault()?;
    Ok(Verdict::Noted(
        "a NaN input, an infinite input and a zero input with a zero epsilon raised norm_quant only, NaN in every slot's rows, both quants".to_string(),
    ))
}

/// An id past the stack writes NaN in that slot only and raises
/// [`FaultSite::ExpertId`], the other slots the clean launch's bits; the
/// norm's fault inputs (a NaN input, an infinite one, a zero input with a
/// zero epsilon) raise [`FaultSite::NormQuant`] with NaN in every slot's
/// rows. One quant's pair over `weights`.
pub fn pair_faults(
    h: &Harness<'_>,
    quant: Quant,
    weights: &PairWeights,
    norm: &PairNorm,
) -> Result<(), String> {
    let e = h.geom().experts;
    let x = oracle::activations(PAIR_COLS, 0x1B1D, 1.0);
    let y_len = e.slots * e.k;

    // An id past the stack: that slot's rows NaN, the others the clean bits.
    h.clear_fault()?;
    let clean = launch_pair(
        h,
        PairIn {
            quant,
            shared_quant: quant,
            n_experts: N_EXPERTS,
            ids: &IDS,
            gate: &weights.gate,
            up: &weights.up,
            shared_gate: &weights.shared_gate,
            shared_up: &weights.shared_up,
            x: &x,
            norm,
            y0: &vec![f32::from_bits(SENTINEL_BITS); y_len],
        },
    )?;
    if let Some(f) = h.fault()? {
        return Err(format!("the clean pair raised {f}"));
    }
    for bad in [-1, N_EXPERTS as i32] {
        let mut bad_ids = IDS;
        bad_ids[3] = bad;
        h.clear_fault()?;
        let got = launch_pair(
            h,
            PairIn {
                quant,
                shared_quant: quant,
                n_experts: N_EXPERTS,
                ids: &bad_ids,
                gate: &weights.gate,
                up: &weights.up,
                shared_gate: &weights.shared_gate,
                shared_up: &weights.shared_up,
                x: &x,
                norm,
                y0: &vec![1.0; y_len],
            },
        )?;
        common::expect_fault(h, &format!("id {bad}"), FaultSite::ExpertId)?;
        for slot in 0..e.slots {
            let row = &got[slot * e.k..(slot + 1) * e.k];
            let clean_row = &clean[slot * e.k..(slot + 1) * e.k];
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

    // The norm's fault inputs: NaN rows in every slot's rows.
    for (what, mut x, eps) in [
        (
            "a NaN input",
            oracle::activations(PAIR_COLS, 0x1A0, 1.0),
            norm.eps,
        ),
        (
            "an infinite input",
            oracle::activations(PAIR_COLS, 0x1A1, 1.0),
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
        h.clear_fault()?;
        let got = launch_pair(
            h,
            PairIn {
                quant,
                shared_quant: quant,
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
        common::expect_fault(h, what, FaultSite::NormQuant)?;
        if let Some(k) = got.iter().position(|v| !v.is_nan()) {
            return Err(format!(
                "{what}: slot {} row {} is {:?}, not NaN",
                k / e.k,
                k % e.k,
                got[k]
            ));
        }
    }
    h.clear_fault()?;
    Ok(())
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
    let e = h.geom().experts;
    let [gate, up, shared_gate, shared_up] = rows;
    let ids = vec![0; e.routed_ids];
    let y_len = e.slots * e.k;
    let got = launch_pair(
        h,
        PairIn {
            quant,
            shared_quant: quant,
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
    Ok(got[..e.k].to_vec())
}

pub fn check_pair_shared(h: &Harness<'_>, fix: &Fixture, quant: Quant) -> Result<(), String> {
    let e = h.geom().experts;
    let weights = pair_weights_for(&e, fix, quant)?;
    let norm = ffn_norm(fix)?;
    let x = oracle::activations(PAIR_COLS, 0x5E1 + tag(quant), 1.0);
    let y_len = e.slots * e.k;
    h.clear_fault()?;
    let full = launch_pair(
        h,
        PairIn {
            quant,
            shared_quant: quant,
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
    for slot in [0usize, 1, e.shared_slot]
        .into_iter()
        .filter(|&s| s < e.slots)
    {
        let (gate, up, shared_gate, shared_up) = if slot == e.shared_slot {
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
        let got = &full[slot * e.k..(slot + 1) * e.k];
        expect_bits(got, &solo, &format!("{} slot {slot}", quant.name()))?;
    }
    Ok(())
}

pub fn pair_shared(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let e = h.geom().experts;
    check_pair_shared(h, fix, Quant::Q4K)?;
    check_pair_shared(h, fix, Quant::Q6K)?;
    Ok(Verdict::Noted(if e.has_shared {
        "slots 0, 1 and the shared 8 bit-equal to a solo launch of the same rows, both quants"
            .to_string()
    } else {
        "slots 0 and 1 bit-equal to a solo launch of the same rows, both quants; no shared slot"
            .to_string()
    }))
}

pub fn hidden_states(e: &ExpertsGeom, quant: Quant) -> Vec<f32> {
    let mut out = Vec::with_capacity(e.slots * e.k);
    for slot in 0..e.slots {
        out.extend(oracle::activations(
            e.k,
            0xB00 + slot as u64 + tag(quant),
            1.0,
        ));
    }
    out
}

pub fn check_fold_quant(
    h: &Harness<'_>,
    fix: &Fixture,
    quant: Quant,
    rows: usize,
) -> Result<String, String> {
    check_fold_weights(
        h,
        quant,
        &fold_weights_for(&h.geom().experts, fix, quant, rows)?,
        rows,
    )
}

/// The fold over `weights` (`rows` rows an expert) against the oracle band:
/// the worst error over the band.
pub fn check_fold_weights(
    h: &Harness<'_>,
    quant: Quant,
    weights: &FoldWeights,
    rows: usize,
) -> Result<String, String> {
    check_fold_mixed(h, quant, quant, weights, rows)
}

/// [`check_fold_weights`] for a down stack of `quant` and a shared down of
/// `shared_quant`: the shared slot's rows are read in the shared type, and
/// the band is the two ranges' paths end to end.
pub fn check_fold_mixed(
    h: &Harness<'_>,
    quant: Quant,
    shared_quant: Quant,
    weights: &FoldWeights,
    rows: usize,
) -> Result<String, String> {
    let e = h.geom().experts;
    let route = route_weights(&e);
    let hidden = hidden_states(&e, quant);
    let resid = oracle::activations(rows, 0xF01D + tag(quant), 2.0);
    h.clear_fault()?;
    let got = launch_fold(
        h,
        FoldIn {
            quant,
            shared_quant,
            n_experts: N_EXPERTS,
            ids: &IDS,
            down: &weights.down,
            shared: &weights.shared,
            h: &hidden,
            w: &route,
            y0: &resid,
        },
    )?;
    if let Some(f) = h.fault()? {
        return Err(format!("{} fold raised {f}", quant.name()));
    }
    let mut worst = 0.0f64;
    for row in 0..rows {
        let mut want = f64::from(resid[row]);
        let mut magnitude = 0.0;
        for slot in 0..e.slots {
            let slot_quant = if slot == e.shared_slot {
                shared_quant
            } else {
                quant
            };
            let rb = row_bytes(slot_quant, e.k);
            let bytes = if slot == e.shared_slot {
                &weights.shared[row * rb..(row + 1) * rb]
            } else {
                let expert = IDS[slot] as usize;
                let at = (expert * rows + row) * rb;
                &weights.down[at..at + rb]
            };
            let x: Vec<f64> = hidden[slot * e.k..(slot + 1) * e.k]
                .iter()
                .map(|&v| f64::from(route[slot]) * f64::from(v))
                .collect();
            let reference = oracle::reference(slot_quant, bytes, 1, e.k, &x)?;
            want += reference.y[0];
            magnitude += reference.magnitude[0];
        }
        let width = if shared_quant == quant {
            oracle_experts::fold_width(&e, quant, magnitude, want)
        } else {
            oracle_experts::fold_width_apart(&e, quant, shared_quant, magnitude, want)
        };
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

pub fn fold_bad_id(h: &Harness<'_>, weights: &FoldWeights, quant: Quant) -> Result<(), String> {
    let e = h.geom().experts;
    let route = route_weights(&e);
    let hidden = hidden_states(&e, quant);
    for bad in [-1, N_EXPERTS as i32] {
        let mut ids = IDS;
        ids[0] = bad;
        h.clear_fault()?;
        let got = launch_fold(
            h,
            FoldIn {
                quant,
                shared_quant: quant,
                n_experts: N_EXPERTS,
                ids: &ids,
                down: &weights.down,
                shared: &weights.shared,
                h: &hidden,
                w: &route,
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

/// A row count a threadgroup's four rows do not divide: the last SIMD
/// group's live rows are one, and its dead rows are never stored.
pub const FOLD_TAIL_ROWS: usize = 2045;

const _: () = assert!(!FOLD_TAIL_ROWS.is_multiple_of(ROWS_PER_SIMD_GROUP));

pub fn fold(h: &Harness<'_>, fix: &Fixture) -> Clause {
    let q4 = check_fold_quant(h, fix, Quant::Q4K, FOLD_ROWS)?;
    let q6 = check_fold_quant(h, fix, Quant::Q6K, FOLD_ROWS)?;
    // The row tail: the rows the last SIMD group does not fill, both quants.
    let tail_q4 = check_fold_quant(h, fix, Quant::Q4K, FOLD_TAIL_ROWS)?;
    let tail_q6 = check_fold_quant(h, fix, Quant::Q6K, FOLD_TAIL_ROWS)?;
    fold_bad_id(
        h,
        &fold_weights(&h.geom().experts, fix, Quant::Q4K)?,
        Quant::Q4K,
    )?;
    Ok(Verdict::Noted(format!(
        "worst error over its band: {q4}, {q6}; tail {FOLD_TAIL_ROWS} rows: {tail_q4}, {tail_q6}; \
         a bad id writes NaN and raises expert_id"
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
    /// The element the input's window starts at in its buffer.
    x_at: usize,
    /// The element the gain's window starts at in its buffer.
    gain_at: usize,
    y_len: usize,
}

impl PairArgs {
    /// The table's arguments: every window short of its good length, so a
    /// case's own check refuses before any window check can.
    fn whole(e: &ExpertsGeom, gate_len: usize) -> Self {
        Self {
            slots: e.slots,
            k: e.k,
            n_experts: 1,
            id_len: e.routed_ids,
            gate_len,
            stride: e.k,
            x: 16,
            x_at: 0,
            gain_at: 0,
            y_len: 64,
        }
    }

    /// The well-formed launch of the same buffers.
    fn good(e: &ExpertsGeom, gate_len: usize) -> Self {
        Self {
            x: PAIR_COLS,
            y_len: e.slots * e.k,
            ..Self::whole(e, gate_len)
        }
    }
}

fn refuse_pair(
    h: &Harness<'_>,
    b: &mut Refusals,
    a: &PairArgs,
    encoder: &mut bloomery_metal::encode::ComputeEncoder<'_>,
) -> Result<(), bloomery_device::DeviceError> {
    let e = h.geom().experts;
    h.kernels.pair(
        encoder,
        PairParams {
            quant: Quant::Q4K,
            shared_quant: Quant::Q4K,
            slots: a.slots,
            k: a.k,
            cols: PAIR_COLS,
            n_experts: a.n_experts,
            ids: b.ids.window(Elems(0), Elems(a.id_len))?,
            gate: b.bytes.window(Elems(0), Elems(a.gate_len))?,
            up: b.bytes.window(Elems(0), Elems(a.gate_len))?,
            shared_gate: e
                .has_shared
                .then(|| b.bytes.window(Elems(0), Elems(a.gate_len)))
                .transpose()?,
            shared_up: e
                .has_shared
                .then(|| b.bytes.window(Elems(0), Elems(a.gate_len)))
                .transpose()?,
            x: b.values.window(Elems(a.x_at), Elems(a.x))?,
            norm: LoadNorm::Gain {
                gain: b.values.window(Elems(a.gain_at), Elems(a.x))?,
                eps: 1e-6,
            },
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
    /// The element the hidden window starts at in its buffer.
    h_at: usize,
    y_len: usize,
}

impl FoldArgs {
    /// The table's arguments: every window short of its good length, so a
    /// case's own check refuses before any window check can.
    fn whole(e: &ExpertsGeom) -> Self {
        Self {
            slots: e.slots,
            k: e.k,
            n_experts: 1,
            id_len: e.routed_ids,
            rows: 8,
            down_len: 16,
            h_len: 16,
            h_at: 0,
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
    let e = h.geom().experts;
    h.kernels.fold(
        encoder,
        FoldParams {
            quant: Quant::Q4K,
            shared: shared_plan(&e, b.bytes.window(Elems(0), Elems(a.down_len))?),
            rows: a.rows,
            slots: a.slots,
            k: a.k,
            n_experts: a.n_experts,
            ids: b.ids.window(Elems(0), Elems(a.id_len))?,
            down: b.bytes.window(Elems(0), Elems(a.down_len))?,
            h: b.values.window(Elems(a.h_at), Elems(a.h_len))?,
            w: b.route.window(Elems(0), Elems(e.slots))?,
            y: &b.marker.window_mut(Elems(0), Elems(a.y_len))?,
            fault: h.device.fault_word(),
            layer: LAYER,
        },
    )
}

/// The launches the pair and the fold must refuse, each by name, with nothing
/// written. A pair's input or gain window and a fold's hidden window that do
/// not start on 16 bytes are refused: deleting the `float4_aligned` calls in
/// `experts.rs` turns those cases red.
pub fn experts_bounds(h: &Harness<'_>, _fix: &Fixture) -> Clause {
    let e = h.geom().experts;
    let (device, stream) = (h.device, h.stream);
    let gate_len = e.k * row_bytes(Quant::Q4K, PAIR_COLS);
    let mut bufs = Refusals {
        marker: device
            .upload(stream, &vec![1.0f32; e.slots * e.k])
            .map_err(why)?,
        bytes: device.upload(stream, &vec![0u8; gate_len]).map_err(why)?,
        // Four floats past the longest window, so a window can start one
        // float in.
        values: device
            .upload(stream, &vec![0f32; e.slots * e.k + 4])
            .map_err(why)?,
        route: device.upload(stream, &vec![1.0f32; e.slots]).map_err(why)?,
        ids: device
            .upload(stream, &vec![0i32; e.routed_ids])
            .map_err(why)?,
    };
    let pair_whole = PairArgs::whole(&e, gate_len);
    let pair_good = PairArgs::good(&e, gate_len);
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
            named(e.slots),
            "",
            "window",
            Box::new(|a| a.slots = 3),
        ),
        bad_pair(
            "a pair of width 256",
            named(e.k),
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
        BadArg::new(
            "a pair input window one float into its buffer",
            "16-byte aligned",
            Box::new(move |b: &mut Refusals, encoder| {
                let mut a = pair_good;
                a.x_at = 1;
                refuse_pair(h, b, &a, encoder)
            }),
        )
        .naming("the x window"),
        BadArg::new(
            "a pair gain window one float into its buffer",
            "16-byte aligned",
            Box::new(move |b: &mut Refusals, encoder| {
                let mut a = pair_good;
                a.gain_at = 1;
                refuse_pair(h, b, &a, encoder)
            }),
        )
        .naming("the gain window"),
    ];
    let fold_whole = FoldArgs::whole(&e);
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
            named(e.slots),
            "",
            "window",
            Box::new(|a| a.slots = 3),
        ),
        bad_fold(
            "a fold of width 256",
            named(e.k),
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
        BadArg::new(
            "a fold hidden window one float into its buffer",
            "16-byte aligned",
            Box::new(move |b: &mut Refusals, encoder| {
                let mut a = fold_whole;
                a.down_len = a.rows * row_bytes(Quant::Q4K, e.k);
                a.h_len = e.slots * e.k;
                a.y_len = a.rows;
                a.h_at = 1;
                refuse_fold(h, b, &a, encoder)
            }),
        )
        .naming("the hidden window"),
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
                                shared: shared_plan(&e, b.bytes.window(Elems(0), Elems(16))?),
                                rows: FOLD_ROWS,
                                slots: e.slots,
                                k: e.k,
                                n_experts: 1,
                                ids: b.ids.window(Elems(0), Elems(e.routed_ids))?,
                                down: b.bytes.window(Elems(0), Elems(16))?,
                                h: same,
                                w: b.route.window(Elems(0), Elems(e.slots))?,
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
        a.down_len = a.rows * row_bytes(Quant::Q4K, e.k);
        a.h_len = e.slots * e.k;
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
    let e = h.geom().experts;
    let (device, stream) = (h.device, h.stream);
    let down_len = 4 * row_bytes(Quant::Q4K, e.k);
    let down = fix.rows("blk.0.ffn_down_exps.weight", GgmlType::Q4_K, e.k, 4)?;
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
            .upload(stream, &vec![1.0f32; e.slots * e.k])
            .map_err(why)?,
        route: device.upload(stream, &vec![1.0f32; e.slots]).map_err(why)?,
        ids: device
            .upload(stream, &vec![0i32; e.routed_ids])
            .map_err(why)?,
    };
    let cases = [
        (
            "an Apart pair of no entry",
            e.slots,
            e.k,
            1usize,
            e.routed_ids,
        ),
        (
            "an Apart pair of no entry with a bad slot count",
            1,
            e.k,
            1,
            e.routed_ids,
        ),
        (
            "an Apart pair of no entry with a bad width",
            e.slots,
            256,
            1,
            e.routed_ids,
        ),
        (
            "an Apart pair of no entry with an empty stack",
            e.slots,
            e.k,
            0,
            e.routed_ids,
        ),
        (
            "an Apart pair of no entry with an empty id list",
            e.slots,
            e.k,
            1,
            0,
        ),
    ]
    .into_iter()
    .map(|(what, slots, k, n_experts, id_len)| {
        BadArg::new(
            what,
            "no entry",
            Box::new(move |b: &mut Bufs, encoder| {
                let y = b.marker.window_mut(Elems(0), Elems(8))?;
                h.kernels.fold(
                    encoder,
                    FoldParams {
                        quant: Quant::Q4K,
                        // Q4_K stacks beside a Q6_K shared expert: not a pair
                        // of the down table.
                        shared: SharedPlan::Apart {
                            quant: Quant::Q6K,
                            weights: b.down.window(Elems(0), Elems(16))?,
                        },
                        rows: 4,
                        slots,
                        k,
                        n_experts,
                        ids: b.ids.window(Elems(0), Elems(id_len))?,
                        down: b.down.window(Elems(0), Elems(16))?,
                        h: b.hidden.window(Elems(0), Elems(16))?,
                        w: b.route.window(Elems(0), Elems(e.slots))?,
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
                shared: shared_plan(&e, b.down.window(Elems(0), Elems(down_len))?),
                rows: 4,
                slots: e.slots,
                k: e.k,
                n_experts: 1,
                ids: b.ids.window(Elems(0), Elems(e.routed_ids))?,
                down: b.down.window(Elems(0), Elems(down_len))?,
                h: b.hidden.window(Elems(0), Elems(e.slots * e.k))?,
                w: b.route.window(Elems(0), Elems(e.slots))?,
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
        "an Apart pair with no entry is refused by name before the slot count, the width, an empty stack and an empty id list".to_string(),
    ))
}

/// A mixed pair's weights from synthetic rows: the stacks in `quant`, the
/// shared gate and up in `shared`.
fn synthetic_pair_apart(
    e: &ExpertsGeom,
    (quant, shared): (Quant, Quant),
    seed: u64,
) -> PairWeights {
    let stack = N_EXPERTS * e.k;
    PairWeights {
        gate: synthetic::matrix(quant, stack, PAIR_COLS, seed),
        up: synthetic::matrix(quant, stack, PAIR_COLS, seed + 1),
        shared_gate: synthetic::matrix(shared, e.k, PAIR_COLS, seed + 2),
        shared_up: synthetic::matrix(shared, e.k, PAIR_COLS, seed + 3),
    }
}

/// A mixed fold's weights from synthetic rows at `rows` rows an expert: the
/// down stack in `quant`, the shared down in `shared`.
fn synthetic_fold_apart(
    e: &ExpertsGeom,
    (quant, shared): (Quant, Quant),
    rows: usize,
    seed: u64,
) -> FoldWeights {
    FoldWeights {
        down: synthetic::matrix(quant, N_EXPERTS * rows, e.k, seed),
        shared: synthetic::matrix(shared, rows, e.k, seed + 1),
    }
}

/// The epsilon the Apart clauses' norm takes.
const APART_EPS: f32 = 1e-6;

/// The pair over stacks of one type and a shared expert of another: every
/// pair of the gate-up table against the oracle, the shared slot held to the
/// shared type's band, and a pair of types outside the table refused by name.
pub fn apart_pair(h: &Harness<'_>) -> Clause {
    let e = h.geom().experts;
    if !e.has_shared {
        return Err("the Apart pair needs an instance with a shared slot".to_string());
    }
    let norm = synthetic_norm(APART_EPS);
    let mut parts = Vec::new();
    for (i, &(quant, shared)) in APART_GATE_UP.iter().enumerate() {
        let weights = synthetic_pair_apart(&e, (quant, shared), 0xA9A0 + 8 * i as u64);
        let worst = check_pair_mixed(h, quant, shared, &weights, &norm)?;
        parts.push(format!("{worst} beside {}", shared.name()));
    }
    let (quant, shared) = (Quant::Q4K, Quant::Q6K);
    let weights = synthetic_pair_apart(&e, (quant, shared), 0xA9B0);
    let x = oracle::activations(PAIR_COLS, 0xA9B1, 1.0);
    let launched = launch_pair(
        h,
        PairIn {
            quant,
            shared_quant: shared,
            n_experts: N_EXPERTS,
            ids: &IDS,
            gate: &weights.gate,
            up: &weights.up,
            shared_gate: &weights.shared_gate,
            shared_up: &weights.shared_up,
            x: &x,
            norm: &norm,
            y0: &vec![f32::from_bits(SENTINEL_BITS); e.slots * e.k],
        },
    );
    match launched {
        Err(msg) if msg.contains("no entry") => {}
        Err(msg) => {
            return Err(format!(
                "a pair of no entry was refused for another reason: {msg}"
            ));
        }
        Ok(_) => return Err("stacks of q4k and a shared expert of q6k ran".to_string()),
    }
    Ok(Verdict::Noted(format!(
        "worst error over its band: {}; a pair of no entry is refused by name",
        parts.join(", ")
    )))
}

/// The fold over a down stack of one type and a shared down of another:
/// every pair of the down table against the oracle at the engine's rows and
/// at a row count no threadgroup divides, and a pair outside the table
/// refused by name.
pub fn apart_fold(h: &Harness<'_>) -> Clause {
    let e = h.geom().experts;
    if !e.has_shared {
        return Err("the Apart fold needs an instance with a shared slot".to_string());
    }
    let mut parts = Vec::new();
    for (i, &(quant, shared)) in APART_DOWN.iter().enumerate() {
        let seed = 0xA9C0 + 8 * i as u64;
        for rows in [FOLD_ROWS, FOLD_TAIL_ROWS] {
            let weights = synthetic_fold_apart(&e, (quant, shared), rows, seed);
            let worst = check_fold_mixed(h, quant, shared, &weights, rows)?;
            parts.push(format!("{worst} beside {} at {rows} rows", shared.name()));
        }
    }
    let (quant, shared) = (Quant::Q4K, Quant::Q8_0);
    let weights = synthetic_fold_apart(&e, (quant, shared), FOLD_ROWS, 0xA9D0);
    let launched = launch_fold(
        h,
        FoldIn {
            quant,
            shared_quant: shared,
            n_experts: N_EXPERTS,
            ids: &IDS,
            down: &weights.down,
            shared: &weights.shared,
            h: &hidden_states(&e, quant),
            w: &route_weights(&e),
            y0: &oracle::activations(FOLD_ROWS, 0xA9D1, 2.0),
        },
    );
    match launched {
        Err(msg) if msg.contains("no entry") => {}
        Err(msg) => {
            return Err(format!(
                "a pair of no entry was refused for another reason: {msg}"
            ));
        }
        Ok(_) => return Err("a down stack of q4k and a shared down of q8_0 ran".to_string()),
    }
    Ok(Verdict::Noted(format!(
        "worst error over its band: {}; a pair of no entry is refused by name",
        parts.join(", ")
    )))
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
