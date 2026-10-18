//! `gate-quant`: every row of `gguf::QROWS` against llama.cpp mainline's quant oracle.
//!
//! The oracle (`tools/ref/quant_oracle.cpp`, set `ref-quant`, read through `refset::quant`) runs mainline's ggml on seeded
//! rows per weight type: rows ggml quantized, rows of random codes with finite scales, their `to_float`, f32 activation rows
//! in the type's `vec_dot_type` and every `vec_dot`. It also writes a `type` row for every type ggml's own table names, from
//! its traits: the block size and size, and either `dumped` (the set holds cases) or `skip` with the traits ggml leaves NULL
//! (`to_float NULL`, `vec_dot NULL`). A row is held to it by four clauses, one test each (`hw_quant_sizes`,
//! `hw_quant_dequant`, `hw_quant_act`, `hw_quant_dot`), each over every row and ending red with every red row's message.
//! `QUANT_ONLY=<ggml name>[,<name>...]` (`just gate-quant <name>`) takes those rows alone; a name that is neither a row nor a
//! type of ggml's table, and a value that is not UTF-8, are named errors. It is read here and named by the recipe, and is no
//! `BLOOMERY_*` lever.
//!
//! - (a) the row's `blck`, `bytes`, name and tag are what the harness's table of ggml's types states for the type (every case
//!   states what its type's row does, which the reader holds);
//! - (b) `dequant_row` is bit-equal to `to_float` on every weight row of the type; a row with `dequant: false` keeps its named
//!   `QuantError::Unsupported`;
//! - (c) the host activation encoder (`qdot::quantize_col`, and its scalar twin byte for byte) is bit-equal to `from_float`
//!   of the row's `vec_dot_type` when the row's `host_act` is ggml's; where it is not ggml's, the clause names why and holds
//!   the encoder to the half-step of its own scale instead;
//! - (d) qdot's scalar and AVX2 dots (and the dispatching `dot_row`) against `vec_dot`.
//!
//! Coverage runs both ways, and neither is a skip. A QROWS row ggml's table does not name is red by name. A type ggml's table
//! names that has no QROWS row is red by name in all four clauses. A type the table states `skip` has no case to equal, so
//! its row must say the same: `dequant: false` and `host_act: None`, refused by name in (b), (c) and (d). A kernel-less row
//! (`qdot::supports` false) keeps the clauses it has: (b)'s refusal or equality, (c)'s named refusal of an activation form and
//! (d)'s `UnsupportedType`.
//!
//! # The dot band, clause (d)
//!
//! The weight decode and the integer products of every kernel are exact (`crates/gguf/src/quant.rs` header: each product of a
//! scale and a code fits the f32 significand, and the codes of both sides are integers), so two kernels that sum in the same
//! order write the same bits, and two that do not differ only by f32 rounding. Let `w` be the oracle's `to_float` row, `x~` an
//! activation as its kernel reads it (code times scale), `V = sum w_j x~_j` in f64 and `V_abs = sum |w_j x~_j|`.
//!
//! - A kernel forms at most one f32 term per 32 weight values (`T0 = k / 32`), each from at most three f32 roundings (the
//!   scale product, the integer-to-float convert, the fused add), and adds them in some order through eight lanes and a lane
//!   reduction (at most seven more additions); an f32 float row (`dot_f32`) adds `k / 8` terms per lane. So the result has
//!   `T = T0 + 8` roundings in its chain, and Higham's bound for a sum of rounded terms gives
//!   `|r - V| <= (T + 2) u S` with `u = 2^-24` and `S` the sum of the absolute values of the f32 terms.
//! - `S` is `V_abs` for a format whose block term is one scale times an integer sum. A format that carries a separate
//!   offset (`w = a c - b`: Q2_K, Q4_K, Q5_K, Q5_1) forms the two parts as separate f32 terms, each as large as the part. For
//!   codes `c` uniform in `[0, M]`, which is how the harness draws the random rows, the parts' sum over `|w|` is
//!   `(M/2 + b/a) / E|c - b/a|`, at most 4.24 over all offsets; `R = 8` covers it with a factor two for the fluctuation of 32
//!   draws. Every quantized row uses `R = 8`, so no per-type list exists; an f32 row uses `R = 1`.
//! - ggml's Q8_1 activation (Q5_1's `vec_dot_type`) stores the code sum in f16, and the offset term reads it: one more rounding,
//!   of relative size `2^-11`, on a part as large as `R V_abs`. Ggml's band for such a form adds `2^-11 R V_abs`.
//! - So `band(V_abs) = (T + 2) u R V_abs`, applied to ours against `V` of the activation our encoder wrote, and to ggml's
//!   `vec_dot` against `V` of the activation ggml wrote: ours and ggml's each within their band of their own `V`.
//! - Where the two activations are the same values `V` is shared and `|ours - ggml| <= band_ours + band_ggml`. Where
//!   they are not (a row whose `host_act` is not ggml's, clause (c)) the gap between the two `V` is a known number,
//!   `G = sum |w_j| |x~ours_j - x~ggml_j|`, and `|ours - ggml| <= band_ours + band_ggml + G` still holds; `G` is zero when the
//!   decoded activations are bit-equal.
//! - A type listed in [`BIT_EQUAL`] is held to bit equality on every pair: its sum order is ggml's.
//!
//! Nothing here is fitted to a run: the constants are `u`, `R`, `T` and the half-steps of the activation forms below, and a
//! type that needs more is a finding about the engine or about this derivation.

#![cfg(target_arch = "x86_64")]

mod common;

use std::sync::OnceLock;

use gguf::quant::{QuantError, dequant_row, half_to_f32, quantize_activations};
use gguf::{GgmlType, HostAct, QROWS, QRow, activation_format, ggml_type_info};
use qdot::{
    QdotError, col_bytes, dot_f32, dot_row, dot_row_avx2, dot_row_scalar, quantize_col,
    quantize_col_scalar, supports,
};
use refset::arch::quant::MAINLINE;
use refset::quant::{Case, Part, QuantSet, TypeRow};

/// The unit roundoff of f32 round-to-nearest.
const U: f64 = 1.0 / 16_777_216.0;
/// The ratio of the f32 terms' absolute sum to `V_abs` a quantized row's band allows (module header).
const R_QUANT: f64 = 8.0;
/// Additions and roundings a chain holds beyond its block terms: the lane reduction and the per-term scale products.
const CHAIN_EXTRA: usize = 8;

/// The types whose dot is bit-equal to ggml's on every pair of the set: one eight-lane accumulator, the same fused steps in the
/// same order and `hsum_float_8` closing the row, as mainline's `ggml_vec_dot_<type>_q8_K`. Every other fused type adds in
/// another order (or contracts a step the other does not) and is held to the band.
// PIN(2026-10-19): the first run of this gate (`bit_equal` equals `pairs` for exactly these three); a type here that stops
// being bit-equal is red, and a type that becomes bit-equal is listed by the change that made it so.
const BIT_EQUAL: &[&str] = &["q2_K", "iq2_xxs", "iq4_xs"];

fn set() -> &'static QuantSet {
    static SET: OnceLock<QuantSet> = OnceLock::new();
    SET.get_or_init(|| {
        QuantSet::open(&MAINLINE.path("ref-quant"), &MAINLINE).unwrap_or_else(|e| panic!("{e}"))
    })
}

/// The row's line in the set's table of ggml's types, through the reader's own lookup. A row ggml's table does not name is red
/// by the reader's message, and the clause has nothing more to check.
fn ggml_type(row: &QRow, reds: &mut Reds) -> Option<&'static TypeRow> {
    match set().type_row(row.name) {
        Ok(t) => Some(t),
        Err(e) => {
            reds.red_case(e.to_string());
            None
        }
    }
}

/// The cases of a row's type, one per row length: none for a type the table states `skip`, and at least one for any other
/// (the reader refuses a set that holds a dumped type without one).
fn cases(row: &QRow) -> Vec<&'static Case> {
    set().of_type(row.name).collect()
}

/// The reds of one clause, kept so a run names every one, not the first. A statement about a whole case comes before the
/// lines of its pairs, so the cap on lines shown never hides it.
#[derive(Default)]
struct Reds {
    case: Vec<String>,
    pairs: Vec<String>,
}

impl Reds {
    fn red(&mut self, msg: String) {
        self.pairs.push(msg);
    }

    fn red_case(&mut self, msg: String) {
        self.case.push(msg);
    }

    /// The row's reds as one message, or `None` when the clause is green.
    fn message(self, row: &QRow, clause: &str) -> Option<String> {
        const SHOWN: usize = 12;
        let n = self.case.len() + self.pairs.len();
        if n == 0 {
            return None;
        }
        let mut lines = self.case;
        lines.extend(self.pairs);
        lines.truncate(SHOWN);
        let more = if n > SHOWN {
            format!("\n... and {} more", n - SHOWN)
        } else {
            String::new()
        };
        Some(format!(
            "{} {clause}: {n} red\n{}{more}",
            row.name,
            lines.join("\n")
        ))
    }
}

/// One case's five files.
struct Loaded {
    case: &'static Case,
    blocks: Vec<u8>,
    deq: Vec<f32>,
    x: Vec<f32>,
    act: Vec<u8>,
    dot: Vec<f32>,
}

fn load(case: &'static Case) -> Loaded {
    let s = set();
    let bytes = |p| s.bytes(case, p).unwrap_or_else(|e| panic!("{e}"));
    let f32s = |p| s.f32s(case, p).unwrap_or_else(|e| panic!("{e}"));
    Loaded {
        case,
        blocks: bytes(Part::Blocks),
        deq: f32s(Part::Deq),
        x: f32s(Part::X),
        act: bytes(Part::Act),
        dot: f32s(Part::Dot),
    }
}

impl Loaded {
    fn weight_row(&self, w: usize) -> &[u8] {
        let n = self.case.row_bytes;
        &self.blocks[w * n..(w + 1) * n]
    }

    fn deq_row(&self, w: usize) -> &[f32] {
        let k = self.case.k;
        &self.deq[w * k..(w + 1) * k]
    }

    fn x_row(&self, a: usize) -> &[f32] {
        let k = self.case.k;
        &self.x[a * k..(a + 1) * k]
    }

    fn act_row(&self, a: usize) -> &[u8] {
        let n = self.case.act_row_bytes;
        &self.act[a * n..(a + 1) * n]
    }

    fn dot_at(&self, w: usize, a: usize) -> f32 {
        self.dot[w * self.case.act_rows + a]
    }
}

/// An activation form: what ggml's `vec_dot_type` or this engine's encoder writes, as the half-step bound on one element
/// and a decoder for the oracle's bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Form {
    F32,
    F16,
    /// ggml's `block_q8_K`: f32 scale, 256 codes, 16 i16 sums.
    Q8K,
    /// ggml's `block_q8_0`: f16 scale, 32 codes.
    Q8_0,
    /// ggml's `block_q8_1`: f16 scale, f16 sum, 32 codes.
    Q8_1,
    /// ik's `block_q8_2` family: bf16 scale, i16 code sum, 32 codes. Mainline has no such type.
    Q82X4,
}

impl Form {
    /// The form of an oracle case's `vec_dot_type`; one the gate cannot decode is red by name.
    fn of_vdt(name: &str) -> Form {
        match name {
            "f32" => Form::F32,
            "f16" => Form::F16,
            "q8_K" => Form::Q8K,
            "q8_0" => Form::Q8_0,
            "q8_1" => Form::Q8_1,
            other => panic!(
                "the oracle's vec_dot_type {other:?} has no decoder in this gate (add it to Form::of_vdt and Form::decode)"
            ),
        }
    }

    /// The form a row's `host_act` is, when ggml has it under that name (`None`: ggml has no such type).
    fn ggml_of_host(h: HostAct) -> Option<Form> {
        match h {
            HostAct::F32 => Some(Form::F32),
            HostAct::Q8K => Some(Form::Q8K),
            HostAct::Q8_0 => Some(Form::Q8_0),
            HostAct::Q8_2X4 | HostAct::None => None,
        }
    }

    /// The oracle's name of the form, for a message.
    fn name(self) -> &'static str {
        match self {
            Form::F32 => "f32",
            Form::F16 => "f16",
            Form::Q8K => "q8_K",
            Form::Q8_0 => "q8_0",
            Form::Q8_1 => "q8_1",
            Form::Q82X4 => "q8_2_x4",
        }
    }

    /// Values sharing one scale: the amax block.
    fn block(self) -> usize {
        match self {
            Form::Q8K => 256,
            Form::Q8_0 | Form::Q8_1 | Form::Q82X4 => 32,
            Form::F32 | Form::F16 => 1,
        }
    }

    /// The distance from a value to its decoded code, in units of the block's `d = amax / 127`, at most:
    /// - Q8_K: the product `x * iscale` rounds in f32 (`2^-17` at 127) and the code is the nearest integer, so half a step plus
    ///   that, and a final `2^-12` for the scale's own f32 division;
    /// - Q8_0 and Q8_1: the scale is stored as f16 (relative error `2^-11`), which moves each of up to 127 steps by that much,
    ///   so half a step plus `127 * 2^-11`;
    /// - Q8_2: the scale is stored as bf16 (relative error `2^-8`) and the code multiplier is its exact inverse, so half a
    ///   step of the stored scale, which is at most `1 + 2^-8` of `d`; the top element's code is at most 127.5 and saturates
    ///   at 127, within the same half step.
    fn half_step(self) -> f64 {
        let slack = 2f64.powi(-12);
        match self {
            Form::Q8K => 0.5 + slack,
            Form::Q8_0 | Form::Q8_1 => 0.5 + 127.0 * 2f64.powi(-11) + slack,
            Form::Q82X4 => (0.5 + slack) * (1.0 + 2f64.powi(-8)),
            Form::F32 | Form::F16 => 0.0,
        }
    }

    /// The relative error of a sum the form stores in f16 and a dot reads back (Q8_1's `s`); zero for a form without one.
    fn stored_sum_rel(self) -> f64 {
        match self {
            Form::Q8_1 => 2f64.powi(-11),
            _ => 0.0,
        }
    }

    /// The bound on `|x_j - decoded_j|` for each element of `x`: the amax block's half step, or f16's relative `2^-11`.
    fn bounds(self, x: &[f32]) -> Vec<f64> {
        match self {
            Form::F32 => vec![0.0; x.len()],
            Form::F16 => x
                .iter()
                .map(|v| f64::from(v.abs()) * 2f64.powi(-11))
                .collect(),
            _ => x
                .chunks(self.block())
                .flat_map(|b| {
                    let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let step = f64::from(amax) / 127.0 * self.half_step();
                    std::iter::repeat_n(step, b.len())
                })
                .collect(),
        }
    }

    /// The `k` values of one activation row's bytes (the oracle's, or the engine's encoder's for `Q82X4`), each as the f32
    /// product of the scale and the code (the convention of the engine's own round trip, `quantize_activations`), so two
    /// equal activations decode to equal values.
    fn decode(self, bytes: &[u8], k: usize) -> Vec<f64> {
        let half = |b: &[u8]| half_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let scaled = |d: f32, q: u8| f64::from(d * f32::from(q as i8));
        let values: Vec<f64> = match self {
            // The weight types of the same names: their dequantizer is the decoder (clause (b) holds it to ggml's `to_float`).
            Form::F32 | Form::F16 | Form::Q8_0 => {
                let ty = GgmlType::from_u32(
                    set()
                        .type_row(self.name())
                        .unwrap_or_else(|e| panic!("{e}"))
                        .id,
                );
                let mut out = vec![0.0; k];
                dequant_row(ty, bytes, &mut out)
                    .unwrap_or_else(|e| panic!("{} activation row: {e}", self.name()));
                out.into_iter().map(f64::from).collect()
            }
            Form::Q8_1 => bytes
                .as_chunks::<36>()
                .0
                .iter()
                .flat_map(|b| b[4..].iter().map(move |&q| scaled(half(b), q)))
                .collect(),
            Form::Q8K => bytes
                .as_chunks::<292>()
                .0
                .iter()
                .flat_map(|b| {
                    let d = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    b[4..260].iter().map(move |&q| scaled(d, q))
                })
                .collect(),
            // The encoder's own bytes (the oracle holds none: mainline has no such type), read by the qdot gates' one reader.
            // A bf16 scale times an i8 code is exact in f32, so narrowing the f64 product is the f32 product the others use.
            Form::Q82X4 => common::restore_q82x4(bytes)
                .into_iter()
                .map(|v| f64::from(v as f32))
                .collect(),
        };
        assert_eq!(
            values.len(),
            k,
            "{} activation row of {} values",
            self.name(),
            values.len()
        );
        values
    }
}

/// The activation values our encoder's bytes stand for: the engine's own reference of the rule (`quantize_activations`).
fn ours_values(ty: GgmlType, x: &[f32]) -> Vec<f64> {
    let mut out = vec![0f32; x.len()];
    quantize_activations(ty, x, &mut out).unwrap_or_else(|e| panic!("{ty}: {e}"));
    out.iter().map(|&v| f64::from(v)).collect()
}

/// (a) The row's sizes are the sizes the harness wrote for the type. The reader holds every case equal to its type's row, so
/// the row equal to the type's row is the row equal to every case; a type with no case is held the same way.
fn sizes(row: &QRow) -> Reds {
    let mut reds = Reds::default();
    let Some(t) = ggml_type(row, &mut reds) else {
        return reds;
    };
    let at = &row.name;
    if t.id != row.ty.as_u32() {
        reds.red(format!(
            "{at}: ggml type id {}, the row's tag is {}",
            t.id,
            row.ty.as_u32()
        ));
    }
    if GgmlType::from_u32(t.id) != row.ty {
        reds.red(format!(
            "{at}: GgmlType::from_u32({}) is {:?}, not the row's {:?}",
            t.id,
            GgmlType::from_u32(t.id),
            row.ty
        ));
    }
    if t.blck as u64 != row.blck {
        reds.red(format!(
            "{at}: blck_size {}, the row's blck {}",
            t.blck, row.blck
        ));
    }
    if t.bytes as u64 != row.bytes {
        reds.red(format!(
            "{at}: type_size {}, the row's bytes {}",
            t.bytes, row.bytes
        ));
    }
    if ggml_type_info(t.id) != Some((row.name, row.blck, row.bytes)) {
        reds.red(format!(
            "{at}: ggml_type_info({}) is {:?}, not the row's",
            t.id,
            ggml_type_info(t.id)
        ));
    }
    reds
}

/// (b) `dequant_row` is bit-equal to `to_float`; a sized-only row is refused by name.
fn dequant(row: &QRow) -> Reds {
    let mut reds = Reds::default();
    let Some(t) = ggml_type(row, &mut reds) else {
        return reds;
    };
    if let Some(why) = &t.skip {
        // ggml has no to_float for the type: the set holds nothing to equal, so the row must hold no decoder either.
        if row.dequant {
            reds.red_case(format!(
                "{}: ggml states '{why}' for the type, so the set holds no to_float, but the row says dequant: true",
                row.name
            ));
        }
        let got = dequant_row(row.ty, &[], &mut []);
        if got != Err(QuantError::Unsupported(row.ty)) {
            reds.red_case(format!(
                "{}: a skipped type gives {got:?}, not Unsupported({})",
                row.name, row.name
            ));
        }
        return reds;
    }
    for c in cases(row) {
        let d = load(c);
        let at = &c.name;
        let mut out = vec![0f32; c.k];
        for w in 0..c.rows() {
            let got = dequant_row(row.ty, d.weight_row(w), &mut out);
            if !row.dequant {
                if got != Err(QuantError::Unsupported(row.ty)) {
                    reds.red(format!(
                        "{at} row {w}: a row without a decoder gives {got:?}, not Unsupported({})",
                        row.name
                    ));
                }
                continue;
            }
            if let Err(e) = got {
                reds.red(format!("{at} row {w}: {e}"));
                continue;
            }
            let want = d.deq_row(w);
            let bad = out
                .iter()
                .zip(want)
                .enumerate()
                .filter(|(_, (g, t))| g.to_bits() != t.to_bits())
                .count();
            if bad > 0 {
                let i = out
                    .iter()
                    .zip(want)
                    .position(|(g, t)| g.to_bits() != t.to_bits())
                    .unwrap_or(0);
                reds.red(format!(
                    "{at} row {w}: {bad} of {} values differ from to_float; first at {i}: ours {:#010x} ({}), ggml {:#010x} ({})",
                    c.k, out[i].to_bits(), out[i], want[i].to_bits(), want[i]
                ));
            }
        }
    }
    reds
}

/// A Q8_K column: ours (296 bytes a block, f32 `d` at 0, codes at 8, sums at 264) against ggml's (292 bytes, codes at 4,
/// sums at 260), field by field. `None` when equal.
fn q8k_gap(ours: &[u8], ggml: &[u8]) -> Option<String> {
    for (b, (o, g)) in ours
        .as_chunks::<296>()
        .0
        .iter()
        .zip(ggml.as_chunks::<292>().0)
        .enumerate()
    {
        if o[0..4] != g[0..4] {
            return Some(format!("block {b}: d {:?} vs {:?}", &o[0..4], &g[0..4]));
        }
        if let Some(i) = (0..256).find(|&i| o[8 + i] != g[4 + i]) {
            return Some(format!(
                "block {b}: code {i} is {} vs {}",
                o[8 + i] as i8,
                g[4 + i] as i8
            ));
        }
        if o[264..296] != g[260..292] {
            return Some(format!("block {b}: the 16 code sums differ"));
        }
    }
    None
}

/// A row no CPU matmul takes here (`host_act` None): the activation form is refused by name, not guessed, and qdot has no kernel.
fn refuses_activation(row: &QRow, at: &str, reds: &mut Reds) {
    let af = activation_format(row.ty);
    if af != Err(QuantError::NoActivationFormat(row.ty)) {
        reds.red(format!(
            "{at}: host_act None but activation_format gives {af:?}"
        ));
    }
    if supports(row.ty) {
        reds.red(format!("{at}: host_act None but qdot has a kernel for it"));
    }
}

/// (c) The host activation encoder against ggml's `from_float(vec_dot_type)`.
fn act(row: &QRow) -> Reds {
    let mut reds = Reds::default();
    let Some(t) = ggml_type(row, &mut reds) else {
        return reds;
    };
    let af = activation_format(row.ty);
    if let Some(why) = &t.skip {
        // ggml has no vec_dot for the type, so no vec_dot_type to be equal to: the row must take no activation either.
        if row.host_act != HostAct::None {
            reds.red_case(format!(
                "{}: ggml states '{why}' for the type, so it has no vec_dot_type, but the row's host_act is {:?}",
                row.name, row.host_act
            ));
        }
        refuses_activation(row, row.name, &mut reds);
        return reds;
    }
    for c in cases(row) {
        let d = load(c);
        let at = &c.name;
        match row.host_act {
            HostAct::None => refuses_activation(row, at, &mut reds),
            HostAct::F32 => {
                if af != Ok(None) {
                    reds.red(format!(
                        "{at}: host_act F32 but activation_format gives {af:?}"
                    ));
                }
                for a in 0..c.act_rows {
                    let x = d.x_row(a);
                    let kept = ours_values(row.ty, x);
                    if kept.iter().zip(x).any(|(k, v)| *k != f64::from(*v)) {
                        reds.red(format!(
                            "{at} act row {a}: the engine's rule changes an f32 activation"
                        ));
                    }
                    // ggml's f32 activation is the row copied; for a weight whose vec_dot_type is another form (f16 for f16)
                    // ggml rounds the activation to it and the engine does not: the difference is clause (d)'s gap G.
                    if c.vdt_name == "f32"
                        && d.act_row(a)
                            != d.x_row(a)
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<u8>>()
                    {
                        reds.red(format!(
                            "{at} act row {a}: ggml's f32 activation is not the row itself"
                        ));
                    }
                }
            }
            HostAct::Q8K | HostAct::Q8_0 | HostAct::Q8_2X4 => {
                if af.is_err() || !supports(row.ty) {
                    reds.red(format!(
                        "{at}: host_act {:?} but activation_format gives {af:?}, supports {}",
                        row.host_act,
                        supports(row.ty)
                    ));
                    continue;
                }
                let n = col_bytes(row.ty, c.k);
                let theirs = Form::of_vdt(&c.vdt_name);
                let mine = Form::ggml_of_host(row.host_act);
                for a in 0..c.act_rows {
                    let x = d.x_row(a);
                    let (mut avx, mut sca) = (vec![0u8; n], vec![0u8; n]);
                    quantize_col(row.ty, x, &mut avx);
                    quantize_col_scalar(row.ty, x, &mut sca);
                    if avx != sca {
                        reds.red(format!("{at} act row {a}: the AVX2 encoder and its scalar twin write different bytes"));
                    }
                    match mine {
                        Some(f) if f == theirs => {
                            let gap = if f == Form::Q8K {
                                q8k_gap(&avx, d.act_row(a))
                            } else {
                                (avx != d.act_row(a)).then(|| "the bytes differ".to_string())
                            };
                            if let Some(g) = gap {
                                reds.red(format!("{at} act row {a}: not bit-equal to ggml's {}: {g}", f.name()));
                            }
                        }
                        Some(f) => reds.red(format!(
                            "{at}: the row's host_act is ggml's {} but the oracle's vec_dot_type for {} is {}",
                            f.name(), row.name, theirs.name()
                        )),
                        None => {
                            // Not ggml's: ik's block_q8_2 family (bf16 scale, i16 code sum, four-block interleave); mainline
                            // pairs this weight with `theirs`. The encoder is held to the half step of its own scale per
                            // element, and the difference to ggml's values enters (d) as G. The values are read back from the
                            // bytes the AVX2 encoder wrote, so a defect both encoders share (the bf16 scale, the multiplier)
                            // is seen here and not only in the round trip they are both checked against in (d).
                            let rt = Form::Q82X4.decode(&avx, c.k);
                            let bound = Form::Q82X4.bounds(x);
                            for (j, ((r, v), b)) in rt.iter().zip(x).zip(&bound).enumerate() {
                                if !r.is_finite() || (r - f64::from(*v)).abs() > *b {
                                    reds.red(format!("{at} act row {a}: element {j} is {r} for {v}, over the half step {b}"));
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    reds
}

/// `(V, V_abs)` of a weight row against activation values, in f64.
fn sums(w: &[f32], x: &[f64]) -> (f64, f64) {
    w.iter().zip(x).fold((0.0, 0.0), |(v, va), (w, x)| {
        let p = f64::from(*w) * x;
        (v + p, va + p.abs())
    })
}

/// The ratio of `diff` to `band`, infinite when only the band is zero.
fn ratio(diff: f64, band: f64) -> f64 {
    if diff == 0.0 {
        0.0
    } else if band == 0.0 {
        f64::INFINITY
    } else {
        diff / band
    }
}

/// A qdot dot entry: weight type, weight row, activation column, `k`.
type DotFn = fn(GgmlType, &[u8], &[u8], usize) -> Result<f32, QdotError>;

/// The results of qdot's dots for one pair: the AVX2 kernel's, and the scalar and dispatching entries it must equal bit
/// for bit (none for the f32 row, whose one entry picks its own path).
struct Ours {
    scalar: Option<f32>,
    avx2: f32,
    dispatch: Option<f32>,
}

fn usize_of(v: u64) -> usize {
    usize::try_from(v).unwrap_or_else(|_| panic!("{v} does not fit usize"))
}

/// A row qdot has no kernel for gives `QdotError::UnsupportedType` from every entry.
fn refuses_every_entry(row: &QRow, w: &[u8], k: usize, reds: &mut Reds) {
    let refused = |r: Result<f32, QdotError>| r == Err(QdotError::UnsupportedType(row.ty));
    if !refused(dot_row(row.ty, w, &[], k))
        || !refused(dot_row_scalar(row.ty, w, &[], k))
        || !refused(dot_row_avx2(row.ty, w, &[], k))
    {
        reds.red(format!(
            "{}: a row without a kernel does not give QdotError::UnsupportedType from every entry",
            row.name
        ));
    }
}

/// (d) The scalar and AVX2 dots against `vec_dot`.
fn dot(row: &QRow) -> Reds {
    let mut reds = Reds::default();
    let Some(t) = ggml_type(row, &mut reds) else {
        return reds;
    };
    let fused = supports(row.ty);
    let f32_row = row.ty == GgmlType::F32;
    let engine_has_dot = matches!(row.host_act, HostAct::Q8K | HostAct::Q8_2X4 | HostAct::Q8_0);
    if fused != engine_has_dot {
        reds.red(format!(
            "{}: host_act {:?} and qdot::supports {fused} disagree about a fused dot",
            row.name, row.host_act
        ));
    }
    if let Some(why) = &t.skip {
        // ggml has no vec_dot for the type: the set holds no dot to equal, so the row must hold no kernel either.
        if engine_has_dot {
            reds.red_case(format!(
                "{}: ggml states '{why}' for the type, so it has no vec_dot, but the row's host_act is {:?}",
                row.name, row.host_act
            ));
        }
        let w = vec![0u8; usize_of(row.bytes)];
        refuses_every_entry(row, &w, usize_of(row.blck), &mut reds);
        return reds;
    }
    let cs = cases(row);
    if !fused && !f32_row {
        // The engine dequantizes this row and runs its f32 path; qdot has no dot, and says so by name.
        let d = load(cs[0]);
        refuses_every_entry(row, d.weight_row(0), cs[0].k, &mut reds);
        return reds;
    }
    for c in cs {
        let d = load(c);
        let k = c.k;
        let theirs = Form::of_vdt(&c.vdt_name);
        let (terms, r_factor) = if f32_row {
            (k / 8 + CHAIN_EXTRA, 1.0)
        } else {
            (k / 32 + CHAIN_EXTRA, R_QUANT)
        };
        let band = |v_abs: f64| (terms as f64 + 2.0) * U * r_factor * v_abs;
        let (mut pairs, mut bit_equal) = (0usize, 0usize);
        let (mut worst_o, mut worst_g, mut worst_d) = (0f64, 0f64, 0f64);
        for a in 0..c.act_rows {
            let x = d.x_row(a);
            let xo = ours_values(row.ty, x);
            let xm = theirs.decode(d.act_row(a), k);
            // The oracle's own activation is within its half step of x: the decoder and the bound are the gate's, so check them.
            for (j, ((m, v), b)) in xm.iter().zip(x).zip(theirs.bounds(x)).enumerate() {
                if (m - f64::from(*v)).abs() > b {
                    reds.red(format!(
                        "{} act row {a}: the oracle's {} activation, element {j}, is {m} for {v}",
                        c.name,
                        theirs.name()
                    ));
                    break;
                }
            }
            let col = if f32_row {
                Vec::new()
            } else {
                let mut col = vec![0u8; col_bytes(row.ty, k)];
                quantize_col(row.ty, x, &mut col);
                col
            };
            for w in 0..c.rows() {
                let wd = d.deq_row(w);
                let (v_o, va_o) = sums(wd, &xo);
                let (v_m, va_m) = sums(wd, &xm);
                let gap: f64 = wd
                    .iter()
                    .zip(xo.iter().zip(&xm))
                    .map(|(w, (o, m))| f64::from(w.abs()) * (o - m).abs())
                    .sum();
                let ours = if f32_row {
                    let r = dot_f32(d.weight_row(w), x)
                        .unwrap_or_else(|| panic!("{}: dot_f32 takes k = {k}", c.name));
                    Ours {
                        scalar: None,
                        avx2: r,
                        dispatch: None,
                    }
                } else {
                    let call = |f: DotFn| {
                        f(row.ty, d.weight_row(w), &col, k)
                            .unwrap_or_else(|e| panic!("{} row {w}: {e}", c.name))
                    };
                    Ours {
                        scalar: Some(call(dot_row_scalar)),
                        avx2: call(dot_row_avx2),
                        dispatch: Some(call(dot_row)),
                    }
                };
                let g = d.dot_at(w, a);
                let r = ours.avx2;
                pairs += 1;
                bit_equal += usize::from(r.to_bits() == g.to_bits());
                let (b_o, b_m) = (
                    band(va_o),
                    band(va_m) + theirs.stored_sum_rel() * r_factor * va_m,
                );
                let (e_o, e_m, e_d) = (
                    (f64::from(r) - v_o).abs(),
                    (f64::from(g) - v_m).abs(),
                    (f64::from(r) - f64::from(g)).abs(),
                );
                worst_o = worst_o.max(ratio(e_o, b_o));
                worst_g = worst_g.max(ratio(e_m, b_m));
                worst_d = worst_d.max(ratio(e_d, b_o + b_m + gap));
                let at = format!("{} w{w} a{a}", c.name);
                // A NaN result makes every comparison below false, so it is refused by name first.
                if ![e_o, e_m, e_d, b_o, b_m, gap]
                    .into_iter()
                    .all(f64::is_finite)
                {
                    reds.red(format!(
                        "{at}: a non-finite dot error or band: ours {r}, ggml {g}, errors {e_o:e} {e_m:e} {e_d:e}, bands {b_o:e} {b_m:e}, activation gap G {gap:e}"
                    ));
                }
                if e_o > b_o {
                    reds.red(format!(
                        "{at}: ours {r} is {e_o:e} from its own exact {v_o}, band {b_o:e}"
                    ));
                }
                if e_m > b_m {
                    reds.red(format!("{at}: ggml's {g} is {e_m:e} from its own exact {v_m}, band {b_m:e} (the oracle or this gate's decoder)"));
                }
                if e_d > b_o + b_m + gap {
                    reds.red(format!("{at}: ours {r} vs ggml {g}: {e_d:e} apart, band {:e} (activation gap G {gap:e})", b_o + b_m + gap));
                }
                for (name, other) in [("scalar", ours.scalar), ("dot_row", ours.dispatch)] {
                    if let Some(o) = other
                        && o.to_bits() != r.to_bits()
                    {
                        reds.red(format!(
                            "{at}: {name} {o} and AVX2 {r} are not the same bits"
                        ));
                    }
                }
            }
        }
        eprintln!(
            "quant dot {:<12} pairs={pairs} bit_equal={bit_equal} ours/own-band={worst_o:.3e} ggml/own-band={worst_g:.3e} ours-ggml/band={worst_d:.3e}",
            c.name
        );
        if BIT_EQUAL.contains(&row.name) && bit_equal != pairs {
            reds.red_case(format!(
                "{}: listed in BIT_EQUAL, but {bit_equal} of {pairs} pairs are bit-equal to ggml",
                c.name
            ));
        }
    }
    reds
}

/// What a run takes: QROWS rows, and the types of ggml's table that have no QROWS row.
struct Selection {
    rows: Vec<&'static QRow>,
    uncovered: Vec<&'static TypeRow>,
}

/// Everything, or what `QUANT_ONLY` names (ggml's type names, comma-separated). A name that is neither a QROWS row nor a type
/// of ggml's table in the set, and a value that is not UTF-8, are named errors, never an empty or a full run.
fn selected() -> Selection {
    let uncovered: Vec<&'static TypeRow> = set()
        .types
        .iter()
        .filter(|t| QROWS.iter().all(|r| r.name != t.name))
        .collect();
    let only = match std::env::var("QUANT_ONLY") {
        Ok(only) if !only.is_empty() => only,
        Ok(_) | Err(std::env::VarError::NotPresent) => {
            return Selection {
                rows: QROWS.iter().collect(),
                uncovered,
            };
        }
        Err(e @ std::env::VarError::NotUnicode(_)) => {
            panic!("QUANT_ONLY is not valid UTF-8, so it names no type of ggml's table: {e}")
        }
    };
    let mut sel = Selection {
        rows: Vec::new(),
        uncovered: Vec::new(),
    };
    for name in only.split(',') {
        if let Some(row) = QROWS.iter().find(|r| r.name == name) {
            sel.rows.push(row);
        } else if let Some(t) = uncovered.iter().find(|t| t.name == name) {
            sel.uncovered.push(t);
        } else {
            let names: Vec<&str> = QROWS
                .iter()
                .map(|r| r.name)
                .chain(uncovered.iter().map(|t| t.name.as_str()))
                .collect();
            panic!(
                "QUANT_ONLY names {name:?}, which is neither a QROWS row nor a type of ggml's table: {names:?}"
            );
        }
    }
    sel
}

/// A type ggml's table names that QROWS has no row for: red by name, with what the engine's sized-only table answers.
fn uncovered_message(t: &TypeRow, clause: &str) -> String {
    let state = t
        .skip
        .as_ref()
        .map_or_else(|| "dumped".to_string(), |why| format!("skipped: {why}"));
    format!(
        "{} {clause}: ggml's table names this type (id {}, {} values in {} bytes, {state}) and QROWS has no row for it; \
         ggml_type_info({}) answers {:?}",
        t.name,
        t.id,
        t.blck,
        t.bytes,
        t.id,
        ggml_type_info(t.id)
    )
}

/// One clause over the selected rows. A row that panics inside a kernel or a reader is red under its name, and the other
/// rows still run; the test ends red with every red row's message.
fn gate(clause: &str, check: fn(&QRow) -> Reds) {
    let sel = selected();
    let mut reds: Vec<String> = sel
        .rows
        .iter()
        .filter_map(
            |row| match std::panic::catch_unwind(|| check(row).message(row, clause)) {
                Ok(msg) => msg,
                Err(payload) => {
                    let what = payload
                        .downcast_ref::<String>()
                        .map(String::as_str)
                        .or_else(|| payload.downcast_ref::<&str>().copied())
                        .unwrap_or("a panic with no message");
                    Some(format!("{} {clause}: panicked: {what}", row.name))
                }
            },
        )
        .collect();
    reds.extend(sel.uncovered.iter().map(|t| uncovered_message(t, clause)));
    assert!(
        reds.is_empty(),
        "{clause}: {} of {} rows red\n{}",
        reds.len(),
        sel.rows.len() + sel.uncovered.len(),
        reds.join("\n")
    );
}

#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref-quant (just dump-ref-quant)"]
fn hw_quant_sizes() {
    gate("(a) sizes", sizes);
}

#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref-quant (just dump-ref-quant)"]
fn hw_quant_dequant() {
    gate("(b) dequant", dequant);
}

#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref-quant (just dump-ref-quant)"]
fn hw_quant_act() {
    gate("(c) activation", act);
}

#[test]
#[ignore = "hw: needs the box and $BLOOMERY_DATA/ref-quant (just dump-ref-quant)"]
fn hw_quant_dot() {
    gate("(d) dot", dot);
}
