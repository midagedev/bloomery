//! Block / end-to-end oracle harness (docs/gpu-design.md decision 3,
//! block layer): compare a whole forward pass, tap by tap, against the ik
//! dump sets — our engine against the CUDA set (`ref_cuda_v2`), or one
//! oracle against the other. The kernel gates judge single ops against
//! exact references;
//! this is the layer where quantization noise is the signal, and the
//! question is "which op left its band first".
//!
//! Taps are fixed lists per block kind, in forward order, verified
//! against the manifest (type, op, shape template) before any number is
//! read — a tap absent from the dump is an error naming it. View taps are
//! read through `ref_tensor_logical_in`: the block layer consumes the
//! LOGICAL tensor, never the flat VIEW file. The last block and the head
//! run their FFN on the last token only (`ffn_inp-26` onward are [.,1]);
//! those taps carry `last_token_only` and compare the dump's single
//! column against the engine's last token column.
//!
//! `gate_block` is the instrument self-check: a dump against itself is
//! exactly 0 everywhere, CPU-vs-CUDA reproduces the measured distance
//! between ik's own backends, and a corrupted mid-chain tensor is named
//! as the first divergence. The engine block gate pins its bands from
//! the same tables.

use crate::{
    GateError, RefRow, find_ref_row_in, max_rel_err, ref_tensor_logical_in, ref_tensor_of_in,
    view_flat,
};
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

/// The dump's prompt length; every tap's token axis is checked against
/// it (or against 1 for `last_token_only` taps).
pub const M_TOKENS: usize = 6;

// Architecture constants of the model the dump sets were made for
// (DeepSeek-V2-Lite): hidden 2048, MLA latent 512 + rope 64, 16 heads,
// q rows 16*192 with a 128-wide nope head, router 64 experts / top-6,
// vocab 102400. Hardcoded like the kernel gates' shape checks — the
// manifest verification pins them against the dump.
const HIDDEN: u64 = 2048;
const LATENT: u64 = 512;
const ROPE: u64 = 64;
const N_HEADS: u64 = 16;
const Q_ROWS: u64 = 16 * 192;
const NOPE: u64 = 128;
const KQ_HEAD: u64 = 192;
const KV_WIDTH: u64 = LATENT + ROPE;
const N_USED: u64 = 6;
const VOCAB: u64 = 102400;

/// One observable in the forward order of a block. `rank` (declaration
/// order) is the forward order `assert_within` reports the first
/// violation in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TapKind {
    AttnNorm,
    Q,
    KvRopeCompressed,
    QRope,
    KRope,
    KvCompressed,
    KqvCompressed,
    KqvOut,
    FfnInp,
    FfnNorm,
    FfnMoeLogits,
    FfnMoeWeights,
    FfnMoeOut,
    FfnShexp,
    FfnOut,
    LOut,
    ResultNorm,
    ResultOutput,
}

/// Every kind in forward order — the ranking key of the first-divergence
/// report.
const ALL_KINDS: [TapKind; 18] = [
    TapKind::AttnNorm,
    TapKind::Q,
    TapKind::KvRopeCompressed,
    TapKind::QRope,
    TapKind::KRope,
    TapKind::KvCompressed,
    TapKind::KqvCompressed,
    TapKind::KqvOut,
    TapKind::FfnInp,
    TapKind::FfnNorm,
    TapKind::FfnMoeLogits,
    TapKind::FfnMoeWeights,
    TapKind::FfnMoeOut,
    TapKind::FfnShexp,
    TapKind::FfnOut,
    TapKind::LOut,
    TapKind::ResultNorm,
    TapKind::ResultOutput,
];

/// The manifest contract of a tap kind: manifest `ne` with the token axis
/// set to 1 (multiply that axis by the compared token count for the
/// check), the axis index, the producing op, and the occurrence.
struct TapShape {
    base: [u64; 4],
    token_axis: usize,
    op: &'static str,
    occurrence: u32,
}

impl TapKind {
    fn shape(self) -> TapShape {
        use TapKind::*;
        match self {
            AttnNorm => TapShape {
                base: [HIDDEN, 1, 1, 1],
                token_axis: 1,
                op: "FUSED_RMS_NORM",
                occurrence: 0,
            },
            Q => TapShape {
                base: [Q_ROWS, 1, 1, 1],
                token_axis: 1,
                op: "MUL_MAT",
                occurrence: 0,
            },
            KvRopeCompressed => TapShape {
                base: [KV_WIDTH, 1, 1, 1],
                token_axis: 1,
                op: "MUL_MAT",
                occurrence: 0,
            },
            // Post-ROPE / post-norm: occurrence 1; occurrence 0 is the VIEW
            // of the base matmul and is only read through its logical twin.
            QRope => TapShape {
                base: [ROPE, N_HEADS, 1, 1],
                token_axis: 2,
                op: "ROPE",
                occurrence: 1,
            },
            KRope => TapShape {
                base: [ROPE, 1, 1, 1],
                token_axis: 2,
                op: "ROPE",
                occurrence: 1,
            },
            KvCompressed => TapShape {
                base: [LATENT, 1, 1, 1],
                token_axis: 1,
                op: "FUSED_RMS_NORM",
                occurrence: 1,
            },
            KqvCompressed => TapShape {
                base: [LATENT, N_HEADS, 1, 1],
                token_axis: 2,
                op: "FLASH_ATTN_EXT",
                occurrence: 0,
            },
            KqvOut => TapShape {
                base: [HIDDEN, 1, 1, 1],
                token_axis: 1,
                op: "MUL_MAT",
                occurrence: 0,
            },
            FfnInp => TapShape {
                base: [HIDDEN, 1, 1, 1],
                token_axis: 1,
                op: "ADD",
                occurrence: 0,
            },
            FfnNorm => TapShape {
                base: [HIDDEN, 1, 1, 1],
                token_axis: 1,
                op: "FUSED_RMS_NORM",
                occurrence: 0,
            },
            FfnMoeLogits => TapShape {
                base: [64, 1, 1, 1],
                token_axis: 1,
                op: "MUL_MAT",
                occurrence: 0,
            },
            FfnMoeWeights => TapShape {
                base: [1, N_USED, 1, 1],
                token_axis: 2,
                op: "GET_ROWS",
                occurrence: 0,
            },
            FfnMoeOut => TapShape {
                base: [HIDDEN, 1, 1, 1],
                token_axis: 1,
                op: "MUL_MULTI_ADD",
                occurrence: 0,
            },
            FfnShexp => TapShape {
                base: [HIDDEN, 1, 1, 1],
                token_axis: 1,
                op: "MUL_MAT",
                occurrence: 0,
            },
            FfnOut => TapShape {
                base: [HIDDEN, 1, 1, 1],
                token_axis: 1,
                op: "ADD",
                occurrence: 0,
            },
            LOut => TapShape {
                base: [HIDDEN, 1, 1, 1],
                token_axis: 1,
                op: "ADD",
                occurrence: 0,
            },
            ResultNorm => TapShape {
                base: [HIDDEN, 1, 1, 1],
                token_axis: 1,
                op: "FUSED_RMS_NORM",
                occurrence: 0,
            },
            ResultOutput => TapShape {
                base: [VOCAB, 1, 1, 1],
                token_axis: 1,
                op: "MUL_MAT",
                occurrence: 0,
            },
        }
    }

    /// The dump tensor's name for this kind at `layer` (head tensors have
    /// no layer suffix).
    pub fn tensor_name(self, layer: usize) -> String {
        let stem = match self {
            TapKind::AttnNorm => "attn_norm",
            TapKind::Q => "q",
            TapKind::KvRopeCompressed => "kv_rope_compressed",
            TapKind::QRope => "q_rope",
            TapKind::KRope => "k_rope",
            TapKind::KvCompressed => "kv_compressed",
            TapKind::KqvCompressed => "kqv_compressed",
            TapKind::KqvOut => "kqv_out",
            TapKind::FfnInp => "ffn_inp",
            TapKind::FfnNorm => "ffn_norm",
            TapKind::FfnMoeLogits => "ffn_moe_logits",
            TapKind::FfnMoeWeights => "ffn_moe_weights",
            TapKind::FfnMoeOut => "ffn_moe_out",
            TapKind::FfnShexp => "ffn_shexp",
            TapKind::FfnOut => "ffn_out",
            TapKind::LOut => "l_out",
            TapKind::ResultNorm => "result_norm",
            TapKind::ResultOutput => "result_output",
        };
        if matches!(self, TapKind::ResultNorm | TapKind::ResultOutput) {
            stem.to_string()
        } else {
            format!("{stem}-{layer}")
        }
    }

    /// Values per compared token: the product of the shape's non-token
    /// axes.
    pub fn per_token(self) -> usize {
        self.shape().base.iter().product::<u64>() as usize
    }

    /// Position in the block's forward order.
    pub fn forward_rank(self) -> usize {
        ALL_KINDS
            .iter()
            .position(|&k| k == self)
            .expect("TapKind in ALL_KINDS")
    }
}

/// The block flavours the harness knows; `Head` sorts after the last
/// block (layer value 27 with this model's 27 blocks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// Layer 0: dense FFN, no router, no shared expert.
    Dense0,
    /// Layers 1..n-1: routed MoE plus the shared expert, all tokens.
    Moe,
    /// The last layer: attention over all tokens, FFN on the last token
    /// only (its `ffn_inp` onward are [.,1] in the dump).
    Last,
    /// `result_norm` / `result_output`, the last token only.
    Head,
}

/// One comparison point: a tap kind at a layer. `occurrence` is fixed by
/// the kind (post-op tensors are occurrence 1); `last_token_only` narrows
/// the compared span to the dump's single last-token column. `op` is the
/// producing op the manifest row must carry — carried per tap because one
/// name means different graph nodes in different blocks (`ffn_out-0`, the
/// dense block's down-projection output, is MUL_MAT; the MoE blocks'
/// `ffn_out-L`, the moe+shexp sum, is ADD).
#[derive(Debug, Clone)]
pub struct Tap {
    pub kind: TapKind,
    pub layer: usize,
    pub occurrence: u32,
    pub op: &'static str,
    pub last_token_only: bool,
}

impl Tap {
    fn new(kind: TapKind, layer: usize, last_token_only: bool) -> Tap {
        Tap {
            kind,
            layer,
            occurrence: kind.shape().occurrence,
            op: kind.shape().op,
            last_token_only,
        }
    }

    /// `new` with the manifest op overridden (the dense block's ffn_out).
    fn with_op(mut self, op: &'static str) -> Tap {
        self.op = op;
        self
    }

    /// The dump tensor's name.
    pub fn name(&self) -> String {
        self.kind.tensor_name(self.layer)
    }

    /// (layer, forward rank) — the report's ordering key.
    pub fn order_key(&self) -> (usize, usize) {
        (self.layer, self.kind.forward_rank())
    }
}

/// The fixed tap list of one block, in forward order. Every entry is
/// verified against the manifest before use; a tap absent from the dump
/// is an error naming it — the lists below are the complete contract, so
/// a rename in ik's graph fails loudly here, not silently as a skipped
/// comparison.
pub fn taps(kind: BlockKind, layer: usize) -> Vec<Tap> {
    use TapKind::*;
    match kind {
        BlockKind::Dense0 => [
            AttnNorm,
            Q,
            KvRopeCompressed,
            QRope,
            KRope,
            KvCompressed,
            KqvCompressed,
            KqvOut,
            FfnInp,
            FfnNorm,
            FfnOut,
            LOut,
        ]
        .into_iter()
        .map(|k| {
            let t = Tap::new(k, layer, false);
            // ffn_out-0 is the down projection itself; the MoE blocks'
            // ffn_out-L is the moe+shexp sum. Same name, different node.
            if k == FfnOut { t.with_op("MUL_MAT") } else { t }
        })
        .collect(),
        BlockKind::Moe => [
            AttnNorm,
            Q,
            KvRopeCompressed,
            QRope,
            KRope,
            KvCompressed,
            KqvCompressed,
            KqvOut,
            FfnInp,
            FfnNorm,
            FfnMoeLogits,
            FfnMoeWeights,
            FfnMoeOut,
            FfnShexp,
            FfnOut,
            LOut,
        ]
        .into_iter()
        .map(|k| Tap::new(k, layer, false))
        .collect(),
        // The last layer's attention still runs over every token (later
        // queries need the earlier keys); from ffn_inp on the dump holds
        // the last token only.
        BlockKind::Last => [
            (AttnNorm, false),
            (Q, false),
            (KvRopeCompressed, false),
            (QRope, false),
            (KRope, false),
            (KvCompressed, false),
            (KqvCompressed, false),
            (KqvOut, false),
            (FfnInp, true),
            (FfnNorm, true),
            (FfnMoeLogits, true),
            (FfnMoeWeights, true),
            (FfnMoeOut, true),
            (FfnShexp, true),
            (FfnOut, true),
            (LOut, true),
        ]
        .into_iter()
        .map(|(k, lto)| Tap::new(k, layer, lto))
        .collect(),
        BlockKind::Head => [ResultNorm, ResultOutput]
            .into_iter()
            .map(|k| Tap::new(k, layer, true))
            .collect(),
    }
}

/// Verify a manifest row against a tap's contract: f32, the tap's op,
/// and the shape template with the token axis at the compared width. The
/// error names the tap, not just the row.
pub fn check_row(row: &RefRow, tap: &Tap, m_tokens: usize) -> Result<(), GateError> {
    let s = tap.kind.shape();
    let m_eff = if tap.last_token_only { 1 } else { m_tokens };
    let mut ne = s.base;
    ne[s.token_axis] *= m_eff as u64;
    if row.ty != "f32" || row.ne != ne || row.op != tap.op {
        return Err(format!(
            "check_row: {} is {} {} {:?}, want f32 {} {:?}{}",
            tap.name(),
            row.ty,
            row.op,
            row.ne,
            tap.op,
            ne,
            if tap.last_token_only {
                " (last-token tap)"
            } else {
                ""
            }
        )
        .into());
    }
    Ok(())
}

/// One comparison's outcome. `rel` follows `max_rel_err` semantics
/// (`max|got - ref| / max|ref|`; any non-finite side or an all-zero
/// reference is an error, not a score); `worst_index` is the first flat
/// index of that maximum inside the compared span; `n` is the span's
/// length.
#[derive(Debug, Clone)]
pub struct TapResult {
    pub tap: Tap,
    pub rel: f32,
    pub worst_index: usize,
    pub n: usize,
}

impl TapResult {
    /// The worst index rendered as (token, offset within the token) —
    /// the useful form when a span holds more than one token.
    pub fn worst(&self) -> (usize, usize) {
        let per = self.tap.kind.per_token();
        if per == 0 {
            return (0, self.worst_index);
        }
        (self.worst_index / per, self.worst_index % per)
    }
}

/// Compare `got` against the dump tensor of `tap` from the set at `dir`.
/// `got` must be exactly the compared span in the same logical order:
/// for a `last_token_only` tap that is the engine's LAST token column,
/// which is what the dump's single column holds. `m_tokens` is the
/// prompt width the engine ran.
pub fn compare_in(
    dir: &Path,
    man: &[RefRow],
    tap: &Tap,
    got: &[f32],
    m_tokens: usize,
) -> Result<TapResult, GateError> {
    let name = tap.name();
    let row = find_ref_row_in(dir, man, &name, tap.occurrence)?;
    check_row(row, tap, m_tokens)?;
    let ref_vals = ref_tensor_logical_in(dir, row)?;
    let per = tap.kind.per_token();
    let m_eff = if tap.last_token_only { 1 } else { m_tokens };
    if ref_vals.len() != per * m_eff {
        return Err(format!(
            "compare: {} holds {} values, the contract says {}x{m_eff}",
            name,
            ref_vals.len(),
            per
        )
        .into());
    }
    if got.len() != ref_vals.len() {
        return Err(format!(
            "compare: {} got {} values against {m_eff} token column(s) of {per} — \
             a last_token_only tap is compared on the engine's LAST column only",
            name,
            got.len()
        )
        .into());
    }
    let rel = max_rel_err(got, &ref_vals)?;
    // First index of the maximum |diff| (max_rel_err already rejected
    // non-finite values on both sides).
    let mut worst_index = 0usize;
    let mut worst_d = -1.0f32;
    for (i, (&g, &r)) in got.iter().zip(&ref_vals).enumerate() {
        let d = (g - r).abs();
        if d > worst_d {
            worst_d = d;
            worst_index = i;
        }
    }
    Ok(TapResult {
        tap: tap.clone(),
        rel,
        worst_index,
        n: got.len(),
    })
}

// ------------------------------------------------------------- printing

/// Print one comparison table: a line per tap with layer, tap, op,
/// compared shape (values x compared tokens), rel, worst (token:offset),
/// n, and the verdict against `bands` when given. `a` is the `got` side's
/// label, `b` the reference's.
pub fn print_table(title: &str, a: &str, b: &str, results: &[TapResult], bands: Option<&Bands>) {
    println!("table {title}: got={a} ref={b} taps={}", results.len());
    println!(
        "{:>2} {:<16} {:<15} {:>12} {:>10} {:>9} {:>8}",
        "L", "tap", "op", "span", "rel", "worst", "n"
    );
    for r in results {
        let per = r.tap.kind.per_token();
        let (wt, wo) = r.worst();
        let verdict = match bands {
            Some(bs) => match bs.band(r.tap.kind, r.tap.layer) {
                Some(band) if r.rel <= band => "ok",
                Some(_) => "VIOLATION",
                None => "NO BAND",
            },
            None => "",
        };
        println!(
            "{:>2} {:<16} {:<15} {:>7}x{:<4} {:>10.3e} {:>4}:{:<4} {:>8} {}",
            r.tap.layer,
            r.tap.kind.tensor_name(r.tap.layer),
            r.tap.op,
            per,
            r.n / per.max(1),
            r.rel,
            wt,
            wo,
            r.n,
            verdict
        );
    }
}

// --------------------------------------------------------------- bands

/// Block-layer band table: one band per (tap kind, layer). Built from
/// measured rels by a factor, or pinned by the lead from a measured
/// table; the engine gate asserts engine-vs-oracle rels against it.
#[derive(Debug, Clone, Default)]
pub struct Bands {
    map: HashMap<(TapKind, usize), f32>,
}

/// Every result that left its band, with the report's key property: the
/// FIRST tap in forward order, so a divergence is attributed to the op
/// that broke, not to everything downstream of it.
#[derive(Debug, Clone)]
pub struct Violations {
    /// (tap, rel, band) per violating result, in the results' order.
    pub all: Vec<(Tap, f32, f32)>,
    /// The lowest (layer, forward rank) among them.
    pub first: Option<(TapKind, usize)>,
}

impl fmt::Display for Violations {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} tap(s) left their band:", self.all.len())?;
        for (tap, rel, band) in &self.all {
            writeln!(
                f,
                "  L{} {} rel={rel:.3e} band={band:.3e}",
                tap.layer,
                tap.name()
            )?;
        }
        match self.first {
            Some((kind, layer)) => write!(
                f,
                "first divergence in forward order: L{layer} {}",
                kind.tensor_name(layer)
            ),
            None => write!(f, "no violations"),
        }
    }
}

impl Bands {
    /// Bands at `factor` times each result's measured rel — a harness
    /// self-check device (a rerun of the same comparison must stay inside
    /// factor x itself; a corrupted tensor must not).
    pub fn from_results(results: &[TapResult], factor: f32) -> Bands {
        Bands {
            map: results
                .iter()
                .map(|r| ((r.tap.kind, r.tap.layer), r.rel * factor))
                .collect(),
        }
    }

    /// Bands pinned by the lead from a measured table — `(kind, layer,
    /// band)` triples, each carrying its `PIN(date)` at the call site.
    pub fn pinned(table: &[(TapKind, usize, f32)]) -> Bands {
        Bands {
            map: table.iter().map(|&(k, l, b)| ((k, l), b)).collect(),
        }
    }

    pub fn band(&self, kind: TapKind, layer: usize) -> Option<f32> {
        self.map.get(&(kind, layer)).copied()
    }

    /// Every violation, plus the first in forward order. A result without
    /// a band is NOT a violation — it was never measured into the table.
    pub fn check(&self, results: &[TapResult]) -> Violations {
        let mut all = Vec::new();
        let mut first_idx: Option<usize> = None;
        for (i, r) in results.iter().enumerate() {
            let Some(band) = self.band(r.tap.kind, r.tap.layer) else {
                continue;
            };
            if r.rel > band {
                all.push((r.tap.clone(), r.rel, band));
                if first_idx.is_none_or(|j| r.tap.order_key() < results[j].tap.order_key()) {
                    first_idx = Some(i);
                }
            }
        }
        Violations {
            all,
            first: first_idx.map(|i| (results[i].tap.kind, results[i].tap.layer)),
        }
    }

    /// `check` as a Result: `Err` carries the full violation list and
    /// names the first tap in forward order.
    pub fn assert_within(&self, results: &[TapResult]) -> Result<(), Violations> {
        let v = self.check(results);
        if v.all.is_empty() { Ok(()) } else { Err(v) }
    }
}

// ------------------------------------------------------ view conventions

/// Prove the v2 dumper's logical twins for the three VIEW families the
/// block chain reads through, against their base MUL_MAT tensors — the
/// same reconstruction the kernel gates do by hand, so the block layer
/// never re-derives it: `kv_compressed-L/0` is the latent head of each
/// `kv_rope_compressed-L` row, `k_rope-L/0` the rope tail, `q_rope-L/0`
/// per (token, head) the rope slice of `q-L`. Also re-proves the PLAIN
/// files still hold the flat base memory (the pre-v2 convention the
/// kernel gates' `view_flat` assertions stand on). Errors name the
/// tensor.
pub fn check_logical_views(
    dir: &Path,
    man: &[RefRow],
    layer: usize,
    m_tokens: usize,
) -> Result<(), GateError> {
    let m = m_tokens;
    let find = |name: &str, occ: u32| find_ref_row_in(dir, man, name, occ);

    let krc_name = format!("kv_rope_compressed-{layer}");
    let krc_row = find(&krc_name, 0)?;
    if krc_row.ty != "f32" || krc_row.op != "MUL_MAT" || krc_row.ne != [KV_WIDTH, m as u64, 1, 1] {
        return Err(format!(
            "check_logical_views: {krc_name} is {} {} {:?}, want f32 MUL_MAT [{KV_WIDTH}, {m}]",
            krc_row.ty, krc_row.op, krc_row.ne
        )
        .into());
    }
    let krc = ref_tensor_logical_in(dir, krc_row)?;

    // kv_compressed-L/0: the latent head of each row.
    let kv_name = format!("kv_compressed-{layer}");
    let kv_view_row = find(&kv_name, 0)?;
    if kv_view_row.op != "VIEW" || kv_view_row.ne != [LATENT, m as u64, 1, 1] {
        return Err(format!(
            "check_logical_views: {kv_name}/0 is {} {:?}, want VIEW [{LATENT}, {m}]",
            kv_view_row.op, kv_view_row.ne
        )
        .into());
    }
    let kv_logical = ref_tensor_logical_in(dir, kv_view_row)?;
    for t in 0..m {
        for d in 0..LATENT as usize {
            let want = krc[t * KV_WIDTH as usize + d];
            let got = kv_logical[t * LATENT as usize + d];
            if got.to_bits() != want.to_bits() {
                return Err(format!(
                    "check_logical_views: {kv_name}/0 logical at (t={t}, d={d}) is {got}, \
                     want the base row head {want}"
                )
                .into());
            }
        }
    }
    let kv_plain = ref_tensor_of_in(dir, kv_view_row)?;
    view_flat(&kv_plain, &krc, 0, &format!("{kv_name}/0 plain"))?;

    // k_rope-L/0: the rope tail of each row.
    let kr_name = format!("k_rope-{layer}");
    let kr_view_row = find(&kr_name, 0)?;
    if kr_view_row.op != "VIEW" || kr_view_row.ne != [ROPE, 1, m as u64, 1] {
        return Err(format!(
            "check_logical_views: {kr_name}/0 is {} {:?}, want VIEW [{ROPE}, 1, {m}]",
            kr_view_row.op, kr_view_row.ne
        )
        .into());
    }
    let kr_logical = ref_tensor_logical_in(dir, kr_view_row)?;
    for t in 0..m {
        for d in 0..ROPE as usize {
            let want = krc[t * KV_WIDTH as usize + LATENT as usize + d];
            let got = kr_logical[t * ROPE as usize + d];
            if got.to_bits() != want.to_bits() {
                return Err(format!(
                    "check_logical_views: {kr_name}/0 logical at (t={t}, d={d}) is {got}, \
                     want the base row tail {want}"
                )
                .into());
            }
        }
    }
    let kr_plain = ref_tensor_of_in(dir, kr_view_row)?;
    view_flat(
        &kr_plain,
        &krc,
        LATENT as usize,
        &format!("{kr_name}/0 plain"),
    )?;

    // q_rope-L/0: per (token, head) the rope slice of q-L's head rows.
    let q_name = format!("q-{layer}");
    let q_row = find(&q_name, 0)?;
    if q_row.ty != "f32" || q_row.op != "MUL_MAT" || q_row.ne != [Q_ROWS, m as u64, 1, 1] {
        return Err(format!(
            "check_logical_views: {q_name}/0 is {} {} {:?}, want f32 MUL_MAT [{Q_ROWS}, {m}]",
            q_row.ty, q_row.op, q_row.ne
        )
        .into());
    }
    let q = ref_tensor_logical_in(dir, q_row)?;
    let qr_name = format!("q_rope-{layer}");
    let qr_view_row = find(&qr_name, 0)?;
    if qr_view_row.op != "VIEW" || qr_view_row.ne != [ROPE, N_HEADS, m as u64, 1] {
        return Err(format!(
            "check_logical_views: {qr_name}/0 is {} {:?}, want VIEW [{ROPE}, {N_HEADS}, {m}]",
            qr_view_row.op, qr_view_row.ne
        )
        .into());
    }
    let qr_logical = ref_tensor_logical_in(dir, qr_view_row)?;
    for t in 0..m {
        for h in 0..N_HEADS as usize {
            for d in 0..ROPE as usize {
                let want = q[t * Q_ROWS as usize + h * KQ_HEAD as usize + NOPE as usize + d];
                let got = qr_logical[(t * N_HEADS as usize + h) * ROPE as usize + d];
                if got.to_bits() != want.to_bits() {
                    return Err(format!(
                        "check_logical_views: {qr_name}/0 logical at (t={t}, h={h}, d={d}) is \
                         {got}, want the head's rope slice {want}"
                    )
                    .into());
                }
            }
        }
    }
    let qr_plain = ref_tensor_of_in(dir, qr_view_row)?;
    view_flat(&qr_plain, &q, NOPE as usize, &format!("{qr_name}/0 plain"))?;
    Ok(())
}
