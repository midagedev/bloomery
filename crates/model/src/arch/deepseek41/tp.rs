//! How tensor parallel splits each deepseek41 tensor ([`cut`]): along the dim and by the unit a
//! rank's part holds whole there, or whole on every rank. The dims are ggml's order, the GGUF
//! file's and the checkpoint map's alike (`model_hf::arch::deepseek41`), so one rule serves both;
//! the plan of the cuts is [`plan_tp`](crate::placement::tp::plan_tp)'s.
//!
//! Attention splits by head: `attn_q_b`'s rows and the sinks by head, `attn_output_a`'s rows by
//! output group and `attn_output_b` along its input by the same groups. The shared expert and
//! every routed expert split by inner width: gate and up by row, down along its input. The head
//! splits by vocabulary row. An Engram site's `engram_wkv` splits along its input by hash
//! column (a rank's columns are consecutive), the rows a rank gathers. The low-rank inputs every
//! head shares (`attn_q_a`, `attn_kv`), the compressor, the indexer, the router and its biases,
//! the hyper-connection mixes, the norms, the Engram gains and the embedding stay whole. A rank
//! holds every trunk tensor on the card but the embedding and the Engram tables, which are read
//! on the host from their files ([`card_holds`]).
//!
//! A part keeps whole blocks of the tensor's card type, and along K at least its role's kernel
//! unit ([`cut_of`]); which indices a rank holds is then [`Cut::span`]'s ([`rank_span`]).

use std::ops::Range;

use gguf::{GgmlType, Split};

use super::hparams::Hparams;
use super::kv::KvLayout;
use super::roles;
use crate::placement::tp::{self, Cut, RankNeed, TpTensor, check_rank};
use crate::placement::workstation::{CONTEXT_SELF, SCRATCH};
use crate::placement::{KvBytes, PlacementError, Role, RowTableSource};

/// The counts a cut divides by: attention heads, output groups and Engram hash columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpUnits {
    pub heads: u64,
    pub groups: u64,
    /// Rows one site gathers a token: `(max_ngram − 1) · n_head`; 0 for a model of no Engram.
    pub engram_cols: u64,
}

impl TpUnits {
    /// The units of the model `hp` describes.
    #[must_use]
    pub fn of(hp: &Hparams) -> TpUnits {
        TpUnits {
            heads: hp.n_head as u64,
            groups: hp.o_groups as u64,
            engram_cols: hp.engram.as_ref().map_or(0, |e| e.rows_per_token() as u64),
        }
    }
}

/// Whether a TP rank holds the tensor `name` on the card: every trunk tensor but the token
/// embedding and the Engram and hash tables, which stay in their files and are read by row, and
/// the tensors no step reads.
#[must_use]
pub fn card_holds(name: &str) -> bool {
    !matches!(
        roles::tensor_role(name),
        Some(
            Role::TokenEmbedding
                | Role::EngramTable
                | Role::EngramHash
                | Role::HashTable
                | Role::Unread
                | Role::Unused
        )
    )
}

/// The name past `blk.N.`, or the whole name for a model-level tensor.
fn stem_of(name: &str) -> &str {
    name.strip_prefix("blk.")
        .and_then(|r| r.split_once('.'))
        .map_or(name, |(_, stem)| stem)
}

/// Where the deepseek41 tensor `name` of `dims` splits among ranks, `None` for whole on every
/// rank. A part along a dim is that dim over the unit count `u` gives its family. A stem with no
/// rule, a split tensor of other dims than its stem has, and an extent that is not whole units
/// are refused, the reason the error.
pub fn cut(name: &str, dims: &[u64], u: &TpUnits) -> Result<Option<Cut>, PlacementError> {
    let stem = stem_of(name);
    // The dims the tensor has, the dim it splits along, and the unit count along it (0 for
    // single values).
    let (rank, dim, units) = match stem {
        "token_embd.weight"
        | "output_norm.weight"
        | "attn_norm.weight"
        | "ffn_norm.weight"
        | "attn_q_a.weight"
        | "attn_q_a_norm.weight"
        | "attn_kv.weight"
        | "attn_kv_a_norm.weight"
        | "ffn_gate_inp.weight"
        | "exp_probs_b.bias"
        | "exp_probs_b_vl.bias"
        | "engram_q.weight"
        | "engram_k.weight"
        | "engram_embd.weight" => return Ok(None),
        _ if stem.starts_with("indexer")
            || stem.starts_with("attn_compressor")
            || stem.starts_with("hc_")
            || stem.starts_with("output_hc") =>
        {
            return Ok(None);
        }
        "attn_q_b.weight" => (2, 1, u.heads),
        "attn_sinks.weight" => (1, 0, u.heads),
        "attn_output_a.weight" => (2, 1, u.groups),
        "attn_output_b.weight" => (2, 0, u.groups),
        "engram_wkv.weight" => (2, 0, u.engram_cols),
        "ffn_gate_shexp.weight" | "ffn_up_shexp.weight" | "output.weight" => (2, 1, 0),
        "ffn_down_shexp.weight" => (2, 0, 0),
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" => (3, 1, 0),
        "ffn_down_exps.weight" => (3, 0, 0),
        _ => {
            return Err(PlacementError::Tensor {
                name: name.into(),
                detail: "no tensor-parallel split rule".into(),
            });
        }
    };
    tp::cut_units(name, dims, rank, dim, (units != 0).then_some(units))
}

/// The [`cut`] of the deepseek41 tensor `name` of `dims` held in card type `ty`, with the values
/// a part keeps whole along dims 0 and 1: whole blocks of the type, along K at least the K unit
/// of its role's kernel ([`card_blocks`]); `None` for whole on every rank. A refusal of [`cut`],
/// a split tensor with no role or of a type with no block are refused by name.
pub fn cut_of(
    name: &str,
    ty: GgmlType,
    dims: &[u64],
    u: &TpUnits,
) -> Result<Option<(Cut, [u64; 2])>, PlacementError> {
    tp::cut_blocks(name, ty, cut(name, dims, u)?, roles::tensor_role(name))
}

/// Rank `rank` of `ranks`' indices of the deepseek41 tensor `name` of card type `ty` and `dims`:
/// the dim its [`cut_of`] splits and the indices along it ([`Cut::span`]); `None` for the whole
/// tensor, at one rank or where the family keeps it whole. A rank that is not one of the ranks,
/// and a refusal of [`cut_of`] or [`Cut::span`], are refused by name.
pub fn rank_span(
    name: &str,
    ty: GgmlType,
    dims: &[u64],
    u: &TpUnits,
    rank: usize,
    ranks: usize,
) -> Result<Option<(usize, Range<u64>)>, PlacementError> {
    tp::span_of(name, dims, rank, ranks, || cut_of(name, ty, dims, u))
}

/// The hash columns rank `rank` of `ranks` gathers at each Engram site: its consecutive run of
/// the `u.engram_cols` columns, the inputs of its part of `engram_wkv`. Columns that do not
/// split evenly, and a rank that is not one of the ranks, are refused by name.
pub fn engram_cols(u: &TpUnits, rank: usize, ranks: usize) -> Result<Range<u64>, PlacementError> {
    let name = "engram columns";
    check_rank(name, rank, ranks)?;
    let n = ranks as u64;
    if u.engram_cols == 0 || !u.engram_cols.is_multiple_of(n) {
        return Err(PlacementError::Tensor {
            name: name.into(),
            detail: format!("{} columns do not split into {ranks} ranks", u.engram_cols),
        });
    }
    let part = u.engram_cols / n;
    Ok(part * rank as u64..part * (rank as u64 + 1))
}

/// The rows of a site's table its hash columns `cols` read, the site's buckets of `primes` rows
/// starting at `offsets` (one each a column): from the first column's bucket to the end of the
/// last's, the table's original row ids. `None` for columns past the site's.
#[must_use]
pub fn engram_rows(primes: &[u64], offsets: &[u64], cols: Range<u64>) -> Option<Range<u64>> {
    let (first, last) = (
        usize::try_from(cols.start).ok()?,
        usize::try_from(cols.end).ok()?,
    );
    let last = last.checked_sub(1).filter(|&l| l >= first)?;
    Some(*offsets.get(first)?..offsets.get(last)? + primes.get(last)?)
}

/// The tensors of the deepseek41 header `split` (a GGUF file's or a checkpoint folder's) a rank
/// holds on the card ([`card_holds`]), each in its card type, whose resident bytes the plan
/// counts, and cut by [`cut_of`].
///
/// # Errors
///
/// A cut [`cut_of`] refuses, by name.
pub fn tp_tensors(split: &Split, u: &TpUnits) -> Result<Vec<TpTensor>, PlacementError> {
    split
        .iter_tensors()
        .map(|(_, t)| t)
        .filter(|t| card_holds(&t.name))
        .map(|t| {
            let split = cut_of(&t.name, t.ty, &t.dims, u)?;
            Ok(TpTensor {
                name: t.name.clone(),
                dims: t.dims.clone(),
                bytes: t.nbytes,
                card: Some(t.ty),
                whole_bytes: 0,
                block: split.map_or([t.ty.blck_size().unwrap_or(1), 1], |(_, b)| b),
                cut: split.map(|(c, _)| c),
            })
        })
        .collect()
}

/// The run a rank's need is counted at ([`rank_need`]): the positions a sequence holds, the
/// sequences resident at once, and the positions one step feeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunShape {
    pub ctx: u64,
    pub slots: u64,
    pub batch: u64,
}

/// What rank `rank` of `ranks` needs beside its weights to run the deepseek41 header `split`
/// at `run`; `split`, `whole` and `load_peak` are left 0 for the plan and the loader to fill
/// ([`crate::placement::tp::preflight`]).
///
/// - `kv` and `shadow`: every layer's cache and its ring shadow ([`KvLayout`]), whole on every
///   rank (the latent is shared by every head), a sequence each.
/// - `row_staging`: the Engram rows of the rank's hash columns ([`engram_cols`]) a step reads at
///   each site, twice (the step's and the next one's), at the table's row bytes.
/// - `row_cache`: the page-cache pages those row reads hold until evicted, every page each plane
///   of a row can touch on this host's page
///   ([`row_room`](crate::placement::row_room) of the rank's columns, read from the file's parts).
/// - `gathers`: one gather slot of `ranks` F32 partials a boundary kind in flight: the attention
///   and the FFN sums at `n_embd`, and each Engram site's at its `engram_wkv` output.
/// - `scratch`: the card's context and its stage's m = 1 scratch ([`CONTEXT_SELF`], [`SCRATCH`]).
///
/// # Errors
///
/// A context the step planner refuses ([`Planner::from_file`](super::plan::Planner::from_file)),
/// a layer whose cache the layout cannot count, Engram columns [`engram_cols`] refuses, an
/// Engram tensor the header lacks, and a byte count past u64, each by name.
pub fn rank_need(
    split: &Split,
    hp: &Hparams,
    run: RunShape,
    rank: usize,
    ranks: usize,
) -> Result<RankNeed, PlacementError> {
    let refuse = |name: &str, detail: &str| PlacementError::Tensor {
        name: name.into(),
        detail: detail.into(),
    };
    let over = || refuse("rank need", "bytes pass u64");
    super::plan::Planner::from_file(split, hp, run.ctx)
        .map_err(|e| refuse("rank need", &e.to_string()))?;
    let kv = KvLayout::from_file(split, hp)?;
    let (mut cache, mut shadow) = (0u64, 0u64);
    for l in 0..hp.n_layer {
        cache = cache
            .checked_add(kv.layer_bytes(l, run.ctx))
            .ok_or_else(over)?;
        shadow = shadow
            .checked_add(kv.shadow_bytes(l, run.ctx))
            .ok_or_else(over)?;
    }
    let (mut row_staging, mut row_cache, mut engram_out) = (0u64, 0u64, 0u64);
    if let Some(e) = &hp.engram {
        let cols = engram_cols(&TpUnits::of(hp), rank, ranks)?;
        let rows = run
            .batch
            .checked_mul(cols.end - cols.start)
            .and_then(|n| n.checked_mul(e.layer_ids.len() as u64))
            .ok_or_else(over)?;
        for &site in &e.layer_ids {
            let table = format!("blk.{site}.engram_embd.weight");
            let wkv = format!("blk.{site}.engram_wkv.weight");
            let (_, t) = split
                .find(&table)
                .ok_or_else(|| refuse(&table, "is not in the header"))?;
            let (_, w) = split
                .find(&wkv)
                .ok_or_else(|| refuse(&wkv, "is not in the header"))?;
            let rows_of = t
                .dims
                .get(1)
                .copied()
                .filter(|&r| r > 0)
                .ok_or_else(|| refuse(&table, "is not a table of rows"))?;
            let row_bytes = t.nbytes / rows_of;
            row_staging = row_staging.max(row_bytes);
            let wkv_out = w
                .dims
                .get(1)
                .copied()
                .ok_or_else(|| refuse(&wkv, "is not a 2-D projection"))?;
            engram_out = engram_out.checked_add(wkv_out).ok_or_else(over)?;
        }
        row_staging = rows
            .checked_mul(2)
            .and_then(|n| n.checked_mul(row_staging))
            .ok_or_else(over)?;
        let mut tables = roles::classify(split, hp)?;
        for t in &mut tables.tensors {
            if t.role == Role::EngramTable {
                t.gathered_rows = Some(cols.end - cols.start);
            }
        }
        row_cache = crate::placement::row_room(
            &tables,
            run.batch,
            crate::placement::host_lock::page_bytes()?,
            RowTableSource::Parts,
        )?;
    }
    let partial = (ranks as u64)
        .checked_mul(run.batch)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(over)?;
    let gathers = (2 * hp.n_embd as u64)
        .checked_add(engram_out)
        .and_then(|w| w.checked_mul(partial))
        .ok_or_else(over)?;
    Ok(RankNeed {
        kv: cache.checked_mul(run.slots).ok_or_else(over)?,
        shadow: shadow.checked_mul(run.slots).ok_or_else(over)?,
        row_staging,
        row_cache,
        gathers,
        scratch: CONTEXT_SELF + SCRATCH,
        ..RankNeed::default()
    })
}

#[cfg(test)]
mod tests {
    use super::{GgmlType, TpUnits, card_holds, cut, engram_cols, engram_rows, rank_span};
    use crate::placement::PlacementError;
    use crate::placement::tp::Cut;

    /// DeepSeek-V4.1-Flash's units: 64 heads, 8 output groups, 24 Engram columns.
    const V41: TpUnits = TpUnits {
        heads: 64,
        groups: 8,
        engram_cols: 24,
    };

    /// Each family at the real model's dims cuts by its rule, the rest stays whole, and a stem
    /// with no rule, a split tensor of another dim count and an extent off its units are refused.
    #[test]
    fn every_stem_cuts_by_its_rule() {
        let at = |dim, unit| Some(Cut { dim, unit });
        for (name, dims, want) in [
            ("blk.0.attn_q_b.weight", &[1280u64, 32768][..], at(1, 512)),
            ("blk.0.attn_sinks.weight", &[64], at(0, 1)),
            ("blk.0.attn_output_a.weight", &[4096, 8192], at(1, 1024)),
            ("blk.0.attn_output_b.weight", &[8192, 5120], at(0, 1024)),
            ("blk.1.engram_wkv.weight", &[6144, 25600], at(0, 256)),
            ("blk.0.ffn_gate_shexp.weight", &[5120, 2304], at(1, 1)),
            ("blk.0.ffn_down_shexp.weight", &[2304, 5120], at(0, 1)),
            ("blk.0.ffn_up_exps.weight", &[5120, 2304, 384], at(1, 1)),
            ("blk.0.ffn_down_exps.weight", &[2304, 5120, 384], at(0, 1)),
            ("output.weight", &[5120, 129280], at(1, 1)),
        ] {
            assert_eq!(cut(name, dims, &V41).unwrap(), want, "{name}");
        }
        for name in [
            "token_embd.weight",
            "blk.0.attn_q_a.weight",
            "blk.0.attn_kv.weight",
            "blk.2.attn_compressor_kv.weight",
            "blk.2.indexer.attn_q_b.weight",
            "blk.0.hc_attn_fn.weight",
            "blk.0.ffn_gate_inp.weight",
            "blk.1.engram_embd.weight",
            "blk.1.engram_q.weight",
        ] {
            assert_eq!(cut(name, &[5120, 512], &V41).unwrap(), None, "{name}");
        }
        assert!(!card_holds("token_embd.weight") && !card_holds("blk.14.engram_embd.weight"));
        assert!(card_holds("blk.14.engram_wkv.weight"));
        assert!(matches!(
            cut("blk.0.attn_q.weight", &[64], &V41),
            Err(PlacementError::Tensor { detail, .. }) if detail == "no tensor-parallel split rule"
        ));
        assert!(cut("blk.0.ffn_up_exps.weight", &[5120, 2304], &V41).is_err());
        assert!(cut("blk.1.engram_wkv.weight", &[6100, 25600], &V41).is_err());
    }

    /// One rank reads every tensor whole; at four, a rank reads its quarter of each cut: its 16
    /// heads of `attn_q_b`, its 2 groups of `attn_output_b`, its 6 Engram columns of
    /// `engram_wkv` (1,536 inputs) and its 576 of each routed expert's 2,304 inner rows.
    #[test]
    fn a_rank_reads_its_indices_of_each_cut() {
        let span =
            |name, ty, dims: &[u64], rank, ranks| rank_span(name, ty, dims, &V41, rank, ranks);
        let mx8 = GgmlType::MXFP8;
        let q_b = ("blk.0.attn_q_b.weight", &[1280u64, 32768][..]);
        assert_eq!(span(q_b.0, mx8, q_b.1, 0, 1).unwrap(), None);
        assert_eq!(
            span(q_b.0, mx8, q_b.1, 1, 4).unwrap(),
            Some((1, 8192..16384))
        );
        let o_b = ("blk.0.attn_output_b.weight", &[8192u64, 5120][..]);
        assert_eq!(
            span(o_b.0, mx8, o_b.1, 3, 4).unwrap(),
            Some((0, 6144..8192))
        );
        let wkv = ("blk.1.engram_wkv.weight", &[6144u64, 25600][..]);
        assert_eq!(
            span(wkv.0, mx8, wkv.1, 2, 4).unwrap(),
            Some((0, 3072..4608))
        );
        let up = ("blk.0.ffn_up_exps.weight", &[5120u64, 2304, 384][..]);
        let down = ("blk.0.ffn_down_exps.weight", &[2304u64, 5120, 384][..]);
        let mx4 = GgmlType::MXFP4;
        assert_eq!(span(up.0, mx4, up.1, 1, 4).unwrap(), Some((1, 576..1152)));
        assert_eq!(span(down.0, mx4, down.1, 3, 2).ok(), None);
        assert_eq!(
            span("blk.0.attn_kv.weight", mx8, &[5120, 512], 1, 4).unwrap(),
            None
        );
        // Rank 2 of 4 gathers columns 12..18, the rows of its six buckets.
        assert_eq!(engram_cols(&V41, 2, 4).unwrap(), 12..18);
        assert!(engram_cols(&V41, 0, 5).is_err());
        let primes = [3, 5, 7, 11];
        let offsets = [0, 3, 8, 15];
        assert_eq!(engram_rows(&primes, &offsets, 1..3), Some(3..15));
        assert_eq!(engram_rows(&primes, &offsets, 3..5), None);
    }
}
