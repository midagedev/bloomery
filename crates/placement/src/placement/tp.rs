//! Tensor-parallel placement: N ranks, one a node, each holding a slice of every split tensor and
//! all of every whole one in its own node's pool. No host or NVMe tier: a rank holds its bytes
//! resident, and a plan that does not fit a rank's pool is refused by name.
//!
//! A tensor is split along one of its dims (ggml's order: dim 0 is the inner width a row's values
//! run along, dim 1 the rows, the dims past them whole matrices) into N equal parts. A part must
//! hold whole units of the tensor's role along that dim (a head's rows, say) and whole blocks of
//! its format (a 64-value NVFP4 block along dim 0, a 128-row FP8 block along dim 1): a cut off
//! either is refused by name. The family says which dim and unit ([`TpTensor::cut`]), the
//! format its bytes and blocks, and [`card_blocks`] includes the role's kernel K unit. Which
//! indices a rank holds is [`Cut::span`]'s, for every source of a rank's parts; [`part_of`] cuts
//! a file tensor's bytes to them.

use std::ops::Range;

use gguf::{GgmlType, TensorInfo};
use models::Role;

use super::devices::DeviceId;
use super::machine::file_card_format;
use super::workstation::{CardSpec, unified_of};
use crate::kernels;

/// A tensor as the plan takes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TpTensor {
    pub name: String,
    /// Logical element counts, ggml's order.
    pub dims: Vec<u64>,
    /// The bytes that split with the values (the values and their block scales).
    pub bytes: u64,
    /// The card type a rank loads its part in. A part then counts its own shape's resident bytes
    /// in that type's card layout ([`file_card_format`]), each row's padding and scale words
    /// included; `None` splits `bytes` evenly.
    pub card: Option<GgmlType>,
    /// The bytes every rank holds whole beside its part (a tensor scale a matrix).
    pub whole_bytes: u64,
    /// The values a part keeps whole along dims 0 and 1: the format's block, along dim 0 at
    /// least the role's kernel K unit ([`card_blocks`]).
    pub block: [u64; 2],
    /// `None` keeps the tensor whole on every rank.
    pub cut: Option<Cut>,
}

/// Where a tensor splits: along `dim`, in parts of whole `unit`s (the role's granule there).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cut {
    pub dim: usize,
    pub unit: u64,
}

impl Cut {
    /// One rank's extent along the cut of tensor `name` of `dims` into `n` ranks, its format and
    /// kernel blocks `blocks` values along dims 0 and 1 (1 past them). No rank, a dim the tensor
    /// lacks, a part of no units or of other than whole units, and a part off a block are refused by
    /// name.
    ///
    /// # Errors
    ///
    /// [`TpError::NoRank`], [`TpError::NoDim`], [`TpError::Unit`] or [`TpError::Block`].
    pub fn part(
        self,
        name: &str,
        dims: &[u64],
        blocks: [u64; 2],
        n: usize,
    ) -> Result<u64, TpError> {
        if n == 0 {
            return Err(TpError::NoRank);
        }
        let dim = self.dim;
        let extent = *dims.get(dim).ok_or_else(|| TpError::NoDim {
            name: name.into(),
            dim,
            dims: dims.to_vec(),
        })?;
        let part = extent / n as u64;
        if part == 0 || !extent.is_multiple_of(n as u64) || !part.is_multiple_of(self.unit) {
            return Err(TpError::Unit {
                name: name.into(),
                dim,
                extent,
                ranks: n,
                unit: self.unit,
            });
        }
        let block = blocks.get(dim).copied().unwrap_or(1);
        if !part.is_multiple_of(block) {
            return Err(TpError::Block {
                name: name.into(),
                dim,
                extent,
                ranks: n,
                block,
            });
        }
        Ok(part)
    }

    /// Rank `rank`'s indices along the cut of tensor `name` into `n` ranks: its [`Cut::part`],
    /// `rank` parts in. A rank that is not one of the `n` is refused by name, as is every cut
    /// [`Cut::part`] refuses.
    ///
    /// # Errors
    ///
    /// [`TpError::Rank`] and [`Cut::part`]'s.
    pub fn span(
        self,
        name: &str,
        dims: &[u64],
        blocks: [u64; 2],
        rank: usize,
        n: usize,
    ) -> Result<Range<u64>, TpError> {
        let part = self.part(name, dims, blocks, n)?;
        check_rank(name, rank, n)?;
        // `part · n` is the dim's extent, so the start does not wrap.
        let start = part * rank as u64;
        Ok(start..start + part)
    }
}

/// Refuse a rank of `name` that is not one of `ranks`.
///
/// # Errors
///
/// [`TpError::Rank`].
pub fn check_rank(name: &str, rank: usize, ranks: usize) -> Result<(), TpError> {
    if rank >= ranks {
        return Err(TpError::Rank {
            name: name.into(),
            rank,
            ranks,
        });
    }
    Ok(())
}

/// Refuse a file tensor whose dims do not hold its bytes: no dim or a zero one, a dim 0 off its
/// type's blocks, or a byte count those give past u64 or other than `t.nbytes`. Returns the
/// checked type's block width and byte size.
///
/// # Errors
///
/// [`TpError::Tensor`] naming the tensor.
pub fn check_bytes(t: &TensorInfo) -> Result<(u64, u64), TpError> {
    let layout =
        t.ty.blck_size()
            .zip(t.ty.type_size())
            .filter(|&(block, _)| {
                block > 0
                    && !t.dims.is_empty()
                    && t.dims.iter().all(|&d| d > 0)
                    && t.dims[0].is_multiple_of(block)
            })
            .filter(|&(block, size)| gguf::tensor_bytes(block, size, &t.dims) == Some(t.nbytes));
    let Some(layout) = layout else {
        return Err(TpError::Tensor {
            name: t.name.clone(),
            detail: format!("{} {:?} do not hold {} file bytes", t.ty, t.dims, t.nbytes),
        });
    };
    Ok(layout)
}

/// The part of the file tensor `t` that holds the indices `at.1` along dim `at.0` (a rank's,
/// [`Cut::span`]), or the whole tensor for `None`: the part's header entry (its dims, its bytes,
/// the tensor's offset) and the byte ranges of its rows in the tensor's bytes, in row order. A
/// tensor [`check_bytes`] refuses, indices that are empty, past the dim or along dim 0 off the
/// type's blocks, and bytes past usize are refused by name.
///
/// # Errors
///
/// [`check_bytes`]' and [`TpError::Tensor`].
pub fn part_of(
    t: &TensorInfo,
    at: Option<(usize, Range<u64>)>,
) -> Result<(TensorInfo, Vec<Range<usize>>), TpError> {
    let (block, size) = check_bytes(t)?;
    let refuse = |detail: String| TpError::Tensor {
        name: t.name.clone(),
        detail,
    };
    let fit = |v: u64| usize::try_from(v).map_err(|_| refuse(format!("{v} bytes pass usize")));
    let Some((dim, at)) = at else {
        return Ok((t.clone(), std::iter::once(0..fit(t.nbytes)?).collect()));
    };
    let runs = gguf::tensor_runs(block, size, &t.dims, dim, at.clone()).ok_or_else(|| {
        refuse(format!(
            "indices {at:?} along dim {dim} of {:?} are not whole {} blocks of it",
            t.dims, t.ty
        ))
    })?;
    let mut dims = t.dims.clone();
    dims[dim] = at.end - at.start;
    // Each run lies inside the tensor's bytes, so neither the sum nor an end passes them.
    let nbytes = runs.iter().map(|r| r.end - r.start).sum();
    let ranges = runs
        .into_iter()
        .map(|r| Ok(fit(r.start)?..fit(r.end)?))
        .collect::<Result<Vec<_>, TpError>>()?;
    Ok((
        TensorInfo {
            name: t.name.clone(),
            dims,
            ty: t.ty,
            offset: t.offset,
            nbytes,
        },
        ranges,
    ))
}

/// The values a part of a tensor of card type `ty`, whose format block spans `block` values along
/// dims 0 and 1, keeps whole: along dim 0 the least multiple of the block and the K unit of the
/// role's kernel ([`kernels::card_k_unit`]), so a cut never leaves a rank a part it cannot run. A
/// block past u64 is refused.
///
/// # Errors
///
/// [`TpError::Overflow`].
pub fn card_blocks(ty: GgmlType, block: [u64; 2], role: Role) -> Result<[u64; 2], TpError> {
    let unit = kernels::card_k_unit(ty, role);
    let (mut gcd, mut b) = (block[0], unit);
    while b != 0 {
        (gcd, b) = (b, gcd % b);
    }
    let k = (block[0] / gcd)
        .checked_mul(unit)
        .ok_or(TpError::Overflow)?;
    Ok([k, block[1]])
}

/// One rank's node, by the name its lines print, and the bytes its pool holds for the rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankPool {
    pub node: String,
    pub pool_bytes: u64,
    /// The rank's device, when its census named one: two ranks on one device are refused.
    pub device: Option<DeviceId>,
}

impl RankPool {
    /// The pool of the rank whose one device is `spec`, from its node's census: on a device that
    /// allocates from host memory, the node's pool ([`unified_of`]); else the card's usable bytes,
    /// capped by its free reading.
    #[must_use]
    pub fn of(node: String, spec: &CardSpec) -> RankPool {
        let pool_bytes = match unified_of(spec) {
            Some(pool) => pool.bytes,
            None => spec
                .free_bytes
                .map_or(spec.usable_bytes(), |free| free.min(spec.usable_bytes())),
        };
        RankPool {
            node,
            pool_bytes,
            device: spec.device,
        }
    }
}

/// One rank of a plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rank {
    pub rank: usize,
    pub node: String,
    pub pool_bytes: u64,
    /// The parts of the split tensors the rank holds.
    pub split_bytes: u64,
    /// The whole tensors and the whole bytes of the split ones.
    pub whole_bytes: u64,
    pub split_tensors: usize,
    pub whole_tensors: usize,
}

impl Rank {
    /// Every byte the rank holds.
    ///
    /// # Panics
    ///
    /// If the rank's byte sum exceeds u64.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.split_bytes
            .checked_add(self.whole_bytes)
            .expect("tp: rank weight byte sum exceeds u64")
    }
}

/// Why a plan was refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TpError {
    #[error("tensor parallel: no rank to plan")]
    NoRank,
    #[error("tp: rank weight byte sum exceeds u64")]
    Overflow,
    #[error("tp: {name}: dim {dim} is past its dims {dims:?}")]
    NoDim {
        name: String,
        dim: usize,
        dims: Vec<u64>,
    },
    #[error(
        "tp: {name}: dim {dim} of {extent} into {ranks} ranks is not a whole number of {unit}-value units a rank"
    )]
    Unit {
        name: String,
        dim: usize,
        extent: u64,
        ranks: usize,
        unit: u64,
    },
    #[error(
        "tp: {name}: dim {dim} of {extent} into {ranks} ranks cuts off its format's {block}-value block"
    )]
    Block {
        name: String,
        dim: usize,
        extent: u64,
        ranks: usize,
        block: u64,
    },
    #[error("tp: {name}: rank {rank} is not one of {ranks} ranks")]
    Rank {
        name: String,
        rank: usize,
        ranks: usize,
    },
    #[error("tp: {name}: {detail}")]
    Tensor { name: String, detail: String },
    #[error("tp: rank {rank} on {node} is the device of rank {other}")]
    SameDevice {
        rank: usize,
        node: String,
        other: usize,
    },
    #[error("tp: rank {rank} on {node} holds {bytes} B, past its pool of {pool} B")]
    Over {
        rank: usize,
        node: String,
        bytes: u64,
        pool: u64,
    },
}

/// The plan of `tensors` over one rank a pool of `pools`, in rank order.
///
/// # Errors
///
/// A cut [`Cut::part`] refuses, a rank past its pool and two ranks on one device, each by name; no
/// rank or a byte sum past u64.
pub fn plan_tp(pools: &[RankPool], tensors: &[TpTensor]) -> Result<Vec<Rank>, TpError> {
    let n = pools.len();
    if n == 0 {
        return Err(TpError::NoRank);
    }
    for (rank, p) in pools.iter().enumerate() {
        let same = |q: &RankPool| {
            p.device
                .is_some_and(|d| q.device.is_some_and(|e| e.uuid == d.uuid))
        };
        if let Some(other) = pools[..rank].iter().position(same) {
            return Err(TpError::SameDevice {
                rank,
                node: p.node.clone(),
                other,
            });
        }
    }
    let (mut split, mut whole, mut split_n, mut whole_n) = (0u64, 0u64, 0, 0);
    for t in tensors {
        let Some(cut) = t.cut else {
            whole = whole
                .checked_add(t.bytes)
                .and_then(|sum| sum.checked_add(t.whole_bytes))
                .ok_or(TpError::Overflow)?;
            whole_n += 1;
            continue;
        };
        let part = cut.part(&t.name, &t.dims, t.block, n)?;
        let bytes = match t.card {
            Some(ty) => part_bytes(t, cut.dim, part, ty)?,
            // Whole blocks a part: the bytes split evenly with the values.
            None => t.bytes / n as u64,
        };
        split = split.checked_add(bytes).ok_or(TpError::Overflow)?;
        whole = whole.checked_add(t.whole_bytes).ok_or(TpError::Overflow)?;
        split_n += 1;
    }
    split.checked_add(whole).ok_or(TpError::Overflow)?;
    pools
        .iter()
        .enumerate()
        .map(|(rank, p)| {
            let r = Rank {
                rank,
                node: p.node.clone(),
                pool_bytes: p.pool_bytes,
                split_bytes: split,
                whole_bytes: whole,
                split_tensors: split_n,
                whole_tensors: whole_n,
            };
            if r.bytes() > p.pool_bytes {
                return Err(TpError::Over {
                    rank,
                    node: p.node.clone(),
                    bytes: r.bytes(),
                    pool: p.pool_bytes,
                });
            }
            Ok(r)
        })
        .collect()
}

/// The resident bytes of `t`'s part of `part` values along `dim` in card type `ty`'s layout, as a
/// rank's load allocates them; a shape past u64 or with no layout is refused.
fn part_bytes(t: &TpTensor, dim: usize, part: u64, ty: GgmlType) -> Result<u64, TpError> {
    let mut dims = t.dims.clone();
    dims[dim] = part;
    let k = dims[0];
    let rows = dims[1..]
        .iter()
        .try_fold(1u64, |a, &b| a.checked_mul(b))
        .ok_or(TpError::Overflow)?;
    file_card_format(&t.name, ty, k)
        .and_then(|f| f.resident_bytes(ty, k, rows))
        .ok_or_else(|| TpError::Tensor {
            name: t.name.clone(),
            detail: format!("no resident card form for {ty} part {dims:?}"),
        })
}

#[cfg(test)]
mod tests {
    use super::{
        Cut, GgmlType, Range, RankPool, Role, TensorInfo, TpError, TpTensor, card_blocks, part_of,
        plan_tp,
    };

    fn pools(n: usize, bytes: u64) -> Vec<RankPool> {
        (0..n)
            .map(|i| RankPool {
                node: format!("node{i}"),
                pool_bytes: bytes,
                device: None,
            })
            .collect()
    }

    /// A 2-D tensor of `rows` rows of `k` values, `bytes` a value, cut along `cut`.
    fn t(name: &str, k: u64, rows: u64, block: [u64; 2], cut: Option<Cut>) -> TpTensor {
        TpTensor {
            name: name.into(),
            dims: vec![k, rows],
            bytes: k * rows,
            card: None,
            whole_bytes: 4,
            block,
            cut,
        }
    }

    /// Every rank holds a 1/N part of each split tensor and all of each whole one; N = 1, 2 and 4
    /// give one sum by one rule, and a rank past its pool is refused naming it.
    #[test]
    fn ranks_hold_a_part_of_each_split_tensor_and_every_whole_one() {
        let rows = Some(Cut { dim: 1, unit: 128 });
        let cols = Some(Cut { dim: 0, unit: 1 });
        let ts = [
            t("q", 4096, 8192, [1, 1], rows),
            t("o", 8192, 4096, [64, 1], cols),
            t("norm", 4096, 1, [1, 1], None),
        ];
        for n in [1, 2, 4] {
            let plan = plan_tp(&pools(n, 1 << 40), &ts).unwrap();
            assert_eq!(plan.len(), n);
            for (i, r) in plan.iter().enumerate() {
                assert_eq!(r.rank, i);
                assert_eq!(r.split_bytes, 2 * 4096 * 8192 / n as u64);
                assert_eq!(r.whole_bytes, 4096 + 4 * 3);
                assert_eq!((r.split_tensors, r.whole_tensors), (2, 1));
            }
        }
        let need = 4096 * 8192 + 4096 + 12;
        assert_eq!(plan_tp(&pools(2, need), &ts).unwrap()[1].bytes(), need);
        let mut short = pools(2, need);
        short[1].pool_bytes = need - 1;
        assert_eq!(
            plan_tp(&short, &ts).unwrap_err(),
            TpError::Over {
                rank: 1,
                node: "node1".into(),
                bytes: need,
                pool: need - 1
            }
        );
    }

    /// A part of a tensor with a card type counts its own rows' padding and scale words: K 2304
    /// over 4 ranks leaves 576, which MXFP8 planes hold in 576 + 4 · ceil(576 / 128) = 596 bytes a
    /// row and q5_1 in one 1024-value code window and 18 scale pairs, 1168, not a quarter of the
    /// tensor's 594 and 912. A pool one byte short of that is refused. Mutant, the tensor's bytes
    /// over N: 3,041,280 and 4,669,440 pass both pools.
    #[test]
    fn a_part_counts_its_own_rows_padding() {
        const ROWS: u64 = 5120;
        for (ty, whole, need) in [
            (GgmlType::MXFP8, 12_165_120, 3_051_520),
            (GgmlType::Q5_1, 18_677_760, 5_980_160),
        ] {
            let ts = [TpTensor {
                name: "down".into(),
                dims: vec![2304, ROWS],
                bytes: whole,
                card: Some(ty),
                whole_bytes: 0,
                block: [32, 1],
                cut: Some(Cut { dim: 0, unit: 32 }),
            }];
            let plan = plan_tp(&pools(4, need), &ts).unwrap();
            assert!(plan.iter().all(|r| r.bytes() == need), "{ty}: {plan:?}");
            assert_eq!(
                plan_tp(&pools(4, need - 1), &ts).unwrap_err(),
                TpError::Over {
                    rank: 0,
                    node: "node0".into(),
                    bytes: need,
                    pool: need - 1
                },
                "{ty}"
            );
        }
    }

    /// A unified node's rank plans against its node's pool, a card's against its usable bytes
    /// capped by its free reading.
    #[test]
    fn a_rank_plans_against_its_own_nodes_pool() {
        use crate::placement::devices::DeviceId;
        use crate::placement::workstation::{CardSpec, gb10};
        let unified = gb10(125_303_951_360);
        assert_eq!(
            RankPool::of("a".into(), &unified).pool_bytes,
            125_303_951_360
        );
        let card = CardSpec {
            integrated: false,
            ..gb10(1 << 40)
        };
        assert_eq!(
            RankPool::of("b".into(), &card).pool_bytes,
            card.usable_bytes()
        );
        let busy = CardSpec {
            free_bytes: Some(1 << 30),
            ..card
        };
        assert_eq!(RankPool::of("c".into(), &busy).pool_bytes, 1 << 30);
        // A node answering for this process's own device is not a rank of its own.
        let id = |uuid| Some(DeviceId { ordinal: 0, uuid });
        let on = |node: &str, uuid| RankPool {
            device: id(uuid),
            ..RankPool::of(node.into(), &unified)
        };
        let pools = [
            on("local", [1; 16]),
            on("other", [2; 16]),
            on("loop", [1; 16]),
        ];
        assert_eq!(
            plan_tp(&pools, &[]).unwrap_err(),
            TpError::SameDevice {
                rank: 2,
                node: "loop".into(),
                other: 0
            }
        );
        assert!(plan_tp(&pools[..2], &[]).is_ok());
    }

    #[test]
    fn rank_byte_sums_must_fit_u64() {
        let tensor = |bytes, whole_bytes, cut| TpTensor {
            name: "large".into(),
            dims: vec![1 << 62],
            bytes,
            card: None,
            whole_bytes,
            block: [1, 1],
            cut,
        };
        let big = 1 << 63;
        let cut = Some(Cut { dim: 0, unit: 1 });
        let whole = tensor(big, 0, None);
        let split = tensor(big, 0, cut);
        let companion = tensor(0, big, cut);
        let results = [
            vec![whole.clone(), whole.clone()],
            vec![tensor(big, big, None)],
            vec![split.clone(), split.clone()],
            vec![companion.clone(), companion],
            vec![split, whole],
        ]
        .map(|ts| plan_tp(&pools(1, 1), &ts));
        assert!(
            results.iter().all(|r| matches!(r, Err(TpError::Overflow))),
            "{results:?}"
        );
        for cut in [None, cut] {
            let plan = plan_tp(&pools(1, u64::MAX), &[tensor(big, big - 1, cut)]).unwrap();
            assert_eq!(plan[0].bytes(), u64::MAX);
        }
    }

    /// A part that is not whole units of the role, or that cuts a format block, is refused by name;
    /// so are a part of no units, a dim the tensor lacks and no rank.
    #[test]
    fn cuts_off_a_unit_or_a_block_are_refused_by_name() {
        let heads = Some(Cut { dim: 1, unit: 128 });
        // 64 heads of 128 rows: 3 ranks part no whole head, 128 ranks take half a head each.
        for n in [3, 128] {
            let e = plan_tp(&pools(n, u64::MAX), &[t("q", 4096, 8192, [1, 1], heads)]).unwrap_err();
            assert!(
                matches!(e, TpError::Unit { ranks, unit: 128, .. } if ranks == n),
                "{e}"
            );
        }
        // A down of K 1024 at N = 4 is 256 a rank, whole 64-value NVFP4 blocks; K 1088 is 272.
        let down = |k| t("down", k, 4096, [64, 1], Some(Cut { dim: 0, unit: 1 }));
        assert!(plan_tp(&pools(4, u64::MAX), &[down(1024)]).is_ok());
        let e = plan_tp(&pools(4, u64::MAX), &[down(1088)]).unwrap_err();
        assert_eq!(
            e.to_string(),
            "tp: down: dim 0 of 1088 into 4 ranks cuts off its format's 64-value block"
        );
        // A 128-row FP8 block: 2048 rows into 4 is 512, into 32 is 64.
        let gate = t(
            "gate",
            4096,
            2048,
            [128, 128],
            Some(Cut { dim: 1, unit: 1 }),
        );
        assert!(plan_tp(&pools(4, u64::MAX), std::slice::from_ref(&gate)).is_ok());
        assert!(matches!(
            plan_tp(&pools(32, u64::MAX), &[gate]).unwrap_err(),
            TpError::Block { block: 128, .. }
        ));
        let e = plan_tp(
            &pools(2, u64::MAX),
            &[t("k_b", 4096, 64, [1, 1], Some(Cut { dim: 2, unit: 1 }))],
        )
        .unwrap_err();
        assert!(matches!(e, TpError::NoDim { dim: 2, .. }), "{e}");
        // An extent of no values leaves every rank no unit.
        let empty = t("empty", 0, 64, [1, 1], Some(Cut { dim: 0, unit: 1 }));
        assert!(matches!(
            plan_tp(&pools(2, u64::MAX), &[empty]).unwrap_err(),
            TpError::Unit { extent: 0, .. }
        ));
        assert_eq!(plan_tp(&[], &[]).unwrap_err(), TpError::NoRank);
    }

    /// A cut along K keeps the card layout's unit whole: MXFP8 planes and an MXFP4 routed down,
    /// whose part reads the 32-value activation, keep their 32-value block (K 2304 at N = 8 is
    /// 288 a rank, at N = 16 144); a type whose block is its layout keeps its block.
    // PIN(2026-11-03): MXFP8 planes took 128 values and an MXFP4 stack 256 (K 4096 at N = 32
    // and 1024 at N = 4 the last fits); both now take 32-value parts.
    #[test]
    fn a_cut_keeps_the_card_layouts_k_unit_whole() {
        for (ty, role) in [
            (GgmlType::MXFP8, Role::DenseFfn),
            (GgmlType::MXFP4, Role::RoutedExperts),
        ] {
            let (k, fits, cuts) = (2304, 8, 16);
            let block = card_blocks(ty, [32, 1], role).unwrap();
            let down = || t("down", k, 64, block, Some(Cut { dim: 0, unit: 1 }));
            assert!(plan_tp(&pools(fits, u64::MAX), &[down()]).is_ok(), "{ty}");
            assert!(
                matches!(
                    plan_tp(&pools(cuts, u64::MAX), &[down()]).unwrap_err(),
                    TpError::Block { .. }
                ),
                "{ty}"
            );
        }
        // Dense NVFP4 reads Q8Act; routed down reads Q8Blocks32.
        let dense = card_blocks(GgmlType::NVFP4, [64, 1], Role::DenseFfn).unwrap();
        let routed = card_blocks(GgmlType::NVFP4, [32, 1], Role::RoutedExperts).unwrap();
        for role in [Role::Attention, Role::SharedExpert] {
            assert_eq!(card_blocks(GgmlType::NVFP4, [64, 1], role), Ok(dense));
        }
        for n in [1, 2, 4] {
            let cut = Cut { dim: 0, unit: 1 };
            assert_eq!(
                cut.part("routed down", &[32 * n as u64, 64, 2], routed, n),
                Ok(32)
            );
            assert_eq!(
                cut.part("dense down", &[256 * n as u64, 64], dense, n),
                Ok(256)
            );
            for k in [64, 128, 192] {
                assert!(matches!(
                    cut.part("dense down", &[k * n as u64, 64], dense, n),
                    Err(TpError::Block { block: 256, .. })
                ));
            }
        }
        assert_eq!(
            card_blocks(GgmlType::Q4_K, [256, 1], Role::DenseFfn),
            Ok([256, 1])
        );
        assert_eq!(
            card_blocks(GgmlType::MXFP4, [48, 128], Role::RoutedExperts),
            Ok([96, 128])
        );
        assert_eq!(
            card_blocks(GgmlType::MXFP4, [u64::MAX, 1], Role::RoutedExperts),
            Err(TpError::Overflow)
        );
    }

    /// A file tensor of `dims` of `ty`, its bytes numbered so a part's bytes name where they came
    /// from.
    fn tensor(ty: GgmlType, dims: &[u64]) -> (TensorInfo, Vec<u8>) {
        let rows: u64 = dims[1..].iter().product();
        let row = dims[0] / ty.blck_size().unwrap() * ty.type_size().unwrap();
        let nbytes = rows * row;
        let t = TensorInfo {
            name: "t.weight".into(),
            dims: dims.to_vec(),
            ty,
            offset: 0,
            nbytes,
        };
        let bytes = (0..nbytes).map(|i| (i * 7 % 251) as u8).collect();
        (t, bytes)
    }

    /// Rank `rank` of `n`'s part of `t` cut by `cut` as role `role` reads it: [`Cut::span`] at
    /// the type's blocks and the role's K unit, then [`part_of`].
    fn rank_part(
        t: &TensorInfo,
        cut: Option<Cut>,
        role: Role,
        rank: usize,
        n: usize,
    ) -> Result<(TensorInfo, Vec<Range<usize>>), TpError> {
        let at = match cut {
            None => None,
            Some(c) => {
                let blocks = card_blocks(t.ty, [super::check_bytes(t)?.0, 1], role)?;
                Some((c.dim, c.span(&t.name, &t.dims, blocks, rank, n)?))
            }
        };
        part_of(t, at)
    }

    /// Element `(i0, i1, i2)`'s bytes of a tensor of `dims`: `[row(i1, i2)][i0 block]`.
    fn block_bytes(t: &TensorInfo, bytes: &[u8], i: [u64; 3]) -> Vec<u8> {
        let (bs, ts) = (t.ty.blck_size().unwrap(), t.ty.type_size().unwrap());
        let d = |k: usize| t.dims.get(k).copied().unwrap_or(1);
        let row = d(0) / bs * ts;
        let at = (i[2] * d(1) + i[1]) * row + i[0] / bs * ts;
        bytes[at as usize..(at + ts) as usize].to_vec()
    }

    /// At N = 1, 2 and 4, along every dim of a 3-D stack and of a 2-D matrix, each rank's part
    /// is the dims with the cut dim over N, its bytes the tensor's blocks at those indices, and
    /// the parts' bytes add to the tensor's; whole tensors are their bytes on every rank.
    #[test]
    fn every_rank_holds_its_indices_of_the_cut_dim() {
        for (ty, dims) in [
            (GgmlType::Q5_K, vec![1024u64, 8, 4]),
            (GgmlType::Q8_0, vec![256, 16]),
            (GgmlType::F32, vec![8, 4, 16]),
        ] {
            let (t, bytes) = tensor(ty, &dims);
            let bs = ty.blck_size().unwrap();
            for n in [1usize, 2, 4] {
                for dim in 0..dims.len() {
                    let cut = Some(Cut { dim, unit: 1 });
                    let mut total = 0;
                    for r in 0..n {
                        let (p, spans) = rank_part(&t, cut, Role::DenseFfn, r, n).unwrap();
                        let part = dims[dim] / n as u64;
                        let mut want_dims = dims.clone();
                        want_dims[dim] = part;
                        assert_eq!(p.dims, want_dims, "{ty} {dims:?} dim {dim} rank {r}/{n}");
                        let got: Vec<u8> = spans
                            .iter()
                            .flat_map(|s| bytes[s.clone()].to_vec())
                            .collect();
                        assert_eq!(got.len() as u64, p.nbytes);
                        let mut want = Vec::new();
                        let d = |k: usize| want_dims.get(k).copied().unwrap_or(1);
                        for i2 in 0..d(2) {
                            for i1 in 0..d(1) {
                                for i0 in (0..d(0)).step_by(bs as usize) {
                                    let mut i = [i0, i1, i2];
                                    i[dim] += r as u64 * part;
                                    want.extend(block_bytes(&t, &bytes, i));
                                }
                            }
                        }
                        assert_eq!(got, want, "{ty} {dims:?} dim {dim} rank {r}/{n}");
                        total += p.nbytes;
                    }
                    assert_eq!(total, t.nbytes);
                }
                let (p, spans) = rank_part(&t, None, Role::DenseFfn, n - 1, n).unwrap();
                assert_eq!((&p.dims, p.nbytes), (&t.dims, t.nbytes));
                assert_eq!(spans, vec![0..bytes.len()]);
            }
        }
    }

    #[test]
    fn malformed_tensor_sizes_are_refused_before_cutting() {
        let mut accepted = Vec::new();
        for (dims, nbytes) in [
            (vec![256, 1], 143),
            (vec![256, 1], 145),
            (vec![], 0),
            (vec![0, 1], 0),
            (vec![256, 0], 0),
            (vec![257, 1], 144),
            (vec![256, u64::MAX, 2], 144),
        ] {
            let t = TensorInfo {
                name: "bad".into(),
                dims: dims.clone(),
                ty: GgmlType::Q4_K,
                offset: 0,
                nbytes,
            };
            for cut in [None, Some(Cut { dim: 1, unit: 1 })] {
                if !std::panic::catch_unwind(|| rank_part(&t, cut, Role::DenseFfn, 0, 1))
                    .is_ok_and(|result| result.is_err())
                {
                    accepted.push((dims.clone(), nbytes, cut));
                }
            }
        }
        assert!(accepted.is_empty(), "not refused safely: {accepted:?}");
        assert!(gguf::tensor_runs(1, 0, &[64, 2, 1], 1, 0..1).is_none());
        for dims in [[0, 2, 1], [64, 2, 0]] {
            assert!(
                gguf::tensor_runs(1, 4, &dims, 1, 0..1).is_none(),
                "{dims:?}"
            );
        }
        let (t, _) = tensor(GgmlType::Q5_K, &[512, 4]);
        assert_eq!(super::check_bytes(&t).unwrap(), (256, 176));
        for at in [
            (0, 128..384),
            (0, 256..768),
            (1, 2..2),
            (1, 3..5),
            (2, 0..1),
        ] {
            assert!(part_of(&t, Some(at.clone())).is_err(), "{at:?}");
        }
        let (p, spans) = part_of(&t, Some((0, 256..512))).unwrap();
        assert_eq!((p.dims, spans.len()), (vec![256, 4], 4));
        let (t, _) = tensor(GgmlType::Q4_K, &[256, 1]);
        assert!(rank_part(&t, Some(Cut { dim: 1, unit: 1 }), Role::DenseFfn, 0, 4).is_err());
    }

    /// An extent the ranks do not divide, a part of other than whole units, a dim-0 part off the
    /// type's block or its card layout's K unit, a dim past the tensor's and a rank that is not
    /// one of the ranks are refused by name.
    #[test]
    fn a_cut_off_its_unit_or_block_is_refused() {
        let (t, _) = tensor(GgmlType::Q5_K, &[512, 6, 4]);
        for (cut, rank, n) in [
            (Cut { dim: 1, unit: 1 }, 0, 4),
            (Cut { dim: 1, unit: 2 }, 0, 2),
            (Cut { dim: 0, unit: 1 }, 0, 4),
            (Cut { dim: 3, unit: 1 }, 0, 2),
            (Cut { dim: 2, unit: 0 }, 0, 2),
            (Cut { dim: 2, unit: 1 }, 2, 2),
        ] {
            assert!(
                rank_part(&t, Some(cut), Role::DenseFfn, rank, n).is_err(),
                "{cut:?} {rank}/{n}"
            );
        }
        assert_eq!(
            rank_part(&t, Some(Cut { dim: 2, unit: 1 }), Role::DenseFfn, 2, 2).unwrap_err(),
            TpError::Rank {
                name: t.name.clone(),
                rank: 2,
                ranks: 2
            }
        );
        // PIN(2026-11-03): an MXFP4 stack kept 256 values along K (1024 into 4 ranks, not 8, as a
        // DenseFfn); a routed down keeps its 32-value block now: 1024 into 32 ranks, not into 64.
        let (mx, _) = tensor(GgmlType::MXFP4, &[1024, 4]);
        let k = Some(Cut { dim: 0, unit: 1 });
        assert!(rank_part(&mx, k, Role::RoutedExperts, 0, 32).is_ok());
        assert!(rank_part(&mx, k, Role::RoutedExperts, 0, 64).is_err());
        let (nv, _) = tensor(GgmlType::NVFP4, &[1024, 64]);
        for role in [Role::Attention, Role::DenseFfn, Role::SharedExpert] {
            for n in [1, 2, 4] {
                assert!(rank_part(&nv, k, role, 0, n).is_ok(), "{role:?} N={n}");
            }
            for n in [8, 16] {
                let e = rank_part(&nv, k, role, 0, n).unwrap_err().to_string();
                assert!(e.contains(&nv.name) && e.contains("256"), "{e}");
            }
        }
        let (routed, _) = tensor(GgmlType::NVFP4, &[1024, 64, 2]);
        assert!(rank_part(&routed, k, Role::RoutedExperts, 0, 16).is_ok());
    }
}
