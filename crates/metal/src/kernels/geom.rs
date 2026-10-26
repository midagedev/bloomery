use crate::attention_geom;
use crate::delta_geom;
use crate::experts_geom;
use crate::geometry;
use crate::instance::{HOST, Instance};
use crate::router_geom;

/// The numbers of decode attention for one instance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttnGeom {
    /// Values in one head.
    pub head: usize,
    /// The partial rope: the values of a head that turn, in pairs from the
    /// first.
    pub rot: usize,
    /// Query heads.
    pub query_heads: usize,
    /// Key and value heads.
    pub kv_heads: usize,
    /// Query heads per key head.
    pub gqa: usize,
    /// SIMD groups in the flash threadgroup.
    pub simd_groups: usize,
    /// Values one lane of the flash holds.
    pub lane_dims: usize,
    /// One segment partial: the max, the weight sum, then the head.
    pub partial_elems: usize,
    /// One head of the query projection: the query, then the gate when the
    /// layer writes one.
    pub qg_stride: usize,
    /// Threads in the flash threadgroup.
    pub flash_threads: usize,
    /// The score's scale.
    pub scale: f32,
    /// The query projection writes a gate beside each head's query.
    pub gated: bool,
}

/// The numbers of the gated delta step for one instance: its own axes, or the
/// axes its kernels are compiled at when its files have no delta layer
/// ([`Instance::delta_built`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeltaGeom {
    /// The value heads of the out projection.
    pub value_heads: usize,
    /// The columns of a value head.
    pub head_cols: usize,
    /// Taps of the causal conv.
    pub conv_taps: usize,
    /// Slots of the conv ring.
    pub ring_rows: usize,
    /// Threadgroups of the step: one per value head and column block.
    pub groups: usize,
    /// Threads in the step's threadgroup: one per column of a head.
    pub group_threads: usize,
    /// f32s of partial sums a row of the step writes: one per threadgroup.
    pub partials_stride: usize,
    /// The query's scale: `1 / sqrt(head_cols)` rounded to f32.
    pub q_scale: f32,
}

/// The numbers of the router for one instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouterGeom {
    /// The routed experts.
    pub experts: usize,
    /// The rows the router reads: the routed experts, and a gated shared
    /// expert's gate after them.
    pub rows: usize,
    /// One when the router has a shared expert's row, else zero.
    pub shared: usize,
    /// The columns of every row: the hidden width.
    pub k: usize,
    /// The partials of one launch.
    pub partials: usize,
    /// The threadgroups of one launch.
    pub groups: usize,
    /// The most routed experts one launch selects.
    pub top_k_max: usize,
    /// The float4s one lane loads from one split of a row.
    pub float4s: usize,
    /// The logits row stride a pass keeps: the rows rounded up to a multiple
    /// of four floats.
    pub logit_stride: usize,
}

/// The numbers of the routed gate-up pair and the folded down for one
/// instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertsGeom {
    /// Slots in one launch: the routed picks, and a gated shared expert's
    /// slot after them.
    pub slots: usize,
    /// Ids a launch reads: the picks.
    pub routed_ids: usize,
    /// The slot that reads the shared expert.
    pub shared_slot: usize,
    /// Whether the slots run a shared expert after the routed ones.
    pub has_shared: bool,
    /// The expert intermediate width: the pair's rows and the fold's K.
    pub k: usize,
    /// Blocks of K' of the fold.
    pub folded_blocks: usize,
}

/// The launch geometry of one instance: every number the host sizes a launch,
/// a window or a buffer by, derived once from its row.
///
/// [`Geometry::of`] is a `const fn` over the same functions the kernels'
/// source header is written from: `build.rs` prints each instance-derived
/// `#define` from this struct, so a launch and the kernel it dispatches
/// cannot disagree about a number. `Kernels` holds the geometry of its
/// library's instance (`Kernels::geom`), resolved when the pipelines are
/// made.
///
/// The names the host read before a geometry existed (`HEAD_DIM`,
/// `EXPERT_K`, `LOGIT_STRIDE`, `partials_len`, and the rest of the geometry
/// files' constants) remain as shims of the build's host instance ([`HOST`])
/// for the callers that have not moved to a geometry: the arch
/// (`src/arch/qwen3moe`), the e2e gate, the kernel gate and its oracles, and
/// the probes. Each shim's doc names the callers left.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    /// The instance the numbers are derived from.
    pub instance: &'static Instance,
    /// The residual width: the router's K and the head's.
    pub hidden: usize,
    /// Decode attention.
    pub attn: AttnGeom,
    /// The gated delta step.
    pub delta: DeltaGeom,
    /// The router.
    pub router: RouterGeom,
    /// The experts.
    pub experts: ExpertsGeom,
}

impl Geometry {
    /// The geometry of `inst`, from the functions the kernels' source header
    /// is written from, so the values equal the header's by construction.
    #[must_use]
    pub const fn of(inst: &'static Instance) -> Geometry {
        let key = &inst.key;
        let hidden = key.hidden as usize;
        let head = key.attn.row.head as usize;
        let delta = inst.delta_built();
        let value_heads = delta.value_heads as usize;
        let head_cols = delta.head_cols as usize;
        let conv_taps = delta.conv_taps as usize;
        let rows = router_geom::rows(inst);
        let slots = experts_geom::slots(inst);
        let routed_ids = key.moe.top_k as usize;
        Geometry {
            instance: inst,
            hidden,
            attn: AttnGeom {
                head,
                rot: key.attn.rot as usize,
                query_heads: key.attn.n_head as usize,
                kv_heads: key.attn.n_kv as usize,
                gqa: attention_geom::gqa(inst),
                simd_groups: attention_geom::simd_groups(inst),
                lane_dims: attention_geom::lane_dims(inst),
                partial_elems: attention_geom::partial_elems(head),
                qg_stride: attention_geom::qg_stride(inst),
                flash_threads: attention_geom::flash_threads(inst),
                scale: attention_geom::scale_of(head),
                gated: key.attn.q_gate,
            },
            delta: DeltaGeom {
                value_heads,
                head_cols,
                conv_taps,
                ring_rows: delta_geom::ring_rows(conv_taps),
                groups: value_heads * geometry::PARTIALS_PER_HEAD,
                group_threads: head_cols,
                partials_stride: value_heads * geometry::PARTIALS_PER_HEAD,
                q_scale: attention_geom::scale_of(head_cols),
            },
            router: RouterGeom {
                experts: key.moe.router.experts as usize,
                rows,
                shared: key.moe.router.rule.gated as usize,
                k: hidden,
                partials: router_geom::partials(inst),
                groups: router_geom::groups(inst),
                top_k_max: routed_ids,
                float4s: router_geom::float4s(hidden),
                logit_stride: router_geom::logit_stride(rows),
            },
            experts: ExpertsGeom {
                slots,
                routed_ids,
                shared_slot: experts_geom::shared_slot(inst),
                has_shared: slots > routed_ids,
                k: key.moe.expert_ff as usize,
                folded_blocks: experts_geom::folded_blocks(inst),
            },
        }
    }
}

/// The geometry of the build's host instance ([`HOST`]): what the shims of
/// the pre-geometry names read until their callers read a [`Geometry`].
pub const HOST_GEOMETRY: Geometry = Geometry::of(&HOST);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::REGISTERED;

    #[test]
    fn each_geometry_is_its_row() {
        for inst in REGISTERED {
            let g = Geometry::of(inst);
            let key = &inst.key;
            let name = inst.name;
            assert_eq!(g.instance, inst, "{name}: the instance");
            assert_eq!(g.hidden, key.hidden as usize, "{name}: hidden");

            let a = g.attn;
            assert_eq!(a.head, key.attn.row.head as usize, "{name}: head");
            assert_eq!(a.rot, key.attn.rot as usize, "{name}: rot");
            assert_eq!(a.query_heads, key.attn.n_head as usize, "{name}: heads");
            assert_eq!(a.kv_heads, key.attn.n_kv as usize, "{name}: kv heads");
            assert_eq!(a.gqa * a.kv_heads, a.query_heads, "{name}: gqa");
            assert_eq!(
                a.simd_groups * attention_geom::HEADS_PER_SG,
                a.gqa,
                "{name}: simd groups"
            );
            assert_eq!(a.lane_dims * 32, a.head, "{name}: lane dims");
            assert_eq!(a.partial_elems, a.head + 2, "{name}: partial");
            let gate = usize::from(key.attn.q_gate);
            assert_eq!(a.qg_stride, a.head * (1 + gate), "{name}: qg stride");
            assert_eq!(a.flash_threads, 32 * a.simd_groups, "{name}: flash threads");
            assert_eq!(a.gated, key.attn.q_gate, "{name}: gated");
            let ik = 1.0f32 / (a.head as f32).sqrt();
            assert_eq!(a.scale.to_bits(), ik.to_bits(), "{name}: scale");

            let d = g.delta;
            let built = inst.delta_built();
            assert_eq!(
                d.value_heads, built.value_heads as usize,
                "{name}: value heads"
            );
            assert_eq!(d.head_cols, built.head_cols as usize, "{name}: head cols");
            assert_eq!(d.conv_taps, built.conv_taps as usize, "{name}: taps");
            assert_eq!(
                d.ring_rows,
                d.conv_taps - 1 + delta_geom::PASS_ROWS,
                "{name}: ring"
            );
            assert_eq!(
                d.groups,
                d.value_heads * geometry::PARTIALS_PER_HEAD,
                "{name}: groups"
            );
            assert_eq!(d.group_threads, d.head_cols, "{name}: group threads");
            assert_eq!(d.partials_stride, d.groups, "{name}: partials stride");
            let ik = 1.0f32 / (d.head_cols as f32).sqrt();
            assert_eq!(d.q_scale.to_bits(), ik.to_bits(), "{name}: q scale");

            let r = g.router;
            let moe = &key.moe;
            assert_eq!(r.experts, moe.router.experts as usize, "{name}: experts");
            assert_eq!(
                r.shared,
                usize::from(moe.router.rule.gated),
                "{name}: shared"
            );
            assert_eq!(r.rows, r.experts + r.shared, "{name}: rows");
            assert_eq!(r.k, key.hidden as usize, "{name}: router k");
            assert_eq!(r.partials, r.rows, "{name}: partials");
            assert_eq!(
                r.groups,
                r.rows.div_ceil(router_geom::SIMD_GROUPS),
                "{name}: groups"
            );
            assert_eq!(r.top_k_max, moe.top_k as usize, "{name}: top k");
            assert_eq!(r.float4s * 4 * 32, r.k, "{name}: float4s");
            assert_eq!(r.logit_stride % 4, 0, "{name}: logit stride");
            assert!(
                r.logit_stride >= r.rows && r.logit_stride < r.rows + 4,
                "{name}: stride"
            );

            let e = g.experts;
            assert_eq!(e.routed_ids, moe.top_k as usize, "{name}: ids");
            assert_eq!(e.has_shared, moe.router.rule.gated, "{name}: has shared");
            assert_eq!(
                e.slots,
                e.routed_ids + usize::from(e.has_shared),
                "{name}: slots"
            );
            assert_eq!(e.shared_slot, e.routed_ids, "{name}: shared slot");
            assert_eq!(e.k, moe.expert_ff as usize, "{name}: expert k");
            assert_eq!(
                e.folded_blocks,
                e.slots * (e.k / 256),
                "{name}: folded blocks"
            );

            if *inst == HOST {
                assert_eq!(g, HOST_GEOMETRY, "{name}: the host's geometry");
                assert_eq!(a.head, attention_geom::HEAD_DIM, "shim HEAD_DIM");
                assert_eq!(a.rot, attention_geom::ROT, "shim ROT");
                assert_eq!(
                    a.query_heads,
                    attention_geom::QUERY_HEADS,
                    "shim QUERY_HEADS"
                );
                assert_eq!(a.kv_heads, attention_geom::KV_HEADS, "shim KV_HEADS");
                assert_eq!(a.qg_stride, attention_geom::QG_STRIDE, "shim QG_STRIDE");
                assert_eq!(
                    a.flash_threads,
                    attention_geom::FLASH_THREADS,
                    "shim FLASH_THREADS"
                );
                assert_eq!(
                    a.scale.to_bits(),
                    attention_geom::SCALE.to_bits(),
                    "shim SCALE"
                );
                assert_eq!(d.value_heads, geometry::VALUE_HEADS, "shim VALUE_HEADS");
                assert_eq!(d.head_cols, geometry::HEAD_COLS, "shim HEAD_COLS");
                assert_eq!(d.conv_taps, delta_geom::CONV_TAPS, "shim CONV_TAPS");
                assert_eq!(d.ring_rows, delta_geom::RING_ROWS, "shim RING_ROWS");
                assert_eq!(r.experts, router_geom::EXPERTS, "shim EXPERTS");
                assert_eq!(r.rows, router_geom::ROWS, "shim ROWS");
                assert_eq!(r.shared, router_geom::SHARED, "shim SHARED");
                assert_eq!(r.k, router_geom::K, "shim K");
                assert_eq!(r.partials, router_geom::PARTIALS, "shim PARTIALS");
                assert_eq!(r.groups, router_geom::GROUPS, "shim GROUPS");
                assert_eq!(r.top_k_max, router_geom::TOP_K_MAX, "shim TOP_K_MAX");
                assert_eq!(e.slots, experts_geom::EXPERT_SLOTS, "shim EXPERT_SLOTS");
                assert_eq!(e.routed_ids, experts_geom::ROUTED_IDS, "shim ROUTED_IDS");
                assert_eq!(e.shared_slot, experts_geom::SHARED_SLOT, "shim SHARED_SLOT");
                assert_eq!(e.k, experts_geom::EXPERT_K, "shim EXPERT_K");
                #[cfg(target_os = "macos")]
                {
                    use crate::kernels;
                    assert_eq!(g.hidden, kernels::HEAD_K, "shim HEAD_K");
                    assert_eq!(
                        d.q_scale.to_bits(),
                        kernels::Q_SCALE.to_bits(),
                        "shim Q_SCALE"
                    );
                    assert_eq!(r.logit_stride, kernels::LOGIT_STRIDE, "shim LOGIT_STRIDE");
                }
            }
        }
    }
}
