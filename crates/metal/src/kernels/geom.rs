use core::fmt;

use crate::attention_geom;
use crate::delta_geom;
use crate::experts_geom;
use crate::geometry;
use crate::instance::{HOST, Instance};
use crate::router_geom;

/// A geometry one kernel family cannot take: which kernel, which axis, the
/// value the geometry has and what the kernel needs there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelRefused {
    /// The kernel family: `attention`, `router`, `experts` or `delta`.
    pub kernel: &'static str,
    /// The axis the kernel cannot take.
    pub axis: &'static str,
    /// The axis's value in the geometry, as text.
    pub value: String,
    /// What the kernel needs of the axis.
    pub needs: &'static str,
}

impl KernelRefused {
    /// The refusal of `kernel` at `axis`, which is `value` where `needs` is
    /// asked.
    #[must_use]
    pub fn new(
        kernel: &'static str,
        axis: &'static str,
        value: impl fmt::Display,
        needs: &'static str,
    ) -> Self {
        Self {
            kernel,
            axis,
            value: value.to_string(),
            needs,
        }
    }
}

/// A flag as a refusal prints it.
#[must_use]
pub const fn yes_no(flag: bool) -> &'static str {
    if flag { "yes" } else { "no" }
}

impl fmt::Display for KernelRefused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} is {}; it must be {}",
            self.kernel, self.axis, self.value, self.needs
        )
    }
}

impl std::error::Error for KernelRefused {}

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
    /// The flash cuts a window a launch names.
    pub windowed: bool,
    /// The layer's output gate is one value a query head, from a projection
    /// of its own that the launch binds.
    pub head_gate: bool,
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

/// The numbers of the router for one instance: its own, or the axes its
/// kernels are compiled at when its files have a dense FFN and launch none
/// ([`Instance::router_built`]).
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
    /// The FFN is dense: its one slot is expert 0 of one-expert stacks at
    /// weight 1, no router picks it and no shared expert follows it.
    pub dense: bool,
}

/// The launch geometry of one instance: every number the host sizes a launch,
/// a window or a buffer by, derived once from its row.
///
/// [`Geometry::of`] is a `const fn` over the same functions the kernels'
/// source header is written from: [`crate::header::render`] prints each
/// instance-derived `#define` from this struct, at build for a registered
/// instance and at open for the file's, so a launch and the kernel it
/// dispatches cannot disagree about a number. `Kernels` holds the geometry of
/// its library's instance (`Kernels::geom`), resolved when the pipelines are
/// made. A geometry the kernels cannot take is refused by
/// [`Geometry::check`], which every open runs before the header is written.
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
    pub instance: Instance,
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
    ///
    /// The numbers are derived, not judged: a geometry no kernel takes still
    /// has them (a score scale none is written for is NaN), and
    /// [`Geometry::check`] refuses it.
    #[must_use]
    pub const fn of(inst: &Instance) -> Geometry {
        let key = &inst.key;
        let hidden = key.hidden as usize;
        let head = key.attn.row.head as usize;
        let delta = inst.delta_built();
        let value_heads = delta.value_heads as usize;
        let head_cols = delta.head_cols as usize;
        let conv_taps = delta.conv_taps as usize;
        let router = inst.router_built();
        let rows = router_geom::rows(inst);
        let slots = experts_geom::slots(inst);
        let routed_ids = experts_geom::routed_ids(inst);
        Geometry {
            instance: *inst,
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
                windowed: attention_geom::windowed(inst),
                head_gate: attention_geom::head_gate(inst),
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
                experts: router.router.experts as usize,
                rows,
                shared: router.router.rule.gated as usize,
                k: hidden,
                partials: router_geom::partials(inst),
                groups: router_geom::groups(inst),
                top_k_max: router.top_k as usize,
                float4s: router_geom::float4s(hidden),
                logit_stride: router_geom::logit_stride(rows),
            },
            experts: ExpertsGeom {
                slots,
                routed_ids,
                shared_slot: experts_geom::shared_slot(inst),
                has_shared: slots > routed_ids,
                k: experts_geom::expert_k(inst),
                dense: key.is_dense(),
            },
        }
    }

    /// Whether every kernel family takes this geometry: the kernels' own
    /// `static_assert`s, stated where the host sizes its launches.
    ///
    /// # Errors
    ///
    /// [`KernelRefused`] naming the first kernel family, axis and value a
    /// kernel cannot take.
    pub fn check(&self) -> Result<(), KernelRefused> {
        attention_geom::holds(self)?;
        router_geom::holds(self)?;
        experts_geom::holds(self)?;
        delta_geom::holds(self)
    }
}

/// The geometry of the build's host instance ([`HOST`]): what the shims of
/// the pre-geometry names read until their callers read a [`Geometry`].
pub const HOST_GEOMETRY: Geometry = Geometry::of(&HOST);

#[cfg(test)]
mod tests {
    use models::shape::{self, RouterBody};

    use super::*;
    use crate::instance::{DeltaAxes, FfnAxes, MoeAxes, QWEN35MOE_A3B, REGISTERED};

    /// A case of the refusal table: how Qwen3.6's row is moved, and the kernel,
    /// axis and value (a prefix of it) the geometry is then refused at.
    type Case = (
        fn(&mut Instance),
        (&'static str, &'static str, &'static str),
    );

    /// `inst`'s routed axes, edited by `edit`.
    fn moe(inst: &mut Instance, edit: impl FnOnce(&mut MoeAxes)) {
        let mut axes = inst.key.routed().expect("a routed row");
        edit(&mut axes);
        inst.key.ffn = FfnAxes::Routed(axes);
    }

    /// Attention of `n_head` heads of `head` values over the shape table's
    /// row of a group of eight.
    fn attention(inst: &mut Instance, head: u32, n_head: u32) {
        inst.key.attn.row = shape::gqa_row(head, 8).expect("the shape table has the row");
        inst.key.attn.n_head = n_head;
    }

    /// Qwen3-30B's attention over the shape table's row that cuts a window and
    /// folds sinks.
    fn window_row(inst: &mut Instance) {
        inst.key.attn.row = shape::GQA[1];
        inst.key.attn.n_head = 32;
        inst.key.attn.n_kv = 4;
        inst.key.attn.rot = 64;
    }

    /// A delta layer of the given axes.
    fn delta(inst: &mut Instance, value_heads: u32, head_cols: u32, conv_taps: u32) {
        inst.key.delta = Some(DeltaAxes {
            value_heads,
            head_cols,
            conv_taps,
        });
    }

    #[test]
    fn every_registered_geometry_passes_the_check() {
        for inst in REGISTERED {
            assert_eq!(Geometry::of(inst).check(), Ok(()), "{}", inst.name);
        }
        // A shape no row registers, with the axes the kernels take free.
        let mut inst = QWEN35MOE_A3B;
        inst.key.hidden = 3072;
        moe(&mut inst, |m| m.expert_ff = 1024);
        inst.key.attn.rot = 32;
        assert_eq!(Geometry::of(&inst).check(), Ok(()));
    }

    #[test]
    fn a_geometry_a_kernel_cannot_take_is_refused_by_name() {
        // One case a clause of each kernel family's `holds`, each moving the
        // axes of Qwen3.6's row (hidden 2048, expert ff 512, 16 heads of 256
        // over 2, rope 64, 32 delta heads of 128 and 4 taps) to the named one.
        // PIN(2026-10-31): lane-held fold factors drop the key-head divisor and factor-memory
        // guards and the three-key-head case; the attention_geom tests accept 24/3 and 128/16.
        let cases: [Case; 14] = [
            (|i| i.key.hidden = 3136, ("router", "hidden", "3136")),
            (
                |i| {
                    moe(i, |m| {
                        m.router = shape::router_row(RouterBody::BiasedSigmoid, 256);
                    });
                },
                ("router", "router rule", "Sigmoid scores, bias yes"),
            ),
            (
                |i| moe(i, |m| m.expert_ff = 700),
                ("experts", "expert ff", "700"),
            ),
            (|i| attention(i, 64, 16), ("attention", "head", "64")),
            (
                |i| i.key.attn.row.value = 192,
                ("attention", "value width", "192"),
            ),
            (
                |i| i.key.attn.n_head = 15,
                ("attention", "query heads", "15"),
            ),
            (
                |i| i.key.attn.n_head = 32,
                ("attention", "query heads a key head", "16"),
            ),
            (|i| i.key.attn.rot = 128, ("attention", "rope dims", "128")),
            (|i| i.key.attn.rot = 33, ("attention", "rope dims", "33")),
            (
                window_row,
                (
                    "attention",
                    "flash row",
                    "window yes, sinks yes, value scale no",
                ),
            ),
            (|i| delta(i, 32, 128, 5), ("delta", "conv taps", "5")),
            // PIN(2026-10-28): the 48-value-head case is now an axis; the odd count is its refusal.
            (|i| delta(i, 33, 128, 4), ("delta", "value heads", "33")),
            (|i| delta(i, 0, 128, 4), ("delta", "value heads", "0")),
            (|i| delta(i, 32, 64, 4), ("delta", "head columns", "64")),
        ];
        let mut wrong = Vec::new();
        for (moved, (kernel, axis, value)) in cases {
            let mut inst = QWEN35MOE_A3B;
            moved(&mut inst);
            match Geometry::of(&inst).check() {
                Err(got)
                    if got.kernel == kernel && got.axis == axis && got.value.starts_with(value) => {
                }
                got => wrong.push(format!("{kernel}/{axis} {value}: {got:?}")),
            }
        }
        assert!(
            wrong.is_empty(),
            "not refused as named:\n{}",
            wrong.join("\n")
        );
        let mut inst = QWEN35MOE_A3B;
        inst.key.hidden = 3136;
        assert_eq!(
            Geometry::of(&inst).check().unwrap_err().to_string(),
            "router: hidden is 3136; it must be a multiple of 1024 (a float4 a lane over 256 threads)"
        );
    }

    #[test]
    fn a_dense_ffn_is_one_slot_of_its_width_and_launches_no_router() {
        let dense = |hidden: u32, ff: u32| {
            let mut inst = QWEN35MOE_A3B;
            inst.key.hidden = hidden;
            inst.key.ffn = FfnAxes::Dense { ff };
            inst
        };
        // Qwen3.8-27B's FFN: 17,408 values over a residual of 5,120.
        let inst = dense(5120, 17_408);
        let g = Geometry::of(&inst);
        assert_eq!(g.check(), Ok(()));
        let e = g.experts;
        assert_eq!(
            (
                e.slots,
                e.routed_ids,
                e.shared_slot,
                e.has_shared,
                e.k,
                e.dense
            ),
            (1, 1, 1, false, 17_408, true)
        );
        // The router's kernels are in the program at Qwen3.6's axes and launch
        // nowhere: the header's router rows are the unlaunched ones.
        let built = QWEN35MOE_A3B.router_built();
        assert_eq!(inst.router_built(), built);
        assert_eq!(g.router.experts, built.router.experts as usize);
        assert_eq!(g.router.top_k_max, built.top_k as usize);
        assert_eq!(g.router.k, 5120);
        // A width the fold's blocks do not divide is refused by name.
        let got = Geometry::of(&dense(5120, 17_400)).check();
        let Err(why) = &got else {
            panic!("a dense FFN of 17400 values: {got:?}");
        };
        assert_eq!(
            (why.kernel, why.axis, why.value.as_str()),
            ("experts", "expert ff", "17400")
        );
    }

    #[test]
    fn delta_value_heads_are_an_axis() {
        // 48 value heads over 16 key heads (three to a key head) and 64 over
        // 16 hold beside Qwen3.6's 32: the step's grid, its partials and its
        // prepared row follow the value heads.
        for value_heads in [32u32, 48, 64] {
            let mut inst = QWEN35MOE_A3B;
            delta(&mut inst, value_heads, 128, 4);
            let g = Geometry::of(&inst);
            assert_eq!(g.check(), Ok(()), "{value_heads} value heads");
            let (d, n_v, n_k) = (g.delta, value_heads as usize, 16);
            assert_eq!(
                n_v % n_k,
                0,
                "{value_heads}: key heads divide the value heads"
            );
            assert_eq!(d.groups, n_v * geometry::PARTIALS_PER_HEAD, "{n_v}: groups");
            assert_eq!(d.partials_stride, d.groups, "{n_v}: partials");
            assert_eq!(d.group_threads, 128, "{n_v}: threads a group");
            // The step's buffers are the Mac side's (`kernels::delta`).
            #[cfg(target_os = "macos")]
            {
                assert_eq!(
                    d.channels(n_k),
                    Some((2 * n_k + n_v) * 128),
                    "{n_v}: channels"
                );
                let prepared = d.prepared_elems(n_k).expect("the prepared row fits");
                assert_eq!(
                    prepared % 4,
                    0,
                    "{n_v}: a prepared row is whole 16-byte words"
                );
            }
        }
    }

    #[test]
    fn each_geometry_is_its_row() {
        for inst in REGISTERED {
            let g = Geometry::of(inst);
            let key = &inst.key;
            let name = inst.name;
            assert_eq!(g.instance, *inst, "{name}: the instance");
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
            assert_eq!(a.windowed, key.attn.row.window, "{name}: windowed");
            assert_eq!(a.head_gate, key.attn.head_gate, "{name}: head gate");
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
            let moe = key.routed().expect("a registered row is routed");
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
            assert!(!e.dense, "{name}: a routed row is not dense");

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
