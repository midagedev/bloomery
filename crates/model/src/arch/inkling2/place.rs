//! An Inkling 2 file's placement inputs: the family's half of the path from the file's headers to
//! a plan. The tensors' roles, the typed description and the trunk's stores over its layer visits
//! ([`place_hybrid::KvLayout::of_visits`]) are read once ([`Inputs::describe`];
//! [`Inputs::read`] refuses a file whose features no program runs, every one listed — the
//! coverage list is the work queue of the rounds that build one). The plan itself
//! ([`Inputs::plan`]) is the Inkling 2 body's ([`Program::Inkling2Body`]): every routed stack of a
//! type the body's card leg runs goes to the card by the expert rule beside the dense weights,
//! the visits' K and V and their convolution states — the weights once per stored layer (a
//! looped layer's passes share them: the file holds one set), the stores once per visit — and,
//! beside them on the card, the prompt pass's buffers ([`pass_bytes`] of [`pass_shape`] at
//! [`pass_rows`]), the one formula the load holds the pass it makes to.

use gguf::{GgmlType, Split};
use models::{LoopSpec, ModelSpec, visits};

use super::hparams::{Hparams, Loop};
use super::names::{self, Inp};
use super::spec;
use crate::arch::coverage;
use crate::arch::place_hybrid::{self, RingForms};
use crate::placement::accepts::{Program, routed_rule};
use crate::placement::kernels::wide_act_bytes;
use crate::placement::{
    Machine, ModelTensors, PlacementError, Plan, PlanLevers, Unimplemented, plan_routed_reserving,
};

pub use crate::arch::place_hybrid::KvLayout;

/// What a plan of an Inkling 2 file is made from: every tensor with its role, the typed
/// description the coverage check reads, and the trunk's stores over its layer visits.
#[derive(Clone, Debug)]
pub struct Inputs {
    /// The file's hyperparameters, carried for the body and never read here.
    pub hp: Hparams,
    /// Every tensor with its role, and the model's layer and expert counts.
    pub model: ModelTensors,
    /// The typed model description.
    pub spec: ModelSpec,
    /// The trunk's stores, a visit a slot ([`KvLayout::of_visits`]).
    pub kv: KvLayout,
}

impl Inputs {
    /// `split`'s tensors' roles, then its description, then its stores, the first that fails being
    /// the error.
    pub fn describe(split: &Split) -> Result<Inputs, PlacementError> {
        let read = spec::read(split)?;
        let hp = Hparams::read(split)?;
        let kv = KvLayout::of_visits(&read.spec)?;
        Ok(Inputs {
            hp,
            model: read.tensors,
            spec: read.spec,
            kv,
        })
    }

    /// [`Inputs::describe`] with the refusals: `Err` for a file with a feature no program runs,
    /// every one listed.
    pub fn read(split: &Split) -> Result<Inputs, PlacementError> {
        let inputs = Inputs::describe(split)?;
        let missing = inputs.unimplemented();
        if missing.is_empty() {
            Ok(inputs)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// Every feature of the file no program in this tree runs, as [`coverage::check`] lists them.
    /// Empty for a file a program runs.
    pub fn unimplemented(&self) -> Vec<Unimplemented> {
        coverage::check(&self.spec, &self.model)
    }

    /// `self`, or — for a file with a feature the engine does not run —
    /// [`PlacementError::Unimplemented`] with every one listed.
    pub fn runnable(self) -> Result<Inputs, PlacementError> {
        let missing = self.unimplemented();
        if missing.is_empty() {
            Ok(self)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// The placement of the file on `machine` at `ctx_max` positions under the placement's
    /// `levers`: the routed stacks by the body's rule ([`routed_rule`] of
    /// [`Program::Inkling2Body`], each stack by its own type, so a fused gate·up stack plans whole)
    /// through the expert rule, the dense tensors where their roles say, the visits' stores on the
    /// cards that run their layers, and the prompt pass's buffers kept beside them on every card
    /// ([`Inputs::pass_reserve`]).
    pub fn plan<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlacementError> {
        let rule = routed_rule(Program::Inkling2Body);
        let reserve = self.pass_reserve(ctx_max)?;
        plan_routed_reserving(
            &self.model,
            machine,
            ctx_max,
            &self.kv,
            levers,
            rule,
            reserve,
        )
    }

    /// The card bytes of the prompt pass a load at `ctx_max` positions makes: [`pass_bytes`] of
    /// the file's [`pass_shape`] at [`pass_rows`]. A site tensor missing or of a type no site
    /// entry runs is refused by name.
    pub fn pass_reserve(&self, ctx_max: u64) -> Result<u64, PlacementError> {
        let shape = pass_shape(&self.hp, |name| {
            self.model
                .tensors
                .iter()
                .find(|t| t.name == name)
                .map(|t| t.ty)
        })?;
        let ctx = usize::try_from(ctx_max).unwrap_or(usize::MAX);
        Ok(pass_bytes(&shape, pass_rows(ctx)))
    }
}

/// The most rows one prompt pass runs: the prompt batch a window ring keeps room for beside its
/// window ([`runtime::stores::PROMPT_BATCH`]), so a pass's appended keys never overwrite a key one
/// of its rows reads.
pub const PASS_ROWS: usize = 512;
const _: () = assert!(PASS_ROWS as u64 == runtime::stores::PROMPT_BATCH);

/// The rows a load's pass is cut for at `ctx` positions: [`PASS_ROWS`], or the context where it is
/// shorter.
#[must_use]
pub fn pass_rows(ctx: usize) -> usize {
    PASS_ROWS.min(ctx)
}

/// The most blocks a walk of `hp`'s visits banks at once: the stem's and the coda's block heads
/// plus, at the core, the prelude's and one pass's (every pass past the first restarts from the
/// prelude's banks). The step's bank and the pass's are cut by it.
#[must_use]
pub fn banked_most(hp: &Hparams) -> usize {
    banks_of(hp.n_layer, hp.block, hp.loop_)
}

/// [`banked_most`] of `n_layer` stored layers in blocks of `block` around the loop `lp`.
fn banks_of(n_layer: usize, block: usize, lp: Option<Loop>) -> usize {
    let core = lp.filter(|lp| lp.passes > 1);
    let spec = lp.map(|lp| LoopSpec {
        core_start: lp.core_start as u32,
        core_end: lp.core_end as u32,
        passes: lp.passes as u32,
        gate_rank: lp.gate_rank as u32,
    });
    let (mut pushed, mut prelude, mut most) = (0usize, 0usize, 0usize);
    for v in visits(n_layer, spec) {
        let l = v.layer as usize;
        if let Some(lp) = core
            && l == lp.core_start
        {
            if v.pass == 0 {
                prelude = pushed;
            } else {
                pushed = prelude;
            }
        }
        if l.is_multiple_of(block) {
            pushed += 1;
        }
        most = most.max(pushed);
    }
    most
}

/// The widths and forms the prompt pass's buffers are cut from: what every visit shares (the
/// body refuses layers whose widths differ), the bank's rows ([`banked_most`]), whether the core
/// loops (the message bank, the controllers' scratch and the pass edge's rows), and the file
/// types of the projection sites by the width they read: the model's (q, k, v, the relative rows,
/// the sinks, the gate and a windowless layer's conv taps) and the attention's (the output).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassShape {
    pub n: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head: usize,
    pub rel: usize,
    pub taps: usize,
    pub ff: usize,
    pub n_used: usize,
    pub n_expert: usize,
    pub banks: usize,
    pub rank: usize,
    pub looped: bool,
    pub inner: Vec<GgmlType>,
    pub outer: Vec<GgmlType>,
    /// The forms the route GEMMs read the routed rows in, over every layer's stacks: a role's
    /// 32-value form where every layer's type reads it, else the grouped GEMM's.
    pub forms: RingForms,
}

/// The shape of `hp`'s prompt pass, its site types read by name through `ty_of` (the plan's
/// tensor list or the file's header). A site tensor `ty_of` does not find, or a routed stack
/// whose types no route GEMM reads, is refused by name.
pub fn pass_shape(
    hp: &Hparams,
    ty_of: impl Fn(&str) -> Option<GgmlType>,
) -> Result<PassShape, PlacementError> {
    let first = |v: &[usize], key: &str| {
        v.first().copied().ok_or_else(|| PlacementError::Metadata {
            key: key.to_string(),
            detail: "names no visit's value".to_string(),
        })
    };
    let ty = |name: String| {
        ty_of(&name).ok_or_else(|| PlacementError::Tensor {
            name,
            detail: "is not in the file: a prompt pass's projection reads it".to_string(),
        })
    };
    let (mut inner, mut outer) = (Vec::new(), Vec::new());
    let add = |set: &mut Vec<GgmlType>, t: GgmlType| {
        if !set.contains(&t) {
            set.push(t);
        }
    };
    let mut forms = RingForms::BLOCKS32;
    for l in 0..hp.n_layer {
        let blk = |s: &str| format!("blk.{l}.{s}");
        let mut sites = vec![
            blk("attn_q.weight"),
            blk("attn_k.weight"),
            blk("attn_v.weight"),
            names::attn_rel(l),
            names::attn_sink_proj(l),
            blk("attn_gate.weight"),
        ];
        if !hp.windowed[l] {
            sites.push(names::inp(l, Inp::ConvK));
            sites.push(names::inp(l, Inp::ConvV));
        }
        for s in sites {
            add(&mut inner, ty(s)?);
        }
        add(&mut outer, ty(blk("attn_output.weight"))?);
        let (fused, down) = (names::ffn_gate_up_exps(l), blk("ffn_down_exps.weight"));
        let f =
            RingForms::of(ty(fused.clone())?, ty(down)?).ok_or_else(|| PlacementError::Tensor {
                name: fused,
                detail: "is of a type no route GEMM reads: the prompt pass's experts run there"
                    .to_string(),
            })?;
        forms.gate_up32 &= f.gate_up32;
        forms.down32 &= f.down32;
    }
    Ok(PassShape {
        n: hp.n_embd,
        n_head: first(&hp.n_head, "attention.head_count")?,
        n_kv: first(&hp.n_head_kv, "attention.head_count_kv")?,
        head: hp.head_dim,
        rel: hp.d_rel,
        taps: hp.conv,
        ff: first(&hp.expert_ff, "expert_feed_forward_length")?,
        n_used: first(&hp.n_used, "expert_used_count")?,
        n_expert: hp.n_expert,
        banks: banked_most(hp),
        rank: hp.loop_.map_or(0, |lp| lp.gate_rank),
        looped: hp.loop_.is_some_and(|lp| lp.passes > 1),
        inner,
        outer,
        forms,
    })
}

/// The card bytes of the buffers of a prompt pass of `rows` rows over `s`, four bytes a value:
///
/// - f32 rows of the model's width: the stream, the attention mix's three inputs, the three
///   controllers' outputs, the three normed inputs, the output projection (the embedding's rows
///   before their norm), the attention block's output, the feed-forward's input, its mix (the
///   head's mix after the last visit), its normed input, the routed sum and the post conv's
///   output — fourteen — and, where the core loops, the prelude's stream, the bank's last row and
///   the four message rows; the bank's rows ([`PassShape::banks`] of them) and, where the core
///   loops, the message bank's as many;
/// - f32 rows of the attention's widths: the query, the gate, the attention and its gated rows
///   (`n_head · head` each), the key and value before and after their convs (`n_kv · head` each),
///   the relative rows, the sinks, the two conv taps' rows (`n_kv · taps` each), each row's
///   factor and the windowed visits' unit factor; where the core loops, the controllers' scratch
///   (three modules of `rank`);
/// - the router's scores (`n_expert` a row) and weights (`n_used` a row) and the slots' down rows
///   (`n_used · n` a row);
/// - u32 words: the ids, the positions, the live key counts, the ring slots, the first position
///   word and the router's picks (`n_used` a row);
/// - the forms: the wide forms of the model's and the attention's site widths
///   ([`wide_act_bytes`] of their types), the one-expert route table
///   ([`place_hybrid::route_bytes`]) and the route block's buffers over the experts
///   ([`place_hybrid::ring_pass_bytes_in`] over every expert: the rows' and the slots' forms,
///   the gate and up rows, the route table).
///
/// The one formula of those bytes: the plan reserves them ([`Inputs::pass_reserve`]) and the load
/// refuses a pass that makes more.
#[must_use]
pub fn pass_bytes(s: &PassShape, rows: usize) -> u64 {
    let (f, u) = (size_of::<f32>(), size_of::<u32>());
    let looped = usize::from(s.looped);
    let q = s.n_head * s.head;
    let kv = s.n_kv * s.head;
    let model_rows = 14 + looped * 6 + s.banks * (1 + looped);
    let f32s = rows
        * (model_rows * s.n
            + 4 * q
            + 4 * kv
            + s.n_head * s.rel
            + s.n_head
            + 2 * s.n_kv * s.taps
            + 2
            + looped * 3 * s.rank
            + s.n_expert
            + s.n_used
            + s.n_used * s.n);
    let u32s = rows * (4 + s.n_used) + 1;
    let (cols, n, q64) = (rows as u64, s.n as u64, q as u64);
    let wide = wide_act_bytes(s.inner.iter().copied(), cols, n)
        + wide_act_bytes(s.outer.iter().copied(), cols, q64);
    let route = place_hybrid::route_bytes(rows);
    let experts = place_hybrid::ring_pass_bytes_in(rows, s.n_used, s.n, s.ff, s.n_expert, s.forms);
    (f32s * f + u32s * u + route + experts) as u64 + wide
}

#[cfg(test)]
mod tests {
    use models::LoopSpec;

    use super::super::hparams::Loop;
    use super::super::hparams::tests::{keys, tensors};
    use super::{Inputs, PassShape, banks_of, pass_bytes};
    use crate::arch::place_hybrid::RingForms;
    use crate::arch::synthetic::{V, header_shaped};
    use crate::placement::{KvBytes, PlacementError, PlanLevers, workstation};
    use gguf::GgmlType;

    /// The small header's file, opened; each test names its own file (the tests run in threads of
    /// one process). `typed` names a file whose routed stacks are Q8_0 (a type the host stack
    /// runs): the header-only writer's F32 default has no host kernel, so a plan of it is
    /// refused by the stack's name.
    fn opened(name: &str) -> gguf::Split {
        opened_typed(name, false)
    }

    /// [`opened`] with the routed stacks in Q8_0.
    fn opened_typed(name: &str, typed: bool) -> gguf::Split {
        let global = [
            ("tokenizer.ggml.pre", V::Str("inkling2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = if typed {
            use crate::arch::synthetic::Ty;
            let tensors: Vec<(String, Vec<u64>, Ty)> = tensors(true)
                .into_iter()
                .map(|(n, d)| {
                    let ty = if n.ends_with("ffn_gate_up_exps.weight")
                        || n.ends_with("ffn_down_exps.weight")
                    {
                        Ty::Q8_0
                    } else {
                        Ty::F32
                    };
                    (n, d, ty)
                })
                .collect();
            crate::arch::synthetic::header_typed(name, "inkling2", &keys(true), &global, &tensors)
        } else {
            header_shaped(name, "inkling2", &keys(true), &global, &tensors(true))
        };
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let _ = std::fs::remove_file(&path);
        split
    }

    /// The described inputs of the small header: 6 stored layers, the core [2, 4) three passes,
    /// 10 visits — layers 2 and 3 twice more each — of 2 key heads of 16 (a K and V row of 128 B
    /// a position), every visit's conv state the widest layer's channels (2·32 + 2·64)·3 f32 =
    /// 2,304 B [derived from the header's widths].
    fn inputs(name: &str) -> Inputs {
        Inputs::describe(&opened(name)).expect("the synthetic header describes")
    }

    /// [`inputs`] of the typed header (the routed stacks Q8_0).
    fn typed_inputs(name: &str) -> Inputs {
        Inputs::describe(&opened_typed(name, true)).expect("the synthetic header describes")
    }

    /// The description's loop is the file's, and the stores are one a visit: the core's two
    /// layers hold their three visits' rows and states (3·(rows·128 + 2,304)), the other four
    /// layers one (rows·128 + 2,304), the windowed layers' rows on the ring. Mutant:
    /// the visits not counted (a core layer sized as its stem's).
    #[test]
    fn the_stores_are_one_a_visit() {
        let inputs = inputs("inkling2-place-visits");
        assert_eq!(
            inputs.spec.loop_,
            Some(LoopSpec {
                core_start: 2,
                core_end: 4,
                passes: 3,
                gate_rank: 8,
            })
        );
        let (row, conv) = (2 * 2 * 16 * 2, (2 * 32 + 2 * 64) * 3 * 4);
        assert_eq!((row, conv), (128, 2_304));
        let kv = &inputs.kv;
        // Layer 2 is windowless (its visits hold every position), layer 3 windowed, both core.
        assert_eq!(kv.layer_bytes(2, 64), 3 * (64 * row + conv));
        assert_eq!(
            kv.layer_bytes(3, 64),
            3 * (runtime_rows(16, 64) * row + conv)
        );
        assert_eq!(kv.layer_bytes(1, 64), runtime_rows(16, 64) * row + conv);
        assert_eq!(
            kv.layer_bytes(4, 64),
            64 * row + conv,
            "layer 4 is windowless"
        );
        assert_eq!(kv.layer_bytes(6, 64), 0, "past the trunk");
        // At 64 positions the ring holds every position, so each of the trunk's 10 visits holds
        // 64 rows and its state.
        assert_eq!(runtime_rows(16, 64), 64);
        let whole: u64 = (0..6).map(|l| kv.layer_bytes(l, 64)).sum();
        assert_eq!(whole, 10 * (64 * row + conv));
    }

    /// The ring's row count of a window of `window` at `ctx` positions
    /// ([`runtime::stores::ring_rows`], the layout's own rule).
    fn runtime_rows(window: u64, ctx: u64) -> u64 {
        runtime::stores::ring_rows(window, ctx)
    }

    /// The body's plan of the described file holds on the gate machine with every expert on the
    /// card: the Q8_0 stacks are the card leg's, the small header's experts fit beside the dense
    /// tensors and the stores, so no stack is hosted. The read still refuses the small header
    /// through the coverage check (its widths are no kernel's). Mutant: the host-routed plan
    /// keeps every layer's experts on the host.
    #[test]
    fn the_described_file_plans_its_experts_on_the_card() {
        let inputs = typed_inputs("inkling2-place-plan");
        let machine = workstation::plan_gate(6);
        let plan = inputs
            .plan(&machine, 64, &PlanLevers::default())
            .expect("the gate machine plans the file");
        let experts = inputs.model.experts;
        assert!(
            plan.n_l.iter().all(|&n| n == experts),
            "every layer keeps its {experts} experts on the card: {:?}",
            plan.n_l
        );
        assert_eq!(plan.host.expert_bytes, 0, "no routed stack is hosted");
        match Inputs::read(&opened("inkling2-place-read")) {
            Err(PlacementError::Unimplemented(items)) => assert!(!items.is_empty()),
            other => panic!("{other:?}, not the coverage refusal"),
        }
    }

    /// The Mini's walk banks at most four blocks (the stem's, the prelude's pass's two, the
    /// coda's), every pass restarting from the prelude's one; the fixture's at most three.
    /// Mutant: a pass that keeps the blocks before it (no restart) counts eight.
    #[test]
    fn the_walk_banks_a_pass_over_the_prelude() {
        let lp = |start: usize, end: usize| Loop {
            core_start: start,
            core_end: end,
            passes: 3,
            gate_rank: 64,
        };
        assert_eq!(banks_of(24, 6, Some(lp(6, 18))), 4);
        assert_eq!(banks_of(13, 6, Some(lp(6, 12))), 3);
    }

    /// The Mini's pass of 512 rows [derived term by term from `pass_bytes`' list]: f32 values a
    /// row 28 x 2,304 (fourteen model rows, six loop rows, four bank rows twice) + 4 x 3,072 +
    /// 4 x 1,024 + 384 + 24 + 64 + 2 + 192 + 128 + 8 + 18,432 = 100,130; u32 words 12 a row and
    /// one; the forms: the q8 blocks of 32 (80 B a 64-value step) of 512 rows at 2,304 and at
    /// 3,072, 1,474,560 + 1,966,080 (the BF16 sites read the f32 rows), the one-expert table of
    /// 512 slots 4 x (512 + 2 x 9 + 2 + 512) = 4,176, the route block over 4,096 slots of 128
    /// experts 1,474,560 + 2 x 4 x 4,096 x 1,152 + 4,096 x 1,440 + 4 x (4,096 + 2 x 192 + 2) =
    /// 45,139,464: 253,675,100 B in all. Mutant: a model row left out moves it by 4,718,592.
    #[test]
    fn the_mini_pass_is_its_buffers() {
        let mini = PassShape {
            n: 2304,
            n_head: 24,
            n_kv: 8,
            head: 128,
            rel: 16,
            taps: 4,
            ff: 1152,
            n_used: 8,
            n_expert: 128,
            banks: 4,
            rank: 64,
            looped: true,
            inner: vec![GgmlType::Q8_0, GgmlType::BF16],
            outer: vec![GgmlType::Q8_0],
            forms: RingForms::BLOCKS32,
        };
        let f32s =
            512 * (28 * 2304 + 4 * 3072 + 4 * 1024 + 384 + 24 + 64 + 2 + 192 + 128 + 8 + 18_432);
        assert_eq!(f32s, 512 * 100_130);
        let forms = 3_440_640 + 4_176 + 45_139_464;
        assert_eq!(
            pass_bytes(&mini, 512),
            4 * (f32s + 512 * 12 + 1) as u64 + forms
        );
        assert_eq!(pass_bytes(&mini, 512), 253_675_100);
    }
}
