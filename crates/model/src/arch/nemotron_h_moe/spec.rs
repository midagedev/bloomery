//! The `nemotron_h_moe` reader's [`ModelSpec`]. A Mamba-2 layer and an
//! attention layer are the hybrid trunk's: a [`Mixer::Mamba2`], and grouped query
//! attention with no rope, each with no feed-forward block. A MoE layer has
//! no token mixer and a routed mixture of its own expert count and width:
//! sigmoid scores with a selection bias, the kept weights renormalized
//! (`expert_weights_norm`) and scaled (`expert_weights_scale`), squared-ReLU
//! experts with no gate inside the `moe_latent_size` latent, and one
//! squared-ReLU shared expert with no gate on the hidden row
//! (nemotron-h.cpp:282-334). A file without that key has no latent: its
//! experts act on the hidden row itself. The file's next-token blocks are
//! carried, never read. The tool calls are
//! the `<tool_call>` parser and the `<think>` span is read.

use gguf::Split;
use models::{
    Act, Arch, Ffn, LayerSpec, Mixer, ModelSpec, Moe, ReasoningFormat, Residual, Router, Score,
    Shared,
};

use super::hparams::{Hparams, Kind};
use super::roles;
use crate::arch::hybrid::{NO_BLOCK, TOOLS, mamba2, nope_gqa};
use crate::arch::{Read, chat_of, spec_u32};
use crate::placement::PlacementError;

/// `split`'s description and roles, with the key defaults the reader took.
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let chat = chat_of(split, TOOLS, Some(ReasoningFormat::THINK))?;
    let spec = spec_of(&hp, chat)?;
    Ok(Read {
        spec,
        tensors,
        defaults: hp.defaults,
    })
}

/// The description of the file `hp` was read from.
pub fn spec_of(hp: &Hparams, chat: models::ChatSpec) -> Result<ModelSpec, PlacementError> {
    let mamba = mamba2(&hp.ssm)?;
    let attn = hp.attn();
    let router = Router {
        score: Score::Sigmoid,
        bias: true,
        norm: models::RouterNorm::from_legacy_bool(hp.weights_norm),
        scale: hp.weights_scale,
        hash: false,
    };
    let shared = Shared {
        ff: spec_u32("expert_shared_feed_forward_length", hp.shared_ff)?,
        act: Act::ReluSqr,
        sigmoid_gate: false,
    };
    let latent = match hp.latent {
        Some(v) => Some(spec_u32("moe_latent_size", v)?),
        None => None,
    };
    let layer = |l: usize| -> Result<LayerSpec, PlacementError> {
        let (mixer, ffn) = match hp.kinds[l] {
            Kind::Mamba => (Mixer::Mamba2(mamba), NO_BLOCK),
            Kind::Attention => (Mixer::Gqa(nope_gqa(&attn, hp.kv[l])?), NO_BLOCK),
            Kind::Moe => {
                let x = hp.experts[l].expect("a MoE layer's experts are read");
                (
                    Mixer::None,
                    Ffn::Moe(Moe {
                        experts: spec_u32("the layer's router rows", x.n_expert)?,
                        top_k: spec_u32("expert_used_count", x.top_k)?,
                        expert_ff: spec_u32("expert_feed_forward_length", x.ff)?,
                        latent,
                        pre_exp_norm: false,
                        post_exp_norm: None,
                        act: Act::ReluSqr,
                        router,
                        shared: Some(shared),
                    }),
                )
            }
        };
        Ok(LayerSpec {
            mixer,
            ffn,
            residual: Residual::Plain,
            extras: Vec::new(),
        })
    };
    Ok(ModelSpec {
        arch: Arch::NemotronHMoe,
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        ctx_train: spec_u32("context_length", hp.n_ctx)?,
        rms_eps: hp.rms_eps,
        layers: (0..hp.n_trunk).map(layer).collect::<Result<Vec<_>, _>>()?,
        mtp: Vec::new(),
        hc: None,
        engram: None,
        encoder: None,
        logit_epilogue: None,
        chat,
    })
}

/// The host split's clauses, one owner with the real file's gate
/// (`tests/nemotron_h_moe_meta.rs`).
#[cfg(test)]
#[path = "../../../tests/common/host_split.rs"]
mod host_split;

#[cfg(test)]
mod tests {
    use models::{Act, Ffn, Mixer, RopeMode, Score};

    use super::super::names;
    use super::super::place::Experts;
    use super::Hparams;
    use super::host_split::{half_expert_budget, prefix_split};
    use crate::arch::coverage;
    use crate::arch::synthetic::{Ty, V, header_typed};
    use crate::placement::Role;

    /// Six layers: Mamba-2, MoE, attention, Mamba-2, MoE, MoE.
    const KINDS: [char; 6] = ['m', 'e', 'a', 'm', 'e', 'e'];
    const EMBD: u64 = 32;
    const HEADS: u64 = 4;
    const KV: i32 = 2;
    const HEAD: u64 = 8;
    const INNER: u64 = 32;
    const MAMBA_HEADS: u64 = 4;
    const GROUPS: u64 = 4;
    const STATE: u64 = 128;
    const CONV: u64 = 4;
    const LATENT: u64 = 16;
    const SHARED: u64 = 24;
    /// Each MoE layer's (experts, width): two shapes share a count, two a
    /// width. The largest count is `expert_count`.
    const SHAPES: [(u64, u64); 3] = [(4, 16), (6, 32), (6, 16)];
    const TOP_K: i32 = 2;

    /// A synthetic file's layers and widths: [`SMALL`], the reader's tests'
    /// header, and [`FIXTURE`], the gate fixture's layout at Bolt's shapes.
    struct Geo {
        kinds: &'static [char],
        /// Each MoE layer's (experts, width), in layer order.
        shapes: &'static [(u64, u64)],
        embd: u64,
        vocab: u64,
        heads: u64,
        kv: i32,
        head: u64,
        inner: u64,
        mamba_heads: u64,
        groups: u64,
        /// `moe_latent_size`: the latent the routed experts run in, `None`
        /// where the file has none and its experts act on the hidden row.
        latent: Option<u64>,
        shared: u64,
        top_k: i32,
        /// The Bolt file's types (NVFP4 Mamba projections, Q8_0 attention,
        /// latent pair and globals, IQ2_XXS experts); f32 throughout if not.
        quant: bool,
    }

    /// The reader's tests' header: [`KINDS`] at the small widths, all f32.
    const SMALL: Geo = Geo {
        kinds: &KINDS,
        shapes: &SHAPES,
        embd: EMBD,
        vocab: EMBD,
        heads: HEADS,
        kv: KV,
        head: HEAD,
        inner: INNER,
        mamba_heads: MAMBA_HEADS,
        groups: GROUPS,
        latent: Some(LATENT),
        shared: SHARED,
        top_k: TOP_K,
        quant: false,
    };

    /// The same header without a latent: the experts act on the hidden row
    /// itself and the file holds no latent pair.
    const NO_LATENT: Geo = Geo {
        latent: None,
        ..SMALL
    };

    /// The gate fixture's layout (`fixture::LAYER_MAP`: Mamba-2, MoE of 240,
    /// attention of 32 heads over 2, MoE of 496) at Bolt's types, routers,
    /// group 16, head 128, state 128 and top 9; the widths the body does not
    /// fix (hidden, inner, latent, the experts' and the shared expert's) at
    /// the least the block sizes take, so the header stays small.
    const FIXTURE: Geo = Geo {
        kinds: &['m', 'e', 'a', 'e'],
        shapes: &[(240, 256), (496, 512)],
        embd: 256,
        vocab: 64,
        heads: 32,
        kv: 2,
        head: 128,
        inner: 256,
        mamba_heads: 4,
        groups: 1,
        latent: Some(256),
        shared: 256,
        top_k: 9,
        quant: true,
    };

    impl Geo {
        /// Each layer's (experts, width), `None` off the MoE layers.
        fn layer_shapes(&self) -> Vec<Option<(u64, u64)>> {
            let mut moe = self.shapes.iter();
            self.kinds
                .iter()
                .map(|&k| (k == 'e').then(|| *moe.next().expect("a shape a MoE layer")))
                .collect()
        }

        fn per_layer(&self, f: impl Fn(char, Option<(u64, u64)>) -> i32) -> V {
            V::I32s(
                self.kinds
                    .iter()
                    .zip(self.layer_shapes())
                    .map(|(&k, s)| f(k, s))
                    .collect(),
            )
        }

        fn keys(&self) -> Vec<(&'static str, V)> {
            let width = |_, s: Option<(u64, u64)>| s.map_or(0, |(_, w)| w as i32);
            let n_expert = self.shapes.iter().map(|&(n, _)| n).max().unwrap_or(0);
            let (kv, top_k) = (self.kv, self.top_k);
            let mut keys = vec![
                ("block_count", V::U32(self.kinds.len() as u32)),
                ("context_length", V::U32(128)),
                ("embedding_length", V::U32(self.embd as u32)),
                ("feed_forward_length", self.per_layer(width)),
                ("attention.head_count", V::U32(self.heads as u32)),
                (
                    "attention.head_count_kv",
                    self.per_layer(|k, _| if k == 'a' { kv } else { 0 }),
                ),
                ("attention.layer_norm_rms_epsilon", V::F32(1e-5)),
                ("attention.layer_norm_epsilon", V::F32(1e-5)),
                ("expert_count", V::U32(n_expert as u32)),
                ("expert_group_count", V::U32(1)),
                ("expert_group_used_count", V::U32(1)),
                ("expert_gating_func", V::U32(2)),
                ("vocab_size", V::U32(self.vocab as u32)),
                ("rope.dimension_count", V::U32(self.head as u32)),
                ("ssm.conv_kernel", V::U32(CONV as u32)),
                ("ssm.state_size", V::U32(STATE as u32)),
                ("ssm.group_count", V::U32(self.groups as u32)),
                ("ssm.inner_size", V::U32(self.inner as u32)),
                ("ssm.time_step_rank", V::U32(self.mamba_heads as u32)),
                ("attention.key_length", V::U32(self.head as u32)),
                ("attention.value_length", V::U32(self.head as u32)),
                ("expert_feed_forward_length", self.per_layer(width)),
                (
                    "expert_used_count",
                    self.per_layer(|k, _| if k == 'e' { top_k } else { 0 }),
                ),
                (
                    "expert_shared_feed_forward_length",
                    V::U32(self.shared as u32),
                ),
                ("expert_shared_count", V::U32(1)),
                ("expert_weights_norm", V::Bool(true)),
                ("expert_weights_scale", V::F32(5.0)),
            ];
            // The latent, where the file has one.
            if let Some(latent) = self.latent {
                keys.push(("moe_latent_size", V::U32(latent as u32)));
            }
            keys
        }

        /// A tensor's type: `quant`'s for a quantized family, f32 otherwise.
        fn ty(&self, quant: Ty) -> Ty {
            if self.quant { quant } else { Ty::F32 }
        }

        /// Layer `l`'s tensors for `kind`, at `shape` on a MoE layer.
        fn layer(&self, l: usize, kind: char, shape: Option<(u64, u64)>, out: &mut Tensors) {
            let e = self.embd;
            let mut one = |stem: &str, dims: Vec<u64>, ty: Ty| {
                out.push((format!("blk.{l}.{stem}"), dims, ty));
            };
            let (q8, nv, iq) = (self.ty(Ty::Q8_0), self.ty(Ty::Nvfp4), self.ty(Ty::Iq2Xxs));
            one(names::ATTN_NORM, vec![e], Ty::F32);
            match kind {
                'm' => {
                    let (inner, heads, groups) = (self.inner, self.mamba_heads, self.groups);
                    let gs = 2 * groups * STATE;
                    let (proj, conv) = (2 * inner + gs + heads, inner + gs);
                    one(names::SSM_A, vec![1, heads], Ty::F32);
                    one(names::SSM_D, vec![1, heads], Ty::F32);
                    one(names::SSM_DT_BIAS, vec![heads], Ty::F32);
                    one(names::SSM_CONV, vec![CONV, conv], Ty::F32);
                    one(names::SSM_CONV_BIAS, vec![conv], Ty::F32);
                    one(names::SSM_IN, vec![e, proj], nv);
                    one(names::SSM_IN_SCALE, vec![1], Ty::F32);
                    one(names::SSM_NORM, vec![inner / groups, groups], Ty::F32);
                    one(names::SSM_OUT, vec![inner, e], nv);
                    one(names::SSM_OUT_SCALE, vec![1], Ty::F32);
                }
                'a' => {
                    let (h, kv) = (self.heads * self.head, self.kv as u64 * self.head);
                    one(names::ATTN_Q, vec![e, h], q8);
                    one(names::ATTN_K, vec![e, kv], q8);
                    one(names::ATTN_V, vec![e, kv], q8);
                    one(names::ATTN_OUTPUT, vec![h, e], q8);
                }
                'e' => {
                    let (n, w) = shape.expect("a MoE layer's shape");
                    // The width the experts run in: the latent, or the hidden
                    // row where the file has no latent.
                    let lat = self.latent.unwrap_or(self.embd);
                    let sh = self.shared;
                    one(names::FFN_GATE_INP, vec![e, n], Ty::F32);
                    one(names::EXP_PROBS_B, vec![n], Ty::F32);
                    if let Some(latent) = self.latent {
                        one(names::FFN_LATENT_DOWN, vec![e, latent], q8);
                        one(names::FFN_LATENT_UP, vec![latent, e], q8);
                    }
                    one(names::FFN_UP_EXPS, vec![lat, w, n], iq);
                    one(names::FFN_DOWN_EXPS, vec![w, lat, n], iq);
                    one(names::FFN_UP_SHEXP, vec![e, sh], iq);
                    one(names::FFN_DOWN_SHEXP, vec![sh, e], iq);
                }
                _ => panic!("kind {kind}"),
            }
        }

        fn tensors(&self) -> Tensors {
            let q8 = self.ty(Ty::Q8_0);
            let mut out = vec![
                (names::token_embd(), vec![self.embd, self.vocab], q8),
                (names::output(), vec![self.embd, self.vocab], q8),
                (names::output_norm(), vec![self.embd], Ty::F32),
            ];
            for (l, (&kind, shape)) in self.kinds.iter().zip(self.layer_shapes()).enumerate() {
                self.layer(l, kind, shape, &mut out);
            }
            out
        }
    }

    type Tensors = Vec<(String, Vec<u64>, Ty)>;

    fn keys() -> Vec<(&'static str, V)> {
        SMALL.keys()
    }

    fn tensors() -> Tensors {
        SMALL.tensors()
    }

    /// The no-latent header's keys: the same file without `moe_latent_size`.
    fn nolatent_keys() -> Vec<(&'static str, V)> {
        NO_LATENT.keys()
    }

    /// The no-latent header's tensors: the experts at the hidden width and no
    /// latent pair.
    fn nolatent_tensors() -> Tensors {
        NO_LATENT.tensors()
    }

    fn load(tag: &str, kv: Vec<(&str, V)>, tensors: &Tensors) -> Result<crate::arch::Read, String> {
        let global = [("tokenizer.ggml.pre", V::Str("pixtral"))];
        let path = header_typed(tag, "nemotron_h_moe", &kv, &global, tensors);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let read = crate::arch::spec(&split).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        read
    }

    /// `keys()` with `key` set to `value`, or dropped when `None`.
    fn with(key: &str, value: Option<V>) -> Vec<(&'static str, V)> {
        let mut kv = keys();
        let i = kv.iter().position(|(k, _)| *k == key).expect(key);
        match value {
            Some(v) => kv[i].1 = v,
            None => {
                kv.remove(i);
            }
        }
        kv
    }

    fn refused(tag: &str, kv: Vec<(&str, V)>, tensors: &Tensors, want: &[&str]) {
        let err = load(tag, kv, tensors).expect_err(tag);
        for w in want {
            assert!(err.contains(w), "{tag}: want {w:?} in: {err}");
        }
    }

    /// A MoE layer is a latent mixture of its own expert count and width,
    /// no token mixer, squared-ReLU experts and shared expert with no gate,
    /// a sigmoid router with the selection bias, renormalized and scaled;
    /// the Mamba-2 and attention layers are `nemotron_h`'s. The latent pair
    /// is a router-role matrix and `attn_norm` the pre-FFN norm on a MoE
    /// layer. A MoE layer read at `expert_count`'s width fails this.
    #[test]
    fn a_moe_layer_is_a_latent_mixture_of_its_own_shape() {
        let read = load("nhm-ok", keys(), &tensors()).expect("the header reads");
        let s = &read.spec;
        assert_eq!(s.arch, models::Arch::NemotronHMoe);
        assert!(read.defaults.is_empty(), "{:?}", read.defaults);
        let mut moe = SHAPES.iter();
        for (l, &kind) in KINDS.iter().enumerate() {
            match (kind, &s.layers[l].mixer, &s.layers[l].ffn) {
                ('m', Mixer::Mamba2(m), Ffn::Dense { ff: 0, .. }) => {
                    assert_eq!((m.n_heads, m.n_groups, m.d_state), (4, 4, 128));
                }
                ('a', Mixer::Gqa(q), Ffn::Dense { ff: 0, .. }) => {
                    assert_eq!((q.heads, q.kv_heads, q.head_dim), (4, 2, 8));
                    assert_eq!(q.rope.mode, RopeMode::None);
                }
                ('e', Mixer::None, Ffn::Moe(m)) => {
                    let &(n, w) = moe.next().unwrap();
                    assert_eq!((m.experts, m.expert_ff), (n as u32, w as u32), "layer {l}");
                    assert_eq!((m.top_k, m.latent), (2, Some(LATENT as u32)));
                    assert_eq!(m.act, Act::ReluSqr);
                    assert_eq!(m.router.score, Score::Sigmoid);
                    assert!(m.router.bias && !m.router.hash);
                    assert_eq!(m.router.norm, models::RouterNorm::SelectedSum);
                    assert_eq!(m.router.scale, 5.0);
                    let sh = m.shared.expect("a shared expert");
                    assert_eq!((sh.ff, sh.act, sh.sigmoid_gate), (24, Act::ReluSqr, false));
                }
                other => panic!("layer {l} ({kind}): {other:?}"),
            }
        }
        let t = &read.tensors;
        assert_eq!((t.layers, t.experts, t.experts_used), (6, 6, 2));
        let role = |name: String| {
            t.tensors
                .iter()
                .find(|x| x.name == name)
                .map(|x| x.role)
                .expect("a classified tensor")
        };
        assert_eq!(
            role(format!("blk.1.{}", names::FFN_LATENT_DOWN)),
            Role::Router
        );
        assert_eq!(
            role(format!("blk.1.{}", names::FFN_LATENT_UP)),
            Role::Router
        );
        assert_eq!(
            role(format!("blk.1.{}", names::FFN_UP_EXPS)),
            Role::RoutedExperts
        );
        assert_eq!(
            role(format!("blk.1.{}", names::FFN_UP_SHEXP)),
            Role::SharedExpert
        );
        assert_eq!(role(format!("blk.1.{}", names::ATTN_NORM)), Role::FfnNorm);
        assert_eq!(role(format!("blk.0.{}", names::ATTN_NORM)), Role::Attention);
        assert_eq!(
            role(format!("blk.0.{}", names::SSM_IN_SCALE)),
            Role::Attention
        );
    }

    /// The latent MoE body runs a MoE layer at the shapes it is built for:
    /// on the small header the coverage check names each part at a shape it
    /// is not — the routers at 4 and 6 experts, top 2, a 16-value latent, a
    /// shared expert of 24, the group-2 head-8 flash — at the layers that
    /// carry each, and never the program, the squared-ReLU experts, the
    /// per-layer shapes or the Mamba-2 mixer, which it runs. Emitting a MoE
    /// layer's needs as a dense layer's, or dropping one of them, fails this.
    #[test]
    fn coverage_names_the_latent_moe_parts() {
        let read = load("nhm-cover", keys(), &tensors()).expect("the header reads");
        let items = coverage::check(&read.spec, &read.tensors);
        let at = |feature: &str| -> Vec<Option<usize>> {
            items
                .iter()
                .filter(|u| u.feature == feature)
                .map(|u| u.layer)
                .collect()
        };
        let moe = vec![Some(1), Some(4), Some(5)];
        assert_eq!(
            at(
                "routed experts in a 16-value latent (ffn_latent_down before them, \
                ffn_latent_up after their sum)"
            ),
            moe
        );
        assert_eq!(at("shared expert, ff 24"), moe);
        assert_eq!(at("GQA flash, head 8, group 2"), vec![Some(2)]);
        for (l, n) in [(1, 4), (4, 6), (5, 6)] {
            let want = format!("router: sigmoid, {n} experts, top 2, with a selection bias");
            assert!(at(&want).contains(&Some(l)), "{want} at {l}: {items:?}");
        }
        for run in [
            "a layer program for nemotron_h_moe",
            "routed experts of squared ReLU, no gate",
            "shared expert of squared ReLU, no gate",
            "routed stacks whose expert count and width differ by layer (3 shapes)",
        ] {
            assert!(at(run).is_empty(), "{run}: {items:?}");
        }
        assert!(
            items.iter().all(|u| !u.feature.starts_with("Mamba-2")
                && !u.feature.starts_with("dense")
                && !u.feature.starts_with("squared-ReLU dense")),
            "{items:?}"
        );
    }

    /// The gate fixture's layout at Bolt's types and shapes ([`FIXTURE`]) is
    /// one the program runs whole: no coverage item. It plans whole-card on the
    /// gate machine, every routed stack and the shared expert in the file's
    /// blocks; under every arm of [`Experts`] a context of no position, a tier
    /// card and a second card are refused by name. A router row, a format pin
    /// or the group-16 flash row lost names its part here.
    #[test]
    fn the_fixture_layout_plans_whole_card() {
        use crate::placement::{CardFormat, Device, Format, PlanLevers, workstation};
        let inputs = fixture_inputs("nhm-fixture");
        let machine = workstation::plan_gate(inputs.model.layers);
        let levers = PlanLevers::default();
        let plan = inputs
            .plan(&machine, 4, &levers)
            .expect("a whole-card plan");
        assert_eq!(plan.cards.len(), 1);
        assert_eq!(plan.host.experts + plan.host.expert_bytes, 0);
        assert_eq!(plan.nvme_bytes, 0);
        for row in &plan.rows {
            let t = &inputs.model.tensors[row.tensor];
            if matches!(t.role, Role::RoutedExperts | Role::SharedExpert) {
                for seg in &row.segments {
                    assert_eq!(seg.device, Device::Card(0), "{}", t.name);
                    assert_eq!(seg.format, Format::Card(CardFormat::KQuant), "{}", t.name);
                }
            }
        }
        let mut tiered = machine.clone();
        let mut tier = tiered.cards[0].clone();
        tier.layers = 0..0;
        tier.head = false;
        tier.token_embedding = false;
        tiered.tiers.push(tier);
        let mut two = machine.clone();
        let mut second = two.cards[0].clone();
        second.layers = 0..0;
        second.head = false;
        second.token_embedding = false;
        two.cards.push(second);
        for experts in [Experts::Card, Experts::Placed, Experts::Host] {
            assert!(
                matches!(
                    inputs.plan_with(&machine, 0, &levers, experts),
                    Err(super::super::place::PlaceError::CtxZero)
                ),
                "{experts:?}: not the context refusal"
            );
            match inputs.plan_with(&tiered, 4, &levers, experts) {
                Err(e @ super::super::place::PlaceError::Tier { tier: 0, .. }) => assert!(
                    e.to_string()
                        .contains("the nemotron_h_moe body runs every layer on one card"),
                    "{experts:?}: {e}"
                ),
                other => panic!("{experts:?}: {other:?}, not the tier refusal"),
            }
            match inputs.plan_with(&two, 4, &levers, experts) {
                Err(e @ super::super::place::PlaceError::Cards(2, _)) => assert_eq!(
                    e.to_string(),
                    "the placement puts the layers on 2 cards; the nemotron_h_moe chain runs on one",
                    "{experts:?}"
                ),
                other => panic!("{experts:?}: {other:?}, not the card-count refusal"),
            }
        }
    }

    /// A card too small for every routed expert of the fixture layout is
    /// refused by name — the body uploads every expert stack, so the planner
    /// refuses the spill — while the gate machine, which holds them all,
    /// plans. The card is shrunk past half the experts' bytes, so only some
    /// fit: a planner that accepts the spilled plan fails this.
    #[test]
    fn a_card_too_small_for_every_expert_is_refused() {
        use crate::placement::{PlanLevers, workstation};
        let inputs = fixture_inputs("nhm-spill");
        let machine = workstation::plan_gate(inputs.model.layers);
        let levers = PlanLevers::default();
        let whole = inputs
            .plan(&machine, 4, &levers)
            .expect("the gate machine holds every expert");
        assert_eq!(whole.host.experts, 0);
        // The pass's reserve comes out of the card first: the budget keeps half the experts' bytes
        // beside it, as this clause means.
        let budget = half_expert_budget(&whole) + inputs.prompt_reserve;
        let mut small = machine;
        small.cards[0].usable_bytes = budget;
        match inputs.plan(&small, 4, &levers) {
            Err(e) => assert!(
                e.to_string()
                    .contains("nemotron_h_moe requires every expert on one card; ")
                    && e.to_string().contains(" experts remain on the host"),
                "{e}"
            ),
            other => panic!("{other:?}, a spilled plan, not the expert refusal"),
        }
    }

    /// The fixture layout's plan inputs, read as the three plan gates do.
    fn fixture_inputs(tag: &str) -> super::super::place::PlanInputs {
        geo_inputs(&FIXTURE, tag)
    }

    /// `geo`'s plan inputs, read as the three plan gates do.
    fn geo_inputs(geo: &Geo, tag: &str) -> super::super::place::PlanInputs {
        let path = header_typed(
            tag,
            "nemotron_h_moe",
            &geo.keys(),
            &[("tokenizer.ggml.pre", V::Str("pixtral"))],
            &geo.tensors(),
        );
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let _ = std::fs::remove_file(&path);
        super::super::place::read(&split)
            .unwrap_or_else(|e| panic!("the fixture layout is refused: {e}"))
    }

    /// The plan inputs carry the prompt pass's reserve, the formula's figure at the file's headers
    /// (85,859,372 B [derived] at the fixture layout's widths and 512 rows: 256 wide hidden, inner,
    /// latent and shared expert, 32 query heads over 2 of 128, 9 of the 240 and 496 experts of 256
    /// and 512, a router of 496 at most), and the plan keeps it out of the expert budget: on a card
    /// that holds half the experts beside it the reserved plan holds fewer than the same inputs
    /// with none. Mutant: `describe` leaves the reserve out, or the plan ignores it.
    #[test]
    fn the_plan_inputs_reserve_the_prompt_pass() {
        use crate::placement::{PlanLevers, workstation};
        let inputs = fixture_inputs("nhm-reserve");
        assert_eq!(inputs.prompt_reserve, 85_859_372);
        let machine = workstation::plan_gate(inputs.model.layers);
        let levers = PlanLevers::default();
        let whole = inputs
            .plan(&machine, 4, &levers)
            .expect("the gate machine holds every expert");
        let budget = half_expert_budget(&whole) + inputs.prompt_reserve;
        let mut small = machine;
        small.cards[0].usable_bytes = budget;
        let mut free = inputs.clone();
        free.prompt_reserve = 0;
        let kept = inputs
            .plan_with(&small, 4, &levers, Experts::Placed)
            .unwrap_or_else(|e| panic!("a card that holds some experts is refused: {e}"));
        let unreserved = free
            .plan_with(&small, 4, &levers, Experts::Placed)
            .unwrap_or_else(|e| panic!("a card that holds some experts is refused: {e}"));
        assert!(
            kept.cards[0].experts < unreserved.cards[0].experts,
            "{} experts with the reserve, {} without",
            kept.cards[0].experts,
            unreserved.cards[0].experts
        );
    }

    /// A host room under the host need pages the host experts it cannot hold on the NVMe expert
    /// tier ([`crate::placement::expert_nvme_tier`]): a layer's up and down stacks alike, the host
    /// keeping the rest; a room past the need pages none. FAIL-first: the hybrid plan read no room
    /// and kept every host expert in RAM.
    #[test]
    fn a_room_under_the_host_need_pages_experts_on_the_nvme_tier() {
        use crate::placement::workstation::{self, HostNeed, HostRead};
        use crate::placement::{Device, PlanLevers};
        const PAGED: Geo = Geo {
            kinds: &['m', 'e', 'a', 'e', 'e', 'e', 'e', 'e'],
            shapes: &[(496, 256); 6],
            ..FIXTURE
        };
        let mut inputs = geo_inputs(&PAGED, "nhm-paged");
        let machine = workstation::plan_gate(inputs.model.layers);
        let levers = PlanLevers::default();
        inputs.room = (u64::MAX, HostRead::Given);
        let whole = inputs
            .plan_with(&machine, 4, &levers, Experts::Host)
            .unwrap_or_else(|e| panic!("the host arm is refused: {e}"));
        assert_eq!(
            whole.host.nvme_expert_bytes, 0,
            "a room past the need pages none"
        );
        let held = whole.host.expert_bytes;
        let layer = held / 6;
        inputs.room = (HostNeed::of(&whole, 0).bytes() - layer, HostRead::Given);
        let plan = inputs
            .plan_with(&machine, 4, &levers, Experts::Host)
            .unwrap_or_else(|e| panic!("a room a layer short is refused: {e}"));
        assert!(
            plan.host.nvme_expert_bytes > 0,
            "a room a layer short pages none"
        );
        assert_eq!(plan.host.expert_bytes + plan.host.nvme_expert_bytes, held);
        let paged: Vec<&str> = plan
            .rows
            .iter()
            .filter(|r| r.segments.iter().any(|s| s.device == Device::Nvme))
            .map(|r| inputs.model.tensors[r.tensor].name.as_str())
            .collect();
        assert!(
            !paged.is_empty()
                && paged.iter().all(|n| {
                    let up = n.ends_with(names::FFN_UP_EXPS);
                    let twin = if up {
                        n.replace(names::FFN_UP_EXPS, names::FFN_DOWN_EXPS)
                    } else {
                        n.replace(names::FFN_DOWN_EXPS, names::FFN_UP_EXPS)
                    };
                    (up || n.ends_with(names::FFN_DOWN_EXPS)) && paged.contains(&twin.as_str())
                }),
            "{paged:?}"
        );
    }

    /// A card that holds every routed expert plans [`Experts::Placed`] as it
    /// plans [`Experts::Card`], byte for byte, on both cards' bytes: no
    /// expert on the host.
    #[test]
    fn the_placed_plan_of_a_whole_card_is_the_card_plan() {
        use crate::placement::{PlanLevers, workstation};
        let inputs = fixture_inputs("nhm-placed-whole");
        let layers = inputs.model.layers;
        let levers = PlanLevers::default();
        for (name, machine) in [
            ("3090", workstation::plan_gate(layers)),
            ("A6000", workstation::plan_a(layers)),
        ] {
            let card = inputs
                .plan(&machine, 4, &levers)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let placed = inputs
                .plan_with(&machine, 4, &levers, Experts::Placed)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(placed.host.experts, 0, "{name}");
            assert_eq!(format!("{placed:?}"), format!("{card:?}"), "{name}");
        }
    }

    /// A card shrunk past half the experts' bytes plans [`Experts::Placed`]:
    /// each routed layer keeps the id prefix `[0, n_l)` its budget holds and
    /// the rest is the host's, each expert once, a layer's stacks the same
    /// lists, the host segments the file's share, the margin kept. A budget
    /// that drops the margin, a host count at the model's widest layer and a
    /// refusal of any host expert each fail a clause.
    #[test]
    fn a_small_card_keeps_each_layers_prefix_and_hosts_the_rest() {
        use crate::placement::{PlanLevers, workstation};
        let inputs = fixture_inputs("nhm-placed-small");
        let machine = workstation::plan_gate(inputs.model.layers);
        let levers = PlanLevers::default();
        let whole = inputs
            .plan(&machine, 4, &levers)
            .expect("the gate machine holds every expert");
        // The pass's reserve comes out of the card first: the budget keeps half the experts' bytes
        // beside it, as this clause means.
        let budget = half_expert_budget(&whole) + inputs.prompt_reserve;
        let mut small = machine;
        small.cards[0].usable_bytes = budget;
        let plan = inputs
            .plan_with(&small, 4, &levers, Experts::Placed)
            .unwrap_or_else(|e| panic!("a card that holds some experts is refused: {e}"));
        let (layers, bad) = prefix_split(&plan);
        assert!(bad.is_empty(), "{bad:#?}");
        assert!(
            layers
                .iter()
                .any(|l| 0 < l.on_card && l.on_card < l.experts),
            "no layer is split: {layers:?}"
        );
        assert_eq!(
            plan.host.experts,
            layers.iter().map(|l| l.experts - l.on_card).sum::<u64>()
        );
        assert_eq!(
            plan.cards[0].experts,
            layers.iter().map(|l| l.on_card).sum::<u64>()
        );
    }

    /// [`Experts::Host`] keeps no routed expert on the card: every layer's
    /// count is 0, the card holds no expert byte and no routed segment, and
    /// the host holds each routed layer's every expert.
    #[test]
    fn the_host_arm_keeps_no_routed_expert_on_the_card() {
        use crate::placement::{Device, PlanLevers, workstation};
        let inputs = fixture_inputs("nhm-host-arm");
        let machine = workstation::plan_gate(inputs.model.layers);
        let plan = inputs
            .plan_with(&machine, 4, &PlanLevers::default(), Experts::Host)
            .unwrap_or_else(|e| panic!("the host arm is refused: {e}"));
        assert!(plan.n_l.iter().all(|&n| n == 0), "{:?}", plan.n_l);
        assert_eq!(plan.cards[0].expert_bytes, 0);
        assert_eq!(plan.cards[0].experts, 0);
        for row in &plan.rows {
            let t = &inputs.model.tensors[row.tensor];
            if t.role == Role::RoutedExperts {
                assert!(
                    row.segments.iter().all(|s| s.device == Device::Host),
                    "{}: {:?}",
                    t.name,
                    row.segments
                );
            }
        }
        let (layers, bad) = prefix_split(&plan);
        assert!(bad.is_empty(), "{bad:#?}");
        assert_eq!(
            layers
                .iter()
                .map(|l| (l.layer, l.experts))
                .collect::<Vec<_>>(),
            [(1, 240), (3, 496)]
        );
        assert_eq!(plan.host.experts, 240 + 496);
    }

    /// A header without `moe_latent_size` reads as the no-latent form: the
    /// routed experts act on the hidden row itself, the file holds no latent
    /// pair and the description says the experts have no latent. A reader
    /// that still wants the key refuses the file by its name.
    #[test]
    fn a_missing_moe_latent_size_reads_as_the_no_latent_form() {
        let read = load("nhm-nolatent", nolatent_keys(), &nolatent_tensors())
            .expect("the header reads without a latent");
        let moe = read
            .spec
            .layers
            .iter()
            .filter_map(|l| match &l.ffn {
                Ffn::Moe(m) => Some(m),
                Ffn::Dense { .. } => None,
            })
            .next()
            .expect("a MoE layer");
        assert_eq!(moe.latent, None, "the experts act on the hidden row");
        assert!(
            read.tensors
                .tensors
                .iter()
                .all(|t| !t.name.contains("ffn_latent")),
            "the file holds no latent pair: {:?}",
            read.tensors
                .tensors
                .iter()
                .map(|t| &t.name)
                .collect::<Vec<_>>()
        );
    }

    /// A file with a next-token block: the trunk is the blocks before it, the
    /// next-token block's tensors are carried and never loaded, and the
    /// description runs the trunk's layers only. A reader that reads every
    /// block as a trunk layer fails this.
    #[test]
    fn a_next_token_block_is_carried_and_never_run() {
        let mut kv = keys();
        {
            let row = kv.iter_mut().find(|(k, _)| *k == "block_count").unwrap();
            row.1 = V::U32((KINDS.len() + 1) as u32);
        }
        kv.push(("nextn_predict_layers", V::U32(1)));
        // The per-layer arrays run over the file's blocks, the next-token
        // block's included.
        for key in ["feed_forward_length", "attention.head_count_kv"] {
            let row = kv.iter_mut().find(|(k, _)| *k == key).unwrap();
            let V::I32s(xs) = &mut row.1 else {
                panic!("{key} is an array")
            };
            xs.push(0);
        }
        let mut t = tensors();
        // The next-token block: a MoE layer's stems and its own.
        let nextn = KINDS.len();
        let (n, w) = SHAPES[0];
        let e = EMBD;
        let sh = SHARED;
        t.push((
            format!("blk.{nextn}.{}", names::ATTN_NORM),
            vec![e],
            Ty::F32,
        ));
        t.push((
            format!("blk.{nextn}.{}", names::FFN_GATE_INP),
            vec![e, n],
            Ty::F32,
        ));
        t.push((
            format!("blk.{nextn}.{}", names::EXP_PROBS_B),
            vec![n],
            Ty::F32,
        ));
        t.push((
            format!("blk.{nextn}.{}", names::FFN_UP_EXPS),
            vec![LATENT, w, n],
            Ty::F32,
        ));
        t.push((
            format!("blk.{nextn}.{}", names::FFN_DOWN_EXPS),
            vec![w, LATENT, n],
            Ty::F32,
        ));
        t.push((
            format!("blk.{nextn}.{}", names::FFN_UP_SHEXP),
            vec![e, sh],
            Ty::F32,
        ));
        t.push((
            format!("blk.{nextn}.{}", names::FFN_DOWN_SHEXP),
            vec![sh, e],
            Ty::F32,
        ));
        for stem in names::NEXTN {
            let dims = if stem.ends_with("eh_proj.weight") {
                vec![2 * e, e]
            } else {
                vec![e]
            };
            t.push((format!("blk.{nextn}.{stem}"), dims, Ty::F32));
        }
        let read = load("nhm-nextn", kv, &t).expect("the header reads");
        assert_eq!(read.spec.layers.len(), KINDS.len(), "the trunk's layers");
        assert_eq!(read.tensors.layers, KINDS.len(), "the trunk's layer count");
        let role = |name: String| {
            read.tensors
                .tensors
                .iter()
                .find(|x| x.name == name)
                .map(|x| x.role)
                .expect("a classified tensor")
        };
        for stem in names::NEXTN {
            assert_eq!(role(format!("blk.{nextn}.{stem}")), Role::Unused, "{stem}");
        }
        assert_eq!(
            role(format!("blk.{nextn}.{}", names::FFN_UP_EXPS)),
            Role::Unused
        );
        // The next-token block's tensors carry no bytes in a plan.
        let nextn_bytes: u64 = read
            .tensors
            .tensors
            .iter()
            .filter(|t| t.name.starts_with(&format!("blk.{nextn}.")))
            .map(|t| t.file_bytes)
            .sum();
        assert!(nextn_bytes > 0, "the file holds the next-token block");
        assert!(
            read.tensors
                .tensors
                .iter()
                .filter(|t| t.name.starts_with(&format!("blk.{nextn}.")))
                .all(|t| t.role == Role::Unused),
            "the next-token block is carried, never loaded"
        );
    }

    /// FAIL-first: each per-layer array one value short of `block_count` is
    /// refused by its name. An array read without its length check lets the
    /// header read (the missing last layer reads as 0, a Mamba-2 layer's).
    #[test]
    fn a_per_layer_array_of_the_wrong_length_is_named() {
        for key in [
            "feed_forward_length",
            "attention.head_count_kv",
            "expert_feed_forward_length",
            "expert_used_count",
        ] {
            let mut kv = keys();
            let row = kv.iter_mut().find(|(k, _)| *k == key).expect(key);
            let V::I32s(xs) = &mut row.1 else {
                panic!("{key} is an array")
            };
            xs.pop();
            refused(
                "nhm-short",
                kv,
                &tensors(),
                &[
                    &format!("nemotron_h_moe.{key}"),
                    "has 5 values for 6 trunk layers",
                ],
            );
        }
    }

    /// FAIL-first: a routed stack whose last dim is not its layer's expert
    /// count (the router's rows) is refused naming the stack, both dims
    /// listed. A stack sized by `expert_count` (6 at layer 1, of 4 experts)
    /// is refused the same way.
    #[test]
    fn a_stack_whose_expert_dim_disagrees_is_named() {
        for (stem, dims) in [
            (names::FFN_UP_EXPS, vec![LATENT, 32, 5]),
            (names::FFN_DOWN_EXPS, vec![32, LATENT, 5]),
            (names::EXP_PROBS_B, vec![5]),
        ] {
            let mut t = tensors();
            let name = format!("blk.4.{stem}");
            t.iter_mut().find(|(n, ..)| *n == name).expect(stem).1 = dims;
            refused("nhm-stack", keys(), &t, &[&name, "has other dims", "5]"]);
        }
        let mut t = tensors();
        let name = format!("blk.1.{}", names::FFN_UP_EXPS);
        t.iter_mut().find(|(n, ..)| *n == name).unwrap().1 = vec![LATENT, 16, 6];
        refused("nhm-stack-max", keys(), &t, &[&name, "(want [16, 16, 4])"]);
    }

    /// The routed arrays agree with the layer's kind: a top-k on a Mamba-2
    /// layer, a width on an attention layer, and a 0 on a MoE layer are each
    /// refused naming the key and the layer.
    #[test]
    fn routed_arrays_off_their_kind_are_named() {
        let cases = [
            (
                "expert_used_count",
                [2, 2, 0, 0, 2, 2],
                "is 2 at layer 0, a Mamba-2 layer, which routes nothing",
            ),
            (
                "expert_feed_forward_length",
                [0, 16, 8, 0, 32, 16],
                "is 8 at layer 2, an attention layer, which routes nothing",
            ),
            (
                "expert_used_count",
                [0, 2, 0, 0, 0, 2],
                "is 0 at layer 4, a MoE layer",
            ),
        ];
        for (key, xs, want) in cases {
            refused(
                "nhm-arrays",
                with(key, Some(V::I32s(xs.to_vec()))),
                &tensors(),
                &[&format!("nemotron_h_moe.{key}"), want],
            );
        }
    }

    /// `expert_count` is the largest layer's count: one above every router
    /// is refused, one below a router names that router; a top-k past its
    /// layer's experts is refused naming the layer.
    #[test]
    fn expert_counts_are_held_to_the_routers() {
        refused(
            "nhm-count-hi",
            with("expert_count", Some(V::U32(8))),
            &tensors(),
            &["nemotron_h_moe.expert_count", "is 8", "has 6 rows"],
        );
        refused(
            "nhm-count-lo",
            with("expert_count", Some(V::U32(5))),
            &tensors(),
            &["blk.4.ffn_gate_inp.weight", "6 rows, past expert_count 5"],
        );
        refused(
            "nhm-topk",
            with("expert_used_count", Some(V::I32s(vec![0, 5, 0, 0, 2, 2]))),
            &tensors(),
            &[
                "nemotron_h_moe.expert_used_count",
                "is 5 at layer 1",
                "4 experts",
            ],
        );
    }

    /// The card pass is sized for the largest top-k: in a file whose MoE
    /// layers keep 1 and 2 experts, the layer that keeps 1 is refused by
    /// name and the layers that keep 2 are not.
    #[test]
    fn a_layer_keeping_fewer_than_the_largest_top_k_is_refused_on_the_card() {
        let kv = with("expert_used_count", Some(V::I32s(vec![0, 1, 0, 0, 2, 2])));
        let global = [("tokenizer.ggml.pre", V::Str("pixtral"))];
        let path = header_typed("nhm-mixed-topk", "nemotron_h_moe", &kv, &global, &tensors());
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let hp = Hparams::read(&split).expect("a mixed top-k is a file the reader takes");
        let _ = std::fs::remove_file(&path);
        let err = hp
            .refuse_card_keeps(1, 1)
            .expect_err("layer 1 keeps 1 of the pass's 2");
        assert!(
            err.contains("layer 1 keeps 1 experts") && err.contains("the 2 the pass is sized for"),
            "{err}"
        );
        for l in [4, 5] {
            hp.refuse_card_keeps(l, 2)
                .expect("a layer that keeps the largest top-k");
        }
    }

    /// Every MoE key the reader needs is required by name, and a value the
    /// description cannot hold is refused by name: another gating function,
    /// a second shared expert, a group limit.
    #[test]
    fn the_moe_keys_are_named() {
        for key in [
            "expert_count",
            "expert_shared_feed_forward_length",
            "expert_shared_count",
            "expert_weights_norm",
            "expert_weights_scale",
            "attention.layer_norm_epsilon",
        ] {
            refused(
                "nhm-key",
                with(key, None),
                &tensors(),
                &[&format!("nemotron_h_moe.{key}"), "absent"],
            );
        }
        // The gating key is the dialect's to default: the dialect builds the
        // sigmoid router whatever the key says, so a file that carries no key
        // reads with that router and the line that says where it came from.
        let read = load("nhm-gate", with("expert_gating_func", None), &tensors())
            .expect("the header reads without the gating key");
        assert_eq!(
            read.defaults,
            [
                "expert_gating_func = sigmoid (nemotron-h.cpp:312, the sigmoid router the dialect \
             builds)"
            ]
        );
        let moe = read
            .spec
            .layers
            .iter()
            .filter_map(|l| match &l.ffn {
                Ffn::Moe(m) => Some(m),
                Ffn::Dense { .. } => None,
            })
            .next()
            .expect("a MoE layer");
        assert_eq!(moe.router.score, Score::Sigmoid);
        for (key, v, want) in [
            (
                "expert_gating_func",
                V::U32(1),
                "is 1; this reader reads the sigmoid router",
            ),
            (
                "expert_shared_count",
                V::U32(2),
                "is 2; a layer runs one shared expert",
            ),
            (
                "expert_group_count",
                V::U32(4),
                "is 4; the nemotron_h_moe router takes one group",
            ),
            (
                "expert_weights_scale",
                V::F32(0.0),
                "is 0, not a finite positive value",
            ),
        ] {
            refused("nhm-value", with(key, Some(v)), &tensors(), &[key, want]);
        }
    }

    /// A projection's scale is optional, and when present one f32 value: an
    /// f16 scale is refused by name. The unread keys are listed.
    #[test]
    fn a_projection_scale_is_one_optional_f32() {
        let mut t = tensors();
        t.retain(|(n, ..)| !n.ends_with(".scale"));
        load("nhm-noscale", keys(), &t).expect("the scales are optional");
        let mut t = tensors();
        let name = format!("blk.3.{}", names::SSM_OUT_SCALE);
        t.iter_mut().find(|(n, ..)| *n == name).unwrap().2 = Ty::F16;
        refused(
            "nhm-f16scale",
            keys(),
            &t,
            &[&name, "is f16, not the f32 scale"],
        );
    }
}
