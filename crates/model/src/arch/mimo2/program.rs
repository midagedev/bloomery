//! The facts of the program that runs a mimo2 file, as the file's description
//! fixes them: what each layer's attention launches take ([`attn_args`]), what
//! each block's take ([`block_args`]), and how many nodes and memory-operation
//! batches the captured step holds ([`step_launches`], [`step_memops`]).
//! `crates/gpu-mimo2` builds its layers from these and the gate reads them
//! again; the tests here are the program's unit tests, which need no device.
//!
//! MiMo facts that fix this file: the score head is wider than the value head
//! (192 and 128) and the rope turns 64 of the 192 score values at the layer's
//! own base; the nine full layers have 4 key/value heads over 64 query heads
//! and the 39 window layers 8, with a window of positions and per-head sinks;
//! every layer multiplies its value rows by one constant before the cache;
//! the routed block has no shared expert and no clamp; layer 0's block is
//! dense.

use models::{Act, Ffn, LayerSpec, Mixer, RopeMode, Score};
use runtime::layer::FfnKind;

/// A layer the program has no launch for, named by its layer.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("layer {layer}: {what}")]
pub struct Refusal {
    /// The layer's index.
    pub layer: usize,
    /// What the program does not run there.
    pub what: String,
}

fn refuse(layer: usize, what: String) -> Refusal {
    Refusal { layer, what }
}

/// The shapes the attention kernels are built for, which the description is
/// held to.
#[derive(Clone, Copy, Debug)]
pub struct AttnBuilt {
    /// Score head width.
    pub score_head: usize,
    /// Value head width.
    pub value_head: usize,
    /// Score values the rope turns.
    pub rotated: usize,
    /// Query heads of the model.
    pub heads: usize,
}

/// What an attention layer's launches take besides the buffers, read from the
/// layer's description by [`attn_args`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttnArgs {
    /// Key/value heads the layer holds (`attention.head_count_kv`).
    pub kv_heads: usize,
    /// Positions a query attends; 0 for every key.
    pub window: usize,
    /// The layer folds its per-head sinks into the merge.
    pub sinks: bool,
    /// The rope's base θ.
    pub theta: f32,
    /// The multiplier on the value rows before they are cached.
    pub v_scale: f32,
}

/// Layer `l`'s [`AttnArgs`] from its description `s`, held to the shapes
/// `built` names. Refused by name: another mixer, a head or rotated width the
/// kernels are not built for, a rope but NEOX, a QK norm, an output gate or a
/// selector, a window without sinks (the merge without them cuts from key 0)
/// or sinks without a window (a layer the program has no merge for), a window
/// of 0, and a base or multiplier that is not finite and positive. A
/// description with no multiplier multiplies by 1.
pub fn attn_args(l: usize, s: &LayerSpec, built: AttnBuilt) -> Result<AttnArgs, Refusal> {
    let Mixer::Gqa(g) = &s.mixer else {
        return Err(refuse(
            l,
            "a mixer that is not GQA; every mimo2 layer is".to_string(),
        ));
    };
    let fixed = [
        ("score head", g.head_dim as usize, built.score_head),
        ("value head", g.value_dim as usize, built.value_head),
        ("rotated values", g.rope.dims as usize, built.rotated),
        ("query heads", g.heads as usize, built.heads),
    ];
    for (what, got, want) in fixed {
        if got != want {
            return Err(refuse(
                l,
                format!("{what} is {got}; the K192 kernels are built for {want}"),
            ));
        }
    }
    if g.rope.mode != RopeMode::Neox
        || g.rope.yarn.is_some()
        || g.qk_norm
        || g.out_gate
        || g.select.is_some()
    {
        return Err(refuse(
            l,
            "a rope, norm, gate or selector the K192 append does not run (NEOX, no scaling, \
             no QK norm, no output gate, no selector)"
                .to_string(),
        ));
    }
    if g.window == Some(0) {
        return Err(refuse(
            l,
            "a window of 0 positions; a layer with no window has none".to_string(),
        ));
    }
    if g.window.is_some() != g.sinks {
        return Err(refuse(
            l,
            format!(
                "window {:?} with sinks {}; a window layer folds its sinks and a full layer has \
                 none",
                g.window, g.sinks
            ),
        ));
    }
    let v_scale = g.value_scale.unwrap_or(1.0);
    for (what, v) in [("rope base", g.rope.base), ("value multiplier", v_scale)] {
        if !(v.is_finite() && v > 0.0) {
            return Err(refuse(
                l,
                format!("the {what} is {v}, not finite and positive"),
            ));
        }
    }
    Ok(AttnArgs {
        kv_heads: g.kv_heads as usize,
        window: g.window.map_or(0, |w| w as usize),
        sinks: g.sinks,
        theta: g.rope.base,
        v_scale,
    })
}

/// Layer `l`'s sinks row of `len` values is one per query head of `heads`,
/// or refused by name.
pub fn sinks_row(l: usize, len: usize, heads: usize) -> Result<(), Refusal> {
    if len == heads {
        return Ok(());
    }
    Err(refuse(l, format!("{len} sinks for {heads} query heads")))
}

/// The router the routed block's kernels are built for.
#[derive(Clone, Copy, Debug)]
pub struct RouterBuilt {
    /// Experts the router scores.
    pub n_expert: usize,
    /// Picks a token.
    pub n_used: usize,
    /// The routed scale on the picks' weights.
    pub scale: f32,
}

/// What a block's launches take besides the buffers, read from the layer's
/// description by [`block_args`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockArgs {
    /// The dense block's SwiGLU limit; 0 for the plain combine, and on a
    /// routed block.
    pub limit: f32,
    /// The dense block's width; 0 on a routed block.
    pub ff: usize,
    /// The router's selection bias is in the file.
    pub bias: bool,
}

/// The layer's SwiGLU limit, 0 (the plain combine) when it has none.
#[must_use]
pub fn swiglu_limit(act: Act) -> f32 {
    let Act::SwiGlu { limit } = act;
    limit.unwrap_or(0.0)
}

/// Layer `l`'s [`BlockArgs`] from its description `s`, held to the router
/// `built` names. Refused by name: a router that is not the sigmoid one that
/// renormalizes its picks, unhashed, at another width, count or scale than
/// the program's; a shared expert (the routed block has none, and a layer with
/// one would lose its output); a clamp on the routed experts (the host tier
/// runs the plain combine).
pub fn block_args(l: usize, s: &LayerSpec, built: RouterBuilt) -> Result<BlockArgs, Refusal> {
    match &s.ffn {
        Ffn::Dense { ff, act } => Ok(BlockArgs {
            limit: swiglu_limit(*act),
            ff: *ff as usize,
            bias: false,
        }),
        Ffn::Moe(m) => {
            let r = &m.router;
            if r.score != Score::Sigmoid || !r.norm || r.hash {
                return Err(refuse(
                    l,
                    format!(
                        "a router scoring by {:?}, renormalizing {}, hashed {}; the routed \
                         block runs the sigmoid router that renormalizes its picks, unhashed",
                        r.score, r.norm, r.hash
                    ),
                ));
            }
            if m.experts as usize != built.n_expert
                || m.top_k as usize != built.n_used
                || r.scale != built.scale
            {
                return Err(refuse(
                    l,
                    format!(
                        "{} experts, top {}, scale {}; the router is built for {}, top {}, \
                         scale {}",
                        m.experts, m.top_k, r.scale, built.n_expert, built.n_used, built.scale
                    ),
                ));
            }
            if m.shared.is_some() {
                return Err(refuse(
                    l,
                    "a shared expert; the routed block has none".to_string(),
                ));
            }
            if swiglu_limit(m.act) != 0.0 {
                return Err(refuse(
                    l,
                    "a clamp on the routed experts; the host tier runs the plain combine"
                        .to_string(),
                ));
            }
            Ok(BlockArgs {
                limit: 0.0,
                ff: 0,
                bias: r.bias,
            })
        }
    }
}

/// The attention sub-layer's launches: the norm, the fused projection, the
/// rope-and-append, the flash's segment pass and its merge, the output
/// projection and the add.
pub const ATTN_LAUNCHES: usize = 7;

/// The dense block's launches: the norm, the gate·up·SwiGLU, the down
/// projection and the add.
pub const DENSE_LAUNCHES: usize = 4;

/// A routed block's launches: the front's norm, router and handoff, the go,
/// the wait and the add.
pub const MOE_LAUNCHES: usize = 3 + 1 + 1 + 1;

/// The head's launches: the norm, the q8_0 gemv and the argmax.
pub const HEAD_LAUNCHES: usize = 3;

/// One layer's launches: its attention's and its block's.
#[must_use]
pub fn layer_launches(ffn: FfnKind) -> usize {
    ATTN_LAUNCHES
        + match ffn {
            FfnKind::Dense => DENSE_LAUNCHES,
            FfnKind::Moe => MOE_LAUNCHES,
        }
}

/// The nodes of the step's captured chain: every layer's launches
/// ([`layer_launches`]) and the head's.
#[must_use]
pub fn step_launches(ffns: &[FfnKind]) -> usize {
    ffns.iter().map(|&f| layer_launches(f)).sum::<usize>() + HEAD_LAUNCHES
}

/// The stream memory-operation batches of the step's captured chain: each
/// routed layer's go and its wait.
#[must_use]
pub fn step_memops(ffns: &[FfnKind]) -> usize {
    2 * ffns.iter().filter(|&&f| f == FfnKind::Moe).count()
}

#[cfg(test)]
mod tests {
    use models::{
        Act, Ffn, Gqa, LayerSpec, Mixer, Moe, Residual, Rope, RopeMode, Router, Score, Shared,
    };
    use runtime::layer::FfnKind;

    use super::{
        AttnArgs, AttnBuilt, BlockArgs, HEAD_LAUNCHES, RouterBuilt, attn_args, block_args,
        layer_launches, sinks_row, step_launches, step_memops,
    };

    /// The kernels' shapes: 192-wide scores over 128-wide values, 64 of the
    /// 192 turned, 64 query heads.
    const BUILT: AttnBuilt = AttnBuilt {
        score_head: 192,
        value_head: 128,
        rotated: 64,
        heads: 64,
    };
    const ROUTER: RouterBuilt = RouterBuilt {
        n_expert: 256,
        n_used: 8,
        scale: 1.0,
    };
    /// MiMo-V2.6-Flash's full layers, by index.
    const FULL: [usize; 9] = [0, 5, 11, 17, 23, 29, 35, 41, 47];

    fn gqa(kv: u32, window: Option<u32>, base: f32) -> Gqa {
        Gqa {
            heads: 64,
            kv_heads: kv,
            head_dim: 192,
            value_dim: 128,
            rope: Rope {
                mode: RopeMode::Neox,
                dims: 64,
                base,
                yarn: None,
            },
            qk_norm: false,
            out_gate: false,
            select: None,
            window,
            sinks: window.is_some(),
            value_scale: Some(0.707),
        }
    }

    fn with(mixer: Gqa, ffn: Ffn) -> LayerSpec {
        LayerSpec {
            mixer: Mixer::Gqa(mixer),
            ffn,
            residual: Residual::Plain,
            extras: Vec::new(),
        }
    }

    fn dense() -> Ffn {
        Ffn::Dense {
            ff: 16384,
            act: Act::SwiGlu { limit: None },
        }
    }

    /// A layer of the shapes the real file's description carries: `window`
    /// positions with sinks, or none of either.
    fn layer(kv: u32, window: Option<u32>, base: f32) -> LayerSpec {
        with(gqa(kv, window, base), dense())
    }

    fn moe(f: impl FnOnce(&mut Moe)) -> LayerSpec {
        let mut m = Moe {
            experts: 256,
            top_k: 8,
            expert_ff: 2048,
            act: Act::SwiGlu { limit: None },
            router: Router {
                score: Score::Sigmoid,
                bias: true,
                norm: true,
                scale: 1.0,
                hash: false,
            },
            shared: None,
        };
        f(&mut m);
        with(gqa(4, None, 1.0e7), Ffn::Moe(m))
    }

    /// The 48 layers' arguments follow the description's facts: the window,
    /// the sinks, the base and the multiplier of each layer are what its
    /// description says, nine full layers among the window ones.
    #[test]
    fn the_per_layer_arguments_are_the_descriptions() {
        let layers: Vec<LayerSpec> = (0..48)
            .map(|l| {
                if FULL.contains(&l) {
                    layer(4, None, 1.0e7)
                } else {
                    layer(8, Some(128), 1.0e4)
                }
            })
            .collect();
        for (l, s) in layers.iter().enumerate() {
            let got = attn_args(l, s, BUILT).expect("a real layer");
            let want = if FULL.contains(&l) {
                AttnArgs {
                    kv_heads: 4,
                    window: 0,
                    sinks: false,
                    theta: 1.0e7,
                    v_scale: 0.707,
                }
            } else {
                AttnArgs {
                    kv_heads: 8,
                    window: 128,
                    sinks: true,
                    theta: 1.0e4,
                    v_scale: 0.707,
                }
            };
            assert_eq!(got, want, "layer {l}");
        }
    }

    /// What the attention launches have no row for is refused by name, never
    /// run as another layer kind.
    #[test]
    fn a_layer_the_kernels_do_not_run_is_refused_by_name() {
        let refuses = |s: LayerSpec, says: &str| {
            let e = attn_args(3, &s, BUILT).expect_err(says).to_string();
            assert!(e.contains("layer 3") && e.contains(says), "{says}: {e}");
        };
        let edit = |f: &dyn Fn(&mut Gqa)| {
            let mut g = gqa(8, Some(128), 1.0e4);
            f(&mut g);
            with(g, dense())
        };
        refuses(
            edit(&|g| g.sinks = false),
            "window Some(128) with sinks false",
        );
        refuses(edit(&|g| g.window = None), "window None with sinks true");
        refuses(edit(&|g| g.window = Some(0)), "a window of 0");
        refuses(edit(&|g| g.head_dim = 128), "score head is 128");
        refuses(edit(&|g| g.value_dim = 192), "value head is 192");
        refuses(edit(&|g| g.rope.dims = 192), "rotated values is 192");
        refuses(edit(&|g| g.heads = 32), "query heads is 32");
        refuses(
            edit(&|g| g.qk_norm = true),
            "a rope, norm, gate or selector",
        );
        refuses(
            edit(&|g| g.rope.mode = RopeMode::NormTail),
            "a rope, norm, gate",
        );
        refuses(
            edit(&|g| g.value_scale = Some(f32::NAN)),
            "value multiplier",
        );
        refuses(edit(&|g| g.rope.base = f32::INFINITY), "rope base");
    }

    /// A description with no value multiplier multiplies the value rows by 1.
    #[test]
    fn no_value_multiplier_is_one() {
        let mut g = gqa(4, None, 1.0e7);
        g.value_scale = None;
        let got = attn_args(0, &with(g, dense()), BUILT).expect("a layer");
        assert_eq!(got.v_scale, 1.0);
    }

    /// A sinks row is one value per query head; a short or long row is
    /// refused by its layer.
    #[test]
    fn a_sinks_row_is_one_per_query_head() {
        sinks_row(7, 64, 64).expect("one per head");
        for len in [32, 65] {
            let e = sinks_row(7, len, 64).expect_err("a row of another width");
            assert!(
                e.to_string().contains("layer 7") && e.to_string().contains("sinks for 64"),
                "{e}"
            );
        }
    }

    /// The real file's blocks: layer 0's dense width and the routed layers'
    /// selection bias come from the description.
    #[test]
    fn the_blocks_are_the_descriptions() {
        assert_eq!(
            block_args(0, &layer(4, None, 1.0e7), ROUTER).expect("the dense block"),
            BlockArgs {
                limit: 0.0,
                ff: 16384,
                bias: false
            }
        );
        let routed = block_args(1, &moe(|_| {}), ROUTER).expect("a routed block");
        assert_eq!((routed.ff, routed.bias), (0, true));
        let unbiased = block_args(1, &moe(|m| m.router.bias = false), ROUTER).expect("no bias");
        assert!(!unbiased.bias);
    }

    /// What the routed block does not run is refused by name.
    #[test]
    fn a_block_the_program_does_not_run_is_refused_by_name() {
        let refuses = |s: LayerSpec, says: &str| {
            let e = block_args(5, &s, ROUTER).expect_err(says).to_string();
            assert!(e.contains("layer 5") && e.contains(says), "{says}: {e}");
        };
        refuses(moe(|m| m.router.score = Score::Softmax), "sigmoid router");
        refuses(moe(|m| m.router.norm = false), "sigmoid router");
        refuses(moe(|m| m.router.hash = true), "sigmoid router");
        refuses(moe(|m| m.experts = 288), "288 experts");
        refuses(moe(|m| m.top_k = 6), "top 6");
        refuses(moe(|m| m.router.scale = 2.5), "scale 2.5");
        refuses(
            moe(|m| {
                m.shared = Some(Shared {
                    ff: 2048,
                    act: Act::SwiGlu { limit: None },
                    sigmoid_gate: false,
                });
            }),
            "a shared expert",
        );
        refuses(
            moe(|m| m.act = Act::SwiGlu { limit: Some(7.0) }),
            "a clamp on the routed experts",
        );
    }

    /// MiMo-V2.6-Flash's 48 layers: the dense layer 0 and 47 routed.
    fn real() -> Vec<FfnKind> {
        let mut ffns = vec![FfnKind::Moe; 48];
        ffns[0] = FfnKind::Dense;
        ffns
    }

    /// The step's nodes are the sum of its layers' launches and the head's:
    /// 48 attention layers of 7, the dense block's 4, 47 routed blocks of 6
    /// and the head's 3, of which the 47 routed layers' go and wait are the
    /// 94 memory-operation batches.
    #[test]
    fn the_step_is_625_nodes_and_94_memops() {
        let ffns = real();
        assert_eq!(step_launches(&ffns), 48 * 7 + 4 + 47 * 6 + 3);
        assert_eq!(step_launches(&ffns), 625);
        assert_eq!(step_memops(&ffns), 94);
        assert_eq!(step_launches(&ffns) - step_memops(&ffns), 531);
    }

    /// A layer's launches are its attention's and its block's alone.
    #[test]
    fn a_layer_is_its_attention_and_its_block() {
        assert_eq!(layer_launches(FfnKind::Dense), 7 + 4);
        assert_eq!(layer_launches(FfnKind::Moe), 7 + 6);
        assert_eq!(step_launches(&[]), HEAD_LAUNCHES);
        assert_eq!(step_memops(&[FfnKind::Dense, FfnKind::Dense]), 0);
    }
}
