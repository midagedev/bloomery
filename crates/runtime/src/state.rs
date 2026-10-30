//! What a layer keeps across positions, read off its [`LayerSpec`]: each
//! store and the rule by which later positions read it ([`stores`]). A
//! schedule derives what it may skip from the rules alone, never from the
//! mixer's name — so a mixer a later model brings needs its stores listed
//! here, and nothing in the schedules.

use models::{IndexKeys, LayerSpec, Mixer, Selector, Source};

/// How the positions after the one that wrote a store's row read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreRule {
    /// One row per position (or per group of positions), kept for good:
    /// every later position may read it.
    Positional,
    /// A ring of the last `slots` positions' rows: only the next `slots`
    /// positions read a row.
    Window { slots: u32 },
    /// A ring of the last `ratio` projections a compressor pools: the
    /// position that completes a group reads the group's.
    Ratio { ratio: u32 },
    /// One state overwritten in place by every position: it carries every
    /// position before it.
    Recurrent,
}

impl StoreRule {
    /// Whether a position more than any window behind the writer reads it.
    #[must_use]
    pub fn reads_past_window(self) -> bool {
        match self {
            StoreRule::Positional | StoreRule::Recurrent => true,
            StoreRule::Window { .. } | StoreRule::Ratio { .. } => false,
        }
    }
}

/// A store a layer keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Store {
    /// A GQA layer's key and value planes.
    Kv,
    /// A latent layer's latent rows.
    Latent,
    /// The compressed rows a latent layer's own compressor writes.
    Compressed,
    /// That compressor's pooling state.
    CompressorState,
    /// The index keys a latent layer writes, one per compressed row.
    IndexKeys,
    /// The pooling state of the indexer's own compressor.
    IndexCompressorState,
    /// The pooled keys a token-pool selector scores.
    PooledKeys,
    /// A delta-rule, Mamba-2 or short-conv layer's causal conv inputs.
    Conv,
    /// A delta-rule layer's recurrent state.
    State,
}

/// Layer `layer`'s stores and their rules, in the order above. A store
/// another layer writes and this one reads (a [`Source::From`]) is that
/// layer's.
#[must_use]
pub fn stores(layer: &LayerSpec) -> Vec<(Store, StoreRule)> {
    let mut v = Vec::new();
    match &layer.mixer {
        Mixer::Gqa(g) => {
            v.push((
                Store::Kv,
                g.window
                    .map_or(StoreRule::Positional, |slots| StoreRule::Window { slots }),
            ));
            // A block-max selector scores every position, so each one keeps an index key.
            if matches!(g.select, Some(Selector::BlockMax { .. })) {
                v.push((Store::IndexKeys, StoreRule::Positional));
            }
        }
        Mixer::DeltaRule(models::DeltaRule { conv, .. })
        | Mixer::Mamba2(models::Mamba2 { d_conv: conv, .. }) => {
            v.push((
                Store::Conv,
                StoreRule::Window {
                    slots: conv.saturating_sub(1),
                },
            ));
            v.push((Store::State, StoreRule::Recurrent));
        }
        Mixer::ShortConv(c) => v.push((
            Store::Conv,
            StoreRule::Window {
                slots: c.taps.saturating_sub(1),
            },
        )),
        Mixer::Latent(a) => {
            v.push((
                Store::Latent,
                a.window
                    .map_or(StoreRule::Positional, |slots| StoreRule::Window { slots }),
            ));
            let ratio = a.compress.as_ref().map(|c| c.ratio);
            let pooled = |store| {
                ratio
                    .filter(|&r| r > 1)
                    .map(|ratio| (store, StoreRule::Ratio { ratio }))
            };
            if let Some(c) = &a.compress
                && matches!(c.rows, Source::Own(_))
            {
                v.push((Store::Compressed, StoreRule::Positional));
                v.extend(pooled(Store::CompressorState));
            }
            match &a.select {
                Some(Selector::StreamTopK {
                    keys: Source::Own(k),
                    ..
                }) => {
                    v.push((Store::IndexKeys, StoreRule::Positional));
                    if matches!(k, IndexKeys::Compressor(_)) {
                        v.extend(pooled(Store::IndexCompressorState));
                    }
                }
                Some(Selector::TokenPool { .. }) => {
                    v.push((Store::PooledKeys, StoreRule::Positional));
                }
                Some(Selector::StreamTopK { .. } | Selector::BlockMax { .. }) | None => {}
            }
        }
        Mixer::None => {}
    }
    v
}

/// Whether a prompt call runs layer `layer`'s store writes at every position:
/// some store of it is read past any window ([`StoreRule::reads_past_window`]).
/// A layer whose stores are windows alone writes only what the window of a
/// later reader reaches — the CED triangle's rows.
#[must_use]
pub fn every_position(layer: &LayerSpec) -> bool {
    stores(layer).iter().any(|&(_, r)| r.reads_past_window())
}

/// Whether a checkpoint copies layer `layer`'s stores. A layer is copied
/// when one of its stores keeps its history nowhere else: a [`StoreRule::Recurrent`]
/// store, copied whole, or a [`StoreRule::Window`] store that no shadow holds,
/// copied as its last `slots` rows. `shadowed` is the caller's
/// `KvBytes::shadow_bytes(layer, ctx) > 0`. Runtime takes that fact as an
/// argument and gains no placement dependency.
#[must_use]
pub fn copied(layer: &LayerSpec, shadowed: bool) -> bool {
    stores(layer).iter().any(|&(_, r)| match r {
        StoreRule::Recurrent => true,
        StoreRule::Window { .. } => !shadowed,
        StoreRule::Positional | StoreRule::Ratio { .. } => false,
    })
}

/// The layers a checkpoint copies ([`copied`]), ascending. `shadowed(layer)`
/// is that layer's `KvBytes::shadow_bytes(layer, ctx) > 0`.
#[must_use]
pub fn copied_layers(layers: &[LayerSpec], shadowed: &dyn Fn(usize) -> bool) -> Vec<usize> {
    (0..layers.len())
        .filter(|&l| copied(&layers[l], shadowed(l)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use models::{
        Act, Candidates, Compress, Compressor, DeltaKind, DeltaRule, Ffn, GdnGate, Gqa, KHeadMap,
        KdaGate, Latent, LatentOut, LatentUp, Moe, OutGate, PoolRule, Residual, Rope, RopeMode,
        Router, Score, ShortConv, Source,
    };

    fn moe() -> Ffn {
        Ffn::Moe(Moe {
            experts: 256,
            top_k: 8,
            expert_ff: 512,
            latent: None,
            pre_exp_norm: false,
            post_exp_norm: None,
            act: Act::SwiGlu { limit: None },
            router: Router {
                score: Score::Softmax,
                bias: false,
                norm: models::RouterNorm::SelectedSum,
                scale: 1.0,
                hash: false,
            },
            shared: None,
        })
    }

    fn rope(mode: RopeMode, dims: u32) -> Rope {
        Rope {
            mode,
            dims,
            base: 10_000.0,
            yarn: None,
            factors: false,
        }
    }

    fn layer(mixer: Mixer, residual: Residual) -> LayerSpec {
        LayerSpec {
            mixer,
            ffn: moe(),
            residual,
            extras: Vec::new(),
        }
    }

    /// A V4.1 latent layer: window 128; `cmp` the compressed stream (ratio,
    /// owner), `sel` the index keys' and the list's owners, `cand` whether a
    /// candidate pool applies.
    fn v41_layer(
        l: u32,
        cmp: Option<(u32, u32)>,
        sel: Option<(u32, u32)>,
        cand: bool,
    ) -> LayerSpec {
        let src = |s: u32| {
            if s == l {
                Source::Own(())
            } else {
                Source::From(s)
            }
        };
        let own = Compressor {
            gated: true,
            ape: false,
            overlap: false,
        };
        let compress = cmp.map(|(ratio, s)| Compress {
            ratio,
            rows: if s == l {
                Source::Own(own)
            } else {
                Source::From(s)
            },
        });
        let select = sel.map(|(keys, list)| Selector::StreamTopK {
            heads: 32,
            d: 128,
            k: 512,
            keys: if keys == l {
                Source::Own(IndexKeys::FromRows)
            } else {
                Source::From(keys)
            },
            list: src(list),
            candidates: cand.then_some(Candidates {
                source: 20,
                blocks: 2048,
                block: 8,
            }),
        });
        layer(
            Mixer::Latent(Latent {
                heads: 64,
                q_lora: 1280,
                latent: 512,
                up: LatentUp::KeqV,
                rope: Some(rope(RopeMode::NormTail, 64)),
                q_head_norm: false,
                ssmax: false,
                pe: 0,
                cache: Source::Own(()),
                out_gate: OutGate::None,
                out: LatentOut::Grouped {
                    groups: 8,
                    rank: 1024,
                },
                window: Some(128),
                sinks: true,
                compress,
                select,
            }),
            Residual::Hc,
        )
    }

    /// V4.1's 40 layers as `ds41_meta.rs`'s `V41_VIEW` pins them: 0 and 1
    /// window-only; compressors of ratio 2 owned at 2, 8 and 14 and of ratio 1
    /// at 20, each layer reading the last owner at or below it; index keys
    /// with the compressors; lists owned at 2, 8, 14, 20, 24, 28, 32 and 36;
    /// candidate pools after 20.
    fn v41() -> Vec<LayerSpec> {
        (0..40)
            .map(|l| {
                if l < 2 {
                    return v41_layer(l, None, None, false);
                }
                let owner = [2, 8, 14, 20]
                    .into_iter()
                    .filter(|&s| s <= l)
                    .max()
                    .unwrap();
                let ratio = if owner == 20 { 1 } else { 2 };
                let list = [2, 8, 14, 20, 24, 28, 32, 36]
                    .into_iter()
                    .filter(|&s| s <= l)
                    .max()
                    .unwrap();
                v41_layer(l, Some((ratio, owner)), Some((owner, list)), l > 20)
            })
            .collect()
    }

    /// Qwen3.6's 40 layers as `qwen35moe_meta.rs`'s `VIEW` pins them: every
    /// fourth layer (3, 7, …, 39) GQA 16/2 × 256, the others GDN.
    fn qwen36() -> Vec<LayerSpec> {
        (0..40)
            .map(|l| {
                let mixer = if l % 4 == 3 {
                    Mixer::Gqa(Gqa {
                        heads: 16,
                        kv_heads: 2,
                        head_dim: 256,
                        value_dim: 256,
                        rope: rope(
                            RopeMode::Imrope {
                                sections: [11, 11, 10, 0],
                            },
                            64,
                        ),
                        qk_norm: true,
                        o_norm: false,
                        out_gate: OutGate::Packed,
                        select: None,
                        window: None,
                        sinks: false,
                        value_scale: None,
                    })
                } else {
                    Mixer::DeltaRule(DeltaRule {
                        kind: DeltaKind::Gdn {
                            khead_map: KHeadMap::Tiled,
                            gate: GdnGate::Silu,
                        },
                        k_heads: 16,
                        v_heads: 32,
                        d: 128,
                        conv: 4,
                    })
                };
                layer(mixer, Residual::Plain)
            })
            .collect()
    }

    /// The CED walk's rule before the stores: a layer that owns a compressor
    /// or index keys runs every position.
    fn owner_rule(l: &LayerSpec) -> bool {
        l.owns_rows() || l.owns_keys()
    }

    /// T-d: V4.1's layers derive the owner rule's vector, true exactly at 2,
    /// 8, 14 and 20.
    #[test]
    fn v41_every_is_the_owners() {
        let every: Vec<bool> = v41().iter().map(every_position).collect();
        let owners: Vec<bool> = v41().iter().map(owner_rule).collect();
        assert_eq!(every, owners);
        let at: Vec<usize> = (0..40).filter(|&l| every[l]).collect();
        assert_eq!(at, [2, 8, 14, 20]);
    }

    /// T-d: every Qwen3.6 layer keeps a store read past any window — the GQA
    /// layers their planes, the GDN layers their state — where the owner rule
    /// finds none.
    #[test]
    fn qwen36_every_layer() {
        let q = qwen36();
        assert!(q.iter().all(every_position));
        assert!(q.iter().all(|l| !owner_rule(l)));
    }

    /// The stores of each kind, and the ratio-1 compressor's want of a state.
    #[test]
    fn stores_of_each_kind() {
        let v = v41();
        assert_eq!(
            stores(&v[0]),
            [(Store::Latent, StoreRule::Window { slots: 128 })]
        );
        assert_eq!(
            stores(&v[2]),
            [
                (Store::Latent, StoreRule::Window { slots: 128 }),
                (Store::Compressed, StoreRule::Positional),
                (Store::CompressorState, StoreRule::Ratio { ratio: 2 }),
                (Store::IndexKeys, StoreRule::Positional),
            ]
        );
        assert_eq!(
            stores(&v[20]),
            [
                (Store::Latent, StoreRule::Window { slots: 128 }),
                (Store::Compressed, StoreRule::Positional),
                (Store::IndexKeys, StoreRule::Positional),
            ]
        );
        assert_eq!(
            stores(&v[21]),
            [(Store::Latent, StoreRule::Window { slots: 128 })]
        );
        let q = qwen36();
        assert_eq!(
            stores(&q[0]),
            [
                (Store::Conv, StoreRule::Window { slots: 3 }),
                (Store::State, StoreRule::Recurrent),
            ]
        );
        assert_eq!(stores(&q[3]), [(Store::Kv, StoreRule::Positional)]);
    }

    /// A GQA window is a `Window` store: not every position, and copied only
    /// while no shadow holds it. A full GQA layer stays positional.
    #[test]
    fn a_gqa_window_is_copied_when_nothing_shadows_it() {
        let mut q = qwen36();
        let Mixer::Gqa(g) = &mut q[3].mixer else {
            panic!("layer 3 is the GQA layer");
        };
        g.window = Some(128);
        assert_eq!(
            stores(&q[3]),
            [(Store::Kv, StoreRule::Window { slots: 128 })]
        );
        assert!(!every_position(&q[3]));
        assert!(copied(&q[3], false));
        assert!(!copied(&q[3], true));
        assert!(copied_layers(&q, &|l| l == 3).iter().all(|&l| l != 3));
    }

    /// A short-conv layer keeps the last `taps - 1` inputs of its conv and nothing else: a window
    /// store a checkpoint copies, read past no window. Mutant: the arm adds a recurrent state or
    /// sizes the ring `taps`.
    #[test]
    fn a_short_conv_layer_keeps_a_conv_ring_of_taps_minus_one() {
        let s = layer(Mixer::ShortConv(ShortConv { taps: 3 }), Residual::Plain);
        assert_eq!(stores(&s), [(Store::Conv, StoreRule::Window { slots: 2 })]);
        assert!(!every_position(&s));
        assert!(copied(&s, false));
    }

    /// A GLM-5.3-Flash-like trunk: KDA layers, every fourth a latent layer
    /// with its own index keys.
    fn glm() -> Vec<LayerSpec> {
        (0..45)
            .map(|l| {
                if l % 4 == 3 {
                    let mut s = v41_layer(l, None, Some((l, l)), false);
                    if let Mixer::Latent(a) = &mut s.mixer {
                        a.window = None;
                    }
                    s
                } else {
                    layer(
                        Mixer::DeltaRule(DeltaRule {
                            kind: DeltaKind::Kda {
                                gate: KdaGate::Sigmoid { lower_bound: -5.0 },
                                beta_scale: 1.0,
                            },
                            k_heads: 64,
                            v_heads: 64,
                            d: 128,
                            conv: 4,
                        }),
                        Residual::Hc,
                    )
                }
            })
            .collect()
    }

    /// A Qwen3.8-like trunk: Qwen3.6's, its GQA layers selecting by token
    /// pools.
    fn qwen38() -> Vec<LayerSpec> {
        let mut q = qwen36();
        for s in &mut q {
            if let Mixer::Gqa(g) = &mut s.mixer {
                g.select = Some(Selector::TokenPool {
                    heads: 4,
                    d: 128,
                    top_k: 2048,
                    pool: 64,
                    rule: PoolRule::Learned { key_eps: 1e-6 },
                });
            }
        }
        q
    }

    /// A GQA layer keeps one plane of K and V; one selecting by a block-max index also keeps an
    /// index key a position. Mutant: the arm for the selector dropped.
    #[test]
    fn a_block_max_gqa_layer_keeps_an_index_key_a_position() {
        let mut q = qwen36();
        let gqa = q
            .iter_mut()
            .find(|s| matches!(s.mixer, Mixer::Gqa(_)))
            .expect("Qwen3.6 has GQA layers");
        assert_eq!(stores(gqa), [(Store::Kv, StoreRule::Positional)]);
        if let Mixer::Gqa(g) = &mut gqa.mixer {
            g.select = Some(Selector::BlockMax {
                heads: 4,
                d: 128,
                top_k: 16,
                block: 128,
                local: 1,
                rope: rope(RopeMode::Neox, 64),
            });
        }
        assert_eq!(
            stores(gqa),
            [
                (Store::Kv, StoreRule::Positional),
                (Store::IndexKeys, StoreRule::Positional),
            ]
        );
    }

    /// The checkpoint copies a recurrent store whatever the shadow, and a
    /// window only when nothing shadows it. V4.1's latent windows are
    /// shadowed, so none of its layers are copied. Qwen3.6's and Qwen3.8's
    /// thirty GDN layers and GLM's thirty-four KDA layers are copied for
    /// their state. A Qwen GQA layer has no window here, so it is not.
    #[test]
    fn copied_layers_are_the_recurrent_ones() {
        let shadow = |_: usize| true;
        let bare = |_: usize| false;
        assert!(copied_layers(&v41(), &shadow).is_empty());
        assert_eq!(copied_layers(&v41(), &bare), (0..40).collect::<Vec<_>>());
        let delta: Vec<usize> = (0..40).filter(|l| l % 4 != 3).collect();
        assert_eq!(copied_layers(&qwen36(), &bare), delta);
        assert_eq!(copied_layers(&qwen38(), &shadow), delta);
        let kda: Vec<usize> = (0..45).filter(|l| l % 4 != 3).collect();
        assert_eq!(kda.len(), 34);
        assert_eq!(copied_layers(&glm(), &bare), kda);
        let g = glm();
        assert_eq!(
            stores(&g[3]),
            [
                (Store::Latent, StoreRule::Positional),
                (Store::IndexKeys, StoreRule::Positional),
            ]
        );
    }
}
