//! Each layer's tensor names, made once at load from its programs: the
//! step looks its weights up by these, and formats none.

use bloomery_gpu::GpuError;
use bloomery_gpu::site::{Form, SiteTy};
use bloomery_gpu::weights::Weights;
use gguf::{GgmlType, Split};
use model::arch::glm5next::names::{self, Sub};
use runtime::layer::{FfnKind, Layer, MixerKind};

/// A sub-layer's hyper-connection tensors.
pub(crate) struct HcNames {
    pub fn_: String,
    pub scale: String,
    pub base: String,
}

impl HcNames {
    fn of(l: usize, sub: Sub) -> HcNames {
        HcNames {
            fn_: names::hc_fn(l, sub),
            scale: names::hc_scale(l, sub),
            base: names::hc_base(l, sub),
        }
    }
}

/// A projection of one input whose weights the load joins row-wise into one
/// weight (`whole`) when its parts share a resident format ([`Body::derive`]),
/// and leaves as the parts' own weights otherwise: the launch is the one
/// joined weight's, or one by-name launch a part into its rows of the joined
/// layout.
pub(crate) struct Joined {
    pub whole: String,
    pub parts: Vec<String>,
}

/// The weights a joined projection launches, each with its first row, and the joined rows ([`Joined::launches`]).
pub(crate) type PartRows<'a> = (Vec<(&'a str, usize)>, usize);

impl Joined {
    /// The weights the projection launches, in row order, each with its
    /// first row in the joined layout, and the joined row count: the joined
    /// weight alone when it is resident, else the parts.
    pub(crate) fn launches<'a>(
        &'a self,
        what: &'static str,
        w: &Weights,
    ) -> Result<PartRows<'a>, GpuError> {
        if w.get(&self.whole).is_some() {
            let rows = w.resident(what, &self.whole)?.rows();
            return Ok((vec![(self.whole.as_str(), 0)], rows));
        }
        let mut at = 0;
        let mut out = Vec::with_capacity(self.parts.len());
        for p in &self.parts {
            out.push((p.as_str(), at));
            at += w.resident(what, p)?.rows();
        }
        Ok((out, at))
    }

    /// Whether the load joined the parts.
    pub(crate) fn is_joined(&self, w: &Weights) -> bool {
        w.get(&self.whole).is_some()
    }

    /// The names of the weights resident for this projection.
    pub(crate) fn resident<'a>(&'a self, w: &Weights) -> Vec<&'a str> {
        if self.is_joined(w) {
            vec![self.whole.as_str()]
        } else {
            self.parts.iter().map(String::as_str).collect()
        }
    }
}

/// A KDA mixer's tensors, the joined q·k·v and conv taps among them.
pub(crate) struct KdaNames {
    pub norm: String,
    pub qkv: Joined,
    pub f_a: String,
    pub g_a: String,
    pub beta: String,
    pub f_b: String,
    pub g_b: String,
    pub conv: String,
    pub dt_bias: String,
    pub a: String,
    pub gate_norm: String,
    pub out: String,
}

/// A latent mixer's tensors, the joined projection of the normed input
/// among them.
pub(crate) struct LatentNames {
    pub norm: String,
    pub stack: Joined,
    pub q_a_norm: String,
    pub kv_a_norm: String,
    pub index_norm: String,
    pub index_norm_bias: String,
    pub q_b: String,
    pub k_b: String,
    pub v_b: String,
    pub out: String,
    pub sel: SelNames,
}

/// The k-pool selector's own tensors: the indexer query's projection of the
/// low-rank query, the head weights' projection of the normed input, and
/// the pool gate's position bias.
pub(crate) struct SelNames {
    pub q_b: String,
    pub proj: String,
    pub ape: String,
}

/// A layer's mixer's tensors, by its kind.
pub(crate) enum MixerNames {
    Kda(KdaNames),
    Latent(LatentNames),
}

/// A layer's feed-forward block's tensors, by its kind: the routed block's
/// router, its selection bias and its shared expert (the card experts'
/// stacks are the card's own, [`crate::ffn`]).
pub(crate) enum FfnNames {
    Dense {
        norm: String,
        gate: String,
        up: String,
        down: String,
    },
    Moe {
        norm: String,
        router: String,
        bias: String,
        sh_gate: String,
        sh_up: String,
        sh_down: String,
    },
}

/// One layer's tensor names.
pub(crate) struct LayerNames {
    pub hc_attn: HcNames,
    pub hc_ffn: HcNames,
    pub mixer: MixerNames,
    pub ffn: FfnNames,
}

impl LayerNames {
    /// Layer `l`'s names by its programs `kind`; a GQA mixer, which
    /// glm5next has none of, is refused by name.
    pub(crate) fn of(l: usize, kind: Layer) -> Result<LayerNames, GpuError> {
        let mixer = match kind.mixer {
            MixerKind::DeltaRule => MixerNames::Kda(KdaNames {
                norm: names::attn_norm(l),
                qkv: Joined {
                    whole: names::attn_qkv(l),
                    parts: vec![names::attn_q(l), names::attn_k(l), names::attn_v(l)],
                },
                f_a: names::ssm_f_a(l),
                g_a: names::ssm_g_a(l),
                beta: names::ssm_beta(l),
                f_b: names::ssm_f_b(l),
                g_b: names::ssm_g_b(l),
                conv: names::ssm_conv1d_qkv(l),
                dt_bias: names::ssm_dt_bias(l),
                a: names::ssm_a(l),
                gate_norm: names::ssm_norm(l),
                out: names::attn_output(l),
            }),
            MixerKind::Latent => MixerNames::Latent(LatentNames {
                norm: names::attn_norm(l),
                stack: Joined {
                    whole: names::attn_a_stack(l),
                    parts: vec![
                        names::attn_q_a(l),
                        names::attn_kv_a_mqa(l),
                        names::indexer_attn_k(l),
                        names::indexer_compressor_gate(l),
                    ],
                },
                q_a_norm: names::attn_q_a_norm(l),
                kv_a_norm: names::attn_kv_a_norm(l),
                index_norm: names::indexer_k_norm(l),
                index_norm_bias: names::indexer_k_norm_bias(l),
                q_b: names::attn_q_b(l),
                k_b: names::attn_k_b(l),
                v_b: names::attn_v_b(l),
                out: names::attn_output(l),
                sel: SelNames {
                    q_b: names::indexer_attn_q_b(l),
                    proj: names::indexer_proj(l),
                    ape: names::indexer_compressor_ape(l),
                },
            }),
            MixerKind::Gqa => {
                return Err(GpuError::Shape {
                    what: "glm5next LayerNames",
                    detail: format!("layer {l}: a GQA mixer, which glm5next has none of"),
                });
            }
            MixerKind::Mamba2 => {
                return Err(GpuError::Shape {
                    what: "glm5next LayerNames",
                    detail: format!("layer {l}: a Mamba-2 mixer, which glm5next has none of"),
                });
            }
            MixerKind::ShortConv => {
                return Err(GpuError::Shape {
                    what: "glm5next LayerNames",
                    detail: format!("layer {l}: a short-conv mixer, which glm5next has none of"),
                });
            }
            MixerKind::None => {
                return Err(GpuError::Shape {
                    what: "glm5next LayerNames",
                    detail: format!("layer {l}: no token mixer, which glm5next has none of"),
                });
            }
        };
        let ffn = match kind.ffn {
            FfnKind::Dense => FfnNames::Dense {
                norm: names::ffn_norm(l),
                gate: names::ffn_gate(l),
                up: names::ffn_up(l),
                down: names::ffn_down(l),
            },
            FfnKind::Moe => FfnNames::Moe {
                norm: names::ffn_norm(l),
                router: names::ffn_gate_inp(l),
                bias: names::exp_probs_b(l),
                sh_gate: names::ffn_gate_shexp(l),
                sh_up: names::ffn_up_shexp(l),
                sh_down: names::ffn_down_shexp(l),
            },
        };
        Ok(LayerNames {
            hc_attn: HcNames::of(l, Sub::Attn),
            hc_ffn: HcNames::of(l, Sub::Ffn),
            mixer,
            ffn,
        })
    }

    /// The mixer's resident dense weights a by-name projection launches: the
    /// joined or the unjoined parts of the projections the load may leave
    /// apart, and the sites of one weight. The absorbed pair (`k_b`, `v_b`)
    /// is not among them: its per-head launch reads q8_0 planes.
    pub(crate) fn mixer_sites<'a>(&'a self, w: &Weights) -> Vec<&'a str> {
        match &self.mixer {
            MixerNames::Kda(k) => {
                let mut v = k.qkv.resident(w);
                v.extend([&k.f_a, &k.g_a, &k.beta, &k.f_b, &k.g_b, &k.out].map(String::as_str));
                v
            }
            MixerNames::Latent(n) => {
                let mut v = n.stack.resident(w);
                v.extend([&n.q_b, &n.sel.q_b, &n.sel.proj, &n.out].map(String::as_str));
                v
            }
        }
    }

    /// The launches the mixer's by-name projections add to the one launch a
    /// projection of one weight makes, from `file`'s header: each projection
    /// the load leaves apart ([`Joined`]) as many launches as it has parts,
    /// and each launch of a type that reads q8_1 rows (a K-quant, NVFP4) one
    /// more, the quantization of its input.
    pub(crate) fn mixer_extra_launches(&self, file: &Split) -> usize {
        let ty = |name: &str| file.find(name).map(|(_, t)| t.ty);
        let q8_1 = |t: GgmlType| SiteTy::of_ggml(t).is_some_and(|s| s.reads() == Form::Q8x128);
        let single = |name: &String| usize::from(ty(name).is_some_and(q8_1));
        // A joined projection launches once at its parts' common type; the
        // latent stack joins only as q8_0 planes ([`crate::body::joins_stack`]).
        let joined = |j: &Joined, only_q8_0: bool| {
            let types: Vec<Option<GgmlType>> = j.parts.iter().map(|p| ty(p)).collect();
            let same = types.windows(2).all(|p| p[0] == p[1]);
            let joins = same && (!only_q8_0 || types[0] == Some(GgmlType::Q8_0));
            if joins {
                usize::from(types[0].is_some_and(q8_1))
            } else {
                (j.parts.len() - 1) + j.parts.iter().map(&single).sum::<usize>()
            }
        };
        match &self.mixer {
            MixerNames::Kda(k) => {
                joined(&k.qkv, false)
                    + [&k.f_a, &k.g_a, &k.beta, &k.f_b, &k.g_b, &k.out]
                        .into_iter()
                        .map(single)
                        .sum::<usize>()
            }
            MixerNames::Latent(n) => {
                joined(&n.stack, true) + [&n.q_b, &n.out].into_iter().map(single).sum::<usize>()
            }
        }
    }

    /// The mixer's dense tensors by their file names, whatever the load
    /// joined: what the file's header says each site's type is.
    pub(crate) fn file_sites(&self) -> Vec<&str> {
        match &self.mixer {
            MixerNames::Kda(k) => {
                let mut v: Vec<&str> = k.qkv.parts.iter().map(String::as_str).collect();
                v.extend([&k.f_a, &k.g_a, &k.beta, &k.f_b, &k.g_b, &k.out].map(String::as_str));
                v
            }
            MixerNames::Latent(n) => {
                let mut v: Vec<&str> = n.stack.parts.iter().map(String::as_str).collect();
                v.extend([&n.q_b, &n.sel.q_b, &n.sel.proj, &n.out].map(String::as_str));
                v
            }
        }
    }

    /// Sub-layer `sub`'s hyper-connection names.
    pub(crate) fn hc(&self, sub: Sub) -> &HcNames {
        match sub {
            Sub::Attn => &self.hc_attn,
            Sub::Ffn => &self.hc_ffn,
        }
    }
}

/// The error of a launch that finds another kind's names at its layer: the
/// names follow the programs, so this is a wiring fault, named.
pub(crate) fn other_kind(what: &'static str, l: usize) -> GpuError {
    GpuError::Shape {
        what,
        detail: format!("layer {l}: its names are another kind's than this launch runs"),
    }
}
