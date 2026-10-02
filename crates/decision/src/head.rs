//! Clef's joint schema head (`JointSchemaHead.forward` for one record), on the host in f32.
//!
//! The layer list is Clef's: `hidden_norm`, the six 4096 → width projections, the type embedding,
//! `routing_layers` evidence routing layers (pre-norm attention over a LayerNorm'd memory, then a
//! GELU feed-forward), `option_summary_norm`, `layers` torch `TransformerDecoderLayer`s
//! (`norm_first`, gelu: self-attention, cross-attention over the raw memory, feed-forward),
//! `field_norm`, `option_norm`, the residual scorer and three scalars. The weight names are the
//! module's `state_dict` names; a missing or extra name is refused by name.

use std::path::Path;
use std::time::Instant;

use crate::Error;
use crate::encode::Encoded;
use crate::json::{self, Json};
use crate::ops::{self, Mha, dot, gelu, layer_norm, layer_normed, linear, normalized};
use crate::safetensors::Safetensors;

/// `joint_head_config.json`: the head's shape (`dropout` is accepted and has no effect at inference).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadConfig {
    pub hidden_size: usize,
    pub width: usize,
    pub routing_layers: usize,
    pub layers: usize,
    pub heads: usize,
    pub feedforward: usize,
}

impl HeadConfig {
    /// Parse the config file's text; an unknown or missing key is refused by name.
    pub fn parse(text: &str) -> Result<HeadConfig, Error> {
        let Json::Object(pairs) = json::parse(text)? else {
            return Err(Error::HeadConfig("the config is not a JSON object".into()));
        };
        let mut vals = [None; 6];
        const KEYS: [&str; 6] = [
            "hidden_size",
            "width",
            "routing_layers",
            "layers",
            "heads",
            "feedforward",
        ];
        for (k, v) in &pairs {
            if k == "dropout" {
                continue;
            }
            let i = KEYS
                .iter()
                .position(|key| key == k)
                .ok_or_else(|| Error::HeadConfig(format!("unknown key {k:?}")))?;
            let n = v
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| {
                    Error::HeadConfig(format!("{k} is {}, not a whole number", v.kind()))
                })?;
            vals[i] = Some(n);
        }
        let get = |i: usize| vals[i].ok_or_else(|| Error::HeadConfig(format!("no {}", KEYS[i])));
        let cfg = HeadConfig {
            hidden_size: get(0)?,
            width: get(1)?,
            routing_layers: get(2)?,
            layers: get(3)?,
            heads: get(4)?,
            feedforward: get(5)?,
        };
        if cfg.heads == 0 || !cfg.width.is_multiple_of(cfg.heads) {
            return Err(Error::HeadConfig(format!(
                "width {} does not split into {} heads",
                cfg.width, cfg.heads
            )));
        }
        Ok(cfg)
    }

    /// Every `state_dict` name of the head with its shape (`[]` is a scalar).
    #[must_use]
    pub fn weights(&self) -> Vec<(String, Vec<usize>)> {
        let (h, w, f) = (self.hidden_size, self.width, self.feedforward);
        let mut v: Vec<(String, Vec<usize>)> = Vec::new();
        let mut add = |name: String, shape: &[usize]| v.push((name, shape.to_vec()));
        let norm = |add: &mut dyn FnMut(String, &[usize]), p: &str, n: usize| {
            add(format!("{p}.weight"), &[n]);
            add(format!("{p}.bias"), &[n]);
        };
        let mha = |add: &mut dyn FnMut(String, &[usize]), p: &str| {
            add(format!("{p}.in_proj_weight"), &[3 * w, w]);
            add(format!("{p}.in_proj_bias"), &[3 * w]);
            add(format!("{p}.out_proj.weight"), &[w, w]);
            add(format!("{p}.out_proj.bias"), &[w]);
        };
        norm(&mut add, "hidden_norm", h);
        for p in [
            "memory_projection",
            "question_projection",
            "option_question_projection",
            "global_projection",
            "option_context_projection",
            "option_lexical_projection",
        ] {
            add(format!("{p}.weight"), &[w, h]);
        }
        add("type_embedding.weight".into(), &[3, w]);
        for i in 0..self.routing_layers {
            let p = format!("evidence_layers.{i}");
            norm(&mut add, &format!("{p}.query_norm"), w);
            norm(&mut add, &format!("{p}.memory_norm"), w);
            mha(&mut add, &format!("{p}.attention"));
            norm(&mut add, &format!("{p}.feedforward_norm"), w);
            add(format!("{p}.feedforward.0.weight"), &[f, w]);
            add(format!("{p}.feedforward.0.bias"), &[f]);
            add(format!("{p}.feedforward.3.weight"), &[w, f]);
            add(format!("{p}.feedforward.3.bias"), &[w]);
        }
        norm(&mut add, "option_summary_norm", w);
        for i in 0..self.layers {
            let p = format!("layers.{i}");
            mha(&mut add, &format!("{p}.self_attn"));
            mha(&mut add, &format!("{p}.multihead_attn"));
            add(format!("{p}.linear1.weight"), &[f, w]);
            add(format!("{p}.linear1.bias"), &[f]);
            add(format!("{p}.linear2.weight"), &[w, f]);
            add(format!("{p}.linear2.bias"), &[w]);
            for n in ["norm1", "norm2", "norm3"] {
                norm(&mut add, &format!("{p}.{n}"), w);
            }
        }
        norm(&mut add, "field_norm", w);
        norm(&mut add, "option_norm", w);
        add("residual_scorer.0.weight".into(), &[w, 4 * w]);
        add("residual_scorer.0.bias".into(), &[w]);
        add("residual_scorer.3.weight".into(), &[1, w]);
        add("residual_scorer.3.bias".into(), &[1]);
        for s in ["prior_logit_scale", "joint_logit_scale", "residual_gate"] {
            add(s.into(), &[]);
        }
        v
    }
}

struct Norm {
    w: Vec<f32>,
    b: Vec<f32>,
}

struct Ffn {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl Ffn {
    fn apply(&self, x: &[f32], width: usize) -> Vec<f32> {
        let mut h = linear(x, width, &self.w1, Some(&self.b1));
        for v in &mut h {
            *v = gelu(*v);
        }
        linear(&h, self.w1.len() / width, &self.w2, Some(&self.b2))
    }
}

struct Evidence {
    query_norm: Norm,
    memory_norm: Norm,
    attention: Mha,
    feedforward_norm: Norm,
    ffn: Ffn,
}

struct Decoder {
    self_attn: Mha,
    cross: Mha,
    ffn: Ffn,
    norm1: Norm,
    norm2: Norm,
    norm3: Norm,
}

/// The loaded head.
pub struct ClefHead {
    cfg: HeadConfig,
    hidden_norm: Norm,
    memory_projection: Vec<f32>,
    question_projection: Vec<f32>,
    option_question_projection: Vec<f32>,
    global_projection: Vec<f32>,
    option_context_projection: Vec<f32>,
    option_lexical_projection: Vec<f32>,
    type_embedding: Vec<f32>,
    evidence: Vec<Evidence>,
    option_summary_norm: Norm,
    decoder: Vec<Decoder>,
    field_norm: Norm,
    option_norm: Norm,
    scorer0_w: Vec<f32>,
    scorer0_b: Vec<f32>,
    scorer3_w: Vec<f32>,
    scorer3_b: f32,
    prior_logit_scale: f32,
    joint_logit_scale: f32,
    residual_gate: f32,
}

/// The output rows of a list of token ids: `[ids.len(), hidden]` row-major, or why not.
pub type Rows<'a> = dyn FnMut(&[u32]) -> Result<Vec<f32>, String> + 'a;

impl ClefHead {
    /// The head at `path` with its config: `config`, or `joint_head_config.json` beside the file.
    pub fn open(path: &Path, config: Option<&Path>) -> Result<ClefHead, Error> {
        let cfg_path = config.map_or_else(
            || path.with_file_name("joint_head_config.json"),
            Path::to_path_buf,
        );
        let text = std::fs::read_to_string(&cfg_path)
            .map_err(|e| Error::Io(cfg_path.display().to_string(), e))?;
        ClefHead::from_safetensors(&Safetensors::open(path)?, HeadConfig::parse(&text)?)
    }

    /// The head from a parsed file: its names and shapes must be exactly `cfg.weights()`.
    pub fn from_safetensors(st: &Safetensors, cfg: HeadConfig) -> Result<ClefHead, Error> {
        let want = cfg.weights();
        let mut missing = Vec::new();
        let mut shapes = Vec::new();
        for (name, shape) in &want {
            match st.entry(name) {
                None => missing.push(name.clone()),
                Some(e) if e.shape != *shape => {
                    shapes.push(format!("{name} {:?} (want {shape:?})", e.shape))
                }
                Some(_) => {}
            }
        }
        let extra: Vec<String> = st
            .names()
            .filter(|n| !want.iter().any(|(w, _)| w == n))
            .map(str::to_string)
            .collect();
        if !(missing.is_empty() && extra.is_empty() && shapes.is_empty()) {
            return Err(Error::Weights {
                missing,
                extra,
                shapes,
            });
        }
        let t = |name: &str| st.tensor(name).map(|t| t.data);
        let s = |name: &str| -> Result<f32, Error> { Ok(t(name)?[0]) };
        let norm = |p: &str| -> Result<Norm, Error> {
            Ok(Norm {
                w: t(&format!("{p}.weight"))?,
                b: t(&format!("{p}.bias"))?,
            })
        };
        let mha = |p: &str| -> Result<Mha, Error> {
            Ok(Mha {
                in_w: t(&format!("{p}.in_proj_weight"))?,
                in_b: t(&format!("{p}.in_proj_bias"))?,
                out_w: t(&format!("{p}.out_proj.weight"))?,
                out_b: t(&format!("{p}.out_proj.bias"))?,
                width: cfg.width,
                heads: cfg.heads,
            })
        };
        let ffn = |p: &str, a: &str, b: &str| -> Result<Ffn, Error> {
            Ok(Ffn {
                w1: t(&format!("{p}.{a}.weight"))?,
                b1: t(&format!("{p}.{a}.bias"))?,
                w2: t(&format!("{p}.{b}.weight"))?,
                b2: t(&format!("{p}.{b}.bias"))?,
            })
        };
        let evidence = (0..cfg.routing_layers)
            .map(|i| {
                let p = format!("evidence_layers.{i}");
                Ok(Evidence {
                    query_norm: norm(&format!("{p}.query_norm"))?,
                    memory_norm: norm(&format!("{p}.memory_norm"))?,
                    attention: mha(&format!("{p}.attention"))?,
                    feedforward_norm: norm(&format!("{p}.feedforward_norm"))?,
                    ffn: ffn(&format!("{p}.feedforward"), "0", "3")?,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let decoder = (0..cfg.layers)
            .map(|i| {
                let p = format!("layers.{i}");
                Ok(Decoder {
                    self_attn: mha(&format!("{p}.self_attn"))?,
                    cross: mha(&format!("{p}.multihead_attn"))?,
                    ffn: ffn(&p, "linear1", "linear2")?,
                    norm1: norm(&format!("{p}.norm1"))?,
                    norm2: norm(&format!("{p}.norm2"))?,
                    norm3: norm(&format!("{p}.norm3"))?,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(ClefHead {
            cfg,
            hidden_norm: norm("hidden_norm")?,
            memory_projection: t("memory_projection.weight")?,
            question_projection: t("question_projection.weight")?,
            option_question_projection: t("option_question_projection.weight")?,
            global_projection: t("global_projection.weight")?,
            option_context_projection: t("option_context_projection.weight")?,
            option_lexical_projection: t("option_lexical_projection.weight")?,
            type_embedding: t("type_embedding.weight")?,
            evidence,
            option_summary_norm: norm("option_summary_norm")?,
            decoder,
            field_norm: norm("field_norm")?,
            option_norm: norm("option_norm")?,
            scorer0_w: t("residual_scorer.0.weight")?,
            scorer0_b: t("residual_scorer.0.bias")?,
            scorer3_w: t("residual_scorer.3.weight")?,
            scorer3_b: s("residual_scorer.3.bias")?,
            prior_logit_scale: s("prior_logit_scale")?,
            joint_logit_scale: s("joint_logit_scale")?,
            residual_gate: s("residual_gate")?,
        })
    }

    /// The head's shape.
    #[must_use]
    pub fn config(&self) -> HeadConfig {
        self.cfg
    }

    /// The three scalars as the logits use them: `exp(min(prior_logit_scale, ln 100))`,
    /// `exp(min(joint_logit_scale, ln 100))` and `sigmoid(residual_gate)`.
    #[must_use]
    pub fn scales(&self) -> (f32, f32, f32) {
        let cap = 100f32.ln();
        (
            self.prior_logit_scale.min(cap).exp(),
            self.joint_logit_scale.min(cap).exp(),
            1.0 / (1.0 + (-self.residual_gate).exp()),
        )
    }

    /// Per question, one logit per option in span order. `hidden` is the backbone's last hidden
    /// state, `[enc.ids.len(), hidden_size]` row-major; `rows` gives the output embedding rows of the
    /// ids inside the option spans, all spans in order, in one call.
    pub fn forward(
        &self,
        hidden: &[f32],
        enc: &Encoded,
        rows: &mut Rows<'_>,
    ) -> Result<Vec<Vec<f32>>, Error> {
        self.forward_staged(hidden, enc, rows, &mut Vec::new())
    }

    /// [`ClefHead::forward`], each stage's wall in ms appended to `stages` in order: `norm`,
    /// `memory`, `rows`, `means`, `route`, `fields`, `score`.
    pub fn forward_staged(
        &self,
        hidden: &[f32],
        enc: &Encoded,
        rows: &mut Rows<'_>,
        stages: &mut Vec<(&'static str, f64)>,
    ) -> Result<Vec<Vec<f32>>, Error> {
        let hs = self.cfg.hidden_size;
        let n = enc.ids.len();
        if hidden.len() != n * hs {
            return Err(Error::HiddenShape {
                values: hidden.len(),
                ids: n,
                width: hs,
            });
        }
        check_spans(enc)?;
        let mut clock = Clock(Instant::now(), stages);
        let p = self.pool(hidden, enc, rows, &mut clock)?;
        let options = self.route(&p);
        clock.lap("route");
        let fields = self.fields(&p, enc, &options);
        clock.lap("fields");
        let logits = self.score(&p, enc, &options, &fields);
        clock.lap("score");
        logits
    }

    /// The normalized hidden states, the memory and the span means.
    fn pool(
        &self,
        hidden: &[f32],
        enc: &Encoded,
        rows: &mut Rows<'_>,
        clock: &mut Clock<'_>,
    ) -> Result<Pooled, Error> {
        let hs = self.cfg.hidden_size;
        let nh = layer_normed(hidden, hs, &self.hidden_norm.w, &self.hidden_norm.b);
        clock.lap("norm");
        let memory = linear(&nh, hs, &self.memory_projection, None);
        clock.lap("memory");
        let mean = |src: &[f32], (a, b): (usize, usize)| -> Vec<f32> {
            let mut m = vec![0f32; hs];
            for row in src[a * hs..b * hs].chunks_exact(hs) {
                for (m, v) in m.iter_mut().zip(row) {
                    *m += v;
                }
            }
            let inv = (b - a) as f32;
            m.iter_mut().for_each(|v| *v /= inv);
            m
        };
        let qvecs: Vec<f32> = enc
            .questions
            .iter()
            .flat_map(|q| mean(&nh, q.question_span))
            .collect();
        let lex_ids: Vec<u32> = enc
            .questions
            .iter()
            .flat_map(|q| {
                q.option_spans
                    .iter()
                    .flat_map(|&(a, b)| enc.ids[a..b].iter().copied())
            })
            .collect();
        let lex_rows = rows(&lex_ids).map_err(Error::Rows)?;
        clock.lap("rows");
        if lex_rows.len() != lex_ids.len() * hs {
            return Err(Error::Rows(format!(
                "{} values for {} ids of width {hs}",
                lex_rows.len(),
                lex_ids.len()
            )));
        }
        let (mut ctx, mut lex, mut owner) = (Vec::new(), Vec::new(), Vec::new());
        let mut at = 0;
        for (qi, q) in enc.questions.iter().enumerate() {
            for &(a, b) in &q.option_spans {
                ctx.extend(mean(&nh, (a, b)));
                lex.extend(mean(&lex_rows[at * hs..], (0, b - a)));
                at += b - a;
                owner.push(qi);
            }
        }
        let global = nh[(enc.ids.len() - 1) * hs..].to_vec();
        clock.lap("means");
        Ok(Pooled {
            memory,
            global,
            qvecs,
            ctx,
            lex,
            owner,
        })
    }

    /// The option queries through the evidence routing layers.
    fn route(&self, p: &Pooled) -> Vec<f32> {
        let (hs, w) = (self.cfg.hidden_size, self.cfg.width);
        let ctx_p = linear(&p.ctx, hs, &self.option_context_projection, None);
        let lex_p = linear(&p.lex, hs, &self.option_lexical_projection, None);
        let q_p = linear(&p.qvecs, hs, &self.option_question_projection, None);
        let mut options: Vec<f32> = (0..p.owner.len() * w)
            .map(|i| ctx_p[i] + lex_p[i] + q_p[p.owner[i / w] * w + i % w])
            .collect();
        for layer in &self.evidence {
            let mem_n = layer_normed(&p.memory, w, &layer.memory_norm.w, &layer.memory_norm.b);
            let qn = layer_normed(&options, w, &layer.query_norm.w, &layer.query_norm.b);
            add(&mut options, &layer.attention.attend_memory(&qn, &mem_n));
            let fin = layer_normed(
                &options,
                w,
                &layer.feedforward_norm.w,
                &layer.feedforward_norm.b,
            );
            add(&mut options, &layer.ffn.apply(&fin, w));
        }
        options
    }

    /// The fields: each question's projection, its routed option summary, the global vector and
    /// the type embedding, through the decoder layers and `field_norm`.
    fn fields(&self, p: &Pooled, enc: &Encoded, options: &[f32]) -> Vec<f32> {
        let (hs, w) = (self.cfg.hidden_size, self.cfg.width);
        let base = linear(&p.qvecs, hs, &self.question_projection, None);
        let mut summaries = Vec::with_capacity(enc.questions.len() * w);
        let mut first = 0;
        for (qi, q) in enc.questions.iter().enumerate() {
            let opts = &options[first * w..(first + q.option_spans.len()) * w];
            let field = &base[qi * w..(qi + 1) * w];
            let root = (w as f32).sqrt();
            let mut weights: Vec<f32> =
                opts.chunks_exact(w).map(|o| dot(o, field) / root).collect();
            ops::softmax(&mut weights);
            let mut s = vec![0f32; w];
            for (o, p) in opts.chunks_exact(w).zip(&weights) {
                for (s, v) in s.iter_mut().zip(o) {
                    *s += p * v;
                }
            }
            summaries.extend(s);
            first += q.option_spans.len();
        }
        layer_norm(
            &mut summaries,
            w,
            &self.option_summary_norm.w,
            &self.option_summary_norm.b,
        );
        let g = linear(&p.global, hs, &self.global_projection, None);
        let mut fields: Vec<f32> = (0..enc.questions.len() * w)
            .map(|i| {
                let t = enc.questions[i / w].kind.index();
                base[i] + summaries[i] + g[i % w] + self.type_embedding[t * w + i % w]
            })
            .collect();
        for layer in &self.decoder {
            let x1 = layer_normed(&fields, w, &layer.norm1.w, &layer.norm1.b);
            let self_kv = layer.self_attn.kv(&x1);
            add(&mut fields, &layer.self_attn.attend(&x1, &self_kv));
            let x2 = layer_normed(&fields, w, &layer.norm2.w, &layer.norm2.b);
            add(&mut fields, &layer.cross.attend_memory(&x2, &p.memory));
            let x3 = layer_normed(&fields, w, &layer.norm3.w, &layer.norm3.b);
            add(&mut fields, &layer.ffn.apply(&x3, w));
        }
        layer_norm(&mut fields, w, &self.field_norm.w, &self.field_norm.b);
        fields
    }

    /// The logits: `prior + sigmoid(gate) · (joint_scale · cosine + residual)` per option.
    fn score(
        &self,
        p: &Pooled,
        enc: &Encoded,
        options: &[f32],
        fields: &[f32],
    ) -> Result<Vec<Vec<f32>>, Error> {
        let (hs, w) = (self.cfg.hidden_size, self.cfg.width);
        let (prior_scale, joint_scale, gate) = self.scales();
        let mut logits = Vec::with_capacity(enc.questions.len());
        let mut first = 0;
        for (qi, q) in enc.questions.iter().enumerate() {
            let k = q.option_spans.len();
            let anchor_in: Vec<f32> = p.qvecs[qi * hs..(qi + 1) * hs]
                .iter()
                .zip(&p.global)
                .map(|(a, b)| a + b)
                .collect();
            let anchor = normalized(&anchor_in, 1e-12);
            let field = &fields[qi * w..(qi + 1) * w];
            let field_n = normalized(field, 1e-8);
            let opts = layer_normed(
                &options[first * w..(first + k) * w],
                w,
                &self.option_norm.w,
                &self.option_norm.b,
            );
            let mut features = Vec::with_capacity(k * 4 * w);
            for o in opts.chunks_exact(w) {
                features.extend_from_slice(field);
                features.extend_from_slice(o);
                features.extend(field.iter().zip(o).map(|(f, o)| f * o));
                features.extend(field.iter().zip(o).map(|(f, o)| (f - o).abs()));
            }
            let mut hidden_r = linear(&features, 4 * w, &self.scorer0_w, Some(&self.scorer0_b));
            hidden_r.iter_mut().for_each(|v| *v = gelu(*v));
            let residual = linear(&hidden_r, w, &self.scorer3_w, None);
            let mut out = Vec::with_capacity(k);
            for (oi, o) in opts.chunks_exact(w).enumerate() {
                let lex_a = normalized(&p.lex[(first + oi) * hs..(first + oi + 1) * hs], 1e-12);
                let prior = prior_scale * dot(&lex_a, &anchor);
                let cosine = dot(&field_n, &normalized(o, 1e-8));
                let joint = joint_scale * cosine + (residual[oi] + self.scorer3_b);
                let logit = prior + gate * joint;
                if !logit.is_finite() {
                    return Err(Error::NonFinite {
                        question: q.id.clone(),
                        option: q.option_ids[oi].clone(),
                    });
                }
                out.push(logit);
            }
            logits.push(out);
            first += k;
        }
        Ok(logits)
    }
}

/// What the head pools from the hidden states: the memory `[n, width]`, the last position's
/// normalized state, the question span means `[questions, hidden]`, the option span means and the
/// options' lexical means `[options, hidden]`, and each option's question.
/// `name=ms` for each of [`ClefHead::forward_staged`]'s stages, space-separated.
#[must_use]
pub fn stage_line(stages: &[(&'static str, f64)]) -> String {
    stages
        .iter()
        .map(|(s, ms)| format!("{s}={ms:.1}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The stage walls of one forward: each lap appends the ms since the last.
struct Clock<'a>(Instant, &'a mut Vec<(&'static str, f64)>);

impl Clock<'_> {
    fn lap(&mut self, stage: &'static str) {
        let now = Instant::now();
        self.1.push((stage, (now - self.0).as_secs_f64() * 1e3));
        self.0 = now;
    }
}

struct Pooled {
    memory: Vec<f32>,
    global: Vec<f32>,
    qvecs: Vec<f32>,
    ctx: Vec<f32>,
    lex: Vec<f32>,
    owner: Vec<usize>,
}

fn add(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// Every span lies inside the ids and is not empty (an empty one would make the release's mean a NaN).
fn check_spans(enc: &Encoded) -> Result<(), Error> {
    let n = enc.ids.len();
    for q in &enc.questions {
        let spans = std::iter::once(("instructions", q.question_span))
            .chain(q.option_spans.iter().map(|&s| ("an option", s)));
        for (what, (a, b)) in spans {
            if a >= b {
                return Err(Error::EmptySpan {
                    question: q.id.clone(),
                    what,
                });
            }
            if b > n {
                return Err(Error::SpanOutside {
                    question: q.id.clone(),
                    span: (a, b),
                    ids: n,
                });
            }
        }
        if q.option_spans.len() != q.option_ids.len() || q.option_spans.is_empty() {
            return Err(Error::Options {
                question: q.id.clone(),
                spans: q.option_spans.len(),
                ids: q.option_ids.len(),
            });
        }
    }
    if enc.questions.is_empty() || n == 0 {
        return Err(Error::Request(
            "the record produced no model input or questions".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::encode_with;
    use crate::request::Request;
    use crate::safetensors::write;

    const CFG: HeadConfig = HeadConfig {
        hidden_size: 16,
        width: 8,
        routing_layers: 2,
        layers: 1,
        heads: 2,
        feedforward: 12,
    };

    fn bf16(n: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .flat_map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let x = (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                ((x.to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect()
    }

    fn file(skip: Option<&str>, extra: bool) -> Safetensors {
        let weights = CFG.weights();
        let raw: Vec<Vec<u8>> = weights
            .iter()
            .enumerate()
            .map(|(i, (_, s))| bf16(s.iter().product(), u32::try_from(i).unwrap()))
            .collect();
        let mut list: Vec<(&str, &str, &[usize], &[u8])> = weights
            .iter()
            .zip(&raw)
            .filter(|((n, _), _)| Some(n.as_str()) != skip)
            .map(|((n, s), r)| (n.as_str(), "BF16", s.as_slice(), r.as_slice()))
            .collect();
        let two = bf16(1, 99);
        if extra {
            list.push(("stray.weight", "BF16", &[], &two));
        }
        Safetensors::parse(write(&list)).unwrap()
    }

    #[test]
    fn the_config_reads_and_refuses_by_name() {
        let cfg = HeadConfig::parse(
            r#"{"hidden_size": 16, "width": 8, "routing_layers": 2, "layers": 1, "heads": 2, "feedforward": 12, "dropout": 0.1}"#,
        )
        .unwrap();
        assert_eq!(cfg, CFG);
        for (text, want) in [
            (r#"{"hidden_size": 16}"#, "no width"),
            (
                r#"{"hidden_size": 16, "width": 8, "routing_layers": 2, "layers": 1, "heads": 3, "feedforward": 12}"#,
                "does not split",
            ),
            (r#"{"hidden": 16}"#, "unknown key"),
        ] {
            let e = HeadConfig::parse(text).unwrap_err().to_string();
            assert!(e.contains(want), "{text}: {e}");
        }
    }

    #[test]
    fn a_missing_or_extra_weight_is_refused_by_name() {
        assert!(ClefHead::from_safetensors(&file(None, false), CFG).is_ok());
        match ClefHead::from_safetensors(&file(Some("layers.0.norm2.bias"), true), CFG) {
            Err(Error::Weights {
                missing,
                extra,
                shapes,
            }) => {
                assert_eq!(missing, ["layers.0.norm2.bias"]);
                assert_eq!(extra, ["stray.weight"]);
                assert!(shapes.is_empty());
            }
            other => panic!("{:?}", other.err()),
        }
        let wide = HeadConfig {
            feedforward: 10,
            ..CFG
        };
        assert!(
            matches!(ClefHead::from_safetensors(&file(None, false), wide), Err(Error::Weights { ref shapes, .. }) if !shapes.is_empty())
        );
    }

    #[test]
    fn forward_scores_every_option_and_checks_its_inputs() {
        let head = ClefHead::from_safetensors(&file(None, false), CFG).unwrap();
        let req = Request::from_json(
            &json::parse(r#"{"model":"m","state":"st","questions":{"a":{"type":"choice","criteria":{"x":1,"y":2,"z":3}},"b":{"type":"noul"}}}"#)
                .unwrap(),
        )
        .unwrap();
        let enc = encode_with(
            &mut |t: &str| t.bytes().map(u32::from).collect(),
            &req,
            16384,
        )
        .unwrap();
        let n = enc.ids.len();
        let hidden: Vec<f32> = (0..n * 16)
            .map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0)
            .collect();
        let mut rows = |ids: &[u32]| -> Result<Vec<f32>, String> {
            Ok(ids
                .iter()
                .flat_map(|&id| (0..16).map(move |j| (id as f32 * 0.01 + j as f32 * 0.1).sin()))
                .collect())
        };
        let logits = head.forward(&hidden, &enc, &mut rows).unwrap();
        assert_eq!(logits.iter().map(Vec::len).collect::<Vec<_>>(), [3, 2]);
        assert!(logits.iter().flatten().all(|l| l.is_finite()));
        assert!(matches!(
            head.forward(&hidden[16..], &enc, &mut rows),
            Err(Error::HiddenShape { .. })
        ));
        let mut short =
            |ids: &[u32]| -> Result<Vec<f32>, String> { Ok(vec![0.0; (ids.len() - 1) * 16]) };
        assert!(matches!(
            head.forward(&hidden, &enc, &mut short),
            Err(Error::Rows(_))
        ));
        let mut nan = hidden.clone();
        nan[3] = f32::NAN;
        assert!(matches!(
            head.forward(&nan, &enc, &mut rows),
            Err(Error::NonFinite { .. })
        ));
        let mut empty = enc.clone();
        empty.questions[1].option_spans[0].1 = empty.questions[1].option_spans[0].0;
        assert!(matches!(
            head.forward(&hidden, &empty, &mut rows),
            Err(Error::EmptySpan {
                what: "an option",
                ..
            })
        ));
    }
}
