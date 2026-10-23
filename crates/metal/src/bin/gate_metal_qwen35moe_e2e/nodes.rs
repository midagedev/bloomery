//! `--nodes`: its set's prompt host-fed one token at a time through the
//! plain step (no store is loaded from the set; `--set` names it, the step
//! set the default), then its step token through the traced step, every
//! named node it can read against ik's dump of the same node. One line per
//! node: the layer, the name, the relative distance
//! (on ik's norm), the largest absolute error and the band; then the first
//! node past its band, the place a wrong value enters. A router whose picks
//! differ from ik's at a near tie (clause a's rule) changes every value after
//! it, so a node read after such a flip is past its band by the flip, not by
//! a defect: the summary names the flip and the first node past its band
//! before it.
//!
//! A tool, not a clause: its bands are a reading aid. A node of layer `l`
//! is held to `4e-2 · √(l + 1)`, the per-layer error [`super::free::FREE_BAND`]
//! is derived from, composed over the layers that feed it; the head's
//! logits to `FREE_BAND` itself.

use std::process::ExitCode;
use std::time::Instant;

use bloomery_metal::arch::qwen3moe::session::{Node, Op, Probe, Session, SessionError, StepMode};
use bloomery_metal::kernels::{
    EXPERT_K, HEAD_COLS, PARTIALS_PER_HEAD, ROUTED_IDS, SHARED_SLOT, router,
};
use refset::ik::RefManifest;

use super::flip::{self, rel_base};
use super::free::FREE_BAND;
use super::says;
use super::sets::{self, Set};

/// The per-layer error the free band composes.
const LAYER_ERROR: f64 = 4e-2;

/// One node's reading.
struct Reading {
    layer: Option<usize>,
    name: String,
    rel: f64,
    max_abs: f64,
    band: f64,
    /// Read after the step's first route flip.
    after_flip: bool,
}

/// The step's first router whose picks differ from ik's.
struct RouteFlip {
    layer: usize,
    margin: f64,
    err: f32,
    tie: bool,
}

fn band_of(l: usize) -> f64 {
    LAYER_ERROR * ((l + 1) as f64).sqrt()
}

fn shape_err(detail: String) -> SessionError {
    SessionError::Shape {
        what: "--nodes",
        detail,
    }
}

/// ik's node `name`, as the hook's error type.
fn ik(man: &RefManifest, name: &str) -> Result<Vec<f32>, SessionError> {
    sets::tap(man, name).map_err(shape_err)
}

/// The readings the step leaves, kept in order.
struct Nodes<'m> {
    man: &'m RefManifest,
    pos: u32,
    eps: f32,
    out: Vec<Reading>,
    /// ik's residual stream entering the layer running.
    ik_resid: Vec<f32>,
    flip: Option<RouteFlip>,
}

impl Nodes<'_> {
    fn push(
        &mut self,
        layer: Option<usize>,
        name: String,
        ours: &[f32],
        theirs: &[f32],
        band: f64,
    ) {
        let (rel, max_abs) = if ours.len() == theirs.len() {
            let rel = flip::worse(0.0, rel_base(ours, theirs, |i| f64::from(theirs[i])));
            let max_abs = ours
                .iter()
                .zip(theirs)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                .fold(0.0f64, flip::worse);
            (rel, max_abs)
        } else {
            (f64::INFINITY, f64::INFINITY)
        };
        self.out.push(Reading {
            layer,
            name,
            rel,
            max_abs,
            band,
            after_flip: self.flip.is_some(),
        });
    }

    /// `name-l` against our `node`.
    fn node(&mut self, l: usize, stem: &str, ours: &[f32]) -> Result<(), SessionError> {
        let name = format!("{stem}-{l}");
        let theirs = ik(self.man, &name)?;
        self.push(Some(l), name, ours, &theirs, band_of(l));
        Ok(())
    }

    fn after(&mut self, op: Op, p: &Probe<'_>) -> Result<(), SessionError> {
        match op {
            Op::Embed => {
                let theirs = ik(self.man, "inp_embd")?;
                self.push(
                    None,
                    "inp_embd".into(),
                    &p.read(Node::Residual)?,
                    &theirs,
                    band_of(0),
                );
                self.ik_resid = theirs;
            }
            Op::DeltaIn(l) => {
                self.node(l, "qkv_mixed", &p.read(Node::DeltaQkv(l))?)?;
                self.node(l, "z", &p.read(Node::DeltaZ(l))?)?;
                self.node(l, "beta", &p.read(Node::DeltaBeta(l))?)?;
                self.node(l, "alpha", &p.read(Node::DeltaAlpha(l))?)?;
            }
            Op::DeltaCore(l) => {
                // ik's attn_out_norm is the gated numerator times each head's
                // norm scale, which the out projection applies here.
                let g = p.read(Node::DeltaG)?;
                let parts = p.read(Node::DeltaPartials)?;
                let scaled: Vec<f32> = g
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let h = i / HEAD_COLS;
                        let sum: f32 = parts[h * PARTIALS_PER_HEAD..(h + 1) * PARTIALS_PER_HEAD]
                            .iter()
                            .sum();
                        v / (sum / HEAD_COLS as f32 + self.eps).sqrt()
                    })
                    .collect();
                self.node(l, "attn_out_norm", &scaled)?;
            }
            Op::DeltaOut(l) | Op::Experts(l) => {
                let resid = p.read(Node::Residual)?;
                let stem = if matches!(op, Op::DeltaOut(_)) {
                    "ssm_output"
                } else {
                    self.experts(l, p)?;
                    "l_out"
                };
                let name = format!("{stem}-{l}");
                let theirs = ik(self.man, &name)?;
                self.push(Some(l), name, &resid, &theirs, band_of(l));
                self.ik_resid = theirs;
            }
            Op::AttnIn(l) => {
                self.node(l, "Qaux", &p.read(Node::AttnQ(l))?)?;
                self.node(l, "Kcur", &p.read(Node::AttnK(l))?)?;
                self.node(l, "Vcur", &p.read(Node::AttnV(l))?)?;
            }
            Op::AttnCore(l) => {
                self.node(l, "Kcur_roped", &p.read(Node::KeyRow(l, self.pos))?)?;
                self.node(l, "Vcur", &p.read(Node::ValueRow(l, self.pos))?)?;
                self.node(l, "qkv_gated", &p.read(Node::AttnOut)?)?;
            }
            Op::AttnOut(l) => {
                // ik names the sum unstably; it is the residual entering the
                // layer plus attn_out.
                let attn_out = ik(self.man, &format!("attn_out-{l}"))?;
                let theirs: Vec<f32> = self
                    .ik_resid
                    .iter()
                    .zip(&attn_out)
                    .map(|(a, b)| a + b)
                    .collect();
                self.push(
                    Some(l),
                    format!("residual after attn_out-{l}"),
                    &p.read(Node::Residual)?,
                    &theirs,
                    band_of(l),
                );
                self.ik_resid = theirs;
            }
            Op::Route(l) => self.route(l, p)?,
            Op::Head => {
                let theirs = ik(self.man, "result_output")?;
                self.push(
                    None,
                    "result_output".into(),
                    &p.read(Node::Logits)?,
                    &theirs,
                    FREE_BAND,
                );
            }
        }
        Ok(())
    }

    /// The router's logits, its shared gate, and its picks against ik's.
    fn route(&mut self, l: usize, p: &Probe<'_>) -> Result<(), SessionError> {
        let logits = p.read(Node::RouterLogits(l))?;
        let experts = router::EXPERTS;
        self.node(l, "ffn_moe_logits", &logits[..experts])?;
        self.node(l, "shared_expert_gate", &logits[experts..])?;
        let ids = p.router_ids(l)?;
        let theirs = sets::topk(self.man, l, experts as u32).map_err(shape_err)?;
        let ours = &ids[..ROUTED_IDS];
        if ours.iter().all(|&e| theirs.contains(&(e as i32))) {
            let w = p.read(Node::RouterWeights(l))?;
            let ik_w = ik(self.man, &format!("ffn_moe_weights_norm-{l}"))?;
            // Our weights in ik's order: the sets are equal, so every one of
            // ik's ids is one of ours.
            let by_ik = theirs
                .iter()
                .map(|&e| {
                    ours.iter()
                        .position(|&o| o as i32 == e)
                        .map(|s| w[s])
                        .ok_or_else(|| shape_err(format!("layer {l}: ik's id {e} is not ours")))
                })
                .collect::<Result<Vec<f32>, _>>()?;
            self.push(
                Some(l),
                format!("ffn_moe_weights_norm-{l}"),
                &by_ik,
                &ik_w,
                band_of(l),
            );
            let gate = ik(self.man, &format!("shared_expert_gate-{l}"))?;
            let sig: Vec<f32> = gate.iter().map(|&g| 1.0 / (1.0 + (-g).exp())).collect();
            self.push(
                Some(l),
                format!("sigmoid(shared_expert_gate-{l})"),
                &w[SHARED_SLOT..],
                &sig,
                band_of(l),
            );
        } else {
            let lg = ik(self.man, &format!("ffn_moe_logits-{l}"))?;
            let margin = flip::margin(&lg, &theirs);
            let err = logits[..experts]
                .iter()
                .zip(&lg)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let tie = flip::router_tie(margin as f32, err);
            println!(
                "nodes: layer {l} picks differ: ours {ours:?}, ik {theirs:?}, ik margin {margin:.3e}, \
                 our logits' error {err:.3e}: {}",
                if tie { "a near tie" } else { "no near tie" }
            );
            if self.flip.is_none() {
                self.flip = Some(RouteFlip {
                    layer: l,
                    margin,
                    err,
                    tie,
                });
            }
        }
        Ok(())
    }

    /// The experts' activations, slot by id, and the shared expert's.
    fn experts(&mut self, l: usize, p: &Probe<'_>) -> Result<(), SessionError> {
        let h = p.read(Node::ExpertRows)?;
        let ids = p.router_ids(l)?;
        let theirs = sets::topk(self.man, l, router::EXPERTS as u32).map_err(shape_err)?;
        let ik_h = ik(self.man, &format!("ffn_moe_gate_par-{l}"))?;
        let mut ours_rows = Vec::new();
        let mut ik_rows = Vec::new();
        for (s, &e) in ids[..ROUTED_IDS].iter().enumerate() {
            if let Some(k) = theirs.iter().position(|&t| t == e as i32) {
                ours_rows.extend_from_slice(&h[s * EXPERT_K..(s + 1) * EXPERT_K]);
                ik_rows.extend_from_slice(&ik_h[k * EXPERT_K..(k + 1) * EXPERT_K]);
            }
        }
        self.push(
            Some(l),
            format!("ffn_moe_gate_par-{l} (the ids both picked)"),
            &ours_rows,
            &ik_rows,
            band_of(l),
        );
        let shared = &h[SHARED_SLOT * EXPERT_K..(SHARED_SLOT + 1) * EXPERT_K];
        self.node(l, "ffn_up_gate", shared)
    }
}

pub fn run(s: &mut Session, set: Set) -> ExitCode {
    let t0 = Instant::now();
    match go(s, set) {
        Ok(()) => {
            println!("nodes: one run in {:.1} s", t0.elapsed().as_secs_f64());
            ExitCode::SUCCESS
        }
        Err(e) => {
            println!("nodes: FAIL ({e})");
            ExitCode::FAILURE
        }
    }
}

fn go(s: &mut Session, set: Set) -> Result<(), String> {
    let man = sets::open(set)?;
    let (pos, step, prompt) = man.step().map_err(|e| format!("the set's tokens: {e}"))?;
    let (step, prompt) = (step.to_vec(), prompt.to_vec());
    let [tok] = step[..] else {
        return Err(format!(
            "the step set holds {} step tokens, the tool reads one",
            step.len()
        ));
    };
    s.reset().map_err(says)?;
    for &id in &prompt {
        s.step(id, StepMode::HostFed).map_err(says)?;
    }
    if s.pos() != pos {
        return Err(format!(
            "the prompt left position {}, the set steps at {pos}",
            s.pos()
        ));
    }
    let mut nodes = Nodes {
        man: &man,
        pos,
        eps: s.eps(),
        out: Vec::new(),
        ik_resid: Vec::new(),
        flip: None,
    };
    let stepped = s
        .step_traced(tok, |op, p| nodes.after(op, p))
        .map_err(says)?;
    let mut first = None;
    let mut past_after = 0usize;
    for r in &nodes.out {
        let past = r.rel.is_nan() || r.rel > r.band;
        println!(
            "node {:>5} {:<44} rel {:.3e} max_abs {:.3e} band {:.2e}{}",
            r.layer.map_or("-".to_string(), |l| l.to_string()),
            r.name,
            r.rel,
            r.max_abs,
            r.band,
            match (past, r.after_flip) {
                (false, _) => "",
                (true, false) => " PAST",
                (true, true) => " PAST (after the route flip)",
            }
        );
        if past && r.after_flip {
            past_after += 1;
        } else if past && first.is_none() {
            first = Some(r);
        }
    }
    let ik_logits = ik(&man, "result_output").map_err(says)?;
    println!(
        "nodes: step token {tok} at position {pos}: our pick {}, ik's argmax {}",
        stepped.id,
        flip::argmax(&ik_logits)
    );
    let flip_line = match &nodes.flip {
        None => "no route flip".to_string(),
        Some(f) => format!(
            "the first route flip at layer {} is {} (ik margin {:.3e}, our logits' error {:.3e}), \
             {past_after} nodes after it past their bands",
            f.layer,
            if f.tie { "a near tie" } else { "NO near tie" },
            f.margin,
            f.err
        ),
    };
    match first {
        Some(r) => println!(
            "nodes: first node past its band before any route flip: layer {} {} rel {:.3e} \
             max_abs {:.3e} band {:.2e}; {flip_line} ({} nodes read)",
            r.layer.map_or("-".to_string(), |l| l.to_string()),
            r.name,
            r.rel,
            r.max_abs,
            r.band,
            nodes.out.len()
        ),
        None => println!(
            "nodes: every node before the first route flip within its band; {flip_line} ({} \
             nodes read)",
            nodes.out.len()
        ),
    }
    Ok(())
}
