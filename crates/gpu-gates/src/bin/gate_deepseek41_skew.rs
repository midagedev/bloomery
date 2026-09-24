//! GPU gate for the V4.1 pair pass (`Body::enqueue_pair`, C4 `ktok-skew`):
//! two tokens, `t` at position `p` and `t1` at `p + 1`, run as rows one
//! layer apart on one stream, against the same two tokens run as two
//! one-token steps in turn — on the gate placement (`workstation::plan_gate`),
//! the model opened through the engine's own entry (`body::open`).
//!
//! - `--sets`: on the decode-step sets `step4` (position 4) and `d1`
//!   (position 301, ik's `top_k` overridden to 64, so every indexer layer
//!   selects), each set's state injected as the step gate injects it, `t` the
//!   set's token and `t1` the greedy token after it. (i) two steps in turn,
//!   (ii) the pair pass eagerly, (iii) a capture of the pair pass replayed:
//!   bit for bit, both rows' logits, every layer's window ring, compressed
//!   rows, index keys and compressor state, each row's streams, folds and
//!   lists, and the history. Then (iv) the pair pass, the rollback of
//!   position `p + 1`, and one step of another token `t2` (the runner-up
//!   after `t`) against the steps `t`, `t2` in turn, bit for bit on the
//!   same buffers; (v) the pair pass and a cut of both its positions, granted
//!   exactly where the ring slots the step at `p` reads still hold their rows
//!   (`step4`: the ring never wrapped) and refused where a restore would need
//!   a shadow row of the injected history (`d1`), then `t`, `t2` against the
//!   same two steps.
//! - `--structure`: the pair pass captured holds, per node kind and per
//!   kernel name, exactly twice the one-token step's capture, and nothing
//!   else; its memory-operation batches come in the pass's order (both rows'
//!   layer-0 goes, then per layer each row's wait of the layer before and go,
//!   then both last waits); its prologue is each row's gather and
//!   broadcast, then row 0's layer 0 up to its go; and each go's shadow —
//!   the kernels after it up to the next batch — holds the one-token step's
//!   shadow table for that layer (the MoE piece's own work and, before an
//!   engram site, the site's token-only work), row 0's first go followed by
//!   row 1's layer 0 up to its go: each row carries its sites' work once, in
//!   its own shadow.
//! - `--api`: the engine's own entry from a reset, graph mode and eager
//!   mode: `step_pair(t, t1)` after the set's prompt against `step(t)`,
//!   `step(t1)`, and `step_pair` + `rollback` + `step(t2)` against `step(t)`,
//!   `step(t2)`: logits and tokens bit for bit, the two captured graphs
//!   (one-token and pair) replayed in one process. Then deep cuts: from a
//!   reset, `m = 4·W + 7` positions (`d1`'s ids, then greedy ids, at `d1`'s
//!   `top_k` so the indexer selects; a run whose logits repeat or stop being
//!   finite fails, naming the first layer where it died), the
//!   last one eagerly with every layer's `l_out` read; then cuts to `m − 2`,
//!   `m − W`, `W/2` and 0 (each rounded down by `Body::keep_point`), each
//!   followed by the taken-back tokens again: every re-fed position's argmax
//!   and logits, the last position's `l_out` of every layer, and every
//!   layer's caches and compressor state afterwards equal the uninterrupted
//!   run's, bit for bit.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_skew: built without the `deepseek41` feature; see `just gate-gpu-ds41-skew`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_skew", gate::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_finite.rs"]
mod finite;

#[cfg(feature = "deepseek41")]
mod gate {
    use std::collections::BTreeMap;

    use bloomery_gpu::head::Head;
    use bloomery_gpu::model::{ChainBody, StepMode};
    use bloomery_gpu::weights::Weights;
    use bloomery_gpu::{Gpu, GpuError};
    use bloomery_gpu_deepseek41::body::{self, Body, Deepseek41Model, PAIR_ROWS, Seam};
    use bloomery_gpu_gates::oracle::deepseek41::{D1, STEP4};
    use bloomery_gpu_gates::oracle::for_arch;
    use bloomery_gpu_gates::{
        GateError, RefManifest, checks_failed, ref_tensor_of_in, verdict, widened_f16_rows_in,
    };
    use cuda_core::sys;
    use gguf::Split;
    use model::arch::Arch;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::names;
    use model::arch::deepseek41::plan::{Planner, StepPlan};
    use model::placement::workstation;

    use crate::finite;

    /// The sets the pair pass runs on: one whose lists are the identity and
    /// one whose indexer layers select.
    const SETS: [&str; 2] = [STEP4, D1];
    /// The build every V4.1 oracle set must name: the sink-fixed ik tree.
    const ORACLE_BUILD: &str = "db517b69";
    /// The serving context: the gate placement's, and the body's caches'.
    const CTX_MAX: u64 = workstation::CTX_MAX;
    /// f16 NaN: every ring slot and compressed row the gate does not inject.
    const NAN16: u16 = 0x7e00;
    /// The port's names of the compressed streams, in the planner's order.
    const STREAMS: [&str; 2] = ["csa", "hca"];
    /// Operations in a go batch and in a wait batch (`bloomery_gpu::hybrid`).
    const GO_OPS: u32 = 5;
    const WAIT_OPS: u32 = 2;
    /// The MoE piece's own shadow work with card experts and without, and
    /// one engram site's token-only work (the step gate's tables).
    const SHADOW_CARD: [&str; 6] = [
        "ds41_hc_pre",
        "ds41_expert_gate_up",
        "q3k_quantize_q8_1",
        "q4k_gemv_sel",
        "ds41_shexp_gate_up",
        "q8_0_gemv",
    ];
    const SHADOW_HOST: [&str; 3] = ["ds41_hc_pre", "ds41_shexp_gate_up", "q8_0_gemv"];
    const ENGRAM_KV: [&str; 3] = ["ds41_glue_engram_rows", "q8_0_gemv", "ds41_engram_key_norm"];
    /// A row's own launches before its first layer: the attention piece's
    /// gather of the step words and the embedding broadcast.
    const ROW_START: usize = 2;

    struct Args {
        sets: bool,
        structure: bool,
        api: bool,
    }

    fn parse_args() -> Result<Args, GateError> {
        const USAGE: &str = "usage: gate_deepseek41_skew [--sets] [--structure] [--api]";
        let mut a = Args {
            sets: false,
            structure: false,
            api: false,
        };
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "--sets" => a.sets = true,
                "--structure" => a.structure = true,
                "--api" => a.api = true,
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        if !(a.sets || a.structure || a.api) {
            return Err(USAGE.into());
        }
        Ok(a)
    }

    pub fn run() -> Result<(), GateError> {
        let args = parse_args()?;
        let path = workstation::model_v41();
        let split = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let hp = Hparams::read(&split)?;
        let file = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let mut m = body::open(file, workstation::plan_gate, CTX_MAX as usize)?;
        let mut heads = {
            let (gpu, w, body) = m.body_parts("gate_deepseek41_skew")?;
            println!(
                "load: {} layers; a row adds: {}",
                body.layers().len(),
                body.row_bytes()
                    .iter()
                    .map(|(what, n)| format!("{what} {n} B"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            [
                Head::new(gpu, w, hp.rms_eps)?,
                Head::new(gpu, w, hp.rms_eps)?,
            ]
        };
        let mut pass = true;
        if args.structure {
            pass &= structure(&mut m, &mut heads, &hp)?;
        }
        if args.sets {
            for name in SETS {
                pass &= set_arms(&mut m, &mut heads, &split, &hp, name)?;
            }
        }
        if args.api {
            pass &= api(&mut m, &split, &hp)?;
            pass &= cuts(&mut m, &mut heads[0], &split, &hp)?;
        }
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: gate_deepseek41_skew");
        Ok(())
    }

    // ------------------------------------------------------- the set's state

    /// One decode-step set: its manifest, the step, the host plan of it and
    /// the indexer's `top_k` ik ran with.
    struct Set {
        name: &'static str,
        man: RefManifest,
        pos: u32,
        token: u32,
        before: Vec<u32>,
        plan: StepPlan,
        top_k: usize,
    }

    /// The top-k override in a set's `# flags`, if there is one (the step
    /// gate's `top_k_override`).
    fn top_k_override(flags: &str) -> Result<Option<usize>, GateError> {
        let key = format!("{}.attention.indexer.top_k=int:", Arch::Deepseek41.name());
        let mut it = flags.split_whitespace();
        let mut found = None;
        while let Some(tok) = it.next() {
            if tok == "--override-kv" {
                let kv = it.next().ok_or("# flags: --override-kv without a value")?;
                if let Some(v) = kv.strip_prefix(key.as_str()) {
                    found = Some(v.parse::<usize>()?);
                }
            }
        }
        Ok(found)
    }

    fn open_set(split: &Split, hp: &Hparams, name: &'static str) -> Result<Set, GateError> {
        let man = for_arch(Arch::Deepseek41)?.open_named(name)?;
        if !man
            .build
            .as_deref()
            .is_some_and(|b| b.starts_with(ORACLE_BUILD))
        {
            return Err(format!(
                "{name}: build {:?}, not the sink-fixed tree {ORACLE_BUILD}: a stale set",
                man.build
            )
            .into());
        }
        let (pos, tokens, before) = man.step()?;
        let (tokens, before) = (tokens.to_vec(), before.to_vec());
        if tokens.len() != 1 {
            return Err(format!("{name}: a step of {} tokens", tokens.len()).into());
        }
        let ctx = man
            .header
            .ctx
            .unwrap_or((before.len() + tokens.len()) as u64);
        let planner = Planner::from_file(split, hp, ctx)?;
        let mut plan = StepPlan::default();
        planner.plan_into(&tokens, pos, &before, &mut plan)?;
        let top_k = match man.header.flags.as_deref() {
            Some(flags) => top_k_override(flags)?,
            None => None,
        };
        println!(
            "set {name}: {} (build {}) position {pos} token {}{}",
            man.dir.display(),
            man.build.as_deref().unwrap_or("-"),
            tokens[0],
            top_k.map_or(String::new(), |k| format!(" top_k {k} (overridden)"))
        );
        Ok(Set {
            name,
            man,
            pos,
            token: tokens[0],
            before,
            plan,
            top_k: top_k.unwrap_or(hp.indexer.top_k),
        })
    }

    /// Tensor row `name`/0, made by `op`, with its position.
    fn node<'m>(
        man: &'m RefManifest,
        name: &str,
        op: &str,
    ) -> Result<(usize, &'m bloomery_gpu_gates::RefRow), GateError> {
        let (at, r) = man.tensor_at(name, 0)?;
        if r.op != op {
            return Err(format!("{}: {name} is {}, want {op}", man.dir.display(), r.op).into());
        }
        Ok((at, r))
    }

    /// One layer's state before the step, in the body's layout.
    struct LayerState {
        ring: Vec<u16>,
        rows: Option<Vec<u16>>,
        keys: Option<Vec<u16>>,
        comp: Option<(Vec<f32>, Vec<f32>)>,
    }

    /// A cache of compressed rows or index keys as the set holds it (the step
    /// gate's `cache_rows`).
    fn cache_rows(
        man: &RefManifest,
        write: &str,
        view: &str,
    ) -> Result<Option<Vec<u16>>, GateError> {
        if let Ok((_, w)) = node(man, write, "SET_ROWS") {
            return Ok(Some(widened_f16_rows_in(&man.dir, w)?));
        }
        match node(man, view, "VIEW") {
            Ok((_, v)) => {
                let leaf = v.src0.as_deref().ok_or("a cache view has no src0")?;
                Ok(Some(widened_f16_rows_in(&man.dir, man.input(leaf, 0)?)?))
            }
            Err(_) => Ok(None),
        }
    }

    /// `n_vis` rows of `width` from `from` into a cache of `cap` rows, every
    /// other row — and the step's written row `written` — f16 NaN.
    fn seat_rows(
        from: &[u16],
        width: usize,
        cap: usize,
        n_vis: usize,
        written: Option<usize>,
    ) -> Vec<u16> {
        let mut rows = vec![NAN16; cap * width];
        for r in (0..n_vis).filter(|&r| Some(r) != written) {
            rows[r * width..(r + 1) * width].copy_from_slice(&from[r * width..(r + 1) * width]);
        }
        rows
    }

    /// Every layer's state the set's prefill left (the step gate's
    /// `set_state`).
    fn set_state(set: &Set, hp: &Hparams) -> Result<Vec<LayerState>, GateError> {
        let man = &set.man;
        let width = hp.head_dim;
        let pos = set.pos as usize;
        let len = (set.plan.pos[0] - set.plan.raw_first[0] + 1) as usize;
        let ring_rows = hp.window.min(CTX_MAX as usize);
        let first = pos + 1 - len;
        let mut out = Vec::with_capacity(hp.n_layer);
        for (l, kind) in hp.layers.iter().enumerate() {
            let before = widened_f16_rows_in(&man.dir, man.input(&format!("cache_k_l{l}"), 0)?)?;
            let mut ring = vec![NAN16; ring_rows * width];
            for c in first..pos {
                let slot = c % ring_rows;
                ring[slot * width..(slot + 1) * width]
                    .copy_from_slice(&before[c * width..(c + 1) * width]);
            }
            let (rows, keys, comp) = match (kind.stream, kind.compressor) {
                (Some(st), Some(c)) if st.kv_source == l => {
                    let s = stream_index(&set.plan, st.ratio)?;
                    let tag = STREAMS[s];
                    let pst = &set.plan.streams[s];
                    let pre = match node(man, &format!("{tag}_k-{l}"), "VIEW") {
                        Ok((_, view)) => {
                            let leaf = view.src0.as_deref().ok_or("the rows view has no src0")?;
                            widened_f16_rows_in(&man.dir, man.input(leaf, 0)?)?
                        }
                        Err(_) => cache_rows(man, &format!("{tag}_k_write-{l}"), "-")?
                            .ok_or_else(|| format!("layer {l}: no cache of compressed rows"))?,
                    };
                    let n_vis = pst.n_visible[0] as usize;
                    let written = pst.state_write.first().map(|&w| w as usize);
                    let cap = (CTX_MAX as usize).div_ceil(st.ratio as usize);
                    let rows = seat_rows(&pre, width, cap, n_vis, written);
                    let keys = if kind.index_keys {
                        cache_rows(man, &format!("lid_k_write-{l}"), &format!("lid_k-{l}"))?
                            .map(|k| seat_rows(&k, hp.indexer.head_dim, cap, n_vis, written))
                    } else {
                        None
                    };
                    let comp = if c.gated && st.ratio > 1 {
                        Some(comp_ring(man, l, tag, st.ratio as usize, width)?)
                    } else {
                        None
                    };
                    (Some(rows), keys, comp)
                }
                _ => (None, None, None),
            };
            out.push(LayerState {
                ring,
                rows,
                keys,
                comp,
            });
        }
        Ok(out)
    }

    /// The plan's stream of ratio `ratio`.
    fn stream_index(plan: &StepPlan, ratio: u32) -> Result<usize, GateError> {
        plan.streams
            .iter()
            .position(|s| s.ratio == ratio)
            .ok_or_else(|| format!("the plan has no stream of ratio {ratio}").into())
    }

    /// A ratio-`r` compressor's ring before the step (the step gate's
    /// `comp_ring`).
    fn comp_ring(
        man: &RefManifest,
        l: usize,
        tag: &str,
        r: usize,
        width: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), GateError> {
        let ring_ne = [width as u64, r as u64, 1, 1];
        let by_src0 = |src0: &str| {
            man.tensors
                .iter()
                .find(|t| t.op == "MUL_MAT" && t.src0.as_deref() == Some(src0))
        };
        let kv = by_src0(&names::attn_compressor_kv(l))
            .ok_or_else(|| format!("layer {l}: no compressor kv projection"))?;
        let sc = by_src0(&names::attn_compressor_gate(l))
            .ok_or_else(|| format!("layer {l}: no compressor gate projection"))?;
        let (at_pk, _) = node(man, &format!("{tag}_k_state_persist-{l}"), "SET_ROWS")?;
        let (at_ps, _) = node(man, &format!("{tag}_score_state_persist-{l}"), "SET_ROWS")?;
        let first = |concat: &str, src1: &str, persist: usize| {
            let by_concat = man.tensors.iter().position(|t| {
                t.op == "CONCAT" && t.name == concat && t.src1.as_deref() == Some(src1)
            });
            by_concat
                .and_then(|c| {
                    man.first_touched_by(c)
                        .into_iter()
                        .find(|i| i.ty == "f32" && i.ne == ring_ne)
                })
                .or_else(|| {
                    man.first_touched_by(persist)
                        .into_iter()
                        .find(|i| i.ty == "f32" && i.ne == ring_ne)
                })
        };
        let v = first(&format!("{tag}_source_kv"), &kv.name, at_pk)
            .ok_or_else(|| format!("layer {l}: no ring of values"))?;
        let s = first(&format!("{tag}_source_score"), &sc.name, at_ps)
            .ok_or_else(|| format!("layer {l}: no ring of scores"))?;
        Ok((
            ref_tensor_of_in(&man.dir, v)?,
            ref_tensor_of_in(&man.dir, s)?,
        ))
    }

    /// Reset the body, write `state` into its buffers and seat the set's
    /// history; the next step is the set's.
    fn inject(
        gpu: &Gpu,
        body: &mut Body,
        set: &Set,
        state: &[LayerState],
    ) -> Result<(), GateError> {
        let stream = gpu.stream();
        body.reset(gpu)?;
        for (l, s) in state.iter().enumerate() {
            let b = body
                .state_mut(l)
                .ok_or_else(|| format!("the body does not run layer {l}"))?;
            b.ring.buf_mut().copy_from_host(stream, &s.ring)?;
            match (&s.rows, b.rows) {
                (Some(h), Some(d)) => d.buf_mut().copy_from_host(stream, h)?,
                (None, _) => {}
                (Some(_), None) => return Err(format!("layer {l}: the body holds no rows").into()),
            }
            match (&s.keys, b.keys) {
                (Some(h), Some(d)) => d.buf_mut().copy_from_host(stream, h)?,
                (None, _) => {}
                (Some(_), None) => {
                    return Err(format!("layer {l}: the body holds no index keys").into());
                }
            }
            match (&s.comp, b.values, b.scores) {
                (Some((v, sc)), Some(dv), Some(ds)) => {
                    dv.buf_mut().copy_from_host(stream, v)?;
                    ds.buf_mut().copy_from_host(stream, sc)?;
                }
                (None, _, _) => {}
                _ => return Err(format!("layer {l}: the body holds no compressor ring").into()),
            }
        }
        body.set_history(&set.before)?;
        stream.synchronize()?;
        Ok(())
    }

    // ------------------------------------------------------------ the arms

    /// What a row's step leaves that the pair pass must reproduce: the
    /// logits, and the row's streams, folds and lists (bits).
    #[derive(PartialEq)]
    struct RowOut {
        logits: Vec<u32>,
        streams: Vec<u32>,
        folds: Vec<u32>,
        lists: Vec<u32>,
    }

    /// What the whole run leaves: every layer's caches and compressor state
    /// (bits), the history.
    #[derive(PartialEq)]
    struct Caches {
        state: Vec<Vec<u16>>,
        comp: Vec<Vec<u32>>,
        history: Vec<u32>,
    }

    fn bits(v: Vec<f32>) -> Vec<u32> {
        v.into_iter().map(f32::to_bits).collect()
    }

    fn row_out(gpu: &Gpu, body: &Body, row: usize, head: &Head) -> Result<RowOut, GateError> {
        let stream = gpu.stream();
        let b = body
            .row_buffers(row)
            .ok_or_else(|| format!("the body holds no row {row}"))?;
        let mut streams = Vec::new();
        for s in b.streams {
            streams.extend(bits(s.to_host_vec(stream)?));
        }
        let mut folds = Vec::new();
        for f in b.folds {
            folds.extend(bits(f.to_host_vec(stream)?));
        }
        let mut lists = Vec::new();
        for l in b.lists {
            lists.extend(l.to_host_vec(stream)?);
        }
        Ok(RowOut {
            logits: bits(head.logits_to_host(gpu)?),
            streams,
            folds,
            lists,
        })
    }

    fn caches(gpu: &Gpu, body: &mut Body, layers: usize) -> Result<Caches, GateError> {
        let stream = gpu.stream();
        let mut state = Vec::new();
        let mut comp = Vec::new();
        for l in 0..layers {
            let b = body
                .state_mut(l)
                .ok_or_else(|| format!("the body does not run layer {l}"))?;
            state.push(b.ring.buf().to_host_vec(stream)?);
            for t in [b.rows, b.keys].into_iter().flatten() {
                state.push(t.buf().to_host_vec(stream)?);
            }
            for t in [b.values, b.scores].into_iter().flatten() {
                comp.push(bits(t.buf().to_host_vec(stream)?));
            }
        }
        Ok(Caches {
            state,
            comp,
            history: body.history().to_vec(),
        })
    }

    /// One one-token step of `token` at `pos` on row 0, eagerly, into
    /// `head`.
    fn one_step(
        gpu: &Gpu,
        w: &Weights,
        body: &mut Body,
        head: &mut Head,
        token: u32,
        pos: u32,
    ) -> Result<(), GateError> {
        let input = body.decode_input(token, pos)?;
        body.refresh(gpu.stream(), &input)?;
        body.enqueue_chain(gpu, w, head)?;
        gpu.stream().synchronize()?;
        Ok(())
    }

    /// Differing words between two runs' outputs, by what differs.
    fn diff_rows(what: &str, a: &RowOut, b: &RowOut) -> Vec<String> {
        let count = |x: &[u32], y: &[u32]| {
            x.iter().zip(y).filter(|(p, q)| p != q).count() + x.len().abs_diff(y.len())
        };
        [
            ("logits", count(&a.logits, &b.logits)),
            ("streams", count(&a.streams, &b.streams)),
            ("folds", count(&a.folds, &b.folds)),
            ("lists", count(&a.lists, &b.lists)),
        ]
        .into_iter()
        .filter(|&(_, n)| n > 0)
        .map(|(k, n)| format!("{what} {k} {n} words"))
        .collect()
    }

    fn diff_caches(a: &Caches, b: &Caches) -> Vec<String> {
        let mut out = Vec::new();
        let state = a
            .state
            .iter()
            .zip(&b.state)
            .map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p != q).count())
            .sum::<usize>();
        let comp = a
            .comp
            .iter()
            .zip(&b.comp)
            .map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p != q).count())
            .sum::<usize>();
        if state > 0 || a.state.len() != b.state.len() {
            out.push(format!("caches {state} words"));
        }
        if comp > 0 || a.comp.len() != b.comp.len() {
            out.push(format!("compressor state {comp} words"));
        }
        if a.history != b.history {
            out.push("history".to_string());
        }
        out
    }

    /// The largest logit's id, and the runner-up's.
    fn top2(logits: &[u32]) -> (u32, u32) {
        let v: Vec<f32> = logits.iter().map(|&b| f32::from_bits(b)).collect();
        let mut best = (0usize, f32::NEG_INFINITY);
        let mut second = (0usize, f32::NEG_INFINITY);
        for (i, &x) in v.iter().enumerate() {
            if x > best.1 {
                second = best;
                best = (i, x);
            } else if x > second.1 {
                second = (i, x);
            }
        }
        (best.0 as u32, second.0 as u32)
    }

    fn set_arms(
        m: &mut Deepseek41Model,
        heads: &mut [Head; PAIR_ROWS],
        split: &Split,
        hp: &Hparams,
        name: &'static str,
    ) -> Result<bool, GateError> {
        let set = open_set(split, hp, name)?;
        let name = set.name;
        let state = set_state(&set, hp)?;
        let (gpu, w, body) = m.body_parts("set_arms")?;
        body.set_indexer_top_k(gpu, set.top_k)?;
        let (p, t) = (set.pos, set.token);
        let [ha, hb] = heads;

        // (i) two steps in turn.
        inject(gpu, body, &set, &state)?;
        one_step(gpu, w, body, ha, t, p)?;
        let seq_a = row_out(gpu, body, 0, ha)?;
        let (t1, t2) = top2(&seq_a.logits);
        one_step(gpu, w, body, ha, t1, p + 1)?;
        let seq_b = row_out(gpu, body, 0, ha)?;
        let seq = caches(gpu, body, hp.n_layer)?;
        println!("set {name}: t {t} at {p}, t1 {t1} (greedy after t), t2 {t2} (runner-up after t)");

        // (ii) the pair pass, eagerly.
        inject(gpu, body, &set, &state)?;
        body.decode_pair(gpu.stream(), [t, t1], p)?;
        body.enqueue_pair(gpu, w, [&mut *ha, &mut *hb])?;
        gpu.stream().synchronize()?;
        let eager = (
            row_out(gpu, body, 0, ha)?,
            row_out(gpu, body, 1, hb)?,
            caches(gpu, body, hp.n_layer)?,
        );
        let mut d = diff_rows("row 0", &seq_a, &eager.0);
        d.extend(diff_rows("row 1", &seq_b, &eager.1));
        d.extend(diff_caches(&seq, &eager.2));
        let ok_eager = d.is_empty();
        println!(
            "set {name} pair eager == two steps (both rows' logits, streams, folds, lists; {} \
             cache and {} compressor buffers; history): {}{}",
            seq.state.len(),
            seq.comp.len(),
            verdict(ok_eager),
            if ok_eager {
                String::new()
            } else {
                format!(" — {}", d.join(", "))
            }
        );

        // (iii) a capture of the pair pass, replayed.
        let graph = gpu.capture(|_| body.enqueue_pair(gpu, w, [&mut *ha, &mut *hb]))?;
        inject(gpu, body, &set, &state)?;
        body.decode_pair(gpu.stream(), [t, t1], p)?;
        graph.launch(gpu.stream())?;
        body.serve_replay_pair()?;
        gpu.stream().synchronize()?;
        let replay = (
            row_out(gpu, body, 0, ha)?,
            row_out(gpu, body, 1, hb)?,
            caches(gpu, body, hp.n_layer)?,
        );
        drop(graph);
        let mut d = diff_rows("row 0", &eager.0, &replay.0);
        d.extend(diff_rows("row 1", &eager.1, &replay.1));
        d.extend(diff_caches(&eager.2, &replay.2));
        let ok_replay = d.is_empty();
        println!(
            "set {name} pair replay == pair eager: {}{}",
            verdict(ok_replay),
            if ok_replay {
                String::new()
            } else {
                format!(" — {}", d.join(", "))
            }
        );

        // (iv) the pair pass, t1 taken back, t2 at p + 1 == t, t2 in turn.
        inject(gpu, body, &set, &state)?;
        one_step(gpu, w, body, ha, t, p)?;
        one_step(gpu, w, body, ha, t2, p + 1)?;
        let ref_row = row_out(gpu, body, 0, ha)?;
        let ref_caches = caches(gpu, body, hp.n_layer)?;
        inject(gpu, body, &set, &state)?;
        body.decode_pair(gpu.stream(), [t, t1], p)?;
        body.enqueue_pair(gpu, w, [&mut *ha, &mut *hb])?;
        gpu.stream().synchronize()?;
        body.rollback(p + 1)?;
        one_step(gpu, w, body, ha, t2, p + 1)?;
        let rb_row = row_out(gpu, body, 0, ha)?;
        let rb_caches = caches(gpu, body, hp.n_layer)?;
        let mut d = diff_rows("row 0", &ref_row, &rb_row);
        d.extend(diff_caches(&ref_caches, &rb_caches));
        let ok_rollback = d.is_empty();
        println!(
            "set {name} pair + rollback({}) + step(t2) == step(t) + step(t2): {}{}",
            p + 1,
            verdict(ok_rollback),
            if ok_rollback {
                String::new()
            } else {
                format!(" — {}", d.join(", "))
            }
        );
        // (v) Both pair positions taken back. PIN(2026-09-24): a cut past the
        // last position was refused outright until the ring gained its shadow;
        // now it is granted where the ring slots the step at p reads hold
        // their rows (step4: position 4, the ring never wrapped) and refused
        // where it would restore one from a shadow row of the injected
        // history (d1: rows 173 and 174, which positions 301 and 302 overwrote).
        let grant = name == STEP4;
        inject(gpu, body, &set, &state)?;
        body.decode_pair(gpu.stream(), [t, t1], p)?;
        body.enqueue_pair(gpu, w, [&mut *ha, &mut *hb])?;
        gpu.stream().synchronize()?;
        let kept = body.keep_point(p as usize);
        let granted = body.rollback(p).is_ok();
        let ok_cut2 = if granted {
            one_step(gpu, w, body, ha, t, p)?;
            one_step(gpu, w, body, ha, t2, p + 1)?;
            let row = row_out(gpu, body, 0, ha)?;
            let caches = caches(gpu, body, hp.n_layer)?;
            let mut d = diff_rows("row 0", &ref_row, &row);
            d.extend(diff_caches(&ref_caches, &caches));
            if !d.is_empty() {
                println!("FAIL: set {name} pair + rollback({p}): {}", d.join(", "));
            }
            d.is_empty()
        } else {
            true
        };
        let ok_grant = granted == grant && (kept == p as usize) == grant;
        println!(
            "set {name} rollback of both pair positions ({p} after {}): keep_point {kept}, {} \
             (pinned {}); then step(t) + step(t2) == step(t) + step(t2): {}",
            p + 2,
            if granted { "granted" } else { "refused" },
            if grant { "granted" } else { "refused" },
            verdict(ok_grant && ok_cut2)
        );

        body.set_indexer_top_k(gpu, hp.indexer.top_k)?;
        Ok(ok_eager && ok_replay && ok_rollback && ok_grant && ok_cut2)
    }

    // ------------------------------------------------------------ structure

    fn drv(rc: sys::CUresult, what: &str) -> Result<(), GateError> {
        if rc == sys::cudaError_enum_CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{what}: CUresult {rc}").into())
        }
    }

    /// One node of a captured chain, as the driver reports it.
    #[derive(Debug)]
    enum StepNode {
        Kernel(String),
        /// A stream memory-operation batch and its operation count.
        Memop(u32),
        Other(String),
    }

    /// The nodes `enqueue` records on the engine stream, in stream order:
    /// the template walked from its one root along its edges, then
    /// destroyed without being instantiated (the step gate's `step_order`).
    fn chain_order(
        gpu: &Gpu,
        enqueue: impl FnOnce() -> Result<(), GpuError>,
    ) -> Result<Vec<StepNode>, GateError> {
        let hs = gpu.stream().cu_stream();
        // SAFETY: `hs` is the engine's live stream, which is not capturing
        // (every capture before this one ended).
        let rc = unsafe {
            sys::cuStreamBeginCapture_v2(
                hs,
                sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        };
        drv(rc, "cuStreamBeginCapture_v2")?;
        let enqueued = enqueue();
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        // SAFETY: the stream is capturing (begun above); this ends it on
        // every path and writes the template, or null, into `graph`.
        let ended = unsafe { sys::cuStreamEndCapture(hs, &mut graph) };
        let order = match enqueued {
            Err(e) => Err(e.into()),
            Ok(()) => drv(ended, "cuStreamEndCapture").and_then(|()| walk(graph)),
        };
        if !graph.is_null() {
            // SAFETY: a non-null handle from the end of the capture is a
            // template, destroyed exactly once here.
            unsafe { sys::cuGraphDestroy(graph) };
        }
        order
    }

    /// `graph`'s nodes from its one root along its one-successor edges; a
    /// template that is not a chain is refused.
    fn walk(graph: sys::CUgraph) -> Result<Vec<StepNode>, GateError> {
        let mut total = 0usize;
        // SAFETY: a null array asks only for the count.
        let rc = unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut total) };
        drv(rc, "cuGraphGetNodes")?;
        let mut roots = 0usize;
        // SAFETY: a null array asks only for the count.
        let rc = unsafe { sys::cuGraphGetRootNodes(graph, std::ptr::null_mut(), &mut roots) };
        drv(rc, "cuGraphGetRootNodes")?;
        if roots != 1 {
            return Err(format!("the graph has {roots} roots; a one-stream capture has 1").into());
        }
        let mut node: sys::CUgraphNode = std::ptr::null_mut();
        // SAFETY: one slot, the count the driver just reported.
        let rc = unsafe { sys::cuGraphGetRootNodes(graph, &mut node, &mut roots) };
        drv(rc, "cuGraphGetRootNodes")?;
        let mut out = Vec::with_capacity(total);
        loop {
            out.push(describe(node)?);
            let mut k = 0usize;
            // SAFETY: `node` is a node of the live template; null arrays ask
            // only for the count.
            let rc = unsafe {
                sys::cuGraphNodeGetDependentNodes_v2(
                    node,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut k,
                )
            };
            drv(rc, "cuGraphNodeGetDependentNodes_v2")?;
            match k {
                0 => break,
                1 => {
                    let mut next: sys::CUgraphNode = std::ptr::null_mut();
                    // SAFETY: one slot, the count just reported; the edge data
                    // is not asked for.
                    let rc = unsafe {
                        sys::cuGraphNodeGetDependentNodes_v2(
                            node,
                            &mut next,
                            std::ptr::null_mut(),
                            &mut k,
                        )
                    };
                    drv(rc, "cuGraphNodeGetDependentNodes_v2")?;
                    node = next;
                }
                _ => {
                    return Err(format!(
                        "node {} of the graph has {k} successors; a one-stream capture has one",
                        out.len() - 1
                    )
                    .into());
                }
            }
        }
        if out.len() != total {
            return Err(format!(
                "the walk from the root reached {} of the graph's {total} nodes",
                out.len()
            )
            .into());
        }
        Ok(out)
    }

    /// A node's kind, with its kernel's entry name or its batch's operation
    /// count (the step gate's `describe`).
    fn describe(node: sys::CUgraphNode) -> Result<StepNode, GateError> {
        let mut kind: sys::CUgraphNodeType = 0;
        // SAFETY: `node` is a node of a live template.
        let rc = unsafe { sys::cuGraphNodeGetType(node, &mut kind) };
        drv(rc, "cuGraphNodeGetType")?;
        if kind == sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL {
            // SAFETY: all-zero is a valid value of this plain C struct
            // (integers and nullable pointers), which the call then fills.
            let mut p: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
            // SAFETY: `node` is a kernel node and `p` the struct the call fills.
            let rc = unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut p) };
            drv(rc, "cuGraphKernelNodeGetParams_v2")?;
            let mut name: *const std::ffi::c_char = std::ptr::null();
            if p.func.is_null() {
                // SAFETY: the node launches the library kernel `p.kern`, a
                // live handle; `name` is a local the call writes.
                let rc = unsafe { sys::cuKernelGetName(&mut name, p.kern) };
                drv(rc, "cuKernelGetName")?;
            } else {
                // SAFETY: the node launches the module function `p.func`, a
                // live handle; `name` is a local the call writes.
                let rc = unsafe { sys::cuFuncGetName(&mut name, p.func) };
                drv(rc, "cuFuncGetName")?;
            }
            if name.is_null() {
                return Err("a kernel node whose function has no name".into());
            }
            // SAFETY: the driver hands back a NUL-terminated name it owns for
            // the function's lifetime; it is copied out at once.
            let name = unsafe { std::ffi::CStr::from_ptr(name) };
            Ok(StepNode::Kernel(name.to_string_lossy().into_owned()))
        } else if kind == sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP {
            // SAFETY: all-zero is a valid value of this plain C struct (a
            // context, a count, a nullable array pointer, flags), which the
            // call then fills.
            let mut p: sys::CUDA_BATCH_MEM_OP_NODE_PARAMS = unsafe { std::mem::zeroed() };
            // SAFETY: `node` is a batch memory-operation node and `p` the
            // struct the call fills.
            let rc = unsafe { sys::cuGraphBatchMemOpNodeGetParams(node, &mut p) };
            drv(rc, "cuGraphBatchMemOpNodeGetParams")?;
            Ok(StepNode::Memop(p.count))
        } else {
            Ok(StepNode::Other(bloomery_gpu_gates::nodes::kind_name(kind)))
        }
    }

    /// Per kernel name its count, the memop batches and the other nodes'
    /// kinds.
    fn census(nodes: &[StepNode]) -> (BTreeMap<&str, usize>, usize, Vec<&str>) {
        let mut kernels = BTreeMap::new();
        let (mut memops, mut other) = (0, Vec::new());
        for n in nodes {
            match n {
                StepNode::Kernel(k) => *kernels.entry(k.as_str()).or_insert(0) += 1,
                StepNode::Memop(_) => memops += 1,
                StepNode::Other(k) => other.push(k.as_str()),
            }
        }
        (kernels, memops, other)
    }

    fn structure(
        m: &mut Deepseek41Model,
        heads: &mut [Head; PAIR_ROWS],
        hp: &Hparams,
    ) -> Result<bool, GateError> {
        if !bloomery_gpu::hybrid::levers()?.overlap {
            return Err("the shadow check needs BLOOMERY_HYBRID_OVERLAP unset or 1".into());
        }
        let (gpu, w, body) = m.body_parts("structure")?;
        let layers = body.layers();
        let [ha, hb] = heads;
        let one = chain_order(gpu, || body.enqueue_chain(gpu, w, &mut *ha))?;
        let pair = chain_order(gpu, || body.enqueue_pair(gpu, w, [&mut *ha, &mut *hb]))?;
        let (k1, m1, o1) = census(&one);
        let (k2, m2, o2) = census(&pair);
        let (n1, n2) = (k1.values().sum::<usize>(), k2.values().sum::<usize>());
        println!(
            "predict pair: every launch of the one-token step once per row — nodes {} = kernels \
             {} + memops {}, other 0 (twice the step's {} = {} + {})",
            2 * one.len(),
            2 * n1,
            2 * m1,
            one.len(),
            n1,
            m1
        );
        let names_twice =
            k1.len() == k2.len() && k1.iter().all(|(k, &c)| k2.get(k) == Some(&(2 * c)));
        let ok_nodes = pair.len() == 2 * one.len()
            && n2 == 2 * n1
            && m2 == 2 * m1
            && o1.is_empty()
            && o2.is_empty()
            && names_twice;
        println!(
            "graph pair: nodes={} kernels={n2} memops={m2} other={} [{}]; {} kernel names, each \
             twice the step's: {names_twice}: {}",
            pair.len(),
            o2.len(),
            o2.join(","),
            k2.len(),
            verdict(ok_nodes)
        );
        if !names_twice {
            for (k, c) in &k1 {
                let got = k2.get(k).copied().unwrap_or(0);
                if got != 2 * c {
                    println!("  kernel {k}: step {c}, pair {got}: FAIL");
                }
            }
        }

        // The batches in the pass's order: go A0, go B0; per later layer
        // wait A, go A, wait B, go B; then wait A, wait B.
        let ops: Vec<u32> = pair
            .iter()
            .filter_map(|n| match n {
                StepNode::Memop(c) => Some(*c),
                _ => None,
            })
            .collect();
        let mut want_ops = vec![GO_OPS, GO_OPS];
        for _ in 1..layers.len() {
            want_ops.extend([WAIT_OPS, GO_OPS, WAIT_OPS, GO_OPS]);
        }
        want_ops.extend([WAIT_OPS, WAIT_OPS]);
        let ok_order = ops == want_ops;
        println!(
            "graph pair batches: {} in the pass's order (go {GO_OPS} ops, wait {WAIT_OPS}): {}",
            ops.len(),
            verdict(ok_order)
        );

        // The prologue: the one-token step's kernels before its first go are
        // the gather, the embedding broadcast, then layer 0 up to the go. The
        // pass gathers and broadcasts both rows first, then runs row 0's
        // layer 0 up to its go.
        let pre_go = |nodes: &[StepNode]| -> Vec<String> {
            nodes
                .iter()
                .take_while(|n| !matches!(n, StepNode::Memop(_)))
                .filter_map(|n| match n {
                    StepNode::Kernel(k) => Some(k.clone()),
                    _ => None,
                })
                .collect()
        };
        let (one_head, pair_head) = (pre_go(&one), pre_go(&pair));
        let own = ROW_START.min(one_head.len());
        let (starts, layer0) = one_head.split_at(own);
        let want_head: Vec<String> = [starts, starts, layer0].concat();
        let ok_head = pair_head == want_head;
        println!(
            "graph pair prologue: [{}] per row, then row 0's layer 0 up to its go ({} kernels): {}",
            starts.join(","),
            layer0.len(),
            verdict(ok_head)
        );

        // Each go's shadow: the kernels after it up to the next batch.
        let sites = &hp.engram.layer_ids;
        let mut shadows: Vec<Vec<&str>> = Vec::new();
        let mut open: Option<Vec<&str>> = None;
        for n in &pair {
            match n {
                StepNode::Memop(c) => {
                    if let Some(s) = open.take() {
                        shadows.push(s);
                    }
                    if *c == GO_OPS {
                        open = Some(Vec::new());
                    }
                }
                StepNode::Kernel(k) => {
                    if let Some(s) = open.as_mut() {
                        s.push(k.as_str());
                    }
                }
                StepNode::Other(_) => {}
            }
        }
        let mut ok_shadow = shadows.len() == PAIR_ROWS * layers.len();
        let mut rows_ok = [0usize; PAIR_ROWS];
        for (g, got) in shadows.iter().enumerate() {
            let (l, row) = (layers.start + g / PAIR_ROWS, g % PAIR_ROWS);
            let launches = body
                .ffn_launches(l)
                .ok_or_else(|| format!("the ffn piece does not run layer {l}"))?;
            let mut want: Vec<&str> = if launches > 7 {
                SHADOW_CARD.to_vec()
            } else {
                SHADOW_HOST.to_vec()
            };
            if sites.contains(&(l + 1)) {
                want.extend(ENGRAM_KV);
            }
            // Row 0's first go has no wait of row 1 behind it: row 1's layer
            // 0 up to its go runs there too, in the same host leg's shadow.
            if g == 0 {
                want.extend(layer0.iter().map(String::as_str));
            }
            if *got == want {
                rows_ok[row] += 1;
            } else {
                ok_shadow = false;
                println!(
                    "shadow row={row} layer={l} got=[{}] want=[{}]: FAIL",
                    got.join(","),
                    want.join(",")
                );
            }
        }
        let names: Vec<Option<&str>> = pair
            .iter()
            .map(|n| match n {
                StepNode::Kernel(k) => Some(k.as_str()),
                _ => None,
            })
            .collect();
        let engram_runs = (0..names.len().saturating_sub(ENGRAM_KV.len() - 1))
            .filter(|&i| {
                ENGRAM_KV
                    .iter()
                    .zip(&names[i..])
                    .all(|(want, got)| *got == Some(*want))
            })
            .count();
        let in_shadows = shadows
            .iter()
            .map(|s| {
                (0..s.len().saturating_sub(ENGRAM_KV.len() - 1))
                    .filter(|&i| ENGRAM_KV.iter().zip(&s[i..]).all(|(a, b)| a == b))
                    .count()
            })
            .sum::<usize>();
        let ok_engram = engram_runs == PAIR_ROWS * sites.len() && in_shadows == engram_runs;
        println!(
            "shadow pair: {} goes, row 0 {} and row 1 {} of {} layers match the step's shadow \
             table; engram token-only work {engram_runs} runs (want {} = {PAIR_ROWS} rows x {} \
             sites), {in_shadows} inside a shadow: {}",
            shadows.len(),
            rows_ok[0],
            rows_ok[1],
            layers.len(),
            PAIR_ROWS * sites.len(),
            sites.len(),
            verdict(ok_shadow && ok_engram)
        );
        Ok(ok_nodes && ok_order && ok_head && ok_shadow && ok_engram)
    }

    // ------------------------------------------------------------------ api

    /// The engine's own entry, graph and eager, from a reset: the set's
    /// prompt, then the pair against two steps, and the pair + rollback +
    /// step against two steps.
    fn api(m: &mut Deepseek41Model, split: &Split, hp: &Hparams) -> Result<bool, GateError> {
        let set = open_set(split, hp, STEP4)?;
        let prompt = &set.before;
        let t = set.token;
        let p = u32::try_from(prompt.len())?;
        let mut all = true;
        for mode in [StepMode::Graph, StepMode::Eager] {
            m.set_mode(mode);
            let lbits =
                |m: &Deepseek41Model| -> Result<Vec<u32>, GateError> { Ok(bits(m.logits()?)) };
            // Two steps in turn.
            m.reset()?;
            m.step(prompt)?;
            let a_tok = m.step(&[t])?;
            let a = lbits(m)?;
            let (t1, t2) = top2(&a);
            let b_tok = m.step(&[t1])?;
            let b = lbits(m)?;
            // The pair.
            m.reset()?;
            m.step(prompt)?;
            let toks = m.step_pair(t, t1)?;
            let [pa, pb] = m.pair_logits()?;
            let at = m.pos();
            let ok_pair = toks == [a_tok, b_tok] && bits(pa) == a && bits(pb) == b && at == p + 2;
            // Pair, rollback, t2 against t, t2.
            m.reset()?;
            m.step(prompt)?;
            m.step(&[t])?;
            let c_tok = m.step(&[t2])?;
            let c = lbits(m)?;
            m.reset()?;
            m.step(prompt)?;
            m.step_pair(t, t1)?;
            m.rollback(p + 1)?;
            let r_tok = m.step(&[t2])?;
            let r = lbits(m)?;
            let ok_rb = r_tok == c_tok && r == c && m.pos() == p + 2;
            println!(
                "api {mode:?}: step_pair({t}, {t1}) at {p} == step, step (tokens {toks:?} vs \
                 [{a_tok}, {b_tok}], both logits bit for bit, pos {at}): {}; step_pair + \
                 rollback({}) + step({t2}) == step, step (token {r_tok} vs {c_tok}, logits): {}",
                verdict(ok_pair),
                p + 1,
                verdict(ok_rb)
            );
            all &= ok_pair && ok_rb;
        }
        m.set_mode(StepMode::Graph);
        Ok(all)
    }
    // ----------------------------------------------------------------- cuts

    /// FNV-1a over 32-bit words: one number per tap a run keeps.
    fn digest(words: &[u32]) -> u64 {
        words.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &w| {
            (h ^ u64::from(w)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// The digest [`logits_digest`] gives logits that hold a NaN or an
    /// infinity.
    fn nonfinite_mark() -> u64 {
        u64::MAX
    }

    /// A logits row's digest, or [`nonfinite_mark`] when a value is not
    /// finite.
    fn logits_digest(logits: &[u32]) -> u64 {
        if logits.iter().all(|&b| f32::from_bits(b).is_finite()) {
            digest(logits)
        } else {
            nonfinite_mark()
        }
    }

    /// What the last position of a run leaves: its argmax, its logits and
    /// every layer's `l_out` (the streams after the MoE sub-layer), digested.
    #[derive(PartialEq, Debug)]
    struct LastStep {
        token: u32,
        logits: u64,
        l_out: Vec<u64>,
        /// The first seam whose streams hold a NaN or an infinity: its layer
        /// and sub-layer.
        first_nonfinite: Option<finite::Site>,
        /// Where the streams are not finite, the probe's reading of them.
        fault: Option<String>,
    }

    /// The step of `token` at `pos` on the body, eagerly, every layer's
    /// `l_out` read at its seam ([`finite::observed_step`]).
    fn observed_step(
        m: &mut Deepseek41Model,
        head: &mut Head,
        token: u32,
        pos: u32,
    ) -> Result<LastStep, GateError> {
        let mut l_out = Vec::new();
        let o = finite::observed_step(m, head, token, pos, &mut |_, seam, v| {
            if let Seam::Ffn { .. } = seam {
                l_out.push(digest(&v.iter().map(|x| x.to_bits()).collect::<Vec<_>>()));
            }
            Ok(())
        })?;
        Ok(LastStep {
            token: o.token(),
            logits: logits_digest(&o.logits),
            l_out,
            first_nonfinite: o.first_nonfinite(),
            fault: o.first_nonfinite().map(|_| o.describe()),
        })
    }

    /// Feed `tokens[from..]` through the engine in graph mode, all but the
    /// last one step at a time and the last as [`observed_step`]: each
    /// position's argmax and logits digest, and the last step.
    fn feed(
        m: &mut Deepseek41Model,
        head: &mut Head,
        tokens: &[u32],
        from: usize,
    ) -> Result<(Vec<(u32, u64)>, LastStep), GateError> {
        let (&last, body) = tokens.split_last().ok_or("no tokens to feed")?;
        let mut seen = Vec::with_capacity(tokens.len() - from);
        for &t in &body[from..] {
            let a = m.step(&[t])?;
            seen.push((a, logits_digest(&bits(m.logits()?))));
        }
        let pos = u32::try_from(tokens.len() - 1)?;
        Ok((seen, observed_step(m, head, last, pos)?))
    }

    /// The deep cuts ([`Body::keep_point`], [`Body::rollback`]) from a reset,
    /// through the engine's entry in graph mode.
    fn cuts(
        m: &mut Deepseek41Model,
        head: &mut Head,
        split: &Split,
        hp: &Hparams,
    ) -> Result<bool, GateError> {
        // The d1 set's ids of text, then greedy ids, at d1's top_k, so every
        // indexer layer selects once its stream holds more rows than that.
        let set = open_set(split, hp, D1)?;
        let w = hp.window;
        let total = 4 * w + 7;
        m.set_mode(StepMode::Graph);
        {
            let (gpu, _, body) = m.body_parts("cuts")?;
            body.set_indexer_top_k(gpu, set.top_k)?;
        }
        m.reset()?;
        // The uninterrupted run: the set's prompt and token, then greedy ids.
        let mut tokens = set.before.clone();
        tokens.push(set.token);
        let mut first = Vec::with_capacity(total);
        for &t in &tokens[..tokens.len() - 1] {
            let a = m.step(&[t])?;
            first.push((a, logits_digest(&bits(m.logits()?))));
        }
        while tokens.len() < total {
            let a = m.step(&[*tokens.last().ok_or("no tokens")?])?;
            first.push((a, logits_digest(&bits(m.logits()?))));
            tokens.push(a);
        }
        let pos = u32::try_from(total - 1)?;
        let last = observed_step(m, head, tokens[total - 1], pos)?;
        let whole = {
            let (gpu, _, body) = m.body_parts("cuts")?;
            caches(gpu, body, hp.n_layer)?
        };
        let distinct = {
            let mut d: Vec<u64> = first.iter().map(|&(_, h)| h).collect();
            d.sort_unstable();
            d.dedup();
            d.len()
        };
        // Two positions with the same logits bits are a dead run (a NaN
        // spreads to every later step alike), and a cut into it compares
        // nothing.
        let repeat = first
            .iter()
            .enumerate()
            .find_map(|(i, x)| first[..i].iter().position(|y| y.1 == x.1).map(|j| (j, i)));
        let dead = first.iter().position(|&(_, h)| h == nonfinite_mark());
        println!(
            "cuts: the run's first non-finite logits at {dead:?}, first repeated logits at {repeat:?}; \
             tokens {:?} .. {:?}",
            &tokens[..12],
            &tokens[total - 12..]
        );
        if let Some((j, i)) = repeat.or(dead.map(|p| (p, p))) {
            // Where the run dies: the two positions' streams layer by layer.
            m.reset()?;
            for &t in &tokens[..j] {
                m.step(&[t])?;
            }
            let a = observed_step(m, head, tokens[j], u32::try_from(j)?)?;
            m.rollback(u32::try_from(j)?)?;
            m.step(&[tokens[j]])?;
            for &t in &tokens[j + 1..i] {
                m.step(&[t])?;
            }
            let b = observed_step(m, head, tokens[i], u32::try_from(i)?)?;
            let same = a.l_out.iter().zip(&b.l_out).position(|(x, y)| x == y);
            println!(
                "FAIL: cuts: the uninterrupted run dies: positions {j} and {i} (tokens {} and {}) \
                 leave the same l_out from layer {same:?} on; first non-finite streams {:?} and \
                 {:?}",
                tokens[j], tokens[i], a.first_nonfinite, b.first_nonfinite
            );
            for (p, f) in [(j, &a.fault), (i, &b.fault)] {
                if let Some(f) = f {
                    println!("  position {p}: {f}");
                }
            }
            let (gpu, _, body) = m.body_parts("cuts")?;
            body.set_indexer_top_k(gpu, hp.indexer.top_k)?;
            return Ok(false);
        }
        println!(
            "cuts: {total} positions (window {w}, top_k {}): the set's {} prompt ids, then greedy; \
             {distinct} distinct logits over the first {} positions; the last step reads {} l_out \
             taps",
            set.top_k,
            set.before.len() + 1,
            first.len(),
            last.l_out.len()
        );
        let ratios: Vec<usize> = Planner::from_file(split, hp, CTX_MAX)?
            .stream_ratios()
            .iter()
            .map(|&r| r as usize)
            .collect();
        let mut all = !last.l_out.is_empty() && dead.is_none() && repeat.is_none();
        for n in [total - 2, total - w, w / 2, 0] {
            let (kept, stale) = {
                let (_, _, body) = m.body_parts("cuts")?;
                (body.keep_point(n), body.history().len())
            };
            let want = (0..=n)
                .rev()
                .find(|k| ratios.iter().all(|&r| r == 0 || k % r == 0))
                .unwrap_or(0);
            m.rollback(u32::try_from(kept)?)?;
            let (again, last_again) = feed(m, head, &tokens, kept)?;
            let whole_again = {
                let (gpu, _, body) = m.body_parts("cuts")?;
                caches(gpu, body, hp.n_layer)?
            };
            let steps = again.as_slice() == &first[kept..];
            let pairs = || again.iter().zip(&first[kept..]);
            let tokens_off = pairs().filter(|(a, b)| a.0 != b.0).count();
            let logits_off = pairs().filter(|(a, b)| a.1 != b.1).count();
            let bad_at = pairs().position(|(a, b)| a != b).map(|i| kept + i);
            let d = diff_caches(&whole, &whole_again);
            let ok = kept == want && steps && last_again == last && d.is_empty();
            println!(
                "cuts: cut to {n} from {stale} kept {kept} (rounded down to every ratio \
                 {ratios:?}: {want}); positions {kept}..{} again: argmax and logits {}{}, the \
                 last step's token, logits and {} l_out taps {}, caches and compressor state {}: \
                 {}",
                total - 1,
                if steps {
                    "equal".to_string()
                } else {
                    format!("differ ({tokens_off} argmax, {logits_off} logits)")
                },
                bad_at.map_or(String::new(), |p| format!(" (first at {p})")),
                last_again.l_out.len(),
                if last_again == last {
                    "equal"
                } else {
                    "differ"
                },
                if d.is_empty() {
                    "equal".to_string()
                } else {
                    d.join(", ")
                },
                verdict(ok)
            );
            all &= ok;
        }
        let (gpu, _, body) = m.body_parts("cuts")?;
        body.set_indexer_top_k(gpu, hp.indexer.top_k)?;
        Ok(all)
    }
}
