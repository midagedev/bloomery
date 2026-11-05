//! The Mistral Large 4 prefill gate: the prompt batch (`app::Prompt for Body`, the common walk over units of up to
//! [`T_MAX`] rows with the family's attention, `bloomery_gpu_mistrallarge4::prefill`) is the one-token step run per id
//! to within the error model's band, and the stores it leaves are the ones the next step reads.
//!
//! The batch keeps its own arithmetic at the dense sites (`project_wide`: every site of a layer's input one GEMM over
//! the unit's rows, its input rounded to the forms the site's type reads, where the step's gemv reads the row as it is);
//! the absorption and the values are the per-head q8_0 launches over the rows, the attention the selected latent flash
//! (roped layers) and the selected 512-wide walk (NoPE layers) over each row's own list, the key path, the indexer's
//! keys, scores and top-k, the SSMax pairs and the norms the step's rules at each row's own position. So the batch is the
//! steps' within a band, not bit for bit: a subset of the roundings ours and mainline's differ by, which
//! `gate_mistrallarge4_e2e` holds to its bands. A router of four picks of 128 can exchange an expert at a near tie
//! between the two paths' roundings, and the exchange is carried on: from the first routed layer on no error model bands
//! the free chain, and the logits' comparison is deferred (`gate_minimaxm3_prefill`'s reason).
//!
//! The prompt lengths are the fixture's selection boundaries and the unit's: `1` (one row, the step's shape), `7` (rows
//! short of a launch's eight), `16` (the fixture's top-k: every row still reads its whole causal prefix), `17` (the first
//! row whose list is a selection), `40` (the end-to-end gate's walk), `512` (one whole unit), `513` (a unit and one
//! row: the second unit's lists and records are its own) and `529` (a unit and seventeen rows, the longest: under group 2
//! the second unit's rows past the top-k select while the first unit's still read their lists, so a list of the first
//! the second's would take shows in the first's rows past its first). The ids are the fixed walk of `e2e::walk`.
//!
//! The clauses (all self-consistency, so both tiers run them), the body's own first:
//! - `prefill_selection_is_its_rule`: at the own lengths past the top-k, every full layer's list at the call's last row
//!   is, as a set, the rule's top-k of the batch's own scores over the visible rows (descending, a tie to the lower row),
//!   bit for bit; and layer 0's scores (its input the embedding on both paths) are within [`SCORE_BAND`] of the steps',
//!   the rows the two lists exchange printed.
//! - `prefill_rows_near_steps`: at the own lengths, the store rows no routed expert reaches — layer 0's latent rows and
//!   index keys at every position, layer 1's at the positions whose layer-0 attention reads the whole causal prefix (below
//!   the top-k, no selection to exchange) — within [`ROW_BAND`] of the steps' rows (layer 1: [`TAP_CEIL`], layer 0's
//!   output band), each roped latent row's rope span on its own as well.
//! - `prefill_head_caps_its_logits`: at the own lengths, the batch's logits row is the head's own projection of the
//!   call's last row (the head's input as the walk leaves it, projected again alone) under the file's softcap
//!   `cap·tanh(x/cap)`, within [`CAP_BAND`]; a file with no cap, the projection bit for bit.
//! - the common clauses ([`prefill_clauses`]): `prefill_near_steps` (layer 0's output within [`TAP_CEIL`], the store
//!   list, the position; the logits deferred), `prefill_then_step_near_steps`, `prefill_group_is_group_one` at group 2
//!   over the longest length (two units of 512 and 17: every logit, layer output and store digest bit for bit) and
//!   `fault_word_clean`. Where layer 0's pick at the call's last position (the batch's last row, the step after the
//!   prompt) exchanged rows with the steps' at a near tie of its scores, the free run's layer outputs are deferred and
//!   the position runs again with the steps' list forced into layer 0 (`Body::force_selection`), judged at the same
//!   ceiling; an exchange past a near tie is red.
//!
//! The load: the gate plan (the gate card's bytes, the routed experts the card does not hold on the host tier) priced for
//! the prompt group of two (`place::configure_group`), the step eager with its taps (`Body::arm_taps`: no capture holds
//! the copies) and the prompt batch's taps armed (`prefill::set_prefill_taps`). The fixture tier keeps half the experts
//! the gate card would hold, as the end-to-end gate's plan does, so both legs of every routed layer run.
//!
//! FAIL-first (box): the one-line mutants under the round's `.mutants/` name the clause each reddens.

#[cfg(feature = "mistrallarge4")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_mistrallarge4_prefill", gate::run())
}

#[cfg(feature = "mistrallarge4")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "mistrallarge4")]
#[allow(
    dead_code,
    reason = "the card the gate plans on; the gate reads its spec"
)]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "mistrallarge4")]
#[allow(
    dead_code,
    reason = "the engine serves several gates; each takes the calls it reads"
)]
#[path = "shared/lcpp_e2e.rs"]
mod lcpp_e2e;

#[cfg(feature = "mistrallarge4")]
#[allow(
    dead_code,
    reason = "the prefill clauses serve several gates; each takes the helpers they read"
)]
#[path = "shared/prefill_clauses.rs"]
mod prefill_clauses;

#[cfg(feature = "mistrallarge4")]
mod gate {
    use crate::e2e::{rel, walk};
    use crate::gate_card;
    use crate::lcpp_e2e::{Dims, Seat, Selection, fnv1a};
    use crate::prefill_clauses::{self, Case};
    use bloomery_gpu::GpuModel;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::prompts::SIGMA_LOGIT_V41;
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{GateError, bits_equal, checks_failed, open_split, verdict};
    use bloomery_gpu_mistrallarge4::body::Body;
    use bloomery_gpu_mistrallarge4::prefill::{T_MAX, set_prefill_group, set_prefill_taps};
    use gguf::quant::half_to_f32;
    use model::arch::Arch;
    use model::arch::mistrallarge4::hparams::Hparams;
    use model::arch::mistrallarge4::place::{self, PlanInputs};
    use model::placement::{Machine, PlanLevers, workstation};

    type Model = GpuModel<Body>;

    const NAME: &str = "gate_mistrallarge4_prefill";

    /// The recipe the model file is opened for.
    const RECIPE: &str = "gate-gpu-mistrallarge4-prefill";

    /// The prompt lengths (module doc).
    const LENGTHS: [usize; 8] = [1, 7, 16, 17, 40, T_MAX, T_MAX + 1, T_MAX + 17];

    /// The lengths the body's own clauses run: rows short of a launch, the first selecting length of the fixture, and a
    /// unit and one row.
    const OWN_LENGTHS: [usize; 3] = [7, 17, T_MAX + 1];

    /// The prompt group the group clause runs, which the plan is priced for.
    const GROUP: usize = 2;

    /// Positions of the cache beyond the longest prompt and the step after it.
    const CTX_SPARE: usize = 8;

    /// The host tier's configuration the load opens with: no lever is read.
    const HOST_CFG: bloomery_levers::HostCfg = bloomery_levers::HostCfg {
        populate: false,
        lock: false,
        card_dontneed: false,
        r8: false,
        route_trace: false,
    };

    /// The ceiling of layer 0's output distance between the batch and the steps on the free chain.
    ///
    /// [derived]: layer 0 reads the embedding on both paths and is the dense lead, so no expert exchange reaches it; the
    /// two paths differ at its dense sites (the GEMM's input forms and sum order against the step's gemv). A selection
    /// exchange at a near tie of the index scores is not in this ceiling: `prefill_selection_is_its_rule` holds the
    /// batch's list to the rule, and the common clauses judge a position with such an exchange on the run with the
    /// steps' list forced in. Ours against mainline's differ by those roundings and more: `gate_mistrallarge4_e2e` holds
    /// the dense nodes to 0.03 and the attention's value row to 0.06 where the selection is the dump's own, and the
    /// layer's output carries the larger share at most.
    // PIN(2026-11-02): provisional, the e2e gate's ATTN_REL_BAND; the first run prints layer 0's distance at every
    // length, and the ceiling becomes 1.2x its worst with this line naming the run. FAIL-first: rope_row_shift's step
    // after the prompt reads the batch's keys a position off.
    const TAP_CEIL: f64 = 0.06;

    /// The ceiling of a layer-0 store row's relative distance between the batch and the steps.
    ///
    /// [derived]: the row's input is the embedding on both paths; the projection rounds apart at the dense site (the e2e
    /// gate's dense nodes, 0.03, bound ours against mainline's, a superset), and the norm, the rope and the f16 append
    /// are the step's rule at the row's own position and rope row. A rope row a position off rotates the span's
    /// fastest pairs by about a radian: a relative distance of order one on the rope span.
    // PIN(2026-11-02): provisional, the e2e gate's DENSE_REL_BAND; the first run prints the worst row of each store.
    const ROW_BAND: f64 = 0.03;

    /// The ceiling of layer 0's index scores' relative distance over the visible rows between the batch and the steps.
    ///
    /// [derived]: the scores are the index queries' and keys' dot products, both projected at dense sites from the
    /// embedding's norms on both paths, weighted by the head weights' projection: each factor within the dense band, a
    /// product of two within twice it, which the attention's 0.06 covers.
    // PIN(2026-11-02): provisional, as TAP_CEIL; the first run prints layer 0's score distance at each length.
    const SCORE_BAND: f64 = 0.06;

    /// The ceiling of a logit's distance between the batch's capped row and the head's projection capped on the host.
    ///
    /// [derived]: both sides round `cap·tanh(x/cap)` in f32 from the same projection bits; the device's tanh is
    /// libdevice's and the host's differs by ulps (`gate_p4`'s softcap case), a few ulps of the cap apart (30 · 2⁻²³ ≈
    /// 3.6e-6 an ulp); a missing cap moves a logit `x` by about `x³ / (3·cap²)`, past this once `|x|` is past ~1.
    const CAP_BAND: f32 = 1e-4;

    /// The free chain's ceiling: layer 0 only (module doc).
    fn tap_ceil(_tier: Tier, layer: usize) -> f64 {
        if layer == 0 { TAP_CEIL } else { f64::INFINITY }
    }

    fn tap_ceils(_tier: Tier) -> String {
        format!(
            "ceiling {TAP_CEIL} at layer 0, none after it (an expert exchanged at a near tie is carried on)"
        )
    }

    /// The path the last call ran: the steps write the step's taps, the prompt batch the shell's.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Path {
        Step,
        Prompt,
    }

    /// The load behind the seat's calls.
    struct Ml4 {
        m: Model,
        dims: Dims,
        last: Path,
    }

    impl Ml4 {
        /// The last call's logits row.
        fn row(&self) -> Result<Vec<f32>, GateError> {
            let mut row = vec![0.0; self.dims.vocab];
            self.m.logits_into(&mut row)?;
            Ok(row)
        }

        /// Drop the forced list: it stands for one call ([`Seat::force_selection`]).
        fn release_forced(&mut self) -> Result<(), GateError> {
            let (_, _, body) = self.m.body_parts(NAME)?;
            body.clear_forced_selection();
            Ok(())
        }
    }

    impl Seat for Ml4 {
        fn dims(&self) -> Dims {
            self.dims
        }

        fn reset(&mut self) -> Result<(), GateError> {
            Ok(self.m.reset()?)
        }

        fn pos(&self) -> u32 {
            self.m.pos()
        }

        /// The eager step, its taps armed.
        fn step(&mut self, id: u32) -> Result<Vec<f32>, GateError> {
            let ran = self.m.step(&[id]);
            let released = self.release_forced();
            ran?;
            released?;
            self.last = Path::Step;
            self.row()
        }

        /// The prompt in batches (`app::Prompt for Body`), one residency pass a call.
        fn prompt(&mut self, ids: &[u32]) -> Result<Vec<f32>, GateError> {
            let ran = <Body as app::Prompt>::prompt(&mut self.m, ids);
            let released = self.release_forced();
            ran?;
            released?;
            self.last = Path::Prompt;
            self.row()
        }

        fn set_group(&mut self, group: usize) -> Result<(), GateError> {
            set_prefill_group(&mut self.m, group)?;
            Ok(())
        }

        /// Every layer's output at the last position of the last call, from the taps its path wrote.
        fn taps(&mut self) -> Result<Vec<f32>, GateError> {
            let layers = self.dims.layers;
            let last = self.last;
            let (gpu, _, body) = self.m.body_parts(NAME)?;
            Ok(match last {
                Path::Step => body
                    .nodes(gpu, 0..layers)?
                    .into_iter()
                    .flat_map(|n| n.l_out)
                    .collect(),
                Path::Prompt => body.prefill_taps(gpu)?,
            })
        }

        fn stores(&mut self) -> Result<Vec<(String, u64)>, GateError> {
            let pos = usize::try_from(self.m.pos())?;
            let (gpu, _, body) = self.m.body_parts(NAME)?;
            Ok(body
                .store_rows(gpu, pos)?
                .into_iter()
                .map(|(name, bytes)| (name, fnv1a(&bytes)))
                .collect())
        }

        /// Layer 0's list and scores at the last position of the last call, from the taps its path wrote (the header
        /// reader refuses a file whose layer 0 is not a full indexer layer).
        fn selection(&mut self) -> Result<Option<Selection>, GateError> {
            let visible = usize::try_from(self.m.pos())?;
            let last = self.last;
            let (gpu, _, body) = self.m.body_parts(NAME)?;
            let (list, scores) = match last {
                Path::Step => (
                    body.lists(gpu, 0..1)?
                        .pop()
                        .ok_or("the steps' lists hold no row for layer 0")?
                        .1,
                    body.index_scores(gpu, 0)?,
                ),
                Path::Prompt => body
                    .prefill_selection(gpu)?
                    .into_iter()
                    .find(|x| x.0 == 0)
                    .map(|(_, list, scores)| (list, scores))
                    .ok_or("the batch's selection taps hold no row for layer 0")?,
            };
            let scores = scores
                .get(..visible)
                .ok_or("layer 0's scores hold fewer values than the visible rows")?
                .to_vec();
            Ok(Some(Selection {
                layer: 0,
                list,
                scores,
            }))
        }

        fn force_selection(&mut self, layer: usize, list: &[u32]) -> Result<(), GateError> {
            let (gpu, _, body) = self.m.body_parts(NAME)?;
            Ok(body.force_selection(gpu, layer, list)?)
        }

        fn fault(&mut self) -> Result<(), GateError> {
            if let Some(fault) = self.m.poisoned() {
                return Err(format!("the model is poisoned by {fault:?}").into());
            }
            match self.m.gpu().fault()? {
                None => Ok(()),
                Some(fault) => Err(format!("the card's fault word holds {fault:?}").into()),
            }
        }
    }

    // ------------------------------------------------------------ the body's own clauses

    /// What a path left that the own clauses read: layers 0 and 1's store rows decoded (name, layer, values), each
    /// full layer's list and scores at the last row, and the last row's logits.
    struct Left {
        rows: Vec<(String, usize, Vec<f32>)>,
        sel: Vec<(usize, Vec<u32>, Vec<f32>)>,
        logits: Vec<f32>,
    }

    /// Layers 0 and 1's store rows over the first `pos` positions, decoded from f16.
    fn early_rows(seat: &mut Ml4, pos: usize) -> Result<Vec<(String, usize, Vec<f32>)>, GateError> {
        let (gpu, _, body) = seat.m.body_parts(NAME)?;
        let mut out = Vec::new();
        for (name, bytes) in body.store_rows(gpu, pos)? {
            let layer = if name.starts_with("blk.0.") {
                0
            } else if name.starts_with("blk.1.") {
                1
            } else {
                continue;
            };
            let values = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half_to_f32(u16::from_le_bytes(*b)))
                .collect();
            out.push((name, layer, values));
        }
        Ok(out)
    }

    /// A reset, then `ids` one a step: the rows and each full layer's list and scores after the last.
    fn by_steps(seat: &mut Ml4, ids: &[u32], full: &[usize]) -> Result<Left, GateError> {
        seat.reset()?;
        let mut logits = Vec::new();
        for &id in ids {
            logits = seat.step(id)?;
        }
        let rows = early_rows(seat, ids.len())?;
        let (gpu, _, body) = seat.m.body_parts(NAME)?;
        let mut sel = Vec::with_capacity(full.len());
        for &l in full {
            let list = body
                .lists(gpu, l..l + 1)?
                .pop()
                .ok_or("the steps' lists hold no row for a full layer")?
                .1;
            sel.push((l, list, body.index_scores(gpu, l)?));
        }
        Ok(Left { rows, sel, logits })
    }

    /// A reset, then `ids` as one prompt call: the rows and the selection taps.
    fn by_prompt(seat: &mut Ml4, ids: &[u32]) -> Result<Left, GateError> {
        seat.reset()?;
        let logits = seat.prompt(ids)?;
        let rows = early_rows(seat, ids.len())?;
        let (gpu, _, body) = seat.m.body_parts(NAME)?;
        Ok(Left {
            rows,
            sel: body.prefill_selection(gpu)?,
            logits,
        })
    }

    /// `prefill_head_caps_its_logits` at `p` ids, right after the prompt call that left `capped` (module doc): the head
    /// alone over the row the walk left in its input, uncapped, then capped on the host. The position moves one past the
    /// call; the caller resets.
    fn head_caps_its_logits(
        seat: &mut Ml4,
        p: usize,
        cap: f32,
        capped: &[f32],
    ) -> Result<bool, GateError> {
        seat.m.run_rows(1, NAME, |gpu, w, _body, head, _pos| {
            head.enqueue(gpu, w)?;
            Ok(true)
        })?;
        let raw = seat.row()?;
        let pass = if cap > 0.0 {
            let want: Vec<f32> = raw.iter().map(|&x| cap * (x / cap).tanh()).collect();
            // A NaN distance is out of band and wins the fold.
            let far = |a: &[f32], b: &[f32]| {
                a.iter().zip(b).fold(0.0f32, |w, (&x, &y)| {
                    let d = (x - y).abs();
                    if d.is_nan() || d > w { d } else { w }
                })
            };
            let (worst, moved) = (far(capped, &want), far(&raw, &want));
            // A cap that moves no logit past the band cannot be told from a missing one.
            let pass = capped.len() == want.len() && worst <= CAP_BAND && moved > CAP_BAND;
            println!(
                "prefill_head_caps_its_logits: P={p}: the row against the head's projection capped at {cap} on the host, \
                 worst {worst:.2e} (band {CAP_BAND}), the cap's own largest move {moved:.2e} {}",
                verdict(pass)
            );
            pass
        } else {
            let pass = bits_equal(capped, &raw);
            println!(
                "prefill_head_caps_its_logits: P={p}: no cap in the file, the row against the head's projection bit for \
                 bit {}",
                verdict(pass)
            );
            pass
        };
        Ok(pass)
    }

    /// The `k` best of `scores` by the indexer's rule (descending score, a tie to the lower row), in row order.
    fn rule_top(scores: &[f32], k: usize) -> Vec<usize> {
        let mut ids: Vec<usize> = (0..scores.len()).collect();
        ids.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        ids.truncate(k);
        ids.sort_unstable();
        ids
    }

    /// A list's rows in row order.
    fn sorted(list: &[u32]) -> Vec<usize> {
        let mut v: Vec<usize> = list.iter().map(|&r| r as usize).collect();
        v.sort_unstable();
        v
    }

    /// `prefill_selection_is_its_rule` at `p` ids (module doc): `s` the steps', `b` the batch's.
    fn selection_is_its_rule(p: usize, top_k: usize, s: &Left, b: &Left) -> bool {
        let mut ok = s.sel.len() == b.sel.len() && !b.sel.is_empty();
        for ((ls, step_list, step_scores), (lb, list, scores)) in s.sel.iter().zip(&b.sel) {
            let rule = rule_top(scores.get(..p).unwrap_or(&[]), top_k);
            let own = sorted(list) == rule && *ls == *lb;
            let mut line = format!(
                "prefill_selection_is_its_rule: P={p} layer {lb}: the list is the rule's top {top_k} of its own scores {}",
                verdict(own)
            );
            let mut pass = own;
            if *lb == 0 {
                let d = rel(
                    scores.get(..p).unwrap_or(&[]),
                    step_scores.get(..p).unwrap_or(&[]),
                );
                let near = d.is_finite() && d <= SCORE_BAND;
                let (bs, ss) = (sorted(list), sorted(step_list));
                let exchanged = bs.iter().filter(|r| !ss.contains(r)).count();
                line.push_str(&format!(
                    "; scores against the steps' {d:.2e} (band {SCORE_BAND}) {}, {exchanged} rows exchanged",
                    verdict(near)
                ));
                pass &= near;
            }
            println!("{line}");
            ok &= pass;
        }
        ok
    }

    /// `prefill_rows_near_steps` at `p` ids (module doc).
    fn rows_near_steps(p: usize, hp: &Hparams, s: &Left, b: &Left) -> bool {
        let same = s.rows.len() == b.rows.len()
            && !b.rows.is_empty()
            && s.rows.iter().zip(&b.rows).all(|(x, y)| x.0 == y.0);
        let mut ok = same;
        if !same {
            println!(
                "prefill_rows_near_steps: P={p}: the store lists differ ({} against {}) FAIL",
                s.rows.len(),
                b.rows.len()
            );
        }
        for ((name, layer, sv), (_, _, bv)) in s.rows.iter().zip(&b.rows) {
            let (band, upto) = if *layer == 0 {
                (ROW_BAND, p)
            } else {
                (TAP_CEIL, p.min(hp.indexer_top_k))
            };
            let width = sv.len() / p.max(1);
            let sized = sv.len() == bv.len() && width * p == sv.len() && upto > 0;
            // A roped latent row is `[rope | latent]`; an index key is transformed after its rope,
            // so only the whole row is compared.
            let pe = if name.ends_with(".latent") {
                hp.pe(*layer)
            } else {
                0
            };
            let (mut worst, mut worst_rope, mut at) = (0.0f64, 0.0f64, 0usize);
            for t in 0..if sized { upto } else { 0 } {
                let (x, y) = (
                    &bv[t * width..(t + 1) * width],
                    &sv[t * width..(t + 1) * width],
                );
                // `rel` reads a NaN as infinite.
                let d = rel(x, y);
                let dr = if pe > 0 { rel(&x[..pe], &y[..pe]) } else { 0.0 };
                if d > worst || dr > worst_rope {
                    at = t;
                }
                worst = worst.max(d);
                worst_rope = worst_rope.max(dr);
            }
            let pass = sized && worst <= band && worst_rope <= band;
            println!(
                "prefill_rows_near_steps: P={p} {name}: worst row {worst:.2e}, rope span {worst_rope:.2e} \
                 (at position {at} of {upto}, band {band}) {}",
                verdict(pass)
            );
            ok &= pass;
        }
        ok
    }

    /// The body's own clauses over `lengths`: each length's steps and prompt once, both clauses read from them.
    fn own_clauses(seat: &mut Ml4, hp: &Hparams, lengths: &[usize]) -> Result<bool, GateError> {
        const SEL: &str = "prefill_selection_is_its_rule";
        const ROWS: &str = "prefill_rows_near_steps";
        const HEAD: &str = "prefill_head_caps_its_logits";
        let (sel_on, rows_on, head_on) = (
            tier::run_clause(SEL, Tag::SelfConsistency)?,
            tier::run_clause(ROWS, Tag::SelfConsistency)?,
            tier::run_clause(HEAD, Tag::SelfConsistency)?,
        );
        let full: Vec<usize> = (0..hp.n_layer).filter(|&l| hp.indexer_full[l]).collect();
        let (mut sel_ok, mut rows_ok, mut head_ok) = (true, true, true);
        for &p in lengths {
            let sel_here = sel_on && p > hp.indexer_top_k;
            if !(rows_on || sel_here || head_on) {
                continue;
            }
            let ids = walk(p, hp.n_vocab);
            let s = if rows_on || sel_here {
                Some(by_steps(seat, &ids, &full)?)
            } else {
                None
            };
            let b = by_prompt(seat, &ids)?;
            if let Some(s) = &s {
                if sel_here {
                    sel_ok &= selection_is_its_rule(p, hp.indexer_top_k, s, &b);
                }
                if rows_on {
                    rows_ok &= rows_near_steps(p, hp, s, &b);
                }
            }
            if head_on {
                head_ok &= head_caps_its_logits(seat, p, hp.softcap, &b.logits)?;
            }
        }
        for (clause, on, pass) in [
            (SEL, sel_on, sel_ok),
            (ROWS, rows_on, rows_ok),
            (HEAD, head_on, head_ok),
        ] {
            if on && !pass {
                println!("clause {clause} FAIL");
            }
        }
        Ok(sel_ok && rows_ok && head_ok)
    }

    // ------------------------------------------------------------ the load

    /// The plan's levers: in the fixture tier the card keeps half of the experts the gate card would hold (so the host
    /// leg serves the rest and both legs run), in the real tier the card's own bytes (`gate_mistrallarge4_e2e`'s).
    fn levers(
        inputs: &PlanInputs,
        machine: &Machine,
        (ctx, tier_now): (u64, Tier),
    ) -> Result<PlanLevers, GateError> {
        if tier_now == Tier::Real {
            return Ok(PlanLevers::default());
        }
        let all = inputs.plan(machine, ctx, &PlanLevers::default())?;
        let c = &all.cards[0];
        let card = &machine.cards[0];
        let kept = c.dense_bytes
            + c.rounding_bytes
            + c.kv_bytes
            + c.stage_rows_bytes
            + card
                .set_aside_bytes()
                .ok_or("the card's set-aside is unbounded")?
            + card.margin_bytes
            + inputs.prompt_reserve
            + c.expert_bytes / 2;
        Ok(PlanLevers {
            card_budget_bytes: Some(kept).into(),
            ..PlanLevers::default()
        })
    }

    pub fn run() -> Result<(), GateError> {
        let tier_now = Tier::from_env()?;
        let file = open_split(Arch::MistralLarge4, RECIPE)?;
        let hp = Hparams::read(&file).map_err(|e| format!("mistral-large4 header: {e}"))?;
        let longest = LENGTHS.iter().copied().max().unwrap_or(1);
        let ctx = longest + 1 + CTX_SPARE;
        // The gate plan, priced for the group clause's two units before it is made.
        let mut inputs = PlanInputs::describe(&file)?;
        place::configure_group(&mut inputs, GROUP)?;
        let machine = workstation::placed_on(gate_card::card()?, hp.n_layer, workstation::SCRATCH);
        let ctx64 = u64::try_from(ctx)?;
        let plan = inputs.plan(
            &machine,
            ctx64,
            &levers(&inputs, &machine, (ctx64, tier_now))?,
        )?;
        println!(
            "plan: ctx={ctx} card experts {} ({} B) host experts {} ({} B) prompt reserve {} B (plan values)",
            plan.cards[0].experts,
            plan.cards[0].expert_bytes,
            plan.host.experts,
            plan.host.expert_bytes,
            inputs.prompt_reserve
        );
        let t = std::time::Instant::now();
        let mut m = Body::open_placed(file, &plan, &inputs, 0, HOST_CFG)?;
        // The step eager with its taps (no capture holds their copies), the batch's taps beside them.
        m.set_mode(StepMode::Eager);
        {
            let (gpu, _, body) = m.body_parts(NAME)?;
            body.arm_taps(gpu, true)?;
        }
        set_prefill_taps(&mut m, true)?;
        let prompt_bytes = {
            let (_, _, body) = m.body_parts(NAME)?;
            body.prompt_bytes()
        };
        println!(
            "load resident_bytes={} prompt_bytes={prompt_bytes} ctx={ctx} layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            hp.n_layer,
            t.elapsed().as_secs_f64()
        );
        let mut seat = Ml4 {
            m,
            dims: Dims {
                layers: hp.n_layer,
                embd: hp.n_embd,
                vocab: hp.n_vocab,
            },
            last: Path::Step,
        };
        // The body's own clauses first, so the common run's tally line counts them.
        let own = own_clauses(&mut seat, &hp, &OWN_LENGTHS)?;
        let case = Case {
            lengths: LENGTHS.to_vec(),
            tap_ceil,
            tap_ceils,
            group: GROUP,
            // A top-4 near tie can flip between the batch's GEMM sum order and the step's gemv, and a flip is a
            // discrete exchange of experts: the comparison is deferred.
            logit_bound: None,
            sigma_logit: SIGMA_LOGIT_V41,
            vocab: hp.n_vocab,
            embd: hp.n_embd,
            tier: tier_now,
        };
        let common = prefill_clauses::run(NAME, &case, &mut seat);
        match (common, own) {
            (Err(e), _) => Err(e),
            (Ok(()), false) => Err(checks_failed()),
            (Ok(()), true) => Ok(()),
        }
    }
}
