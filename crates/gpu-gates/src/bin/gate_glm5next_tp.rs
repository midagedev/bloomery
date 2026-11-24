//! The tensor-parallel GLM-5.3-Flash body (`bloomery_gpu_glm5next::tp`) on the fixture, every
//! rank a context of its own on the one visible card and the ranks' parts carried by a loopback
//! (`bloomery_gpu::host::GatherLoop`). Each run is the batch set's tokens stepped from a reset,
//! the taps armed: every layer's streams, every routed layer's picks and the logits at each
//! position. The rank counts are data: [`RANKS`].
//!
//! - (n1) one rank against the one-card body (`Body::open_placed`, the gate plan without a card
//!   budget, so every expert the plan admits is on the card): the layers before the first one
//!   whose routed experts the one-card body leaves on the host are its taps bit for bit at every
//!   position, and every token is its token. From that layer on, the rank computes on the card
//!   what the one-card body computes on the host (the fixture's Q6_K downs, which only a rank's
//!   leg admits), so those layers are held within their stream bands ([`FxBands`]) and the
//!   logits within the head band, the argmax a tie at it. The gate prints both ranges.
//! - (q6) the first such layer alone: its input and its mixer are the one-card body's bit for
//!   bit, so its streams differ by the routed experts' engine only, the rank's card leg with the
//!   Q6_K down against the host's rule, held within the layer's own block band.
//! - (hm) every run's step token, at every position, the argmax of the ranks' logits laid in
//!   rank order (the largest value, of equal values the lowest id): the head's merge of the
//!   ranks' picks names the whole vocabulary's id.
//! - (nN) N ranks for each N past one: every rank's taps, picks and tokens bit for bit rank 0's,
//!   and the run within the bands of the one-rank run, layer by layer and at the head. Every
//!   count loads and runs but the one whose part of the fixture's expert width is off its format
//!   block ([`REFUSED`]), whose open is refused by name: the gate prints the refusal and holds it
//!   to name the tensor and the count; a refusal at any other count fails the gate.
//! - (rr) the first N past one twice from fresh opens: the two runs bit for bit.
//! - (st) one rank's positions: opens of no positions or past the reference sets' refused by
//!   name; on a fresh open its reads refused for want of a step, and a first step at position 1
//!   or at the stores' end refused by name before any launch; after the step at 0 the same
//!   position refused again, its logits read, and the step at 1 taken. Taps armed on the fresh
//!   open; after a step, arming again refused by name and the step's taps read bit for bit as
//!   before it, and once disarmed, the taps' read and a new arming refused by name. Wrong GPUs,
//!   shared rank streams, invalid tokens, and missing, foreign or released links are refused
//!   before a launch; a loopback refuses short pages.
//! - (fx) every run that loaded against ik's batch set of the fixture (`fx_ref_glm5next`), as the
//!   e2e gate's (cfx) holds the one-card body: a flip first on its path held to the pair rule
//!   under the layer's router band, every layer output off every flip's path within its stream
//!   band, and the last position's argmax a tie at the head band.

#[cfg(feature = "glm5next")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_glm5next_tp", gate::run())
}

#[cfg(not(feature = "glm5next"))]
fn main() {}

#[cfg(feature = "glm5next")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "glm5next")]
#[path = "shared/e2e.rs"]
#[allow(
    dead_code,
    reason = "the gate reads the compare, tap, layer-table and path helpers; the rest serve the \
              other gates"
)]
mod e2e;

#[cfg(feature = "glm5next")]
#[path = "shared/glm5next_tier.rs"]
mod glm5next_tier;

#[cfg(feature = "glm5next")]
mod gate {
    use std::sync::Arc;

    use crate::e2e::{argmax, layer_table, on_path, rel, same_bits};
    use crate::glm5next_tier::{self, FxBands, IkRoute, fx_route_cap, ik_bands};
    use bloomery_gpu::hc::HC_STREAMS as STREAMS;
    use bloomery_gpu::host::{GatherLoop, PostCols};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, GpuError, MappedHost};
    use bloomery_gpu_deepseek41::router::glm5next::N_EXPERT;
    use bloomery_gpu_gates::act_rule::Arm;
    use bloomery_gpu_gates::flip::{self, Flip};
    use bloomery_gpu_gates::tier::Tier;
    use bloomery_gpu_gates::{GateError, verdict};
    use bloomery_gpu_glm5next::tp::{self, Rank, RankOf, RoutePick};
    use bloomery_gpu_glm5next::{Body, set_taps};
    use gguf::{GgmlType, Split};
    use model::arch::glm5next::names;
    use model::arch::glm5next::place::{ORACLE_POSITIONS, PlanInputs};
    use model::placement::PlanLevers;
    use refset::arch::glm5next::fixture as fx;
    use runtime::layer::FfnKind;

    /// The rank counts the gate loads: none is special to the program.
    const RANKS: [usize; 3] = [1, 2, 4];

    /// The one count of [`RANKS`] the fixture refuses at the open: its part of the expert width,
    /// 512 values over 4, cuts off the down stacks' 256-value Q5_K block. Every other count loads
    /// and runs.
    const REFUSED: usize = 4;

    /// Positions of the stores: past the batch set's tokens.
    const CTX: usize = 64;

    const FX: &str = "(fx) ik's batch set of the fixture";

    /// One run of the batch set's tokens from a reset: each position's token, logits over the
    /// whole vocabulary, layer streams and routed picks (`None` for a dense layer).
    struct Run {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
        taps: Vec<Vec<f32>>,
        routes: Vec<Vec<Option<RoutePick>>>,
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[])?;
        if Tier::from_env()? != Tier::Fixture {
            return Err(
                "gate_glm5next_tp runs on the fixture tier only: the real file's ranks \
                        are loaded across nodes, not on one card"
                    .into(),
            );
        }
        crate::gate_card::init()?;
        glm5next_tier::init_file()?;
        let toks = glm5next_tier::batch_tokens()?;
        let file = Arc::new(glm5next_tier::open()?);
        let inputs = PlanInputs::describe(&file)?;
        let kinds: Vec<_> = inputs
            .spec
            .layers
            .iter()
            .map(runtime::layer::Layer::of)
            .collect();
        let bands = ik_bands(&file, &kinds, Arm::Gemv)?;
        let embd = inputs.hp.n_embd;
        let (one, host_from) = one_card(&levers, &inputs, &toks)?;
        let mut ok = true;
        let n1 = ranks_run(&file, 1, &toks)?.remove(0);
        ok &= merge_clause(1, &n1);
        ok &= n1_clause(&one, &n1, host_from, &bands, embd);
        ok &= q6_clause(&file, &one, &n1, host_from, embd)?;
        let first = *toks.first().ok_or("no batch tokens")?;
        ok &= state_clause(&file, first)?;
        let mut loaded = vec![(1, n1)];
        for &n in RANKS.iter().filter(|&&n| n > 1) {
            match ranks_run(&file, n, &toks) {
                Ok(runs) => {
                    ok &= merge_clause(n, &runs[0]);
                    ok &= ranks_clause(n, &runs, &loaded[0].1, &bands, embd);
                    if loaded.len() == 1 {
                        let again = ranks_run(&file, n, &toks)?;
                        let same = runs.iter().zip(&again).all(|(a, b)| same_run(a, b));
                        println!(
                            "(rr) {n} ranks twice from fresh opens: every rank's tokens, logits, \
                             taps and picks bit for bit {}",
                            verdict(same)
                        );
                        ok &= same;
                    }
                    loaded.push((n, runs.into_iter().next().ok_or("no rank's run")?));
                }
                Err(e) if n == REFUSED => ok &= refused_clause(n, &e),
                Err(e) => return Err(e),
            }
        }
        let man = glm5next_tier::fx_set(FX, &fx::IK, fx::BATCH)?;
        for (n, r) in &loaded {
            ok &= fx_clause(&man, &file, *n, r, &bands, embd)?;
        }
        if ok {
            Ok(())
        } else {
            Err("a clause failed".into())
        }
    }

    /// The one-card body's run of `toks` on the gate plan without a card budget, and its first
    /// routed layer whose experts are not all on the card (the trunk's count when none is).
    fn one_card(
        levers: &bloomery_levers::Levers,
        inputs: &PlanInputs,
        toks: &[u32],
    ) -> Result<(Run, usize), GateError> {
        let machine = crate::gate_card::plan_gate(inputs.model.layers);
        let plan = inputs.plan(&machine, CTX as u64, &PlanLevers::default())?;
        let experts = plan.model.experts;
        let host_from = inputs
            .spec
            .layers
            .iter()
            .enumerate()
            .position(|(l, s)| {
                runtime::layer::Layer::of(s).ffn == FfnKind::Moe
                    && plan.n_l.get(l).copied() != Some(experts)
            })
            .unwrap_or(inputs.spec.layers.len());
        println!(
            "one-card plan: card experts by layer {:?} of {experts}; the first layer with host \
             experts {host_from}",
            plan.n_l
        );
        let mut m = Body::open_placed(glm5next_tier::open()?, &plan, inputs, 0, levers.host())?;
        m.reset()?;
        set_taps(&mut m, true)?;
        m.set_mode(StepMode::Eager);
        let mut r = Run {
            tokens: Vec::new(),
            logits: Vec::new(),
            taps: Vec::new(),
            routes: Vec::new(),
        };
        for &t in toks {
            r.tokens.push(m.step(&[t])?);
            r.logits.push(m.logits()?);
            let (gpu, _, b) = m.body_parts("one-card run")?;
            r.taps.push(b.taps(gpu)?);
        }
        Ok((r, host_from))
    }

    /// `n` ranks' runs of `toks` from fresh opens, rank 0's first; the ranks' loopback released
    /// and every card drained on a failed step before the error returns.
    fn ranks_run(file: &Arc<Split>, n: usize, toks: &[u32]) -> Result<Vec<Run>, GateError> {
        let gpus = (0..n).map(|_| Gpu::new()).collect::<Result<Vec<_>, _>>()?;
        let mut ranks = Vec::with_capacity(n);
        for (r, gpu) in gpus.iter().enumerate() {
            ranks.push(Rank::open(gpu, Arc::clone(file), RankOf::new(r, n)?, CTX)?);
        }
        let ring = join(&ranks)?;
        for (gpu, r) in gpus.iter().zip(ranks.iter_mut()) {
            r.set_taps(gpu, true)?;
        }
        let mut runs: Vec<Run> = (0..n)
            .map(|_| Run {
                tokens: Vec::new(),
                logits: Vec::new(),
                taps: Vec::new(),
                routes: Vec::new(),
            })
            .collect();
        for (pos, &t) in toks.iter().enumerate() {
            let mut pairs: Vec<(&Gpu, &mut Rank)> = gpus.iter().zip(ranks.iter_mut()).collect();
            let token = match tp::step(&mut pairs, ring.as_ref(), t, pos as u32) {
                Ok(token) => token,
                Err(e) => {
                    let released = ring.as_ref().map(GatherLoop::release);
                    let broken = ring.as_ref().and_then(GatherLoop::broken);
                    for g in &gpus {
                        g.stream().synchronize().map_err(GpuError::from)?;
                    }
                    return Err(format!(
                        "{n} ranks, position {pos}: {e} (the loopback {broken:?}, released \
                         {released:?})"
                    )
                    .into());
                }
            };
            let mut logits = Vec::new();
            for (gpu, r) in gpus.iter().zip(&ranks) {
                logits.extend(r.logits(gpu)?);
            }
            for (run, (gpu, r)) in runs.iter_mut().zip(gpus.iter().zip(&ranks)) {
                run.tokens.push(token);
                run.logits.push(logits.clone());
                run.taps.push(r.taps(gpu)?);
                run.routes.push(r.routes(gpu)?);
            }
        }
        drop(ring);
        Ok(runs)
    }

    /// The ranks' pages joined by the loopback, each end posting one row; `None` at one rank.
    fn join(ranks: &[Rank]) -> Result<Option<GatherLoop>, GateError> {
        if ranks.len() < 2 {
            return Ok(None);
        }
        let mut layouts = Vec::with_capacity(ranks.len());
        let mut pages: Vec<Arc<MappedHost>> = Vec::with_capacity(ranks.len());
        for r in ranks {
            let (layout, page) = r
                .link()
                .ok_or("a rank of several without its gather page")?;
            layouts.push(layout.clone());
            pages.push(Arc::clone(page));
        }
        let cols = PostCols::default();
        cols.set(1);
        Ok(Some(GatherLoop::new(&layouts, &pages, &cols)?))
    }

    /// Layer `l`'s streams of a run's step `t`.
    fn layer(r: &Run, t: usize, l: usize, embd: usize) -> &[f32] {
        let row = STREAMS * embd;
        &r.taps[t][l * row..(l + 1) * row]
    }

    /// Each layer's worst relative distance of `a` from `b` over the positions, with each
    /// position's logits' worst against `b`'s and whether every argmax is a tie at `head`.
    fn against(a: &Run, b: &Run, embd: usize, head: f64) -> (Vec<f64>, f64, bool) {
        let layers = b.taps.first().map_or(0, |t| t.len() / (STREAMS * embd));
        let worst = (0..layers)
            .map(|l| {
                (0..b.taps.len())
                    .map(|t| rel(layer(a, t, l, embd), layer(b, t, l, embd)))
                    .fold(0.0f64, f64::max)
            })
            .collect();
        let logits = a
            .logits
            .iter()
            .zip(&b.logits)
            .map(|(x, y)| rel(x, y))
            .fold(0.0f64, f64::max);
        let ties = a
            .logits
            .iter()
            .zip(&b.logits)
            .all(|(x, y)| flip::head_tie(y, argmax(x), head));
        (worst, logits, ties)
    }

    /// (n1) (module doc).
    fn n1_clause(one: &Run, n1: &Run, host_from: usize, bands: &FxBands, embd: usize) -> bool {
        let positions = one.taps.len();
        let bits = (0..host_from).all(|l| {
            (0..positions).all(|t| same_bits(layer(n1, t, l, embd), layer(one, t, l, embd)))
        });
        let tokens = (0..positions).all(|t| {
            n1.tokens[t] == one.tokens[t]
                || flip::head_tie(&one.logits[t], n1.tokens[t], bands.head)
        });
        let (worst, logits, ties) = against(n1, one, embd, bands.head);
        let mut held = bits && tokens;
        println!(
            "(n1) bit-equal range: layers 0..{host_from} (every expert on the card in both), {} \
             positions: taps bit for bit {}",
            positions,
            verdict(bits)
        );
        for (l, &e) in worst.iter().enumerate().skip(host_from) {
            let pass = e <= bands.stream[l];
            held &= pass;
            println!(
                "(n1) band range: layer {l} l_out_rel {e:.3e} of the stream band {:.3e} (the rank \
                 computes on the card what the one-card body computes on the host) {}",
                bands.stream[l],
                verdict(pass)
            );
        }
        let head = logits <= bands.head && ties;
        held &= head;
        println!(
            "(n1) band range: logits rel {logits:.3e} of the head band {:.3e}, every argmax a tie \
             at it {ties}; tokens the one-card body's or a tie {tokens} {}",
            bands.head,
            verdict(head && tokens)
        );
        held
    }

    /// (q6) (module doc): the first layer the one-card body serves on the host, if any.
    fn q6_clause(
        file: &Split,
        one: &Run,
        n1: &Run,
        host_from: usize,
        embd: usize,
    ) -> Result<bool, GateError> {
        if host_from >= one.taps.first().map_or(0, |t| t.len() / (STREAMS * embd)) {
            println!(
                "(q6) no layer's experts on the host in the one-card plan: nothing to hold PASS"
            );
            return Ok(true);
        }
        let down = names::ffn_down_exps(host_from);
        let ty = file.find(&down).map(|(_, t)| t.ty);
        let band = glm5next_tier::block_terms(file, host_from, FfnKind::Moe, Arm::Gemv)?
            .var()
            .sqrt();
        let worst = (0..one.taps.len())
            .map(|t| {
                rel(
                    layer(n1, t, host_from, embd),
                    layer(one, t, host_from, embd),
                )
            })
            .fold(0.0f64, f64::max);
        let pass = ty == Some(GgmlType::Q6_K) && worst <= band;
        println!(
            "(q6) layer {host_from} ({down} {ty:?}): the rank's card leg against the host's rule, \
             l_out_rel {worst:.3e} of the layer's block band {band:.3e} {}",
            verdict(pass)
        );
        Ok(pass)
    }

    /// (hm) (module doc) for the run of `n` ranks.
    fn merge_clause(n: usize, run: &Run) -> bool {
        let off = run
            .tokens
            .iter()
            .zip(&run.logits)
            .filter(|&(&t, l)| t != argmax(l))
            .count();
        let pass = off == 0 && !run.tokens.is_empty() && run.tokens.len() == run.logits.len();
        println!(
            "(hm) {n} ranks: the step's token the argmax of the ranks' logits in rank order at \
             every position, {off} of {} otherwise {}",
            run.tokens.len(),
            verdict(pass)
        );
        pass
    }

    /// (nN) (module doc) at `n` ranks, `n1` the one-rank run.
    fn ranks_clause(n: usize, runs: &[Run], n1: &Run, bands: &FxBands, embd: usize) -> bool {
        let first = &runs[0];
        let alike = runs.iter().all(|r| same_run(r, first));
        println!(
            "({n}) {n} ranks: every rank's tokens, taps and picks bit for bit rank 0's {}",
            verdict(alike)
        );
        let (worst, logits, ties) = against(first, n1, embd, bands.head);
        let mut held = alike;
        for (l, &e) in worst.iter().enumerate() {
            let pass = e <= bands.stream[l];
            held &= pass;
            println!(
                "({n}) layer {l} l_out_rel {e:.3e} against one rank, of the stream band {:.3e} {}",
                bands.stream[l],
                verdict(pass)
            );
        }
        let flips = first
            .routes
            .iter()
            .zip(&n1.routes)
            .flat_map(|(a, b)| a.iter().zip(b))
            .filter(|(a, b)| a.as_ref().map(chosen) != b.as_ref().map(chosen))
            .count();
        let head = logits <= bands.head && ties;
        held &= head;
        println!(
            "({n}) logits rel {logits:.3e} against one rank, of the head band {:.3e}, every argmax \
             a tie at it {ties}; routed cells picked otherwise than one rank's {flips} {}",
            bands.head,
            verdict(head)
        );
        held
    }

    /// A routed layer's chosen experts as a set, whatever their slot order.
    fn chosen(r: &RoutePick) -> Vec<u32> {
        let mut ids = r.ids.clone();
        ids.sort_unstable();
        ids
    }

    /// [`REFUSED`] ranks: refused at the open by name, naming the tensor and the count.
    fn refused_clause(n: usize, e: &GateError) -> bool {
        let says = e.to_string();
        let named = says.contains("blk.") && says.contains(&format!("into {n} ranks"));
        println!(
            "({n}) {n} ranks: the open refused \"{says}\" (a part of other than whole units or \
             format blocks; the loopback runs no step at this count) {}",
            verdict(named)
        );
        named
    }

    /// (st) (module doc), stepping `token`: each refusal names what it refuses.
    fn state_clause(file: &Arc<Split>, token: u32) -> Result<bool, GateError> {
        let says = |r: Result<(), GpuError>, want: &str| match r {
            Err(e) => {
                let e = e.to_string();
                (e.contains(want), e)
            }
            Ok(()) => (false, "accepted".to_string()),
        };
        let taken = |r: Result<(), GpuError>| match r {
            Ok(()) => (true, "accepted".to_string()),
            Err(e) => (false, e.to_string()),
        };
        let gpu = Gpu::new()?;
        let at = RankOf::new(0, 1)?;
        let open = |ctx| Rank::open(&gpu, Arc::clone(file), at, ctx);
        let over = usize::try_from(ORACLE_POSITIONS)? + 1;
        let mut held = vec![
            (
                "an open of no positions",
                says(open(0).map(drop), "ctx 0: at least 1"),
            ),
            (
                "an open past the reference sets' positions",
                says(open(over).map(drop), &format!("ctx {over}: at least 1")),
            ),
        ];
        let mut r = open(CTX)?;
        let step = |r: &mut Rank, pos: u32| tp::step(&mut [(&gpu, r)], None, token, pos).map(drop);
        let late = "taps armed before the first step";
        held.push(("tap arming on a fresh open", taken(r.set_taps(&gpu, true))));
        held.push((
            "logits before a step",
            says(r.logits(&gpu).map(drop), "a completed step"),
        ));
        held.push((
            "a first step at position 1",
            says(
                step(&mut r, 1),
                "position 1, where the stores hold 0 positions",
            ),
        ));
        let end = u32::try_from(CTX)?;
        held.push((
            "a first step at the stores' end",
            says(
                step(&mut r, end),
                &format!("position {end} in stores of {CTX}"),
            ),
        ));
        held.push(("the step at 0", taken(step(&mut r, 0))));
        held.push((
            "position 0 after its step",
            says(
                step(&mut r, 0),
                "position 0, where the stores hold 1 positions",
            ),
        ));
        held.push((
            "logits after the step at 0",
            taken(r.logits(&gpu).map(drop)),
        ));
        held.push(("the step at 1", taken(step(&mut r, 1))));
        let traced = r.taps(&gpu)?;
        held.push((
            "tap arming after a step",
            says(r.set_taps(&gpu, true), late),
        ));
        held.push((
            "the step's taps after the refused arming",
            match r.taps(&gpu) {
                Ok(t) => {
                    let kept = same_bits(&t, &traced);
                    (
                        kept,
                        format!("{} values, bit for bit the read before it: {kept}", t.len()),
                    )
                }
                Err(e) => (false, e.to_string()),
            },
        ));
        held.push(("disarming after a step", taken(r.set_taps(&gpu, false))));
        held.push((
            "taps once disarmed",
            says(r.taps(&gpu).map(drop), "armed taps"),
        ));
        held.push((
            "tap arming once disarmed",
            says(r.set_taps(&gpu, true), late),
        ));
        let other = Gpu::new()?;
        let gpu_ok = gpu_clause(&mut r, &other, 2);
        held.push((
            "a token past the vocabulary",
            says(
                tp::step(&mut [(&gpu, &mut r)], None, u32::MAX, 2).map(drop),
                "token",
            ),
        ));
        held.push(("the step after refusals", taken(step(&mut r, 2))));
        let mut shared = (0..2)
            .map(|i| Rank::open(&gpu, Arc::clone(file), RankOf::new(i, 2)?, CTX))
            .collect::<Result<Vec<_>, GpuError>>()?;
        let foreign = join(&shared)?;
        let mut pairs: Vec<_> = shared.iter_mut().map(|r| (&gpu, r)).collect();
        held.push((
            "two ranks on one GPU",
            says(
                tp::step(&mut pairs, foreign.as_ref(), token, 0).map(drop),
                "each rank must use a distinct stream",
            ),
        ));
        if let Some(link) = &foreign {
            link.release()?;
        }
        drop(pairs);
        let mut separate = vec![
            Rank::open(&gpu, Arc::clone(file), RankOf::new(0, 2)?, CTX)?,
            Rank::open(&other, Arc::clone(file), RankOf::new(1, 2)?, CTX)?,
        ];
        let mut pairs: Vec<_> = [&gpu, &other]
            .into_iter()
            .zip(separate.iter_mut())
            .collect();
        pairs.swap(0, 1);
        held.push((
            "ranks out of order",
            says(
                tp::step(&mut pairs, foreign.as_ref(), token, 0).map(drop),
                "at place 0",
            ),
        ));
        pairs.swap(0, 1);
        held.push((
            "a rank of another count",
            says(
                tp::step(&mut pairs[..1], None, token, 0).map(drop),
                "step over 1",
            ),
        ));
        held.push((
            "unjoined ranks",
            says(
                tp::step(&mut pairs, None, token, 0).map(drop),
                "needs its joined link",
            ),
        ));
        drop(pairs);
        let ring = join(&separate)?;
        let link = ring.as_ref().ok_or("two ranks without a link")?;
        let (layout, _) = separate[0].link().ok_or("no gather page")?;
        let short = Arc::new(MappedHost::new(
            gpu.context(),
            layout.page_bytes() - 4,
            "short gather page",
        )?);
        let short_pages = [
            short,
            Arc::clone(separate[1].link().ok_or("no gather page")?.1),
        ];
        let layouts = vec![
            layout.clone(),
            separate[1].link().ok_or("no gather page")?.0.clone(),
        ];
        held.push((
            "a short gather page",
            says(
                GatherLoop::new(&layouts, &short_pages, &PostCols::default()).map(drop),
                "page has",
            ),
        ));
        let mut pairs: Vec<_> = [&gpu, &other]
            .into_iter()
            .zip(separate.iter_mut())
            .collect();
        held.push((
            "a foreign link",
            says(
                tp::step(&mut pairs, foreign.as_ref(), token, 0).map(drop),
                "page is not joined",
            ),
        ));
        link.release()?;
        held.push((
            "a released link",
            says(
                tp::step(&mut pairs, Some(link), token, 0).map(drop),
                "link was released",
            ),
        ));
        drop(pairs);
        let mut ok = gpu_ok;
        for (what, (pass, said)) in held {
            println!("(st) one rank: {what}: {said} {}", verdict(pass));
            ok &= pass;
        }
        Ok(ok)
    }

    /// A wrong GPU is refused before token decoding, state reads or tap changes.
    fn gpu_clause(r: &mut Rank, other: &Gpu, pos: u32) -> bool {
        let mut ok = true;
        for (what, result) in [
            (
                "step",
                tp::step(&mut [(other, &mut *r)], None, u32::MAX, pos).map(drop),
            ),
            ("set_taps", r.set_taps(other, false)),
            ("taps", r.taps(other).map(drop)),
            ("routes", r.routes(other).map(drop)),
            ("logits", r.logits(other).map(drop)),
        ] {
            let said = match result {
                Err(e) => e.to_string(),
                Ok(()) => "accepted".into(),
            };
            let pass = said.contains("use the GPU that opened it");
            println!("(st) {what} on a different GPU: {said} {}", verdict(pass));
            ok &= pass;
        }
        ok
    }

    #[test]
    #[ignore = "needs an NVIDIA card and the GLM fixture"]
    fn hw_gpu_inputs_use_the_rank_opener() -> Result<(), GateError> {
        crate::gate_card::init()?;
        glm5next_tier::init_file()?;
        let file = Arc::new(glm5next_tier::open()?);
        let gpu = Gpu::new()?;
        let other = Gpu::new()?;
        let mut r = Rank::open(&gpu, Arc::clone(&file), RankOf::new(0, 1)?, CTX)?;
        assert!(gpu_clause(&mut r, &other, 0));
        let mut ranks = vec![
            Rank::open(&gpu, Arc::clone(&file), RankOf::new(0, 2)?, CTX)?,
            Rank::open(&other, file, RankOf::new(1, 2)?, CTX)?,
        ];
        let ring = join(&ranks)?;
        let layouts: Vec<_> = ranks
            .iter()
            .map(|r| r.link().expect("a rank of two has a page").0.clone())
            .collect();
        let pages = [&gpu, &other]
            .into_iter()
            .zip(&layouts)
            .map(|(g, l)| {
                MappedHost::new(g.context(), l.page_bytes(), "foreign gather page").map(Arc::new)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let cols = PostCols::default();
        cols.set(1);
        let foreign = GatherLoop::new(&layouts, &pages, &cols)?;
        let mut pairs: Vec<_> = [&gpu, &other].into_iter().zip(ranks.iter_mut()).collect();
        for (link, want) in [
            (None, "needs its joined link"),
            (Some(&foreign), "page is not joined"),
        ] {
            let e = tp::step(&mut pairs, link, u32::MAX, 0).unwrap_err();
            println!("(st) link preflight: {e}");
            assert!(e.to_string().contains(want), "{e}");
        }
        let link = ring.as_ref().ok_or("two ranks without a link")?;
        link.release()?;
        let e = tp::step(&mut pairs, Some(link), u32::MAX, 0).unwrap_err();
        println!("(st) link preflight: {e}");
        assert!(e.to_string().contains("link was released"), "{e}");
        Ok(())
    }

    /// Two runs' tokens, logits, taps and picks bit for bit.
    fn same_run(a: &Run, b: &Run) -> bool {
        a.tokens == b.tokens
            && a.routes == b.routes
            && a.logits.len() == b.logits.len()
            && a.logits.iter().zip(&b.logits).all(|(x, y)| same_bits(x, y))
            && a.taps.len() == b.taps.len()
            && a.taps.iter().zip(&b.taps).all(|(x, y)| same_bits(x, y))
    }

    /// (fx) (module doc) for the run of `n` ranks.
    fn fx_clause(
        man: &bloomery_gpu_gates::RefManifest,
        file: &Split,
        n: usize,
        run: &Run,
        bands: &FxBands,
        embd: usize,
    ) -> Result<bool, GateError> {
        let layers = bands.stream.len();
        let table = layer_table(man, &run.taps, STREAMS * embd, layers)?;
        let mut flips = Vec::new();
        let mut logits_at = Vec::new();
        for l in 0..layers {
            let routed = run.routes.first().and_then(|r| r.get(l));
            if !routed.is_some_and(Option::is_some) {
                continue;
            }
            let ik = IkRoute::read(man, l)?;
            if ik.tokens() != run.routes.len() {
                return Err(format!(
                    "layer {l}: ik routes {} tokens, the run {}",
                    ik.tokens(),
                    run.routes.len()
                )
                .into());
            }
            let z = crate::e2e::tap(man, &format!("ffn_moe_logits-{l}"))?;
            let bias = bloomery_gpu_gates::split_f32(file, &names::exp_probs_b(l), N_EXPERT)?;
            for (t, routes) in run.routes.iter().enumerate() {
                let ours = routes[l].as_ref().ok_or("a routed layer with no picks")?;
                let ranked: Vec<f32> = ours.probs.iter().zip(&bias).map(|(p, b)| p + b).collect();
                let (iv, ids) = ik.at(t);
                if let Some(f) =
                    Flip::between((l, t), (&ours.ids, &ranked), (ids, iv), ik.margin(t))
                {
                    flips.push(f);
                    logits_at.push(z[t * N_EXPERT..(t + 1) * N_EXPERT].to_vec());
                }
            }
        }
        let mut first_ok = true;
        for (f, z) in flips.iter().zip(&logits_at) {
            let cap = fx_route_cap(bands.router[f.layer], z);
            let first = f.layer == 0 || !on_path(&flips, f.layer - 1, f.token, false);
            if first {
                first_ok &= f.allowed(cap);
            }
            println!(
                "{} ({})",
                f.line(&format!("(fx) {n} ranks"), cap),
                if first {
                    "first on its path: held"
                } else {
                    "past an earlier flip: printed"
                }
            );
        }
        let mut off_ok = true;
        let mut exempt = 0usize;
        for (l, row) in table.iter().enumerate() {
            for (t, e) in row.iter().enumerate() {
                let Some(e) = *e else { continue };
                if on_path(&flips, l, t, false) {
                    exempt += 1;
                } else {
                    off_ok &= e <= bands.stream[l];
                }
            }
            let cells: Vec<String> = row
                .iter()
                .map(|e| e.map_or("-".into(), |e| format!("{e:.3e}")))
                .collect();
            println!(
                "(fx) {n} ranks layer={l} l_out_rel by position {} (stream band {:.3e})",
                cells.join(" "),
                bands.stream[l]
            );
        }
        let ik_last = crate::e2e::ik_last(man, run.logits.last().map_or(0, Vec::len))?;
        let ours = run.logits.last().ok_or("no logits")?;
        let tie = flip::head_tie(&ik_last, argmax(ours), bands.head);
        let pass = first_ok && off_ok && tie;
        println!(
            "(fx) {n} ranks: {} flips, the first on their paths within the pair rule {first_ok}; \
             every output off a flip's path within its stream band {off_ok} ({exempt} on a path); \
             the last argmax {} a tie with ik's at the head band {tie} {}",
            flips.len(),
            argmax(ours),
            verdict(pass)
        );
        Ok(pass)
    }
}
