//! GPU gate for V4.1's host streaming in a prompt call (`body::prefill`,
//! `BLOOMERY_HOSTSTREAM=on`) on the real model: plan (b′) on both cards
//! (`--place bp`), the residency machine over the stage card's routed stacks
//! at `mid-p40-s1`, groups of [`GROUP`] batches, streaming on at the load
//! (set here; the lever itself is refused, so the environment cannot move
//! it), the host set locked (set here too, `BLOOMERY_HOST_LOCK` refused: a
//! pick refuses a victim the host would serve from the file, and another
//! process's reads reclaim populated pages). One load. The residency's own
//! clauses (`shared/ds41_residency.rs`) run first: its host-set refusal
//! before the load, the rest on the load before the clauses below, with
//! streaming off; `--only residency` stops after them. `--place a` loads plan
//! (a) on the A6000 alone (the serving plan, no tier card) instead, for the
//! residency's clauses only: it needs `--only residency`, and the clauses
//! below run under plan (b′). There the residency's `static` clause ends
//! after the teardown, on a load of its own. Each clause below
//! starts from a clear (the residency back to its seed), streaming on, and
//! names its mutant:
//!
//! - `s2` (DEMOTE: a streamed expert gives the bits a card-resident one
//!   does): call A, [`S2_P`] prose ids in one batch, streams — its pick
//!   admits experts in place of pool residents, their slots are read by the
//!   streamed pass after their copies. The serve seat's reset
//!   ([`runtime::Target::reset`]) keeps that placement, and call B on the
//!   same ids finds every expert it wants on the card already: it admits
//!   none, and its card sum reads only slots that were resident before the
//!   call. B's argmax and logits (FNV-1a 64) equal A's. Premises, each
//!   named: A admitted at least one expert; B admitted none; the card sets
//!   after B are those after A (mutant: the card sum launched before the
//!   streamed pass).
//! - `s1` (scripted replay: the placement a call reaches is a function of
//!   its ids and its start, not of when the copies land): [`S1_HALF`] prose
//!   ids then as many code ids, two groups, then [`S1_STEPS`] greedy steps,
//!   twice — free, and with the machine's copy stream held by a host flag
//!   for [`HOLD`] from before the call, so every pick's copies land late (a
//!   guard raises the flag on every path out of the clause). The two give
//!   the same picks per (group, layer) (admitted, kept, bytes, the counts'
//!   digest), the same call argmax and logits and the same steps' tokens and
//!   logits; the free call admitted at least one expert and admitted in a
//!   group past the first (mutant: `s4`'s, under which the call reads
//!   what the call before it left in the block buffers a group shares, and
//!   the two arms follow different calls). Both arms read a slot alike when
//!   the engine does not wait for a pick's copies, so that mutant is `s2`'s:
//!   its call A reads slots mid-copy and faults.
//! - `s3` (end to end against `off`): on [`S3_WINDOWS`] windows of [`S3_P`]
//!   ids each of the prose and the code corpus, the call's argmax and
//!   [`S3_STEPS`] greedy steps with streaming off and on, each from a clear;
//!   where the two first differ, the `on` arm's top-1 margin is below
//!   [`GREEDY_MARGIN`] (a near tie, the long gate's rule), and every `on`
//!   call admitted at least one expert (mutant: the pick's admitted experts
//!   left out of the streamed places).
//! - `s4` (a group of two streams what it runs off: a streamed pass that
//!   admits nothing is the plain pass, bit for bit): [`S4_P`] prose ids,
//!   two batches, one group of two, so the next batch's route is enqueued
//!   ahead of this one's host serve while the batches share the block's
//!   norm, routing and places. Call A streams and admits; the seat's reset
//!   keeps its placement; call B, the same ids streaming, admits none; after
//!   another reset call C, the same ids with streaming off, runs on the same
//!   placement. B's argmax and logits equal C's. Premises, each named: A
//!   admitted an expert; B admitted none; the card sets after C are those
//!   after B (mutant: the next batch's route enqueued between the pick and
//!   the streamed pass, which then reads that batch's routing and places;
//!   `s3`'s near-tie rule did not see it on four windows of this size).
//!
//! Each clause runs under a watchdog ([`watched`]): past [`CLAUSE_BOUND`] it
//! names the clause, waits [`CLAUSE_GRACE`] (a stack watch,
//! `BLOOMERY_GATE_STACKS`, dumps the threads in between) and ends the
//! process, so a wait with no bound is a named red, not the runner's bound.
//! So does the teardown, after a reset: a flip's copy waiting for the
//! staging of a job not yet due holds the machine's copy stream, and every
//! free of device or pinned memory (a `DeviceBuffer`'s drop synchronizes the
//! context) waits for it.
//!
//! `off` keeps a prompt call bit for bit the decode steps' (the prefill
//! gate); this gate judges `on`, whose bits are the band's.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_ds41_callstream: built without the `deepseek41` feature; see `just \
         gate-gpu-ds41-callstream`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_ds41_callstream", gate::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_open.rs"]
mod ds41_open;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_residency.rs"]
mod residency;

#[cfg(feature = "deepseek41")]
mod gate {
    use crate::ds41_open::{fnv, open};
    use crate::residency::{self, StaticProbe};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    use app::Session;
    use bloomery_gpu::HostFlags;
    use bloomery_gpu::host::slots::Slot;
    use bloomery_gpu::host::swap::{CallReport, Residency};
    use bloomery_gpu_deepseek41::body::{Body, OpenCfg, PrefillMode};
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::{
        GREEDY_MARGIN, GateError, checks_failed, data_dir, ref_model_path, verdict,
    };
    use bloomery_levers::{CARD_DONTNEED, ENGRAM_HELPER, HOST_POPULATE, R8};
    use gguf::Split;
    use model::arch::deepseek41::place::{self, PlanInputs};
    use runtime::{Out, Target, Want};

    const NAME: &str = "gate_ds41_callstream";
    /// The residency the gate runs: 40 pinned, one spare a layer.
    const RESIDENCY: Residency = Residency::Mid {
        pinned: 40,
        spares: 1,
    };
    /// Batches a prompt group holds.
    const GROUP: usize = 2;
    /// `s2`'s prompt: one batch, one group, one pick a layer.
    const S2_P: usize = 512;
    /// Half of `s1`'s prompt: prose ids, then as many code ids. A lone last
    /// batch joins the group before it, so a second group needs four
    /// batches: 2048 ids are four of 512, groups {0, 1} and {2, 3}, each
    /// picking from its first batch at a floor of 64 (`STREAM_FLOOR` a batch
    /// of the group). The first group's pick moves each pool toward what
    /// prose routes most; the second counts a code batch, whose hottest
    /// experts are others (why code windows are their own tables), so an
    /// expert off the card clears the floor and a pool resident's count in
    /// a later group [derived]. At 1536 prose ids (three batches, one group
    /// of three) no later group exists.
    const S1_HALF: usize = 1024;
    /// Greedy steps after `s1`'s call.
    const S1_STEPS: usize = 8;
    /// How long `s1` holds the copy stream: past the call's first picks, far
    /// inside the machine's 30 s deadline on every host wait.
    const HOLD: Duration = Duration::from_secs(1);
    /// `s3`'s windows per corpus, each its own ids.
    const S3_WINDOWS: usize = 4;
    /// `s3`'s prompt: one batch.
    const S3_P: usize = 512;
    /// Greedy steps after each of `s3`'s calls.
    const S3_STEPS: usize = 8;
    /// `s4`'s prompt: two batches, one group of two.
    const S4_P: usize = 1024;
    /// A clause's bound: several times the slowest clause's time in a clean
    /// run (each prints its own), so past it one waits on something that
    /// will not come.
    const CLAUSE_BOUND: Duration = Duration::from_secs(300);
    /// How long the watchdog waits after naming a clause past its bound
    /// before it ends the process.
    const CLAUSE_GRACE: Duration = Duration::from_secs(60);

    /// `f`, the clause `clause`, under a watchdog: past [`CLAUSE_BOUND`] it
    /// names the clause red and, [`CLAUSE_GRACE`] later, ends the process —
    /// a host blocked with no bound (inside the driver, say) never returns.
    fn watched<T>(
        clause: &'static str,
        f: impl FnOnce() -> Result<T, GateError>,
    ) -> Result<T, GateError> {
        let (done, watch) = mpsc::channel::<()>();
        let dog = std::thread::spawn(move || {
            if let Err(mpsc::RecvTimeoutError::Timeout) = watch.recv_timeout(CLAUSE_BOUND) {
                println!(
                    "{clause}: the clause did not finish in {CLAUSE_BOUND:?}: a wait with no \
                     bound: FAIL"
                );
                std::thread::sleep(CLAUSE_GRACE);
                std::process::abort();
            }
        });
        let t0 = Instant::now();
        let r = f();
        let _ = done.send(());
        dog.join()
            .map_err(|_| format!("{clause}: the watchdog thread panicked"))?;
        println!("{clause}: {:.1} s", t0.elapsed().as_secs_f64());
        r
    }

    /// The hold on the machine's copy stream: flag 0 of the flags a copy
    /// stream wait was enqueued on, raised by [`Hold::release`] and again on
    /// drop, so every path out of the clause — a FAIL, an early `?`, a
    /// panic — lets the stream go on.
    struct Hold<'a>(&'a HostFlags);

    impl Hold<'_> {
        fn release(&self) {
            if let Err(e) = self.0.raise(0) {
                eprintln!("{NAME}: the copy stream's hold was not released: {e}");
            }
        }
    }

    impl Drop for Hold<'_> {
        fn drop(&mut self) {
            self.release();
        }
    }

    /// Ids `skip .. skip + n` of `$BLOOMERY_DATA/engram/corpus-<name>.ids`.
    fn corpus(name: &str, skip: usize, n: usize) -> Result<Vec<u32>, GateError> {
        let path = data_dir().join("engram").join(format!("corpus-{name}.ids"));
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let ids = text
            .split_whitespace()
            .skip(skip)
            .take(n)
            .map(str::parse::<u32>)
            .collect::<Result<Vec<_>, _>>()?;
        if ids.len() < n {
            return Err(format!(
                "{}: {} ids from {skip}, the gate reads {n}",
                path.display(),
                ids.len()
            )
            .into());
        }
        Ok(ids)
    }

    /// The row's top-1 logit less its top-2; a non-finite logit is refused
    /// by name.
    fn margin(row: &[f32]) -> Result<f32, GateError> {
        let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for (i, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(format!("logit {i} is {v}").into());
            }
            if v > a {
                b = a;
                a = v;
            } else if v > b {
                b = v;
            }
        }
        Ok(a - b)
    }

    /// One position's output: its argmax, its logits' FNV and top-1 margin.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Pos {
        argmax: u32,
        fnv: u64,
        margin: f32,
    }

    fn pos_of(out: Out<'_>) -> Result<Pos, GateError> {
        match out {
            Out::Logits { argmax, row } => Ok(Pos {
                argmax,
                fnv: fnv(row),
                margin: margin(row)?,
            }),
            Out::Argmax(_) => Err("a call asked for its logits returned the argmax alone".into()),
        }
    }

    /// A pick's record without its times: (group, layer, admitted, kept,
    /// bytes, the counts' digest).
    type Pick = (usize, usize, usize, usize, u64, u64);

    /// A prompt call and the steps after it.
    struct Run {
        /// The call's last position, then each step's.
        out: Vec<Pos>,
        picks: Vec<Pick>,
        end: Option<CallReport>,
    }

    impl Run {
        fn admitted(&self) -> usize {
            self.picks.iter().map(|p| p.2).sum()
        }
    }

    /// The prompt call of `ids` from where the session stands, then `steps` greedy
    /// steps; the call's picks and end.
    fn call(s: &mut Session<Body>, ids: &[u32], steps: usize) -> Result<Run, GateError> {
        let first = pos_of(s.prompt(ids, Want::Logits)?)?;
        let (picks, end) = s.model_mut().body_parts(NAME)?.2.take_stream_records();
        let mut out = Vec::with_capacity(steps + 1);
        out.push(first);
        let mut next = first.argmax;
        for _ in 0..steps {
            let p = pos_of(s.step(next, Want::Logits)?)?;
            next = p.argmax;
            out.push(p);
        }
        if let Some(r) = &end {
            record::call_report(r).print();
        }
        Ok(Run {
            out,
            picks: picks
                .iter()
                .map(|(g, p)| (*g, p.layer, p.admitted, p.kept, p.bytes, p.counts))
                .collect(),
            end,
        })
    }

    /// Per layer of the host map, its card experts, ascending.
    fn card_sets(s: &Session<Body>) -> Result<Vec<Vec<u32>>, GateError> {
        let m = s.model();
        let map = m.body(NAME)?.hybrid().slots();
        Ok(map
            .layers()
            .map(|l| {
                (0..map.n_expert() as u32)
                    .filter(|&id| matches!(map.slot(l, id), Some(Slot::Card(_))))
                    .collect()
            })
            .collect())
    }

    fn set_stream(s: &mut Session<Body>, on: bool) -> Result<(), GateError> {
        s.model_mut().body_parts(NAME)?.2.set_hoststream(on)?;
        Ok(())
    }

    /// `s2`: a call that streams, then the same call on the placement it
    /// left.
    fn s2(s: &mut Session<Body>) -> Result<bool, GateError> {
        let ids = corpus("prose", 0, S2_P)?;
        s.clear()?;
        let a = call(s, &ids, 0)?;
        let after_a = card_sets(s)?;
        Target::reset(s)?;
        if s.pos() != 0 {
            return Err(format!("the seat's reset left the session at {}", s.pos()).into());
        }
        let b = call(s, &ids, 0)?;
        let after_b = card_sets(s)?;
        let premises = [
            ("A admitted an expert", a.admitted() > 0),
            ("B admitted none", b.admitted() == 0),
            ("the card sets after B are A's", after_a == after_b),
        ];
        let failed: Vec<&str> = premises
            .iter()
            .filter(|(_, ok)| !ok)
            .map(|(what, _)| *what)
            .collect();
        let same = a.out == b.out;
        let ok = failed.is_empty() && same;
        println!(
            "s2: call A admitted {} experts over {} picks, B {}; B's argmax {} and logits {:016x} \
             against A's {} {:016x}: same {same}{}: {}",
            a.admitted(),
            a.picks.len(),
            b.admitted(),
            b.out[0].argmax,
            b.out[0].fnv,
            a.out[0].argmax,
            a.out[0].fnv,
            if failed.is_empty() {
                String::new()
            } else {
                format!("; premises failed: {}", failed.join(", "))
            },
            verdict(ok)
        );
        Ok(ok)
    }

    /// `s1`: the call and its steps, free and with the copy stream held on
    /// flag 0 of `flags`.
    fn s1(s: &mut Session<Body>, flags: &HostFlags) -> Result<bool, GateError> {
        let mut ids = corpus("prose", 0, S1_HALF)?;
        ids.extend(corpus("code", 0, S1_HALF)?);
        s.clear()?;
        let free = call(s, &ids, S1_STEPS)?;
        s.clear()?;
        flags.clear(0)?;
        let hold = Hold(flags);
        {
            let m = s.model();
            let machine = m
                .body(NAME)?
                .hybrid()
                .swap()
                .ok_or("the load runs no residency machine")?;
            flags.enqueue_wait(machine.copy_stream(), 0)?;
        }
        let t = Instant::now();
        let held = std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(HOLD);
                hold.release();
            });
            call(s, &ids, S1_STEPS)
        });
        drop(hold);
        let held = held?;
        let held_ms = t.elapsed().as_secs_f64() * 1e3;
        if let Some(k) = (0..free.picks.len().max(held.picks.len()))
            .find(|&k| free.picks.get(k) != held.picks.get(k))
        {
            println!(
                "s1: first pick that differs, #{k} (group, layer, admitted, kept, bytes, \
                 counts): free {:x?}, held {:x?}",
                free.picks.get(k),
                held.picks.get(k)
            );
        }
        let later = free.picks.iter().any(|p| p.0 >= 1 && p.2 > 0);
        let kept = free.end.is_some_and(|r| r.kept && r.restored == 0);
        let ok = free.picks == held.picks
            && free.out == held.out
            && free.admitted() > 0
            && later
            && kept;
        println!(
            "s1: {} picks, {} experts admitted (a group past the first admitting {later}, the \
             placement kept {kept}); held {HOLD:?} ({held_ms:.0} ms): same picks {}, same call \
             and {S1_STEPS} steps {}: {}",
            free.picks.len(),
            free.admitted(),
            free.picks == held.picks,
            free.out == held.out,
            verdict(ok)
        );
        Ok(ok)
    }

    /// `s4`: a streaming call that admits, then the same call streaming on
    /// its placement and with streaming off.
    fn s4(s: &mut Session<Body>) -> Result<bool, GateError> {
        let ids = corpus("prose", 0, S4_P)?;
        s.clear()?;
        let a = call(s, &ids, 0)?;
        Target::reset(s)?;
        let b = call(s, &ids, 0)?;
        let after_b = card_sets(s)?;
        Target::reset(s)?;
        set_stream(s, false)?;
        let c = call(s, &ids, 0);
        set_stream(s, true)?;
        let c = c?;
        let after_c = card_sets(s)?;
        let premises = [
            ("A admitted an expert", a.admitted() > 0),
            ("B admitted none", b.admitted() == 0),
            ("the card sets after C are B's", after_b == after_c),
        ];
        let failed: Vec<&str> = premises
            .iter()
            .filter(|(_, ok)| !ok)
            .map(|(what, _)| *what)
            .collect();
        let same = b.out == c.out;
        let ok = failed.is_empty() && same;
        println!(
            "s4: {S4_P} ids in a group of two; call A admitted {} experts over {} picks, B {}; \
             B (streaming) argmax {} and logits {:016x} against C (off) {} {:016x}: same \
             {same}{}: {}",
            a.admitted(),
            a.picks.len(),
            b.admitted(),
            b.out[0].argmax,
            b.out[0].fnv,
            c.out[0].argmax,
            c.out[0].fnv,
            if failed.is_empty() {
                String::new()
            } else {
                format!("; premises failed: {}", failed.join(", "))
            },
            verdict(ok)
        );
        Ok(ok)
    }

    /// `s3`: [`S3_STEPS`] steps after each of `windows` windows of `p` ids of
    /// each corpus, with streaming off, then on, each from a clear.
    fn on_off(
        s: &mut Session<Body>,
        clause: &str,
        p: usize,
        windows: usize,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let (mut differ, mut idle) = (0, 0);
        for name in ["prose", "code"] {
            for w in 0..windows {
                let ids = corpus(name, w * p, p)?;
                set_stream(s, false)?;
                s.clear()?;
                let off = call(s, &ids, S3_STEPS)?;
                set_stream(s, true)?;
                s.clear()?;
                let on = call(s, &ids, S3_STEPS)?;
                if on.admitted() == 0 {
                    idle += 1;
                }
                let first = off
                    .out
                    .iter()
                    .zip(&on.out)
                    .position(|(a, b)| a.argmax != b.argmax);
                let line = match first {
                    None => "same tokens".to_string(),
                    Some(i) => {
                        differ += 1;
                        let near = on.out[i].margin < GREEDY_MARGIN;
                        ok &= near;
                        format!(
                            "first differs at {i} ({} off, {} on), on's margin {:.3} (off's {:.3}): \
                             {}",
                            off.out[i].argmax,
                            on.out[i].argmax,
                            on.out[i].margin,
                            off.out[i].margin,
                            if near { "a near tie" } else { "not a near tie" }
                        )
                    }
                };
                println!(
                    "{clause} {name} window {w}: on admitted {} over {} picks; {line}",
                    on.admitted(),
                    on.picks.len()
                );
            }
        }
        ok &= idle == 0;
        println!(
            "{clause}: {} windows of {p} ids, {differ} differ from off within the call and \
             {S3_STEPS} steps, {idle} calls admitted nothing; every difference a near tie below \
             {GREEDY_MARGIN}: {}",
            2 * windows,
            verdict(ok)
        );
        Ok(ok)
    }

    /// The placement the gate loads (`--place a|bp`, plan (b′) when not
    /// given) and whether the residency's clauses run alone (`--only
    /// residency`); `--place a` without it is refused by name.
    fn parse_args() -> Result<(Place, bool), GateError> {
        const USAGE: &str = "usage: gate_ds41_callstream [--place a|bp] [--only residency]";
        let args: Vec<String> = std::env::args().skip(1).collect();
        let (mut place, mut only) = (Place::Bp, false);
        let mut it = args.iter().map(String::as_str);
        while let Some(flag) = it.next() {
            match (flag, it.next()) {
                ("--place", Some("a")) => place = Place::A,
                ("--place", Some("bp")) => place = Place::Bp,
                ("--only", Some("residency")) => only = true,
                _ => return Err(format!("{USAGE}, not {args:?}").into()),
            }
        }
        if place == Place::A && !only {
            return Err(format!(
                "--place a runs the residency's clauses alone; the streaming clauses run under \
                 plan (b′): add --only residency ({USAGE})"
            )
            .into());
        }
        Ok((place, only))
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[ENGRAM_HELPER, HOST_POPULATE, CARD_DONTNEED, R8])?;
        let (at, only_residency) = parse_args()?;
        let mut cfg = OpenCfg::from_levers(&levers)?;
        cfg.body.residency = RESIDENCY;
        cfg.body.prefill = PrefillMode::Batch;
        cfg.body.group = GROUP;
        cfg.body.hoststream = true;
        // Every pick's victims must be host-resident, and populated pages
        // alone are reclaimed under another process's reads.
        cfg.body.host.lock = true;
        // Declared before the session, so freed after it: freeing mapped
        // host memory waits for every stream of the context, and the
        // machine's copy stream holds copies whose staging waits for the
        // next pass (a job not yet due) until the machine drops.
        let flags: HostFlags;
        let path = ref_model_path()?;
        let inputs = PlanInputs::read(
            &Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?,
        )?;
        // Plan (b′)'s tier reserves the prompt batch's bytes; plan (a) has no tier.
        let batch = (at == Place::Bp).then(|| place::tier_batch(&inputs.hp));
        let machine = at.machine(None, batch)?;
        let refused = residency::refuse_clause(&path, &inputs, &cfg, at, &machine)?;
        let t0 = Instant::now();
        let mut s = open(&path, at, machine, &cfg)?;
        println!("load in {:.1} s", t0.elapsed().as_secs_f64());
        flags = HostFlags::new(s.model().gpu().context(), 1)?;
        let clauses = |s: &mut Session<Body>| -> Result<(bool, Option<StaticProbe>), GateError> {
            let (residency_ok, probe) = watched("residency", || residency::clauses(s, &flags, at))?;
            let mut pass = refused && residency_ok;
            if only_residency {
                return Ok((pass, probe));
            }
            set_stream(s, true)?;
            pass &= watched("s2", || s2(s))?;
            pass &= watched("s1", || s1(s, &flags))?;
            pass &= watched("s3", || on_off(s, "s3", S3_P, S3_WINDOWS))?;
            pass &= watched("s4", || s4(s))?;
            Ok((pass, probe))
        };
        let pass = clauses(&mut s);
        // Named before the teardown, so a failure there does not hide it.
        if let Err(e) = &pass {
            println!("{NAME}: a clause failed: {e}");
        }
        // Every free of device or pinned host memory waits for every stream
        // of the context, and the machine's copy stream can hold a copy that
        // waits for the staging of a flip not yet due, which only a pass or
        // a reset lets through: the reset stages and drains it before the
        // session frees anything.
        // Workaround: r3fix's stop-first Drop (HostTier::stop_swap) removes it.
        let down = watched("teardown", move || {
            s.clear()?;
            drop(s);
            Ok(())
        });
        // `static`'s own load, once the residency's is gone: two V4.1 loads
        // never stand at once.
        let pass = match (pass, &down) {
            (Ok((ok, Some(p))), Ok(())) => watched("static", || {
                residency::static_clause(&path, at, machine, &cfg, &p)
            })
            .map(|st| ok && st),
            (Ok((ok, _)), _) => Ok(ok),
            (Err(e), _) => Err(e),
        };
        match (pass, down) {
            (Ok(true), Ok(())) => {
                println!("{NAME}: every clause passed");
                Ok(())
            }
            (Ok(false), Ok(())) => Err(checks_failed()),
            (Err(e), Ok(())) => Err(e),
            (Ok(_), Err(t)) => Err(t),
            (Err(e), Err(t)) => Err(format!("{e}; then the teardown: {t}").into()),
        }
    }
}
