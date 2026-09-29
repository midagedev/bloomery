//! GPU gate for V4.1's host streaming in a prompt call (`body::prefill`,
//! `BLOOMERY_HOSTSTREAM=on`) on the real model: plan (b′) on both cards
//! (`--place bp`), the residency machine over the stage card's routed stacks
//! at `mid-p40-s1`, groups of [`GROUP`] batches, streaming on at the load
//! (set here; the lever itself is refused, so the environment cannot move
//! it). One load. Each clause starts from a clear (the residency back to its
//! seed) and names its mutant:
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
//!   its ids and its start, not of when the copies land): [`S1_P`] prose ids
//!   in groups past the first, then [`S1_STEPS`] greedy steps, twice — free,
//!   and with the machine's copy stream held by a host flag for [`HOLD`] from
//!   before the call, so every pick's copies land late. The two give the
//!   same picks per (group, layer) (admitted, kept, bytes), the same call
//!   argmax and logits and the same steps' tokens and logits; the free call
//!   admitted at least one expert and picked in a group past the first
//!   (mutant: the engine stream does not wait for a pick's copies before the
//!   streamed pass).
//! - `s3` (end to end against `off`): on [`S3_WINDOWS`] windows of [`S3_P`]
//!   ids each of the prose and the code corpus, the call's argmax and
//!   [`S3_STEPS`] greedy steps with streaming off and on, each from a clear;
//!   where the two first differ, the `on` arm's top-1 margin is below
//!   [`GREEDY_MARGIN`] (a near tie, the long gate's rule), and every `on`
//!   call admitted at least one expert (mutant: the pick's admitted experts
//!   left out of the streamed places).
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
mod gate {
    use crate::ds41_open::{fnv, open};
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
    use bloomery_levers::{CARD_DONTNEED, ENGRAM_HELPER, HOST_LOCK, HOST_POPULATE, R8};
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
    /// `s1`'s prompt: three batches, two groups, so a layer picks twice.
    const S1_P: usize = 1536;
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
    /// bytes).
    type Pick = (usize, usize, usize, usize, u64);

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
                .map(|(g, p)| (*g, p.layer, p.admitted, p.kept, p.bytes))
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

    /// `s1`: the call and its steps, free and with the copy stream held.
    fn s1(s: &mut Session<Body>) -> Result<bool, GateError> {
        let ids = corpus("prose", 0, S1_P)?;
        s.clear()?;
        let free = call(s, &ids, S1_STEPS)?;
        s.clear()?;
        let flags = HostFlags::new(s.model().gpu().context(), 1)?;
        flags.clear(0)?;
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
                // A raise that fails leaves the stream held: the machine's
                // deadline names it.
                let _ = flags.raise(0);
            });
            call(s, &ids, S1_STEPS)
        })?;
        let held_ms = t.elapsed().as_secs_f64() * 1e3;
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

    /// `s3`: each window with streaming off, then on.
    fn s3(s: &mut Session<Body>) -> Result<bool, GateError> {
        let mut ok = true;
        let (mut differ, mut idle) = (0, 0);
        for name in ["prose", "code"] {
            for w in 0..S3_WINDOWS {
                let ids = corpus(name, w * S3_P, S3_P)?;
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
                    "s3 {name} window {w}: on admitted {} over {} picks; {line}",
                    on.admitted(),
                    on.picks.len()
                );
            }
        }
        ok &= idle == 0;
        println!(
            "s3: {} windows, {differ} differ from off within the call and {S3_STEPS} steps, {idle} \
             calls admitted nothing; every difference a near tie below {GREEDY_MARGIN}: {}",
            2 * S3_WINDOWS,
            verdict(ok)
        );
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[
            ENGRAM_HELPER,
            HOST_POPULATE,
            HOST_LOCK,
            CARD_DONTNEED,
            R8,
        ])?;
        let mut cfg = OpenCfg::from_levers(&levers)?;
        cfg.body.residency = RESIDENCY;
        cfg.body.prefill = PrefillMode::Batch;
        cfg.body.group = GROUP;
        cfg.body.hoststream = true;
        let path = ref_model_path()?;
        let inputs = PlanInputs::read(
            &Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?,
        )?;
        let bp = Place::Bp.machine(None, Some(place::tier_batch(&inputs.hp)))?;
        let t0 = Instant::now();
        let mut s = open(&path, bp, &cfg)?;
        println!("load in {:.1} s", t0.elapsed().as_secs_f64());
        let mut pass = s2(&mut s)?;
        pass &= s1(&mut s)?;
        pass &= s3(&mut s)?;
        if pass {
            println!("{NAME}: every clause passed");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
