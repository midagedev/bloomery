//! The relay of a mirrored link ([`super::Link`]): what one end's pump does
//! on its own page, each pass.
//!
//! A counter of the tier protocol has two writers on the shared page, at
//! different times: the stage card adds one to a row's go and the tier takes
//! it back, and the tier adds one to a row's done and the stage card takes it
//! back. Across two pages a copy of the writer's absolute value would erase
//! the reader's take back. So the counter crosses as a generation: on the
//! writer's page only the writing card adds to it, the writer's end posts each
//! new value, data first, to the inbox word on the reader's page, and the
//! reader's end adds one to the counter on the reader's page for each new
//! generation it finds there ([`Relay`]). The reading card's wait for at least
//! one and its take back are the shared page's, so a captured wait keeps its
//! one fixed target on every replay.
//!
//! An end ([`Half`]) holds one page: it posts the crossings whose reader is
//! the other side and delivers the crossings it reads. It reaches the other
//! page only through its [`Wire`], which carries a post's byte ranges in order:
//! each span the counter publishes, then for a done the tier's progress, then
//! the generation into the inbox. The progress the host reads is the writer's
//! dones posted, so it never runs ahead of the bytes posted with them.
//!
//! The inbox and two staging words per crossing (the generation and the
//! progress a post sends) are the link's tail, past the tier page's own
//! bytes ([`TAIL_LINE`]).

use std::ops::Range;
use std::sync::atomic::{AtomicU32, Ordering};

/// A page of 4-byte words an end reads and writes by byte offset.
pub(super) trait Page: Send + Sync {
    /// The u32 at byte `off`, which the caller keeps 4-aligned and inside the
    /// page.
    fn word(&self, off: usize) -> &AtomicU32;
}

/// One byte range a post carries: `len` bytes at byte `from` of this end's
/// page to byte `to` of the other end's, each 4-aligned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Move {
    pub from: usize,
    pub to: usize,
    pub len: usize,
}

/// How an end's posts reach the other end's page.
pub(super) trait Wire: Send {
    /// Carry `moves` in order, the last landing after every one before it.
    fn post(&mut self, moves: &[Move]) -> Result<(), String>;

    /// Take what the wire finished: `Ok(true)` when no post is in flight, or
    /// a failed post by name.
    fn reap(&mut self) -> Result<bool, String>;
}

/// Bytes of the link's tail per crossing: the inbox word, then the staged
/// generation and the staged progress.
pub(super) const TAIL_LINE: usize = 16;

/// The side a counter's reader sits on, and the side an end holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reader {
    /// The tier card reads it: a row's go.
    Tier,
    /// The stage card reads it: a row's done.
    Stage,
}

/// One counter that crosses the link, as the tier page lays it out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Crossing {
    /// The counter's byte offset: on the writer's page the generation the
    /// writing card adds one to, on the reader's the counter the reader's end
    /// adds one to and the reading card's wait takes back.
    pub counter: usize,
    pub reader: Reader,
    /// Byte spans the writing card fills before it adds to the counter,
    /// carried to the reader's page before each generation.
    pub spans: Vec<Range<usize>>,
    /// The progress word on the reader's page that counts this counter's
    /// generations with every other done of the same tier; `None` for a go.
    pub prog: Option<usize>,
}

/// The reader's side of one counter: the generation it last delivered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Relay {
    seen: u32,
}

impl Relay {
    /// A relay that has delivered every generation up to `inbox`.
    pub(super) fn at(inbox: u32) -> Relay {
        Relay { seen: inbox }
    }

    /// What the inbox's value `now` delivers: `Ok(false)` nothing new (a
    /// repeated post included), `Ok(true)` one generation more, and the step
    /// as the error for anything else, since a row has at most one go or done
    /// in flight. Wrapping.
    pub(super) fn take(&mut self, now: u32) -> Result<bool, u32> {
        match now.wrapping_sub(self.seen) {
            0 => Ok(false),
            1 => {
                self.seen = now;
                Ok(true)
            }
            step => Err(step),
        }
    }

    /// Whether every generation up to `inbox` was delivered.
    pub(super) fn holds(self, inbox: u32) -> bool {
        self.seen == inbox
    }
}

/// One crossing as this end sees it.
struct Line {
    c: Crossing,
    /// Byte offset of the crossing's tail line, on both pages.
    tail: usize,
    /// Whether this end holds the writer's page.
    writes: bool,
    /// The index of the counter's progress word in [`Half::progs`].
    prog: Option<usize>,
    /// The writer's side: the last generation posted.
    posted: u32,
    /// The reader's side.
    relay: Relay,
    /// Set by a release: nothing is carried for this line until its reset.
    stopped: bool,
}

impl Line {
    fn inbox(&self) -> usize {
        self.tail
    }

    fn gen_slot(&self) -> usize {
        self.tail + 4
    }

    fn prog_slot(&self) -> usize {
        self.tail + 8
    }
}

/// A tier's progress as the writer posts it: the writer's progress when the
/// count started, and the dones posted since.
struct Prog {
    off: usize,
    base: u32,
    dones: u32,
}

/// One end's relay over its own page.
pub(super) struct Half {
    lines: Vec<Line>,
    progs: Vec<Prog>,
    /// The first failure by name: a step that was neither 0 nor 1, or a
    /// post the wire refused or lost. It stays, and the end carries nothing
    /// more.
    broken: Option<String>,
}

impl Half {
    /// The end on the side `side` over `crossings`, whose tail lines start at
    /// byte `tail` of both pages, in crossing order. Every generation and
    /// inbox is taken as `page` holds it now, so nothing already there is
    /// delivered again. A crossing whose counter, progress word or span is
    /// not 4-aligned before the tail is refused by name.
    pub(super) fn new(
        crossings: Vec<Crossing>,
        side: Reader,
        tail: usize,
        page: &dyn Page,
    ) -> Result<Half, String> {
        let fits = |r: &Range<usize>| {
            r.start.is_multiple_of(4)
                && r.end.is_multiple_of(4)
                && r.start <= r.end
                && r.end <= tail
        };
        if let Some(c) = crossings.iter().find(|c| {
            !fits(&(c.counter..c.counter + 4))
                || c.prog.is_some_and(|p| !fits(&(p..p + 4)))
                || !c.spans.iter().all(fits)
        }) {
            return Err(format!(
                "the crossing {c:?} is not 4-aligned before the link's tail at byte {tail}"
            ));
        }
        let mut progs: Vec<Prog> = Vec::new();
        let lines = crossings
            .into_iter()
            .enumerate()
            .map(|(i, c)| {
                let writes = c.reader != side;
                let prog = c.prog.filter(|_| writes).map(|off| {
                    progs.iter().position(|p| p.off == off).unwrap_or_else(|| {
                        progs.push(Prog {
                            off,
                            base: 0,
                            dones: 0,
                        });
                        progs.len() - 1
                    })
                });
                Line {
                    c,
                    tail: tail + i * TAIL_LINE,
                    writes,
                    prog,
                    posted: 0,
                    relay: Relay::at(0),
                    stopped: false,
                }
            })
            .collect();
        let mut h = Half {
            lines,
            progs,
            broken: None,
        };
        for i in 0..h.lines.len() {
            let l = &mut h.lines[i];
            if l.writes {
                l.posted = page.word(l.c.counter).load(Ordering::Acquire);
                if let Some(p) = l.prog {
                    h.progs[p].base = page.word(h.progs[p].off).load(Ordering::Acquire);
                }
            } else {
                l.relay = Relay::at(page.word(l.inbox()).load(Ordering::Acquire));
            }
        }
        Ok(h)
    }

    /// One pass: post each new generation this end writes, deliver what each
    /// inbox it reads holds, then reap the wire. `true` when anything moved.
    pub(super) fn pass(&mut self, page: &dyn Page, wire: &mut dyn Wire) -> bool {
        let mut moved = false;
        for i in 0..self.lines.len() {
            if self.broken.is_some() {
                return false;
            }
            if self.lines[i].stopped {
                continue;
            }
            moved |= if self.lines[i].writes {
                self.post(i, page, wire)
            } else {
                self.deliver(i, page)
            };
        }
        if let Err(e) = wire.reap() {
            self.break_with(e);
        }
        moved
    }

    fn break_with(&mut self, why: String) {
        self.broken.get_or_insert(why);
    }

    /// The writer's side of line `i`: a generation one past the last posted
    /// goes to the reader's inbox after the bytes it publishes and, for a
    /// done, after the progress that counts it. The step is checked before
    /// any byte moves.
    fn post(&mut self, i: usize, page: &dyn Page, wire: &mut dyn Wire) -> bool {
        let line = &self.lines[i];
        let g = page.word(line.c.counter).load(Ordering::Acquire);
        match g.wrapping_sub(line.posted) {
            0 => return false,
            1 => {}
            step => {
                let why = format!(
                    "the tier page's counter at byte {} stepped by {step} between two posts: a row \
                     has at most one go or done in flight",
                    line.c.counter
                );
                self.break_with(why);
                return false;
            }
        }
        let mut moves: Vec<Move> = line
            .c
            .spans
            .iter()
            .map(|s| Move {
                from: s.start,
                to: s.start,
                len: s.len(),
            })
            .collect();
        page.word(line.gen_slot()).store(g, Ordering::Relaxed);
        if let Some(p) = line.prog {
            let p = &mut self.progs[p];
            p.dones = p.dones.wrapping_add(1);
            page.word(line.prog_slot())
                .store(p.base.wrapping_add(p.dones), Ordering::Relaxed);
            moves.push(Move {
                from: line.prog_slot(),
                to: p.off,
                len: 4,
            });
        }
        moves.push(Move {
            from: line.gen_slot(),
            to: line.inbox(),
            len: 4,
        });
        if let Err(e) = wire.post(&moves) {
            self.break_with(e);
            return false;
        }
        self.lines[i].posted = g;
        true
    }

    /// The reader's side of line `i`: one more generation in the inbox adds
    /// one to the counter; any other step breaks the end by name.
    fn deliver(&mut self, i: usize, page: &dyn Page) -> bool {
        let line = &mut self.lines[i];
        match line
            .relay
            .take(page.word(line.inbox()).load(Ordering::Acquire))
        {
            Ok(false) => false,
            Ok(true) => {
                page.word(line.c.counter).fetch_add(1, Ordering::Release);
                true
            }
            Err(step) => {
                let why = format!(
                    "the tier page's counter at byte {} stepped by {step} between two relays: a \
                     row has at most one go or done in flight",
                    line.c.counter
                );
                self.break_with(why);
                false
            }
        }
    }

    /// The line of the counter at byte `off`; every counter the tier
    /// protocol names crosses.
    fn line(&self, off: usize) -> usize {
        self.lines
            .iter()
            .position(|l| l.c.counter == off)
            .expect("every counter of the tier page crosses the link")
    }

    /// Stop carrying the counter at byte `off` until its reset; on the end
    /// that reads it, set its counter to `v`.
    pub(super) fn release(&mut self, off: usize, v: u32, page: &dyn Page) {
        let i = self.line(off);
        self.lines[i].stopped = true;
        if !self.lines[i].writes {
            page.word(off).store(v, Ordering::Release);
        }
    }

    /// Stop carrying the counter at byte `off` until its reset.
    pub(super) fn stop(&mut self, off: usize) {
        let i = self.line(off);
        self.lines[i].stopped = true;
    }

    /// Whether this end's side of the counter at byte `off` is at rest after
    /// one more pass: the writer's every generation posted and nothing in
    /// flight, or the reader's counter at 0 and its inbox delivered; and the
    /// end not broken.
    pub(super) fn at_rest(&mut self, off: usize, page: &dyn Page, wire: &mut dyn Wire) -> bool {
        self.pass(page, wire);
        let idle = match wire.reap() {
            Ok(idle) => idle,
            Err(e) => {
                self.break_with(e);
                false
            }
        };
        let line = &self.lines[self.line(off)];
        let side = if line.writes {
            page.word(off).load(Ordering::Acquire) == line.posted && idle
        } else {
            page.word(off).load(Ordering::Acquire) == 0
                && line
                    .relay
                    .holds(page.word(line.inbox()).load(Ordering::Acquire))
        };
        self.broken.is_none() && side
    }

    /// The writer's side of the counter at byte `off` back to rest once both
    /// cards' streams have drained: its generation taken as posted, its
    /// tier's progress restarted from the writer's page, and both sent to the
    /// reader's page through the wire, which is drained before and after.
    /// `spin_until` bounds each drain; a drain past it breaks the end.
    /// Nothing on the end that reads the counter.
    pub(super) fn reset_writer(
        &mut self,
        off: usize,
        page: &dyn Page,
        wire: &mut dyn Wire,
        spin_until: &dyn Fn() -> bool,
    ) {
        let i = self.line(off);
        if !self.lines[i].writes || self.broken.is_some() || !self.drain(wire, spin_until) {
            return;
        }
        let line = &mut self.lines[i];
        let g = page.word(off).load(Ordering::Acquire);
        line.posted = g;
        page.word(line.gen_slot()).store(g, Ordering::Relaxed);
        let mut moves = Vec::with_capacity(2);
        if let Some(p) = line.prog {
            let p = &mut self.progs[p];
            p.base = page.word(p.off).load(Ordering::Acquire);
            p.dones = 0;
            page.word(line.prog_slot()).store(p.base, Ordering::Relaxed);
            moves.push(Move {
                from: line.prog_slot(),
                to: p.off,
                len: 4,
            });
        }
        moves.push(Move {
            from: line.gen_slot(),
            to: line.inbox(),
            len: 4,
        });
        if let Err(e) = wire.post(&moves) {
            self.break_with(e);
            return;
        }
        if self.drain(wire, spin_until) {
            self.lines[i].stopped = false;
        }
    }

    /// The reader's side of the counter at byte `off` back to rest once the
    /// writer's reset landed: every generation in the inbox taken as
    /// delivered and the counter at 0. Nothing on the end that writes it.
    pub(super) fn reset_reader(&mut self, off: usize, page: &dyn Page) {
        let i = self.line(off);
        if self.lines[i].writes {
            return;
        }
        let line = &mut self.lines[i];
        line.relay = Relay::at(page.word(line.inbox()).load(Ordering::Acquire));
        page.word(off).store(0, Ordering::Release);
        line.stopped = false;
    }

    /// Reap until nothing is in flight, while `spin_until` holds; `false`
    /// with the end broken otherwise.
    fn drain(&mut self, wire: &mut dyn Wire, spin_until: &dyn Fn() -> bool) -> bool {
        loop {
            match wire.reap() {
                Ok(true) => return true,
                Ok(false) if spin_until() => std::hint::spin_loop(),
                Ok(false) => {
                    self.break_with("the link's posts did not finish before a reset".into());
                    return false;
                }
                Err(e) => {
                    self.break_with(e);
                    return false;
                }
            }
        }
    }

    /// The broken end's description, if it broke.
    pub(super) fn broken(&self) -> Option<&str> {
        self.broken.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};

    use super::super::{End, Memory, Mirror, Pump};
    use super::{Crossing, Half, Move, Page, Reader, Relay, TAIL_LINE, Wire};
    use crate::host::step::RELEASE;

    struct Mem(Vec<AtomicU32>);

    impl Page for Mem {
        fn word(&self, off: usize) -> &AtomicU32 {
            &self.0[off / 4]
        }
    }

    /// A wire whose every post fails by name.
    struct Refuse;

    impl Wire for Refuse {
        fn post(&mut self, _: &[Move]) -> Result<(), String> {
            Err("the test wire failed a post".into())
        }

        fn reap(&mut self) -> Result<bool, String> {
            Ok(true)
        }
    }

    #[test]
    fn a_relay_delivers_each_generation_once_across_the_wrap() {
        let mut r = Relay::at(u32::MAX - 1);
        assert_eq!(r.take(u32::MAX - 1), Ok(false));
        assert_eq!(r.take(u32::MAX), Ok(true));
        assert_eq!(r.take(u32::MAX), Ok(false));
        assert_eq!(r.take(0), Ok(true));
        assert!(r.holds(0));
        assert_eq!(r.take(2), Err(2));
        assert_eq!(r.take(u32::MAX), Err(u32::MAX));
    }

    // Byte offsets of the model page: one row of one tier, two crossings.
    const GO: usize = 0;
    const CNT: usize = 64;
    const PROG: usize = 128;
    const IMAGE: usize = 192;
    const ROWS: usize = 256;
    const TAIL: usize = 320;
    const WORDS: usize = (TAIL + 2 * TAIL_LINE) / 4;

    fn page(seed: &[(usize, u32)]) -> Arc<Mem> {
        let m = Mem((0..WORDS).map(|_| AtomicU32::new(0)).collect());
        for &(off, v) in seed {
            m.word(off).store(v, Ordering::Relaxed);
        }
        Arc::new(m)
    }

    // The row's go, which publishes its image, and its done, which publishes
    // its rows and counts in the progress word.
    fn row() -> Vec<Crossing> {
        let (image, rows) = (IMAGE..IMAGE + 64, ROWS..ROWS + 64);
        vec![
            Crossing {
                counter: GO,
                reader: Reader::Tier,
                spans: vec![image],
                prog: None,
            },
            Crossing {
                counter: CNT,
                reader: Reader::Stage,
                spans: vec![rows],
                prog: Some(PROG),
            },
        ]
    }

    /// The loopback's wire from `from` into `to`.
    fn wire(from: &Arc<Mem>, to: &Arc<Mem>) -> Memory<Mem> {
        Memory {
            from: Arc::clone(from),
            to: Arc::clone(to),
        }
    }

    /// Mirrored pages as the link holds them, with no pump thread: the stage
    /// card's end over `stage`, posting through `stage_wire`, and the tier
    /// cards' end over `tier`, posting into `stage` on the loopback's wire.
    fn link(stage: &Arc<Mem>, tier: &Arc<Mem>, stage_wire: impl Wire + 'static) -> Mirror<Mem> {
        let half = |side, page: &Arc<Mem>| {
            Half::new(row(), side, TAIL, &**page).expect("the model row fits before the tail")
        };
        Mirror {
            pumps: Vec::new(),
            stage: Arc::new(End::new(
                half(Reader::Stage, stage),
                stage_wire,
                Arc::clone(stage),
            )),
            tier: Arc::new(End::new(
                half(Reader::Tier, tier),
                wire(tier, stage),
                Arc::clone(tier),
            )),
            run_on: None,
        }
    }

    fn wait_take(p: &Mem, off: usize, until: Instant) -> bool {
        while p.word(off).load(Ordering::Acquire) < 1 {
            if Instant::now() > until {
                return false;
            }
            std::hint::spin_loop();
        }
        p.word(off).fetch_sub(1, Ordering::AcqRel);
        true
    }

    // The model cards run the tier protocol's ops: add one to the counter
    // each writes, wait for at least one on the counter each reads and take
    // one back. Goes `ks` carry image k and dones rows k + ROWS_OF; the bytes
    // are checked for every go up to `checked`, the ones before a release.
    const ROWS_OF: u32 = 1 << 20;

    fn tier_card(
        tier: Arc<Mem>,
        ks: std::ops::RangeInclusive<u32>,
        checked: u32,
        until: Instant,
    ) -> JoinHandle<Result<(), String>> {
        std::thread::spawn(move || {
            for k in ks {
                if !wait_take(&tier, GO, until) {
                    return Err(format!("the tier's go {k} never came"));
                }
                let img = tier.word(IMAGE).load(Ordering::Relaxed);
                if k <= checked && img != k {
                    return Err(format!("the tier's go {k} carried the image of go {img}"));
                }
                tier.word(ROWS).store(k + ROWS_OF, Ordering::Relaxed);
                tier.word(CNT).fetch_add(1, Ordering::Release);
                tier.word(PROG).fetch_add(1, Ordering::Release);
            }
            Ok(())
        })
    }

    fn stage_card(
        stage: &Mem,
        ks: std::ops::RangeInclusive<u32>,
        checked: u32,
        until: Instant,
    ) -> Result<(), String> {
        for k in ks {
            stage.word(IMAGE).store(k, Ordering::Relaxed);
            stage.word(GO).fetch_add(1, Ordering::Release);
            if !wait_take(stage, CNT, until) {
                return Err(format!("the stage's done {k} never came"));
            }
            let rows = stage.word(ROWS).load(Ordering::Relaxed);
            if k <= checked && rows != k + ROWS_OF {
                return Err(format!("the stage's done {k} carried rows {rows:#x}"));
            }
        }
        Ok(())
    }

    // Two model cards and the link's two ends, each end on its own pump
    // thread with its own page, from generations near the wrap: every go
    // carries its image and every done its rows. Then the link's release, and
    // the cancelled graph runs K more layers on both cards with nothing
    // carried; once both drain, the link's reset of the go and of the done are
    // taken apart, the pumps running between them, and the next N goes and
    // dones are carried as before. Both counters end at rest and the progress
    // counts every done.
    #[test]
    fn the_ends_carry_every_go_and_done_once_with_its_bytes_across_a_recovery() {
        const N: u32 = 2000;
        const K: u32 = 3;
        let seed = u32::MAX - 700;
        let stage = page(&[(GO, seed), (PROG, seed), (TAIL + TAIL_LINE, seed)]);
        let tier = page(&[(CNT, seed), (PROG, seed), (TAIL, seed)]);
        let mut m = link(&stage, &tier, wire(&stage, &tier));
        m.pumps = [&m.stage, &m.tier]
            .map(|e| Pump::start(vec![Arc::clone(e)]).expect("a pump thread"))
            .into();
        let until = Instant::now() + Duration::from_secs(20);
        let run = || -> Result<(), String> {
            let card = tier_card(tier.clone(), 1..=N + K, N, until);
            stage_card(&stage, 1..=N, N, until)?;
            m.release(&[GO, CNT], RELEASE);
            stage_card(&stage, N + 1..=N + K, N, until)?;
            card.join().unwrap()?;
            m.reset(GO)?;
            // The pumps run between the two resets, as they do while the
            // host drains the next tier's stream.
            let window = Instant::now() + Duration::from_millis(100);
            while Instant::now() < window && m.broken().is_none() {
                std::thread::yield_now();
            }
            m.reset(CNT)?;
            let card = tier_card(tier.clone(), N + K + 1..=2 * N + K, 2 * N + K, until);
            stage_card(&stage, N + K + 1..=2 * N + K, 2 * N + K, until)?;
            card.join().unwrap()
        };
        let ran = run();
        m.pumps.clear();
        assert_eq!(ran, Ok(()), "the ends: {:?}", m.broken());
        assert_eq!(m.broken(), None);
        assert!(m.at_rest(GO) && m.at_rest(CNT));
        assert_eq!(
            stage.word(PROG).load(Ordering::Acquire),
            seed.wrapping_add(2 * N + K)
        );
    }

    // A release stops its own counter on both ends: what its writer adds
    // meanwhile is not carried. A reset of another counter leaves it stopped,
    // and its own reset takes what was added as carried, so the next
    // generation is delivered once and counted once. A repeated post of that
    // generation delivers nothing.
    #[test]
    fn a_reset_reopens_its_own_counter_alone() {
        let stage = page(&[]);
        let tier = page(&[]);
        let m = link(&stage, &tier, wire(&stage, &tier));
        m.release(&[GO, CNT], RELEASE);
        assert_eq!(stage.word(CNT).load(Ordering::Acquire), RELEASE);
        tier.word(CNT).store(2, Ordering::Release);
        tier.word(PROG).store(2, Ordering::Release);
        assert!(!m.pass());
        assert_eq!(m.reset(GO), Ok(()));
        m.pass();
        assert_eq!(m.broken(), None);
        assert_eq!(stage.word(CNT).load(Ordering::Acquire), RELEASE);
        assert!(!m.at_rest(CNT));
        assert_eq!(m.reset(CNT), Ok(()));
        assert!(m.at_rest(GO) && m.at_rest(CNT));
        assert_eq!(stage.word(PROG).load(Ordering::Acquire), 2);
        tier.word(CNT).store(3, Ordering::Release);
        tier.word(PROG).store(3, Ordering::Release);
        m.pass();
        m.pass();
        assert_eq!(m.broken(), None);
        assert_eq!(stage.word(CNT).load(Ordering::Acquire), 1);
        assert_eq!(stage.word(PROG).load(Ordering::Acquire), 3);
        let inbox = TAIL + TAIL_LINE;
        let again = Move {
            from: inbox + 4,
            to: inbox,
            len: 4,
        };
        wire(&tier, &stage).post(&[again]).unwrap();
        m.pass();
        assert_eq!(stage.word(CNT).load(Ordering::Acquire), 1);
    }

    // A done whose generation steps by two publishes nothing, not the rows it
    // carries (a raised fault copy among them), the progress, the generation
    // or the counter: the host sees the tier behind, not a done the stage
    // card never gets. Its end stops by name, and a reset fails by that name.
    // A reset whose post the wire fails stops its end by name too.
    #[test]
    fn a_doubled_done_publishes_nothing() {
        let stage = page(&[]);
        let tier = page(&[]);
        let m = link(&stage, &tier, Refuse);
        tier.word(ROWS).store(7, Ordering::Relaxed);
        tier.word(CNT).store(2, Ordering::Release);
        tier.word(PROG).store(2, Ordering::Release);
        m.pass();
        for off in [ROWS, PROG, CNT, TAIL + TAIL_LINE] {
            assert_eq!(stage.word(off).load(Ordering::Acquire), 0, "byte {off}");
        }
        assert!(m.broken().is_some_and(|b| b.contains("stepped by 2")));
        assert!(!m.at_rest(CNT));
        assert!(m.reset(CNT).is_err_and(|b| b.contains("stepped by 2")));
        assert!(m.reset(GO).is_err());
        assert!(
            m.stage
                .broken()
                .is_some_and(|b| b.contains("the test wire failed"))
        );
    }
}
