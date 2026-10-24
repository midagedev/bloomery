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
//! each span the counter publishes, then the generation into the inbox. The
//! reader's end counts a done in the tier's progress when it delivers it, so
//! the progress the host reads never runs ahead of a done the stage card can
//! take. An end delivers only while none of its posts is in flight, so a card
//! a delivery releases, and the line's next staged words, rewrite no range a
//! post made before the delivery still reads, but for the tier's fault copy
//! in a pass of two rows (`docs/contrib/dgx-spark.md` 9.5, item 2).
//!
//! A span may hold a row's columns one after another ([`Span`]): a post
//! carries only the columns the chain in flight uses ([`PostCols`]).
//!
//! The inbox and two staging words per crossing (the generation a post sends
//! and the progress a reset sends) are the link's tail, past the tier page's
//! own bytes ([`TAIL_LINE`]).

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// A page of 4-byte words an end reads and writes by byte offset.
pub(super) trait Page: Send + Sync {
    /// The page's byte length.
    fn bytes(&self) -> usize;

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

    /// Stop the wire for good: every post still in flight finishes, failed,
    /// at a later reap.
    fn abort(&mut self) -> Result<(), String>;
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
    pub spans: Vec<Span>,
    /// The progress word on the reader's page that the reader's end adds one
    /// to for each generation it delivers, with every other done of the same
    /// tier; `None` for a go.
    pub prog: Option<usize>,
}

/// A byte range a crossing publishes, laid out as `cols` equal columns one
/// after another: a post carries its first columns, as many as the end's
/// [`PostCols`] holds when it posts, and all of a span of one column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    pub bytes: Range<usize>,
    pub cols: usize,
}

impl Span {
    /// The span carried whole: one column.
    #[must_use]
    pub fn whole(bytes: Range<usize>) -> Span {
        Span { bytes, cols: 1 }
    }

    /// Bytes a post carries when its chain uses `cols` columns.
    #[must_use]
    pub fn bytes_at(&self, cols: usize) -> usize {
        self.bytes.len() / self.cols * cols.min(self.cols)
    }
}

/// The columns of a row the chain in flight uses, which every end over one
/// page reads when it posts ([`Span::bytes_at`]); every column until a chain
/// sets it. The tier's host sets it before it launches or enqueues a chain,
/// so each done that chain writes is posted after the set, and every done of
/// a pass is delivered before the host starts the next.
#[derive(Clone, Debug)]
pub struct PostCols(Arc<AtomicUsize>);

impl Default for PostCols {
    fn default() -> PostCols {
        PostCols(Arc::new(AtomicUsize::new(usize::MAX)))
    }
}

impl PostCols {
    /// The chain about to run uses `cols` columns of a row.
    pub fn set(&self, cols: usize) {
        self.0.store(cols, Ordering::Release);
    }

    pub(crate) fn get(&self) -> usize {
        self.0.load(Ordering::Acquire)
    }
}

/// One side's last generation of a counter: delivered on the reader,
/// posted on the writer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Relay {
    seen: u32,
}

impl Relay {
    /// A relay whose last generation is `seen`.
    pub(super) fn at(seen: u32) -> Relay {
        Relay { seen }
    }

    /// What the value `now` carries: `Ok(false)` nothing new (a
    /// repeated post included), `Ok(true)` one generation more, and the step
    /// as the error for anything else, since a row has at most one go or done
    /// in flight. Wrapping. The writer checks here before posting.
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

/// One end's relay over its own page.
pub(super) struct Half {
    lines: Vec<Line>,
    /// One post's moves, with room at load for the longest post.
    moves: Vec<Move>,
    /// The first failure by name: a step that was neither 0 nor 1, or a
    /// post the wire refused or lost. It stays, and the end carries nothing
    /// more.
    broken: Option<String>,
    /// The last settle found posts it could not show over, so the release
    /// that asked for it left this end's counters where they were: a later
    /// settle does not wait for them again ([`Half::settle`]).
    unsettled: bool,
    cols: PostCols,
}

impl Half {
    /// The end on the side `side` over `crossings`, whose tail lines start at
    /// byte `tail` of both pages, in crossing order. Every generation and
    /// inbox is taken as `page` holds it now, so nothing already there is
    /// delivered again. A crossing whose counter, progress word or span's
    /// columns are not 4-aligned before the tail is refused by name. Each
    /// post carries the columns `cols` holds.
    pub(super) fn new(
        crossings: Vec<Crossing>,
        side: Reader,
        tail: usize,
        page: &dyn Page,
        cols: PostCols,
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
                || !c.spans.iter().all(|s| {
                    fits(&s.bytes)
                        && s.cols > 0
                        && s.bytes.len().is_multiple_of(s.cols)
                        && (s.bytes.len() / s.cols).is_multiple_of(4)
                })
        }) {
            return Err(format!(
                "the crossing {c:?} is not 4-aligned columns before the link's tail at byte {tail}"
            ));
        }
        let moves = Vec::with_capacity(
            crossings
                .iter()
                .map(|c| c.spans.len() + 2)
                .max()
                .unwrap_or(0),
        );
        let lines = crossings
            .into_iter()
            .enumerate()
            .map(|(i, c)| {
                let writes = c.reader != side;
                Line {
                    c,
                    tail: tail + i * TAIL_LINE,
                    writes,
                    posted: 0,
                    relay: Relay::at(0),
                    stopped: false,
                }
            })
            .collect();
        let mut h = Half {
            lines,
            moves,
            broken: None,
            unsettled: false,
            cols,
        };
        for i in 0..h.lines.len() {
            let l = &mut h.lines[i];
            if l.writes {
                l.posted = page.word(l.c.counter).load(Ordering::Acquire);
            } else {
                l.relay = Relay::at(page.word(l.inbox()).load(Ordering::Acquire));
            }
        }
        Ok(h)
    }

    /// One pass: post each new generation this end writes, reap the wire,
    /// and only when none of the end's posts is in flight deliver what each
    /// inbox it reads holds: a card a delivery releases rewrites the ranges
    /// its end's earlier posts read. `true` when anything moved.
    pub(super) fn pass(&mut self, page: &dyn Page, wire: &mut dyn Wire) -> bool {
        let mut moved = false;
        for i in 0..self.lines.len() {
            if self.broken.is_some() {
                return false;
            }
            if self.lines[i].writes && !self.lines[i].stopped {
                moved |= self.post(i, page, wire);
            }
        }
        match wire.reap() {
            Ok(true) => {}
            Ok(false) => return moved,
            Err(e) => {
                self.break_with(e);
                return false;
            }
        }
        for i in 0..self.lines.len() {
            if self.broken.is_some() {
                return false;
            }
            if !self.lines[i].writes && !self.lines[i].stopped {
                moved |= self.deliver(i, page);
            }
        }
        moved
    }

    fn break_with(&mut self, why: String) {
        self.broken.get_or_insert(why);
    }

    /// The writer's side of line `i`: a generation one past the last posted
    /// goes to the reader's inbox after the bytes it publishes. The step is
    /// checked before any byte moves.
    fn post(&mut self, i: usize, page: &dyn Page, wire: &mut dyn Wire) -> bool {
        let line = &self.lines[i];
        let g = page.word(line.c.counter).load(Ordering::Acquire);
        match Relay::at(line.posted).take(g) {
            Ok(false) => return false,
            Ok(true) => {}
            Err(step) => {
                let why = format!(
                    "the tier page's counter at byte {} stepped by {step} between two posts: a row \
                     has at most one go or done in flight",
                    line.c.counter
                );
                self.break_with(why);
                return false;
            }
        }
        let cols = self.cols.get();
        let moves = &mut self.moves;
        moves.clear();
        moves.extend(line.c.spans.iter().map(|s| Move {
            from: s.bytes.start,
            to: s.bytes.start,
            len: s.bytes_at(cols),
        }));
        page.word(line.gen_slot()).store(g, Ordering::Relaxed);
        moves.push(Move {
            from: line.gen_slot(),
            to: line.inbox(),
            len: 4,
        });
        if let Err(e) = wire.post(moves) {
            self.break_with(e);
            return false;
        }
        self.lines[i].posted = g;
        true
    }

    /// The reader's side of line `i`: one more generation in the inbox adds
    /// one to the counter and, for a done, to the tier's progress; any other
    /// step breaks the end by name.
    fn deliver(&mut self, i: usize, page: &dyn Page) -> bool {
        let line = &mut self.lines[i];
        match line
            .relay
            .take(page.word(line.inbox()).load(Ordering::Acquire))
        {
            Ok(false) => false,
            Ok(true) => {
                if let Some(off) = line.c.prog {
                    page.word(off).fetch_add(1, Ordering::Release);
                }
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
        if let Some(prog) = line.c.prog {
            let progress = page.word(prog).load(Ordering::Acquire);
            page.word(line.prog_slot())
                .store(progress, Ordering::Relaxed);
            moves.push(Move {
                from: line.prog_slot(),
                to: prog,
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
                    self.break_with("the link's posts in flight did not finish in time".into());
                    return false;
                }
                Err(e) => {
                    self.break_with(e);
                    return false;
                }
            }
        }
    }

    /// Bring the end to where nothing it posted can still be read: reap until
    /// nothing is in flight, at most `bound`. Past it, or on a failed post,
    /// the end breaks, aborts its wire ([`Wire::abort`]) and reaps the
    /// aborted posts, at most `bound` again. `false`, with the end broken,
    /// when a post is still not known to be over; such an end later reaps
    /// once, without waiting.
    pub(super) fn settle(&mut self, wire: &mut dyn Wire, bound: Duration) -> bool {
        if self.unsettled {
            self.unsettled = !matches!(wire.reap(), Ok(true));
            return !self.unsettled;
        }
        let until = Instant::now() + bound;
        if self.drain(wire, &|| Instant::now() < until) {
            return true;
        }
        if let Err(e) = wire.abort() {
            self.break_with(e);
        }
        let until = Instant::now() + bound;
        while Instant::now() < until {
            if matches!(wire.reap(), Ok(true)) {
                return true;
            }
            std::hint::spin_loop();
        }
        self.unsettled = true;
        let held = "a settle found the link's posts in flight not over after an abort and left \
                    this end's counters where they were";
        match &mut self.broken {
            Some(why) => *why = format!("{why}; {held}"),
            None => self.broken = Some(held.into()),
        }
        false
    }

    /// Whether the last settle left this end's counters where they were
    /// ([`Half::settle`]).
    pub(super) fn unsettled(&self) -> bool {
        self.unsettled
    }

    /// The counter of every crossing this end carries.
    pub(super) fn counters(&self) -> Vec<usize> {
        self.lines.iter().map(|l| l.c.counter).collect()
    }

    /// The broken end's description, if it broke.
    pub(super) fn broken(&self) -> Option<&str> {
        self.broken.as_deref()
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};

    use super::super::{DRAIN, End, Memory, Mirror, Pump, Side};
    use super::{Crossing, Half, Move, Page, PostCols, Reader, Relay, Span, TAIL_LINE, Wire};
    use crate::host::step::RELEASE;

    pub(in crate::host::link) struct Mem(pub(in crate::host::link) Vec<AtomicU32>);

    impl Page for Mem {
        fn bytes(&self) -> usize {
            self.0.len() * 4
        }

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

        fn abort(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    /// A wire whose posts land at their second reap, as a NIC reads a posted
    /// range any time before the completion. As each post lands it records
    /// the word at byte `watch` of its own page.
    struct Late {
        wire: Memory<Mem>,
        held: Vec<Move>,
        reaps: u32,
        watch: usize,
        seen: Arc<AtomicU32>,
    }

    impl Late {
        fn new(from: &Arc<Mem>, to: &Arc<Mem>, watch: usize) -> (Late, Arc<AtomicU32>) {
            let seen = Arc::new(AtomicU32::new(u32::MAX));
            let late = Late {
                wire: wire(from, to),
                held: Vec::new(),
                reaps: 0,
                watch,
                seen: Arc::clone(&seen),
            };
            (late, seen)
        }
    }

    impl Wire for Late {
        fn post(&mut self, moves: &[Move]) -> Result<(), String> {
            self.held.extend_from_slice(moves);
            Ok(())
        }

        fn reap(&mut self) -> Result<bool, String> {
            if self.held.is_empty() {
                return Ok(true);
            }
            self.reaps += 1;
            if self.reaps < 2 {
                return Ok(false);
            }
            self.reaps = 0;
            let now = self.wire.from.word(self.watch).load(Ordering::Acquire);
            self.seen.store(now, Ordering::Release);
            self.wire.post(&std::mem::take(&mut self.held))?;
            Ok(true)
        }

        fn abort(&mut self) -> Result<(), String> {
            self.held.clear();
            Ok(())
        }
    }

    /// A wire whose writes fail and whose posts are never over, not even once
    /// aborted.
    struct Stuck(Arc<AtomicBool>);

    impl Wire for Stuck {
        fn post(&mut self, _: &[Move]) -> Result<(), String> {
            Ok(())
        }

        fn reap(&mut self) -> Result<bool, String> {
            Err("the test wire's write failed".into())
        }

        fn abort(&mut self) -> Result<(), String> {
            self.0.store(true, Ordering::Release);
            Ok(())
        }
    }

    /// A wire that takes part of a post and refuses the rest, and whose taken
    /// part is over only once aborted; it records the word at byte `watch`
    /// of its own page at the abort.
    struct Partial {
        from: Arc<Mem>,
        held: usize,
        watch: usize,
        seen: Arc<AtomicU32>,
    }

    impl Wire for Partial {
        fn post(&mut self, moves: &[Move]) -> Result<(), String> {
            self.held += moves.len() - 1;
            Err("the test wire refused a post's last write".into())
        }

        fn reap(&mut self) -> Result<bool, String> {
            Ok(self.held == 0)
        }

        fn abort(&mut self) -> Result<(), String> {
            let now = self.from.word(self.watch).load(Ordering::Acquire);
            self.seen.store(now, Ordering::Release);
            self.held = 0;
            Ok(())
        }
    }

    /// A wire that runs a card step and pump pass at a chosen reap.
    struct StepOnReap {
        wire: Memory<Mem>,
        step: Option<Box<dyn FnOnce() + Send>>,
        skip_first: bool,
    }

    impl Wire for StepOnReap {
        fn post(&mut self, moves: &[Move]) -> Result<(), String> {
            self.wire.post(moves)
        }

        fn reap(&mut self) -> Result<bool, String> {
            if self.skip_first {
                self.skip_first = false;
            } else if let Some(step) = self.step.take() {
                step();
            }
            self.wire.reap()
        }

        fn abort(&mut self) -> Result<(), String> {
            self.wire.abort()
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

    // Byte offsets of the model page: one row of one tier, two crossings,
    // and a second tier's flags for the pages of two tiers.
    const GO: usize = 0;
    const CNT: usize = 64;
    const PROG: usize = 128;
    const IMAGE: usize = 192;
    const ROWS: usize = 256;
    const SPAN: usize = 64;
    const TAIL: usize = 320;
    const GO_1: usize = 8;
    const CNT_1: usize = 72;
    const PROG_1: usize = 136;
    const WORDS: usize = (TAIL + 4 * TAIL_LINE) / 4;

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
        let (image, rows) = (IMAGE..IMAGE + SPAN, ROWS..ROWS + SPAN);
        vec![
            Crossing {
                counter: GO,
                reader: Reader::Tier,
                spans: vec![Span::whole(image)],
                prog: None,
            },
            Crossing {
                counter: CNT,
                reader: Reader::Stage,
                spans: vec![Span::whole(rows)],
                prog: Some(PROG),
            },
        ]
    }

    // The row's crossings for two tiers, which post the same image and rows
    // spans, each tier with its own counters and progress word.
    fn two_tiers() -> Vec<Crossing> {
        let mut c = row();
        let mut t1 = row();
        t1[0].counter = GO_1;
        t1[1].counter = CNT_1;
        t1[1].prog = Some(PROG_1);
        c.extend(t1);
        c
    }

    /// The loopback's wire from `from` into `to`.
    fn wire(from: &Arc<Mem>, to: &Arc<Mem>) -> Memory<Mem> {
        Memory {
            from: Arc::clone(from),
            to: Arc::clone(to),
        }
    }

    /// Mirrored pages over the model row, with no pump thread: the stage
    /// card's end over `stage`, posting through `stage_wire`, and the tier
    /// cards' end over `tier`, posting through `tier_wire`.
    fn link(
        stage: &Arc<Mem>,
        tier: &Arc<Mem>,
        stage_wire: impl Wire + 'static,
        tier_wire: impl Wire + 'static,
    ) -> Mirror<Mem> {
        let half = |side, page: &Arc<Mem>| {
            Half::new(row(), side, TAIL, &**page, PostCols::default())
                .expect("the model row fits before the tail")
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
                tier_wire,
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
    // one back. Goes `ks` carry image k and dones rows k + ROWS_OF in every
    // word of the span; the bytes are checked for every go up to `checked`,
    // the ones before a release.
    const ROWS_OF: u32 = 1 << 20;

    fn fill(p: &Mem, at: usize, v: u32) {
        for off in (at..at + SPAN).step_by(4) {
            p.word(off).store(v, Ordering::Relaxed);
        }
    }

    /// The first word of the span at `at` that is not `v`, as (byte, word).
    fn stray(p: &Mem, at: usize, v: u32) -> Option<(usize, u32)> {
        (at..at + SPAN)
            .step_by(4)
            .map(|off| (off, p.word(off).load(Ordering::Relaxed)))
            .find(|&(_, w)| w != v)
    }

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
                if k <= checked
                    && let Some((off, img)) = stray(&tier, IMAGE, k)
                {
                    return Err(format!(
                        "the tier's go {k} carried the image of go {img} at byte {off}"
                    ));
                }
                fill(&tier, ROWS, k + ROWS_OF);
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
            fill(stage, IMAGE, k);
            stage.word(GO).fetch_add(1, Ordering::Release);
            if !wait_take(stage, CNT, until) {
                return Err(format!("the stage's done {k} never came"));
            }
            if k <= checked
                && let Some((off, rows)) = stray(stage, ROWS, k + ROWS_OF)
            {
                return Err(format!(
                    "the stage's done {k} carried rows {rows:#x} at byte {off}"
                ));
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
        let mut m = link(&stage, &tier, wire(&stage, &tier), wire(&tier, &stage));
        m.pumps = [&m.stage, &m.tier]
            .map(|e| Pump::start(vec![Arc::clone(e)]).expect("a pump thread"))
            .into();
        let until = Instant::now() + Duration::from_secs(20);
        let run = || -> Result<(), String> {
            let card = tier_card(tier.clone(), 1..=N + K, N, until);
            stage_card(&stage, 1..=N, N, until)?;
            m.release(RELEASE)?;
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
        let m = link(&stage, &tier, wire(&stage, &tier), wire(&tier, &stage));
        assert_eq!(m.release(RELEASE), Ok(()));
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

    // A go and a done still in flight at a release land with the bytes they
    // published, not the ones the released cards write next: each end
    // raises the counter its card waits on only once its own posts are over.
    // Both ends settled, so the link's drop frees both pages.
    #[test]
    fn a_release_lands_the_posts_in_flight_first() {
        let stage = page(&[]);
        let tier = page(&[]);
        let (go_wire, go_seen) = Late::new(&stage, &tier, CNT);
        let (done_wire, done_seen) = Late::new(&tier, &stage, GO);
        let m = link(&stage, &tier, go_wire, done_wire);
        stage.word(IMAGE).store(1, Ordering::Relaxed);
        stage.word(GO).store(1, Ordering::Release);
        tier.word(ROWS).store(7, Ordering::Relaxed);
        tier.word(CNT).store(1, Ordering::Release);
        tier.word(PROG).store(1, Ordering::Release);
        m.pass();
        assert_eq!(tier.word(TAIL).load(Ordering::Acquire), 0);
        assert_eq!(stage.word(TAIL + TAIL_LINE).load(Ordering::Acquire), 0);
        assert_eq!(m.release(RELEASE), Ok(()));
        stage.word(IMAGE).store(2, Ordering::Relaxed);
        tier.word(ROWS).store(8, Ordering::Relaxed);
        m.pass();
        assert_eq!(m.broken(), None);
        assert_eq!(
            go_seen.load(Ordering::Acquire),
            0,
            "the stage's done at the go's landing"
        );
        assert_eq!(
            done_seen.load(Ordering::Acquire),
            0,
            "the tier's go at the done's landing"
        );
        assert_eq!(tier.word(IMAGE).load(Ordering::Acquire), 1);
        assert_eq!(stage.word(ROWS).load(Ordering::Acquire), 7);
        assert_eq!(tier.word(GO).load(Ordering::Acquire), RELEASE);
        assert_eq!(stage.word(CNT).load(Ordering::Acquire), RELEASE);
        drop(m);
        assert_eq!(
            (Arc::strong_count(&stage), Arc::strong_count(&tier)),
            (1, 1)
        );
    }

    // A drop with no release lands both ends' posts before freeing their
    // pages, without raising the counters their cards wait on.
    #[test]
    fn a_drop_lands_the_posts_in_flight_before_freeing_the_pages() {
        let stage = page(&[]);
        let tier = page(&[]);
        let (go_wire, go_seen) = Late::new(&stage, &tier, CNT);
        let (done_wire, done_seen) = Late::new(&tier, &stage, GO);
        let m = link(&stage, &tier, go_wire, done_wire);
        stage.word(IMAGE).store(1, Ordering::Relaxed);
        stage.word(GO).store(1, Ordering::Release);
        tier.word(ROWS).store(7, Ordering::Relaxed);
        tier.word(CNT).store(1, Ordering::Release);
        tier.word(PROG).store(1, Ordering::Release);
        m.pass();
        assert_eq!(tier.word(TAIL).load(Ordering::Acquire), 0);
        assert_eq!(stage.word(TAIL + TAIL_LINE).load(Ordering::Acquire), 0);
        drop(m);
        assert_eq!(go_seen.load(Ordering::Acquire), 0);
        assert_eq!(done_seen.load(Ordering::Acquire), 0);
        assert_eq!(tier.word(IMAGE).load(Ordering::Acquire), 1);
        assert_eq!(stage.word(ROWS).load(Ordering::Acquire), 7);
        assert_eq!(tier.word(TAIL).load(Ordering::Acquire), 1);
        assert_eq!(stage.word(TAIL + TAIL_LINE).load(Ordering::Acquire), 1);
        assert_eq!(stage.word(CNT).load(Ordering::Acquire), 0);
        assert_eq!(tier.word(GO).load(Ordering::Acquire), 0);
        assert_eq!(
            (Arc::strong_count(&stage), Arc::strong_count(&tier)),
            (1, 1)
        );
    }

    // A drop whose outstanding writes remain after the abort keeps both
    // ends and their pages alive, without a release beforehand.
    #[test]
    fn a_drop_keeps_the_pages_of_posts_it_cannot_settle() {
        let stage = page(&[]);
        let tier = page(&[]);
        let aborted = Arc::new(AtomicBool::new(false));
        let m = link(
            &stage,
            &tier,
            Stuck(Arc::clone(&aborted)),
            wire(&tier, &stage),
        );
        stage.word(GO).store(1, Ordering::Release);
        m.pass();
        assert!(!aborted.load(Ordering::Acquire));
        drop(m);
        assert!(aborted.load(Ordering::Acquire));
        assert_eq!(stage.word(CNT).load(Ordering::Acquire), 0);
        assert_eq!(tier.word(GO).load(Ordering::Acquire), 0);
        assert!(Arc::strong_count(&stage) > 1 && Arc::strong_count(&tier) > 1);
    }

    // A done found in the stage's inbox while the stage end's go is still in
    // flight, an answer that came before the end reaped the go's completion,
    // is delivered only once the go is over: the stage card it releases
    // writes the next image and go, which the go's writes and a resend of
    // them may still read.
    #[test]
    fn a_delivery_waits_for_the_ends_posts_in_flight() {
        let stage = page(&[]);
        let tier = page(&[]);
        let (go_wire, go_seen) = Late::new(&stage, &tier, CNT);
        let m = link(&stage, &tier, go_wire, wire(&tier, &stage));
        stage.word(IMAGE).store(1, Ordering::Relaxed);
        stage.word(GO).store(1, Ordering::Release);
        stage.word(TAIL + TAIL_LINE).store(1, Ordering::Release);
        m.pass();
        assert_eq!(
            stage.word(CNT).load(Ordering::Acquire),
            0,
            "the done delivered while the go is in flight"
        );
        m.pass();
        assert_eq!(m.broken(), None);
        assert_eq!(
            go_seen.load(Ordering::Acquire),
            0,
            "the stage's done at the go's landing"
        );
        assert_eq!(tier.word(IMAGE).load(Ordering::Acquire), 1);
        assert_eq!(stage.word(CNT).load(Ordering::Acquire), 1);
    }

    // The tiers of a row post the same spans. A release stops every tier's
    // counters on both ends before any counter moves. Each card and pump
    // run during the other end's settle, before the first raise, and the
    // tier card runs again after release; neither end carries a span.
    #[test]
    fn a_release_stops_every_tiers_counters_before_any_moves() {
        let stage = page(&[]);
        let tier = page(&[]);
        let tier_end = Arc::new(End::new(
            Half::new(two_tiers(), Reader::Tier, TAIL, &*tier, PostCols::default())
                .expect("the tier's crossings"),
            StepOnReap {
                wire: wire(&tier, &stage),
                step: None,
                skip_first: true,
            },
            Arc::clone(&tier),
        ));
        let stage_end = Arc::new(End::new(
            Half::new(
                two_tiers(),
                Reader::Stage,
                TAIL,
                &*stage,
                PostCols::default(),
            )
            .expect("the stage's crossings"),
            StepOnReap {
                wire: wire(&stage, &tier),
                step: Some(Box::new({
                    let stage = Arc::clone(&stage);
                    let tier = Arc::clone(&tier_end);
                    move || {
                        assert_eq!(stage.word(CNT).load(Ordering::Acquire), 0);
                        let page = tier.page();
                        assert_eq!(page.word(GO).load(Ordering::Acquire), 0);
                        page.word(ROWS).store(7, Ordering::Relaxed);
                        page.word(CNT_1).fetch_add(1, Ordering::Release);
                        page.word(PROG_1).fetch_add(1, Ordering::Release);
                        assert!(!tier.pass(), "a tier carried a done during release");
                    }
                })),
                skip_first: false,
            },
            Arc::clone(&stage),
        ));
        tier_end.lock().1.step = Some(Box::new({
            let stage = Arc::clone(&stage_end);
            let tier = Arc::clone(&tier);
            move || {
                let page = stage.page();
                assert_eq!(page.word(CNT).load(Ordering::Acquire), 0);
                assert_eq!(tier.word(GO).load(Ordering::Acquire), 0);
                page.word(IMAGE).store(11, Ordering::Relaxed);
                page.word(GO_1).fetch_add(1, Ordering::Release);
                assert!(!stage.pass(), "the stage carried a go during release");
            }
        }));
        let m = Mirror {
            pumps: Vec::new(),
            stage: stage_end,
            tier: tier_end,
            run_on: Some((CNT_1, 1)),
        };
        assert_eq!(m.release(RELEASE), Ok(()));
        assert_eq!(m.broken(), None);
        assert_eq!(tier.word(CNT_1).load(Ordering::Acquire), 2);
        assert_eq!(stage.word(GO_1).load(Ordering::Acquire), 1);
        assert_eq!(tier.word(IMAGE).load(Ordering::Acquire), 0, "image carried");
        assert_eq!(stage.word(ROWS).load(Ordering::Acquire), 0, "rows carried");
        assert_eq!(
            stage.word(PROG_1).load(Ordering::Acquire),
            0,
            "a done counted"
        );
        for off in [CNT, CNT_1] {
            assert_eq!(stage.word(off).load(Ordering::Acquire), RELEASE);
        }
        for off in [GO, GO_1] {
            assert_eq!(tier.word(off).load(Ordering::Acquire), RELEASE);
        }
    }

    // An end whose posts are not known to be over, even after it aborts its
    // wire, leaves the counter its card waits on where it is: the card keeps
    // waiting rather than rewrite a span a write may still read, and the
    // release names the end. The other end settles and releases its own, a
    // later release does not wait for the stuck end again, and the link's
    // drop leaks both ends and their pages.
    #[test]
    fn a_release_leaves_the_counters_of_an_end_it_cannot_settle() {
        let stage = page(&[]);
        let tier = page(&[]);
        let aborted = Arc::new(AtomicBool::new(false));
        let m = link(
            &stage,
            &tier,
            Stuck(Arc::clone(&aborted)),
            wire(&tier, &stage),
        );
        stage.word(GO).store(1, Ordering::Release);
        m.pass();
        let held = m.release(RELEASE).expect_err("an end that did not settle");
        assert!(held.contains("the test wire's write failed"), "{held}");
        assert!(aborted.load(Ordering::Acquire));
        assert_eq!(stage.word(CNT).load(Ordering::Acquire), 0);
        assert_eq!(tier.word(GO).load(Ordering::Acquire), RELEASE);
        assert!(
            m.broken()
                .is_some_and(|b| b.contains("left this end's counters"))
        );
        let again = Instant::now();
        assert!(m.release(RELEASE).is_err());
        assert!(again.elapsed() < DRAIN);
        assert_eq!(stage.word(CNT).load(Ordering::Acquire), 0);
        drop(m);
        assert!(Arc::strong_count(&stage) > 1 && Arc::strong_count(&tier) > 1);
    }

    // A post the wire took in part breaks its end, and a release raises the
    // end's counters only once the wire is aborted and the part is over.
    #[test]
    fn a_release_after_a_partial_post_waits_for_the_abort() {
        let stage = page(&[]);
        let tier = page(&[]);
        let seen = Arc::new(AtomicU32::new(u32::MAX));
        let partial = Partial {
            from: Arc::clone(&stage),
            held: 0,
            watch: CNT,
            seen: Arc::clone(&seen),
        };
        let m = link(&stage, &tier, partial, wire(&tier, &stage));
        stage.word(GO).store(1, Ordering::Release);
        m.pass();
        assert!(m.broken().is_some_and(|b| b.contains("refused")));
        assert_eq!(m.release(RELEASE), Ok(()));
        assert_eq!(
            seen.load(Ordering::Acquire),
            0,
            "the stage's done at the abort"
        );
        assert_eq!(stage.word(CNT).load(Ordering::Acquire), RELEASE);
        assert_eq!(tier.word(GO).load(Ordering::Acquire), RELEASE);
    }

    /// A wire that lands every write of a post but its last, the inbox
    /// word's, and refuses that one.
    struct LoseInbox(Memory<Mem>);

    impl Wire for LoseInbox {
        fn post(&mut self, moves: &[Move]) -> Result<(), String> {
            self.0.post(&moves[..moves.len() - 1])?;
            Err("the test wire lost a post's inbox write".into())
        }

        fn reap(&mut self) -> Result<bool, String> {
            Ok(true)
        }

        fn abort(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    // The host's progress counts a done when the reader's end delivers it, the
    // moment the stage card can take it: a done posted before its reader
    // passes leaves the progress where it was, and a done whose inbox write
    // is lost leaves it behind although its rows, and every other write,
    // landed, so the host never finds the progress caught up with a done the
    // stage card does not get.
    #[test]
    fn the_progress_counts_a_done_when_it_is_delivered() {
        let stage = page(&[]);
        let tier = page(&[]);
        let m = link(&stage, &tier, wire(&stage, &tier), wire(&tier, &stage));
        tier.word(ROWS).store(7, Ordering::Relaxed);
        tier.word(CNT).store(1, Ordering::Release);
        tier.word(PROG).store(1, Ordering::Release);
        m.tier.pass();
        assert_eq!(stage.word(ROWS).load(Ordering::Acquire), 7);
        assert_eq!(stage.word(TAIL + TAIL_LINE).load(Ordering::Acquire), 1);
        assert_eq!(stage.word(PROG).load(Ordering::Acquire), 0);
        m.stage.pass();
        assert_eq!(stage.word(CNT).load(Ordering::Acquire), 1);
        assert_eq!(stage.word(PROG).load(Ordering::Acquire), 1);

        let stage = page(&[]);
        let tier = page(&[]);
        let m = link(
            &stage,
            &tier,
            wire(&stage, &tier),
            LoseInbox(wire(&tier, &stage)),
        );
        tier.word(ROWS).store(7, Ordering::Relaxed);
        tier.word(CNT).store(1, Ordering::Release);
        tier.word(PROG).store(1, Ordering::Release);
        m.tier.pass();
        assert!(m.broken().is_some_and(|b| b.contains("lost")));
        assert_eq!(stage.word(ROWS).load(Ordering::Acquire), 7);
        assert_eq!(stage.word(TAIL + TAIL_LINE).load(Ordering::Acquire), 0);
        assert_eq!(stage.word(PROG).load(Ordering::Acquire), 0);
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
        let m = link(&stage, &tier, Refuse, wire(&tier, &stage));
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

    // A done whose rows hold four columns carries the columns its chain
    // uses: a chain of one column leaves the other three as the stage's page
    // held them, and a chain of all four carries every one.
    #[test]
    fn a_done_carries_the_columns_its_chain_uses() {
        let stage = page(&[]);
        let tier = page(&[]);
        let mut crossings = row();
        crossings[1].spans = vec![Span {
            bytes: ROWS..ROWS + 64,
            cols: 4,
        }];
        let cols = PostCols::default();
        let mut end = Half::new(crossings, Reader::Tier, TAIL, &*tier, cols.clone())
            .expect("the model row fits before the tail");
        let mut done = wire(&tier, &stage);
        for (generation, used) in [(1u32, 1usize), (2, 4)] {
            let value = |w: usize| generation << 8 | w as u32;
            for w in 0..16 {
                tier.word(ROWS + 4 * w).store(value(w), Ordering::Relaxed);
            }
            cols.set(used);
            tier.word(CNT).store(generation, Ordering::Release);
            assert!(end.pass(&*tier, &mut done));
            for w in 0..16 {
                let want = if w < 4 * used { value(w) } else { 0 };
                let got = stage.word(ROWS + 4 * w).load(Ordering::Acquire);
                assert_eq!(got, want, "word {w} of the rows at {used} columns");
            }
        }
        assert_eq!(end.broken(), None);
    }
}
