//! The one load of a part source onto a card (`model::part::Source`: a GGUF file or a
//! checkpoint folder). The caller's rule says which tensors the card holds, whole, the part of
//! a rank, or a stack's listed experts, as the source plans them; every planned part's card
//! format is checked before any is read; then parts are read while the parts before them still
//! hold their read bytes, up to the planned read-ahead, and staged in order as they arrive through
//! one staging ring, which one synchronize ends ([`load`]). A rank of a tensor-parallel load of N
//! ranks passes its family's cut; a one-card load passes its placement plan's card
//! ([`OnCard`]), and a folder whose host reads tensors where its located header holds them
//! passes those ([`InPlace`]): the card takes its share of a stack the plan splits between it
//! and that host, the host reads the rest where the folder stores it.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Condvar, Mutex, PoisonError, mpsc};
use std::time::{Duration, Instant};

use cuda_core::CudaStream;
use gguf::{GgmlType, Place, Split, TensorInfo};
use model::part::{Part, Planned, Source, Take, read_ahead};
use model::placement::ctx_ask::Unpaged;
use model::placement::workstation::{HostRead, host_available_read};
use model::placement::{CardFormat, Device, ExpertList, Format, ModelTensor, Plan, Role, Segment};

use crate::GpuError;
use crate::upload::UploadRing;
use crate::weights::{CardOf, TensorPart, Weights, part_format, stage_parts};

/// What the errors name.
const WHAT: &str = "parts::load";

/// The parts selected by `rule`, each read once the parts before it that are not yet staged
/// leave it room under the planned read-ahead and the live host reading, and staged in plan order
/// while later parts still read ([`read_staged`]); a part over the read-ahead reads with none
/// held. The placed caller reserves `LoadPlan::host_bytes` in its host need; a part alone over the live
/// reading is refused. A refusal drains pending copies before freeing device buffers. Under
/// `release`, each part's source pages are dropped after it is staged.
pub fn load<S: Source + Sync, E: std::fmt::Display>(
    stream: &CudaStream,
    source: &S,
    rule: impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
    card: impl Fn(&TensorInfo) -> Option<CardFormat>,
    release: bool,
) -> Result<Weights, GpuError> {
    let reading = host_available_read().map_err(refused)?;
    let plan = LoadPlan::new(source, rule, reading.0)?;
    Ok(load_plan(stream, source, &plan, card, release, &reading)?.0)
}

/// What a part source's load read: the wall from its first part's read to its last's, and the
/// most host bytes its reads were planned to hold at once (`LoadPlan::host_bytes`).
#[derive(Clone, Copy, Debug)]
pub struct PartsRead {
    pub read: Duration,
    pub peak: u64,
}

fn load_plan<S: Source + Sync>(
    stream: &CudaStream,
    source: &S,
    plan: &LoadPlan,
    card: impl Fn(&TensorInfo) -> Option<CardFormat>,
    release: bool,
    room: &(u64, HostRead),
) -> Result<(Weights, PartsRead), GpuError> {
    plan.check(room)?;
    for p in plan.parts.iter() {
        if let Some(format) = card(&p.info) {
            card_layout(&p.info, format)?;
        }
    }
    stream.context().bind_to_thread()?;
    let mut staged = BTreeMap::new();
    // Dropped before the staged device buffers: draining the ring protects refused loads.
    let mut ring = UploadRing::new(stream, plan.upload_bytes)?;
    let live = || host_available_read().map_err(refused);
    let read = read_staged(source, plan, live, release, |_, part| {
        let one = TensorPart {
            info: part.info.clone(),
            spans: part.spans(),
            scale: part.scale.clone(),
        };
        let held = |name: &str| staged.contains_key(name);
        let added = stage_parts(stream, &mut ring, &[one], held, &card)?;
        staged.extend(added);
        Ok(())
    })?;
    ring.finish(stream)?;
    let read = PartsRead {
        read,
        peak: plan.host_bytes,
    };
    Ok((Weights::of_staged(staged), read))
}

/// Whether a part may start its read ([`LoadPlan::admits`]).
#[derive(Debug, PartialEq, Eq)]
enum Admit {
    Read,
    /// Over the planned read-ahead: it waits for the parts before it to be staged.
    Wait,
    /// Within the planned read-ahead, over the live host reading: it waits as `Wait` does.
    Short,
}

/// What the readers and the stage of [`read_staged`] share.
#[derive(Default)]
struct Window {
    /// The next part a reader takes.
    next: usize,
    /// The parts staged: every part before it has dropped its read bytes.
    staged: usize,
    /// Staged parts whose source pages a reader drops next; the readers end once every part is
    /// staged and this is empty.
    releases: Vec<usize>,
    /// The first refusal: every reader ends, and the stage ends at its next wait.
    failed: Option<GpuError>,
    /// The stage refused; the readers end.
    stop: bool,
    /// The live reading has held a read back once (named once on stderr).
    short: bool,
    /// The first part's read start and the last part's read end.
    reads: Option<(Instant, Instant)>,
}

/// `plan`'s parts read on threads of their own while `stage` takes each, in plan order, as soon
/// as it and every part before it are read, so reads, normalizing and staging overlap across the
/// whole load. A reader takes the next part once [`LoadPlan::admits`] lets it beside the read
/// parts not yet staged, reading `available` as it admits; a part's read bytes are freed when
/// its stage returns, and under `release` a reader then drops its source pages. Staging packs
/// and copies over the pool, which is not reentrant, so the readers are not the pool's; each
/// first sets its mask to the pool's cpus, since a spawned thread inherits the caller's pin. The
/// first refusal, a reader's, an admission's, a read's, a release's or the stage's, stops the
/// readers taking more parts; a reader's or the stage's panic stops them the same way and then
/// unwinds. The wall from the first part's read to the last's.
fn read_staged<'s, 'p, S: Source + Sync>(
    source: &'s S,
    plan: &'p LoadPlan,
    available: impl Fn() -> Result<(u64, HostRead), GpuError> + Sync,
    release: bool,
    mut stage: impl FnMut(&'p Planned, Part<'s>) -> Result<(), GpuError>,
) -> Result<Duration, GpuError> {
    let parts: &[Planned] = &plan.parts;
    let pool = threads::pool();
    let cpus: Vec<u32> = pool.topology().concat();
    let shared = Mutex::new(Window::default());
    let changed = Condvar::new();
    let lock = || shared.lock().unwrap_or_else(PoisonError::into_inner);
    let fail = |e: GpuError| {
        lock().failed.get_or_insert(e);
        changed.notify_all();
    };
    /// Runs its call when dropped by a panic's unwinding.
    struct OnUnwind<F: Fn()>(F);
    impl<F: Fn()> Drop for OnUnwind<F> {
        fn drop(&mut self) {
            if std::thread::panicking() {
                (self.0)();
            }
        }
    }
    let (tx, rx) = mpsc::channel();
    std::thread::scope(|scope| {
        let _stage_unwind = OnUnwind(|| fail(refused("the parts' stage panicked".into())));
        for _ in 0..pool.threads().clamp(1, parts.len().max(1)) {
            let tx = tx.clone();
            let (cpus, changed, available, lock, fail) =
                (&cpus, &changed, &available, &lock, &fail);
            let reader = std::thread::Builder::new().spawn_scoped(scope, move || {
                let _reader_unwind = OnUnwind(|| fail(refused("a part reader panicked".into())));
                if let Err(e) = os::set_mask(cpus) {
                    return fail(refused(format!("a part reader's cpus: {e}")));
                }
                let mut w = lock();
                while !w.stop && w.failed.is_none() {
                    if let Some(j) = w.releases.pop() {
                        drop(w);
                        if let Err(e) = source.release(&parts[j]) {
                            return fail(refused(e.to_string()));
                        }
                        w = lock();
                        continue;
                    }
                    if w.next == parts.len() {
                        if w.staged == parts.len() {
                            break;
                        }
                        w = changed.wait(w).unwrap_or_else(PoisonError::into_inner);
                        continue;
                    }
                    let i = w.next;
                    match plan.admits(w.staged..i, available) {
                        Ok(Admit::Read) => {
                            w.next += 1;
                            drop(w);
                            let start = Instant::now();
                            let read = source.read(&parts[i]).map_err(|e| e.to_string());
                            let end = Instant::now();
                            if tx.send((i, read)).is_err() {
                                return;
                            }
                            w = lock();
                            let (first, last) = w.reads.get_or_insert((start, end));
                            *first = (*first).min(start);
                            *last = (*last).max(end);
                        }
                        Ok(admit) => {
                            if admit == Admit::Short && !std::mem::replace(&mut w.short, true) {
                                eprintln!(
                                    "{WHAT}: the parts' read-ahead of up to {} B is over the live \
                                     host room; fewer parts read at once",
                                    plan.window
                                );
                            }
                            w = changed.wait(w).unwrap_or_else(PoisonError::into_inner);
                        }
                        Err(e) => {
                            drop(w);
                            return fail(e);
                        }
                    }
                }
            });
            if let Err(e) = reader {
                fail(refused(format!("a part reader could not start: {e}")));
                break;
            }
        }
        drop(tx);
        let mut read: Vec<Option<Result<Part<'s>, String>>> = parts.iter().map(|_| None).collect();
        let mut at = 0;
        let staged = (|| {
            while at < parts.len() {
                let Ok((i, part)) = rx.recv() else {
                    return Err(lock()
                        .failed
                        .take()
                        .unwrap_or_else(|| refused("a part reader stopped".into())));
                };
                read[i] = Some(part);
                while let Some(part) = read.get_mut(at).and_then(Option::take) {
                    stage(&parts[at], part.map_err(refused)?)?;
                    let mut w = lock();
                    w.staged = at + 1;
                    if release {
                        w.releases.push(at);
                    }
                    drop(w);
                    changed.notify_all();
                    at += 1;
                }
            }
            Ok(())
        })();
        lock().stop = staged.is_err();
        changed.notify_all();
        drop(rx);
        staged
    })?;
    let w = shared.into_inner().unwrap_or_else(PoisonError::into_inner);
    match w.failed {
        Some(e) => Err(e),
        None => Ok(w.reads.map_or(Duration::ZERO, |(first, last)| last - first)),
    }
}

/// The least host bytes a load of card `card` of `plan` reads from `source`, whose located header
/// holds `in_place` where the folder stores it, beside the plan's own terms: its parts one at a
/// time with the staging ring, the peak of [`PlacedSource`]'s read at no room. A plan reserves
/// them before its NVMe split (`workstation::reserve_part_reads`); the card's parts, and so these
/// bytes, are the same at every host room.
pub fn least_reads<S: Source>(
    source: &S,
    in_place: &InPlace,
    plan: &Plan<'_>,
    card: usize,
) -> Result<u64, GpuError> {
    let on = OnCard::of_in_place(plan, card, in_place)?;
    Ok(LoadPlan::reading(source, plan, &on, 0)?.host_bytes)
}

/// A placed source shares one admission and upload implementation across model components.
pub(crate) struct PlacedSource<S> {
    pub(crate) source: S,
    pub(crate) in_place: InPlace,
    /// The last read planned, with the card and card tensors it was planned for: a context
    /// search asks the host bytes of many plans that hold the same tensors on the card, the load
    /// then reads that plan's parts, and planning the parts checks every part the source makes.
    pub(crate) last: Mutex<Option<(MemoKey, u64, LoadPlan)>>,
}

/// The card share and the card tensors a read was planned for.
type MemoKey = (usize, OnCard);

impl<S: Source> PlacedSource<S> {
    /// The read of `plan`'s card `card` under `room`, and the card tensors it reads: the parts
    /// and their costs are planned once for a card and its tensors (`last`), and a read under
    /// another room takes their read-ahead again ([`LoadPlan::under`]).
    fn reading(
        &self,
        plan: &Plan<'_>,
        card: usize,
        room: u64,
    ) -> Result<(OnCard, LoadPlan), GpuError> {
        let key = (card, OnCard::of_in_place(plan, card, &self.in_place)?);
        let mut last = self
            .last
            .lock()
            .map_err(|_| refused("a part source's last read lock is poisoned".into()))?;
        let read = match last.as_ref() {
            Some((held, at, read)) if *held == key && *at == room => read.clone(),
            Some((held, _, read)) if *held == key => LoadPlan::under(
                read.parts.clone(),
                read.costs.clone(),
                read.upload_bytes,
                room,
            )?,
            _ => LoadPlan::reading(&self.source, plan, &key.1, room)?,
        };
        let on = key.1.clone();
        *last = Some((key, room, read.clone()));
        Ok((on, read))
    }
}

impl<S: Source + Send + Sync> crate::model::PartsLoad for PlacedSource<S> {
    fn host_bytes(&self, plan: &Plan<'_>, card: usize, room: u64) -> Result<u64, GpuError> {
        Ok(self.reading(plan, card, room)?.1.host_bytes)
    }

    fn load(
        &self,
        stream: &CudaStream,
        plan: &Plan<'_>,
        card: usize,
        release: bool,
        room: u64,
    ) -> Result<(Weights, PartsRead), GpuError> {
        let (on, read) = self.reading(plan, card, room)?;
        let (available, reading) = host_available_read().map_err(refused)?;
        let (w, read) = load_plan(
            stream,
            &self.source,
            &read,
            |t| on.format(t),
            release,
            &(available.min(room), reading),
        )?;
        on.check(&w)?;
        Ok((w, read))
    }

    fn load_whole(&self, stream: &CudaStream, card: CardOf<'_>) -> Result<Weights, GpuError> {
        load(
            stream,
            &self.source,
            |_, _, _| Ok::<_, std::convert::Infallible>(Take::Whole),
            card,
            false,
        )
    }
}

/// Parts, their read costs, the owned read bytes they may hold at once, the peak owned host bytes
/// reserved before loading and the least peak, one part at a time.
#[derive(Clone)]
pub(crate) struct LoadPlan {
    parts: Arc<[Planned]>,
    costs: Arc<[(u64, u64)]>,
    /// The read-ahead: the most owned read and packing bytes the parts read and not yet staged
    /// hold at once ([`Self::admits`]), the largest a leading run of parts takes under the
    /// planned room.
    window: u64,
    host_bytes: u64,
    serial_bytes: u64,
    upload_bytes: usize,
}

impl LoadPlan {
    /// Whether part `held.end` may start its read while the parts `held` are read and not yet
    /// staged: with none held it reads alone, refused when its peak is over the live reading
    /// `available` gives; with some held, the held parts and it within the read-ahead
    /// ([`read_ahead`] under [`Self::window`]) and within the live reading, or it waits. The live
    /// reading already lacks what the held parts hold, so it is read only once the read-ahead
    /// admits the part.
    fn admits(
        &self,
        held: std::ops::Range<usize>,
        available: impl Fn() -> Result<(u64, HostRead), GpuError>,
    ) -> Result<Admit, GpuError> {
        let (at, i) = (held.start, held.end);
        let parts = &self.parts[at..=i];
        let costs = &self.costs[at..=i];
        let fits = |room: u64| {
            read_ahead(parts, costs, room)
                .map(|(n, bytes)| n == parts.len() && bytes <= room)
                .map_err(|e| refused(e.to_string()))
        };
        if at < i && !fits(self.window)? {
            return Ok(Admit::Wait);
        }
        let (live, read) = available()?;
        if at == i {
            let bytes = read_ahead(parts, costs, 0)
                .map_err(|e| refused(e.to_string()))?
                .1;
            if bytes > live {
                return Err(refused(format!(
                    "part read-ahead needs {bytes} B, the host room is {live} B ({read})"
                )));
            }
            return Ok(Admit::Read);
        }
        Ok(if fits(live)? {
            Admit::Read
        } else {
            Admit::Short
        })
    }

    /// Refused unless one part at a time fits the host room `room`, named by the reading that
    /// gave it; [`Self::admits`] holds reads back to the live reading.
    fn check(&self, (available, read): &(u64, HostRead)) -> Result<(), GpuError> {
        if self.serial_bytes > *available {
            return Err(refused(format!(
                "part read-ahead needs {} B, the host room is {available} B ({read})",
                self.serial_bytes
            )));
        }
        Ok(())
    }

    fn new<S: Source, E: std::fmt::Display>(
        source: &S,
        rule: impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
        room: u64,
    ) -> Result<Self, GpuError> {
        let parts = source.plan(rule).map_err(|e| refused(e.to_string()))?;
        let mut upload_bytes = 0usize;
        let mut costs = Vec::with_capacity(parts.len());
        for p in &parts {
            let staged = part_format(&p.info)?.1;
            upload_bytes = upload_bytes
                .checked_add(staged)
                .ok_or_else(|| refused("the parts' staging bytes pass usize".into()))?;
            // Packing capacity and scale copies coexist with every held part's read results.
            let read = source.read_bytes(p).map_err(|e| refused(e.to_string()))?;
            let packing = (staged as u64)
                .checked_mul(2)
                .ok_or_else(|| refused(format!("{}: packing bytes pass u64", p.info.name)))?;
            costs.push((read, packing));
        }
        Self::under(parts.into(), costs.into(), upload_bytes, room)
    }

    /// `parts` and their `costs`, `upload_bytes` staged in all, read under `room`.
    fn under(
        parts: Arc<[Planned]>,
        costs: Arc<[(u64, u64)]>,
        upload_bytes: usize,
        room: u64,
    ) -> Result<Self, GpuError> {
        let (mut window, mut serial_bytes) = (0, 0);
        for at in 0..parts.len() {
            // A part runs alone under no room.
            let one = read_ahead(&parts[at..=at], &costs[at..=at], 0)
                .map_err(|e| refused(e.to_string()))?
                .1;
            serial_bytes = serial_bytes.max(one);
        }
        let mut at = 0;
        while at < parts.len() {
            let (n, bytes) = read_ahead(
                &parts[at..],
                &costs[at..],
                room.saturating_sub(UploadRing::HOST_MAX),
            )
            .map_err(|e| refused(e.to_string()))?;
            window = window.max(bytes);
            at += n;
        }
        let with_ring = |bytes: u64| match bytes {
            0 => Ok(0),
            _ => bytes
                .checked_add(UploadRing::HOST_MAX)
                .ok_or_else(|| refused("the parts' read-ahead and ring bytes pass u64".into())),
        };
        Ok(Self {
            parts,
            costs,
            window,
            host_bytes: with_ring(window)?,
            serial_bytes: with_ring(serial_bytes)?,
            upload_bytes,
        })
    }

    /// The parts of `plan`'s tensors that `on`, a card of it, reads from `source` under `room`.
    fn reading<S: Source>(
        source: &S,
        plan: &Plan<'_>,
        on: &OnCard,
        room: u64,
    ) -> Result<Self, GpuError> {
        let selected = model::part::Filtered {
            source,
            keep: |name: &str| plan.model.tensors.iter().any(|t| t.name == name),
        };
        Self::new(&selected, |name, _, _| on.take(name), room)
    }
}

/// The tensors a located header holds in place (`gguf::Place::File` or `gguf::Place::Stack`),
/// which the host reads where the folder stores them: a load from a part source skips such a
/// tensor the plan holds on the host, and takes its card's experts of such a stack the plan
/// splits between the card and the host ([`OnCard::of_in_place`]). A tensor the part source
/// alone makes (`gguf::Place::Made`) has no host reader.
#[derive(Debug, Default)]
pub struct InPlace(HashMap<String, Held>);

/// How a located header holds a tensor in place ([`InPlace`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Held {
    /// One run of a file.
    File,
    /// A stack of this many matrices, each read where the file keeps it.
    Stack(usize),
}

impl InPlace {
    /// The tensors `header` holds in place.
    #[must_use]
    pub fn of(header: &Split) -> InPlace {
        InPlace(
            header
                .iter_tensors()
                .filter_map(|(_, t)| {
                    let held = match header.place(&t.name)? {
                        Place::File { .. } => Held::File,
                        Place::Stack { codes_at, .. } => Held::Stack(codes_at.len()),
                        Place::Made => return None,
                    };
                    Some((t.name.clone(), held))
                })
                .collect(),
        )
    }

    /// Whether the header holds tensor `name` in place.
    #[must_use]
    pub fn holds(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }

    /// The matrices of the stack `name` the header holds in place; `None` for a tensor it holds
    /// as one file run, or not at all.
    fn stack(&self, name: &str) -> Option<usize> {
        match self.0.get(name)? {
            Held::Stack(n) => Some(*n),
            Held::File => None,
        }
    }
}

/// Card `card` of `plan` as a load from a part source reads it ([`load`]): each tensor the plan
/// holds whole on that card, in the plan's card format and resident bytes, and the card's
/// experts of a stack the plan splits between that card, other cards (a stage card and its
/// expert tier card), a host that reads the stack where a located header holds it
/// ([`InPlace`]) and the NVMe expert tier, whose arena fills from the same place, in the plan's
/// slot order; a tensor the plan uses nowhere, or reads by rows from the NVMe tier (a
/// row-gathered table, which its own reader maps where the source stores it), is skipped, as is
/// a tensor the plan holds on the host and the NVMe tier where a located header holds it in
/// place, which they read there, and a tensor or stack of which this card holds nothing, which
/// the other cards' loads read. A source fills no host set, so a tensor the plan splits
/// otherwise, or holds off the cards but the source alone makes, is refused by name.
#[derive(Clone, PartialEq)]
pub struct OnCard {
    held: HashMap<String, OnThisCard>,
    skipped: HashSet<String>,
}

/// A tensor the card holds ([`OnCard`]): the plan's card format and resident bytes, and the
/// stack's experts the card holds of a split stack (`None` the whole tensor).
#[derive(Clone, PartialEq)]
struct OnThisCard {
    format: CardFormat,
    bytes: u64,
    experts: Option<ExpertList>,
}

impl OnCard {
    /// Card `card` of `plan`, each tensor's segments checked as the type says; no tensor on the
    /// host.
    pub fn of(plan: &Plan<'_>, card: usize) -> Result<OnCard, GpuError> {
        OnCard::of_in_place(plan, card, &InPlace::default())
    }

    /// [`OnCard::of`] whose host and NVMe tier read the tensors `in_place` names where they are
    /// stored: a tensor the plan holds wholly off the cards is skipped when `in_place` holds it,
    /// and refused by name otherwise; of a stack `in_place` holds that the plan splits between
    /// this card and the host or the tier, the card takes its segment's experts
    /// ([`OnCard::take`]). A plan that pages a routed stack `in_place` does not hold, which the
    /// tier's arena could fill from nowhere the folder stores it, is refused as short on the
    /// host ([`Unpaged`]).
    pub fn of_in_place(
        plan: &Plan<'_>,
        card: usize,
        in_place: &InPlace,
    ) -> Result<OnCard, GpuError> {
        let model = plan.model;
        let made = plan.rows.iter().any(|row| {
            row.segments.iter().any(|s| s.device == Device::Nvme)
                && model.tensors.get(row.tensor).is_some_and(|t| {
                    t.role == Role::RoutedExperts && in_place.stack(&t.name).is_none()
                })
        });
        if let Some(paged) = Unpaged::of(plan).filter(|_| made) {
            return Err(GpuError::plan(WHAT, paged));
        }
        let rows = plan.rows.iter().map(|row| {
            model
                .tensors
                .get(row.tensor)
                .map(|t| (t, row.segments.as_slice(), model.experts_of(t)))
                .ok_or_else(|| refused(format!("a placement row names tensor {}", row.tensor)))
        });
        OnCard::of_rows(rows, card, in_place)
    }

    /// The tensors of `rows` (each with its segments and its stack's experts) on card `card`,
    /// the host reading the tensors `in_place` names.
    fn of_rows<'a>(
        rows: impl Iterator<Item = Result<(&'a ModelTensor, &'a [Segment], u64), GpuError>>,
        card: usize,
        in_place: &InPlace,
    ) -> Result<OnCard, GpuError> {
        let mut held = HashMap::new();
        let mut skipped = HashSet::new();
        for row in rows {
            let (t, segments, experts) = row?;
            let elsewhere = |placed: String| {
                refused(format!(
                    "{} is placed {placed}: a load from a part source holds every tensor whole \
                     on card {card}, and reads a row-gathered table from the NVMe tier",
                    t.name
                ))
            };
            match segments {
                [] => {
                    skipped.insert(t.name.clone());
                }
                [s] if s.device == Device::Card(card) => {
                    let Format::Card(format) = s.format else {
                        return Err(refused(format!(
                            "{} is on the card as {}",
                            t.name, s.format
                        )));
                    };
                    let whole = s
                        .experts
                        .as_ref()
                        .is_none_or(|e| e.ids().len() as u64 == experts);
                    if !whole {
                        return Err(elsewhere(format!(
                            "on card {card} in part, the rest of its experts elsewhere"
                        )));
                    }
                    held.insert(
                        t.name.clone(),
                        OnThisCard {
                            format,
                            bytes: s.resident_bytes,
                            experts: None,
                        },
                    );
                }
                [s] if s.device == Device::Unused
                    || (s.device == Device::Nvme && t.gathered_rows.is_some()) =>
                {
                    skipped.insert(t.name.clone());
                }
                segments if elsewhere_only(segments, card) => {
                    skipped.insert(t.name.clone());
                }
                segments if !segments.is_empty() && segments.iter().all(off_cards) => {
                    if !in_place.holds(&t.name) {
                        let on: Vec<String> =
                            segments.iter().map(|s| format!("{:?}", s.device)).collect();
                        return Err(elsewhere(format!(
                            "on {}, where the host and the NVMe tier read only a tensor a \
                             located header holds in place (a file run or a stack), and the part \
                             source makes this one",
                            on.join(" and ")
                        )));
                    }
                    skipped.insert(t.name.clone());
                }
                segments => {
                    if let Some(share) = card_share(t, segments, card, in_place)? {
                        held.insert(t.name.clone(), share);
                        continue;
                    }
                    let on: Vec<String> =
                        segments.iter().map(|s| format!("{:?}", s.device)).collect();
                    return Err(elsewhere(format!("on {}", on.join(" and "))));
                }
            }
        }
        Ok(OnCard { held, skipped })
    }

    /// The rule [`load`] plans by: a tensor the card holds whole, the card's experts of a split
    /// stack in the plan's slot order, or none; a name the plan does not hold is refused by name.
    pub fn take(&self, name: &str) -> Result<Take, String> {
        if let Some(on) = self.held.get(name) {
            Ok(on
                .experts
                .as_ref()
                .map_or(Take::Whole, |list| Take::Experts(list.clone())))
        } else if self.skipped.contains(name) {
            Ok(Take::Skip)
        } else {
            Err(format!("{name} is not a tensor of the plan"))
        }
    }

    /// The plan's card format of `t`.
    pub fn format(&self, t: &TensorInfo) -> Option<CardFormat> {
        self.held.get(&t.name).map(|on| on.format)
    }

    /// Refused by name unless `w` holds every tensor the card holds, at the plan's resident
    /// bytes: a tensor the source does not make, or makes otherwise than the plan's file did.
    pub fn check(&self, w: &Weights) -> Result<(), GpuError> {
        for (name, &OnThisCard { bytes, .. }) in &self.held {
            let got = w.get(name).map(|dw| dw.resident_bytes() as u64);
            if got != Some(bytes) {
                return Err(refused(format!(
                    "{name}: the source uploaded {got:?} device bytes, the plan holds {bytes}"
                )));
            }
        }
        Ok(())
    }
}

/// Whether `s` is on the host or the NVMe expert tier, which read an in-place stack where the
/// located header holds it.
fn off_cards(s: &Segment) -> bool {
    matches!(s.device, Device::Host | Device::Nvme)
}

/// Whether `segments` put something on a card other than `card` and nothing on `card`, the rest
/// on the host or the NVMe tier: another card's load reads its segments, and the load of a card
/// that holds a share checks the rest ([`card_share`]).
fn elsewhere_only(segments: &[Segment], card: usize) -> bool {
    let other = |s: &Segment| matches!(s.device, Device::Card(c) if c != card);
    segments.iter().any(other) && segments.iter().all(|s| other(s) || off_cards(s))
}

/// The card's share of stack `t`, whose segments are `segments`, when the plan splits it between
/// card `card`, other cards, the host and the NVMe tier and `in_place` holds it as a stack, which
/// the host reads and the tier's arena fills from where the folder stores it: one segment on the
/// card, in a card format, holding some of the stack's experts, every other segment on another
/// card, the host or the tier. `None` for segments of any other split, which the caller refuses
/// by name; a card expert past the located stack's matrices, and a card segment in no card
/// format, are refused by name.
fn card_share(
    t: &ModelTensor,
    segments: &[Segment],
    card: usize,
    in_place: &InPlace,
) -> Result<Option<OnThisCard>, GpuError> {
    let Some(matrices) = in_place.stack(&t.name) else {
        return Ok(None);
    };
    let mut on_card = segments.iter().filter(|s| s.device == Device::Card(card));
    let (Some(s), None) = (on_card.next(), on_card.next()) else {
        return Ok(None);
    };
    let elsewhere = segments.iter().filter(|s| s.device != Device::Card(card));
    let Some(list) = s.experts.as_ref() else {
        return Ok(None);
    };
    let placed = |s: &Segment| matches!(s.device, Device::Card(_)) || off_cards(s);
    if !elsewhere.clone().all(placed) || elsewhere.count() == 0 {
        return Ok(None);
    }
    let Format::Card(format) = s.format else {
        return Err(refused(format!(
            "{} is on the card as {}",
            t.name, s.format
        )));
    };
    if let Some(&e) = list.ids().iter().find(|&&e| e as usize >= matrices) {
        return Err(refused(format!(
            "{}: the plan puts expert {e} on card {card}, and the located stack holds {matrices} \
             matrices",
            t.name
        )));
    }
    Ok(Some(OnThisCard {
        format,
        bytes: s.resident_bytes,
        experts: Some(list.clone()),
    }))
}

/// Refused by name unless card format `format` lays out the part `t` (its type, its row width
/// and rows): a layout the plan's card format cannot take, before any part is read.
fn card_layout(t: &TensorInfo, format: CardFormat) -> Result<(), GpuError> {
    let rows: u64 = t.dims.iter().skip(1).product();
    let k = t.dims.first().copied().unwrap_or(0);
    format
        .resident_bytes(t.ty, k, rows)
        .map(|_| ())
        .ok_or_else(|| {
            refused(format!(
                "{}: {} {:?} has no layout in the plan's card format {format:?}",
                t.name, t.ty, t.dims
            ))
        })
}

fn refused(detail: String) -> GpuError {
    GpuError::shape(WHAT, detail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use model::placement::Role;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A plan's KV terms of no bytes.
    struct NoKv;

    impl model::placement::KvBytes for NoKv {
        fn layer_bytes(&self, _: usize, _: u64) -> u64 {
            0
        }
    }

    /// Parts are staged in plan order while later parts still read, a staged part's pages are
    /// released once, and a read's refusal stops the stage at that part. The last part's read
    /// waits, at most 10 s, for the first part's stage. A read-ahead of two parts reads four once
    /// each stage frees its part, holding two read and not yet staged, and the last part's failed
    /// release is the load's error. Under a live reading short of a second part, one part is held
    /// at a time; a part alone over it is the load's error.
    /// Mutants: staging in arrival order stages the later, faster reads first; staging once every
    /// read is in leaves the last read waiting until it is refused; a release queued twice or
    /// never changes the released list; a stage that never frees its read-ahead stalls a load,
    /// and the test aborts after 10 s; a release error dropped after the last stage returns `Ok`;
    /// a reader's or the stage's panic that leaves the other readers waiting stalls its load;
    /// admitting a part without the held parts, or reading on a `Wait` or a `Short`, holds more
    /// parts than the peaks; never reading beside a held part holds fewer; a reader's admission
    /// error not kept as the load's error names a stopped reader instead; a reads' wall left at
    /// zero falls under the longest read.
    #[test]
    fn parts_stage_in_order_as_their_reads_arrive() {
        let (finished, watch) = mpsc::channel::<()>();
        std::thread::spawn(move || {
            let wait = watch.recv_timeout(std::time::Duration::from_secs(10));
            if wait == Err(mpsc::RecvTimeoutError::Timeout) {
                eprintln!("parts_stage_in_order_as_their_reads_arrive stalled a load");
                std::process::abort();
            }
        });
        /// Signals the first part's stage, for the last part's read.
        struct Slow(Mutex<mpsc::Receiver<()>>);
        impl Source for Slow {
            type Error = std::io::Error;
            fn plan<E: std::fmt::Display>(
                &self,
                _: impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
            ) -> Result<Vec<Planned>, Self::Error> {
                unreachable!()
            }
            fn read_bytes(&self, _: &Planned) -> Result<u64, Self::Error> {
                unreachable!()
            }
            fn read(&self, p: &Planned) -> Result<model::part::Part<'_>, Self::Error> {
                assert_ne!(p.info.name, "r0", "a read panicked");
                PEAK.fetch_max(HELD.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                let i = p.info.offset;
                std::thread::sleep(std::time::Duration::from_millis(2 * (12 - i)));
                if i == 8 {
                    let first = self.0.lock().unwrap();
                    first
                        .recv_timeout(std::time::Duration::from_secs(10))
                        .map_err(|_| std::io::Error::other("part 8 read before part 0 staged"))?;
                }
                match i {
                    9 => Err(std::io::Error::other("part 9 refused")),
                    _ => Ok(model::part::Part::made(p.info.clone(), vec![i as u8], None)),
                }
            }
            fn release(&self, p: &Planned) -> Result<(), Self::Error> {
                if p.info.name == "q3" {
                    return Err(std::io::Error::other("q3 release refused"));
                }
                RELEASED.lock().unwrap().push(p.info.offset);
                Ok(())
            }
        }
        static RELEASED: Mutex<Vec<u64>> = Mutex::new(Vec::new());
        static HELD: AtomicUsize = AtomicUsize::new(0);
        static PEAK: AtomicUsize = AtomicUsize::new(0);
        let peak_of = |load: &mut dyn FnMut()| {
            HELD.store(0, Ordering::SeqCst);
            PEAK.store(0, Ordering::SeqCst);
            load();
            PEAK.load(Ordering::SeqCst)
        };
        let plan = |n: u64, window: u64, name: &str| LoadPlan {
            parts: (0..n)
                .map(|i| Planned {
                    at: None,
                    info: TensorInfo {
                        name: format!("{name}{i}"),
                        ty: GgmlType::F32,
                        dims: vec![1],
                        offset: i,
                        nbytes: 4,
                    },
                })
                .collect(),
            costs: vec![(4, 0); n as usize].into(),
            window,
            host_bytes: window + UploadRing::HOST_MAX,
            serial_bytes: 4 + UploadRing::HOST_MAX,
            upload_bytes: 0,
        };
        let room = || Ok((1 << 20, HostRead::MemAvailable));
        let (staged, first) = mpsc::channel();
        let slow = Slow(Mutex::new(first));
        let mut order = Vec::new();
        let mut stage = |p: &Planned, part: model::part::Part<'_>| {
            assert_eq!(part.spans(), [[p.info.offset as u8]]);
            if p.info.offset == 0 {
                staged.send(()).unwrap();
            }
            order.push(p.info.offset);
            HELD.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        };
        let (release, keep) = (true, false);
        let whole = plan(9, 36, "p");
        let refusing = plan(12, 48, "p");
        let turning = plan(4, 8, "q");
        // The reads' wall spans the longest read (part 0's sleep) and lies inside the load.
        let started = Instant::now();
        let reads = read_staged(&slow, &whole, room, release, &mut stage).unwrap();
        assert!(
            reads >= Duration::from_millis(24) && reads <= started.elapsed(),
            "{reads:?}"
        );
        let mut released = std::mem::take(&mut *RELEASED.lock().unwrap());
        released.sort_unstable();
        assert_eq!(released, (0..9).collect::<Vec<u64>>());
        let error = read_staged(&slow, &refusing, room, keep, &mut stage)
            .unwrap_err()
            .to_string();
        assert!(error.contains("part 9 refused"), "{error}");
        assert!(RELEASED.lock().unwrap().is_empty());
        let mut error = String::new();
        let peak = peak_of(&mut || {
            error = read_staged(&slow, &turning, room, release, &mut stage)
                .unwrap_err()
                .to_string();
        });
        assert!(error.contains("q3 release refused"), "{error}");
        assert!(
            peak == 2 || (peak == 1 && threads::pool().threads() == 1),
            "the read-ahead of 2 parts held {peak}"
        );
        // Parts 0 and 1 need their 3 B of reads and part 0's 2 B packing, over the live 3 B.
        let mut tight = plan(2, 8, "t");
        tight.costs = vec![(1, 2), (2, 0)].into();
        let short = || Ok((3, HostRead::Given));
        let peak = peak_of(&mut || {
            read_staged(&slow, &tight, short, keep, &mut stage).unwrap();
        });
        assert_eq!(peak, 1, "a live reading short of two parts held {peak}");
        let error = read_staged(&slow, &plan(1, 4, "u"), short, keep, &mut stage)
            .unwrap_err()
            .to_string();
        assert!(error.contains("part read-ahead needs 4 B"), "{error}");
        let read_panics = plan(2, 8, "r");
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            read_staged(&slow, &read_panics, room, keep, &mut stage)
        }));
        assert!(unwound.is_err());
        let stage_panics = plan(2, 8, "s");
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            read_staged(&slow, &stage_panics, room, keep, |_, _| {
                panic!("a stage panicked")
            })
        }));
        assert!(unwound.is_err());
        let want: Vec<u64> = (0..9).chain(0..9).chain(0..4).chain(0..2).collect();
        assert_eq!(order, want);
        let mut released = std::mem::take(&mut *RELEASED.lock().unwrap());
        released.sort_unstable();
        // A reader may take q3's failing release before q2's and stop with q2 still queued.
        assert!(released == [0, 1] || released == [0, 1, 2], "{released:?}");
        finished.send(()).unwrap();
    }

    /// A part reads beside the held parts only within the planned read-ahead and then the live
    /// reading, which is not read while the read-ahead holds it back; with none held it reads
    /// alone unless its own peak is over the live reading, which refuses it by name. Mutants:
    /// admitting past the read-ahead turns the `Wait` red, ignoring the live reading the `Short`,
    /// and dropping the lone part's check the refusal.
    // PIN(2026-10-28): replaces later_reads_use_their_own_peak_under_a_falling_host_room; a load
    // reads beside the held parts instead of in runs, so the live reading is checked per part.
    #[test]
    fn a_part_reads_beside_the_held_parts_within_the_read_ahead_and_the_live_room() {
        let mib = 1u64 << 20;
        let decls = [64, 8]
            .into_iter()
            .map(|n| gguf::write::TensorDecl {
                name: format!("p{n}.weight"),
                dims: vec![64, n * mib / 256],
                type_id: GgmlType::F32.as_u32(),
                nbytes: n * mib,
            })
            .collect();
        let layout = gguf::write::Layout::new(&[], decls).unwrap();
        let source =
            Split::header_only(std::path::PathBuf::from("/synthetic/falling-room"), &layout)
                .unwrap();
        let shared =
            LoadPlan::new(&source, |_, _, _| Ok::<_, String>(Take::Whole), 296 * mib).unwrap();
        assert_eq!(shared.window, 200 * mib);
        assert_eq!(shared.host_bytes, 296 * mib);
        let read = |bytes: u64| (bytes, HostRead::MemAvailable);
        shared.check(&read(288 * mib)).unwrap();
        assert!(shared.check(&read(288 * mib - 1)).is_err());
        let plan =
            LoadPlan::new(&source, |_, _, _| Ok::<_, String>(Take::Whole), 288 * mib).unwrap();
        assert_eq!(plan.window, 192 * mib);
        assert_eq!(plan.host_bytes, 288 * mib);
        plan.check(&read(300 * mib)).unwrap();
        assert!(plan.check(&read(288 * mib - 1)).is_err());
        let live = |mb: u64| move || Ok(read(mb * mib));
        let unread = || -> Result<(u64, HostRead), GpuError> { unreachable!("read-ahead first") };
        for (load, held, room, want) in [
            (&shared, 0..1, 200, Admit::Read),
            (&shared, 0..1, 199, Admit::Short),
            (&plan, 1..1, 24, Admit::Read),
            (&shared, 0..0, 192, Admit::Read),
        ] {
            assert_eq!(load.admits(held, live(room)).unwrap(), want);
        }
        assert_eq!(plan.admits(0..1, unread).unwrap(), Admit::Wait);
        let error = plan
            .admits(1..1, || Ok((23 * mib, HostRead::Given)))
            .unwrap_err()
            .to_string();
        assert!(error.contains("needs 25165824 B"), "{error}");
        assert!(
            error.contains("the host room is 24117248 B (given by BLOOMERY_HOST_ROOM"),
            "{error}"
        );
    }

    #[test]
    #[ignore = "needs a CUDA card"]
    fn hw_a_parts_load_refuses_a_short_host_before_reading() {
        struct Owned;
        impl Source for Owned {
            type Error = std::io::Error;
            fn plan<E: std::fmt::Display>(
                &self,
                rule: impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
            ) -> Result<Vec<Planned>, Self::Error> {
                let info = TensorInfo {
                    name: "owned.weight".into(),
                    ty: GgmlType::BF16,
                    dims: vec![64, 262144],
                    offset: 0,
                    nbytes: 32 << 20,
                };
                rule(&info.name, info.ty, &info.dims)
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                Ok(vec![Planned { at: None, info }; 256])
            }
            fn read_bytes(&self, p: &Planned) -> Result<u64, Self::Error> {
                Ok(p.info.nbytes * 2)
            }
            fn read(&self, _: &Planned) -> Result<model::part::Part<'_>, Self::Error> {
                Err(std::io::Error::other("read before host admission"))
            }
        }
        let plan =
            LoadPlan::new(&Owned, |_, _, _| Ok::<_, String>(Take::Whole), 128 << 20).unwrap();
        assert_eq!(plan.window + UploadRing::HOST_MAX, plan.serial_bytes);
        assert_eq!(plan.host_bytes, (192 + 3 * 32) << 20);
        let gpu = crate::Gpu::with_device(0).unwrap();
        let room = (128 << 20, HostRead::MemAvailable);
        let error = load_plan(gpu.stream(), &Owned, &plan, |_| None, false, &room)
            .err()
            .expect("a short host must refuse the load")
            .to_string();
        assert!(
            error.contains("part read-ahead needs 301989888 B"),
            "{error}"
        );
        assert!(
            error.contains("the host room is 134217728 B (MemAvailable"),
            "{error}"
        );
        assert!(plan.check(&(plan.host_bytes, HostRead::Given)).is_ok());
        assert!(plan.check(&(plan.host_bytes - 1, HostRead::Given)).is_err());
    }

    /// A parts load stages every part and drops each part's source pages only under `release`.
    /// Mutant: the load passing a fixed `release` to its reader loop changes the released list of
    /// one of the two loads.
    #[test]
    #[ignore = "needs a CUDA card"]
    fn hw_a_parts_load_releases_its_parts_only_under_release() {
        struct Small(Mutex<Vec<String>>);
        impl Source for Small {
            type Error = std::io::Error;
            fn plan<E: std::fmt::Display>(
                &self,
                _: impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
            ) -> Result<Vec<Planned>, Self::Error> {
                let part = |name: &str| Planned {
                    at: None,
                    info: TensorInfo {
                        name: name.into(),
                        ty: GgmlType::F32,
                        dims: vec![64, 1],
                        offset: 0,
                        nbytes: 256,
                    },
                };
                Ok(vec![part("a.weight"), part("b.weight")])
            }
            fn read_bytes(&self, p: &Planned) -> Result<u64, Self::Error> {
                Ok(p.info.nbytes)
            }
            fn read(&self, p: &Planned) -> Result<model::part::Part<'_>, Self::Error> {
                Ok(model::part::Part::made(p.info.clone(), vec![0; 256], None))
            }
            fn release(&self, p: &Planned) -> Result<(), Self::Error> {
                self.0.lock().unwrap().push(p.info.name.clone());
                Ok(())
            }
        }
        let source = Small(Mutex::new(Vec::new()));
        let room = (1 << 30, HostRead::Given);
        let plan = LoadPlan::new(&source, |_, _, _| Ok::<_, String>(Take::Whole), room.0).unwrap();
        let gpu = crate::Gpu::with_device(0).unwrap();
        for (release, want) in [(false, vec![]), (true, vec!["a.weight", "b.weight"])] {
            let (w, _) = load_plan(gpu.stream(), &source, &plan, |_| None, release, &room).unwrap();
            assert_eq!(w.names().collect::<Vec<_>>(), ["a.weight", "b.weight"]);
            let mut released = std::mem::take(&mut *source.0.lock().unwrap());
            released.sort_unstable();
            assert_eq!(released, want);
        }
    }

    /// The rule a load from a part source plans by: a tensor whole on the card is taken in the
    /// plan's card format, an unused tensor, a row-gathered table on the NVMe tier and a tensor
    /// whole on another card (that card's load reads it) are skipped, and a tensor the plan
    /// places on the host, split, or in part on the card is refused by name, as is a name the
    /// plan does not hold; the check refuses weights that miss a held tensor. Mutant: the
    /// other-card arm dropped — the stage's attention on the tier card's load is refused "on
    /// Card(0)".
    #[test]
    fn on_card_takes_the_card_s_tensors_and_refuses_the_rest() {
        let tensor = |name: &str, role, gathered_rows| ModelTensor {
            name: name.into(),
            shard: 0,
            layer: Some(0),
            role,
            ty: GgmlType::BF16,
            dims: vec![64, 4, 8],
            file_bytes: 4096,
            gathered_rows,
        };
        let seg = |device, format, experts: Option<Vec<u32>>| Segment {
            device,
            format,
            experts: experts.map(|ids| ExpertList::new(ids, 8).unwrap()),
            resident_bytes: 4096,
        };
        let bf16 = Format::Card(CardFormat::Bf16Raw);
        let held = tensor("a.weight", Role::Attention, None);
        let stack = tensor("x.weight", Role::RoutedExperts, None);
        let table = tensor("t.weight", Role::EngramTable, Some(1));
        let unused = tensor("u.weight", Role::Attention, None);
        let stage = tensor("s.weight", Role::Attention, None);
        let ok = [
            (&held, vec![seg(Device::Card(0), bf16, None)]),
            (
                &stack,
                vec![seg(Device::Card(0), bf16, Some((0..8).collect()))],
            ),
            (&table, vec![seg(Device::Nvme, Format::NvmeFile, None)]),
            (&unused, vec![seg(Device::Unused, Format::Unused, None)]),
            (&stage, vec![seg(Device::Card(1), bf16, None)]),
        ];
        let of = |r: &[(&ModelTensor, Vec<Segment>)]| {
            OnCard::of_rows(
                r.iter().map(|(t, s)| Ok((*t, s.as_slice(), 8))),
                0,
                &InPlace::default(),
            )
        };
        let on = of(&ok).unwrap();
        assert!(matches!(on.take("a.weight"), Ok(Take::Whole)));
        assert!(matches!(on.take("x.weight"), Ok(Take::Whole)));
        assert!(matches!(on.take("t.weight"), Ok(Take::Skip)));
        assert!(matches!(on.take("u.weight"), Ok(Take::Skip)));
        assert!(matches!(on.take("s.weight"), Ok(Take::Skip)));
        assert!(on.take("z.weight").unwrap_err().contains("z.weight"));
        let info = |name: &str| TensorInfo {
            name: name.into(),
            dims: vec![64],
            ty: GgmlType::BF16,
            offset: 0,
            nbytes: 128,
        };
        assert_eq!(on.format(&info("a.weight")), Some(CardFormat::Bf16Raw));
        assert_eq!(on.format(&info("t.weight")), None);
        let e = on.check(&Weights::default()).unwrap_err().to_string();
        assert!(e.contains("uploaded None device bytes"), "{e}");
        let host_table = tensor("t.weight", Role::EngramTable, Some(1));
        for (t, segments, named) in [
            (
                &held,
                vec![seg(Device::Host, Format::HostFile, None)],
                "on Host",
            ),
            (
                &host_table,
                vec![seg(Device::Host, Format::HostFile, None)],
                "on Host",
            ),
            (
                &held,
                vec![
                    seg(Device::Card(0), bf16, None),
                    seg(Device::Host, Format::HostFile, None),
                ],
                "on Card(0) and Host",
            ),
            (
                &stack,
                vec![seg(Device::Card(0), bf16, Some((0..6).collect()))],
                "in part",
            ),
        ] {
            let e = of(&[(t, segments)]).err().expect("refused").to_string();
            assert!(e.contains(&t.name) && e.contains(named), "{e}");
        }
    }

    /// A tensor the plan holds wholly on the host is skipped when the located header holds it in
    /// place, which the host reads there, and refused by name when the part source alone makes
    /// it; a tensor the header holds as one file run that the plan splits between the card and
    /// the host is refused by name. Mutant: the host arm dropped — the in-place embedding is
    /// refused "on Host" and the folder's load never starts.
    #[test]
    fn on_card_skips_the_host_s_in_place_tensors_and_refuses_made_ones() {
        let tensor = |name: &str| ModelTensor {
            name: name.into(),
            shard: 0,
            layer: None,
            role: Role::TokenEmbedding,
            ty: GgmlType::BF16,
            dims: vec![64, 8],
            file_bytes: 1024,
            gathered_rows: None,
        };
        let seg = |device, format, experts: Option<Vec<u32>>| Segment {
            device,
            format,
            experts: experts.map(|ids| ExpertList::new(ids, 8).unwrap()),
            resident_bytes: 1024,
        };
        let host = || vec![seg(Device::Host, Format::HostFile, None)];
        let (stored, made) = (tensor("token_embd.weight"), tensor("made.weight"));
        let in_place = InPlace(
            [("token_embd.weight".to_string(), Held::File)]
                .into_iter()
                .collect(),
        );
        let of = |r: &[(&ModelTensor, Vec<Segment>)]| {
            OnCard::of_rows(
                r.iter().map(|(t, s)| Ok((*t, s.as_slice(), 8))),
                0,
                &in_place,
            )
        };
        let on = of(&[(&stored, host())]).unwrap();
        assert!(matches!(on.take("token_embd.weight"), Ok(Take::Skip)));
        let e = of(&[(&made, host())]).err().expect("refused").to_string();
        assert!(
            e.contains("made.weight") && e.contains("on Host") && e.contains("makes"),
            "{e}"
        );
        let bf16 = Format::Card(CardFormat::Bf16Raw);
        let split = vec![
            seg(Device::Card(0), bf16, Some((0..6).collect())),
            seg(Device::Host, Format::HostFile, Some((6..8).collect())),
        ];
        let e = of(&[(&stored, split)]).err().expect("refused").to_string();
        assert!(e.contains("on Card(0) and Host"), "{e}");
    }

    /// Of a stack the located header holds in place that the plan splits between the card, other
    /// cards (plan (b′)'s stage and expert tier cards), the host and the NVMe tier, the card takes
    /// its segment's experts in the plan's slot order, at that segment's card format and resident
    /// bytes, the other cards' loads take theirs, and the host and the tier read the rest where
    /// the folder stores them; a card that holds none of the stack skips it, also when the host
    /// and the tier hold all of it. Refused by name: a card expert past the located stack's
    /// matrices, a card segment in no card format, two segments on this card, and a split of a
    /// stack the header does not hold in place (a GGUF's, or a folder's whose part source makes
    /// it). Red against the load before the split arm, which refused every split "on Card(0)
    /// and Host"; mutant: other cards left out of the split arm — the stage's share of a tiered
    /// split is refused "on Card(0) and Card(1) and Host"; FAIL-first: a share beside the tier
    /// was refused "on Card(0) and Host and Nvme", and a stack the host and the tier hold whole
    /// "on Host and Nvme".
    #[test]
    fn on_card_takes_its_experts_of_a_stack_split_with_the_in_place_host() {
        let name = "stack.weight";
        let stack = ModelTensor {
            name: name.into(),
            shard: 0,
            layer: Some(3),
            role: Role::RoutedExperts,
            ty: GgmlType::BF16,
            dims: vec![64, 4, 8],
            file_bytes: 4096,
            gathered_rows: None,
        };
        let seg = |device, format, ids: &[u32], experts: u64| Segment {
            device,
            format,
            experts: Some(ExpertList::new(ids.to_vec(), experts).unwrap()),
            resident_bytes: 512 * ids.len() as u64,
        };
        let card = Format::Card(CardFormat::Bf16Raw);
        let host = Format::HostFile;
        let in_place =
            |n: usize| InPlace([(name.to_string(), Held::Stack(n))].into_iter().collect());
        let of = |t: &ModelTensor, segments: Vec<Segment>, in_place: &InPlace| {
            OnCard::of_rows(
                std::iter::once(Ok((t, segments.as_slice(), t.dims[2]))),
                0,
                in_place,
            )
        };
        let split = || {
            vec![
                seg(Device::Card(0), card, &[1, 4, 6], 8),
                seg(Device::Host, host, &[0, 2, 3, 5, 7], 8),
            ]
        };
        let on = of(&stack, split(), &in_place(8)).unwrap();
        let want = ExpertList::new(vec![1, 4, 6], 8).unwrap();
        assert_eq!(on.take(name), Ok(Take::Experts(want)));
        let info = TensorInfo {
            name: name.into(),
            dims: vec![64, 4, 3],
            ty: GgmlType::BF16,
            offset: 0,
            nbytes: 1536,
        };
        assert_eq!(on.format(&info), Some(CardFormat::Bf16Raw));
        let e = on.check(&Weights::default()).unwrap_err().to_string();
        assert!(e.contains(name) && e.contains("the plan holds 1536"), "{e}");
        let two_hosts = vec![
            seg(Device::Card(0), card, &[0], 8),
            seg(Device::Host, host, &[1, 2, 3], 8),
            seg(Device::Host, host, &[4, 5, 6, 7], 8),
        ];
        let on = of(&stack, two_hosts, &in_place(8)).unwrap();
        let want = ExpertList::new(vec![0], 8).unwrap();
        assert_eq!(on.take(name), Ok(Take::Experts(want)));
        let tiered = || {
            vec![
                seg(Device::Card(0), card, &[1, 4], 8),
                seg(Device::Card(1), card, &[6], 8),
                seg(Device::Host, host, &[0, 2, 3, 5, 7], 8),
            ]
        };
        let on = of(&stack, tiered(), &in_place(8)).unwrap();
        let want = ExpertList::new(vec![1, 4], 8).unwrap();
        assert_eq!(on.take(name), Ok(Take::Experts(want)));
        let on = OnCard::of_rows(
            std::iter::once(Ok((&stack, tiered().as_slice(), 8))),
            1,
            &in_place(8),
        )
        .unwrap();
        let want = ExpertList::new(vec![6], 8).unwrap();
        assert_eq!(on.take(name), Ok(Take::Experts(want)));
        let stage_only = [
            seg(Device::Card(0), card, &[1, 4], 8),
            seg(Device::Host, host, &[0, 2, 3, 5, 6, 7], 8),
        ];
        let on = OnCard::of_rows(
            std::iter::once(Ok((&stack, stage_only.as_slice(), 8))),
            1,
            &in_place(8),
        )
        .unwrap();
        assert_eq!(on.take(name), Ok(Take::Skip));
        let nvme = Format::NvmeFile;
        let paged = || {
            vec![
                seg(Device::Card(0), card, &[1, 4], 8),
                seg(Device::Host, host, &[0, 2, 3], 8),
                seg(Device::Nvme, nvme, &[5, 6, 7], 8),
            ]
        };
        let on = of(&stack, paged(), &in_place(8)).unwrap();
        let want = ExpertList::new(vec![1, 4], 8).unwrap();
        assert_eq!(on.take(name), Ok(Take::Experts(want)));
        let off_cards = vec![
            seg(Device::Host, host, &[0, 1, 2, 3], 8),
            seg(Device::Nvme, nvme, &[4, 5, 6, 7], 8),
        ];
        let on = of(&stack, off_cards, &in_place(8)).unwrap();
        assert_eq!(on.take(name), Ok(Take::Skip));
        for (segments, in_place, says) in [
            (
                split(),
                in_place(6),
                "expert 6 on card 0, and the located stack holds 6",
            ),
            (
                vec![
                    seg(Device::Card(0), host, &[1, 4, 6], 8),
                    seg(Device::Host, host, &[0, 2, 3, 5, 7], 8),
                ],
                in_place(8),
                "on the card as host_file",
            ),
            // PIN(2026-11-01): a located stack in place on the card and the NVMe tier ("on Card(0)
            // and Nvme") is no longer refused: the tier reads the rest where the folder stores it.
            (paged(), InPlace::default(), "on Card(0) and Host and Nvme"),
            (
                vec![
                    seg(Device::Card(0), card, &[1, 4], 8),
                    seg(Device::Card(0), card, &[6], 8),
                    seg(Device::Host, host, &[0, 2, 3, 5, 7], 8),
                ],
                in_place(8),
                "on Card(0) and Card(0) and Host",
            ),
            (split(), InPlace::default(), "on Card(0) and Host"),
        ] {
            let e = of(&stack, segments, &in_place)
                .err()
                .expect("refused")
                .to_string();
            assert!(e.contains(name) && e.contains(says), "{says}: {e}");
        }
    }

    /// A part whose type and dims the plan's card format lays out passes; one it does not is
    /// refused by name before any part is read. Mutant: `card_layout` passes every part — the
    /// refusal comes from the upload of that part, after the parts before it moved.
    #[test]
    fn a_layout_the_card_format_cannot_take_is_refused_by_name() {
        let t = |ty, dims: Vec<u64>| TensorInfo {
            name: "stack.weight".into(),
            nbytes: 0,
            ty,
            dims,
            offset: 0,
        };
        assert!(
            card_layout(
                &t(GgmlType::NVFP4, vec![256, 4, 3]),
                CardFormat::Nvfp4Planes
            )
            .is_ok()
        );
        let e = card_layout(
            &t(GgmlType::NVFP4, vec![256, 4, 3]),
            CardFormat::Mxfp8Planes,
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("stack.weight") && e.contains("Mxfp8Planes"),
            "{e}"
        );
    }

    /// A placed source plans a card's parts once for the plans that hold the same tensors on it,
    /// as a context search asks of plans at other contexts and rooms and the load then reads at
    /// its own room, and again for a plan whose card tensors differ; another room takes its
    /// read-ahead again; a churn pool beside the plan leaves the reads their room. Mutants: the
    /// last read never reused, so a second ask plans the parts again; the last read reused at
    /// another room, or its read-ahead not taken again, so a room for one part reads ahead as far
    /// as a room for both; `OnCard` left out of the key, so the larger tensor reuses the smaller
    /// one's read; the reads' room less the pool, so they read one part at a time.
    #[test]
    fn a_placed_source_plans_a_cards_parts_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        use model::placement::{ModelTensors, plan_with, workstation};

        use crate::model::PartsLoad;

        struct Counted(AtomicUsize);
        impl Source for Counted {
            type Error = std::io::Error;
            fn plan<E: std::fmt::Display>(
                &self,
                rule: impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
            ) -> Result<Vec<Planned>, Self::Error> {
                self.0.fetch_add(1, Ordering::Relaxed);
                rule("output.weight", GgmlType::Q8_0, &[256, 16])
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                // Two halves of the rows, so a room can hold one part or both.
                let half = |rows: std::ops::Range<u64>| Planned {
                    at: Some(model::part::At::Along(1, rows)),
                    info: TensorInfo {
                        name: "output.weight".into(),
                        ty: GgmlType::Q8_0,
                        dims: vec![256, 8],
                        offset: 0,
                        nbytes: 2176,
                    },
                };
                Ok(vec![half(0..8), half(8..16)])
            }
            fn read_bytes(&self, p: &Planned) -> Result<u64, Self::Error> {
                Ok(p.info.nbytes)
            }
            fn read(&self, _: &Planned) -> Result<model::part::Part<'_>, Self::Error> {
                Err(std::io::Error::other("no read"))
            }
        }
        let model_of = |rows: u64| ModelTensors {
            tensors: vec![ModelTensor {
                name: "output.weight".into(),
                shard: 0,
                layer: None,
                role: Role::Head,
                ty: GgmlType::Q8_0,
                dims: vec![256, rows],
                file_bytes: 272 * rows,
                gathered_rows: None,
            }],
            layers: 1,
            experts: 8,
            experts_used: 2,
            bf16_words: |_| false,
        };
        let model = model_of(16);
        let larger = model_of(32);
        let machine = workstation::plan_a(1);
        let plan = |ctx| plan_with(&model, &machine, ctx, &NoKv, None).expect("a plan");
        let load = PlacedSource {
            source: Counted(AtomicUsize::new(0)),
            in_place: InPlace::default(),
            last: Mutex::default(),
        };
        let least = load.host_bytes(&plan(4096), 0, 0).unwrap();
        assert!(least > 0);
        assert_eq!(load.host_bytes(&plan(8192), 0, 0).unwrap(), least);
        assert_eq!(load.source.0.load(Ordering::Relaxed), 1);
        let (_, read) = load.reading(&plan(4096), 0, least).unwrap();
        assert_eq!(read.host_bytes, least, "the load's read at its peak");
        let (_, small) = load.reading(&plan(4096), 0, UploadRing::HOST_MAX).unwrap();
        let (_, large) = load.reading(&plan(4096), 0, 1 << 30).unwrap();
        assert!(
            small.window < large.window,
            "a room for one part reads ahead {} B, a room for both {} B",
            small.window,
            large.window
        );
        assert_eq!(load.source.0.load(Ordering::Relaxed), 1);
        // The need plans the reads in the room its other terms leave: a churn pool, read in once
        // the reads end, takes none of it.
        let base = workstation::HostNeed::of(&plan(4096), 0).bytes();
        let (need, peak) = load
            .host_need(&plan(4096), 0, large.host_bytes, base + large.host_bytes)
            .unwrap();
        assert_eq!(peak, large.host_bytes);
        assert_eq!(need.read_ahead, large.window);
        // A plan that holds other bytes of the tensor on the card is planned again.
        let other = plan_with(&larger, &machine, 4096, &NoKv, None).expect("a plan");
        load.host_bytes(&other, 0, 1 << 30).unwrap();
        assert_eq!(load.source.0.load(Ordering::Relaxed), 2);
    }

    /// A plan that pages routed experts to the NVMe expert tier is refused as short on the host,
    /// so a context or slot search reads it as a plan to fit below: the paged bytes, the host
    /// need with them on the host, and the room the plan was made at. A plan that pages none is
    /// not, nor one whose paged stacks a located header holds in place, which the tier's arena
    /// fills from where the folder stores them. Mutants: the paged stack refused by name only
    /// (no `Unpaged` in the chain); the need left without the paged experts, under the room;
    /// every paged plan refused, the in-place one too.
    #[test]
    fn a_paged_plan_is_short_on_the_host() {
        use model::placement::workstation::{self, HostNeed, HostRead};
        use model::placement::{
            ArenaFloor, ModelTensors, PlanLevers, expert_nvme_tier, plan_host_routed,
        };

        // Eight layers of 256 experts: a stack passes the NVMe arena's floor of its layer.
        let (k, rows, experts, layers) = (640, 2560, 256, 8);
        let stack = k * rows * experts / 32 * 34;
        let model = ModelTensors {
            tensors: (0..layers)
                .map(|l| ModelTensor {
                    name: format!("layer.{l}.routed.weight"),
                    shard: 0,
                    layer: Some(l),
                    role: Role::RoutedExperts,
                    ty: GgmlType::Q8_0,
                    dims: vec![k, rows, experts],
                    file_bytes: stack,
                    gathered_rows: None,
                })
                .collect(),
            layers,
            experts,
            experts_used: 2,
            bf16_words: |_| false,
        };
        let machine = workstation::plan_on(workstation::A6000, layers);
        let mut plan =
            plan_host_routed(&model, &machine, 4096, &NoKv, &PlanLevers::default()).expect("plans");
        let unpaged = |plan: &Plan<'_>| {
            OnCard::of(plan, 0)
                .err()
                .and_then(|e| std::error::Error::source(&e).map(ToString::to_string))
        };
        assert_eq!(Unpaged::of(&plan), None);
        assert_eq!(unpaged(&plan), None, "a plan that pages nothing");
        let floor = ArenaFloor::of(&plan).expect("the arena's floor").bytes;
        let room = HostNeed::of(&plan, 0).bytes() - 2 * stack + floor;
        let room = plan.plan_at(&(room, HostRead::Given));
        expert_nvme_tier(&mut plan, room, None).expect("the split");
        let paged = plan.host.nvme_expert_bytes;
        assert!(paged > 0, "the room pages experts");
        let e = OnCard::of(&plan, 0).err().expect("a paged plan is refused");
        let GpuError::Plan { source, .. } = &e else {
            panic!("{e}");
        };
        let refusal = source
            .downcast_ref::<Unpaged>()
            .expect("refused as unpaged");
        assert_eq!(refusal.paged, paged);
        assert_eq!(refusal.room, room);
        assert_eq!(
            refusal.need,
            HostNeed {
                experts: plan.host.expert_bytes + paged,
                arena: 0,
                ..HostNeed::of(&plan, 0)
            }
            .bytes()
        );
        let short = refusal.short();
        assert!(short.need > short.room, "{refusal}");
        assert_eq!(unpaged(&plan), Some(refusal.to_string()));
        let in_place = InPlace(
            model
                .tensors
                .iter()
                .map(|t| (t.name.clone(), Held::Stack(experts as usize)))
                .collect(),
        );
        assert!(
            OnCard::of_in_place(&plan, 0, &in_place).is_ok(),
            "a plan that pages stacks held in place"
        );
    }

    /// A folder's plan that pages experts holds its load's need: the plan reserves the least its
    /// card's parts read ([`least_reads`] as `workstation::PART_READS`), the split keeps them
    /// room, and at a 4 GiB room (an emulated small host, its experts in place on the host and
    /// the NVMe tier, a dense head on the card) the load's need, its read-ahead counted at the
    /// room it reads, is the plan's to the byte and within the room. The same plan made without
    /// the reserve pages less and its load is short of the room by the reads. FAIL-first: the
    /// plan's need left out the reads, so every paged folder load was refused ("part read-ahead").
    #[test]
    fn a_paged_folder_plan_holds_its_loads_reads() {
        use model::placement::workstation::{self, HostNeed, HostRead};
        use model::placement::{ModelTensors, PlanLevers, expert_nvme_tier, plan_host_routed};

        use crate::model::PartsLoad;

        // Twelve layers of 256 Q8_0 experts, 5.3 GB on the host; the head, 285 MB, on the card.
        let (k, rows, experts, layers) = (640, 2560, 256, 12);
        let stack = k * rows * experts / 32 * 34;
        let head = ModelTensor {
            name: "output.weight".into(),
            shard: 0,
            layer: None,
            role: Role::Head,
            ty: GgmlType::Q8_0,
            dims: vec![4096, 65536],
            file_bytes: 4096 * 65536 / 32 * 34,
            gathered_rows: None,
        };
        let model = ModelTensors {
            tensors: (0..layers)
                .map(|l| ModelTensor {
                    name: format!("layer.{l}.routed.weight"),
                    shard: 0,
                    layer: Some(l),
                    role: Role::RoutedExperts,
                    ty: GgmlType::Q8_0,
                    dims: vec![k, rows, experts],
                    file_bytes: stack,
                    gathered_rows: None,
                })
                .chain([head.clone()])
                .collect(),
            layers,
            experts,
            experts_used: 2,
            bf16_words: |_| false,
        };
        struct Head(TensorInfo);
        impl Source for Head {
            type Error = std::io::Error;
            fn plan<E: std::fmt::Display>(
                &self,
                rule: impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
            ) -> Result<Vec<Planned>, Self::Error> {
                rule(&self.0.name, self.0.ty, &self.0.dims)
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                Ok(vec![Planned {
                    at: None,
                    info: self.0.clone(),
                }])
            }
            fn read_bytes(&self, p: &Planned) -> Result<u64, Self::Error> {
                Ok(p.info.nbytes)
            }
            fn read(&self, _: &Planned) -> Result<model::part::Part<'_>, Self::Error> {
                Err(std::io::Error::other("no read"))
            }
        }
        let source = Head(TensorInfo {
            name: head.name.clone(),
            ty: head.ty,
            dims: head.dims.clone(),
            offset: 0,
            nbytes: head.file_bytes,
        });
        let in_place = InPlace(
            (0..layers)
                .map(|l| {
                    (
                        format!("layer.{l}.routed.weight"),
                        Held::Stack(experts as usize),
                    )
                })
                .collect(),
        );
        let given = (4 << 30, HostRead::Given);
        let split = |machine| {
            let mut plan = plan_host_routed(&model, machine, 4096, &NoKv, &PlanLevers::default())
                .expect("plans");
            let room = plan.plan_at(&given);
            expert_nvme_tier(&mut plan, room, None).expect("the split");
            (plan, room)
        };
        let load = PlacedSource {
            source,
            in_place,
            last: Mutex::default(),
        };
        let small = workstation::plan_on(workstation::A6000, layers);
        let (bare, room) = split(&small);
        assert!(bare.host.nvme_expert_bytes > 0, "the room pages experts");
        let short = load.host_need(&bare, 0, 0, room).expect("the need").0;
        let reads = least_reads(&load.source, &load.in_place, &bare, 0).expect("the least reads");
        assert_eq!(reads, load.host_bytes(&bare, 0, 0).expect("the reads"));
        assert!(HostNeed::of(&bare, 0).bytes() <= room);
        assert!(short.bytes() > room, "{short:?} in a room of {room} B");
        let mut machine = small.clone();
        workstation::reserve_part_reads(&mut machine, reads);
        let (plan, room) = split(&machine);
        let planned = HostNeed::of(&plan, 0);
        let (need, peak) = load.host_need(&plan, 0, 0, room).expect("the need");
        assert_eq!(planned.read_ahead, reads);
        assert_eq!(peak, reads);
        assert_eq!(need.bytes(), planned.bytes());
        assert!(need.bytes() <= room, "{need:?} in a room of {room} B");
        assert!(plan.host.nvme_expert_bytes > bare.host.nvme_expert_bytes);
    }
}
