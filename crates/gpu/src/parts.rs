//! The one load of a part source onto a card (`model::part::Source`: a GGUF file or a
//! checkpoint folder). The caller's rule says which tensors the card holds, whole, the part of
//! a rank, or a stack's listed experts, as the source plans them; every planned part's card
//! format is checked before any is read; then parts are read a run at a time and staged in
//! order as they arrive through one staging ring, which one synchronize ends ([`load`]). A rank of a tensor-parallel load of N
//! ranks passes its family's cut; a one-card load passes its placement plan's card
//! ([`OnCard`]), and a folder whose host reads tensors where its located header holds them
//! passes those ([`InPlace`]): the card takes its share of a stack the plan splits between it
//! and that host, the host reads the rest where the folder stores it.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;

use cuda_core::CudaStream;
use gguf::{GgmlType, Place, Split, TensorInfo};
use model::part::{Part, Planned, Source, Take, read_ahead};
use model::placement::workstation::{HostRead, host_available_read};
use model::placement::{CardFormat, Device, ExpertList, Format, ModelTensor, Plan, Segment};

use crate::GpuError;
use crate::upload::{UploadRing, pool_map};
use crate::weights::{TensorPart, Weights, part_format, stage_parts};

/// What the errors name.
const WHAT: &str = "parts::load";

/// The parts selected by `rule`, read in runs bounded by owned read and packing bytes and staged
/// in plan order while the run still reads ([`read_staged`]). Oversized parts run alone. The
/// placed caller reserves `LoadPlan::host_bytes` in its host need. Each run is checked against
/// the live host reading and, when it does not fit, read in smaller runs down to one part at a
/// time; a part alone over the reading is refused. A refusal drains pending copies before freeing
/// device buffers. Under `release`, each run's source pages are dropped after staging.
pub fn load<S: Source + Sync, E: std::fmt::Display>(
    stream: &CudaStream,
    source: &S,
    rule: impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
    card: impl Fn(&TensorInfo) -> Option<CardFormat>,
    release: bool,
) -> Result<Weights, GpuError> {
    let reading = host_available_read().map_err(refused)?;
    let plan = LoadPlan::new(source, rule, reading.0)?;
    load_plan(stream, source, &plan, card, release, &reading)
}

fn load_plan<S: Source + Sync>(
    stream: &CudaStream,
    source: &S,
    plan: &LoadPlan,
    card: impl Fn(&TensorInfo) -> Option<CardFormat>,
    release: bool,
    room: &(u64, HostRead),
) -> Result<Weights, GpuError> {
    plan.check(room)?;
    for p in &plan.parts {
        if let Some(format) = card(&p.info) {
            card_layout(&p.info, format)?;
        }
    }
    stream.context().bind_to_thread()?;
    let mut staged = BTreeMap::new();
    // Dropped before the staged device buffers: draining the ring protects refused loads.
    let mut ring = UploadRing::new(stream, plan.upload_bytes)?;
    for run in plan.checked_runs(|| host_available_read().map_err(refused)) {
        let run = run?;
        let mut done = Vec::with_capacity(run.len());
        read_staged(source, run, |p, part| {
            let one = TensorPart {
                info: part.info.clone(),
                spans: part.spans(),
                scale: part.scale.clone(),
            };
            let held = |name: &str| staged.contains_key(name);
            let added = stage_parts(stream, &mut ring, &[one], held, &card)?;
            staged.extend(added);
            done.push((p, part));
            Ok(())
        })?;
        let released = pool_map(done, |(p, part)| {
            drop(part);
            match release {
                true => source.release(p).map_err(|e| e.to_string()),
                false => Ok(()),
            }
        });
        released
            .into_iter()
            .collect::<Result<(), _>>()
            .map_err(refused)?;
    }
    ring.finish(stream)?;
    Ok(Weights::of_staged(staged))
}

/// `run`'s parts read on threads of their own while `stage` takes each, in run order, as soon as
/// it and every part before it are read, so the staging of a run overlaps its reads. Staging
/// packs and copies over the pool, which is not reentrant, so the readers are not the pool's;
/// each first sets its mask to the pool's cpus, since a spawned thread inherits the caller's pin.
/// The first refusal, a reader's, a read's or the stage's, stops the readers taking more parts.
fn read_staged<'s, 'r, S: Source + Sync>(
    source: &'s S,
    run: &'r [Planned],
    mut stage: impl FnMut(&'r Planned, Part<'s>) -> Result<(), GpuError>,
) -> Result<(), GpuError> {
    let pool = threads::pool();
    let cpus: Vec<u32> = pool.topology().concat();
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let (tx, rx) = mpsc::channel();
    std::thread::scope(|scope| {
        for _ in 0..pool.threads().clamp(1, run.len().max(1)) {
            let tx = tx.clone();
            let (next, stop, cpus) = (&next, &stop, &cpus);
            scope.spawn(move || {
                let placed = os::set_mask(cpus).map_err(|e| format!("a part reader's cpus: {e}"));
                while !stop.load(Ordering::Relaxed) {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(p) = run.get(i) else { break };
                    let read = placed
                        .clone()
                        .map(|()| (i, source.read(p).map_err(|e| e.to_string())));
                    if tx.send(read).is_err() || placed.is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        let mut read: Vec<Option<Result<Part<'s>, String>>> = run.iter().map(|_| None).collect();
        let mut at = 0;
        let staged = (|| {
            while at < run.len() {
                let (i, part) = rx
                    .recv()
                    .map_err(|_| refused("a part reader stopped".into()))?
                    .map_err(refused)?;
                read[i] = Some(part);
                while let Some(part) = read.get_mut(at).and_then(Option::take) {
                    stage(&run[at], part.map_err(refused)?)?;
                    at += 1;
                }
            }
            Ok(())
        })();
        stop.store(true, Ordering::Relaxed);
        drop(rx);
        staged
    })
}

/// A placed source shares one admission and upload implementation across model components.
pub(crate) struct PlacedSource<S> {
    pub(crate) source: S,
    pub(crate) in_place: InPlace,
}

impl<S: Source + Send + Sync> crate::model::PartsLoad for PlacedSource<S> {
    fn host_bytes(&self, plan: &Plan<'_>, card: usize, room: u64) -> Result<u64, GpuError> {
        Ok(
            LoadPlan::placed(&self.source, plan, card, &self.in_place, room)?
                .1
                .host_bytes,
        )
    }

    fn load(
        &self,
        stream: &CudaStream,
        plan: &Plan<'_>,
        card: usize,
        release: bool,
        room: u64,
    ) -> Result<Weights, GpuError> {
        load_placed_in_place(
            stream,
            &self.source,
            plan,
            card,
            release,
            &self.in_place,
            room,
        )
    }
}

/// Parts, their read costs, their bounded parallel reads, the peak owned host bytes reserved
/// before loading and the least peak, one part at a time.
struct LoadPlan {
    parts: Vec<Planned>,
    costs: Vec<(u64, u64)>,
    runs: Vec<(std::ops::Range<usize>, u64)>,
    host_bytes: u64,
    serial_bytes: u64,
    upload_bytes: usize,
}

impl LoadPlan {
    fn checked_runs<'a>(
        &'a self,
        mut available: impl FnMut() -> Result<(u64, HostRead), GpuError> + 'a,
    ) -> impl Iterator<Item = Result<&'a [Planned], GpuError>> + 'a {
        let mut runs = self.runs.iter();
        let mut left = 0..0;
        std::iter::from_fn(move || {
            let mut planned = None;
            if left.is_empty() {
                let (range, bytes) = runs.next()?;
                (left, planned) = (range.clone(), Some(*bytes));
            }
            // The ring is already paid for in the live reading.
            let run = available().and_then(|(live, read)| {
                let (n, bytes) =
                    read_ahead(&self.parts[left.clone()], &self.costs[left.clone()], live)
                        .map_err(|e| refused(e.to_string()))?;
                if bytes > live {
                    return Err(refused(format!(
                        "part read-ahead needs {bytes} B, the host room is {live} B ({read})"
                    )));
                }
                if let Some(planned) = planned.filter(|_| n < left.len()) {
                    eprintln!(
                        "{WHAT}: a read-ahead run of {} parts needs {planned} B, the host room \
                         is {live} B ({read}); it reads in smaller runs",
                        left.len()
                    );
                }
                let at = left.start;
                left.start += n;
                Ok(&self.parts[at..left.start])
            });
            Some(run)
        })
    }

    /// Refused unless one part at a time fits the host room `room`, named by the reading that
    /// gave it; [`Self::checked_runs`] shrinks the planned runs to the live reading.
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
            // Packing capacity and scale copies coexist with the entire run's read results.
            let read = source.read_bytes(p).map_err(|e| refused(e.to_string()))?;
            let packing = (staged as u64)
                .checked_mul(2)
                .ok_or_else(|| refused(format!("{}: packing bytes pass u64", p.info.name)))?;
            costs.push((read, packing));
        }
        let mut runs = Vec::new();
        let (mut host_bytes, mut serial_bytes) = (0, 0);
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
            runs.push((at..at + n, bytes));
            host_bytes = host_bytes.max(bytes);
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
            runs,
            host_bytes: with_ring(host_bytes)?,
            serial_bytes: with_ring(serial_bytes)?,
            upload_bytes,
        })
    }

    fn placed<S: Source>(
        source: &S,
        plan: &Plan<'_>,
        card: usize,
        in_place: &InPlace,
        room: u64,
    ) -> Result<(OnCard, Self), GpuError> {
        let on = OnCard::of_in_place(plan, card, in_place)?;
        let selected = model::part::Filtered {
            source,
            keep: |name: &str| plan.model.tensors.iter().any(|t| t.name == name),
        };
        let read = Self::new(&selected, |name, _, _| on.take(name), room)?;
        Ok((on, read))
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
/// expert tier card) and a host that reads the stack where a located header holds it
/// ([`InPlace`]), in the plan's slot order; a tensor the plan uses nowhere, or reads by rows from
/// the NVMe tier (a row-gathered table, which its own reader maps where the source stores it), is
/// skipped, as is a tensor the plan holds on the host where a located header holds it in place,
/// which the host reads there, and a tensor or stack of which this card holds nothing, which the
/// other cards' loads read. A source fills no host set, so a tensor the plan splits otherwise, or
/// holds on the host but the source alone makes, is refused by name.
pub struct OnCard {
    held: HashMap<String, OnThisCard>,
    skipped: HashSet<String>,
}

/// A tensor the card holds ([`OnCard`]): the plan's card format and resident bytes, and the
/// stack's experts the card holds of a split stack (`None` the whole tensor).
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

    /// [`OnCard::of`] whose host reads the tensors `in_place` names where they are stored: a
    /// tensor the plan holds wholly on the host is skipped when `in_place` holds it, and refused
    /// by name otherwise; of a stack `in_place` holds that the plan splits between this card and
    /// the host, the card takes its segment's experts ([`OnCard::take`]).
    pub fn of_in_place(
        plan: &Plan<'_>,
        card: usize,
        in_place: &InPlace,
    ) -> Result<OnCard, GpuError> {
        let model = plan.model;
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
                segments
                    if !segments.is_empty()
                        && segments.iter().all(|s| s.device == Device::Host) =>
                {
                    if !in_place.holds(&t.name) {
                        return Err(elsewhere(
                            "on Host, where the host reads only a tensor a located header holds \
                             in place (a file run or a stack), and the part source makes this one"
                                .to_string(),
                        ));
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

/// Card `card` of `plan` from `source` ([`OnCard::of_in_place`], [`load`], each part's pages
/// released under `release`), checked against the plan; the host reads the tensors `in_place`
/// names where they are stored.
pub fn load_placed_in_place<S: Source + Sync>(
    stream: &CudaStream,
    source: &S,
    plan: &Plan<'_>,
    card: usize,
    release: bool,
    in_place: &InPlace,
    room: u64,
) -> Result<Weights, GpuError> {
    let (on, read) = LoadPlan::placed(source, plan, card, in_place, room)?;
    let (available, reading) = host_available_read().map_err(refused)?;
    let w = load_plan(
        stream,
        source,
        &read,
        |t| on.format(t),
        release,
        &(available.min(room), reading),
    )?;
    on.check(&w)?;
    Ok(w)
}

/// Whether `segments` put something on a card other than `card` and nothing on `card`, the rest
/// on the host: another card's load reads its segments, and the load of a card that holds a share
/// checks the host's ([`card_share`]).
fn elsewhere_only(segments: &[Segment], card: usize) -> bool {
    let other = |s: &Segment| matches!(s.device, Device::Card(c) if c != card);
    segments.iter().any(other)
        && segments
            .iter()
            .all(|s| other(s) || s.device == Device::Host)
}

/// The card's share of stack `t`, whose segments are `segments`, when the plan splits it between
/// card `card`, other cards and the host and `in_place` holds it as a stack, which the host reads
/// where the folder stores it: one segment on the card, in a card format, holding some of the
/// stack's experts, every other segment on another card or the host. `None` for segments of any
/// other split, which the caller refuses by name; a card expert past the located stack's
/// matrices, and a card segment in no card format, are refused by name.
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
    let card_or_host = |s: &Segment| matches!(s.device, Device::Card(_) | Device::Host);
    if !elsewhere.clone().all(card_or_host) || elsewhere.count() == 0 {
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

    /// A run's parts are staged in run order while later parts still read, and a read's refusal
    /// stops the stage at that part. The last part's read waits, at most 10 s, for the first
    /// part's stage. Mutants: staging in arrival order stages the later, faster reads first;
    /// staging once every read is in leaves the last read waiting until it is refused.
    #[test]
    fn a_run_stages_in_order_as_its_reads_arrive() {
        /// Signals the first part's stage, for the last part's read.
        struct Slow(std::sync::Mutex<mpsc::Receiver<()>>);
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
        }
        let run: Vec<Planned> = (0..12)
            .map(|i| Planned {
                at: None,
                info: TensorInfo {
                    name: format!("p{i}"),
                    ty: GgmlType::F32,
                    dims: vec![1],
                    offset: i,
                    nbytes: 4,
                },
            })
            .collect();
        let (staged, first) = mpsc::channel();
        let slow = Slow(std::sync::Mutex::new(first));
        let mut order = Vec::new();
        let mut stage = |p: &Planned, part: model::part::Part<'_>| {
            assert_eq!(part.spans(), [[p.info.offset as u8]]);
            if p.info.offset == 0 {
                staged.send(()).unwrap();
            }
            order.push(p.info.offset);
            Ok(())
        };
        read_staged(&slow, &run[..9], &mut stage).unwrap();
        let error = read_staged(&slow, &run, &mut stage)
            .unwrap_err()
            .to_string();
        assert!(error.contains("part 9 refused"), "{error}");
        let want: Vec<u64> = (0..9).chain(0..9).collect();
        assert_eq!(order, want);
    }

    #[test]
    fn later_reads_use_their_own_peak_under_a_falling_host_room() {
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
        assert_eq!(shared.runs, [(0..2, 200 * mib)]);
        assert_eq!(shared.host_bytes, 296 * mib);
        let read = |bytes: u64| (bytes, HostRead::MemAvailable);
        shared.check(&read(288 * mib)).unwrap();
        assert!(shared.check(&read(288 * mib - 1)).is_err());
        let plan =
            LoadPlan::new(&source, |_, _, _| Ok::<_, String>(Take::Whole), 288 * mib).unwrap();
        assert_eq!(plan.runs, [(0..1, 192 * mib), (1..2, 24 * mib)]);
        assert_eq!(plan.host_bytes, 288 * mib);
        plan.check(&read(300 * mib)).unwrap();
        assert!(plan.check(&read(288 * mib - 1)).is_err());
        for (load, readings, lens) in [
            (&shared, &[200][..], &[2][..]),
            (&shared, &[199, 199][..], &[1, 1][..]),
            (&plan, &[228, 164][..], &[1, 1][..]),
            (&plan, &[192, 24][..], &[1, 1][..]),
        ] {
            let mut rooms = readings.iter();
            let runs = load
                .checked_runs(|| Ok(read(rooms.next().unwrap() * mib)))
                .collect::<Result<Vec<_>, _>>()
                .expect("each run fits its live room, shrunk to it when over");
            assert_eq!(runs.iter().map(|r| r.len()).collect::<Vec<_>>(), lens);
            assert_eq!(
                runs.iter()
                    .flat_map(|run| run.iter())
                    .map(|p| p.info.name.as_str())
                    .collect::<Vec<_>>(),
                ["p64.weight", "p8.weight"]
            );
        }
        let mut rooms = [(228, HostRead::MemAvailable), (23, HostRead::Given)].into_iter();
        let mut runs = plan.checked_runs(|| {
            let (room, read) = rooms.next().unwrap();
            Ok((room * mib, read))
        });
        assert!(runs.next().unwrap().is_ok());
        let error = runs.next().unwrap().unwrap_err().to_string();
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
        assert_eq!(plan.runs.len(), 256);
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
    /// cards (plan (b′)'s stage and expert tier cards) and the host, the card takes its segment's
    /// experts in the plan's slot order, at that segment's card format and resident bytes, the
    /// other cards' loads take theirs and the host reads the rest where the folder stores them;
    /// a card that holds none of the stack skips it. Refused by name: a card expert past the
    /// located stack's matrices, a card segment in no card format, the rest on the NVMe tier,
    /// two segments on this card, and a split of a stack the header does not hold in place (a
    /// GGUF's, or a folder's whose part source makes it). Red against the load before the split
    /// arm, which refused every split "on Card(0) and Host"; mutant: other cards left out of
    /// the split arm — the stage's share of a tiered split is refused "on Card(0) and Card(1)
    /// and Host".
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
            (
                vec![
                    seg(Device::Card(0), card, &[1, 4, 6], 8),
                    seg(Device::Nvme, Format::NvmeFile, &[0, 2, 3, 5, 7], 8),
                ],
                in_place(8),
                "on Card(0) and Nvme",
            ),
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
}
