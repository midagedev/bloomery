use std::fmt;

use gguf::GgmlType;

use super::Role;
use super::machine::{Card, Device, ExpertList, Format, Heap, ModelTensor, ModelTensors};
use super::refusal::PlacementError;
use super::rule::{card_buffers, per_expert, row_bytes, rows_of};
use super::table::{Plan, Row, Segment};
use super::workstation;

/// What a program's whole load holds on its card past its weights, its
/// cache and its card's set-aside ([`Card::set_aside_bytes`]: the context
/// and the step arenas' scratch), as its own load counts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WholeLoad {
    /// Its ubatch arena, which the load checks against the card's free
    /// bytes before allocating, and what that check keeps free past it.
    Checked { arena: u64, reserve: u64 },
    /// A program whose load allocates its arena with no such check and whose
    /// arena's bytes have no formula outside the allocation: the card's
    /// margin is what covers it.
    Unchecked,
}

impl WholeLoad {
    /// The bytes it holds past the set-aside on `card`: the arena and the
    /// reserve, or the card's margin. `None` past u64 bytes.
    #[must_use]
    pub fn past_bytes(self, card: &Card) -> Option<u64> {
        match self {
            WholeLoad::Checked { arena, reserve } => arena.checked_add(reserve),
            WholeLoad::Unchecked => Some(card.margin_bytes),
        }
    }
}

/// What a load of the whole model asks of one card ([`whole_need`]) and
/// what the card had; its `Display` is the verdict in one line, term by
/// term.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WholeNeed {
    /// The card, its context, scratch, reserves, margin and free reading.
    pub card: Card,
    /// The granules the uploads take through the card's allocator.
    pub weights: u64,
    /// The same uploads' buffer bytes, before the allocator's rounding.
    pub weights_unrounded: u64,
    /// The cache the caller counted: the load's context, every slot.
    pub kv: u64,
    /// What the program holds past them.
    pub load: WholeLoad,
    /// The card's floor over all of it ([`Card::floor_bytes`]).
    pub need: u64,
    /// What the card had: [`Card::read_capped_bytes`].
    pub budget: u64,
}

impl WholeNeed {
    /// Whether the load fits: its need within what the card had.
    #[must_use]
    pub fn fits(&self) -> bool {
        self.need <= self.budget
    }
}

impl fmt::Display for WholeNeed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = &self.card;
        write!(
            f,
            "the whole load on {}: {}: need {} B = weights {} B (the allocator's granules; {} B \
             of buffers) + cache {} B + context {} B + scratch {} B + reserves ",
            c.name,
            if self.fits() { "fits" } else { "does not fit" },
            self.need,
            self.weights,
            self.weights_unrounded,
            self.kv,
            c.context_bytes,
            c.scratch_bytes,
        )?;
        match c.reserve_bytes() {
            Some(bytes) => write!(f, "{bytes} B + ")?,
            None => write!(f, "past u64 bytes + ")?,
        }
        match self.load {
            WholeLoad::Checked { arena, reserve } => {
                write!(f, "arena {arena} B + reserve past it {reserve} B")?;
            }
            WholeLoad::Unchecked => write!(
                f,
                "margin {} B (the program's arena under it)",
                c.margin_bytes
            )?,
        }
        write!(f, ", of {} B the card had (", self.budget)?;
        match c.free_bytes {
            Some(free) => write!(
                f,
                "{free} B free of its usable {} B at plan time",
                c.usable_bytes
            )?,
            None => write!(f, "no free reading, its usable {} B", c.usable_bytes)?,
        }
        if let Some(held) = c.held_by {
            write!(f, ", held by {held}")?;
        }
        f.write_str(")")
    }
}

/// What a load of the whole of `model` on `card` with `kv` bytes of cache
/// asks of the card, against what the card had: the one owner of the
/// whole-fit verdict. The load uploads every tensor its loader takes — each
/// of the model's layers and every model-level tensor, whatever its role —
/// in the model's order, each in the card format its role gives its type
/// ([`ModelTensors::card_format`]) and each one's buffers through the card's
/// allocator ([`Heap`]), so the granules are counted; then the cache, the
/// card's set-aside and what the program holds past them (`load`), summed
/// by the card's floor ([`Card::floor_bytes`]). The budget is the card's
/// usable bytes capped by the census's free reading as taken
/// ([`Card::read_capped_bytes`]), not the plan's cap ([`capped`]), which
/// adds the context's own creation back: between the reading and its arena
/// a whole load takes more than these terms — its joined stacks' and its
/// caches' rounding, its rope rows and prompt image, its step arenas — and
/// the context term's whole size is what covers that. A tensor no card
/// format loads is refused by name ([`PlacementError::NoCardFormat`]): the
/// whole load cannot upload it.
pub fn whole_need(
    model: &ModelTensors,
    card: &Card,
    kv: u64,
    load: WholeLoad,
) -> Result<WholeNeed, PlacementError> {
    let mut heap = Heap::new(card.granule_bytes);
    let mut unrounded = 0u64;
    let loaded = model
        .tensors
        .iter()
        .filter(|t| t.layer.is_none_or(|l| l < model.layers));
    for t in loaded {
        let format = model
            .card_format(t)
            .ok_or_else(|| PlacementError::NoCardFormat {
                name: t.name.clone(),
                ty: t.ty,
                role: t.role,
            })?;
        for b in card_buffers(t, format, rows_of(t))? {
            unrounded += b;
            heap.alloc(b);
        }
    }
    let need = load
        .past_bytes(card)
        .and_then(|past| card.floor_bytes(heap.taken, kv, past))
        .ok_or_else(|| card.bytes_past_u64())?;
    Ok(WholeNeed {
        card: card.clone(),
        weights: heap.taken,
        weights_unrounded: unrounded,
        kv,
        load,
        need,
        budget: card.read_capped_bytes(None),
    })
}

/// Whether `r` places a row-gathered table the program reads by row id
/// ([`Role::EngramTable`]): what [`row_table_tier`] places and the row
/// reserve is for.
pub(super) fn row_read(model: &ModelTensors, r: &Row) -> bool {
    model
        .tensors
        .get(r.tensor)
        .is_some_and(|t| t.role == Role::EngramTable)
}

/// The row reserve [`totals`] sets aside: [`workstation::ROW_CACHE`], the
/// engram row cache — the hot rows a reader keeps beside the drive — when
/// any row-gathered table lies on the NVMe tier, 0 when none does. A plan
/// whose maker sizes its tables' reader ([`row_table_tier`]) replaces it.
pub(super) fn row_cache_reserve(model: &ModelTensors, rows: &[Row]) -> u64 {
    let on_nvme = rows
        .iter()
        .filter(|r| row_read(model, r))
        .flat_map(|r| &r.segments)
        .any(|s| s.device == Device::Nvme);
    if on_nvme { workstation::ROW_CACHE } else { 0 }
}

/// The page-cache bytes a reader of `model`'s row-gathered tables
/// ([`Role::EngramTable`]) holds while it reads one call of up to
/// `positions` positions from the NVMe tier: per table, its rows a position
/// ([`ModelTensor::gathered_rows`]) for every position, each physical plane of
/// a row on at most `1 + ceil((plane bytes - 1) / page)` host base pages.
/// Split rows are packed; NVFP4 part-source rows have separate code and scale
/// planes. The ids are hashed, so a call's rows are counted on pages of their
/// own, and each plane may straddle a page boundary. The reader
/// advises and copies one call at a time, so one call's pages are what must
/// stay resident until its copy. A table with no per-token row count, or a
/// room past u64, is refused by name.
pub fn row_room(
    model: &ModelTensors,
    positions: u64,
    page: u64,
    source: RowTableSource,
) -> Result<u64, PlacementError> {
    model
        .tensors
        .iter()
        .filter(|t| t.role == Role::EngramTable)
        .try_fold(0u64, |sum, t| {
            let rows = t.gathered_rows.ok_or_else(|| {
                PlacementError::tensor(t, "a row-gathered table without a per-token row count")
            })?;
            let bytes = row_bytes(t)?;
            let span = |bytes: u64| 1 + bytes.saturating_sub(1).div_ceil(page);
            let pages = match (source, t.ty) {
                (RowTableSource::Parts, GgmlType::NVFP4_ROWS) => {
                    if !bytes.is_multiple_of(9) {
                        return Err(PlacementError::tensor(
                            t,
                            "NVFP4 row bytes are not whole groups",
                        ));
                    }
                    let groups = bytes / 9;
                    span(groups * 8) + span(groups)
                }
                _ => span(bytes),
            };
            positions
                .checked_mul(rows)
                .and_then(|n| n.checked_mul(pages))
                .and_then(|n| n.checked_mul(page))
                .and_then(|n| sum.checked_add(n))
                .ok_or_else(|| {
                    PlacementError::tensor(
                        t,
                        format!("the row room of a call of {positions} positions passes u64 bytes"),
                    )
                })
        })
}

/// Where a row reader obtains its table. Part sources have no split host set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowTableSource {
    /// A split's mappings, held on the host when its room permits.
    Split,
    /// A part source's file ranges, gathered from the NVMe tier.
    Parts,
}

/// Where `plan`'s row-gathered tables ([`Role::EngramTable`]) are read
/// from, the one rule for a plan whose maker sizes their reader: on the
/// host in their file bytes when `room` — the host's available bytes
/// ([`workstation::host_room`]'s reading, or a room the caller gives) —
/// holds the plan's host need with them there ([`workstation::HostNeed`],
/// no churn pool beside it) and, on a unified machine, `floor` bytes more;
/// else on the NVMe tier, where the role rule placed them ([`place_whole`]),
/// with their reader's room for a call of up to `positions` positions
/// ([`row_room`], on `page` pages) as the plan's row reserve, in place of
/// the row cache [`totals`] set aside. [`RowTableSource::Parts`] stays on
/// NVMe, with its row window sized for the stored planes.
///
/// `floor` is the prompt cache's: the caller's two saved sequence states at
/// the plan's context. The cache's default budget is half of what the room
/// leaves past the need and it holds nothing under one state, so a table
/// that takes the room whole leaves a load with no cache to serve from. A
/// machine whose cards have memory of their own takes the room alone, as
/// every such plan did.
///
/// `page` is the host's base page ([`row_room`]): the row room counts the
/// pages a row sits on, and a host whose base page is not the one this
/// workstation has counts its own.
///
/// A plan with no such table is left as it is; [`Plan::row_tier`] reads the
/// outcome. A table of other than one NVMe file segment, or a sum past u64,
/// is refused by name.
pub fn row_table_tier(
    plan: &mut Plan<'_>,
    room: u64,
    floor: u64,
    positions: u64,
    page: u64,
    source: RowTableSource,
) -> Result<(), PlacementError> {
    let model = plan.model;
    let tables: Vec<usize> = (0..plan.rows.len())
        .filter(|&i| row_read(model, &plan.rows[i]))
        .collect();
    let mut moved = 0u64;
    for &i in &tables {
        let r = &plan.rows[i];
        let t = &model.tensors[r.tensor];
        let [s] = r.segments.as_slice() else {
            return Err(PlacementError::tensor(
                t,
                format!(
                    "the row-gathered table in {} segments, not one",
                    r.segments.len()
                ),
            ));
        };
        if (s.device, s.format) != (Device::Nvme, Format::NvmeFile) {
            return Err(PlacementError::tensor(
                t,
                format!(
                    "the row-gathered table on {:?} as {}, not the NVMe tier's file bytes",
                    s.device, s.format
                ),
            ));
        }
        moved = moved
            .checked_add(s.resident_bytes)
            .ok_or_else(|| PlacementError::tensor(t, "the row-gathered tables' bytes pass u64"))?;
    }
    let Some(&first) = tables.first() else {
        return Ok(());
    };
    let named = &model.tensors[plan.rows[first].tensor];
    let past = |what: &str| PlacementError::tensor(named, format!("{what} passes u64 bytes"));
    let old = plan.host.row_reserve_bytes;
    let mut host_arm = workstation::HostNeed::of(plan, 0);
    host_arm.tables = host_arm
        .tables
        .checked_add(moved)
        .ok_or_else(|| past("the host's tables with the row-gathered ones"))?;
    host_arm.reserves = host_arm.reserves.checked_sub(old).ok_or_else(|| {
        PlacementError::tensor(
            named,
            format!(
                "the plan's row reserve {old} B is past its host reserves {} B",
                host_arm.reserves
            ),
        )
    })?;
    let floor = if plan.machine.unified().is_some() {
        floor
    } else {
        0
    };
    let held = host_arm
        .bytes()
        .checked_add(floor)
        .ok_or_else(|| past("the host's need with the prompt cache's floor"))?;
    let new = if source == RowTableSource::Split && held <= room {
        plan.nvme_bytes = plan.nvme_bytes.checked_sub(moved).ok_or_else(|| {
            PlacementError::tensor(
                named,
                format!(
                    "the row-gathered tables' {moved} B pass the NVMe tier's {} B",
                    plan.nvme_bytes
                ),
            )
        })?;
        plan.host.table_bytes = host_arm.tables;
        plan.host.headroom_bytes -= i128::from(moved);
        for &i in &tables {
            let s = &mut plan.rows[i].segments[0];
            (s.device, s.format) = (Device::Host, Format::HostFile);
        }
        0
    } else {
        row_room(model, positions, page, source)?
    };
    // The plan's own reserves take the new room: the need's leave out the part reads
    // (`workstation::PART_READS`, its `read_ahead`), which stay the plan's.
    plan.host.reserve_bytes = (plan.host.reserve_bytes - old)
        .checked_add(new)
        .ok_or_else(|| past("the host's reserves with the row room"))?;
    plan.host.row_reserve_bytes = new;
    plan.host.headroom_bytes += i128::from(old) - i128::from(new);
    Ok(())
}

/// The room a caller gave through the lever `name` (bytes, `M` and `G`
/// binary units), read as [`workstation::HostRead::Given`]: `Ok(None)` for
/// an unset one — the machine's reading stands. A value that is not bytes,
/// is past u64 or is not UTF-8 is refused by name.
pub fn host_room_given(
    name: &str,
    value: Option<&std::ffi::OsStr>,
) -> Result<Option<(u64, workstation::HostRead)>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let text = value
        .to_str()
        .ok_or_else(|| format!("{name}={value:?} is not UTF-8; it takes bytes"))?;
    match bloomery_levers::parse_bytes(text) {
        Ok(bytes) => Ok(Some((bytes, workstation::HostRead::Given))),
        Err(e) => Err(format!(
            "{name}={e}; it takes bytes: digits, `_` separators allowed, then nothing, `M` or `G`"
        )),
    }
}

/// One layer's routed stacks as the NVMe expert tier splits them: each
/// stack's row and the index of its host segment, the ids that segment holds
/// (the same for every stack of the layer), and the bytes of one expert
/// across the stacks.
struct HostLayer {
    stacks: Vec<(usize, usize)>,
    ids: Vec<u32>,
    unit: u64,
}

impl HostLayer {
    fn bytes(&self) -> u64 {
        self.unit * self.ids.len() as u64
    }
}

/// Puts the routed experts the host cannot hold on the NVMe tier, for a plan
/// whose host need ([`workstation::HostNeed`], what `HostNeed::check` holds
/// the host's available bytes to) passes `room`: each layer's host-served
/// ids split into the first `r_l`, which stay on the host, and the rest,
/// which the host leg reads from the NVMe tier ([`Device::Nvme`],
/// [`Format::NvmeFile`]). The `r_l` are spread over the layers by their host
/// bytes, so the host's resident bytes fall by the overflow, to within one
/// layer's expert; every stack of a layer splits at the same ids. A plan
/// whose need `room` holds is left as it is.
///
/// Runs after [`row_table_tier`], which puts the row-gathered tables on the
/// NVMe tier first: a plan that still needs a split with such a table on the
/// host is refused by name. The need is the plan's, with no residency churn
/// pool beside it and no transient window: a reader of the NVMe tier takes
/// its two layers of transient reads from the room beside the plan's need
/// ([`PlacementError::HostRoomFloor`] keeps the room for them).
///
/// The page every window of an NVMe-tier arena slot is aligned to: the
/// `O_DIRECT` alignment the slot's fills read at (engram's `DIRECT_ALIGN`,
/// which the arena's crate holds equal to it at compile time).
pub const ARENA_ALIGN: u64 = 4096;

/// The step port's width: the columns past it do not enter the arena's floor
/// ([`HostTotals::step_slots`]). The model crate's `ops::DEFER_MAX_COLS`
/// holds it equal, by a compile-time assertion in the gpu crate's tier.
pub const STEP_PORT_COLUMNS: usize = 8;

/// The slots of a paged layer one residency lane holds at once beside the
/// step's call: `4 + 1`. Four are open reads: each lane thread pins the slot
/// of the id it stages from `open_read` until its `end_read`, and a flush's
/// or a landing's jobs stage on all `LANE_THREADS` (4) threads at once
/// (`host/lane.rs`, `Shared::serve`). One is a victim claim: the machine's
/// pick check (`SwapMachine::check_victims`) prepares a victim on its own
/// thread while the lane's threads hold their pins (`NvTier::ensure_victim`,
/// published unpinned). There is one lane per swap machine, so a paged layer
/// sees one lane's count. Placement cannot depend on the gpu crate: a test
/// there holds `LANE_THREADS + 1` equal to this value. A claim that finds
/// every candidate pinned is refused by name.
pub const ARENA_LANE_SLOTS: u64 = 4 + 1;

/// The experts of the NVMe tier's arena one batch of a prompt union holds:
/// the unit a claim, a fill and a pin move in. A union over more of a
/// layer's arena ids than its slots less two batches runs in batches, each
/// pinned while it computes and the next claimed and filled beside it, so
/// two batches are pinned at once ([`ArenaFloor::union`]). The model side's
/// batched union and the arena's floor read this one value.
pub const UNION_BATCH_IDS: u64 = 16;

/// The bytes one part of `len` bytes takes in an arena slot: the part and
/// the two partial pages an aligned read of it can span, in whole
/// [`ARENA_ALIGN`] pages. The one shape of a slot's window, which the
/// arena carves and its floor counts.
#[must_use]
pub fn arena_window_bytes(len: u64) -> u64 {
    (len + 2 * ARENA_ALIGN).div_ceil(ARENA_ALIGN) * ARENA_ALIGN
}

/// The bytes one expert's part takes in an arena slot: a window
/// ([`arena_window_bytes`]) for each run its fill reads, `codes` bytes and,
/// for a matrix kept as two planes, `scales` bytes of block scales (0 for a
/// part of one run), whose read lands in a window of its own before the fill
/// moves it to follow the codes. The one shape of a part's place in a slot,
/// which the arena carves and its floor counts.
#[must_use]
pub fn arena_part_bytes(codes: u64, scales: u64) -> u64 {
    let scales = if scales == 0 {
        0
    } else {
        arena_window_bytes(scales)
    };
    arena_window_bytes(codes) + scales
}

/// The runs one expert of routed stack `t`, `per` bytes, is filled from,
/// `(codes, scales)`: NVFP4 rows ([`GgmlType::NVFP4_ROWS`]), which a
/// checkpoint folder keeps as two planes a matrix, its codes (8 of a group's
/// 9 bytes) and its block scales (the ninth); one run of `per` bytes, `(per,
/// 0)`, for every other type. NVFP4 rows a file keeps in one run are counted
/// at the two planes' windows, the larger.
fn stored_planes(t: &ModelTensor, per: u64) -> (u64, u64) {
    match (t.ty, t.ty.type_size()) {
        (GgmlType::NVFP4_ROWS, Some(group)) => {
            let scales = per / group;
            (per - scales, scales)
        }
        _ => (per, 0),
    }
}

/// The NVMe tier arena's floor: the slots a paged layer needs so that one
/// call reads every distinct id it names from its own slot while the lane's
/// pins and claim beside it take others — the call's `k × columns` ids and
/// the lane's slots — or, for a prompt union run in batches, the two batches
/// it pins at once and the lane's slots ([`ArenaFloor::union`]), whichever
/// is larger, and their bytes over one slot of every layer the arena may page.
/// The one owner of the floor: the split dial gives no arena under it
/// ([`nvme_arena_of`]), and the arena's carve at load refuses a budget
/// under its slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaFloor {
    /// The routed width: the ids one column names.
    pub k: u64,
    /// The widest call's columns.
    pub columns: u64,
    /// The slots the lane holds beside the call: [`ARENA_LANE_SLOTS`]
    /// where a stage card holds a routed expert (the residency machine can
    /// run), else none.
    pub lane_slots: u64,
    /// One slot of every layer the host leg serves routed experts of, each
    /// part in its windows ([`arena_part_bytes`]).
    pub slot_set: u64,
    /// The slots a prompt union run in batches holds at once: two batches
    /// ([`UNION_BATCH_IDS`]) and the lane's slots ([`ARENA_LANE_SLOTS`],
    /// whether or not a lane runs: the union's own refusal counts it so).
    pub union: u64,
    /// The larger of `k × columns + lane_slots` and `union`.
    pub slots: u64,
    /// `slots × slot_set`.
    pub bytes: u64,
}

impl ArenaFloor {
    /// The floor of calls of `columns` columns of `k` ids beside `lane_slots`
    /// lane slots, over a slot set of `slot_set` bytes. Refused by name: a
    /// count or a byte total past u64.
    pub fn new(
        k: u64,
        columns: u64,
        lane_slots: u64,
        slot_set: u64,
    ) -> Result<ArenaFloor, PlacementError> {
        let past = || PlacementError::Metadata {
            key: "the NVMe tier arena's floor".to_string(),
            detail: format!(
                "{k} × {columns} ids + {lane_slots} lane slots over a {slot_set} B slot set pass u64"
            ),
        };
        let call = k
            .checked_mul(columns)
            .and_then(|ids| ids.checked_add(lane_slots))
            .ok_or_else(past)?;
        let union = 2 * UNION_BATCH_IDS + ARENA_LANE_SLOTS;
        let slots = call.max(union);
        let bytes = slots.checked_mul(slot_set).ok_or_else(past)?;
        Ok(ArenaFloor {
            k,
            columns,
            lane_slots,
            slot_set,
            union,
            slots,
            bytes,
        })
    }

    /// `plan`'s floor for calls of [`HostTotals::arena_columns`] columns: `k`
    /// the model's routed width, the lane slots where a stage card holds a routed
    /// expert, and the slot set over every layer whose routed stacks the
    /// host leg serves (a host or an NVMe segment) — on a plan the split
    /// has not run on yet, the layers it can page, a bound over the ones it
    /// does, so the arena's carve at load never falls under the slots the
    /// dial sized it for.
    pub fn of(plan: &Plan<'_>) -> Result<ArenaFloor, PlacementError> {
        let model = plan.model;
        let mut slot_set = 0u64;
        for r in &plan.rows {
            let t = &model.tensors[r.tensor];
            if t.role != Role::RoutedExperts
                || t.layer.is_none()
                || !r
                    .segments
                    .iter()
                    .any(|s| matches!(s.device, Device::Host | Device::Nvme))
            {
                continue;
            }
            let (_, per) = per_expert(t, model.experts_of(t))?;
            let (codes, scales) = stored_planes(t, per);
            slot_set = slot_set
                .checked_add(arena_part_bytes(codes, scales))
                .ok_or_else(|| PlacementError::tensor(t, "the arena's slot set passes u64"))?;
        }
        let lane_slots = if plan.n_l.iter().any(|&n| n > 0) {
            ARENA_LANE_SLOTS
        } else {
            0
        };
        ArenaFloor::new(
            model.experts_used,
            plan.host.arena_columns,
            lane_slots,
            slot_set,
        )
    }
}

/// The slots `plan`'s arena holds: its bytes over one slot of every layer the
/// host leg serves routed experts of ([`ArenaFloor::slot_set`]); 0 on a plan
/// with no arena. Refused by name: an arena on a plan with no such layer.
pub fn arena_slots(plan: &Plan<'_>) -> Result<u64, PlacementError> {
    let arena = plan.host.nvme_arena_bytes;
    if arena == 0 {
        return Ok(0);
    }
    let set = ArenaFloor::of(plan)?.slot_set;
    arena
        .checked_div(set)
        .ok_or_else(|| PlacementError::Metadata {
            key: "the NVMe tier arena's slots".to_string(),
            detail: format!("an arena of {arena} B over a slot set of 0 B"),
        })
}

impl fmt::Display for ArenaFloor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} slots a paged layer (the larger of {} × {} ids a call + {} lane slots and {} for \
             a union in batches) × {} B a slot of every paged layer = {} B",
            self.slots,
            self.k,
            self.columns,
            self.lane_slots,
            self.union,
            self.slot_set,
            self.bytes
        )
    }
}

/// [`PlacementError::HostRoomFloor`]'s term for the arena's least, when
/// the floor counts it.
pub(super) fn least_term(least: Option<&ArenaFloor>) -> String {
    least.map_or_else(String::new, |a| {
        format!(
            " + the RAM arena's least, {a}, which BLOOMERY_NVTIER_BYTES=0 leaves out to page \
             through the file mapping"
        )
    })
}

/// What the split dial reads of a plan the NVMe expert tier splits
/// ([`expert_nvme_tier`]).
#[derive(Clone, Copy, Debug)]
pub struct ArenaDial {
    /// The host's room the split reads.
    pub room: u64,
    /// The plan's host terms beside the routed experts.
    pub base: u64,
    /// On a unified machine, the cards' share of the pool that `room` and
    /// `base` both carry (`HostNeed::cards`); 0 where each card has memory
    /// of its own. The dial's step and deep side read the host's room past
    /// it, so a unified card gets the arena a discrete one gets beside the
    /// same card bytes.
    pub cards: u64,
    /// `W`: the host-served expert bytes of the layer with the most.
    pub heavy: u64,
    /// `H`: the host leg's routed-expert bytes before the split.
    pub held: u64,
    /// The plan's host need before the split.
    pub need: u64,
    /// The arena's floor.
    pub least: ArenaFloor,
}

impl ArenaDial {
    /// The tier's room floor, `base + 3 W` ([`expert_nvme_tier`]'s terms).
    /// Refused by name past u64.
    pub fn floor(&self) -> Result<u64, PlacementError> {
        3u64.checked_mul(self.heavy)
            .and_then(|w| w.checked_add(self.base))
            .ok_or_else(|| PlacementError::Metadata {
                key: "the NVMe expert tier's floor".to_string(),
                detail: "passes u64 bytes".to_string(),
            })
    }

    /// The room the dial steps at, `H/2 + cards`: where the host's room past
    /// the cards reaches half the host leg. Refused by name past u64.
    fn half(&self) -> Result<u64, PlacementError> {
        (self.held / 2)
            .checked_add(self.cards)
            .ok_or_else(|| PlacementError::Metadata {
                key: "the NVMe expert tier's step".to_string(),
                detail: "passes u64 bytes".to_string(),
            })
    }

    /// What the room leaves above the floor at the step,
    /// `H/2 + cards − floor` (the deep side's arenas lie under it), when the
    /// arena's least fits under it; `None` on a plan the dial gives no
    /// arena.
    fn peak(&self, floor: u64) -> Result<Option<u64>, PlacementError> {
        Ok(self
            .half()?
            .checked_sub(floor)
            .filter(|&peak| peak >= self.least.bytes))
    }

    /// The least room the dial plans at under the lever's `given` arena:
    /// the floor, and with the lever unset on a plan with a deep side, the
    /// arena's least above it.
    pub fn room_floor(&self, given: Option<u64>) -> Result<u64, PlacementError> {
        let floor = self.floor()?;
        Ok(match (given, self.peak(floor)?) {
            (None, Some(_)) => floor + self.least.bytes,
            _ => floor,
        })
    }

    /// The last room the dial gives an arena, `H/2 + cards − 1` — the deep
    /// side's largest, `H/2 + cards − 1 − F`, its one step to none at
    /// `H/2 + cards` — or `None` on a plan with no deep side.
    pub fn step_room(&self) -> Result<Option<u64>, PlacementError> {
        let floor = self.floor()?;
        let half = self.half()?;
        Ok(self.peak(floor)?.map(|_| half - 1))
    }
}

/// The arena bytes the split dial gives `dial`'s room: the lever
/// `BLOOMERY_NVTIER_BYTES` (`given`) when set, else the default. The one
/// owner of the arena's bytes; the plan states what it chose in
/// [`HostTotals::nvme_arena_bytes`].
///
/// The default, with `F` the floor, `H` the host leg and `M` the arena's
/// least ([`ArenaFloor`]): the room above the floor, `room − F`, while the
/// host's room past the cards ([`ArenaDial::cards`]) is under `H/2` (a deep
/// split: the prompt is NVMe-bound whatever the split, so the arena's warm
/// set costs prefill little); from there on, none (the split keeps every
/// host expert it can and pages through the file mapping, which serves
/// several columns and the draft). No room gets an arena under `M`: a room
/// under `F + M` on a plan with a deep side is refused. The one step is at
/// `H/2 + cards`, from `H/2 + cards − 1 − F` to none; a byte of room moves
/// the arena by one byte everywhere else. A plan whose deep side cannot
/// hold `M` gets none at any room.
///
/// Refused by name: a room under [`ArenaDial::room_floor`]
/// ([`PlacementError::HostRoomFloor`], naming the arena's least when it
/// counts it); a given arena under `M` ([`PlacementError::NvTierArenaUnder`])
/// or past `room − F` ([`PlacementError::NvTierArena`]). A given 0 pages
/// through the file mapping at any room the floor holds.
pub fn nvme_arena_of(dial: &ArenaDial, given: Option<u64>) -> Result<u64, PlacementError> {
    let floor = dial.floor()?;
    let room_floor = dial.room_floor(given)?;
    let room = dial.room;
    if room < room_floor {
        return Err(PlacementError::HostRoomFloor {
            room,
            floor: room_floor,
            base: dial.base,
            layer: dial.heavy,
            need: dial.need,
            arena: (room_floor > floor).then_some(dial.least),
        });
    }
    let least = dial.least;
    match given {
        Some(0) => return Ok(0),
        Some(arena) if arena < least.bytes => {
            return Err(PlacementError::NvTierArenaUnder { arena, least });
        }
        Some(arena) if arena > room - floor => {
            return Err(PlacementError::NvTierArena { arena, room, floor });
        }
        Some(arena) => return Ok(arena),
        None => {}
    }
    if dial.peak(floor)?.is_none() || room >= dial.half()? {
        return Ok(0);
    }
    Ok(room - floor)
}

/// The floor is `base + 3 W`: `base` the need with no routed expert on the
/// host, `W` the host bytes of the layer with the most — the room's host
/// terms beside the routed experts hold `1 W` of host-served expert
/// segments, `2 W` of the prompt run-ahead's page-cache window (a host
/// reserve of the split plan) and the RAM arena [`nvme_arena_of`] picks:
/// `1 W + 2 W + arena = room − base` at the largest arena,
/// `room − base − 3 W`. A room under it, or under it and the arena's least
/// where the dial would give the room an arena, is refused by name
/// ([`ArenaDial::room_floor`]); the plan's `host.experts` counts the
/// experts the host leg serves, on either tier.
/// The split plan's headroom is what the room leaves past the arena and the
/// host need, the budget a churn pool beside them must fit.
///
/// `given` is the lever `BLOOMERY_NVTIER_BYTES`'s value ([`nvme_arena_of`]),
/// read once by the family's `describe` beside the room: this function reads
/// no environment, so a gate plans each arm by its own arena.
pub fn expert_nvme_tier(
    plan: &mut Plan<'_>,
    room: u64,
    given: Option<u64>,
) -> Result<(), PlacementError> {
    let model = plan.model;
    let need = workstation::HostNeed::of(plan, 0);
    // The exact sum: a need past u64 bytes reads `u64::MAX` in `bytes()`,
    // which a room of `u64::MAX` would hold.
    if need.exact(true) <= u128::from(room) {
        return Ok(());
    }
    if let Some(r) = plan
        .rows
        .iter()
        .find(|r| row_read(model, r) && r.segments.iter().any(|s| s.device == Device::Host))
    {
        return Err(PlacementError::tensor(
            &model.tensors[r.tensor],
            format!(
                "the host need {} B passes the room {room} B with a row-gathered table still on \
                 the host: row_table_tier places it on the NVMe tier first",
                need.bytes()
            ),
        ));
    }
    let mut layers: Vec<HostLayer> = Vec::new();
    let mut layer_of: Vec<Option<usize>> = vec![None; model.layers];
    for (i, r) in plan.rows.iter().enumerate() {
        let t = &model.tensors[r.tensor];
        let Some(l) = t.layer.filter(|_| t.role == Role::RoutedExperts) else {
            continue;
        };
        let Some((at, s)) = r
            .segments
            .iter()
            .enumerate()
            .find(|(_, s)| s.device == Device::Host)
        else {
            continue;
        };
        if s.format != Format::HostFile {
            return Err(PlacementError::tensor(
                t,
                format!("a host segment as {}, not the file's bytes", s.format),
            ));
        }
        let Some(list) = s.experts.as_ref() else {
            return Err(PlacementError::tensor(
                t,
                "a host segment of a routed stack with no list",
            ));
        };
        let (_, per) = per_expert(t, model.experts_of(t))?;
        let Some(slot) = layer_of.get_mut(l) else {
            return Err(PlacementError::tensor(
                t,
                format!("layer {l} is past the model's"),
            ));
        };
        match *slot {
            None => {
                *slot = Some(layers.len());
                layers.push(HostLayer {
                    stacks: vec![(i, at)],
                    ids: list.ids().to_vec(),
                    unit: per,
                });
            }
            Some(k) => {
                let h = &mut layers[k];
                if h.ids != list.ids() {
                    return Err(PlacementError::tensor(
                        t,
                        format!("host experts {list} are not the other stacks' of layer {l}"),
                    ));
                }
                h.stacks.push((i, at));
                h.unit += per;
            }
        }
    }
    let base = workstation::HostNeed { experts: 0, ..need }.bytes();
    let heavy = layers.iter().map(HostLayer::bytes).max().unwrap_or(0);
    // The split dial: the arena's bytes come out of the host segments the
    // room holds, the split running on the room that remains.
    let held = plan.host.expert_bytes;
    let dial = ArenaDial {
        room,
        base,
        cards: need.cards,
        heavy,
        held,
        need: need.bytes(),
        least: ArenaFloor::of(plan)?,
    };
    let arena = nvme_arena_of(&dial, given)?;
    let room = room - arena;
    plan.host.nvme_arena_bytes = arena;
    // The prompt run-ahead's page-cache window, `2 W`, is a host reserve of
    // the split plan: the host keeps `keep` of its `held` expert bytes in
    // what the room leaves past the arena and the window, which the floor
    // keeps at least one layer's worth.
    // The split's sums are exact (the need may pass u64 bytes with the floor
    // under it): the reserve is checked, `keep` is computed in u128 from the
    // exact need, which the room does not hold (the early return above), so
    // `over` has no negative part. `3 W` fit u64 (the dial's floor).
    let past = |what: &str| PlacementError::Metadata {
        key: "the NVMe expert tier's split".to_string(),
        detail: format!("{what} passes u64 bytes"),
    };
    let window = 2 * heavy;
    plan.host.reserve_bytes = plan
        .host
        .reserve_bytes
        .checked_add(window)
        .ok_or_else(|| past("the host reserve beside the window"))?;
    let over = need.exact(true) + u128::from(window) - u128::from(room);
    let keep = u128::from(held)
        .checked_sub(over)
        .and_then(|keep| u64::try_from(keep).ok())
        .ok_or_else(|| past("the bytes the host keeps"))?;
    let share = |l: &HostLayer| u128::from(keep) * u128::from(l.bytes()) / u128::from(held);
    let mut r: Vec<u64> = layers
        .iter()
        .map(|l| u64::try_from(share(l) / u128::from(l.unit)).unwrap_or(u64::MAX))
        .collect();
    let mut kept: u64 = r.iter().zip(&layers).map(|(&r, l)| r * l.unit).sum();
    // The rest of the bytes to keep go one expert at a time to the layers
    // whose share was cut the most.
    let mut order: Vec<usize> = (0..layers.len()).collect();
    let cut = |k: usize| share(&layers[k]) - u128::from(r[k] * layers[k].unit);
    order.sort_by_key(|&k| std::cmp::Reverse(cut(k) * 1_000_000 / u128::from(layers[k].unit)));
    for k in order {
        let l = &layers[k];
        if r[k] < l.ids.len() as u64 && kept + l.unit <= keep {
            r[k] += 1;
            kept += l.unit;
        }
    }
    let mut moved = 0u64;
    for (l, &keep_l) in layers.iter().zip(&r) {
        let n = usize::try_from(keep_l)
            .unwrap_or(usize::MAX)
            .min(l.ids.len());
        for &(i, at) in &l.stacks {
            let t = &model.tensors[plan.rows[i].tensor];
            let (_, per) = per_expert(t, model.experts_of(t))?;
            let on_host = &l.ids[..n];
            let off_host = &l.ids[n..];
            let mut pieces = Vec::with_capacity(2);
            if !on_host.is_empty() {
                pieces.push(Segment {
                    device: Device::Host,
                    format: Format::HostFile,
                    experts: Some(ExpertList::new(on_host.to_vec(), model.experts)?),
                    resident_bytes: on_host.len() as u64 * per,
                });
            }
            if !off_host.is_empty() {
                let bytes = off_host.len() as u64 * per;
                moved += bytes;
                pieces.push(Segment {
                    device: Device::Nvme,
                    format: Format::NvmeFile,
                    experts: Some(ExpertList::new(off_host.to_vec(), model.experts)?),
                    resident_bytes: bytes,
                });
            }
            plan.rows[i].segments.splice(at..=at, pieces);
        }
    }
    plan.host.expert_bytes -= moved;
    plan.host.nvme_expert_bytes += moved;
    plan.nvme_bytes += moved;
    // The room binds a split plan: its headroom is what the room leaves
    // past the arena (`room` here) and the plan's host need.
    plan.host.headroom_bytes = workstation::HostNeed::of(plan, 0).headroom(room);
    Ok(())
}
