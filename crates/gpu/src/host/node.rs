//! Tier nodes: an expert tier whose card is in another process
//! (`docs/contrib/dgx-spark.md` §9.5). The process that runs the stage card
//! (node 0) drives the tier as it drives a card of its own
//! ([`super::tier::TierCard`]); the tier's work ([`TierWork`]) runs in a tier
//! server's process (node 1, [`serve`]), and each step node 0 asks of it is a
//! request over a TCP control channel ([`TierNode`]). A shared page is a
//! file under `/dev/shm` both processes map ([`super::link::Link::across`]).
//! A remote bind gives each process its own page, whose end of the link,
//! RDMA or TCP, exchanges what the other end needs to meet it on this channel
//! ([`super::link::Peer`]: a queue pair's endpoint, or a TCP listener's
//! address and nonce); each end carries only its tier's crossings. The
//! channel has no authentication.
//!
//! On connect the server sends its census ([`NodeCensus`]): node 0 plans
//! the tier on the device it names ([`node_cards`]). The open
//! carries the tier's rows of node 0's plan ([`NodeOpen`]); the server loads
//! them from its own copy of the model file, which must match the tier's
//! set, each row's tensor and the sha256 of its header and bytes, and builds the
//! family's tier computation over them ([`NodeFamily`]).
//!
//! Every read and write of the channel is bounded: a request and its reply
//! by one deadline, the open by one its rows' bytes give ([`open_bound`]).
//! A lost node ends both sides by name. Node 0 marks a tier whose channel
//! failed lost ([`TierNode::lost`]) and ends that channel, and the host tier
//! names it a lost card; a node that stops signalling mid-step is the go
//! deadline's, as for a card of node 0's. A step sends the node only the
//! layers whose picks name its slots, one enqueue frame each, so node 0 ends
//! each pass that sent it one with an end frame ([`TierNode::send_end`]),
//! and sends nothing more for a pass whose card it named lost. The server
//! ends when node 0's channel ends without a close, when a request fails
//! after it reached the tier's stream, or when work is outstanding — a layer
//! asked of it unserved, or a pass opened and not ended — with no frame or
//! progress past the go deadline and its grace: it writes
//! [`super::step::RELEASE`] to its own counters and go words, so every wait
//! its stream holds passes and node 0 names the release when it runs again
//! ([`super::tier::TierCard::released`]); it drains its stream and returns
//! the loss by name. A server between passes is idle for any time. A
//! half-open control connection is bounded by its request deadline, and an
//! unresponsive tier by the go deadline.
//!
//! A prompt batch's block runs on the node as one request a tiered
//! layer-batch ([`block`]): node 0 sends it before its host's union and
//! reads its reply after, so the link, the node's card and node 0's host
//! work beside each other; no other request goes out while a block's reply
//! is unread.

pub mod block;
pub mod wire;

use std::net::{SocketAddr, TcpListener, TcpStream};
use std::num::NonZeroU64;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use gguf::Split;
use gguf::digest::sha256_hex_of;
use model::ops::DEFER_MAX_COLS;
use model::placement::devices::{DeviceId, DeviceInfo, spec_of_device};
use model::placement::workstation::{self, ArenaGrant, CardSpec, HostRead};
use model::placement::{
    Device, ExpertList, Format, ModelTensor, ModelTensors, Plan, Role, Row, Segment,
};

use super::DRAIN_DEADLINE;
use super::batch::{BlockStaging, dtoh, htod_bytes};
use super::link::{Peer, TcpPeer};
use super::slots::HOST;
use super::step::{Chain, GO_DEADLINE};
use super::tier::{
    BlockRows, ReplyRows, TIER_CHAINS, TIER_GRACE, TierAct, TierBlock, TierCard, TierExperts,
    TierLayout, TierSet, TierShape, TierWork, chain_slot,
};
use crate::fault::{FAULT_NONE, Fault};
use crate::model::CardBytes;
use crate::tensor::TIER_ACT_MAX_COLS;
use crate::upload::bytes_of;
use crate::weights::{SegmentFile, Weights};
use crate::words::{WordOps, WordSide};
use crate::{Gpu, GpuError, Q8Act};
use block::{BlockRequest, NodeBlock};
use cuda_core::PinnedHostBuffer;
use wire::{Channel, In, Out, Polled, Tag};

/// What node 0's errors name.
const WHAT: &str = "TierNode";

/// What the server's errors name.
const SERVER: &str = "tier_server";

/// How long a request waits for its reply: a step's request launches or
/// enqueues and returns at once, so a node that has not answered by the go
/// deadline is lost.
const REPLY: Duration = GO_DEADLINE;

/// The open's bound beyond reading its rows: the card's open and the
/// family's build. Derived, not measured.
const OPEN_BASE: Duration = Duration::from_secs(120);

/// The read rate the open's bound takes for its rows, bytes a second, their
/// digest and their upload together: a load slower than this is a lost
/// node. Derived, not measured.
const OPEN_RATE: u64 = 100 << 20;

/// How long node 0 waits for the open of `open`: [`OPEN_BASE`] and each
/// row's larger file-read or resident-upload span at [`OPEN_RATE`].
fn open_bound(open: &NodeOpen) -> Result<Duration, GpuError> {
    let bad = || GpuError::shape(WHAT, "the assigned open bytes are invalid or overflow");
    let bytes = open.rows.iter().try_fold(0u64, |sum, row| {
        let file = match &row.experts {
            None => row.file_bytes,
            Some(ids) => {
                let n = row
                    .dims
                    .last()
                    .copied()
                    .filter(|&n| n > 0)
                    .ok_or_else(bad)?;
                if !row.file_bytes.is_multiple_of(n) {
                    return Err(bad());
                }
                (row.file_bytes / n)
                    .checked_mul(ids.len() as u64)
                    .ok_or_else(bad)?
            }
        };
        sum.checked_add(file.max(row.resident_bytes))
            .ok_or_else(bad)
    })?;
    Ok(OPEN_BASE + Duration::from_secs(bytes / OPEN_RATE))
}

// ---------------------------------------------------------------- census

/// A tier node's census: its visible devices as its driver enumerates them
/// ([`crate::census`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeCensus {
    pub devices: Vec<DeviceInfo>,
    /// The node's available host bytes and the reading that supplied them.
    pub host_room: (u64, HostRead),
    pub arena_given: Option<u64>,
}

impl NodeCensus {
    /// This process's census, as a tier server sends it.
    pub fn here() -> Result<NodeCensus, GpuError> {
        Ok(NodeCensus {
            devices: crate::census()?,
            host_room: workstation::host_available_read().map_err(|e| GpuError::plan(SERVER, e))?,
            arena_given: workstation::nvtier_arena_given()
                .map_err(|e| GpuError::plan(SERVER, e))?,
        })
    }

    /// Append this census to the node protocol payload.
    pub fn write(&self, o: &mut Out) {
        o.u32(wire::VERSION).u64(self.host_room.0);
        match &self.host_room.1 {
            HostRead::Given => o.u8(0),
            HostRead::MemAvailable => o.u8(1),
            HostRead::VmStats => o.u8(2),
            HostRead::Cgroup { path } => o.u8(3).str(path),
            HostRead::Machine => o.u8(4),
        };
        match self.arena_given {
            Some(bytes) => o.u8(1).u64(bytes),
            None => o.u8(0),
        };
        o.len(self.devices.len());
        for d in &self.devices {
            o.u32(d.ordinal)
                .str(&d.name)
                .u64(d.total_bytes)
                .u64(d.free_bytes)
                .u8(u8::from(d.integrated))
                .u64(d.granule_bytes.map_or(0, NonZeroU64::get))
                .bytes(&d.uuid)
                .str(&d.pci_bus)
                .str(d.held_by.as_deref().unwrap_or(""));
        }
    }

    /// A census as [`NodeCensus::write`] wrote it. Another protocol version,
    /// no device, a device with more free bytes than total and a uuid
    /// listed twice are refused by name.
    fn read(i: &mut In<'_>) -> Result<NodeCensus, String> {
        let v = i.u32("version")?;
        if v != wire::VERSION {
            return Err(format!(
                "a node of protocol version {v}; this process speaks {}",
                wire::VERSION
            ));
        }
        let bytes = i.u64("host room")?;
        let read = match i.u8("host reading")? {
            0 => HostRead::Given,
            1 => HostRead::MemAvailable,
            2 => HostRead::VmStats,
            3 => HostRead::Cgroup {
                path: i.str("host cgroup path")?,
            },
            4 => HostRead::Machine,
            tag => return Err(format!("unknown host reading {tag}")),
        };
        let host_room = (bytes, read);
        let arena_given = if i.flag("arena given")? {
            Some(i.u64("arena bytes")?)
        } else {
            None
        };
        let n = i.len("device count")?;
        if n == 0 {
            return Err("a census of no device".to_string());
        }
        let mut devices: Vec<DeviceInfo> = Vec::new();
        for _ in 0..n {
            let ordinal = i.u32("ordinal")?;
            let name = i.str("device name")?;
            let total_bytes = i.u64("total bytes")?;
            let free_bytes = i.u64("free bytes")?;
            let integrated = i.flag("integrated")?;
            let granule_bytes = NonZeroU64::new(i.u64("allocation granule")?);
            let uuid = <[u8; 16]>::try_from(i.bytes("uuid")?)
                .map_err(|_| "a uuid of other than 16 bytes".to_string())?;
            let pci_bus = i.str("pci bus")?;
            let held = i.str("holders")?;
            if free_bytes > total_bytes {
                return Err(format!(
                    "device {ordinal} {name:?}: {free_bytes} bytes free of {total_bytes}"
                ));
            }
            if devices.iter().any(|d| d.uuid == uuid) {
                return Err(format!("device {ordinal} {name:?}: a uuid listed twice"));
            }
            devices.push(DeviceInfo {
                ordinal,
                name,
                total_bytes,
                free_bytes,
                integrated,
                uuid,
                pci_bus,
                held_by: (!held.is_empty()).then_some(held),
                compute_capability: None,
                granule_bytes,
            });
        }
        Ok(NodeCensus {
            devices,
            host_room,
            arena_given,
        })
    }
}

/// The expert tier cards of the nodes at `addrs`, in tier order, as a plan
/// takes them: each node's census, read over its channel
/// ([`TierNode::census_of`]), and the spec of its one device
/// ([`spec_of_device`]). A node whose census lists more than one device is
/// refused by name: a node serves one tier card. The load path plans its
/// tier cards from it when `BLOOMERY_TIER_NODES` is set.
pub fn node_cards(addrs: &[SocketAddr]) -> Result<Vec<(NodeCensus, CardSpec)>, GpuError> {
    addrs
        .iter()
        .map(|&addr| {
            let census = TierNode::census_of(addr)?;
            match &census.devices[..] {
                [d] => {
                    let spec = spec_of_device(d);
                    Ok((census, spec))
                }
                ds => Err(GpuError::shape(
                    WHAT,
                    format!(
                        "the node at {addr} lists {} devices: a node serves one tier card",
                        ds.len()
                    ),
                )),
            }
        })
        .collect()
}

// ------------------------------------------------------------------ open

/// One tier segment of node 0's plan as the open carries it: the tensor it
/// is a piece of, by name and by what node 0's file holds (its shard, type,
/// dims and file bytes, which the node's file must match), the segment's
/// card format, experts and resident bytes, and the sha256 of the header
/// identity and segment bytes in node 0's file ([`segment_sha256`]), which
/// the node's file must match.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeRow {
    pub name: String,
    pub shard: usize,
    pub ty: String,
    pub dims: Vec<u64>,
    pub file_bytes: u64,
    pub format: String,
    pub experts: Option<Vec<u32>>,
    pub resident_bytes: u64,
    pub sha256: String,
}

/// The sha256 of the header identity followed by the bytes segment `seg`
/// of `t`, a stack of `experts` when it is one, holds in `file`: its spans'
/// bytes in slot order, the bytes its upload reads ([`SegmentFile::of`]).
fn segment_sha256(
    file: &Split,
    experts: u64,
    t: &ModelTensor,
    seg: &Segment,
) -> Result<String, GpuError> {
    let f = SegmentFile::of(file, experts, t, seg)?;
    let header = model::fixture::header_sha256(file);
    Ok(sha256_hex_of(
        std::iter::once(header.as_bytes()).chain(f.bytes),
    ))
}

/// What a tier node opens its tier from ([`TierNode::open`]): the family
/// whose computation it runs, the card by its plan name and its device in
/// the node's census, the plan's device index of the tier's segments and
/// those segments ([`NodeOpen::rows_of`]), the tier's set, the shapes it is
/// cut for and its columns, the model's layers, and the block route's
/// columns with the plan's reserve for it.
#[derive(Clone, Debug)]
pub struct NodeOpen {
    pub nvme: Option<NodeNvme>,
    pub family: String,
    pub name: String,
    pub device: Option<DeviceId>,
    pub card: usize,
    pub rows: Vec<NodeRow>,
    pub set: TierSet,
    pub shape: TierShape,
    pub cols: usize,
    pub layers: usize,
    pub block: (usize, usize),
}

/// A routed host plan scoped to one node. Its row assignment is the open's
/// set; the common split derives the resident and paged rows from this room.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeNvme {
    pub room: u64,
    pub arena: ArenaGrant,
    pub reserve: u64,
}

impl NodeOpen {
    /// Project the assigned routed stacks onto this node's own host room.
    /// Row tables and attention are not part of an expert tier's request.
    pub fn project(&mut self, plan: &Plan<'_>, census: &NodeCensus) -> Result<(), GpuError> {
        self.nvme = None;
        if self
            .rows
            .iter()
            .all(|r| r.format != Format::HostFile.to_string())
        {
            return Ok(());
        }
        let device = census
            .devices
            .iter()
            .find(|d| self.device.is_some_and(|id| id.uuid == d.uuid))
            .ok_or_else(|| GpuError::shape(WHAT, "the projected tier is absent from the census"))?;
        let machine = workstation::plan_on(spec_of_device(device), self.layers);
        let card = plan.machine.card(self.card).ok_or_else(|| {
            GpuError::shape(WHAT, "the projected tier card is absent from the plan")
        })?;
        // Host-only projections have no card totals; their fixed reserves
        // still belong to the machine. Dynamic scratch needs selected totals.
        let reserve = if card.tier_leg.is_some() {
            let totals = plan.cards.get(self.card).ok_or_else(|| {
                GpuError::shape(WHAT, "the projected tier card is absent from the plan")
            })?;
            totals
                .context_bytes
                .checked_add(totals.scratch_bytes)
                .and_then(|bytes| bytes.checked_add(totals.reserve_bytes))
        } else {
            card.set_aside_bytes()
        }
        .ok_or_else(|| GpuError::shape(WHAT, "the projected tier reserve overflows"))?;
        let mut projected = model::placement::nodes::routed_host(
            plan.model,
            &plan.rows,
            self.card,
            &machine,
            self.cols as u64,
        )
        .map_err(|e| GpuError::plan(WHAT, e))?;
        projected.host.reserve_bytes = workstation::HostNeed::of(&projected, 0)
            .os
            .checked_add(reserve)
            .ok_or_else(|| GpuError::shape(WHAT, "the projected host reserve overflows"))?;
        model::placement::expert_nvme_tier(&mut projected, census.host_room.0, census.arena_given)
            .map_err(|e| {
                GpuError::plan(
                    WHAT,
                    format!(
                        "the node's host room {} B ({}): {e}",
                        census.host_room.0, census.host_room.1
                    ),
                )
            })?;
        self.nvme = Some(NodeNvme {
            room: census.host_room.0,
            arena: ArenaGrant::whole(projected.host.nvme_arena_bytes),
            reserve,
        });
        Ok(())
    }

    /// Open the assigned tier locally or on its node, with the same paging projection.
    pub fn open_tier(
        self,
        stage: &Gpu,
        file: &Arc<Split>,
        plan: &Plan<'_>,
        node: Option<SocketAddr>,
        family: &NodeFamily,
        card_dontneed: bool,
    ) -> Result<TierCard, GpuError> {
        self.open_tier_from(
            stage,
            (file, &CardBytes::File),
            plan,
            node,
            family,
            card_dontneed,
        )
    }

    /// [`NodeOpen::open_tier`] whose card tensors come from `bytes`: the file's own segments, or
    /// a checkpoint folder's part source, which loads the tier's segments as it loads the stage
    /// card's. A part source's tier runs in this process: a node reads its own model file, and a
    /// paged tier its file's runs, so either is refused by name for one.
    pub fn open_tier_from(
        mut self,
        stage: &Gpu,
        (file, bytes): (&Arc<Split>, &CardBytes<'_>),
        plan: &Plan<'_>,
        node: Option<SocketAddr>,
        family: &NodeFamily,
        card_dontneed: bool,
    ) -> Result<TierCard, GpuError> {
        let parts = match bytes {
            CardBytes::File => None,
            CardBytes::Parts(load) => Some(*load),
        };
        if parts.is_some() && node.is_some() {
            return Err(GpuError::shape(
                WHAT,
                "a tier node of a checkpoint folder: a node loads its tier from its own GGUF file",
            ));
        }
        let census = match node {
            Some(addr) => TierNode::census_of(addr)?,
            None => NodeCensus::here()?,
        };
        self.project(plan, &census)?;
        if parts.is_some() && self.nvme.is_some() {
            return Err(GpuError::shape(
                WHAT,
                "a paged tier of a checkpoint folder: the paged tier reads its GGUF file's runs",
            ));
        }
        let card = if let Some(addr) = node {
            TierCard::on_node(TierNode::open(addr, &self)?, self.name, self.set)
        } else {
            let gpu = Gpu::open_card(&self.name, self.device)?;
            let tensors = (family.tensors)(file)?;
            let (w, experts) = if self.nvme.is_some() {
                self.load_host(&gpu, file, &tensors, family)?
            } else {
                let w = match parts {
                    None => {
                        Weights::load_placed(gpu.stream(), file, plan, self.card, card_dontneed)?
                    }
                    Some(load) => {
                        let available =
                            workstation::host_available().map_err(|e| GpuError::plan(WHAT, e))?;
                        let room = load.host_bytes(plan, self.card, available)?;
                        load.load(gpu.stream(), plan, self.card, card_dontneed, room)?
                            .0
                    }
                };
                let experts = (family.experts)(&gpu, file, &tensors, &w, &self)?;
                (w, experts)
            };
            TierCard::open_cols(gpu, self.name, w, self.set, experts, self.shape, self.cols)?
        };
        stage.context().bind_to_thread()?;
        Ok(card)
    }

    /// Load the projected host tier through the same executor locally or on a node.
    pub(crate) fn load_host(
        &self,
        gpu: &Gpu,
        file: &Arc<Split>,
        tensors: &ModelTensors,
        family: &NodeFamily,
    ) -> Result<(Weights, Box<dyn TierExperts>), GpuError> {
        let paging = self
            .nvme
            .as_ref()
            .ok_or_else(|| GpuError::shape(SERVER, "no node NVMe plan"))?;
        let devices = crate::census()?;
        let device = devices
            .iter()
            .find(|d| self.device.is_some_and(|id| id.uuid == d.uuid))
            .ok_or_else(|| {
                GpuError::shape(SERVER, "the tier device disappeared from its census")
            })?;
        let machine = workstation::plan_on(spec_of_device(device), self.layers);
        let plan = self.host_plan(tensors, &machine)?;
        let floor = if plan.host.nvme_expert_bytes == 0 {
            0
        } else {
            model::placement::ArenaFloor::of(&plan)
                .map_err(|e| GpuError::plan(SERVER, e))?
                .bytes
        };
        // The projection's room was this host's reading too, so the fresh
        // reading names the room either way.
        let (now, read) =
            workstation::host_available_read().map_err(|e| GpuError::plan(SERVER, e))?;
        let mut arena = workstation::HostNeed::of(&plan, 0)
            .grant(&(now.min(paging.room), read), floor)
            .map_err(|e| GpuError::plan(SERVER, e))?;
        arena.granted = arena.granted.min(paging.arena.granted);
        let tier = crate::model::nvme_tier(gpu.context(), &plan, file, false, arena)?;
        let run = (family.host)(Arc::clone(file), self)?;
        let experts = super::tier_cpu::CpuExperts::new(
            gpu,
            run,
            tier,
            self.set.clone(),
            self.shape,
            self.cols,
        )?;
        let w = Weights::load_rows(gpu.stream(), file, tensors, &[], self.card)?;
        Ok((w, Box::new(experts)))
    }

    pub(super) fn host_plan<'a>(
        &self,
        model: &'a ModelTensors,
        machine: &'a model::placement::Machine,
    ) -> Result<Plan<'a>, GpuError> {
        let paging = self
            .nvme
            .as_ref()
            .ok_or_else(|| GpuError::shape(SERVER, "no node NVMe plan"))?;
        if (paging.arena.granted == 0) != (paging.arena.planned == 0)
            || paging.arena.granted > paging.arena.planned
        {
            return Err(GpuError::shape(SERVER, "invalid node NVMe arena grant"));
        }
        let rows = self.rows_on(model)?;
        let mut projected = model::placement::nodes::routed_host(
            model,
            &rows,
            self.card,
            machine,
            self.cols as u64,
        )
        .map_err(|e| GpuError::plan(SERVER, e))?;
        projected.host.reserve_bytes = workstation::HostNeed::of(&projected, 0)
            .os
            .checked_add(paging.reserve)
            .ok_or_else(|| GpuError::shape(SERVER, "the projected host reserve overflows"))?;
        model::placement::expert_nvme_tier(&mut projected, paging.room, Some(paging.arena.planned))
            .map_err(|e| GpuError::plan(SERVER, e))?;
        if projected.host.nvme_arena_bytes != paging.arena.planned {
            return Err(GpuError::shape(
                SERVER,
                "the node's routed split disagrees with its arena grant",
            ));
        }
        Ok(projected)
    }

    /// The segments `plan` puts on its card `card`, each with its tensor's
    /// name and file identity and the sha256 of its header and bytes in `file`, the
    /// model file the plan was made from.
    pub fn rows_of(plan: &Plan<'_>, file: &Split, card: usize) -> Result<Vec<NodeRow>, GpuError> {
        Self::rows_by(plan, card, |t, s| {
            segment_sha256(file, plan.model.experts, t, s)
        })
    }

    /// [`NodeOpen::rows_of`] for a tier whose segments a part source loads
    /// ([`NodeOpen::open_tier_from`]): no digest, which only a node that
    /// reads its own file checks, and a part source's tier runs in this
    /// process.
    pub fn rows_unhashed(plan: &Plan<'_>, card: usize) -> Result<Vec<NodeRow>, GpuError> {
        Self::rows_by(plan, card, |_, _| Ok(String::new()))
    }

    /// The segments `plan` puts on card `card`, each row's digest `digest`'s.
    fn rows_by(
        plan: &Plan<'_>,
        card: usize,
        digest: impl Fn(&ModelTensor, &Segment) -> Result<String, GpuError>,
    ) -> Result<Vec<NodeRow>, GpuError> {
        let mut rows = Vec::new();
        for row in &plan.rows {
            let t = plan.model.tensors.get(row.tensor).ok_or_else(|| {
                GpuError::shape(WHAT, format!("a plan row names tensor {}", row.tensor))
            })?;
            for s in row
                .segments
                .iter()
                .filter(|s| s.device == Device::Card(card))
            {
                rows.push(NodeRow {
                    name: t.name.clone(),
                    shard: t.shard,
                    ty: t.ty.to_string(),
                    dims: t.dims.clone(),
                    file_bytes: t.file_bytes,
                    format: s.format.to_string(),
                    experts: s.experts.as_ref().map(|e| e.ids().to_vec()),
                    resident_bytes: s.resident_bytes,
                    sha256: digest(t, s)?,
                });
            }
        }
        Ok(rows)
    }

    /// The open's segments as placement rows over `model`, the node's own
    /// file's tensors: each tensor found by name and held as node 0's plan
    /// holds it, else refused by name: the two processes read other files.
    /// Each segment is a routed stack's experts of its layer, exactly the
    /// ids the open's set gives that layer, in its order, else refused by
    /// name: the stage side's slots name the set's experts. The layer count
    /// must match this file, the set must fit it, and columns must fit the
    /// step port, before any card is opened. Every routed stack on a layer
    /// with assigned ids appears once; empty layers need no row.
    pub fn rows_on(&self, model: &ModelTensors) -> Result<Vec<Row>, GpuError> {
        if self.layers != model.layers
            || self.set.layers().end > model.layers
            || !(1..=DEFER_MAX_COLS).contains(&self.cols)
        {
            return Err(GpuError::shape(
                SERVER,
                format!(
                    "an open of {} layers (set {:?}), {} columns; this file has {} layers, \
                     the step port serves 1..={DEFER_MAX_COLS} columns",
                    self.layers,
                    self.set.layers(),
                    self.cols,
                    model.layers
                ),
            ));
        }
        let mut rows = Vec::with_capacity(self.rows.len());
        let mut seen = vec![false; model.tensors.len()];
        for r in &self.rows {
            let Some((i, t)) = model
                .tensors
                .iter()
                .enumerate()
                .find(|(_, t)| t.name == r.name)
            else {
                return Err(GpuError::shape(
                    SERVER,
                    format!("{}: node 0's plan names a tensor this file lacks", r.name),
                ));
            };
            if std::mem::replace(&mut seen[i], true) {
                return Err(GpuError::shape(
                    SERVER,
                    format!("{}: a routed stack assigned more than once", r.name),
                ));
            }
            if t.shard != r.shard
                || t.ty.to_string() != r.ty
                || t.dims != r.dims
                || t.file_bytes != r.file_bytes
            {
                return Err(GpuError::shape(
                    SERVER,
                    format!(
                        "{}: this file holds shard {} {} {:?} ({} bytes), node 0's plan shard {} \
                         {} {:?} ({} bytes): the two processes read other files",
                        r.name,
                        t.shard,
                        t.ty,
                        t.dims,
                        t.file_bytes,
                        r.shard,
                        r.ty,
                        r.dims,
                        r.file_bytes
                    ),
                ));
            }
            let set = t.layer.and_then(|l| self.set.ids(l));
            if set.is_none() || r.experts.as_deref() != set {
                return Err(GpuError::shape(
                    SERVER,
                    format!(
                        "{} (layer {:?}): node 0's plan gives the tier the experts {:?}, its set \
                         {set:?}",
                        r.name, t.layer, r.experts
                    ),
                ));
            }
            let format = Format::of_word(&r.format).ok_or_else(|| {
                GpuError::shape(SERVER, format!("{}: a format {:?}", r.name, r.format))
            })?;
            let experts = r
                .experts
                .clone()
                .map(|ids| ExpertList::new(ids, model.experts))
                .transpose()
                .map_err(|e| GpuError::plan(SERVER, e))?;
            rows.push(Row {
                tensor: i,
                segments: vec![Segment {
                    device: Device::Card(self.card),
                    format,
                    experts,
                    resident_bytes: r.resident_bytes,
                }],
                read_bytes: 0,
                stage: 0,
            });
        }
        if let Some((_, t)) = model.tensors.iter().enumerate().find(|(i, t)| {
            t.role == Role::RoutedExperts
                && t.layer
                    .and_then(|l| self.set.ids(l))
                    .is_some_and(|ids| !ids.is_empty())
                && !seen[*i]
        }) {
            return Err(GpuError::shape(
                SERVER,
                format!("{}: missing routed stack for its set", t.name),
            ));
        }
        Ok(rows)
    }

    /// The open as its request carries it; a layer of the set's range with
    /// no row is refused by name.
    pub fn write(&self, o: &mut Out) -> Result<(), GpuError> {
        match &self.nvme {
            Some(p) => {
                o.u8(1)
                    .u64(p.room)
                    .u64(p.arena.planned)
                    .u64(p.arena.granted)
                    .u64(p.reserve);
            }
            None => {
                o.u8(0);
            }
        }
        o.str(&self.family).str(&self.name);
        match self.device {
            Some(d) => o.u8(1).u32(d.ordinal).bytes(&d.uuid),
            None => o.u8(0),
        };
        o.len(self.card).len(self.rows.len());
        for r in &self.rows {
            o.str(&r.name)
                .len(r.shard)
                .str(&r.ty)
                .u64s(&r.dims)
                .u64(r.file_bytes)
                .str(&r.format);
            match &r.experts {
                Some(ids) => o.u8(1).u32s(ids),
                None => o.u8(0),
            };
            o.u64(r.resident_bytes).str(&r.sha256);
        }
        let layers = self.set.layers();
        o.len(layers.start).len(layers.end).len(self.set.n_expert());
        for l in layers {
            let ids = self.set.ids(l).ok_or_else(|| {
                GpuError::shape(
                    WHAT,
                    format!("layer {l}: the tier's set holds no row of it"),
                )
            })?;
            o.u32s(ids);
        }
        let s = self.shape;
        o.len(s.hidden).len(s.n_used).len(s.rows).u8(match s.act {
            TierAct::Q8 => 0,
            TierAct::F32 => 1,
        });
        o.u8(match s.reply {
            ReplyRows::Whole => 0,
            ReplyRows::Packed => 1,
        });
        o.len(self.cols)
            .len(self.layers)
            .len(self.block.0)
            .len(self.block.1);
        Ok(())
    }

    fn read(i: &mut In<'_>) -> Result<NodeOpen, String> {
        let nvme = if i.flag("node NVMe")? {
            let room = i.u64("node room")?;
            let arena = ArenaGrant {
                planned: i.u64("planned arena")?,
                granted: i.u64("granted arena")?,
            };
            let reserve = i.u64("node reserve")?;
            if (arena.granted == 0) != (arena.planned == 0) || arena.granted > arena.planned {
                return Err("invalid node NVMe arena grant".into());
            }
            Some(NodeNvme {
                room,
                arena,
                reserve,
            })
        } else {
            None
        };
        let family = i.str("family")?;
        let name = i.str("card name")?;
        let device = if i.flag("device")? {
            Some(DeviceId {
                ordinal: i.u32("ordinal")?,
                uuid: <[u8; 16]>::try_from(i.bytes("uuid")?)
                    .map_err(|_| "a uuid of other than 16 bytes".to_string())?,
            })
        } else {
            None
        };
        let card = i.len("card")?;
        let n = i.len("row count")?;
        let mut rows = Vec::new();
        for _ in 0..n {
            let name = i.str("tensor name")?;
            let shard = i.len("shard")?;
            let ty = i.str("type")?;
            let dims = i.u64s("dims")?;
            let file_bytes = i.u64("file bytes")?;
            let format = i.str("format")?;
            let experts = if i.flag("experts")? {
                Some(i.u32s("expert ids")?)
            } else {
                None
            };
            rows.push(NodeRow {
                name,
                shard,
                ty,
                dims,
                file_bytes,
                format,
                experts,
                resident_bytes: i.u64("resident bytes")?,
                sha256: i.str("sha256")?,
            });
        }
        let (start, end, n_expert) = (
            i.len("first layer")?,
            i.len("end layer")?,
            i.len("experts")?,
        );
        let ids = (start..end)
            .map(|_| i.u32s("set row"))
            .collect::<Result<Vec<_>, _>>()?;
        let set = TierSet::new(start..end, n_expert, ids).map_err(|e| e.to_string())?;
        let (hidden, n_used, rows_) = (i.len("hidden")?, i.len("n_used")?, i.len("rows")?);
        let act = match i.u8("activation")? {
            0 => TierAct::Q8,
            1 => TierAct::F32,
            a => return Err(format!("an activation form {a}")),
        };
        let reply = match i.u8("reply rows")? {
            0 => ReplyRows::Whole,
            1 => ReplyRows::Packed,
            r => return Err(format!("a reply of rows {r}")),
        };
        Ok(NodeOpen {
            nvme,
            family,
            name,
            device,
            card,
            rows,
            set,
            shape: TierShape {
                hidden,
                n_used,
                rows: rows_,
                act,
                reply,
            },
            cols: i.len("columns")?,
            layers: i.len("layers")?,
            block: (i.len("block columns")?, i.len("block reserve")?),
        })
    }
}

/// The words a request carries a chain as: 0 the step, 1 the pair, 2 the
/// quad, `16 + m` a chain of `m` columns.
const COLS_WORD: u32 = 16;

/// A chain as a request carries it; a column count past the word is
/// refused by name.
fn chain_word(c: Chain) -> Result<u32, GpuError> {
    match c {
        Chain::Step => Ok(0),
        Chain::Pair => Ok(1),
        Chain::Quad => Ok(2),
        Chain::Cols(m) => u32::try_from(m)
            .ok()
            .and_then(|m| m.checked_add(COLS_WORD))
            .ok_or_else(|| GpuError::shape(WHAT, format!("a chain of {m} columns"))),
    }
}

fn chain_of(w: u32) -> Result<Chain, String> {
    let chain = match w {
        0 => Chain::Step,
        1 => Chain::Pair,
        2 => Chain::Quad,
        COLS_WORD.. => Chain::Cols((w - COLS_WORD) as usize),
        _ => return Err(format!("a chain word {w}")),
    };
    chain_slot(chain).map_err(|e| e.to_string())?;
    Ok(chain)
}

/// A page-word side as a capture's reply carries it.
fn side_byte(side: WordSide) -> u8 {
    match side {
        WordSide::Memop => 0,
        WordSide::Kernel => 1,
    }
}

fn side_of(b: u8) -> Result<WordSide, String> {
    match b {
        0 => Ok(WordSide::Memop),
        1 => Ok(WordSide::Kernel),
        _ => Err(format!("a page-word side {b}")),
    }
}

// ---------------------------------------------------------------- errors

/// The kinds of error a [`Tag::Err`] reply carries, so node 0 raises the
/// server's kind: a refusal, a fault of the tier's card (with its words),
/// or the card and the protocol out of step.
const ERR_REFUSED: u8 = 0;
const ERR_FAULT: u8 = 1;
const ERR_PROTOCOL: u8 = 2;

/// `e` as a [`Tag::Err`] reply's payload: its kind, a fault's words, and
/// its text.
fn write_err(e: &GpuError, o: &mut Out) {
    o.clear();
    match e {
        GpuError::Fault { fault, .. } => o.u8(ERR_FAULT).u32(fault.word()).u32(fault.sites),
        GpuError::Protocol { .. } | GpuError::Stalled { .. } => o.u8(ERR_PROTOCOL),
        _ => o.u8(ERR_REFUSED),
    };
    o.str(&e.to_string());
}

/// The error a [`Tag::Err`] reply `body` from the node at `addr` to the
/// request `what` carries, of the kind the server raised; a payload of
/// another form is the error's text.
/// Run a block on the tier's card: the request's tier places and q8_1
/// planes landed in the staging from column 0, the tier's experts over the
/// block of `cols` columns ([`TierWork::enqueue_block`], the path a tier
/// card of node 0 runs), the down outputs of the first `copied` slots
/// copied to the page-locked rows. Returns once the tier's stream drained,
/// so the request's bytes and the rows are free.
fn run_block(
    work: &mut TierWork,
    b: &mut ServerBlock,
    (layer, cols): (usize, usize),
    places: &[u32],
    (q3, d8): (&[u8], &[u8]),
    copied: usize,
) -> Result<(), GpuError> {
    let n = work.shape().hidden;
    {
        let g = work.gpu();
        g.context().bind_to_thread()?;
        let st = &mut b.staging;
        let (aq3, ad8) = st.act.q3_d8_mut();
        // SAFETY: the request's bytes outlive the copies: this fn drains the
        // tier's stream before it returns.
        unsafe {
            htod_bytes(g.stream(), &mut st.tsel, bytes_of(places))?;
            htod_bytes(g.stream(), aq3, q3)?;
            htod_bytes(g.stream(), ad8, d8)?;
        }
    }
    let st = &mut b.staging;
    work.enqueue_block(
        layer,
        TierBlock {
            x: &st.x,
            act: &st.act,
            sel: &st.tsel,
            cols,
            down: &mut st.y,
        },
    )?;
    if copied > 0 {
        // SAFETY: the rows are read only after the drain below.
        unsafe { dtoh(work.gpu().stream(), &mut b.host, &st.y, 0..copied * n)? };
    }
    work.drain(DRAIN_DEADLINE, "tier_server: a block")
}

fn read_err(addr: SocketAddr, what: &str, body: &[u8]) -> Result<GpuError, String> {
    let mut i = In::new(body);
    let kind = i.u8("error kind")?;
    let words = if kind == ERR_FAULT {
        Some((i.u32("fault word")?, i.u32("fault sites")?))
    } else {
        None
    };
    let text = i.str("error text")?;
    i.end()?;
    let detail = format!("the node at {addr} refused the {what} request: {text}");
    match (kind, words) {
        (ERR_REFUSED, _) => Ok(GpuError::shape(WHAT, detail)),
        (ERR_PROTOCOL, _) => Ok(GpuError::protocol(WHAT, detail)),
        (ERR_FAULT, Some((w, s))) => match Fault::from_words(w, s) {
            Some(fault) => Ok(GpuError::Fault {
                what: WHAT,
                fault,
                behind: Some(Box::new(GpuError::link(WHAT, detail))),
            }),
            None => Err(format!("a fault reply with a clean word: {text}")),
        },
        (k, _) => Err(format!("an error of kind {k}: {text}")),
    }
}

/// Write a shared-file Bind payload, refusing a path that is not UTF-8.
pub fn write_file_bind(
    path: &Path,
    bytes: usize,
    tiers: usize,
    index: usize,
    out: &mut Out,
) -> Result<(), GpuError> {
    let path = path.to_str().ok_or_else(|| {
        GpuError::shape(WHAT, format!("a page path {} not UTF-8", path.display()))
    })?;
    out.u8(0).str(path).len(bytes).len(tiers).len(index);
    Ok(())
}

/// The bind mode of a remote bind over `peer`'s wire: 1 RDMA, 2 TCP. A
/// remote bind's request is its mode, the page's tiers and the tier's index,
/// then the stage's end's peer; its reply is the mode and the tier's end's
/// peer.
fn peer_mode(peer: &Peer) -> u8 {
    match peer {
        Peer::Rdma(_) => 1,
        Peer::Tcp(_) => 2,
    }
}

/// `peer`'s fields as a remote bind of its mode carries them.
fn write_peer(peer: &Peer, out: &mut Out) {
    match peer {
        Peer::Rdma(e) => write_endpoint(e, out),
        Peer::Tcp(t) => write_tcp_peer(t, out),
    }
}

/// The peer a remote bind of mode `mode` carries; a mode that is not a
/// remote bind's is refused by name.
fn read_peer(mode: u8, i: &mut In<'_>) -> Result<Peer, String> {
    match mode {
        1 => read_endpoint(i).map(Peer::Rdma),
        2 => read_tcp_peer(i).map(Peer::Tcp),
        m => Err(format!("unknown remote bind mode {m}")),
    }
}

/// A remote bind's reply: its mode and the tier's end's peer, the whole
/// payload.
fn read_peer_reply(body: &[u8]) -> Result<Peer, String> {
    let mut i = In::new(body);
    let mode = i.u8("bind mode")?;
    let peer = read_peer(mode, &mut i)?;
    i.end()?;
    Ok(peer)
}

/// A TCP end's address, as 16 bytes (an IPv4 address IPv6-mapped), its
/// port and its nonce.
pub(crate) fn write_tcp_peer(t: &TcpPeer, out: &mut Out) {
    let ip = match t.addr.ip() {
        std::net::IpAddr::V4(a) => a.to_ipv6_mapped(),
        std::net::IpAddr::V6(a) => a,
    };
    out.bytes(&ip.octets())
        .u32(u32::from(t.addr.port()))
        .u64(t.nonce);
}

/// [`write_tcp_peer`]'s fields, an IPv6-mapped address read back as the
/// IPv4 address it maps; an unspecified address or a port of 0 or past
/// `u16` is refused by name.
pub(crate) fn read_tcp_peer(i: &mut In<'_>) -> Result<TcpPeer, String> {
    let octets: [u8; 16] = i
        .bytes("TCP address")?
        .try_into()
        .map_err(|_| "a TCP address is 16 bytes")?;
    let ip = std::net::Ipv6Addr::from(octets).to_canonical();
    let port = i.u32("TCP port")?;
    let nonce = i.u64("TCP nonce")?;
    match u16::try_from(port) {
        Ok(port) if port != 0 && !ip.is_unspecified() => Ok(TcpPeer {
            addr: SocketAddr::new(ip, port),
            nonce,
        }),
        _ => Err(format!("an invalid TCP endpoint: {ip} port {port}")),
    }
}

pub(super) fn write_endpoint(e: &bloomery_verbs::Endpoint, out: &mut Out) {
    out.u32(e.qpn)
        .u32(e.psn)
        .bytes(&e.gid.raw)
        .u64(e.addr)
        .u32(e.rkey);
}

pub(super) fn read_endpoint(i: &mut In<'_>) -> Result<bloomery_verbs::Endpoint, String> {
    let qpn = i.u32("queue pair")?;
    let psn = i.u32("packet sequence")?;
    let gid = bloomery_verbs::Gid {
        raw: i
            .bytes("GID")?
            .try_into()
            .map_err(|_| "a GID is 16 bytes")?,
    };
    let addr = i.u64("region address")?;
    let rkey = i.u32("region key")?;
    if qpn == 0 || qpn > 0xff_ffff || psn > 0xff_ffff || addr == 0 {
        return Err("an invalid RDMA endpoint".to_string());
    }
    Ok(bloomery_verbs::Endpoint {
        qpn,
        psn,
        gid,
        addr,
        rkey,
    })
}

// ---------------------------------------------------------------- client

/// Node 0's end of a tier node's channel: each [`TierWork`] step the stage
/// side asks of the tier is a request, answered once the server has taken
/// it. A channel that fails, times out or carries what the protocol does
/// not marks the node lost ([`TierNode::lost`]) and is ended, so a node
/// that runs again finds node 0's end; every later request fails by name.
pub struct TierNode {
    addr: SocketAddr,
    ch: Channel,
    /// A request's payload, its buffer kept from one request to the next.
    req: Out,
    /// The device of the node's census the tier opened on.
    device: DeviceInfo,
    shape: TierShape,
    cols: usize,
    planes: (usize, usize),
    resident_bytes: usize,
    device_bytes: usize,
    /// Per chain, the nodes of the graph the server holds for it.
    graphs: [Option<usize>; TIER_CHAINS],
    /// How the node's card adds to the page's words, as its captures'
    /// replies name it; `None` before the first.
    words: Option<WordSide>,
    /// The node's block staging, once its block open made one.
    block: Option<NodeBlock>,
    /// A block request sent whose reply is unread: its slots and the tier
    /// slots it sent.
    pending: Option<(usize, usize)>,
    /// The frames of the step's passes ([`TierNode::send_enqueue`],
    /// [`TierNode::send_end`]).
    frames: PassFrames,
    control_round_trips: u64,
    lost: Option<String>,
}

/// A node tier's pass frames, each sent without waiting for its reply: the
/// replies unread, and whether the node holds a pass open.
#[derive(Default)]
struct PassFrames {
    /// Enqueues sent whose replies are unread.
    enqueues: u32,
    /// An end sent whose reply is unread. Every send reads the replies before
    /// it, so the end was sent before each unread enqueue and its reply comes
    /// first.
    end: bool,
    /// An enqueue was sent since the last end: the node holds the pass open
    /// until this process ends it.
    open: bool,
}

/// Connect to the node at `addr` and read its census.
fn connect(addr: SocketAddr) -> Result<(Channel, NodeCensus), GpuError> {
    let lost = |e: String| GpuError::link(WHAT, format!("the node at {addr}: {e}"));
    let s = TcpStream::connect_timeout(&addr, REPLY).map_err(|e| lost(e.to_string()))?;
    let mut ch = Channel::new(s).map_err(|e| lost(e.to_string()))?;
    let tag = ch
        .recv(Instant::now() + REPLY)
        .map_err(|e| lost(format!("its census: {e}")))?;
    if tag != Tag::Census {
        return Err(lost(format!("a {tag:?} frame where its census belongs")));
    }
    let mut i = In::new(ch.body());
    let census = NodeCensus::read(&mut i)
        .and_then(|c| i.end().map(|()| c))
        .map_err(|e| lost(format!("its census: {e}")))?;
    Ok((ch, census))
}

impl TierNode {
    /// The census of the node at `addr`: what it sends on connect, before
    /// any open. The plan of a tier on that node reads it ([`node_cards`]).
    pub fn census_of(addr: SocketAddr) -> Result<NodeCensus, GpuError> {
        let (mut ch, census) = connect(addr)?;
        let deadline = Instant::now() + REPLY;
        drop(
            ch.send(Tag::Close, &[], deadline)
                .and_then(|()| ch.recv(deadline)),
        );
        Ok(census)
    }

    /// The tier `open` names, opened on the node at `addr`: its device must
    /// be one the node sends in its census on this connection. The node's
    /// load is waited for under [`open_bound`]. Load-time only.
    pub fn open(addr: SocketAddr, open: &NodeOpen) -> Result<TierNode, GpuError> {
        let (ch, census) = connect(addr)?;
        let device = open
            .device
            .and_then(|d| census.devices.iter().find(|c| c.uuid == d.uuid))
            .ok_or_else(|| {
                GpuError::shape(
                    WHAT,
                    format!(
                        "the tier {} on {:?}: the node at {addr} lists no such device in its \
                         census of {} devices, and a tier on a node opens by its device",
                        open.name,
                        open.device,
                        census.devices.len()
                    ),
                )
            })?
            .clone();
        let mut node = TierNode {
            addr,
            ch,
            req: Out::default(),
            device,
            shape: open.shape,
            cols: open.cols,
            planes: (0, 0),
            resident_bytes: 0,
            device_bytes: 0,
            graphs: [None; TIER_CHAINS],
            words: None,
            block: None,
            pending: None,
            frames: PassFrames::default(),
            control_round_trips: 0,
            lost: None,
        };
        open.write(node.req.clear())?;
        node.call(Tag::Open, open_bound(open)?, "open")?;
        let mut i = In::new(node.ch.body());
        let r = (|| {
            let planes = (i.len("codes")?, i.len("scales")?);
            let resident = i.len("resident bytes")?;
            let device = i.len("device bytes")?;
            i.end()?;
            Ok::<_, String>((planes, resident, device))
        })();
        let (planes, resident, device) = r.map_err(|e| node.lose_now("open", e))?;
        node.planes = planes;
        node.resident_bytes = resident;
        node.device_bytes = device;
        Ok(node)
    }

    /// Mark the node lost by `why` in the request `what` and end its
    /// channel, as an error.
    fn lose_now(&mut self, what: &str, why: String) -> GpuError {
        let why = format!("the {what} request: {why}");
        let e = GpuError::link(WHAT, format!("the node at {}: {why}", self.addr));
        self.ch.shutdown();
        self.lost = Some(why);
        e
    }

    /// Send the request `tag` with the payload [`TierNode::req`] holds and
    /// read its reply, both within `within`; the reply's payload is then
    /// [`Channel::body`]. A refusal by the server is its error, of the kind
    /// it raised; a channel that fails loses the node.
    fn call(&mut self, tag: Tag, within: Duration, what: &'static str) -> Result<(), GpuError> {
        self.ready(what)?;
        let deadline = Instant::now() + within;
        let r = self
            .ch
            .send(tag, self.req.done(), deadline)
            .and_then(|()| self.ch.recv(deadline));
        match r {
            Ok(Tag::Ok) => {
                self.control_round_trips += 1;
                Ok(())
            }
            Ok(Tag::Err) => match read_err(self.addr, what, self.ch.body()) {
                Ok(e) => Err(e),
                Err(why) => Err(self.lose_now(what, why)),
            },
            Ok(t) => Err(self.lose_now(what, format!("a {t:?} frame where a reply belongs"))),
            Err(e) => Err(self.lose_now(what, e.to_string())),
        }
    }

    /// Refuse a request `what` to a lost node by name; with a block's reply
    /// or a pass frame's unread, read it first and set it aside, whatever it
    /// carries, so the request's reply is the next frame. A channel that
    /// fails loses the node.
    fn ready(&mut self, what: &str) -> Result<(), GpuError> {
        if let Some(why) = &self.lost {
            return Err(GpuError::link(
                WHAT,
                format!(
                    "the node at {} is lost ({why}): no {what} request",
                    self.addr
                ),
            ));
        }
        self.drain_pass(Instant::now() + REPLY)?;
        if self.pending.take().is_some() {
            match self.ch.recv(Instant::now() + REPLY) {
                Ok(Tag::Ok | Tag::Err) => self.control_round_trips += 1,
                Ok(t) => {
                    let why = format!("a {t:?} frame where a block's reply belongs");
                    return Err(self.lose_now("block", why));
                }
                Err(e) => return Err(self.lose_now("block", e.to_string())),
            }
        }
        Ok(())
    }

    /// Open the node's block staging for a prompt batch's blocks of up to
    /// `cap` tokens: its staging and block scratch bytes and where its
    /// architecture leaves a block's rows ([`NodeBlock`]), or `None` when
    /// its tier runs no block from the q8_1 planes, whose experts a batch's
    /// host then serves. A node that cannot hold the staging refuses by
    /// name. Load-time only.
    pub(super) fn block_open(&mut self, cap: usize) -> Result<Option<NodeBlock>, GpuError> {
        self.req.clear().len(cap);
        self.call(Tag::BlockOpen, REPLY, "block open")?;
        let block = block::read_open_reply(self.ch.body(), cap)
            .map_err(|e| self.lose_now("block open", e))?;
        self.block = block;
        Ok(block)
    }

    /// The node's block staging, once its block open made one.
    pub(super) fn block(&self) -> Option<NodeBlock> {
        self.block
    }

    /// Send layer `layer`'s block of `cols` tokens: each slot's tier place
    /// and the columns' q3 plane and scales, of whose slots `sent` are the
    /// node's. Returns once the request is written; [`TierNode::block_rows`]
    /// reads its reply.
    pub(super) fn block_send(
        &mut self,
        (layer, cols): (usize, usize),
        places: &[u32],
        (q3, d8): (&[u8], &[u8]),
        sent: usize,
    ) -> Result<(), GpuError> {
        self.ready("block")?;
        self.req.clear();
        block::write_request(&mut self.req, (layer, cols), places, q3, d8);
        let written = self
            .ch
            .send(Tag::Block, self.req.done(), Instant::now() + REPLY);
        written.map_err(|e| self.lose_now("block", e.to_string()))?;
        self.pending = Some((places.len(), sent));
        Ok(())
    }

    /// The rows of the block [`TierNode::block_send`] sent, its reply read
    /// by `deadline`: one a tier slot it sent, in slot order. A refusal by
    /// the node is its error; a reply of other rows, or a channel that
    /// fails, loses the node.
    pub(super) fn block_rows(&mut self, deadline: Instant) -> Result<&[u8], GpuError> {
        let (slots, sent) = self.pending.take().ok_or(GpuError::State {
            what: WHAT,
            missing: "a block request sent (TierNode::block_send)",
        })?;
        match self.ch.recv(deadline) {
            Ok(Tag::Ok) => self.control_round_trips += 1,
            Ok(Tag::Err) => {
                self.control_round_trips += 1;
                return match read_err(self.addr, "block", self.ch.body()) {
                    Ok(e) => Err(e),
                    Err(why) => Err(self.lose_now("block", why)),
                };
            }
            Ok(t) => {
                let why = format!("a {t:?} frame where a block's reply belongs");
                return Err(self.lose_now("block", why));
            }
            Err(e) => return Err(self.lose_now("block", e.to_string())),
        }
        let hidden = self.shape.hidden;
        if let Err(why) = block::read_reply(self.ch.body(), slots, hidden, sent) {
            return Err(self.lose_now("block", why));
        }
        block::read_reply(self.ch.body(), slots, hidden, sent)
            .map_err(|e| GpuError::protocol(WHAT, e))
    }

    /// A request whose reply carries nothing.
    fn ask(&mut self, tag: Tag, within: Duration, what: &'static str) -> Result<(), GpuError> {
        self.call(tag, within, what)?;
        let end = In::new(self.ch.body()).end();
        end.map_err(|e| self.lose_now(what, e))
    }

    /// [`TierWork::enqueue`] on the node, sent without its reply: one layer a
    /// node tier serves on the host's own reading of the picks, whose frame
    /// names its layer, so a layer the host skips is simply never sent. The
    /// first since the last end opens a pass on the node, which
    /// [`TierNode::send_end`] ends. The replies are read by
    /// [`TierNode::drain_pass`], in the order the frames were sent; a refusal
    /// surfaces there.
    pub(super) fn send_enqueue(
        &mut self,
        layer: usize,
        row: usize,
        cols: usize,
    ) -> Result<(), GpuError> {
        self.ready("enqueue")?;
        self.req.clear().len(layer).len(row).len(cols);
        let written = self
            .ch
            .send(Tag::Enqueue, self.req.done(), Instant::now() + REPLY);
        written.map_err(|e| self.lose_now("enqueue", e.to_string()))?;
        self.frames.enqueues += 1;
        self.frames.open = true;
        Ok(())
    }

    /// End the pass the node holds open: one end frame, sent without its
    /// reply, which [`TierNode::drain_pass`] reads. The node takes no frame
    /// for a layer the host skips, so this frame alone tells it the pass is
    /// over and it is idle, not stalled. Nothing is sent when no enqueue went
    /// out since the last end, nor to a lost node.
    pub(super) fn send_end(&mut self) -> Result<(), GpuError> {
        if !std::mem::take(&mut self.frames.open) || self.lost.is_some() {
            return Ok(());
        }
        self.ready("end")?;
        let written = self.ch.send(Tag::End, &[], Instant::now() + REPLY);
        written.map_err(|e| self.lose_now("end", e.to_string()))?;
        self.frames.end = true;
        Ok(())
    }

    /// Leave the pass the node holds open unended: this process named the
    /// node's card lost, and the node, finding its pass not ended past the go
    /// deadline and its grace, ends by its own rule.
    pub(super) fn leave_pass_open(&mut self) {
        self.frames.open = false;
    }

    /// The replies of the pass frames [`TierNode::send_enqueue`] and
    /// [`TierNode::send_end`] sent, read by `deadline` in the order the
    /// frames were sent: each an empty acknowledgement, or the server's
    /// refusal of the frame it belongs to. A channel that fails loses the
    /// node.
    pub(super) fn drain_pass(&mut self, deadline: Instant) -> Result<(), GpuError> {
        if std::mem::take(&mut self.frames.end) {
            self.read_ack("end", deadline)?;
        }
        while self.frames.enqueues > 0 {
            self.frames.enqueues -= 1;
            self.read_ack("enqueue", deadline)?;
        }
        Ok(())
    }

    /// The reply of one frame `what` sent without waiting for it, read by
    /// `deadline`: an empty acknowledgement, or the server's refusal as its
    /// error. Any other reply, or a channel that fails, loses the node.
    fn read_ack(&mut self, what: &'static str, deadline: Instant) -> Result<(), GpuError> {
        match self.ch.recv(deadline) {
            Ok(Tag::Ok) => {
                self.control_round_trips += 1;
                let end = In::new(self.ch.body()).end();
                end.map_err(|why| self.lose_now(what, why))
            }
            Ok(Tag::Err) => {
                self.control_round_trips += 1;
                match read_err(self.addr, what, self.ch.body()) {
                    Ok(e) => Err(e),
                    Err(why) => Err(self.lose_now(what, why)),
                }
            }
            Ok(t) => {
                let why = format!("a {t:?} frame where an {what}'s reply belongs");
                Err(self.lose_now(what, why))
            }
            Err(e) => Err(self.lose_now(what, e.to_string())),
        }
    }

    /// Bind the tier to the page shared through the file at `path`, `bytes`
    /// long, laid out for `tiers` tiers, as tier `index`
    /// ([`TierWork::bind_opened`]).
    pub(super) fn bind(
        &mut self,
        path: &Path,
        bytes: usize,
        tiers: usize,
        index: usize,
    ) -> Result<(), GpuError> {
        self.req.clear();
        write_file_bind(path, bytes, tiers, index, &mut self.req)?;
        self.ask(Tag::Bind, REPLY, "bind")
    }

    /// Bind a page held by this node's own end, on the wire of the stage's
    /// end `peer` (bind mode 1 RDMA, 2 TCP), as tier `index` of `tiers`:
    /// the peers of the two ends exchanged on the control channel, the
    /// node's end connected to the stage's before it answers. Returns the
    /// node's end's peer.
    pub(super) fn bind_remote(
        &mut self,
        tiers: usize,
        index: usize,
        peer: &Peer,
    ) -> Result<Peer, GpuError> {
        self.req.clear().u8(peer_mode(peer)).len(tiers).len(index);
        write_peer(peer, &mut self.req);
        self.call(Tag::Bind, REPLY, "remote bind")?;
        read_peer_reply(self.ch.body()).map_err(|e| self.lose_now("remote bind", e))
    }

    pub(super) fn local_ip(&self) -> Result<std::net::IpAddr, GpuError> {
        self.ch
            .local_ip()
            .map_err(|e| GpuError::link(WHAT, e.to_string()))
    }

    pub(super) fn release_remote(&mut self) -> Result<(), GpuError> {
        self.req.clear();
        self.ask(Tag::Release, 2 * DRAIN_DEADLINE + REPLY, "remote release")
    }

    /// Reset both drained ends, stopping both before either writer or reader resets.
    pub(super) fn reset_remote(
        &mut self,
        link: &super::link::Link,
        counters: &[usize],
    ) -> Result<(), GpuError> {
        for phase in 0..=2 {
            link.reset_remote(phase, counters)
                .map_err(|e| GpuError::protocol(WHAT, e))?;
            self.req.clear().u8(phase);
            self.ask(Tag::Reset, REPLY, "remote reset")?;
        }
        Ok(())
    }

    /// [`TierWork::launch`] on the node.
    pub(super) fn launch(&mut self, chain: Chain) -> Result<(), GpuError> {
        let w = chain_word(chain)?;
        self.req.clear().u32(w);
        self.ask(Tag::Launch, REPLY, "launch")
    }

    /// [`TierWork::capture`] on the node: the graph's node count. The reply
    /// names the node's page-word side after the count; a side other than
    /// an earlier capture's loses the node.
    pub(super) fn capture(
        &mut self,
        chain: Chain,
        cols: usize,
        list: &[(usize, usize)],
    ) -> Result<usize, GpuError> {
        let i = chain_slot(chain)?;
        self.graphs[i] = None;
        let w = chain_word(chain)?;
        self.req.clear().u32(w).len(cols).len(list.len());
        for &(layer, row) in list {
            self.req.len(layer).len(row);
        }
        self.call(Tag::Capture, REPLY, "capture")?;
        let mut r = In::new(self.ch.body());
        let reply = (|| {
            let n = r.len("nodes")?;
            let side = side_of(r.u8("page-word side")?)?;
            r.end()?;
            Ok::<_, String>((n, side))
        })();
        let (n, side) = reply.map_err(|e| self.lose_now("capture", e))?;
        if let Some(was) = self.words.filter(|&was| was != side) {
            let why = format!(
                "a page-word side of {} where an earlier capture's was {}",
                side.word(),
                was.word()
            );
            return Err(self.lose_now("capture", why));
        }
        self.words = Some(side);
        self.graphs[i] = (!list.is_empty()).then_some(n);
        Ok(n)
    }

    /// [`TierWork::enqueue`] on the node.
    pub(super) fn enqueue(
        &mut self,
        layer: usize,
        row: usize,
        cols: usize,
    ) -> Result<(), GpuError> {
        self.req.clear().len(layer).len(row).len(cols);
        self.ask(Tag::Enqueue, REPLY, "enqueue")
    }

    /// [`TierWork::drain`] on the node, under `deadline` (the server's
    /// drain deadline at most).
    pub(super) fn drain(&mut self, deadline: Duration) -> Result<(u64, u64), GpuError> {
        let deadline = deadline.min(DRAIN_DEADLINE);
        let ms = u64::try_from(deadline.as_millis()).unwrap_or(u64::MAX);
        self.req.clear().u64(ms);
        self.call(Tag::Drain, deadline + REPLY, "drain")?;
        let mut i = In::new(self.ch.body());
        let fills = (|| {
            let fills = (i.u64("host fills")?, i.u64("host fill bytes")?);
            i.end()?;
            Ok::<_, String>(fills)
        })();
        fills.map_err(|e| self.lose_now("drain", e))
    }

    /// [`TierWork::clear_fault_within`] on the node, refused while work
    /// asked of the tier is in flight.
    pub(super) fn clear_fault(&mut self) -> Result<(), GpuError> {
        self.req.clear();
        self.ask(Tag::ClearFault, 2 * DRAIN_DEADLINE + REPLY, "clear-fault")
    }

    /// [`TierWork::fault`] on the node, once its stream drained.
    pub(super) fn fault(&mut self) -> Result<Option<Fault>, GpuError> {
        self.req.clear();
        self.call(Tag::Fault, DRAIN_DEADLINE + REPLY, "fault")?;
        let mut i = In::new(self.ch.body());
        let words = (|| {
            let w = (i.u32("fault word")?, i.u32("fault sites")?);
            i.end()?;
            Ok::<_, String>(w)
        })();
        let (w, s) = words.map_err(|e| self.lose_now("fault", e))?;
        Ok(Fault::from_words(w, s))
    }

    pub(super) fn control_round_trips(&self) -> u64 {
        self.control_round_trips
    }

    /// The node's address.
    #[must_use]
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The device of the node's census the tier opened on.
    #[must_use]
    pub fn device(&self) -> &DeviceInfo {
        &self.device
    }

    pub(super) fn cols(&self) -> usize {
        self.cols
    }

    /// The layout of a page of `tiers` tiers the server's staging reads.
    pub(super) fn page_layout(&self, tiers: usize) -> Result<TierLayout, GpuError> {
        let (q3, d8) = self.planes;
        TierLayout::with_cols(self.shape, self.cols, q3, d8, tiers)
    }

    /// Nodes of the graph the server holds in slot `i`; `None` when none.
    pub(super) fn graph_nodes(&self, i: usize) -> Option<usize> {
        self.graphs.get(i).copied().flatten()
    }

    /// The node's page-word side its captures' replies named; `None`
    /// before the first.
    pub(super) fn word_side(&self) -> Option<WordSide> {
        self.words
    }

    pub(super) fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.device_bytes
    }

    /// Why the node is lost; `None` while its channel holds.
    #[must_use]
    pub fn lost(&self) -> Option<&str> {
        self.lost.as_deref()
    }

    /// End the load: a close the node answers once its stream has drained,
    /// which ends its serving of the load ([`serve`]). Nothing to a lost
    /// node. The channel ends with it, so the node is lost to every later
    /// request.
    pub(super) fn close(&mut self) -> Result<(), GpuError> {
        if self.lost.is_some() {
            return Ok(());
        }
        self.req.clear();
        let closed = self.ask(Tag::Close, DRAIN_DEADLINE + REPLY, "close");
        if self.lost.is_none() {
            self.ch.shutdown();
            self.lost = Some("its load was closed".to_owned());
        }
        closed
    }
}

impl Drop for TierNode {
    /// The load's close ([`TierNode::close`]).
    fn drop(&mut self) {
        drop(self.close());
    }
}

// ---------------------------------------------------------------- server

/// A family whose tier a tier node opens: its word ([`NodeOpen::family`]),
/// the model file's tensors as its placement classifies them, and its tier
/// computation over the stacks the node loaded for an open.
pub struct NodeFamily {
    pub word: &'static str,
    pub tensors: fn(&Split) -> Result<ModelTensors, GpuError>,
    pub experts: OpenExperts,
    pub host: fn(Arc<Split>, &NodeOpen) -> Result<super::run::HostRun, GpuError>,
}

/// A family's tier computation over the stacks a node loaded for an open
/// ([`NodeFamily::experts`]).
pub(crate) type OpenExperts =
    fn(&Gpu, &Split, &ModelTensors, &Weights, &NodeOpen) -> Result<Box<dyn TierExperts>, GpuError>;

/// How a connection ended.
enum Ended {
    /// Closed before any open: a census read.
    Census,
    /// Closed after an open: the load is done.
    Load,
}

/// How a request failed: refused before it reached the tier's stream, so
/// the session goes on, or after, so node 0 is lost to it.
enum Failed {
    Refused(GpuError),
    Fatal(GpuError),
}

impl From<GpuError> for Failed {
    fn from(e: GpuError) -> Failed {
        Failed::Refused(e)
    }
}

/// The tier a connection opened: its work, its card's name, its set, its
/// columns and, once a block open made it, its block staging.
struct Opened {
    work: TierWork,
    name: String,
    set: TierSet,
    cols: usize,
    block: Option<ServerBlock>,
}

/// A connection's block staging, made by its block open: the staging on
/// the tier's card, the page-locked rows a block's down outputs land in,
/// and what the open's reply named.
struct ServerBlock {
    staging: BlockStaging,
    host: PinnedHostBuffer<f32>,
    meta: NodeBlock,
}

impl Opened {
    /// Refuse by name, before anything is enqueued, a service of layer
    /// `layer` of row `row` over `cols` columns the tier cannot serve: a
    /// layer the set holds no expert of, a row past the page's, columns
    /// past the tier's, or a tier not yet bound.
    fn check_service(&self, layer: usize, row: usize, cols: usize) -> Result<(), GpuError> {
        let rows = self.work.page_rows().ok_or(GpuError::State {
            what: SERVER,
            missing: "the tier page (the bind request)",
        })?;
        if row >= rows || !(1..=self.cols).contains(&cols) || self.set.on_tier(layer)? == 0 {
            return Err(GpuError::shape(
                SERVER,
                format!(
                    "a service of layer {layer}, row {row} of {rows}, {cols} columns of {}: the \
                     tier holds {} experts of that layer",
                    self.cols,
                    self.set.on_tier(layer)?
                ),
            ));
        }
        Ok(())
    }
}

/// One connection's state: the tier once opened, the layers asked of it,
/// the pass node 0 holds open, its progress as last seen and when it last
/// moved, per chain the layers its captured graph serves, and the reply
/// being written.
struct Session<'a> {
    local_ip: std::net::IpAddr,
    reset_phase: u8,
    file: &'a Arc<Split>,
    families: &'a [NodeFamily],
    opened: Option<Opened>,
    asked: u32,
    /// Enqueue frames taken since node 0 last ended a pass ([`Tag::End`]):
    /// the layers of the pass it holds open, 0 between passes.
    pass: u32,
    seen: (u32, Instant),
    listed: Vec<(u32, u32)>,
    reply: Out,
    /// Given the tier's page-word operations once an open succeeds.
    on_open: &'a mut dyn FnMut(&WordOps),
}

/// Serve the expert tier of one load on `listener`, from `file`: each
/// connection gets this process's census; a connection that closes before
/// an open is a census read, and the server waits for the next. The first
/// open's load is served until node 0 closes it, and the server returns.
/// `on_open` is given the opened tier's page-word operations, whose display
/// is `word_ops=<side> driver=<cuDriverGetVersion>`, once its open succeeds.
/// Node 0's channel ended with no close, a protocol it does not speak, a
/// request that failed after it reached the tier's stream, or outstanding
/// work with no frame or progress past the go deadline and its grace
/// ([`Session::stopped_at`]) is node 0 lost: the tier's words are released,
/// its stream drained, and the loss returned by name.
pub fn serve(
    listener: &TcpListener,
    file: &Arc<Split>,
    families: &[NodeFamily],
    mut on_open: impl FnMut(&WordOps),
) -> Result<(), GpuError> {
    loop {
        let (s, peer) = listener
            .accept()
            .map_err(|e| GpuError::link(SERVER, format!("accept: {e}")))?;
        let lost = |e: std::io::Error| GpuError::link(SERVER, format!("node 0 at {peer}: {e}"));
        let local_ip = s.local_addr().map_err(lost)?.ip();
        let mut ch = Channel::new(s).map_err(lost)?;
        let mut session = Session {
            local_ip,
            reset_phase: 0,
            file,
            families,
            opened: None,
            asked: 0,
            pass: 0,
            seen: (0, Instant::now()),
            listed: Vec::new(),
            reply: Out::default(),
            on_open: &mut on_open,
        };
        NodeCensus::here()?.write(&mut session.reply);
        ch.send(Tag::Census, session.reply.done(), Instant::now() + REPLY)
            .map_err(lost)?;
        match session.run(&mut ch) {
            Ok(Ended::Census) => {}
            Ok(Ended::Load) => return Ok(()),
            Err(why) => return Err(session.end_lost(peer, &why)),
        }
    }
}

impl Session<'_> {
    /// Serve requests until the connection ends: in order, or with node 0
    /// lost (`Err`, why). While no whole request is in, it watches the
    /// tier's progress ([`Session::check_stopped`]) and the age of a request
    /// part read.
    fn run(&mut self, ch: &mut Channel) -> Result<Ended, String> {
        loop {
            let tag = match ch.poll() {
                Ok(Polled::Frame(t)) => t,
                Ok(Polled::Waiting) => {
                    self.check_stopped()?;
                    if let Some(t0) = ch.started()
                        && t0.elapsed() > REPLY
                    {
                        return Err(format!(
                            "a request unfinished {:.1?} after its first byte",
                            t0.elapsed()
                        ));
                    }
                    continue;
                }
                Ok(Polled::Closed) => {
                    return match &self.opened {
                        None => Ok(Ended::Census),
                        Some(o) => Err(format!(
                            "the channel ended with no close, {}",
                            self.state(o.work.opened_progress().unwrap_or(0))
                        )),
                    };
                }
                Err(e) => return Err(format!("the channel failed: {e}")),
            };
            if tag == Tag::Close {
                if !ch.body().is_empty() {
                    let e = GpuError::shape(
                        SERVER,
                        format!("a close with {} bytes of payload", ch.body().len()),
                    );
                    write_err(&e, &mut self.reply);
                    ch.send(Tag::Err, self.reply.done(), Instant::now() + REPLY)
                        .map_err(|e| format!("a reply: {e}"))?;
                    continue;
                }
                if let Some(o) = &self.opened {
                    o.work
                        .drain(DRAIN_DEADLINE, "tier_server: the close's drain")
                        .map_err(|e| format!("the close: {e}"))?;
                }
                ch.send(Tag::Ok, &[], Instant::now() + REPLY)
                    .map_err(|e| e.to_string())?;
                return Ok(match self.opened {
                    None => Ended::Census,
                    Some(_) => Ended::Load,
                });
            }
            let r = self.handle(tag, ch.body());
            // The reply's deadline runs from the request's end: an open or a
            // drain may take longer than a reply's whole bound.
            let deadline = Instant::now() + REPLY;
            let (sent, fatal) = match r {
                Ok(()) => (ch.send(Tag::Ok, self.reply.done(), deadline), None),
                Err(Failed::Refused(e)) => {
                    write_err(&e, &mut self.reply);
                    (ch.send(Tag::Err, self.reply.done(), deadline), None)
                }
                Err(Failed::Fatal(e)) => {
                    write_err(&e, &mut self.reply);
                    (ch.send(Tag::Err, self.reply.done(), deadline), Some(e))
                }
            };
            if let Some(e) = fatal {
                return Err(format!(
                    "a {tag:?} request failed after it reached the tier's stream: {e}"
                ));
            }
            sent.map_err(|e| format!("a reply: {e}"))?;
        }
    }

    /// Node 0 stopped ([`Session::stopped_at`]), read at the tier's progress
    /// now.
    fn check_stopped(&mut self) -> Result<(), String> {
        let Some(p) = self.opened.as_ref().and_then(|o| o.work.opened_progress()) else {
            return Ok(());
        };
        self.stopped_at(p, Instant::now())
    }

    /// Node 0 stopped, at the tier's progress `p` and the time `now`: work is
    /// outstanding — a layer asked of the tier unserved, or a pass node 0
    /// opened with an enqueue and has not ended — and neither a frame nor the
    /// tier's progress has moved past the go deadline and its grace. A layer
    /// node 0 skips sends nothing, so only its end frame tells node 1 a pass
    /// is over; a node between passes is idle for any time.
    fn stopped_at(&mut self, p: u32, now: Instant) -> Result<(), String> {
        if p != self.seen.0 || (p == self.asked && self.pass == 0) {
            self.seen = (p, now);
            return Ok(());
        }
        let waited = now.saturating_duration_since(self.seen.1);
        if waited > GO_DEADLINE + TIER_GRACE {
            let sent = if p == self.asked {
                "no layer or end of the pass"
            } else {
                "no go"
            };
            return Err(format!(
                "{}, and node 0 sent {sent} for {waited:.1?}",
                self.state(p)
            ));
        }
        Ok(())
    }

    /// The tier's progress `p` against the layers asked of it, and the pass
    /// node 0 holds open, as a loss names them.
    fn state(&self, p: u32) -> String {
        let pass = match self.pass {
            0 => String::new(),
            n => format!(", {n} of them in a pass node 0 has not ended"),
        };
        format!(
            "the tier past {p} of the {} layers asked of it{pass}",
            self.asked
        )
    }

    /// The opened tier.
    fn opened(&mut self) -> Result<&mut Opened, GpuError> {
        self.opened.as_mut().ok_or(GpuError::State {
            what: SERVER,
            missing: "an opened tier (the open request)",
        })
    }

    /// Work was asked of the tier: `n` layers more.
    fn ask(&mut self, n: u32) {
        self.asked = self.asked.wrapping_add(n);
        self.seen.1 = Instant::now();
    }

    /// One request, its reply's payload left in [`Session::reply`]. Every
    /// field is read and checked before the tier's stream is touched.
    fn handle(&mut self, tag: Tag, body: &[u8]) -> Result<(), Failed> {
        let mut i = In::new(body);
        let bad = |e: String| GpuError::shape(SERVER, format!("a {tag:?} request: {e}"));
        self.reply.clear();
        if self.reset_phase != 0 && tag != Tag::Reset {
            return Err(bad("the remote reset is incomplete".to_string()).into());
        }
        match tag {
            Tag::Open => self.handle_open(&mut i, bad)?,
            Tag::Bind => {
                let mode = i.u8("bind mode").map_err(bad)?;
                match mode {
                    0 => {
                        if !self.local_ip.is_loopback() {
                            return Err(bad(
                                "a shared-file bind requires a loopback connection".to_string()
                            )
                            .into());
                        }
                        let path = i.str("path").map_err(bad)?;
                        let (bytes, tiers, index) = (
                            i.len("bytes").map_err(bad)?,
                            i.len("tiers").map_err(bad)?,
                            i.len("index").map_err(bad)?,
                        );
                        i.end().map_err(bad)?;
                        self.opened()?
                            .work
                            .bind_opened(Path::new(&path), bytes, tiers, index)?;
                    }
                    1 | 2 => {
                        let tiers = i.len("tiers").map_err(bad)?;
                        let index = i.len("index").map_err(bad)?;
                        let remote = read_peer(mode, &mut i).map_err(bad)?;
                        i.end().map_err(bad)?;
                        let ip = self.local_ip;
                        let endpoint =
                            self.opened()?.work.bind_remote(tiers, index, &remote, ip)?;
                        self.reply.u8(peer_mode(&endpoint));
                        write_peer(&endpoint, &mut self.reply);
                    }
                    _ => return Err(bad(format!("unknown bind mode {mode}")).into()),
                }
                let p = self.opened()?.work.opened_progress().unwrap_or(0);
                self.seen = (p, Instant::now());
                self.asked = p;
            }
            Tag::Reset => {
                let phase = i.u8("reset phase").map_err(bad)?;
                i.end().map_err(bad)?;
                if phase != self.reset_phase || phase > 2 {
                    return Err(bad(format!(
                        "remote reset phase {phase}, expected {}",
                        self.reset_phase
                    ))
                    .into());
                }
                self.opened()?.work.reset_remote(phase)?;
                self.reset_phase = (phase + 1) % 3;
            }
            Tag::Launch => {
                let c = i.u32("chain").map_err(bad)?;
                i.end().map_err(bad)?;
                let chain = chain_of(c).map_err(bad)?;
                let Some(&(_, n)) = self.listed.iter().find(|(w, _)| *w == c) else {
                    return Err(bad(format!("a launch of {chain:?}, which holds no graph")).into());
                };
                self.opened()?.work.launch(chain).map_err(Failed::Fatal)?;
                self.ask(n);
            }
            Tag::Capture => self.handle_capture(&mut i, bad)?,
            Tag::Enqueue => {
                // The frame is the pass's whatever its reply: node 0 counts
                // it sent.
                self.pass = self.pass.saturating_add(1);
                let (layer, row, cols) = (
                    i.len("layer").map_err(bad)?,
                    i.len("row").map_err(bad)?,
                    i.len("columns").map_err(bad)?,
                );
                i.end().map_err(bad)?;
                let o = self.opened()?;
                o.check_service(layer, row, cols)?;
                o.work.enqueue(layer, row, cols).map_err(Failed::Fatal)?;
                self.ask(1);
            }
            Tag::End => {
                i.end().map_err(bad)?;
                if self.pass == 0 {
                    return Err(bad(
                        "no pass is open: node 0 sent no enqueue since its last end".to_string()
                    )
                    .into());
                }
                self.pass = 0;
                self.seen.1 = Instant::now();
            }
            Tag::Release => {
                i.end().map_err(bad)?;
                let work = &self.opened()?.work;
                if !work.release_opened().map_err(Failed::Fatal)? {
                    return Err(Failed::Fatal(GpuError::state(SERVER, "a bound link")));
                }
                work.drain(DRAIN_DEADLINE, "tier_server: release")
                    .map_err(Failed::Fatal)?;
            }
            Tag::Drain => {
                let ms = i.u64("deadline").map_err(bad)?;
                i.end().map_err(bad)?;
                let deadline = Duration::from_millis(ms).min(DRAIN_DEADLINE);
                let work = &self.opened()?.work;
                work.drain(deadline, "tier_server: the tier's stream")?;
                let (fills, bytes) = work.host_fills()?;
                self.reply.u64(fills).u64(bytes);
            }
            Tag::ClearFault => {
                i.end().map_err(bad)?;
                let asked = self.asked;
                let o = self.opened()?;
                let p = o.work.opened_progress().unwrap_or(0);
                if p != asked {
                    return Err(bad(format!(
                        "the tier is past {p} of the {asked} layers asked of it: work is in \
                         flight"
                    ))
                    .into());
                }
                o.work
                    .clear_fault_within(DRAIN_DEADLINE, "tier_server: the clear-fault")?;
            }
            Tag::Fault => {
                i.end().map_err(bad)?;
                let o = self.opened()?;
                o.work
                    .drain(DRAIN_DEADLINE, "tier_server: the fault's drain")?;
                let f = o.work.fault()?;
                let (w, s) = f.map_or((FAULT_NONE, 0), |f| (f.word(), f.sites));
                self.reply.u32(w).u32(s);
            }
            Tag::BlockOpen => {
                let cap = i.len("block columns").map_err(bad)?;
                i.end().map_err(bad)?;
                self.handle_block_open(cap)?;
            }
            Tag::Block => self.handle_block(&mut i, bad)?,
            Tag::Census | Tag::Close | Tag::Ok | Tag::Err => {
                return Err(bad("not a request".to_string()).into());
            }
        }
        Ok(())
    }

    /// A block open: the block staging for blocks of up to `cap` columns
    /// made on the tier's card, its bytes the reply
    /// ([`block::write_open_reply`]); a tier whose work runs no block from
    /// the q8_1 planes ([`TierWork::block_kind`]) answers that it runs none.
    /// A second block open, columns past the staging's or a frame's
    /// ([`block::check_frames`]), and a staging the card cannot hold are
    /// refused by name.
    fn handle_block_open(&mut self, cap: usize) -> Result<(), Failed> {
        let o = self.opened.as_mut().ok_or(GpuError::State {
            what: SERVER,
            missing: "an opened tier (the open request)",
        })?;
        if o.block.is_some() {
            return Err(GpuError::shape(SERVER, "a second block open on one connection").into());
        }
        let Some((scratch, rows)) = o.work.block_kind() else {
            block::write_open_reply(&mut self.reply, None);
            return Ok(());
        };
        let TierShape {
            hidden: n,
            n_used: s,
            ..
        } = o.work.shape();
        let (q3, d8) = Q8Act::q3_d8_words(n)?;
        let refused =
            |why: String| GpuError::shape(SERVER, format!("a block open on {}: {why}", o.name));
        if !(1..=TIER_ACT_MAX_COLS).contains(&cap) {
            return Err(refused(format!("{cap} columns, 1 to {TIER_ACT_MAX_COLS}")).into());
        }
        block::check_frames(cap, s, n, (q3 * 8, d8 * 4)).map_err(refused)?;
        let g = o.work.gpu();
        g.context().bind_to_thread().map_err(GpuError::from)?;
        let staging = BlockStaging::new(g.stream(), cap, n, s)
            .map_err(|e| refused(format!("the staging of {cap} columns: {e}")))?;
        let host = PinnedHostBuffer::<f32>::zeroed(g.context(), cap * s * n)
            .map_err(|e| refused(format!("the page-locked rows of {cap} columns: {e}")))?;
        let meta = NodeBlock {
            staging: staging.device_bytes(),
            scratch,
            rows,
            cap,
        };
        o.block = Some(ServerBlock {
            staging,
            host,
            meta,
        });
        block::write_open_reply(&mut self.reply, Some(meta));
        Ok(())
    }

    /// A block request ([`BlockRequest`]): checked against the block
    /// staging and the tier's set before the tier's stream is touched, its
    /// places and planes landed, the tier's experts run over the block
    /// ([`run_block`]), and the rows of the slots it places on the tier, in
    /// slot order, the reply ([`block::write_reply_head`]).
    fn handle_block(
        &mut self,
        i: &mut In<'_>,
        bad: impl Fn(String) -> GpuError,
    ) -> Result<(), Failed> {
        let r = BlockRequest::read(i).map_err(&bad)?;
        let o = self.opened.as_mut().ok_or(GpuError::State {
            what: SERVER,
            missing: "an opened tier (the open request)",
        })?;
        let b = o.block.as_mut().ok_or(GpuError::State {
            what: SERVER,
            missing: "the block staging (the block open request)",
        })?;
        let TierShape {
            hidden: n,
            n_used: s,
            ..
        } = o.work.shape();
        let (q3, d8) = Q8Act::q3_d8_words(n)?;
        let on_tier = o.set.on_tier(r.layer)?;
        let cols = r.cols;
        if !(1..=b.meta.cap).contains(&cols)
            || r.places.len() != cols * s
            || r.q3.len() != cols * q3 * 8
            || r.d8.len() != cols * d8 * 4
            || on_tier == 0
        {
            return Err(bad(format!(
                "layer {}'s block of {cols} columns of up to {}: {} places, {} q3 and {} scale \
                 bytes, the tier holding {on_tier} of the layer's experts",
                r.layer,
                b.meta.cap,
                r.places.len(),
                r.q3.len(),
                r.d8.len()
            ))
            .into());
        }
        if let Some(p) = r
            .places
            .iter()
            .find(|&&p| p != HOST && p as usize >= on_tier)
        {
            return Err(bad(format!(
                "layer {}'s tier place {p}, past the tier's {on_tier} experts of the layer",
                r.layer
            ))
            .into());
        }
        let sent = r.places.iter().filter(|&&p| p != HOST).count();
        let copied = match b.meta.rows {
            BlockRows::Staged => cols * s,
            BlockRows::Packed => sent,
        };
        let planes = (r.q3, r.d8);
        run_block(&mut o.work, b, (r.layer, cols), &r.places, planes, copied)
            .map_err(Failed::Fatal)?;
        block::write_reply_head(&mut self.reply, sent, n);
        match b.meta.rows {
            BlockRows::Packed => {
                self.reply.raw(bytes_of(&b.host[..sent * n]));
            }
            BlockRows::Staged => {
                for (j, _) in r.places.iter().enumerate().filter(|&(_, &p)| p != HOST) {
                    self.reply.raw(bytes_of(&b.host[j * n..(j + 1) * n]));
                }
            }
        }
        Ok(())
    }

    /// An open request: the tier it names opened ([`Session::open`]), its
    /// planes and bytes the reply.
    fn handle_open(
        &mut self,
        i: &mut In<'_>,
        bad: impl Fn(String) -> GpuError,
    ) -> Result<(), Failed> {
        let open = NodeOpen::read(i).map_err(&bad)?;
        i.end().map_err(&bad)?;
        let work = self.open(&open)?;
        (self.on_open)(work.word_ops());
        let (q3, d8) = work.planes();
        self.reply
            .len(q3)
            .len(d8)
            .len(work.weights().resident_bytes())
            .len(work.device_bytes());
        self.opened = Some(Opened {
            work,
            name: open.name,
            set: open.set,
            cols: open.cols,
            block: None,
        });
        Ok(())
    }

    /// A capture request: each service it lists checked, the chain's graph
    /// captured and listed for its launches, its node count the reply.
    fn handle_capture(
        &mut self,
        i: &mut In<'_>,
        bad: impl Fn(String) -> GpuError,
    ) -> Result<(), Failed> {
        let c = i.u32("chain").map_err(&bad)?;
        let cols = i.len("columns").map_err(&bad)?;
        let n = i.len("services").map_err(&bad)?;
        let mut list = Vec::new();
        for _ in 0..n {
            list.push((i.len("layer").map_err(&bad)?, i.len("row").map_err(&bad)?));
        }
        i.end().map_err(&bad)?;
        let chain = chain_of(c).map_err(&bad)?;
        if cols != chain.cols() {
            return Err(bad(format!("a {chain:?} capture of {cols} columns")).into());
        }
        let len = u32::try_from(list.len()).map_err(|_| bad("a list past u32".into()))?;
        let o = self.opened()?;
        for &(layer, row) in &list {
            o.check_service(layer, row, cols)?;
        }
        let nodes = o.work.capture(chain, cols, &list).map_err(Failed::Fatal)?;
        let side = o.work.word_side();
        self.listed.retain(|(w, _)| *w != c);
        if len > 0 {
            self.listed.push((c, len));
        }
        self.reply.len(nodes).u8(side_byte(side));
        Ok(())
    }

    /// The tier `open` names: its card by name and device, checked against
    /// this process's census ([`Gpu::open_card`]), its segments loaded from
    /// this process's file, which must hold them as node 0's plan does
    /// ([`NodeOpen::rows_on`]) and with the bytes node 0's file holds (each
    /// segment's sha256, [`NodeRow::sha256`]), and its family's computation
    /// over them. A second open on one connection is refused by name.
    fn open(&self, open: &NodeOpen) -> Result<TierWork, GpuError> {
        if self.opened.is_some() {
            return Err(GpuError::shape(SERVER, "a second open on one connection"));
        }
        let family = self
            .families
            .iter()
            .find(|f| f.word == open.family)
            .ok_or_else(|| {
                GpuError::shape(
                    SERVER,
                    format!(
                        "a tier of the family {:?}; this server opens {:?}",
                        open.family,
                        self.families.iter().map(|f| f.word).collect::<Vec<_>>()
                    ),
                )
            })?;
        let tensors = (family.tensors)(self.file)?;
        let rows = open.rows_on(&tensors)?;
        for (r, row) in open.rows.iter().zip(&rows) {
            let (t, seg) = (&tensors.tensors[row.tensor], &row.segments[0]);
            let sha256 = segment_sha256(self.file, tensors.experts, t, seg)?;
            if sha256 != r.sha256 {
                return Err(GpuError::shape(
                    SERVER,
                    format!(
                        "{}: the experts {:?} hold other bytes in this file (sha256 {sha256}) than \
                         in node 0's ({}): the two processes read other files",
                        r.name, r.experts, r.sha256
                    ),
                ));
            }
        }
        let gpu = Gpu::open_card(&open.name, open.device)?;
        let (w, experts) = if open.nvme.is_some() {
            open.load_host(&gpu, self.file, &tensors, family)?
        } else {
            let w = Weights::load_rows(gpu.stream(), self.file, &tensors, &rows, open.card)?;
            let experts = (family.experts)(&gpu, self.file, &tensors, &w, open)?;
            (w, experts)
        };
        TierWork::open(gpu, w, experts, open.shape, open.cols)
    }

    /// Node 0 is lost (`why`): release the tier's words, so every wait its
    /// stream holds passes and node 0's next settle refuses the step, and
    /// drain the stream; the loss is returned by name.
    /// A stream that did not drain may still touch the work's buffers and
    /// the page, so the work is then never freed: the process ends with it.
    fn end_lost(&mut self, peer: SocketAddr, why: &str) -> GpuError {
        let Some(o) = self.opened.take() else {
            return GpuError::link(SERVER, format!("node 0 at {peer} is lost: {why}"));
        };
        let released = match o.work.release_opened() {
            Ok(true) => "released its words",
            Ok(false) => "held no page",
            Err(e) => {
                let error = GpuError::link(
                    SERVER,
                    format!("node 0 at {peer} is lost: {why}; release failed: {e}"),
                );
                std::mem::forget(o);
                return error;
            }
        };
        let drained = o
            .work
            .drain(DRAIN_DEADLINE, "tier_server: the lost load's drain");
        let state = match &drained {
            Ok(()) => format!("the tier {released} and its stream drained"),
            Err(e) => format!("the tier {released} and its stream did not drain: {e}"),
        };
        let e = GpuError::link(
            SERVER,
            format!(
                "node 0 at {peer} is lost: {why}; the tier {}: {state}",
                o.name
            ),
        );
        if drained.is_err() {
            std::mem::forget(o);
        }
        e
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::net::SocketAddr;
    use std::time::{Duration, Instant};

    use super::wire::{In, Out, Tag};
    use super::{
        Failed, NodeCensus, NodeOpen, NodeRow, OPEN_BASE, OPEN_RATE, Session, chain_of, chain_word,
        open_bound, read_err, side_byte, side_of, write_err,
    };
    use crate::GpuError;
    use crate::fault::{Fault, FaultSite};
    use crate::host::link::{Peer, TcpPeer};
    use crate::host::step::Chain;
    use crate::host::tier::{ReplyRows, TierAct, TierSet, TierShape};
    use crate::words::WordSide;
    use gguf::write::{Layout, Writer};
    use gguf::{GgmlType, Split};
    use model::placement::devices::{DeviceId, DeviceInfo};
    use model::placement::workstation::HostRead;
    use model::placement::{Format, ModelTensor, ModelTensors, Role};

    pub(in crate::host) fn empty_file(tag: &str) -> (std::path::PathBuf, std::sync::Arc<Split>) {
        let path = std::env::temp_dir().join(format!("bloomery-{tag}-{}.gguf", std::process::id()));
        let file = std::fs::File::create_new(&path).expect("a new model file");
        Writer::new(file, Layout::new(&[], vec![]).expect("an empty layout"))
            .expect("a writer")
            .finish()
            .expect("the model header");
        let file = Split::open(&path).expect("the model file");
        (path, std::sync::Arc::new(file))
    }

    #[test]
    fn a_node_refuses_other_expert_metadata_with_identical_tensor_bytes() {
        use gguf::Value;
        use gguf::write::TensorDecl;
        use model::arch::{Arch, names};
        use model::placement::{Device, Segment};

        let tensor = ModelTensor {
            name: names::routed_stacks(0)[1].clone(),
            shard: 0,
            layer: Some(0),
            role: Role::RoutedExperts,
            ty: GgmlType::F32,
            dims: vec![2, 2, 2],
            file_bytes: 32,
            gathered_rows: None,
        };
        let tensors = ModelTensors {
            tensors: vec![tensor.clone()],
            layers: 1,
            experts: 2,
            experts_used: 1,
            bf16_words: |_| false,
        };
        let segment = Segment {
            device: Device::Card(1),
            format: Format::Card(model::placement::CardFormat::F32),
            experts: Some(model::placement::ExpertList::range(1..2).unwrap()),
            resident_bytes: 16,
        };
        let dir =
            std::env::temp_dir().join(format!("bloomery-node-metadata-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let files: Vec<_> = [10.0, 1.0]
            .into_iter()
            .map(|clamp| {
                let path = dir.join(format!("clamp-{clamp}.gguf"));
                let layout = Layout::new(
                    &[(
                        format!("{}.swiglu_clamp_exp", Arch::Glm5next.name()),
                        Value::F32(clamp),
                    )],
                    vec![TensorDecl {
                        name: tensor.name.clone(),
                        dims: tensor.dims.clone(),
                        type_id: 0,
                        nbytes: 32,
                    }],
                )
                .unwrap();
                let mut writer =
                    Writer::new(std::fs::File::create_new(&path).unwrap(), layout).unwrap();
                writer.tensor(&tensor.name, &[7; 32]).unwrap();
                writer.finish().unwrap();
                std::sync::Arc::new(Split::open(&path).unwrap())
            })
            .collect();
        let bytes = |file: &Split| {
            crate::weights::SegmentFile::of(file, 2, &tensor, &segment)
                .unwrap()
                .bytes
                .into_iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>()
        };
        assert_eq!(bytes(&files[0]), bytes(&files[1]));
        let open = NodeOpen {
            nvme: None,
            family: "metadata-test".into(),
            name: "unused".into(),
            device: None,
            card: 1,
            rows: vec![NodeRow {
                name: tensor.name.clone(),
                shard: 0,
                ty: tensor.ty.to_string(),
                dims: tensor.dims.clone(),
                file_bytes: 32,
                format: segment.format.to_string(),
                experts: Some(vec![1]),
                resident_bytes: 16,
                sha256: super::segment_sha256(&files[0], 2, &tensor, &segment).unwrap(),
            }],
            set: TierSet::new(0..1, 2, vec![vec![1]]).unwrap(),
            shape: TierShape {
                hidden: 2,
                n_used: 1,
                rows: 1,
                act: TierAct::F32,
                reply: ReplyRows::Whole,
            },
            cols: 1,
            layers: 1,
            block: (0, 0),
        };
        let family = super::NodeFamily {
            word: "metadata-test",
            tensors: |_| {
                Ok(ModelTensors {
                    tensors: vec![ModelTensor {
                        name: names::routed_stacks(0)[1].clone(),
                        shard: 0,
                        layer: Some(0),
                        role: Role::RoutedExperts,
                        ty: GgmlType::F32,
                        dims: vec![2, 2, 2],
                        file_bytes: 32,
                        gathered_rows: None,
                    }],
                    layers: 1,
                    experts: 2,
                    experts_used: 1,
                    bf16_words: |_| false,
                })
            },
            experts: |_, _, _, _, _| panic!("metadata must be checked before loading experts"),
            host: |_, _| panic!("metadata must be checked before opening host experts"),
        };
        assert!(open.rows_on(&tensors).is_ok());
        let session = Session {
            local_ip: std::net::Ipv4Addr::LOCALHOST.into(),
            reset_phase: 0,
            file: &files[1],
            families: &[family],
            opened: None,
            asked: 0,
            pass: 0,
            seen: (0, Instant::now()),
            listed: Vec::new(),
            reply: Out::default(),
            on_open: &mut |_| {},
        };
        let error = session
            .open(&open)
            .err()
            .expect("different expert metadata must be refused");
        assert!(
            error
                .to_string()
                .contains("the two processes read other files"),
            "{error}"
        );
        drop(session);
        drop(files);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_an_unsettled_lost_node_keeps_its_work_without_unwinding() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        let (path, file) = empty_file("unsettled-node");
        let dropped = Arc::new(AtomicBool::new(false));
        let work = super::super::tier::tests::remote_work(&file, Arc::clone(&dropped), true);
        let mut session = Session {
            local_ip: std::net::Ipv4Addr::LOCALHOST.into(),
            reset_phase: 0,
            file: &file,
            families: &[],
            opened: Some(super::Opened {
                work,
                name: "test tier".into(),
                set: TierSet::new(0..1, 1, vec![vec![0]]).expect("a set"),
                cols: 1,
                block: None,
            }),
            asked: 0,
            pass: 0,
            seen: (0, Instant::now()),
            listed: Vec::new(),
            reply: Out::default(),
            on_open: &mut |_| {},
        };
        assert!(matches!(
            session.handle(Tag::Release, &[]),
            Err(Failed::Fatal(GpuError::Protocol { .. }))
        ));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            session.end_lost("127.0.0.1:7340".parse().expect("a peer"), "channel closed")
        }));
        assert!(result.is_ok(), "a failed release must return, not unwind");
        let error = result.expect("no unwind");
        assert!(error.to_string().contains("release failed"), "{error}");
        assert!(
            !dropped.load(Ordering::Acquire),
            "live work must be retained"
        );
        assert!(session.opened.is_none());
        drop(session);
        drop(file);
        std::fs::remove_file(path).expect("remove the fixture");
    }

    /// A node refuses, by name, a service of a layer its set holds no expert
    /// of: the frames a step sends name layers the routing picked, and a
    /// layer the node does not hold is not one of them
    /// ([`TierCard::send_layer`]'s skip keeps the node's stream in step).
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_node_refuses_a_layer_it_holds_none_of() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        let (path, file) = empty_file("node-refuse-layer");
        let dropped = Arc::new(AtomicBool::new(false));
        let work = super::super::tier::tests::remote_work(&file, Arc::clone(&dropped), true);
        let set = TierSet::new(0..2, 4, vec![vec![0], vec![]]).expect("a set with an empty layer");
        let mut session = Session {
            local_ip: std::net::Ipv4Addr::LOCALHOST.into(),
            reset_phase: 0,
            file: &file,
            families: &[],
            opened: Some(super::Opened {
                work,
                name: "test tier".into(),
                set,
                cols: 1,
                block: None,
            }),
            asked: 0,
            pass: 0,
            seen: (0, Instant::now()),
            listed: Vec::new(),
            reply: Out::default(),
            on_open: &mut |_| {},
        };
        let mut req = Out::default();
        req.len(1).len(0).len(1);
        match session.handle(Tag::Enqueue, req.done()) {
            Err(Failed::Refused(e)) => assert!(
                e.to_string()
                    .contains("the tier holds 0 experts of that layer"),
                "{e}"
            ),
            Err(_) => panic!("an enqueue of an empty layer must be refused, not failed"),
            Ok(()) => panic!("an enqueue of an empty layer must be refused"),
        }
        // The same check passes a layer the set holds an expert of. It is
        // read without an enqueue: a served layer's stream would park on a go
        // no test raises, and every later test on the card would wait behind it.
        let opened = session.opened.as_ref().expect("the opened tier");
        assert!(
            opened.check_service(0, 0, 1).is_ok(),
            "a layer the set holds an expert of passes the service check"
        );
        let _ = session.end_lost(
            "127.0.0.1:7340".parse().expect("a peer"),
            "the test ends its load",
        );
        assert!(session.opened.is_none());
        drop(session);
        drop(file);
        std::fs::remove_file(path).expect("remove the fixture");
    }

    fn gb10(uuid: u8) -> DeviceInfo {
        DeviceInfo {
            ordinal: 0,
            name: "NVIDIA GB10".into(),
            total_bytes: 128 << 30,
            free_bytes: 100 << 30,
            integrated: true,
            uuid: [uuid; 16],
            pci_bus: "000F:01:00.0".into(),
            held_by: None,
            compute_capability: None,
            granule_bytes: std::num::NonZeroU64::new(2 << 20),
        }
    }

    /// Over loopback, with no card: a node's load is closed once. The node
    /// gets one close with no payload, and neither a second close nor the
    /// node's Drop sends anything more; the closed node is lost, so a later
    /// request is refused.
    #[test]
    fn a_node_gets_one_close() {
        use super::wire::Channel;
        use std::net::TcpListener;

        let l = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let addr = l.local_addr().expect("its address");
        let peer = std::thread::spawn(move || {
            let mut ch = Channel::new(l.accept().expect("an accept").0).expect("a channel");
            let mut frames = Vec::new();
            while let Ok(tag) = ch.recv(Instant::now() + Duration::from_secs(10)) {
                frames.push((tag, ch.body().len()));
                ch.send(Tag::Ok, &[], Instant::now() + Duration::from_secs(10))
                    .expect("a reply");
            }
            frames
        });
        let mut node = loopback_node(addr);
        // A payload left by an earlier request, which the close does not send.
        node.req.u8(1);
        node.close().expect("the close");
        assert!(node.lost().is_some(), "a closed node is lost");
        node.close().expect("a second close sends nothing");
        assert!(node.release_remote().is_err(), "a request after the close");
        drop(node);
        assert_eq!(
            peer.join().expect("the peer"),
            [(Tag::Close, 0)],
            "the frames the node got"
        );
    }

    /// A node tier over a loopback channel to `addr`, opened on no card: one
    /// row of one column.
    fn loopback_node(addr: SocketAddr) -> super::TierNode {
        use super::wire::Channel;
        super::TierNode {
            addr,
            ch: Channel::new(std::net::TcpStream::connect(addr).expect("a connect"))
                .expect("a channel"),
            req: Out::default(),
            device: gb10(1),
            shape: TierShape {
                hidden: 2,
                n_used: 1,
                rows: 1,
                act: TierAct::F32,
                reply: ReplyRows::Whole,
            },
            cols: 1,
            planes: (0, 0),
            resident_bytes: 0,
            device_bytes: 0,
            graphs: [None; super::TIER_CHAINS],
            words: None,
            block: None,
            pending: None,
            frames: super::PassFrames::default(),
            control_round_trips: 0,
            lost: None,
        }
    }

    /// The frames a loopback peer got, each tag with its payload, once its
    /// channel ends.
    type PeerFrames = std::thread::JoinHandle<Vec<(Tag, Vec<u8>)>>;

    /// A loopback peer that answers every frame with an empty
    /// acknowledgement until the channel ends, and returns the frames it got.
    fn loopback_peer() -> (SocketAddr, PeerFrames) {
        use super::wire::Channel;
        let l = std::net::TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let addr = l.local_addr().expect("its address");
        let peer = std::thread::spawn(move || {
            let mut ch = Channel::new(l.accept().expect("an accept").0).expect("a channel");
            let mut frames = Vec::new();
            while let Ok(tag) = ch.recv(Instant::now() + Duration::from_secs(10)) {
                frames.push((tag, ch.body().to_vec()));
                ch.send(Tag::Ok, &[], Instant::now() + Duration::from_secs(10))
                    .expect("a reply");
            }
            frames
        });
        (addr, peer)
    }

    /// Over loopback, with no card: how a node tier's pass ends. A pass that
    /// sent the node a layer ends with one end frame and a second end sends
    /// nothing; a pass whose layers were all skipped sent nothing, so it ends
    /// with nothing; a pass whose card the host named lost is left open, so
    /// the node ends by its own rule.
    #[test]
    fn a_pass_that_sent_the_node_a_layer_ends_with_one_end_frame() {
        use super::{Chain, TierCard, TierSet};

        let (addr, peer) = loopback_peer();
        let mut card = TierCard::on_node(
            loopback_node(addr),
            "test tier".into(),
            TierSet::new(0..3, 4, vec![vec![0], vec![1], vec![2]]).expect("a set"),
        );
        assert!(card.send_layer(Chain::Step, 0, 0, 1).expect("layer 0"));
        card.end_pass().expect("the first pass's end");
        card.end_pass().expect("a second end sends nothing");
        assert!(!card.send_layer(Chain::Step, 1, 0, 0).expect("layer 1"));
        card.end_pass().expect("a skipped pass sends no end");
        assert!(card.send_layer(Chain::Step, 2, 0, 1).expect("layer 2"));
        card.leave_pass_open();
        card.end_pass().expect("a pass left open sends no end");
        drop(card);
        let enqueue = |layer| {
            let mut o = Out::default();
            o.len(layer).len(0).len(1);
            (Tag::Enqueue, o.done().to_vec())
        };
        assert_eq!(
            peer.join().expect("the peer"),
            vec![
                enqueue(0),
                (Tag::End, Vec::new()),
                enqueue(2),
                (Tag::Close, Vec::new()),
            ],
            "the frames the node got: one end, after the pass that sent layer 0"
        );
    }

    /// Node 1's loss rule, with no card: a node between passes, every layer
    /// asked of it served, is idle for any time; a pass node 0 opened and has
    /// not ended loses node 0 once neither a frame nor the tier's progress
    /// has moved past the go deadline and its grace; an end closes the pass,
    /// and an end with no pass open, or with a payload, is refused by name;
    /// a layer asked and unserved is the go rule's, as before.
    #[test]
    fn a_pass_node_0_does_not_end_loses_node_0_past_the_deadline() {
        use super::super::step::GO_DEADLINE;
        use super::super::tier::TIER_GRACE;

        let (path, file) = empty_file("node-pass");
        let t0 = Instant::now();
        let past = t0 + GO_DEADLINE + TIER_GRACE + Duration::from_millis(1);
        let hour = Duration::from_secs(3600);
        let mut session = Session {
            local_ip: std::net::Ipv4Addr::LOCALHOST.into(),
            reset_phase: 0,
            file: &file,
            families: &[],
            opened: None,
            asked: 4,
            pass: 0,
            seen: (4, t0),
            listed: Vec::new(),
            reply: Out::default(),
            on_open: &mut |_| {},
        };
        assert!(
            session.stopped_at(4, past + hour).is_ok(),
            "between passes, every layer served, a node is idle for any time"
        );
        session.pass = 1;
        session.seen = (4, t0);
        assert!(
            session.stopped_at(4, t0 + GO_DEADLINE).is_ok(),
            "an open pass within the deadline and its grace is not a loss"
        );
        let lost = session
            .stopped_at(4, past)
            .expect_err("a pass not ended past the deadline and its grace loses node 0");
        assert!(
            lost.contains(
                "the tier past 4 of the 4 layers asked of it, 1 of them in a pass node 0 has not \
                 ended, and node 0 sent no layer or end of the pass"
            ),
            "{lost}"
        );
        assert!(matches!(session.handle(Tag::End, &[]), Ok(())));
        assert_eq!(session.pass, 0, "the end closes the pass");
        assert!(
            session.stopped_at(4, past + hour).is_ok(),
            "an ended pass leaves the node idle"
        );
        match session.handle(Tag::End, &[]) {
            Err(Failed::Refused(e)) => assert!(e.to_string().contains("no pass is open"), "{e}"),
            _ => panic!("an end with no pass open must be refused by name"),
        }
        session.pass = 1;
        assert!(
            matches!(session.handle(Tag::End, &[0]), Err(Failed::Refused(_))),
            "an end with a payload must be refused"
        );
        assert_eq!(session.pass, 1, "a refused end leaves the pass open");
        session.pass = 0;
        session.seen = (3, t0);
        let lost = session
            .stopped_at(3, past)
            .expect_err("a layer asked and unserved past the deadline loses node 0");
        assert!(lost.contains("node 0 sent no go"), "{lost}");
        drop(session);
        drop(file);
        std::fs::remove_file(path).expect("remove the fixture");
    }

    /// Over loopback, with no card: the frames a node tier's layers produce.
    /// A layer whose picks name a slot of the node's is sent its frame, which
    /// names its layer; a layer whose picks name none is sent nothing — no
    /// frame, so no round trip — and the layer after it still names its own.
    #[test]
    fn a_node_free_layer_sends_no_frame() {
        use super::{Chain, TierCard, TierSet};

        let (addr, peer) = loopback_peer();
        let mut card = TierCard::on_node(
            loopback_node(addr),
            "test tier".into(),
            TierSet::new(0..3, 4, vec![vec![0], vec![], vec![1]]).expect("a set"),
        );
        // The hook's calls for one step: layer 5 holds two of the node's
        // slots, layer 6 none (skipped), layer 7 one.
        assert!(
            card.send_layer(Chain::Step, 5, 0, 2)
                .expect("layer 5's decision"),
            "layer 5's picks name slots: sent"
        );
        assert!(
            !card
                .send_layer(Chain::Step, 6, 0, 0)
                .expect("layer 6's decision"),
            "layer 6 names no slot of the node's: skipped"
        );
        assert!(
            card.send_layer(Chain::Step, 7, 0, 1)
                .expect("layer 7's decision"),
            "layer 7's pick names a slot: sent"
        );
        let mut want = Vec::new();
        for layer in [5, 7] {
            let mut o = Out::default();
            o.len(layer).len(0).len(1);
            want.push((Tag::Enqueue, o.done().to_vec()));
        }
        drop(card);
        let got = peer.join().expect("the peer");
        assert_eq!(
            got,
            vec![want[0].clone(), want[1].clone(), (Tag::Close, Vec::new()),],
            "the frames the node got: layer 6 (no node pick) must send none"
        );
    }

    /// A census and an open read back as written; every chain's word names
    /// it back; an error reply reads back as the server's fault (its words
    /// kept), protocol error or refusal, with its text. A census of another
    /// version, of no device, of a device with more free bytes than total or
    /// of one uuid twice is refused.
    #[test]
    fn file_bind_payload_carries_its_mode_and_layout() {
        let path = std::path::Path::new("/dev/shm/bloomery-tier-41");
        let mut out = Out::default();
        super::write_file_bind(path, 4096, 3, 2, &mut out).expect("a file bind");
        let mut input = In::new(out.done());
        assert_eq!(input.u8("mode").expect("mode"), 0);
        assert_eq!(
            input.str("path").expect("path"),
            path.to_str().expect("ASCII")
        );
        assert_eq!(input.len("bytes").expect("bytes"), 4096);
        assert_eq!(input.len("tiers").expect("tiers"), 3);
        assert_eq!(input.len("index").expect("index"), 2);
        input.end().expect("the exact payload");
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            let path = std::path::Path::new(std::ffi::OsStr::from_bytes(b"/dev/shm/\xff"));
            out.clear();
            assert!(super::write_file_bind(path, 4096, 1, 0, &mut out).is_err());
            assert!(out.done().is_empty());
        }
    }

    #[test]
    fn rdma_endpoints_require_complete_valid_fields() {
        let endpoint = bloomery_verbs::Endpoint {
            qpn: 0xff_ffff,
            psn: 0xff_ffff,
            gid: bloomery_verbs::Gid { raw: [0x7b; 16] },
            addr: 0x1234_5678_9abc_def0,
            rkey: 0xfedc_ba98,
        };
        let mut out = Out::default();
        super::write_endpoint(&endpoint, &mut out);
        let mut input = In::new(out.done());
        assert_eq!(
            super::read_endpoint(&mut input).expect("an endpoint"),
            endpoint
        );
        input.end().expect("the exact payload");
        for n in 0..out.done().len() {
            assert!(
                super::read_endpoint(&mut In::new(&out.done()[..n])).is_err(),
                "{n}"
            );
        }
        for invalid in [
            bloomery_verbs::Endpoint { qpn: 0, ..endpoint },
            bloomery_verbs::Endpoint {
                qpn: 1 << 24,
                ..endpoint
            },
            bloomery_verbs::Endpoint {
                psn: 1 << 24,
                ..endpoint
            },
            bloomery_verbs::Endpoint {
                addr: 0,
                ..endpoint
            },
        ] {
            out.clear();
            super::write_endpoint(&invalid, &mut out);
            assert!(super::read_endpoint(&mut In::new(out.done())).is_err());
        }
    }

    /// A remote bind carries each wire's peer under its own mode: a TCP
    /// peer's IPv4 address comes back as that IPv4 address (the stage's end
    /// compares it with the address the dial came from), an IPv6 one as
    /// itself; a reply reads back as the peer of its mode, every truncation
    /// and an unspecified address or a port of 0 or past `u16` refused.
    #[test]
    fn remote_binds_carry_each_wires_peer_under_its_mode() {
        use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
        let rdma = Peer::Rdma(bloomery_verbs::Endpoint {
            qpn: 7,
            psn: 0x100,
            gid: bloomery_verbs::Gid { raw: [1; 16] },
            addr: 0x1000,
            rkey: 9,
        });
        let v4 = Peer::Tcp(TcpPeer {
            addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(198, 18, 0, 2)), 40_001),
            nonce: 0xfeed_beef_0123_4567,
        });
        let v6 = Peer::Tcp(TcpPeer {
            addr: SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 9),
            nonce: 1,
        });
        let mut out = Out::default();
        for peer in [rdma, v4, v6] {
            out.clear().u8(super::peer_mode(&peer));
            super::write_peer(&peer, &mut out);
            assert_eq!(super::read_peer_reply(out.done()), Ok(peer), "{peer:?}");
            for n in 0..out.done().len() {
                assert!(super::read_peer_reply(&out.done()[..n]).is_err(), "{n}");
            }
        }
        out.clear().u8(super::peer_mode(&v4));
        super::write_peer(&v4, &mut out);
        assert_eq!(out.done()[0], 2, "TCP is bind mode 2");
        assert_eq!(
            &out.done()[1 + 8..1 + 8 + 16],
            &Ipv4Addr::new(198, 18, 0, 2).to_ipv6_mapped().octets(),
            "16 bytes after their length, IPv6-mapped"
        );
        for (ip, port) in [
            (Ipv6Addr::UNSPECIFIED, 1),
            (Ipv4Addr::LOCALHOST.to_ipv6_mapped(), 0),
            (Ipv4Addr::LOCALHOST.to_ipv6_mapped(), 1 << 16),
        ] {
            out.clear().u8(2).bytes(&ip.octets()).u32(port).u64(0);
            let read = super::read_peer_reply(out.done());
            assert!(
                read.as_ref()
                    .is_err_and(|e| e.contains("an invalid TCP endpoint")),
                "{ip} {port}: {read:?}"
            );
        }
        out.clear().u8(0);
        assert!(
            super::read_peer_reply(out.done())
                .is_err_and(|e| e.contains("unknown remote bind mode 0"))
        );
    }

    #[test]
    fn a_census_keeps_its_host_reading() {
        for read in [
            HostRead::Given,
            HostRead::MemAvailable,
            HostRead::VmStats,
            HostRead::Cgroup { path: "/".into() },
            HostRead::Cgroup {
                path: "/user.slice/job.scope".into(),
            },
            HostRead::Machine,
        ] {
            let census = NodeCensus {
                devices: vec![gb10(7)],
                host_room: (1234, read),
                arena_given: Some(0),
            };
            let mut out = Out::default();
            census.write(&mut out);
            let mut input = In::new(out.done());
            assert_eq!(
                NodeCensus::read(&mut input),
                Ok(census),
                "the census must preserve the room and its reading"
            );
            input.end().expect("the exact census payload");
            for n in 0..out.done().len() {
                assert!(
                    NodeCensus::read(&mut In::new(&out.done()[..n])).is_err(),
                    "a truncated census at byte {n} must be refused"
                );
            }
        }
        let mut out = Out::default();
        out.u32(super::wire::VERSION).u64(1234).u8(255);
        let refusal = NodeCensus::read(&mut In::new(out.done())).unwrap_err();
        assert!(
            refusal.contains("unknown host reading 255"),
            "an undefined host reading must be named: {refusal}"
        );
        out.clear().u32(5).u64(1234).u8(0);
        let refusal = NodeCensus::read(&mut In::new(out.done())).unwrap_err();
        assert!(
            refusal.contains("protocol version 5"),
            "a census without a host reading must be refused by version: {refusal}"
        );
    }

    #[test]
    fn messages_read_back_as_written() {
        let census = NodeCensus {
            host_room: (u64::MAX, HostRead::Given),
            arena_given: None,
            devices: vec![gb10(7)],
        };
        let mut o = Out::default();
        census.write(&mut o);
        let mut i = In::new(o.done());
        assert_eq!(NodeCensus::read(&mut i), Ok(census.clone()));
        assert!(i.end().is_ok());
        let mut full = gb10(8);
        full.free_bytes = full.total_bytes + 1;
        for bad in [Vec::new(), vec![full], vec![gb10(7), gb10(7)]] {
            let bad = NodeCensus {
                devices: bad,
                ..census.clone()
            };
            let mut o = Out::default();
            bad.write(&mut o);
            assert!(NodeCensus::read(&mut In::new(o.done())).is_err());
        }
        let mut o = Out::default();
        census.write(&mut o);
        let mut other = o.done().to_vec();
        other[0] ^= 0xFF;
        let refused = NodeCensus::read(&mut In::new(&other)).unwrap_err();
        assert!(refused.contains("protocol version"), "{refused}");

        let set = TierSet::new(1..3, 8, vec![vec![2, 5], vec![]]).expect("a set");
        let open = NodeOpen {
            nvme: None,
            family: "qwen4exp".into(),
            name: "GB10".into(),
            device: Some(DeviceId {
                ordinal: 0,
                uuid: [7; 16],
            }),
            card: 1,
            rows: vec![NodeRow {
                name: "t.1.up".into(),
                shard: 2,
                ty: "q5_1".into(),
                dims: vec![512, 2560, 8],
                file_bytes: 1 << 20,
                format: "kquant".into(),
                experts: Some(vec![2, 5]),
                resident_bytes: 4096,
                sha256: "9f".repeat(32),
            }],
            set,
            shape: TierShape {
                hidden: 2560,
                n_used: 10,
                rows: 1,
                act: TierAct::F32,
                reply: ReplyRows::Whole,
            },
            cols: 4,
            layers: 3,
            block: (512, 1 << 20),
        };
        let mut timed = open.clone();
        timed.rows[0].format = Format::HostFile.to_string();
        timed.rows[0].file_bytes = 32 * OPEN_RATE;
        timed.rows[0].resident_bytes = 0;
        assert_eq!(
            open_bound(&timed).unwrap(),
            OPEN_BASE + Duration::from_secs(8)
        );
        timed.rows[0].experts = None;
        assert_eq!(
            open_bound(&timed).unwrap(),
            OPEN_BASE + Duration::from_secs(32)
        );
        timed.rows[0].resident_bytes = 64 * OPEN_RATE;
        assert_eq!(
            open_bound(&timed).unwrap(),
            OPEN_BASE + Duration::from_secs(64)
        );
        timed.rows[0].resident_bytes = u64::MAX;
        timed.rows.push(timed.rows[0].clone());
        assert!(open_bound(&timed).is_err());
        timed.rows.truncate(1);
        timed.rows[0].experts = Some(vec![1]);
        *timed.rows[0].dims.last_mut().unwrap() = 0;
        assert!(open_bound(&timed).is_err());

        let mut o = Out::default();
        open.write(&mut o).expect("the open");
        let mut i = In::new(o.done());
        let back = NodeOpen::read(&mut i).expect("the open");
        assert!(i.end().is_ok());
        assert_eq!(
            (back.family, back.name, back.device, back.card, back.rows),
            (
                open.family.clone(),
                open.name.clone(),
                open.device,
                open.card,
                open.rows.clone()
            )
        );
        assert_eq!(
            (back.set, back.shape, back.cols, back.layers, back.block),
            (
                open.set.clone(),
                open.shape,
                open.cols,
                open.layers,
                open.block
            )
        );
        for c in [
            Chain::Step,
            Chain::Pair,
            Chain::Quad,
            Chain::Cols(2),
            Chain::Cols(5),
            Chain::Cols(model::ops::DEFER_MAX_COLS),
        ] {
            assert_eq!(chain_of(chain_word(c).expect("a word")), Ok(c));
        }
        assert!(chain_of(3).is_err());
        for c in [
            Chain::Cols(0),
            Chain::Cols(1),
            Chain::Cols(model::ops::DEFER_MAX_COLS + 1),
        ] {
            assert!(chain_of(chain_word(c).expect("a word")).is_err());
        }
        assert!(chain_word(Chain::Cols(usize::MAX)).is_err());
        for side in [WordSide::Memop, WordSide::Kernel] {
            assert_eq!(side_of(side_byte(side)), Ok(side));
        }
        assert!(side_of(2).is_err());

        let (path, file) = empty_file("node");
        let mut session = Session {
            local_ip: std::net::Ipv4Addr::LOCALHOST.into(),
            reset_phase: 0,
            file: &file,
            families: &[],
            opened: None,
            asked: 0,
            pass: 0,
            seen: (0, Instant::now()),
            listed: Vec::new(),
            reply: Out::default(),
            on_open: &mut |_| {},
        };
        session.reset_phase = 1;
        for (tag, payload) in [(Tag::Launch, vec![]), (Tag::Reset, vec![0])] {
            let Err(Failed::Refused(e)) = session.handle(tag, &payload) else {
                panic!("an out-of-order remote reset must be refused");
            };
            assert!(e.to_string().contains("reset"), "{e}");
        }
        session.reset_phase = 0;
        for payload in [vec![2], vec![1], vec![1, 0]] {
            let Err(Failed::Refused(GpuError::Shape { .. })) = session.handle(Tag::Bind, &payload)
            else {
                panic!("an unknown or incomplete bind must be refused before opening a card");
            };
        }
        for cols in [1, 4] {
            let mut o = Out::default();
            o.u32(20).len(cols).len(1).len(1).len(0);
            let Err(Failed::Refused(e)) = session.handle(Tag::Capture, o.done()) else {
                panic!("the capture must be refused before opening a tier");
            };
            if cols == 1 {
                assert!(matches!(e, GpuError::Shape { .. }), "{e}");
                assert!(
                    e.to_string().contains("a Cols(4) capture of 1 columns"),
                    "{e}"
                );
            } else {
                assert!(matches!(e, GpuError::State { .. }), "{e}");
            }
            assert!(session.listed.is_empty());
        }
        drop(session);
        drop(file);
        std::fs::remove_file(&path).expect("the model file removed");

        let fault = Fault::of_sites(5, &[FaultSite::Q5Quant, FaultSite::HcQuant]);
        let addr = SocketAddr::from(([127, 0, 0, 1], 9));
        for e in [
            GpuError::fault("t", fault),
            GpuError::protocol("t", "a go out of order"),
            GpuError::shape("t", "a chain of 0 columns"),
        ] {
            let mut o = Out::default();
            write_err(&e, &mut o);
            let back = read_err(addr, "step", o.done()).expect("an error reply");
            let kind = match (&e, &back) {
                (GpuError::Fault { .. }, GpuError::Fault { fault: f, .. }) => *f == fault,
                (GpuError::Protocol { .. }, GpuError::Protocol { .. })
                | (GpuError::Shape { .. }, GpuError::Shape { .. }) => true,
                _ => false,
            };
            assert!(
                kind && back.to_string().contains(&e.to_string()),
                "{e} read back as {back:?}"
            );
        }

        for arena in [
            model::placement::workstation::ArenaGrant {
                planned: 4096,
                granted: 4096,
            },
            model::placement::workstation::ArenaGrant {
                planned: 8192,
                granted: 4096,
            },
        ] {
            let mut paged = open.clone();
            paged.nvme = Some(super::NodeNvme {
                room: 1 << 30,
                arena,
                reserve: 65536,
            });
            let mut out = Out::default();
            paged.write(&mut out).unwrap();
            let mut header = Out::default();
            header
                .u64(1 << 30)
                .u64(arena.planned)
                .u64(arena.granted)
                .u64(65536);
            assert_eq!(
                &out.done()[1..33],
                header.done(),
                "only the NVMe budget is on the wire"
            );
            let mut resident = Out::default();
            open.write(&mut resident).unwrap();
            assert_eq!(out.done().len(), resident.done().len() + 32);
            let mut input = In::new(out.done());
            let back = NodeOpen::read(&mut input).unwrap();
            assert_eq!(back.nvme, paged.nvme);
            assert!(input.end().is_ok());
            // PIN(2026-10-17): node-zero is retired with the unused ordinal; arena bounds remain.
            for granted in [0, arena.planned + 1] {
                paged.nvme.as_mut().unwrap().arena.granted = granted;
                out.clear();
                paged.write(&mut out).unwrap();
                assert!(
                    NodeOpen::read(&mut In::new(out.done()))
                        .unwrap_err()
                        .contains("arena grant")
                );
            }
        }
        rows_on_refuses_another_file_and_other_experts(&open);
    }

    /// [`NodeOpen::rows_on`] takes a segment its file holds as node 0's
    /// plan does, of the set's experts; another file's tensor, and the
    /// set's count of other experts, are refused by name.
    fn rows_on_refuses_another_file_and_other_experts(open: &NodeOpen) {
        let r = &open.rows[0];
        let model = |dims: Vec<u64>| ModelTensors {
            tensors: vec![ModelTensor {
                name: r.name.clone(),
                shard: r.shard,
                layer: Some(1),
                role: Role::RoutedExperts,
                ty: GgmlType::Q5_1,
                dims,
                file_bytes: r.file_bytes,
                gathered_rows: None,
            }],
            layers: 3,
            experts: 8,
            experts_used: 2,
            bf16_words: |_| false,
        };
        assert!(open.rows_on(&model(r.dims.clone())).is_ok());
        let mut absent = open.clone();
        absent.rows.clear();
        let missing = absent.rows_on(&model(r.dims.clone())).unwrap_err();
        assert!(
            missing.to_string().contains("missing routed stack"),
            "{missing}"
        );
        let mut repeated = open.clone();
        repeated.rows.push(r.clone());
        let duplicate = repeated.rows_on(&model(r.dims.clone())).unwrap_err();
        assert!(
            duplicate.to_string().contains("more than once"),
            "{duplicate}"
        );

        let mut wide = model(r.dims.clone());
        let template = wide.tensors[0].clone();
        wide.layers = 48;
        wide.tensors = (0..wide.layers)
            .map(|layer| ModelTensor {
                name: format!("t.{layer}.up"),
                layer: Some(layer),
                ..template.clone()
            })
            .collect();
        let mut trimmed = open.clone();
        trimmed.layers = wide.layers;
        trimmed.set = TierSet::new(
            3..46,
            8,
            (3..46)
                .map(|l| if l == 24 { vec![] } else { vec![2, 5] })
                .collect(),
        )
        .unwrap();
        trimmed.rows = (3..46)
            .filter(|&l| l != 24)
            .map(|layer| NodeRow {
                name: format!("t.{layer}.up"),
                ..r.clone()
            })
            .collect();
        assert_eq!(trimmed.rows_on(&wide).unwrap().len(), trimmed.rows.len());
        let mut missing = trimmed.clone();
        missing.rows.pop();
        assert!(
            missing
                .rows_on(&wide)
                .unwrap_err()
                .to_string()
                .contains("missing routed stack")
        );
        let mut duplicate = trimmed.clone();
        duplicate.rows.push(trimmed.rows[0].clone());
        assert!(
            duplicate
                .rows_on(&wide)
                .unwrap_err()
                .to_string()
                .contains("more than once")
        );
        for layers in [0, 2, 4, usize::MAX] {
            let mut bad = open.clone();
            bad.layers = layers;
            let e = bad.rows_on(&model(r.dims.clone())).unwrap_err();
            assert!(e.to_string().contains("layers"), "{e}");
        }
        let mut bad = open.clone();
        bad.set = TierSet::new(3..4, 8, vec![vec![]]).expect("a set past the file");
        let e = bad.rows_on(&model(r.dims.clone())).unwrap_err();
        assert!(e.to_string().contains("layers"), "{e}");
        for cols in [0, model::ops::DEFER_MAX_COLS + 1, usize::MAX] {
            let mut bad = open.clone();
            bad.cols = cols;
            let e = bad.rows_on(&model(r.dims.clone())).unwrap_err();
            assert!(e.to_string().contains("columns"), "{e}");
        }
        for cols in [1, model::ops::DEFER_MAX_COLS] {
            let mut valid = open.clone();
            valid.cols = cols;
            assert!(valid.rows_on(&model(r.dims.clone())).is_ok());
        }
        let other = open.rows_on(&model(vec![512, 2560, 16])).unwrap_err();
        assert!(other.to_string().contains("other files"), "{other}");
        let mut swapped = open.clone();
        swapped.rows[0].experts = Some(vec![3, 6]);
        let ids = swapped.rows_on(&model(r.dims.clone())).unwrap_err();
        assert!(ids.to_string().contains("its set"), "{ids}");
    }
}
