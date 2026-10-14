//! Tier nodes: an expert tier whose card is in another process
//! (`docs/contrib/dgx-spark.md` §9.5). The process that runs the stage card
//! (node 0) drives the tier as it drives a card of its own
//! ([`super::tier::TierCard`]); the tier's work ([`TierWork`]) runs in a tier
//! server's process (node 1, [`serve`]), and each step node 0 asks of it is a
//! request over a TCP control channel ([`TierNode`]). The page the
//! protocol's words and rows live on is a file under `/dev/shm` both
//! processes map ([`super::link::Link::across`]), so both processes share one
//! host's memory: `BLOOMERY_TIER_NODES` takes loopback addresses only, and
//! the channel has no authentication.
//!
//! On connect the server sends its census ([`NodeCensus`]): node 0 plans
//! the tier on the device it names ([`node_cards`]). The open
//! carries the tier's rows of node 0's plan ([`NodeOpen`]); the server loads
//! them from its own copy of the model file, which must match the tier's
//! set, each row's tensor and the sha256 of each row's bytes, and builds the
//! family's tier computation over them ([`NodeFamily`]).
//!
//! Every read and write of the channel is bounded: a request and its reply
//! by one deadline, the open by one its rows' bytes give ([`open_bound`]).
//! A lost node ends both sides by name. Node 0 marks a tier whose channel
//! failed lost ([`TierNode::lost`]), and the host tier names it a lost card;
//! a node that stops signalling mid-step is the go deadline's, as for a card
//! of node 0's. The server ends when node 0's channel ends without a close,
//! when a request fails after it reached the tier's stream, or when work it
//! was asked for has made no progress past the go deadline and its grace: it
//! writes [`super::step::RELEASE`] to its own counters and go words, so
//! every wait its stream holds passes and node 0's next settle refuses the
//! step by name ([`super::tier::TierCard::released`]); it drains its stream
//! and returns the loss by name. Over loopback a peer's end is always seen;
//! a half-open connection to another host is the transport's to bound once
//! one runs there.

pub mod wire;

use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::Path;
use std::time::{Duration, Instant};

use gguf::Split;
use model::fileio::sha256_hex_of;
use model::placement::devices::{DeviceId, DeviceInfo, spec_of_device};
use model::placement::workstation::CardSpec;
use model::placement::{Device, ExpertList, Format, ModelTensor, ModelTensors, Plan, Row, Segment};

use super::DRAIN_DEADLINE;
use super::step::{Chain, GO_DEADLINE};
use super::tier::{
    TIER_CHAINS, TIER_GRACE, TierAct, TierExperts, TierLayout, TierSet, TierShape, TierWork,
    chain_slot,
};
use crate::fault::{FAULT_NONE, Fault};
use crate::weights::{SegmentFile, Weights};
use crate::{Gpu, GpuError};
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

/// How long node 0 waits for the open of `open`: [`OPEN_BASE`] and its
/// rows' resident bytes at [`OPEN_RATE`].
fn open_bound(open: &NodeOpen) -> Duration {
    let bytes: u64 = open.rows.iter().map(|r| r.resident_bytes).sum();
    OPEN_BASE + Duration::from_secs(bytes / OPEN_RATE)
}

// ---------------------------------------------------------------- census

/// A tier node's census: its visible devices as its driver enumerates them
/// ([`crate::census`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeCensus {
    pub devices: Vec<DeviceInfo>,
}

impl NodeCensus {
    /// This process's census, as a tier server sends it.
    pub fn here() -> Result<NodeCensus, GpuError> {
        Ok(NodeCensus {
            devices: crate::census()?,
        })
    }

    fn write(&self, o: &mut Out) {
        o.u32(wire::VERSION).len(self.devices.len());
        for d in &self.devices {
            o.u32(d.ordinal)
                .str(&d.name)
                .u64(d.total_bytes)
                .u64(d.free_bytes)
                .u8(u8::from(d.integrated))
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
            });
        }
        Ok(NodeCensus { devices })
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
/// card format, experts and resident bytes, and the sha256 of the file bytes
/// the segment holds in node 0's file ([`segment_sha256`]), which the bytes
/// the node loads must match.
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

/// The sha256 of the file bytes segment `seg` of `t`, a stack of `experts`
/// when it is one, holds in `file`: its spans' bytes in slot order, the
/// bytes its upload reads ([`SegmentFile::of`]).
fn segment_sha256(
    file: &Split,
    experts: u64,
    t: &ModelTensor,
    seg: &Segment,
) -> Result<String, GpuError> {
    let f = SegmentFile::of(file, experts, t, seg)?;
    Ok(sha256_hex_of(f.bytes))
}

/// What a tier node opens its tier from ([`TierNode::open`]): the family
/// whose computation it runs, the card by its plan name and its device in
/// the node's census, the plan's device index of the tier's segments and
/// those segments ([`NodeOpen::rows_of`]), the tier's set, the shapes it is
/// cut for and its columns, the model's layers, and the block route's
/// columns with the plan's reserve for it.
#[derive(Clone, Debug)]
pub struct NodeOpen {
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

impl NodeOpen {
    /// The segments `plan` puts on its card `card`, each with its tensor's
    /// name and file identity and the sha256 of its bytes in `file`, the
    /// model file the plan was made from.
    pub fn rows_of(plan: &Plan<'_>, file: &Split, card: usize) -> Result<Vec<NodeRow>, GpuError> {
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
                    sha256: segment_sha256(file, plan.model.experts, t, s)?,
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
    /// name: the stage side's slots name the set's experts.
    pub fn rows_on(&self, model: &ModelTensors) -> Result<Vec<Row>, GpuError> {
        let mut rows = Vec::with_capacity(self.rows.len());
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
        Ok(rows)
    }

    /// The open as its request carries it; a layer of the set's range with
    /// no row is refused by name.
    fn write(&self, o: &mut Out) -> Result<(), GpuError> {
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
        o.len(self.cols)
            .len(self.layers)
            .len(self.block.0)
            .len(self.block.1);
        Ok(())
    }

    fn read(i: &mut In<'_>) -> Result<NodeOpen, String> {
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
        Ok(NodeOpen {
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
    match w {
        0 => Ok(Chain::Step),
        1 => Ok(Chain::Pair),
        2 => Ok(Chain::Quad),
        COLS_WORD.. => Ok(Chain::Cols((w - COLS_WORD) as usize)),
        _ => Err(format!("a chain word {w}")),
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

// ---------------------------------------------------------------- client

/// Node 0's end of a tier node's channel: each [`TierWork`] step the stage
/// side asks of the tier is a request, answered once the server has taken
/// it. A channel that fails, times out or carries what the protocol does
/// not marks the node lost ([`TierNode::lost`]), and every later request
/// fails by name.
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
    lost: Option<String>,
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
            lost: None,
        };
        open.write(node.req.clear())?;
        node.call(Tag::Open, open_bound(open), "open")?;
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

    /// Mark the node lost by `why` in the request `what`, as an error.
    fn lose_now(&mut self, what: &str, why: String) -> GpuError {
        let why = format!("the {what} request: {why}");
        let e = GpuError::link(WHAT, format!("the node at {}: {why}", self.addr));
        self.lost = Some(why);
        e
    }

    /// Send the request `tag` with the payload [`TierNode::req`] holds and
    /// read its reply, both within `within`; the reply's payload is then
    /// [`Channel::body`]. A refusal by the server is its error, of the kind
    /// it raised; a channel that fails loses the node.
    fn call(&mut self, tag: Tag, within: Duration, what: &'static str) -> Result<(), GpuError> {
        if let Some(why) = &self.lost {
            return Err(GpuError::link(
                WHAT,
                format!(
                    "the node at {} is lost ({why}): no {what} request",
                    self.addr
                ),
            ));
        }
        let deadline = Instant::now() + within;
        let r = self
            .ch
            .send(tag, self.req.done(), deadline)
            .and_then(|()| self.ch.recv(deadline));
        match r {
            Ok(Tag::Ok) => Ok(()),
            Ok(Tag::Err) => match read_err(self.addr, what, self.ch.body()) {
                Ok(e) => Err(e),
                Err(why) => Err(self.lose_now(what, why)),
            },
            Ok(t) => Err(self.lose_now(what, format!("a {t:?} frame where a reply belongs"))),
            Err(e) => Err(self.lose_now(what, e.to_string())),
        }
    }

    /// A request whose reply carries nothing.
    fn ask(&mut self, tag: Tag, within: Duration, what: &'static str) -> Result<(), GpuError> {
        self.call(tag, within, what)?;
        let end = In::new(self.ch.body()).end();
        end.map_err(|e| self.lose_now(what, e))
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
        let path = path.to_str().ok_or_else(|| {
            GpuError::shape(WHAT, format!("a page path {} not UTF-8", path.display()))
        })?;
        self.req.clear().str(path).len(bytes).len(tiers).len(index);
        self.ask(Tag::Bind, REPLY, "bind")
    }

    /// [`TierWork::launch`] on the node.
    pub(super) fn launch(&mut self, chain: Chain) -> Result<(), GpuError> {
        let w = chain_word(chain)?;
        self.req.clear().u32(w);
        self.ask(Tag::Launch, REPLY, "launch")
    }

    /// [`TierWork::capture`] on the node: the graph's node count.
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
        let n = r.len("nodes").and_then(|n| r.end().map(|()| n));
        let n = n.map_err(|e| self.lose_now("capture", e))?;
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

    /// [`TierWork::sync`] on the node, under the drain deadline.
    pub(super) fn sync(&mut self) -> Result<(), GpuError> {
        self.req.clear();
        self.ask(Tag::Sync, DRAIN_DEADLINE + REPLY, "sync")
    }

    /// [`TierWork::drain`] on the node, under `deadline` (the server's
    /// drain deadline at most).
    pub(super) fn drain(&mut self, deadline: Duration) -> Result<(), GpuError> {
        let deadline = deadline.min(DRAIN_DEADLINE);
        let ms = u64::try_from(deadline.as_millis()).unwrap_or(u64::MAX);
        self.req.clear().u64(ms);
        self.ask(Tag::Drain, deadline + REPLY, "drain")
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
}

impl Drop for TierNode {
    /// A close the node answers once its stream has drained; nothing to a
    /// lost node.
    fn drop(&mut self) {
        if self.lost.is_none() {
            let deadline = Instant::now() + DRAIN_DEADLINE + REPLY;
            drop(
                self.ch
                    .send(Tag::Close, &[], deadline)
                    .and_then(|()| self.ch.recv(deadline)),
            );
        }
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
}

/// A family's tier computation over the stacks a node loaded for an open
/// ([`NodeFamily::experts`]).
pub type OpenExperts =
    fn(&Gpu, &ModelTensors, &Weights, &NodeOpen) -> Result<Box<dyn TierExperts>, GpuError>;

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

/// The tier a connection opened: its work, its card's name, its set and
/// its columns.
struct Opened {
    work: TierWork,
    name: String,
    set: TierSet,
    cols: usize,
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
/// its progress as last seen and when it last moved, per chain the layers
/// its captured graph serves, and the reply being written.
struct Session<'a> {
    file: &'a Split,
    families: &'a [NodeFamily],
    opened: Option<Opened>,
    asked: u32,
    seen: (u32, Instant),
    listed: Vec<(u32, u32)>,
    reply: Out,
}

/// Serve the expert tier of one load on `listener`, from `file`: each
/// connection gets this process's census; a connection that closes before
/// an open is a census read, and the server waits for the next. The first
/// open's load is served until node 0 closes it, and the server returns.
/// Node 0's channel ended with no close, a protocol it does not speak, a
/// request that failed after it reached the tier's stream, or work asked of
/// the tier with no progress past the go deadline and its grace is node 0
/// lost: the tier's words are released, its stream drained, and the loss
/// returned by name.
pub fn serve(
    listener: &TcpListener,
    file: &Split,
    families: &[NodeFamily],
) -> Result<(), GpuError> {
    loop {
        let (s, peer) = listener
            .accept()
            .map_err(|e| GpuError::link(SERVER, format!("accept: {e}")))?;
        let lost = |e: std::io::Error| GpuError::link(SERVER, format!("node 0 at {peer}: {e}"));
        let mut ch = Channel::new(s).map_err(lost)?;
        let mut session = Session {
            file,
            families,
            opened: None,
            asked: 0,
            seen: (0, Instant::now()),
            listed: Vec::new(),
            reply: Out::default(),
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
                            "the channel ended with no close, the tier past {} of the {} layers \
                             asked of it",
                            o.work.opened_progress().unwrap_or(0),
                            self.asked
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

    /// Node 0 stopped: work was asked of the tier and its progress has not
    /// moved past the go deadline and its grace.
    fn check_stopped(&mut self) -> Result<(), String> {
        let Some(p) = self.opened.as_ref().and_then(|o| o.work.opened_progress()) else {
            return Ok(());
        };
        let now = Instant::now();
        if p != self.seen.0 || p == self.asked {
            self.seen = (p, now);
            return Ok(());
        }
        let waited = now - self.seen.1;
        if waited > GO_DEADLINE + TIER_GRACE {
            return Err(format!(
                "the tier past {p} of the {} layers asked of it and node 0 sent no go for \
                 {waited:.1?}",
                self.asked
            ));
        }
        Ok(())
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
        match tag {
            Tag::Open => self.handle_open(&mut i, bad)?,
            Tag::Bind => {
                let path = i.str("path").map_err(bad)?;
                let (bytes, tiers, index) = (
                    i.len("bytes").map_err(bad)?,
                    i.len("tiers").map_err(bad)?,
                    i.len("index").map_err(bad)?,
                );
                i.end().map_err(bad)?;
                let o = self.opened()?;
                o.work.bind_opened(Path::new(&path), bytes, tiers, index)?;
                let p = o.work.opened_progress().unwrap_or(0);
                self.seen = (p, Instant::now());
                self.asked = p;
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
            Tag::Sync => {
                i.end().map_err(bad)?;
                self.opened()?
                    .work
                    .drain(DRAIN_DEADLINE, "tier_server: the sync")?;
            }
            Tag::Drain => {
                let ms = i.u64("deadline").map_err(bad)?;
                i.end().map_err(bad)?;
                let deadline = Duration::from_millis(ms).min(DRAIN_DEADLINE);
                self.opened()?
                    .work
                    .drain(deadline, "tier_server: the tier's stream")?;
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
            Tag::Census | Tag::Close | Tag::Ok | Tag::Err => {
                return Err(bad("not a request".to_string()).into());
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
        let len = u32::try_from(list.len()).map_err(|_| bad("a list past u32".into()))?;
        let o = self.opened()?;
        for &(layer, row) in &list {
            o.check_service(layer, row, cols)?;
        }
        let nodes = o.work.capture(chain, cols, &list).map_err(Failed::Fatal)?;
        self.listed.retain(|(w, _)| *w != c);
        if len > 0 {
            self.listed.push((c, len));
        }
        self.reply.len(nodes);
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
        let w = Weights::load_rows(gpu.stream(), self.file, &tensors, &rows, open.card)?;
        let experts = (family.experts)(&gpu, &tensors, &w, open)?;
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
        o.work.release_opened();
        let drained = o
            .work
            .drain(DRAIN_DEADLINE, "tier_server: the lost load's drain");
        let state = match &drained {
            Ok(()) => "the tier released its words and its stream drained".to_string(),
            Err(e) => format!("the tier released its words and its stream did not drain: {e}"),
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
mod tests {
    use std::net::SocketAddr;

    use super::wire::{In, Out};
    use super::{NodeCensus, NodeOpen, NodeRow, chain_of, chain_word, read_err, write_err};
    use crate::GpuError;
    use crate::fault::{Fault, FaultSite};
    use crate::host::step::Chain;
    use crate::host::tier::{TierAct, TierSet, TierShape};
    use gguf::GgmlType;
    use model::placement::devices::{DeviceId, DeviceInfo};
    use model::placement::{ModelTensor, ModelTensors, Role};

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
        }
    }

    /// A census and an open read back as written; every chain's word names
    /// it back; an error reply reads back as the server's fault (its words
    /// kept), protocol error or refusal, with its text. A census of another
    /// version, of no device, of a device with more free bytes than total or
    /// of one uuid twice is refused.
    #[test]
    fn messages_read_back_as_written() {
        let census = NodeCensus {
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
            let bad = NodeCensus { devices: bad };
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
            },
            cols: 4,
            layers: 3,
            block: (512, 1 << 20),
        };
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
        for c in [Chain::Step, Chain::Pair, Chain::Quad, Chain::Cols(5)] {
            assert_eq!(chain_of(chain_word(c).expect("a word")), Ok(c));
        }
        assert!(chain_of(3).is_err());
        assert!(chain_word(Chain::Cols(usize::MAX)).is_err());

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
        };
        assert!(open.rows_on(&model(r.dims.clone())).is_ok());
        let other = open.rows_on(&model(vec![512, 2560, 16])).unwrap_err();
        assert!(other.to_string().contains("other files"), "{other}");
        let mut swapped = open.clone();
        swapped.rows[0].experts = Some(vec![3, 6]);
        let ids = swapped.rows_on(&model(r.dims.clone())).unwrap_err();
        assert!(ids.to_string().contains("its set"), "{ids}");
    }
}
