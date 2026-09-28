//! The route trace: for every position the engine runs, each hybrid layer's
//! routed ids in router rank order and the slot each one ran in, as a router
//! set on disk — `tools/ref/router_trace.cpp`'s layout, which
//! `tools/ref/router-coverage.py` (`read_manifest`, `read_topk`) reads:
//!
//! ```text
//! MANIFEST.tsv      the header by name, a `layer` row per layer, a `call` row per prompt call,
//!                   `# positions` last, then `# complete` once the writer is finished
//! topk-<layer>.u16  positions × n_used ids, little-endian, position-major, in step order
//! slots-<layer>.u8  the same shape: `C` the stage card, `T` the tier card, `H` the host
//! ```
//!
//! The step port records a one-row step's service of each layer before its
//! signal (the page holds the handoff until then) and appends the row to the
//! files once its last layer is served; `MANIFEST.tsv` is then rewritten
//! whole (a new file renamed over the old) with `# positions` the rows on
//! disk. A row whose append or manifest fails is cut back off every layer
//! file, and the trace then refuses to complete. A process killed between a
//! row's appends and its manifest leaves files longer than the manifest:
//! `tools/bloomery/route_trace.py trim` cuts them back and says so. `# complete`
//! is written once, by [`RouteTrace::finish`]: a writer dropped without it
//! (a failed run's unwinding, a process ending on an error) leaves the set
//! incomplete and says so on stderr, and a server stopped by a signal is
//! sealed by its driver (`route_trace.py seal`).
//! A `call` row marks a prompt call: `prompt_call` positions from `first`
//! ran its ids one step each; every later position up to the next call is a
//! step on the id the one before it answered, or on a prompt id a caller
//! feeds outside the call (a server's last prompt id). A call that ran
//! fewer ids than it named — a step refused them — is closed at the ids it
//! ran when the next call is marked or the writer finishes, its row dropped
//! when it ran none, each with a `route trace:` line on stderr naming it.
//!
//! What the trace does not record is refused by name, never skipped: a
//! position before any prompt call, a service out of layer order, a pass of
//! two rows or of several columns, a prompt batch (the host tier refuses its
//! services while a trace is attached).

use super::slots::{Slot, SlotMap};
use super::step::Chain;
use crate::GpuError;
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// What every error of the trace names.
const WHAT: &str = "RouteTrace";

/// The manifest's first line: `router-coverage.py` opens a set whose title
/// starts with `# router_trace`.
const TITLE: &str = "# router_trace — bloomery engine: the routed ids of every position the \
                     engine ran, in step order, as the host step port read them";

/// The header keys the trace writes itself; a caller's extra line may not
/// take one, nor a column line's key.
const RESERVED: [&str; 22] = [
    "model",
    "model_file",
    "arch",
    "build",
    "engine",
    "tokens",
    "n_expert",
    "n_expert_used",
    "n_layer",
    "feed",
    "slots",
    "positions",
    "sealed",
    "complete",
    "layer",
    "call",
    "kind",
    "input",
    "int",
    "draft",
    "verify",
    "plain",
];

/// The most routed ids a position the writer's row buffer holds.
const MAX_USED: usize = 32;

/// The slot kinds as the `slots-<layer>.u8` files hold them.
pub const KIND_CARD: u8 = b'C';
pub const KIND_TIER: u8 = b'T';
pub const KIND_HOST: u8 = b'H';

/// What the manifest's header says about the run besides the positions.
#[derive(Clone, Debug)]
pub struct TraceHeader {
    /// The model's first shard, its full path.
    pub model: PathBuf,
    /// The file's architecture.
    pub arch: String,
    /// The engine's build, as the binary knows it.
    pub build: String,
    pub n_expert: usize,
    /// Routed ids a position selects.
    pub n_used: usize,
    /// Layers `0 .. n_layer`, each a hybrid layer every step serves.
    pub n_layer: usize,
    /// The placement and the levers that move the routing's numerics, as
    /// header lines in order.
    pub extra: Vec<(String, String)>,
}

/// A prompt call: `prompt` positions from `first` ran its ids, the first at
/// cache position `pos0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Call {
    pub first: u64,
    pub prompt: u64,
    pub pos0: u32,
}

/// The files of one layer, appended a position at a time.
struct LayerFiles {
    topk: File,
    slots: File,
}

/// The trace's writer: one per process, attached to the host tier
/// ([`super::HostTier::attach_route_trace`]).
pub struct RouteTrace {
    dir: PathBuf,
    header: TraceHeader,
    files: Vec<LayerFiles>,
    /// The position being served, layer-major: `n_layer × n_used` ids and
    /// their slot kinds.
    ids: Vec<u16>,
    kinds: Vec<u8>,
    /// The layer the position's next service must be.
    next: usize,
    /// Positions on disk.
    rows: u64,
    calls: Vec<Call>,
    /// The first write that failed: the trace then never completes.
    failed: Option<String>,
    /// [`RouteTrace::finish`] ran: `# complete` is on disk, or it returned
    /// why not.
    done: bool,
}

fn io_err(path: &Path, e: &std::io::Error) -> GpuError {
    GpuError::shape(WHAT, format!("{}: {e}", path.display()))
}

impl RouteTrace {
    /// The trace into `dir`, which is created here as a new directory — an
    /// existing path or a missing parent is refused by name — with its empty
    /// files and a manifest of no position.
    pub fn create(dir: &Path, header: TraceHeader) -> Result<RouteTrace, GpuError> {
        let h = &header;
        if h.n_layer == 0
            || !(1..=MAX_USED).contains(&h.n_used)
            || h.n_expert == 0
            || h.n_expert > 1 << 16
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} layers of {} experts, {} a position: the trace holds u16 ids of at least \
                     one layer and 1 to {MAX_USED} slots",
                    h.n_layer, h.n_expert, h.n_used
                ),
            ));
        }
        for (k, v) in &h.extra {
            let bad = k.is_empty()
                || k.contains(['\t', '\n', ' '])
                || v.contains('\n')
                || RESERVED.contains(&k.as_str())
                || h.extra.iter().filter(|(o, _)| o == k).count() > 1;
            if bad {
                return Err(GpuError::shape(
                    WHAT,
                    format!("the header line {k:?} = {v:?} is not a key of its own on one line"),
                ));
            }
        }
        fs::create_dir(dir).map_err(|e| {
            GpuError::shape(
                WHAT,
                format!(
                    "{} cannot be created as a new directory: {e}",
                    dir.display()
                ),
            )
        })?;
        let mut files = Vec::with_capacity(h.n_layer);
        for l in 0..h.n_layer {
            let open = |name: String| {
                let p = dir.join(name);
                OpenOptions::new()
                    .append(true)
                    .create_new(true)
                    .open(&p)
                    .map_err(|e| io_err(&p, &e))
            };
            files.push(LayerFiles {
                topk: open(format!("topk-{l}.u16"))?,
                slots: open(format!("slots-{l}.u8"))?,
            });
        }
        let n = h.n_layer * h.n_used;
        let t = RouteTrace {
            dir: dir.to_path_buf(),
            ids: vec![0; n],
            kinds: vec![0; n],
            header,
            files,
            next: 0,
            rows: 0,
            calls: Vec::new(),
            failed: None,
            done: false,
        };
        t.write_manifest(false)?;
        Ok(t)
    }

    /// The directory the trace writes.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Positions on disk.
    #[must_use]
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// The prompt calls marked so far.
    #[must_use]
    pub fn calls(&self) -> &[Call] {
        &self.calls
    }

    /// The trace's layers and routed ids a position, for the tier that
    /// attaches it: its slot map must cover exactly those layers of
    /// `n_expert` experts, and its handoff carry `n_used` ids.
    pub(crate) fn fits(&self, slots: &SlotMap, n_used: usize) -> Result<(), GpuError> {
        let h = &self.header;
        if slots.layers() != (0..h.n_layer) || slots.n_expert() != h.n_expert || n_used != h.n_used
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a trace of layers 0..{} of {} experts, {} a position, on a host tier of \
                     layers {:?} of {} experts, {n_used} a position",
                    h.n_layer,
                    h.n_expert,
                    h.n_used,
                    slots.layers(),
                    slots.n_expert()
                ),
            ));
        }
        Ok(())
    }

    /// A prompt call of `n` ids from cache position `pos0` begins: the next
    /// `n` positions are its. Refused between two layers of a position and
    /// for a call of no id. The last call, when it ran fewer ids than it
    /// named, is closed first (`RouteTrace::close_short_call`).
    pub fn prompt(&mut self, pos0: u32, n: usize) -> Result<(), GpuError> {
        if self.next != 0 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a prompt call marked after layer {} of a position was served",
                    self.next - 1
                ),
            ));
        }
        if n == 0 {
            return Err(GpuError::shape(WHAT, "a prompt call of no id"));
        }
        self.close_short_call(&format!(
            "before the call of {n} ids from cache position {pos0}"
        ));
        self.calls.push(Call {
            first: self.rows,
            prompt: n as u64,
            pos0,
        });
        Ok(())
    }

    /// The last call closed at the ids it ran when a step refused the rest:
    /// its row dropped when it ran none, else its `prompt_call` cut to the
    /// positions it holds. Either is a `route trace:` line on stderr, `when`
    /// saying what closed it.
    fn close_short_call(&mut self, when: &str) {
        let (n, rows) = (self.calls.len(), self.rows);
        let Some(c) = self.calls.last_mut() else {
            return;
        };
        let (i, ran) = (n - 1, rows - c.first);
        if ran >= c.prompt {
            return;
        }
        if ran == 0 {
            eprintln!(
                "route trace: call {i} (position {}, {} ids from cache position {}) ran none of \
                 its ids, dropped {when}",
                c.first, c.prompt, c.pos0
            );
            self.calls.pop();
        } else {
            eprintln!(
                "route trace: call {i} (position {}, {} ids from cache position {}) ran {ran} of \
                 its ids, closed at {ran} {when}",
                c.first, c.prompt, c.pos0
            );
            c.prompt = ran;
        }
    }

    /// The position being served is dropped: its layers recorded so far
    /// were never written, and the next service is layer 0 of a new one.
    /// The host tier's reset calls it when it lifts a refusal.
    pub(crate) fn abandon_position(&mut self) {
        self.next = 0;
    }

    /// Record `chain`'s service of layer `layer`: its `n_used` routed ids,
    /// id `s` the page's `id_at(s)`, and the slot `slots` runs each in.
    /// `true` once the position's last layer is recorded: the caller then
    /// writes it ([`RouteTrace::write_row`]) after its signal.
    pub(crate) fn record(
        &mut self,
        layer: usize,
        chain: Chain,
        slots: &SlotMap,
        id_at: impl Fn(usize) -> Option<u32>,
    ) -> Result<bool, GpuError> {
        if chain != Chain::Step {
            return Err(GpuError::shape(
                WHAT,
                format!("a {chain:?} service: the trace records one-row steps"),
            ));
        }
        if self.calls.is_empty() {
            return Err(GpuError::shape(
                WHAT,
                "a step before any prompt call: the trace has no call its position belongs to",
            ));
        }
        if layer != self.next {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {layer}'s service while the position expects layer {}",
                    self.next
                ),
            ));
        }
        let k = self.header.n_used;
        for s in 0..k {
            let id = id_at(s)
                .ok_or_else(|| GpuError::shape(WHAT, "the routing is outside the handoff"))?;
            let kind = match slots.slot(layer, id) {
                Some(Slot::Card(_)) => KIND_CARD,
                Some(Slot::Tier(_)) => KIND_TIER,
                Some(Slot::Host) => KIND_HOST,
                None => {
                    return Err(GpuError::shape(
                        WHAT,
                        format!("layer {layer} slot {s} names expert {id}, which no slot holds"),
                    ));
                }
            };
            let id = u16::try_from(id)
                .map_err(|_| GpuError::shape(WHAT, format!("expert {id} passes u16")))?;
            self.ids[layer * k + s] = id;
            self.kinds[layer * k + s] = kind;
        }
        self.next += 1;
        if self.next == self.header.n_layer {
            self.next = 0;
            return Ok(true);
        }
        Ok(false)
    }

    /// Append the recorded position to every layer's files, then rewrite
    /// the manifest with it. On an error every layer file is cut back to the
    /// row's start, the manifest keeps the positions before it, and the
    /// trace no longer completes.
    pub(crate) fn write_row(&mut self) -> Result<(), GpuError> {
        if let Some(why) = &self.failed {
            return Err(GpuError::shape(
                WHAT,
                format!("a row after a failed write ({why})"),
            ));
        }
        let start = self.rows;
        let written = self.append_row().and_then(|()| {
            self.rows = start + 1;
            self.write_manifest(false)
        });
        let Err(e) = written else {
            return Ok(());
        };
        self.rows = start;
        let e = match self.cut_back(start) {
            Ok(()) => GpuError::shape(
                WHAT,
                format!("{e}; every layer file cut back to its {start} positions"),
            ),
            Err(cut) => GpuError::shape(
                WHAT,
                format!("{e}; cutting the layer files back to {start} positions failed too: {cut}"),
            ),
        };
        self.failed = Some(e.to_string());
        Err(e)
    }

    /// The recorded position appended to every layer's two files.
    fn append_row(&mut self) -> Result<(), GpuError> {
        let k = self.header.n_used;
        let mut bytes = [0u8; 2 * MAX_USED];
        for (l, f) in self.files.iter_mut().enumerate() {
            for (b, id) in bytes
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(&self.ids[l * k..][..k])
            {
                *b = id.to_le_bytes();
            }
            let at = |name: String| self.dir.join(name);
            f.topk
                .write_all(&bytes[..2 * k])
                .map_err(|e| io_err(&at(format!("topk-{l}.u16")), &e))?;
            f.slots
                .write_all(&self.kinds[l * k..][..k])
                .map_err(|e| io_err(&at(format!("slots-{l}.u8")), &e))?;
        }
        Ok(())
    }

    /// Every layer's files cut to `rows` positions, each opened by its
    /// path: the append handle may be what failed. An append handle writes
    /// at the new end.
    fn cut_back(&self, rows: u64) -> Result<(), GpuError> {
        let k = self.header.n_used as u64;
        for l in 0..self.header.n_layer {
            for (name, len) in [
                (format!("topk-{l}.u16"), rows * 2 * k),
                (format!("slots-{l}.u8"), rows * k),
            ] {
                let p = self.dir.join(name);
                OpenOptions::new()
                    .write(true)
                    .open(&p)
                    .and_then(|f| f.set_len(len))
                    .map_err(|e| io_err(&p, &e))?;
            }
        }
        Ok(())
    }

    /// Mark the set complete: a last call short of its ids is closed
    /// (`RouteTrace::close_short_call`) and the manifest gets
    /// `# complete`. Refused after a failed write. Returns the positions on
    /// disk.
    pub fn finish(mut self) -> Result<u64, GpuError> {
        self.done = true;
        if let Some(why) = &self.failed {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{}: not completed: a write failed ({why})",
                    self.dir.display()
                ),
            ));
        }
        self.close_short_call("as the writer finished");
        self.write_manifest(true)?;
        Ok(self.rows)
    }

    /// The manifest of the positions on disk, written beside and renamed
    /// over the old one; with `# complete` when `complete`.
    fn write_manifest(&self, complete: bool) -> Result<(), GpuError> {
        let h = &self.header;
        let mut m = String::new();
        let file = h.model.file_name().map_or_else(
            || h.model.display().to_string(),
            |f| f.to_string_lossy().into_owned(),
        );
        let _ = writeln!(m, "{TITLE}");
        let _ = writeln!(m, "# model\t{}", h.model.display());
        let _ = writeln!(m, "# model_file\t{file}");
        let _ = writeln!(m, "# arch\t{}", h.arch);
        let _ = writeln!(m, "# build\t{}", h.build);
        let _ = writeln!(m, "# engine\tbloomery (not an ik build)");
        let _ = writeln!(m, "# tokens\t{}", self.rows);
        let _ = writeln!(m, "# n_expert\t{}", h.n_expert);
        let _ = writeln!(m, "# n_expert_used\t{}", h.n_used);
        let _ = writeln!(m, "# n_layer\t{}", h.n_layer);
        let _ = writeln!(
            m,
            "# feed\tone step per position: a prompt call's ids one step each (call rows)"
        );
        let _ = writeln!(
            m,
            "# slots\tslots-<layer>.u8 beside topk-<layer>.u16: C stage card, T tier card, H host"
        );
        for (k, v) in &h.extra {
            let _ = writeln!(m, "# {k}\t{v}");
        }
        let _ = writeln!(m, "# layer\tlayer\ttokens\tfile\tslots");
        for l in 0..h.n_layer {
            let _ = writeln!(m, "layer\t{l}\t{}\ttopk-{l}.u16\tslots-{l}.u8", self.rows);
        }
        let _ = writeln!(m, "# call\tindex\tfirst\tprompt_call\tend\tpos0");
        for (i, c) in self.calls.iter().enumerate() {
            let end = self.calls.get(i + 1).map_or(self.rows, |n| n.first);
            let _ = writeln!(m, "call\t{i}\t{}\t{}\t{end}\t{}", c.first, c.prompt, c.pos0);
        }
        let _ = writeln!(m, "# positions\t{}", self.rows);
        if complete {
            let _ = writeln!(m, "# complete\t{}\t{}", self.rows, h.n_layer);
        }
        let tmp = self.dir.join("MANIFEST.tsv.tmp");
        let path = self.dir.join("MANIFEST.tsv");
        fs::write(&tmp, m).map_err(|e| io_err(&tmp, &e))?;
        fs::rename(&tmp, &path).map_err(|e| io_err(&path, &e))
    }
}

impl Drop for RouteTrace {
    /// A writer dropped without [`RouteTrace::finish`] leaves its set
    /// without `# complete` — a drop cannot tell a finished run from a failed
    /// one — and says so on stderr.
    fn drop(&mut self) {
        if !self.done {
            eprintln!(
                "route trace: {} dropped without finish: {} positions on disk, not complete",
                self.dir.display(),
                self.rows
            );
        }
    }
}

/// A trace set read back as [`RouteTrace`] wrote it — the gate's view of
/// what is on disk.
#[derive(Clone, Debug)]
pub struct TraceSet {
    /// The `# tokens` line.
    pub tokens: u64,
    pub n_used: usize,
    /// Per layer `0 .. n`, its ids and slot kinds, position-major.
    pub ids: Vec<Vec<u16>>,
    pub kinds: Vec<Vec<u8>>,
    /// The `call` rows: first, prompt_call, end, pos0.
    pub calls: Vec<(u64, u64, u64, u32)>,
    /// The `# positions` line.
    pub positions: Option<u64>,
    /// The `# complete` line's positions.
    pub complete: Option<u64>,
}

impl TraceSet {
    /// The set in `dir`: its manifest's header, `layer` and `call` rows by
    /// their column lines' names, and every layer's two files, whose lengths
    /// must be the manifest's positions.
    pub fn read(dir: &Path) -> Result<TraceSet, GpuError> {
        let path = dir.join("MANIFEST.tsv");
        let text = fs::read_to_string(&path).map_err(|e| io_err(&path, &e))?;
        let bad = |n: usize, why: String| {
            GpuError::shape(WHAT, format!("{}:{}: {why}", path.display(), n + 1))
        };
        let mut lines = text.lines().enumerate();
        match lines.next() {
            Some((_, t)) if t.starts_with("# router_trace") => {}
            _ => return Err(bad(0, "not a router set manifest".to_string())),
        }
        let (mut tokens, mut n_used, mut positions, mut complete) = (None, None, None, None);
        let (mut layer_cols, mut call_cols): (Vec<&str>, Vec<&str>) = (Vec::new(), Vec::new());
        let (mut layers, mut calls) = (Vec::new(), Vec::new());
        let field = |cols: &[&str], f: &[&str], name: &str, n: usize| -> Result<u64, GpuError> {
            // Field 0 is the row's kind; a name is looked up past it, as
            // `tools/bloomery/manifest.py` reads the `layer` line.
            let i = cols
                .iter()
                .skip(1)
                .position(|c| *c == name)
                .ok_or_else(|| bad(n, format!("no {name} column")))?
                + 1;
            f.get(i)
                .and_then(|v| v.parse::<u64>().ok())
                .ok_or_else(|| bad(n, format!("{name} is not a whole number")))
        };
        for (n, line) in lines {
            if let Some(rest) = line.strip_prefix("# ") {
                let (key, value) = rest.split_once('\t').unwrap_or((rest, ""));
                let num = || {
                    value
                        .split('\t')
                        .next()
                        .and_then(|v| v.parse::<u64>().ok())
                        .ok_or_else(|| bad(n, format!("# {key} {value:?}")))
                };
                match key {
                    "tokens" => tokens = Some(num()?),
                    "n_expert_used" => n_used = Some(num()?),
                    "positions" => positions = Some(num()?),
                    "complete" => complete = Some(num()?),
                    "layer" => layer_cols = rest.split('\t').collect(),
                    "call" => call_cols = rest.split('\t').collect(),
                    _ => {}
                }
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            match f.first().copied() {
                Some("layer") if f.len() == layer_cols.len() => {
                    layers.push(field(&layer_cols, &f, "layer", n)?);
                }
                Some("call") if f.len() == call_cols.len() => calls.push((
                    field(&call_cols, &f, "first", n)?,
                    field(&call_cols, &f, "prompt_call", n)?,
                    field(&call_cols, &f, "end", n)?,
                    u32::try_from(field(&call_cols, &f, "pos0", n)?)
                        .map_err(|_| bad(n, "pos0 passes u32".to_string()))?,
                )),
                _ => return Err(bad(n, format!("a row no column line reads: {line:?}"))),
            }
        }
        let (tokens, n_used) = match (tokens, n_used) {
            (Some(t), Some(k)) => (t, usize::try_from(k).unwrap_or(usize::MAX)),
            _ => return Err(bad(0, "no # tokens or # n_expert_used".to_string())),
        };
        if layers != (0..layers.len() as u64).collect::<Vec<_>>() {
            return Err(bad(0, format!("the layer rows are {layers:?}, not 0..n")));
        }
        let want = usize::try_from(tokens).unwrap_or(usize::MAX) * n_used;
        let (mut ids, mut kinds) = (Vec::new(), Vec::new());
        for l in 0..layers.len() {
            let t = dir.join(format!("topk-{l}.u16"));
            let s = dir.join(format!("slots-{l}.u8"));
            let tb = fs::read(&t).map_err(|e| io_err(&t, &e))?;
            let sb = fs::read(&s).map_err(|e| io_err(&s, &e))?;
            if tb.len() != 2 * want || sb.len() != want {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {l}: {} and {} bytes, not the manifest's {tokens} positions × {n_used}",
                        tb.len(),
                        sb.len()
                    ),
                ));
            }
            ids.push(
                tb.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| u16::from_le_bytes(*b))
                    .collect(),
            );
            kinds.push(sb);
        }
        Ok(TraceSet {
            tokens,
            n_used,
            ids,
            kinds,
            calls,
            positions,
            complete,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trace of 2 layers of 8 experts, 2 ids a position, in a new
    /// directory under the system's temp dir named for `name`.
    fn trace(name: &str) -> RouteTrace {
        let dir = std::env::temp_dir().join(format!(
            "bloomery-route-trace-{}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let header = TraceHeader {
            model: PathBuf::from("/models/t/m-00001-of-00001.gguf"),
            arch: "test".to_owned(),
            build: "test".to_owned(),
            n_expert: 8,
            n_used: 2,
            n_layer: 2,
            extra: Vec::new(),
        };
        RouteTrace::create(&dir, header).unwrap()
    }

    /// One position recorded as the step port would, ids `id, id + 1` on
    /// every layer, and written.
    fn row(t: &mut RouteTrace, id: u16) -> Result<(), GpuError> {
        for (i, v) in t.ids.iter_mut().enumerate() {
            *v = id + (i % 2) as u16;
        }
        t.kinds.fill(KIND_HOST);
        t.write_row()
    }

    fn manifest(t: &RouteTrace) -> String {
        fs::read_to_string(t.dir().join("MANIFEST.tsv")).unwrap()
    }

    #[test]
    fn complete_only_when_finished() {
        let mut t = trace("finish");
        t.prompt(0, 1).unwrap();
        row(&mut t, 1).unwrap();
        row(&mut t, 3).unwrap();
        let dir = t.dir().to_path_buf();
        let open = manifest(&t);
        assert!(
            open.contains("# positions\t2\n") && !open.contains("# complete"),
            "{open}"
        );
        let set = TraceSet::read(&dir).unwrap();
        assert_eq!((set.positions, set.complete), (Some(2), None));
        assert_eq!(t.finish().unwrap(), 2);
        let set = TraceSet::read(&dir).unwrap();
        assert_eq!((set.positions, set.complete), (Some(2), Some(2)));
        assert_eq!(set.ids[1], [1, 2, 3, 4]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn drop_leaves_incomplete() {
        let mut t = trace("drop");
        t.prompt(0, 1).unwrap();
        row(&mut t, 1).unwrap();
        let dir = t.dir().to_path_buf();
        drop(t);
        let set = TraceSet::read(&dir).unwrap();
        assert_eq!((set.positions, set.complete), (Some(1), None));
        fs::remove_dir_all(&dir).unwrap();
    }

    /// A failed append cuts every layer file back to the positions before
    /// it; the manifest keeps them, and the trace refuses to complete.
    #[test]
    fn failed_write_cuts_back() {
        let mut t = trace("cut");
        t.prompt(0, 3).unwrap();
        row(&mut t, 1).unwrap();
        let dir = t.dir().to_path_buf();
        // Layer 1's slots file behind a read-only handle: layer 0 and layer
        // 1's ids are appended, then the write fails.
        t.files[1].slots = File::open(dir.join("slots-1.u8")).unwrap();
        let e = row(&mut t, 5).unwrap_err().to_string();
        assert!(e.contains("cut back to its 1 positions"), "{e}");
        for (name, len) in [
            ("topk-0.u16", 4),
            ("slots-0.u8", 2),
            ("topk-1.u16", 4),
            ("slots-1.u8", 2),
        ] {
            assert_eq!(fs::metadata(dir.join(name)).unwrap().len(), len, "{name}");
        }
        assert_eq!(t.rows(), 1);
        assert!(manifest(&t).contains("# positions\t1\n"));
        assert!(
            row(&mut t, 7)
                .unwrap_err()
                .to_string()
                .contains("a row after a failed write")
        );
        assert!(
            t.finish()
                .unwrap_err()
                .to_string()
                .contains("not completed: a write failed")
        );
        assert_eq!(TraceSet::read(&dir).unwrap().complete, None);
        fs::remove_dir_all(&dir).unwrap();
    }

    /// A call a step refused is closed at the ids it ran: dropped at none,
    /// cut at some; the next call is then marked.
    #[test]
    fn short_calls_close() {
        let mut t = trace("short");
        t.prompt(0, 2).unwrap();
        t.prompt(0, 3).unwrap();
        assert_eq!(
            t.calls(),
            [Call {
                first: 0,
                prompt: 3,
                pos0: 0
            }]
        );
        row(&mut t, 1).unwrap();
        t.prompt(0, 2).unwrap();
        row(&mut t, 1).unwrap();
        row(&mut t, 1).unwrap();
        t.prompt(4, 5).unwrap();
        assert_eq!(
            t.calls(),
            [
                Call {
                    first: 0,
                    prompt: 1,
                    pos0: 0
                },
                Call {
                    first: 1,
                    prompt: 2,
                    pos0: 0
                },
                Call {
                    first: 3,
                    prompt: 5,
                    pos0: 4
                },
            ]
        );
        let dir = t.dir().to_path_buf();
        assert_eq!(t.finish().unwrap(), 3);
        assert_eq!(
            TraceSet::read(&dir).unwrap().calls,
            [(0, 1, 1, 0), (1, 2, 3, 0)]
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    /// A position cut short between two layers refuses a call until it is
    /// abandoned, as the host tier's reset does.
    #[test]
    fn abandoned_position() {
        let mut t = trace("abandon");
        t.prompt(0, 1).unwrap();
        row(&mut t, 1).unwrap();
        t.next = 1;
        let e = t.prompt(1, 1).unwrap_err().to_string();
        assert!(e.contains("marked after layer 0 of a position"), "{e}");
        t.abandon_position();
        t.prompt(1, 1).unwrap();
        let dir = t.dir().to_path_buf();
        drop(t);
        fs::remove_dir_all(&dir).unwrap();
    }
}
