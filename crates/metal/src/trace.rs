//! The encode trace: what the host asks of Metal, as text.
//!
//! The kernels' pipelines are what `bin/metal_scan.rs` hashes; the other half
//! of a run's program is the host's dispatch: which pipeline, how many groups
//! of how many threads, which buffer window and which `setBytes` value at each
//! slot, the barriers, and the command buffers' commit, event and wait order.
//! This module writes that, one line per event, while a trace is started
//! ([`start`], [`finish`]).
//!
//! A buffer is named by its allocation order (`#n`, the nth buffer
//! [`crate::MetalBuffer`] made in the process), an offset and its length, never
//! by an address; a command buffer by the order it was made. A dispatch line
//! states the whole binding table the dispatch runs under (Metal keeps a slot's
//! binding until it is set again), so the order of the `set` calls between two
//! dispatches is not part of the trace. The host calls no
//! `setThreadgroupMemoryLength`; a dispatch's threadgroup memory is its
//! pipeline's static length, printed on the line.
//!
//! Every line is one of:
//!
//! ```text
//! cb N begin | cb N wait-event NAME=V | cb N signal NAME=V | cb N commit
//! enc N.M open serial|concurrent | enc N.M end
//! dispatch N.M.K LABEL grid=GXxGY tpg=T max=… width=… tg_static=… b0=… b1=…
//! barrier N.M
//! host settle V | host sync
//! ```

use core::fmt::Write as _;
use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, PoisonError};

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLComputePipelineState, MTLDispatchType};

use crate::encode::MAX_BINDINGS;
use crate::library::Pipeline;

/// The format's version, the first line of a trace.
pub const VERSION: &str = "# bloomery encode trace v1";

struct State {
    /// A trace is being written.
    active: bool,
    lines: Vec<String>,
    /// Buffer address to (allocation order, length): every buffer made since
    /// the process began, the last one at an address winning.
    buffers: BTreeMap<usize, (u64, usize)>,
    made: u64,
    /// Buffers bound but never made by [`crate::MetalBuffer`], by first sight.
    strangers: BTreeMap<usize, u64>,
    /// Command buffer address to its order.
    command_buffers: BTreeMap<usize, u64>,
    command_buffers_made: u64,
    /// Encoders opened on each command buffer so far.
    encoders: BTreeMap<u64, u64>,
}

static STATE: Mutex<State> = Mutex::new(State {
    active: false,
    lines: Vec::new(),
    buffers: BTreeMap::new(),
    made: 0,
    strangers: BTreeMap::new(),
    command_buffers: BTreeMap::new(),
    command_buffers_made: 0,
    encoders: BTreeMap::new(),
});

fn state() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

impl State {
    fn emit(&mut self, line: impl FnOnce() -> String) {
        if self.active {
            let line = line();
            self.lines.push(line);
        }
    }

    /// The order of the command buffer at `address`.
    fn command_buffer(&mut self, address: usize) -> u64 {
        match self.command_buffers.get(&address) {
            Some(&n) => n,
            None => {
                let n = self.command_buffers_made;
                self.command_buffers_made += 1;
                self.command_buffers.insert(address, n);
                n
            }
        }
    }
}

/// Start a trace, dropping the lines an earlier one wrote.
pub fn start() {
    let mut state = state();
    state.active = true;
    state.lines.clear();
}

/// The trace written since [`start`], as text; the trace stops.
#[must_use]
pub fn finish() -> String {
    let mut state = state();
    state.active = false;
    let mut text = String::from(VERSION);
    text.push('\n');
    for line in state.lines.drain(..) {
        text.push_str(&line);
        text.push('\n');
    }
    text
}

/// A buffer of `length` bytes at `address` was made.
pub(crate) fn buffer_made(address: usize, length: usize) {
    let mut state = state();
    let n = state.made;
    state.made += 1;
    state.buffers.insert(address, (n, length));
}

/// A command buffer was made.
pub(crate) fn command_buffer_made(buffer: &ProtocolObject<dyn MTLCommandBuffer>) {
    let mut state = state();
    // A new command buffer at an address a finished one had is a new order.
    let address = core::ptr::from_ref(buffer).addr();
    state.command_buffers.remove(&address);
    let n = state.command_buffer(address);
    state.emit(|| format!("cb {n} begin"));
}

/// A command buffer encodes a wait for the event `name` to reach `value`.
pub(crate) fn wait_event(buffer: &ProtocolObject<dyn MTLCommandBuffer>, name: &str, value: u64) {
    let mut state = state();
    let n = state.command_buffer(core::ptr::from_ref(buffer).addr());
    state.emit(|| format!("cb {n} wait-event {name}={value}"));
}

/// A command buffer encodes a signal of the event `name` at `value`.
pub(crate) fn signal_event(buffer: &ProtocolObject<dyn MTLCommandBuffer>, name: &str, value: u64) {
    let mut state = state();
    let n = state.command_buffer(core::ptr::from_ref(buffer).addr());
    state.emit(|| format!("cb {n} signal {name}={value}"));
}

/// A command buffer was committed.
pub(crate) fn commit(buffer: &ProtocolObject<dyn MTLCommandBuffer>) {
    let mut state = state();
    let n = state.command_buffer(core::ptr::from_ref(buffer).addr());
    state.emit(|| format!("cb {n} commit"));
}

/// The host waits: `what` is `settle V` or `sync`.
pub(crate) fn host(what: impl FnOnce() -> String) {
    state().emit(|| format!("host {what}", what = what()));
}

/// What one compute encoder has been asked: the trace's side of
/// [`crate::encode::ComputeEncoder`].
pub(crate) struct EncoderTrace {
    /// The encoder's name, `N.M`: command buffer N, its Mth encoder.
    name: String,
    pipeline: Option<String>,
    bound: [Option<String>; MAX_BINDINGS],
    dispatches: u64,
}

impl EncoderTrace {
    /// An encoder just made on `buffer` of dispatch type `dispatch`.
    pub(crate) fn open(
        buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        dispatch: MTLDispatchType,
    ) -> Self {
        let mut state = state();
        let n = state.command_buffer(core::ptr::from_ref(buffer).addr());
        let m = *state.encoders.entry(n).and_modify(|m| *m += 1).or_insert(0);
        let name = format!("{n}.{m}");
        let kind = if dispatch == MTLDispatchType::Concurrent {
            "concurrent"
        } else {
            "serial"
        };
        state.emit(|| format!("enc {name} open {kind}"));
        Self {
            name,
            pipeline: None,
            bound: [const { None }; MAX_BINDINGS],
            dispatches: 0,
        }
    }

    /// The pipeline the next dispatches run.
    pub(crate) fn set_pipeline(&mut self, pipeline: &Pipeline) {
        let state = pipeline.metal();
        self.pipeline = Some(format!(
            "{} max={} width={} tg_static={}",
            pipeline.label(),
            pipeline.max_threads_per_group(),
            state.threadExecutionWidth(),
            state.staticThreadgroupMemoryLength()
        ));
    }

    /// Slot `index` holds a window of `buffer` from `offset` bytes.
    pub(crate) fn set_buffer(
        &mut self,
        index: usize,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
    ) {
        let address = core::ptr::from_ref(buffer).addr();
        let mut state = state();
        let name = match state.buffers.get(&address) {
            Some(&(n, length)) => format!("#{n}+{offset}/{length}"),
            None => {
                let next = state.strangers.len() as u64;
                let n = *state.strangers.entry(address).or_insert(next);
                format!("#?{n}+{offset}/{}", buffer.length())
            }
        };
        self.bound[index] = Some(name);
    }

    /// Slot `index` holds the value `data`.
    pub(crate) fn set_bytes(&mut self, index: usize, data: &[u8]) {
        self.bound[index] = Some(format!("bytes:{}", gguf::digest::hex(data)));
    }

    /// A dispatch of `groups_x` by `groups_y` groups of `threads` threads.
    pub(crate) fn dispatch(&mut self, groups_x: usize, groups_y: usize, threads: usize) {
        let k = self.dispatches;
        self.dispatches += 1;
        let mut line = format!(
            "dispatch {}.{k} {} grid={groups_x}x{groups_y} tpg={threads}",
            self.name,
            self.pipeline.as_deref().unwrap_or("-"),
        );
        for (index, slot) in self.bound.iter().enumerate() {
            if let Some(slot) = slot {
                let _ = write!(line, " b{index}={slot}");
            }
        }
        state().emit(|| line);
    }

    /// A barrier.
    pub(crate) fn barrier(&self) {
        state().emit(|| format!("barrier {}", self.name));
    }
}

impl Drop for EncoderTrace {
    fn drop(&mut self) {
        state().emit(|| format!("enc {} end", self.name));
    }
}
