//! V4.1's draft layers for the common DSpark owner
//! ([`bloomery_gpu::dspark::DraftLayers`]): the three window-only blocks and
//! their context rings, without the parts the owner holds (fc and the hidden
//! norm before the append, the head's chain after the block).
//!
//! The owner hands the layers rows; the layers keep the positions:
//!
//! - [`V41Layers::stage_kv`] writes each context row's ring slot and rope
//!   table into a pinned image and copies it to the card; nothing of the
//!   rows themselves goes through it, they are the owner's `x`.
//!   [`V41Layers::append_kv`] then runs, per layer, `attn_kv` over `x` and
//!   the norm, rope and ring append ([`super::kv::AppendCx`], the launches
//!   [`super::kv::KvAppend`] runs over its own rows).
//! - [`V41Layers::stage_block`] reads the target's embedding row of the
//!   block's first id and writes it with the rope tables and visible counts
//!   ([`BlockPass::stage`]). [`V41Layers::enqueue_block`] decodes row 0
//!   (the 4-stream broadcast), runs the layers and leaves the last layer's
//!   fold in the owner's `x`: the last ffn's HC post writes it there, so no
//!   copy follows ([`BlockPass::enqueue_layers`]).

use std::mem::size_of;
use std::sync::Arc;

use bloomery_gpu::dspark::DraftLayers;
use bloomery_gpu::dspark::stage::{Inbox, put_f32};
use bloomery_gpu::{Gpu, GpuError, Window, capturing};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::dspark::DraftHparams;

use super::block::{BlockInput, BlockPass, MAX_WIDTH, Rule, embedding_row};
use super::kv::{AppendCx, AppendDims, DraftRings, GROUP, GroupRows};
use super::load::DraftWeights;
use crate::rope::{Direction, RopeKernels, RopeSpec, RopeTable};

const WHAT: &str = "draft::layers";

/// A block's own first rope position and visible ring rows, in place of the
/// ones the owner stages: an oracle set's, which differ from the engine's
/// (ik turns each committed row after the prompt at `p + 1` and its block
/// one further on).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OracleBlock {
    /// Row 0's rope position; row `t` turns at `first_pos + t`.
    pub first_pos: u32,
    /// Ring rows every block row attends.
    pub ring_rows: usize,
}

/// Where a group's inputs sit in the append's image, in words: from
/// `g · words`, each row's ring slot ([`GROUP`] words; the kernel takes them
/// modulo the window), then each row's rope table ([`RopeTable::push`]'s
/// layout, room for [`GROUP`] rows). The context rows are the owner's, so
/// the image holds no feature rows.
#[derive(Clone, Copy)]
struct Layout {
    slots: usize,
    cs: usize,
    rope_dims: usize,
    words: usize,
}

impl Layout {
    fn of(rope_dims: usize) -> Layout {
        Layout {
            slots: 0,
            cs: GROUP,
            rope_dims,
            words: GROUP + GROUP * rope_dims,
        }
    }

    /// Words of the image an append of `n` rows reads: every full group,
    /// then the last group up to its last row's rope table.
    fn used(&self, n: usize) -> usize {
        let (full, rest) = (n / GROUP, n % GROUP);
        full * self.words
            + if rest == 0 {
                0
            } else {
                self.cs + rest * self.rope_dims
            }
    }
}

/// One group's scratch: up to [`GROUP`] rows' `attn_kv` output and normed,
/// turned rows, per layer.
struct Group {
    kv: Vec<DeviceBuffer<f32>>,
    out: Vec<DeviceBuffer<f32>>,
}

/// The context append's buffers and constants.
struct Append {
    groups: Vec<Group>,
    /// Every group's slots and rope tables, one copy an append.
    inbox: Inbox,
    layout: Layout,
    /// The rope table of the row being staged; kept to stage without
    /// allocating.
    cs: Vec<f32>,
    table: RopeTable,
    rope: RopeKernels,
    dims: AppendDims,
    n_layer: usize,
    window: usize,
    /// Rows the last stage wrote; 0 before the first.
    staged: usize,
}

impl Append {
    /// Scratch for up to `hp.window` rows per append. Load-time only.
    fn new(gpu: &Gpu, hp: &DraftHparams) -> Result<Append, GpuError> {
        let s = gpu.stream();
        let n_groups = hp.window.div_ceil(GROUP);
        let per_layer = || {
            (0..hp.n_layer)
                .map(|_| DeviceBuffer::zeroed(s, GROUP * hp.head_dim))
                .collect::<Result<Vec<_>, _>>()
        };
        let groups = (0..n_groups)
            .map(|_| {
                Ok(Group {
                    kv: per_layer()?,
                    out: per_layer()?,
                })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        let layout = Layout::of(hp.rope_dims);
        Ok(Append {
            groups,
            inbox: Inbox::new(gpu, n_groups * layout.words)?,
            layout,
            cs: Vec::with_capacity(GROUP * hp.rope_dims),
            table: RopeTable::new(&RopeSpec::window(hp.rope_base, hp.rope_dims))?,
            rope: RopeKernels::load(gpu.context())?,
            dims: AppendDims {
                n_embd: hp.n_embd,
                head_dim: hp.head_dim,
                rope_dims: hp.rope_dims,
                eps: hp.rms_eps,
            },
            n_layer: hp.n_layer,
            window: hp.window,
            staged: 0,
        })
    }

    /// Rows one append takes: the ring's window, in whole groups.
    fn capacity(&self) -> usize {
        self.groups.len() * GROUP
    }

    /// Write the slots and rope tables of `n` rows at positions
    /// `first .. first + n` — row `i` in ring slot `(first % window) + i`,
    /// turned at `first + i` — and enqueue the image's one copy on `stream`.
    fn stage(&mut self, stream: &CudaStream, first: u32, n: usize) -> Result<(), GpuError> {
        let shape = |detail: String| GpuError::Shape { what: WHAT, detail };
        if n == 0 || n > self.capacity() {
            return Err(shape(format!(
                "an append of {n} rows; 1..={}",
                self.capacity()
            )));
        }
        let window = u32::try_from(self.window)
            .map_err(|_| shape(format!("a window of {} rows", self.window)))?;
        if u32::try_from(n)
            .ok()
            .and_then(|n| first.checked_add(n))
            .is_none()
        {
            return Err(shape(format!("positions {first} + {n} pass a u32")));
        }
        let first_slot = first % window;
        let lay = self.layout;
        let host = self.inbox.host_mut()?;
        for gi in 0..self.groups.len() {
            let r0 = gi * GROUP;
            let rows = n.saturating_sub(r0).min(GROUP);
            if rows == 0 {
                continue;
            }
            let img = &mut host[gi * lay.words..(gi + 1) * lay.words];
            let (slots, cs) = img.split_at_mut(lay.cs);
            self.cs.clear();
            for (i, slot) in slots.iter_mut().enumerate() {
                if i >= rows {
                    *slot = 0;
                    continue;
                }
                let off = (r0 + i) as u32;
                self.table
                    .push(first + off, Direction::Forward, &mut self.cs);
                *slot = first_slot + off;
            }
            put_f32(cs, &self.cs);
        }
        self.inbox.upload(stream, lay.used(n))?;
        self.staged = n;
        Ok(())
    }

    /// Every layer's `attn_kv` and append of `n` rows of `x` (token-major,
    /// `n_embd` a row) into `rings`, the launches of `n` rows whatever was
    /// staged: a capture before any stage records the same ones a replay
    /// after a stage of `n` rows runs. Eager, `n` is the staged count.
    /// Capturable.
    fn enqueue(
        &mut self,
        gpu: &Gpu,
        w: &DraftWeights,
        rings: &mut DraftRings,
        x: &DeviceBuffer<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        let shape = |detail: String| GpuError::Shape { what: WHAT, detail };
        let n_embd = self.dims.n_embd;
        if n == 0 || n > self.capacity() {
            return Err(shape(format!(
                "an append of {n} rows; 1..={}",
                self.capacity()
            )));
        }
        if n != self.staged && !(self.staged == 0 && capturing(gpu.stream())?) {
            return Err(shape(format!(
                "an append of {n} rows after a stage of {}",
                self.staged
            )));
        }
        if x.len() < n * n_embd {
            return Err(shape(format!(
                "{n} rows of {n_embd} in a buffer of {} values",
                x.len()
            )));
        }
        let (ring_set, shadow) = rings.split_mut();
        if ring_set.len() != self.n_layer {
            return Err(shape(format!(
                "{} rings for {} layers",
                ring_set.len(),
                self.n_layer
            )));
        }
        let cx = AppendCx {
            gpu,
            rope: &self.rope,
            w,
            dims: self.dims,
        };
        let lay = self.layout;
        for (l, ring) in ring_set.iter_mut().enumerate() {
            for (gi, g) in self.groups.iter_mut().enumerate().take(n.div_ceil(GROUP)) {
                let rows = (n - gi * GROUP).min(GROUP);
                let at = gi * lay.words;
                let cs = self.inbox.f32s(at + lay.cs, GROUP * lay.rope_dims)?;
                let slots = self.inbox.u32s(at + lay.slots, GROUP)?;
                let window =
                    Window::<f32>::of(x, gi * GROUP * n_embd * size_of::<f32>(), rows * n_embd)?;
                let group = GroupRows {
                    x: &window,
                    rows,
                    cs: &cs,
                    slots: &slots,
                };
                cx.enqueue_layer(
                    l,
                    &group,
                    (&mut g.kv[l], &mut g.out[l]),
                    (&mut *ring, &mut *shadow),
                )?;
            }
        }
        Ok(())
    }

    /// Kernel launches one append of `n` rows makes: per group, per layer,
    /// `attn_kv` and the append.
    fn launches(&self, n: usize) -> usize {
        n.div_ceil(GROUP) * 2 * self.n_layer
    }
}

/// V4.1's draft layers. See the module comment.
pub struct V41Layers {
    w: DraftWeights,
    rings: DraftRings,
    append: Append,
    pass: BlockPass,
    /// The target's file, for the embedding row of each block's first id.
    target: Arc<Split>,
    /// The rule the next staged block follows.
    rule: Rule,
    /// An oracle set's positions for the next staged blocks, when one is set.
    oracle: Option<OracleBlock>,
    /// Card bytes the layers hold: the weights they keep and their buffers.
    bytes: u64,
}

impl V41Layers {
    /// Rings, the append's scratch and the block pass over `w`'s draft.
    /// Load-time only.
    pub fn new(gpu: &Gpu, w: DraftWeights, target: Arc<Split>) -> Result<V41Layers, GpuError> {
        let (free, _) = gpu.mem_info()?;
        let rings = DraftRings::new(gpu.stream(), w.hp())?;
        let append = Append::new(gpu, w.hp())?;
        let pass = BlockPass::new(gpu, &w)?;
        let (left, _) = gpu.mem_info()?;
        let bytes = w.layer_bytes() + free.saturating_sub(left) as u64;
        Ok(V41Layers {
            w,
            rings,
            append,
            pass,
            target,
            rule: Rule::Reference,
            oracle: None,
            bytes,
        })
    }

    /// The weights.
    #[must_use]
    pub fn weights(&self) -> &DraftWeights {
        &self.w
    }

    /// The block pass, as the last staged block left it.
    #[must_use]
    pub fn pass(&self) -> &BlockPass {
        &self.pass
    }

    /// The rings.
    #[must_use]
    pub fn rings(&self) -> &DraftRings {
        &self.rings
    }

    /// The rings, writable: the gates' way to seat an oracle's rows. Graphs
    /// read them in place, so a replay sees what is seated.
    pub fn rings_mut(&mut self) -> &mut DraftRings {
        &mut self.rings
    }

    /// The block pass, writable: the gates' way to stage a row of their own
    /// into it. A replay or an eager pass then runs over what is staged,
    /// until the owner's next [`DraftLayers::stage_block`] writes it again.
    pub fn pass_mut(&mut self) -> &mut BlockPass {
        &mut self.pass
    }

    /// The rule the next staged block follows.
    #[must_use]
    pub fn rule(&self) -> Rule {
        self.rule
    }

    /// Follow `rule` from the next staged block on. Only the pass of
    /// [`Rule::Reference`] is a graph's: see [`DraftLayers::replayable`].
    pub fn set_rule(&mut self, rule: Rule) {
        self.rule = rule;
    }

    /// Stage the next blocks at `oracle`'s positions instead of the ones the
    /// owner passes, or at the owner's again with `None`: the gates' way to
    /// feed an oracle set's own rope positions and ring rows.
    pub fn set_oracle(&mut self, oracle: Option<OracleBlock>) {
        self.oracle = oracle;
    }

    /// The oracle positions in force, if any.
    #[must_use]
    pub fn oracle(&self) -> Option<OracleBlock> {
        self.oracle
    }

    /// Kernel launches one block pass of `m` rows makes: the pass's, without
    /// the head chain the owner holds.
    #[must_use]
    pub fn launches(&self, m: usize) -> usize {
        self.pass.launches(m)
    }

    /// Kernel launches one append of `n` rows makes: per group, per layer,
    /// `attn_kv` and the append; fc and the norm are the owner's.
    #[must_use]
    pub fn append_launches(&self, n: usize) -> usize {
        self.append.launches(n)
    }
}

impl DraftLayers for V41Layers {
    const MAX_ROWS: usize = MAX_WIDTH;
    const APPEND_GROUP: usize = GROUP;

    fn hidden(&self) -> usize {
        self.w.hp().n_embd
    }

    fn stage_kv(&mut self, gpu: &Gpu, first: u32, n: usize) -> Result<(), GpuError> {
        self.append.stage(gpu.stream(), first, n)
    }

    fn append_kv(&mut self, gpu: &Gpu, x: &DeviceBuffer<f32>, n: usize) -> Result<(), GpuError> {
        self.append.enqueue(gpu, &self.w, &mut self.rings, x, n)
    }

    fn stage_block(
        &mut self,
        gpu: &Gpu,
        last: u32,
        pos: u32,
        m: usize,
        ctx: usize,
    ) -> Result<(), GpuError> {
        let (first_pos, ring_rows) = self
            .oracle
            .map_or((pos, ctx), |o| (o.first_pos, o.ring_rows));
        let row = embedding_row(&self.target, self.w.hp().n_embd, last)?;
        self.pass.stage(
            gpu.stream(),
            &BlockInput {
                row,
                width: m,
                first_pos,
                ring_rows,
                rule: self.rule,
            },
        )
    }

    fn enqueue_block(
        &mut self,
        gpu: &Gpu,
        x: &mut DeviceBuffer<f32>,
        m: usize,
    ) -> Result<(), GpuError> {
        let n_embd = self.w.hp().n_embd;
        let staged = self.pass.width();
        if m != staged && !(staged == 0 && capturing(gpu.stream())?) {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("a block pass of {m} rows after a stage of {staged}"),
            });
        }
        if x.len() < m * n_embd {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("{m} rows of {n_embd} in a buffer of {} values", x.len()),
            });
        }
        self.pass.enqueue_layers(gpu, &self.w, &self.rings, m, x)
    }

    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        self.rings.reset(gpu.stream())?;
        gpu.stream().synchronize()?;
        Ok(())
    }

    /// The captures are the pass of [`Rule::Reference`]: its clamp limits are
    /// launch scalars, so a block staged under another rule runs eager.
    fn replayable(&self) -> Result<(), GpuError> {
        if self.rule == Rule::Reference {
            return Ok(());
        }
        Err(GpuError::State {
            what: WHAT,
            missing: "the reference rule for a graph pass (ik's rule runs eager)",
        })
    }

    /// The rings are the draft's window whatever `ctx` is, so the bytes do
    /// not depend on it.
    fn card_bytes(&self, _ctx: usize) -> u64 {
        self.bytes
    }
}
