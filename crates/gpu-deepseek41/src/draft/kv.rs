//! The draft's feature-to-KV graph: committed target positions into every
//! draft layer's window ring.
//!
//! For `n` committed positions (`n <= window`, [`KvAppend::capacity`]), in
//! row groups of at most [`GROUP`] positions, one launch each:
//!
//! 1. `fc` (Q8_0, `features → n_embd`) over the group's feature rows,
//!    token-major (`q8_0_gemv_mcol`);
//! 2. `rms_norm` with `enc.output_norm` — the draft's `main_x`;
//! 3. per layer, `attn_kv` (Q8_0, `n_embd → head_dim`) over `main_x`,
//!    token-major, then `ds41_kv_norm_rope_append`: the norm with
//!    `attn_kv_a_norm`, the plain tail rope at each row's position, the row
//!    in f32 and its f16 in ring slot `slot % window`.
//!
//! A row's slot and its rope position are separate inputs: the engine gives
//! both the target position (slot `p % window`), and the gate feeds the
//! oracle set's own pair. Every gemv column is the `m = 1` launch of that
//! column bit for bit (`q8f32`'s contract), so how rows are grouped changes
//! no bit.
//!
//! Every group's inputs — feature rows, rope tables, slots — sit in one
//! pinned image, copied to the card in one transfer by [`KvAppend::stage`]
//! ([`Inbox`]); the launches read only device memory, so an append of up
//! to [`GROUP`] rows captures ([`KvAppend::capture`]) and replays after any
//! stage of that many rows.

use bloomery_gpu::q8f32::{GemvOut, Q8_0GemvMcolArgs};
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, Graph};
use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::dspark::{DraftHparams, names};

use super::load::DraftWeights;
use super::stage::{Inbox, put_f32};
use crate::rope::{Direction, KvAppendArgs, RopeKernels, RopeSpec, RopeTable};

const WHAT: &str = "draft::kv";

/// Positions one group's launches carry: the Q8_0 gemv's column limit.
pub const GROUP: usize = 8;

/// Every draft layer's raw window ring: `window` rows of `head_dim` f16,
/// the row of position `p` in slot `p % window`.
pub struct DraftRings {
    rings: Vec<DeviceTensor<u16>>,
    /// The append's shadow: no rows, so nothing is written beside the rings.
    none: DeviceTensor<u16>,
}

impl DraftRings {
    /// Zeroed rings for `hp`'s layers. Load-time only.
    pub fn new(stream: &CudaStream, hp: &DraftHparams) -> Result<DraftRings, GpuError> {
        let rings = (0..hp.n_layer)
            .map(|_| DeviceTensor::zeroed(stream, hp.window, hp.head_dim))
            .collect::<Result<Vec<_>, _>>()?;
        let none = DeviceTensor::zeroed(stream, 0, hp.head_dim)?;
        Ok(DraftRings { rings, none })
    }

    /// Layer `l`'s ring.
    #[must_use]
    pub fn ring(&self, l: usize) -> Option<&DeviceTensor<u16>> {
        self.rings.get(l)
    }

    /// Layer `l`'s ring, writable: the gate's way to seat an oracle's rows.
    pub fn ring_mut(&mut self, l: usize) -> Option<&mut DeviceTensor<u16>> {
        self.rings.get_mut(l)
    }

    /// Device bytes of every ring.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.rings.iter().map(|r| r.buf().num_bytes()).sum()
    }

    /// Zero every ring. Asynchronous.
    pub fn reset(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        for r in &mut self.rings {
            r.buf_mut().zero_async(stream)?;
        }
        Ok(())
    }
}

/// One group's buffers: up to [`GROUP`] positions. Its inputs are in the
/// append's image ([`Layout`]).
struct Group {
    /// `fc · feat`, token-major.
    fc: DeviceBuffer<f32>,
    /// `main_x`, token-major.
    main: DeviceBuffer<f32>,
    /// Per layer, `attn_kv · main_x` and the normed, turned row.
    kv: Vec<DeviceBuffer<f32>>,
    out: Vec<DeviceBuffer<f32>>,
    /// Rows the last [`KvAppend::stage`] wrote.
    rows: usize,
}

/// Where group `g`'s inputs sit in the append's image, in words: from
/// `g · words`, each row's ring slot (the kernel takes it modulo the
/// window), each row's rope table ([`RopeTable::push`]'s layout) and the
/// feature rows (token-major), room for [`GROUP`] rows of each. The feature
/// rows come last, so the copy of an append stops after its last row
/// ([`Layout::used`]).
#[derive(Clone, Copy)]
struct Layout {
    slots: usize,
    cs: usize,
    feat: usize,
    features: usize,
    words: usize,
}

impl Layout {
    fn of(features: usize, rope_dims: usize) -> Layout {
        let slots = 0;
        let cs = slots + GROUP;
        let feat = cs + GROUP * rope_dims;
        Layout {
            slots,
            cs,
            feat,
            features,
            words: feat + GROUP * features,
        }
    }

    /// Words of the image an append of `n` rows reads: every full group,
    /// then the last group up to its last feature row.
    fn used(&self, n: usize) -> usize {
        let (full, rest) = (n / GROUP, n % GROUP);
        full * self.words
            + if rest == 0 {
                0
            } else {
                self.feat + rest * self.features
            }
    }
}

/// The feature-to-KV graph's buffers and constants. See the module comment.
pub struct KvAppend {
    groups: Vec<Group>,
    /// Every group's inputs, one copy an append.
    inbox: Inbox,
    layout: Layout,
    /// The rope table of the row being staged; kept to stage without
    /// allocating.
    cs: Vec<f32>,
    table: RopeTable,
    rope: RopeKernels,
    features: usize,
    n_embd: usize,
    head_dim: usize,
    rope_dims: usize,
    eps: f32,
    n_layer: usize,
    /// Rows the last [`KvAppend::stage`] wrote.
    staged: usize,
}

impl KvAppend {
    /// Buffers for up to `hp.window` positions per append. Load-time only.
    pub fn new(gpu: &Gpu, hp: &DraftHparams) -> Result<KvAppend, GpuError> {
        let s = gpu.stream();
        let features = hp.target_layers.len() * hp.n_embd;
        let n_groups = hp.window.div_ceil(GROUP);
        let per_layer = |w: usize| {
            (0..hp.n_layer)
                .map(|_| DeviceBuffer::zeroed(s, GROUP * w))
                .collect::<Result<Vec<_>, _>>()
        };
        let groups = (0..n_groups)
            .map(|_| {
                Ok(Group {
                    fc: DeviceBuffer::zeroed(s, GROUP * hp.n_embd)?,
                    main: DeviceBuffer::zeroed(s, GROUP * hp.n_embd)?,
                    kv: per_layer(hp.head_dim)?,
                    out: per_layer(hp.head_dim)?,
                    rows: 0,
                })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        let layout = Layout::of(features, hp.rope_dims);
        Ok(KvAppend {
            groups,
            inbox: Inbox::new(gpu, n_groups * layout.words)?,
            layout,
            cs: Vec::with_capacity(GROUP * hp.rope_dims),
            table: RopeTable::new(&RopeSpec::window(hp.rope_base, hp.rope_dims))?,
            rope: RopeKernels::load(gpu.context())?,
            features,
            n_embd: hp.n_embd,
            head_dim: hp.head_dim,
            rope_dims: hp.rope_dims,
            eps: hp.rms_eps,
            n_layer: hp.n_layer,
            staged: 0,
        })
    }

    /// Positions one append takes: the ring's window.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.groups.len() * GROUP
    }

    /// Values of one feature row.
    #[must_use]
    pub fn features(&self) -> usize {
        self.features
    }

    /// Device bytes of the buffers, the image's device side included.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        4 * self.groups.len() * self.layout.words
            + self
                .groups
                .iter()
                .map(|g| {
                    g.fc.num_bytes()
                        + g.main.num_bytes()
                        + g.kv
                            .iter()
                            .chain(&g.out)
                            .map(DeviceBuffer::num_bytes)
                            .sum::<usize>()
                })
                .sum::<usize>()
    }

    /// Write the inputs of `n = feats.len() / features()` positions into the
    /// image — row `i` takes ring slot `first_slot + i` and rope position
    /// `first_pos + i` — and enqueue its one copy to the card on `stream`.
    /// Asynchronous, except that it waits for the previous append's copy to
    /// have read the image; never inside a capture.
    pub fn stage(
        &mut self,
        stream: &CudaStream,
        feats: &[f32],
        first_slot: u32,
        first_pos: u32,
    ) -> Result<usize, GpuError> {
        let n = feats.len() / self.features;
        if n == 0 || n * self.features != feats.len() || n > self.capacity() {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "{} values are not 1..={} rows of {}",
                    feats.len(),
                    self.capacity(),
                    self.features
                ),
            });
        }
        let (lay, f) = (self.layout, self.features);
        let host = self.inbox.host_mut()?;
        for (gi, g) in self.groups.iter_mut().enumerate() {
            let r0 = gi * GROUP;
            g.rows = n.saturating_sub(r0).min(GROUP);
            if g.rows == 0 {
                continue;
            }
            let img = &mut host[gi * lay.words..(gi + 1) * lay.words];
            let (slots, rest) = img.split_at_mut(lay.cs);
            let (cs, feat) = rest.split_at_mut(lay.feat - lay.cs);
            // Only the rows staged are copied and read.
            put_f32(&mut feat[..g.rows * f], &feats[r0 * f..(r0 + g.rows) * f]);
            self.cs.clear();
            for (i, slot) in slots.iter_mut().enumerate() {
                if i >= g.rows {
                    *slot = 0;
                    continue;
                }
                let off = (r0 + i) as u32;
                self.table
                    .push(first_pos + off, Direction::Forward, &mut self.cs);
                *slot = first_slot + off;
            }
            put_f32(cs, &self.cs);
        }
        self.inbox.upload(stream, lay.used(n))?;
        self.staged = n;
        Ok(n)
    }

    /// Write `main_x` of the staged rows directly (the gate's way to run the
    /// layers alone): `main.len()` must be the staged rows times `n_embd`.
    /// Synchronizing copies; never inside a capture.
    pub fn set_main(&mut self, stream: &CudaStream, main: &[f32]) -> Result<(), GpuError> {
        if main.len() != self.staged * self.n_embd {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "{} values for {} staged rows of {}",
                    main.len(),
                    self.staged,
                    self.n_embd
                ),
            });
        }
        for (gi, g) in self.groups.iter_mut().filter(|g| g.rows > 0).enumerate() {
            let r0 = gi * GROUP * self.n_embd;
            g.main.copy_from_host(
                stream,
                &padded(&main[r0..r0 + g.rows * self.n_embd], GROUP * self.n_embd),
            )?;
        }
        Ok(())
    }

    /// Enqueue the whole graph over the staged rows: `fc`, the norm, and
    /// every layer's append into `rings`. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue(
        &mut self,
        gpu: &Gpu,
        w: &DraftWeights,
        rings: &mut DraftRings,
    ) -> Result<(), GpuError> {
        self.enqueue_fc(gpu, w)?;
        self.enqueue_layers(gpu, w, rings)
    }

    /// Enqueue `fc` and the norm over the staged rows into `main_x`.
    pub fn enqueue_fc(&mut self, gpu: &Gpu, w: &DraftWeights) -> Result<(), GpuError> {
        let s = gpu.stream();
        let (qs, d) = w.q8(&names::fc(), self.features, self.n_embd)?;
        let gain = w.gain(&names::enc_output_norm(), self.n_embd)?;
        let lay = self.layout;
        for (gi, g) in self
            .groups
            .iter_mut()
            .enumerate()
            .filter(|(_, g)| g.rows > 0)
        {
            let feat = self
                .inbox
                .f32s(gi * lay.words + lay.feat, GROUP * self.features)?;
            gpu.q8f32().enqueue_q8_0_gemv_mcol(
                s,
                Q8_0GemvMcolArgs {
                    qs,
                    d,
                    x: &feat,
                    m: g.rows,
                    out: GemvOut::TokenMajor,
                    y: &mut g.fc,
                },
            )?;
            gpu.elem().enqueue_rms_norm(
                s,
                &g.fc,
                gain,
                self.eps,
                self.n_embd,
                g.rows,
                &mut g.main,
            )?;
        }
        Ok(())
    }

    /// Enqueue every layer's `attn_kv` and append over the staged rows'
    /// `main_x` into `rings`.
    pub fn enqueue_layers(
        &mut self,
        gpu: &Gpu,
        w: &DraftWeights,
        rings: &mut DraftRings,
    ) -> Result<(), GpuError> {
        let s = gpu.stream();
        if rings.rings.len() != self.n_layer {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("{} rings for {} layers", rings.rings.len(), self.n_layer),
            });
        }
        let lay = self.layout;
        for (l, ring) in rings.rings.iter_mut().enumerate() {
            let (qs, d) = w.q8(&names::attn_kv(l), self.n_embd, self.head_dim)?;
            let gain = w.gain(&names::attn_kv_a_norm(l), self.head_dim)?;
            for (gi, g) in self
                .groups
                .iter_mut()
                .enumerate()
                .filter(|(_, g)| g.rows > 0)
            {
                let at = gi * lay.words;
                let cs = self.inbox.f32s(at + lay.cs, GROUP * self.rope_dims)?;
                let slots = self.inbox.u32s(at + lay.slots, GROUP)?;
                gpu.q8f32().enqueue_q8_0_gemv_mcol(
                    s,
                    Q8_0GemvMcolArgs {
                        qs,
                        d,
                        x: &g.main,
                        m: g.rows,
                        out: GemvOut::TokenMajor,
                        y: &mut g.kv[l],
                    },
                )?;
                self.rope.enqueue_kv_norm_rope_append(
                    s,
                    KvAppendArgs {
                        kv: &g.kv[l],
                        gain,
                        cs: &cs,
                        pos: &slots,
                        eps: self.eps,
                        n_dims: self.rope_dims,
                        m: g.rows,
                        out: &mut g.out[l],
                        cache: ring,
                        shadow: &mut rings.none,
                    },
                )?;
            }
        }
        Ok(())
    }

    /// Capture the append of `n` rows (`1..=GROUP`, one group) into `rings`:
    /// [`KvAppend::enqueue`] with the staging left out, so a replay after
    /// any [`KvAppend::stage`] of `n` rows is that append. The graph names
    /// these buffers, `w` and `rings`: the caller keeps all three alive and
    /// in place while it holds the graph. Leaves the append staged as `n`
    /// rows of whatever the image holds. Load-time only.
    pub fn capture(
        &mut self,
        gpu: &Gpu,
        w: &DraftWeights,
        rings: &mut DraftRings,
        n: usize,
    ) -> Result<Graph, GpuError> {
        if !(1..=GROUP).contains(&n) {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("a capture of {n} rows; 1..={GROUP}"),
            });
        }
        for (gi, g) in self.groups.iter_mut().enumerate() {
            g.rows = if gi == 0 { n } else { 0 };
        }
        self.staged = n;
        gpu.capture(|_| self.enqueue(gpu, w, rings))
    }

    /// Rows the last [`KvAppend::stage`] wrote.
    #[must_use]
    pub fn staged(&self) -> usize {
        self.staged
    }

    /// Kernel launches one [`KvAppend::enqueue`] of `n` rows makes.
    #[must_use]
    pub fn launches(&self, n: usize) -> usize {
        n.div_ceil(GROUP) * (2 + 2 * self.n_layer)
    }

    /// The staged rows' `fc · feat`, `main_x`, and per layer the `attn_kv`
    /// rows and the normed, turned rows, token-major. Blocking reads; gate
    /// use.
    pub fn to_host(&self, stream: &CudaStream) -> Result<KvReadback, GpuError> {
        let mut r = KvReadback {
            fc: Vec::new(),
            main: Vec::new(),
            kv: vec![Vec::new(); self.n_layer],
            out: vec![Vec::new(); self.n_layer],
        };
        for g in self.groups.iter().filter(|g| g.rows > 0) {
            r.fc.extend_from_slice(&g.fc.to_host_vec(stream)?[..g.rows * self.n_embd]);
            r.main
                .extend_from_slice(&g.main.to_host_vec(stream)?[..g.rows * self.n_embd]);
            for l in 0..self.n_layer {
                r.kv[l].extend_from_slice(&g.kv[l].to_host_vec(stream)?[..g.rows * self.head_dim]);
                r.out[l]
                    .extend_from_slice(&g.out[l].to_host_vec(stream)?[..g.rows * self.head_dim]);
            }
        }
        Ok(r)
    }
}

/// [`KvAppend::to_host`]'s buffers.
pub struct KvReadback {
    pub fc: Vec<f32>,
    pub main: Vec<f32>,
    pub kv: Vec<Vec<f32>>,
    pub out: Vec<Vec<f32>>,
}

/// `v` followed by zeros to `len` values.
fn padded(v: &[f32], len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    out.extend_from_slice(v);
    out.resize(len, 0.0);
    out
}
