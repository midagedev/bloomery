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

use bloomery_gpu::q8f32::{GemvOut, Q8_0GemvMcolArgs};
use bloomery_gpu::{DeviceTensor, Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::dspark::{DraftHparams, names};

use super::load::DraftWeights;
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

/// One group's buffers: up to [`GROUP`] positions.
struct Group {
    /// The feature rows, token-major.
    feat: DeviceBuffer<f32>,
    /// `fc · feat`, token-major.
    fc: DeviceBuffer<f32>,
    /// `main_x`, token-major.
    main: DeviceBuffer<f32>,
    /// Per layer, `attn_kv · main_x` and the normed, turned row.
    kv: Vec<DeviceBuffer<f32>>,
    out: Vec<DeviceBuffer<f32>>,
    /// Each row's ring slot (the kernel takes it modulo the window).
    slots: DeviceBuffer<u32>,
    /// Each row's rope table, [`RopeTable::push`]'s layout.
    cs: DeviceBuffer<f32>,
    /// Rows the last [`KvAppend::stage`] wrote.
    rows: usize,
}

/// The feature-to-KV graph's buffers and constants. See the module comment.
pub struct KvAppend {
    groups: Vec<Group>,
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
                    feat: DeviceBuffer::zeroed(s, GROUP * features)?,
                    fc: DeviceBuffer::zeroed(s, GROUP * hp.n_embd)?,
                    main: DeviceBuffer::zeroed(s, GROUP * hp.n_embd)?,
                    kv: per_layer(hp.head_dim)?,
                    out: per_layer(hp.head_dim)?,
                    slots: DeviceBuffer::zeroed(s, GROUP)?,
                    cs: DeviceBuffer::zeroed(s, GROUP * hp.rope_dims)?,
                    rows: 0,
                })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        Ok(KvAppend {
            groups,
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

    /// Device bytes of the buffers.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.groups
            .iter()
            .map(|g| {
                g.feat.num_bytes()
                    + g.fc.num_bytes()
                    + g.main.num_bytes()
                    + g.kv
                        .iter()
                        .chain(&g.out)
                        .map(DeviceBuffer::num_bytes)
                        .sum::<usize>()
                    + g.slots.num_bytes()
                    + g.cs.num_bytes()
            })
            .sum()
    }

    /// Write the inputs of `n = feats.len() / features()` positions: row `i`
    /// takes ring slot `first_slot + i` and rope position `first_pos + i`.
    /// Synchronizing copies on the engine stream; never inside a capture.
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
        for (gi, g) in self.groups.iter_mut().enumerate() {
            let r0 = gi * GROUP;
            g.rows = n.saturating_sub(r0).min(GROUP);
            if g.rows == 0 {
                continue;
            }
            g.feat.copy_from_host(
                stream,
                &padded(
                    &feats[r0 * self.features..(r0 + g.rows) * self.features],
                    GROUP * self.features,
                ),
            )?;
            let mut cs = Vec::with_capacity(GROUP * self.rope_dims);
            let mut slots = vec![0u32; GROUP];
            for (i, slot) in slots.iter_mut().take(g.rows).enumerate() {
                let off = (r0 + i) as u32;
                self.table
                    .push(first_pos + off, Direction::Forward, &mut cs);
                *slot = first_slot + off;
            }
            cs.resize(GROUP * self.rope_dims, 0.0);
            g.cs.copy_from_host(stream, &cs)?;
            g.slots.copy_from_host(stream, &slots)?;
        }
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
        for g in self.groups.iter_mut().filter(|g| g.rows > 0) {
            gpu.q8f32().enqueue_q8_0_gemv_mcol(
                s,
                Q8_0GemvMcolArgs {
                    qs,
                    d,
                    x: &g.feat,
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
        for (l, ring) in rings.rings.iter_mut().enumerate() {
            let (qs, d) = w.q8(&names::attn_kv(l), self.n_embd, self.head_dim)?;
            let gain = w.gain(&names::attn_kv_a_norm(l), self.head_dim)?;
            for g in self.groups.iter_mut().filter(|g| g.rows > 0) {
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
                        cs: &g.cs,
                        pos: &g.slots,
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
