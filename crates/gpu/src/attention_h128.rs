//! Loaded head-128 GQA: independent rotation and readable-window policies over f16 KV stores.
//!
//! KV retention belongs to the runtime's Store policy. This binding owns no logical cursor:
//! the caller supplies absolute positions and causal counts, and exchanges only cache planes
//! between slots. Ring appends are cut to `capacity - window + 1` rows before attention, so
//! a later row never overwrites a key still readable by the first row of the chunk.
//!
//! The flash takes eight or four query heads a key head, with or without per-head sinks. The
//! rope table the caller passes to [`Binding::enqueue`] is built from the layer's own `Rope`,
//! factors included when the layer has them: the kernels read the table, whatever made it.

use crate::attention_h128_ring;
use crate::flash_gqa::{HEAD, window_segments};
use crate::flash_gqa_h128_swa::{H128SwaArgs, H128SwaKernels, served_group};
use crate::gemm::{GEMM_MAX_SLOTS, Gemm32Kernels, GemmInput, GemmKernels, GemmRoute};
use crate::qk_rms::{QkRmsArgs, QkRmsKernels};
use crate::rope_neox::{NeoxArgs, RopeNeoxKernels};
use crate::rope_table::RopeRows;
use crate::site::{self, SiteTy, SiteWeight, WideActs, WideForms, WideKernels};
use crate::weights::Weights;
use crate::{Gpu, GpuError};
use cuda_core::DeviceBuffer;
use gguf::Split;
use model::arch::models::{Gqa, LayerSpec, Mixer, OutGate, RopeMode};
use model::arch::names;
use model::placement::kernels::attention::{Forms, Head128Scratch};

const WHAT: &str = "attention_h128";
const SITES: &[SiteTy] = &[
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Iq4Xs,
    SiteTy::Q5_0,
    SiteTy::Q8_0,
    SiteTy::F32,
    SiteTy::Mxfp8,
    SiteTy::Fp8B128,
    SiteTy::Bf16,
];

struct Projection {
    name: String,
    ty: SiteTy,
    rows: usize,
}

impl Projection {
    fn load(
        file: &Split,
        w: &Weights,
        name: String,
        rows: usize,
        cols: usize,
    ) -> Result<Self, GpuError> {
        let ty = site::file_site(file, WHAT, &name, (rows, cols), SITES)?;
        site::site(w, WHAT, &name, (rows, cols), ty)?;
        Ok(Self { name, ty, rows })
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "one typed projection's borrowed launch state (rust-quality R8)"
    )]
    fn enqueue(
        &self,
        gpu: &Gpu,
        w: &Weights,
        k: &WideKernels<'_>,
        a: &WideActs,
        x: &DeviceBuffer<f32>,
        route: &GemmRoute,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        site::gemm(
            gpu,
            k,
            (self.ty, w, SiteWeight::Resident(&self.name)),
            self.rows,
            &a.input(x),
            (route, GemmInput::PerSlot),
            m,
            y,
        )
    }
}

/// A layer's loaded sites, gains, optional sinks, and independent attention policies.
pub struct Binding {
    layer: usize,
    hidden: usize,
    heads: usize,
    kv: usize,
    ctx: usize,
    capacity: usize,
    window: usize,
    rotate: bool,
    eps: f32,
    norm: String,
    qnorm: String,
    knorm: String,
    sinks: Option<String>,
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
}

fn vector(file: &Split, w: &Weights, name: String, len: usize) -> Result<String, GpuError> {
    let (_, t) = file
        .find(&name)
        .ok_or_else(|| GpuError::tensor(WHAT, &name, "a vector"))?;
    if t.dims != [len as u64] {
        return Err(GpuError::shape(
            WHAT,
            format!("{name} has shape {:?}; need [{len}]", t.dims),
        ));
    }
    let mut values = vec![0.0; len];
    gguf::quant::dequant_row(t.ty, file.data(&name)?, &mut values)
        .map_err(::model::ModelError::from)?;
    if let Some(i) = values.iter().position(|x| !x.is_finite()) {
        return Err(GpuError::shape(WHAT, format!("{name}[{i}] is not finite")));
    }
    site::vector(w, WHAT, name, len)
}

/// What the kernels run of one layer's description, once every field they do not run is
/// refused by name.
#[derive(Debug, PartialEq, Eq)]
struct Admitted {
    heads: usize,
    kv: usize,
    window: usize,
    capacity: usize,
    rotate: bool,
    sinks: bool,
}

fn admit(layer: usize, g: &Gqa, hidden: usize, eps: f32, ctx: usize) -> Result<Admitted, GpuError> {
    let heads = g.heads as usize;
    let kv = g.kv_heads as usize;
    // The table is the caller's, so a layer with rope factors is the same rotation launch
    // as one without: only the values in the table differ.
    let rotate = match g.rope.mode {
        RopeMode::None if g.rope.dims == 0 && g.rope.yarn.is_none() && !g.rope.factors => false,
        RopeMode::Neox
            if g.rope.dims as usize == HEAD
                && g.rope.yarn.is_none()
                && g.rope.base.is_finite()
                && g.rope.base > 0.0 =>
        {
            true
        }
        _ => {
            return Err(GpuError::shape(
                WHAT,
                "need NoPE or whole-head NeoX without YaRN",
            ));
        }
    };
    if g.out_gate != OutGate::None {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "layer {layer} has output gate {:?}; the gate is the output-gate kernel's launch, \
                 which this binding does not run",
                g.out_gate
            ),
        ));
    }
    let window = g.window.map_or(0, |n| n as usize);
    if hidden == 0
        || ctx == 0
        || u32::try_from(ctx).is_err()
        || kv == 0
        || served_group(heads, kv).is_none()
        || g.head_dim as usize != HEAD
        || g.value_dim as usize != HEAD
        || !g.qk_norm
        || g.o_norm
        || g.select.is_some()
        || g.value_scale.is_some()
        || !eps.is_finite()
        || eps <= 0.0
        || g.window == Some(0)
    {
        return Err(GpuError::shape(
            WHAT,
            format!("unsupported head128 layer {layer}: {g:?}"),
        ));
    }
    // The same capacity owner sizes the plan and the physical allocation.
    let capacity = runtime::stores::Rows::Ring.count(g.window.map(u64::from), ctx as u64) as usize;
    Ok(Admitted {
        heads,
        kv,
        window,
        capacity,
        rotate,
        sinks: g.sinks,
    })
}

impl Binding {
    /// Bind actual file and resident formats once. No architecture-name dispatch is involved.
    #[allow(
        clippy::too_many_arguments,
        reason = "the loaded layer and its geometry (rust-quality R8)"
    )]
    pub fn load(
        file: &Split,
        w: &Weights,
        layer: usize,
        spec: &LayerSpec,
        hidden: usize,
        eps: f32,
        ctx: usize,
    ) -> Result<Self, GpuError> {
        let Mixer::Gqa(g) = &spec.mixer else {
            return Err(GpuError::shape(WHAT, "a GQA layer is required"));
        };
        let a = admit(layer, g, hidden, eps, ctx)?;
        let qwidth = a
            .heads
            .checked_mul(HEAD)
            .ok_or_else(|| GpuError::shape(WHAT, "query width overflow"))?;
        let kvwidth =
            a.kv.checked_mul(HEAD)
                .ok_or_else(|| GpuError::shape(WHAT, "KV width overflow"))?;
        Ok(Self {
            layer,
            hidden,
            heads: a.heads,
            kv: a.kv,
            ctx,
            capacity: a.capacity,
            window: a.window,
            rotate: a.rotate,
            eps,
            norm: vector(file, w, names::attn_norm(layer), hidden)?,
            qnorm: vector(file, w, names::attn_q_norm(layer), HEAD)?,
            knorm: vector(file, w, names::attn_k_norm(layer), HEAD)?,
            sinks: a
                .sinks
                .then(|| vector(file, w, names::attn_sinks(layer), a.heads))
                .transpose()?,
            q: Projection::load(file, w, names::attn_q(layer), qwidth, hidden)?,
            k: Projection::load(file, w, names::attn_k(layer), kvwidth, hidden)?,
            v: Projection::load(file, w, names::attn_v(layer), kvwidth, hidden)?,
            o: Projection::load(file, w, names::attn_output(layer), hidden, qwidth)?,
        })
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    #[must_use]
    pub fn window(&self) -> usize {
        self.window
    }
    /// Maximum append-then-attend chunk. Plane rows have no overwrite hazard.
    #[must_use]
    pub fn chunk_rows(&self) -> usize {
        if self.capacity == self.ctx {
            self.ctx
        } else {
            self.capacity - self.window + 1
        }
    }

    /// Full attention sub-layer: RMS, Q/K/V, head RMS and optional NeoX, append,
    /// attention (sink-folded when the layer has sinks), O projection, residual. Every
    /// scratch cell read is written.
    ///
    /// # Safety
    /// `pos[0..m]` are consecutive absolute positions and `n_keys[t] == pos[t] + 1`.
    /// Cache rows for earlier readable positions are initialized. The runtime Store owner
    /// has established retention before a rewind; a wrapped ring cannot restore overwritten keys.
    #[allow(
        clippy::too_many_arguments,
        reason = "borrowed layer launch state (rust-quality R8)"
    )]
    pub unsafe fn enqueue(
        &self,
        gpu: &Gpu,
        w: &Weights,
        k: &Kernels,
        s: &mut Scratch,
        cache: &mut Cache,
        x: &DeviceBuffer<f32>,
        x1: &mut DeviceBuffer<f32>,
        pos: &DeviceBuffer<u32>,
        n_keys: &DeviceBuffer<u32>,
        m: usize,
        rope: &RopeRows,
    ) -> Result<(), GpuError> {
        if m == 0
            || m > s.rows
            || m > self.chunk_rows()
            || s.hidden != self.hidden
            || s.heads != self.heads
            || s.kv != self.kv
            || cache.capacity != self.capacity
            || cache.kv != self.kv
            || x.len() < m * self.hidden
            || x1.len() < m * self.hidden
            || pos.len() < m
            || n_keys.len() < m
            || (self.rotate && (rope.width != HEAD || rope.rows() < self.ctx))
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {} incompatible scratch/cache/rows: m={m}",
                    self.layer
                ),
            ));
        }
        let stream = gpu.stream();
        let fault = gpu.layer_sink(self.layer)?;
        let wide = WideKernels {
            gemm: &k.gemm,
            g32: Some(&k.g32),
        };
        gpu.elem().enqueue_rms_norm(
            stream,
            x,
            w.f32_buf(WHAT, &self.norm)?,
            self.eps,
            self.hidden,
            m,
            &mut s.xn,
        )?;
        k.gemm.enqueue_route_dense(stream, m, &mut s.route, fault)?;
        s.input.quantize(
            gpu,
            &wide,
            fault,
            &s.xn,
            m,
            &[self.q.ty, self.k.ty, self.v.ty],
        )?;
        self.q
            .enqueue(gpu, w, &wide, &s.input, &s.xn, &s.route, m, &mut s.q)?;
        self.k
            .enqueue(gpu, w, &wide, &s.input, &s.xn, &s.route, m, &mut s.k)?;
        self.v
            .enqueue(gpu, w, &wide, &s.input, &s.xn, &s.route, m, &mut s.v)?;
        let gq = w.f32_buf(WHAT, &self.qnorm)?;
        let gk = w.f32_buf(WHAT, &self.knorm)?;
        let sinks = self
            .sinks
            .as_ref()
            .map(|name| w.f32_buf(WHAT, name))
            .transpose()?;
        // SAFETY: positions and initialized retained rows are the caller's contract;
        // chunk_rows prevents wrapped appends from destroying an earlier row's window.
        unsafe {
            k.enqueue_append(
                stream,
                NeoxArgs {
                    q: &mut s.q,
                    k: &mut s.k,
                    v: &s.v,
                    gq,
                    gk,
                    table: &rope.table,
                    pos,
                    eps: self.eps,
                    n_head: self.heads,
                    n_kv: self.kv,
                    ctx: self.ctx,
                    m,
                    fault,
                    cache_k: &mut cache.k,
                    cache_v: &mut cache.v,
                },
                self.capacity,
                self.rotate,
            )?;
        }
        k.enqueue_attention(
            stream,
            H128SwaArgs {
                q: &s.q,
                kc: &cache.k,
                vc: &cache.v,
                n_keys,
                scale: 1.0 / (HEAD as f32).sqrt(),
                n_kv: self.kv,
                ctx: self.ctx,
                m,
                window: self.window,
                sinks,
                part_v: &mut s.part_v,
                part_ms: &mut s.part_ms,
                fault,
                y: &mut s.attended,
            },
            self.heads,
            self.capacity,
        )?;
        s.output
            .quantize(gpu, &wide, fault, &s.attended, m, &[self.o.ty])?;
        self.o.enqueue(
            gpu,
            w,
            &wide,
            &s.output,
            &s.attended,
            &s.route,
            m,
            &mut s.delta,
        )?;
        gpu.elem()
            .enqueue_add(stream, x, &s.delta, m * self.hidden, x1)
    }
}

/// Only the physical cache travels with a slot. Logical retention stays in runtime::Store.
pub struct Cache {
    pub k: DeviceBuffer<u16>,
    pub v: DeviceBuffer<u16>,
    capacity: usize,
    kv: usize,
}

impl Cache {
    pub fn new(gpu: &Gpu, binding: &Binding) -> Result<Self, GpuError> {
        let n = binding
            .kv
            .checked_mul(binding.capacity)
            .and_then(|n| n.checked_mul(HEAD))
            .ok_or_else(|| GpuError::shape(WHAT, "cache allocation overflow"))?;
        Ok(Self {
            k: DeviceBuffer::zeroed(gpu.stream(), n)?,
            v: DeviceBuffer::zeroed(gpu.stream(), n)?,
            capacity: binding.capacity,
            kv: binding.kv,
        })
    }
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.k.num_bytes() + self.v.num_bytes()
    }
}

/// Loaded launch modules. Their module handles allocate no device scratch.
pub struct Kernels {
    gemm: GemmKernels,
    g32: Gemm32Kernels,
    qk: QkRmsKernels,
    rope: RopeNeoxKernels,
    flash: H128SwaKernels,
    ring: attention_h128_ring::Kernels,
}

impl Kernels {
    pub fn load(gpu: &Gpu) -> Result<Self, GpuError> {
        Ok(Self {
            gemm: GemmKernels::load(gpu.context())?,
            g32: Gemm32Kernels::load(gpu.context())?,
            qk: QkRmsKernels::load(gpu.context())?,
            rope: RopeNeoxKernels::load(gpu.context())?,
            flash: H128SwaKernels::load(gpu.context())?,
            ring: attention_h128_ring::Kernels::load(gpu.context())?,
        })
    }
    #[must_use]
    pub fn bytes(&self) -> usize {
        0
    }

    /// Head RMS and append, with independent rotation and physical capacity.
    /// # Safety
    /// Valid absolute positions are distinct modulo `rows`.
    pub unsafe fn enqueue_append(
        &self,
        stream: &cuda_core::CudaStream,
        args: NeoxArgs<'_>,
        rows: usize,
        rotate: bool,
    ) -> Result<(), GpuError> {
        if rows != args.ctx {
            // SAFETY: this method's caller guarantees distinct physical rows.
            return unsafe { self.ring.append(stream, args, rows, rotate) };
        }
        if rotate {
            // SAFETY: distinct modulo the full plane means distinct plane positions.
            unsafe { self.rope.enqueue_head_norm_neox_append(stream, args) }
        } else {
            let NeoxArgs {
                q,
                k,
                v,
                gq,
                gk,
                pos,
                eps,
                n_head,
                n_kv,
                ctx,
                m,
                fault,
                cache_k,
                cache_v,
                ..
            } = args;
            self.qk.enqueue(
                stream,
                QkRmsArgs {
                    q,
                    k,
                    v,
                    gq,
                    gk,
                    pos,
                    eps,
                    n_head,
                    n_kv,
                    ctx,
                    m,
                    fault,
                    cache_k,
                    cache_v,
                },
            )
        }
    }

    pub fn enqueue_attention(
        &self,
        stream: &cuda_core::CudaStream,
        args: H128SwaArgs<'_>,
        heads: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        if rows == args.ctx {
            self.flash.enqueue(stream, args, heads)
        } else {
            self.ring.attention(stream, args, heads, rows)
        }
    }
}

/// Reused step/prompt storage. No allocation occurs during enqueue.
pub struct Scratch {
    rows: usize,
    hidden: usize,
    heads: usize,
    kv: usize,
    xn: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    attended: DeviceBuffer<f32>,
    delta: DeviceBuffer<f32>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
    input: WideActs,
    output: WideActs,
    route: GemmRoute,
}

fn layout(
    bindings: &[Binding],
    rows: usize,
) -> Result<(usize, usize, usize, usize, WideForms, WideForms), GpuError> {
    let first = bindings
        .first()
        .ok_or_else(|| GpuError::shape(WHAT, "at least one binding required"))?;
    if !(1..=GEMM_MAX_SLOTS).contains(&rows)
        || bindings
            .iter()
            .any(|b| b.hidden != first.hidden || b.heads != first.heads || b.kv != first.kv)
    {
        return Err(GpuError::shape(
            WHAT,
            format!("scratch needs 1..={GEMM_MAX_SLOTS} rows and equal layer widths"),
        ));
    }
    let mut input = WideForms::NONE;
    let mut output = WideForms::NONE;
    let mut segs = 1;
    for b in bindings {
        input = input.union(WideActs::wants_of(&[b.q.ty, b.k.ty, b.v.ty]));
        output = output.union(WideActs::wants(b.o.ty));
        segs = segs.max(window_segments(b.window));
    }
    // The f32 source buffers are already owned above; no duplicate form is needed.
    input.f32 = false;
    output.f32 = false;
    Ok((first.hidden, first.heads, first.kv, segs, input, output))
}

impl Scratch {
    pub fn new(gpu: &Gpu, bindings: &[Binding], rows: usize) -> Result<Self, GpuError> {
        let (hidden, heads, kv, segs, input, output) = layout(bindings, rows)?;
        let _ = Self::planned_bytes(bindings, rows)?;
        let stream = gpu.stream();
        Ok(Self {
            rows,
            hidden,
            heads,
            kv,
            xn: DeviceBuffer::zeroed(stream, rows * hidden)?,
            q: DeviceBuffer::zeroed(stream, rows * heads * HEAD)?,
            k: DeviceBuffer::zeroed(stream, rows * kv * HEAD)?,
            v: DeviceBuffer::zeroed(stream, rows * kv * HEAD)?,
            attended: DeviceBuffer::zeroed(stream, rows * heads * HEAD)?,
            delta: DeviceBuffer::zeroed(stream, rows * hidden)?,
            part_v: DeviceBuffer::zeroed(stream, rows * heads * segs * HEAD)?,
            part_ms: DeviceBuffer::zeroed(stream, rows * heads * segs * 2)?,
            input: WideActs::new(stream, input, rows, hidden)?,
            output: WideActs::new(stream, output, rows, heads * HEAD)?,
            route: GemmRoute::new(stream, rows, 1)?,
        })
    }
    /// Exact device allocation, including typed activation forms and the dense route table.
    pub fn planned_bytes(bindings: &[Binding], rows: usize) -> Result<usize, GpuError> {
        let (hidden, heads, kv, segs, input, output) = layout(bindings, rows)?;
        let shape = Head128Scratch {
            hidden: hidden as u64,
            query: (heads * HEAD) as u64,
            key_value: (kv * HEAD) as u64,
            heads: heads as u64,
            segments: segs as u64,
            input: Forms {
                q128: input.q128,
                q32: input.q32,
            },
            output: Forms {
                q128: output.q128,
                q32: output.q32,
            },
        };
        shape
            .bytes(rows as u64)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| GpuError::shape(WHAT, "scratch allocation overflow"))
    }
    #[must_use]
    pub fn bytes(&self) -> usize {
        [
            &self.xn,
            &self.q,
            &self.k,
            &self.v,
            &self.attended,
            &self.delta,
            &self.part_v,
            &self.part_ms,
        ]
        .iter()
        .map(|b| b.num_bytes())
        .sum::<usize>()
            + self.input.bytes()
            + self.output.bytes()
            + self.route.bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model::arch::models::Rope;

    const BASE: f32 = 1_000_000.0;

    fn neox(factors: bool) -> Rope {
        Rope {
            mode: RopeMode::Neox,
            dims: 128,
            base: BASE,
            yarn: None,
            factors,
        }
    }

    /// A layer of 16 query heads over 4 key heads of head 128: no sinks, no gate, q/k norms.
    fn layer(rope: Rope, window: Option<u32>) -> Gqa {
        Gqa {
            heads: 16,
            kv_heads: 4,
            head_dim: 128,
            value_dim: 128,
            rope,
            qk_norm: true,
            o_norm: false,
            out_gate: OutGate::None,
            select: None,
            window,
            sinks: false,
            value_scale: None,
        }
    }

    fn admitted(g: &Gqa) -> Result<Admitted, GpuError> {
        admit(3, g, 2048, 1e-5, 4096)
    }

    fn refusal(r: Result<Admitted, GpuError>) -> String {
        r.expect_err("a refusal").to_string()
    }

    /// A group of four runs, a sliding layer on the ring and a full layer on the plane, with
    /// no sinks.
    #[test]
    fn a_group_of_four_runs_sliding_and_full() {
        let sliding = admitted(&layer(neox(false), Some(256))).expect("sliding");
        assert_eq!((sliding.heads, sliding.kv, sliding.window), (16, 4, 256));
        assert_eq!(sliding.capacity, 768);
        assert!(sliding.rotate && !sliding.sinks);
        let full = admitted(&layer(neox(true), None)).expect("full");
        assert_eq!((full.window, full.capacity), (0, 4096));
        assert!(full.rotate && !full.sinks);
    }

    /// The group of eight with sinks keeps its admission, and the sinks are carried.
    #[test]
    fn a_group_of_eight_with_sinks_is_admitted() {
        let g = Gqa {
            heads: 32,
            kv_heads: 4,
            sinks: true,
            ..layer(neox(false), Some(512))
        };
        let a = admitted(&g).expect("eight");
        assert_eq!((a.heads, a.kv, a.sinks), (32, 4, true));
    }

    /// A group other than eight and four is refused by name, whatever else holds.
    #[test]
    fn another_group_is_refused() {
        for (heads, kv) in [(20, 4), (24, 4), (16, 8), (4, 4), (16, 3)] {
            let g = Gqa {
                heads,
                kv_heads: kv,
                ..layer(neox(false), None)
            };
            assert!(
                refusal(admitted(&g)).contains("unsupported head128 layer 3"),
                "{heads}/{kv}"
            );
        }
    }

    /// Rope factors ride a whole-head NeoX layer; they do not make a NoPE layer one, and YaRN
    /// is still refused.
    #[test]
    fn rope_factors_need_the_neox_table() {
        assert!(admitted(&layer(neox(true), None)).is_ok());
        let nope = Rope {
            mode: RopeMode::None,
            dims: 0,
            base: 0.0,
            yarn: None,
            factors: false,
        };
        assert!(!admitted(&layer(nope, None)).expect("nope").rotate);
        let nope_factors = Rope {
            factors: true,
            ..nope
        };
        assert!(refusal(admitted(&layer(nope_factors, None))).contains("NoPE or whole-head NeoX"));
        let partial = Rope {
            dims: 64,
            ..neox(true)
        };
        assert!(refusal(admitted(&layer(partial, None))).contains("NoPE or whole-head NeoX"));
    }

    /// The per-head output gate is the gate kernel's launch: refused by name, not run silently
    /// without it.
    #[test]
    fn an_output_gate_is_refused_by_name() {
        for gate in [OutGate::ProjHead, OutGate::Proj, OutGate::Packed] {
            let g = Gqa {
                out_gate: gate,
                ..layer(neox(false), Some(256))
            };
            let e = refusal(admitted(&g));
            assert!(
                e.contains(&format!("{gate:?}")) && e.contains("output-gate kernel"),
                "{e}"
            );
        }
    }
}
