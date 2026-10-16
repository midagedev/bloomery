//! bloomery-gpu-gemma-embedding2: the EmbeddingGemma 2 encoder
//! (`gemma-embedding2`, llama.cpp's `src/models/gemma-embedding2.cpp`): one
//! pass over an input's token ids, every position attending both ways, to
//! the token rows of the output projection and their mean. It has no device
//! code of its own: every launch is `bloomery-gpu`'s.
//!
//! The pass, in the graph's order:
//! - the token rows, decoded on the host from the file's table
//!   (`gguf::quant::dequant_row`) and multiplied by `√hidden` (`inp_scaled`);
//! - per layer: the attention (RMS norm, the q, k and v GEMMs, the per-head
//!   q and k norms with their gains and the v norm with none, the whole NEOX
//!   turn, the flash at scale 1 over the keys within the layer's reach on
//!   each side of a row (every key on a full layer), the output GEMM, its
//!   post norm and the residual); the GELU GLU, its post norm and the
//!   residual; the per-layer input, `gelu(inp_gate·x) · ple_l` through
//!   `proj` and `post_norm`, added to `x`, where `ple_l` is the layer's
//!   slice of `per_layer_model_proj · inp_scaled`, times `1/√hidden` and
//!   RMS-normed by `per_layer_proj_norm`; then the layer's output times its
//!   `layer_output_scale` (`l_out`);
//! - the output norm (`result_norm`), the `output` GEMM of every position
//!   (`result_embd`), and on the host their mean ([`Pass::pooled`]).
//!
//! - [`Encoder`]: the load and the pass ([`Encoder::pass`]);
//! - [`card_bytes`]: what a load at a context holds on its card.
//!
//! Built with `cargo oxide`, as the device crate it links is.

mod pass;

pub use pass::{Pass, Taps};

use std::collections::BTreeMap;

use bloomery_gpu::flash_gqa_prefill::FlashGqaPrefill;
use bloomery_gpu::flash_gqa_prefill_512::FlashGqaPrefill512;
use bloomery_gpu::gemm::{GEMM_MAX_SLOTS, Gemm32Kernels, GemmAct32, GemmKernels, GemmRoute};
use bloomery_gpu::rope_neox::RopeNeoxKernels;
use bloomery_gpu::rope_table::{RopeRows, RopeSpec, RopeTable};
use bloomery_gpu::weights::{Weights, resident_size};
use bloomery_gpu::{DeviceTensor, Gpu, GpuError};
use cuda_core::DeviceBuffer;
use gguf::{GgmlType, Split};
use model::arch::gemma_embedding2::hparams::{CARD_CTX, Hparams};
use model::arch::gemma_embedding2::names;

/// The refusal `what` of a load.
const OPEN: &str = "Encoder::open";

fn size_overflow() -> GpuError {
    GpuError::Shape {
        what: "card_bytes",
        detail: "encoder byte count overflows usize".to_string(),
    }
}

fn size_product(factors: &[usize]) -> Result<usize, GpuError> {
    factors
        .iter()
        .try_fold(1usize, |n, &factor| n.checked_mul(factor))
        .ok_or_else(size_overflow)
}

fn size_sum(sizes: impl IntoIterator<Item = usize>) -> Result<usize, GpuError> {
    sizes
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(size_overflow)
}

/// How layer `l` attends: its head, its key heads, and the keys a row
/// reaches on each side of its own position (`None`: every key).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Attn {
    pub(crate) head: usize,
    pub(crate) kv: usize,
    pub(crate) reach: Option<usize>,
}

/// Each layer's layout; the program's coverage check owns which shapes run.
pub(crate) fn layers(hp: &Hparams, reach: &[Option<u32>]) -> Vec<Attn> {
    (0..hp.n_layer)
        .map(|l| Attn {
            head: hp.attn(l).head,
            kv: hp.kv[l],
            reach: reach[l].map(|r| r as usize),
        })
        .collect()
}

/// The distinct reaches of `layers`, each once: a pass uploads one pair of
/// key bound rows a reach.
pub(crate) fn reaches(layers: &[Attn]) -> Vec<Option<usize>> {
    let mut r: Vec<Option<usize>> = layers.iter().map(|a| a.reach).collect();
    r.sort_unstable();
    r.dedup();
    r
}

/// Whether `name` is resident on the card: every tensor but the token table
/// (read on the host), `per_layer_model_proj` (uploaded per layer as f32,
/// [`Encoder`]'s `ple`) and the layers' output scalars (read on the host).
fn on_card(name: &str) -> bool {
    name != names::token_embd()
        && name != names::PER_LAYER_MODEL_PROJ
        && !name.ends_with(&format!(".{}", names::LAYER_OUTPUT_SCALE))
}

/// The tensor types the pass reads, refused by name otherwise: the token
/// table and `per_layer_model_proj` any type the host decodes, every other
/// matrix q8_0 (the GEMM's 32-value path), every vector f32.
fn check_types(split: &Split) -> Result<(), GpuError> {
    for (_, t) in split.iter_tensors() {
        let ok = if t.name == names::token_embd() || t.name == names::PER_LAYER_MODEL_PROJ {
            t.ty.has_dequant()
        } else if t.dims.len() >= 2 {
            t.ty == GgmlType::Q8_0
        } else {
            t.ty == GgmlType::F32
        };
        if !ok {
            return Err(GpuError::Tensor {
                what: OPEN,
                name: t.name.clone(),
                need: "q8_0 for a matrix, f32 for a vector, a host-decoded type for the token \
                       table and per_layer_model_proj",
            });
        }
    }
    Ok(())
}

/// The widths a pass's buffers are sized by, at `ctx` positions.
pub(crate) struct Sizes {
    pub(crate) ctx: usize,
    pub(crate) hidden: usize,
    /// Query values a position: the widest layer's `n_head · head`.
    pub(crate) q: usize,
    /// Key (and value) values a position: the widest layer's `kv · head`.
    pub(crate) kv: usize,
    pub(crate) ff: usize,
    pub(crate) per_layer: usize,
    pub(crate) out: usize,
    /// The widest head: the ones the gainless v norm reads.
    pub(crate) head: usize,
    /// The widths a GEMM's input is quantized at.
    pub(crate) act: Vec<usize>,
    /// The distinct reaches ([`reaches`]), one pair of key bound rows each.
    pub(crate) reaches: usize,
}

impl Sizes {
    pub(crate) fn of(hp: &Hparams, layers: &[Attn], ctx: usize) -> Sizes {
        let max = |f: fn(&Attn, usize) -> usize| {
            layers.iter().map(|a| f(a, hp.n_head)).max().unwrap_or(0)
        };
        let mut act: Vec<usize> = layers
            .iter()
            .map(|a| hp.n_head * a.head)
            .chain([hp.n_embd, hp.n_ff, hp.per_layer])
            .collect();
        act.sort_unstable();
        act.dedup();
        Sizes {
            ctx,
            hidden: hp.n_embd,
            q: max(|a, h| h * a.head),
            kv: max(|a, _| a.kv * a.head),
            ff: hp.n_ff,
            per_layer: hp.per_layer,
            out: hp.n_out,
            head: max(|a, _| a.head),
            act,
            reaches: reaches(layers).len(),
        }
    }

    /// The f32 scratch values a position: the pass's buffers, six rows of
    /// the hidden width and three of the query, key and feed-forward widths
    /// and of the per-layer input.
    pub(crate) fn f32_per_pos(&self) -> Result<usize, GpuError> {
        size_sum([
            size_product(&[6, self.hidden])?,
            size_product(&[3, size_sum([self.q, self.kv, self.ff, self.per_layer])?])?,
        ])
    }

    /// The scratch bytes at [`Sizes::ctx`]: the f32 rows, the two f16 cache
    /// planes, the positions, the two scalar rows and the per-layer scale
    /// rows of `n_layer`, the ones, and the per-pass rows (the token rows,
    /// the first keys and key counts of each reach, the output rows); the
    /// quantizer scratch and the route table bounded above by two bytes a
    /// quantized value and sixteen a position.
    fn bytes(&self, n_layer: usize) -> Result<usize, GpuError> {
        let ctx = self.ctx;
        let quant = self.act.iter().try_fold(0usize, |bytes, &k| {
            size_sum([bytes, size_product(&[2, k, ctx])?])
        })?;
        size_sum([
            size_product(&[4, self.f32_per_pos()?, ctx])?,
            size_product(&[4, self.kv, ctx])?,
            size_product(&[4, ctx, size_sum([2, n_layer])?])?,
            size_product(&[4, self.head])?,
            size_product(&[
                4,
                ctx,
                size_sum([self.hidden, size_product(&[2, self.reaches])?, self.out])?,
            ])?,
            quant,
            size_product(&[16, ctx])?,
        ])
    }
}

/// The bytes a load of `split` at `ctx` positions holds on its card: the
/// resident tensors in their device formats ([`resident_size`]), the
/// per-layer projection as f32, the two rope tables and the pass's scratch
/// ([`Sizes`]). The figure a plan reserves for the encoder, and the one its
/// open holds itself to.
pub fn card_bytes(split: &Split, ctx: usize) -> Result<u64, GpuError> {
    let (hp, layers) = header(split, ctx)?;
    let mut bytes = 0usize;
    for (_, t) in split.iter_tensors().filter(|(_, t)| on_card(&t.name)) {
        let k = t.dims.first().copied().unwrap_or(1);
        let rows = t
            .dims
            .iter()
            .skip(1)
            .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))
            .ok_or_else(size_overflow)?;
        let k = usize::try_from(k).map_err(|_| size_overflow())?;
        let resident = resident_size(t.ty, k, rows).ok_or_else(|| GpuError::Tensor {
            what: "card_bytes",
            name: t.name.clone(),
            need: "a type and shape with a device format",
        })?;
        bytes = size_sum([bytes, resident])?;
    }
    let s = Sizes::of(&hp, &layers, ctx);
    card_figure(bytes, &hp, &s)
}

fn card_figure(bytes: usize, hp: &Hparams, s: &Sizes) -> Result<u64, GpuError> {
    let ple = size_product(&[4, hp.n_layer, hp.per_layer, hp.n_embd])?;
    let rope = size_product(&[4, s.ctx, size_sum([hp.swa.head, hp.full.head])?])?;
    u64::try_from(size_sum([bytes, ple, rope, s.bytes(hp.n_layer)?])?).map_err(|_| size_overflow())
}

// A pass's GEMMs take every position as one column.
const _: () = assert!(CARD_CTX <= GEMM_MAX_SLOTS);

fn check_program(read: &model::arch::Read) -> Result<(), GpuError> {
    let hidden = read.spec.hidden as usize;
    let line = bloomery_gpu::gemm::F32_TILE_LINE;
    if hidden == 0 || !hidden.is_multiple_of(line) {
        return Err(GpuError::Shape {
            what: OPEN,
            detail: format!(
                "embedding_length {hidden}: the per-layer f32 projection requires a positive multiple of {line}"
            ),
        });
    }
    let missing = model::arch::coverage::check(&read.spec, &read.tensors);
    if missing.is_empty() {
        Ok(())
    } else {
        Err(GpuError::plan(
            OPEN,
            model::placement::PlacementError::Unimplemented(missing).to_string(),
        ))
    }
}

/// `split`'s header and layers, and `ctx` checked: at least one position,
/// at most the model card's [`CARD_CTX`] and the file's `context_length`.
fn header(split: &Split, ctx: usize) -> Result<(Hparams, Vec<Attn>), GpuError> {
    let hp = Hparams::read(split).map_err(|e| GpuError::plan(OPEN, e.to_string()))?;
    if ctx == 0 || ctx > CARD_CTX || ctx > hp.n_ctx {
        return Err(GpuError::Shape {
            what: OPEN,
            detail: format!(
                "a context of {ctx} positions: one pass takes 1 to the model card's {CARD_CTX}, \
                 within the file's {}",
                hp.n_ctx
            ),
        });
    }
    let read = model::arch::spec(split).map_err(|e| GpuError::plan(OPEN, e.to_string()))?;
    check_program(&read)?;
    check_types(split)?;
    let encoder = read.spec.encoder.as_ref().ok_or(GpuError::State {
        what: OPEN,
        missing: "the encoder description",
    })?;
    let layers = layers(&hp, &encoder.reach);
    Ok((hp, layers))
}

/// The kernels a pass launches besides the [`Gpu`]'s own.
pub(crate) struct Kernels {
    pub(crate) gemm: GemmKernels,
    pub(crate) g32: Gemm32Kernels,
    pub(crate) rope: RopeNeoxKernels,
    pub(crate) flash: FlashGqaPrefill,
    pub(crate) flash512: FlashGqaPrefill512,
}

/// A pass's device buffers, sized at the load's context ([`Sizes`]).
pub(crate) struct Scratch {
    pub(crate) route: GemmRoute,
    /// The quantized GEMM input of each width.
    pub(crate) acts: BTreeMap<usize, GemmAct32>,
    pub(crate) x: DeviceBuffer<f32>,
    pub(crate) xn: DeviceBuffer<f32>,
    pub(crate) a: DeviceBuffer<f32>,
    pub(crate) c: DeviceBuffer<f32>,
    pub(crate) o: DeviceBuffer<f32>,
    pub(crate) on: DeviceBuffer<f32>,
    pub(crate) qs: DeviceBuffer<f32>,
    pub(crate) q: DeviceBuffer<f32>,
    pub(crate) att: DeviceBuffer<f32>,
    pub(crate) kk: DeviceBuffer<f32>,
    pub(crate) vv: DeviceBuffer<f32>,
    pub(crate) vn: DeviceBuffer<f32>,
    pub(crate) gate: DeviceBuffer<f32>,
    pub(crate) up: DeviceBuffer<f32>,
    pub(crate) gu: DeviceBuffer<f32>,
    pub(crate) p1: DeviceBuffer<f32>,
    pub(crate) p2: DeviceBuffer<f32>,
    pub(crate) p3: DeviceBuffer<f32>,
    pub(crate) cache_k: DeviceBuffer<u16>,
    pub(crate) cache_v: DeviceBuffer<u16>,
}

impl Scratch {
    /// The f32 bytes held by the buffers, independent of their size formula.
    fn f32_bytes(&self) -> Result<usize, GpuError> {
        let Scratch {
            route: _,
            acts: _,
            x,
            xn,
            a,
            c,
            o,
            on,
            qs,
            q,
            att,
            kk,
            vv,
            vn,
            gate,
            up,
            gu,
            p1,
            p2,
            p3,
            cache_k: _,
            cache_v: _,
        } = self;
        size_product(&[
            4,
            size_sum(
                [
                    x, xn, a, c, o, on, qs, q, att, kk, vv, vn, gate, up, gu, p1, p2, p3,
                ]
                .map(DeviceBuffer::len),
            )?,
        ])
    }

    fn new(gpu: &Gpu, s: &Sizes) -> Result<Scratch, GpuError> {
        let stream = gpu.stream();
        let f = |w: usize| DeviceBuffer::<f32>::zeroed(stream, w * s.ctx);
        let mut acts = BTreeMap::new();
        for &k in &s.act {
            acts.insert(k, GemmAct32::new(stream, s.ctx, k)?);
        }
        Ok(Scratch {
            route: GemmRoute::new(stream, s.ctx, 1)?,
            acts,
            x: f(s.hidden)?,
            xn: f(s.hidden)?,
            a: f(s.hidden)?,
            c: f(s.hidden)?,
            o: f(s.hidden)?,
            on: f(s.hidden)?,
            qs: f(s.q)?,
            q: f(s.q)?,
            att: f(s.q)?,
            kk: f(s.kv)?,
            vv: f(s.kv)?,
            vn: f(s.kv)?,
            gate: f(s.ff)?,
            up: f(s.ff)?,
            gu: f(s.ff)?,
            p1: f(s.per_layer)?,
            p2: f(s.per_layer)?,
            p3: f(s.per_layer)?,
            cache_k: DeviceBuffer::<u16>::zeroed(stream, s.kv * s.ctx)?,
            cache_v: DeviceBuffer::<u16>::zeroed(stream, s.kv * s.ctx)?,
        })
    }
}

/// The rows every pass reads unchanged: the ones the v norm's gain is, the
/// scalar `1/√hidden` and each layer's output scale, one a position (the
/// per-token weights `weighted_sum` takes), and the positions `0..ctx`.
pub(crate) struct Consts {
    pub(crate) ones: DeviceBuffer<f32>,
    pub(crate) inv: DeviceBuffer<f32>,
    pub(crate) scale: Vec<DeviceBuffer<f32>>,
    pub(crate) pos: DeviceBuffer<u32>,
}

/// The encoder of one `gemma-embedding2` file on one card, sized for passes
/// of up to [`Encoder::ctx`] positions.
pub struct Encoder {
    pub(crate) gpu: Gpu,
    pub(crate) split: Split,
    pub(crate) hp: Hparams,
    pub(crate) layers: Vec<Attn>,
    pub(crate) w: Weights,
    /// Each layer's rows of `per_layer_model_proj`, `per_layer × hidden` f32.
    pub(crate) ple: Vec<DeviceTensor<f32>>,
    pub(crate) k: Kernels,
    /// The rope rows of the sliding layers and of the full layers.
    pub(crate) rope_swa: RopeRows,
    pub(crate) rope_full: RopeRows,
    pub(crate) consts: Consts,
    pub(crate) s: Scratch,
    pub(crate) ctx: usize,
}

impl Encoder {
    /// The model of `split` on `gpu`, for passes of up to `ctx` positions.
    /// A header the reader refuses, a part the program's coverage does not run, a
    /// tensor of a type the pass does not read, a context out of range, or a
    /// load that holds more than [`card_bytes`] is refused by name.
    pub fn open(split: Split, ctx: usize, gpu: Gpu) -> Result<Encoder, GpuError> {
        let (hp, layers) = header(&split, ctx)?;
        let budget = card_bytes(&split, ctx)?;
        let stream = gpu.stream();
        let w = Weights::load_where(stream, &split, on_card)?;
        let ple = ple_rows(&gpu, &split, &hp)?;
        let rope = |a: model::arch::gemma_embedding2::hparams::Attn| {
            let table = RopeTable::new(&RopeSpec::window(a.rope_base, a.head))?;
            RopeRows::new(stream, &table, a.head, ctx)
        };
        let (rope_swa, rope_full) = (rope(hp.swa)?, rope(hp.full)?);
        let sizes = Sizes::of(&hp, &layers, ctx);
        let s = Scratch::new(&gpu, &sizes)?;
        let scale = (0..hp.n_layer)
            .map(|l| {
                let v = output_scale(&split, l)?;
                Ok(DeviceBuffer::from_host(stream, &vec![v; ctx])?)
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        let pos: Vec<u32> = (0..u32::try_from(ctx).map_err(|_| GpuError::Shape {
            what: OPEN,
            detail: format!("{ctx} positions do not fit u32"),
        })?)
            .collect();
        let consts = Consts {
            ones: DeviceBuffer::from_host(stream, &vec![1.0f32; sizes.head])?,
            inv: DeviceBuffer::from_host(stream, &vec![1.0 / (hp.n_embd as f32).sqrt(); ctx])?,
            scale,
            pos: DeviceBuffer::from_host(stream, &pos)?,
        };
        let held = w.resident_bytes()
            + ple.iter().map(|t| 4 * t.rows() * t.cols()).sum::<usize>()
            + 4 * (rope_swa.table.len() + rope_full.table.len())
            + s.acts.values().map(GemmAct32::bytes).sum::<usize>()
            + s.route.bytes()
            + s.f32_bytes()?
            + 2 * (s.cache_k.len() + s.cache_v.len())
            + 4 * (consts.ones.len()
                + consts.inv.len()
                + consts.pos.len()
                + consts.scale.iter().map(|s| s.len()).sum::<usize>());
        if held as u64 > budget {
            return Err(GpuError::Shape {
                what: OPEN,
                detail: format!("the load holds {held} B, past its card figure of {budget} B"),
            });
        }
        let k = Kernels {
            gemm: GemmKernels::load(gpu.context())?,
            g32: Gemm32Kernels::load(gpu.context())?,
            rope: RopeNeoxKernels::load(gpu.context())?,
            flash: FlashGqaPrefill::load(gpu.context())?,
            flash512: FlashGqaPrefill512::load(gpu.context())?,
        };
        Ok(Encoder {
            gpu,
            split,
            hp,
            layers,
            w,
            ple,
            k,
            rope_swa,
            rope_full,
            consts,
            s,
            ctx,
        })
    }

    /// The most positions one pass takes.
    #[must_use]
    pub fn ctx(&self) -> usize {
        self.ctx
    }

    /// The context the encoder's pass owns; a tower call binds it back after its encode.
    pub fn context(&self) -> &std::sync::Arc<cuda_core::CudaContext> {
        self.gpu.context()
    }
}

/// Rows `rows` of `name` in `split`, `k` values each, decoded to f32 and
/// appended to `out` through the common row layout.
pub(crate) fn decode_rows(
    split: &Split,
    name: &str,
    k: usize,
    rows: impl Iterator<Item = usize>,
    out: &mut Vec<f32>,
) -> Result<(), GpuError> {
    let layout = model::embed::RowLayout::open(split, name, k)?;
    for r in rows {
        let id = u32::try_from(r).map_err(|_| GpuError::Shape {
            what: "decode_rows",
            detail: format!("{name}: row {r} exceeds u32"),
        })?;
        let src = layout.row(split, id)?;
        let at = out.len();
        out.resize(at + k, 0.0);
        gguf::quant::dequant_row(layout.ty(), src, &mut out[at..]).map_err(|e| {
            GpuError::Shape {
                what: "decode_rows",
                detail: format!("{name}: {e}"),
            }
        })?;
    }
    Ok(())
}

/// Each layer's `per_layer` rows of `per_layer_model_proj` (`[hidden,
/// per_layer · n_layer]`, layer `l` the rows `l·per_layer ..`), decoded to
/// f32 on the host and uploaded as one tensor a layer: the f32 tile then
/// writes the layer's slice contiguous, each value the whole projection's.
fn ple_rows(gpu: &Gpu, split: &Split, hp: &Hparams) -> Result<Vec<DeviceTensor<f32>>, GpuError> {
    let (e, pl) = (hp.n_embd, hp.per_layer);
    (0..hp.n_layer)
        .map(|l| {
            let mut rows = Vec::with_capacity(pl * e);
            decode_rows(
                split,
                names::PER_LAYER_MODEL_PROJ,
                e,
                l * pl..(l + 1) * pl,
                &mut rows,
            )?;
            DeviceTensor::upload(gpu.stream(), &rows, pl, e)
        })
        .collect()
}

/// Layer `l`'s `layer_output_scale`, one f32.
fn output_scale(split: &Split, l: usize) -> Result<f32, GpuError> {
    let name = format!("blk.{l}.{}", names::LAYER_OUTPUT_SCALE);
    let missing = || GpuError::Tensor {
        what: OPEN,
        name: name.clone(),
        need: "a tensor of the file",
    };
    let (shard, t) = split.find(&name).ok_or_else(missing)?;
    let data = split.shard(shard).ok_or_else(missing)?.data(t)?;
    let b: [u8; 4] = data
        .get(..4)
        .and_then(|b| b.try_into().ok())
        .ok_or_else(|| GpuError::Shape {
            what: OPEN,
            detail: format!("{name} holds {} bytes, not one f32", data.len()),
        })?;
    Ok(f32::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> (Hparams, Sizes) {
        let a = model::arch::gemma_embedding2::hparams::Attn {
            head: 256,
            rope_dims: 256,
            rope_base: 1.0e6,
        };
        let hp = Hparams {
            n_layer: 3,
            n_embd: 32,
            n_ctx: 2,
            n_vocab: 4,
            rms_eps: 1.0e-6,
            n_head: 2,
            n_ff: 64,
            kv: vec![1; 3],
            sliding: vec![true, false, true],
            full: a,
            swa: a,
            window: 4,
            per_layer: 32,
            n_out: 16,
        };
        let s = Sizes::of(&hp, &layers(&hp, &[Some(2), None, Some(2)]), 2);
        (hp, s)
    }

    pub(super) fn with_model<T>(
        hidden: usize,
        ff: usize,
        check: impl FnOnce(Split, &Hparams) -> T,
    ) -> T {
        use gguf::Value as V;
        use gguf::write::{Layout, TensorDecl, Writer};
        use model::arch::names::blk;

        let (mut hp, _) = geometry();
        hp.n_embd = hidden;
        hp.n_ff = ff;
        hp.per_layer = 512;
        hp.n_out = 768;
        let mut kvs = vec![
            (
                gguf::GENERAL_ARCHITECTURE.to_owned(),
                V::String("gemma-embedding2".to_owned()),
            ),
            (
                "tokenizer.ggml.model".to_owned(),
                V::String("gemma4".to_owned()),
            ),
            (
                "tokenizer.ggml.tokens".to_owned(),
                V::Array((0..hp.n_vocab).map(|i| V::String(i.to_string())).collect()),
            ),
        ];
        kvs.extend(
            [
                ("block_count", hp.n_layer),
                ("embedding_length", hp.n_embd),
                ("context_length", hp.n_ctx),
                ("feed_forward_length", hp.n_ff),
                ("attention.head_count", hp.n_head),
                ("attention.sliding_window", hp.window),
                ("embedding_length_per_layer_input", hp.per_layer),
                ("embedding_length_out", hp.n_out),
                ("pooling_type", 1),
            ]
            .map(|(key, n)| (format!("gemma-embedding2.{key}"), V::U64(n as u64))),
        );
        kvs.extend([
            (
                "gemma-embedding2.attention.head_count_kv".to_owned(),
                V::Array(hp.kv.iter().map(|&n| V::U64(n as u64)).collect()),
            ),
            (
                "gemma-embedding2.attention.sliding_window_pattern".to_owned(),
                V::Array(hp.sliding.iter().map(|&b| V::Bool(b)).collect()),
            ),
            (
                "gemma-embedding2.attention.layer_norm_rms_epsilon".to_owned(),
                V::F32(hp.rms_eps),
            ),
        ]);
        for (suffix, a) in [("", hp.full), ("_swa", hp.swa)] {
            kvs.extend(
                [
                    ("attention.key_length", a.head),
                    ("attention.value_length", a.head),
                    ("rope.dimension_count", a.rope_dims),
                ]
                .map(|(key, n)| (format!("gemma-embedding2.{key}{suffix}"), V::U64(n as u64))),
            );
            kvs.push((
                format!("gemma-embedding2.rope.freq_base{suffix}"),
                V::F32(a.rope_base),
            ));
        }
        let e = hp.n_embd as u64;
        let ple = hp.per_layer as u64;
        let mut tensors = vec![
            (names::token_embd(), vec![e, hp.n_vocab as u64]),
            (
                names::PER_LAYER_MODEL_PROJ.to_owned(),
                vec![e, ple * hp.n_layer as u64],
            ),
            (names::PER_LAYER_PROJ_NORM.to_owned(), vec![ple]),
            (names::output_norm(), vec![e]),
            (names::output(), vec![e, hp.n_out as u64]),
        ];
        for l in 0..hp.n_layer {
            let d = hp.attn(l).head as u64;
            let q = d * hp.n_head as u64;
            let k = d * hp.kv[l] as u64;
            tensors.extend(
                [
                    (names::ATTN_NORM, vec![e]),
                    (names::ATTN_Q, vec![e, q]),
                    (names::ATTN_K, vec![e, k]),
                    (names::ATTN_V, vec![e, k]),
                    (names::ATTN_OUTPUT, vec![q, e]),
                    (names::ATTN_Q_NORM, vec![d]),
                    (names::ATTN_K_NORM, vec![d]),
                    (names::POST_ATTENTION_NORM, vec![e]),
                    (names::FFN_NORM, vec![e]),
                    (names::FFN_GATE, vec![e, hp.n_ff as u64]),
                    (names::FFN_UP, vec![e, hp.n_ff as u64]),
                    (names::FFN_DOWN, vec![hp.n_ff as u64, e]),
                    (names::POST_FFW_NORM, vec![e]),
                    (names::INP_GATE, vec![e, ple]),
                    (names::PROJ, vec![ple, e]),
                    (names::POST_NORM, vec![e]),
                    (names::LAYER_OUTPUT_SCALE, vec![1]),
                ]
                .map(|(stem, dims)| (blk(l, stem), dims)),
            );
        }
        let tensors: Vec<_> = tensors
            .into_iter()
            .map(|(name, dims)| {
                let ty = if dims.len() >= 2
                    && name != names::token_embd()
                    && name != names::PER_LAYER_MODEL_PROJ
                {
                    GgmlType::Q8_0
                } else {
                    GgmlType::F32
                };
                TensorDecl {
                    name,
                    nbytes: gguf::tensor_bytes(
                        ty.blck_size().unwrap(),
                        ty.type_size().unwrap(),
                        &dims,
                    )
                    .unwrap(),
                    dims,
                    type_id: ty.as_u32(),
                }
            })
            .collect();
        let layout = Layout::new(&kvs, tensors.clone()).unwrap();
        let path = std::env::temp_dir().join(format!(
            "gemma-embedding2-coverage-{}-{:?}-{hidden}-{ff}.gguf",
            std::process::id(),
            std::thread::current().id(),
        ));
        let file = std::fs::File::create_new(&path).unwrap();
        let mut writer = Writer::new(file, layout).unwrap();
        for t in tensors {
            let nbytes = usize::try_from(t.nbytes).unwrap();
            let bytes = match t.name.as_str() {
                name if name == names::token_embd() || name == names::PER_LAYER_PROJ_NORM => {
                    1.0f32.to_le_bytes().repeat(nbytes / 4)
                }
                names::PER_LAYER_MODEL_PROJ => (1.0f32 / 65536.0).to_le_bytes().repeat(nbytes / 4),
                _ => vec![0; nbytes],
            };
            writer.tensor(&t.name, &bytes).unwrap();
        }
        drop(writer.finish().unwrap());
        let result = check(Split::open(&path).unwrap(), &hp);
        std::fs::remove_file(&path).unwrap();
        result
    }

    #[test]
    fn a_shape_outside_the_program_is_refused_by_name() {
        for (hidden, ff, refused) in [
            (64, 2048, None),
            (32, 2048, Some("embedding_length")),
            (64, 4096, Some("dense GELU GLU layer: ff 4096")),
        ] {
            let result = with_model(hidden, ff, |split, hp| {
                if refused.is_none() {
                    let (_, layers) = header(&split, hp.n_ctx).unwrap();
                    assert_eq!(
                        layers.iter().map(|a| a.reach).collect::<Vec<_>>(),
                        [Some(2), None, Some(2)]
                    );
                }
                card_bytes(&split, hp.n_ctx)
            });
            if let Some(what) = refused {
                let err = result.unwrap_err().to_string();
                assert!(err.contains(OPEN), "{err}");
                assert!(err.contains(what), "{err}");
            } else {
                assert!(result.unwrap() > 0);
            }
        }
    }

    #[test]
    fn card_figure_counts_the_buffers_once() {
        let (hp, s) = geometry();
        assert_eq!(s.f32_per_pos().unwrap(), 2784);
        assert_eq!(s.bytes(hp.n_layer).unwrap(), 28264);
        assert_eq!(card_figure(0, &hp, &s).unwrap(), 44648);
    }

    #[test]
    fn byte_counts_refuse_overflow_by_name() {
        let (hp, s) = geometry();
        let mut results = vec![
            size_product(&[4, usize::MAX / 4 + 1]),
            size_sum([usize::MAX, 1]),
            card_figure(usize::MAX, &hp, &s).map(|n| n as usize),
        ];
        for change in [
            |hp: &mut Hparams| hp.n_layer = usize::MAX,
            |hp: &mut Hparams| hp.per_layer = usize::MAX,
            |hp: &mut Hparams| hp.n_embd = usize::MAX,
            |hp: &mut Hparams| hp.swa.head = usize::MAX,
        ] {
            let mut hp = hp.clone();
            change(&mut hp);
            results.push(card_figure(0, &hp, &s).map(|n| n as usize));
        }
        for change in [
            |s: &mut Sizes| s.hidden = usize::MAX,
            |s: &mut Sizes| s.q = usize::MAX,
            |s: &mut Sizes| s.kv = usize::MAX,
            |s: &mut Sizes| s.ff = usize::MAX,
            |s: &mut Sizes| s.per_layer = usize::MAX,
            |s: &mut Sizes| s.out = usize::MAX,
            |s: &mut Sizes| s.head = usize::MAX,
            |s: &mut Sizes| s.ctx = usize::MAX,
            |s: &mut Sizes| s.act = vec![usize::MAX],
        ] {
            let (_, mut s) = geometry();
            change(&mut s);
            results.push(s.bytes(hp.n_layer));
        }
        for result in results {
            assert!(matches!(
                result,
                Err(GpuError::Shape { what: "card_bytes", detail })
                    if detail.contains("overflows")
            ));
        }
    }

    #[test]
    #[ignore = "needs a CUDA card"]
    fn hw_scratch_census_reads_buffer_lengths() {
        let gpu = Gpu::new().unwrap();
        let (_, sizes) = geometry();
        let mut scratch = Scratch::new(&gpu, &sizes).unwrap();
        let bytes = scratch.f32_bytes().unwrap();
        assert_eq!(bytes, 22272);
        scratch.x = DeviceBuffer::zeroed(gpu.stream(), scratch.x.len() + 1).unwrap();
        assert_eq!(scratch.f32_bytes().unwrap(), bytes + 4);
    }
}
