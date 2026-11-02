//! Mistral Large 4's DSpark draft on a card: the file's dense MLA stages as
//! the common owner's layers ([`Ml4Layers`], a
//! [`DraftLayers`](bloomery_gpu::dspark::DraftLayers)), and [`open`], which
//! loads the file against its target's and builds the owner
//! ([`bloomery_gpu::dspark::DsparkBody`]) over them. The file and its
//! statement are `model::arch::mistrallarge4::dspark`'s; the reference is
//! llama.cpp's `graph_mistral_large4` (`src/models/dflash.cpp`).
//!
//! The architecture facts this module holds, and nothing else: the
//! attention is MLA (a latent of `kv_lora` values and a roped tail of
//! `rope` values, shared by every head), the block's rows `1..` are the
//! file's `mask_embd` rows, and the stages are dense. fc, the hidden norm,
//! the positions, the head (the target's output, the soft cap, the
//! recurrent Markov loop) and the graphs are the owner's.
//!
//! # A stage
//!
//! For `m` block rows `x` (token-major, `n_embd` each), in the reference's
//! order:
//!
//! 1. `xn = rms(x) · attn_norm`;
//! 2. `q = attn_q · xn` (Q8_0), each head `[nope | rope]` of `head_k`; every
//!    `rope`-wide column of it turned (`elem::rope`, NORM pairs, the
//!    reference's rope kind for this file) into `q_rope`;
//! 3. the flash row `[q_rope | q_nope · k_b]` of each head: the turned tail
//!    gathered to the row's head (`gather_pairs`), the absorbed nope through
//!    `attn_k_b` per head (`q8_0_gemv_heads_mcol`) behind it;
//! 4. the block's own key rows: `kv_a = attn_kv_a_mqa · xn`, then the latent's
//!    norm (`attn_kv_a_norm`), the tail's turn and the f16 `[k_rope | latent]`
//!    row appended in the ring's slot `ctx + t`
//!    (`kv_norm_rope_append_rows`);
//! 5. the attention (`flash_latent`): every head of every block row is one
//!    query row over the same keys — the `ctx` visible context rows at the
//!    ring's head and the block's `m` rows behind them — so the launch takes
//!    `m · heads` query rows of one token, which sees all of its keys: the
//!    block is not causal, and its window is its first row's (module doc of
//!    `model::arch::mistrallarge4::dspark`);
//! 6. `attn_v_b` per head over the latent output (`q8_0_gemv_heads_mcol`),
//!    `attn_output`, the residual;
//! 7. `rms(·) · ffn_norm`, `ffn_down · (silu(ffn_gate ·) ⊙ ffn_up ·)`, the
//!    residual.
//!
//! A context row (the owner's fc and hidden norm over the target's taps)
//! takes step 4 alone, with no norm before `attn_kv_a_mqa`, at its own
//! position and in slot `p % window`.
//!
//! # The ring
//!
//! Each stage holds `window + MAX_ROWS` f16 rows of `kv_lora + rope`. The
//! row of position `p` sits in slot `p % window`; the owner's positions
//! (`ring::Positions`) make the visible rows the ring's first
//! `min(committed, window)` slots, so a block's keys are one run from slot
//! 0: the visible rows, then the block's own rows written right behind them
//! by the pass. Attention is a softmax over a set of keys, so the ring
//! needs no unrolling. A block row in a slot a later context row owns is
//! overwritten by that row's append before any block reads it.
//!
//! # Staging
//!
//! Every position-dependent word a replayed pass reads — each row's cos/sin
//! row, the slots its key rows go to, the live key count, the anchor's
//! embedded row — is copied to the card by the stage calls, never inside a
//! capture. The mask rows sit in the block's input buffer from the load.

use std::sync::Arc;

use bloomery_gpu::dspark::dense::{gguf_f32, gguf_tensor};
use bloomery_gpu::dspark::head::{HeadTail, HeadWeights, RnnTail};
use bloomery_gpu::dspark::{
    DraftLayers, DsparkBody, EmbedSource, FcWeights, HeadSource, OwnHeld, Q8Rows, TargetHead,
};
use bloomery_gpu::flash::{FlashGeom, FlashInputs, FlashLatentArgs};
use bloomery_gpu::fused::FusedKernels;
use bloomery_gpu::model::StepKernels;
use bloomery_gpu::q8f32::{GemvOut, Q8_0GemvHeadsMcolArgs, Q8_0GemvMcolArgs};
use bloomery_gpu::rope_table::{Direction, RopeSpec, RopeTable};
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, Window, WindowMut, capturing};
use cuda_core::DeviceBuffer;
use gguf::quant::dequant_row;
use gguf::{GgmlType, Split};
use model::arch::dspark::DraftHeld;
use model::arch::mistrallarge4::dspark::{self as file, Hparams, names};
use model::arch::mistrallarge4::hparams::Hparams as TargetHparams;
use model::arch::mistrallarge4::names as stem;
use model::arch::names::{blk, output, token_embd};
use model::embed::EmbedTable;

const WHAT: &str = "mistrallarge4::draft";

/// Rows of one block pass and of one append group: the Q8_0 row kernel's
/// column bound.
const ROWS: usize = 8;

/// The draft on its card: the common owner over [`Ml4Layers`].
pub type Ml4Draft = DsparkBody<Ml4Layers>;

fn refuse(detail: String) -> GpuError {
    GpuError::shape(WHAT, detail)
}

/// The Q8_0 tensor `name` of `split`, of ggml shape `dims`, as `rows` rows
/// of `k` on the card (the common reader, [`Q8Rows::from_gguf_dims`]).
fn q8(
    gpu: &Gpu,
    split: &Split,
    name: &str,
    dims: &[usize],
    (rows, k): (usize, usize),
) -> Result<Q8Rows, GpuError> {
    Q8Rows::from_gguf_dims(gpu, split, name, dims, rows, k)
}

/// The f32 vector `name` of `split`, `n` values, on the card.
fn f32s(gpu: &Gpu, split: &Split, name: &str, n: usize) -> Result<DeviceBuffer<f32>, GpuError> {
    Ok(DeviceBuffer::from_host(
        gpu.stream(),
        &gguf_f32(split, name, &[n])?,
    )?)
}

/// The f32 vector `name` of `split`, `n` values, as the one-row tensor the
/// owner's norms take.
fn gain(gpu: &Gpu, split: &Split, name: &str, n: usize) -> Result<DeviceTensor<f32>, GpuError> {
    DeviceTensor::upload(gpu.stream(), &gguf_f32(split, name, &[n])?, 1, n)
}

/// A stage weight on the card, as [`Ml4Layers::loaded`] names it.
pub enum Loaded<'a> {
    /// A Q8_0 matrix in the row kernels' two planes.
    Q8(&'a Q8Rows),
    /// A gain vector.
    F32(&'a DeviceBuffer<f32>),
}

/// One stage's weights and its ring.
struct Stage {
    attn_norm: DeviceBuffer<f32>,
    /// `attn_q`: `heads · head_k` rows over `n_embd`.
    q: Q8Rows,
    /// `attn_kv_a_mqa`: `kv_lora + rope` rows over `n_embd`.
    kv_a: Q8Rows,
    kv_a_norm: DeviceBuffer<f32>,
    /// `attn_k_b`: `heads · kv_lora` rows over the nope part.
    k_b: Q8Rows,
    /// `attn_v_b`: `heads · head_v` rows over `kv_lora`.
    v_b: Q8Rows,
    /// `attn_output`: `n_embd` rows over `heads · head_v`.
    o: Q8Rows,
    ffn_norm: DeviceBuffer<f32>,
    gate: Q8Rows,
    up: Q8Rows,
    down: Q8Rows,
    /// `window + ROWS` f16 rows of `kv_lora + rope` (module doc).
    ring: DeviceTensor<u16>,
}

impl Stage {
    fn load(gpu: &Gpu, split: &Split, hp: &Hparams, l: usize) -> Result<Stage, GpuError> {
        let (e, h) = (hp.n_embd, hp.n_head);
        let (lat, rope, nope) = (hp.kv_lora, hp.rope_dims, hp.nope());
        let n = |s: &str| blk(l, s);
        Ok(Stage {
            attn_norm: f32s(gpu, split, &n(stem::ATTN_NORM), e)?,
            q: q8(
                gpu,
                split,
                &n(stem::ATTN_Q),
                &[e, h * hp.head_k],
                (h * hp.head_k, e),
            )?,
            kv_a: q8(
                gpu,
                split,
                &n(stem::ATTN_KV_A_MQA),
                &[e, lat + rope],
                (lat + rope, e),
            )?,
            kv_a_norm: f32s(gpu, split, &n(stem::ATTN_KV_A_NORM), lat)?,
            k_b: q8(
                gpu,
                split,
                &n(stem::ATTN_K_B),
                &[nope, lat, h],
                (h * lat, nope),
            )?,
            v_b: q8(
                gpu,
                split,
                &n(stem::ATTN_V_B),
                &[lat, hp.head_v, h],
                (h * hp.head_v, lat),
            )?,
            o: q8(
                gpu,
                split,
                &n(stem::ATTN_OUTPUT),
                &[h * hp.head_v, e],
                (e, h * hp.head_v),
            )?,
            ffn_norm: f32s(gpu, split, &n(stem::FFN_NORM), e)?,
            gate: q8(gpu, split, &n(stem::FFN_GATE), &[e, hp.n_ff], (hp.n_ff, e))?,
            up: q8(gpu, split, &n(stem::FFN_UP), &[e, hp.n_ff], (hp.n_ff, e))?,
            down: q8(gpu, split, &n(stem::FFN_DOWN), &[hp.n_ff, e], (e, hp.n_ff))?,
            ring: DeviceTensor::zeroed(gpu.stream(), hp.window + ROWS, lat + rope)?,
        })
    }

    /// Device bytes of each weight buffer, in [`Stage::load`]'s order.
    fn weight_buffers(&self, out: &mut Vec<u64>) {
        let f = |b: &DeviceBuffer<f32>| b.num_bytes() as u64;
        let q = |r: &Q8Rows, out: &mut Vec<u64>| {
            out.push(r.qs.buf().num_bytes() as u64);
            out.push(r.d.buf().num_bytes() as u64);
        };
        out.push(f(&self.attn_norm));
        q(&self.q, out);
        q(&self.kv_a, out);
        out.push(f(&self.kv_a_norm));
        q(&self.k_b, out);
        q(&self.v_b, out);
        q(&self.o, out);
        out.push(f(&self.ffn_norm));
        q(&self.gate, out);
        q(&self.up, out);
        q(&self.down, out);
    }
}

/// The pass's working buffers, every one for [`ROWS`] rows.
struct Scratch {
    xn: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    /// The flash rows, `heads · (rope + kv_lora)` a row.
    q_cur: DeviceBuffer<f32>,
    kv_a: DeviceBuffer<f32>,
    kv_s: DeviceBuffer<f32>,
    kvr: DeviceBuffer<f32>,
    att: DeviceBuffer<f32>,
    av: DeviceBuffer<f32>,
    r: DeviceBuffer<f32>,
    ffn_inp: DeviceBuffer<f32>,
    g: DeviceBuffer<f32>,
    u: DeviceBuffer<f32>,
    act: DeviceBuffer<f32>,
    /// The flash row's tail gather: each head's turned tail column into the
    /// first `rope` values of its flash row, for [`ROWS`] rows.
    from: DeviceBuffer<u32>,
    to: DeviceBuffer<u32>,
}

impl Scratch {
    fn new(gpu: &Gpu, hp: &Hparams) -> Result<Scratch, GpuError> {
        let s = gpu.stream();
        let z = |n: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(s, ROWS * n)?)
        };
        let (h, rope, lat) = (hp.n_head, hp.rope_dims, hp.kv_lora);
        let width = rope + lat;
        let (mut from, mut to) = (Vec::new(), Vec::new());
        for t in 0..ROWS {
            for head in 0..h {
                for j in 0..rope {
                    from.push(t * h * hp.head_k + head * hp.head_k + hp.nope() + j);
                    to.push(t * h * width + head * width + j);
                }
            }
        }
        let words = |v: Vec<usize>| -> Result<DeviceBuffer<u32>, GpuError> {
            let w = v
                .into_iter()
                .map(|i| u32::try_from(i).map_err(|_| refuse(format!("a gather index {i}"))))
                .collect::<Result<Vec<u32>, _>>()?;
            Ok(DeviceBuffer::from_host(s, &w)?)
        };
        Ok(Scratch {
            xn: z(hp.n_embd)?,
            q: z(h * hp.head_k)?,
            q_rope: z(h * hp.head_k)?,
            q_cur: z(h * width)?,
            kv_a: z(width)?,
            kv_s: z(width)?,
            kvr: z(width)?,
            att: z(h * lat)?,
            av: z(h * hp.head_v)?,
            r: z(hp.n_embd)?,
            ffn_inp: z(hp.n_embd)?,
            g: z(hp.n_ff)?,
            u: z(hp.n_ff)?,
            act: z(hp.n_ff)?,
            from: words(from)?,
            to: words(to)?,
        })
    }

    fn bytes(&self) -> u64 {
        let f = [
            &self.xn,
            &self.q,
            &self.q_rope,
            &self.q_cur,
            &self.kv_a,
            &self.kv_s,
            &self.kvr,
            &self.att,
            &self.av,
            &self.r,
            &self.ffn_inp,
            &self.g,
            &self.u,
            &self.act,
        ]
        .iter()
        .map(|b| b.num_bytes())
        .sum::<usize>();
        (f + self.from.num_bytes() + self.to.num_bytes()) as u64
    }
}

/// What the last stage call prepared, so an eager pass runs only on its own
/// stage and a replay is refused before any.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Staged {
    None,
    Kv(usize),
    Block(usize),
}

/// The stage calls' card images (module doc, Staging).
struct Images {
    /// The block rows' cos/sin rows, [`ROWS`] of `rope`.
    blk_cs: DeviceBuffer<f32>,
    /// The ring slots the block's key rows go to, [`ROWS`] words.
    blk_slots: DeviceBuffer<u32>,
    /// The block's live key count: the visible rows and the block's.
    blk_keys: DeviceBuffer<u32>,
    /// The block's input rows: the anchor's at row 0 (staged), the file's
    /// mask rows at rows `1..block_size` (written at load).
    rows: DeviceBuffer<f32>,
    /// A context append's cos/sin rows and slots, `cap` rows.
    ctx_cs: DeviceBuffer<f32>,
    ctx_slots: DeviceBuffer<u32>,
}

impl Images {
    fn bytes(&self) -> u64 {
        (self.blk_cs.num_bytes()
            + self.blk_slots.num_bytes()
            + self.blk_keys.num_bytes()
            + self.rows.num_bytes()
            + self.ctx_cs.num_bytes()
            + self.ctx_slots.num_bytes()) as u64
    }
}

/// Mistral Large 4's draft stages on the card (module doc).
pub struct Ml4Layers {
    hp: Hparams,
    stages: Vec<Stage>,
    s: Scratch,
    img: Images,
    fused: FusedKernels,
    step: StepKernels,
    rope: RopeTable,
    /// The target's `token_embd`, read on the host: the anchor's row.
    embed: EmbedTable,
    /// The anchor's row on the host, `n_embd` values.
    anchor: Vec<f32>,
    /// The most context rows one append takes: the window in whole groups.
    cap: usize,
    staged: Staged,
}

impl Ml4Layers {
    /// The stages of `draft` (its statement checked, `hp` read from it) on
    /// `gpu`, the anchor rows read from `embed`, the target's `token_embd`.
    /// Load-time only.
    fn load(
        gpu: &Gpu,
        draft: &Split,
        hp: &Hparams,
        embed: EmbedTable,
    ) -> Result<Ml4Layers, GpuError> {
        if hp.block_size > ROWS {
            return Err(refuse(format!(
                "a block of {} rows; a pass holds 1..={ROWS}",
                hp.block_size
            )));
        }
        let stages = (0..hp.n_layer)
            .map(|l| Stage::load(gpu, draft, hp, l))
            .collect::<Result<Vec<_>, _>>()?;
        let ctx = gpu.context();
        let s = gpu.stream();
        // The block's input rows: the mask rows from the file at rows
        // `1..block_size`, row 0 the anchor's at each stage.
        let e = hp.n_embd;
        let masks = gguf_tensor(
            draft,
            &names::mask_embd(),
            GgmlType::Q8_0,
            &[e, hp.block_size - 1],
        )?;
        let mut rows = vec![0.0f32; ROWS * e];
        let row_bytes = e / 32 * 34;
        for k in 0..hp.block_size - 1 {
            dequant_row(
                GgmlType::Q8_0,
                &masks[k * row_bytes..(k + 1) * row_bytes],
                &mut rows[(k + 1) * e..(k + 2) * e],
            )
            .map_err(|err| refuse(format!("{}: {err}", names::mask_embd())))?;
        }
        let cap = hp.window.div_ceil(ROWS) * ROWS;
        let img = Images {
            blk_cs: DeviceBuffer::zeroed(s, ROWS * hp.rope_dims)?,
            blk_slots: DeviceBuffer::zeroed(s, ROWS)?,
            blk_keys: DeviceBuffer::zeroed(s, 1)?,
            rows: DeviceBuffer::from_host(s, &rows)?,
            ctx_cs: DeviceBuffer::zeroed(s, cap * hp.rope_dims)?,
            ctx_slots: DeviceBuffer::zeroed(s, cap)?,
        };
        // The draft's rope: plain YaRN at the file's keys, no LeRoPE factors
        // (the reader holds the net magnitude to 1, as the target's).
        let spec = RopeSpec::yarn(
            hp.rope_base,
            hp.yarn.factor,
            hp.yarn.orig_ctx as i32,
            hp.yarn.beta_fast,
            hp.yarn.beta_slow,
            hp.rope_dims,
        );
        Ok(Ml4Layers {
            stages,
            s: Scratch::new(gpu, hp)?,
            img,
            fused: FusedKernels::load(ctx)?,
            step: StepKernels::load(ctx)?,
            rope: RopeTable::new(&spec)?,
            anchor: vec![0.0; e],
            embed,
            cap,
            staged: Staged::None,
            hp: hp.clone(),
        })
    }

    /// The file's hyperparameters.
    #[must_use]
    pub fn hp(&self) -> &Hparams {
        &self.hp
    }

    /// Stage `l`'s ring, the gates' readout.
    #[must_use]
    pub fn ring(&self, l: usize) -> Option<&DeviceTensor<u16>> {
        self.stages.get(l).map(|s| &s.ring)
    }

    /// Every stage weight by the file's tensor name, in load order: the
    /// gates' readout of what the load put on the card.
    #[must_use]
    pub fn loaded(&self) -> Vec<(String, Loaded<'_>)> {
        let mut out = Vec::new();
        for (l, st) in self.stages.iter().enumerate() {
            let n = |s: &str| blk(l, s);
            out.extend([
                (n(stem::ATTN_NORM), Loaded::F32(&st.attn_norm)),
                (n(stem::ATTN_Q), Loaded::Q8(&st.q)),
                (n(stem::ATTN_KV_A_MQA), Loaded::Q8(&st.kv_a)),
                (n(stem::ATTN_KV_A_NORM), Loaded::F32(&st.kv_a_norm)),
                (n(stem::ATTN_K_B), Loaded::Q8(&st.k_b)),
                (n(stem::ATTN_V_B), Loaded::Q8(&st.v_b)),
                (n(stem::ATTN_OUTPUT), Loaded::Q8(&st.o)),
                (n(stem::FFN_NORM), Loaded::F32(&st.ffn_norm)),
                (n(stem::FFN_GATE), Loaded::Q8(&st.gate)),
                (n(stem::FFN_UP), Loaded::Q8(&st.up)),
                (n(stem::FFN_DOWN), Loaded::Q8(&st.down)),
            ]);
        }
        out
    }

    /// The block's mask rows as the load wrote them, `block_size - 1` rows
    /// of `n_embd`: the gates' readout of `mask_embd`. Blocking.
    pub fn mask_rows(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        let e = self.hp.n_embd;
        let all = self.img.rows.to_host_vec(gpu.stream())?;
        Ok(all[e..self.hp.block_size * e].to_vec())
    }

    /// The cos/sin rows of `n` positions from `first`, `rope` each, padded to
    /// `rows` rows.
    fn cos_sin(&self, first: u32, n: usize, rows: usize) -> Result<Vec<f32>, GpuError> {
        let mut out = Vec::with_capacity(rows * self.hp.rope_dims);
        for t in 0..n {
            let p = u32::try_from(t)
                .ok()
                .and_then(|t| first.checked_add(t))
                .ok_or_else(|| refuse(format!("position {first} + {t} passes u32")))?;
            self.rope.push(p, Direction::Forward, &mut out);
        }
        out.resize(rows * self.hp.rope_dims, 0.0);
        Ok(out)
    }

    /// Device bytes of every weight buffer, stage by stage in load order.
    fn weight_buffers(&self) -> Vec<u64> {
        let mut out = Vec::new();
        for st in &self.stages {
            st.weight_buffers(&mut out);
        }
        out
    }

    /// Device bytes of every stage's ring.
    fn ring_bytes(&self) -> Vec<u64> {
        self.stages
            .iter()
            .map(|s| s.ring.buf().num_bytes() as u64)
            .collect()
    }

    /// Device bytes of the pass's and the append's working buffers and the
    /// stage images.
    fn working_bytes(&self) -> u64 {
        self.s.bytes() + self.img.bytes()
    }

    /// The buffers the load allocated: these layers' weights, then the
    /// owner's (`own`, [`DsparkBody::own_held`]), the rings, and every
    /// working buffer.
    #[must_use]
    pub fn held(&self, own: &OwnHeld) -> DraftHeld {
        let mut weights = self.weight_buffers();
        weights.extend(own.weights.iter().copied().filter(|&n| n > 0));
        DraftHeld {
            weights,
            rings: self.ring_bytes(),
            working: self.working_bytes() + own.working,
        }
    }

    /// Refuse an eager launch no stage call prepared; a capture runs before
    /// any stage call.
    fn ready(&self, gpu: &Gpu, want: Staged) -> Result<(), GpuError> {
        if self.staged == want || capturing(gpu.stream())? {
            return Ok(());
        }
        Err(GpuError::State {
            what: WHAT,
            missing: "a launch of the rows the last stage call staged",
        })
    }
}

/// `m` columns of the Q8_0 rows `w` over `x` (token-major) into `y`,
/// token-major.
fn gemv(
    gpu: &Gpu,
    w: &Q8Rows,
    x: &DeviceBuffer<f32>,
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    gpu.q8f32().enqueue_q8_0_gemv_mcol(
        gpu.stream(),
        Q8_0GemvMcolArgs {
            qs: &w.qs,
            d: &w.d,
            x,
            m,
            out: GemvOut::TokenMajor,
            y,
        },
    )
}

impl DraftLayers for Ml4Layers {
    const MAX_ROWS: usize = ROWS;
    const APPEND_GROUP: usize = ROWS;

    fn hidden(&self) -> usize {
        self.hp.n_embd
    }

    fn replayable(&self) -> Result<(), GpuError> {
        if self.staged == Staged::None {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a stage call before the replay",
            });
        }
        Ok(())
    }

    fn stage_kv(&mut self, gpu: &Gpu, first: u32, n: usize) -> Result<(), GpuError> {
        if n == 0 || n > self.cap {
            return Err(refuse(format!(
                "an append of {n} rows; one takes 1..={}",
                self.cap
            )));
        }
        let window = u32::try_from(self.hp.window)
            .map_err(|_| refuse(format!("a window of {}", self.hp.window)))?;
        let mut slots = vec![0u32; self.cap];
        for (t, slot) in slots.iter_mut().take(n).enumerate() {
            let p = u32::try_from(t)
                .ok()
                .and_then(|t| first.checked_add(t))
                .ok_or_else(|| refuse(format!("position {first} + {t} passes u32")))?;
            *slot = p % window;
        }
        let cs = self.cos_sin(first, n, self.cap)?;
        let s = gpu.stream();
        self.img.ctx_cs.copy_from_host(s, &cs)?;
        self.img.ctx_slots.copy_from_host(s, &slots)?;
        self.staged = Staged::Kv(n);
        Ok(())
    }

    fn append_kv(&mut self, gpu: &Gpu, x: &DeviceBuffer<f32>, n: usize) -> Result<(), GpuError> {
        self.ready(gpu, Staged::Kv(n))?;
        let (e, lat, rope) = (self.hp.n_embd, self.hp.kv_lora, self.hp.rope_dims);
        if n == 0 || x.len() < n * e {
            return Err(refuse(format!(
                "an append of {n} rows from {} values of {e}-wide rows",
                x.len()
            )));
        }
        let (stream, fault, eps) = (gpu.stream(), gpu.unlabelled_sink(), self.hp.rms_eps);
        let Ml4Layers {
            stages,
            s,
            img,
            fused,
            ..
        } = self;
        for r0 in (0..n).step_by(ROWS) {
            let g = ROWS.min(n - r0);
            let rows = Window::<f32>::of(x, 4 * r0 * e, g * e)?;
            let cs = Window::<f32>::of(&img.ctx_cs, 4 * r0 * rope, g * rope)?;
            let slots = Window::<u32>::of(&img.ctx_slots, 4 * r0, g)?;
            for st in stages.iter_mut() {
                gemv(gpu, &st.kv_a, &rows, g, &mut s.kv_a)?;
                fused.enqueue_kv_norm_rope_append_rows(
                    stream,
                    &s.kv_a,
                    &st.kv_a_norm,
                    &cs,
                    &slots,
                    eps,
                    lat,
                    rope,
                    g,
                    &mut s.kv_s,
                    &mut s.kvr,
                    &mut st.ring,
                    fault,
                )?;
            }
        }
        Ok(())
    }

    fn stage_block(
        &mut self,
        gpu: &Gpu,
        last: u32,
        pos: u32,
        m: usize,
        ctx: usize,
    ) -> Result<(), GpuError> {
        if m == 0 || m > self.hp.block_size {
            return Err(refuse(format!(
                "a block of {m} rows; the file's blocks hold 1..={} (its mask rows)",
                self.hp.block_size
            )));
        }
        if ctx > self.hp.window {
            return Err(refuse(format!(
                "a block over {ctx} context rows; the ring keeps {}",
                self.hp.window
            )));
        }
        let mut slots = vec![0u32; ROWS];
        for (t, slot) in slots.iter_mut().take(m).enumerate() {
            *slot = u32::try_from(ctx + t).map_err(|_| refuse(format!("ring slot {}", ctx + t)))?;
        }
        let keys = u32::try_from(ctx + m).map_err(|_| refuse(format!("{} keys", ctx + m)))?;
        let cs = self.cos_sin(pos, m, ROWS)?;
        self.embed.row_into(last, &mut self.anchor)?;
        let s = gpu.stream();
        self.img.blk_cs.copy_from_host(s, &cs)?;
        self.img.blk_slots.copy_from_host(s, &slots)?;
        self.img.blk_keys.copy_from_host(s, &[keys])?;
        let mut first = WindowMut::<f32>::of_mut(&mut self.img.rows, 0, self.hp.n_embd)?;
        first.copy_from_host(s, &self.anchor)?;
        self.staged = Staged::Block(m);
        Ok(())
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one stage's launches in the reference's order, nothing shared"
    )]
    fn enqueue_block(
        &mut self,
        gpu: &Gpu,
        x: &mut DeviceBuffer<f32>,
        m: usize,
    ) -> Result<(), GpuError> {
        self.ready(gpu, Staged::Block(m))?;
        let hp = &self.hp;
        let (e, h, lat, rope) = (hp.n_embd, hp.n_head, hp.kv_lora, hp.rope_dims);
        let (head_k, head_v, n_ff, eps) = (hp.head_k, hp.head_v, hp.n_ff, hp.rms_eps);
        let width = lat + rope;
        if m == 0 || m > ROWS || x.len() < m * e {
            return Err(refuse(format!(
                "a block of {m} rows into {} values of {e}-wide rows",
                x.len()
            )));
        }
        let kq_scale = 1.0 / (head_k as f32).sqrt();
        let (stream, fault) = (gpu.stream(), gpu.unlabelled_sink());
        let (elem, flash) = (gpu.elem(), gpu.flash());
        let Ml4Layers {
            stages,
            s,
            img,
            fused,
            step,
            ..
        } = self;
        {
            let rows = Window::<f32>::of(&img.rows, 0, m * e)?;
            let mut input = WindowMut::<f32>::of_mut(&mut *x, 0, m * e)?;
            input.copy_from_device_async(&rows, stream)?;
        }
        let blk_cs = Window::<f32>::of(&img.blk_cs, 0, m * rope)?;
        let blk_slots = Window::<u32>::of(&img.blk_slots, 0, m)?;
        for st in stages.iter_mut() {
            // 1–2: the norm, the query and its turned columns.
            elem.enqueue_rms_norm(stream, x, &st.attn_norm, eps, e, m, &mut s.xn)?;
            gemv(gpu, &st.q, &s.xn, m, &mut s.q)?;
            let n_vec = u32::try_from(h * head_k / rope)
                .map_err(|_| refuse(format!("{h} heads of {head_k}")))?;
            elem.enqueue_rope(stream, &s.q, &blk_cs, rope, n_vec, m, &mut s.q_rope)?;
            // 3: the flash rows, the turned tail then the absorbed nope.
            step.enqueue_gather(
                stream,
                &s.q_rope,
                &s.from,
                &s.to,
                m * h * rope,
                &mut s.q_cur,
            )?;
            gpu.q8f32().enqueue_q8_0_gemv_heads_mcol(
                stream,
                Q8_0GemvHeadsMcolArgs {
                    qs: &st.k_b.qs,
                    d: &st.k_b.d,
                    x: &s.q,
                    rows_per_head: lat,
                    x_head_stride: head_k,
                    y_head_stride: width,
                    y_off: rope,
                    m,
                    x_col_stride: h * head_k,
                    y_col_stride: h * width,
                    y: &mut s.q_cur,
                },
            )?;
            // 4: the block's key rows behind the visible ones.
            gemv(gpu, &st.kv_a, &s.xn, m, &mut s.kv_a)?;
            fused.enqueue_kv_norm_rope_append_rows(
                stream,
                &s.kv_a,
                &st.kv_a_norm,
                &blk_cs,
                &blk_slots,
                eps,
                lat,
                rope,
                m,
                &mut s.kv_s,
                &mut s.kvr,
                &mut st.ring,
                fault,
            )?;
            // 5: every head of every row one query row over the same keys.
            flash.enqueue_flash_latent(
                stream,
                FlashLatentArgs {
                    inputs: FlashInputs {
                        q: &s.q_cur,
                        kv: &st.ring,
                        n_keys_buf: &img.blk_keys,
                        kq_scale,
                        geom: FlashGeom {
                            tokens: 1,
                            heads: m * h,
                            rope_dims: rope,
                            latent_dims: lat,
                        },
                    },
                    y: &mut s.att,
                },
            )?;
            // 6: the values, the output projection, the residual.
            gpu.q8f32().enqueue_q8_0_gemv_heads_mcol(
                stream,
                Q8_0GemvHeadsMcolArgs {
                    qs: &st.v_b.qs,
                    d: &st.v_b.d,
                    x: &s.att,
                    rows_per_head: head_v,
                    x_head_stride: lat,
                    y_head_stride: head_v,
                    y_off: 0,
                    m,
                    x_col_stride: h * lat,
                    y_col_stride: h * head_v,
                    y: &mut s.av,
                },
            )?;
            gemv(gpu, &st.o, &s.av, m, &mut s.r)?;
            elem.enqueue_add(stream, &s.r, x, m * e, &mut s.ffn_inp)?;
            // 7: the dense FFN and its residual.
            elem.enqueue_rms_norm(stream, &s.ffn_inp, &st.ffn_norm, eps, e, m, &mut s.xn)?;
            gemv(gpu, &st.gate, &s.xn, m, &mut s.g)?;
            gemv(gpu, &st.up, &s.xn, m, &mut s.u)?;
            elem.enqueue_swiglu(stream, &s.g, &s.u, m * n_ff, &mut s.act)?;
            gemv(gpu, &st.down, &s.act, m, &mut s.r)?;
            elem.enqueue_add(stream, &s.r, &s.ffn_inp, m * e, x)?;
        }
        Ok(())
    }

    fn reset(&mut self, _gpu: &Gpu) -> Result<(), GpuError> {
        self.staged = Staged::None;
        Ok(())
    }

    fn card_bytes(&self, _ctx: usize) -> u64 {
        self.weight_buffers().iter().sum::<u64>()
            + self.ring_bytes().iter().sum::<u64>()
            + self.working_bytes()
    }
}

/// Open `draft`, Mistral Large 4's DSpark file, on `gpu` for the target file
/// `target`: the draft's keys and statement held to the file
/// ([`file::check`]) and to the target's hyperparameters ([`file::fit`]),
/// the target's `output` (the head's projection, in the target file's type:
/// Q6_K, else refused by name) copied to the draft's card, its
/// `token_embd` read on the host for each block's anchor row (a type the
/// host reader decodes, else refused by name), and the owner over the
/// stages. Load-time only.
pub fn open(gpu: &Gpu, draft: &Split, target: Arc<Split>) -> Result<Ml4Draft, GpuError> {
    let hp = Hparams::read(draft).map_err(|e| GpuError::plan(WHAT, e))?;
    file::check(draft, &hp).map_err(|e| GpuError::plan(WHAT, e))?;
    let target_hp = TargetHparams::read(&target).map_err(|e| GpuError::plan(WHAT, e))?;
    file::fit(&hp, &target_hp).map_err(|e| GpuError::plan(WHAT, e))?;
    let shape = file::shape(&hp);
    let e = hp.n_embd;
    let fc = FcWeights {
        fc: q8(gpu, draft, &names::fc(), &[hp.fc_in(), e], (e, hp.fc_in()))?,
        norm: gain(gpu, draft, &names::enc_output_norm(), e)?,
        eps: hp.rms_eps,
    };
    let rank = hp.markov_rank;
    let tail = RnnTail {
        w1: q8(
            gpu,
            draft,
            &names::markov_w1(),
            &[rank, hp.n_vocab],
            (hp.n_vocab, rank),
        )?,
        w2: q8(
            gpu,
            draft,
            &names::markov_w2(),
            &[rank, hp.n_vocab],
            (hp.n_vocab, rank),
        )?,
        joint: q8(
            gpu,
            draft,
            &names::joint_proj(),
            &[hp.joint_in(), 3 * rank],
            (3 * rank, hp.joint_in()),
        )?,
        joint_b: f32s(gpu, draft, &names::joint_proj_bias(), 3 * rank)?,
        softcap: Some(hp.softcap),
    };
    let head = HeadWeights {
        norm: gain(gpu, draft, &names::output_norm(), e)?,
        eps: hp.rms_eps,
        source: HeadSource::Target(TargetHead::Q6K(projection(gpu, &target)?)),
        tail: HeadTail::MarkovRnn(tail),
    };
    let embed = EmbedTable::new(target, &token_embd(), e)?;
    let layers = Ml4Layers::load(gpu, draft, &hp, embed)?;
    DsparkBody::new(gpu, &shape, layers, fc, head, EmbedSource::Target)
}

/// The target file's `output` on the draft's card as the head's row kernel
/// takes it: a Q6_K matrix; any other type is refused by name.
fn projection(gpu: &Gpu, target: &Split) -> Result<DeviceTensor<u32>, GpuError> {
    let name = output();
    let mut w = Weights::load_where(gpu.stream(), target, |n| n == name)?;
    match w.take(&name) {
        Some(DevWeight::KQuant {
            ty: GgmlType::Q6_K,
            w,
            ..
        }) => Ok(w),
        Some(_) => Err(refuse(format!(
            "the target's {name} is not Q6_K: the draft head's row kernel reads Q6_K"
        ))),
        None => Err(GpuError::tensor(WHAT, &name, "in the target file")),
    }
}

/// The buffers `body`'s load allocated, counted from the buffers themselves:
/// the stages' weights, the owner's ([`DsparkBody::own_held`]), the rings
/// and every working buffer. The captured graphs are not buffers.
#[must_use]
pub fn held(body: &Ml4Draft) -> DraftHeld {
    body.layers().held(&body.own_held())
}
