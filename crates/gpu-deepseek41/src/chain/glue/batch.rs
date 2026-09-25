//! The glue over a chunk of a prompt batch ([`GlueBatch`]): the embedding
//! broadcast and the engram step for up to [`HC_MAX_TOKENS`] consecutive
//! tokens whose rows one image of the batch layout carries. Per token, the
//! launches that read a token's rows out of the image are the step's own
//! (`ds41_glue_embed`, `ds41_glue_engram_rows`, their Q3_K forms) at that
//! token's offsets; `engram_wkv` runs over the chunk's columns in one launch
//! ([`DenseKernels::enqueue_m`], column c the one-token launch on token c)
//! and its row-major output is copied token-major ([`crate::transpose`]); the
//! key norm, the gate and the fold take the chunk's tokens in one launch
//! each, token by token what their one-token launches write.

use super::*;
use crate::hc::HC_MAX_TOKENS;
use crate::span::{span, span_mut};
use crate::transpose::TransposeKernels;

const WHAT: &str = "deepseek41 GlueBatch";

/// A chunk's engram scratch and the batch layout's per-token offsets.
pub struct GlueBatch {
    transpose: TransposeKernels,
    /// Tokens an image of the batch layout carries.
    tokens: usize,
    /// Per token of the image: its embedding row's first word, and its
    /// engram rows' first byte.
    embd_at: Vec<u32>,
    engram_byte: Vec<usize>,
    /// Per token of a chunk: a site's rows dequantized (`rows · key_len`), and
    /// per token count `m` (index `m − 1`) their q8_1 form.
    x: DeviceBuffer<f32>,
    acts: Vec<Q8Act>,
    /// `engram_wkv`'s output, row-major, then token-major; the normalized
    /// keys; the gates.
    raw: DeviceBuffer<f32>,
    kv: DeviceBuffer<f32>,
    kn: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
}

impl GlueBatch {
    /// Device bytes of the scratch.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        [&self.x, &self.raw, &self.kv, &self.kn, &self.gate]
            .iter()
            .map(|b| b.num_bytes())
            .sum::<usize>()
            + self
                .acts
                .iter()
                .map(|a| q8act_bytes(a.m(), self.x.len() / self.tokens))
                .sum::<usize>()
    }
}

impl Glue {
    /// The chunk scratch for images of `layout`, the batch layout: its tokens
    /// (1..=[`HC_MAX_TOKENS`]) carry each token's rows as this piece's
    /// one-token layout carries them. Load-time or first-batch only.
    pub fn batch(&self, gpu: &Gpu, layout: &ImageLayout) -> Result<GlueBatch, GpuError> {
        let dims = layout.dims();
        let m = dims.tokens;
        let rows_values = self.rows as usize * self.key_len as usize;
        let site_bytes = self.rows as usize * self.row_bytes as usize;
        if !(1..=HC_MAX_TOKENS).contains(&m)
            || dims.n_embd != self.n_embd
            || layout.embd_words() != self.embd_words as usize
            || dims.engram_bytes != self.sites.len() * site_bytes
        {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "images of {m} tokens (1..={HC_MAX_TOKENS}), rows of {} values, {} embedding \
                     words and {} engram bytes a token; the piece reads {}, {} and {}",
                    dims.n_embd,
                    layout.embd_words(),
                    dims.engram_bytes,
                    self.n_embd,
                    self.embd_words,
                    self.sites.len() * site_bytes
                ),
            });
        }
        let embd_at = (0..m)
            .map(|t| launch_u32(WHAT, "embedding word", layout.embd_at(t)))
            .collect::<Result<Vec<_>, _>>()?;
        let engram_byte: Vec<usize> = (0..m).map(|t| 4 * layout.engram_at(t)).collect();
        if self.table == TableRows::Q3K && engram_byte.iter().any(|b| !b.is_multiple_of(2)) {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: "a token's Q3_K engram rows start off a half word".to_string(),
            });
        }
        let stream = gpu.stream();
        let wide = (HC_STREAMS + 1) * ROW;
        Ok(GlueBatch {
            transpose: TransposeKernels::load(gpu.context())?,
            tokens: m,
            embd_at,
            engram_byte,
            x: DeviceBuffer::zeroed(stream, m * rows_values)?,
            acts: (1..=m)
                .map(|c| Q8Act::with_k(stream, c, rows_values))
                .collect::<Result<_, _>>()?,
            raw: DeviceBuffer::zeroed(stream, m * wide)?,
            kv: DeviceBuffer::zeroed(stream, m * wide)?,
            kn: DeviceBuffer::zeroed(stream, m * HC_STREAMS * ROW)?,
            gate: DeviceBuffer::zeroed(stream, m * HC_STREAMS)?,
        })
    }

    /// Enqueue the embedding broadcast of the first `m` tokens of the image
    /// `params`: token `t`'s row into its four streams, `streams[4·n·t ..]`,
    /// and into its layer-0 input, `input[n·t ..]`. One launch a token.
    /// Asynchronous, allocation-free.
    pub fn enqueue_batch_embed(
        &self,
        gpu: &Gpu,
        b: &GlueBatch,
        params: &DeviceBuffer<u32>,
        m: usize,
        streams: &mut DeviceBuffer<f32>,
        input: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "Glue::enqueue_batch_embed";
        let n = self.n_embd;
        check_tokens(WHAT, b, m)?;
        need(WHAT, "streams", streams.len(), m * HC_STREAMS * n)?;
        need(WHAT, "input", input.len(), m * n)?;
        for (t, &at) in b.embd_at.iter().take(m).enumerate() {
            need(
                WHAT,
                "params",
                params.len(),
                at as usize + self.embd_words as usize,
            )?;
            let mut s = span_mut(WHAT, streams, t * HC_STREAMS * n, HC_STREAMS * n)?;
            let mut x = span_mut(WHAT, input, t * n, n)?;
            match self.embd {
                EmbdRows::Bf16 => {
                    let half = self.embd_words;
                    let prep = self.module.prepare_ds41_glue_embed(LaunchConfig1D::new(
                        half.div_ceil(THREADS),
                        THREADS,
                        0,
                    ))?;
                    self.module.ds41_glue_embed(
                        gpu.stream(),
                        &prep,
                        params,
                        at,
                        half,
                        HC_STREAMS as u32,
                        &mut s,
                        &mut x,
                    )?;
                }
                EmbdRows::Q3K => {
                    let n_sb = launch_u32(WHAT, "super-blocks", n / Q3_K_BLOCK)?;
                    let grid = launch_u32(WHAT, "grid", n.div_ceil(THREADS as usize))?;
                    let prep = self
                        .module
                        .prepare_ds41_glue_embed_q3k(LaunchConfig1D::new(grid, THREADS, 0))?;
                    self.module.ds41_glue_embed_q3k(
                        gpu.stream(),
                        &prep,
                        params,
                        at,
                        n_sb,
                        HC_STREAMS as u32,
                        &mut s,
                        &mut x,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Enqueue the engram step at `layer` for the first `m` tokens of the
    /// image `params`: each token's site rows dequantized, `engram_wkv` over
    /// the `m` columns, the key norm, then the gate over `a.streams` into
    /// `a.out` and the fold of `a.out` by `a.pre` into `a.input` — every
    /// buffer of `a` the chunk's `m` tokens, token-major. Asynchronous,
    /// allocation-free.
    #[allow(
        clippy::too_many_arguments,
        reason = "the engram step's arguments, the chunk's scratch, image and token count (rust-quality R8)"
    )]
    pub fn enqueue_batch_engram(
        &self,
        gpu: &Gpu,
        w: &Weights,
        b: &mut GlueBatch,
        layer: usize,
        params: &DeviceBuffer<u32>,
        m: usize,
        a: EngramStep<'_>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "Glue::enqueue_batch_engram";
        check_tokens(WHAT, b, m)?;
        let stream = gpu.stream();
        let rows_values = (self.rows * self.key_len) as usize;
        let site_bytes = self.rows as usize * self.row_bytes as usize;
        let s = self
            .sites
            .iter()
            .position(|s| s.layer == layer)
            .ok_or_else(|| GpuError::Shape {
                what: WHAT,
                detail: format!("layer {layer} carries no engram site"),
            })?;
        let site = &self.sites[s];
        let grid = (self.rows * self.key_len).div_ceil(THREADS);
        for (t, &token_byte) in b.engram_byte.iter().take(m).enumerate() {
            let at_byte = token_byte + s * site_bytes;
            need(WHAT, "params", 4 * params.len(), at_byte + site_bytes)?;
            let at_byte = launch_u32(WHAT, "site rows", at_byte)?;
            let mut x = span_mut(WHAT, &mut b.x, t * rows_values, rows_values)?;
            match self.table {
                TableRows::Q8_0 => {
                    let prep = self
                        .module
                        .prepare_ds41_glue_engram_rows(LaunchConfig1D::new(grid, THREADS, 0))?;
                    self.module.ds41_glue_engram_rows(
                        stream,
                        &prep,
                        params,
                        at_byte / 4,
                        self.rows,
                        self.row_bytes / 4,
                        self.key_len,
                        &mut x,
                    )?;
                }
                TableRows::Q3K => {
                    let prep = self
                        .module
                        .prepare_ds41_glue_engram_rows_q3k(LaunchConfig1D::new(grid, THREADS, 0))?;
                    self.module.ds41_glue_engram_rows_q3k(
                        stream,
                        &prep,
                        params,
                        at_byte,
                        self.rows,
                        self.key_len / Q3_K_BLOCK as u32,
                        &mut x,
                    )?;
                }
            }
        }
        let wide = (HC_STREAMS + 1) * ROW;
        let wkv = Dense::of(w, &site.wkv, rows_values, wide)?;
        let act = if wkv.reads_q8_1() {
            let act = b.acts.get_mut(m - 1).ok_or_else(|| GpuError::Shape {
                what: WHAT,
                detail: format!("a chunk of {m} tokens"),
            })?;
            gpu.enqueue_quantize_q8_1_layer(&b.x, act, layer)?;
            Some(&*act)
        } else {
            None
        };
        if m == 1 {
            self.dense.enqueue_m(gpu, wkv, &b.x, act, 1, &mut b.kv)?;
        } else {
            self.dense.enqueue_m(gpu, wkv, &b.x, act, m, &mut b.raw)?;
            b.transpose.enqueue(stream, &b.raw, wide, m, &mut b.kv)?;
        }
        let kv = span(WHAT, &b.kv, 0, m * wide)?;
        let mut kn = span_mut(WHAT, &mut b.kn, 0, m * HC_STREAMS * ROW)?;
        self.engram.enqueue_key_norm(
            stream,
            KeyNormArgs {
                kv: &kv,
                gain: gain(w, &site.gain_k, HC_STREAMS * ROW)?,
                eps: self.eps,
                hc: HC_STREAMS,
                m,
                kn: &mut kn,
            },
        )?;
        let mut gates = span_mut(WHAT, &mut b.gate, 0, m * HC_STREAMS)?;
        self.engram.enqueue_gate(
            stream,
            GateArgs {
                x: a.streams,
                kn: &kn,
                kv: &kv,
                gain: gain(w, &site.gain_q, HC_STREAMS * ROW)?,
                eps: self.eps,
                hc: HC_STREAMS,
                m,
                out: &mut *a.out,
                gate: &mut gates,
            },
        )?;
        self.hc
            .enqueue_fold(stream, a.out, a.pre, self.n_embd, m, a.input)
    }
}

/// Refuse a chunk of `m` tokens that the batch layout's images do not carry.
fn check_tokens(what: &'static str, b: &GlueBatch, m: usize) -> Result<(), GpuError> {
    if m == 0 || m > b.tokens {
        return Err(GpuError::Shape {
            what,
            detail: format!("a chunk of {m} tokens; the batch images carry {}", b.tokens),
        });
    }
    Ok(())
}

/// The rows a prompt batch's images carry, read for the whole batch at once:
/// every token's embedding row, then every site's rows of every token hashed
/// first, the reads of all of them started together (one `WILLNEED` a row,
/// [`engram::Site::prefetch`]) and only then copied, so a cold row's read is
/// in flight beside every other's rather than behind the one before it. The
/// bytes are the table's, as [`StepRows`] reads them for a step.
pub struct PromptRows {
    engram: Arc<Engram>,
    embd: (usize, TensorInfo),
    embd_bytes: usize,
    n_vocab: usize,
    n_cols: usize,
    row_bytes: usize,
    cap: usize,
    ctx: Vec<u64>,
    /// `[token][site][n_cols]` for the batch's tokens, and per site the same
    /// ids in token order, the site's one prefetch.
    ids: Vec<u32>,
    site_ids: Vec<Vec<u32>>,
    embd_rows: Vec<u8>,
    engram_rows: Vec<u8>,
    /// Tokens the last fill read.
    tokens: usize,
}

impl StepRows {
    /// The batch reader over the same tables, for batches of up to `cap`
    /// tokens: every buffer a fill writes allocated here.
    #[must_use]
    pub fn prompt_rows(&self, cap: usize) -> PromptRows {
        let sites = self.engram.sites().len();
        PromptRows {
            engram: Arc::clone(&self.engram),
            embd: self.embd.clone(),
            embd_bytes: self.embd_bytes,
            n_vocab: self.n_vocab,
            n_cols: self.n_cols,
            row_bytes: self.row_bytes,
            cap,
            ctx: vec![0; self.ctx.len()],
            ids: vec![0; cap * sites * self.n_cols],
            site_ids: (0..sites)
                .map(|_| Vec::with_capacity(cap * self.n_cols))
                .collect(),
            embd_rows: vec![0; cap * self.embd_bytes],
            engram_rows: vec![0; cap * sites * self.n_cols * self.row_bytes],
            tokens: 0,
        }
    }
}

impl PromptRows {
    /// Read the rows of `plans`' steps, consecutive and in order, at most the
    /// reader's `cap` tokens in all, from `file`, the file the step's reader
    /// opened.
    pub fn fill(&mut self, file: &Split, plans: &[StepPlan]) -> Result<(), GpuError> {
        const WHAT: &str = "PromptRows::fill";
        let n_gram = self.ctx.len();
        let total: usize = plans.iter().map(StepPlan::len).sum();
        if total == 0 || total > self.cap {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("{total} tokens in a batch of at most {}", self.cap),
            });
        }
        let sites = self.engram.sites().len();
        let token_ids = sites * self.n_cols;
        let mut t = 0;
        for plan in plans {
            let m = plan.len();
            if plan.engram_window.len() != m * n_gram {
                return Err(GpuError::Shape {
                    what: WHAT,
                    detail: format!(
                        "a plan of {m} tokens and {} n-gram slots of {n_gram}",
                        plan.engram_window.len()
                    ),
                });
            }
            embd_rows_into(
                &self.embd,
                [self.embd_bytes, self.n_vocab],
                &mut self.embd_rows[t * self.embd_bytes..][..m * self.embd_bytes],
                file,
                plan,
            )?;
            for (k, window) in plan.engram_window.chunks_exact(n_gram).enumerate() {
                let ids = &mut self.ids[(t + k) * token_ids..][..token_ids];
                token_rows(&self.engram, &mut self.ctx, window, ids, self.n_cols)?;
            }
            t += m;
        }
        for (s, list) in self.site_ids.iter_mut().enumerate() {
            list.clear();
            for k in 0..total {
                list.extend_from_slice(&self.ids[(k * sites + s) * self.n_cols..][..self.n_cols]);
            }
        }
        for (site, list) in self.engram.sites().iter().zip(&self.site_ids) {
            site.prefetch(list).map_err(|e| GpuError::plan(WHAT, e))?;
        }
        let site_bytes = self.n_cols * self.row_bytes;
        for k in 0..total {
            let out = &mut self.engram_rows[k * sites * site_bytes..][..sites * site_bytes];
            for ((site, ids), dst) in self
                .engram
                .sites()
                .iter()
                .zip(self.ids[k * token_ids..][..token_ids].chunks_exact(self.n_cols))
                .zip(out.chunks_exact_mut(site_bytes))
            {
                site.copy_rows(ids, dst)
                    .map_err(|e| GpuError::plan(WHAT, e))?;
            }
        }
        self.tokens = total;
        Ok(())
    }

    /// Tokens `range` of the last fill's embedding rows.
    pub fn embd(&self, range: std::ops::Range<usize>) -> Result<&[u8], GpuError> {
        self.rows_of(&self.embd_rows, self.embd_bytes, range)
    }

    /// Tokens `range` of the last fill's engram rows: per token, every site's
    /// rows in site order.
    pub fn engram(&self, range: std::ops::Range<usize>) -> Result<&[u8], GpuError> {
        let token = self.engram.sites().len() * self.n_cols * self.row_bytes;
        self.rows_of(&self.engram_rows, token, range)
    }

    fn rows_of<'a>(
        &self,
        rows: &'a [u8],
        per: usize,
        range: std::ops::Range<usize>,
    ) -> Result<&'a [u8], GpuError> {
        if range.start >= range.end || range.end > self.tokens {
            return Err(GpuError::Shape {
                what: "PromptRows",
                detail: format!("tokens {range:?} of a fill of {}", self.tokens),
            });
        }
        Ok(&rows[range.start * per..range.end * per])
    }
}
