//! The host plan of one DeepSeek-V4.1 step: every integer the step's graph reads, for a batch
//! of consecutive positions of one sequence. The port builds the same set in
//! `llama_prepare_dsv4_graph_inputs` (`src/llama-dsv4.cpp:1742-1852` in our port's tree at
//! `c10fbbcc`: the raw context `:329-480`, a stream's plan `:664-832`, its mask `:842-882`) and
//! in `llama_set_inputs` (`src/llama.cpp`: the output ids `:5836-5857`, the window mask
//! `:5873-6200`, the engram history `llama_kv_prev_tokens` `:5333`).
//!
//! The sequence fills its cache in order from position 0, so a token's KV cell is its position
//! (the port's `kv.head + i` on a cache filled that way). Per step:
//!
//! * **Raw window.** The token at `p` writes its latent row to cell `p` and sees the cells
//!   `max(0, p + 1 − W) ..= p`, `W` the file's `attention.sliding_window` — the port hides
//!   `p − cell ≥ W` (`llama.cpp:6010`). Where a cell's row lives is the attention's choice: the
//!   port keeps a row per position, a ring of `W` rows keeps cell `c` in row `c % W`.
//! * **Compressed streams**, one per distinct nonzero ratio in layer order. The port names the
//!   first `csa` and the second `hca`, and a layer reads the stream of its own ratio
//!   (`llama-hparams.cpp:2140-2193`, `graphs/build_deepseek4.cpp:1132-1134`). A stream of
//!   ratio `r` keeps a state ring of `r` projections, slot `p % r`, and a cache of compressed
//!   rows. The token at `p` completes a group when `p % r == r − 1`: row `p / r` pools the
//!   positions `p + 1 − r ..= p`, read from the ring ⧺ this step's projections, and is roped at
//!   `p + 1 − r`. Then each ring slot keeps the step's last token of its residue. The token at
//!   `p` sees the rows `0 .. (p + 1) / r`.
//! * **Per token**: its id and position, the output ids (the last token's logits only), and the
//!   engram n-gram ending at it, which the engram row ids are hashed from.
//!
//! The counts are the plan. The f16 masks the port feeds its graph are renderings of them
//! ([`StepPlan::raw_mask_into`], [`StreamStep::kq_mask_into`]) whose width grows with the
//! depth, so a kernel's step parameters are the counts.
//!
//! Every model value comes from the file, and a missing key is an error, never a literal.
//! [`MASK_TOKEN_PAD`] and [`KV_PAD`] are the port's mask layout, not model values.

use gguf::Split;

use super::kv::KvLayout;
use super::meta_usize;
use crate::placement::PlacementError;

/// A mask's token lines are padded to a multiple of this (`GGML_KQ_MASK_PAD`, `ggml.h`).
pub const MASK_TOKEN_PAD: usize = 16;
/// The cells or rows a mask covers are padded to a multiple of this, and never fewer
/// (`llama_kv_cache::get_padding` under flash attention, `llama-context.h:77`).
pub const KV_PAD: usize = 256;
/// The mask value of a key a token sees: f16 `0.0`.
pub const F16_VISIBLE: u16 = 0x0000;
/// The mask value of a key a token does not see: f16 `-inf`.
pub const F16_HIDDEN: u16 = 0xfc00;

/// Why a plan cannot be made.
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    /// The file lacks a value the plan is laid out from.
    #[error(transparent)]
    File(#[from] PlacementError),
    /// Values no plan can be laid out from.
    #[error("plan layout: {0}")]
    Layout(String),
    /// A step the plan cannot lay out.
    #[error("step of {tokens} token(s) at position {pos0}: {detail}")]
    Step {
        pos0: u32,
        tokens: usize,
        detail: String,
    },
}

/// What every step's plan is laid out from: the file's window, compression ratios and engram
/// n-gram length, and the positions the caches hold.
#[derive(Clone, Debug)]
pub struct Planner {
    window: u32,
    /// The distinct nonzero ratios in layer order: stream `s` pools `stream_ratios[s]`
    /// positions into a compressed row.
    stream_ratios: Vec<u32>,
    /// Per layer: the stream it reads, `None` on a window-only layer.
    layer_streams: Vec<Option<usize>>,
    /// Tokens in an engram n-gram, the current one included.
    ngram: usize,
    /// Every position a step plans is below this.
    ctx_max: u32,
}

impl Planner {
    /// The planner of the model `split` holds, for caches of `ctx_max` positions: the window
    /// and every layer's ratio from [`KvLayout`], the n-gram length from
    /// `engram.max_ngram_size`.
    pub fn from_file(split: &Split, ctx_max: u64) -> Result<Planner, PlanError> {
        let kv = KvLayout::from_file(split)?;
        let ngram = meta_usize(split, "engram.max_ngram_size")?;
        let narrow = |suffix: &str, v: u64| {
            u32::try_from(v).map_err(|_| PlacementError::Metadata {
                key: split.arch_key(suffix),
                detail: format!("{v} does not fit u32"),
            })
        };
        let window = narrow("attention.sliding_window", kv.window())?;
        let ratios = kv
            .ratios()
            .iter()
            .map(|&r| narrow("attention.compress_ratios", r))
            .collect::<Result<Vec<_>, _>>()?;
        Planner::new(window, &ratios, ngram, ctx_max)
    }

    /// The planner of explicit values: a window of `window` positions, `layer_ratios[l]` the
    /// ratio of layer `l` (0: window only), `ngram` tokens per engram n-gram, and caches of
    /// `ctx_max` positions — at least one, and few enough for the port's i32 positions.
    pub fn new(
        window: u32,
        layer_ratios: &[u32],
        ngram: usize,
        ctx_max: u64,
    ) -> Result<Planner, PlanError> {
        if window == 0 {
            return Err(PlanError::Layout(
                "a window of 0 positions sees no key".to_string(),
            ));
        }
        if ngram == 0 {
            return Err(PlanError::Layout(
                "an engram n-gram of 0 tokens".to_string(),
            ));
        }
        let ctx_max = u32::try_from(ctx_max)
            .ok()
            .filter(|&c| c > 0 && i32::try_from(c).is_ok())
            .ok_or_else(|| {
                PlanError::Layout(format!(
                    "a context of {ctx_max} positions: it holds at least one, and a position is an i32"
                ))
            })?;
        let mut stream_ratios: Vec<u32> = Vec::new();
        let mut layer_streams = Vec::with_capacity(layer_ratios.len());
        for &r in layer_ratios {
            if r == 0 {
                layer_streams.push(None);
                continue;
            }
            let known = stream_ratios.iter().position(|&s| s == r);
            let stream = known.unwrap_or_else(|| {
                stream_ratios.push(r);
                stream_ratios.len() - 1
            });
            layer_streams.push(Some(stream));
        }
        Ok(Planner {
            window,
            stream_ratios,
            layer_streams,
            ngram,
            ctx_max,
        })
    }

    /// `attention.sliding_window`: the positions a token's window spans, its own included.
    pub fn window(&self) -> u32 {
        self.window
    }

    /// The compressed streams' ratios, in the order [`StepPlan::streams`] holds them.
    pub fn stream_ratios(&self) -> &[u32] {
        &self.stream_ratios
    }

    /// The stream layer `layer` reads: `None` on a window-only layer and past the last layer.
    pub fn layer_stream(&self, layer: usize) -> Option<usize> {
        self.layer_streams.get(layer).copied().flatten()
    }

    /// Tokens in an engram n-gram, the current one included.
    pub fn ngram(&self) -> usize {
        self.ngram
    }

    /// Positions the caches hold.
    pub fn ctx_max(&self) -> u32 {
        self.ctx_max
    }

    /// Plan the step that runs `tokens` at positions `pos0 ..`. `before` ends with the token
    /// at `pos0 − 1` and holds at least the `ngram − 1` tokens before `pos0`, or all of them
    /// when fewer exist. `out`'s buffers are reused, so a steady step allocates nothing.
    pub fn plan_into(
        &self,
        tokens: &[u32],
        pos0: u32,
        before: &[u32],
        out: &mut StepPlan,
    ) -> Result<(), PlanError> {
        let refuse = |detail: String| PlanError::Step {
            pos0,
            tokens: tokens.len(),
            detail,
        };
        if tokens.is_empty() {
            return Err(refuse("a step runs at least one token".to_string()));
        }
        let last = u32::try_from(tokens.len() - 1)
            .ok()
            .and_then(|n| pos0.checked_add(n))
            .filter(|&l| l < self.ctx_max)
            .ok_or_else(|| refuse(format!("the caches hold positions below {}", self.ctx_max)))?;
        let needed = (self.ngram - 1).min(pos0 as usize);
        if before.len() < needed {
            return Err(refuse(format!(
                "its engram n-grams reach {needed} token(s) back, and {} are given",
                before.len()
            )));
        }
        let tail = &before[before.len() - needed..];

        out.clear();
        for (p, &token) in (pos0..).zip(tokens) {
            out.tokens.push(token);
            out.pos.push(p);
            out.raw_write.push(p);
            out.raw_first.push(p.saturating_sub(self.window - 1));
            for back in 0..self.ngram {
                out.engram_window
                    .push(token_at(p, back, pos0, tokens, tail));
            }
        }
        out.raw_n_kv = padded(last as usize + 1);
        out.out_ids.push(last - pos0);
        out.streams
            .resize_with(self.stream_ratios.len(), StreamStep::default);
        for (stream, &ratio) in out.streams.iter_mut().zip(&self.stream_ratios) {
            stream.plan(ratio, pos0, last);
        }
        self.bounds(out).map_err(refuse)
    }

    /// The plan's indices against the caches they address, which the port checks of every
    /// plan too (`dsv4_validate_comp_plan`): an index past its cache is a write outside it on
    /// the card.
    fn bounds(&self, plan: &StepPlan) -> Result<(), String> {
        let tokens = plan.len();
        if let Some(c) = plan.raw_write.iter().find(|&&c| c >= self.ctx_max) {
            return Err(format!(
                "cell {c} is past the {} the window cache holds",
                self.ctx_max
            ));
        }
        for s in &plan.streams {
            let r = s.ratio;
            let rows = u64::from(self.ctx_max.div_ceil(r));
            let sources = r as usize + tokens;
            let bad = s
                .state_write
                .iter()
                .find(|&&w| w >= rows)
                .map(|w| format!("writes row {w} of {rows}"))
                .or_else(|| {
                    s.state_read
                        .iter()
                        .find(|&&i| i as usize >= sources)
                        .map(|i| format!("reads source {i} of {sources}"))
                })
                .or_else(|| {
                    s.persist_src
                        .iter()
                        .find(|&&i| i as usize >= tokens)
                        .map(|i| format!("keeps token {i} of {tokens}"))
                })
                .or_else(|| {
                    s.persist_dst
                        .iter()
                        .find(|&&d| d >= r)
                        .map(|d| format!("keeps it in slot {d} of {r}"))
                });
            if let Some(detail) = bad {
                return Err(format!("the stream of ratio {r} {detail}"));
            }
        }
        Ok(())
    }
}

/// `n` cells or rows as a mask covers them: rounded up to [`KV_PAD`], and never fewer.
fn padded(n: usize) -> usize {
    n.div_ceil(KV_PAD).max(1) * KV_PAD
}

/// The token `back` positions before position `p` of a step that runs `tokens` from `pos0`,
/// `tail` ending at `pos0 − 1`; `None` before the sequence starts. `plan_into` gives `tail`
/// the `min(ngram − 1, pos0)` tokens a `back` below `ngram` can reach.
fn token_at(p: u32, back: usize, pos0: u32, tokens: &[u32], tail: &[u32]) -> Option<u32> {
    let q = (p as usize).checked_sub(back)?;
    let pos0 = pos0 as usize;
    Some(if q >= pos0 {
        tokens[q - pos0]
    } else {
        tail[tail.len() - (pos0 - q)]
    })
}

/// One step's plan. The per-token vectors are in batch order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StepPlan {
    /// The token ids (`inp_tokens`).
    pub tokens: Vec<u32>,
    /// Each token's position (`inp_pos`).
    pub pos: Vec<u32>,
    /// The cell each token's latent row lands in: its position (`dsv4_raw_k_write_idxs`).
    pub raw_write: Vec<u32>,
    /// The first cell each token's window sees; the last is its own.
    pub raw_first: Vec<u32>,
    /// Cells the port's window mask covers: through the last token's, padded to [`KV_PAD`].
    pub raw_n_kv: usize,
    /// One per compressed stream, in [`Planner::stream_ratios`] order.
    pub streams: Vec<StreamStep>,
    /// The tokens whose logits the step returns: the last (`inp_out_ids`).
    pub out_ids: Vec<u32>,
    /// Per token, the engram n-gram ending at it, newest first: `[i * ngram + back]` is the
    /// token `back` positions before token `i` (`back` 0 is token `i` itself), `None` before
    /// the sequence starts. The engram row ids are hashed from these (the engram crate's
    /// `Hash::rows_into`, after its token map).
    pub engram_window: Vec<Option<u32>>,
}

impl StepPlan {
    /// Tokens in the step.
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    /// No step has been planned into this.
    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    /// The port's window mask (`KQ_mask_swa`, dumped as `dsv4_raw_mask_padded-<layer>`) as f16
    /// bits: a line of [`raw_n_kv`](Self::raw_n_kv) cells per token, the lines padded to a
    /// multiple of [`MASK_TOKEN_PAD`]; [`F16_VISIBLE`] on the cells a token sees,
    /// [`F16_HIDDEN`] everywhere else, the padding lines included.
    pub fn raw_mask_into(&self, out: &mut Vec<u16>) {
        out.clear();
        if self.is_empty() {
            return;
        }
        let lines = self.len().div_ceil(MASK_TOKEN_PAD) * MASK_TOKEN_PAD;
        out.resize(lines * self.raw_n_kv, F16_HIDDEN);
        for ((line, &first), &last) in out
            .chunks_mut(self.raw_n_kv)
            .zip(&self.raw_first)
            .zip(&self.pos)
        {
            line[first as usize..=last as usize].fill(F16_VISIBLE);
        }
    }

    fn clear(&mut self) {
        self.tokens.clear();
        self.pos.clear();
        self.raw_write.clear();
        self.raw_first.clear();
        self.raw_n_kv = 0;
        self.out_ids.clear();
        self.engram_window.clear();
    }
}

/// One compressed stream's part of a step.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StreamStep {
    /// Positions pooled into one compressed row.
    pub ratio: u32,
    /// Per token, the compressed rows it sees: `0 .. n_visible`.
    pub n_visible: Vec<u32>,
    /// Rows the port's mask for this stream covers: the most any token sees, padded to
    /// [`KV_PAD`].
    pub n_kv: usize,
    /// Per completed group in batch order, the `ratio` projections it pools, oldest first, as
    /// indices into the state ring (`0 .. ratio`) ⧺ this step's projections (`ratio ..`)
    /// (`dsv4_<s>_state_read`).
    pub state_read: Vec<u32>,
    /// Per completed group, the compressed row it writes (`dsv4_<s>_state_write`, an i64 in the
    /// port's graph).
    pub state_write: Vec<u64>,
    /// Per completed group, the position its row is roped at, its first
    /// (`dsv4_<s>_write_pos`).
    pub write_pos: Vec<u32>,
    /// The step's tokens whose projections the state ring keeps (`dsv4_<s>_persist_src`),
    /// paired with [`persist_dst`](Self::persist_dst).
    pub persist_src: Vec<u32>,
    /// The ring slot each kept projection lands in, ascending (`dsv4_<s>_persist_dst`).
    pub persist_dst: Vec<u32>,
}

impl StreamStep {
    /// Groups the step completes: the compressed rows it writes. The port builds its
    /// compressor only into a step with one; a captured graph runs it every step and has it
    /// skip on zero.
    pub fn groups(&self) -> usize {
        self.state_write.len()
    }

    /// The port's mask for this stream (`dsv4_<s>_kq_mask`) as f16 bits: a line of
    /// [`n_kv`](Self::n_kv) rows per token, no padding lines; [`F16_VISIBLE`] on the rows a
    /// token sees, [`F16_HIDDEN`] on the rest.
    pub fn kq_mask_into(&self, out: &mut Vec<u16>) {
        out.clear();
        if self.n_visible.is_empty() {
            return;
        }
        out.resize(self.n_visible.len() * self.n_kv, F16_HIDDEN);
        for (line, &n) in out.chunks_mut(self.n_kv).zip(&self.n_visible) {
            line[..n as usize].fill(F16_VISIBLE);
        }
    }

    /// This stream's part of the step at positions `pos0 ..= last`.
    fn plan(&mut self, ratio: u32, pos0: u32, last: u32) {
        self.ratio = ratio;
        self.n_visible.clear();
        self.state_read.clear();
        self.state_write.clear();
        self.write_pos.clear();
        self.persist_src.clear();
        self.persist_dst.clear();
        for p in pos0..=last {
            self.n_visible.push((p + 1) / ratio);
            if (p + 1) % ratio != 0 {
                continue;
            }
            let first = p + 1 - ratio;
            self.state_write.push(u64::from(p / ratio));
            self.write_pos.push(first);
            self.state_read.extend((first..=p).map(|q| {
                if q >= pos0 {
                    ratio + (q - pos0)
                } else {
                    q % ratio
                }
            }));
        }
        self.n_kv = padded(((last + 1) / ratio) as usize);
        for slot in 0..ratio {
            // The latest position up to `last` in this slot's residue, if the step holds it.
            let back = (last % ratio + ratio - slot) % ratio;
            if back <= last - pos0 {
                self.persist_src.push(last - back - pos0);
                self.persist_dst.push(slot);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The file's values as `just gate-ds41-plan` reads them from shard 1:
    /// `attention.sliding_window`, the first `block_count` (40) of
    /// `attention.compress_ratios`, and `engram.max_ngram_size`.
    const WINDOW: u32 = 128;
    const NGRAM: usize = 4;
    fn file_ratios() -> Vec<u32> {
        [[0; 2].as_slice(), &[2; 18], &[1; 20]].concat()
    }

    /// The walks' last position.
    const LAST: u32 = 2100;

    fn planner(ctx_max: u64) -> Planner {
        Planner::new(WINDOW, &file_ratios(), NGRAM, ctx_max)
            .expect("the file's values make a planner")
    }

    /// The walks' token sequence: ids from a fixed LCG.
    fn sequence(n: usize) -> Vec<u32> {
        let mut s = 0x2545_f491_u32;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                s >> 8
            })
            .collect()
    }

    /// What the caches hold after the steps so far, rebuilt from the plans alone.
    struct World {
        /// The position whose latent row each cell holds.
        cells: Vec<Option<u32>>,
        /// Per stream, the position whose projection each ring slot holds.
        rings: Vec<Vec<Option<u32>>>,
        /// Per stream, the first and last position each compressed row pooled.
        rows: Vec<Vec<Option<(u32, u32)>>>,
    }

    /// `n_kv` covers `needed` cells or rows the way the port pads them: a multiple of
    /// [`KV_PAD`], never fewer, and the smallest such.
    fn padding_holds(n_kv: usize, needed: usize) -> bool {
        let floor = needed.max(KV_PAD);
        n_kv.is_multiple_of(KV_PAD) && n_kv >= floor && n_kv < floor + KV_PAD
    }

    impl World {
        fn new(pl: &Planner) -> World {
            World {
                cells: Vec::new(),
                rings: pl
                    .stream_ratios()
                    .iter()
                    .map(|&r| vec![None; r as usize])
                    .collect(),
                rows: vec![Vec::new(); pl.stream_ratios().len()],
            }
        }

        /// Take one step's plan, checking every index against what the caches hold before it
        /// and every visibility against what they hold after its writes.
        fn step(&mut self, pl: &Planner, seq: &[u32], pos0: u32, plan: &StepPlan) {
            let t = plan.len();
            let positions: Vec<u32> = (pos0..).take(t).collect();
            let last = positions[t - 1];
            assert_eq!(plan.pos, positions, "positions at {pos0}");
            assert_eq!(plan.tokens, seq[pos0 as usize..][..t], "tokens at {pos0}");
            assert_eq!(
                plan.out_ids,
                [last - pos0],
                "the last token's logits at {pos0}"
            );

            // Raw window: a token writes its own cell and sees the cells holding the
            // positions less than the window before its own.
            for &c in &plan.raw_write {
                let c = c as usize;
                if self.cells.len() <= c {
                    self.cells.resize(c + 1, None);
                }
                self.cells[c] = Some(c as u32);
            }
            assert_eq!(plan.raw_write, positions, "a token's cell is its position");
            assert!(
                padding_holds(plan.raw_n_kv, last as usize + 1),
                "raw mask of {} cells through position {last}",
                plan.raw_n_kv
            );
            let mut mask = Vec::new();
            plan.raw_mask_into(&mut mask);
            assert_eq!(
                mask.len(),
                t.div_ceil(MASK_TOKEN_PAD) * MASK_TOKEN_PAD * plan.raw_n_kv,
                "raw mask lines at {pos0}"
            );
            for (j, line) in mask.chunks(plan.raw_n_kv).enumerate() {
                for (c, &v) in line.iter().enumerate() {
                    let sees = positions.get(j).is_some_and(|&p| {
                        self.cells
                            .get(c)
                            .copied()
                            .flatten()
                            .is_some_and(|q| q <= p && p - q < pl.window())
                    });
                    let want = if sees { F16_VISIBLE } else { F16_HIDDEN };
                    assert_eq!(v, want, "raw mask line {j} cell {c} at {pos0}");
                }
            }

            for (s, &r) in pl.stream_ratios().iter().enumerate() {
                self.stream(s, r, &positions, &plan.streams[s]);
            }

            let n = pl.ngram();
            assert_eq!(plan.engram_window.len(), t * n);
            for (i, &p) in positions.iter().enumerate() {
                for back in 0..n {
                    let want = (p as usize).checked_sub(back).map(|q| seq[q]);
                    assert_eq!(
                        plan.engram_window[i * n + back],
                        want,
                        "engram n-gram of position {p}, {back} back"
                    );
                }
            }
        }

        /// One stream's part: the reads of each completed group address the positions it
        /// pools, a token sees exactly the rows whose groups ended at or before it, and after
        /// the step each ring slot holds the latest position of its residue.
        fn stream(&mut self, s: usize, r: u32, positions: &[u32], st: &StreamStep) {
            let (pos0, last) = (positions[0], positions[positions.len() - 1]);
            let ring = &mut self.rings[s];
            let rows = &mut self.rows[s];
            assert_eq!(st.ratio, r);
            let completing: Vec<u32> = positions
                .iter()
                .copied()
                .filter(|p| p % r == r - 1)
                .collect();
            assert_eq!(
                st.groups(),
                completing.len(),
                "groups of ratio {r} at {pos0}"
            );
            assert_eq!(st.write_pos.len(), completing.len());
            assert_eq!(st.state_read.len(), completing.len() * r as usize);
            for (g, &p) in completing.iter().enumerate() {
                let pooled: Vec<Option<u32>> = st.state_read[g * r as usize..][..r as usize]
                    .iter()
                    .map(|&k| {
                        if k < r {
                            ring[k as usize]
                        } else {
                            positions.get((k - r) as usize).copied()
                        }
                    })
                    .collect();
                let want: Vec<Option<u32>> = (p + 1 - r..=p).map(Some).collect();
                assert_eq!(pooled, want, "ratio {r}: the group ending at {p} pools");
                assert_eq!(
                    st.write_pos[g],
                    p + 1 - r,
                    "ratio {r}: rope position of {p}'s group"
                );
                assert_eq!(
                    st.state_write[g],
                    u64::from(p / r),
                    "ratio {r}: row of {p}'s group"
                );
                let row = (p / r) as usize;
                if rows.len() <= row {
                    rows.resize(row + 1, None);
                }
                rows[row] = Some((p + 1 - r, p));
            }

            assert_eq!(st.n_visible.len(), positions.len());
            for (&p, &nv) in positions.iter().zip(&st.n_visible) {
                assert_eq!(nv, (p + 1) / r, "ratio {r}: rows position {p} sees");
                let nv = nv as usize;
                assert!(
                    rows[..nv]
                        .iter()
                        .all(|row| row.is_some_and(|(_, l)| l <= p)),
                    "ratio {r}: a row position {p} sees is unwritten or ends after it"
                );
                assert!(
                    rows.get(nv).copied().flatten().is_none_or(|(_, l)| l > p),
                    "ratio {r}: position {p} misses a row that ended before it"
                );
            }
            let seen = st.n_visible.iter().copied().max().unwrap_or(0) as usize;
            assert!(
                padding_holds(st.n_kv, seen),
                "ratio {r}: mask of {} rows",
                st.n_kv
            );
            let mut mask = Vec::new();
            st.kq_mask_into(&mut mask);
            assert_eq!(mask.len(), positions.len() * st.n_kv);
            for (line, &nv) in mask.chunks(st.n_kv).zip(&st.n_visible) {
                for (c, &v) in line.iter().enumerate() {
                    let want = if c < nv as usize {
                        F16_VISIBLE
                    } else {
                        F16_HIDDEN
                    };
                    assert_eq!(v, want, "ratio {r} mask row {c} at {pos0}");
                }
            }

            assert_eq!(st.persist_src.len(), st.persist_dst.len());
            assert!(
                st.persist_dst.windows(2).all(|w| w[0] < w[1]),
                "ratio {r}: kept slots ascend"
            );
            for (&src, &dst) in st.persist_src.iter().zip(&st.persist_dst) {
                ring[dst as usize] = Some(positions[src as usize]);
            }
            for (k, slot) in (0u32..).zip(ring.iter()) {
                let want = (0..=last).rev().find(|q| q % r == k);
                assert_eq!(*slot, want, "ratio {r}: slot {k} after position {last}");
            }
        }
    }

    /// Plan positions `0 ..= LAST` in steps of the sizes in `sizes` (cycled), taking each
    /// step through the world; the plans with their first positions.
    fn walk(pl: &Planner, seq: &[u32], sizes: &[usize]) -> Vec<(u32, StepPlan)> {
        let mut world = World::new(pl);
        let mut steps = Vec::new();
        let mut pos0 = 0u32;
        for &size in sizes.iter().cycle() {
            if pos0 > LAST {
                break;
            }
            let n = size.min((LAST - pos0 + 1) as usize);
            let mut plan = StepPlan::default();
            pl.plan_into(
                &seq[pos0 as usize..][..n],
                pos0,
                &seq[..pos0 as usize],
                &mut plan,
            )
            .expect("the walk's steps fit the context");
            world.step(pl, seq, pos0, &plan);
            steps.push((pos0, plan));
            pos0 += n as u32;
        }
        steps
    }

    /// Everything a walk planned that depends on a position and not on the steps it came in:
    /// per position the token, window and n-gram, and per stream the rows each position sees
    /// and each written row with its rope position, in order.
    fn by_position(steps: &[(u32, StepPlan)]) -> Vec<Vec<u64>> {
        let streams = steps.first().map_or(0, |(_, plan)| plan.streams.len());
        let mut out = vec![Vec::new(); 4 + 2 * streams];
        for (_, plan) in steps {
            out[0].extend(plan.pos.iter().map(|&v| u64::from(v)));
            out[1].extend(plan.tokens.iter().map(|&v| u64::from(v)));
            out[2].extend(plan.raw_first.iter().map(|&v| u64::from(v)));
            out[3].extend(
                plan.engram_window
                    .iter()
                    .map(|v| v.map_or(u64::MAX, u64::from)),
            );
            for (s, st) in plan.streams.iter().enumerate() {
                out[4 + 2 * s].extend(st.n_visible.iter().map(|&v| u64::from(v)));
                for (&row, &pos) in st.state_write.iter().zip(&st.write_pos) {
                    out[5 + 2 * s].extend([row, u64::from(pos)]);
                }
            }
        }
        out
    }

    /// A decode walk over positions `0 ..= 2100`, both streams at the file's ratios, checked
    /// against the caches it builds: cells are positions, the window hides `p − cell ≥ 128`, a
    /// group completes at `p % r == r − 1` and writes row `p / r` roped at `p + 1 − r` from the
    /// positions it pools, each ring slot keeps its residue's last token, a token sees
    /// `(p + 1) / r` rows, and every mask pads to 256. Three walks — a 5-token prefill then
    /// short batches mixed into single steps, single steps only, and one batch — must agree
    /// position by position.
    #[test]
    fn the_walk_matches_what_the_caches_hold() {
        let pl = planner(u64::from(LAST) + 1);
        let seq = sequence(LAST as usize + 1);
        let mixed = walk(&pl, &seq, &[5, 1, 1, 1, 2, 1, 3, 1, 1, 16, 1, 7, 1, 1, 1]);
        let single = walk(&pl, &seq, &[1]);
        let whole = walk(&pl, &seq, &[LAST as usize + 1]);
        assert_eq!(single.len(), LAST as usize + 1);
        assert_eq!(whole.len(), 1);
        assert_eq!(by_position(&mixed), by_position(&single));
        assert_eq!(by_position(&whole), by_position(&single));
    }

    /// The plan's integers do not depend on the context beyond its bound: the same steps under
    /// a tight and a roomy context are equal, and the tight one refuses the position past it.
    #[test]
    fn the_plan_does_not_depend_on_the_context() {
        let seq = sequence(LAST as usize + 2);
        let tight = planner(u64::from(LAST) + 1);
        let roomy = planner(1 << 20);
        let (mut a, mut b) = (StepPlan::default(), StepPlan::default());
        for (pos0, n) in [(0, 5), (4, 1), (127, 3), (301, 1), (1025, 1), (LAST, 1)] {
            let (tokens, before) = (&seq[pos0 as usize..][..n], &seq[..pos0 as usize]);
            tight
                .plan_into(tokens, pos0, before, &mut a)
                .expect("inside the tight context");
            roomy
                .plan_into(tokens, pos0, before, &mut b)
                .expect("inside the roomy context");
            assert_eq!(a, b, "step of {n} at {pos0}");
        }
        let past = LAST + 1;
        assert!(
            tight
                .plan_into(
                    &seq[past as usize..][..1],
                    past,
                    &seq[..past as usize],
                    &mut a
                )
                .is_err()
        );
    }

    /// Streams are the distinct nonzero ratios in layer order, and a layer reads the stream of
    /// its own ratio — the port's `csa` first, `hca` second.
    #[test]
    fn streams_follow_the_ratios_in_layer_order() {
        let pl = planner(4096);
        assert_eq!(pl.stream_ratios(), [2, 1]);
        for l in 0..40 {
            let want = match l {
                0 | 1 => None,
                2..=19 => Some(0),
                _ => Some(1),
            };
            assert_eq!(pl.layer_stream(l), want, "layer {l}");
        }
        assert_eq!(pl.layer_stream(40), None);
        let pl = Planner::new(WINDOW, &[0, 4, 128, 4, 0, 128], NGRAM, 4096).expect("planner");
        assert_eq!(pl.stream_ratios(), [4, 128]);
        let streams: Vec<_> = (0..6).map(|l| pl.layer_stream(l)).collect();
        assert_eq!(streams, [None, Some(0), Some(1), Some(0), None, Some(1)]);
    }

    /// What no plan can be made of is an error, not a guess.
    #[test]
    fn it_refuses_what_it_cannot_plan() {
        let ratios = file_ratios();
        assert!(
            Planner::new(0, &ratios, NGRAM, 4096).is_err(),
            "a window of 0"
        );
        assert!(
            Planner::new(WINDOW, &ratios, 0, 4096).is_err(),
            "an n-gram of 0"
        );
        assert!(
            Planner::new(WINDOW, &ratios, NGRAM, 0).is_err(),
            "no position"
        );
        assert!(
            Planner::new(WINDOW, &ratios, NGRAM, 1 << 31).is_err(),
            "past i32"
        );
        assert!(Planner::new(WINDOW, &ratios, NGRAM, (1 << 31) - 1).is_ok());
        let pl = planner(64);
        let mut plan = StepPlan::default();
        assert!(
            pl.plan_into(&[], 0, &[], &mut plan).is_err(),
            "an empty step"
        );
        assert!(
            pl.plan_into(&[1, 2], 63, &[0; 63], &mut plan).is_err(),
            "past the context"
        );
        assert!(
            pl.plan_into(&[1], 63, &[0; 63], &mut plan).is_ok(),
            "the context's last position"
        );
        assert!(
            pl.plan_into(&[1], 10, &[7, 8], &mut plan).is_err(),
            "an n-gram past the history"
        );
        assert!(
            pl.plan_into(&[1], 2, &[7, 8], &mut plan).is_ok(),
            "the whole, shorter history"
        );
    }

    /// The mask values are the engine's own f16 roundings of 0 and −∞.
    #[test]
    fn mask_values_are_the_f16_bits_of_zero_and_minus_infinity() {
        assert_eq!(gguf::quant::f32_to_f16_bits(0.0), F16_VISIBLE);
        assert_eq!(gguf::quant::f32_to_f16_bits(f32::NEG_INFINITY), F16_HIDDEN);
    }
}
