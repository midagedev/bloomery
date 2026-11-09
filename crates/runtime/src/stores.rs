//! The size rules of the stores a layer keeps beside its weights, one owner
//! for the plan that counts their bytes and the card body that allocates
//! them: the recurrent conv ring's rows, the PLE conv ring's rows, a window
//! layer's key ring ([`window_rows`], [`ring_rows`], [`Rows`],
//! [`cut_depth_keeps`]), the pooled plane's rows, a selecting layer's raw
//! index key ring ([`index_ring_rows`]), and each layer's bytes by
//! kind. Pure arithmetic over the file's widths; a card constant that must
//! equal one of these holds itself to it with a `const` assertion where it is
//! defined.

/// Bytes of an f16 value.
pub const F16_BYTES: u64 = 2;
/// Bytes of an f32 value.
pub const F32_BYTES: u64 = 4;
/// Bytes of a u32 word.
pub const U32_BYTES: u64 = 4;

/// u64 words of a delta replay table row: state, stamp, carry, layer and
/// record.
pub const DELTA_REPLAY_WORDS: usize = 5;
/// Device bytes of one delta replay table row.
pub const DELTA_REPLAY_BYTES: u64 = (DELTA_REPLAY_WORDS * std::mem::size_of::<u64>()) as u64;

/// The widest call a later call may roll back into, in positions: a ring
/// keeps its conv's reach plus this many inputs, so a cut anywhere inside a
/// pass of up to `PASS_ROWS` positions still finds every input its next
/// position reads.
pub const PASS_ROWS: usize = 8;

/// The columns a prompt reader and host batch port hold for a load: its
/// ubatch and at least one verify pass.
#[must_use]
pub const fn reader_columns(ubatch: u64) -> u64 {
    if ubatch < PASS_ROWS as u64 {
        PASS_ROWS as u64
    } else {
        ubatch
    }
}

/// Rows of a recurrent layer's conv ring: the reach of a `conv`-tap conv and
/// a pass.
///
/// # Panics
///
/// On a conv of no tap, by name.
#[must_use]
pub const fn conv_ring_rows(conv: usize) -> usize {
    assert!(conv >= 1, "conv_ring_rows: a conv of at least one tap");
    conv - 1 + PASS_ROWS
}

/// A recurrent layer's conv ring in bytes: [`conv_ring_rows`] rows of `channels` f32 values.
///
/// # Panics
///
/// On a conv of no tap, by name.
#[must_use]
pub const fn conv_ring_bytes(conv: usize, channels: u64) -> u64 {
    conv_ring_rows(conv) as u64 * channels * F32_BYTES
}

/// Rows of a PLE conv ring: the reach of `taps` taps `dilation` apart and a
/// pass.
///
/// # Panics
///
/// On a conv of no tap, by name.
#[must_use]
pub const fn ple_ring_rows(taps: usize, dilation: usize) -> usize {
    assert!(taps >= 1, "ple_ring_rows: a conv of at least one tap");
    (taps - 1) * dilation + PASS_ROWS
}

/// The positions one prompt pass appends before it reads, which a window ring
/// keeps room for beside its window. It equals `model::moe::UNION_MAX_COLS`,
/// the batch every card prompt walk takes; the plan that sizes a ring from
/// it holds the two equal with a `const` assertion.
pub const PROMPT_BATCH: u64 = 512;

/// The rows a window ring's length is padded up to a multiple of.
const RING_ALIGN: u64 = 256;

/// Rows a ring of `window` rows holds once `positions` positions have been
/// appended: every position until it fills, `window` after. The rows of a
/// ring that keeps exactly its window (the DSpark context ring) and the
/// clamp of [`ring_rows`]. Any row-count width: the callers hold theirs as
/// `usize` and `u64`.
#[must_use]
pub fn window_rows<T: Ord>(window: T, positions: T) -> T {
    positions.min(window)
}

/// Rows a window layer's ring holds at `ctx_max`: the window plus one prompt
/// batch ([`PROMPT_BATCH`]), padded up to a multiple of `RING_ALIGN`, and
/// never past `ctx_max`. A pass of up to a batch appends its rows before it
/// reads, so the window's oldest row survives it.
#[must_use]
pub fn ring_rows(window: u64, ctx_max: u64) -> u64 {
    let span = window.saturating_add(PROMPT_BATCH - 1);
    window_rows(span.div_ceil(RING_ALIGN) * RING_ALIGN, ctx_max)
}

/// How many rows a window layer holds. A full layer holds `ctx_max` under
/// either choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rows {
    /// The engine's default: [`ring_rows`] rows.
    Ring,
    /// Every position. The default-context search and the oracle sequence
    /// use this term.
    Plane,
}

impl Rows {
    /// Rows held under this choice. No window means a full layer.
    #[must_use]
    pub fn count(self, window: Option<u64>, ctx_max: u64) -> u64 {
        match (self, window) {
            (Self::Ring, Some(w)) => ring_rows(w, ctx_max),
            _ => ctx_max,
        }
    }
}

/// Whether a cut of `depth` positions (`pos - n`) still leaves the keys the
/// pass at `n` reads. A plane (`rows == ctx`) does. A ring holds
/// `[pos - rows, pos)` and that pass appends row `n` and reads
/// `[n - window + 1, n]`, so the cut is free when
/// `depth <= rows - window + 1`. The ring is at least the window plus one
/// prompt batch minus one, which is why a cut inside that slack writes
/// nothing.
#[must_use]
pub fn cut_depth_keeps(depth: u64, rows: u64, ctx: u64, window: u64) -> bool {
    if rows == ctx {
        return true;
    }
    depth <= rows.saturating_sub(window).saturating_add(1)
}

/// Rows of a pooled key plane over `ctx` positions in pools of `pool`, the
/// last pool counted whole. `pool` is at least one.
#[must_use]
pub const fn pooled_rows(ctx: usize, pool: usize) -> usize {
    ctx.div_ceil(pool)
}

/// A gated-delta-rule layer's state and conv ring over `v_heads` value heads
/// and `k_heads` key heads of `state` values with a `conv`-tap conv, in
/// bytes: `v_heads` heads of `state × state` f32 and the ring's rows of the
/// conv's channels, `2·k_heads·state + v_heads·state`, in f32.
#[must_use]
pub const fn recurrent_bytes(v_heads: usize, k_heads: usize, state: usize, conv: usize) -> u64 {
    let (v, k, d) = (v_heads as u64, k_heads as u64, state as u64);
    let channels = 2 * k * d + v * d;
    v * d * d * F32_BYTES + conv_ring_bytes(conv, channels)
}

/// A Mamba-2 layer's stores in f32 values, `[state, ring]`: `d_inner` channels
/// of `d_state` scan state, and the conv ring's rows
/// ([`conv_ring_rows`] of the `conv`-tap conv) of `conv_dim` channels. The
/// plan counts a layer's store by these lengths and the card body allocates
/// it by them.
#[must_use]
pub const fn mamba2_lens(
    d_inner: usize,
    d_state: usize,
    conv: usize,
    conv_dim: usize,
) -> [usize; 2] {
    [d_inner * d_state, conv_ring_rows(conv) * conv_dim]
}

/// A Mamba-2 layer's stores in bytes ([`mamba2_lens`], f32): what it holds
/// whatever the context, one sequence's.
#[must_use]
pub const fn mamba2_bytes(d_inner: usize, d_state: usize, conv: usize, conv_dim: usize) -> u64 {
    let [state, ring] = mamba2_lens(d_inner, d_state, conv, conv_dim);
    (state + ring) as u64 * F32_BYTES
}

/// Rows of a verify on a model whose gated-delta-rule layers record it
/// (Qwen3.8's `Body38`): each row's key, update and head decay go into the
/// layer's record ([`delta_record_bytes`]), and the commit replays the kept
/// rows into the state's one lane.
pub const DELTA_VERIFY_ROWS: usize = 4;

/// F32s of a gated-delta-rule layer's verify record over `v_heads` value
/// heads of `state` values for up to `rows` rows: per row and value head the
/// key, the update and the head decay, `3·state` f32 (the decay's word
/// padded to the key's alignment). `None` past `usize`.
#[must_use]
pub const fn delta_record_words(v_heads: usize, state: usize, rows: usize) -> Option<usize> {
    let Some(row) = v_heads.checked_mul(state) else {
        return None;
    };
    let Some(row) = row.checked_mul(3) else {
        return None;
    };
    rows.checked_mul(row)
}

/// Bytes of [`delta_record_words`].
///
/// # Panics
///
/// A record past `usize`, by name.
#[must_use]
pub const fn delta_record_bytes(v_heads: usize, state: usize, rows: usize) -> u64 {
    match delta_record_words(v_heads, state, rows) {
        Some(words) => words as u64 * F32_BYTES,
        None => panic!("delta_record_bytes: a record past usize"),
    }
}

/// Bytes a gated-delta-rule layer holds past [`recurrent_bytes`]' one state
/// lane when it keeps `lanes` lanes (at least one) over `v_heads` value
/// heads of `state × state` f32: the other `lanes − 1` lanes and a u32
/// stamp a lane (the position the lane's state stands at).
///
/// # Panics
///
/// On no lane, by name.
#[must_use]
pub const fn delta_lane_bytes(v_heads: usize, state: usize, lanes: usize) -> u64 {
    assert!(lanes >= 1, "delta_lane_bytes: a state of at least one lane");
    let (v, d, l) = (v_heads as u64, state as u64, lanes as u64);
    (l - 1) * v * d * d * F32_BYTES + l * U32_BYTES
}

/// Tokens of a chunk of the chunked gated delta rule a prompt runs on a
/// model with a carry (Qwen3.8's `Body38`).
pub const DELTA_CHUNK_ROWS: usize = 16;

/// Rows of a gated-delta-rule layer's carry: the tokens of the chunk a call
/// left out of the state, a chunk at most (a call that ends inside a chunk,
/// or a prompt's closing call that keeps the chunk its last id completes).
pub const DELTA_CARRY_ROWS: usize = DELTA_CHUNK_ROWS;

/// Words before a carry's rows: the rows' count, the position after them
/// and two spare words.
pub const DELTA_CARRY_HEAD: usize = 4;

/// Bytes of a gated-delta-rule layer's carry over `v_heads` value heads and
/// `k_heads` key heads of `state` values: [`DELTA_CARRY_HEAD`] words, then
/// [`DELTA_CARRY_ROWS`] rows of the key and value heads and each row's β
/// and decay a value head, in f32.
#[must_use]
pub const fn delta_carry_bytes(v_heads: usize, k_heads: usize, state: usize) -> u64 {
    let row = (k_heads + v_heads) * state + 2 * v_heads;
    (DELTA_CARRY_HEAD + DELTA_CARRY_ROWS * row) as u64 * F32_BYTES
}

/// A PLE conv ring of `taps` taps `dilation` apart over `streams` streams
/// of `n_embd` values, in bytes (f32).
#[must_use]
pub const fn ple_ring_bytes(taps: usize, dilation: usize, streams: usize, n_embd: usize) -> u64 {
    ple_ring_rows(taps, dilation) as u64 * (streams * n_embd) as u64 * F32_BYTES
}

/// A GQA position's K and V over `kv_heads` heads of `head_dim`, in f16.
#[must_use]
pub const fn kv_row_bytes(kv_heads: usize, head_dim: usize) -> u64 {
    2 * (kv_heads * head_dim) as u64 * F16_BYTES
}

/// A GQA position's K and V over `kv_heads` heads of `head_dim` in q8_0's
/// two-plane layout: per side a code byte and a f16 scale's half byte every
/// 32 values (`2 + 32·1 = 34` bytes a block, `17/16` a value), both planes
/// quantized together by the one format choice.
#[must_use]
pub const fn kv_q8_row_bytes(kv_heads: usize, head_dim: usize) -> u64 {
    assert!(head_dim.is_multiple_of(32), "a q8_0 block covers 32 values");
    2 * (kv_heads * head_dim) as u64 * 17 / 16
}

/// A GQA layer's K and V over `ctx` positions: [`kv_q8_row_bytes`] a position when `q8`, else
/// [`kv_row_bytes`].
#[must_use]
pub const fn kv_plane_bytes(kv_heads: usize, head_dim: usize, q8: bool, ctx: u64) -> u64 {
    ctx * if q8 {
        kv_q8_row_bytes(kv_heads, head_dim)
    } else {
        kv_row_bytes(kv_heads, head_dim)
    }
}

/// A dense attention layer's bytes at `ctx` positions: each position's K and
/// V ([`kv_row_bytes`]) and nothing else — the MTP draft layer's store, which
/// attends to every position it holds.
#[must_use]
pub const fn dense_kv_bytes(kv_heads: usize, head_dim: usize, ctx: usize) -> u64 {
    ctx as u64 * kv_row_bytes(kv_heads, head_dim)
}

/// Rows of a selecting layer's raw index key ring: the keys of a pool not
/// yet complete (`pool − 1` positions) and a pass, so a cut anywhere inside
/// a pass of up to [`PASS_ROWS`] positions still finds every key its next
/// pool reads.
///
/// # Panics
///
/// On a pool of no position, by name.
#[must_use]
pub const fn index_ring_rows(pool: usize) -> usize {
    assert!(
        pool >= 1,
        "index_ring_rows: a pool of at least one position"
    );
    pool - 1 + PASS_ROWS
}

/// A selecting layer's raw index key ring in bytes: [`index_ring_rows`] rows
/// of `idx_dim` f16 values.
#[must_use]
pub const fn index_ring_bytes(idx_dim: usize, pool: usize) -> u64 {
    index_ring_rows(pool) as u64 * idx_dim as u64 * F16_BYTES
}

/// A selecting attention layer's positional bytes at `ctx` positions: each
/// position's K and V ([`kv_plane_bytes`] in the layout `q8` names), and one
/// pooled key of `idx_dim` f16 a pool of `pool` positions ([`pooled_rows`]).
/// What a sequence state carries of the layer by position. The index keys
/// stay f16 in either layout.
#[must_use]
pub const fn selecting_rows_bytes(
    kv_heads: usize,
    head_dim: usize,
    q8: bool,
    idx_dim: usize,
    pool: usize,
    ctx: usize,
) -> u64 {
    kv_plane_bytes(kv_heads, head_dim, q8, ctx as u64)
        + pooled_rows(ctx, pool) as u64 * idx_dim as u64 * F16_BYTES
}

/// A selecting attention layer's bytes at `ctx` positions: its positional
/// rows ([`selecting_rows_bytes`]) and its raw index key ring
/// ([`index_ring_bytes`]).
#[must_use]
pub const fn selecting_bytes(
    kv_heads: usize,
    head_dim: usize,
    q8: bool,
    idx_dim: usize,
    pool: usize,
    ctx: usize,
) -> u64 {
    selecting_rows_bytes(kv_heads, head_dim, q8, idx_dim, pool, ctx)
        + index_ring_bytes(idx_dim, pool)
}

/// How a step reads a by-position buffer: every row (`Scan`) or a bounded list of rows (`Select`).
/// The architecture fact that decides where a buffer's pages may live: a scanned buffer is read
/// whole every step, so a host page would cost the step its whole size at the host link's rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Scan,
    Select { rows: u64 },
}

/// Where a by-position buffer's pages may live, read from its [`Access`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// The card only.
    Card,
    /// The card or the host.
    Host,
}

impl Access {
    /// A `Scan` buffer is card-only, a `Select` buffer may sit on the host.
    #[must_use]
    pub const fn class(self) -> Class {
        match self {
            Access::Scan => Class::Card,
            Access::Select { .. } => Class::Host,
        }
    }
}

/// One by-position buffer of a layer's store: `runs` runs of `rows(ctx)` rows of `row` bytes, one
/// row every `per` positions (a pooled key's row covers `per`). `runs` is the head-run count a
/// room pages each run of apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PosBuf {
    pub what: &'static str,
    pub row: u64,
    pub per: u64,
    pub runs: u64,
    pub access: Access,
}

impl PosBuf {
    /// Rows of one run over `positions` positions: the last partial row counted whole.
    #[must_use]
    pub fn rows(&self, positions: u64) -> u64 {
        positions.div_ceil(self.per)
    }

    /// One run's bytes over `positions` positions.
    #[must_use]
    pub fn run_bytes(&self, positions: u64) -> u64 {
        self.rows(positions).saturating_mul(self.row)
    }

    /// The buffer's exact bytes over `positions` positions.
    #[must_use]
    pub fn bytes(&self, positions: u64) -> u64 {
        self.run_bytes(positions).saturating_mul(self.runs)
    }
}

/// Every layer's by-position buffers, times the sequences that hold them; the fixed stores (rings,
/// recurrent and compressor states) are not in it. One declaration for the plan's floor and
/// padding terms and the card body's room constructors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PosLayout {
    layers: std::sync::Arc<[Vec<PosBuf>]>,
    slots: u64,
}

impl PosLayout {
    /// The layout of one sequence: `layers[l]` is layer `l`'s buffers.
    #[must_use]
    pub fn new(layers: Vec<Vec<PosBuf>>) -> PosLayout {
        PosLayout {
            layers: layers.into(),
            slots: 1,
        }
    }

    /// The same layout held by `slots` sequences: every byte count times `slots`.
    #[must_use]
    pub fn times(&self, slots: u64) -> PosLayout {
        PosLayout {
            layers: std::sync::Arc::clone(&self.layers),
            slots: self.slots.saturating_mul(slots),
        }
    }

    /// Layer `layer`'s buffers; none past the layers it names.
    #[must_use]
    pub fn buffers(&self, layer: usize) -> &[PosBuf] {
        self.layers.get(layer).map_or(&[], Vec::as_slice)
    }

    /// `layer`'s exact by-position bytes over `positions` positions.
    #[must_use]
    pub fn bytes(&self, layer: usize, positions: u64) -> u64 {
        self.sum(layer, |_| true, |b| b.bytes(positions))
    }

    /// The bytes of `layer`'s `Select` buffers over `positions` positions: the part a host-backed
    /// room holds off the card.
    #[must_use]
    pub fn host_able(&self, layer: usize, positions: u64) -> u64 {
        self.sum(
            layer,
            |b| b.access.class() == Class::Host,
            |b| b.bytes(positions),
        )
    }

    /// `layer`'s bytes over `positions` positions with every run rounded up to whole pages of
    /// `page(class)`, the class of a buffer being its [`Access::class`]; a buffer whose class
    /// has no page (`None`) holds none. A room backs whole pages run by run.
    #[must_use]
    pub fn paged(&self, layer: usize, positions: u64, page: impl Fn(Class) -> Option<u64>) -> u64 {
        self.sum(
            layer,
            |b| page(b.access.class()).is_some(),
            |b| {
                let page = page(b.access.class()).expect("filtered to the classes with a page");
                b.run_bytes(positions)
                    .div_ceil(page)
                    .saturating_mul(page)
                    .saturating_mul(b.runs)
            },
        )
    }

    /// The bytes `layer`'s buffers hold past their exact rows at `positions` positions when every
    /// buffer (its runs together) is rounded up to whole pages of `page(class)`: a class with no
    /// page (`None`) holds none and adds none. The rounding the plan reserves at the cap; `None`
    /// past u64.
    #[must_use]
    pub fn padding(
        &self,
        layer: usize,
        positions: u64,
        page: impl Fn(Class) -> Option<u64>,
    ) -> Option<u64> {
        let mut extra = 0u64;
        for b in self.buffers(layer) {
            let Some(page) = page(b.access.class()) else {
                continue;
            };
            let bytes = b.bytes(positions);
            let whole = bytes.div_ceil(page).checked_mul(page)?;
            extra = extra.checked_add(whole - bytes)?;
        }
        extra.checked_mul(self.slots)
    }

    /// `layer`'s `Select` buffers over `positions` positions, each buffer (its runs together)
    /// rounded up to whole pages of `page`, per sequence, then times the slots: what a host room holds
    /// once its sequences reach every position; `None` past u64.
    #[must_use]
    pub fn host_whole(&self, layer: usize, positions: u64, page: u64) -> Option<u64> {
        let mut whole = 0u64;
        for b in self.buffers(layer) {
            if b.access.class() != Class::Host {
                continue;
            }
            let bytes = b.bytes(positions);
            whole = whole.checked_add(bytes.div_ceil(page).checked_mul(page)?)?;
        }
        whole.checked_mul(self.slots)
    }

    /// Layers the layout names.
    #[must_use]
    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    fn sum(
        &self,
        layer: usize,
        keep: impl Fn(&PosBuf) -> bool,
        of: impl Fn(&PosBuf) -> u64,
    ) -> u64 {
        self.buffers(layer)
            .iter()
            .filter(|b| keep(b))
            .map(of)
            .fold(0u64, u64::saturating_add)
            .saturating_mul(self.slots)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DELTA_VERIFY_ROWS, PROMPT_BATCH, RING_ALIGN, Rows, conv_ring_bytes, conv_ring_rows,
        cut_depth_keeps, delta_carry_bytes, delta_lane_bytes, delta_record_bytes,
        delta_record_words, dense_kv_bytes, index_ring_bytes, index_ring_rows, kv_plane_bytes,
        kv_row_bytes, mamba2_bytes, mamba2_lens, ple_ring_bytes, ple_ring_rows, pooled_rows,
        recurrent_bytes, ring_rows, selecting_bytes, selecting_rows_bytes, window_rows,
    };

    /// A GQA layer's planes over `ctx` positions are `ctx` rows of the layout asked for: f16 K and V,
    /// or q8_0's 17/16 byte a value. Mutant: the two layouts swapped, or `ctx` not applied.
    #[test]
    fn a_gqa_layers_planes_are_ctx_rows_of_its_layout() {
        assert_eq!(kv_plane_bytes(4, 128, false, 32_768), 32_768 * 2048);
        assert_eq!(kv_plane_bytes(4, 128, true, 32_768), 32_768 * 1088);
        assert_eq!(kv_plane_bytes(4, 128, false, 0), 0);
    }

    /// A conv ring is `conv_ring_rows` rows of `channels` f32: a seven-tap conv over 2,048 channels
    /// holds 14 rows, 14 · 2,048 · 4 = 114,688 bytes; a four-tap conv over 10,240 channels 11
    /// rows, 450,560 bytes (the ring term of [`recurrent_bytes`]). Mutant: f16 bytes, or `conv`
    /// rows (the pass left out).
    #[test]
    fn a_conv_ring_is_its_rows_of_f32_channels() {
        assert_eq!(conv_ring_bytes(7, 2048), 114_688);
        assert_eq!(conv_ring_bytes(4, 10_240), 450_560);
        assert_eq!(
            recurrent_bytes(48, 16, 128, 4) - 48 * 128 * 128 * 4,
            conv_ring_bytes(4, 10_240)
        );
    }

    /// A Mamba-2 layer of 32 channels at state 128 with a 4-tap conv over
    /// 1,056 conv channels: 4,096 f32 of scan state and eleven conv rows.
    #[test]
    fn mamba2_sizes() {
        assert_eq!(mamba2_lens(32, 128, 4, 1056), [4096, 11 * 1056]);
        assert_eq!(mamba2_bytes(32, 128, 4, 1056), (4096 + 11 * 1056) * 4);
    }

    /// Qwen3.8's sizes: a GDN layer holds 48 heads of 128 × 128 f32,
    /// eleven conv rows of 10,240 f32 channels and a carry of four words and
    /// sixteen rows of 64 heads of 128 f32 and 2 × 48 f32; the PLE ring 17
    /// rows of four 2,560-value streams; an attention position 2 × 2 × 256 f16 of K and V
    /// (17/16 B a value in q8_0), a 128-value pooled key per four positions, the last pool
    /// whole, and eleven raw index key rows of 128 f16 whatever the context.
    #[test]
    fn qwen38_sizes() {
        assert_eq!(conv_ring_rows(4), 11);
        assert_eq!(ple_ring_rows(4, 3), 17);
        assert_eq!(recurrent_bytes(48, 16, 128, 4), 3_145_728 + 11 * 10_240 * 4);
        assert_eq!(delta_carry_bytes(48, 16, 128), (4 + 16 * (8192 + 96)) * 4);
        assert_eq!(ple_ring_bytes(4, 3, 4, 2560), 696_320);
        assert_eq!(kv_row_bytes(2, 256), 2048);
        assert_eq!(index_ring_rows(4), 11);
        assert_eq!(index_ring_bytes(128, 4), 2816);
        assert_eq!(
            selecting_rows_bytes(2, 256, false, 128, 4, 4096),
            4096 * 2048 + 1024 * 256
        );
        assert_eq!(
            selecting_bytes(2, 256, false, 128, 4, 4096),
            4096 * 2048 + 1024 * 256 + 2816
        );
        assert_eq!(
            selecting_bytes(2, 256, false, 128, 4, 4097),
            4097 * 2048 + 1025 * 256 + 2816
        );
        // q8_0: 2 · 2 · 256 values at 17/16 B, the index keys unchanged.
        assert_eq!(
            selecting_bytes(2, 256, true, 128, 4, 4096),
            4096 * 1088 + 1024 * 256 + 2816
        );
    }

    /// Qwen3.8's MTP layer: 2 × 2 × 256 f16 of K and V a position and no
    /// index key, at every position to the context.
    #[test]
    fn qwen38_mtp_store() {
        assert_eq!(dense_kv_bytes(2, 256, 4096), 8_388_608);
        assert_eq!(dense_kv_bytes(2, 256, 32_768), 67_108_864);
        assert_eq!(dense_kv_bytes(2, 256, 1), 2048);
        assert_eq!(dense_kv_bytes(2, 256, 0), 0);
    }

    /// Qwen3.8's delta store past the one lane `recurrent_bytes` counts: the
    /// lane's u32 stamp and a record of four verify rows, 48 heads of 3 × 128
    /// f32 a row.
    #[test]
    fn qwen38_lane_and_record() {
        assert_eq!(DELTA_VERIFY_ROWS, 4);
        assert_eq!(delta_lane_bytes(48, 128, 1), 4);
        assert_eq!(
            delta_record_bytes(48, 128, DELTA_VERIFY_ROWS),
            4 * 48 * 384 * 4
        );
        assert_eq!(delta_record_words(usize::MAX, 128, 1), None);
    }

    #[test]
    #[should_panic(expected = "delta_lane_bytes: a state of at least one lane")]
    fn a_state_of_no_lane_is_refused() {
        let _ = delta_lane_bytes(48, 128, std::hint::black_box(0));
    }

    /// A conv of no tap has no reach: both ring sizes refuse it by name
    /// rather than wrap to a ring of `usize::MAX` rows.
    #[test]
    #[should_panic(expected = "conv_ring_rows: a conv of at least one tap")]
    fn a_conv_of_no_tap_is_refused() {
        let _ = conv_ring_rows(std::hint::black_box(0));
    }

    #[test]
    #[should_panic(expected = "ple_ring_rows: a conv of at least one tap")]
    fn a_ple_conv_of_no_tap_is_refused() {
        let _ = ple_ring_rows(std::hint::black_box(0), 3);
    }

    /// A window ring holds the positions appended so far until it fills, then
    /// its window: counted out as the length of the last `window` positions of
    /// `0..positions`, over a sweep that crosses the fill point.
    #[test]
    fn window_rows_are_the_last_window_positions() {
        for window in 0u64..=20 {
            for positions in 0u64..=40 {
                let kept = (positions.saturating_sub(window)..positions).count() as u64;
                assert_eq!(
                    window_rows(window, positions),
                    kept,
                    "window {window}, {positions} positions"
                );
            }
        }
    }

    /// The padded ring is the least multiple of 256 that holds the window and
    /// a prompt batch's other 511 rows, clamped to the context. The oracle
    /// counts the multiples out with the numbers written in, so a change to
    /// either constant or an off-by-one in the span turns this red; the
    /// anchors are the published MiMo trunk's window (128 positions in 768
    /// rows) and the edge where 513 fills 1,024 rows exactly.
    #[test]
    fn ring_rows_pad_the_window_and_a_batch() {
        assert_eq!((PROMPT_BATCH, RING_ALIGN), (512, 256));
        let least = |window: u64| (0u64..).map(|k| k * 256).find(|&r| r >= window + 511);
        let contexts = [
            0u64,
            1,
            255,
            256,
            257,
            511,
            512,
            639,
            640,
            767,
            768,
            769,
            1_023,
            1_024,
            1_025,
            1_279,
            1_280,
            1_281,
            4_160,
            1 << 20,
            u64::MAX,
        ];
        for window in 1u64..=1_300 {
            let span = least(window).expect("a multiple of 256 reaches any window");
            for ctx in contexts {
                assert_eq!(
                    ring_rows(window, ctx),
                    ctx.min(span),
                    "window {window}, context {ctx}"
                );
            }
        }
        assert_eq!(ring_rows(128, 4_160), 768);
        assert_eq!(ring_rows(513, 1 << 20), 1_024);
        assert_eq!(ring_rows(514, 1 << 20), 1_280);
        assert_eq!(ring_rows(8, 9), 9, "a ring never passes the context");
    }

    /// The row choice: a window layer on the ring holds [`ring_rows`], on the
    /// plane every position; a full layer holds the context either way.
    #[test]
    fn rows_choose_ring_or_plane() {
        assert_eq!(Rows::Ring.count(Some(128), 4_160), 768);
        assert_eq!(Rows::Plane.count(Some(128), 4_160), 4_160);
        assert_eq!(Rows::Ring.count(None, 4_160), 4_160);
        assert_eq!(Rows::Plane.count(None, 4_160), 4_160);
    }

    /// A ring cut keeps `n` through `rows - window + 1` positions and refuses
    /// the next one. A plane (`rows == ctx`) keeps a cut of any depth. At a
    /// context the ring still fills, the two are the same term.
    #[test]
    fn a_ring_cut_past_the_slack_is_refused() {
        let window = 128;
        let ctx = 4_160;
        let rows = ring_rows(window, ctx);
        assert_eq!(
            rows,
            (window + PROMPT_BATCH - 1).div_ceil(RING_ALIGN) * RING_ALIGN
        );
        assert!(rows < ctx, "the ring is shorter than this context");
        let slack = rows - window + 1;
        assert!(cut_depth_keeps(slack, rows, ctx, window));
        assert!(!cut_depth_keeps(slack + 1, rows, ctx, window));
        assert!(cut_depth_keeps(slack + 1, ctx, ctx, window));
        let filled = ring_rows(8, 9);
        assert_eq!(filled, 9);
        assert!(cut_depth_keeps(1_000, filled, 9, 8));
    }

    /// A feed of 1,600 leaves the written end at 1,600. The cut to 960 is
    /// depth 640, inside the slack. The end stays 1,600, so the cut to 950
    /// is depth 650 and is refused. Depth 10, measured from 960, is the
    /// overwritten-key reading.
    #[test]
    fn a_second_cut_is_measured_from_the_written_end() {
        let window = 128;
        let ctx = 4_160;
        let rows = ring_rows(window, ctx);
        let written = 1_600;
        let first = 960;
        let second = 950;
        assert!(cut_depth_keeps(written - first, rows, ctx, window));
        assert!(!cut_depth_keeps(written - second, rows, ctx, window));
        assert!(cut_depth_keeps(first - second, rows, ctx, window));
    }

    /// The pooled plane counts a partial last pool whole and nothing past it.
    #[test]
    fn pooled_rows_round_up() {
        for (ctx, want) in [
            (0usize, 0usize),
            (1, 1),
            (4, 1),
            (5, 2),
            (8, 2),
            (2051, 513),
        ] {
            assert_eq!(pooled_rows(ctx, 4), want, "ctx {ctx}");
        }
    }
}

#[cfg(test)]
mod pos_layout_tests {
    use super::{Access, Class, PosBuf, PosLayout, kv_plane_bytes, selecting_rows_bytes};

    const MIB2: u64 = 2 << 20;

    /// One selecting layer as Qwen3.8 holds it: K and V planes of `[ctx][head]` runs, one run a
    /// kv head, f16 or q8_0 (data and scales), and a pooled key a pool of 4, scanned.
    fn selecting_layer(q8: bool) -> Vec<PosBuf> {
        let select = Access::Select { rows: 2_051 };
        let plane = |what, row| PosBuf {
            what,
            row,
            per: 1,
            runs: 2,
            access: select,
        };
        let mut bufs = if q8 {
            vec![
                plane("k data", 256),
                plane("k scales", 16),
                plane("v data", 256),
                plane("v scales", 16),
            ]
        } else {
            vec![plane("k", 512), plane("v", 512)]
        };
        bufs.push(PosBuf {
            what: "pooled",
            row: 256,
            per: 4,
            runs: 1,
            access: Access::Scan,
        });
        bufs
    }

    fn layout(q8: bool) -> PosLayout {
        PosLayout::new(vec![vec![], selecting_layer(q8)])
    }

    /// The descriptor's bytes are the size rules' bytes in both cache layouts, its host-able part
    /// is the planes alone, and a layer it does not name is empty.
    #[test]
    fn bytes_and_host_able_follow_the_size_rules() {
        for q8 in [false, true] {
            let l = layout(q8);
            for ctx in [0u64, 1, 4, 5, 2_051, 8_192, 262_144] {
                assert_eq!(
                    l.bytes(1, ctx),
                    selecting_rows_bytes(2, 256, q8, 128, 4, ctx as usize),
                    "q8 {q8} ctx {ctx}"
                );
                assert_eq!(l.host_able(1, ctx), kv_plane_bytes(2, 256, q8, ctx));
            }
            assert_eq!((l.bytes(0, 4096), l.bytes(7, 4096)), (0, 0));
        }
    }

    /// The class is the access fact: scanned is card-only, selected may sit on the host.
    #[test]
    fn a_scanned_buffer_is_card_class_and_a_selected_one_host_class() {
        assert_eq!(Access::Scan.class(), Class::Card);
        assert_eq!(Access::Select { rows: 1 }.class(), Class::Host);
    }

    /// A room backs whole pages run by run: at 8,192 positions each 512 B plane run is 4 MiB, 2
    /// pages, and the pooled run (2,048 rows of 256 B = 0.5 MiB) one page, the design's 226.5 MB a
    /// 12-layer slot.
    #[test]
    fn paged_rounds_each_run_not_the_buffer() {
        let l = layout(false);
        let page = |_| Some(MIB2);
        // K and V, 2 runs each, 2 pages a run, and one pooled page.
        assert_eq!(l.paged(1, 8_192, page), (2 * 2 * 2 + 1) * MIB2);
        assert_eq!(l.paged(1, 8_192, page) * 12, 226_492_416);
        // A class with no page holds none: the host classes off the card.
        let card_only = |c| (c == Class::Card).then_some(MIB2);
        assert_eq!(l.paged(1, 8_192, card_only), MIB2);
        // The runs of a buffer round apart: two runs of 3 MiB are 4 MiB each, though the buffer's
        // 6 MiB is whole pages (no padding).
        let odd = PosLayout::new(vec![vec![PosBuf {
            what: "odd",
            row: 3 << 20,
            per: 1,
            runs: 2,
            access: Access::Scan,
        }]]);
        assert_eq!(odd.paged(0, 1, |_| Some(MIB2)), 8 << 20);
        assert_eq!(odd.padding(0, 1, |_| Some(MIB2)), Some(0));
    }

    /// The padding rounds each buffer, its runs together (the plan's reserve at the cap), per
    /// sequence and then times the slots; a class with no page adds none.
    #[test]
    fn padding_rounds_each_buffer_per_sequence() {
        let l = layout(false);
        let page = |_| Some(MIB2);
        // ctx 5: K = 5 · 1,024 B → 2 MiB − 5,120; V the same; pooled 2 rows · 256 B.
        let want = 2 * (MIB2 - 5 * 1_024) + (MIB2 - 2 * 256);
        assert_eq!(l.padding(1, 5, page), Some(want));
        assert_eq!(l.times(3).padding(1, 5, page), Some(3 * want));
        let card_only = |c| (c == Class::Card).then_some(MIB2);
        assert_eq!(l.padding(1, 5, card_only), Some(MIB2 - 2 * 256));
        assert_eq!(l.padding(1, 5, |_| None), Some(0));
        assert_eq!(l.times(u64::MAX).padding(1, 5, page), None);
    }

    /// The host room's whole bytes: the selected buffers rounded per buffer per sequence, times the
    /// slots; the scanned ones none.
    #[test]
    fn host_whole_counts_the_selected_buffers_only() {
        let l = layout(false);
        // ctx 5: K and V each 5,120 B of 2 runs → one page apiece.
        assert_eq!(l.host_whole(1, 5, MIB2), Some(2 * MIB2));
        assert_eq!(l.times(4).host_whole(1, 5, MIB2), Some(8 * MIB2));
        assert_eq!(l.host_whole(0, 5, MIB2), Some(0));
    }

    /// Slots scale every byte count and compose.
    #[test]
    fn times_scales_every_count_and_composes() {
        let l = layout(false);
        let two = l.times(2).times(3);
        assert_eq!(two.bytes(1, 4_096), 6 * l.bytes(1, 4_096));
        assert_eq!(two.host_able(1, 4_096), 6 * l.host_able(1, 4_096));
        assert_eq!(l.times(u64::MAX).bytes(1, 4_096), u64::MAX);
    }
}
