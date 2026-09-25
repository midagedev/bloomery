//! The step image: every value one decode step's chain reads that changes
//! from one step to the next, in one host buffer that the body uploads with a
//! single host-to-device copy before a launch, never inside a capture. The
//! captured chain reads it at the fixed offsets [`ImageLayout`] computes once
//! at load.
//!
//! The image copies; it does not plan. Its integers are the host step plan's
//! (`model::arch::deepseek41::plan`, the one owner of that arithmetic), its
//! rope tables are [`RopeTable::push`]'s, and its embedding and engram rows
//! are the caller's.
//!
//! Layout, in little-endian u32 words, every section starting on a 16-byte
//! boundary:
//!
//! 1. **tokens** — per token: its id, its position, the slot of the window
//!    ring its latent row lands in (its cell modulo the window) and the
//!    length of the window it sees (its position − the first cell it sees +
//!    1).
//! 2. **streams** — per compressed stream, in the plan's order: the groups the
//!    step completes and the ring slots it keeps (two words, then two zero
//!    words); then, each at its capacity for the step's token count,
//!    `n_visible` per token, `state_read` (`ratio` per group), `state_write`,
//!    `write_pos`, `persist_src` and `persist_dst`. A slot past a count holds
//!    zero.
//! 3. **rope** — per token, the [`Table::ALL`] tables at its position; then,
//!    per stream whose ratio exceeds one, per group slot, the YaRN table at
//!    the group's write position, zero in a slot the step does not complete.
//!    A stream of ratio one pools a token into its own row, so its rows are
//!    roped at the token's position: that token's YaRN table.
//! 4. **embedding** — per token, the file's `token_embd` row as the file
//!    stores it (bf16 values, or Q3_K blocks), four bytes to a word.
//! 5. **engram** — per token, every site's gathered rows in site order, as the
//!    table holds them, four bytes to a word.

use bloomery_gpu::GpuError;
use model::arch::deepseek41::hparams::{Hparams, Rope};
use model::arch::deepseek41::plan::{Planner, StepPlan};

use crate::rope::{Direction, RopeSpec, RopeTable};

const WHAT: &str = "StepImage";

/// Words per token in the tokens section: id, position, ring slot, window
/// length.
const TOKEN_WORDS: usize = 4;

/// Words of a stream's header: groups and kept slots, then two zero words.
const STREAM_HEADER: usize = 4;

/// Sections start on a multiple of this many words.
const ALIGN: usize = 4;

/// The rope tables every token carries, in image order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Table {
    /// The window-only layers' rope at the token's position: their query
    /// heads and their latent row.
    WindowForward,
    /// Its inverse, which turns those layers' attention output back.
    WindowBack,
    /// The compressed layers' YaRN rope at the token's position: their query
    /// heads, their latent row, the indexer query, and a ratio-one stream's
    /// row and index key.
    YarnForward,
    /// Its inverse, which turns those layers' attention output back.
    YarnBack,
}

impl Table {
    /// Every table a token carries, in image order. The rope kernels take a
    /// table with its direction folded into the sines, so each inverse is a
    /// table of its own.
    pub const ALL: [Table; 4] = [
        Table::WindowForward,
        Table::WindowBack,
        Table::YarnForward,
        Table::YarnBack,
    ];

    /// Its place among a token's tables.
    fn index(self) -> usize {
        match self {
            Table::WindowForward => 0,
            Table::WindowBack => 1,
            Table::YarnForward => 2,
            Table::YarnBack => 3,
        }
    }

    fn direction(self) -> Direction {
        match self {
            Table::WindowForward | Table::YarnForward => Direction::Forward,
            Table::WindowBack | Table::YarnBack => Direction::Back,
        }
    }
}

/// What an image is laid out from: the tokens one step runs and the file's
/// sizes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageDims {
    /// Tokens per step.
    pub tokens: usize,
    /// `attention.sliding_window`: the ring's slots.
    pub window: u32,
    /// The compressed streams' ratios, in the plan's stream order.
    pub stream_ratios: Vec<u32>,
    /// Values per rope table: `rope.dimension_count`.
    pub rope_dims: usize,
    /// Values per embedding row.
    pub n_embd: usize,
    /// Bytes of one token's embedding row, as the file stores it.
    pub embd_bytes: usize,
    /// Engram bytes per token: every site's gathered rows.
    pub engram_bytes: usize,
}

impl ImageDims {
    /// The dims of a step of `tokens` tokens of the model `hp` describes,
    /// with `planner`'s streams and window, embedding rows of `token_embd`'s
    /// type ([`Hparams::rows`]) and engram table rows of `engram_row_bytes`
    /// bytes. A type with no block size gives an embedding row of 0 bytes,
    /// which [`ImageLayout::new`] refuses.
    #[must_use]
    pub fn of(hp: &Hparams, planner: &Planner, tokens: usize, engram_row_bytes: usize) -> Self {
        let ty = hp.rows.token_embd;
        let embd_bytes = ty
            .blck_size()
            .zip(ty.type_size())
            .filter(|&(b, _)| b > 0 && (hp.n_embd as u64).is_multiple_of(b))
            .map_or(0, |(b, t)| hp.n_embd / b as usize * t as usize);
        ImageDims {
            tokens,
            window: planner.window(),
            stream_ratios: planner.stream_ratios().to_vec(),
            rope_dims: hp.rope_dims,
            n_embd: hp.n_embd,
            embd_bytes,
            engram_bytes: hp.engram.layer_ids.len() * hp.engram.rows_per_token() * engram_row_bytes,
        }
    }
}

/// One stream's block in the streams section.
#[derive(Clone, Debug)]
pub struct StreamLayout {
    /// Positions pooled into one compressed row.
    pub ratio: u32,
    /// Groups one step can complete: `⌈tokens / ratio⌉`.
    pub group_slots: usize,
    /// Ring slots one step can keep: `min(ratio, tokens)`.
    pub persist_slots: usize,
    /// Word offset of the block.
    at: usize,
    /// Word offset of its row tables, for a ratio above one.
    row_tables: Option<usize>,
}

impl StreamLayout {
    fn n_visible_at(&self) -> usize {
        self.at + STREAM_HEADER
    }

    fn state_read_at(&self, tokens: usize) -> usize {
        self.n_visible_at() + tokens
    }

    fn state_write_at(&self, tokens: usize) -> usize {
        self.state_read_at(tokens) + self.group_slots * self.ratio as usize
    }

    fn write_pos_at(&self, tokens: usize) -> usize {
        self.state_write_at(tokens) + self.group_slots
    }

    fn persist_src_at(&self, tokens: usize) -> usize {
        self.write_pos_at(tokens) + self.group_slots
    }

    fn persist_dst_at(&self, tokens: usize) -> usize {
        self.persist_src_at(tokens) + self.persist_slots
    }

    fn end(&self, tokens: usize) -> usize {
        self.persist_dst_at(tokens) + self.persist_slots
    }
}

/// Where every field of a step's image lies, fixed at load: the captured
/// chain's kernels read these offsets, so they never move between steps.
#[derive(Clone, Debug)]
pub struct ImageLayout {
    dims: ImageDims,
    streams: Vec<StreamLayout>,
    rope_at: usize,
    embd_at: usize,
    /// Words of one token's embedding row.
    embd_words: usize,
    engram_at: usize,
    /// Words of one token's engram rows.
    engram_words: usize,
    words: usize,
}

impl ImageLayout {
    /// The layout of `dims`. A step runs at least one token and no more than
    /// the window has slots — the ring append writes one slot per token, so a
    /// longer step would write a slot twice; the rope tables hold cos/sin
    /// pairs, and an embedding row is some bytes.
    pub fn new(dims: ImageDims) -> Result<ImageLayout, GpuError> {
        let refuse = |detail: String| GpuError::Shape {
            what: "ImageLayout::new",
            detail,
        };
        let m = dims.tokens;
        if m == 0 || m > dims.window as usize {
            return Err(refuse(format!(
                "a step of {m} tokens: at least one, and at most the window's {} slots",
                dims.window
            )));
        }
        if dims.rope_dims < 2 || !dims.rope_dims.is_multiple_of(2) {
            return Err(refuse(format!(
                "{} rope values per table: cos/sin pairs",
                dims.rope_dims
            )));
        }
        if dims.embd_bytes == 0 {
            return Err(refuse(format!(
                "an embedding row of {} values in no bytes: its type has no block that tiles it",
                dims.n_embd
            )));
        }
        let mut at = align(TOKEN_WORDS * m);
        let mut streams = Vec::with_capacity(dims.stream_ratios.len());
        for &ratio in &dims.stream_ratios {
            if ratio == 0 {
                return Err(refuse("a stream of ratio 0 pools nothing".to_string()));
            }
            let s = StreamLayout {
                ratio,
                group_slots: m.div_ceil(ratio as usize),
                persist_slots: m.min(ratio as usize),
                at,
                row_tables: None,
            };
            at = align(s.end(m));
            streams.push(s);
        }
        let rope_at = at;
        at += m * Table::ALL.len() * dims.rope_dims;
        for s in streams.iter_mut().filter(|s| s.ratio > 1) {
            s.row_tables = Some(at);
            at += s.group_slots * dims.rope_dims;
        }
        let embd_at = align(at);
        let embd_words = dims.embd_bytes.div_ceil(4);
        let engram_at = align(embd_at + m * embd_words);
        let engram_words = dims.engram_bytes.div_ceil(4);
        let words = align(engram_at + m * engram_words);
        Ok(ImageLayout {
            dims,
            streams,
            rope_at,
            embd_at,
            embd_words,
            engram_at,
            engram_words,
            words,
        })
    }

    /// What the layout was made from.
    #[must_use]
    pub fn dims(&self) -> &ImageDims {
        &self.dims
    }

    /// The streams' blocks, in the plan's order.
    #[must_use]
    pub fn streams(&self) -> &[StreamLayout] {
        &self.streams
    }

    /// Words in the image.
    #[must_use]
    pub fn words(&self) -> usize {
        self.words
    }

    /// Bytes in the image: what one refresh uploads.
    #[must_use]
    pub fn bytes(&self) -> usize {
        4 * self.words
    }

    /// Word of token `t`'s rope table `table`.
    #[must_use]
    pub fn table_at(&self, t: usize, table: Table) -> usize {
        self.rope_at + (t * Table::ALL.len() + table.index()) * self.dims.rope_dims
    }

    /// Word of token `t`'s id; its position, ring slot and window length
    /// follow it.
    #[must_use]
    pub fn token_at(&self, t: usize) -> usize {
        TOKEN_WORDS * t
    }

    /// Word of stream `s`'s visible counts, one per token; `None` past the
    /// streams.
    #[must_use]
    pub fn n_visible_at(&self, s: usize) -> Option<usize> {
        self.streams.get(s).map(StreamLayout::n_visible_at)
    }

    /// Word of token `t`'s embedding row.
    #[must_use]
    pub fn embd_at(&self, t: usize) -> usize {
        self.embd_at + t * self.embd_words
    }

    /// Words of one token's embedding row.
    #[must_use]
    pub fn embd_words(&self) -> usize {
        self.embd_words
    }

    /// Word of token `t`'s engram rows.
    #[must_use]
    pub fn engram_at(&self, t: usize) -> usize {
        self.engram_at + t * self.engram_words
    }

    /// `words` — an image of this layout, a host copy or one read back from
    /// the card — seen field by field.
    pub fn view<'a>(&'a self, words: &'a [u32]) -> Result<ImageView<'a>, GpuError> {
        if words.len() != self.words {
            return Err(GpuError::Shape {
                what: "ImageLayout::view",
                detail: format!("{} words, the layout holds {}", words.len(), self.words),
            });
        }
        Ok(ImageView {
            layout: self,
            words,
        })
    }
}

/// `n` words rounded up to a section boundary.
fn align(n: usize) -> usize {
    n.next_multiple_of(ALIGN)
}

/// One token's fields in the tokens section.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenFields {
    pub token: u32,
    pub pos: u32,
    pub slot: u32,
    pub len: u32,
}

/// One stream's fields, each cut to the counts the image holds.
#[derive(Clone, Copy, Debug)]
pub struct StreamFields<'a> {
    pub groups: u32,
    pub persists: u32,
    pub n_visible: &'a [u32],
    pub state_read: &'a [u32],
    pub state_write: &'a [u32],
    pub write_pos: &'a [u32],
    pub persist_src: &'a [u32],
    pub persist_dst: &'a [u32],
}

/// An image read field by field at its layout's offsets.
#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a> {
    layout: &'a ImageLayout,
    words: &'a [u32],
}

impl<'a> ImageView<'a> {
    fn span(&self, at: usize, len: usize) -> &'a [u32] {
        &self.words[at..at + len]
    }

    /// Token `t`'s id, position, ring slot and window length.
    #[must_use]
    pub fn token(&self, t: usize) -> TokenFields {
        let w = self.span(TOKEN_WORDS * t, TOKEN_WORDS);
        TokenFields {
            token: w[0],
            pos: w[1],
            slot: w[2],
            len: w[3],
        }
    }

    /// Stream `s`'s fields, cut to the counts in its header — at most its
    /// capacities.
    #[must_use]
    pub fn stream(&self, s: usize) -> StreamFields<'a> {
        let l = &self.layout.streams[s];
        let m = self.layout.dims.tokens;
        let groups = (self.words[l.at] as usize).min(l.group_slots);
        let kept = (self.words[l.at + 1] as usize).min(l.persist_slots);
        StreamFields {
            groups: self.words[l.at],
            persists: self.words[l.at + 1],
            n_visible: self.span(l.n_visible_at(), m),
            state_read: self.span(l.state_read_at(m), groups * l.ratio as usize),
            state_write: self.span(l.state_write_at(m), groups),
            write_pos: self.span(l.write_pos_at(m), groups),
            persist_src: self.span(l.persist_src_at(m), kept),
            persist_dst: self.span(l.persist_dst_at(m), kept),
        }
    }

    /// Token `t`'s `table`, as f32 bits.
    #[must_use]
    pub fn table(&self, t: usize, table: Table) -> &'a [u32] {
        self.span(self.layout.table_at(t, table), self.layout.dims.rope_dims)
    }

    /// Stream `s`'s row table for group slot `g`, as f32 bits; `None` for a
    /// stream of ratio one, whose rows take the token's YaRN table.
    #[must_use]
    pub fn row_table(&self, s: usize, g: usize) -> Option<&'a [u32]> {
        let l = &self.layout.streams[s];
        let dims = self.layout.dims.rope_dims;
        l.row_tables
            .filter(|_| g < l.group_slots)
            .map(|at| self.span(at + g * dims, dims))
    }

    /// Token `t`'s embedding row, four bytes to a word.
    #[must_use]
    pub fn embd(&self, t: usize) -> &'a [u32] {
        let n = self.layout.embd_words;
        self.span(self.layout.embd_at + t * n, n)
    }

    /// Token `t`'s engram rows, four bytes to a word.
    #[must_use]
    pub fn engram(&self, t: usize) -> &'a [u32] {
        let n = self.layout.engram_words;
        self.span(self.layout.engram_at + t * n, n)
    }
}

/// The rope spec of `rope` — one of the file's two, as `Hparams` read it —
/// over `n_dims` values.
pub fn rope_spec(rope: &Rope, n_dims: usize) -> Result<RopeSpec, GpuError> {
    let n_ctx_orig = i32::try_from(rope.n_ctx_orig).map_err(|_| GpuError::Shape {
        what: "rope_spec",
        detail: format!("original context {} passes i32", rope.n_ctx_orig),
    })?;
    Ok(RopeSpec {
        n_dims,
        freq_base: rope.base,
        freq_scale: rope.freq_scale,
        ext_factor: rope.ext_factor,
        attn_factor: rope.attn_factor,
        beta_fast: rope.beta_fast,
        beta_slow: rope.beta_slow,
        n_ctx_orig,
    })
}

/// The window-only layers' rope and the compressed layers' rope, each the
/// one every layer of its kind shares; a model that lacks either kind, or
/// whose layers of one kind disagree, is refused.
pub fn rope_specs(hp: &Hparams) -> Result<(RopeSpec, RopeSpec), GpuError> {
    let one = |compressed: bool| -> Result<RopeSpec, GpuError> {
        let mut kind = hp
            .layers
            .iter()
            .filter(|k| k.stream.is_some() == compressed)
            .map(|k| k.rope);
        let name = if compressed {
            "compressed"
        } else {
            "window-only"
        };
        let first = kind.next().ok_or_else(|| GpuError::Shape {
            what: "rope_specs",
            detail: format!("the model has no {name} layer to take that rope from"),
        })?;
        if kind.any(|r| r != first) {
            return Err(GpuError::Shape {
                what: "rope_specs",
                detail: format!("the {name} layers do not share one rope"),
            });
        }
        rope_spec(&first, hp.rope_dims)
    };
    Ok((one(false)?, one(true)?))
}

/// One step's image on the host, and what builds it: the layout, the two
/// ropes' tables, and the buffers a build reuses, so a steady step allocates
/// nothing.
#[derive(Clone, Debug)]
pub struct StepImage {
    layout: ImageLayout,
    window: RopeTable,
    yarn: RopeTable,
    words: Vec<u32>,
    table: Vec<f32>,
    /// The first position of the step the words hold, once built.
    pos: Option<u32>,
}

impl StepImage {
    /// An image of `layout`, with the window-only layers' rope `window` and
    /// the compressed layers' rope `yarn`; every word zero until the first
    /// build.
    pub fn new(
        layout: ImageLayout,
        window: &RopeSpec,
        yarn: &RopeSpec,
    ) -> Result<StepImage, GpuError> {
        let dims = layout.dims.rope_dims;
        let (window, yarn) = (RopeTable::new(window)?, RopeTable::new(yarn)?);
        if window.n_dims() != dims || yarn.n_dims() != dims {
            return Err(GpuError::Shape {
                what: "StepImage::new",
                detail: format!(
                    "tables of {} and {} values, the layout's hold {dims}",
                    window.n_dims(),
                    yarn.n_dims()
                ),
            });
        }
        Ok(StepImage {
            words: vec![0; layout.words],
            table: Vec::with_capacity(dims),
            layout,
            window,
            yarn,
            pos: None,
        })
    }

    /// The layout the words follow.
    #[must_use]
    pub fn layout(&self) -> &ImageLayout {
        &self.layout
    }

    /// The image as built: what a refresh uploads.
    #[must_use]
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// The first position of the step the image holds; `None` before the
    /// first build and after a refused one.
    #[must_use]
    pub fn pos(&self) -> Option<u32> {
        self.pos
    }

    /// Build the image of `plan`'s step, with `embd` (the step's embedding
    /// rows, `tokens · embd_bytes` bytes) and `engram` (its engram rows,
    /// `tokens · engram_bytes` bytes). The step runs `1 ..=` the layout's
    /// tokens: a prompt batch's last chunk may be short, and the words past
    /// its tokens stay zero — its launches run its own token count. The
    /// step's positions must be consecutive — the window ring's append gives
    /// each token its own slot only then — and every count must fit the
    /// layout; a refused build leaves no step in the image.
    pub fn build(&mut self, plan: &StepPlan, embd: &[u8], engram: &[u8]) -> Result<(), GpuError> {
        self.pos = None;
        let refuse = |detail: String| GpuError::Shape { what: WHAT, detail };
        let dims = &self.layout.dims;
        let m = plan.len();
        if !(1..=dims.tokens).contains(&m)
            || plan.tokens.len() != m
            || plan.raw_write.len() != m
            || plan.raw_first.len() != m
        {
            return Err(refuse(format!(
                "a plan of {} positions, {} tokens, {} cells and {} window starts; the layout runs \
                 1..={} tokens",
                plan.len(),
                plan.tokens.len(),
                plan.raw_write.len(),
                plan.raw_first.len(),
                dims.tokens
            )));
        }
        let pos0 = plan.pos[0];
        if (0u32..)
            .zip(&plan.pos)
            .any(|(t, &p)| pos0.checked_add(t) != Some(p))
        {
            return Err(refuse(format!(
                "positions {:?} are not consecutive: the window ring's append gives each token its own slot only when they are",
                plan.pos
            )));
        }
        if plan.streams.len() != self.layout.streams.len() {
            return Err(refuse(format!(
                "a plan of {} streams, the layout holds {}",
                plan.streams.len(),
                self.layout.streams.len()
            )));
        }
        if embd.len() != m * dims.embd_bytes || engram.len() != m * dims.engram_bytes {
            return Err(refuse(format!(
                "{} embedding bytes and {} engram bytes; the step's {m} tokens take {} and {}",
                embd.len(),
                engram.len(),
                m * dims.embd_bytes,
                m * dims.engram_bytes
            )));
        }
        self.words.fill(0);
        self.tokens(plan)?;
        self.streams(plan)?;
        self.ropes(plan);
        self.rows(embd, engram);
        self.pos = Some(pos0);
        Ok(())
    }

    fn tokens(&mut self, plan: &StepPlan) -> Result<(), GpuError> {
        let window = self.layout.dims.window;
        let fields =
            (plan.tokens.iter().zip(&plan.pos)).zip(plan.raw_write.iter().zip(&plan.raw_first));
        for (w, ((&token, &p), (&cell, &first))) in self
            .words
            .as_chunks_mut::<TOKEN_WORDS>()
            .0
            .iter_mut()
            .zip(fields)
        {
            if first > p {
                return Err(GpuError::Shape {
                    what: WHAT,
                    detail: format!("the window of position {p} starts at cell {first}"),
                });
            }
            *w = [token, p, cell % window, p - first + 1];
        }
        Ok(())
    }

    fn streams(&mut self, plan: &StepPlan) -> Result<(), GpuError> {
        // Offsets are the layout's, at its capacity; the counts are the plan's.
        let (cap, m) = (self.layout.dims.tokens, plan.len());
        for (s, (l, st)) in self.layout.streams.iter().zip(&plan.streams).enumerate() {
            let refuse = |detail: String| GpuError::Shape {
                what: WHAT,
                detail: format!("stream {s}: {detail}"),
            };
            let groups = st.groups();
            let kept = st.persist_src.len();
            if st.ratio != l.ratio
                || st.n_visible.len() != m
                || groups > l.group_slots
                || st.state_read.len() != groups * l.ratio as usize
                || st.write_pos.len() != groups
                || kept > l.persist_slots
                || st.persist_dst.len() != kept
            {
                return Err(refuse(format!(
                    "ratio {} with {} visible counts, {groups} groups, {} reads, {} write positions \
                     and {kept}/{} kept slots; the layout holds ratio {} for {cap} tokens, the \
                     plan runs {m}",
                    st.ratio,
                    st.n_visible.len(),
                    st.state_read.len(),
                    st.write_pos.len(),
                    st.persist_dst.len(),
                    l.ratio
                )));
            }
            let w = &mut self.words;
            w[l.at] = count(groups, &refuse)?;
            w[l.at + 1] = count(kept, &refuse)?;
            w[l.n_visible_at()..][..m].copy_from_slice(&st.n_visible);
            w[l.state_read_at(cap)..][..st.state_read.len()].copy_from_slice(&st.state_read);
            for (dst, &row) in w[l.state_write_at(cap)..].iter_mut().zip(&st.state_write) {
                *dst = u32::try_from(row).map_err(|_| refuse(format!("row {row} passes u32")))?;
            }
            w[l.write_pos_at(cap)..][..groups].copy_from_slice(&st.write_pos);
            w[l.persist_src_at(cap)..][..kept].copy_from_slice(&st.persist_src);
            w[l.persist_dst_at(cap)..][..kept].copy_from_slice(&st.persist_dst);
        }
        Ok(())
    }

    fn ropes(&mut self, plan: &StepPlan) {
        for (t, &p) in plan.pos.iter().enumerate() {
            for table in Table::ALL {
                let source = match table {
                    Table::WindowForward | Table::WindowBack => &self.window,
                    Table::YarnForward | Table::YarnBack => &self.yarn,
                };
                self.table.clear();
                source.push(p, table.direction(), &mut self.table);
                let at = self.layout.table_at(t, table);
                put_f32(&mut self.words[at..], &self.table);
            }
        }
        for (l, st) in self.layout.streams.iter().zip(&plan.streams) {
            let Some(at) = l.row_tables else {
                continue;
            };
            for (g, &p) in st.write_pos.iter().enumerate() {
                self.table.clear();
                self.yarn.push(p, Direction::Forward, &mut self.table);
                put_f32(
                    &mut self.words[at + g * self.layout.dims.rope_dims..],
                    &self.table,
                );
            }
        }
    }

    fn rows(&mut self, embd: &[u8], engram: &[u8]) {
        let dims = &self.layout.dims;
        let n = self.layout.embd_words;
        for (t, row) in embd.chunks_exact(dims.embd_bytes).enumerate() {
            put_bytes(&mut self.words[self.layout.embd_at + t * n..][..n], row);
        }
        if dims.engram_bytes == 0 {
            return;
        }
        let n = self.layout.engram_words;
        for (t, rows) in engram.chunks_exact(dims.engram_bytes).enumerate() {
            put_bytes(&mut self.words[self.layout.engram_at + t * n..][..n], rows);
        }
    }
}

/// `n` as an image word.
fn count(n: usize, refuse: &impl Fn(String) -> GpuError) -> Result<u32, GpuError> {
    u32::try_from(n).map_err(|_| refuse(format!("{n} passes u32")))
}

/// `bytes` into `dst`, four to a little-endian word, the last word
/// zero-padded.
fn put_bytes(dst: &mut [u32], bytes: &[u8]) {
    for (w, b) in dst.iter_mut().zip(bytes.chunks(4)) {
        let mut le = [0u8; 4];
        le[..b.len()].copy_from_slice(b);
        *w = u32::from_le_bytes(le);
    }
}

/// `values`' bits into the front of `dst`.
fn put_f32(dst: &mut [u32], values: &[f32]) {
    for (w, v) in dst.iter_mut().zip(values) {
        *w = v.to_bits();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW: u32 = 128;
    const NGRAM: usize = 4;
    const ROPE_DIMS: usize = 64;
    const N_EMBD: usize = 8;
    const ENGRAM_BYTES: usize = 6;

    fn image(tokens: usize) -> StepImage {
        let layout = ImageLayout::new(ImageDims {
            tokens,
            window: WINDOW,
            stream_ratios: vec![2, 1],
            rope_dims: ROPE_DIMS,
            n_embd: N_EMBD,
            embd_bytes: 2 * N_EMBD,
            engram_bytes: ENGRAM_BYTES,
        })
        .expect("the test dims make a layout");
        let window = RopeSpec::window(10_000.0, ROPE_DIMS);
        let yarn = RopeSpec::yarn(160_000.0, 16.0, 65_536, 32.0, 1.0, ROPE_DIMS);
        StepImage::new(layout, &window, &yarn).expect("the test ropes make tables")
    }

    /// Each offset accessor names the word the view reads the same field at,
    /// for every token of a two-token image: an image whose every word holds
    /// its own index shows it.
    #[test]
    fn offsets_are_the_views() {
        let img = image(2);
        let layout = img.layout();
        let probe: Vec<u32> = (0..u32::try_from(layout.words()).expect("u32 words")).collect();
        let view = layout.view(&probe).expect("a view");
        let at = |w: u32| w as usize;
        for t in 0..2 {
            assert_eq!(at(view.token(t).token), layout.token_at(t), "token {t}");
            assert_eq!(at(view.embd(t)[0]), layout.embd_at(t), "embd {t}");
            assert_eq!(at(view.engram(t)[0]), layout.engram_at(t), "engram {t}");
            for table in Table::ALL {
                assert_eq!(
                    at(view.table(t, table)[0]),
                    layout.table_at(t, table),
                    "{table:?} {t}"
                );
            }
        }
        for s in 0..layout.streams().len() {
            assert_eq!(
                Some(at(view.stream(s).n_visible[0])),
                layout.n_visible_at(s),
                "stream {s}"
            );
        }
        assert_eq!(layout.n_visible_at(layout.streams().len()), None);
    }

    /// The ring append gives each token of a step its own slot only when the
    /// step's positions are consecutive, so a plan whose positions skip one
    /// is refused, and the image then holds no step.
    #[test]
    fn build_refuses_positions_that_are_not_consecutive() {
        let planner = Planner::new(WINDOW, &[0, 2, 1], NGRAM, 4096).expect("a planner");
        let mut plan = StepPlan::default();
        planner
            .plan_into(&[7, 9], 300, &[1, 2, 3], &mut plan)
            .expect("a plan of two tokens");
        let (embd, engram) = (vec![0u8; 2 * 2 * N_EMBD], vec![0u8; 2 * ENGRAM_BYTES]);
        let mut img = image(2);
        img.build(&plan, &embd, &engram)
            .expect("consecutive positions build");
        assert_eq!(img.pos(), Some(300));
        plan.pos[1] += 1;
        let err = img
            .build(&plan, &embd, &engram)
            .expect_err("positions 300 and 302 must be refused");
        assert!(
            err.to_string().contains("not consecutive"),
            "the refusal names the positions: {err}"
        );
        assert_eq!(img.pos(), None);
    }
}
