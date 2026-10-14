//! The MiniCPM-V 4.7 prompt sets: llama.cpp mainline's text model on a prompt whose images are mtmd's canvas chunks,
//! dumped by `tools/ref/clefvis/dump_mtmd.cpp` (`just dump-ref-mcvis`) at the mainline tree [`MC47_LCPP_BUILD`], on the
//! text file [`TEXT_MODEL`] with the f16 projector the tower sets name. Two kinds, one family each ([`PromptKind`]):
//!
//! - [`PromptKind::Hidden`] (set C, `# clefvis canvas-hidden`): `result_norm` `[2048, ids]` of every position of a chat
//!   prompt, and `mrope_pos` `[4, ids]`, the four position lanes each row was fed, `(t, y, x, z)` in the order of the
//!   batch's sections. A text row is fed `[p, p, p, 0]` (the batch allocator expands the one position the dump gives it);
//!   an image row is fed what `mtmd_image_tokens_get_decoder_pos` names, with the `<image>` token's position `origin` in
//!   `z` for a tile row and `p` in every lane for a separator row.
//! - [`PromptKind::ChatIds`] (set E, `# clefvis canvas-chatids`): `ids` `[ids]`, the ids llama-server's chat path gives a
//!   request: the model's template, then mtmd's text `<image_id>N</image_id><image>` before each image, and the image
//!   token [`IMAGE_PAD_ID`] repeated for each of the image chunk's rows. The `# prompt` line is the rendered text.
//!
//! An image is one pad run: the chunk's rows are every tile's merger rows followed by that tile's separator rows, so a run
//! is the `n_tokens` of the tower sets' `# image` line for the same image (`canvas-taps`), and the position cursor
//! advances by the chunk's `n_pos` (one a tile, one a separator), not by its rows. The header lines this reader owns beyond
//! the tower sets' (`# clefvis`, `# mmproj`, `# projector`, `# positions`, `# device`): `# tokens_count`, `# image_pad_id`,
//! `# rows`, `# pos_lanes`, `# span columns` and `# span`, `# image columns` and `# image` (no `# tile` lines: the tiles
//! are the taps set's, and a prompt set's positions give the same layout), and for the chat ids `# chat`, `# prompt`,
//! `# prompt_sha256`. [`PromptSet::open`] checks every one; [`PromptSet::check_lanes`] is the oracle's own positions
//! against the canvas rules.

use super::vis::{At, MC47_LCPP_BUILD, N_EMBD, canvas_suffix, one, open_seat, rows_of};
use crate::RefError;
use crate::clefvis::{ChatLine, parse_chat, unescape};
use crate::family::{Build, Family, Identity};
use crate::ik::{self, FileElem, Layout, RefManifest, RowKind};
use std::path::Path;

/// The text file every prompt set is dumped from: the checkpoint's Q4_K_M, a `qwen35moe` file with its own vocabulary.
pub const TEXT_MODEL: &str = "/models/MiniCPM-V-4.7-35B-A3B/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf";

/// The architecture the text file states (`general.architecture`), the `# arch` line of a prompt set.
pub const TEXT_ARCH: &str = "qwen35moe";

/// The image token of the prompt sets: `<unk>` of the text file's vocabulary, the id an image's rows are expanded to.
// PIN(2026-10-13): `llama-tokenize -m <TEXT_MODEL> -p "<unk>" --ids` prints 248077 (mainline 358a420ec); `<image>` is
// 248078, `<|image_pad|>` 248056.
pub const IMAGE_PAD_ID: u32 = 248_077;

/// The marker the chat path leaves in the rendered prompt where an image goes (`mtmd_default_marker`).
pub const MEDIA_MARKER: &str = "<__media__>";

/// The position lanes of `mrope_pos`, in the order of the batch's sections: the time `t`, the line `y`, the column `x`, the
/// canvas origin `z`.
pub const LANES: usize = 4;

/// The kinds of prompt set the seat dumps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptKind {
    /// Set C: `result_norm` and the four position lanes of every position.
    Hidden,
    /// Set E: the ids of a chat request.
    ChatIds,
}

impl PromptKind {
    /// The `# clefvis` line of the kind's sets.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            PromptKind::Hidden => "canvas-hidden",
            PromptKind::ChatIds => "canvas-chatids",
        }
    }

    /// The family of the kind.
    #[must_use]
    pub fn family(self) -> &'static Family {
        match self {
            PromptKind::Hidden => &HIDDEN,
            PromptKind::ChatIds => &CHATIDS,
        }
    }
}

/// [`TEXT_MODEL`], as a family's `runs`.
fn text() -> Result<String, RefError> {
    Ok(TEXT_MODEL.to_string())
}

/// Set C: the hidden states of three chat prompts on the card, and the CPU twin of the first (the other two cases'
/// twins are not dumped: the twin is a distance to report, not a second oracle).
pub static HIDDEN: Family = Family {
    name: "minicpmv47-vis-hidden",
    sets: &[
        "ref_minicpmv47_vis_hidden_mc1",
        "ref_minicpmv47_vis_hidden_mc2",
        "ref_minicpmv47_vis_hidden_mc3",
        "ref_minicpmv47_vis_hidden_mc1.cpu",
    ],
    resolve: None,
    recipe: "just dump-ref-mcvis [--cpu-twin] ref_minicpmv47_vis_hidden_mc<N>",
    identity: Identity::Manifest,
    arch: Some(TEXT_ARCH),
    build: Some(Build::Is(MC47_LCPP_BUILD)),
    runs: Some(text),
    draft_runs: None,
    consumers: &[],
};

/// Set E: the ids llama-server's chat path gives each request, with its rendered prompt; no card, no twin.
pub static CHATIDS: Family = Family {
    name: "minicpmv47-vis-chatids",
    sets: &[
        "ref_minicpmv47_vis_chatids_mc1",
        "ref_minicpmv47_vis_chatids_mc2",
        "ref_minicpmv47_vis_chatids_mc3",
    ],
    resolve: None,
    recipe: "just dump-ref-mcvis --ids",
    identity: Identity::Manifest,
    arch: Some(TEXT_ARCH),
    build: Some(Build::Is(MC47_LCPP_BUILD)),
    runs: Some(text),
    draft_runs: None,
    consumers: &[],
};

/// One `# image` line of a prompt set: an image of the prompt and the layout of its chunk. The tiles' sizes and rows are
/// the tower sets' (`canvas-taps`, same image name).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptImage {
    pub name: String,
    /// The PNG's sha256, so a gate can prove the file it reads is the one dumped.
    pub png_sha256: String,
    /// The sha256 of the RGB8 bytes stb_image decoded from it.
    pub rgb8_sha256: String,
    pub w: usize,
    pub h: usize,
    /// The slice grid, `0 x 0` for an overview alone.
    pub grid_x: usize,
    pub grid_y: usize,
    /// The tiles: the overview and the slices.
    pub n_tiles: usize,
    /// The rows of the chunk (every tile's merger rows and separator rows), and how far it advances the decoder (one a
    /// tile, one a separator).
    pub n_tokens: usize,
    pub n_pos: usize,
}

impl PromptImage {
    /// The separator rows of the chunk: one after an overview alone, else `2 * slices + lines`.
    #[must_use]
    pub fn separators(&self) -> usize {
        if self.n_tiles == 1 {
            1
        } else {
            2 * (self.n_tiles - 1) + self.grid_y
        }
    }
}

/// One `# span` line: where an image's pad run sits in the prompt's ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptSpan {
    pub index: usize,
    pub image: String,
    /// The first sequence index of the run, and its length (image-pad ids): the image's `n_tokens`.
    pub at: usize,
    pub len: usize,
    pub n_tiles: usize,
    pub grid_x: usize,
    pub grid_y: usize,
    pub n_pos: usize,
    /// The decoder position of the chunk's first row, and the one after its last (`start_pos + n_pos`). The `<image>`
    /// token is at `start_pos - 1`: the canvas origin.
    pub start_pos: i64,
    pub end_pos: i64,
}

/// A MiniCPM-V 4.7 prompt set: the node-dump manifest and what this module read of it.
#[derive(Debug, Clone)]
pub struct PromptSet {
    pub man: RefManifest,
    pub kind: PromptKind,
    /// `# device cpu`: the CPU twin.
    pub cpu: bool,
    /// The card the dump ran on (`# device … card <name>`), `cpu` for a twin.
    pub card: String,
    /// The prompt's ids (`# tokens_count`).
    pub n_ids: usize,
    pub images: Vec<PromptImage>,
    pub spans: Vec<PromptSpan>,
    /// The `# chat` line of a [`PromptKind::ChatIds`] set.
    pub chat: Option<ChatLine>,
}

impl PromptSet {
    /// Read the set at `dir` as a set of `kind`'s family: against the family ([`RefManifest::open`]: the trailer, the text
    /// file as the model, the architecture and the mainline build), then the seat's lines (the projector file and digest,
    /// `# projector`, `# positions`, `# device`), then this module's. A set of another kind, of the tower kinds or of
    /// another seat is [`RefError::Foreign`] on `clefvis`, `projector` or `positions`; one made with another mmproj
    /// [`RefError::Stale`]; a line or row missing or of another shape [`RefError::Malformed`] naming it.
    pub fn open(dir: &Path, kind: PromptKind) -> Result<PromptSet, RefError> {
        let (man, seat) = open_seat(dir, kind.family(), kind.as_str())?;
        let at = At(dir);
        let n_ids = man
            .header
            .tokens_count
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| {
                RefError::missing(dir, format!("{}: no `# tokens_count` line", dir.display()))
            })?;
        let pad = one(&man, "image_pad_id")?;
        if pad != [IMAGE_PAD_ID.to_string()] {
            return Err(RefError::malformed(
                &at,
                format!("image_pad_id {pad:?}, the vocabulary's <unk> is {IMAGE_PAD_ID}"),
            ));
        }
        let images = read_images(&man)?;
        let spans = read_spans(&man)?;
        let chat = if kind == PromptKind::ChatIds {
            Some(read_chat(&man)?)
        } else {
            let rows = one(&man, "rows")?;
            let lanes = one(&man, "pos_lanes")?;
            if rows != ["mtmd"] || lanes.first() != Some(&"t y x z") {
                return Err(RefError::malformed(
                    &at,
                    format!("# rows {rows:?}, # pos_lanes {lanes:?}: want `mtmd` and `t y x z`"),
                ));
            }
            None
        };
        let set = PromptSet {
            man,
            kind,
            cpu: seat.cpu,
            card: seat.card,
            n_ids,
            images,
            spans,
            chat,
        };
        set.check_rows()?;
        Ok(set)
    }

    /// The `# image` line named `name`.
    pub fn image(&self, name: &str) -> Result<&PromptImage, RefError> {
        self.images.iter().find(|i| i.name == name).ok_or_else(|| {
            RefError::missing(
                &self.man.dir,
                format!("{}: no image {name:?}", self.man.dir.display()),
            )
        })
    }

    fn need(&self, kind: PromptKind, what: &str) -> Result<(), RefError> {
        if self.kind == kind {
            return Ok(());
        }
        Err(RefError::malformed(
            At(&self.man.dir),
            format!(
                "{what} reads a {} set, this is {}",
                kind.as_str(),
                self.kind.as_str()
            ),
        ))
    }

    /// `result_norm` of every position, `[2048, ids]` row-major by position ([`PromptKind::Hidden`]).
    pub fn result_norm(&self) -> Result<Vec<f32>, RefError> {
        self.need(PromptKind::Hidden, "result_norm")?;
        Ok(ik::load_ref_in(&self.man, "result_norm", 0)?.1)
    }

    /// The `(t, y, x, z)` each position was fed ([`PromptKind::Hidden`]).
    pub fn positions(&self) -> Result<Vec<[i32; LANES]>, RefError> {
        self.need(PromptKind::Hidden, "positions")?;
        let v = ik::ref_ints(&self.man, "mrope_pos", 0, RowKind::Tensor, Layout::Flat)?;
        v.as_chunks::<LANES>()
            .0
            .iter()
            .map(|c| {
                let mut lane = [0i32; LANES];
                for (l, &p) in lane.iter_mut().zip(c) {
                    *l = i32::try_from(p).map_err(|_| {
                        RefError::malformed(
                            At(&self.man.dir),
                            format!("position {p} is not an i32"),
                        )
                    })?;
                }
                Ok(lane)
            })
            .collect()
    }

    /// The ids the chat path gave the request, each image's run the pad id repeated ([`PromptKind::ChatIds`]).
    pub fn ids(&self) -> Result<Vec<u32>, RefError> {
        self.need(PromptKind::ChatIds, "ids")?;
        ik::ref_ints(&self.man, "ids", 0, RowKind::Tensor, Layout::Flat)?
            .into_iter()
            .map(|t| {
                u32::try_from(t).map_err(|_| {
                    RefError::malformed(At(&self.man.dir), format!("ids: {t} is not a token id"))
                })
            })
            .collect()
    }

    /// The prompt the chat template rendered, before tokenization ([`PromptKind::ChatIds`]).
    pub fn prompt(&self) -> Result<String, RefError> {
        self.need(PromptKind::ChatIds, "prompt")?;
        let line = one(&self.man, "prompt")?;
        let [raw] = line[..] else {
            return Err(RefError::malformed(
                At(&self.man.dir),
                format!("a `# prompt` line of {} fields, want one", line.len()),
            ));
        };
        unescape(raw).map_err(|e| RefError::malformed(At(&self.man.dir), format!("# prompt: {e}")))
    }

    /// The `# chat` line ([`PromptKind::ChatIds`]).
    pub fn chat(&self) -> Result<&ChatLine, RefError> {
        self.need(PromptKind::ChatIds, "chat")?;
        self.chat.as_ref().ok_or_else(|| {
            RefError::missing(&self.man.dir, "a chat-ids set without its `# chat` line")
        })
    }

    /// Every row the kind needs is there, of its shape, and the spans, images and ids agree.
    fn check_rows(&self) -> Result<(), RefError> {
        let at = At(&self.man.dir);
        let n = self.n_ids as u64;
        self.check_images()?;
        self.check_spans()?;
        match self.kind {
            PromptKind::Hidden => {
                self.want_row("result_norm", "f32", [N_EMBD as u64, n, 1, 1])?;
                self.want_row("mrope_pos", "i32", [LANES as u64, n, 1, 1])?;
                self.check_int_row("mrope_pos", LANES as u64 * n)?;
                if self.man.tensors.len() != 2 {
                    return Err(RefError::malformed(
                        &at,
                        format!(
                            "{} tensor rows, a hidden set holds 2",
                            self.man.tensors.len()
                        ),
                    ));
                }
            }
            PromptKind::ChatIds => {
                self.want_row("ids", "i32", [n, 1, 1, 1])?;
                self.check_int_row("ids", n)?;
                self.check_chat_ids()?;
            }
        }
        Ok(())
    }

    /// Row `name` is of type `ty` and shape `ne`.
    fn want_row(&self, name: &str, ty: &str, ne: [u64; 4]) -> Result<(), RefError> {
        let r = self.man.tensor(name, 0)?;
        if r.ty != ty || r.ne != ne {
            return Err(RefError::malformed(
                At(&self.man.dir),
                format!("{name} is {} {:?}, want {ty} {ne:?}", r.ty, r.ne),
            ));
        }
        Ok(())
    }

    /// The int row `name`, of `count` elements, has its lossless i32 twin.
    fn check_int_row(&self, name: &str, count: u64) -> Result<(), RefError> {
        let int = ik::find_int_row(&self.man, name, 0, RowKind::Tensor, Layout::Flat)?;
        if int.twin != FileElem::I32 || int.count != count {
            return Err(RefError::malformed(
                At(&self.man.dir),
                format!(
                    "{name} twin: {} elements of {:?}, want {count} of I32",
                    int.count, int.twin
                ),
            ));
        }
        Ok(())
    }

    /// The image lines keep the canvas arithmetic: a grid of lines and columns or none, a tile for the overview and each
    /// slice, and `n_pos` one a tile and one a separator row.
    fn check_images(&self) -> Result<(), RefError> {
        let at = At(&self.man.dir);
        if self.images.is_empty() {
            return Err(RefError::malformed(&at, "no `# image` line"));
        }
        for (k, i) in self.images.iter().enumerate() {
            let bad = |what: String| RefError::malformed(&at, format!("image {}: {what}", i.name));
            if self.images[..k].iter().any(|o| o.name == i.name) {
                return Err(bad("twice".to_string()));
            }
            if (i.grid_x == 0) != (i.grid_y == 0) || i.n_tiles != 1 + i.grid_x * i.grid_y {
                return Err(bad(format!(
                    "{} tiles in a {} x {} grid",
                    i.n_tiles, i.grid_x, i.grid_y
                )));
            }
            if i.n_pos != i.n_tiles + i.separators() || i.n_tokens < i.n_tiles + i.separators() {
                return Err(bad(format!(
                    "{} rows and {} positions for {} tiles and {} separator rows",
                    i.n_tokens,
                    i.n_pos,
                    i.n_tiles,
                    i.separators()
                )));
            }
        }
        Ok(())
    }

    /// The spans lie inside the ids in order, hold their image's rows, and start where the images before them leave the
    /// decoder: a span starts at its index less what those images saved (`rows - n_pos` each).
    fn check_spans(&self) -> Result<(), RefError> {
        let at = At(&self.man.dir);
        let (mut end, mut saved) = (0usize, 0i64);
        for (k, s) in self.spans.iter().enumerate() {
            if s.index != k || s.at < end || s.at + s.len > self.n_ids || s.len == 0 {
                return Err(RefError::malformed(
                    &at,
                    format!(
                        "span {}: {} + {} of {} ids after index {end}",
                        s.index, s.at, s.len, self.n_ids
                    ),
                ));
            }
            end = s.at + s.len;
            let i = self.image(&s.image)?;
            if (s.len, s.n_tiles, s.grid_x, s.grid_y, s.n_pos)
                != (i.n_tokens, i.n_tiles, i.grid_x, i.grid_y, i.n_pos)
            {
                return Err(RefError::malformed(
                    &at,
                    format!(
                        "span {} holds {} ids as {} tiles of a {} x {} grid, image {} is {} rows as {} tiles of {} x {}",
                        s.index,
                        s.len,
                        s.n_tiles,
                        s.grid_x,
                        s.grid_y,
                        s.image,
                        i.n_tokens,
                        i.n_tiles,
                        i.grid_x,
                        i.grid_y
                    ),
                ));
            }
            let start = s.at as i64 - saved;
            if (s.start_pos, s.end_pos) != (start, start + s.n_pos as i64) {
                return Err(RefError::malformed(
                    &at,
                    format!(
                        "span {} runs from position {} to {}, its {} ids and the images before it give {start} to {}",
                        s.index,
                        s.start_pos,
                        s.end_pos,
                        s.len,
                        start + s.n_pos as i64
                    ),
                ));
            }
            saved += s.len as i64 - s.n_pos as i64;
        }
        if self.images.len() != self.spans.len() {
            return Err(RefError::malformed(
                &at,
                format!(
                    "{} image lines, {} spans",
                    self.images.len(),
                    self.spans.len()
                ),
            ));
        }
        Ok(())
    }

    /// A chat-ids set's pad id fills each span and nothing else, the rendered prompt holds one media marker for each image
    /// (mtmd's `<image_id>N</image_id><image>` text is added at tokenization, after the template), and the request either
    /// opens the next turn or continues the last.
    fn check_chat_ids(&self) -> Result<(), RefError> {
        let at = At(&self.man.dir);
        let ids = self.ids()?;
        let mut in_span = vec![false; ids.len()];
        for s in &self.spans {
            in_span[s.at..s.at + s.len].fill(true);
        }
        for (i, (&id, &inside)) in ids.iter().zip(&in_span).enumerate() {
            if (id == IMAGE_PAD_ID) != inside {
                return Err(RefError::malformed(
                    &at,
                    format!(
                        "ids[{i}] is {id}: the image-pad id {IMAGE_PAD_ID} fills the spans and nothing else"
                    ),
                ));
            }
        }
        let markers = self.prompt()?.matches(MEDIA_MARKER).count();
        if markers != self.spans.len() {
            return Err(RefError::malformed(
                &at,
                format!(
                    "# prompt holds {markers} {MEDIA_MARKER} markers, the set {} spans",
                    self.spans.len()
                ),
            ));
        }
        let chat = self.chat()?;
        if chat.continue_final_message == chat.add_generation_prompt {
            return Err(RefError::malformed(
                &at,
                "# chat: a request either opens the next turn or continues the last, never both or neither",
            ));
        }
        Ok(())
    }

    /// The oracle's own positions against the canvas rules ([`PromptKind::Hidden`]): a text row is `[p, p, p, 0]` with `p`
    /// its index less what the images before it saved; in a span the rows are tiles and separators — a tile's rows carry
    /// the span's origin (`start_pos - 1`) in `z` and one `t` (the first tile's is `start_pos`, the next one a tile and its
    /// separators later), a separator row is `p` in every lane with `p` the next after its tile — and their counts are the
    /// span's tiles and positions, each tile's separators the ones its place in the grid names.
    pub fn check_lanes(&self) -> Result<(), RefError> {
        let pos = self.positions()?;
        let at = At(&self.man.dir);
        let bad = |row: usize, what: String| RefError::malformed(&at, format!("row {row}: {what}"));
        let text = |row: usize, saved: i64| -> Result<(), RefError> {
            let p = row as i64 - saved;
            let want = [p, p, p, 0].map(|v| i32::try_from(v).unwrap_or(i32::MAX));
            if pos[row] != want {
                return Err(bad(
                    row,
                    format!("a text row is fed {:?}, want {want:?}", pos[row]),
                ));
            }
            Ok(())
        };
        let (mut row, mut saved) = (0usize, 0i64);
        for s in &self.spans {
            for r in row..s.at {
                text(r, saved)?;
            }
            let origin = i32::try_from(s.start_pos - 1)
                .map_err(|_| bad(s.at, format!("span {} starts at {}", s.index, s.start_pos)))?;
            let (mut tiles, mut seps, mut tile_t) = (0usize, 0usize, origin + 1);
            let (mut tile_seps, mut in_tile) = (0usize, false);
            for (r, &lane) in pos.iter().enumerate().skip(s.at).take(s.len) {
                let [t, y, x, z] = lane;
                if z == origin {
                    if !in_tile {
                        if tiles > 0 {
                            self.check_tile_seps(s, tiles - 1, tile_seps, r)?;
                            tile_t += 1 + i32::try_from(tile_seps).unwrap_or(i32::MAX);
                        }
                        tiles += 1;
                        tile_seps = 0;
                    }
                    if t != tile_t || y < origin || x < origin {
                        return Err(bad(
                            r,
                            format!(
                                "a tile row of span {} is fed {:?}, its tile's t is {tile_t}",
                                s.index, lane
                            ),
                        ));
                    }
                    in_tile = true;
                } else {
                    let p = tile_t + 1 + i32::try_from(tile_seps).unwrap_or(i32::MAX);
                    if tiles == 0 || [t, y, x, z] != [p; LANES] {
                        return Err(bad(
                            r,
                            format!(
                                "a separator row of span {} is fed {:?}, want {p} in every lane",
                                s.index, lane
                            ),
                        ));
                    }
                    tile_seps += 1;
                    seps += 1;
                    in_tile = false;
                }
            }
            self.check_tile_seps(s, tiles - 1, tile_seps, s.at + s.len)?;
            if tiles != s.n_tiles || tiles + seps != s.n_pos {
                return Err(bad(
                    s.at,
                    format!(
                        "span {} holds {tiles} tiles and {seps} separator rows, its line says {} tiles and {} positions",
                        s.index, s.n_tiles, s.n_pos
                    ),
                ));
            }
            saved += s.len as i64 - s.n_pos as i64;
            row = s.at + s.len;
        }
        for r in row..self.n_ids {
            text(r, saved)?;
        }
        Ok(())
    }

    /// Tile `tile` of span `s` has the separator rows its place in the span's grid names.
    fn check_tile_seps(
        &self,
        s: &PromptSpan,
        tile: usize,
        seps: usize,
        row: usize,
    ) -> Result<(), RefError> {
        let want = canvas_suffix(tile, s.grid_x, s.grid_y).len();
        if seps == want {
            return Ok(());
        }
        Err(RefError::malformed(
            At(&self.man.dir),
            format!(
                "row {row}: tile {tile} of span {} has {seps} separator rows, a {} x {} grid gives it {want}",
                s.index, s.grid_x, s.grid_y
            ),
        ))
    }
}

fn read_images(man: &RefManifest) -> Result<Vec<PromptImage>, RefError> {
    rows_of(man, "image", |r| {
        Ok(PromptImage {
            name: r.text("name")?.to_string(),
            png_sha256: r.text("png_sha256")?.to_string(),
            rgb8_sha256: r.text("rgb8_sha256")?.to_string(),
            w: r.parse("w")?,
            h: r.parse("h")?,
            grid_x: r.parse("grid_x")?,
            grid_y: r.parse("grid_y")?,
            n_tiles: r.parse("n_tiles")?,
            n_tokens: r.parse("n_tokens")?,
            n_pos: r.parse("n_pos")?,
        })
    })
}

fn read_spans(man: &RefManifest) -> Result<Vec<PromptSpan>, RefError> {
    rows_of(man, "span", |r| {
        Ok(PromptSpan {
            index: r.parse("index")?,
            image: r.text("image")?.to_string(),
            at: r.parse("at")?,
            len: r.parse("len")?,
            n_tiles: r.parse("n_tiles")?,
            grid_x: r.parse("grid_x")?,
            grid_y: r.parse("grid_y")?,
            n_pos: r.parse("n_pos")?,
            start_pos: r.parse("start_pos")?,
            end_pos: r.parse("end_pos")?,
        })
    })
}

/// `# chat\ttemplate_sha256 <hex>\tuse_jinja 1\tenable_thinking 0|1\treasoning_format deepseek\t
/// add_generation_prompt 0|1\tcontinue_final_message none|auto`, parsed by [`parse_chat`].
fn read_chat(man: &RefManifest) -> Result<ChatLine, RefError> {
    parse_chat(&one(man, "chat")?, &At(&man.dir))
}
