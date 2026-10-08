//! The Clef-Flash image-input oracle sets (`tools/ref/clefvis/dump_mtmd.cpp`):
//! llama.cpp mainline's mtmd `qwen3vl_merger` tower and its qwen35 text model on
//! Clef-Flash's Q8_0 file, written in the node-dump set format
//! (`tools/ref/dump_ref.cpp`), so a family's identity check
//! ([`crate::arch::qwen35::clefvis`], through [`RefManifest::open`]) already
//! refuses a set of another file, another build or another architecture and
//! a set without its trailer. This module reads what the format leaves to the
//! kind of set: the `# clefvis <kind>` line, the mmproj the set names, the
//! images and spans, and the rows each kind must carry, and hands the values
//! out by name.
//!
//! - [`Kind::Preproc`] (set A): per image `<name>/inp_raw`, the f32 image after
//!   mtmd's preprocess, channel-planar `[W, H, 3, 1]`, the tower's graph input.
//! - [`Kind::Taps`] (set B): per image `<name>/<node>` for the nodes of
//!   [`tap_names`], and `<name>/embd`, the final embeddings `[4096, tokens]`.
//! - [`Kind::Hidden`], [`Kind::Prose`] and [`Kind::Bf16Rows`] (sets C, C′, C″):
//!   one prompt's `result_norm` of every position `[4096, ids]` and the
//!   positions the decode was fed, `mrope_pos` `[3, ids]` (t, y, x), with the
//!   spans of its images.
//!
//! The header lines this reader owns: `# clefvis`, `# mmproj`, `# device`,
//! `# image columns` and `# image`, `# span columns` and `# span`,
//! `# tap_effect`; the rest are [`RefManifest`]'s.

use crate::RefError;
use crate::columns::{Columns, Row};
use crate::family::{Family, Identity};
use crate::ik::{self, FileElem, Layout, RefManifest, RefRow, RowKind};
use std::fmt;
use std::path::Path;

/// The projector file every clefvis set is dumped with: Clef-Flash's bf16
/// mmproj from bartowski's `Cloudflare_clef-flash-GGUF`, under the box's
/// `/root/models` (the box's `/models` is not a round's).
pub const MMPROJ_BF16: &str = "/root/models/clef-flash/mmproj-Cloudflare_clef-flash-bf16.gguf";

/// The sha256 of [`MMPROJ_BF16`], the LFS oid of that file (921,704,928 bytes).
// PIN(2026-10-08): bartowski/Cloudflare_clef-flash-GGUF at rev d7f376ea, read
// from its tree listing (crates/hf/tests/fixtures/bartowski-clef-flash-gguf.json)
// and equal to `sha256sum` of the downloaded file.
pub const MMPROJ_BF16_SHA256: &str =
    "3c45b34aee6f353a0d41d6b96ba712a498a17a82f0a6152bf68a5021f9652c0f";

/// The token id of `<|image_pad|>` in the Clef vocabulary.
pub const IMAGE_PAD_ID: u32 = 248_056;

/// The text model's width, and so the width of the tower's output rows.
pub const N_EMBD: usize = 4096;

/// Patch size times the merge size: the sides of mtmd's planned image are
/// multiples of it, and a token covers this many pixels a side.
pub const ALIGN: usize = 32;

/// The blocks whose nodes set B taps.
pub const TAP_BLOCKS: [u32; 4] = [0, 1, 13, 26];

/// The nodes tapped in each of [`TAP_BLOCKS`], mtmd's names (`<node>-<block>`).
pub const TAP_NODES: [&str; 6] = [
    "ln1",
    "Qcur_rope",
    "attn_out",
    "ffn_inp",
    "ffn_out",
    "layer_out",
];

/// The patch embedding plus the learned positions, before block 0.
pub const POS_EMB: &str = "inp_pos_emb";

/// The post-LN output, after block 26 and before the 2x2 merge.
pub const POST_LN: &str = "norm_b-27";

/// The final embeddings of the clean pass (`mtmd_get_output_embd`).
pub const EMBD: &str = "embd";

/// The graph input of the tower: the preprocessed image.
pub const INP_RAW: &str = "inp_raw";

/// The nodes set B holds for an image tapped in full, in the order the dump
/// wrote them: [`POS_EMB`], [`POST_LN`], then each block's [`TAP_NODES`].
#[must_use]
pub fn tap_names() -> Vec<String> {
    let mut v = vec![POS_EMB.to_string(), POST_LN.to_string()];
    for b in TAP_BLOCKS {
        v.extend(TAP_NODES.iter().map(|n| format!("{n}-{b}")));
    }
    v
}

/// What a set holds, as its `# clefvis` line spells it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Set A: mtmd's preprocess of each test image.
    Preproc,
    /// Set B: the tower's taps.
    Taps,
    /// Set C: one prompt, the image rows mtmd's tower's.
    Hidden,
    /// Set C′: the same prompt with each image's rows prose-id text rows.
    Prose,
    /// Set C″: the same prompt with the tower's rows rounded to bf16.
    Bf16Rows,
}

impl Kind {
    /// The `# clefvis` spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Preproc => "preproc",
            Kind::Taps => "taps",
            Kind::Hidden => "hidden",
            Kind::Prose => "prose",
            Kind::Bf16Rows => "bf16rows",
        }
    }

    /// Every kind, in the order the recipe writes them; the table holds one family of each.
    pub const ALL: [Kind; 5] = [
        Kind::Preproc,
        Kind::Taps,
        Kind::Hidden,
        Kind::Prose,
        Kind::Bf16Rows,
    ];

    fn is_prompt(self) -> bool {
        matches!(self, Kind::Hidden | Kind::Prose | Kind::Bf16Rows)
    }
}

/// One `# image` line: a test image, the size mtmd planned for it and its
/// token grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub name: String,
    /// The PNG's sha256, so a gate can prove the file it reads is the one dumped.
    pub png_sha256: String,
    /// The sha256 of the RGB8 bytes stb_image decoded from it.
    pub rgb8_sha256: String,
    pub w: usize,
    pub h: usize,
    /// The planned size (mtmd's `calc_size_preserved_ratio`): a multiple of 32 a side.
    pub best_w: usize,
    pub best_h: usize,
    /// The merged token grid: `best_w / 32` by `best_h / 32`.
    pub nx: usize,
    pub ny: usize,
    pub n_tokens: usize,
    /// How far the image advances the decoder's position: `max(nx, ny)`.
    pub n_pos: usize,
}

/// One `# span` line: where an image sits in a prompt and how the decode
/// advanced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub index: usize,
    pub image: String,
    /// The first sequence index of the span, and its length (image-pad ids).
    pub at: usize,
    pub len: usize,
    /// The span's token grid; `0 x 0` in a prose set (no tower ran).
    pub nx: usize,
    pub ny: usize,
    pub n_pos: usize,
    /// The decoder position before the span and after it.
    pub start_pos: i64,
    pub end_pos: i64,
}

/// One `# tap_effect` line: whether the tap pass's final embeddings equal the
/// clean pass's, and whether the image carries its block taps.
#[derive(Debug, Clone, PartialEq)]
pub struct TapEffect {
    pub image: String,
    pub values: u64,
    pub differing: u64,
    pub max_abs_diff: f64,
    /// `true` when the image carries every node of [`tap_names`].
    pub full: bool,
}

/// A clefvis set: the node-dump manifest and what this module read of it.
#[derive(Debug, Clone)]
pub struct ClefvisSet {
    pub man: RefManifest,
    pub kind: Kind,
    /// `# device cpu`: the CPU twin.
    pub cpu: bool,
    /// The card the dump ran on (`# device … card <name>`), `cpu` for a twin.
    pub card: String,
    pub images: Vec<Image>,
    pub spans: Vec<Span>,
    pub tap_effect: Vec<TapEffect>,
}

struct At<'a>(&'a Path);

impl fmt::Display for At<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "clefvis: {}", self.0.display())
    }
}

/// The lines of the header that open with `# <key>\t`, their fields after the key.
fn lines<'a>(man: &'a RefManifest, key: &str) -> Vec<Vec<&'a str>> {
    let open = format!("# {key}\t");
    man.header
        .other
        .iter()
        .filter_map(|l| l.strip_prefix(open.as_str()))
        .map(|rest| rest.split('\t').collect())
        .collect()
}

/// The one line of the header that opens with `# <key>\t`.
fn one<'a>(man: &'a RefManifest, key: &str) -> Result<Vec<&'a str>, RefError> {
    let mut v = lines(man, key);
    match v.len() {
        1 => Ok(v.remove(0)),
        0 => Err(RefError::missing(
            &man.dir,
            format!("{}: no `# {key}` line", man.dir.display()),
        )),
        n => Err(RefError::malformed(
            At(&man.dir),
            format!("{n} `# {key}` lines, want one"),
        )),
    }
}

impl ClefvisSet {
    /// Read the set at `dir` as a set of `family`, a family of identity [`Identity::Clefvis`]: against the family
    /// ([`RefManifest::open`]: the trailer, the model file, the architecture and the mainline build), then against
    /// the family's kind. A set of another kind is [`RefError::Foreign`] on `clefvis`, one made with another mmproj
    /// [`RefError::Stale`], a row the kind needs that is missing or of another shape [`RefError::Malformed`]
    /// naming it, and so is a family of another identity.
    pub fn open(dir: &Path, family: &Family) -> Result<ClefvisSet, RefError> {
        let Identity::Clefvis(kind) = family.identity else {
            return Err(RefError::malformed(
                At(dir),
                format!("the {} family is not a clefvis family", family.name),
            ));
        };
        let man = RefManifest::open(dir, family)?;
        let at = At(dir);
        let line = one(&man, "clefvis")?;
        if line != [kind.as_str()] {
            return Err(RefError::Foreign {
                set: dir.display().to_string(),
                family: family.name,
                field: "clefvis",
                got: line.join(" "),
                want: kind.as_str().to_string(),
            });
        }
        let mm = one(&man, "mmproj")?;
        let [path, "sha256", digest] = mm[..] else {
            return Err(RefError::malformed(
                &at,
                format!("{mm:?}: want <path>\tsha256\t<digest>"),
            ));
        };
        if path != MMPROJ_BF16 || digest != MMPROJ_BF16_SHA256 {
            return Err(RefError::Stale {
                set: dir.display().to_string(),
                dumped_from: format!("mmproj {path} sha256 {digest}"),
                runs: format!("mmproj {MMPROJ_BF16} sha256 {MMPROJ_BF16_SHA256}"),
            });
        }
        let dev = one(&man, "device")?;
        let (cpu, card) = match dev[..] {
            [d @ ("cuda" | "cpu"), "threads", _, "card", card, ..] => {
                (d == "cpu", card.to_string())
            }
            _ => {
                return Err(RefError::malformed(
                    &at,
                    format!("{dev:?}: want cuda|cpu\tthreads\t<n>\tcard\t<name>…"),
                ));
            }
        };
        let images = read_images(&man)?;
        let spans = read_spans(&man)?;
        let tap_effect = read_tap_effect(&man)?;
        let set = ClefvisSet {
            man,
            kind,
            cpu,
            card,
            images,
            spans,
            tap_effect,
        };
        set.check_rows()?;
        Ok(set)
    }

    /// The `# image` line named `name`.
    pub fn image(&self, name: &str) -> Result<&Image, RefError> {
        self.images.iter().find(|i| i.name == name).ok_or_else(|| {
            RefError::missing(
                &self.man.dir,
                format!("{}: no image {name:?}", self.man.dir.display()),
            )
        })
    }

    /// The `# tap_effect` line of image `name`.
    pub fn tap_effect_of(&self, name: &str) -> Result<&TapEffect, RefError> {
        self.tap_effect
            .iter()
            .find(|t| t.image == name)
            .ok_or_else(|| {
                RefError::missing(
                    &self.man.dir,
                    format!("{}: no tap_effect line of {name:?}", self.man.dir.display()),
                )
            })
    }

    /// Row `<image>/<node>`, read and checked against its own row (f32, finite).
    fn tensor(&self, image: &str, node: &str) -> Result<(RefRow, Vec<f32>), RefError> {
        ik::load_ref_in(&self.man, &format!("{image}/{node}"), 0)
    }

    /// Image `name`'s preprocessed input, planar `3 x best_h x best_w` f32 ([`Kind::Preproc`]).
    pub fn inp_raw(&self, name: &str) -> Result<Vec<f32>, RefError> {
        self.need(Kind::Preproc, "inp_raw")?;
        Ok(self.tensor(name, INP_RAW)?.1)
    }

    /// Node `node` of image `name`'s tower ([`Kind::Taps`]): a name of [`tap_names`] or
    /// [`EMBD`]; the row's shape is `[1152, patches]` (`Qcur_rope`: `[72, 16, patches]`),
    /// `[4096, tokens]` for [`EMBD`]. An image the set holds `embd` only for
    /// ([`TapEffect::full`] false) refuses every other node by name.
    pub fn tap(&self, name: &str, node: &str) -> Result<(RefRow, Vec<f32>), RefError> {
        self.need(Kind::Taps, "tap")?;
        if node != EMBD && !self.tap_effect_of(name)?.full {
            return Err(RefError::missing(
                &self.man.dir,
                format!(
                    "{}: {name} carries {EMBD} only, not {node}",
                    self.man.dir.display()
                ),
            ));
        }
        self.tensor(name, node)
    }

    /// `result_norm` of every position `[4096, ids]`, row-major by position (a prompt kind).
    pub fn result_norm(&self) -> Result<Vec<f32>, RefError> {
        self.need_prompt("result_norm")?;
        Ok(ik::load_ref_in(&self.man, "result_norm", 0)?.1)
    }

    /// The (t, y, x) the decode was fed for each position (a prompt kind): the image rows' from
    /// the helper's own batches, a text row's three all its position.
    pub fn positions(&self) -> Result<Vec<[i32; 3]>, RefError> {
        self.need_prompt("mrope_pos")?;
        let v = ik::ref_ints(&self.man, "mrope_pos", 0, RowKind::Tensor, Layout::Flat)?;
        let to_i32 = |p: i64| {
            i32::try_from(p).map_err(|_| {
                RefError::malformed(At(&self.man.dir), format!("position {p} is not an i32"))
            })
        };
        v.as_chunks::<3>()
            .0
            .iter()
            .map(|c| Ok([to_i32(c[0])?, to_i32(c[1])?, to_i32(c[2])?]))
            .collect()
    }

    /// The number of ids of the prompt (a prompt kind): its `# tokens_count`.
    pub fn n_ids(&self) -> Result<usize, RefError> {
        self.need_prompt("n_ids")?;
        self.man
            .header
            .tokens_count
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| {
                RefError::missing(
                    &self.man.dir,
                    format!("{}: no `# tokens_count` line", self.man.dir.display()),
                )
            })
    }

    fn need(&self, kind: Kind, what: &str) -> Result<(), RefError> {
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

    fn need_prompt(&self, what: &str) -> Result<(), RefError> {
        if self.kind.is_prompt() {
            return Ok(());
        }
        Err(RefError::malformed(
            At(&self.man.dir),
            format!("{what} reads a prompt set, this is {}", self.kind.as_str()),
        ))
    }

    /// Every row the kind needs is there, of its shape.
    fn check_rows(&self) -> Result<(), RefError> {
        let at = At(&self.man.dir);
        let want = |name: String, ne: [u64; 4]| -> Result<(), RefError> {
            let r = self.man.tensor(&name, 0)?;
            if r.ne != ne {
                return Err(RefError::malformed(
                    &at,
                    format!("{name} is {:?}, want {ne:?}", r.ne),
                ));
            }
            Ok(())
        };
        let mut seen = std::collections::HashSet::new();
        for i in &self.images {
            if !seen.insert(&i.name) {
                return Err(RefError::malformed(&at, format!("image {} twice", i.name)));
            }
            if i.nx * i.ny != i.n_tokens
                || i.best_w != i.nx * ALIGN
                || i.best_h != i.ny * ALIGN
                || i.n_pos != i.nx.max(i.ny)
            {
                return Err(RefError::malformed(
                    &at,
                    format!(
                        "image {}: grid {}x{} of {} tokens, planned {}x{}, n_pos {}",
                        i.name, i.nx, i.ny, i.n_tokens, i.best_w, i.best_h, i.n_pos
                    ),
                ));
            }
        }
        match self.kind {
            Kind::Preproc => {
                for i in &self.images {
                    want(
                        format!("{}/{INP_RAW}", i.name),
                        [i.best_w as u64, i.best_h as u64, 3, 1],
                    )?;
                }
            }
            Kind::Taps => {
                for i in &self.images {
                    want(
                        format!("{}/{EMBD}", i.name),
                        [N_EMBD as u64, i.n_tokens as u64, 1, 1],
                    )?;
                    let full = self.tap_effect_of(&i.name)?.full;
                    if full {
                        for n in tap_names() {
                            let r = self.man.tensor(&format!("{}/{n}", i.name), 0)?;
                            if r.ne.iter().product::<u64>() % (4 * i.n_tokens as u64) != 0 {
                                return Err(RefError::malformed(
                                    &at,
                                    format!(
                                        "{}/{n} is {:?}, not a whole row per patch",
                                        i.name, r.ne
                                    ),
                                ));
                            }
                        }
                    }
                }
                if self.tap_effect.len() != self.images.len() {
                    return Err(RefError::malformed(
                        &at,
                        format!(
                            "{} images, {} tap_effect lines",
                            self.images.len(),
                            self.tap_effect.len()
                        ),
                    ));
                }
            }
            Kind::Hidden | Kind::Prose | Kind::Bf16Rows => {
                let pad = one(&self.man, "image_pad_id")?;
                if pad != [IMAGE_PAD_ID.to_string()] {
                    return Err(RefError::malformed(
                        &at,
                        format!(
                            "image_pad_id {pad:?}, the Clef vocabulary's <|image_pad|> is {IMAGE_PAD_ID}"
                        ),
                    ));
                }
                let n = self.n_ids()? as u64;
                want("result_norm".to_string(), [N_EMBD as u64, n, 1, 1])?;
                want("mrope_pos".to_string(), [3, n, 1, 1])?;
                let int =
                    ik::find_int_row(&self.man, "mrope_pos", 0, RowKind::Tensor, Layout::Flat)?;
                if int.twin != FileElem::I32 || int.count != 3 * n {
                    return Err(RefError::malformed(
                        &at,
                        format!(
                            "mrope_pos twin: {} elements of {:?}, want {} of I32",
                            int.count,
                            int.twin,
                            3 * n
                        ),
                    ));
                }
                let mut end = 0usize;
                for (k, s) in self.spans.iter().enumerate() {
                    if s.index != k || s.at < end || s.at + s.len > n as usize || s.len == 0 {
                        return Err(RefError::malformed(
                            &at,
                            format!(
                                "span {}: {} + {} of {n} ids after index {end}",
                                s.index, s.at, s.len
                            ),
                        ));
                    }
                    end = s.at + s.len;
                    if self.kind != Kind::Prose {
                        let i = self.image(&s.image)?;
                        if s.len != i.n_tokens || (s.nx, s.ny, s.n_pos) != (i.nx, i.ny, i.n_pos) {
                            return Err(RefError::malformed(
                                &at,
                                format!(
                                    "span {} holds {} ids as {}x{}, image {} is {} tokens as {}x{}",
                                    s.index, s.len, s.nx, s.ny, s.image, i.n_tokens, i.nx, i.ny
                                ),
                            ));
                        }
                    }
                }
                if (self.kind == Kind::Prose) != self.images.is_empty() {
                    return Err(RefError::malformed(
                        &at,
                        format!(
                            "a {} set with {} image lines",
                            self.kind.as_str(),
                            self.images.len()
                        ),
                    ));
                }
            }
        }
        Ok(())
    }
}

fn columns_of(man: &RefManifest, kind: &str) -> Result<Option<Columns>, RefError> {
    let v = lines(man, &format!("{kind} columns"));
    match &v[..] {
        [] => Ok(None),
        [one] => Ok(Some(Columns::new(
            kind,
            one.first().copied().unwrap_or("").split(' '),
        ))),
        _ => Err(RefError::malformed(
            At(&man.dir),
            format!("a second `# {kind} columns` line"),
        )),
    }
}

/// Every `# <kind>` line of the set as a row by the names of its `# <kind> columns` line;
/// none at all is fine, a line before its column line or of another width is not.
fn rows_of<T>(
    man: &RefManifest,
    kind: &str,
    parse: impl Fn(&Row<'_>) -> Result<T, RefError>,
) -> Result<Vec<T>, RefError> {
    let found: Vec<&String> = man
        .header
        .other
        .iter()
        .filter(|l| l.starts_with(&format!("# {kind}\t")))
        .collect();
    if found.is_empty() {
        return Ok(Vec::new());
    }
    let cols = columns_of(man, kind)?.ok_or_else(|| {
        RefError::malformed(
            At(&man.dir),
            format!("a `# {kind}` line with no `# {kind} columns` line"),
        )
    })?;
    let at = At(&man.dir);
    found
        .into_iter()
        .map(|l| parse(&cols.row(l.trim_start_matches("# "), &at)?))
        .collect()
}

fn read_images(man: &RefManifest) -> Result<Vec<Image>, RefError> {
    rows_of(man, "image", |r| {
        Ok(Image {
            name: r.text("name")?.to_string(),
            png_sha256: r.text("png_sha256")?.to_string(),
            rgb8_sha256: r.text("rgb8_sha256")?.to_string(),
            w: r.parse("w")?,
            h: r.parse("h")?,
            best_w: r.parse("best_w")?,
            best_h: r.parse("best_h")?,
            nx: r.parse("nx")?,
            ny: r.parse("ny")?,
            n_tokens: r.parse("n_tokens")?,
            n_pos: r.parse("n_pos")?,
        })
    })
}

fn read_spans(man: &RefManifest) -> Result<Vec<Span>, RefError> {
    rows_of(man, "span", |r| {
        Ok(Span {
            index: r.parse("index")?,
            image: r.text("image")?.to_string(),
            at: r.parse("at")?,
            len: r.parse("len")?,
            nx: r.parse("nx")?,
            ny: r.parse("ny")?,
            n_pos: r.parse("n_pos")?,
            start_pos: r.parse("start_pos")?,
            end_pos: r.parse("end_pos")?,
        })
    })
}

/// `# tap_effect\t<image>\tembd values <n>\tdiffering <n>\tmax_abs_diff <x>\ttaps full|inp_raw only`.
fn read_tap_effect(man: &RefManifest) -> Result<Vec<TapEffect>, RefError> {
    lines(man, "tap_effect")
        .into_iter()
        .map(|f| {
            let bad =
                |what: &str| RefError::malformed(At(&man.dir), format!("tap_effect {f:?}: {what}"));
            let value = |i: usize, key: &str| -> Result<&str, RefError> {
                f.get(i)
                    .and_then(|s| s.strip_prefix(key))
                    .map(str::trim)
                    .ok_or_else(|| bad(&format!("field {i} is not `{key} <value>`")))
            };
            let full = match value(4, "taps")? {
                "full" => true,
                "inp_raw only" => false,
                v => return Err(bad(&format!("taps {v:?}, want full or inp_raw only"))),
            };
            Ok(TapEffect {
                image: f.first().copied().unwrap_or("").to_string(),
                values: value(1, "embd values")?
                    .parse()
                    .map_err(|e| bad(&format!("{e}")))?,
                differing: value(2, "differing")?
                    .parse()
                    .map_err(|e| bad(&format!("{e}")))?,
                max_abs_diff: value(3, "max_abs_diff")?
                    .parse()
                    .map_err(|e| bad(&format!("{e}")))?,
                full,
            })
        })
        .collect()
}
