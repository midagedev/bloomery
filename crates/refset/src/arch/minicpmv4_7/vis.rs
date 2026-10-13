//! The MiniCPM-V 4.7 image-tower sets: llama.cpp mainline's mtmd `minicpmv4_7` graph
//! (`tools/mtmd/models/minicpmv.cpp`, `clip_graph_minicpmv4_6::build`) on the f16 projector [`MMPROJ_F16`], dumped by
//! `tools/ref/clefvis/dump_mtmd.cpp` (`just dump-ref-mcvis`, the canvas seat) at the mainline tree [`MC47_LCPP_BUILD`],
//! in the node dumps' set format. One family a kind of set ([`Kind`]); a set of the Qwen or GLM seats
//! ([`crate::clefvis`]) fails these families' check by its model file, its projector or its `# clefvis` kind, and a
//! set of these families fails theirs by the same lines: the kinds here are `canvas-preproc` and `canvas-taps`, never
//! Qwen's `preproc` and `taps`.
//!
//! An image is one chunk of tiles, `[overview, slices row by row]` (mtmd's canvas), and each tile is its own tower
//! graph, so a row is `<image>/t<k>/<node>`: tile 0 the overview, tile `k` the slice at row `(k - 1) / grid_x`, column
//! `(k - 1) % grid_x`. A tile's embeddings are its merger rows followed by its separator rows (`</image>`, `<slice>`,
//! `</slice>`, `\n` of the projector's `tok_embd_sep`), and the image's embeddings are the tiles' in order.
//!
//! - [`Kind::Preproc`] (set A): per tile `<image>/t<k>/inp_raw`, the f32 tile after mtmd's preprocess, channel-planar
//!   `[W, H, 3, 1]`, the tile graph's input.
//! - [`Kind::Taps`] (set B): per tile `<image>/t<k>/<node>` for the nodes of [`tap_names`], and
//!   `<image>/t<k>/embd` `[2048, rows]`, the clean pass's rows of the tile.
//!
//! The header lines this reader owns: `# clefvis`, `# mmproj`, `# projector`, `# positions`, `# device`,
//! `# image columns` and `# image`, `# tile columns` and `# tile`, `# tap_effect`; the rest are [`RefManifest`]'s.
//! [`Family::check_set`] runs [`RefManifest::open`] only (the model file, the architecture, the build and the trailer);
//! what is MiniCPM's is checked here, by [`McvisSet::open`].

use crate::RefError;
use crate::columns::Columns;
use crate::family::{Build, Family, Identity};
use crate::ik::{self, RefManifest, RefRow};
use std::fmt;
use std::path::Path;

/// The architecture a tower set's manifest names in its `# arch` line (the projector file's `general.architecture`).
pub const TOWER_ARCH: &str = "clip";

/// The projector every set is dumped with: openbmb's f16 file, under the box's `/models`.
pub const MMPROJ_F16: &str = "/models/MiniCPM-V-4.7-35B-A3B/mmproj-model-f16.gguf";

/// The sha256 of [`MMPROJ_F16`] (1,119,575,840 bytes).
// PIN(2026-10-13): `sha256sum` of the file on the box; the file came with the checkpoint's download and no remote digest
// has been compared.
pub const MMPROJ_F16_SHA256: &str =
    "bff3b688e14c300eca273092c37c50a0ddfcc2054e079af1a0480edeed5d20f1";

/// The mainline commit every set is dumped from (`git rev-parse --short=9`): 358a420ec4ed59b6b0be08b1d1f71dd478b99a99
/// in `/home/user/llama.cpp-358a420e` (`just build-mc47-lcpp`). Mainline gained `minicpmv4_7` at 69f201a20.
// PIN(2026-10-13): the oldest mainline commit read for this projector; a re-take at another commit changes it here.
pub const MC47_LCPP_BUILD: &str = "358a420ec";

/// The projector's `clip.projector_type`, the first field of the sets' `# projector` line.
pub const PROJECTOR: &str = "minicpmv4_7";
/// ViT patch size; the merger folds `N_MERGE` patches a side (the ViT merger 2x2, then the final merger 2x2), so a tile's
/// side is a multiple of [`ALIGN`].
pub const PATCH: usize = 14;
pub const N_MERGE: usize = 4;
pub const ALIGN: usize = PATCH * N_MERGE;
/// The ViT width, its heads and head width; the projector's output width (the text model's) and the block count.
pub const VIT_WIDTH: usize = 1152;
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = 72;
pub const N_EMBD: usize = 2048;
pub const BLOCKS: usize = 27;
/// The block after which the ViT merger runs: blocks `0..=6` see every patch, blocks `7..` a quarter of them.
pub const INSERT_LAYER: usize = 6;
/// The blocks tapped: the first and last before the ViT merger and after it.
pub const TAP_BLOCKS: [usize; 4] = [0, 6, 7, 26];
/// A tapped block's nodes, as `cb()` names them in `minicpmv.cpp`, each `-<block>`.
pub const BLOCK_NODES: [&str; 6] = [
    "layer_inp_normed",
    "Qcur",
    "attn_out",
    "ffn_inp",
    "ffn_out",
    "layer_out",
];
/// The nodes outside the blocks, in graph order: the position-embedded input, the ViT merger's, the post-LN and the
/// final merger's.
pub const GRAPH_NODES: [&str; 15] = [
    "pos_embed",
    "vit_merger_attn_inp_normed",
    "vit_merger_window_reorder",
    "vit_merger_Qcur",
    "vit_merger_Kcur",
    "vit_merger_Vcur",
    "vit_merger_attn_out",
    "vit_merger_attn_residual",
    "vit_merger_ds_mean_res",
    "vit_merger_ds_normed",
    "vit_merger_ds_mlp_out",
    "vit_merger_ds_out",
    "post_ln",
    "merger_normed",
    "merger_out",
];
/// A tile's clean-pass rows (merger rows then separator rows), `[N_EMBD, rows]`.
pub const EMBD: &str = "embd";
/// A tile's graph input, `[W, H, 3, 1]`.
pub const INP_RAW: &str = "inp_raw";

/// The text model's rope section order the canvas positions are laid out for (the checkpoint's `rope.section_order`).
pub(super) const SECTION_ORDER: &str = "zyxt";

/// The tapped nodes of a tile, in the order the dump writes them (without [`EMBD`]).
#[must_use]
pub fn tap_names() -> Vec<String> {
    let mut v: Vec<String> = GRAPH_NODES.iter().map(|n| (*n).to_string()).collect();
    for b in TAP_BLOCKS {
        v.extend(BLOCK_NODES.iter().map(|n| format!("{n}-{b}")));
    }
    v
}

/// The row of node `node` (or [`EMBD`], or [`INP_RAW`]) of tile `tile` of image `image`.
#[must_use]
pub fn row_name(image: &str, tile: usize, node: &str) -> String {
    format!("{image}/t{tile}/{node}")
}

/// The kinds of set the seat dumps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Set A: the tiles after mtmd's preprocess.
    Preproc,
    /// Set B: the tower's nodes of every tile and the final rows.
    Taps,
}

impl Kind {
    /// The `# clefvis` line of the kind's sets.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Preproc => "canvas-preproc",
            Kind::Taps => "canvas-taps",
        }
    }

    /// The family of the kind.
    #[must_use]
    pub fn family(self) -> &'static Family {
        match self {
            Kind::Preproc => &PREPROC,
            Kind::Taps => &TAPS,
        }
    }
}

/// [`MMPROJ_F16`], as a family's `runs`.
fn mmproj() -> Result<String, RefError> {
    Ok(MMPROJ_F16.to_string())
}

/// Set A: the tiles after mtmd's preprocess, the towers' graph input; the card only.
pub static PREPROC: Family = Family {
    name: "minicpmv47-vis-preproc",
    sets: &["ref_minicpmv47_vis_preproc"],
    resolve: None,
    recipe: "just dump-ref-mcvis ref_minicpmv47_vis_preproc",
    identity: Identity::Manifest,
    arch: Some(TOWER_ARCH),
    build: Some(Build::Is(MC47_LCPP_BUILD)),
    runs: Some(mmproj),
    draft_runs: None,
    consumers: &[],
};

/// Set B: the tower's nodes of every tile and its final rows; the card's set and the CPU twin.
pub static TAPS: Family = Family {
    name: "minicpmv47-vis-taps",
    sets: &["ref_minicpmv47_vis_taps", "ref_minicpmv47_vis_taps.cpu"],
    resolve: None,
    recipe: "just dump-ref-mcvis [--cpu-twin] ref_minicpmv47_vis_taps",
    identity: Identity::Manifest,
    arch: Some(TOWER_ARCH),
    build: Some(Build::Is(MC47_LCPP_BUILD)),
    runs: Some(mmproj),
    draft_runs: None,
    consumers: &[],
};

/// One `# tile` line: a tile of an image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tile {
    pub index: usize,
    /// `None`: the overview; else the slice's (row, column) in the grid.
    pub cell: Option<(usize, usize)>,
    /// The tile's planned size (`inp_raw`), a multiple of [`ALIGN`] a side.
    pub w: usize,
    pub h: usize,
    /// The token grid of the tile's merger rows: `w / ALIGN` by `h / ALIGN`.
    pub nx: usize,
    pub ny: usize,
    pub merger_rows: usize,
    /// The rows of the projector's separator table that follow the merger rows.
    pub suffix: Vec<usize>,
    /// The first row of the tile in the image's embeddings, and its length (merger rows and separator rows).
    pub row0: usize,
    pub rows: usize,
}

impl Tile {
    /// The patches a block before the ViT merger sees.
    #[must_use]
    pub fn patches(&self) -> usize {
        (self.w / PATCH) * (self.h / PATCH)
    }

    /// The tokens a block after the ViT merger sees: a quarter of the patches.
    #[must_use]
    pub fn merged(&self) -> usize {
        self.patches() / 4
    }
}

/// One `# image` line with its tiles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
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
    /// The rows of the image's embeddings, and how far the image advances the decoder (one a tile, one a separator).
    pub n_tokens: usize,
    pub n_pos: usize,
    pub tiles: Vec<Tile>,
}

/// One `# tap_effect` line: the tap pass's final rows against the clean pass's.
#[derive(Debug, Clone, PartialEq)]
pub struct TapEffect {
    pub image: String,
    pub values: usize,
    pub differing: usize,
    pub max_abs_diff: f64,
}

/// A MiniCPM-V 4.7 tower set: the node-dump manifest and what this module read of it.
#[derive(Debug, Clone)]
pub struct McvisSet {
    pub man: RefManifest,
    pub kind: Kind,
    /// `# device cpu`: the CPU twin.
    pub cpu: bool,
    /// The card the dump ran on (`# device … card <name>`), `cpu` for a twin.
    pub card: String,
    pub images: Vec<Image>,
    pub tap_effect: Vec<TapEffect>,
}

pub(super) struct At<'a>(pub(super) &'a Path);

impl fmt::Display for At<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "minicpmv47 vis: {}", self.0.display())
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
pub(super) fn one<'a>(man: &'a RefManifest, key: &str) -> Result<Vec<&'a str>, RefError> {
    let mut v = lines(man, key);
    match v.len() {
        0 => Err(RefError::missing(
            &man.dir,
            format!("{}: no `# {key}` line", man.dir.display()),
        )),
        1 => Ok(v.remove(0)),
        n => Err(RefError::malformed(
            At(&man.dir),
            format!("{n} `# {key}` lines, want one"),
        )),
    }
}

/// Every `# <kind>` line of the set as a row by the names of its `# <kind> columns` line; a line before its column line
/// or of another width is `Malformed`.
pub(super) fn rows_of<T>(
    man: &RefManifest,
    kind: &str,
    parse: impl Fn(&crate::columns::Row<'_>) -> Result<T, RefError>,
) -> Result<Vec<T>, RefError> {
    let at = At(&man.dir);
    let found: Vec<&String> = man
        .header
        .other
        .iter()
        .filter(|l| l.starts_with(&format!("# {kind}\t")))
        .collect();
    let cols = lines(man, &format!("{kind} columns"));
    let cols = match &cols[..] {
        [names] => Columns::new(kind, names.first().copied().unwrap_or("").split(' ')),
        [] if found.is_empty() => return Ok(Vec::new()),
        [] => {
            return Err(RefError::malformed(
                &at,
                format!("a `# {kind}` line with no `# {kind} columns` line"),
            ));
        }
        _ => {
            return Err(RefError::malformed(
                &at,
                format!("a second `# {kind} columns` line"),
            ));
        }
    };
    found
        .into_iter()
        .map(|l| parse(&cols.row(l.trim_start_matches("# "), &at)?))
        .collect()
}

/// mtmd's `add_canvas_chunk` and `clip_suffix_rows`, spelled out: the overview closes with `</image>` (row 0) and opens
/// the slices with `<slice>` (row 1) when there are some; a slice closes with `</slice>` (2) and opens the next with
/// `<slice>` (1), the last of a row adds `\n` (3) between, the very last closes with `</slice>` only.
pub(super) fn canvas_suffix(tile: usize, grid_x: usize, grid_y: usize) -> Vec<usize> {
    if tile == 0 {
        return if grid_x > 0 { vec![0, 1] } else { vec![0] };
    }
    let (y, x) = ((tile - 1) / grid_x, (tile - 1) % grid_x);
    if y == grid_y - 1 && x == grid_x - 1 {
        vec![2]
    } else if x == grid_x - 1 {
        vec![2, 3, 1]
    } else {
        vec![2, 1]
    }
}

/// The shape of node `node` of `tile` (a name of [`tap_names`]); `None` for a name that is not one.
#[must_use]
pub fn tap_shape(node: &str, tile: &Tile) -> Option<[u64; 4]> {
    let (p, n1, r) = (
        tile.patches() as u64,
        tile.merged() as u64,
        tile.merger_rows as u64,
    );
    let d = VIT_WIDTH as u64;
    let qkv = |n: u64| [HEAD_DIM as u64, HEADS as u64, n, 1];
    Some(match node {
        "pos_embed"
        | "vit_merger_attn_inp_normed"
        | "vit_merger_window_reorder"
        | "vit_merger_attn_out"
        | "vit_merger_attn_residual" => [d, p, 1, 1],
        "vit_merger_Qcur" | "vit_merger_Kcur" | "vit_merger_Vcur" => qkv(p),
        "vit_merger_ds_mean_res" | "vit_merger_ds_mlp_out" | "vit_merger_ds_out" | "post_ln" => {
            [d, n1, 1, 1]
        }
        "vit_merger_ds_normed" => [4 * d, n1, 1, 1],
        "merger_normed" => [4 * d, r, 1, 1],
        "merger_out" => [N_EMBD as u64, r, 1, 1],
        _ => {
            let (name, block) = node.rsplit_once('-')?;
            let block: usize = block.parse().ok()?;
            if !TAP_BLOCKS.contains(&block) || !BLOCK_NODES.contains(&name) {
                return None;
            }
            let n = if block <= INSERT_LAYER { p } else { n1 };
            if name == "Qcur" { qkv(n) } else { [d, n, 1, 1] }
        }
    })
}

fn want_shape(man: &RefManifest, name: &str, ne: [u64; 4], at: &At<'_>) -> Result<(), RefError> {
    let r = man.tensor(name, 0)?;
    if r.ty != "f32" || r.ne != ne {
        return Err(RefError::malformed(
            at,
            format!("{name} is {} {:?}, want f32 {ne:?}", r.ty, r.ne),
        ));
    }
    Ok(())
}

/// What a set states about the run that made it.
pub(super) struct Seat {
    /// `# device cpu`: the CPU twin.
    pub cpu: bool,
    /// The card the dump ran on (`# device … card <name>`), `cpu` for a twin.
    pub card: String,
}

/// The identity every set of the seat carries, tower or prompt: against the family ([`RefManifest::open`]: the trailer, the
/// model file, the architecture and the mainline build), then the lines this module owns: `# clefvis` is `kind`, the
/// projector file and its digest, the `# projector` and `# positions` lines, and the device. A set of another kind, or of
/// the Qwen or GLM seats, is [`RefError::Foreign`] on `clefvis`, `projector` or `positions`; one made with another mmproj
/// [`RefError::Stale`].
pub(super) fn open_seat(
    dir: &Path,
    family: &'static Family,
    kind: &str,
) -> Result<(RefManifest, Seat), RefError> {
    let man = RefManifest::open(dir, family)?;
    let at = At(dir);
    let foreign = |field: &'static str, got: String, want: String| RefError::Foreign {
        set: dir.display().to_string(),
        family: family.name,
        field,
        got,
        want,
    };
    let line = one(&man, "clefvis")?;
    if line != [kind] {
        return Err(foreign("clefvis", line.join(" "), kind.into()));
    }
    let mm = one(&man, "mmproj")?;
    let [path, "sha256", digest] = mm[..] else {
        return Err(RefError::malformed(
            &at,
            format!("{mm:?}: want <path>\tsha256\t<digest>"),
        ));
    };
    if path != MMPROJ_F16 || digest != MMPROJ_F16_SHA256 {
        return Err(RefError::Stale {
            set: dir.display().to_string(),
            dumped_from: format!("mmproj {path} sha256 {digest}"),
            runs: format!("mmproj {MMPROJ_F16} sha256 {MMPROJ_F16_SHA256}"),
        });
    }
    let want_projector = format!(
        "{PROJECTOR}\tpatch_size {PATCH}\timage_size 448\tn_merge {N_MERGE}\tprojection_dim {N_EMBD}\tblocks {BLOCKS}\tinsert_layer_id {INSERT_LAYER}"
    );
    let projector = one(&man, "projector")?.join("\t");
    if projector != want_projector {
        return Err(foreign("projector", projector, want_projector));
    }
    let want_positions = format!("canvas\tqwen35moe.rope.section_order {SECTION_ORDER}");
    let positions = one(&man, "positions")?.join("\t");
    if positions != want_positions {
        return Err(foreign("positions", positions, want_positions));
    }
    let dev = one(&man, "device")?;
    let (cpu, card) = match dev[..] {
        [d @ ("cuda" | "cpu"), "threads", _, "card", card, ..] => (d == "cpu", card.to_string()),
        _ => {
            return Err(RefError::malformed(
                &at,
                format!("{dev:?}: want cuda|cpu\tthreads\t<n>\tcard\t<name>…"),
            ));
        }
    };
    Ok((man, Seat { cpu, card }))
}

impl McvisSet {
    /// Read the set at `dir` as a set of `kind`'s family: against the family ([`RefManifest::open`]: the trailer, the
    /// projector file as the model, the architecture and the mainline build), then against this module's pins. A set of
    /// another kind, or of the Qwen or GLM seats, is [`RefError::Foreign`] on `clefvis` or `projector`; one made with
    /// another mmproj [`RefError::Stale`]; a line or row that is missing or of another shape [`RefError::Malformed`]
    /// naming it.
    pub fn open(dir: &Path, kind: Kind) -> Result<McvisSet, RefError> {
        let (man, Seat { cpu, card }) = open_seat(dir, kind.family(), kind.as_str())?;
        let images = read_images(&man)?;
        let tap_effect = if kind == Kind::Taps {
            read_tap_effect(&man, &images)?
        } else {
            Vec::new()
        };
        let set = McvisSet {
            man,
            kind,
            cpu,
            card,
            images,
            tap_effect,
        };
        set.check_rows()?;
        Ok(set)
    }

    /// The `# image` line named `name`, with its tiles.
    pub fn image(&self, name: &str) -> Result<&Image, RefError> {
        self.images.iter().find(|i| i.name == name).ok_or_else(|| {
            RefError::missing(
                &self.man.dir,
                format!("{}: no image {name:?}", self.man.dir.display()),
            )
        })
    }

    /// Tile `tile` of image `name`.
    pub fn tile(&self, name: &str, tile: usize) -> Result<&Tile, RefError> {
        self.image(name)?.tiles.get(tile).ok_or_else(|| {
            RefError::missing(
                &self.man.dir,
                format!("{}: {name} has no tile {tile}", self.man.dir.display()),
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

    /// Tile `tile` of image `name`, planar `3 x H x W` f32 ([`Kind::Preproc`]).
    pub fn inp_raw(&self, name: &str, tile: usize) -> Result<Vec<f32>, RefError> {
        self.need(Kind::Preproc, "inp_raw")?;
        self.tile(name, tile)?;
        Ok(ik::load_ref_in(&self.man, &row_name(name, tile, INP_RAW), 0)?.1)
    }

    /// Node `node` of tile `tile` of image `name` ([`Kind::Taps`]): a name of [`tap_names`] with the shape
    /// [`tap_shape`] gives, or [`EMBD`], `[2048, rows]` of the tile.
    pub fn tap(&self, name: &str, tile: usize, node: &str) -> Result<(RefRow, Vec<f32>), RefError> {
        self.need(Kind::Taps, "tap")?;
        let t = self.tile(name, tile)?;
        if node != EMBD && tap_shape(node, t).is_none() {
            return Err(RefError::missing(
                &self.man.dir,
                format!("{}: {node} is no tapped node", self.man.dir.display()),
            ));
        }
        ik::load_ref_in(&self.man, &row_name(name, tile, node), 0)
    }

    /// The set holds exactly the rows its kind needs, of the shapes the tiles give.
    fn check_rows(&self) -> Result<(), RefError> {
        let at = At(&self.man.dir);
        let mut want = 0usize;
        for im in &self.images {
            for t in &im.tiles {
                match self.kind {
                    Kind::Preproc => {
                        let ne = [t.w as u64, t.h as u64, 3, 1];
                        want_shape(&self.man, &row_name(&im.name, t.index, INP_RAW), ne, &at)?;
                        want += 1;
                    }
                    Kind::Taps => {
                        for n in tap_names() {
                            let Some(ne) = tap_shape(&n, t) else {
                                return Err(RefError::malformed(&at, format!("{n} has no shape")));
                            };
                            want_shape(&self.man, &row_name(&im.name, t.index, &n), ne, &at)?;
                            want += 1;
                        }
                        let ne = [N_EMBD as u64, t.rows as u64, 1, 1];
                        want_shape(&self.man, &row_name(&im.name, t.index, EMBD), ne, &at)?;
                        want += 1;
                    }
                }
            }
        }
        if self.man.tensors.len() != want {
            return Err(RefError::malformed(
                &at,
                format!(
                    "{} tensor rows, the {} tiles of {} images want {want}",
                    self.man.tensors.len(),
                    self.images.iter().map(|i| i.tiles.len()).sum::<usize>(),
                    self.images.len()
                ),
            ));
        }
        Ok(())
    }
}

/// The `# image` lines, each with its `# tile` lines checked against the canvas rules and against the image's totals.
fn read_images(man: &RefManifest) -> Result<Vec<Image>, RefError> {
    let at = At(&man.dir);
    let mut images = rows_of(man, "image", |r| {
        Ok(Image {
            name: r.text("name")?.to_string(),
            png_sha256: r.text("png_sha256")?.to_string(),
            rgb8_sha256: r.text("rgb8_sha256")?.to_string(),
            w: r.parse("w")?,
            h: r.parse("h")?,
            grid_x: r.parse("grid_x")?,
            grid_y: r.parse("grid_y")?,
            n_tokens: r.parse("n_tokens")?,
            n_pos: r.parse("n_pos")?,
            tiles: Vec::new(),
        })
    })?;
    if images.is_empty() {
        return Err(RefError::malformed(&at, "no `# image` line"));
    }
    let n_tiles = rows_of(man, "image", |r| r.parse::<usize>("n_tiles"))?;
    let tiles = rows_of(man, "tile", |r| {
        let cell = |name: &str| -> Result<Option<usize>, RefError> {
            match r.text(name)? {
                "-" => Ok(None),
                v => crate::columns::parse_field(v, name, &at).map(Some),
            }
        };
        let suffix: Result<Vec<usize>, RefError> = r
            .text("suffix")?
            .split(',')
            .map(|v| crate::columns::parse_field(v, "suffix", &at))
            .collect();
        Ok((
            r.text("image")?.to_string(),
            Tile {
                index: r.parse("k")?,
                cell: cell("row")?.zip(cell("col")?),
                w: r.parse("w")?,
                h: r.parse("h")?,
                nx: r.parse("nx")?,
                ny: r.parse("ny")?,
                merger_rows: r.parse("merger_rows")?,
                suffix: suffix?,
                row0: r.parse("row0")?,
                rows: r.parse("rows")?,
            },
        ))
    })?;
    for (name, tile) in tiles {
        let im = images.iter_mut().find(|i| i.name == name).ok_or_else(|| {
            RefError::malformed(&at, format!("a tile of {name:?}, no such image"))
        })?;
        im.tiles.push(tile);
    }
    for (im, &declared) in images.iter().zip(&n_tiles) {
        check_tiles(im, declared, &at)?;
    }
    Ok(images)
}

/// The invariants of one image's tiles: the canvas layout mtmd builds for its slice grid, the rows tiling the image's
/// embeddings and the positions it advances by.
fn check_tiles(im: &Image, declared: usize, at: &At<'_>) -> Result<(), RefError> {
    let bad = |what: String| RefError::malformed(at, format!("{}: {what}", im.name));
    let slices = im.grid_x * im.grid_y;
    if (im.grid_x == 0) != (im.grid_y == 0) {
        return Err(bad(format!("a {} x {} slice grid", im.grid_x, im.grid_y)));
    }
    if im.tiles.len() != declared || declared != 1 + slices {
        return Err(bad(format!(
            "{} tile lines, {declared} tiles declared, a {} x {} grid wants {}",
            im.tiles.len(),
            im.grid_x,
            im.grid_y,
            1 + slices
        )));
    }
    let (mut row0, mut separators) = (0usize, 0usize);
    for (k, t) in im.tiles.iter().enumerate() {
        let cell = (k > 0).then(|| ((k - 1) / im.grid_x, (k - 1) % im.grid_x));
        if t.index != k || t.cell != cell {
            return Err(bad(format!(
                "tile line {k} is tile {} at {:?}, want {cell:?}",
                t.index, t.cell
            )));
        }
        if t.w == 0 || t.h == 0 || t.w % ALIGN != 0 || t.h % ALIGN != 0 {
            return Err(bad(format!(
                "tile {k} is {} x {}, not a multiple of {ALIGN}",
                t.w, t.h
            )));
        }
        if t.nx != t.w / ALIGN || t.ny != t.h / ALIGN || t.merger_rows != t.nx * t.ny {
            return Err(bad(format!(
                "tile {k} is {} x {} tokens with {} merger rows, its {} x {} pixels give {} x {}",
                t.nx,
                t.ny,
                t.merger_rows,
                t.w,
                t.h,
                t.w / ALIGN,
                t.h / ALIGN
            )));
        }
        let suffix = canvas_suffix(k, im.grid_x, im.grid_y);
        if t.suffix != suffix {
            return Err(bad(format!(
                "tile {k} has separator rows {:?}, the canvas wants {suffix:?}",
                t.suffix
            )));
        }
        if t.row0 != row0 || t.rows != t.merger_rows + t.suffix.len() {
            return Err(bad(format!(
                "tile {k} holds rows {}..+{}, want {row0}..+{}",
                t.row0,
                t.rows,
                t.merger_rows + t.suffix.len()
            )));
        }
        row0 += t.rows;
        separators += t.suffix.len();
    }
    if row0 != im.n_tokens || im.n_pos != im.tiles.len() + separators {
        return Err(bad(format!(
            "the tiles hold {row0} rows and advance {} positions, the image line says {} and {}",
            im.tiles.len() + separators,
            im.n_tokens,
            im.n_pos
        )));
    }
    Ok(())
}

/// The text after `key ` in a `<key> <value>` field.
fn after<'a>(field: &'a str, key: &str) -> Option<&'a str> {
    field.strip_prefix(key)?.strip_prefix(' ')
}

/// The `# tap_effect` line of every image: how the tap pass's final rows compare with the clean pass's.
fn read_tap_effect(man: &RefManifest, images: &[Image]) -> Result<Vec<TapEffect>, RefError> {
    let at = At(&man.dir);
    let bad = |what: String| RefError::malformed(&at, what);
    let mut out = Vec::new();
    for f in lines(man, "tap_effect") {
        let [image, values, differing, max_abs_diff, "taps full"] = f[..] else {
            return Err(bad(format!(
                "{f:?}: want <image>\tembd values N\tdiffering N\tmax_abs_diff X\ttaps full"
            )));
        };
        let field = |v: &str, key: &str| -> Result<String, RefError> {
            after(v, key)
                .map(str::to_string)
                .ok_or_else(|| bad(format!("{image}: {v:?}, want `{key} <value>`")))
        };
        let num = |v: String, key: &str| -> Result<usize, RefError> {
            v.parse()
                .map_err(|e| bad(format!("{image}: {key} {v:?}: {e}")))
        };
        let max = field(max_abs_diff, "max_abs_diff")?;
        out.push(TapEffect {
            image: image.to_string(),
            values: num(field(values, "embd values")?, "embd values")?,
            differing: num(field(differing, "differing")?, "differing")?,
            max_abs_diff: max
                .parse()
                .map_err(|e| bad(format!("{image}: max_abs_diff {max:?}: {e}")))?,
        });
    }
    for im in images {
        let e = out
            .iter()
            .find(|e| e.image == im.name)
            .ok_or_else(|| bad(format!("no `# tap_effect` line of {:?}", im.name)))?;
        if e.values != im.n_tokens * N_EMBD {
            return Err(bad(format!(
                "{}: tap_effect counts {} values, the image's rows are {}",
                im.name,
                e.values,
                im.n_tokens * N_EMBD
            )));
        }
    }
    Ok(out)
}
