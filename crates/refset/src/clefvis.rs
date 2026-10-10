//! The image-input oracle sets of the Qwen3-VL family seats
//! (`tools/ref/clefvis/dump_mtmd.cpp`): llama.cpp mainline's mtmd
//! `qwen3vl_merger` tower and a family's text model (Clef-Flash's qwen35 on
//! its Q8_0 file, Qwen3.6's qwen35moe, Qwen3.8's qwen4exp), written in the
//! node-dump set format (`tools/ref/dump_ref.cpp`), so a family's identity
//! check ([`crate::arch::qwen35::clefvis`], [`crate::arch::qwen35moe::vis`],
//! [`crate::arch::qwen4exp::vis`], through [`RefManifest::open`]) already
//! refuses a set of another file, another build or another architecture and
//! a set without its trailer. This module reads what the format leaves to the
//! kind of set: the `# clefvis <kind>` line, the mmproj the set names, the
//! images and spans, and the rows each kind must carry, and hands the values
//! out by name. What differs between the seats is a [`Profile`], the family's
//! `Identity::Clefvis` payload: the projector file and its digest, the text
//! model's width and the image-pad id.
//!
//! - [`Kind::Preproc`] (set A): per image `<name>/inp_raw`, the f32 image after
//!   mtmd's preprocess, channel-planar `[W, H, 3, 1]`, the tower's graph input.
//! - [`Kind::Taps`] (set B): per image `<name>/<node>` for the nodes of
//!   [`tap_names`], and `<name>/embd`, the final embeddings `[n_embd, tokens]`;
//!   a set tapped [`TapScope::Final`] holds `embd` and the post-LN output
//!   [`POST_LN`] of every image.
//! - [`Kind::Hidden`], [`Kind::Prose`] and [`Kind::Bf16Rows`] (sets C, C′, C″):
//!   one prompt's `result_norm` of every position `[n_embd, ids]` and the
//!   positions the decode was fed, `mrope_pos` `[3, ids]` (t, y, x), with the
//!   spans of its images.
//! - [`Kind::ChatIds`] (set E): the ids llama-server's chat path gives one
//!   request, `ids` `[n]`, the rendered prompt, the chat template's digest and
//!   the spans of its images.
//! - [`Kind::Decode`] (set F): greedy decode steps after a prompt, `ids`
//!   `[steps]`, `result_norm` `[n_embd, steps]` and `n_past` `[steps]`, with
//!   the position the prompt left and the spans of its images.
//!
//! The header lines this reader owns: `# clefvis`, `# mmproj`, `# device`,
//! `# image columns` and `# image`, `# span columns` and `# span`,
//! `# tap_effect`, `# prompt`, `# chat`, `# decode`; the rest are
//! [`RefManifest`]'s.

use crate::RefError;
use crate::columns::{Columns, Row};
use crate::family::{Family, Identity};
use crate::ik::{self, FileElem, Layout, RefManifest, RefRow, RowKind};
use std::fmt;
use std::path::Path;

/// The architecture a tower set's manifest names in its `# arch` line (the projector's `general.architecture`).
pub const TOWER_ARCH: &str = "clip";

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

/// What a seat's sets state that no other seat's do: the projector file and its digest, the text model's width and
/// the id of its `<|image_pad|>`. A family's `Identity::Clefvis` carries one.
#[derive(Debug, PartialEq, Eq)]
pub struct Profile {
    /// The projector file every set of the seat is dumped with.
    pub mmproj: &'static str,
    /// The sha256 of [`Profile::mmproj`].
    pub mmproj_sha256: &'static str,
    /// The text model's width, and so the width of the tower's output rows.
    pub n_embd: usize,
    /// The token id of `<|image_pad|>` in the text model's vocabulary.
    pub image_pad_id: u32,
}

/// The mainline commit the Qwen3.6 and Qwen3.8 image-input sets name in their `# build` line.
// PIN(2026-10-10): /home/user/llama.cpp-36a73916 at its HEAD 36a73916ee0c (ggml-org master, 2026-10-07), the tree that
// carries the qwen4exp and GLM5-Next fixes after Clef's pin (`crate::arch::qwen35::LCPP_BUILD`); both trees stay on the box.
pub const QVIS_LCPP_BUILD: &str = "36a73916e";

/// Clef-Flash's seat.
pub static CLEF: Profile = Profile {
    mmproj: MMPROJ_BF16,
    mmproj_sha256: MMPROJ_BF16_SHA256,
    n_embd: N_EMBD,
    image_pad_id: IMAGE_PAD_ID,
};

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
    /// Set E: the ids llama-server's chat path gives one request.
    ChatIds,
    /// Set F: greedy decode steps after a prompt.
    Decode,
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
            Kind::ChatIds => "chatids",
            Kind::Decode => "decode",
        }
    }

    /// Every kind, in the order the recipe writes them; the table holds one family of each.
    pub const ALL: [Kind; 7] = [
        Kind::Preproc,
        Kind::Taps,
        Kind::Hidden,
        Kind::Prose,
        Kind::Bf16Rows,
        Kind::ChatIds,
        Kind::Decode,
    ];

    /// Whether a set of the kind holds `result_norm` and `mrope_pos` of every position of a prompt.
    fn is_prompt(self) -> bool {
        matches!(self, Kind::Hidden | Kind::Prose | Kind::Bf16Rows)
    }

    /// Whether a set of the kind states its prompt's ids: `# tokens_count`, `# image_pad_id` and the spans.
    fn has_prompt_ids(self) -> bool {
        self.is_prompt() || matches!(self, Kind::ChatIds | Kind::Decode)
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

/// Which nodes of the tower an image carries besides its final embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapScope {
    /// Every node of [`tap_names`].
    Full,
    /// The post-LN output [`POST_LN`] only: the tower's output end.
    Final,
    /// None: the image is larger than the dump's `--tap-patches`.
    EmbdOnly,
}

/// One `# tap_effect` line: whether the tap pass's final embeddings equal the
/// clean pass's, and which nodes the image carries.
#[derive(Debug, Clone, PartialEq)]
pub struct TapEffect {
    pub image: String,
    pub values: u64,
    pub differing: u64,
    pub max_abs_diff: f64,
    pub scope: TapScope,
}

/// A clefvis set: the node-dump manifest and what this module read of it.
#[derive(Debug, Clone)]
pub struct ClefvisSet {
    pub man: RefManifest,
    pub kind: Kind,
    /// The seat the family belongs to.
    pub profile: &'static Profile,
    /// `# device cpu`: the CPU twin.
    pub cpu: bool,
    /// The card the dump ran on (`# device … card <name>`), `cpu` for a twin.
    pub card: String,
    pub images: Vec<Image>,
    pub spans: Vec<Span>,
    pub tap_effect: Vec<TapEffect>,
    /// The `# chat` line of a [`Kind::ChatIds`] set.
    pub chat: Option<ChatLine>,
    /// The `# decode` line of a [`Kind::Decode`] set.
    pub decode: Option<DecodeLine>,
}

/// The `# chat` line of a chat-ids set: how the request was rendered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatLine {
    /// The sha256 of the model's chat template source.
    pub template_sha256: String,
    pub enable_thinking: bool,
    pub add_generation_prompt: bool,
    /// A trailing assistant message was continued, not closed.
    pub continue_final_message: bool,
}

/// The `# decode` line of a decode set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeLine {
    pub steps: usize,
    /// The position after the prompt, where the first decoded token sits.
    pub n_past_start: i64,
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
    /// the family's kind and profile. A set of another kind is [`RefError::Foreign`] on `clefvis`, one made with
    /// another mmproj [`RefError::Stale`], a row the kind needs that is missing or of another shape
    /// [`RefError::Malformed`] naming it, and so is a family of another identity.
    pub fn open(dir: &Path, family: &Family) -> Result<ClefvisSet, RefError> {
        let Identity::Clefvis(kind, profile) = family.identity else {
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
        if path != profile.mmproj || digest != profile.mmproj_sha256 {
            return Err(RefError::Stale {
                set: dir.display().to_string(),
                dumped_from: format!("mmproj {path} sha256 {digest}"),
                runs: format!("mmproj {} sha256 {}", profile.mmproj, profile.mmproj_sha256),
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
        let chat = if kind == Kind::ChatIds {
            Some(read_chat(&man)?)
        } else {
            None
        };
        let decode = if kind == Kind::Decode {
            Some(read_decode(&man)?)
        } else {
            None
        };
        let set = ClefvisSet {
            man,
            kind,
            profile,
            cpu,
            card,
            images,
            spans,
            tap_effect,
            chat,
            decode,
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
    /// `[n_embd, tokens]` for [`EMBD`]. An image holds the nodes of its [`TapScope`] and [`EMBD`];
    /// every other node is refused by name.
    pub fn tap(&self, name: &str, node: &str) -> Result<(RefRow, Vec<f32>), RefError> {
        self.need(Kind::Taps, "tap")?;
        let carries = match self.tap_effect_of(name)?.scope {
            TapScope::Full => true,
            TapScope::Final => node == POST_LN,
            TapScope::EmbdOnly => false,
        };
        if node != EMBD && !carries {
            let only = match self.tap_effect_of(name)?.scope {
                TapScope::Final => format!("{POST_LN} and {EMBD} only"),
                _ => format!("{EMBD} only"),
            };
            return Err(RefError::missing(
                &self.man.dir,
                format!(
                    "{}: {name} carries {only}, not {node}",
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

    /// The number of ids of the prompt (a prompt kind, a chat-ids or a decode set): its `# tokens_count`.
    pub fn n_ids(&self) -> Result<usize, RefError> {
        self.need_prompt_ids("n_ids")?;
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

    /// The ids the chat path gave the request ([`Kind::ChatIds`]), image runs as the image-pad id repeated.
    pub fn ids(&self) -> Result<Vec<u32>, RefError> {
        self.need(Kind::ChatIds, "ids")?;
        self.u32s("ids")
    }

    /// The prompt the chat template rendered, before tokenization ([`Kind::ChatIds`]).
    pub fn prompt(&self) -> Result<String, RefError> {
        self.need(Kind::ChatIds, "prompt")?;
        let line = one(&self.man, "prompt")?;
        let [raw] = line[..] else {
            return Err(RefError::malformed(
                At(&self.man.dir),
                format!("a `# prompt` line of {} fields, want one", line.len()),
            ));
        };
        unescape(raw).map_err(|e| RefError::malformed(At(&self.man.dir), format!("# prompt: {e}")))
    }

    /// The sha256 of [`ClefvisSet::prompt`] as the dump computed it ([`Kind::ChatIds`]).
    pub fn prompt_sha256(&self) -> Result<String, RefError> {
        self.need(Kind::ChatIds, "prompt_sha256")?;
        Ok(one(&self.man, "prompt_sha256")?.join("\t"))
    }

    /// The `# chat` line ([`Kind::ChatIds`]).
    pub fn chat(&self) -> Result<&ChatLine, RefError> {
        self.need(Kind::ChatIds, "chat")?;
        self.chat.as_ref().ok_or_else(|| {
            RefError::missing(&self.man.dir, "a chat-ids set without its `# chat` line")
        })
    }

    /// The `# decode` line ([`Kind::Decode`]).
    pub fn decode(&self) -> Result<&DecodeLine, RefError> {
        self.need(Kind::Decode, "decode")?;
        self.decode.as_ref().ok_or_else(|| {
            RefError::missing(&self.man.dir, "a decode set without its `# decode` line")
        })
    }

    /// The token each decode step took, the argmax of the step before's last row ([`Kind::Decode`]).
    pub fn decode_ids(&self) -> Result<Vec<u32>, RefError> {
        self.need(Kind::Decode, "decode_ids")?;
        self.u32s("ids")
    }

    /// `result_norm` of each decoded token's row `[n_embd, steps]`, row-major by step ([`Kind::Decode`]).
    pub fn decode_result_norm(&self) -> Result<Vec<f32>, RefError> {
        self.need(Kind::Decode, "decode_result_norm")?;
        Ok(ik::load_ref_in(&self.man, "result_norm", 0)?.1)
    }

    /// The position after each decoded token ([`Kind::Decode`]).
    pub fn decode_n_past(&self) -> Result<Vec<i32>, RefError> {
        self.need(Kind::Decode, "decode_n_past")?;
        ik::ref_ints(&self.man, "n_past", 0, RowKind::Tensor, Layout::Flat)?
            .into_iter()
            .map(|p| {
                i32::try_from(p).map_err(|_| {
                    RefError::malformed(At(&self.man.dir), format!("position {p} is not an i32"))
                })
            })
            .collect()
    }

    /// The int row `name` as token ids.
    fn u32s(&self, name: &str) -> Result<Vec<u32>, RefError> {
        ik::ref_ints(&self.man, name, 0, RowKind::Tensor, Layout::Flat)?
            .into_iter()
            .map(|t| {
                u32::try_from(t).map_err(|_| {
                    RefError::malformed(At(&self.man.dir), format!("{name}: {t} is not a token id"))
                })
            })
            .collect()
    }

    fn need_prompt_ids(&self, what: &str) -> Result<(), RefError> {
        if self.kind.has_prompt_ids() {
            return Ok(());
        }
        Err(RefError::malformed(
            At(&self.man.dir),
            format!(
                "{what} reads a set that states its prompt's ids, this is {}",
                self.kind.as_str()
            ),
        ))
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
        let n_embd = self.profile.n_embd as u64;
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
                        [n_embd, i.n_tokens as u64, 1, 1],
                    )?;
                    let nodes = match self.tap_effect_of(&i.name)?.scope {
                        TapScope::Full => tap_names(),
                        TapScope::Final => vec![POST_LN.to_string()],
                        TapScope::EmbdOnly => Vec::new(),
                    };
                    for n in nodes {
                        let r = self.man.tensor(&format!("{}/{n}", i.name), 0)?;
                        if r.ne.iter().product::<u64>() % (4 * i.n_tokens as u64) != 0 {
                            return Err(RefError::malformed(
                                &at,
                                format!("{}/{n} is {:?}, not a whole row per patch", i.name, r.ne),
                            ));
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
            Kind::Hidden | Kind::Prose | Kind::Bf16Rows | Kind::ChatIds | Kind::Decode => {
                let pad = one(&self.man, "image_pad_id")?;
                if pad != [self.profile.image_pad_id.to_string()] {
                    return Err(RefError::malformed(
                        &at,
                        format!(
                            "image_pad_id {pad:?}, the vocabulary's <|image_pad|> is {}",
                            self.profile.image_pad_id
                        ),
                    ));
                }
                let n = self.n_ids()? as u64;
                if self.kind.is_prompt() {
                    want("result_norm".to_string(), [n_embd, n, 1, 1])?;
                    want("mrope_pos".to_string(), [3, n, 1, 1])?;
                    self.check_int_row("mrope_pos", 3 * n)?;
                }
                self.check_spans(n as usize)?;
                if (self.kind == Kind::Prose) != self.images.is_empty() && self.kind != Kind::Decode
                {
                    return Err(RefError::malformed(
                        &at,
                        format!(
                            "a {} set with {} image lines",
                            self.kind.as_str(),
                            self.images.len()
                        ),
                    ));
                }
                if self.kind == Kind::ChatIds {
                    self.check_chat_ids(n)?;
                }
                if self.kind == Kind::Decode {
                    self.check_decode(n)?;
                }
            }
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

    /// The spans lie inside the `n` ids in order, hold their image's tokens, and (an image set) carry the decoder
    /// positions the images' `n_pos` give: a span starts at its index less what the images before it saved.
    fn check_spans(&self, n: usize) -> Result<(), RefError> {
        let at = At(&self.man.dir);
        let mut end = 0usize;
        let mut saved = 0i64;
        for (k, s) in self.spans.iter().enumerate() {
            if s.index != k || s.at < end || s.at + s.len > n || s.len == 0 {
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
        }
        if self.kind != Kind::Prose
            && self.kind != Kind::Decode
            && self.images.len() != self.spans.len()
        {
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

    /// A chat-ids set's `ids` row is the `n` ids of the header, the image-pad id fills each span and no id outside one.
    fn check_chat_ids(&self, n: u64) -> Result<(), RefError> {
        let at = At(&self.man.dir);
        let r = self.man.tensor("ids", 0)?;
        if r.ne != [n, 1, 1, 1] {
            return Err(RefError::malformed(
                &at,
                format!("ids is {:?}, want [{n}, 1, 1, 1]", r.ne),
            ));
        }
        self.check_int_row("ids", n)?;
        let ids = self.u32s("ids")?;
        let pad = self.profile.image_pad_id;
        let mut in_span = vec![false; ids.len()];
        for s in &self.spans {
            in_span[s.at..s.at + s.len].fill(true);
        }
        for (i, (&id, &inside)) in ids.iter().zip(&in_span).enumerate() {
            if (id == pad) != inside {
                return Err(RefError::malformed(
                    &at,
                    format!(
                        "ids[{i}] is {id}: the image-pad id {pad} fills the spans and nothing else"
                    ),
                ));
            }
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

    /// A decode set holds one id, one row and one position a step; the first token sits where the prompt's ids and
    /// images leave the decoder (`ids - span lengths + image positions`), and each step advances it by one.
    fn check_decode(&self, n: u64) -> Result<(), RefError> {
        let at = At(&self.man.dir);
        let line = self.decode()?;
        let steps = line.steps as u64;
        want_shape(&self.man, "ids", [steps, 1, 1, 1], &at)?;
        want_shape(&self.man, "n_past", [steps, 1, 1, 1], &at)?;
        want_shape(
            &self.man,
            "result_norm",
            [self.profile.n_embd as u64, steps, 1, 1],
            &at,
        )?;
        self.check_int_row("ids", steps)?;
        self.check_int_row("n_past", steps)?;
        let advance: i64 = self
            .spans
            .iter()
            .map(|s| s.n_pos as i64 - s.len as i64)
            .sum();
        let start = n as i64 + advance;
        if line.n_past_start != start {
            return Err(RefError::malformed(
                &at,
                format!(
                    "# decode: the first token sits at {}, the prompt's {n} ids and {} spans give {start}",
                    line.n_past_start,
                    self.spans.len()
                ),
            ));
        }
        for (k, p) in self.decode_n_past()?.into_iter().enumerate() {
            let want = line.n_past_start + k as i64 + 1;
            if i64::from(p) != want {
                return Err(RefError::malformed(
                    &at,
                    format!("n_past[{k}] is {p}, step {k} ends at position {want}"),
                ));
            }
        }
        self.decode_ids()?;
        Ok(())
    }
}

/// Row `name` of the set has shape `ne`.
fn want_shape(man: &RefManifest, name: &str, ne: [u64; 4], at: &At<'_>) -> Result<(), RefError> {
    let r = man.tensor(name, 0)?;
    if r.ne != ne {
        return Err(RefError::malformed(
            at,
            format!("{name} is {:?}, want {ne:?}", r.ne),
        ));
    }
    Ok(())
}

/// Undo the dump's line escaping: `\\`, `\n`, `\t` and `\r`; another escape, or one cut short, is an error.
fn unescape(raw: &str) -> Result<String, String> {
    let mut out = String::with_capacity(raw.len());
    let mut it = raw.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        out.push(match it.next() {
            Some('\\') => '\\',
            Some('n') => '\n',
            Some('t') => '\t',
            Some('r') => '\r',
            Some(o) => return Err(format!("the escape \\{o} is none of \\\\ \\n \\t \\r")),
            None => return Err("a backslash ends the line".to_string()),
        });
    }
    Ok(out)
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

/// `# tap_effect\t<image>\tembd values <n>\tdiffering <n>\tmax_abs_diff <x>\ttaps full|final|inp_raw only`.
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
            let scope = match value(4, "taps")? {
                "full" => TapScope::Full,
                "final" => TapScope::Final,
                "inp_raw only" => TapScope::EmbdOnly,
                v => {
                    return Err(bad(&format!(
                        "taps {v:?}, want full, final or inp_raw only"
                    )));
                }
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
                scope,
            })
        })
        .collect()
}

/// `# chat\ttemplate_sha256 <hex>\tuse_jinja 1\tenable_thinking 0|1\treasoning_format deepseek\t
/// add_generation_prompt 0|1\tcontinue_final_message none|auto`.
fn read_chat(man: &RefManifest) -> Result<ChatLine, RefError> {
    let f = one(man, "chat")?;
    let bad = |what: &str| RefError::malformed(At(&man.dir), format!("# chat {f:?}: {what}"));
    let value = |i: usize, key: &str| -> Result<&str, RefError> {
        f.get(i)
            .and_then(|s| s.strip_prefix(key))
            .and_then(|s| s.strip_prefix(' '))
            .ok_or_else(|| bad(&format!("field {i} is not `{key} <value>`")))
    };
    let flag = |i: usize, key: &str| -> Result<bool, RefError> {
        match value(i, key)? {
            "1" => Ok(true),
            "0" => Ok(false),
            v => Err(bad(&format!("{key} {v:?}, want 0 or 1"))),
        }
    };
    if value(1, "use_jinja")? != "1" {
        return Err(bad("the oracle renders with the jinja template"));
    }
    Ok(ChatLine {
        template_sha256: value(0, "template_sha256")?.to_string(),
        enable_thinking: flag(2, "enable_thinking")?,
        add_generation_prompt: flag(4, "add_generation_prompt")?,
        continue_final_message: match value(5, "continue_final_message")? {
            "none" => false,
            "auto" => true,
            v => {
                return Err(bad(&format!(
                    "continue_final_message {v:?}, want none or auto"
                )));
            }
        },
    })
}

/// `# decode\tsteps\t<n>\tn_past_start\t<n>\tsampler\t…`.
fn read_decode(man: &RefManifest) -> Result<DecodeLine, RefError> {
    let f = one(man, "decode")?;
    let bad = |what: &str| RefError::malformed(At(&man.dir), format!("# decode {f:?}: {what}"));
    let [steps, n, start, p, _sampler, ..] = f[..] else {
        return Err(bad("want steps\t<n>\tn_past_start\t<n>\tsampler\t…"));
    };
    if steps != "steps" || start != "n_past_start" {
        return Err(bad("want steps\t<n>\tn_past_start\t<n>\tsampler\t…"));
    }
    Ok(DecodeLine {
        steps: n.parse().map_err(|e| bad(&format!("steps: {e}")))?,
        n_past_start: p.parse().map_err(|e| bad(&format!("n_past_start: {e}")))?,
    })
}

#[cfg(test)]
pub(crate) mod testkit;

/// The families of one Qwen seat's sets, `$prefix` the sets' common name (`ref_<arch>_vis`): the tower's [`Kind::Taps`]
/// family states the projector as its model file (`# arch clip`), every other kind states the text model; the card's
/// sets of each kind and the CPU twins the recipe writes sit in one family, the chat ids (no card, no twin) too. The
/// invoking module names the seat's [`Profile`], architecture, mainline build and the two files as functions.
macro_rules! vis_families {
    (
        prefix: $prefix:literal, name: $name:literal, model: $arg:literal,
        profile: $profile:expr, arch: $arch:expr, build: $build:expr,
        mmproj: $mmproj:expr, text: $text:expr $(,)?
    ) => {
        /// Set B of the seat's tower: the post-LN output and the final embeddings of each test image.
        pub static TAPS: $crate::family::Family = $crate::family::Family {
            name: concat!($name, "-taps"),
            sets: &[concat!($prefix, "_taps"), concat!($prefix, "_taps.cpu")],
            resolve: None,
            recipe: concat!(
                "just dump-ref-qvis ",
                $arg,
                " [--cpu-twin] ",
                $prefix,
                "_taps"
            ),
            identity: $crate::family::Identity::Clefvis($crate::clefvis::Kind::Taps, &$profile),
            arch: Some($crate::clefvis::TOWER_ARCH),
            build: Some($crate::family::Build::Is($build)),
            runs: Some($mmproj),
            draft_runs: None,
            consumers: &[],
        };

        /// Set C: `result_norm` of every position of a chat prompt with mainline's tower rows, and the positions fed.
        pub static HIDDEN: $crate::family::Family = $crate::family::Family {
            name: concat!($name, "-hidden"),
            sets: &[
                concat!($prefix, "_hidden_c1"),
                concat!($prefix, "_hidden_c2"),
                concat!($prefix, "_hidden_c3"),
                concat!($prefix, "_hidden_c1.cpu"),
                concat!($prefix, "_hidden_c2.cpu"),
                concat!($prefix, "_hidden_c3.cpu"),
            ],
            resolve: None,
            recipe: concat!(
                "just dump-ref-qvis ",
                $arg,
                " [--cpu-twin] ",
                $prefix,
                "_hidden_c<N>"
            ),
            identity: $crate::family::Identity::Clefvis($crate::clefvis::Kind::Hidden, &$profile),
            arch: Some($arch),
            build: Some($crate::family::Build::Is($build)),
            runs: Some($text),
            draft_runs: None,
            consumers: &[],
        };

        /// Set C′: the same prompts with each image's rows replaced by prose-id text rows.
        pub static PROSE: $crate::family::Family = $crate::family::Family {
            name: concat!($name, "-prose"),
            sets: &[
                concat!($prefix, "_prose_c1"),
                concat!($prefix, "_prose_c2"),
                concat!($prefix, "_prose_c3"),
                concat!($prefix, "_prose_c1.cpu"),
                concat!($prefix, "_prose_c2.cpu"),
                concat!($prefix, "_prose_c3.cpu"),
            ],
            resolve: None,
            recipe: concat!(
                "just dump-ref-qvis ",
                $arg,
                " [--cpu-twin] ",
                $prefix,
                "_prose_c<N>"
            ),
            identity: $crate::family::Identity::Clefvis($crate::clefvis::Kind::Prose, &$profile),
            arch: Some($arch),
            build: Some($crate::family::Build::Is($build)),
            runs: Some($text),
            draft_runs: None,
            consumers: &[],
        };

        /// Set C″: the same prompts with mainline's tower rows rounded to bf16; the card only.
        pub static BF16ROWS: $crate::family::Family = $crate::family::Family {
            name: concat!($name, "-bf16rows"),
            sets: &[
                concat!($prefix, "_bf16rows_c1"),
                concat!($prefix, "_bf16rows_c2"),
                concat!($prefix, "_bf16rows_c3"),
            ],
            resolve: None,
            recipe: concat!("just dump-ref-qvis ", $arg, " ", $prefix, "_bf16rows_c<N>"),
            identity: $crate::family::Identity::Clefvis($crate::clefvis::Kind::Bf16Rows, &$profile),
            arch: Some($arch),
            build: Some($crate::family::Build::Is($build)),
            runs: Some($text),
            draft_runs: None,
            consumers: &[],
        };

        /// Set E: the ids llama-server's chat path gives each request, with its rendered prompt.
        pub static CHATIDS: $crate::family::Family = $crate::family::Family {
            name: concat!($name, "-chatids"),
            sets: &[
                concat!($prefix, "_chatids_c1"),
                concat!($prefix, "_chatids_c2"),
                concat!($prefix, "_chatids_c3"),
                concat!($prefix, "_chatids_e1"),
                concat!($prefix, "_chatids_e2"),
                concat!($prefix, "_chatids_e3"),
            ],
            resolve: None,
            recipe: concat!("just dump-ref-qvis ", $arg, " --ids"),
            identity: $crate::family::Identity::Clefvis($crate::clefvis::Kind::ChatIds, &$profile),
            arch: Some($arch),
            build: Some($crate::family::Build::Is($build)),
            runs: Some($text),
            draft_runs: None,
            consumers: &[],
        };

        /// Set F: 32 greedy decode steps after prompts C1 and C2.
        pub static DECODE: $crate::family::Family = $crate::family::Family {
            name: concat!($name, "-decode"),
            sets: &[
                concat!($prefix, "_decode_c1"),
                concat!($prefix, "_decode_c2"),
                concat!($prefix, "_decode_c1.cpu"),
                concat!($prefix, "_decode_c2.cpu"),
            ],
            resolve: None,
            recipe: concat!(
                "just dump-ref-qvis ",
                $arg,
                " [--cpu-twin] ",
                $prefix,
                "_decode_c<N>"
            ),
            identity: $crate::family::Identity::Clefvis($crate::clefvis::Kind::Decode, &$profile),
            arch: Some($arch),
            build: Some($crate::family::Build::Is($build)),
            runs: Some($text),
            draft_runs: None,
            consumers: &[],
        };
    };
}
pub(crate) use vis_families;
