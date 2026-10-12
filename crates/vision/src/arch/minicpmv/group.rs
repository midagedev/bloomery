//! One image as MiniCPM-V 4.7 sees it: the group of tiles the plan cuts (the overview, then the
//! slices line by line), each normalized and cut into 14-pixel patches, and the tokens the group
//! takes in the prompt.
//!
//! The prompt reference is mtmd's tokenizer (`tools/mtmd/mtmd.cpp:725-740`, `:1476-1483`,
//! `add_canvas_chunk` at `:1733`): for each image the text `<image_id>N</image_id>` (N the 0-based
//! image index) and `<image>`, then one chunk of embedding rows. The chunk holds, per tile, the
//! tile's tokens in raster order followed by its separators, which are rows of `v.tok_embd_sep`:
//! the rows of the text model's embedding table for `</image>`, `<slice>`, `</slice>` and `\n`, so
//! a separator row is the embedding of that token's id. The overview is followed by `</image>`
//! (and `<slice>` when slices follow); a slice by `</slice><slice>`, the last of a line by
//! `</slice>\n<slice>`, the last of all by `</slice>`. A tile of `w × h` pixels takes
//! `w/56 × h/56` tokens.

use super::Hparams;
use super::plan::Plan;
use crate::mrope::Canvas;
use crate::preprocess::{GridPlan, PatchLayout, Patches, patchify};
use crate::resample::resize_bicubic;
use crate::{Rgb8, VisionError};

/// A separator row of `v.tok_embd_sep`, by what it stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sep {
    /// `</image>`, row 0.
    ImageEnd,
    /// `<slice>`, row 1.
    SliceStart,
    /// `</slice>`, row 2.
    SliceEnd,
    /// `\n`, row 3.
    RowEnd,
}

impl Sep {
    /// The row of `v.tok_embd_sep` (the converter's order).
    #[must_use]
    pub const fn row(self) -> usize {
        match self {
            Sep::ImageEnd => 0,
            Sep::SliceStart => 1,
            Sep::SliceEnd => 2,
            Sep::RowEnd => 3,
        }
    }

    fn of_row(row: u8) -> Sep {
        match row {
            0 => Sep::ImageEnd,
            1 => Sep::SliceStart,
            2 => Sep::SliceEnd,
            3 => Sep::RowEnd,
            other => unreachable!("v.tok_embd_sep has 4 rows, not row {other}"),
        }
    }
}

/// A mark that the text model's tokenizer does not turn into exactly one token.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("tokenizer mark {text:?}: {detail}")]
pub struct MarkError {
    pub text: String,
    pub detail: &'static str,
}

/// The ids of the marks of a group, from the text model's tokenizer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Marks {
    pub image_id_open: u32,
    pub image_id_close: u32,
    pub image_open: u32,
    pub image_close: u32,
    pub slice_open: u32,
    pub slice_close: u32,
    pub row_end: u32,
    /// The ids of the digits `0` to `9`: an image index is written in decimal text.
    pub digits: [u32; 10],
}

impl Marks {
    /// The ids by string as mtmd's `lookup_token` takes them: `one` gives the id a string
    /// tokenizes to when it is exactly one token and `None` otherwise; a mark with none is
    /// refused by name.
    pub fn lookup(one: impl Fn(&str) -> Option<u32>) -> Result<Marks, MarkError> {
        let get = |text: &str| {
            one(text).ok_or(MarkError {
                text: text.to_string(),
                detail: "does not tokenize to exactly one token",
            })
        };
        let mut digits = [0u32; 10];
        for (d, slot) in digits.iter_mut().enumerate() {
            *slot = get(&d.to_string())?;
        }
        Ok(Marks {
            image_id_open: get("<image_id>")?,
            image_id_close: get("</image_id>")?,
            image_open: get("<image>")?,
            image_close: get("</image>")?,
            slice_open: get("<slice>")?,
            slice_close: get("</slice>")?,
            row_end: get("\n")?,
            digits,
        })
    }

    /// The id of a separator row.
    #[must_use]
    pub fn sep(&self, sep: Sep) -> u32 {
        match sep {
            Sep::ImageEnd => self.image_close,
            Sep::SliceStart => self.slice_open,
            Sep::SliceEnd => self.slice_close,
            Sep::RowEnd => self.row_end,
        }
    }

    /// The text ids before the group of image `index` (0-based among the prompt's images):
    /// `<image_id>`, the digits of the index, `</image_id>`, `<image>`.
    #[must_use]
    pub fn head(&self, index: usize) -> Vec<u32> {
        let mut ids = vec![self.image_id_open];
        ids.extend(
            index
                .to_string()
                .bytes()
                .map(|b| self.digits[usize::from(b - b'0')]),
        );
        ids.push(self.image_id_close);
        ids.push(self.image_open);
        ids
    }

    /// The group's rows as ids: every tile token is `pad` (the id the tower's rows replace) and
    /// every separator its mark's id. The rows follow [`Marks::head`] in the prompt; the engine
    /// takes the separators' embeddings from the text embedding table (equal to the rows of
    /// `v.tok_embd_sep`) and the tiles' from the tower.
    #[must_use]
    pub fn rows(&self, canvas: &Canvas, pad: u32) -> Vec<u32> {
        let mut ids = Vec::with_capacity(canvas.rows());
        for k in 0..canvas.tiles() {
            let tile = canvas.tile(k);
            ids.extend(std::iter::repeat_n(pad, tile.nx * tile.ny));
            ids.extend(tile.seps.iter().map(|&r| self.sep(Sep::of_row(r))));
        }
        ids
    }
}

/// One image preprocessed: the plan, the group's layout and the tower's input per tile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub plan: Plan,
    /// The tiles' patches, `[overview, slice 0, slice 1, …]` (line by line); a tile's token grid
    /// is `tiles[k].plan.span_grid()`.
    pub tiles: Vec<Patches>,
    canvas: Canvas,
}

impl Group {
    /// The tiles and separators as positions and rows.
    #[must_use]
    pub fn canvas(&self) -> &Canvas {
        &self.canvas
    }

    /// The rows the group takes in the prompt (tokens and separators): the span length.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.canvas.rows()
    }
}

/// The layout of a plan: token grids at `token_px` pixels a side. The sizes are multiples of 56,
/// which the merge of 4 divides.
pub fn canvas_of(hp: &Hparams, plan: &Plan) -> Result<Canvas, VisionError> {
    let tok = |(w, h): (usize, usize)| (w / hp.token_px(), h / hp.token_px());
    let slice = plan.slices.first().map_or((0, 0), |r| tok((r.w, r.h)));
    Canvas::new(tok(plan.overview), plan.grid, slice).map_err(|e| VisionError::Size {
        width: plan.overview.0,
        height: plan.overview.1,
        detail: e.to_string(),
    })
}

/// Crop `w × h` pixels at `(x, y)`.
fn crop(src: &Rgb8, x: usize, y: usize, w: usize, h: usize) -> Rgb8 {
    let mut data = Vec::with_capacity(w * h * 3);
    for row in y..y + h {
        data.extend_from_slice(&src.data[(row * src.width + x) * 3..(row * src.width + x + w) * 3]);
    }
    Rgb8 {
        width: w,
        height: h,
        data,
    }
}

/// An image to its group: planned, stretched (no pad) to the overview size and the refined size
/// with the bicubic resample, the slices cropped from the refined image, each tile normalized and
/// cut in 14-pixel patches.
pub fn preprocess_group(image: &Rgb8, hp: &Hparams) -> Result<Group, VisionError> {
    image.check()?;
    let plan = hp.plan(image.width, image.height)?;
    let canvas = canvas_of(hp, &plan)?;
    let mut imgs = vec![resize_bicubic(image, plan.overview.0, plan.overview.1)?];
    if !plan.slices.is_empty() {
        let refined = resize_bicubic(image, plan.refined.0, plan.refined.1)?;
        imgs.extend(
            plan.slices
                .iter()
                .map(|r| crop(&refined, r.x, r.y, r.w, r.h)),
        );
    }
    let layout = PatchLayout::row_major(hp.patch);
    let tiles = imgs
        .iter()
        .map(|img| {
            let grid = GridPlan {
                n_llm_h: img.height / hp.token_px(),
                n_llm_w: img.width / hp.token_px(),
                best_h: img.height,
                best_w: img.width,
            };
            patchify(img, grid, layout)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Group {
        plan,
        tiles,
        canvas,
    })
}

#[cfg(test)]
mod tests {
    use super::{Marks, preprocess_group};
    use crate::Rgb8;
    use crate::arch::minicpmv::hparams::tests::hp;

    const PAD: u32 = 900;

    fn marks() -> Marks {
        Marks {
            image_id_open: 101,
            image_id_close: 102,
            image_open: 103,
            image_close: 104,
            slice_open: 105,
            slice_close: 106,
            row_end: 107,
            digits: std::array::from_fn(|d| 200 + d as u32),
        }
    }

    fn run(id: u32, n: usize) -> Vec<u32> {
        vec![id; n]
    }

    /// The 2×2-sliced 1000×700 image (overview 70 tokens, four slices of 70) is, after
    /// `<image_id>0</image_id><image>`: the overview and `</image><slice>`, slice (0,0) and
    /// `</slice><slice>`, slice (0,1) and `</slice>\n<slice>`, slice (1,0) and `</slice><slice>`,
    /// slice (1,1) and `</slice>`: 360 rows, the separators of `clip_suffix_rows`.
    #[test]
    fn the_2x2_group_is_laid_out_as_mtmd_lays_it() {
        let hp = hp();
        let g = preprocess_group(&Rgb8::filled(1000, 700, [10, 20, 30]), &hp).expect("group");
        let (m, c) = (marks(), g.canvas());
        assert_eq!((g.rows(), c.advance(), c.tiles()), (360, 15, 5));
        assert_eq!(m.head(0), [101, 200, 102, 103]);
        assert_eq!(m.head(12), [101, 201, 202, 102, 103]);
        let want: Vec<u32> = [
            run(PAD, 70),
            vec![104, 105],
            run(PAD, 70),
            vec![106, 105],
            run(PAD, 70),
            vec![106, 107, 105],
            run(PAD, 70),
            vec![106, 105],
            run(PAD, 70),
            vec![106],
        ]
        .concat();
        assert_eq!(m.rows(c, PAD), want);
        // The tiles: 560×392 pixels each, 40×28 patches, 10×7 tokens.
        assert_eq!(g.tiles.len(), 5);
        for t in &g.tiles {
            assert_eq!(
                (t.n_vit_w, t.n_vit_h, t.plan.span_grid()),
                (40, 28, (10, 7))
            );
            assert_eq!(t.bf16.len(), 40 * 28 * 588);
        }
    }

    /// An image with no slice is its overview and `</image>`; a small one is stretched up.
    #[test]
    fn an_unsliced_image_is_an_overview_and_one_separator() {
        let hp = hp();
        let g = preprocess_group(&Rgb8::filled(200, 200, [255, 0, 0]), &hp).expect("group");
        let m = marks();
        assert_eq!((g.rows(), g.canvas().advance(), g.tiles.len()), (65, 2, 1));
        assert_eq!(m.rows(g.canvas(), PAD), [run(PAD, 64), vec![104]].concat());
        // A flat red image stays flat red through the resize: the red plane of the first patch
        // is +1.0 (bf16 0x3F80) and the other two -1.0 (0xBF80).
        let patch = &g.tiles[0].bf16[..588];
        assert!(patch[..196].iter().all(|&v| v == 0x3F80));
        assert!(patch[196..].iter().all(|&v| v == 0xBF80));
    }

    /// The marks come from the tokenizer by string; one that is not a single token is named.
    #[test]
    fn a_mark_that_is_not_one_token_is_named() {
        let table = |s: &str| match s {
            "<slice>" => None,
            "\n" => Some(198),
            _ => Some(7),
        };
        let e = Marks::lookup(table).unwrap_err();
        assert_eq!(e.text, "<slice>");
        let ok = Marks::lookup(|s| (s != "<slice>").then_some(7).or(Some(8))).expect("lookup");
        assert_eq!((ok.slice_open, ok.digits[3]), (8, 7));
    }
}
