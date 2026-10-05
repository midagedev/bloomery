//! V4.1's image input as the server's media layer sees it: the placeholder `<｜deepseek_image｜>`
//! and its token, `"\n\n"` between flattened content parts, and the reference `load_image` (plan,
//! pad or resize, normalize, patch) as the prepare.

use super::IMAGE_TOKEN_ID;
use crate::grid::GridParams;
use crate::media::{MediaModel, Prepared};
use crate::{Rgb8, VisionError, preprocess};

/// The text of [`IMAGE_TOKEN_ID`], the placeholder V4.1's chat carries for one image.
pub const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

/// The separator between V4.1's flattened content parts.
pub const PART_SEPARATOR: &str = "\n\n";

/// V4.1's [`MediaModel`]: the resize parameters of its encoder file, the rest constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Media {
    grid: GridParams,
}

impl Media {
    /// The media description of an encoder file whose resize parameters are `grid`
    /// ([`super::Hparams::grid`], read after `Hparams::read` refused every value the reference
    /// does not run).
    #[must_use]
    pub fn new(grid: GridParams) -> Media {
        Media { grid }
    }
}

impl MediaModel for Media {
    fn image_placeholder(&self) -> &str {
        IMAGE_PLACEHOLDER
    }

    fn image_token(&self) -> u32 {
        IMAGE_TOKEN_ID
    }

    fn part_separator(&self) -> &str {
        PART_SEPARATOR
    }

    fn prepare(&self, image: &Rgb8) -> Result<Prepared, VisionError> {
        let patches = preprocess(image, &self.grid)?;
        Ok(Prepared {
            span_len: patches.plan.n_tokens(),
            patches,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{IMAGE_PLACEHOLDER, Media};
    use crate::grid::GridParams;
    use crate::media::MediaModel;
    use crate::{Rgb8, image_span};

    /// The reference config's resize parameters.
    const V41: GridParams = GridParams {
        patch: 14,
        downsample: 3,
        max_tokens: 1024,
        min_pixels: 295_936,
    };

    #[test]
    fn data_is_v41s() {
        let m = Media::new(V41);
        assert_eq!(m.image_placeholder(), IMAGE_PLACEHOLDER);
        assert_eq!(m.image_placeholder(), "<\u{FF5C}deepseek_image\u{FF5C}>");
        assert_eq!(m.image_token(), 129_264);
        assert_eq!(m.part_separator(), "\n\n");
    }

    /// A 448×448 image is scaled up to the 546×546 grid: 13×13 aligner tokens, 184 positions with
    /// the delimiters (the research table's first row), and the span the expansion makes has the
    /// length `image_span` gives the plan.
    #[test]
    fn prepare_gives_the_plans_span() {
        let m = Media::new(V41);
        let p = m
            .prepare(&Rgb8::filled(448, 448, [10, 200, 30]))
            .expect("prepare");
        assert_eq!(p.span_len, 184);
        assert_eq!((p.patches.plan.best_w, p.patches.plan.best_h), (546, 546));
        assert_eq!((p.patches.n_vit_h, p.patches.n_vit_w), (39, 39));
        assert_eq!(p.patches.bf16.len(), 39 * 39 * 3 * 14 * 14);
        assert_eq!(
            image_span(&p.patches.plan, m.image_token()).ids.len(),
            p.span_len
        );
    }
}
