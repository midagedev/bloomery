//! Per projector type, what the encoder file holds: its tensor names and its hyperparameters.
//!
//! An encoder file is an mmproj GGUF (architecture `clip`); `clip.projector_type` names the tower
//! and projector it carries, and a module here is named by that string. Each module reads the
//! header once and refuses, by key or by tensor name, anything its encoder does not run.
//!
//! What every projector type does the same way has one owner here, and a module holds its tables
//! as data: [`header`] (the key checks and their refusals), [`naming`] (the tensor-name scheme)
//! and [`table`] (the tensor table and the check of a file against it). The tower's bytes on its
//! card take one shape ([`card::CardBytes`]), and the narrowing of a tensor to bf16 on upload one
//! rule ([`narrow`]). [`open`] picks the module by projector type.

pub mod card;
pub mod deepseek41v;
pub mod glm5v;
pub(crate) mod header;
pub(crate) mod naming;
pub mod narrow;
pub mod qwen3vl;
pub mod table;

use std::sync::Arc;

use gguf::Gguf;

use self::card::CardBytes;
use self::header::{KEY_PROJECTOR, Meta, check_architecture, need, refusal};
use crate::{MediaModel, VisionError};

/// The bounds on how many tokens an image takes: [`open`]'s limits, which every projector that
/// plans from them reads in this one type.
pub use self::qwen3vl::TokenLimits;

/// What an encoder file opens to: the tower's bytes on its card, and the model's image input.
pub type Opened = (CardBytes, Arc<dyn MediaModel + Send + Sync>);

/// Open an encoder file by its projector type: read the header, refuse by name any value or
/// projector this crate does not run, and state the tower's bytes on its card and the model's
/// image input. `limits` bound an image's tokens for the projectors whose plan they steer
/// (`qwen3vl_merger`, `glm5v`); V4.1's plan is the file's own. A caller with no limit given takes
/// them from the projector: [`TokenLimits::DEFAULT`] for `qwen3vl_merger`, the file's own budget
/// ([`glm5v::Hparams::default_limits`]) for `glm5v`.
///
/// The text model's width is the seat's to check (`qwen3vl::Hparams::check_text_width`), and the
/// tensors are checked where the tower loads them.
pub fn open(file: &Gguf, limits: TokenLimits) -> Result<Opened, VisionError> {
    check_architecture(file)?;
    match projector(file)? {
        Projector::Deepseek41v => {
            let hp = deepseek41v::Hparams::read(file)?;
            Ok((
                deepseek41v::card::of(&hp),
                Arc::new(deepseek41v::media::Media::new(hp.grid())),
            ))
        }
        Projector::Glm5v => {
            let hp = glm5v::Hparams::read(file)?;
            let rule = hp.size_rule_at(limits)?;
            Ok((
                glm5v::card::of(&hp, limits.max()),
                Arc::new(glm5v::Media::new(rule)),
            ))
        }
        Projector::Qwen3vl => {
            let hp = qwen3vl::Hparams::read(file)?;
            let media = qwen3vl::Media::new(hp.size_rule(limits)?);
            Ok((qwen3vl::card::of(&hp, limits), Arc::new(media)))
        }
    }
}

/// The projector types this crate runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Projector {
    Deepseek41v,
    Glm5v,
    Qwen3vl,
}

/// The file's `clip.projector_type`, refused by name when it is a tower this crate does not run.
fn projector(m: &impl Meta) -> Result<Projector, VisionError> {
    let v = need(m, KEY_PROJECTOR)?;
    match v.as_str() {
        Some(deepseek41v::PROJECTOR_TYPE) => Ok(Projector::Deepseek41v),
        Some(glm5v::PROJECTOR_TYPE) => Ok(Projector::Glm5v),
        Some(qwen3vl::PROJECTOR_TYPE) => Ok(Projector::Qwen3vl),
        Some(other) => Err(refusal(
            KEY_PROJECTOR,
            format!(
                "is \"{other}\"; this crate reads \"{}\", \"{}\" and \"{}\"",
                deepseek41v::PROJECTOR_TYPE,
                glm5v::PROJECTOR_TYPE,
                qwen3vl::PROJECTOR_TYPE
            ),
        )),
        None => Err(refusal(KEY_PROJECTOR, "is not a string")),
    }
}

#[cfg(test)]
mod tests {
    use gguf::{Gguf, Value};

    use super::{Projector, open, projector};
    use crate::arch::glm5v::hparams::tests::glm;
    use crate::arch::header::testing::Table;
    use crate::arch::qwen3vl::TokenLimits;
    use crate::arch::qwen3vl::hparams::tests::clef;
    use crate::media::PartJoin;

    fn table(projector: &str) -> Table {
        clef().with("clip.projector_type", Value::String(projector.into()))
    }

    /// The projector type picks the module; a stranger is refused by its name, which lists the
    /// three this crate reads.
    #[test]
    fn the_projector_type_picks_the_tower() {
        assert_eq!(
            projector(&table("qwen3vl_merger")).unwrap(),
            Projector::Qwen3vl
        );
        assert_eq!(
            projector(&table("deepseek41v")).unwrap(),
            Projector::Deepseek41v
        );
        assert_eq!(projector(&table("glm5v")).unwrap(), Projector::Glm5v);
        assert_eq!(
            projector(&table("qwen2vl_merger")).unwrap_err().to_string(),
            "metadata clip.projector_type: is \"qwen2vl_merger\"; this crate reads \
             \"deepseek41v\", \"glm5v\" and \"qwen3vl_merger\""
        );
        assert_eq!(
            projector(&clef().with("clip.projector_type", Value::U32(7)))
                .unwrap_err()
                .to_string(),
            "metadata clip.projector_type: is not a string"
        );
        assert_eq!(
            projector(&clef().without("clip.projector_type"))
                .unwrap_err()
                .to_string(),
            "metadata clip.projector_type: is absent"
        );
    }

    fn open_at(
        t: &Table,
        arch: &str,
        limits: TokenLimits,
    ) -> Result<super::Opened, crate::VisionError> {
        let file = t.write(arch);
        let gguf = Gguf::open(&file.0).expect("the header-only file opens");
        open(&gguf, limits)
    }

    fn open_table(t: &Table, arch: &str) -> Result<super::Opened, crate::VisionError> {
        open_at(t, arch, TokenLimits::DEFAULT)
    }

    /// The three Qwen3-VL files open to their own card figures at the default limits (8..4096
    /// tokens), the model's data and llama-server's join rule.
    #[test]
    fn the_three_qwen_files_open_to_their_figures() {
        for (out, weights, scratch) in [
            (4096, 924_123_136, 570_949_632),
            (2048, 905_240_576, 554_172_416),
            (2560, 909_961_216, 558_366_720),
        ] {
            let t = table("qwen3vl_merger").with("clip.vision.projection_dim", Value::U32(out));
            let (card, media) = open_table(&t, "clip").expect("open");
            assert_eq!((card.weights, card.scratch), (weights, scratch), "{out}");
            assert_eq!(media.image_token(), 248_056);
            assert_eq!(media.part_join(), PartJoin::LlamaServer);
        }
    }

    /// GLM's file opens to its own card figure at the limits it is opened with — the tower's
    /// weights, and activations for the most tokens an image may take — and to its own data and
    /// llama-server's join rule; the limits are the plan's, so an image of 1000×600 pixels takes
    /// 792 tokens at the file's own 8000 and no more than 100 at 100.
    #[test]
    fn a_glm_file_opens_to_its_figures() {
        let limits = |min, max| TokenLimits::new(min, max).expect("limits");
        for (max, scratch) in [(8000, 2_016_768_000), (4096, 1_032_585_216)] {
            let (card, media) = open_at(&glm(), "clip", limits(16, max)).expect("open");
            assert_eq!(
                (card.weights, card.scratch),
                (1_128_026_112, scratch),
                "{max}"
            );
            assert_eq!(media.image_token(), 154_854);
            assert_eq!(media.part_join(), PartJoin::LlamaServer);
        }
        let image = crate::Rgb8::filled(1000, 600, [90, 120, 30]);
        let span = |max| {
            let (_, media) = open_at(&glm(), "clip", limits(16, max)).expect("open");
            media.prepare(&image).expect("prepare").span_len
        };
        assert_eq!(span(8000), 792);
        assert!((16..=100).contains(&span(100)), "{}", span(100));
    }

    /// V4.1's file opens to its own figure and its own join rule; a stranger's and a file of another
    /// architecture are refused by name, and a limit that does not fit GLM's pixel count with it.
    #[test]
    fn other_files_open_or_are_refused_by_name() {
        let (card, media) = open_table(&crate::arch::deepseek41v::hparams::tests::v41(), "clip")
            .expect("V4.1 opens");
        assert_eq!(card.total(), 970_924_032 + 339_878_648);
        assert_eq!(media.part_join(), PartJoin::Uniform("\n\n"));
        let refused =
            |t: &Table, arch: &str| open_table(t, arch).err().expect("refused").to_string();
        assert_eq!(
            refused(&table("qwen2vl_merger"), "clip"),
            "metadata clip.projector_type: is \"qwen2vl_merger\"; this crate reads \
             \"deepseek41v\", \"glm5v\" and \"qwen3vl_merger\""
        );
        assert!(refused(&table("qwen3vl_merger"), "llama").starts_with("metadata architecture:"));
        let huge = TokenLimits::new(16, 3_000_000).expect("limits");
        assert_eq!(
            open_at(&glm(), "clip", huge)
                .err()
                .expect("refused")
                .to_string(),
            "image size limits: 16..=3000000 tokens of patch 14 and merge 2 do not fit a 32-bit \
             pixel count"
        );
    }
}
