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
//! [`gemma4a`] owns the host preprocessing of decoded audio; it opens no encoder file.

pub mod card;
pub mod deepseek41v;
pub mod gemma4a;
pub mod gemma4v;
pub mod glm5v;
pub(crate) mod header;
pub mod minicpmv;
pub(crate) mod naming;
pub mod narrow;
pub mod qwen3vl;
pub mod radio;
pub mod table;

use std::sync::Arc;

use gguf::Gguf;

use self::card::CardBytes;
use self::header::{KEY_PROJECTOR, Meta, check_architecture, projector_value, refusal};
use crate::{MediaModel, VisionError};

/// The bounds on how many tokens an image takes: [`open`]'s limits, which every projector that
/// plans from them reads in this one type.
pub use crate::size::TokenLimits;

/// What an encoder file opens to: the tower's bytes on its card, and the model's image input.
pub type Opened = (CardBytes, Arc<dyn MediaModel + Send + Sync>);

/// Open an encoder file by its projector type: read the header, refuse by name any value or
/// projector this crate does not run, and state the tower's bytes on its card and the model's
/// image input. `limits` bound an image's tokens for the projectors whose plan they steer
/// (`qwen3vl_merger`, `glm5v`, `nemotron_omni`); V4.1's plan is the file's own. A caller with no
/// limit given takes them from the projector: [`TokenLimits::DEFAULT`] for `qwen3vl_merger`, the
/// file's own budget ([`glm5v::Hparams::default_limits`], [`radio::Hparams::default_limits`]) for
/// `glm5v` and `nemotron_omni`.
///
/// A `minicpmv4_7` file is read (its header refused by name when it is not the one this crate
/// runs) and then refused: its group's ids come from the text model's tokenizer, which the
/// encoder file does not carry, so the seat builds [`minicpmv::Media`] from [`minicpmv::Hparams`]
/// and the tokenizer's marks itself.
///
/// The text model's width is the seat's to check (`qwen3vl::Hparams::check_text_width`), and the
/// tensors are checked where the tower loads them.
pub fn open(file: &Gguf, limits: TokenLimits) -> Result<Opened, VisionError> {
    check_architecture(file)?;
    match projector(file)? {
        Projector::Gemma4v => {
            let hp = gemma4v::Hparams::read(file)?;
            let media = gemma4v::Media::new(hp.size_rule(limits)?);
            Ok((gemma4v::card::of(file, &hp, limits)?, Arc::new(media)))
        }
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
        Projector::NemotronOmni => {
            let hp = radio::Hparams::read(file)?;
            let media = radio::Media::new(hp.size_rule(limits)?, hp.stats);
            Ok((radio::card::of(&hp, limits), Arc::new(media)))
        }
        Projector::Minicpmv47 => {
            minicpmv::Hparams::read(file)?;
            Err(refusal(
                KEY_PROJECTOR,
                format!(
                    "is \"{}\"; its image ids come from the text model's tokenizer, which the \
                     encoder file does not carry: read it with minicpmv::Hparams::read and build \
                     minicpmv::Media::new",
                    minicpmv::PROJECTOR_TYPE
                ),
            ))
        }
    }
}

/// Open the audio input rule of a multimodal projector, if its header carries one.
pub fn open_audio(file: &Gguf) -> Result<Option<Arc<dyn MediaModel + Send + Sync>>, VisionError> {
    match file.value("clip.audio.projector_type") {
        Some(kind) if kind.as_str() == Some("gemma4a") => {
            Ok(Some(Arc::new(gemma4a::Preprocessor::default())))
        }
        Some(_) => Err(VisionError::Audio(
            "the embedding projector's audio tower must be gemma4a".into(),
        )),
        None => Ok(None),
    }
}

/// The projector types this crate runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Projector {
    Gemma4v,
    Deepseek41v,
    Glm5v,
    Qwen3vl,
    Minicpmv47,
    NemotronOmni,
}

/// The file's `clip.projector_type`, refused by name when it is a tower this crate does not run.
fn projector(m: &impl Meta) -> Result<Projector, VisionError> {
    let v = projector_value(m)?;
    match v.as_str() {
        Some(gemma4v::PROJECTOR_TYPE) => Ok(Projector::Gemma4v),
        Some(deepseek41v::PROJECTOR_TYPE) => Ok(Projector::Deepseek41v),
        Some(glm5v::PROJECTOR_TYPE) => Ok(Projector::Glm5v),
        Some(qwen3vl::PROJECTOR_TYPE) => Ok(Projector::Qwen3vl),
        Some(minicpmv::PROJECTOR_TYPE) => Ok(Projector::Minicpmv47),
        Some(radio::PROJECTOR_TYPE) => Ok(Projector::NemotronOmni),
        Some(other) => Err(refusal(
            KEY_PROJECTOR,
            format!(
                "is \"{other}\"; this crate reads \"{}\", \"{}\", \"{}\", \"{}\", \"{}\" and \"{}\"",
                deepseek41v::PROJECTOR_TYPE,
                gemma4v::PROJECTOR_TYPE,
                glm5v::PROJECTOR_TYPE,
                minicpmv::PROJECTOR_TYPE,
                radio::PROJECTOR_TYPE,
                qwen3vl::PROJECTOR_TYPE
            ),
        )),
        None => Err(refusal(KEY_PROJECTOR, "is not a string")),
    }
}

/// The vision projector type: the common key, or the vision key of a multimodal file.
pub fn projector_type(file: &Gguf) -> Result<&str, VisionError> {
    projector_value(file)?
        .as_str()
        .ok_or_else(|| refusal(KEY_PROJECTOR, "is not a string"))
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

    /// The projector type picks the module; a stranger is refused by its name, which lists
    /// the types this crate reads.
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
            projector(&table("minicpmv4_7")).unwrap(),
            Projector::Minicpmv47
        );
        assert_eq!(
            projector(&table("nemotron_omni")).unwrap(),
            Projector::NemotronOmni
        );
        assert_eq!(
            projector(&table("qwen2vl_merger")).unwrap_err().to_string(),
            "metadata clip.projector_type: is \"qwen2vl_merger\"; this crate reads \
             \"deepseek41v\", \"gemma4v\", \"glm5v\", \"minicpmv4_7\", \"nemotron_omni\" and \
             \"qwen3vl_merger\""
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
            assert_eq!(media.image_token(), Some(248_056));
            assert_eq!(media.part_join(), PartJoin::LlamaServer);
        }
    }

    /// Bolt's file opens to its own card figure at the limits it is opened with — the tower's
    /// weights, and activations for the most tokens an image may take — and to its own data and
    /// llama-server's join rule.
    #[test]
    fn a_bolt_file_opens_to_its_figures() {
        let limits = |min, max| TokenLimits::new(min, max).expect("limits");
        let t = crate::arch::radio::hparams::tests::bolt();
        let file = t.write("clip");
        let hp =
            crate::arch::radio::Hparams::read(&Gguf::open(&file.0).expect("opens")).expect("read");
        for max in [3328, 1024] {
            let (card, media) = open_at(&t, "clip", limits(256, max)).expect("open");
            let want = crate::arch::radio::card::of(&hp, limits(256, max));
            assert_eq!((card.weights, card.scratch), (want.weights, want.scratch));
            assert_eq!(media.image_token(), Some(18));
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
            assert_eq!(media.image_token(), Some(154_854));
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

    /// A MiniCPM-V 4.7 file is refused by name once its header reads, and by its own key when it
    /// does not.
    #[test]
    fn a_minicpm_file_is_read_then_refused_for_its_ids() {
        use crate::arch::minicpmv::hparams::tests::minicpm;
        let refused = |t: &Table| open_table(t, "clip").err().expect("refused").to_string();
        assert!(refused(&minicpm()).starts_with(
            "metadata clip.projector_type: is \"minicpmv4_7\"; its image ids come from the text \
             model's tokenizer"
        ));
        assert_eq!(
            refused(&minicpm().with("clip.vision.block_count", Value::U32(24))),
            "metadata clip.vision.block_count: is 24; minicpmv4_7 runs 27 (the mmproj header)"
        );
    }

    /// V4.1's file opens to its own figure and its own join rule; a stranger's and a file of another
    /// architecture are refused by name, and a limit that does not fit GLM's pixel count with it.
    #[test]
    fn other_files_open_or_are_refused_by_name() {
        let (card, media) = open_table(&crate::arch::deepseek41v::hparams::tests::v41(), "clip")
            .expect("V4.1 opens");
        assert_eq!(card.total(), 970_924_032 + 339_878_648);
        assert!(media.embedding_audio().is_none());
        assert!(
            media
                .prepare_audio(&[0.0, 0.0])
                .unwrap_err()
                .to_string()
                .contains("does not support decoded audio")
        );
        let audio = |table: Table| {
            let file = table.write("clip");
            super::open_audio(&Gguf::open(&file.0).unwrap())
        };
        assert!(audio(clef()).unwrap().is_none());
        let model =
            audio(clef().with("clip.audio.projector_type", Value::String("gemma4a".into())))
                .unwrap()
                .expect("audio input rule");
        assert_eq!(
            model.embedding_audio(),
            Some(("<|audio>", "<audio|>", 16000))
        );
        let chunks = model.prepare_audio(&[0.0, 0.0]).unwrap();
        assert_eq!(
            (chunks[0].frames, chunks[0].span_len, chunks[0].data.len()),
            (1, 1, 128)
        );
        assert!(model.prepare(&crate::Rgb8::filled(1, 1, [0; 3])).is_err());
        for kind in [Value::String("unknown".into()), Value::U32(1)] {
            assert!(
                audio(clef().with("clip.audio.projector_type", kind))
                    .err()
                    .expect("invalid audio projector")
                    .to_string()
                    .contains("must be gemma4a")
            );
        }
        assert_eq!(media.part_join(), PartJoin::Uniform("\n\n"));
        let refused =
            |t: &Table, arch: &str| open_table(t, arch).err().expect("refused").to_string();
        assert_eq!(
            refused(&table("qwen2vl_merger"), "clip"),
            "metadata clip.projector_type: is \"qwen2vl_merger\"; this crate reads \
             \"deepseek41v\", \"gemma4v\", \"glm5v\", \"minicpmv4_7\", \"nemotron_omni\" and \
             \"qwen3vl_merger\""
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
