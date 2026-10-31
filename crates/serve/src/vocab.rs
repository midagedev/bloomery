//! What a model file gives the server beside its engine, for every seat on
//! every backend: [`Vocab`], the file's own vocabulary as the server's
//! [`Tokenizer`], and [`ChatSurface`], its chat template and name
//! (`--chat-template-file`'s text, [`TemplateFile`], in place of the header's
//! template). Host code only: a CUDA seat and the Metal server read the file
//! through the same owner.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::engine::{Decoder, Tokenizer};

/// The file's vocabulary as the server reads it.
pub struct Vocab {
    tok: Arc<tokenizer::Tokenizer>,
    bos: Option<u32>,
    eos: u32,
    user_start: Vec<u32>,
}

impl Vocab {
    /// Wraps a loaded vocabulary; refused when it names no EOS. A BOS it does not name stays
    /// none (the tokenizer itself refuses a vocabulary that adds a BOS without one).
    pub fn new(tok: tokenizer::Tokenizer) -> Result<Vocab, String> {
        let sp = tok.specials();
        let bos = sp.bos;
        let eos = sp
            .eos
            .ok_or_else(|| "the vocabulary names no EOS token".to_owned())?;
        Ok(Vocab {
            tok: Arc::new(tok),
            bos,
            eos,
            user_start: Vec::new(),
        })
    }

    /// The same vocabulary with `marker`, one of its special tokens, as the
    /// token that opens a user message ([`Tokenizer::user_start`]); refused
    /// when no special token has that text.
    pub fn with_user_start(mut self, marker: &str) -> Result<Vocab, String> {
        let id = self
            .tok
            .special_tokens()
            .iter()
            .copied()
            .find(|&id| self.tok.text(id) == Some(marker))
            .ok_or_else(|| format!("the vocabulary has no special token {marker:?}"))?;
        self.user_start = vec![id];
        Ok(self)
    }

    /// Registers ChatML's message start when both the template and vocabulary carry it.
    pub fn with_chat_template(self, template: &str) -> Result<Vocab, String> {
        const MESSAGE_START: &str = "<|im_start|>";
        if template.contains(MESSAGE_START)
            && self
                .tok
                .special_tokens()
                .iter()
                .any(|&id| self.tok.text(id) == Some(MESSAGE_START))
        {
            self.with_user_start(MESSAGE_START)
        } else {
            Ok(self)
        }
    }

    /// The vocabulary itself.
    pub fn tokenizer(&self) -> &tokenizer::Tokenizer {
        &self.tok
    }
}

impl Tokenizer for Vocab {
    fn encode(&self, text: &str) -> Vec<u32> {
        // The rendered prompt carries the BOS string and the role markers as text.
        self.tok.encode(text, false, true)
    }

    fn guard<'a>(&self, text: &'a str, mark: &str) -> std::borrow::Cow<'a, str> {
        self.tok.guard(text, mark)
    }

    fn encode_marked(&self, text: &str, mark: &str) -> Vec<u32> {
        self.tok.encode_marked(text, mark, false, true)
    }

    fn decode(&self, ids: &[u32]) -> String {
        self.tok.decode(ids)
    }

    fn decoder(&self) -> Box<dyn Decoder> {
        Box::new(VocabDecoder {
            tok: Arc::clone(&self.tok),
            pending: Vec::new(),
            cleaner: self.tok.space_cleaner(),
        })
    }

    fn bos(&self) -> Option<u32> {
        self.bos
    }

    fn eos(&self) -> u32 {
        self.eos
    }

    fn stops(&self) -> Vec<u32> {
        // The tokenizer owns the complete end-of-generation set, including explicit HF stops.
        self.tok.eog().to_vec()
    }

    fn add_bos(&self) -> bool {
        self.tok.add_bos()
    }

    fn add_eos(&self) -> bool {
        self.tok.add_eos()
    }

    fn n_vocab(&self) -> usize {
        self.tok.n_vocab()
    }

    fn user_start(&self) -> Vec<u32> {
        self.user_start.clone()
    }

    fn is_special(&self, id: u32) -> bool {
        use tokenizer::attr::{CONTROL, USER_DEFINED};
        self.tok
            .attr(id)
            .is_some_and(|a| a & (CONTROL | USER_DEFINED) != 0)
    }
}

/// `tokenizer::Decoder`'s rule over an owned vocabulary handle: pieces with
/// special tokens rendered, an incomplete UTF-8 tail held, an invalid
/// sequence replaced by U+FFFD, and spaces cleaned where the vocabulary cleans them
/// (`tokenizer::Tokenizer::space_cleaner`). `tokenizer::Decoder` borrows its vocabulary,
/// and the server's decoder must outlive the call that made it.
struct VocabDecoder {
    tok: Arc<tokenizer::Tokenizer>,
    pending: Vec<u8>,
    cleaner: Option<tokenizer::SpaceCleaner>,
}

impl Decoder for VocabDecoder {
    fn push(&mut self, id: u32) -> Option<String> {
        self.pending.extend_from_slice(self.tok.piece(id, true));
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    // The first `valid` bytes were just checked.
                    out.push_str(std::str::from_utf8(&self.pending[..valid]).unwrap_or_default());
                    match e.error_len() {
                        None => {
                            self.pending.drain(..valid);
                            break;
                        }
                        Some(bad) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            self.pending.drain(..valid + bad);
                        }
                    }
                }
            }
        }
        if let Some(cleaner) = &mut self.cleaner {
            let cleaned = cleaner.push(&out);
            out = cleaned;
        }
        (!out.is_empty()).then_some(out)
    }

    fn flush(&mut self) -> String {
        let mut rest = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        if let Some(cleaner) = &mut self.cleaner {
            rest = cleaner.push(&rest);
            rest.push_str(&cleaner.finish());
        }
        rest
    }
}

/// `--chat-template-file PATH`: the chat template read from a file in place of
/// the model file's own ([`ChatSurface::read`]). The seats that take the flag take
/// it through this type, so its spelling and its reading have one owner.
#[derive(Debug, Default)]
pub struct TemplateFile(Option<PathBuf>);

impl TemplateFile {
    /// The flag's spelling.
    pub const FLAG: &'static str = "--chat-template-file";

    /// Takes the file the flag named.
    pub fn set(&mut self, value: &str) {
        self.0 = Some(PathBuf::from(value));
    }

    /// Checks the explicit template and keeps its startup snapshot for the selected seat.
    pub fn check(&self) -> Result<(), String> {
        if let Some(path) = self.path() {
            Self::read(path)?;
        }
        Ok(())
    }

    /// Startup checks and the selected seat share one snapshot, including read failures.
    fn read(file: &Path) -> Result<String, String> {
        crate::flag::file_snapshot(file)
            .map_err(|e| format!("--chat-template-file {}: {e}", file.display()))
    }

    /// The file, when the flag was given.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.0.as_deref()
    }
}

/// The chat surface a seat's file gives the server: the Jinja template and
/// the model's name.
pub struct ChatSurface {
    /// `--chat-template-file`'s text where the seat takes the flag, else the
    /// file's `tokenizer.chat_template`.
    pub template: String,
    /// The file's `general.name`, else the seat's default alias.
    pub name: String,
}

impl ChatSurface {
    /// The surface of the file at `path` from its header's `tokenizer.chat_template`
    /// and `general.name` (`template` and `name`): `template_file`'s text
    /// replaces the header's template, and a file with neither is refused by
    /// name; `name` falls to `default_alias` when the file states none.
    pub fn read(
        path: &Path,
        template: Option<&gguf::Value>,
        name: Option<&gguf::Value>,
        template_file: Option<&Path>,
        default_alias: &str,
    ) -> Result<ChatSurface, String> {
        let template = match template_file {
            Some(file) => TemplateFile::read(file)?,
            None => template
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("{}: no tokenizer.chat_template", path.display()))?
                .to_owned(),
        };
        let name = name
            .and_then(|v| v.as_str())
            .unwrap_or(default_alias)
            .to_owned();
        Ok(ChatSurface { template, name })
    }
}

#[cfg(test)]
mod tests {
    use super::Vocab;
    use crate::engine::Tokenizer;

    #[test]
    fn a_template_file_is_one_startup_snapshot() {
        let dir =
            std::env::temp_dir().join(format!("bloomery-template-snapshot-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("template");
        std::fs::write(&path, "first").unwrap();
        let mut file = super::TemplateFile::default();
        file.set(path.to_str().unwrap());
        file.check().unwrap();
        std::fs::write(&path, "changed").unwrap();
        let chat = super::ChatSurface::read(
            std::path::Path::new("model.gguf"),
            None,
            None,
            file.path(),
            "test",
        )
        .unwrap();
        assert_eq!(chat.template, "first");
        let absent = dir.join("missing");
        file.set(absent.to_str().unwrap());
        let first = file.check().expect_err("the template is absent");
        std::fs::write(&absent, "now present").unwrap();
        assert_eq!(file.check().expect_err("the first read is kept"), first);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn chat_templates_register_only_a_special_message_start() {
        use gguf::Value;

        for (text, kind, template, expected) in [
            ("<|im_start|>", 3, "<|im_start|>user", vec![2]),
            ("<|im_start|>", 3, "{{ messages }}", vec![]),
            ("<|im_start|>", 1, "<|im_start|>user", vec![]),
            ("other", 3, "<|im_start|>user", vec![]),
        ] {
            let tok = tokenizer::Tokenizer::from_keys(&[
                ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
                ("tokenizer.ggml.pre".into(), Value::String("qwen2".into())),
                (
                    "tokenizer.ggml.tokens".into(),
                    Value::Array(
                        ["<s>", "</s>", text]
                            .into_iter()
                            .map(|s| Value::String(s.into()))
                            .collect(),
                    ),
                ),
                (
                    "tokenizer.ggml.token_type".into(),
                    Value::Array([3, 3, kind].into_iter().map(Value::I32).collect()),
                ),
                ("tokenizer.ggml.merges".into(), Value::Array(vec![])),
                ("tokenizer.ggml.bos_token_id".into(), Value::U32(0)),
                ("tokenizer.ggml.eos_token_id".into(), Value::U32(1)),
            ])
            .unwrap();
            let vocab = Vocab::new(tok)
                .unwrap()
                .with_chat_template(template)
                .unwrap();
            assert_eq!(vocab.user_start(), expected, "{text} {kind} {template}");
        }
    }

    /// A vocabulary that names no BOS is wrapped where it adds none, with no BOS id of the
    /// server's own making; one that adds a BOS and names none does not get past the tokenizer,
    /// by name.
    #[test]
    fn a_vocabulary_names_no_bos_only_where_it_adds_none() {
        use gguf::Value;

        let keys = |add_bos: bool| {
            vec![
                ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
                ("tokenizer.ggml.pre".into(), Value::String("qwen2".into())),
                (
                    "tokenizer.ggml.tokens".into(),
                    Value::Array(
                        ["a", "</s>", "<|im_start|>"]
                            .into_iter()
                            .map(|s| Value::String(s.into()))
                            .collect(),
                    ),
                ),
                (
                    "tokenizer.ggml.token_type".into(),
                    Value::Array([1, 3, 3].into_iter().map(Value::I32).collect()),
                ),
                ("tokenizer.ggml.merges".into(), Value::Array(vec![])),
                ("tokenizer.ggml.eos_token_id".into(), Value::U32(1)),
                ("tokenizer.ggml.add_bos_token".into(), Value::Bool(add_bos)),
            ]
        };
        let tok = tokenizer::Tokenizer::from_keys(&keys(false)).unwrap();
        let vocab = Vocab::new(tok).unwrap();
        assert_eq!(vocab.bos(), None);
        assert_eq!(vocab.bos_to_prepend(), Ok(None));
    }
}
