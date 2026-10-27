//! The side seat's flags: `--embd-model` and `--embd-hf` name the embedding model, which the hub
//! may have to supply, so the flags exist in a build with the `hf` feature.

use hf::source;

use super::{ROWS, Side, row};
use crate::GateError;
use crate::model_file;
use crate::vision_seat;
use gguf::Split;

/// The side seat's flags, llama-server's `-m` and `-hf` behind `--embd-`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Flags {
    model: source::Flags,
    image: vision_seat::Flags,
    ctx: Option<usize>,
    cache_bytes: Option<usize>,
}

impl Flags {
    /// `args` with the side seat's flags and their values taken out: the flags and the rest. No
    /// file is read here. A flag with no value or given twice, and `--embd-model` beside
    /// `--embd-hf`, are refused by name.
    pub fn take(args: &[String]) -> Result<(Flags, Vec<String>), GateError> {
        let mut f = Flags::default();
        let mut selected = Vec::new();
        let mut rest = Vec::with_capacity(args.len());
        let mut image_seen = std::collections::HashSet::new();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            let flag = a.as_str();
            if let Some(image_flag) = match flag {
                "--embd-mmproj" => Some("--mmproj"),
                "--embd-no-mmproj" => Some("--no-mmproj"),
                "--embd-image-min-tokens" => Some("--image-min-tokens"),
                "--embd-image-max-tokens" => Some("--image-max-tokens"),
                _ => None,
            } {
                if !image_seen.insert(flag) {
                    return Err(format!("{flag} is given twice").into());
                }
                f.image
                    .take(image_flag, &mut it.by_ref().map(String::as_str))
                    .map_err(|e| format!("{flag}: {e}"))?;
                continue;
            }
            if flag == "--embd-cache-bytes" {
                let v = it.next().ok_or("--embd-cache-bytes needs a value")?;
                let bytes = v
                    .parse::<usize>()
                    .map_err(|e| format!("--embd-cache-bytes {v:?}: {e}"))?;
                if f.cache_bytes.replace(bytes).is_some() {
                    return Err("--embd-cache-bytes is given twice".into());
                }
                continue;
            }
            if flag == "--embd-ctx" {
                let v = it.next().ok_or("--embd-ctx needs a value")?;
                let ctx = v
                    .parse::<usize>()
                    .map_err(|e| format!("--embd-ctx {v:?}: {e}"))?;
                if f.ctx.is_some() {
                    return Err("--embd-ctx is given twice".into());
                }
                if ctx == 0 {
                    return Err("--embd-ctx needs one positive count".into());
                }
                f.ctx = Some(ctx);
                continue;
            }
            let flag = match flag {
                "--embd-model" => "--embd-model",
                "--embd-hf" => "--hf",
                _ => {
                    rest.push(a.clone());
                    if crate::llama_launch::serve_takes_value(a) {
                        rest.extend(it.next().cloned());
                    }
                    continue;
                }
            };
            selected.push(flag.to_owned());
            selected.extend(it.next().cloned());
        }
        let (model, _) = source::take(&selected, &["--embd-model"])
            .map_err(|e| format!("embedding flags (--embd-model/--embd-hf): {e}"))?;
        f.model = model;
        if !f.names() && (!image_seen.is_empty() || f.ctx.is_some() || f.cache_bytes.is_some()) {
            return Err("embedding image/context flags need --embd-model or --embd-hf".into());
        }
        Ok((f, rest))
    }

    /// Whether the flags name an embedding model.
    #[must_use]
    pub fn names(&self) -> bool {
        self.model.path.is_some() || self.model.hf.is_some()
    }

    /// Checks the side model's image flags without opening or fetching either model.
    pub fn check_before_model(&self) -> Result<(), GateError> {
        source::resolve(&self.model, None)
            .map_err(|e| format!("embedding flags (--embd-model/--embd-hf): {e}"))?;
        self.image
            .check_before_model(self.model.hf.is_some())
            .map_err(|e| format!("embedding image flags: {e}").into())
    }

    /// The side model the flags name, `None` when they name none: its file found, or fetched
    /// from `--embd-hf`'s repo; its architecture one of [`ROWS`]; its context the row's
    /// ([`Row::ctx`]); and its bytes. Each step refuses by name. A seat calls it after its levers
    /// and its own arguments, and before its plan.
    pub fn read(&self) -> Result<Option<Side>, GateError> {
        self.check_before_model()?;
        let Some(path) = model_file::resolve(&self.model, None, hf::fetch::Sources::Gguf)? else {
            return Ok(None);
        };
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let arch = split.architecture().unwrap_or("<missing>");
        let row = row(arch).ok_or_else(|| {
            let archs: Vec<&str> = ROWS.iter().map(|r| r.arch).collect();
            format!(
                "--embd-model {}: a {arch} file; a side seat serves {}",
                path.display(),
                archs.join(", ")
            )
        })?;
        let train = split
            .arch_get_u64("context_length")
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| format!("{}: no {arch}.context_length", path.display()))?;
        let limit = row.ctx.min(train);
        let ctx = self.ctx.unwrap_or(limit);
        if ctx > limit {
            return Err(
                format!("--embd-ctx {ctx}: the side model supports at most {limit}").into(),
            );
        }
        let mut bytes = (row.bytes)(&split, ctx)
            .map_err(|e| format!("--embd-model {}: {e}", path.display()))?;
        let projector_path = if let Some(p) = self.image.mmproj()? {
            Some(p.to_path_buf())
        } else if let (Some(hf), Some(mmproj)) = (&self.model.hf, row.projector_file) {
            if cfg!(feature = "vision") && !self.image.disabled() {
                let repo = hf::RepoRef::parse(hf)?;
                Some(
                    model_file::fetch_exact(&repo.repo, &[mmproj])?
                        .into_iter()
                        .next()
                        .ok_or("the projector fetch returned no file")?,
                )
            } else {
                None
            }
        } else {
            None
        };
        let projector = match projector_path.as_deref() {
            Some(p) => {
                let p = vision_seat::Projector::open(p, &self.image, row.projector_types)?;
                let n_embd = split
                    .arch_get_u64("embedding_length")
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or("the embedding width is absent")?;
                p.check_text(
                    &vision_seat::TextFacts {
                        n_embd,
                        imrope: None,
                        ple: vision_seat::PleImage::Absent,
                    },
                    &|_| None,
                )?;
                bytes = bytes
                    .checked_add(p.card.total())
                    .ok_or("embedding model and tower byte count overflows")?;
                Some(p)
            }
            None if self.image.sets_limits() => {
                return Err(
                    "--embd-image-min-tokens/--embd-image-max-tokens need --embd-mmproj".into(),
                );
            }
            None => None,
        };
        let audio_bytes = if let Some(p) = &projector {
            (row.audio_bytes)(&gguf::Gguf::open(&p.path)?)?
        } else {
            0
        };
        bytes = bytes
            .checked_add(audio_bytes)
            .ok_or("embedding model and audio byte count overflows")?;
        let prompts = model_file::sentence_prompts(&self.model, &path)?;
        Ok(Some(Side {
            prompts,
            row,
            path,
            ctx,
            bytes,
            projector,
            audio_bytes,
            cache_bytes: self.cache_bytes.unwrap_or(0),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| (*s).to_owned()).collect()
    }

    /// The flags come out of the seat's arguments and the rest stays in order; each misuse is
    /// refused by name.
    #[test]
    fn the_side_flags_come_out_and_misuse_is_refused() {
        let (f, rest) = Flags::take(&args(&[
            "--ctx",
            "4096",
            "--embd-model",
            "/e.gguf",
            "--port",
            "8080",
        ]))
        .expect("take");
        assert_eq!(rest, args(&["--ctx", "4096", "--port", "8080"]));
        assert_eq!(
            f.model.path,
            Some(("--embd-model".to_owned(), "/e.gguf".to_owned()))
        );
        assert!(f.names());
        let (hf, rest) =
            Flags::take(&args(&["--embd-hf", "x/y:Q8_0", "--ctx", "4096"])).expect("HF model");
        assert_eq!(hf.model.hf.as_deref(), Some("x/y:Q8_0"));
        assert!(hf.model.path.is_none());
        assert!(hf.names());
        assert_eq!(rest, args(&["--ctx", "4096"]));
        let (none, _) = Flags::take(&args(&["--port", "1"])).expect("none");
        assert_eq!(none, Flags::default());
        assert!(!none.names());
        assert_eq!(f.cache_bytes, None);
        for bytes in ["0", "1048576"] {
            let (cached, rest) =
                Flags::take(&args(&["--embd-model", "/a", "--embd-cache-bytes", bytes])).unwrap();
            assert_eq!(cached.cache_bytes, Some(bytes.parse().unwrap()));
            assert!(rest.is_empty());
        }
        for bad in [
            args(&["--embd-cache-bytes", "1"]),
            args(&["--embd-model", "/a", "--embd-cache-bytes"]),
            args(&["--embd-model", "/a", "--embd-cache-bytes", "-1"]),
            args(&[
                "--embd-model",
                "/a",
                "--embd-cache-bytes",
                "0",
                "--embd-cache-bytes",
                "1",
            ]),
        ] {
            assert!(Flags::take(&bad).is_err(), "{bad:?}");
        }
        let (image, rest) = Flags::take(&args(&[
            "--embd-model",
            "/e.gguf",
            "--embd-ctx",
            "8192",
            "--embd-mmproj",
            "/p.gguf",
            "--embd-image-min-tokens",
            "70",
            "--embd-image-max-tokens",
            "1120",
            "--port",
            "8080",
        ]))
        .expect("image flags");
        assert_eq!(rest, args(&["--port", "8080"]));
        assert_eq!(image.ctx, Some(8192));
        assert_eq!(
            image.image.mmproj().expect("projector"),
            Some(std::path::Path::new("/p.gguf"))
        );
        assert!(image.image.sets_limits());
        let default = crate::vision_seat::TokenLimits::new(1, 1).expect("limits");
        let limits = image.image.limits_over(default).expect("image limits");
        assert_eq!((limits.min(), limits.max()), (70, 1120));
        for flag in ["--embd-image-min-tokens", "--embd-image-max-tokens"] {
            let (zero, _) =
                Flags::take(&args(&["--embd-model", "/a", flag, "0"])).expect("count parsed");
            let e = zero
                .image
                .limits_over(default)
                .expect_err("zero limit refused")
                .to_string();
            assert!(
                e.starts_with("--image-min-tokens/--image-max-tokens: image size limits: "),
                "{flag}: {e}"
            );
            assert!(e.contains("need 1 <= min <= max"), "{flag}: {e}");
            let (one, _) =
                Flags::take(&args(&["--embd-model", "/a", flag, "1"])).expect("count parsed");
            let limits = one.image.limits_over(default).expect("one token limit");
            assert_eq!((limits.min(), limits.max()), (1, 1));
        }
        for (bad, why) in [
            (&["--embd-model"][..], "--embd-model needs a value"),
            (&["--embd-hf"], "--hf needs a value"),
            (&["--embd-model", "/a", "--embd-model", "/b"], "named twice"),
            (&["--embd-hf", "x/y", "--embd-hf", "a/b"], "named twice"),
            (&["--embd-model", "/a", "--embd-hf", "x/y"], "named twice"),
            (&["--embd-hf", "x/y", "--embd-model", "/a"], "named twice"),
            (&["--embd-mmproj", "/p"], "need --embd-model or --embd-hf"),
            (&["--embd-no-mmproj"], "need --embd-model or --embd-hf"),
            (
                &["--embd-image-min-tokens", "70"],
                "need --embd-model or --embd-hf",
            ),
            (
                &["--embd-image-max-tokens", "1120"],
                "need --embd-model or --embd-hf",
            ),
            (&["--embd-ctx", "8192"], "need --embd-model or --embd-hf"),
            (
                &["--embd-model", "/a", "--embd-ctx", "1", "--embd-ctx", "2"],
                "--embd-ctx is given twice",
            ),
            (
                &["--embd-model", "/a", "--embd-ctx", "0"],
                "needs one positive count",
            ),
            (
                &[
                    "--embd-model",
                    "/a",
                    "--embd-mmproj",
                    "/p",
                    "--embd-mmproj",
                    "/q",
                ],
                "--embd-mmproj is given twice",
            ),
            (
                &["--embd-model", "/a", "--embd-no-mmproj", "--embd-no-mmproj"],
                "--embd-no-mmproj is given twice",
            ),
            (
                &[
                    "--embd-model",
                    "/a",
                    "--embd-image-min-tokens",
                    "70",
                    "--embd-image-min-tokens",
                    "80",
                ],
                "--embd-image-min-tokens is given twice",
            ),
            (
                &[
                    "--embd-model",
                    "/a",
                    "--embd-image-max-tokens",
                    "1120",
                    "--embd-image-max-tokens",
                    "1200",
                ],
                "--embd-image-max-tokens is given twice",
            ),
            (&["--embd-model", "/a", "--embd-mmproj"], "--embd-mmproj"),
        ] {
            let e = Flags::take(&args(bad)).expect_err("refused").to_string();
            assert!(e.contains(why), "{bad:?}: {e}");
        }
    }
}
