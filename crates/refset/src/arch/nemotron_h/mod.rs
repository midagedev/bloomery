//! The Nemotron 3 Nano family. The logits set is dumped from [`MODEL`] by
//! mainline llama.cpp ([`super::tokenizer::NEMOTRON_BUILD`]), which is the
//! build the tokenizer family already pins. ik has no `nemotron_h`.

use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The architecture every nemotron manifest names in its `# arch` line.
pub const ARCH: &str = "nemotron_h";

/// The file the tree runs. The tokenizer family owns the path.
pub const MODEL: &str = super::tokenizer::NEMOTRON_MODEL;

/// Last-position logits of four prompts, mainline on the CPU.
pub const LOGITS_SET: &str = "ref-nemotron-logits";

/// [`MODEL`], as a family's `runs`.
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// Mainline's last-position logits (`just dump-ref-nemotron`).
pub static LOGITS: Family = Family {
    name: "ref-nemotron-logits",
    sets: &[LOGITS_SET],
    resolve: None,
    recipe: "just dump-ref-nemotron",
    identity: Identity::NemotronLogits,
    arch: Some(ARCH),
    build: Some(Build::Is(super::tokenizer::NEMOTRON_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&LOGITS];

#[cfg(test)]
mod tests {
    use super::LOGITS;
    use crate::RefError;
    use crate::family::Build;
    use crate::nemotron::LogitsSet;
    use std::path::Path;

    fn pin() -> &'static str {
        match LOGITS.build {
            Some(Build::Is(p)) => p,
            _ => panic!("the logits family pins no build"),
        }
    }

    fn runs() -> String {
        LOGITS.runs().unwrap_or_else(|e| panic!("{e}"))
    }

    /// Write one prompt's ids and logits and the manifest. `complete` adds the trailer.
    fn write(dir: &Path, model: &str, build: &str, complete: bool) {
        std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(dir.join("p.ids"), "1\n2\n3\n4\n5\n").unwrap_or_else(|e| panic!("{e}"));
        let logits: Vec<u8> = [0.0f32, 1.5, -0.5, 0.25]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        std::fs::write(dir.join("p.logits.f32"), logits).unwrap_or_else(|e| panic!("{e}"));
        let mut text = format!(
            "# model\t{model}\n# build\t{build}\n# arch\tnemotron_h\n\
             # prompt columns\tname\tn_ids\tn_vocab\tids\tlogits\n\
             prompt\tp\t5\t4\tp.ids\tp.logits.f32\n"
        );
        if complete {
            text.push_str("# complete\t1\t0\n");
        }
        std::fs::write(dir.join("MANIFEST.tsv"), text).unwrap_or_else(|e| panic!("{e}"));
    }

    fn dir(tag: &str) -> std::path::PathBuf {
        let d =
            std::env::temp_dir().join(format!("bloomery-nemotron-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// The build pin is the tokenizer family's constant.
    #[test]
    fn the_build_pin_is_the_tokenizer_familys() {
        let Some(Build::Is(p)) = LOGITS.build else {
            panic!("no pin");
        };
        assert_eq!(p, crate::arch::tokenizer::NEMOTRON_BUILD);
        assert_eq!(LOGITS.sets, &["ref-nemotron-logits"]);
        assert_eq!(runs(), crate::arch::tokenizer::NEMOTRON_MODEL);
    }

    /// A finished set opens, and `logits` is that column's file.
    #[test]
    fn a_finished_set_reads_logits_by_column_name() {
        let d = dir("ok");
        write(&d, &runs(), pin(), true);
        let set = LogitsSet::open(&d, &LOGITS).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(set.prompts.len(), 1);
        let row = set
            .logits(&set.prompts[0])
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(row[1], 1.5);
        std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
    }

    /// No `# complete`: unfinished, even when the other lines are well formed.
    #[test]
    fn a_set_without_its_trailer_is_unfinished() {
        let d = dir("trailer");
        write(&d, &runs(), pin(), false);
        match LogitsSet::open(&d, &LOGITS) {
            Err(RefError::Unfinished { set }) => assert_eq!(set, d.display().to_string()),
            other => panic!("a set without its trailer: {other:?}"),
        }
        std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
    }

    /// Another model path is stale.
    #[test]
    fn a_set_of_another_model_is_stale() {
        let d = dir("model");
        write(&d, "/models/other/file.gguf", pin(), true);
        match LogitsSet::open(&d, &LOGITS) {
            Err(RefError::Stale { dumped_from, .. }) => {
                assert_eq!(dumped_from, "/models/other/file.gguf");
            }
            other => panic!("another file: {other:?}"),
        }
        std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
    }

    /// Another mainline build is foreign.
    #[test]
    fn a_set_of_another_build_is_foreign() {
        let d = dir("build");
        write(
            &d,
            &runs(),
            "llama-tokenize 00000000000000000000000000000000 libllama.so 00000000000000000000000000000000",
            true,
        );
        match LogitsSet::open(&d, &LOGITS) {
            Err(RefError::Foreign { field, .. }) => assert_eq!(field, "build"),
            other => panic!("another build: {other:?}"),
        }
        std::fs::remove_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
    }
}
