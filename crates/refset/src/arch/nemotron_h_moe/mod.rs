//! The Nemotron 3.5 Bolt family (`nemotron_h_moe`). The logits set is dumped from the Bolt GGUF by mainline llama.cpp
//! at `cd0a5b6a1` ([`BUILD`]), which loads files whose layers hold different routed-expert counts; the Nano's family pins the older tree. `tools/ref/dump-nemotron.sh` is the one script for both
//! and its reader is [`crate::nemotron::LogitsSet`]. The Bolt file sets `add_bos_token` false, so its prompts are the
//! Nano's ids without the leading BOS. ik has no `nemotron_h_moe`. The fixture tier's mirror is in [`fixture`].

pub mod fixture;

use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The architecture the Bolt manifest names in its `# arch` line.
pub const ARCH: &str = "nemotron_h_moe";

/// Last-position logits of each prompt, mainline on the CPU.
pub const LOGITS_SET: &str = "ref-nemotron-moe-logits";

/// The build of `/home/user/llama.cpp-mainline-bolt` (`cd0a5b6a1`, CPU only), spelled as [`crate::tokenizer::build_id`].
const BUILD: &str =
    "llama-tokenize f39e6d406cdb6232145cd30a85ca4fb0 libllama.so d846809bb9cd2cb8aa304ed4b9003fed";

/// The Bolt file the set is dumped from (`tools/ref/models/nemotron_h_moe.sh`).
pub const MODEL: &str = "/models/nemotron-3.5-bolt/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf";

fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}

/// Mainline's last-position logits (`just dump-ref-nemotron-moe`).
pub static LOGITS: Family = Family {
    name: "ref-nemotron-moe-logits",
    sets: &[LOGITS_SET],
    resolve: None,
    recipe: "just dump-ref-nemotron-moe",
    identity: Identity::NemotronLogits,
    arch: Some(ARCH),
    build: Some(Build::Is(BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &[],
};

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&LOGITS, &fixture::LOGITS];

#[cfg(test)]
mod tests {
    use super::LOGITS;
    use crate::family::Build;

    /// The build pin is not the Nano's: `NEMOTRON_BUILD`'s tree loads no per-layer expert counts, so a set
    /// pinned to it could not have been dumped from the Bolt file.
    #[test]
    fn the_bolt_build_is_not_the_nanos() {
        let Some(Build::Is(p)) = LOGITS.build else {
            panic!("no pin");
        };
        assert_ne!(p, crate::arch::tokenizer::NEMOTRON_BUILD);
    }
}
