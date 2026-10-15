//! The Nemotron 3 Nano family. The logits set is dumped from the tokenizer
//! family's model ([`super::tokenizer::nemotron_model`]) by mainline llama.cpp
//! ([`super::tokenizer::NEMOTRON_BUILD`]), the build that family already pins.
//! ik has no `nemotron_h`.

use crate::family::{Build, Family, Identity};

/// The architecture every nemotron manifest names in its `# arch` line.
pub const ARCH: &str = "nemotron_h";

/// Last-position logits of each prompt, mainline on the CPU.
pub const LOGITS_SET: &str = "ref-nemotron-logits";

/// Mainline's last-position logits (`just dump-ref-nemotron`).
pub static LOGITS: Family = Family {
    name: "ref-nemotron-logits",
    sets: &[LOGITS_SET],
    resolve: None,
    recipe: "just dump-ref-nemotron",
    identity: Identity::NemotronLogits,
    arch: Some(ARCH),
    build: Some(Build::Is(super::tokenizer::NEMOTRON_BUILD)),
    runs: Some(super::tokenizer::nemotron_model),
    draft_runs: None,
    consumers: &["gate-gpu-nemotron-e2e"],
};

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&LOGITS];

#[cfg(test)]
mod tests {
    use super::LOGITS;
    use crate::family::Build;

    /// The build pin is the tokenizer family's constant.
    #[test]
    fn the_build_pin_is_the_tokenizer_familys() {
        let Some(Build::Is(p)) = LOGITS.build else {
            panic!("no pin");
        };
        assert_eq!(p, crate::arch::tokenizer::NEMOTRON_BUILD);
        assert_eq!(LOGITS.sets, &["ref-nemotron-logits"]);
        assert_eq!(
            LOGITS.runs().unwrap_or_else(|e| panic!("{e}")),
            crate::arch::tokenizer::NEMOTRON
                .runs()
                .unwrap_or_else(|e| panic!("{e}"))
        );
    }
}
