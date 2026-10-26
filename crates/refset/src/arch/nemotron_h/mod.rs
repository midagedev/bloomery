//! The Nemotron 3 Nano family. The logits set is dumped from the tokenizer
//! family's model ([`super::tokenizer::nemotron_model`]) by mainline llama.cpp
//! ([`BUILD`]), the libraries `nemotron_ref` is linked against, built from the
//! tree the tokenizer family pins by commit. ik has no `nemotron_h`. The fixture tier's mirror is in [`fixture`].

pub mod fixture;

use crate::family::{Build, Family, Identity};

/// The architecture every nemotron manifest names in its `# arch` line.
pub const ARCH: &str = "nemotron_h";

/// The executable and library of `/home/user/llama.cpp-mainline` that `nemotron_ref` is linked against
/// (`tools/ref/build-nemotron-ref.sh` records them, `tools/ref/dump-nemotron.sh` holds the live files to the record). The
/// soname is `libllama.so.0`; the pin's word is the one [`crate::tokenizer::build_id`] spells.
// PIN(2026-10-14): /home/user/llama.cpp-mainline's llama-tokenize and libllama.so.0.
pub(crate) const BUILD: &str = "llama-tokenize 5df95d3a942effc971ff170432d840fd \
                                libllama.so 3575b1f39e7773379fec011d39f32a77";

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
    build: Some(Build::Is(BUILD)),
    runs: Some(super::tokenizer::nemotron_model),
    draft_runs: None,
    consumers: &["gate-gpu-nemotron-e2e"],
};

/// The architecture's families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&LOGITS, &fixture::LOGITS];

#[cfg(test)]
mod tests {
    use super::{BUILD, LOGITS};
    use crate::family::Build;

    /// The build pin is the md5s of the libraries the dumper links, as `build_id` spells them, and the
    /// model the tokenizer family's.
    #[test]
    fn the_build_pin_is_the_linked_libraries() {
        let Some(Build::Is(p)) = LOGITS.build else {
            panic!("no pin");
        };
        assert_eq!(p, BUILD);
        let mut words = BUILD.split_whitespace();
        let (binary, library) = (words.nth(1).unwrap_or(""), words.nth(1).unwrap_or(""));
        assert_eq!(BUILD, crate::tokenizer::build_id(binary, library));
        assert_eq!(LOGITS.sets, &["ref-nemotron-logits"]);
        assert_eq!(
            LOGITS.runs().unwrap_or_else(|e| panic!("{e}")),
            crate::arch::tokenizer::NEMOTRON
                .runs()
                .unwrap_or_else(|e| panic!("{e}"))
        );
    }
}
