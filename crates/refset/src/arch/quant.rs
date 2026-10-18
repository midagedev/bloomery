//! The quant oracle's family: llama.cpp mainline's quantization, `to_float`, activation conversion and `vec_dot` per weight type,
//! on seeded rows (`tools/ref/quant_oracle.cpp`, `just dump-ref-quant`). The family pins the harness by the md5 of its source and
//! the library by the mainline commit it was built at: the md5 of one machine's library matches no other build of the commit, so
//! the set states the commit ([`crate::quant`]).

use crate::RefError;
use crate::family::{Build, Family, Identity};
use crate::quant::SYNTHETIC;

/// The key of this family in the table of all of them.
pub const ARCH: &str = "quant";

/// The md5 of `tools/ref/quant_oracle.cpp`: the source the set's rows, sizes and seeds come from.
// PIN(2026-10-19): the harness without the six case fields it never read (the md5 moved, the files it dumps did not);
// `the_harness_pin_is_the_source_in_the_tree` holds it to the tree.
pub const HARNESS: &str = "f87b4d310e51d9d80643fd0f735237cf";

/// The mainline commit the set names in its `# build` line.
// PIN(2026-10-18): /home/user/llama.cpp-36a73916 at its HEAD 36a73916ee0c (ggml-org master, 2026-10-07), the Qwen image-input
// sets' tree (`crate::clefvis::QVIS_LCPP_BUILD`); it holds every type of the table, Q2_0 and NVFP4 among them.
pub const LCPP_BUILD: &str = "36a73916e";

/// The rows ggml's quantizer makes from a fixed seed and the rows of random codes, per type and row length: no model file.
pub static MAINLINE: Family = Family {
    name: "quant-mainline",
    sets: &["ref-quant"],
    resolve: None,
    recipe: "just dump-ref-quant",
    identity: Identity::QuantManifest { harness: HARNESS },
    arch: None,
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(synthetic),
    draft_runs: None,
    consumers: &["gate-refset"],
};

fn synthetic() -> Result<String, RefError> {
    Ok(SYNTHETIC.to_string())
}

/// The families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&MAINLINE];

#[cfg(test)]
mod tests {
    use super::{ARCH, FAMILIES, HARNESS, LCPP_BUILD, MAINLINE};
    use crate::md5::{check_hex, hex_of};
    use std::path::Path;

    /// The table holds the family under its key, and its pins are spelled as a set states them.
    #[test]
    fn the_family_is_a_row_of_the_table() {
        assert_eq!(crate::arch::families(ARCH).len(), 1);
        for f in FAMILIES {
            assert!(std::ptr::eq(
                crate::arch::named(f.name).unwrap_or(&MAINLINE),
                *f
            ));
        }
        check_hex("harness pin", HARNESS, &"pin").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(LCPP_BUILD.len(), 9);
        assert!(LCPP_BUILD.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_eq!(MAINLINE.recipe, "just dump-ref-quant");
        assert_eq!(
            MAINLINE.path("ref-quant"),
            crate::data_dir().join("ref-quant")
        );
    }

    /// The harness pin is the source in the tree: an edit of `quant_oracle.cpp` without a new dump and a new pin is red.
    #[test]
    fn the_harness_pin_is_the_source_in_the_tree() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/quant_oracle.cpp");
        let source = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(
            hex_of(&source),
            HARNESS,
            "tools/ref/quant_oracle.cpp changed: dump the set again (just dump-ref-quant) and pin the new md5"
        );
    }
}
