//! The dequantization families: the rows ggml's `to_float` gives, per type,
//! of a model file the engine opens, and of the synthetic rows that cover
//! the types no model file on the box holds (`tools/ref/dequant_ref.cpp`,
//! `just dump-ref-dequant`). A family pins the harness and the ggml library
//! it loads by md5 ([`crate::dequant::build_id`]): `to_float` is the
//! library's code.

use crate::RefError;
use crate::dequant::SYNTHETIC;
use crate::family::{Build, Family, Identity};

/// The key of these families in the table of all of them.
pub const ARCH: &str = "dequant";

/// The V2-Lite file `dequant_ref` opens by default (the `deepseek2` profile's
/// model).
pub const V2LITE_MODEL: &str = "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf";

/// The harness built by `tools/ref/build-dequant.sh` against ik's `libggml.so`
/// (`/home/user/ik_llama.cpp`): the executable's bytes follow
/// `dequant_ref.cpp` and the compiler, the library's follow ik's build.
// PIN(2026-10-08): the dequant_ref and libggml.so of /home/user/ik_llama.cpp.
pub const BUILD: &str = "dequant_ref e265fd2f62e16c4f8c5ebc5d28705f41 \
                         libggml.so 75ce969e36e0fd4d520b5086a16c8b57";

/// The first rows of one tensor of each type of the V2-Lite file. The set has
/// a directory of its own: `ref` is where the dump before the identity file
/// wrote it, among the other V2-Lite oracles' files.
pub static V2LITE: Family = Family {
    name: "dequant-v2lite",
    sets: &["ref-dequant-v2lite"],
    resolve: None,
    recipe: "just dump-ref-dequant",
    identity: Identity::DequantManifest,
    arch: None,
    build: Some(Build::Is(BUILD)),
    runs: Some(v2lite_model),
    draft_runs: None,
    consumers: &["gate-1-1"],
};

/// The same for every type of the V4.1 file the tree runs; the set's name
/// carries the file's suffix ([`gguf::v41::set`]).
pub static V41: Family = Family {
    name: "dequant-v41",
    sets: &["ref-dequant-v41"],
    resolve: Some(gguf::v41::set),
    recipe: "just dump-ref-dequant",
    identity: Identity::DequantManifest,
    arch: None,
    build: Some(Build::Is(BUILD)),
    runs: Some(super::deepseek41::model),
    draft_runs: None,
    consumers: &["gate-1-1"],
};

/// Rows of the types no model file on the box holds (q2_K, iq2_xs, iq3_xxs,
/// iq4_xs): ggml-quantized rows and rows of random codes, from a fixed seed.
/// The i-quant row cores' gates read its `.blocks` files too.
pub static SYNTH: Family = Family {
    name: "dequant-synth",
    sets: &["ref-synth"],
    resolve: None,
    recipe: "just dump-ref-dequant",
    identity: Identity::DequantManifest,
    arch: None,
    build: Some(Build::Is(BUILD)),
    runs: Some(synthetic),
    draft_runs: None,
    consumers: &[
        "gate-1-1",
        "gate-gpu-gemm",
        "gate-gpu-iq",
        "gate-gpu-iq-sel",
    ],
};

fn v2lite_model() -> Result<String, RefError> {
    Ok(V2LITE_MODEL.to_string())
}

fn synthetic() -> Result<String, RefError> {
    Ok(SYNTHETIC.to_string())
}

/// The families, in the order `refset-check` lists them.
pub static FAMILIES: &[&Family] = &[&V2LITE, &V41, &SYNTH];

#[cfg(test)]
mod tests {
    use super::{ARCH, BUILD, FAMILIES, SYNTH, V2LITE, V2LITE_MODEL, V41};
    use crate::RefError;
    use crate::dequant::{MANIFEST, build_id};
    use crate::family::{Build, Family};
    use crate::md5::{check_hex, hex_of};
    use std::path::Path;

    /// The harness's and the library's md5 the pin names.
    fn md5s(pin: &str) -> (&str, &str) {
        let mut words = pin.split_whitespace();
        (words.nth(1).unwrap_or(""), words.nth(1).unwrap_or(""))
    }

    /// Write a set into `dir` as the dumper would for `model`, stating `build`.
    fn write(dir: &Path, model: &str, build: &str) {
        let (binary, library) = md5s(build);
        let raw = [1u8, 2, 3, 4];
        std::fs::write(dir.join("f32.raw"), raw).unwrap_or_else(|e| panic!("{e}"));
        let manifest = format!(
            "# dequant_ref\t/data/bin/dequant_ref\t{binary}\n\
             # libggml\t/ik/libggml.so\t{library}\n\
             # model\t{model}\n\
             file\tbytes\tmd5\n\
             f32.raw\t{}\t{}\n\
             # complete\n",
            raw.len(),
            hex_of(&raw)
        );
        std::fs::write(dir.join(MANIFEST), manifest).unwrap_or_else(|e| panic!("{e}"));
    }

    fn check(f: &Family, what: &str, model: &str, build: &str) -> Result<(), RefError> {
        let dir = std::env::temp_dir().join(format!(
            "bloomery-refset-arch-dequant-{what}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
        write(&dir, model, build);
        let r = f.check_set(&dir).map(|_| ());
        std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        r
    }

    /// The table holds the three sets of the dequant gate under one key: the
    /// V2-Lite file's in the `ref` directory, the V4.1 file's under the file's
    /// suffix, the synthetic rows'; one pin names the harness and the library
    /// as the reader's `build_id` spells them.
    #[test]
    fn the_three_sets_are_rows_of_the_table() {
        assert_eq!(crate::arch::families(ARCH).len(), 3);
        for f in FAMILIES {
            assert!(std::ptr::eq(
                crate::arch::named(f.name).unwrap_or(&V2LITE),
                *f
            ));
            assert_eq!(f.recipe, "just dump-ref-dequant");
            assert!(f.consumers.contains(&"gate-1-1"));
        }
        let (binary, library) = md5s(BUILD);
        assert_eq!(BUILD, build_id(binary, library));
        for md5 in [binary, library] {
            check_hex("pin", md5, &"pin").unwrap_or_else(|e| panic!("{e}"));
        }
        assert_eq!(
            V2LITE.path(V2LITE.sets[0]),
            crate::data_dir().join("ref-dequant-v2lite")
        );
        assert_eq!(
            V41.path(V41.sets[0]),
            crate::data_dir().join(gguf::v41::set("ref-dequant-v41"))
        );
        assert_eq!(SYNTH.path("ref-synth"), crate::data_dir().join("ref-synth"));
    }

    /// Each family takes a set of its own model with the pinned build, and
    /// refuses the set of another's model and a build that is not the pin.
    #[test]
    fn each_family_takes_its_own_model_and_build_only() -> Result<(), RefError> {
        let runs: Vec<String> = FAMILIES
            .iter()
            .map(|f| f.runs())
            .collect::<Result<_, _>>()?;
        assert_eq!(runs[0], V2LITE_MODEL);
        for (f, own) in FAMILIES.iter().zip(&runs) {
            check(f, &format!("{}-own", f.name), own, BUILD)?;
            let other = BUILD.replace(md5s(BUILD).1, "0128e83c5e71e5f2b72f3d5d2c2de6b2");
            let Some(Build::Is(_)) = f.build else {
                panic!("{} pins no build", f.name);
            };
            match check(f, &format!("{}-lib", f.name), own, &other) {
                Err(RefError::Foreign { field: "build", .. }) => {}
                r => panic!("{}: a set of another library: {r:?}", f.name),
            }
            for model in runs.iter().filter(|m| *m != own) {
                match check(f, &format!("{}-model", f.name), model, BUILD) {
                    Err(RefError::Stale {
                        dumped_from, runs, ..
                    }) => {
                        assert_eq!((&dumped_from, &runs), (model, own));
                    }
                    r => panic!("{} given {model}: {r:?}", f.name),
                }
            }
        }
        Ok(())
    }
}
