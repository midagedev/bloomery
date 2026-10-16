//! Import the pinned upstream arithmetic; this build script never edits the CUDA source.
use std::{env, fs, path::PathBuf};

const CORE_FNV1A: u64 = 0x886002db3d49d572;
const FUNCTIONS: &[&str] = &[
    "half_to_f32",
    "q4k_nibble",
    "q4k_a_chain",
    "q4k_scale_min",
    "q4k_sub_value",
    "q4k_acc",
    "q4k_sb_decode",
    "q4k_iter_term",
    "q4k_row_dot_1col",
];

fn function<'a>(source: &'a str, name: &str) -> &'a str {
    let needle = format!("fn {name}(");
    let start = source
        .find(&needle)
        .unwrap_or_else(|| panic!("HIP Q4 import: missing {name}"));
    assert!(
        !source[start + needle.len()..].contains(&needle),
        "HIP Q4 import: duplicate {name}"
    );
    let brace = start + source[start..].find('{').expect("function body");
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[brace..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=brace + offset];
                }
            }
            _ => {}
        }
    }
    panic!("HIP Q4 import: unclosed body {name}");
}

fn main() {
    let upstream = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .join("../../../crates/gpu/src/cores.rs");
    println!("cargo:rerun-if-changed={}", upstream.display());
    let source = fs::read_to_string(&upstream).expect("HIP Q4 import: upstream cores.rs");
    let hash = source.bytes().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    });
    assert_eq!(
        hash, CORE_FNV1A,
        "HIP Q4 import: unsupported upstream cores.rs drift; review arithmetic and update provenance"
    );
    assert!(
        source.contains("pub(crate) const Q4K_ITER_UNROLL: u32 = 2;"),
        "HIP Q4 import: unroll changed"
    );
    let mut generated = String::from(
        "// Generated from the pinned crates/gpu/src/cores.rs; inspect device/build.rs.\nconst Q4K_ITER_UNROLL: u32 = 2;\n",
    );
    for name in FUNCTIONS {
        if ["q4k_iter_term", "q4k_row_dot_1col"].contains(name) {
            generated.push_str("#[allow(clippy::too_many_arguments, reason = \"original device core: flat kernel arguments (rust-quality R8)\")]\n");
        }
        generated.push_str("#[inline(always)]\n");
        generated.push_str(function(&source, name));
        generated.push('\n');
    }
    let target = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("q4k_core.rs");
    fs::write(target, generated).expect("HIP Q4 import: write generated arithmetic");
}
