//! The build of the Metal backend: the kernels' one source, and the
//! metallib when the Metal Toolchain is installed.
//!
//! Two outputs land in `OUT_DIR`:
//!
//! - `bloomery.metal`: the kernels' source with the fault site's table
//!   written in front of it, from `bloomery-device`'s table — the one owner
//!   of the codes. It is embedded and compiled by the runtime when no
//!   metallib exists.
//! - `bloomery.metallib`: the same source compiled by `xcrun metal`, when
//!   the Metal Toolchain is installed. Its presence is the `metal_lib`
//!   cfg, and the runtime loads it instead of the source.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The mat-vec launch geometry, from the file the host module reads it from:
/// the one owner of the numbers both sides use.
#[allow(
    dead_code,
    reason = "the host module reads every constant; the header takes two of them"
)]
mod geometry {
    include!("src/kernels/geometry.rs");
}

/// The fault site's table, written as the Metal compiler sees it.
fn fault_sites_header() -> String {
    let mut out = String::from(
        "// The fault site's codes, from the device contract's table\n\
         // (bloomery-device). The contract owns the codes; this is their copy.\n",
    );
    for site in bloomery_device::fault::FaultSite::ALL {
        out.push_str(&format!(
            "#define BLOOMERY_FAULT_SITE_{} {}\n",
            site.name().to_uppercase(),
            *site as u32
        ));
    }
    out.push_str(&format!(
        "// The fault word's shape, from the same table.\n\
         #define BLOOMERY_LAYER_NONE {}u\n\
         #define BLOOMERY_FAULT_WORDS {}u\n\
         // The word a raise stores: the layer above the site.\n\
         #define BLOOMERY_FAULT_WORD(layer, site) (((layer) << {}) | (site))\n",
        bloomery_device::fault::LAYER_NONE,
        bloomery_device::fault::FAULT_WORDS,
        bloomery_device::fault::LAYER_SHIFT
    ));
    out
}

/// The mat-vec launch geometry, written as the Metal compiler sees it.
fn matvec_geometry_header() -> String {
    format!(
        "// The mat-vec launch geometry, from src/kernels/geometry.rs.\n\
         #define BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP {}\n\
         #define BLOOMERY_MATVEC_MAX_SEGMENTS {}\n\
         #define BLOOMERY_MATVEC_VALUE_HEADS {}\n\
         #define BLOOMERY_MATVEC_HEAD_COLS {}\n\
         #define BLOOMERY_MATVEC_PARTIALS_PER_HEAD {}\n",
        geometry::ROWS_PER_SIMD_GROUP,
        geometry::MAX_SEGMENTS,
        geometry::VALUE_HEADS,
        geometry::HEAD_COLS,
        geometry::PARTIALS_PER_HEAD
    )
}

/// The kernels' source: the table, then every kernel file in order.
fn combined_source(header: &str, kernels: &[(&str, String)]) -> String {
    let mut out = String::from(header);
    for (name, source) in kernels {
        out.push_str(&format!("\n// {name}\n{source}"));
    }
    out
}

/// Whether the Metal Toolchain is installed: `xcrun metal` answers.
fn metal_toolchain() -> bool {
    Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "--version"])
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

/// Run `xcrun` with `args`, ending the build by name when it fails: the
/// Metal Toolchain answered, so a failure is the kernels' (or a flag's), not
/// an absent tool.
fn xcrun(what: &str, args: &[&str]) {
    let out = Command::new("xcrun")
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("{what}: xcrun did not start: {e}"));
    assert!(
        out.status.success(),
        "{what} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Compile `source` into a metallib under `out_dir`, with the math options
/// the runtime path sets in `library.rs`; `None` when the Metal Toolchain is
/// not installed. A toolchain that answers and then fails ends the build.
fn build_metallib(source: &Path, out_dir: &Path) -> Option<PathBuf> {
    if !metal_toolchain() {
        return None;
    }
    let air = out_dir.join("bloomery.air");
    let lib = out_dir.join("bloomery.metallib");
    let path = |p: &Path| p.to_str().expect("a build path is UTF-8").to_string();
    xcrun(
        "the Metal compiler",
        &[
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.2",
            "-fmetal-math-mode=safe",
            "-fmetal-math-fp32-functions=precise",
            "-c",
            &path(source),
            "-o",
            &path(&air),
        ],
    );
    xcrun(
        "the metallib linker",
        &["-sdk", "macosx", "metallib", &path(&air), "-o", &path(&lib)],
    );
    Some(lib)
}

fn main() {
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-changed=src/kernels/geometry.rs");
    println!("cargo:rerun-if-changed=build.rs");
    // Installing the Metal Toolchain (or pointing at another one) changes
    // what `xcrun` answers: the build runs again.
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rustc-check-cfg=cfg(metal_lib, values(\"metallib\", \"source\"))");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets it"));
    let kernels_dir = manifest.join("kernels");

    let mut kernels = Vec::new();
    // The order is the dependency order: norm defines the argument block the
    // mat-vecs read, matvec_q4k the block layout and macros the Q6_K and embed
    // files use.
    for name in [
        "fault",
        "copy",
        "probe",
        "norm",
        "matvec_q4k",
        "matvec_q6k",
        "embed",
        "delta", // K4-REGISTER
    ] {
        let path = kernels_dir.join(format!("{name}.metal"));
        let source =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        kernels.push((name, source));
    }

    let header = fault_sites_header() + &matvec_geometry_header();
    let combined = combined_source(&header, &kernels);
    let source_path = out_dir.join("bloomery.metal");
    std::fs::write(&source_path, &combined).expect("the source is written");

    match build_metallib(&source_path, &out_dir) {
        Some(_) => println!("cargo:rustc-cfg=metal_lib=\"metallib\""),
        None => println!("cargo:rustc-cfg=metal_lib=\"source\""),
    }
}
