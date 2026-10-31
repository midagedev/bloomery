//! The build of the Metal backend: the kernels' source of every registered
//! instance, and its metallib when the Metal Toolchain is installed.
//!
//! An instance (`src/instance.rs`, a row of `REGISTERED`) is the axes one
//! translation unit is compiled at. The header in front of the kernels'
//! files (the fault site's table and the instance's geometry) is written by
//! `src/kernels/header.rs`, the file the runtime writes the same header with
//! at open. In `OUT_DIR` land the two bodies, the part no instance changes:
//!
//! - `main_body.metal`: the kernels' files joined.
//! - `tensor_body.metal`: the tensor-op GEMMs' files joined.
//!
//! And for each instance:
//!
//! - `<name>.metal`: the header, then `main_body.metal`. It is embedded and
//!   compiled by the runtime when no metallib exists.
//! - `<name>.metallib`: the same source compiled by `xcrun metal`, when the
//!   Metal Toolchain is installed. Its presence is the `metal_lib` cfg, and
//!   the runtime loads it instead of the source.
//! - `<name>_tensor.metal`: the header, then `tensor_body.metal`, which the
//!   runtime compiles at Metal 4 on a GPU that takes them (`library.rs`).
//!
//! And the two tables `library.rs` includes as expressions: `built.rs`, every
//! instance's name and embedded sources, and `metallibs.rs`, each instance's
//! metallib when the toolchain is there. A registered instance whose geometry
//! some kernel cannot take ends the build by name.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The registered instances, from the file the host module reads.
#[allow(
    dead_code,
    reason = "the host module selects a file's instance; the build writes every registered one"
)]
mod instance {
    include!("src/instance.rs");
}

/// The mat-vec launch geometry, from the file the host module reads it from.
#[allow(
    dead_code,
    reason = "the host module reads every constant; the header emits the ones the kernels read"
)]
mod geometry {
    include!("src/kernels/geometry.rs");
}

/// The gated delta step's geometry, from the file the host module reads.
#[allow(
    dead_code,
    reason = "the host module reads every constant; the header emits the ring length and the copies"
)]
mod delta_geom {
    include!("src/kernels/delta_geom.rs");
}

/// The router's geometry, from the file the host module reads.
#[allow(
    dead_code,
    reason = "the host module reads every constant; the header emits the ones the kernel reads"
)]
mod router_geom {
    include!("src/kernels/router_geom.rs");
}

/// The expert pair and fold geometry, from the file the host module reads.
#[allow(
    dead_code,
    reason = "the host module reads every constant; the header emits the ones the kernel reads"
)]
mod experts_geom {
    include!("src/kernels/experts_geom.rs");
}

/// Decode attention's geometry, from the file the host module reads.
#[allow(
    dead_code,
    reason = "the host module reads every constant; the header emits the ones the kernel reads"
)]
mod attention_geom {
    include!("src/kernels/attention_geom.rs");
}

/// The launch geometry of one instance, from the file the host module reads:
/// the header is written from the same struct the launches read.
#[allow(
    dead_code,
    reason = "the host module reads every field; the header emits the ones the kernels read"
)]
mod geom {
    include!("src/kernels/geom.rs");
}

/// The program text of one geometry, from the file the runtime writes it
/// with at open: the header in front of the kernels' files.
#[allow(
    dead_code,
    reason = "the host module reads the header's parts; the build writes the whole program"
)]
mod header {
    include!("src/kernels/header.rs");
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

/// A build path as text.
fn path_text(p: &Path) -> &str {
    p.to_str().expect("a build path is UTF-8")
}

/// Compile `source` into the metallib `<stem>.metallib` under `out_dir`, with
/// the math options the runtime path sets in `library.rs`. The Metal
/// Toolchain answered, so a failure ends the build.
fn build_metallib(source: &Path, out_dir: &Path, stem: &str) -> PathBuf {
    let air = out_dir.join(format!("{stem}.air"));
    let lib = out_dir.join(format!("{stem}.metallib"));
    let path = |p: &Path| path_text(p).to_string();
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
    lib
}

fn main() {
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-changed=src/kernels/geometry.rs");
    println!("cargo:rerun-if-changed=src/kernels/delta_geom.rs");
    println!("cargo:rerun-if-changed=src/kernels/router_geom.rs");
    println!("cargo:rerun-if-changed=src/kernels/experts_geom.rs");
    println!("cargo:rerun-if-changed=src/kernels/attention_geom.rs");
    println!("cargo:rerun-if-changed=src/kernels/geom.rs");
    println!("cargo:rerun-if-changed=src/kernels/header.rs");
    println!("cargo:rerun-if-changed=src/instance.rs");
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
    // mat-vecs read, quant the block layouts and the decode owners every
    // quantized kernel reads (embed's too), matvec_q4k the row loop and the
    // entry templates the Q6_K, A4, A8, A4H, A8H, Q8_0, Q5_K, F32, experts
    // and head files instantiate.
    for name in [
        "fault",
        "copy",
        "probe",
        "norm",
        "quant",
        "gemm_shared",
        "gemm",
        "matvec_q4k",
        "matvec_q6k",
        "matvec_a4",
        "matvec_a8",
        "matvec_a4h",
        "matvec_a8h",
        "matvec_q8_0",
        "matvec_q5k",
        "matvec_f32",
        "embed",
        // Ticket's helpers are what the router and the head call, so ticket
        // comes before either kernel.
        "ticket",
        "router",
        "experts",
        "route",
        "gemm_routed",
        "delta",
        "shortconv",
        "attention",
        "head",
    ] {
        let path = kernels_dir.join(format!("{name}.metal"));
        let source =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        kernels.push((name, source));
    }

    // The tensor-op GEMMs read the fault's raise, the quant owners and the
    // shared GEMM ABI, the engine source's own files, in its order.
    let tensor: Vec<(&str, String)> = kernels
        .iter()
        .filter(|(name, _)| ["fault", "quant", "gemm_shared"].contains(name))
        .cloned()
        .chain(std::iter::once((
            "gemm_tensor",
            std::fs::read_to_string(kernels_dir.join("gemm_tensor.metal"))
                .expect("kernels/gemm_tensor.metal is read"),
        )))
        .collect();

    let main_body = header::files(&kernels);
    let tensor_body = header::files(&tensor);
    std::fs::write(out_dir.join("main_body.metal"), &main_body)
        .expect("the kernels' body is written");
    std::fs::write(out_dir.join("tensor_body.metal"), &tensor_body)
        .expect("the tensor body is written");

    let toolchain = metal_toolchain();
    let mut built = String::from(
        "// Written by build.rs: every registered instance's embedded sources,\n\
         // in instance::REGISTERED's order.\n\
         &[\n",
    );
    let mut metallibs = String::from("&[\n");
    for inst in instance::REGISTERED {
        let stem = inst.name;
        assert!(
            !stem.is_empty()
                && stem
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
            "instance {stem:?}: a name is a file stem of lowercase letters, digits and _"
        );
        assert!(
            stem != instance::UNREGISTERED,
            "instance {stem:?}: the name is the one an instance made at open has"
        );
        let g = geom::Geometry::of(inst);
        if let Err(why) = g.check() {
            panic!("instance {stem:?}: {why}");
        }
        let source_path = out_dir.join(format!("{stem}.metal"));
        std::fs::write(&source_path, header::program(&g, &main_body))
            .expect("the source is written");
        let tensor_path = out_dir.join(format!("{stem}_tensor.metal"));
        std::fs::write(&tensor_path, header::program(&g, &tensor_body))
            .expect("the tensor source is written");
        built.push_str(&format!(
            "    Built {{\n        name: {stem:?},\n        source: include_str!({:?}),\n        \
             tensor: include_str!({:?}),\n    }},\n",
            path_text(&source_path),
            path_text(&tensor_path)
        ));
        if toolchain {
            let lib = build_metallib(&source_path, &out_dir, stem);
            metallibs.push_str(&format!(
                "    ({stem:?}, include_bytes!({:?})),\n",
                path_text(&lib)
            ));
        }
    }
    built.push_str("]\n");
    std::fs::write(out_dir.join("built.rs"), built).expect("the instance table is written");
    if toolchain {
        metallibs.push_str("]\n");
        std::fs::write(out_dir.join("metallibs.rs"), metallibs)
            .expect("the metallib table is written");
    }

    if toolchain {
        println!("cargo:rustc-cfg=metal_lib=\"metallib\"");
    } else {
        println!("cargo:rustc-cfg=metal_lib=\"source\"");
    }
}
