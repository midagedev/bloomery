//! The build of the Metal backend: the kernels' one source, and the
//! metallib when the Metal Toolchain is installed.
//!
//! Three outputs land in `OUT_DIR`:
//!
//! - `bloomery.metal`: the kernels' source with the fault site's table
//!   written in front of it, from `bloomery-device`'s table — the one owner
//!   of the codes. It is embedded and compiled by the runtime when no
//!   metallib exists.
//! - `bloomery.metallib`: the same source compiled by `xcrun metal`, when
//!   the Metal Toolchain is installed. Its presence is the `metal_lib`
//!   cfg, and the runtime loads it instead of the source.
//! - `bloomery_tensor.metal`: the tensor-op GEMMs' source, the same table in
//!   front of the files it reads, which the runtime compiles at Metal 4 on a
//!   GPU that takes them (`library.rs`).

use std::path::{Path, PathBuf};
use std::process::Command;

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

/// The launch geometry, written as the Metal compiler sees it. Each number
/// has one owner: the Rust constant the host module includes.
fn geometry_header() -> String {
    format!(
        "// The mat-vec launch geometry, from src/kernels/geometry.rs.\n\
         #define BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP {rows}\n\
         #define BLOOMERY_MATVEC_MAX_SEGMENTS {segs}\n\
         // The gated delta net's head geometry. The numbers live with the\n\
         // mat-vec geometry; the delta step and the head-normed mat-vec read them.\n\
         #define BLOOMERY_DELTA_VALUE_HEADS {heads}\n\
         #define BLOOMERY_DELTA_HEAD_COLS {cols}\n\
         #define BLOOMERY_DELTA_COLUMN_BLOCKS {blocks}\n\
         #define BLOOMERY_DELTA_COLS_PER_BLOCK {per}\n\
         // The conv ring, from src/kernels/delta_geom.rs.\n\
         #define BLOOMERY_DELTA_RING_ROWS {ring}\n\
         #define BLOOMERY_DELTA_SNAPS {snaps}\n\
         // The router, from src/kernels/router_geom.rs.\n\
         #define BLOOMERY_ROUTER_ROWS {rrows}\n\
         #define BLOOMERY_ROUTER_EXPERTS {rexperts}\n\
         #define BLOOMERY_ROUTER_K {rk}\n\
         #define BLOOMERY_ROUTER_SPLIT {rsplit}\n\
         #define BLOOMERY_ROUTER_SG {rsg}\n\
         #define BLOOMERY_ROUTER_FLOAT4S {rfloat4}\n\
         #define BLOOMERY_ROUTER_TOPK_MAX {rtop}\n\
         #define BLOOMERY_ROUTER_LANES {rlanes}\n\
         // The expert pair and the folded down, from src/kernels/experts_geom.rs.\n\
         #define BLOOMERY_EXPERT_SLOTS {eslots}\n\
         #define BLOOMERY_EXPERT_ROUTED {erouted}\n\
         #define BLOOMERY_EXPERT_SHARED {eshared}\n\
         #define BLOOMERY_EXPERT_K {ek}\n\
         // Decode attention, from src/kernels/attention_geom.rs.\n\
         #define BLOOMERY_ATTN_HEAD {ahead}\n\
         #define BLOOMERY_ATTN_ROT {arot}\n\
         #define BLOOMERY_ATTN_Q {aq}\n\
         #define BLOOMERY_ATTN_KV {akv}\n\
         #define BLOOMERY_ATTN_TILE {atile}\n\
         #define BLOOMERY_ATTN_SG {asg}\n\
         #define BLOOMERY_ATTN_HPT {ahpt}\n\
         #define BLOOMERY_ATTN_GQA {agqa}\n\
         #define BLOOMERY_ATTN_PARTIAL {apartial}\n\
         #define BLOOMERY_ATTN_QG {aqg}\n\
         #define BLOOMERY_ATTN_LANES {alanes}\n\
         #define BLOOMERY_ATTN_THREADS {athreads}\n\
         #define BLOOMERY_ATTN_PF_POSITIONS {apfpos}\n\
         #define BLOOMERY_ATTN_PF_KEYS {apfkeys}\n\
         #define BLOOMERY_ATTN_PF_PAD {apfpad}\n",
        rows = geometry::ROWS_PER_SIMD_GROUP,
        segs = geometry::MAX_SEGMENTS,
        heads = geometry::VALUE_HEADS,
        cols = geometry::HEAD_COLS,
        blocks = geometry::PARTIALS_PER_HEAD,
        per = geometry::HEAD_COLS / geometry::PARTIALS_PER_HEAD,
        ring = delta_geom::RING_ROWS,
        snaps = delta_geom::SNAPS,
        rrows = router_geom::ROWS,
        rexperts = router_geom::EXPERTS,
        rk = router_geom::K,
        rsplit = router_geom::SPLIT,
        rsg = router_geom::SIMD_GROUPS,
        rfloat4 = router_geom::FLOAT4S,
        rtop = router_geom::TOP_K_MAX,
        rlanes = router_geom::SIMD_WIDTH,
        eslots = experts_geom::EXPERT_SLOTS,
        erouted = experts_geom::ROUTED_IDS,
        eshared = experts_geom::SHARED_SLOT,
        ek = experts_geom::EXPERT_K,
        ahead = attention_geom::HEAD_DIM,
        arot = attention_geom::ROT,
        aq = attention_geom::QUERY_HEADS,
        akv = attention_geom::KV_HEADS,
        atile = attention_geom::TILE,
        asg = attention_geom::SIMD_GROUPS,
        ahpt = attention_geom::HEADS_PER_SG,
        agqa = attention_geom::GQA,
        apartial = attention_geom::PARTIAL_ELEMS,
        aqg = attention_geom::QG_STRIDE,
        alanes = attention_geom::SIMD_WIDTH,
        athreads = attention_geom::FLASH_THREADS,
        apfpos = attention_geom::PREFILL_POSITIONS,
        apfkeys = attention_geom::PREFILL_KEYS,
        apfpad = attention_geom::PREFILL_PAD,
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
    println!("cargo:rerun-if-changed=src/kernels/delta_geom.rs");
    println!("cargo:rerun-if-changed=src/kernels/router_geom.rs");
    println!("cargo:rerun-if-changed=src/kernels/experts_geom.rs");
    println!("cargo:rerun-if-changed=src/kernels/attention_geom.rs");
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
    // entry templates the Q6_K, A4, experts and head files instantiate.
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
        "embed",
        // Ticket's helpers are what the router and the head call, so ticket
        // comes before either kernel.
        "ticket",
        "router",
        "experts",
        "route",
        "gemm_routed",
        "delta",
        "attention",
        "head",
    ] {
        let path = kernels_dir.join(format!("{name}.metal"));
        let source =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        kernels.push((name, source));
    }

    let header = fault_sites_header() + &geometry_header();
    let combined = combined_source(&header, &kernels);
    let source_path = out_dir.join("bloomery.metal");
    std::fs::write(&source_path, &combined).expect("the source is written");

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
    std::fs::write(
        out_dir.join("bloomery_tensor.metal"),
        combined_source(&header, &tensor),
    )
    .expect("the tensor source is written");

    match build_metallib(&source_path, &out_dir) {
        Some(_) => println!("cargo:rustc-cfg=metal_lib=\"metallib\""),
        None => println!("cargo:rustc-cfg=metal_lib=\"source\""),
    }
}
