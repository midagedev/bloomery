//! The build of the Metal backend: the kernels' source of every registered
//! instance, and its metallib when the Metal Toolchain is installed.
//!
//! An instance (`src/instance.rs`, a row of `REGISTERED`) is the axes one
//! translation unit is compiled at. For each, these land in `OUT_DIR`:
//!
//! - `<name>.metal`: the kernels' source with the fault site's table, from
//!   `bloomery-device`'s table (the one owner of the codes), and the
//!   instance's geometry written in front of it. It is embedded and compiled
//!   by the runtime when no metallib exists.
//! - `<name>.metallib`: the same source compiled by `xcrun metal`, when the
//!   Metal Toolchain is installed. Its presence is the `metal_lib` cfg, and
//!   the runtime loads it instead of the source.
//! - `<name>_tensor.metal`: the tensor-op GEMMs' source, the same header in
//!   front of the files it reads, which the runtime compiles at Metal 4 on a
//!   GPU that takes them (`library.rs`).
//!
//! And the two tables `library.rs` includes as expressions: `built.rs`, every
//! instance's name and embedded sources, and `metallibs.rs`, each instance's
//! metallib when the toolchain is there.

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

/// The launch geometry of `inst`, written as the Metal compiler sees it.
/// Every instance-derived number is a field of `inst`'s [`geom::Geometry`],
/// the struct the host's launches read; the instance-free ones are the
/// geometry files' constants.
fn geometry_header(inst: &'static instance::Instance) -> String {
    let g = geom::Geometry::of(inst);
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
         #define BLOOMERY_ROUTER_SHARED {rshared}\n\
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
         #define BLOOMERY_ATTN_LANE_DIMS {alanedims}\n\
         #define BLOOMERY_ATTN_GATED {agated}\n\
         #define BLOOMERY_ATTN_SEGMENT_CAP {asegcap}\n\
         #define BLOOMERY_ATTN_PF_POSITIONS {apfpos}\n\
         #define BLOOMERY_ATTN_PF_KEYS {apfkeys}\n\
         #define BLOOMERY_ATTN_PF_PAD {apfpad}\n",
        rows = geometry::ROWS_PER_SIMD_GROUP,
        segs = geometry::MAX_SEGMENTS,
        heads = g.delta.value_heads,
        cols = g.delta.head_cols,
        blocks = geometry::PARTIALS_PER_HEAD,
        per = g.delta.head_cols / geometry::PARTIALS_PER_HEAD,
        ring = g.delta.ring_rows,
        snaps = delta_geom::SNAPS,
        rrows = g.router.rows,
        rexperts = g.router.experts,
        rk = g.router.k,
        rsplit = router_geom::SPLIT,
        rsg = router_geom::SIMD_GROUPS,
        rfloat4 = g.router.float4s,
        rtop = g.router.top_k_max,
        rlanes = router_geom::SIMD_WIDTH,
        rshared = g.router.shared,
        eslots = g.experts.slots,
        erouted = g.experts.routed_ids,
        eshared = g.experts.shared_slot,
        ek = g.experts.k,
        ahead = g.attn.head,
        arot = g.attn.rot,
        aq = g.attn.query_heads,
        akv = g.attn.kv_heads,
        atile = attention_geom::TILE,
        asg = g.attn.simd_groups,
        ahpt = attention_geom::HEADS_PER_SG,
        agqa = g.attn.gqa,
        apartial = g.attn.partial_elems,
        aqg = g.attn.qg_stride,
        alanes = attention_geom::SIMD_WIDTH,
        athreads = g.attn.flash_threads,
        alanedims = g.attn.lane_dims,
        agated = u32::from(g.attn.gated),
        asegcap = attention_geom::SEGMENT_CAP,
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
    // entry templates the Q6_K, A4, Q8_0, Q5_K, F32, experts and head files instantiate.
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
        let header = fault_sites_header() + &geometry_header(inst);
        let source_path = out_dir.join(format!("{stem}.metal"));
        std::fs::write(&source_path, combined_source(&header, &kernels))
            .expect("the source is written");
        let tensor_path = out_dir.join(format!("{stem}_tensor.metal"));
        std::fs::write(&tensor_path, combined_source(&header, &tensor))
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
