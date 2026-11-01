// The kernels' program text: a header in front of the kernels' files. The
// header is the fault site's table and one geometry's numbers as `#define`s,
// so a launch and the kernel it dispatches read one struct (`Geometry`).
// `build.rs` writes it for every registered instance (the metallib's source
// and the embedded sources) and the runtime writes it at open (the source
// path), both through `program`. This file is `include!`d by `build.rs`, so
// it holds no inner doc comment.

use crate::attention_geom;
use crate::delta_geom;
use crate::geom::Geometry;
use crate::geometry;
use crate::router_geom;

/// The fault site's table, written as the Metal compiler sees it.
#[must_use]
pub fn fault_sites() -> String {
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

/// The launch geometry of `g`, written as the Metal compiler sees it. Every
/// instance-derived number is a field of `g`, the struct the host's launches
/// read; the instance-free ones are the geometry files' constants.
#[must_use]
pub fn geometry_defines(g: &Geometry) -> String {
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
         #define BLOOMERY_ATTN_HEAD_GATE {aheadgate}\n\
         #define BLOOMERY_ATTN_QK_NORM {aqknorm}\n\
         #define BLOOMERY_ATTN_PAIRS_NORM {apairsnorm}\n\
         #define BLOOMERY_ATTN_WINDOW {awindow}\n\
         #define BLOOMERY_ATTN_SEGMENT_CAP {asegcap}\n\
         #define BLOOMERY_ATTN_ONE_SG {aonesg}\n\
         #define BLOOMERY_ATTN_ONE_CHUNK {aonechunk}\n\
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
        aheadgate = u32::from(g.attn.head_gate),
        aqknorm = u32::from(g.attn.qk_norm),
        apairsnorm = u32::from(g.attn.norm_pairs),
        awindow = u32::from(g.attn.windowed),
        asegcap = attention_geom::SEGMENT_CAP,
        aonesg = attention_geom::ONE_PASS_SG,
        aonechunk = attention_geom::ONE_PASS_CHUNK,
        apfpos = attention_geom::PREFILL_POSITIONS,
        apfkeys = attention_geom::PREFILL_KEYS,
        apfpad = attention_geom::PREFILL_PAD,
    )
}

/// The whole header of `g`: the fault site's table, then its geometry.
#[must_use]
pub fn render(g: &Geometry) -> String {
    let mut text = fault_sites();
    text.push_str(&geometry_defines(g));
    text
}

/// The kernels' files joined: each under its name, in the order given (the
/// dependency order).
#[must_use]
pub fn files(kernels: &[(&str, String)]) -> String {
    let mut out = String::new();
    for (name, source) in kernels {
        out.push_str(&format!("\n// {name}\n{source}"));
    }
    out
}

/// The program text of `g`: its header in front of `body`, the kernels' files
/// as [`files`] joins them.
#[must_use]
pub fn program(g: &Geometry, body: &str) -> String {
    let mut out = render(g);
    out.push_str(body);
    out
}

/// The value of `#define NAME <n>` in `text`, when one such line names a
/// whole number: the geometry the source's kernels were compiled with.
#[must_use]
pub fn define_of(text: &str, name: &str) -> Option<usize> {
    let mut found = text.lines().filter_map(|line| {
        line.trim()
            .strip_prefix("#define ")?
            .strip_prefix(name)?
            .strip_prefix(' ')
            .and_then(|n| n.trim().parse().ok())
    });
    let n = found.next()?;
    found.next().is_none().then_some(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::REGISTERED;

    /// Every geometry `#define` of the header, with the value its instance's
    /// geometry (or the geometry files' constant) gives it: the header is
    /// the geometry, define by define, and writes no define this table does
    /// not name.
    fn expected(g: &Geometry) -> Vec<(&'static str, usize)> {
        vec![
            (
                "BLOOMERY_MATVEC_ROWS_PER_SIMD_GROUP",
                geometry::ROWS_PER_SIMD_GROUP,
            ),
            ("BLOOMERY_MATVEC_MAX_SEGMENTS", geometry::MAX_SEGMENTS),
            ("BLOOMERY_DELTA_VALUE_HEADS", g.delta.value_heads),
            ("BLOOMERY_DELTA_HEAD_COLS", g.delta.head_cols),
            ("BLOOMERY_DELTA_COLUMN_BLOCKS", geometry::PARTIALS_PER_HEAD),
            (
                "BLOOMERY_DELTA_COLS_PER_BLOCK",
                g.delta.head_cols / geometry::PARTIALS_PER_HEAD,
            ),
            ("BLOOMERY_DELTA_RING_ROWS", g.delta.ring_rows),
            ("BLOOMERY_DELTA_SNAPS", delta_geom::SNAPS),
            ("BLOOMERY_ROUTER_ROWS", g.router.rows),
            ("BLOOMERY_ROUTER_EXPERTS", g.router.experts),
            ("BLOOMERY_ROUTER_K", g.router.k),
            ("BLOOMERY_ROUTER_SPLIT", router_geom::SPLIT),
            ("BLOOMERY_ROUTER_SG", router_geom::SIMD_GROUPS),
            ("BLOOMERY_ROUTER_FLOAT4S", g.router.float4s),
            ("BLOOMERY_ROUTER_TOPK_MAX", g.router.top_k_max),
            ("BLOOMERY_ROUTER_LANES", router_geom::SIMD_WIDTH),
            ("BLOOMERY_ROUTER_SHARED", g.router.shared),
            ("BLOOMERY_EXPERT_SLOTS", g.experts.slots),
            ("BLOOMERY_EXPERT_ROUTED", g.experts.routed_ids),
            ("BLOOMERY_EXPERT_SHARED", g.experts.shared_slot),
            ("BLOOMERY_EXPERT_K", g.experts.k),
            ("BLOOMERY_ATTN_HEAD", g.attn.head),
            ("BLOOMERY_ATTN_ROT", g.attn.rot),
            ("BLOOMERY_ATTN_Q", g.attn.query_heads),
            ("BLOOMERY_ATTN_KV", g.attn.kv_heads),
            ("BLOOMERY_ATTN_TILE", attention_geom::TILE),
            ("BLOOMERY_ATTN_SG", g.attn.simd_groups),
            ("BLOOMERY_ATTN_HPT", attention_geom::HEADS_PER_SG),
            ("BLOOMERY_ATTN_GQA", g.attn.gqa),
            ("BLOOMERY_ATTN_PARTIAL", g.attn.partial_elems),
            ("BLOOMERY_ATTN_QG", g.attn.qg_stride),
            ("BLOOMERY_ATTN_LANES", attention_geom::SIMD_WIDTH),
            ("BLOOMERY_ATTN_THREADS", g.attn.flash_threads),
            ("BLOOMERY_ATTN_LANE_DIMS", g.attn.lane_dims),
            ("BLOOMERY_ATTN_GATED", usize::from(g.attn.gated)),
            ("BLOOMERY_ATTN_HEAD_GATE", usize::from(g.attn.head_gate)),
            ("BLOOMERY_ATTN_QK_NORM", usize::from(g.attn.qk_norm)),
            ("BLOOMERY_ATTN_PAIRS_NORM", usize::from(g.attn.norm_pairs)),
            ("BLOOMERY_ATTN_WINDOW", usize::from(g.attn.windowed)),
            ("BLOOMERY_ATTN_SEGMENT_CAP", attention_geom::SEGMENT_CAP),
            ("BLOOMERY_ATTN_ONE_SG", attention_geom::ONE_PASS_SG),
            ("BLOOMERY_ATTN_ONE_CHUNK", attention_geom::ONE_PASS_CHUNK),
            (
                "BLOOMERY_ATTN_PF_POSITIONS",
                attention_geom::PREFILL_POSITIONS,
            ),
            ("BLOOMERY_ATTN_PF_KEYS", attention_geom::PREFILL_KEYS),
            ("BLOOMERY_ATTN_PF_PAD", attention_geom::PREFILL_PAD),
        ]
    }

    /// The names the header defines for the geometry: every `#define` but the
    /// fault site's.
    fn defined(text: &str) -> Vec<&str> {
        text.lines()
            .filter_map(|line| line.strip_prefix("#define BLOOMERY_"))
            .filter(|rest| !rest.starts_with("FAULT_") && !rest.starts_with("LAYER_NONE"))
            .filter_map(|rest| rest.split([' ', '(']).next())
            .collect()
    }

    #[test]
    fn the_header_is_the_geometry() {
        for inst in REGISTERED {
            let g = Geometry::of(inst);
            let header = render(&g);
            let want = expected(&g);
            for (name, value) in &want {
                assert_eq!(
                    define_of(&header, name),
                    Some(*value),
                    "{}: {name} in the header",
                    inst.name
                );
            }
            let mut written = defined(&header);
            written.sort_unstable();
            let mut named: Vec<_> = want
                .iter()
                .map(|(name, _)| name.trim_start_matches("BLOOMERY_"))
                .collect();
            named.sort_unstable();
            assert_eq!(
                written, named,
                "{}: the defines the header writes",
                inst.name
            );
        }
    }

    #[test]
    fn a_define_names_one_whole_number() {
        let text = "#define BLOOMERY_A 7\n#define BLOOMERY_B 7u\n#define BLOOMERY_C 1\n\
                    #define BLOOMERY_C 2\n  #define BLOOMERY_D  9  \n";
        assert_eq!(define_of(text, "BLOOMERY_A"), Some(7));
        assert_eq!(define_of(text, "BLOOMERY_B"), None, "a suffixed number");
        assert_eq!(define_of(text, "BLOOMERY_C"), None, "a name defined twice");
        assert_eq!(define_of(text, "BLOOMERY_D"), Some(9), "padded");
        assert_eq!(define_of(text, "BLOOMERY_E"), None, "a name not defined");
    }

    #[test]
    fn a_program_is_the_header_then_the_files() {
        let g = Geometry::of(&REGISTERED[0]);
        let body = files(&[("a", "int a;\n".to_string()), ("b", "int b;\n".to_string())]);
        assert_eq!(body, "\n// a\nint a;\n\n// b\nint b;\n");
        let text = program(&g, &body);
        assert_eq!(text, format!("{}{body}", render(&g)));
        assert!(text.starts_with("// The fault site's codes"));
    }
}
