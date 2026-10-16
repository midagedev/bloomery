//! nemotron_h_moe host expert gate: what the host tier serves of the Bolt file
//! (`arch::nemotron_h_moe::host`), on the file's own stacks.
//!
//! - Every layer of the file through `host::layer`: a MoE layer builds as the
//!   relu² form at its own expert count (the router's rows) and width (the
//!   layer's `expert_feed_forward_length`), in the latent the header gives; a
//!   Mamba-2 or attention layer is `None`; a layer past the model is refused
//!   by name.
//! - The narrowest and the widest MoE layer of the file, through one union
//!   scratch made for the widest (the widths differ from layer to layer), at
//!   one column and at nine: nine distinct experts a column (Bolt's top 9),
//!   every cell finite and the union call equal to `experts_into` of each
//!   column, bit for bit, under both deferral arms.
//!
//! The kernels' own values are pinned elsewhere (`gate-qdot` for IQ2_XXS, the
//! union gate's relu² arm for the form against a transcription on qdot's
//! calls); this gate holds the wiring: the names, the latent, the per-layer
//! shapes and the capacity.
//!
//! `hw_`: needs the 24 GB file on the box at [`FILE`] (`just
//! gate-nemotron-h-moe-host`). Reads the stacks of the experts the columns
//! list, in two layers: seconds.

use gguf::Split;
use model::arch::nemotron_h_moe::host;
use model::arch::nemotron_h_moe::hparams::{Experts, Hparams, Kind};
use model::moe::{HostLayer, HostScratch, UnionScratch};
use model::ops::{self, ExpertForm, Tensor2};
use model::r8file::R8Source;

/// The box's path of the file.
const FILE: &str = "/models/nemotron-3.5-bolt/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf";

/// Bolt's routed list: nine experts a token.
const USED: usize = 9;

/// The column counts: one, and Bolt's nine.
const KS: [usize; 2] = [1, 9];

/// `n` seeded values in [-1, 1), every 61st scaled by 8 so a block's scale is
/// set by an outlier the way real activations set it.
fn seeded(n: usize, key: u64) -> Vec<f32> {
    let mut s = key.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|i| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let u = ((s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / 8_388_608.0 - 1.0;
            if i % 61 == 0 { u * 8.0 } else { u }
        })
        .collect()
}

/// Column `j`'s list: [`USED`] distinct experts of the layer's `n_expert`
/// (every count of the file is coprime to 11, so the stride visits distinct
/// ones), weights in [0.05, 0.3).
fn list(j: usize, n_expert: usize) -> Vec<(u32, f32)> {
    (0..USED)
        .map(|i| {
            let e = ((j * 37 + i * 11) % n_expert) as u32;
            (e, 0.05 + 0.025 * ((i + j) % 10) as f32)
        })
        .collect()
}

/// Cells of `got` whose bits differ from `want`, a length mismatch counting
/// every cell.
fn diff_cells(got: &[f32], want: &[f32]) -> usize {
    if got.len() != want.len() {
        return got.len().max(want.len());
    }
    got.iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

#[test]
#[ignore = "hw: needs the Bolt 24GB file on the box (just gate-nemotron-h-moe-host)"]
fn hw_nemotron_h_moe_host() {
    let split = Split::open(FILE).unwrap_or_else(|e| panic!("open {FILE}: {e}"));
    let hp = Hparams::read(&split).unwrap_or_else(|e| panic!("hparams: {e}"));
    let src = R8Source::rows(&split);
    let mut failed: Vec<String> = Vec::new();
    let mut check = |what: String, ok: bool| {
        println!("{what}: {}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            failed.push(what);
        }
    };

    // Every layer through the arch's one wiring.
    let mut layers: Vec<(usize, Experts, HostLayer)> = Vec::new();
    let mut hosted_none = 0usize;
    for l in 0..hp.n_layer {
        let built = host::layer(src, &hp, l)
            .unwrap_or_else(|e| panic!("layer {l} ({:?}): {e}", hp.kinds[l]));
        match (hp.kinds[l], hp.experts[l], built) {
            (Kind::Moe, Some(e), Some(layer)) => {
                check(
                    format!(
                        "layer {l} builds as relu² at {} experts of {}",
                        e.n_expert, e.ff
                    ),
                    layer.form() == ExpertForm::Relu2 && layer.n_expert() == e.n_expert,
                );
                layers.push((l, e, layer));
            }
            (Kind::Mamba | Kind::Attention, None, None) => hosted_none += 1,
            (kind, _, built) => check(
                format!("layer {l} ({kind:?}) is hosted as {}", built.is_some()),
                false,
            ),
        }
    }
    check(
        format!(
            "{} MoE layers built, {hosted_none} others hosted as none, of {} layers",
            layers.len(),
            hp.n_layer
        ),
        layers.len() + hosted_none == hp.n_layer && !layers.is_empty(),
    );
    let past = host::layer(src, &hp, hp.n_layer)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    check(
        format!("a layer past the model is refused by name: {past:?}"),
        past.contains("a layer inside the model's"),
    );

    // The narrowest and the widest layer through one scratch for the widest.
    let by_width = |pick: fn(Experts, Experts) -> bool| {
        layers
            .iter()
            .skip(1)
            .fold(
                &layers[0],
                |best, cur| {
                    if pick(cur.1, best.1) { cur } else { best }
                },
            )
            .0
    };
    let narrow = by_width(|a, b| a.ff < b.ff);
    let wide = by_width(|a, b| a.ff > b.ff);
    let ff_cap = layers.iter().map(|(_, e, _)| e.ff).max().unwrap();
    let mut us = UnionScratch::new_routed(hp.latent, ff_cap, USED, USED)
        .expect("a scratch for the widest layer");
    for l in [narrow, wide] {
        let (_, e, layer) = layers.iter().find(|(i, _, _)| *i == l).unwrap();
        let mut host = HostScratch::new(hp.latent, e.ff, USED).expect("a scratch of nine experts");
        for k in KS {
            let x = Tensor2::from_vec(
                hp.latent,
                k,
                (0..k)
                    .flat_map(|j| seeded(hp.latent, 0xb017 + (l * 16 + j) as u64))
                    .collect(),
            );
            let lists: Vec<Vec<(u32, f32)>> = (0..k).map(|j| list(j, e.n_expert)).collect();
            let slices: Vec<&[(u32, f32)]> = lists.iter().map(Vec::as_slice).collect();
            let mut want = vec![f32::NAN; hp.latent * k];
            for j in 0..k {
                let xj = Tensor2::from_vec(hp.latent, 1, x.col(j).to_vec());
                layer
                    .experts_into(
                        src,
                        &xj,
                        &lists[j],
                        &mut want[j * hp.latent..(j + 1) * hp.latent],
                        &mut host,
                    )
                    .unwrap_or_else(|err| panic!("layer {l} experts_into column {j}: {err}"));
            }
            check(
                format!("layer {l} k={k}: experts_into gives finite values"),
                want.iter().all(|v| v.is_finite()),
            );
            for (arm, on) in [("defer", true), ("prepass", false)] {
                ops::set_defer_quant(Some(on));
                let mut got = vec![f32::NAN; hp.latent * k];
                let r = layer.experts_union_into(src, &x, &slices, &mut got, &mut us);
                ops::set_defer_quant(None);
                r.unwrap_or_else(|err| panic!("layer {l} k {k} {arm}: {err}"));
                let d = diff_cells(&got, &want);
                check(
                    format!(
                        "layer {l} (ff {}, {} experts) k={k} {arm}: union equals experts_into, \
                         diff_cells={d}",
                        e.ff, e.n_expert
                    ),
                    d == 0,
                );
            }
        }
    }
    assert!(failed.is_empty(), "failed: {failed:#?}");
    println!(
        "PASSED: nemotron_h_moe host — {} MoE layers build as relu², the rest as none; layers {narrow} \
         and {wide} (the narrowest and widest ff) through one scratch for ff {ff_cap}: union equal to \
         experts_into at k = 1, 9 under both deferral arms",
        layers.len()
    );
}
