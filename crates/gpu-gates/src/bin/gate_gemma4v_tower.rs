//! The Gemma4 tower from pinned image bytes through every layer to projected rows.
//! Each numerical band is 2.5 times that row's CPU/card oracle distance.

#[path = "shared/emb2_image.rs"]
mod image;

#[allow(
    dead_code,
    reason = "the shared runner serves several gates; this one reads its distance helper"
)]
#[path = "shared/e2e.rs"]
mod e2e;

use bloomery_gpu::Gpu;
use bloomery_gpu_gates::{GateError, checks_failed, exit_with, ref_model_path, verdict};
use bloomery_gpu_vision::{bf16_f32, chain::TapSink, gemma4v::Encoder};
use gguf::Gguf;
use refset::arch::gemma_embedding2::{TOWER, TOWER_CPU, TOWER_IMAGES};
use std::collections::BTreeMap;
use vision::arch::TokenLimits;
use vision::arch::gemma4v::{Hparams, MERGE, conv_pixels, preprocess};

#[derive(Default)]
struct Taps(BTreeMap<String, Vec<f32>>);
impl TapSink for Taps {
    fn wants(&self, _: &str) -> bool {
        true
    }
    fn take(&mut self, name: &str, _: usize, bits: Vec<u16>) {
        self.0
            .insert(name.to_string(), bits.into_iter().map(bf16_f32).collect());
    }
    fn take_f32(&mut self, name: &str, _: usize, values: Vec<f32>) {
        self.0.insert(name.to_string(), values);
    }
}

fn pooling(p: &vision::Patches, taps: &Taps, hp: &Hparams) -> Result<bool, GateError> {
    use bloomery_gpu_vision::{f32_bf16, mean_groups::mean_groups_ref};
    let last = taps
        .0
        .get(&format!("layer_out-{}", hp.n_layer - 1))
        .ok_or("no last layer for pooling")?;
    let pooled = taps.0.get("pooled").ok_or("no pooled tap")?;
    let mut gathered = Vec::with_capacity(last.len());
    for gy in 0..p.n_vit_h / MERGE {
        for gx in 0..p.n_vit_w / MERGE {
            for dy in 0..MERGE {
                for dx in 0..MERGE {
                    let row = ((gy * MERGE + dy) * p.n_vit_w + gx * MERGE + dx) * hp.dim;
                    gathered.extend(last[row..row + hp.dim].iter().map(|&v| f32_bf16(v)));
                }
            }
        }
    }
    let expected: Vec<_> = mean_groups_ref(&gathered, hp.dim, MERGE * MERGE)
        .into_iter()
        .map(|bits| bf16_f32(f32_bf16(bf16_f32(bits) * (hp.dim as f32).sqrt())))
        .collect();
    Ok(pooled
        .iter()
        .map(|v| v.to_bits())
        .eq(expected.iter().map(|v| v.to_bits())))
}

fn run() -> Result<(), GateError> {
    bloomery_levers::at_main(&[])?;
    let card = image::open(&TOWER)?;
    let cpu = image::open(&TOWER_CPU)?;
    let model = ref_model_path()?;
    let file = Gguf::open(image::projector(&model)?)?;
    let hp = Hparams::read(&file)?;
    let limits = TokenLimits::new(70, 1120)?;
    let rule = hp.size_rule(limits)?;
    let gpu = Gpu::new()?;
    let stream = gpu.stream_handle();
    let mut enc = Encoder::load(gpu.context(), &stream, &file, limits)?;
    let mut ok = true;
    for name in TOWER_IMAGES {
        let (rgb, _) = image::image(&card, &cpu, name)?;
        let p = preprocess(&rgb, &rule)?;
        let n = p.n_vit_h * p.n_vit_w;
        let tokens = n / (MERGE * MERGE);
        let raw_shape = [p.plan.best_w as u64, p.plan.best_h as u64, 3, 1];
        let raw = image::row(&card, &format!("{name}/inp_raw_scaled"), raw_shape)?;
        let raw_cpu = image::row(&cpu, &format!("{name}/inp_raw_scaled"), raw_shape)?;
        let mut patches = Vec::with_capacity(raw.len());
        let mut pixels = Vec::with_capacity(raw.len());
        for py in 0..p.n_vit_h {
            for px in 0..p.n_vit_w {
                for c in 0..3 {
                    for y in 0..hp.patch {
                        for x in 0..hp.patch {
                            let at = (c * p.plan.best_h + py * hp.patch + y) * p.plan.best_w
                                + px * hp.patch
                                + x;
                            patches.push(bloomery_gpu_vision::f32_bf16(raw[at]));
                            pixels.push(raw[at]);
                        }
                    }
                }
            }
        }
        let exact = p.bf16 == patches
            && e2e::same_bits(&raw, &raw_cpu)
            && e2e::same_bits(&conv_pixels(&p.bf16)?, &pixels);
        println!(
            "(p) {name}: canvas and normalized patch bits {}",
            verdict(exact)
        );
        ok &= exact;
        let mut taps = Taps::default();
        enc.encode_tapped(&p, &mut taps)?;
        let pool_ok = pooling(&p, &taps, &hp)?;
        println!(
            "(o) {name}: 3x3 raster pooling and scale bits {}",
            verdict(pool_ok)
        );
        ok &= pool_ok;
        for (node, ours) in &taps.0 {
            let cols = if node == "projected" {
                hp.out_dim
            } else {
                hp.dim
            };
            let count = if node == "pos_embd" || node.starts_with("layer_out-") {
                n
            } else {
                tokens
            };
            let shape = [cols as u64, count as u64, 1, 1];
            let key = format!("{name}/{node}");
            let reference = image::row(&card, &key, shape)?;
            let twin = image::row(&cpu, &key, shape)?;
            ok &= image::compare(&format!("(t) {key}"), ours, &reference, &twin);
        }
        for node in ["pos_embd", "pooled", "projected"]
            .into_iter()
            .map(String::from)
            .chain((0..hp.n_layer).map(|l| format!("layer_out-{l}")))
        {
            if !taps.0.contains_key(&node) {
                return Err(format!("{name}: no tap {node}").into());
            }
        }
    }
    if ok { Ok(()) } else { Err(checks_failed()) }
}

fn main() -> std::process::ExitCode {
    exit_with("gate_gemma4v_tower", run())
}
