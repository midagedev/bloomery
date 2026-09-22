//! `exact_ref` — the quantized model's own math, with no engine rounding in
//! it: every weight dequantized exactly (`gguf::quant::dequant_row`), every
//! activation, dot, norm, softmax and attention in f64, the key and value
//! rows never rounded to f16 and never quantized to q8. It is the referee
//! between two engines (ours, ik) that each round activations to q8 and the
//! cache to f16 — a position where they disagree is judged here.
//!
//!     exact_ref --prompt-id ID [--steps S,S,...] [--kv f64|f16] [--threads N]
//!
//! The state is `gate_e2e`'s forced arm: the prompt, then the reference's own
//! tokens (`greedy-ik-cuda-32.tsv`). For every step it prints our exact
//! top-2 margin and where the reference's token ranks; for each `--steps`
//! entry also the top-5. `--kv f16` rounds the latent and rope key rows to
//! f16 before they are used, as both engines' caches do — the one storage
//! rounding the model's graph itself specifies — so the two runs separate
//! that rounding from the engines' arithmetic.
//!
//! Standard (non-absorbed) MLA: `k_nope` and `v` per head come from
//! `attn_kv_b` applied to the cached latent row. Absorbing `wk_b` into the
//! query is the same sum reassociated, so the choice changes nothing in f64.
//! The rope cos/sin table is the engine's own f32 one (`RopeParams::cache`).

use bloomery_gpu_gates::prompts::{read_greedy, read_prompts};
use bloomery_gpu_gates::{GateError, open_model};
use gguf::quant::{dequant_row, half_to_f32};
use gguf::{Gguf, TensorInfo};
use model::attn::{MlaParams, f32_to_f16_bits};
use std::path::Path;

type Res<T> = Result<T, GateError>;

fn flag_value(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn find<'a>(g: &'a Gguf, name: &str) -> Res<&'a TensorInfo> {
    g.find(name)
        .ok_or_else(|| format!("exact_ref: no tensor {name}").into())
}

/// A 2-D weight view: `rows` rows of `k` values each, starting at byte
/// `offset` of the tensor's data. An expert of a stacked tensor is one view.
#[derive(Clone, Copy)]
struct View<'a> {
    t: &'a TensorInfo,
    k: usize,
    rows: usize,
    row_bytes: usize,
    offset: usize,
}

fn view<'a>(t: &'a TensorInfo, expert: Option<usize>) -> Res<View<'a>> {
    let k = t.dims[0] as usize;
    let rows = t.dims[1] as usize;
    let blck = t.ty.blck_size().ok_or("exact_ref: unsupported type")? as usize;
    let tsz = t.ty.type_size().ok_or("exact_ref: unsupported type")? as usize;
    let row_bytes = k / blck * tsz;
    let offset = match expert {
        Some(e) => e * rows * row_bytes,
        None => 0,
    };
    Ok(View {
        t,
        k,
        rows,
        row_bytes,
        offset,
    })
}

/// `y = W x` in f64 over exactly dequantized rows, the rows split across
/// `threads` scoped threads. Each row's dot is one f64 sum, so the split
/// changes no bit.
fn matvec(g: &Gguf, w: View<'_>, x: &[f64], threads: usize) -> Res<Vec<f64>> {
    assert_eq!(
        x.len(),
        w.k,
        "matvec {}: x has {} values",
        w.t.name,
        x.len()
    );
    let data = g.data(w.t)?;
    let mut y = vec![0.0f64; w.rows];
    let chunk = w.rows.div_ceil(threads);
    std::thread::scope(|s| -> Res<()> {
        let mut handles = Vec::new();
        for (c, out) in y.chunks_mut(chunk).enumerate() {
            let r0 = c * chunk;
            handles.push(s.spawn(move || -> Result<(), String> {
                let mut row = vec![0.0f32; w.k];
                for (i, o) in out.iter_mut().enumerate() {
                    let b = w.offset + (r0 + i) * w.row_bytes;
                    dequant_row(w.t.ty, &data[b..b + w.row_bytes], &mut row)
                        .map_err(|e| format!("{}: {e:?}", w.t.name))?;
                    *o = row.iter().zip(x).map(|(a, b)| f64::from(*a) * b).sum();
                }
                Ok(())
            }));
        }
        for h in handles {
            h.join().map_err(|_| "exact_ref: worker panicked")??;
        }
        Ok(())
    })?;
    Ok(y)
}

fn f32_vec(g: &Gguf, t: &TensorInfo) -> Res<Vec<f64>> {
    let n = t.dims.iter().product::<u64>() as usize;
    let mut v = vec![0.0f32; n];
    dequant_row(t.ty, g.data(t)?, &mut v).map_err(|e| format!("{}: {e:?}", t.name))?;
    Ok(v.into_iter().map(f64::from).collect())
}

fn rms_norm(x: &[f64], gain: &[f64], eps: f64) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(gain).map(|(v, g)| v * inv * g).collect()
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// Adjacent-pair rotation, the engine's NORM rope, in f64 over its table.
fn rope(src: &[f64], cs: &[f32]) -> Vec<f64> {
    let mut out = vec![0.0; src.len()];
    for i in (0..src.len()).step_by(2) {
        let (c, s) = (f64::from(cs[i]), f64::from(cs[i + 1]));
        out[i] = src[i] * c - src[i + 1] * s;
        out[i + 1] = src[i] * s + src[i + 1] * c;
    }
    out
}

fn round_f16(v: &mut [f64]) {
    for x in v {
        *x = f64::from(half_to_f32(f32_to_f16_bits(*x as f32)));
    }
}

/// One cached position of one layer: every head's `k_nope` and `v`, and the
/// shared roped key.
struct Row {
    k_nope: Vec<f64>,
    v: Vec<f64>,
    k_pe: Vec<f64>,
}

struct Model<'a> {
    g: &'a Gguf,
    p: MlaParams,
    eps: f64,
    n_layer: usize,
    n_expert: usize,
    n_used: usize,
    expert_scale: f64,
    kv_f16: bool,
    threads: usize,
    cache: Vec<Vec<Row>>,
}

impl Model<'_> {
    fn attn(&mut self, l: usize, x: &[f64], pos: u32) -> Res<Vec<f64>> {
        let (g, t) = (self.g, self.threads);
        let p = &self.p;
        let (nh, nope, rd, vh, lat) = (p.n_head, p.nope, p.rope_dims, p.v_head, p.latent);
        let q = matvec(
            g,
            view(find(g, &format!("blk.{l}.attn_q.weight"))?, None)?,
            x,
            t,
        )?;
        let kva = matvec(
            g,
            view(find(g, &format!("blk.{l}.attn_kv_a_mqa.weight"))?, None)?,
            x,
            t,
        )?;
        let gain = f32_vec(g, find(g, &format!("blk.{l}.attn_kv_a_norm.weight"))?)?;
        let mut c_kv = rms_norm(&kva[..lat], &gain, self.eps);
        let cs = p.rope.cache(pos);
        let mut k_pe = rope(&kva[lat..lat + rd], &cs);
        if self.kv_f16 {
            round_f16(&mut c_kv);
            round_f16(&mut k_pe);
        }
        let kvb = matvec(
            g,
            view(find(g, &format!("blk.{l}.attn_kv_b.weight"))?, None)?,
            &c_kv,
            t,
        )?;
        let mut k_nope = Vec::with_capacity(nh * nope);
        let mut v = Vec::with_capacity(nh * vh);
        for h in 0..nh {
            let b = h * (nope + vh);
            k_nope.extend_from_slice(&kvb[b..b + nope]);
            v.extend_from_slice(&kvb[b + nope..b + nope + vh]);
        }
        self.cache[l].push(Row { k_nope, v, k_pe });

        let scale = f64::from(p.kq_scale);
        let mut out = vec![0.0f64; nh * vh];
        for h in 0..nh {
            let qh = &q[h * p.kq_head..(h + 1) * p.kq_head];
            let q_pe = rope(&qh[nope..], &cs);
            let s: Vec<f64> = self.cache[l]
                .iter()
                .map(|r| {
                    let a: f64 = qh[..nope]
                        .iter()
                        .zip(&r.k_nope[h * nope..(h + 1) * nope])
                        .map(|(a, b)| a * b)
                        .sum();
                    let b: f64 = q_pe.iter().zip(&r.k_pe).map(|(a, b)| a * b).sum();
                    scale * (a + b)
                })
                .collect();
            let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = s.iter().map(|v| (v - m).exp()).collect();
            let z: f64 = w.iter().sum();
            let o = &mut out[h * vh..(h + 1) * vh];
            for (wj, r) in w.iter().zip(&self.cache[l]) {
                for (oi, vi) in o.iter_mut().zip(&r.v[h * vh..(h + 1) * vh]) {
                    *oi += wj / z * vi;
                }
            }
        }
        matvec(
            g,
            view(find(g, &format!("blk.{l}.attn_output.weight"))?, None)?,
            &out,
            t,
        )
    }

    fn mlp(&self, gate: View<'_>, up: View<'_>, down: View<'_>, x: &[f64]) -> Res<Vec<f64>> {
        let a = matvec(self.g, gate, x, self.threads)?;
        let b = matvec(self.g, up, x, self.threads)?;
        let h: Vec<f64> = a.iter().zip(&b).map(|(a, b)| silu(*a) * b).collect();
        matvec(self.g, down, &h, self.threads)
    }

    fn ffn(&self, l: usize, x: &[f64]) -> Res<Vec<f64>> {
        let g = self.g;
        let t = |s: &str| find(g, &format!("blk.{l}.{s}.weight"));
        let Some(router) = g.find(&format!("blk.{l}.ffn_gate_inp.weight")) else {
            return self.mlp(
                view(t("ffn_gate")?, None)?,
                view(t("ffn_up")?, None)?,
                view(t("ffn_down")?, None)?,
                x,
            );
        };
        let logits = matvec(g, view(router, None)?, x, self.threads)?;
        let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = logits.iter().map(|v| (v - m).exp()).collect();
        let z: f64 = e.iter().sum();
        let mut ids: Vec<usize> = (0..self.n_expert).collect();
        ids.sort_by(|&a, &b| e[b].total_cmp(&e[a]).then(a.cmp(&b)));
        let mut out = self.mlp(
            view(t("ffn_gate_shexp")?, None)?,
            view(t("ffn_up_shexp")?, None)?,
            view(t("ffn_down_shexp")?, None)?,
            x,
        )?;
        for &id in &ids[..self.n_used] {
            let w = e[id] / z * self.expert_scale;
            let y = self.mlp(
                view(t("ffn_gate_exps")?, Some(id))?,
                view(t("ffn_up_exps")?, Some(id))?,
                view(t("ffn_down_exps")?, Some(id))?,
                x,
            )?;
            for (o, v) in out.iter_mut().zip(&y) {
                *o += w * v;
            }
        }
        Ok(out)
    }

    /// One position through every layer and the head: the logits.
    fn step(&mut self, token: u32, pos: u32) -> Res<Vec<f64>> {
        let g = self.g;
        let emb = view(find(g, "token_embd.weight")?, None)?;
        let data = g.data(emb.t)?;
        let mut row = vec![0.0f32; emb.k];
        let b = token as usize * emb.row_bytes;
        dequant_row(emb.t.ty, &data[b..b + emb.row_bytes], &mut row)
            .map_err(|e| format!("token_embd: {e:?}"))?;
        let mut x: Vec<f64> = row.into_iter().map(f64::from).collect();
        for l in 0..self.n_layer {
            let gain = f32_vec(g, find(g, &format!("blk.{l}.attn_norm.weight"))?)?;
            let a = self.attn(l, &rms_norm(&x, &gain, self.eps), pos)?;
            for (xi, ai) in x.iter_mut().zip(&a) {
                *xi += ai;
            }
            let gain = f32_vec(g, find(g, &format!("blk.{l}.ffn_norm.weight"))?)?;
            let f = self.ffn(l, &rms_norm(&x, &gain, self.eps))?;
            for (xi, fi) in x.iter_mut().zip(&f) {
                *xi += fi;
            }
        }
        let gain = f32_vec(g, find(g, "output_norm.weight")?)?;
        matvec(
            g,
            view(find(g, "output.weight")?, None)?,
            &rms_norm(&x, &gain, self.eps),
            self.threads,
        )
    }
}

fn ranked(logits: &[f64]) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..logits.len()).collect();
    ids.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    ids
}

fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("exact_ref", run())
}

fn run() -> Res<()> {
    let id: usize = flag_value("--prompt-id")
        .ok_or("exact_ref: --prompt-id is required")?
        .parse()?;
    let steps: Vec<usize> = flag_value("--steps").map_or(Ok(Vec::new()), |s| {
        s.split(',').map(str::parse).collect::<Result<_, _>>()
    })?;
    let kv_f16 = match flag_value("--kv").as_deref() {
        None | Some("f64") => false,
        Some("f16") => true,
        Some(o) => return Err(format!("exact_ref: --kv is f64 or f16, not {o}").into()),
    };
    let threads: usize = flag_value("--threads").map_or_else(
        || Ok(std::thread::available_parallelism().map_or(8, |n| n.get())),
        |s| s.parse(),
    )?;

    let data = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".to_string());
    let reference = read_greedy(&Path::new(&data).join("greedy-ik-cuda-32.tsv"))?;
    let prompts =
        read_prompts(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/prompts.tsv"))?;
    let r = reference
        .iter()
        .find(|r| r.id == id)
        .ok_or_else(|| format!("exact_ref: no reference row {id}"))?;
    let pr = prompts
        .iter()
        .find(|p| p.id == id)
        .ok_or_else(|| format!("exact_ref: no prompt {id}"))?;
    let last = steps.iter().copied().max().unwrap_or(r.gen_ids.len() - 1);
    if last >= r.gen_ids.len() {
        return Err(format!(
            "exact_ref: step {last} is past the row's {}",
            r.gen_ids.len()
        )
        .into());
    }

    let g = open_model()?;
    let n_layer = g.block_count().ok_or("exact_ref: no block_count")? as usize;
    let p = MlaParams::read(&g, 0)?;
    let eps = f64::from(p.eps);
    let expert_scale = g
        .architecture()
        .and_then(|a| g.value(&format!("{a}.expert_weights_scale")))
        .and_then(gguf::Value::as_f32)
        .map_or(1.0, f64::from);
    let mut m = Model {
        g: &g,
        p,
        eps,
        n_layer,
        n_expert: g.expert_count().ok_or("exact_ref: no expert_count")? as usize,
        n_used: g
            .expert_used_count()
            .ok_or("exact_ref: no expert_used_count")? as usize,
        expert_scale,
        kv_f16,
        threads,
        cache: (0..n_layer).map(|_| Vec::new()).collect(),
    };
    println!(
        "exact_ref prompt={id} tokens={} kv={} threads={threads} expert_scale={expert_scale}",
        pr.tokens.len(),
        if kv_f16 { "f16" } else { "f64" }
    );
    println!("step\tpos\ttheirs\texact_top1\texact_margin\ttheirs_gap\tref_margin");
    let n_prompt = pr.tokens.len();
    let mut logits = Vec::new();
    for (i, &tok) in pr.tokens.iter().enumerate() {
        logits = m.step(tok, i as u32)?;
    }
    for s in 0..=last {
        if s > 0 {
            logits = m.step(r.gen_ids[s - 1], (n_prompt - 1 + s) as u32)?;
        }
        let rk = ranked(&logits);
        let theirs = r.gen_ids[s] as usize;
        println!(
            "{s}\t{}\t{theirs}\t{}\t{:.4}\t{:.4}\t{:.4}",
            n_prompt - 1 + s,
            rk[0],
            logits[rk[0]] - logits[rk[1]],
            logits[rk[0]] - logits[theirs],
            r.gen_margins[s]
        );
        if steps.contains(&s) {
            println!(
                "  top5 step {s}: {}",
                rk[..5]
                    .iter()
                    .map(|&t| format!("{t}:{:.4}", logits[t]))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
    Ok(())
}
