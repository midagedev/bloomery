//! `exact_ref` — the quantized model's own math, with no engine rounding in
//! it: every weight dequantized exactly (`gguf::quant::dequant_row`), every
//! activation, dot, norm, softmax and attention in f64, the key and value
//! rows never rounded to f16 and never quantized to q8. It is the referee
//! between two engines (ours, ik) that each round activations to q8 and the
//! cache to f16 — a position where they disagree is judged here.
//!
//!     exact_ref --prompt-id ID [--steps S,S,...] [--kv f64|f16]
//!               [--act f64|ours|ik|ik16] [--dump DIR] [--threads N]
//!               [--emit PATH]
//!
//! The state is `gate_e2e`'s forced arm: the prompt, then the reference's own
//! tokens (`greedy-ik-cuda-32.tsv`). For every step it prints our exact
//! top-2 margin and where the reference's token ranks; for each `--steps`
//! entry also the top-5. Without `--emit` the largest `--steps` entry is the
//! last step run. `--emit PATH` runs every step of the row whatever `--steps`
//! says and writes one `id step exact_top1 exact_top2 exact_margin` line per
//! step to PATH — the per-prompt part `just build-exact-forced` gathers into
//! the truth file `gate_e2e` judges the forced arm on.
//!
//! `--kv f16` rounds the latent and rope key rows to
//! f16 before they are used, as both engines' caches do — the one storage
//! rounding the model's graph itself specifies — so the two runs separate
//! that rounding from the engines' arithmetic.
//!
//! `--act` rounds the input of every quantized-weight product to 8-bit
//! blocks the way an engine's activation quantizer does (`Act`), everything
//! else staying f64: an arm that isolates one engine's activation rounding
//! from its sum orders. `--dump DIR` writes the taps of the last computed
//! step in `forced_probe`'s file layout, so `forced_probe --against DIR`
//! prints our GPU arm's per-layer distance to this one.
//!
//! MLA with the value side absorbed: scores use `k_nope` from `attn_kv_b`
//! applied to each cached latent row; the output is `wv_b` applied to the
//! head's attended latent row (`kqv_compressed`), the product the engines
//! quantize. In f64 both forms are the same sum reassociated.
//! The rope cos/sin table is the engine's own f32 one (`RopeParams::cache`).
//! The engines' Q8_0 requant of `wk_b` is not modelled: both carry it.

use bloomery_gpu_gates::prompts::{read_greedy, read_prompts};
use bloomery_gpu_gates::{GateError, open_model};
use gguf::quant::{GgmlType, dequant_row, half_to_f32};
use gguf::{Gguf, TensorInfo};
use model::attn::{MlaParams, f32_to_f16_bits};
use std::path::{Path, PathBuf};

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

/// Which activation rounding the quantized-weight products see.
///
/// Both engines quantize a product's input to int8 blocks, `d = amax/127`
/// (computed in f32) and `q = round(x/d)`; they differ in the block. Ours
/// feeds the K-quant gemvs (Q3_K, Q4_K, Q6_K) one scale per 128 values and
/// the Q5_0/Q5_1 gemvs one per 32; ik's `quantize_q8_1` is one per 32
/// everywhere and stores `d` as f16 (`ik16` rounds it; `ik` keeps it f32,
/// isolating the block size).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Act {
    F64,
    Ours,
    Ik,
    Ik16,
}

impl Act {
    fn parse(s: Option<&str>) -> Res<Act> {
        match s {
            None | Some("f64") => Ok(Act::F64),
            Some("ours") => Ok(Act::Ours),
            Some("ik") => Ok(Act::Ik),
            Some("ik16") => Ok(Act::Ik16),
            Some(o) => Err(format!("exact_ref: --act is f64, ours, ik or ik16, not {o}").into()),
        }
    }

    /// Values per scale for a product with weights of type `ty`; `None`
    /// leaves the input exact (the F32 router, and every product in `F64`).
    fn block(self, ty: GgmlType) -> Option<usize> {
        let kquant = matches!(ty, GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q6_K);
        let q5 = matches!(ty, GgmlType::Q5_0 | GgmlType::Q5_1);
        match self {
            Act::F64 => None,
            Act::Ours if kquant => Some(128),
            Act::Ours if q5 => Some(32),
            Act::Ik | Act::Ik16 if kquant || q5 => Some(32),
            _ => None,
        }
    }

    /// `x` as the product sees it: each block rounded to `d * q`.
    fn round(self, ty: GgmlType, x: &[f64]) -> Vec<f64> {
        let Some(b) = self.block(ty) else {
            return x.to_vec();
        };
        assert!(
            x.len().is_multiple_of(b),
            "{} values do not split into {b}-value blocks",
            x.len()
        );
        let mut out = Vec::with_capacity(x.len());
        for blk in x.chunks(b) {
            let v: Vec<f32> = blk.iter().map(|&a| a as f32).collect();
            let amax = v.iter().fold(0f32, |m, a| m.max(a.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            let dq = if self == Act::Ik16 {
                half_to_f32(f32_to_f16_bits(d))
            } else {
                d
            };
            out.extend(
                v.iter()
                    .map(|&a| f64::from((a / d).round().clamp(-127.0, 127.0)) * f64::from(dq)),
            );
        }
        out
    }
}

/// One cached position of one layer: every head's `k_nope`, the latent
/// row the values are read through, and the shared roped key.
struct Row {
    k_nope: Vec<f64>,
    c_kv: Vec<f64>,
    k_pe: Vec<f64>,
}

/// The taps of one layer at the dumped step, named and laid out as
/// `forced_probe` writes them. The MoE spans stay empty for a dense layer.
#[derive(Default)]
struct Taps {
    attn_norm: Vec<f64>,
    q: Vec<f64>,
    kv_compressed: Vec<f64>,
    kqv_compressed: Vec<f64>,
    kqv_out: Vec<f64>,
    ffn_inp: Vec<f64>,
    ffn_norm: Vec<f64>,
    moe_logits: Vec<f64>,
    moe_ids: Vec<usize>,
    moe_weights: Vec<f64>,
    l_out: Vec<f64>,
}

impl Taps {
    fn spans(&self) -> [(&'static str, &[f64]); 10] {
        [
            ("attn_norm", &self.attn_norm),
            ("q", &self.q),
            ("kv_compressed", &self.kv_compressed),
            ("kqv_compressed", &self.kqv_compressed),
            ("kqv_out", &self.kqv_out),
            ("ffn_inp", &self.ffn_inp),
            ("ffn_norm", &self.ffn_norm),
            ("moe_logits", &self.moe_logits),
            ("moe_weights", &self.moe_weights),
            ("l_out", &self.l_out),
        ]
    }
}

fn write_f32(path: &Path, v: &[f64]) -> Res<()> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| (*x as f32).to_le_bytes()).collect();
    std::fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()).into())
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
    act: Act,
    threads: usize,
    cache: Vec<Vec<Row>>,
    /// `Some` while a step records its taps, one entry per layer.
    taps: Option<Vec<Taps>>,
}

impl Model<'_> {
    /// `W x` with `x` rounded the way this arm's engine feeds `W`.
    fn mvq(&self, w: View<'_>, x: &[f64]) -> Res<Vec<f64>> {
        matvec(self.g, w, &self.act.round(w.t.ty, x), self.threads)
    }

    fn tap(&mut self) -> Option<&mut Taps> {
        self.taps.as_mut().and_then(|t| t.last_mut())
    }

    fn attn(&mut self, l: usize, x: &[f64], pos: u32) -> Res<Vec<f64>> {
        let (g, t) = (self.g, self.threads);
        let p = self.p.clone();
        let (nh, nope, rd, vh, lat) = (p.n_head, p.nope, p.rope_dims, p.v_head, p.latent);
        let q = self.mvq(
            view(find(g, &format!("blk.{l}.attn_q.weight"))?, None)?,
            x,
        )?;
        let kva = self.mvq(
            view(find(g, &format!("blk.{l}.attn_kv_a_mqa.weight"))?, None)?,
            x,
        )?;
        let gain = f32_vec(g, find(g, &format!("blk.{l}.attn_kv_a_norm.weight"))?)?;
        let normed = rms_norm(&kva[..lat], &gain, self.eps);
        let mut c_kv = normed.clone();
        let cs = p.rope.cache(pos);
        let mut k_pe = rope(&kva[lat..lat + rd], &cs);
        if self.kv_f16 {
            round_f16(&mut c_kv);
            round_f16(&mut k_pe);
        }
        let wkvb = find(g, &format!("blk.{l}.attn_kv_b.weight"))?;
        // The key side reads the cached row itself: no engine quantizes it.
        let kvb = matvec(g, view(wkvb, None)?, &c_kv, t)?;
        let mut k_nope = Vec::with_capacity(nh * nope);
        for h in 0..nh {
            let b = h * (nope + vh);
            k_nope.extend_from_slice(&kvb[b..b + nope]);
        }
        self.cache[l].push(Row { k_nope, c_kv, k_pe });

        let scale = f64::from(p.kq_scale);
        let mut kqvc = vec![0.0f64; nh * lat];
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
            let o = &mut kqvc[h * lat..(h + 1) * lat];
            for (wj, r) in w.iter().zip(&self.cache[l]) {
                for (oi, ci) in o.iter_mut().zip(&r.c_kv) {
                    *oi += wj / z * ci;
                }
            }
        }
        // wv_b per head: rows h*(nope+vh)+nope .. +vh of attn_kv_b against
        // the head's attended latent row.
        let base = view(wkvb, None)?;
        let mut out = Vec::with_capacity(nh * vh);
        for h in 0..nh {
            let wv = View {
                rows: vh,
                offset: (h * (nope + vh) + nope) * base.row_bytes,
                ..base
            };
            out.extend(self.mvq(wv, &kqvc[h * lat..(h + 1) * lat])?);
        }
        let y = self.mvq(
            view(find(g, &format!("blk.{l}.attn_output.weight"))?, None)?,
            &out,
        )?;
        if let Some(tp) = self.tap() {
            tp.q = q;
            tp.kv_compressed = normed;
            tp.kqv_compressed = kqvc;
            tp.kqv_out = y.clone();
        }
        Ok(y)
    }

    fn mlp(&self, gate: View<'_>, up: View<'_>, down: View<'_>, x: &[f64]) -> Res<Vec<f64>> {
        let a = self.mvq(gate, x)?;
        let b = self.mvq(up, x)?;
        let h: Vec<f64> = a.iter().zip(&b).map(|(a, b)| silu(*a) * b).collect();
        self.mvq(down, &h)
    }

    fn ffn(&mut self, l: usize, x: &[f64]) -> Res<Vec<f64>> {
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
        let chosen = ids[..self.n_used].to_vec();
        let weights: Vec<f64> = chosen
            .iter()
            .map(|&id| e[id] / z * self.expert_scale)
            .collect();
        for (&id, &w) in chosen.iter().zip(&weights) {
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
        if let Some(tp) = self.tap() {
            tp.moe_logits = logits;
            tp.moe_ids = chosen;
            tp.moe_weights = weights;
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
            if let Some(t) = self.taps.as_mut() {
                t.push(Taps::default());
            }
            let gain = f32_vec(g, find(g, &format!("blk.{l}.attn_norm.weight"))?)?;
            let an = rms_norm(&x, &gain, self.eps);
            let a = self.attn(l, &an, pos)?;
            for (xi, ai) in x.iter_mut().zip(&a) {
                *xi += ai;
            }
            let routed = g.find(&format!("blk.{l}.ffn_gate_inp.weight")).is_some();
            if let Some(tp) = self.tap() {
                tp.attn_norm = an;
                tp.ffn_inp.clone_from(&x);
            }
            let gain = f32_vec(g, find(g, &format!("blk.{l}.ffn_norm.weight"))?)?;
            let fnorm = rms_norm(&x, &gain, self.eps);
            let f = self.ffn(l, &fnorm)?;
            for (xi, fi) in x.iter_mut().zip(&f) {
                *xi += fi;
            }
            if let Some(tp) = self.tap() {
                // The dense block keeps its normed vector in registers, so
                // the engine has no `ffn_norm` tap there.
                if routed {
                    tp.ffn_norm = fnorm;
                }
                tp.l_out.clone_from(&x);
            }
        }
        let gain = f32_vec(g, find(g, "output_norm.weight")?)?;
        self.mvq(
            view(find(g, "output.weight")?, None)?,
            &rms_norm(&x, &gain, self.eps),
        )
    }

    /// Write the recorded taps and `logits` in `forced_probe`'s layout.
    fn dump(&self, dir: &Path, logits: &[f64]) -> Res<()> {
        std::fs::create_dir_all(dir)?;
        write_f32(&dir.join("logits.f32"), logits)?;
        for (l, t) in self.taps.iter().flatten().enumerate() {
            for (name, v) in t.spans() {
                write_f32(&dir.join(format!("L{l:02}.{name}.f32")), v)?;
            }
            let ids: Vec<String> = t.moe_ids.iter().map(usize::to_string).collect();
            std::fs::write(dir.join(format!("L{l:02}.moe_ids.txt")), ids.join(","))?;
        }
        Ok(())
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
    let act = Act::parse(flag_value("--act").as_deref())?;
    let dump = flag_value("--dump").map(PathBuf::from);
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
    let emit = flag_value("--emit");
    // The emitted file is the gate's truth; a simulated arm is not exact.
    if emit.is_some() && act != Act::F64 {
        return Err("exact_ref: --emit writes the exact truth and needs --act f64".into());
    }
    let last = match (&emit, steps.iter().copied().max()) {
        (None, Some(s)) => s,
        _ => r.gen_ids.len() - 1,
    };
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
        act,
        threads,
        cache: (0..n_layer).map(|_| Vec::new()).collect(),
        taps: None,
    };
    println!(
        "exact_ref prompt={id} tokens={} kv={} act={act:?} threads={threads} \
         expert_scale={expert_scale}",
        pr.tokens.len(),
        if kv_f16 { "f16" } else { "f64" }
    );
    println!("step\tpos\ttheirs\texact_top1\texact_margin\ttheirs_gap\tref_margin");
    let n_prompt = pr.tokens.len();
    let mut emitted = Vec::new();
    let mut logits = Vec::new();
    // The dumped position is the last one computed: step `last`'s feed.
    for (i, &tok) in pr.tokens.iter().enumerate() {
        if dump.is_some() && last == 0 && i + 1 == n_prompt {
            m.taps = Some(Vec::new());
        }
        logits = m.step(tok, i as u32)?;
    }
    for s in 0..=last {
        if s > 0 {
            if dump.is_some() && s == last {
                m.taps = Some(Vec::new());
            }
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
        emitted.push((rk[0], rk[1], logits[rk[0]] - logits[rk[1]]));
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
    if let Some(dir) = dump {
        m.dump(&dir, &logits)?;
        println!("dumped step {last} taps to {}", dir.display());
    }
    if let Some(path) = emit {
        let model = std::env::var("BLOOMERY_REF_MODEL")
            .unwrap_or_else(|_| bloomery_gpu_gates::DEFAULT_MODEL.to_string());
        write_emit(Path::new(&path), id, &model, kv_f16, &emitted)?;
    }
    Ok(())
}

/// The `--emit` part: a header naming what the rows were computed from, then
/// one line per step. Written to `<path>.tmp.<pid>` and renamed, so a reader
/// never sees a half-written part.
fn write_emit(
    path: &Path,
    id: usize,
    model: &str,
    kv_f16: bool,
    rows: &[(usize, usize, f64)],
) -> Res<()> {
    use std::fmt::Write as _;
    let mut text = format!(
        "# exact_ref model={model} kv={}\n",
        if kv_f16 { "f16" } else { "f64" }
    );
    for (s, (t1, t2, m)) in rows.iter().enumerate() {
        writeln!(text, "{id}\t{s}\t{t1}\t{t2}\t{m:.6}")?;
    }
    let tmp = path.with_file_name(format!(
        "{}.tmp.{}",
        path.file_name()
            .ok_or("exact_ref: --emit needs a file path")?
            .to_string_lossy(),
        std::process::id()
    ));
    std::fs::write(&tmp, text)
        .map_err(|e| format!("exact_ref: cannot write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .map_err(|e| format!("exact_ref: cannot rename to {}: {e}", path.display()))?;
    Ok(())
}
