//! `--dump-layers DIR`: the first host-fed step's (from the row's seed) residual and the probe
//! nodes each op leaves, as little-endian f32 files. A tool, not a clause.
//! The step is the traced one, so the read sees the op's writes; the kernels
//! and the session path are the ones a host-fed step runs.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

use bloomery_metal::arch::qwen3moe::count::Layer;
use bloomery_metal::arch::qwen3moe::session::{Node, Op, Session, SessionError};

use super::says;

pub fn run(s: &mut Session, dir: &Path) -> ExitCode {
    match go(s, dir) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            println!("dump-layers: FAIL ({e})");
            ExitCode::FAILURE
        }
    }
}

fn go(s: &mut Session, dir: &Path) -> Result<(), String> {
    s.reset().map_err(says)?;
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut saved: Vec<(String, Vec<f32>)> = Vec::new();
    s.step_traced(super::row::current().seed, |op, probe| {
        let mut take = |name: String, node: Node| -> Result<(), SessionError> {
            saved.push((name, probe.read(node)?));
            Ok(())
        };
        match op {
            Op::Embed => take("embed".to_string(), Node::Residual)?,
            Op::DeltaIn(l) => {
                take(format!("L{l:02}.qkv"), Node::DeltaQkv(l))?;
                take(format!("L{l:02}.z"), Node::DeltaZ(l))?;
                take(format!("L{l:02}.b"), Node::DeltaBeta(l))?;
                take(format!("L{l:02}.a"), Node::DeltaAlpha(l))?;
            }
            Op::DeltaCore(l) => {
                take(format!("L{l:02}.g"), Node::DeltaG)?;
                take(format!("L{l:02}.partials"), Node::DeltaPartials)?;
            }
            Op::DeltaOut(l) => take(format!("L{l:02}.mix"), Node::Residual)?,
            Op::ConvIn(_) | Op::ConvCore(_) => {}
            Op::ConvOut(l) => take(format!("L{l:02}.mix"), Node::Residual)?,
            Op::AttnIn(l) => {
                take(format!("L{l:02}.q"), Node::AttnQ(l))?;
                take(format!("L{l:02}.k"), Node::AttnK(l))?;
                take(format!("L{l:02}.v"), Node::AttnV(l))?;
            }
            Op::AttnCore(l) => take(format!("L{l:02}.attn"), Node::AttnOut)?,
            Op::AttnOut(l) => take(format!("L{l:02}.mix"), Node::Residual)?,
            Op::Route(l) => {
                take(format!("L{l:02}.router"), Node::RouterLogits(l))?;
                take(format!("L{l:02}.weights"), Node::RouterWeights(l))?;
            }
            Op::Experts(l) => take(format!("L{l:02}.out"), Node::Residual)?,
            Op::Head => take("logits".to_string(), Node::Logits)?,
        }
        Ok(())
    })
    .map_err(says)?;
    let mut manifest = String::new();
    for (name, values) in &saved {
        write_f32(dir, name, values)?;
        manifest.push_str(&format!("{name} {}\n", values.len()));
    }
    fs::write(dir.join("manifest.txt"), manifest)
        .map_err(|e| format!("{}: {e}", dir.join("manifest.txt").display()))?;
    write_geom(s, dir, &saved)?;
    println!("dump-layers: {} files in {}", saved.len(), dir.display());
    Ok(())
}

fn write_f32(dir: &Path, name: &str, values: &[f32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let path = dir.join(format!("{name}.f32"));
    fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn write_geom(s: &Session, dir: &Path, saved: &[(String, Vec<f32>)]) -> Result<(), String> {
    let g = s.geom();
    let head = g.delta.head_cols;
    let n_v = g.delta.value_heads;
    let qkv = saved
        .iter()
        .find(|(n, _)| n.ends_with(".qkv"))
        .map(|(_, v)| v.len());
    let n_k = qkv.and_then(|n| {
        if head == 0 || !n.is_multiple_of(head) {
            return None;
        }
        let channels = n / head;
        (channels >= n_v && (channels - n_v).is_multiple_of(2)).then_some((channels - n_v) / 2)
    });
    let path = dir.join("geom.txt");
    let mut f = fs::File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut line = |k: &str, v: &str| -> Result<(), String> {
        writeln!(f, "{k} {v}").map_err(|e| format!("{}: {e}", path.display()))
    };
    line("hidden", &g.hidden.to_string())?;
    line("vocab", &s.vocab().to_string())?;
    line("eps", &format!("{:.8e}", s.eps()))?;
    line("layers", &s.layer_count().to_string())?;
    line("value_heads", &n_v.to_string())?;
    line("head_cols", &head.to_string())?;
    line("conv_taps", &g.delta.conv_taps.to_string())?;
    line("q_scale_bits", &g.delta.q_scale.to_bits().to_string())?;
    let keys = n_k
        .map(|n| n.to_string())
        .unwrap_or_else(|| "none".to_string());
    line("key_heads", &keys)?;
    line("query_heads", &g.attn.query_heads.to_string())?;
    line("kv_heads", &g.attn.kv_heads.to_string())?;
    line("attn_head", &g.attn.head.to_string())?;
    for (i, layer) in s.token_layers().iter().enumerate() {
        let kind = match layer {
            Layer::Delta { .. } => "delta",
            Layer::ShortConv { .. } => "conv",
            Layer::Attention { .. } => "attention",
        };
        line(&format!("kind{i:02}"), kind)?;
    }
    Ok(())
}
