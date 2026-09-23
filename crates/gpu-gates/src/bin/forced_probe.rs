//! `forced_probe` — one teacher-forced position of the end-to-end gate, laid
//! open: our logits' top-k there, where the reference's token ranks in them,
//! every forced step's margin on the way, and each layer's taps at that
//! position. With `--against` it reads other arms' dumps (a comma list) and
//! prints, for each, both arms side by side with the per-layer relative
//! distance. `exact_ref --dump` writes the same layout, so an f64 arm is
//! one of them (`just exact-taps`).
//!
//!     forced_probe --prompt-id ID --step S --dump DIR [--against DIR[,DIR…]]
//!                  [--ctx C] [--top K]
//!
//! Two arms of a process-wide lever (`BLOOMERY_FLASH_MMA`, read once) are two
//! runs: the first writes its dump, the second reads it. `just forced-probe
//! ID S` runs the scalar arm then the tensor-core arm that way.
//!
//! The state is `gate_e2e`'s forced arm exactly: eager mode, `--ctx` defaults
//! to that gate's `CTX_MAX` (the cache height fixes the segment count), fresh
//! caches, the prompt one token at a time, then the reference's own tokens.
//! At step `S` the position is run twice: layer by layer with every tap read
//! back (`step_block0_taps`, `step_layer_taps`), then as the ordinary chain
//! for the logits. Both write the same cache row at the same position, so the
//! second run's answer is the forced arm's answer.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("forced_probe: built without the `gpu` feature; see `just forced-probe`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
use bloomery_gpu::Deepseek2Model;
#[cfg(feature = "gpu")]
use bloomery_gpu::arch::deepseek2::LayerTaps;
#[cfg(feature = "gpu")]
use bloomery_gpu::model::StepMode;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::prompts::{read_greedy, read_prompts};
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{GateError, data_dir, ref_model_path};
#[cfg(feature = "gpu")]
use gguf::Split;
#[cfg(feature = "gpu")]
use std::path::{Path, PathBuf};

#[cfg(feature = "gpu")]
type Res<T> = Result<T, GateError>;

#[cfg(feature = "gpu")]
fn flag_value(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

/// The taps dumped per layer, in forward order. `kqv_compressed` is the
/// flash output (every head's latent row), the first value the lever moves
/// at a position.
#[cfg(feature = "gpu")]
const TAPS: [&str; 10] = [
    "attn_norm",
    "q",
    "kv_compressed",
    "kqv_compressed",
    "kqv_out",
    "ffn_inp",
    "ffn_norm",
    "moe_logits",
    "moe_weights",
    "l_out",
];

#[cfg(feature = "gpu")]
fn tap<'a>(t: &'a LayerTaps, name: &str) -> &'a [f32] {
    match name {
        "attn_norm" => &t.attn_norm,
        "q" => &t.q,
        "kv_compressed" => &t.kv_compressed,
        "kqv_compressed" => &t.kqv_compressed,
        "kqv_out" => &t.kqv_out,
        "ffn_inp" => &t.ffn_inp,
        "ffn_norm" => &t.ffn_norm,
        "moe_logits" => &t.moe_logits,
        "moe_weights" => &t.moe_weights,
        "l_out" => &t.l_out,
        _ => unreachable!("TAPS names only these"),
    }
}

#[cfg(feature = "gpu")]
fn write_f32(path: &Path, v: &[f32]) -> Res<()> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()).into())
}

#[cfg(feature = "gpu")]
fn read_f32(path: &Path) -> Res<Vec<f32>> {
    let b = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

/// Token ids by logit descending, ties by id ascending — the reference
/// writer's order.
#[cfg(feature = "gpu")]
fn ranked(logits: &[f32]) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
    ids.sort_by(|&a, &b| {
        logits[b as usize]
            .total_cmp(&logits[a as usize])
            .then(a.cmp(&b))
    });
    ids
}

/// `‖a − b‖ / ‖b‖` and `max |a − b|`; `(0, 0)` for two empty spans.
#[cfg(feature = "gpu")]
fn distance(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut num, mut den, mut mx) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let d = f64::from(*x) - f64::from(*y);
        num += d * d;
        den += f64::from(*y) * f64::from(*y);
        mx = mx.max(d.abs());
    }
    (if den > 0.0 { (num / den).sqrt() } else { 0.0 }, mx)
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("forced_probe", run())
}

#[cfg(feature = "gpu")]
fn run() -> Res<()> {
    let id: usize = flag_value("--prompt-id")
        .ok_or("forced_probe: --prompt-id is required")?
        .parse()?;
    let step: usize = flag_value("--step")
        .ok_or("forced_probe: --step is required")?
        .parse()?;
    let dump = PathBuf::from(flag_value("--dump").ok_or("forced_probe: --dump DIR is required")?);
    let against: Vec<PathBuf> = flag_value("--against")
        .map(|s| s.split(',').map(PathBuf::from).collect())
        .unwrap_or_default();
    let ctx: usize = flag_value("--ctx").map_or(Ok(256), |s| s.parse())?;
    let top: usize = flag_value("--top").map_or(Ok(5), |s| s.parse())?;

    let reference = read_greedy(&data_dir().join("greedy-ik-cuda-32.tsv"))?;
    let prompts =
        read_prompts(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/prompts.tsv"))?;
    let r = reference
        .iter()
        .find(|r| r.id == id)
        .ok_or_else(|| format!("forced_probe: no reference row {id}"))?;
    let p = prompts
        .iter()
        .find(|p| p.id == id)
        .ok_or_else(|| format!("forced_probe: no prompt {id}"))?;
    if step >= r.gen_ids.len() {
        return Err(format!(
            "forced_probe: step {step} is past the reference row's {} tokens",
            r.gen_ids.len()
        )
        .into());
    }
    std::fs::create_dir_all(&dump)?;

    let mut model = Deepseek2Model::load_full(Split::open(ref_model_path()?)?, ctx)?;
    model.set_mode(StepMode::Eager);
    let layers = model.stages()[0].layers();
    println!(
        "arm flash_mma={} seg_keys={} ctx={ctx} prompt={id} tokens={} step={step} \
         pos={} live_keys={} segments={} live_segments={}",
        bloomery_gpu::flash::flash_mma(),
        bloomery_gpu::flash::seg_keys(),
        p.tokens.len(),
        p.tokens.len() - 1 + step,
        p.tokens.len() + step,
        bloomery_gpu::flash::segments_for(ctx),
        (p.tokens.len() + step).div_ceil(bloomery_gpu::flash::seg_keys()),
    );

    // The forced walk up to the position before `step`, one line per step:
    // our top-2 margin and the reference token's gap to our top-1.
    model.reset()?;
    let (head, feed): (&[u32], u32) = if step == 0 {
        (
            &p.tokens[..p.tokens.len() - 1],
            p.tokens[p.tokens.len() - 1],
        )
    } else {
        (&p.tokens[..], r.gen_ids[step - 1])
    };
    let mut trace = String::from("step\tours\ttheirs\tour_margin\ttheirs_gap\tref_margin\n");
    let mut note = |s: usize, model: &Deepseek2Model, ours: u32| -> Res<()> {
        let lg = model.logits()?;
        let rk = ranked(&lg);
        let theirs = r.gen_ids[s];
        trace.push_str(&format!(
            "{s}\t{ours}\t{theirs}\t{:.4}\t{:.4}\t{:.4}\n",
            lg[rk[0] as usize] - lg[rk[1] as usize],
            lg[rk[0] as usize] - lg[theirs as usize],
            r.gen_margins[s]
        ));
        Ok(())
    };
    if !head.is_empty() {
        let t = model.step(head)?;
        if step > 0 {
            note(0, &model, t)?;
        }
    }
    for s in 1..step {
        let t = model.step(&[r.gen_ids[s - 1]])?;
        note(s, &model, t)?;
    }

    // The position itself, layer by layer with every tap.
    let pos = model.pos();
    let mut taps = Vec::with_capacity(layers.len());
    let b0 = model.step_block0_taps(feed, pos)?;
    let mut x = b0.l_out.clone();
    taps.push(LayerTaps {
        layer: 0,
        attn_norm: b0.attn_norm,
        q: b0.q,
        kv_rope_compressed: b0.kv_rope_compressed,
        q_rope: b0.q_rope,
        k_rope: b0.k_rope,
        kv_compressed: b0.kv_compressed,
        kqv_compressed: b0.kqv_compressed,
        kqv_out: b0.kqv_out,
        ffn_inp: b0.ffn_inp,
        ffn_norm: Vec::new(),
        moe_logits: Vec::new(),
        moe_ids: Vec::new(),
        moe_weights: Vec::new(),
        expert_down: Vec::new(),
        ffn_shexp: Vec::new(),
        l_out: b0.l_out,
    });
    for l in layers.clone().skip(1) {
        let t = model.step_layer_taps(l, &x, pos)?;
        x.clone_from(&t.l_out);
        taps.push(t);
    }
    // The same position as the chain: the logits the forced arm reads.
    let ours = model.step(&[feed])?;
    note(step, &model, ours)?;
    let logits = model.logits()?;

    write_f32(&dump.join("logits.f32"), &logits)?;
    for t in &taps {
        for name in TAPS {
            write_f32(
                &dump.join(format!("L{:02}.{name}.f32", t.layer)),
                tap(t, name),
            )?;
        }
        let ids: Vec<String> = t.moe_ids.iter().map(u32::to_string).collect();
        std::fs::write(
            dump.join(format!("L{:02}.moe_ids.txt", t.layer)),
            ids.join(","),
        )?;
    }
    std::fs::write(dump.join("trace.tsv"), &trace)?;
    print!("{trace}");

    let rk = ranked(&logits);
    let theirs = r.gen_ids[step];
    let theirs_rank = rk.iter().position(|&t| t == theirs).unwrap_or(usize::MAX);
    println!(
        "top{top} {} | ik token {theirs} rank {theirs_rank} logit {:.4} gap_to_top1 {:.4} \
         | ik margin {:.4}",
        rk[..top]
            .iter()
            .map(|&t| format!("{t}:{:.4}", logits[t as usize]))
            .collect::<Vec<_>>()
            .join(" "),
        logits[theirs as usize],
        logits[rk[0] as usize] - logits[theirs as usize],
        r.gen_margins[step]
    );

    for other in &against {
        compare(&logits, &taps, other, top)?;
    }
    Ok(())
}

/// This arm's logits and taps against the dump in `other`: its top-k, the
/// logits' distance, and the per-layer relative distance of every tap.
#[cfg(feature = "gpu")]
fn compare(logits: &[f32], taps: &[LayerTaps], other: &Path, top: usize) -> Res<()> {
    let other_logits = read_f32(&other.join("logits.f32"))?;
    let ork = ranked(&other_logits);
    println!(
        "against {} top{top} {}",
        other.display(),
        ork[..top]
            .iter()
            .map(|&t| format!("{t}:{:.4}", other_logits[t as usize]))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let (lrel, lmax) = distance(logits, &other_logits);
    println!("logits rel={lrel:.3e} max_abs={lmax:.4}");
    print!("{:>3}", "L");
    for name in TAPS {
        print!(" {name:>15}");
    }
    println!("  moe_ids(this | against)");
    for t in taps {
        print!("{:>3}", t.layer);
        for name in TAPS {
            let b = read_f32(&other.join(format!("L{:02}.{name}.f32", t.layer)))?;
            let (rel, _) = distance(tap(t, name), &b);
            print!(" {rel:>15.3e}");
        }
        let ids: Vec<String> = t.moe_ids.iter().map(u32::to_string).collect();
        let oids = std::fs::read_to_string(other.join(format!("L{:02}.moe_ids.txt", t.layer)))?;
        let ids = ids.join(",");
        if ids == oids {
            println!("  =");
        } else {
            println!("  {ids} | {oids}");
        }
    }
    Ok(())
}
