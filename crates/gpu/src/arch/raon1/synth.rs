//! A synthetic Raon-1 file at small widths, written whole and loaded as the engine loads a model
//! file: the case the real file does not hold, a weight the loader admits that the wide arm
//! refuses (a latent layer's Q6_K output projection), and Raon's own types beside it. The
//! widths are the file's where a kernel fixes them (the heads, the latent, the experts) and
//! small where none does; the weights are random with small scales, so every value is finite.

use super::body::Body;
use super::rda::tests::same;
use crate::model::GpuModel;
use crate::{Gpu, GpuError};
use gguf::quant::Q8Block;
use gguf::write::{Layout, TensorDecl, Writer};
use gguf::{GgmlType, Split, Value};
use std::path::PathBuf;

/// Two Q6_K super-blocks a row: the K-quant gemv's even count.
const E: usize = 512;
const HEADS: usize = 40;
const HEAD: usize = 128;
const INNER: usize = HEADS * HEAD;
const CONV: usize = 4;
const RANK: usize = 128;
const Q_LORA: usize = 256;
const NOPE: usize = 128;
const PE: usize = 64;
const KV_LORA: usize = 512;
const FF: usize = 512;
const EFF: usize = 256;
const N_EXP: usize = 128;
const N_USED: usize = 6;
const STREAMS: usize = 4;
const VOCAB: usize = 512;
/// A delta layer with the dense block, a delta layer and a latent layer with the routed one.
const LAYERS: usize = 3;
const CTX: usize = 128;

/// A 64-bit LCG (Knuth's MMIX constants).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 40
    }

    fn uniform(&mut self, (lo, hi): (f32, f32)) -> f32 {
        lo + (hi - lo) * (self.next() as f32 / (1u64 << 24) as f32)
    }
}

/// `n` values of type `ty` as the file holds them: F32 values in `range`, or blocks of random
/// codes under a small scale, so every weight is finite and small.
fn values(r: &mut Lcg, ty: GgmlType, n: usize, range: (f32, f32)) -> Vec<u8> {
    match ty {
        GgmlType::F32 => (0..n)
            .flat_map(|_| r.uniform(range).to_le_bytes())
            .collect(),
        GgmlType::Q8_0 => {
            let mut bytes = Vec::with_capacity(n / 32 * 34);
            for _ in 0..n / 32 {
                let x: [f32; 32] = std::array::from_fn(|_| r.uniform((-0.05, 0.05)));
                let q = Q8Block::quantize_ref(&x).expect("a finite block");
                bytes.extend(q.d.to_le_bytes());
                bytes.extend(q.q.map(|c| c.to_le_bytes()[0]));
            }
            bytes
        }
        GgmlType::Q6_K => (0..n / 256 * 210)
            .map(|i| match i % 210 {
                192..=207 => ((r.next() % 9) as i8 - 4).to_le_bytes()[0],
                209 => 0x14,
                _ => r.next() as u8,
            })
            .collect(),
        GgmlType::Q4_K => (0..n / 256 * 144)
            .map(|i| match i % 144 {
                1 | 3 => 0x14,
                _ => r.next() as u8,
            })
            .collect(),
        GgmlType::Q5_0 => (0..n / 32 * 22)
            .map(|i| if i % 22 == 1 { 0x14 } else { r.next() as u8 })
            .collect(),
        other => panic!("the synthetic file holds no {other:?} tensor"),
    }
}

/// The file's tensors: name, ggml dims (the contiguous axis first), type and the range of an F32
/// one. `out` is the type of the latent layer's output projection.
fn tensors(out: GgmlType) -> Vec<(String, Vec<usize>, GgmlType, (f32, f32))> {
    use GgmlType::{F32, Q4_K, Q5_0, Q6_K, Q8_0};
    let mix = 2 * STREAMS + (STREAMS - 1) * (STREAMS - 1);
    let small = (-0.05, 0.05);
    let mut t: Vec<(String, Vec<usize>, GgmlType, (f32, f32))> = vec![
        ("token_embd.weight".into(), vec![E, VOCAB], Q4_K, small),
        ("output.weight".into(), vec![E, VOCAB], Q6_K, small),
        ("output_norm.weight".into(), vec![E], F32, (0.5, 1.5)),
    ];
    for l in 0..LAYERS {
        let mut layer: Vec<(&str, Vec<usize>, GgmlType, (f32, f32))> =
            vec![("attn_norm.weight", vec![E], F32, (0.5, 1.5))];
        if l == LAYERS - 1 {
            layer.extend([
                ("attn_q_a.weight", vec![E, Q_LORA], Q6_K, small),
                ("attn_q_a_norm.weight", vec![Q_LORA], F32, (0.5, 1.5)),
                (
                    "attn_q_b.weight",
                    vec![Q_LORA, HEADS * (NOPE + PE)],
                    Q8_0,
                    small,
                ),
                ("attn_kv_a_mqa.weight", vec![E, KV_LORA + PE], Q6_K, small),
                ("attn_kv_a_norm.weight", vec![KV_LORA], F32, (0.5, 1.5)),
                ("attn_k_b.weight", vec![NOPE, KV_LORA, HEADS], Q8_0, small),
                ("attn_v_b.weight", vec![KV_LORA, HEAD, HEADS], Q6_K, small),
                ("attn_gate.weight", vec![E, HEADS * HEAD], Q8_0, small),
                ("attn_output.weight", vec![HEADS * HEAD, E], out, small),
            ]);
        } else {
            layer.extend([
                ("attn_q.weight", vec![E, INNER], Q6_K, small),
                ("attn_k.weight", vec![E, INNER], Q6_K, small),
                ("attn_v.weight", vec![E, INNER], Q6_K, small),
                (
                    "ssm_conv1d_q.weight",
                    vec![CONV, 1, INNER],
                    F32,
                    (-0.5, 0.5),
                ),
                (
                    "ssm_conv1d_k.weight",
                    vec![CONV, 1, INNER],
                    F32,
                    (-0.5, 0.5),
                ),
                (
                    "ssm_conv1d_v.weight",
                    vec![CONV, 1, INNER],
                    F32,
                    (-0.5, 0.5),
                ),
                ("ssm_f_a.weight", vec![E, RANK], F32, small),
                ("ssm_f_b.weight", vec![RANK, INNER], F32, (-0.1, 0.1)),
                ("ssm_beta.weight", vec![E, HEADS], F32, small),
                ("ssm_rda.weight", vec![E, HEADS], F32, small),
                ("ssm_a", vec![HEADS], F32, (-2.0, -0.5)),
                ("ssm_dt.bias", vec![INNER], F32, (-0.5, 0.5)),
                ("ssm_g.weight", vec![E, INNER], Q6_K, small),
                ("ssm_norm.weight", vec![HEAD], F32, (0.5, 1.5)),
                ("attn_output.weight", vec![INNER, E], Q8_0, small),
            ]);
        }
        layer.push(("ffn_norm.weight", vec![E], F32, (0.5, 1.5)));
        if l == 0 {
            layer.extend([
                ("ffn_gate.weight", vec![E, FF], Q6_K, small),
                ("ffn_up.weight", vec![E, FF], Q6_K, small),
                ("ffn_down.weight", vec![FF, E], Q6_K, small),
            ]);
        } else {
            layer.extend([
                ("ffn_gate_inp.weight", vec![E, N_EXP], F32, (-0.1, 0.1)),
                ("exp_probs_b.bias", vec![N_EXP], F32, (-0.1, 0.1)),
                ("ffn_gate_exps.weight", vec![E, EFF, N_EXP], Q4_K, small),
                ("ffn_up_exps.weight", vec![E, EFF, N_EXP], Q4_K, small),
                ("ffn_down_exps.weight", vec![EFF, E, N_EXP], Q5_0, small),
                ("ffn_gate_shexp.weight", vec![E, EFF], Q6_K, small),
                ("ffn_up_shexp.weight", vec![E, EFF], Q6_K, small),
                ("ffn_down_shexp.weight", vec![EFF, E], Q8_0, small),
            ]);
        }
        t.extend(
            layer
                .into_iter()
                .map(|(s, d, ty, r)| (format!("blk.{l}.{s}"), d, ty, r)),
        );
        for sub in ["attn", "ffn"] {
            t.extend([
                (
                    format!("blk.{l}.hc_{sub}_fn.weight"),
                    vec![STREAMS * E, mix],
                    F32,
                    (-0.05, 0.05),
                ),
                (
                    format!("blk.{l}.hc_{sub}_base.weight"),
                    vec![mix],
                    F32,
                    (-0.1, 0.1),
                ),
                (
                    format!("blk.{l}.hc_{sub}_scale.weight"),
                    vec![4],
                    F32,
                    (0.3, 0.7),
                ),
            ]);
        }
    }
    t
}

/// The file's keys: the widths above, delta layers at 0 and 1 and a latent one at 2, the first
/// layer dense.
fn keys() -> Vec<(String, Value)> {
    let u = |v: usize| Value::U32(u32::try_from(v).expect("a width"));
    let kv: Vec<Value> = (0..LAYERS)
        .map(|l| Value::I32(i32::from(l == LAYERS - 1)))
        .collect();
    let k = |s: &str, v: Value| (format!("raon1.{s}"), v);
    vec![
        (
            gguf::GENERAL_ARCHITECTURE.to_owned(),
            Value::String("raon1".to_owned()),
        ),
        k("block_count", u(LAYERS)),
        k("context_length", u(4096)),
        k("embedding_length", u(E)),
        k("vocab_size", u(VOCAB)),
        k("feed_forward_length", u(FF)),
        k("leading_dense_block_count", u(1)),
        k("attention.head_count", u(HEADS)),
        k("attention.head_count_kv", Value::Array(kv)),
        k("attention.key_length", u(KV_LORA + PE)),
        k("attention.value_length", u(KV_LORA)),
        k("attention.key_length_mla", u(NOPE + PE)),
        k("attention.value_length_mla", u(HEAD)),
        k("attention.q_lora_rank", u(Q_LORA)),
        k("attention.kv_lora_rank", u(KV_LORA)),
        k("attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
        k("attention.latent_norm_rms_epsilon", Value::F32(1e-6)),
        k("rope.dimension_count", u(PE)),
        k("ssm.conv_kernel", u(CONV)),
        k("ssm.time_step_rank", u(RANK)),
        k("kda.head_dim", u(HEAD)),
        k("kda.gate_lower_bound", Value::F32(-5.0)),
        k("expert_count", u(N_EXP)),
        k("expert_used_count", u(N_USED)),
        k("expert_feed_forward_length", u(EFF)),
        k("expert_shared_count", u(1)),
        k("expert_gating_func", u(2)),
        k("expert_weights_scale", Value::F32(1.0)),
        k("expert_weights_norm", Value::Bool(true)),
        k("hyper_connection.count", u(STREAMS)),
        k("hyper_connection.epsilon", Value::F32(1e-5)),
        k("hyper_connection.magnitude", Value::F32(2.0)),
        k("activation.situ_beta", Value::F32(4.0)),
        k("activation.situ_linear_beta", Value::F32(25.0)),
    ]
}

/// The bytes `n` values of type `ty` take in the file.
fn nbytes(ty: GgmlType, n: usize) -> usize {
    match ty {
        GgmlType::F32 => 4 * n,
        GgmlType::Q8_0 => n / 32 * 34,
        GgmlType::Q6_K => n / 256 * 210,
        GgmlType::Q4_K => n / 256 * 144,
        GgmlType::Q5_0 => n / 32 * 22,
        other => panic!("the synthetic file holds no {other:?} tensor"),
    }
}

/// The file of [`tensors`]`(out)` written to a temporary path named by `tag`.
fn write(tag: &str, out: GgmlType) -> PathBuf {
    let mut r = Lcg(0x0073_796e);
    let table = tensors(out);
    let decls: Vec<TensorDecl> = table
        .iter()
        .map(|(name, dims, ty, _)| TensorDecl {
            name: name.clone(),
            dims: dims.iter().map(|&d| d as u64).collect(),
            type_id: ty.as_u32(),
            nbytes: nbytes(*ty, dims.iter().product()) as u64,
        })
        .collect();
    let layout = Layout::new(&keys(), decls).expect("the file's layout");
    let path = std::env::temp_dir().join(format!("raon1-synth-{tag}-{}.gguf", std::process::id()));
    let file = std::io::BufWriter::new(std::fs::File::create(&path).expect("a temporary file"));
    let mut w = Writer::new(file, layout).expect("the file's writer");
    for (name, dims, ty, range) in &table {
        let n: usize = dims.iter().product();
        let bytes = values(&mut r, *ty, n, *range);
        assert_eq!(bytes.len(), nbytes(*ty, n), "{name}");
        w.tensor(name, &bytes).expect("a tensor's bytes");
    }
    w.finish().expect("the file's end");
    path
}

/// The model of the file `write(tag, out)` made, loaded whole on a card; the file is removed once
/// it is open.
fn open(tag: &str, out: GgmlType) -> Result<GpuModel<Body>, GpuError> {
    let path = write(tag, out);
    let split = Split::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
    let m = GpuModel::<Body>::open(gpu, split, CTX);
    let _ = std::fs::remove_file(&path);
    m
}

/// A prompt of `n` ids of the file's vocabulary.
fn prompt(n: usize) -> Vec<u32> {
    (0..n).map(|i| ((7 * i + 3) % VOCAB) as u32).collect()
}

/// A file whose latent layer's output projection is Q6_K, a type the loader admits and the wide
/// arm refuses, loads with the wide arm off, names the layer, and runs a prompt of nine ids and
/// one past a staging group as the rows pass: the argmax and the last logits row of the steps, by
/// bits. The same file with Q8_0 there (Raon's) loads with the wide arm on and runs the nine ids
/// as a wide unit, to finite logits.
#[test]
#[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
fn hw_a_file_the_wide_arm_refuses_runs_every_prompt_as_passes() {
    let mut m = open("q6k", GgmlType::Q6_K).expect("a file the loader admits");
    let body = m.body("synth").expect("the body");
    assert!(
        !body.wide_ok(),
        "a Q6_K output projection keeps the file off the wide arm"
    );
    let why = body.wide_off().expect("the refusal").to_owned();
    assert!(
        why.contains("layer 2's mixer") && why.contains("an output projection of Q6_K"),
        "{why}"
    );
    for n in [9, 70] {
        let ids = prompt(n);
        m.reset().expect("a reset");
        let steps = m.step(&ids).expect("the steps");
        let steps_row = m.logits().expect("the steps' row");
        m.reset().expect("a reset");
        let rows = m.prompt_rows(&ids).expect("a prompt of the rows pass");
        let rows_row = m.logits().expect("the rows' row");
        assert_eq!(steps, rows, "{n} ids: the argmax");
        assert!(
            steps_row.iter().all(|v| v.is_finite()) && same(&steps_row, &rows_row, f32::to_bits),
            "{n} ids: the last logits row by bits"
        );
    }
    drop(m);
    let mut m = open("q8", GgmlType::Q8_0).expect("a file of Raon's types");
    let body = m.body("synth").expect("the body");
    assert!(body.wide_ok(), "{:?}", body.wide_off());
    m.reset().expect("a reset");
    m.prompt_rows(&prompt(9)).expect("nine ids as a wide unit");
    let row = m.logits().expect("the wide row");
    assert!(
        row.iter().all(|v| v.is_finite()),
        "the wide unit's logits are finite"
    );
}
