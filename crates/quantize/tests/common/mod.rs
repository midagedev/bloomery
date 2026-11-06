//! Shared by the host gates: a seeded generator, the oracle's calling convention and its location.
#![allow(dead_code, reason = "each gate target uses a part of this module")]

use gguf::quant::GgmlType;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

/// xorshift64*: every input of a gate is seeded, so a gate is bit identity.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in [-1, 1).
    pub fn sym(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// `$BLOOMERY_DATA/bin/iq_quant_ref`, the oracle (`just build-iq-quant-ref`).
pub fn oracle_bin() -> PathBuf {
    let data = std::env::var("BLOOMERY_DATA").expect("BLOOMERY_DATA names the data directory");
    let bin = Path::new(&data).join("bin/iq_quant_ref");
    assert!(
        bin.is_file(),
        "no oracle at {} (just build-iq-quant-ref)",
        bin.display()
    );
    bin
}

/// The oracle's ggml type id.
pub fn ggml_id(ty: GgmlType) -> u32 {
    match ty {
        GgmlType::F32 => 0,
        GgmlType::F16 => 1,
        GgmlType::Q8_0 => 8,
        GgmlType::Q4_K => 12,
        GgmlType::Q5_K => 13,
        GgmlType::Q6_K => 14,
        GgmlType::IQ2_XXS => 16,
        GgmlType::IQ1_S => 19,
        GgmlType::BF16 => 30,
        GgmlType::MXFP4 => 39,
        o => panic!("the oracle has no id for {o:?}"),
    }
}

/// ggml's bytes of `src` (whole rows of `n_per_row` values of `src_ty`) quantized to `dst`, `rows_per_expert` rows an
/// expert, with `qw` (`n_per_row` per expert) or none.
pub fn oracle(
    dir: &Path,
    dst: GgmlType,
    src_ty: GgmlType,
    src: &[u8],
    n_per_row: usize,
    rows_per_expert: usize,
    qw: Option<&[f32]>,
) -> Vec<u8> {
    let row = gguf_row_bytes(src_ty, n_per_row);
    let nrows = src.len() / row;
    let n_exp = qw.map_or(0, |q| q.len() / n_per_row);
    let mut buf = b"IQQ1".to_vec();
    buf.extend_from_slice(&ggml_id(dst).to_le_bytes());
    buf.extend_from_slice(&ggml_id(src_ty).to_le_bytes());
    for v in [n_per_row, nrows, rows_per_expert, n_exp] {
        buf.extend_from_slice(&(v as u64).to_le_bytes());
    }
    buf.extend_from_slice(src);
    for v in qw.unwrap_or(&[]) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let (inp, out) = (dir.join("in.bin"), dir.join("out.bin"));
    std::fs::write(&inp, &buf).unwrap();
    let st = std::process::Command::new(oracle_bin())
        .arg(&inp)
        .arg(&out)
        .status()
        .unwrap();
    assert!(st.success(), "iq_quant_ref {dst:?}: {st}");
    std::fs::read(&out).unwrap()
}

fn gguf_row_bytes(ty: GgmlType, n: usize) -> usize {
    n / ty.blck_size().unwrap() as usize * ty.type_size().unwrap() as usize
}

/// A fresh scratch directory under the temp dir, removed by the caller.
pub fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("bq-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

pub fn f32_bytes(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The K3.1 type map's rules over a fixture or the K3-shaped set: one layer's gate and up experts IQ2_XXS, the stacks
/// IQ1_S, the stems the map holds at Q8_0, F32 and Q6_K, every other trunk weight `stem` (`q4_k`, `q5_k`).
pub fn k31_rules(stem: &str) -> String {
    format!(
        "blk\\.1\\.ffn_(gate|up)_exps\\.weight=IQ2_XXS\n\
         ffn_(down|gate|up)_exps\\.weight=IQ1_S\n\
         \\.attn_k_b\\.weight$=q8_0\n\\.ssm_f_b\\.weight$=q8_0\n\\.ssm_beta\\.weight$=f32\n\\.attn_v_b\\.weight$=q6_k\n\
         \\.(attn_gate|attn_k|attn_kv_a_mqa|attn_output|attn_q|attn_q_a|attn_q_b|attn_v|ffn_down|ffn_gate|ffn_up|ffn_down_shexp|ffn_gate_shexp|ffn_up_shexp|ffn_routed_down|ffn_routed_up|ssm_f_a|ssm_g)\\.weight$={stem}\n"
    )
}

/// Run `bin args`, panicking with its stderr tail when it ends non-zero; its wall in seconds and its stderr.
pub fn run_ok(bin: &Path, args: &[&OsStr]) -> (f64, String) {
    let t = Instant::now();
    let o = Command::new(bin).args(args).output().unwrap();
    let s = t.elapsed().as_secs_f64();
    let err = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(
        o.status.success(),
        "{} ended {}: {}",
        bin.display(),
        o.status,
        err.lines().rev().take(8).collect::<Vec<_>>().join(" | ")
    );
    (s, err)
}

/// The fixture as a source the host rules quantize from: the routed stacks MXFP4 as in the real source, every other
/// weight BF16 (the first rule that matches wins), written by llama-quantize into `work/src.gguf`.
pub fn fixture_src(lq: &Path, fixture: &Path, work: &Path) -> PathBuf {
    let src = work.join("src.gguf");
    let rules = work.join("rules-src.txt");
    std::fs::write(
        &rules,
        "ffn_(down|gate|up)_exps\\.weight=mxfp4\n\\.weight$=bf16\n",
    )
    .unwrap();
    run_ok(
        lq,
        &[
            "--allow-requantize".as_ref(),
            "--tensor-type-file".as_ref(),
            rules.as_os_str(),
            fixture.as_os_str(),
            src.as_os_str(),
            "Q4_K_M".as_ref(),
            "16".as_ref(),
        ],
    );
    src
}
