//! Read a dense Qwen file through the native header reader and coverage
//! table without opening a GPU. The table checks every declared feature,
//! including optional chat parsers. Passing is host coverage evidence;
//! device loading and generated tokens need separate gates.

use bloomery_gpu_gates::{GateError, exit_with, qwen_seat};
use gguf::Split;
use model::arch::qwen35moe::hparams::Hparams;
use model::arch::{coverage, spec};
use model::placement::Role;

fn run() -> Result<(), GateError> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [path] = args.as_slice() else {
        return Err("usage: qwen35_admission <dense-qwen35.gguf>".into());
    };
    let file = Split::open(path)?;
    let arch = file.architecture().unwrap_or("<missing>");
    if qwen_seat::Family::from_name(arch) != Some(qwen_seat::Family::Qwen35) {
        return Err(format!("{path}: expected dense qwen35, got {arch}").into());
    }
    let hp = Hparams::read(&file)?;
    let mut types = std::collections::BTreeMap::new();
    for (_, tensor) in file.iter_tensors() {
        *types.entry(format!("{:?}", tensor.ty)).or_insert(0_usize) += 1;
    }
    println!("tensor_types={types:?}");
    println!(
        "metadata arch={arch} layers={} trunk={} embedding={} heads={} kv_heads={} head={} vocab={} experts={} flash_mma={}",
        hp.n_layer,
        hp.n_trunk,
        hp.n_embd,
        hp.n_head,
        hp.n_head_kv,
        hp.head_dim,
        hp.n_vocab,
        hp.n_expert,
        qwen_seat::decode_mma(hp.head_dim, hp.n_head / hp.n_head_kv)?
    );
    let read = spec(&file)?;
    let unused = read
        .tensors
        .tensors
        .iter()
        .filter(|t| t.role == Role::Unused)
        .count();
    let items = coverage::check(&read.spec, &read.tensors);
    println!(
        "coverage unsupported={} unused_tensors={unused}",
        items.len()
    );
    for item in &items {
        println!("unsupported layer={:?} {}", item.layer, item.feature);
    }
    if !items.is_empty() {
        return Err("the native coverage table refuses this file; no device was opened".into());
    }
    println!("host_admission=pass device_load=unverified token_correctness=unverified");
    Ok(())
}

fn main() -> std::process::ExitCode {
    exit_with("qwen35_admission", run())
}
