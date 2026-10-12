//! gate_qwen3moe_types — every routed type a qwen3moe file may carry plans
//! under the `--place`-unset defaults, before any load.
//!
//! For each GGML type the host's routed kernels run here (`qdot::supports`:
//! the types a host expert of any file runs, the card kernel table's among
//! them), a crafted two-layer qwen3moe header whose every routed stack is of
//! that type ([`routed_header`]: the chain's smallest whole super-block widths,
//! every other tensor a Q4_K or F32 one of its chain shape, so nothing but the
//! routed type moves the answer), written to a temp directory and read by the
//! planner's own header reads on this card's census. No weight is uploaded and
//! no model file is opened, so the fixture tier runs it as the real tier does
//! (tools/ref/ref-paths.sh's qwen3moe row is `self`).
//!
//! - (a) the CLI's default (`q3types::cli_default`: `q3place::unplaced_qwen3`
//!   at the CLI's 4096) answers, and holds to what the card runs: the whole
//!   load only where `Qwen3moeModel::whole_load` takes the file, and card
//!   experts only in layers whose routed stacks the card kernel table runs
//!   (the load refuses by name a layer holding card experts of any other).
//! - (b) the serve seat's default (`q3types::serve_default`: the `--ctx`
//!   search under `q3place::SERVE_RULE` for two slots, then the load it
//!   decides at the total) answers and holds as (a).
//! - (c) the whole load's refusal of a Q8_0 gate·up is the load's own text
//!   ([`REFUSED_TEXT`]).
//!
//! The header is written over a sparse file of its full length. Every type
//! prints one line per arm; a refusal or a break of the rule is red by name
//! with the planner's text, and the gate goes on to the next type.
//! FAIL-first: with `header_plan` reading the routed stacks through the whole
//! load's entry check (`header_stacks`), (a) and (b) are red on every type but
//! Q4_K and Q5_K.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen3moe_types", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/qwen3moe_types.rs"]
mod q3types;

#[cfg(feature = "gpu")]
mod gate {
    use super::q3types;
    use bloomery_gpu::Qwen3moeModel;
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use gguf::quant::GgmlType;
    use std::path::PathBuf;

    /// The load's refusal text of a Q8_0 gate·up, as the whole load's header
    /// read names it (the layer wrap around the entry's own `what`-tagged
    /// error).
    const REFUSED_TEXT: &str = "qwen3moe::Body::load: layer 0: qwen3moe::Body::load: a Q8_0 \
                                gate·up, which the card kernel table names no entry for here";

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let types: Vec<GgmlType> = (0..64)
            .map(GgmlType::from_u32)
            .filter(|&t| !matches!(t, GgmlType::Unknown(_)) && qdot::supports(t))
            .collect();
        println!(
            "types: {} routed types the host runs: {types:?}",
            types.len()
        );
        let mut ok = true;
        for &ty in &types {
            ok &= one_type(ty)?;
        }
        ok &= refused_text()?;
        println!("gate_qwen3moe_types: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }

    /// (a) and (b) for routed stacks of `ty`, one line each; a refusal is
    /// printed with the planner's text and counts red.
    fn one_type(ty: GgmlType) -> Result<bool, GateError> {
        let path = routed_header(ty)?;
        let split = gguf::Split::open(&path)?;
        let arms = [
            ("cli", q3types::cli_default(&split)),
            ("serve", q3types::serve_default(&split)),
        ];
        let _ = std::fs::remove_file(&path);
        let mut ok = true;
        for (arm, r) in arms {
            match r {
                Ok(how) => println!("types: {ty:?} {arm}: {how} {}", verdict(true)),
                Err(e) => {
                    println!("types: {ty:?} {arm}: {e} {}", verdict(false));
                    ok = false;
                }
            }
        }
        Ok(ok)
    }

    /// (c): the whole load's refusal of a Q8_0 gate·up is the load's own text.
    fn refused_text() -> Result<bool, GateError> {
        let path = routed_header(GgmlType::Q8_0)?;
        let split = gguf::Split::open(&path)?;
        let text = match Qwen3moeModel::whole_load(&split) {
            Ok(_) => "the whole load read it".to_owned(),
            Err(e) => e.to_string(),
        };
        let _ = std::fs::remove_file(&path);
        let ok = text == REFUSED_TEXT;
        println!(
            "types: the whole load's Q8_0 refusal: {text} {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// A qwen3moe header whose every layer's routed stacks are of `ty`,
    /// written to a fresh temp directory: two layers of the chain's smallest
    /// whole super-block widths (hidden 256, expert width 256), the flash's
    /// head 128 at group 8 (one key head of the eight query heads), the
    /// router's own shape of 128 experts of eight used as an F32 router
    /// weight, each routed stack one expert's matrix by 128 (the expert count
    /// its last dim, as the planner reads a stack), and every other tensor of
    /// the same names a Q4_K or F32 one of its chain shape. A type with no
    /// block info is refused by name.
    fn routed_header(ty: GgmlType) -> Result<PathBuf, GateError> {
        const LAYERS: usize = 2;
        let (q4_k, routed, f32_id) = (GgmlType::Q4_K.as_u32(), ty.as_u32(), GgmlType::F32.as_u32());
        let u32v = |k: &str, v: u32| (format!("qwen3moe.{k}"), gguf::Value::U32(v));
        let kvs = vec![
            (
                gguf::GENERAL_ARCHITECTURE.to_owned(),
                gguf::Value::String("qwen3moe".to_owned()),
            ),
            u32v("block_count", LAYERS as u32),
            u32v("embedding_length", 256),
            u32v("attention.head_count", 8),
            u32v("attention.head_count_kv", 1),
            u32v("attention.key_length", 128),
            u32v("attention.value_length", 128),
            (
                "qwen3moe.attention.layer_norm_rms_epsilon".to_owned(),
                gguf::Value::F32(1e-6),
            ),
            (
                "qwen3moe.rope.freq_base".to_owned(),
                gguf::Value::F32(1_000_000.0),
            ),
            u32v("context_length", 4096),
            u32v("expert_count", 128),
            u32v("expert_used_count", 8),
            u32v("expert_feed_forward_length", 256),
            u32v("vocab_size", 16),
        ];
        let tensor =
            |name: &str, dims: &[u64], ty: u32| -> Result<gguf::write::TensorDecl, GateError> {
                let (_, blck, tsz) = gguf::ggml_type_info(ty)
                    .ok_or_else(|| format!("tensor {name}: ggml type {ty} has no block info"))?;
                let bytes = dims[1..]
                    .iter()
                    .try_fold(tsz * (dims[0] / blck), |acc, d| acc.checked_mul(*d))
                    .ok_or_else(|| format!("tensor {name}: its bytes overflow u64"))?;
                Ok(gguf::write::TensorDecl {
                    name: name.to_owned(),
                    dims: dims.to_vec(),
                    type_id: ty,
                    nbytes: bytes,
                })
            };
        let exps = |l: usize, part: &str| format!("blk.{l}.ffn_{part}_exps.weight");
        let mut tensors = vec![
            tensor("token_embd.weight", &[256, 16], q4_k)?,
            tensor("output_norm.weight", &[256, 1], f32_id)?,
            tensor("output.weight", &[256, 16], q4_k)?,
        ];
        for l in 0..LAYERS {
            let per_layer = [
                (format!("blk.{l}.attn_norm.weight"), vec![256, 1], f32_id),
                (format!("blk.{l}.attn_q.weight"), vec![256, 1024], q4_k),
                (format!("blk.{l}.attn_k.weight"), vec![256, 128], q4_k),
                (format!("blk.{l}.attn_v.weight"), vec![256, 128], q4_k),
                (format!("blk.{l}.attn_q_norm.weight"), vec![128, 1], f32_id),
                (format!("blk.{l}.attn_k_norm.weight"), vec![128, 1], f32_id),
                (format!("blk.{l}.attn_output.weight"), vec![1024, 256], q4_k),
                (format!("blk.{l}.ffn_norm.weight"), vec![256, 1], f32_id),
                (
                    format!("blk.{l}.ffn_gate_inp.weight"),
                    vec![256, 128],
                    f32_id,
                ),
                (exps(l, "gate"), vec![256, 256, 128], routed),
                (exps(l, "up"), vec![256, 256, 128], routed),
                (exps(l, "down"), vec![256, 256, 128], routed),
            ];
            for (name, dims, ty) in per_layer {
                tensors.push(tensor(&name, &dims, ty)?);
            }
        }
        let layout = gguf::write::Layout::new(&kvs, tensors).map_err(|e| e.to_string())?;
        let dir = std::env::temp_dir().join(format!("bloomery-q3types-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("routed-{ty:?}.gguf"));
        // The header, then the file's full length as a sparse run of zeros:
        // the bytes a written zero data section holds, none of them stored.
        let len = layout.file_len();
        let mut file = std::fs::File::create(&path)?;
        drop(gguf::write::Writer::new(&mut file, layout).map_err(|e| e.to_string())?);
        file.set_len(len)?;
        Ok(path)
    }
}
