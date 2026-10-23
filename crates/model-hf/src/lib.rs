//! What the CPU model description does with the Hugging Face hub and its checkpoints, kept out of
//! `model` so that crate, and every gate that links it, does not carry the hf crate: [`remote`]
//! reads a fixture's source headers from a repo by HTTP Range (`fixture fetch`, the `fixture` bin of
//! this crate), [`arch::glm5next`] maps GLM-5.3-Flash's checkpoint onto the glm5next program's names
//! and layouts, [`arch::qwen35moe`] maps Qwen3.8-Flash-Next's onto qwen4exp's and Qwen3.6's MLX
//! folder onto qwen35moe's, the `prepare` bin writes a GGUF split from such a checkpoint folder,
//! and [`part`] reads a load's tensors, whole or a rank's part, from a checkpoint folder as
//! `model::part` reads them from a GGUF file.

pub mod arch;
pub mod part;
pub mod remote;

#[cfg(test)]
mod test_support {
    /// A fresh folder under the system's temporary one.
    pub(crate) fn temp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("model-hf-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        dir
    }
}
