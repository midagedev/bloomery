//! What a server needs of Clef's release to seat it: the route its requests are posted to, the
//! context its encoder fills, the backbone architecture its head reads, the repo and file names of its
//! head, and how a head config is known to be Clef's. The server's own table holds one row of these.

use crate::encode::MAX_LENGTH;
use crate::head::{CONFIG_FILE, HeadConfig};

/// The row's name, as a server's refusals and `/props` give it.
pub const NAME: &str = "clef";

/// The routes a SystemOne request is posted to: the release's.
pub const ROUTES: &[&str] = &["/v1/systemone"];

/// The context unless the server is told otherwise: the release's `max_length`.
pub const CTX: usize = MAX_LENGTH;

/// The file architectures the head reads the hidden states of: Qwen3.5 dense, Clef's backbone.
pub const BACKBONES: &[&str] = &["qwen35"];

/// The repo the release's head is fetched from.
pub const HEAD_REPO: &str = "Cloudflare/clef-flash";

/// A repo whose model card names [`HEAD_REPO`] as the model it quantizes, whose
/// set seats the row under `--hf`: the `--hf` a bare `qwen35` file's refusal
/// points at.
pub const QUANT_REPO: &str = "bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M";

/// The head's weights in [`HEAD_REPO`]; its config is [`HEAD_CONFIG`] beside them.
pub const HEAD_FILE: &str = "joint_head.safetensors";

/// The head config's file name beside the weights.
pub const HEAD_CONFIG: &str = CONFIG_FILE;

/// The file architectures that carry Clef in a layout this engine does not read, with what a
/// server's refusal says: llama.cpp's `clef` (the head inside the GGUF, `ggml-org/Clef-Flash-GGUF`).
pub const UNSERVED: &[(&str, &str)] = &[(
    "clef",
    "llama.cpp's Clef layout (the head inside the GGUF, as ggml-org/Clef-Flash-GGUF publishes it), \
     which this server does not serve yet; serve bartowski's quantization with the release's head: \
     bloomery-serve --hf bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M",
)];

/// Whether the text of a head config is Clef's ([`HeadConfig::recognise`]), or why not.
pub fn knows(config: &str) -> Result<(), String> {
    HeadConfig::recognise(config).map_err(|e| e.to_string())
}
