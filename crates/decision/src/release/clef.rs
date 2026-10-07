//! What a server needs of Clef's release to seat it: the route its requests are posted to, the
//! context its encoder fills, the backbone architectures its head reads and which of them carry the
//! head inside the file, the repo and file names of its head, and how a head config is known to be
//! Clef's. The server's own table holds one row of these.

use crate::encode::MAX_LENGTH;
use crate::head::{CONFIG_FILE, HeadConfig};

/// The row's name, as a server's refusals and `/props` give it.
pub const NAME: &str = "clef";

/// The routes a SystemOne request is posted to: the release's.
pub const ROUTES: &[&str] = &["/v1/systemone"];

/// The context unless the server is told otherwise: the release's `max_length`.
pub const CTX: usize = MAX_LENGTH;

/// The file architectures the head reads the hidden states of: Qwen3.5 dense, Clef's backbone,
/// in the two layouts a GGUF of it comes in ([`IN_FILE`] is the one that holds its head).
pub const BACKBONES: &[&str] = &["qwen35", "clef"];

/// The architecture of llama.cpp's Clef layout, whose file carries the head itself.
pub const LAYOUT_ARCH: &str = "clef";

/// The [`BACKBONES`] whose file carries the head itself, so the seat reads it from the model file
/// ([`crate::head::ClefHead::from_gguf`]) and a head file is not asked for: llama.cpp's `clef`
/// layout. The other backbones (`qwen35`) take the release's head file.
pub const IN_FILE: &[&str] = &[LAYOUT_ARCH];

/// The repo the release's head is fetched from.
pub const HEAD_REPO: &str = "Cloudflare/clef-flash";

/// A repo whose model card names [`HEAD_REPO`] as the model it quantizes, whose
/// set seats the row under `--hf`: the `--hf` a bare `qwen35` file's refusal
/// points at. Its current files are in the `clef` layout.
pub const QUANT_REPO: &str = "bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M";

/// The head's weights in [`HEAD_REPO`]; its config is [`HEAD_CONFIG`] beside them.
pub const HEAD_FILE: &str = "joint_head.safetensors";

/// The head config's file name beside the weights.
pub const HEAD_CONFIG: &str = CONFIG_FILE;

/// The file architectures that carry Clef in a layout this engine does not read, with what a
/// server's refusal says. None: both layouts of the text model ([`BACKBONES`]) are read.
pub const UNSERVED: &[(&str, &str)] = &[];

/// Whether the text of a head config is Clef's ([`HeadConfig::recognise`]), or why not.
pub fn knows(config: &str) -> Result<(), String> {
    HeadConfig::recognise(config).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A layout that carries the head in its file is one the head reads the hidden states of, and
    /// no architecture is both served and listed as unserved.
    #[test]
    fn the_layouts_are_consistent() {
        for a in IN_FILE {
            assert!(
                BACKBONES.contains(a),
                "{a} carries a head and is no backbone"
            );
        }
        for (a, _) in UNSERVED {
            assert!(!BACKBONES.contains(a), "{a} is both served and unserved");
        }
        assert!(IN_FILE.contains(&"clef") && !IN_FILE.contains(&"qwen35"));
    }
}
