//! The MLX-format fixture split the Metal gates read: the qwen35moe fixture
//! written as an MLX folder and converted back to a GGUF split
//! (`tools/mlx-fixture.py`), every typed site MLX_A4 and the routers MLX_A8,
//! in [`DIR`] under the fixture root (`BLOOMERY_FIXTURE_ROOT`).

use std::path::PathBuf;

/// The split's directory under the fixture root.
pub const DIR: &str = "qwen35moe-mlxsplit";

/// The split's first shard: the one `*-00001-of-*.gguf` in [`DIR`] under the
/// fixture root, through the fixture's own resolver.
pub fn path() -> Result<PathBuf, String> {
    let root = refset::fixture::root().map_err(|e| format!("the fixture root: {e}"))?;
    refset::fixture::first_shard_at(
        &format!("{root}/{DIR}"),
        "`tools/mlx-fixture.py make` writes it from the qwen35moe fixture",
    )
    .map(PathBuf::from)
    .map_err(|e| e.to_string())
}
