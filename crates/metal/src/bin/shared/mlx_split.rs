//! The MLX-format fixture splits the Metal gates read: a family's fixture
//! written as an MLX folder and converted back to a GGUF split
//! (`tools/mlx-fixture.py`), every typed site MLX_A4 and the routers MLX_A8,
//! in the family's `<family>-mlxsplit` directory under the fixture root
//! (`BLOOMERY_FIXTURE_ROOT`).

use std::path::PathBuf;

/// The first shard of the split made from the fixture of `family` (a
/// directory name of `tools/fixture-dirs.tsv`): the one `*-00001-of-*.gguf`
/// in `<family>-mlxsplit` under the fixture root, through the fixture's own
/// resolver.
pub fn path(family: &str) -> Result<PathBuf, String> {
    let root = refset::fixture::root().map_err(|e| format!("the fixture root: {e}"))?;
    refset::fixture::first_shard_at(
        &format!("{root}/{family}-mlxsplit"),
        &format!("`tools/mlx-fixture.py make` writes it from the {family} fixture"),
    )
    .map(PathBuf::from)
    .map_err(|e| e.to_string())
}
