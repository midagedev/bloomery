//! Where the synthetic DSpark drafts sit beside a qwen35moe fixture: the layout the writer
//! (`model_hf::dspark_fixture`) lays out and the gates that load the drafts read.

/// The reduced-vocabulary draft's directory under the fixture's: its own head, table and id map.
pub const REDUCED_DIR: &str = "dspark/reduced";
/// The full-vocabulary draft's directory under the fixture's: the target's own ids, its head and
/// embedding borrowed.
pub const FULL_DIR: &str = "dspark/full";
/// The mismatched draft's directory under the fixture's: a vocabulary one id short of the
/// target's, refused by shape before any weight is read.
pub const MISMATCHED_DIR: &str = "dspark/mismatched";
/// The renamed target copy's directory under the fixture's: the target's shards with another
/// `general.name`, every tensor byte for byte.
pub const RENAMED_DIR: &str = "renamed";

/// The renamed copy's `general.name`: a fine-tune's, not the source's — the name a draft written
/// for one target must still fit beside.
pub const RENAMED_NAME: &str = "Sage-A1.5 35B A3B";
/// See [`RENAMED_NAME`]: the basename the renamed copy states.
pub const RENAMED_BASENAME: &str = "Sage_A1.5";
