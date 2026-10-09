//! The Qwen3.8-Flash-Next families of the fixture tier: ik's dumps on the
//! fixture file ([`crate::fixture::first_shard`]), each mirroring one family of
//! the real file with the same ik tree, the same consumers and a set name that
//! carries the `fx_` prefix. The identity is the fixture's: its first shard
//! and its `# fixture` line ([`crate::family::Identity::FixtureManifest`]).
//! The fixture's MTP draft sits beside its target under the real draft's name
//! ([`super::mtp::DRAFT`]), where a run looks for it.

use super::mtp::{self, DRAFT};
use super::{ARCH, IK_BUILD};
use crate::RefError;
use crate::family::{Build, Family, Identity};
use std::path::Path;

/// [`super::BATCH`] on the fixture.
pub const BATCH: &str = "fx_ref_qwen4exp";
/// [`super::STEP4`] on the fixture.
pub const STEP4: &str = "fx_ref_qwen4exp_step4";
/// [`super::STEP4_EVERY_NODE`] on the fixture.
pub const STEP4_EVERY_NODE: &str = "fx_ref_qwen4exp_step4_every_node";
/// [`super::D1K`] on the fixture.
pub const D1K: &str = "fx_ref_qwen4exp_d1k";
/// [`super::D3K`] on the fixture.
pub const D3K: &str = "fx_ref_qwen4exp_d3k";
/// [`mtp::MTP_SET`] on the fixture.
pub const MTP_SET: &str = "ref-mtp/fx_qwen4exp_prose64_n64_k1";

/// The fixture file the tree runs, as a family's `runs`.
fn runs() -> Result<String, RefError> {
    crate::fixture::first_shard(ARCH)
}

/// The fixture's MTP draft, as the family's `draft_runs`: the file named like
/// [`DRAFT`] in the target's directory, which must be there.
fn draft() -> Result<String, RefError> {
    let target = crate::fixture::first_shard(ARCH)?;
    let (dir, _) = target.rsplit_once('/').ok_or_else(|| {
        RefError::missing(&target, format!("fixture: {target} names no directory"))
    })?;
    let name = DRAFT.rsplit_once('/').map_or(DRAFT, |(_, name)| name);
    let draft = format!("{dir}/{name}");
    if Path::new(&draft).is_file() {
        Ok(draft)
    } else {
        Err(RefError::missing(
            &draft,
            format!(
                "fixture: no draft {draft} beside the target {target} (`fixture generate` \
                 writes it under the real draft's name)"
            ),
        ))
    }
}

/// [`super::IK`] on the fixture.
pub static IK: Family = Family {
    name: "fx-ik-qwen4exp",
    sets: &[BATCH, STEP4, STEP4_EVERY_NODE, D1K, D3K],
    resolve: None,
    recipe: "BLOOMERY_TIER=fixture just dump-ref-fixture qwen4exp [VARIANT]",
    identity: Identity::FixtureManifest,
    arch: Some(ARCH),
    build: Some(Build::Is(IK_BUILD)),
    runs: Some(runs),
    draft_runs: None,
    consumers: super::IK.consumers,
};

/// [`mtp::MTP`] on the fixture.
pub static MTP: Family = Family {
    name: "fx-mtp-qwen4exp",
    sets: &[MTP_SET],
    resolve: None,
    recipe: "BLOOMERY_TIER=fixture just dump-ref-mtp-fixture qwen4exp",
    identity: Identity::FixtureMtpManifest,
    arch: Some(ARCH),
    build: Some(Build::Is(mtp::MTP_BUILD)),
    runs: Some(runs),
    draft_runs: Some(draft),
    consumers: mtp::MTP.consumers,
};
