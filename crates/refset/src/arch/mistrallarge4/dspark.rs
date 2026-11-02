//! Mistral Large 4's DSpark draft calls: upstream llama.cpp's own run of the draft file ([`DRAFT`]) beside the target
//! ([`super::MODEL`]), the draft's proposals call by call on each prompt of `tools/ref/ml4-dspark-prompts.txt` with the rows it was
//! fed ([`crate::lcppdspark`] holds the set's format). `tools/ref/dump-ml4-dspark.sh` is the writer (`just dump-ml4-dspark`); the
//! tree is llama.cpp with `tools/ref/ml4-dspark-dump.patch`, a CUDA build (`just build-ml4-dspark-lcpp`), which is not the node
//! dumps' tree ([`super::COMMIT`]): the draft's graph is in the commit [`BUILD`] names. `gate-gpu-mistrallarge4-dspark` reads the
//! set.

use super::{ARCH, model};
use crate::RefError;
use crate::family::{Build, Family, Identity};

/// The DSpark draft file beside the UD-IQ1_S target ([`super::MODEL`]), the identity each set's `# draft_model` line states: the
/// default of the profile's `DSPARK` (`tools/ref/models/mistral-large4.sh`). A set of the file named by another path is stale by
/// this line, not by its name.
pub const DRAFT: &str =
    "/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/dspark-Mistral-Large-4-1T-A52B-Q8_0.gguf";

/// The build every set names: the first nine digits of the oracle tree's commit, and the md5 of the patch the tree is built with.
// PIN(2026-11-02): `LCPP_BUILD_PIN` and `LCPP_BUILD_PATCH` of `just build-ml4-dspark-lcpp` (`the_build_is_the_trees_pin_and_the_
// patchs_md5` holds both), the patch's `md5sum` read once. A tree at another commit or with another patch moves the build, and
// the set is dumped again.
pub const BUILD: &str = "785322d17+patch-d2c7aee326e00f915b8e263df031bb56";

/// The set, under [`crate::data_dir`]: the draft sets' root (`ref-draft`, as ik's DSpark sets), a directory of its own.
pub const SET: &str = "ref-draft/ml4_dspark";

/// [`DRAFT`], as the family's `draft_runs`.
fn draft() -> Result<String, RefError> {
    Ok(DRAFT.to_string())
}

/// The draft's calls on the prompts, from the UD-IQ1_S target and its draft file, by the patched llama.cpp tree: read by the
/// draft's gate, which checks its first proposal on each prompt against upstream's.
pub static CALLS: Family = Family {
    name: "lcpp-dspark-mistrallarge4",
    sets: &[SET],
    resolve: None,
    recipe: "just dump-ml4-dspark",
    identity: Identity::LcppDsparkCalls,
    arch: Some(ARCH),
    build: Some(Build::Is(BUILD)),
    runs: Some(model),
    draft_runs: Some(draft),
    consumers: &["gate-gpu-mistrallarge4-dspark"],
};

#[cfg(test)]
mod tests {
    use super::{BUILD, CALLS, DRAFT, SET};
    use crate::RefError;
    use crate::arch::mistrallarge4::{ARCH, LCPP_BUILD, MODEL, MODEL_MIX256G};
    use crate::family::{Build, Identity, Provenance};
    use crate::lcppdspark::testkit::{Own, edit_manifest, fresh, remove, write_set};
    use std::path::Path;

    /// What the family's sets state.
    const OWN: Own<'static> = Own {
        model: MODEL,
        draft: DRAFT,
        build: BUILD,
        arch: ARCH,
    };

    /// The family's check of a set that states `own`, finished or not, in a directory of its own.
    fn check(what: &str, own: &Own<'_>, complete: bool) -> Result<Provenance, RefError> {
        let dir = fresh("lcpp-dspark-mistrallarge4", what);
        write_set(&dir, own, complete);
        let r = CALLS.check_set(&dir);
        remove(&dir);
        r
    }

    /// `r` is the stale refusal of a set that states `dumped_from` where the tree runs `runs`.
    fn stale(r: Result<Provenance, RefError>, dumped_from: &str, runs: &str) {
        match r {
            Err(e @ RefError::Stale { .. }) => {
                let e = e.to_string();
                assert!(
                    e.contains(&format!("dumped from {dumped_from}"))
                        && e.contains(&format!("the tree runs {runs}")),
                    "{e}"
                );
            }
            other => panic!("a set of {dumped_from}: {other:?}"),
        }
    }

    /// `r` is the foreign refusal of a set that states `got` for `field`.
    fn foreign(r: Result<Provenance, RefError>, field: &str, got: &str, want: &str) {
        match r {
            Err(RefError::Foreign {
                field: f,
                family,
                got: g,
                want: w,
                ..
            }) if f == field && family == CALLS.name && g == got && w == want => {}
            other => panic!("a set of {field} {got}: {other:?}"),
        }
    }

    /// The repository's file `rel`, as bytes.
    fn repo_bytes(rel: &str) -> Vec<u8> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(rel);
        std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    /// The repository's text file `rel`.
    fn repo(rel: &str) -> String {
        String::from_utf8(repo_bytes(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
    }

    /// A complete set of the family's files, build and architecture passes, and states them.
    #[test]
    fn the_family_takes_a_set_of_its_files_build_and_arch() -> Result<(), RefError> {
        let p = check("own", &OWN, true)?;
        assert_eq!(p.dumped_from, MODEL);
        assert_eq!(p.draft.as_deref(), Some(DRAFT));
        assert_eq!(p.build.as_deref(), Some(BUILD));
        Ok(())
    }

    /// A set of another target file is stale: the Mix-256G file and another quant (so only the `# model` line tells them
    /// apart), a shard that is not the first, the first shard's name in another directory, and the draft's own path.
    #[test]
    fn the_family_refuses_a_set_of_another_target_file() {
        for other in [
            MODEL_MIX256G,
            "/models/Mistral-Large-4-1T-A52B-UD-IQ1_M/Mistral-Large-4-1T-A52B-UD-IQ1_M-00001-of-00007.gguf",
            "/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00002-of-00006.gguf",
            "/models/elsewhere/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf",
            DRAFT,
        ] {
            let own = Own {
                model: other,
                ..OWN
            };
            stale(check("target", &own, true), other, MODEL);
        }
    }

    /// A set of another draft file is stale: another quant of the draft, the draft's name in another directory, and the target's
    /// path.
    #[test]
    fn the_family_refuses_a_set_of_another_draft_file() {
        for other in [
            "/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/dspark-Mistral-Large-4-1T-A52B-Q4_0.gguf",
            "/models/elsewhere/dspark-Mistral-Large-4-1T-A52B-Q8_0.gguf",
            MODEL,
        ] {
            let own = Own {
                draft: other,
                ..OWN
            };
            stale(check("draft", &own, true), other, DRAFT);
        }
    }

    /// A set dumped by another llama.cpp tree is foreign by its build: the commit without the patch, another patch, another
    /// commit with this patch, the node dumps' build, and this build with a suffix.
    #[test]
    fn the_family_refuses_a_set_of_another_build() {
        let (commit, patch) = BUILD
            .split_once("+patch-")
            .expect("a build of a commit and a patch");
        let others = [
            commit.to_string(),
            format!("{commit}+patch-{}", "0".repeat(patch.len())),
            format!("78feec062+patch-{patch}"),
            LCPP_BUILD.to_string(),
            format!("{BUILD}+x"),
        ];
        for other in others {
            let own = Own {
                build: &other,
                ..OWN
            };
            foreign(check("build", &own, true), "build", &other, BUILD);
        }
    }

    /// A set whose `# arch` names another architecture is foreign by its arch field.
    #[test]
    fn the_family_refuses_a_set_of_another_arch() {
        for other in ["mistral_large4", "muse-spark", "deepseek41"] {
            let own = Own { arch: other, ..OWN };
            foreign(check("arch", &own, true), "arch", other, ARCH);
        }
    }

    /// A set without its completion trailer is unfinished, by the set's name.
    #[test]
    fn the_family_refuses_a_set_without_its_trailer() {
        let dir = fresh("lcpp-dspark-mistrallarge4", "trailer");
        write_set(&dir, &OWN, false);
        match CALLS.check_set(&dir) {
            Err(RefError::Unfinished { set }) => assert_eq!(set, dir.display().to_string()),
            other => panic!("a set without its trailer: {other:?}"),
        }
        remove(&dir);
    }

    /// A set that states none of the four identity lines is refused by that line's name: nothing says which file, tree or
    /// architecture it came from.
    #[test]
    fn the_family_refuses_a_set_that_states_no_identity_line() {
        for (line, own_value) in [
            ("# model", MODEL),
            ("# draft_model", DRAFT),
            ("# build", BUILD),
            ("# arch", ARCH),
        ] {
            let dir = fresh("lcpp-dspark-mistrallarge4", "unstated");
            write_set(&dir, &OWN, true);
            edit_manifest(&dir, &format!("{line}\t{own_value}\n"), "");
            let r = CALLS.check_set(&dir);
            remove(&dir);
            match (line, r) {
                ("# model", Err(RefError::Stale { dumped_from, .. }))
                | ("# draft_model", Err(RefError::Stale { dumped_from, .. })) => {
                    assert!(
                        dumped_from.contains("unstated") && dumped_from.contains(line),
                        "{dumped_from}"
                    );
                }
                (
                    "# build",
                    Err(RefError::Foreign {
                        field: "build",
                        got,
                        ..
                    }),
                ) => {
                    assert_eq!(got, "(no build line)");
                }
                (
                    "# arch",
                    Err(RefError::Foreign {
                        field: "arch", got, ..
                    }),
                ) => {
                    assert_eq!(got, "(no # arch line)");
                }
                (line, other) => panic!("a set with no {line} line: {other:?}"),
            }
        }
    }

    /// The text after `key` in `body`, up to the next whitespace.
    fn word_after<'a>(body: &'a str, key: &str) -> &'a str {
        let rest = &body[body.find(key).unwrap_or_else(|| panic!("no {key}")) + key.len()..];
        &rest[..rest.find(char::is_whitespace).unwrap_or(rest.len())]
    }

    /// The indented lines of `recipe` in `justfile`, joined.
    fn recipe_body(justfile: &str, recipe: &str) -> String {
        let mut lines = justfile.lines();
        lines
            .find(|l| l.strip_prefix(recipe).is_some_and(|r| r.starts_with(':')))
            .unwrap_or_else(|| panic!("the justfile has no recipe {recipe}"));
        lines
            .take_while(|l| l.is_empty() || l.starts_with(char::is_whitespace))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The build is the oracle tree's: its commit is the first nine digits of the pin the tree's recipe builds, and the md5 is of
    /// the patch that recipe applies, as the file holds it.
    #[test]
    fn the_build_is_the_trees_pin_and_the_patchs_md5() {
        let body = recipe_body(&repo("justfile"), "build-ml4-dspark-lcpp");
        let pin = word_after(&body, "LCPP_BUILD_PIN=");
        let patch = word_after(&body, "LCPP_BUILD_PATCH=");
        let (commit, md5) = BUILD
            .split_once("+patch-")
            .expect("a build of a commit and a patch");
        assert_eq!(pin.len(), 40, "the pin is a whole commit: {pin}");
        assert_eq!(
            &pin[..commit.len()],
            commit,
            "the pin {pin}, the build's commit"
        );
        assert_eq!(commit.len(), 9);
        assert_eq!(patch, "tools/ref/ml4-dspark-dump.patch");
        assert_eq!(
            crate::md5::hex_of(&repo_bytes(patch)),
            md5,
            "the md5 of {patch}"
        );
    }

    /// The draft and the target the family runs are the profile's defaults, which the gate's recipe and the writer open.
    #[test]
    fn the_files_are_the_profiles_defaults() {
        let profile = repo("tools/ref/models/mistral-large4.sh");
        assert!(
            profile.contains(&format!("MODEL=${{BLOOMERY_REF_MODEL:-{MODEL}}}")),
            "the profile's MODEL is not {MODEL}"
        );
        assert!(
            profile.contains(&format!("DSPARK=${{BLOOMERY_REF_DSPARK:-{DRAFT}}}")),
            "the profile's DSPARK is not {DRAFT}"
        );
    }

    /// The row is the writer's and the gate's: the arch table holds it, its set is the writer's default directory, its recipe and
    /// consumer are recipes of the justfile, and the writer states every line the reader needs.
    #[test]
    fn the_row_is_what_the_writer_and_the_gate_use() {
        assert!(crate::arch::named(CALLS.name).is_some_and(|f| std::ptr::eq(f, &CALLS)));
        assert_eq!(CALLS.identity, Identity::LcppDsparkCalls);
        assert_eq!(CALLS.arch, Some(ARCH));
        assert_eq!(CALLS.build, Some(Build::Is(BUILD)));
        assert_eq!(CALLS.sets, [SET]);
        assert_eq!(CALLS.runs.map(|f| f().ok()), Some(Some(MODEL.to_string())));
        assert_eq!(
            CALLS.draft_runs.map(|f| f().ok()),
            Some(Some(DRAFT.to_string()))
        );
        let justfile = repo("justfile");
        let recipe = CALLS.recipe.strip_prefix("just ").expect("a just recipe");
        for name in std::iter::once(recipe).chain(CALLS.consumers.iter().copied()) {
            assert!(
                justfile.lines().any(|l| l.starts_with(&format!("{name}:"))),
                "the justfile has no recipe {name}"
            );
        }
        let writer = repo("tools/ref/dump-ml4-dspark.sh");
        assert!(
            writer.contains(&format!("$BLOOMERY_DATA/{SET}")),
            "the writer's default directory is not {SET}"
        );
        for key in [
            "# model",
            "# draft_model",
            "# build",
            "# arch",
            "# n_max",
            "# prompt columns",
            "prompt",
            "# complete",
        ] {
            assert!(
                writer.contains(&format!("'{key}\\t")),
                "the writer states no `{key}` line"
            );
        }
    }
}
