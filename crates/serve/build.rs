//! Sets `BLOOMERY_SERVE_COMMIT`, the commit `/props` names in
//! `engine.version`: `BLOOMERY_GIT_COMMIT` when the builder sets it, else
//! `git rev-parse --short=8 HEAD` in this package's tree, else `unknown`. A tree
//! without git (the box's copy carries no `.git`) still builds and says `unknown`.

use std::path::{Path, PathBuf};
use std::process::Command;

/// `git -C dir <args>`'s trimmed stdout, or `None` when git is absent, fails
/// or prints nothing.
fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

/// The files whose change moves HEAD: HEAD itself, the branch ref it names
/// and the packed refs, those that exist (a missing one would rerun this
/// script on every build).
fn head_files(dir: &Path) -> Vec<PathBuf> {
    let mut names = vec!["HEAD".to_owned(), "packed-refs".to_owned()];
    if let Some(r) = git(dir, &["rev-parse", "--symbolic-full-name", "HEAD"]) {
        names.push(r);
    }
    names
        .iter()
        .filter_map(|n| git(dir, &["rev-parse", "--git-path", n]))
        .map(|p| dir.join(p))
        .filter(|p| p.exists())
        .collect()
}

fn main() {
    println!("cargo:rerun-if-env-changed=BLOOMERY_GIT_COMMIT");
    let dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_default());
    let set = std::env::var("BLOOMERY_GIT_COMMIT")
        .ok()
        .map(|c| c.trim().to_owned())
        .filter(|c| !c.is_empty());
    let commit = set
        .or_else(|| {
            let c = git(&dir, &["rev-parse", "--short=8", "HEAD"])?;
            for f in head_files(&dir) {
                println!("cargo:rerun-if-changed={}", f.display());
            }
            Some(c)
        })
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=BLOOMERY_SERVE_COMMIT={commit}");
}
