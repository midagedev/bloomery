//! The mirror sidecar of a split set, as the NVMe tier's reader takes it: `<prefix>.shard-dirs`
//! beside the shards (`<prefix>-00001-of-000NN.gguf`) lists the directories, one absolute path a
//! line, that hold a byte-identical copy of every shard by file name. `#` lines and blank lines
//! are passed over. A reader that opens one handle per copy picks the copy of a unit of work by
//! [`copy_of`], so the units of one expert and of one layer's misses spread over the drives.
//!
//! The format is the one the file-mapping route's sidecar reader takes, so one sidecar serves
//! both. Here a copy is checked by name, length and sampled bytes ([`check_copy`]: no digest at
//! a load, no header compare: the tier reads byte ranges, not tensors).

use std::path::{Path, PathBuf};

/// The suffix of the sidecar's name after the prefix the shards share.
pub const SHARD_DIRS_SUFFIX: &str = ".shard-dirs";

/// A sidecar, a directory it lists or a copy of a shard refused by name.
#[derive(Debug, thiserror::Error)]
#[error("{path}: {why}")]
pub struct ShardDirsError {
    /// The sidecar, the listed directory or the copy.
    pub path: String,
    /// What is wrong with it.
    pub why: String,
}

fn bad(path: &Path, why: impl Into<String>) -> ShardDirsError {
    ShardDirsError {
        path: path.display().to_string(),
        why: why.into(),
    }
}

/// The sidecar's path for the set whose first shard is `first`: the shard's path with its
/// `-00001-of-000NN.gguf` ending replaced by [`SHARD_DIRS_SUFFIX`]. `None` when `first` is not
/// named by that rule (a one-shard set named off it has no sidecar).
#[must_use]
pub fn sidecar_path(first: &Path) -> Option<PathBuf> {
    let name = first.file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf")?;
    let (head, count) = stem.rsplit_once("-of-")?;
    if count.is_empty() || !count.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let prefix = head.strip_suffix("-00001")?;
    if prefix.is_empty() {
        return None;
    }
    Some(first.with_file_name(format!("{prefix}{SHARD_DIRS_SUFFIX}")))
}

/// The directories `text` (the sidecar at `sidecar`) lists for a set whose first shard lives in
/// `first_dir`, in order. Refused by name: a relative entry, one that is missing or no
/// directory, the first shard's own directory, and one listed twice. Paths are compared as
/// written; no link is resolved.
pub fn parse_shard_dirs(
    text: &str,
    sidecar: &Path,
    first_dir: &Path,
) -> Result<Vec<PathBuf>, ShardDirsError> {
    let side = sidecar.display();
    let mut dirs: Vec<PathBuf> = Vec::new();
    for entry in text.lines().map(str::trim) {
        if entry.is_empty() || entry.starts_with('#') {
            continue;
        }
        let dir = PathBuf::from(entry);
        if !dir.is_absolute() {
            return Err(bad(&dir, format!("not an absolute path, listed in {side}")));
        }
        match std::fs::metadata(&dir) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(bad(&dir, format!("missing, listed in {side}")));
            }
            Err(e) => return Err(bad(&dir, format!("{e}, listed in {side}"))),
            Ok(m) if !m.is_dir() => {
                return Err(bad(&dir, format!("not a directory, listed in {side}")));
            }
            Ok(_) => {}
        }
        if dir == first_dir {
            return Err(bad(
                &dir,
                format!("the directory of the first shard itself, listed in {side}"),
            ));
        }
        if dirs.contains(&dir) {
            return Err(bad(&dir, format!("listed twice in {side}")));
        }
        dirs.push(dir);
    }
    Ok(dirs)
}

/// The mirror directories of the set whose first shard is `first`: those its sidecar lists, in
/// the sidecar's order (copy 0 is the set's own directory, copy `c` the `c-1`-th entry). Empty
/// when `first` is named off the shard rule or no sidecar sits beside it; a sidecar that cannot
/// be read or lists a bad entry is refused by name ([`parse_shard_dirs`]).
pub fn shard_dirs(first: &Path) -> Result<Vec<PathBuf>, ShardDirsError> {
    let Some(sidecar) = sidecar_path(first) else {
        return Ok(Vec::new());
    };
    let text = match std::fs::read_to_string(&sidecar) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(bad(&sidecar, format!("the sidecar cannot be read: {e}"))),
    };
    let first_dir = match first.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let first_dir = std::path::absolute(first_dir)
        .map_err(|e| bad(first_dir, format!("the first shard's directory: {e}")))?;
    parse_shard_dirs(&text, &sidecar, &first_dir)
}

/// The path of shard `own`'s copy in `dir`: the same file name.
#[must_use]
pub fn copy_path(own: &Path, dir: &Path) -> PathBuf {
    dir.join(own.file_name().unwrap_or_default())
}

/// The bytes of each sampled window of a copy the load compares.
const SAMPLE_BYTES: usize = 64 << 10;

/// The windows between the first and the last that the load compares, evenly spaced.
const SAMPLE_SPREAD: usize = 6;

/// The `(offset, length)` windows of a file of `len` bytes that [`check_copy`] compares, by
/// ascending offset: the first and the last [`SAMPLE_BYTES`] and one at each of
/// [`SAMPLE_SPREAD`] offsets evenly spaced between. A file shorter than a window is one window.
fn sampled_windows(len: u64) -> Vec<(u64, usize)> {
    if len == 0 {
        return Vec::new();
    }
    let window = |at: u64| (at, (len - at).min(SAMPLE_BYTES as u64) as usize);
    let last = len.saturating_sub(SAMPLE_BYTES as u64);
    let spread =
        (1..=SAMPLE_SPREAD as u64).map(|i| (len / (SAMPLE_SPREAD as u64 + 1) * i).min(last));
    let mut at: Vec<u64> = std::iter::once(0)
        .chain(spread)
        .chain(std::iter::once(last))
        .collect();
    at.sort_unstable();
    at.into_iter().map(window).collect()
}

/// The first byte at which `own` and `copy` differ inside a sampled window of a file of `len`
/// bytes; `None` when every window is equal.
fn first_sampled_difference(own: &Path, copy: &Path, len: u64) -> std::io::Result<Option<u64>> {
    use std::os::unix::fs::FileExt;
    let (a, b) = (std::fs::File::open(own)?, std::fs::File::open(copy)?);
    let (mut x, mut y) = (vec![0u8; SAMPLE_BYTES], vec![0u8; SAMPLE_BYTES]);
    for (at, n) in sampled_windows(len) {
        a.read_exact_at(&mut x[..n], at)?;
        b.read_exact_at(&mut y[..n], at)?;
        if let Some(i) = x[..n].iter().zip(&y[..n]).position(|(p, q)| p != q) {
            return Ok(Some(at + i as u64));
        }
    }
    Ok(None)
}

/// Hold the copy at `copy` to shard `own`: it must be a file of the shard's length whose bytes
/// equal the shard's in the sampled windows ([`sampled_windows`]: the first and last 64 KiB and
/// 64 KiB at each of six offsets between). A missing copy, one of another length and one that
/// differs in a window are refused by name, the last with the first differing offset, never
/// read around. The whole-file digest is the copy script's, not a load's.
pub fn check_copy(own: &Path, copy: &Path) -> Result<(), ShardDirsError> {
    let want = std::fs::metadata(own)
        .map_err(|e| bad(own, format!("its length: {e}")))?
        .len();
    match std::fs::metadata(copy) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(bad(
            copy,
            format!("the mirror of {} is missing", own.display()),
        )),
        Err(e) => Err(bad(copy, e.to_string())),
        Ok(m) if !m.is_file() => Err(bad(copy, "not a file")),
        Ok(m) if m.len() != want => Err(bad(
            copy,
            format!("{} B, the shard {}'s is {want} B", m.len(), own.display()),
        )),
        Ok(_) => match first_sampled_difference(own, copy, want) {
            Err(e) => Err(bad(copy, format!("comparing with {}: {e}", own.display()))),
            Ok(Some(at)) => Err(bad(
                copy,
                format!("differs from the shard {} at byte {at}", own.display()),
            )),
            Ok(None) => Ok(()),
        },
    }
}

/// The copy (of `n`, copy 0 the set's own) that unit `part` of expert `id` reads from:
/// `(id + part) % n`, so the parts of one expert and the experts of one layer spread over the
/// copies. `n` is at least 1.
#[must_use]
pub fn copy_of(id: u32, part: usize, n: usize) -> usize {
    (id as usize + part) % n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch directory named by test, removed first and left for the OS after.
    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("shard-dirs-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn parse(text: &str, first_dir: &Path) -> Result<Vec<PathBuf>, ShardDirsError> {
        parse_shard_dirs(text, Path::new("/s/m.shard-dirs"), first_dir)
    }

    #[test]
    fn the_sidecar_sits_beside_the_first_shard_by_the_shard_rule() {
        assert_eq!(
            sidecar_path(Path::new("/m/K-IQ1_S-00001-of-00012.gguf")),
            Some(PathBuf::from("/m/K-IQ1_S.shard-dirs"))
        );
        for off_rule in [
            "/m/K.gguf",
            "/m/K-00002-of-00012.gguf",
            "/m/-00001-of-00012.gguf",
            "/m/K-00001-of-.gguf",
            "/m/K-00001-of-0x.gguf",
        ] {
            assert_eq!(sidecar_path(Path::new(off_rule)), None, "{off_rule}");
        }
    }

    #[test]
    fn one_dir_and_two_dirs_parse_in_order_past_comments_and_blanks() {
        let own = scratch("parse-own");
        let a = scratch("parse-a");
        let b = scratch("parse-b");
        let one = format!("# the mirror\n\n  {}  \n", a.display());
        assert_eq!(parse(&one, &own).unwrap(), vec![a.clone()]);
        let two = format!("{}\n# between\n{}\n", a.display(), b.display());
        assert_eq!(parse(&two, &own).unwrap(), vec![a, b]);
        assert!(parse("# only a comment\n\n", &own).unwrap().is_empty());
    }

    #[test]
    fn a_bad_entry_is_refused_naming_it() {
        let own = scratch("refuse-own");
        let a = scratch("refuse-a");
        let gone = a.join("gone");
        let err = parse(&gone.display().to_string(), &own).unwrap_err();
        assert_eq!(err.path, gone.display().to_string());
        assert!(err.why.contains("missing"), "{err}");
        assert!(err.why.contains("m.shard-dirs"), "{err}");

        let err = parse("relative/dir", &own).unwrap_err();
        assert_eq!(err.path, "relative/dir");
        assert!(err.why.contains("not an absolute path"), "{err}");

        let file = a.join("f");
        std::fs::write(&file, b"x").unwrap();
        let err = parse(&file.display().to_string(), &own).unwrap_err();
        assert!(err.why.contains("not a directory"), "{err}");

        let err = parse(&own.display().to_string(), &own).unwrap_err();
        assert!(err.why.contains("first shard itself"), "{err}");

        let twice = format!("{0}\n{0}\n", a.display());
        let err = parse(&twice, &own).unwrap_err();
        assert!(err.why.contains("listed twice"), "{err}");
    }

    #[test]
    fn shard_dirs_reads_the_sidecar_beside_the_shard_and_none_without_one() {
        let own = scratch("read-own");
        let mirror = scratch("read-mirror");
        let first = own.join("S-00001-of-00002.gguf");
        assert!(shard_dirs(&first).unwrap().is_empty());
        std::fs::write(
            own.join("S.shard-dirs"),
            format!("# copies\n{}\n", mirror.display()),
        )
        .unwrap();
        assert_eq!(shard_dirs(&first).unwrap(), vec![mirror]);
        // A shard named off the rule has no sidecar, even with one beside it.
        assert!(shard_dirs(&own.join("odd.gguf")).unwrap().is_empty());
    }

    #[test]
    fn a_missing_or_short_copy_is_refused_naming_it() {
        let own = scratch("copy-own");
        let mirror = scratch("copy-mirror");
        let shard = own.join("S-00001-of-00001.gguf");
        std::fs::write(&shard, [7u8; 64]).unwrap();
        let copy = copy_path(&shard, &mirror);
        assert_eq!(copy, mirror.join("S-00001-of-00001.gguf"));

        let err = check_copy(&shard, &copy).unwrap_err();
        assert_eq!(err.path, copy.display().to_string());
        assert!(err.why.contains("missing"), "{err}");

        std::fs::write(&copy, [7u8; 63]).unwrap();
        let err = check_copy(&shard, &copy).unwrap_err();
        assert_eq!(err.path, copy.display().to_string());
        assert!(
            err.why.contains("63 B") && err.why.contains("64 B"),
            "{err}"
        );

        std::fs::write(&copy, [7u8; 64]).unwrap();
        check_copy(&shard, &copy).unwrap();
    }

    /// A shard of `len` pseudo-random bytes and its byte-identical copy, in two scratch dirs.
    fn shard_and_copy(name: &str, len: usize) -> (PathBuf, PathBuf, Vec<u8>) {
        let own = scratch(&format!("{name}-own"));
        let mirror = scratch(&format!("{name}-mirror"));
        let shard = own.join("S-00001-of-00001.gguf");
        let mut x = 0x9E37_79B9u32;
        let bytes: Vec<u8> = (0..len)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (x >> 24) as u8
            })
            .collect();
        std::fs::write(&shard, &bytes).unwrap();
        let copy = copy_path(&shard, &mirror);
        std::fs::write(&copy, &bytes).unwrap();
        (shard, copy, bytes)
    }

    #[test]
    fn equal_copies_pass_and_a_changed_sampled_byte_is_refused_at_its_offset() {
        let len = 1 << 20;
        let (shard, copy, bytes) = shard_and_copy("bytes", len);
        check_copy(&shard, &copy).unwrap();
        let windows = sampled_windows(len as u64);
        assert_eq!(windows.len(), 2 + SAMPLE_SPREAD);
        // The start, a window in the middle of the file, and the very last byte.
        for at in [0, windows[4].0 as usize + 17, len - 1] {
            let mut changed = bytes.clone();
            changed[at] ^= 0xFF;
            std::fs::write(&copy, &changed).unwrap();
            let err = check_copy(&shard, &copy).unwrap_err();
            assert_eq!(err.path, copy.display().to_string());
            assert!(
                err.why.contains(&format!("byte {at}"))
                    && err.why.contains(&shard.display().to_string()),
                "{err}"
            );
        }
        std::fs::write(&copy, &bytes).unwrap();
        check_copy(&shard, &copy).unwrap();
    }

    #[test]
    fn the_sampled_windows_stay_inside_the_file_in_order() {
        assert!(sampled_windows(0).is_empty());
        for len in [
            1u64,
            100,
            SAMPLE_BYTES as u64,
            3 * SAMPLE_BYTES as u64 + 5,
            1 << 30,
        ] {
            let windows = sampled_windows(len);
            assert!(windows.iter().all(|&(at, n)| n > 0 && at + n as u64 <= len));
            assert!(windows.windows(2).all(|w| w[0].0 <= w[1].0), "{len}");
            assert_eq!(windows.first().map(|w| w.0), Some(0));
            assert_eq!(windows.last().map(|w| w.0 + w.1 as u64), Some(len));
        }
    }

    #[test]
    fn the_copy_pick_is_id_plus_part_modulo_the_copies() {
        // One copy: always the set's own.
        for id in 0..5 {
            for part in 0..3 {
                assert_eq!(copy_of(id, part, 1), 0);
            }
        }
        // Two copies: the three parts of one expert land on both, and consecutive experts swap.
        let by_id = |id| [0, 1, 2].map(|p| copy_of(id, p, 2));
        assert_eq!(by_id(0), [0, 1, 0]);
        assert_eq!(by_id(1), [1, 0, 1]);
        assert_eq!(by_id(2), [0, 1, 0]);
        // Three copies: the parts of one expert take three different copies.
        let by_id = |id| [0, 1, 2].map(|p| copy_of(id, p, 3));
        assert_eq!(by_id(0), [0, 1, 2]);
        assert_eq!(by_id(1), [1, 2, 0]);
        assert_eq!(by_id(4), [1, 2, 0]);
        // Over many experts each copy takes its share of the units.
        for n in 1..=3usize {
            let mut count = vec![0usize; n];
            for id in 0..96u32 {
                for part in 0..3 {
                    count[copy_of(id, part, n)] += 1;
                }
            }
            assert!(count.iter().all(|&c| c == 96 * 3 / n), "n={n}: {count:?}");
        }
    }
}
