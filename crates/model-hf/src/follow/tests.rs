use super::*;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("follow-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn listing(dir: &Path, text: &str) -> Result<List, Error> {
    let path = dir.join("list.tsv");
    std::fs::write(&path, text).unwrap();
    List::read(&path, dir)
}

/// A listing is `path<TAB>bytes`, its paths relative to the folder or under it; a line with no
/// tab, bytes that are no count, an empty listing and a file listed twice are each refused.
#[test]
fn a_listing_is_read_by_line_and_refused_by_name() {
    let dir = scratch("list");
    let list = listing(
        &dir,
        &format!("a.json\t7\n{}/m/b.safetensors\t9\n\n", dir.display()),
    )
    .unwrap();
    assert_eq!(list.bytes_of("a.json").unwrap(), 7);
    assert_eq!(list.bytes_of("m/b.safetensors").unwrap(), 9);
    let refused = |text: &str| listing(&dir, text).err().expect(text).to_string();
    assert!(refused("a.json 7\n").contains("line 1: no tab"));
    assert!(refused("a.json\t7\nb\tx\n").contains("line 2: \"x\" is no byte count"));
    assert!(refused("a.json\t7\na.json\t8\n").contains("line 2: a.json is listed twice"));
    assert!(refused("\n").contains("lists no file"));
    assert!(
        list.bytes_of("c.json")
            .unwrap_err()
            .to_string()
            .contains("c.json: the listing names no size")
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A file is complete at its listed bytes or with a sidecar, partial below them, and refused above.
#[test]
fn a_file_is_complete_at_its_listed_bytes_or_with_a_sidecar() {
    let dir = scratch("state");
    let list = listing(&dir, "a\t4\nb\t4\nc\t4\n").unwrap();
    assert_eq!(list.state(&dir, "a").unwrap(), State::Absent);
    std::fs::write(dir.join("a"), b"ab").unwrap();
    assert_eq!(list.state(&dir, "a").unwrap(), State::Part(2));
    assert_eq!(list.present(&dir).unwrap(), (2, 12));
    std::fs::write(dir.join("a"), b"abcd").unwrap();
    assert_eq!(list.state(&dir, "a").unwrap(), State::Done);
    std::fs::write(dir.join("b.done"), b"").unwrap();
    assert_eq!(list.state(&dir, "b").unwrap(), State::Done);
    std::fs::write(dir.join("c"), b"abcde").unwrap();
    assert!(
        list.state(&dir, "c")
            .unwrap_err()
            .to_string()
            .contains("more than the 4")
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// What `drive` did: the writes in order, with the shards each kept open, and the waits.
#[derive(Default)]
struct Log {
    present: BTreeSet<String>,
    written: Vec<(usize, Vec<String>)>,
    waits: Vec<Vec<String>>,
    /// Shards that arrive at the n-th wait.
    arrive: Vec<(usize, &'static str)>,
}

fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|s| s.to_string()).collect()
}

fn run(
    needs: &[&[&str]],
    done: Vec<bool>,
    present: &[&str],
    arrive: &[(usize, &'static str)],
) -> Result<Log, Error> {
    let needs: Vec<BTreeSet<String>> = needs.iter().map(|n| set(n)).collect();
    let mut log = Log {
        present: set(present),
        arrive: arrive.to_vec(),
        ..Log::default()
    };
    drive(
        &mut log,
        &needs,
        done,
        |l, s| Ok(l.present.contains(s)),
        |l, no, keep| {
            l.written.push((no, keep.iter().cloned().collect()));
            Ok(())
        },
        |l, missing| {
            l.waits.push(missing.iter().cloned().collect());
            let n = l.waits.len();
            let now: Vec<&str> = l.arrive.iter().filter(|a| a.0 == n).map(|a| a.1).collect();
            l.present.extend(now.iter().map(|s| s.to_string()));
            if l.waits.len() > 5 {
                return Err(Error::Refused("too many waits".into()));
            }
            Ok(())
        },
    )?;
    Ok(log)
}

/// The scheduler writes the first output shard whose inputs are all there, whatever order they
/// arrived in, waits only when none is, tells each write which input shards are still needed,
/// and skips a shard already written.
#[test]
fn an_output_shard_is_written_when_its_inputs_are_there_in_the_order_they_arrive() {
    let needs: &[&[&str]] = &[&["s1", "s2"], &["s3"], &["s2", "s4"]];
    // s3 and s4 are there first, then s2 at the first wait, then s1 at the second.
    let log = run(
        needs,
        vec![false; 3],
        &["s3", "s4"],
        &[(1, "s2"), (2, "s1")],
    )
    .unwrap();
    let order: Vec<usize> = log.written.iter().map(|w| w.0).collect();
    assert_eq!(order, [1, 2, 0]);
    assert_eq!(log.waits, [vec!["s1", "s2"], vec!["s1"]]);
    // Writing shard 1 first leaves shards 0 and 2 needing s1, s2 and s4; then shard 0 alone.
    assert_eq!(log.written[0].1, ["s1", "s2", "s4"]);
    assert_eq!(log.written[1].1, ["s1", "s2"]);
    assert_eq!(log.written[2].1, Vec::<String>::new());

    // Shard 1 is written already: the scheduler never asks for s3.
    let log = run(needs, vec![false, true, false], &["s1", "s2", "s4"], &[]).unwrap();
    assert_eq!(log.written.iter().map(|w| w.0).collect::<Vec<_>>(), [0, 2]);
    assert!(log.waits.is_empty());

    // Everything written: nothing runs, nothing waits.
    let log = run(needs, vec![true; 3], &[], &[]).unwrap();
    assert!(log.written.is_empty() && log.waits.is_empty());
}

/// A wait that never ends is the caller's refusal: the scheduler passes it what is missing.
#[test]
fn a_wait_is_handed_what_is_missing_and_its_refusal_ends_the_run() {
    let err = run(&[&["s1"]], vec![false], &[], &[])
        .err()
        .expect("never arrives");
    assert!(err.to_string().contains("too many waits"));
}

fn derived() -> Vec<(String, Dtype, Vec<usize>)> {
    ["a.weight", "b.weight", "c.weight"]
        .iter()
        .map(|n| (n.to_string(), Dtype::Bf16, vec![2, 2]))
        .collect()
}

fn skip(name: &str) -> Option<&'static str> {
    name.starts_with("vision.").then_some("the vision tower")
}

fn index(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
    entries
        .iter()
        .map(|(n, s)| (n.to_string(), s.to_string()))
        .collect()
}

/// The index and the config's table are read against each other: each derived tensor goes to the
/// shard the index names, a name the map skips is left out, and a name the table lacks that the
/// map does not skip, or a derived tensor the index lacks, is refused by name.
#[test]
fn the_index_and_the_derived_table_are_held_to_each_other() {
    let (shard_of, table) = assign(
        derived(),
        skip,
        &index(&[
            ("a.weight", "s1"),
            ("b.weight", "s1"),
            ("c.weight", "s2"),
            ("vision.w", "s3"),
        ]),
    )
    .unwrap();
    assert_eq!(shard_of.len(), 3);
    let names = |s: &str| -> Vec<&str> { table[s].iter().map(|t| t.0.as_str()).collect() };
    assert_eq!(names("s1"), ["a.weight", "b.weight"]);
    assert_eq!(names("s2"), ["c.weight"]);
    assert!(!table.contains_key("s3"));

    let err = |entries: &[(&str, &str)]| {
        assign(derived(), skip, &index(entries))
            .err()
            .unwrap()
            .to_string()
    };
    assert_eq!(
        err(&[
            ("a.weight", "s1"),
            ("b.weight", "s1"),
            ("c.weight", "s1"),
            ("extra.w", "s2")
        ]),
        "extra.w: the index puts it in s2; the config derives no such tensor and the map does not skip it"
    );
    assert_eq!(
        err(&[("a.weight", "s1"), ("b.weight", "s1")]),
        "c.weight: the config derives it and the index does not name it"
    );
}
