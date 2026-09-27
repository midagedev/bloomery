//! r8 sidecar gates (`model::r8file`), on a synthetic source written with
//! `gguf::write`: two shards holding three Q3_K stacks of random bytes on the
//! grids, one off the 8-row grid and an F32 tensor. `convert`, `Sidecar::open`
//! and `verify` pass and the sidecar holds `repack_q3k_r8` of each stack;
//! every identity difference, every conversion refusal and a flipped sidecar
//! byte are named; the `r8conv` binary converts and verifies the same source
//! end to end. A host tier's load (`HostR8::at_load`) on a synthetic routed
//! layer reads the sidecar at its path, or the source with the lever off,
//! builds the layer's gate and up from it, hands a held sidecar back to
//! another open of the same bytes, and refuses by name a sidecar whose
//! source's header or stack head differs, also while an earlier load holds
//! it open; a gate's reading (`HostR8::at_gate`) refuses no file by name. A
//! sidecar replaced at its path is opened anew, and a sidecar opened for one
//! source is refused by name beside another source's split of the same names
//! and shapes. Releasing a resident sidecar's pages is refused by name and
//! leaves its bytes as they were. No clause reads a model file.

#[path = "common/r8layer.rs"]
mod r8layer;

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::process::Command;

use gguf::write::{Layout, TensorDecl, WriteError, Writer};
use gguf::{GgmlType, Gguf, LoadError, PrivateType, Split, Value, Weights};
use model::ModelError;
use model::moe::HostLayer;
use model::ops::RowLayout;
use model::placement::PlacementError;
use model::placement::host_lock::PageDrop;
use model::r8file::{self, HostR8, Progress, R8Error, R8Source, Sidecar};

const G0: &str = "blk.0.ffn_gate_exps.weight";
const U0: &str = "blk.0.ffn_up_exps.weight";
const G1: &str = "blk.1.ffn_gate_exps.weight";
const F32T: &str = "blk.0.attn.weight";
const OFF: &str = "blk.1.off_grid.weight";
const LAZY: Weights = Weights::Mapped { populate: false };

/// The stacks every clause converts, in sidecar order.
fn names() -> Vec<String> {
    [G0, U0, G1].map(String::from).to_vec()
}

/// A fresh directory for clause `tag`.
fn dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("r8file-{}-{tag}", std::process::id()));
    if d.exists() {
        std::fs::remove_dir_all(&d).unwrap();
    }
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Deterministic bytes (xorshift64*).
fn random(seed: u64, n: u64) -> Vec<u8> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8
        })
        .collect()
}

fn decl(name: &str, dims: &[u64], type_id: u32, nbytes: u64) -> TensorDecl {
    TensorDecl {
        name: name.to_string(),
        dims: dims.to_vec(),
        type_id,
        nbytes,
    }
}

/// A Q3_K stack `[k, rows, experts]` of random bytes.
fn q3k(name: &str, dims: [u64; 3], seed: u64) -> (TensorDecl, Vec<u8>) {
    let nbytes = dims[0] / 256 * 110 * dims[1] * dims[2];
    (decl(name, &dims, 11, nbytes), random(seed, nbytes))
}

/// The source's tensors, shard by shard. `G0` and `U0` are longer than 8 KiB,
/// so a head and a tail never share a byte; `G1` is shorter than 4 KiB.
fn source_tensors() -> [Vec<(TensorDecl, Vec<u8>)>; 2] {
    [
        vec![
            q3k(G0, [512, 16, 3], 1),
            q3k(U0, [512, 16, 3], 2),
            (decl(F32T, &[8, 4], 0, 128), random(3, 128)),
        ],
        vec![q3k(G1, [256, 8, 4], 4), q3k(OFF, [256, 12, 2], 5)],
    ]
}

fn kv(key: &str, v: Value) -> (String, Value) {
    (key.to_string(), v)
}

fn write_gguf(path: &Path, kvs: &[(String, Value)], tensors: &[(TensorDecl, Vec<u8>)]) {
    let layout = Layout::new(kvs, tensors.iter().map(|(t, _)| t.clone()).collect()).unwrap();
    let mut w = Writer::new(BufWriter::new(File::create(path).unwrap()), layout).unwrap();
    for (t, b) in tensors {
        w.tensor(&t.name, b).unwrap();
    }
    w.finish().unwrap();
}

/// The two-shard source in `d`, `<stem>-0000N-of-00002.gguf`; returns the
/// first shard's path.
fn write_source(d: &Path, stem: &str) -> PathBuf {
    let [s0, s1] = source_tensors();
    let split = |no: u16| {
        vec![
            kv("split.no", Value::U16(no)),
            kv("split.count", Value::U16(2)),
            kv("split.tensors.count", Value::I32(5)),
        ]
    };
    let mut kv0 = vec![kv("test.note", Value::String("synthetic source".into()))];
    kv0.extend(split(0));
    let first = d.join(format!("{stem}-00001-of-00002.gguf"));
    write_gguf(&first, &kv0, &s0);
    write_gguf(
        &d.join(format!("{stem}-00002-of-00002.gguf")),
        &split(1),
        &s1,
    );
    first
}

/// The refusal `r` must be, as the r8 error it wraps.
fn r8_err<T>(r: Result<T, ModelError>) -> R8Error {
    match r {
        Ok(_) => panic!("must be refused"),
        Err(ModelError::R8(e)) => e,
        Err(other) => panic!("refused, but not by the r8 format: {other}"),
    }
}

fn part_of(out: &Path) -> PathBuf {
    let mut p = out.as_os_str().to_owned();
    p.push(".part");
    PathBuf::from(p)
}

/// Byte `at` of `path` xor `mask`, in place.
fn flip(path: &Path, at: u64, mask: u8) {
    let mut b = std::fs::read(path).unwrap();
    b[usize::try_from(at).unwrap()] ^= mask;
    std::fs::write(path, b).unwrap();
}

/// Where the first occurrence of `needle` starts in `path`.
fn find_bytes(path: &Path, needle: &[u8]) -> u64 {
    let b = std::fs::read(path).unwrap();
    let at = b
        .windows(needle.len())
        .position(|w| w == needle)
        .unwrap_or_else(|| {
            panic!(
                "{} holds no {:?}",
                path.display(),
                String::from_utf8_lossy(needle)
            )
        });
    at as u64
}

/// The file offset of tensor `name`'s first byte in a file the strict reader
/// opens.
fn data_at(path: &Path, name: &str) -> (u64, u64) {
    let g = Gguf::open(path).unwrap();
    let t = g.find(name).unwrap();
    (g.data_base() + t.offset, t.nbytes)
}

/// The same, in a sidecar.
fn sidecar_data_at(path: &Path, name: &str) -> u64 {
    let table = [PrivateType {
        id: r8file::Q3K_R8_TYPE,
        blck: 256,
        size: 110,
    }];
    let g = Gguf::open_private(path, LAZY, &table).unwrap();
    g.data_base() + g.find(name).unwrap().offset
}

/// The sidecar at `from`, its pairs and declarations passed through `edit`,
/// written to `to` with the same tensor bytes.
fn rewrite(
    from: &Path,
    to: &Path,
    edit: impl FnOnce(&mut Vec<(String, Value)>, &mut Vec<TensorDecl>),
) {
    let table = [PrivateType {
        id: r8file::Q3K_R8_TYPE,
        blck: 256,
        size: 110,
    }];
    let g = Gguf::open_private(from, LAZY, &table).unwrap();
    let mut kvs: Vec<(String, Value)> = g
        .iter_kv()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();
    let mut decls: Vec<TensorDecl> = g
        .iter_tensors()
        .map(|t| decl(&t.name, &t.dims, t.ty.as_u32(), t.nbytes))
        .collect();
    let data: Vec<Vec<u8>> = g
        .iter_tensors()
        .map(|t| g.data(t).unwrap().to_vec())
        .collect();
    edit(&mut kvs, &mut decls);
    let tensors: Vec<(TensorDecl, Vec<u8>)> = decls.into_iter().zip(data).collect();
    write_gguf(to, &kvs, &tensors);
}

/// `convert`, `Sidecar::open` and `verify` pass on the synthetic source, into
/// a directory `convert` makes; each tensor is `repack_q3k_r8` of its stack
/// and reported once, in order; no `.part` is left; the strict reader refuses
/// the sidecar by its first tensor's name; a resident open and its `verify`
/// pass too; a name outside the sidecar is refused.
#[test]
fn a_sidecar_round_trips_its_source() {
    let d = dir("happy");
    let first = write_source(&d, "src");
    let split = Split::open(&first).unwrap();
    let out = d.join("made").join("side.gguf");
    let mut seen = Vec::new();
    let stats = r8file::convert(&split, &names(), &out, &mut |p| {
        if let Progress::Tensor(t) = p {
            seen.push(t.name.clone());
        }
    })
    .unwrap();
    assert_eq!(seen, names());
    assert_eq!(stats.out, out);
    assert_eq!(std::fs::metadata(&out).unwrap().len(), stats.file_bytes);
    assert!(!part_of(&out).exists(), "a .part outlived the conversion");
    let side = Sidecar::open(&out, &split, LAZY).unwrap();
    assert_eq!(side.names().collect::<Vec<_>>(), [G0, U0, G1]);
    for n in names() {
        let (s, t) = split.find(&n).unwrap();
        let src = split.shard(s).unwrap().data(t).unwrap();
        let rows = usize::try_from(t.dims[1] * t.dims[2]).unwrap();
        let mut want = vec![0u8; src.len()];
        qdot::repack_q3k_r8(src, rows, usize::try_from(t.dims[0]).unwrap(), &mut want).unwrap();
        assert!(
            side.data(&n).unwrap() == want.as_slice(),
            "{n}: not repack_q3k_r8 of the source"
        );
    }
    let v = r8file::verify(&split, &side, &mut |_| {}).unwrap();
    assert_eq!((v.tensors.len(), v.bytes), (3, 2 * 10_560 + 3_520));
    match side.data(F32T) {
        Err(ModelError::R8(R8Error::NotInSidecar { tensor, .. })) => assert_eq!(tensor, F32T),
        other => panic!("{F32T} must be refused, got {:?}", other.map(<[u8]>::len)),
    }
    match Gguf::open(&out) {
        Err(LoadError::UnsupportedType { name, ty }) => {
            assert_eq!(
                (name.as_str(), ty),
                (G0, GgmlType::Unknown(r8file::Q3K_R8_TYPE))
            );
        }
        other => panic!(
            "the strict reader must refuse the sidecar, got {:?}",
            other.map(|_| ())
        ),
    }
    let resident = Sidecar::open(&out, &split, Weights::Resident { huge: false }).unwrap();
    assert!(resident.data(G0).unwrap() == side.data(G0).unwrap());
    r8file::verify(&split, &resident, &mut |_| {}).unwrap();
    std::fs::remove_dir_all(&d).unwrap();
}

/// One identity difference: its label, the change it makes to a copy of the
/// source and the sidecar (returning the first shard and the sidecar to
/// open), and the refusal it must meet.
struct Case {
    label: &'static str,
    change: fn(&Path) -> (PathBuf, PathBuf),
    want: fn(&R8Error) -> bool,
}

/// The copy's source and sidecar, unchanged.
fn copied(d: &Path) -> (PathBuf, PathBuf) {
    (d.join("src-00001-of-00002.gguf"), d.join("side.gguf"))
}

/// Each difference `Sidecar::open` checks is refused by its own error: the
/// architecture, the layout version, a source header byte, a shard's length,
/// a shard missing, the shards renamed, a stack's head and tail, the
/// sidecar's dims, a sidecar tensor whose source is not Q3_K, a sidecar
/// tensor of another type id, a name twice, and identity pairs absent or of
/// the wrong length. Each runs on its own copy of one converted source.
#[test]
fn every_identity_difference_is_refused_by_name() {
    let base = dir("identity");
    let first = write_source(&base, "src");
    let split = Split::open(&first).unwrap();
    r8file::convert(&split, &names(), &base.join("side.gguf"), &mut |_| {}).unwrap();
    drop(split);
    let cases = [
        Case {
            label: "architecture",
            change: |d| {
                let (first, side) = copied(d);
                let to = d.join("r9.gguf");
                rewrite(&side, &to, |kvs, _| {
                    let arch = Value::String(r8file::R8_ARCH.to_string());
                    let slot = kvs.iter_mut().find(|(_, v)| *v == arch).unwrap();
                    slot.1 = Value::String("bloomery-r9".to_string());
                });
                (first, to)
            },
            want: |e| matches!(e, R8Error::Architecture { got, .. } if got == "bloomery-r9"),
        },
        Case {
            label: "layout version",
            change: |d| {
                let (first, side) = copied(d);
                let to = d.join("v2.gguf");
                rewrite(&side, &to, |kvs, _| {
                    let slot = kvs
                        .iter_mut()
                        .find(|(k, _)| k == "bloomery.r8.layout")
                        .unwrap();
                    slot.1 = Value::U32(qdot::Q3K_R8_LAYOUT + 1);
                });
                (first, to)
            },
            want: |e| matches!(e, R8Error::Layout { got, .. } if *got == qdot::Q3K_R8_LAYOUT + 1),
        },
        Case {
            label: "source header byte",
            change: |d| {
                let (first, side) = copied(d);
                flip(&first, find_bytes(&first, b"synthetic source"), 0x01);
                (first, side)
            },
            want: |e| matches!(e, R8Error::HeaderDigest { shard, .. } if shard == "src-00001-of-00002.gguf"),
        },
        Case {
            label: "shard length",
            change: |d| {
                let (first, side) = copied(d);
                let second = d.join("src-00002-of-00002.gguf");
                let mut b = std::fs::read(&second).unwrap();
                b.extend_from_slice(&[0; 32]);
                std::fs::write(&second, b).unwrap();
                (first, side)
            },
            want: |e| {
                matches!(e, R8Error::ShardBytes { shard, recorded, found, .. }
                    if shard == "src-00002-of-00002.gguf" && *found == *recorded + 32)
            },
        },
        Case {
            label: "shard missing",
            change: |d| {
                let (_, side) = copied(d);
                let one = d.join("one.gguf");
                let [s0, s1] = source_tensors();
                let all: Vec<(TensorDecl, Vec<u8>)> = s0.into_iter().chain(s1).collect();
                write_gguf(
                    &one,
                    &[kv("test.note", Value::String("synthetic source".into()))],
                    &all,
                );
                (one, side)
            },
            want: |e| {
                matches!(
                    e,
                    R8Error::ShardCount {
                        recorded: 2,
                        found: 1,
                        ..
                    }
                )
            },
        },
        Case {
            label: "shards renamed",
            change: |d| {
                let (_, side) = copied(d);
                for i in 1..=2 {
                    let from = d.join(format!("src-0000{i}-of-00002.gguf"));
                    std::fs::rename(from, d.join(format!("alt-0000{i}-of-00002.gguf"))).unwrap();
                }
                (d.join("alt-00001-of-00002.gguf"), side)
            },
            want: |e| {
                matches!(e, R8Error::ShardName { shard: 0, recorded, found, .. }
                    if recorded == "src-00001-of-00002.gguf" && found == "alt-00001-of-00002.gguf")
            },
        },
        Case {
            label: "stack head",
            change: |d| {
                let (first, side) = copied(d);
                let (at, _) = data_at(&first, G0);
                flip(&first, at, 0x10);
                (first, side)
            },
            want: |e| matches!(e, R8Error::Head { tensor, .. } if tensor == G0),
        },
        Case {
            label: "stack tail",
            change: |d| {
                let (first, side) = copied(d);
                let (at, n) = data_at(&first, G0);
                flip(&first, at + n - 1, 0x10);
                (first, side)
            },
            want: |e| matches!(e, R8Error::Tail { tensor, .. } if tensor == G0),
        },
        Case {
            label: "sidecar dims",
            change: |d| {
                let (first, side) = copied(d);
                let to = d.join("dims.gguf");
                rewrite(&side, &to, |_, decls| decls[0].dims = vec![512, 48, 1]);
                (first, to)
            },
            want: |e| {
                matches!(e, R8Error::Shape { tensor, recorded_dims, found_dims, .. }
                    if tensor == G0 && recorded_dims == &[512, 48, 1] && found_dims == &[512, 16, 3])
            },
        },
        Case {
            label: "source stack not Q3_K",
            change: |d| {
                let (first, side) = copied(d);
                let to = d.join("f32.gguf");
                rewrite(&side, &to, |_, decls| decls[2].name = F32T.to_string());
                (first, to)
            },
            want: |e| matches!(e, R8Error::SourceType { tensor, got: GgmlType::F32, .. } if tensor == F32T),
        },
        Case {
            label: "sidecar type id",
            change: |d| {
                let (first, side) = copied(d);
                let to = d.join("q3k.gguf");
                rewrite(&side, &to, |_, decls| decls[0].type_id = 11);
                (first, to)
            },
            want: |e| matches!(e, R8Error::TensorType { tensor, got: 11, .. } if tensor == G0),
        },
        Case {
            label: "a name twice",
            change: |d| {
                let (first, side) = copied(d);
                // G1's name becomes G0's in the header: the writer refuses a
                // repeated name, a foreign writer need not.
                flip(&side, find_bytes(&side, G1.as_bytes()) + 4, b'1' ^ b'0');
                (first, side)
            },
            want: |e| matches!(e, R8Error::DuplicateTensor { tensor, .. } if tensor == G0),
        },
        Case {
            label: "identity pair absent",
            change: |d| {
                let (first, side) = copied(d);
                let to = d.join("nobytes.gguf");
                rewrite(&side, &to, |kvs, _| {
                    kvs.retain(|(k, _)| k != "bloomery.r8.source.shard_bytes")
                });
                (first, to)
            },
            want: |e| matches!(e, R8Error::Key { key, .. } if *key == "bloomery.r8.source.shard_bytes"),
        },
        Case {
            label: "identity pair short",
            change: |d| {
                let (first, side) = copied(d);
                let to = d.join("short.gguf");
                rewrite(&side, &to, |kvs, _| {
                    let slot = kvs
                        .iter_mut()
                        .find(|(k, _)| k == "bloomery.r8.source.head_sha256")
                        .unwrap();
                    if let Value::Array(items) = &mut slot.1 {
                        items.truncate(2);
                    }
                });
                (first, to)
            },
            want: |e| {
                matches!(e, R8Error::Key { key, detail, .. }
                    if *key == "bloomery.r8.source.head_sha256" && detail.contains("2 entries for 3"))
            },
        },
    ];
    for (i, c) in cases.iter().enumerate() {
        let d = base.join(format!("case{i}"));
        std::fs::create_dir_all(&d).unwrap();
        for f in [
            "src-00001-of-00002.gguf",
            "src-00002-of-00002.gguf",
            "side.gguf",
        ] {
            std::fs::copy(base.join(f), d.join(f)).unwrap();
        }
        let (first, side) = (c.change)(&d);
        let split = Split::open(&first)
            .unwrap_or_else(|e| panic!("{}: the changed source must still open: {e}", c.label));
        match Sidecar::open(&side, &split, LAZY) {
            Ok(_) => panic!("{}: the sidecar must be refused", c.label),
            Err(ModelError::R8(e)) => {
                assert!((c.want)(&e), "{}: wrong refusal: {e}", c.label);
                println!("{}: {e}", c.label);
            }
            Err(other) => panic!("{}: refused, but not by the r8 format: {other}", c.label),
        }
    }
    std::fs::remove_dir_all(&base).unwrap();
}

/// `convert` refuses by name, writing nothing: a stack that is not Q3_K, one
/// off the 8-row grid, a name the source lacks, no names, a name twice, and
/// an `out` that exists, which keeps its bytes.
#[test]
fn convert_refuses_by_name() {
    let d = dir("convert");
    let first = write_source(&d, "src");
    let split = Split::open(&first).unwrap();
    let out = d.join("side.gguf");
    let list = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let refuse = |names: Vec<String>| {
        let e = r8_err(r8file::convert(&split, &names, &out, &mut |_| {}));
        assert!(
            !out.exists() && !part_of(&out).exists(),
            "{e}: a refusal wrote a file"
        );
        e
    };
    match refuse(list(&[G0, F32T])) {
        R8Error::SourceType { tensor, got, .. } => {
            assert_eq!((tensor.as_str(), got), (F32T, GgmlType::F32))
        }
        other => panic!("wrong refusal: {other}"),
    }
    match refuse(list(&[OFF])) {
        R8Error::Grid { tensor, dims, .. } => {
            assert_eq!((tensor.as_str(), dims), (OFF, vec![256, 12, 2]))
        }
        other => panic!("wrong refusal: {other}"),
    }
    match refuse(list(&["blk.9.nothing.weight"])) {
        R8Error::SourceMissing { tensor, .. } => assert_eq!(tensor, "blk.9.nothing.weight"),
        other => panic!("wrong refusal: {other}"),
    }
    assert!(matches!(refuse(Vec::new()), R8Error::NoTensors));
    match refuse(list(&[G0, G0])) {
        R8Error::Write {
            source: WriteError::DuplicateTensor { name },
            ..
        } => assert_eq!(name, G0),
        other => panic!("wrong refusal: {other}"),
    }
    std::fs::write(&out, b"precious").unwrap();
    match r8_err(r8file::convert(&split, &names(), &out, &mut |_| {})) {
        R8Error::Exists { path } => assert_eq!(path, out),
        other => panic!("wrong refusal: {other}"),
    }
    assert_eq!(std::fs::read(&out).unwrap(), b"precious");
    std::fs::remove_dir_all(&d).unwrap();
}

/// A `.part` whose lock a live run holds refuses the conversion and stays; one
/// left by a run that died is removed, reported, and replaced by a finished
/// sidecar at `out`. The live run releases its lock by unlocking, not by
/// closing: a child spawned while the lock is held keeps a copy of the
/// descriptor — here past its exec, which stands for the fork-to-exec window
/// of any concurrent spawn — and a lock left to the close would stay with
/// that copy.
#[test]
fn a_leftover_part_is_removed_and_a_live_one_is_not() {
    let d = dir("part");
    let first = write_source(&d, "src");
    let split = Split::open(&first).unwrap();
    let out = d.join("side.gguf");
    let part = part_of(&out);
    std::fs::write(&part, b"live").unwrap();
    let live = File::open(&part).unwrap();
    live.try_lock().unwrap();
    let mut child = spawn_holding(&live);
    match r8_err(r8file::convert(&split, &names(), &out, &mut |_| {})) {
        R8Error::Busy { path } => assert_eq!(path, part),
        other => panic!("wrong refusal: {other}"),
    }
    assert_eq!(std::fs::read(&part).unwrap(), b"live");
    assert!(!out.exists());
    live.unlock().unwrap();
    drop(live);
    let mut removed = Vec::new();
    r8file::convert(&split, &names(), &out, &mut |p| {
        if let Progress::RemovedPart(p) = p {
            removed.push(p.to_path_buf());
        }
    })
    .unwrap();
    assert_eq!(removed, std::slice::from_ref(&part));
    assert!(!part.exists());
    Sidecar::open(&out, &split, LAZY).unwrap();
    child.kill().unwrap();
    child.wait().unwrap();
    std::fs::remove_dir_all(&d).unwrap();
}

/// A child that holds a copy of `file`'s descriptor — the same open file
/// description, so the same `flock` — until it is killed: `sleep` spawned
/// with that copy kept across its exec.
fn spawn_holding(file: &File) -> std::process::Child {
    use std::os::fd::AsRawFd;
    let copy = file.try_clone().unwrap();
    let fd = copy.as_raw_fd();
    // SAFETY: fcntl reads the descriptor flags of `copy`, a descriptor this
    // function owns; no memory is passed.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    assert!(flags >= 0, "F_GETFD: {}", std::io::Error::last_os_error());
    // SAFETY: as above; it sets them.
    let rc = unsafe { libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) };
    assert_eq!(rc, 0, "F_SETFD: {}", std::io::Error::last_os_error());
    let child = Command::new("sleep").arg("120").spawn().unwrap();
    drop(copy);
    child
}

/// A flipped sidecar byte passes the identity check — it reads no stack — and
/// `verify` names it: row 5's `d` high byte in group super-block 1 of expert
/// 2's second 8-row group of `U0` is Q3_K byte 13 · 220 + 110 + 109 of that
/// expert.
#[test]
fn verify_names_the_first_byte_that_differs() {
    let d = dir("mismatch");
    let first = write_source(&d, "src");
    let split = Split::open(&first).unwrap();
    let out = d.join("side.gguf");
    r8file::convert(&split, &names(), &out, &mut |_| {}).unwrap();
    // U0: k 512 (two super-blocks, 220 B a row), 16 rows an expert (3,520 B),
    // 8-row groups of 1,760 B, group super-blocks of 880 B; row r's d at 2r.
    let (expert, group, sb, row) = (2u64, 1u64, 1u64, 5u64);
    let r8_byte = expert * 3_520 + group * 1_760 + sb * 880 + 2 * row + 1;
    flip(&out, sidecar_data_at(&out, U0) + r8_byte, 0x40);
    let side = Sidecar::open(&out, &split, LAZY).unwrap();
    match r8_err(r8file::verify(&split, &side, &mut |_| {})) {
        R8Error::Mismatch {
            tensor,
            expert: e,
            byte,
            ..
        } => assert_eq!(
            (tensor.as_str(), e, byte),
            (U0, expert, (group * 8 + row) * 220 + sb * 110 + 109)
        ),
        other => panic!("wrong refusal: {other}"),
    }
    std::fs::remove_dir_all(&d).unwrap();
}

/// The sidecar lives beside the source directory, in one of its own, named
/// after the model without the shard suffix.
#[test]
fn the_sidecar_lives_in_a_directory_of_its_own() {
    let cases = [
        (
            "/models/DeepSeek-V4.1-Flash-Q3_K_M/DeepSeek-V4.1-Flash-Q3_K_M-00001-of-00009.gguf",
            "/models/DeepSeek-V4.1-Flash-Q3_K_M-r8/DeepSeek-V4.1-Flash-Q3_K_M-r8.gguf",
        ),
        ("/a/b/model.gguf", "/a/b-r8/model-r8.gguf"),
        ("/a/m-1-of-2.gguf", "/a-r8/m-1-of-2-r8.gguf"),
    ];
    for (first, want) in cases {
        assert_eq!(
            r8file::sidecar_path(Path::new(first)).unwrap(),
            PathBuf::from(want),
            "{first}"
        );
    }
    let cwd = std::fs::canonicalize(".").unwrap();
    let up = cwd.parent().unwrap();
    let beside = |dir: &Path| {
        let mut d = dir.as_os_str().to_owned();
        d.push("-r8");
        PathBuf::from(d).join("m-r8.gguf")
    };
    for (first, dir) in [
        ("m.gguf", cwd.as_path()),
        ("./m.gguf", cwd.as_path()),
        ("../m.gguf", up),
    ] {
        assert_eq!(
            r8file::sidecar_path(Path::new(first)).unwrap(),
            beside(dir),
            "{first}: a directory that ends in no name is the one it resolves to"
        );
    }
    println!(
        "sidecar path: lexical under a named directory; m.gguf, ./m.gguf and ../m.gguf resolved"
    );
}

/// The `r8conv` binary on the synthetic source: `convert --tensors` and
/// `verify` exit 0 with a line per tensor and a summary, `path` prints the
/// convention, and a flipped sidecar byte and an existing `out` exit 1 with
/// the named error.
#[test]
fn the_r8conv_binary_converts_and_verifies() {
    let bin = env!("CARGO_BIN_EXE_r8conv");
    let d = dir("cli");
    let first = write_source(&d, "src");
    let out = d.join("cli-side.gguf");
    let (first_s, out_s) = (first.to_str().unwrap(), out.to_str().unwrap());
    let run = |args: &[&str]| {
        let o = Command::new(bin).args(args).output().unwrap();
        let (stdout, stderr) = (
            String::from_utf8_lossy(&o.stdout).into_owned(),
            String::from_utf8_lossy(&o.stderr).into_owned(),
        );
        println!(
            "$ r8conv {}\n{stdout}{stderr}rc={:?}",
            args.join(" "),
            o.status.code()
        );
        (o.status.code(), stdout, stderr)
    };
    let tensors = [G0, U0, G1].join(",");
    let (rc, stdout, _) = run(&["convert", first_s, out_s, "--tensors", &tensors]);
    assert_eq!(rc, Some(0));
    for n in [G0, U0, G1] {
        assert!(
            stdout.contains(&format!("r8conv: convert {n} bytes=")),
            "no line for {n}"
        );
    }
    assert!(stdout.contains("convert done tensors=3"), "no summary");
    let (rc, stdout, _) = run(&["verify", first_s, out_s]);
    assert_eq!(rc, Some(0));
    assert!(stdout.contains("verify done tensors=3"), "no summary");
    let (rc, stdout, _) = run(&["path", first_s]);
    assert_eq!(rc, Some(0));
    assert_eq!(
        stdout.trim_end(),
        r8file::sidecar_path(&first).unwrap().to_str().unwrap()
    );
    flip(&out, sidecar_data_at(&out, G1) + 1, 0x01);
    let (rc, _, stderr) = run(&["verify", first_s, out_s]);
    assert_eq!(rc, Some(1));
    assert!(
        stderr.contains("unpacks to other bytes"),
        "verify must name the mismatch"
    );
    let (rc, _, stderr) = run(&["convert", first_s, out_s, "--tensors", &tensors]);
    assert_eq!(rc, Some(1));
    assert!(
        stderr.contains("is never overwritten"),
        "convert must name the existing out"
    );
    std::fs::remove_dir_all(&d).unwrap();
}

/// The reading of `split` at load, with the sidecar asked for, must be
/// refused as `want` says, naming `what` refused it.
fn refused(split: &Split, what: &str, by: &str, want: impl Fn(&R8Error) -> bool) {
    match HostR8::at_load(split, true) {
        Err(ModelError::R8(e)) if want(&e) => {}
        Err(other) => panic!("{what}: refused, but not by {by}: {other}"),
        Ok(r8) => panic!("{what}: must be refused, got {r8}"),
    }
    println!("{what}: refused by {by}");
}

/// A host tier's load of a synthetic routed layer: with the sidecar at its
/// path the reading is `r8=on` and the layer's gate and up are its row-lane
/// copies (`HostLayer::layouts`); with the lever off the reading is the
/// source's, named, and the layer reads its rows. While the earlier load's
/// layer holds the sidecar open, a second reading of another open of the
/// same bytes hands back the held sidecar; once the source's header differs
/// in a byte the sidecar does not hold, or its gate's head does, the load is
/// refused by `R8Error::HeaderDigest` naming the shard or by `R8Error::Head`
/// naming the gate — the second reading compares its source with what the
/// held sidecar's check read — and the head's difference on a fresh open
/// too. With no file at the path the load's reading is the source's, named,
/// and the layer reads its rows; a gate's reading is refused by
/// `R8Error::NoSidecar` naming the path, and with the lever off is the
/// source's.
#[test]
#[ignore = "hw: the box's CPU (a host layer's r8 stack needs qdot's fused Q3_K); reads no model file"]
fn hw_a_host_tier_load_refuses_a_sidecar_whose_source_differs() {
    let layer = r8layer::Layer::write("host", 512, 256, 4, GgmlType::Q4_K);
    let spec = layer.spec();
    let split = Split::open(&layer.source).unwrap();
    let r8 = HostR8::at_load(&split, true).unwrap();
    let side = r8.sidecar().expect("the sidecar at its path is read");
    assert_eq!(side.path(), layer.sidecar.as_path());
    assert_eq!(
        r8.to_string(),
        format!("r8=on ({})", layer.sidecar.display())
    );
    let held = HostLayer::build(R8Source::of(&split, &r8).unwrap(), &spec).unwrap();
    assert_eq!(
        held.layouts(),
        [RowLayout::R8, RowLayout::R8, RowLayout::Rows]
    );
    let off = HostR8::at_load(&split, false).unwrap();
    assert!(off.sidecar().is_none(), "the lever off reads the source");
    assert_eq!(off.to_string(), "r8=off (BLOOMERY_R8=off)");
    let rows = HostLayer::build(R8Source::of(&split, &off).unwrap(), &spec).unwrap();
    assert_eq!(rows.layouts(), [RowLayout::Rows; 3]);
    let again = Split::open(&layer.source).unwrap();
    let reuse = HostR8::at_load(&again, true).unwrap();
    assert!(
        reuse
            .sidecar()
            .is_some_and(|s| std::sync::Arc::ptr_eq(s, side)),
        "another open of the same bytes gets the held sidecar"
    );
    println!("reopen: the held sidecar handed back; lever off: r8=off reads rows");
    drop((r8, reuse, split, again, rows));

    let name = find_bytes(&layer.source, r8layer::DOWN.as_bytes());
    flip(&layer.source, name, 0x01);
    let split = Split::open(&layer.source).unwrap();
    refused(
        &split,
        "held open, a header byte",
        "the shard's header digest",
        |e| matches!(e, R8Error::HeaderDigest { shard, .. } if shard == "layer.gguf"),
    );
    drop(split);
    flip(&layer.source, name, 0x01);

    let (at, _) = data_at(&layer.source, r8layer::GATE);
    flip(&layer.source, at, 0x10);
    let split = Split::open(&layer.source).unwrap();
    let head = |e: &R8Error| matches!(e, R8Error::Head { tensor, .. } if tensor == r8layer::GATE);
    refused(
        &split,
        "held open, the gate's head",
        "its head digest",
        head,
    );
    drop(held);
    refused(
        &split,
        "fresh open, the gate's head",
        "its head digest",
        head,
    );

    std::fs::remove_file(&layer.sidecar).unwrap();
    let r8 = HostR8::at_load(&split, true).unwrap();
    assert!(r8.sidecar().is_none());
    assert_eq!(
        r8.to_string(),
        format!(
            "r8=off (no sidecar at {}: just r8-sidecar)",
            layer.sidecar.display()
        )
    );
    let rows = HostLayer::build(R8Source::of(&split, &r8).unwrap(), &spec).unwrap();
    assert_eq!(rows.layouts(), [RowLayout::Rows; 3]);
    match HostR8::at_gate(&split, true) {
        Err(ModelError::R8(R8Error::NoSidecar { path })) => assert_eq!(path, layer.sidecar),
        Err(other) => panic!("a gate's reading with no file: refused, but not by name: {other}"),
        Ok(r8) => panic!("a gate's reading with no file must be refused, got {r8}"),
    }
    assert!(
        matches!(HostR8::at_gate(&split, false), Ok(HostR8::Lever)),
        "a gate's reading with the lever off is the source's"
    );
    println!(
        "host tier load: r8=on reads the sidecar; a changed header or head is refused; no file \
         reads rows, and refuses a gate's reading by name"
    );
}

/// A sidecar opened for one source, beside another source's split of the
/// same tensor names, shapes and header — other stack bytes — cannot be made
/// its pair: `R8Source::of` refuses it by `R8Error::Head` naming the gate
/// (the first stack whose identity differs), and a reader takes a sidecar
/// only through a pair. A layer built beside one open of the sidecar is
/// refused by name (`R8Error::OtherSidecar`) when called with a pair that
/// reads none or another open of the same file — each open checked against
/// the same split — and reads bit for bit what it read beside its own. A
/// page release takes the pair's own sidecar, and a call with a split in
/// place of a pair is a compile error (`R8Source`'s `compile_fail`
/// doctests).
#[test]
#[ignore = "hw: the box's CPU (a host layer's r8 stack needs qdot's fused Q3_K); reads no model file"]
fn hw_a_sidecar_of_another_source_is_refused_beside_its_split() {
    let a = r8layer::Layer::write("pair-a", 512, 256, 4, GgmlType::Q4_K);
    let b = r8layer::Layer::write_salted("pair-b", 512, 256, 4, GgmlType::Q4_K, 0x5a17);
    let (split_a, split_b) = (
        Split::open(&a.source).unwrap(),
        Split::open(&b.source).unwrap(),
    );
    let side_a = HostR8::On(std::sync::Arc::new(
        Sidecar::open(&a.sidecar, &split_a, LAZY).unwrap(),
    ));
    let src_a = R8Source::of(&split_a, &side_a).unwrap();
    let layer = HostLayer::build(src_a, &a.spec()).unwrap();
    let head = |e: &R8Error| matches!(e, R8Error::Head { tensor, .. } if tensor == r8layer::GATE);
    match R8Source::of(&split_b, &side_a) {
        Err(ModelError::R8(e)) if head(&e) => println!("pair beside another source: {e}"),
        Err(other) => panic!("pair beside another source: refused, but not by its head: {other}"),
        Ok(_) => panic!("another source's sidecar was made this split's pair"),
    }

    let embd = 512;
    let x = model::ops::Tensor2::from_vec(
        embd,
        1,
        (0..embd).map(|i| (i % 13) as f32 / 13.0 - 0.5).collect(),
    );
    let list = [(1u32, 0.75f32), (3, 0.25)];
    let mut scratch = model::moe::HostScratch::new(embd, 256, list.len()).unwrap();
    let mut want = vec![f32::NAN; embd];
    layer
        .experts_into(src_a, &x, &list, &mut want, &mut scratch)
        .unwrap();
    let other_open = HostR8::On(std::sync::Arc::new(
        Sidecar::open(&a.sidecar, &split_a, LAZY).unwrap(),
    ));
    let src_other = R8Source::of(&split_a, &other_open).unwrap();
    for (what, src, read) in [
        ("a pair that reads none", R8Source::rows(&split_a), false),
        ("a pair that reads another open", src_other, true),
    ] {
        let mut out = vec![f32::NAN; embd];
        match layer.experts_into(src, &x, &list, &mut out, &mut scratch) {
            Err(ModelError::R8(e @ R8Error::OtherSidecar { .. })) => {
                let R8Error::OtherSidecar { read: got, .. } = &e else {
                    unreachable!()
                };
                assert_eq!(got.is_some(), read, "{what}: {e}");
                println!("layer called with {what}: {e}");
            }
            Err(other) => panic!("{what}: refused, but not by name: {other}"),
            Ok(()) => panic!("{what}: a layer built beside one open read beside another"),
        }
        assert!(out.iter().all(|v| v.is_nan()), "{what}: nothing is written");
    }
    let mut again = vec![f32::NAN; embd];
    layer
        .experts_into(src_a, &x, &list, &mut again, &mut scratch)
        .unwrap();
    assert!(
        want.iter()
            .zip(&again)
            .all(|(w, a)| w.to_bits() == a.to_bits()),
        "the layer's own pair reads what it read"
    );
}

/// A long-lived process holds a sidecar open while the file at its path is
/// replaced by the sidecar of a new source at the same path: the next load
/// of the new source opens the new file — another device and inode — and
/// passes, instead of checking the held one against the new source and
/// refusing a good file.
#[test]
fn a_sidecar_replaced_at_its_path_is_opened_anew() {
    let old = r8layer::Layer::write("replaced", 512, 256, 4, GgmlType::Q4_K);
    let split = Split::open(&old.source).unwrap();
    let held = HostR8::at_load(&split, true).unwrap();
    assert!(held.sidecar().is_some(), "the first load reads the sidecar");
    let new = r8layer::Layer::write_salted("replaced", 512, 256, 4, GgmlType::Q4_K, 0x5a17);
    assert_eq!(
        new.sidecar, old.sidecar,
        "the new sidecar is at the same path"
    );
    let split = Split::open(&new.source).unwrap();
    let again = HostR8::at_load(&split, true)
        .unwrap_or_else(|e| panic!("the new source's own sidecar must be read: {e}"));
    let (was, now) = (held.sidecar().unwrap(), again.sidecar().unwrap());
    assert!(
        !std::sync::Arc::ptr_eq(was, now),
        "the replaced file is opened anew, not the held one handed back"
    );
    println!("replaced sidecar: the new file at the path is opened and passes");
}

/// A sidecar that holds a layer's down stack — a Q3_K down converted beside
/// the gate and the up — passes its identity check, and the host layer built
/// from it is refused by name (`R8Error::HoldsDown`): a host tier reads only
/// a gate and an up from a sidecar.
#[test]
#[ignore = "hw: the box's CPU (a host layer's r8 stack needs qdot's fused Q3_K); reads no model file"]
fn hw_a_host_layer_refuses_a_sidecar_that_holds_its_down() {
    let layer = r8layer::Layer::write("holds-down", 512, 256, 4, GgmlType::Q3_K);
    std::fs::remove_file(&layer.sidecar).unwrap();
    let split = Split::open(&layer.source).unwrap();
    let all = [r8layer::GATE, r8layer::UP, r8layer::DOWN].map(String::from);
    r8file::convert(&split, &all, &layer.sidecar, &mut |_| {}).unwrap();
    let r8 = HostR8::at_load(&split, true).unwrap();
    assert!(
        r8.sidecar().is_some(),
        "the sidecar passes its identity check"
    );
    match HostLayer::build(R8Source::of(&split, &r8).unwrap(), &layer.spec()) {
        Err(ModelError::R8(R8Error::HoldsDown { tensor, .. })) => {
            assert_eq!(tensor, r8layer::DOWN);
        }
        Err(other) => panic!("refused, but not as holding the down: {other}"),
        Ok(_) => panic!("a sidecar holding the down must be refused"),
    }
    println!("host layer: a sidecar holding {} is refused", r8layer::DOWN);
}

/// A resident sidecar is an anonymous copy of its file: `PageDrop` refuses to
/// release its pages by name (`R8Error::ResidentCopy`), releases nothing, and
/// the copy keeps its bytes; the mapped sidecar of the same file is released
/// as a shard is. A pair that reads no sidecar has none to release
/// (`R8Error::PairReadsNone`).
#[test]
fn a_resident_sidecar_is_not_released() {
    let layer = r8layer::Layer::write("resident", 512, 256, 4, GgmlType::Q4_K);
    let split = Split::open(&layer.source).unwrap();
    let mapped = HostR8::On(std::sync::Arc::new(
        Sidecar::open(&layer.sidecar, &split, LAZY).unwrap(),
    ));
    let resident = HostR8::On(std::sync::Arc::new(
        Sidecar::open(&layer.sidecar, &split, Weights::Resident { huge: false }).unwrap(),
    ));
    let gate = |r8: &HostR8| r8.sidecar().unwrap().data(r8layer::GATE).unwrap().to_vec();
    let want = gate(&mapped);
    assert!(
        want.iter().any(|&b| b != 0),
        "the gate's sidecar bytes are not all zero"
    );
    let mut drop = PageDrop::of(R8Source::of(&split, &resident).unwrap());
    match drop.release_sidecar() {
        Err(PlacementError::R8(R8Error::ResidentCopy { path })) => {
            assert_eq!(path, layer.sidecar);
        }
        other => panic!("a resident sidecar must be refused by name, got {other:?}"),
    }
    assert_eq!(drop.bytes(), 0, "nothing is released");
    assert!(gate(&resident) == want, "the resident copy keeps its bytes");
    let mut none = PageDrop::new(&split);
    match none.release_sidecar() {
        Err(PlacementError::R8(R8Error::PairReadsNone { .. })) => {}
        other => panic!("a pair with no sidecar has none to release, got {other:?}"),
    }
    let mut drop = PageDrop::of(R8Source::of(&split, &mapped).unwrap());
    drop.release_sidecar().unwrap();
    assert!(drop.bytes() > 0, "the mapped sidecar's pages are released");
    assert!(
        gate(&mapped) == want,
        "a released mapping reads its file's bytes again"
    );
    println!(
        "page drop: a resident sidecar is refused; the mapped one releases {} B",
        drop.bytes()
    );
}
