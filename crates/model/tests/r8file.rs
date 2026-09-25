//! r8 sidecar gates (`model::r8file`), on a synthetic source written with
//! `gguf::write`: two shards holding three Q3_K stacks of random bytes on the
//! grids, one off the 8-row grid and an F32 tensor. `convert`, `Sidecar::open`
//! and `verify` pass and the sidecar holds `repack_q3k_r8` of each stack;
//! every identity difference, every conversion refusal and a flipped sidecar
//! byte are named; the `r8conv` binary converts and verifies the same source
//! end to end. No clause reads a model file.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::process::Command;

use gguf::write::{Layout, TensorDecl, WriteError, Writer};
use gguf::{GgmlType, Gguf, LoadError, PrivateType, Split, Value, Weights};
use model::ModelError;
use model::r8file::{self, Progress, R8Error, Sidecar};

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
/// sidecar at `out`.
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
    match r8_err(r8file::convert(&split, &names(), &out, &mut |_| {})) {
        R8Error::Busy { path } => assert_eq!(path, part),
        other => panic!("wrong refusal: {other}"),
    }
    assert_eq!(std::fs::read(&part).unwrap(), b"live");
    assert!(!out.exists());
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
    std::fs::remove_dir_all(&d).unwrap();
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
            r8file::sidecar_path(Path::new(first)),
            PathBuf::from(want),
            "{first}"
        );
    }
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
        r8file::sidecar_path(&first).to_str().unwrap()
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
