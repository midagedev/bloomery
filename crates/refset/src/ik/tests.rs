use super::{
    Layout, RefManifest, RowKind, find_int_row, find_ref_row_in, mask_bits_in,
    topk_ids_logical_within, widened_f16_bits_in, widened_f16_rows_in,
};
use crate::RefError;
use crate::family::{Build, Family, Identity};
use std::path::{Path, PathBuf};

/// The column lines `dump_ref` writes: the v1 tensor line (before the v2
/// columns existed), the v2 tensor and input lines, and the int line.
const KIND_V1: &str = "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top";
const KIND_V2: &str = "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1";
const INPUT_V1: &str = "# input\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top";
const INPUT_V2: &str = "# input\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1";
const INT: &str =
    "# int\tname\toccurrence\tof\ttype\ttwin\tlayout\tcount\tbytes\tsum\tabsmax\tfile";

/// A fresh directory for one test's set; tests of one process run in
/// parallel, so each passes its own `what`.
fn set_dir(what: &str) -> Result<PathBuf, RefError> {
    let dir = std::env::temp_dir().join(format!("bloomery-refset-{what}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(dir)
}

fn write(path: &Path, bytes: &[u8]) {
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// `lines` as the set's MANIFEST.tsv, read back.
fn manifest(dir: &Path, lines: &[&str]) -> Result<RefManifest, RefError> {
    write(
        &dir.join("MANIFEST.tsv"),
        (lines.join("\n") + "\n").as_bytes(),
    );
    RefManifest::read(dir)
}

fn f32_file(path: &Path, vals: &[f32]) {
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    write(path, &bytes);
}

/// The top-k reader takes the exact `.logical.i32` twin when the
/// manifest has `int` rows and the `.logical.f32` twin when it has none
/// — the f32 file below disagrees in one id, so the result shows which
/// file was read — and a manifest with twins but none for the row is an
/// error. Ids are bounded by the caller's expert count. The name has a
/// space, so the files resolve only through the dumper's `safe_name` rule.
#[test]
fn topk_ids_take_the_integer_twin_when_the_set_has_twins() -> Result<(), RefError> {
    let dir = set_dir("topk-twin")?;
    let ids: [i32; 4] = [3, 63, 0, 17];
    let stem = "ffn_moe_topk-1_(view).0.logical";
    write(
        &dir.join(format!("{stem}.i32")),
        &ids.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    f32_file(&dir.join(format!("{stem}.f32")), &[3.0, 63.0, 0.0, 16.0]);
    let tensor =
        "tensor\tffn_moe_topk-1 (view)\t0\ti32\t2\t2\t1\t1\t16\t0.000000\tVIEW\t0\t1\t-\t-";
    let int = format!(
        "int\tffn_moe_topk-1 (view)\t0\ttensor\ti32\ti32\tlogical\t4\t16\t83\t63\t{stem}.i32"
    );
    let other = int.replacen("\t0\t", "\t1\t", 1);
    let twins = manifest(&dir, &[KIND_V2, INT, tensor, &int])?;
    assert_eq!(topk_ids_logical_within(&twins, &twins.tensors[0], 64)?, ids);
    assert!(topk_ids_logical_within(&twins, &twins.tensors[0], 63).is_err());
    let cast = manifest(&dir, &[KIND_V2, tensor])?;
    assert_eq!(
        topk_ids_logical_within(&cast, &cast.tensors[0], 64)?,
        [3, 63, 0, 16]
    );
    assert!(topk_ids_logical_within(&cast, &cast.tensors[0], 63).is_err());
    let unmatched = manifest(&dir, &[KIND_V2, INT, tensor, &other])?;
    assert!(topk_ids_logical_within(&unmatched, &unmatched.tensors[0], 64).is_err());
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// The index finds a row by `(kind, name, occurrence)` — a tensor and an
/// input of one name are two rows — and of a key two rows carry, the
/// first, the row a scan in file order finds; the second is counted as a
/// duplicate. An `int` row is found by its twin's kind and layout too.
/// The positional reads agree with a scan: a row's position, the first
/// row of a name whose first written occurrence is not 0, the last row
/// of a name before a reader, and an input's first toucher, the tensor
/// row after it. A name prefix finds its one row, and more than one or
/// none is an error.
#[test]
fn the_index_finds_the_row_a_scan_finds_first() -> Result<(), RefError> {
    let dir = set_dir("index")?;
    let man = manifest(
        &dir,
        &[
            KIND_V1,
            INPUT_V1,
            INT,
            "tensor\tx\t0\tf32\t1\t1\t1\t1\t4\t1.0\tADD",
            "tensor\tx\t1\tf32\t1\t1\t1\t1\t4\t2.0\tADD",
            "input\tx\t0\tf32\t1\t1\t1\t1\t4\t3.0\tNONE",
            "tensor\tx\t0\tf32\t1\t1\t1\t1\t4\t4.0\tADD",
            "int\tp\t0\tinput\ti32\ti32\tflat\t1\t4\t7\t7\tp.0.input.i32",
            "int\tp\t0\tinput\ti32\ti32\tlogical\t1\t4\t7\t7\tp.0.input.logical.i32",
            "skip\ty\t0\tq8_0\tquantized",
            "tensor\ty\t1\tf32\t1\t1\t1\t1\t4\t5.0\tADD",
        ],
    )?;
    let sum = |r: Option<&super::RefRow>| r.map(|r| r.sum);
    assert_eq!(sum(man.find(RowKind::Tensor, "x", 0)), Some(1.0));
    assert_eq!(
        sum(man.find(RowKind::Tensor, "x", 0)),
        Some(find_ref_row_in(&dir, &man.tensors, "x", 0)?.sum)
    );
    assert_eq!(man.tensor("x", 1)?.sum, 2.0);
    assert_eq!(man.input("x", 0)?.sum, 3.0);
    assert!(man.find(RowKind::Input, "x", 1).is_none());
    assert!(man.tensor("y", 0).is_err());
    assert_eq!(man.duplicate_keys(), 1);
    let twin = find_int_row(&man, "p", 0, RowKind::Input, Layout::Logical)?;
    assert_eq!(twin.file, "p.0.input.logical.i32");
    assert!(find_int_row(&man, "p", 0, RowKind::Tensor, Layout::Flat).is_err());

    assert_eq!(man.position(RowKind::Tensor, "x", 1), Some(1));
    assert_eq!(man.tensor_at("y", 1)?.0, 3);
    assert_eq!(sum(man.first_named(RowKind::Tensor, "x")), Some(1.0));
    assert_eq!(sum(man.first_named(RowKind::Tensor, "y")), Some(5.0));
    assert!(man.first_named(RowKind::Input, "y").is_none());
    assert_eq!(man.last_before(2, Some("x"))?.0, 1);
    assert!(man.last_before(0, Some("x")).is_err());
    assert!(man.last_before(3, None).is_err());
    let touched = |at: usize| {
        man.first_touched_by(at)
            .iter()
            .map(|r| r.sum)
            .collect::<Vec<_>>()
    };
    assert_eq!(touched(2), [3.0]);
    assert!(touched(1).is_empty() && touched(3).is_empty());

    assert_eq!(man.only_with_prefix(RowKind::Input, "x")?.sum, 3.0);
    assert_eq!(man.only_with_prefix(RowKind::Tensor, "y")?.sum, 5.0);
    assert!(man.only_with_prefix(RowKind::Tensor, "x").is_err());
    assert!(man.only_with_prefix(RowKind::Input, "y").is_err());
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// Every header line the V4.1 and V2-Lite sets carry, spelled as they
/// spell it (long values shortened): a known key fills its field — `-c`
/// and `-t` out of `# flags`, the step out of `# tokens` and `# decode_pos` —
/// every other `#` line is kept verbatim in file order, and a known key
/// twice or a value that does not parse is an error.
#[test]
fn header_lines_fill_their_fields_or_are_kept() -> Result<(), RefError> {
    let dir = set_dir("header")?;
    let title = "# dump_ref — ik_llama.cpp intermediate tensors, raw f32, little-endian";
    let model = "# model\t/models/M/M-00001-of-00009.gguf";
    let schedule = "# prefill_schedule\tevery-node — each node was asked for and computed alone";
    let row = "tensor\tinp_embd\t0\tf32\t2\t1\t1\t1\t8\t0.5\tGET_ROWS";
    let row_v2 = format!("{row}\t1\t0\t-\t-");
    let step = [
        title,
        model,
        "# build\t49ef19d0",
        "# arch\tdeepseek41",
        "# model_file\tM-00001-of-00009.gguf",
        "# tokens\t5,19415,271",
        "# flags\t-m /models/M/M-00001-of-00009.gguf --expect-arch deepseek41 --tokens-file /d/c.ids \
         --tokens-count 3 -ngl 0 -c 2048 -t 32 --defer-experts --decode-step --no-fused-idx-topk",
        "# tokens_file\t/d/c.ids",
        "# tokens_file_sha256\tf7785d0f",
        "# tokens_count\t3",
        "# prefill\t2",
        "# decode_pos\t2",
        schedule,
        "# state_inputs\tpersistent leaves the step reads are input rows",
        "# fused_idx_topk\t0",
        KIND_V2,
        INT,
        INPUT_V2,
        &row_v2,
        "# complete\t1\t0",
    ];
    let man = manifest(&dir, &step)?;
    let h = &man.header;
    assert_eq!(man.arch.as_deref(), Some("deepseek41"));
    assert_eq!(man.build.as_deref(), Some("49ef19d0"));
    assert_eq!(man.complete, Some((1, 0)));
    assert_eq!(h.model_file.as_deref(), Some("M-00001-of-00009.gguf"));
    assert_eq!(h.model(), Some("/models/M/M-00001-of-00009.gguf"));
    assert_eq!(h.tokens.as_deref(), Some(&[5, 19415, 271][..]));
    assert_eq!(h.tokens_file.as_deref(), Some("/d/c.ids"));
    assert_eq!(h.tokens_file_sha256.as_deref(), Some("f7785d0f"));
    assert_eq!(h.tokens_count, Some(3));
    assert!(
        h.flags
            .as_deref()
            .is_some_and(|f| f.ends_with("--no-fused-idx-topk"))
    );
    assert_eq!((h.ctx, h.threads), (Some(2048), Some(32)));
    assert_eq!((h.prefill, h.decode_pos), (Some(2), Some(2)));
    assert!(h.state_inputs.is_some());
    assert_eq!(h.fused_idx_topk, Some(false));
    assert_eq!(h.other, [title, model, schedule, KIND_V2, INT, INPUT_V2]);
    assert_eq!(man.step()?, (2, &[271][..], &[5, 19415][..]));

    // A V2-Lite set: no V4.1 line, the 11-column rows; its step is the
    // whole sequence from position 0.
    let v2 = manifest(
        &dir,
        &[
            title,
            "# model\t/models/small/L.gguf",
            "# build\tc10fbbcc",
            "# tokens\t100000,549",
            KIND_V1,
            row,
            "# complete\t1\t0",
        ],
    )?;
    let h = &v2.header;
    assert_eq!(
        (v2.arch.as_deref(), v2.build.as_deref()),
        (None, Some("c10fbbcc"))
    );
    assert_eq!(
        (
            h.model_file.as_deref(),
            h.flags.as_deref(),
            h.ctx,
            h.threads
        ),
        (None, None, None, None)
    );
    assert_eq!(
        (h.prefill, h.decode_pos, h.fused_idx_topk),
        (None, None, None)
    );
    assert_eq!(h.other.len(), 3);
    assert_eq!(v2.tensors[0].contig, None);
    assert_eq!(v2.step()?, (0, &[100000, 549][..], &[][..]));

    assert!(manifest(&dir, &[KIND_V1, "# build\tc10fbbcc", row]).is_ok());
    for bad in [
        "# tokens\t5,x",
        "# flags\t-m M.gguf -c",
        "# flags\t-m M.gguf -t x -c 512",
        "# fused_idx_topk\tyes",
        "# decode_pos\t-1",
        "# build\tc10fbbcc",
        KIND_V1,
    ] {
        let lines = [KIND_V1, "# build\tc10fbbcc", bad, row];
        assert!(manifest(&dir, &lines).is_err(), "{bad:?} parsed");
    }
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// Rows are read by the names their column line gives them, not by
/// position: a line that orders the columns otherwise reads the same row.
/// A row of a kind no column line names, a row wider or narrower than its
/// line, a line without a field the reader needs, and a field that does not
/// parse are each `Malformed`, naming the line.
#[test]
fn rows_are_read_by_their_column_names() -> Result<(), RefError> {
    let dir = set_dir("columns")?;
    let row = "tensor\tx\t0\tf32\t2\t3\t1\t1\t24\t1.5\tMUL_MAT\t1\t0\ta\tb";
    let plain = manifest(&dir, &[KIND_V2, row])?;
    let shuffled = manifest(
        &dir,
        &[
            "# kind\top\tname\tsum\toccurrence\tsrc1\ttype\tne0\tlogical\tne1\tcontig\tne2\tsrc0\tne3\tbytes",
            "tensor\tMUL_MAT\tx\t1.5\t0\tb\tf32\t2\t0\t3\t1\t1\ta\t1\t24",
        ],
    )?;
    for m in [&plain, &shuffled] {
        let r = &m.tensors[0];
        assert_eq!(
            (r.name.as_str(), r.occurrence, r.ty.as_str(), r.ne, r.bytes),
            ("x", 0, "f32", [2, 3, 1, 1], 24)
        );
        assert_eq!(
            (r.sum, r.op.as_str(), r.contig, r.logical),
            (1.5, "MUL_MAT", Some(1), Some(0))
        );
        assert_eq!(
            (r.src0.as_deref(), r.src1.as_deref()),
            (Some("a"), Some("b"))
        );
    }
    let malformed = |lines: &[&str]| match manifest(&dir, lines) {
        Err(RefError::Malformed { at, what }) => format!("{at}: {what}"),
        other => panic!("{lines:?}: {other:?}"),
    };
    let e = malformed(&[row]);
    assert!(e.contains("before any # kind column line"), "{e}");
    let e = malformed(&[KIND_V1, row]);
    assert!(e.contains("of 15 fields, its column line names 11"), "{e}");
    let e = malformed(&[&KIND_V2.replace("\top\t", "\toperator\t"), row]);
    assert!(e.contains("names no op field"), "{e}");
    let e = malformed(&[KIND_V2, &row.replace("\t24\t", "\t2x\t")]);
    assert!(e.contains("bytes \"2x\""), "{e}");
    let e = malformed(&[KIND_V2, KIND_V2, row]);
    assert!(e.contains("a second # kind column line"), "{e}");
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// The file a family's test runs.
fn runs() -> String {
    "/models/P/M-00001-of-00009.gguf".to_string()
}

/// A family of the ik node dumps' shape, pinned to build `b0` and
/// architecture `a`.
static FAMILY: Family = Family {
    name: "test",
    sets: &[],
    resolve: None,
    recipe: "",
    identity: Identity::Manifest,
    arch: Some("a"),
    build: Some(Build::Is("b0")),
    runs: Some(runs),
    draft_runs: None,
    consumers: &[],
};

/// A set is checked against its family before any comparison: a set of
/// another file, or of none named, is `Stale`, naming the set, the file it
/// states and the file the tree runs — even when its shard's name is the
/// same; one without its trailer is `Unfinished`; one of another build or
/// architecture is `Foreign`. The family's build is the pin itself: a
/// patched or a `-dirty` tree is not it.
#[test]
fn a_set_of_another_file_is_stale_by_name() -> Result<(), RefError> {
    let dir = set_dir("family")?;
    let row = "tensor\tx\t0\tf32\t1\t1\t1\t1\t4\t0\tNONE";
    let open = |model: &str, build: &str, arch: &str, complete: bool| {
        let mut lines = vec![model, build, arch, KIND_V1, row];
        if complete {
            lines.push("# complete\t1\t0");
        }
        manifest(&dir, &lines).and_then(|m| m.check_family(&FAMILY))
    };
    let ours = "# model\t/models/P/M-00001-of-00009.gguf";
    open(ours, "# build\tb0", "# arch\ta", true)?;
    match open(
        "# model\t/models/Q/M-00001-of-00009.gguf",
        "# build\tb0",
        "# arch\ta",
        true,
    ) {
        Err(e @ RefError::Stale { .. }) => {
            let e = e.to_string();
            assert!(
                e.starts_with("stale reference: ")
                    && e.contains(&dir.display().to_string())
                    && e.contains("dumped from /models/Q/M-00001-of-00009.gguf")
                    && e.contains("the tree runs /models/P/M-00001-of-00009.gguf"),
                "{e}"
            );
        }
        other => panic!("another file: {other:?}"),
    }
    assert!(matches!(
        open(
            "# model_file\tM-00001-of-00009.gguf",
            "# build\tb0",
            "# arch\ta",
            true
        ),
        Err(RefError::Stale { .. })
    ));
    assert!(matches!(
        open(ours, "# build\tb0", "# arch\ta", false),
        Err(RefError::Unfinished { .. })
    ));
    for (build, arch) in [
        ("# build\tb1", "# arch\ta"),
        ("# build\tb0+patch 1", "# arch\ta"),
        ("# build\tb0-dirty", "# arch\ta"),
        ("# build\tb0", "# arch\tz"),
    ] {
        assert!(
            matches!(open(ours, build, arch, true), Err(RefError::Foreign { .. })),
            "{build} {arch}"
        );
    }
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// The mask reader maps `0.0` and `-inf` to their f16 bits and refuses
/// every other value — `-0.0` and NaN too — naming the row and the first
/// offending value, a row that is not f16, and one whose plain file is not
/// its logical order.
#[test]
fn mask_bits_take_zero_and_minus_infinity_only() -> Result<(), RefError> {
    let dir = set_dir("mask")?;
    let node = "tensor\tx\t0\tf32\t1\t1\t1\t1\t4\t0\tNONE";
    let row = "input\tkq_mask\t0\tf16\t2\t2\t1\t1\t16\t-inf\tNONE\t1\t0\t-\t-";
    let man = manifest(&dir, &[KIND_V1, INPUT_V2, node, row])?;
    let file = dir.join(man.inputs[0].file_name());
    let ninf = f32::NEG_INFINITY;
    f32_file(&file, &[0.0, ninf, ninf, 0.0]);
    assert_eq!(
        mask_bits_in(&dir, &man.inputs[0])?,
        [0x0000, 0xfc00, 0xfc00, 0x0000]
    );
    for (bad, shown) in [(-0.0f32, "-0"), (1.0, "1"), (f32::NAN, "NaN")] {
        f32_file(&file, &[0.0, ninf, bad, bad]);
        let e = mask_bits_in(&dir, &man.inputs[0])
            .map(|_| ())
            .map_err(|e| e.to_string())
            .expect_err("a value neither 0 nor -inf");
        assert!(
            e.contains("kq_mask/0") && e.contains(&format!("holds {shown} at 2,")),
            "{e}"
        );
    }
    let not_f16 = manifest(
        &dir,
        &[KIND_V1, INPUT_V2, node, &row.replace("\tf16\t", "\tf32\t")],
    )?;
    assert!(mask_bits_in(&dir, &not_f16.inputs[0]).is_err());
    f32_file(&file, &[0.0, ninf, ninf, 0.0]);
    let flat = manifest(
        &dir,
        &[
            KIND_V1,
            INPUT_V2,
            node,
            &row.replace("NONE\t1\t0", "NONE\t0\t0"),
        ],
    )?;
    assert!(mask_bits_in(&dir, &flat.inputs[0]).is_err());
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// The widened-f16 reader returns the rows asked for, in the order asked,
/// as f16 bits, and reads no other row: a row holding a value no half
/// widens to is refused when asked for and passes unread otherwise; a
/// row past the tensor is an error. The whole-tensor reader asks for
/// every row. Both refuse a row whose plain file is not its logical order.
#[test]
fn widened_f16_bits_read_only_the_rows_asked_for() -> Result<(), RefError> {
    let dir = set_dir("widened")?;
    let row = "tensor\tcache\t0\tf16\t2\t3\t1\t1\t24\t0\tVIEW\t1\t0\t-\t-";
    let man = manifest(&dir, &[KIND_V2, row])?;
    let cache = &man.tensors[0];
    let min_normal = 2.0f32.powi(-14);
    f32_file(
        &dir.join(cache.file_name()),
        &[1.0, -2.0, 0.1, 0.5, 65504.0, min_normal],
    );
    assert_eq!(
        widened_f16_bits_in(&dir, cache, &[2, 0])?,
        [0x7bff, 0x0400, 0x3c00, 0xc000]
    );
    assert!(widened_f16_bits_in(&dir, cache, &[1]).is_err());
    assert!(widened_f16_bits_in(&dir, cache, &[3]).is_err());
    let not_f16 = manifest(&dir, &[KIND_V2, &row.replace("\tf16\t", "\tf32\t")])?;
    assert!(widened_f16_bits_in(&dir, &not_f16.tensors[0], &[0]).is_err());

    assert!(widened_f16_rows_in(&dir, cache).is_err());
    f32_file(
        &dir.join(cache.file_name()),
        &[1.0, -2.0, 0.5, 0.25, 65504.0, min_normal],
    );
    assert_eq!(
        widened_f16_rows_in(&dir, cache)?,
        [0x3c00, 0xc000, 0x3800, 0x3400, 0x7bff, 0x0400]
    );
    let view = manifest(&dir, &[KIND_V2, &row.replace("VIEW\t1\t0", "VIEW\t0\t0")])?;
    assert!(widened_f16_rows_in(&dir, &view.tensors[0]).is_err());
    assert!(widened_f16_bits_in(&dir, &view.tensors[0], &[0]).is_err());
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}
