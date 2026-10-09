use super::{
    Admit, FileElem, Layout, PrefillRoutes, RefManifest, RowKind, dump_file_name, find_int_row,
    find_ref_row_in, mask_bits_in, refused, topk_ids_logical_within, widened_f16_bits_in,
    widened_f16_rows_in,
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
fn runs() -> Result<String, RefError> {
    Ok("/models/P/M-00001-of-00009.gguf".to_string())
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

#[test]
fn a_masked_row_admits_minus_inf_past_its_finite_slots_only() {
    let inf = f32::NEG_INFINITY;
    let rows = [1.0, 2.0, inf, 3.0, 4.0, inf];
    assert_eq!(refused(&rows, 3, Admit::MaskedPast(2)), None);
    assert_eq!(refused(&rows, 3, Admit::Finite), Some(2));
    assert_eq!(refused(&[1.0, inf, inf], 3, Admit::MaskedPast(2)), Some(1));
    assert_eq!(refused(&[1.0, 2.0, 0.0], 3, Admit::MaskedPast(2)), Some(2));
    assert_eq!(
        refused(&[1.0, 2.0, f32::NAN], 3, Admit::MaskedPast(2)),
        Some(2)
    );
    assert_eq!(
        refused(&[f32::NAN, 2.0, inf], 3, Admit::MaskedPast(2)),
        Some(0)
    );
}

const GLM_LAST: &str = "ffn_moe_weights_scaled";
const QWEN_LAST: &str = "ffn_moe_weights_norm";
const ROUTE_HEADER: &str = "# prefill_routes\tik's routing of the prefill's positions: the nodes \
    ffn_moe_topk-<layer>, ffn_moe_weights-<layer>, ffn_moe_weights_scaled-<layer> of each ubatch \
    of the quiet prefill (n_ubatch 512) are tensor rows named prefill.<node>, one occurrence a \
    ubatch in prefill order, their positions following in order from 0";
/// The experts a position picks in every synthetic route set.
const N_USED: usize = 2;

/// The value of cell `j` of position `g` of `layer` in a route set; the picks
/// of a position are distinct and below 7.
fn route_pick(layer: u32, g: usize, j: usize) -> i32 {
    ((layer as usize + g + j) % 7) as i32
}

fn route_raw(layer: u32, g: usize, j: usize) -> f32 {
    (layer as usize * 100 + g * 10 + j + 1) as f32 / 7.0
}

fn route_last(layer: u32, g: usize, j: usize) -> f32 {
    route_raw(layer, g, j) * 2.5 / 3.0
}

fn bits(vals: &[f32]) -> Vec<u32> {
    vals.iter().map(|v| v.to_bits()).collect()
}

/// A route row of the v2 columns; `bytes` is the f32 read of `ne`'s elements.
fn route_row(name: &str, occ: u32, ty: &str, ne: [usize; 4], op: &str, view: bool) -> String {
    let (contig, logical) = if view { (0, 1) } else { (1, 0) };
    let bytes = 4 * ne.iter().product::<usize>();
    format!(
        "tensor\t{name}\t{occ}\t{ty}\t{}\t{}\t{}\t{}\t{bytes}\t0.000000\t{op}\t{contig}\t{logical}\t-\t-",
        ne[0], ne[1], ne[2], ne[3]
    )
}

/// The `int` row of `name`/`occ`'s twin of `ids` in the layout `logical`
/// says, and the file it names, written.
fn int_row(dir: &Path, name: &str, occ: u32, logical: bool, ids: &[i32]) -> String {
    let layout = if logical {
        Layout::Logical
    } else {
        Layout::Flat
    };
    let file = dump_file_name(name, occ, RowKind::Tensor, layout, FileElem::I32);
    write(
        &dir.join(&file),
        &ids.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let sum: i64 = ids.iter().map(|&v| i64::from(v)).sum();
    let absmax = ids.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
    format!(
        "int\t{name}\t{occ}\ttensor\ti32\ti32\t{}\t{}\t{}\t{sum}\t{absmax}\t{file}",
        layout.as_str(),
        ids.len(),
        4 * ids.len()
    )
}

/// The row of a weights file written to `dir`.
fn weights_row(dir: &Path, name: &str, occ: u32, ne: [usize; 4], op: &str, vals: &[f32]) -> String {
    let file = dump_file_name(name, occ, RowKind::Tensor, Layout::Flat, FileElem::F32);
    f32_file(&dir.join(file), vals);
    route_row(name, occ, "f32", ne, op, false)
}

/// The manifest lines of a decode-step set with ik's routing of the prefill
/// in ubatches of `widths` positions, `layers` written in that order, and
/// its files in `dir`. The picks' flat twin is the first position's ids over
/// and over, as the flat read of the view is only a parent's elements. The
/// normalised weights of `QWEN_LAST` are `[n_used, positions]`, the gathered
/// and scaled ones `[1, n_used, positions]`.
fn route_set(dir: &Path, widths: &[usize], layers: &[u32], last_stem: &str) -> Vec<String> {
    let mut lines = vec![
        format!("# prefill\t{}", widths.iter().sum::<usize>()),
        ROUTE_HEADER.to_string(),
        KIND_V2.to_string(),
        INT.to_string(),
    ];
    let mut first = 0;
    for (u, &w) in (0u32..).zip(widths) {
        for &layer in layers {
            let cells = || (first..first + w).flat_map(|g| (0..N_USED).map(move |j| (g, j)));
            let name = format!("prefill.ffn_moe_topk-{layer}");
            let ids: Vec<i32> = cells().map(|(g, j)| route_pick(layer, g, j)).collect();
            let flat: Vec<i32> = ids[..N_USED]
                .iter()
                .cycle()
                .take(ids.len())
                .copied()
                .collect();
            lines.push(route_row(&name, u, "i32", [N_USED, w, 1, 1], "VIEW", true));
            lines.push(int_row(dir, &name, u, false, &flat));
            lines.push(int_row(dir, &name, u, true, &ids));
            let raw: Vec<f32> = cells().map(|(g, j)| route_raw(layer, g, j)).collect();
            lines.push(weights_row(
                dir,
                &format!("prefill.ffn_moe_weights-{layer}"),
                u,
                [1, N_USED, w, 1],
                "GET_ROWS",
                &raw,
            ));
            let last: Vec<f32> = cells().map(|(g, j)| route_last(layer, g, j)).collect();
            let (ne, op) = if last_stem == QWEN_LAST {
                ([N_USED, w, 1, 1], "DIV")
            } else {
                ([1, N_USED, w, 1], "SCALE")
            };
            lines.push(weights_row(
                dir,
                &format!("prefill.{last_stem}-{layer}"),
                u,
                ne,
                op,
                &last,
            ));
        }
        first += w;
    }
    lines
}

fn routes_manifest(dir: &Path, lines: &[String]) -> Result<RefManifest, RefError> {
    let lines: Vec<&str> = lines.iter().map(String::as_str).collect();
    manifest(dir, &lines)
}

/// The routes of the set `lines` with the stems of `last_stem`'s family.
fn routes_of(dir: &Path, lines: &[String], last_stem: &str) -> Result<PrefillRoutes, RefError> {
    routes_manifest(dir, lines)?.prefill_routes("ffn_moe_topk", "ffn_moe_weights", last_stem)
}

/// What reading the set `lines` (GLM's stems) refuses with.
fn refusal(dir: &Path, lines: &[String]) -> Result<RefError, RefError> {
    match routes_of(dir, lines, GLM_LAST) {
        Err(e) => Ok(e),
        Ok(r) => panic!("the set was read: {r:?}"),
    }
}

/// `e` is a `Missing` (`missing`) or a `Malformed` refusal whose text holds
/// every one of `parts`.
fn assert_refusal(e: &RefError, missing: bool, parts: &[&str]) {
    let kind = if missing {
        matches!(e, RefError::Missing { .. })
    } else {
        matches!(e, RefError::Malformed { .. })
    };
    assert!(kind, "wrong kind of refusal: {e:?}");
    let text = e.to_string();
    for part in parts {
        assert!(text.contains(part), "{part:?} is not in {text}");
    }
}

/// The one line of `lines` that opens with `prefix`.
fn line_at<'a>(lines: &'a mut [String], prefix: &str) -> &'a mut String {
    let mut hits = lines.iter_mut().filter(|l| l.starts_with(prefix));
    let hit = hits.next().unwrap_or_else(|| panic!("no line {prefix:?}"));
    assert!(hits.next().is_none(), "two lines {prefix:?}");
    hit
}

/// Row `name`/`occ` of `lines` with the dims `ne` and the bytes they hold.
fn set_ne(lines: &mut [String], name: &str, occ: u32, ne: [usize; 4]) {
    let row = line_at(lines, &format!("tensor\t{name}\t{occ}\t"));
    let mut fields: Vec<String> = row.split('\t').map(str::to_string).collect();
    for (field, dim) in fields[4..8].iter_mut().zip(ne) {
        *field = dim.to_string();
    }
    fields[8] = (4 * ne.iter().product::<usize>()).to_string();
    *row = fields.join("\t");
}

/// ik's routing of a prefill of two ubatches (3 and 2 positions) in layers 5
/// and 2, written in that order: the accessor returns the layers ascending,
/// each with the five positions in order and every pick and weight exactly
/// as written. The picks come from the logical twin: the flat twin the set
/// also holds differs.
#[test]
fn prefill_routes_hold_every_layer_over_the_ubatches_in_order() -> Result<(), RefError> {
    let dir = set_dir("routes-read")?;
    let lines = route_set(&dir, &[3, 2], &[5, 2], GLM_LAST);
    let routes = routes_of(&dir, &lines, GLM_LAST)?;
    let mut reversed = lines.clone();
    reversed[4..].reverse();
    assert_eq!(routes_of(&dir, &reversed, GLM_LAST)?, routes);
    assert_eq!(
        (routes.positions, routes.ubatches, routes.n_used),
        (5, 2, 2)
    );
    assert_eq!(
        routes.layers.iter().map(|l| l.layer).collect::<Vec<_>>(),
        [2, 5]
    );
    for l in &routes.layers {
        let cells = || (0..5).flat_map(|g| (0..N_USED).map(move |j| (g, j)));
        let picks: Vec<u32> = cells()
            .map(|(g, j)| route_pick(l.layer, g, j) as u32)
            .collect();
        let raw: Vec<f32> = cells().map(|(g, j)| route_raw(l.layer, g, j)).collect();
        let last: Vec<f32> = cells().map(|(g, j)| route_last(l.layer, g, j)).collect();
        assert_eq!(l.picks, picks, "layer {}", l.layer);
        assert_eq!(bits(&l.raw), bits(&raw), "layer {}", l.layer);
        assert_eq!(bits(&l.last), bits(&last), "layer {}", l.layer);
    }
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// Qwen3.8's final weights are the normalised ones, `[n_used, positions]`,
/// where the gathered ones are `[1, n_used, positions]`; a ubatch of one
/// position reads the same.
#[test]
fn prefill_routes_take_the_normalised_shape_and_a_ubatch_of_one() -> Result<(), RefError> {
    let dir = set_dir("routes-norm")?;
    let lines = route_set(&dir, &[2, 1], &[0, 1], QWEN_LAST);
    let routes = routes_of(&dir, &lines, QWEN_LAST)?;
    assert_eq!((routes.positions, routes.ubatches), (3, 2));
    for l in &routes.layers {
        let last: Vec<f32> = (0..3 * N_USED)
            .map(|i| route_last(l.layer, i / N_USED, i % N_USED))
            .collect();
        assert_eq!(bits(&l.last), bits(&last), "layer {}", l.layer);
        assert_eq!(l.picks.len(), 3 * N_USED);
    }
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// A set dumped without `--prefill-routes` has no `# prefill_routes` line,
/// and a caller tests for that before planting: the error is `Missing` and
/// names the line and the set.
#[test]
fn a_set_without_the_routes_header_is_missing_by_name() -> Result<(), RefError> {
    let dir = set_dir("routes-header")?;
    let step = [
        "# prefill\t5",
        KIND_V2,
        "tensor\tffn_moe_topk-2\t0\ti32\t2\t1\t1\t1\t8\t0.000000\tVIEW\t0\t1\t-\t-",
    ];
    let man = manifest(&dir, &step)?;
    let e = man
        .prefill_routes("ffn_moe_topk", "ffn_moe_weights", GLM_LAST)
        .expect_err("a step set without routes");
    assert_refusal(
        &e,
        true,
        &["has no # prefill_routes line", &dir.display().to_string()],
    );

    let mut lines = route_set(&dir, &[3, 2], &[2], GLM_LAST);
    lines.retain(|l| !l.starts_with("# prefill_routes\t"));
    let e = refusal(&dir, &lines)?;
    assert_refusal(&e, true, &["has no # prefill_routes line"]);
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// A route set without its `# prefill` line is `Missing` by name: the
/// positions of the prefill are not stated.
#[test]
fn a_route_set_without_the_prefill_line_is_missing_by_name() -> Result<(), RefError> {
    let dir = set_dir("routes-prefill")?;
    let mut lines = route_set(&dir, &[3, 2], &[2], GLM_LAST);
    lines.retain(|l| !l.starts_with("# prefill\t"));
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        true,
        &["has no # prefill line", &dir.display().to_string()],
    );
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// A stem with no row at all is `Missing`, naming the stem, whichever of
/// the three it is.
#[test]
fn a_stem_with_no_row_is_missing_by_name() -> Result<(), RefError> {
    let dir = set_dir("routes-stem")?;
    for stem in ["ffn_moe_topk", "ffn_moe_weights", GLM_LAST] {
        let mut lines = route_set(&dir, &[3, 2], &[2, 5], GLM_LAST);
        lines.retain(|l| !l.contains(&format!("\tprefill.{stem}-")));
        let e = refusal(&dir, &lines)?;
        assert_refusal(
            &e,
            true,
            &[
                &format!("has no tensor row prefill.{stem}-<layer>"),
                &dir.display().to_string(),
            ],
        );
    }
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// The three stems name the same layers: one lacking a layer, or holding
/// another, is refused naming both sets of layers.
#[test]
fn stems_naming_different_layers_are_refused() -> Result<(), RefError> {
    let dir = set_dir("routes-layers")?;
    let mut lines = route_set(&dir, &[3, 2], &[2, 5], GLM_LAST);
    let whole = lines.clone();
    lines.retain(|l| !l.starts_with("tensor\tprefill.ffn_moe_weights-5\t"));
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        false,
        &[
            "different layers",
            "ffn_moe_topk has [2, 5]",
            "ffn_moe_weights has [2]",
            &dir.display().to_string(),
        ],
    );

    let mut lines = whole;
    let extra: Vec<String> = lines
        .iter()
        .filter(|l| l.starts_with(&format!("tensor\tprefill.{GLM_LAST}-2\t")))
        .map(|l| l.replace("-2\t", "-9\t"))
        .collect();
    lines.extend(extra);
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        false,
        &["different layers", &format!("{GLM_LAST} has [2, 5, 9]")],
    );
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// A row's occurrences run 0..U: a gap, a start at 1 and a row written twice
/// are each refused naming the row.
#[test]
fn occurrences_must_run_from_zero_without_a_gap_or_a_repeat() -> Result<(), RefError> {
    let dir = set_dir("routes-occ")?;
    let whole = route_set(&dir, &[3, 2], &[2], GLM_LAST);
    let head = "tensor\tprefill.ffn_moe_topk-2\t";

    let mut lines = whole.clone();
    let row = line_at(&mut lines, &format!("{head}1\t"));
    *row = row.replacen("topk-2\t1\t", "topk-2\t2\t", 1);
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        false,
        &["prefill.ffn_moe_topk-2 has occurrences [0, 2], want 0..2"],
    );

    let mut lines = whole.clone();
    lines.retain(|l| !l.starts_with(&format!("{head}0\t")));
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        false,
        &["prefill.ffn_moe_topk-2 has occurrences [1], want 0..1"],
    );

    let mut lines = whole;
    let twice = line_at(&mut lines, &format!("{head}0\t")).clone();
    lines.push(twice);
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        false,
        &["prefill.ffn_moe_topk-2 has occurrences [0, 0, 1], want 0..3"],
    );
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// Every row of every stem and layer has the same number of occurrences: a
/// stem short of a ubatch, though its own occurrences run from 0, is refused.
#[test]
fn every_row_has_the_same_ubatches() -> Result<(), RefError> {
    let dir = set_dir("routes-ubatches")?;
    let mut lines = route_set(&dir, &[3, 2], &[2, 5], GLM_LAST);
    lines.retain(|l| !l.starts_with(&format!("tensor\tprefill.{GLM_LAST}-5\t1\t")));
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        false,
        &[&format!(
            "prefill.{GLM_LAST}-5 has 1 occurrences, prefill.ffn_moe_topk-2 has 2"
        )],
    );
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// The ubatches' widths sum to `# prefill`.
#[test]
fn ubatch_widths_must_sum_to_the_prefill() -> Result<(), RefError> {
    let dir = set_dir("routes-sum")?;
    let mut lines = route_set(&dir, &[3, 2], &[2], GLM_LAST);
    *line_at(&mut lines, "# prefill\t") = "# prefill\t6".to_string();
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        false,
        &[
            "count [3, 2] positions, # prefill says 6",
            &dir.display().to_string(),
        ],
    );
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// The stems count the same positions at one occurrence, each from its own
/// row's ne.
#[test]
fn the_stems_must_count_the_same_positions_at_an_occurrence() -> Result<(), RefError> {
    let dir = set_dir("routes-agree")?;
    let whole = route_set(&dir, &[3, 2], &[2], GLM_LAST);
    let last = format!("prefill.{GLM_LAST}-2");
    let mut lines = whole.clone();
    set_ne(&mut lines, &last, 0, [1, N_USED, 2, 1]);
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        false,
        &[&format!(
            "{last}/0 counts 2 positions where prefill.ffn_moe_topk-2/0 counts 3"
        )],
    );

    let mut lines = whole;
    set_ne(
        &mut lines,
        "prefill.ffn_moe_weights-2",
        1,
        [1, N_USED, 3, 1],
    );
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        false,
        &["prefill.ffn_moe_weights-2/1 counts 3 positions where prefill.ffn_moe_topk-2/1 counts 2"],
    );
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// Every layer cuts the prefill into the same ubatches, even when each
/// layer's own stems agree and its widths sum to `# prefill`.
#[test]
fn every_layer_must_cut_the_same_ubatches() -> Result<(), RefError> {
    let dir = set_dir("routes-cut")?;
    let mut lines = route_set(&dir, &[3, 2], &[2, 5], GLM_LAST);
    set_ne(&mut lines, "prefill.ffn_moe_topk-5", 0, [N_USED, 2, 1, 1]);
    set_ne(
        &mut lines,
        "prefill.ffn_moe_weights-5",
        0,
        [1, N_USED, 2, 1],
    );
    set_ne(
        &mut lines,
        &format!("prefill.{GLM_LAST}-5"),
        0,
        [1, N_USED, 2, 1],
    );
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        false,
        &["prefill.ffn_moe_topk-5/0 counts 2 positions, layer 2 counts 3"],
    );
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// Every row picks the same number of experts a position.
#[test]
fn n_used_must_not_change_between_rows() -> Result<(), RefError> {
    let dir = set_dir("routes-nused")?;
    let mut lines = route_set(&dir, &[3, 2], &[2, 5], GLM_LAST);
    set_ne(&mut lines, "prefill.ffn_moe_topk-5", 0, [3, 3, 1, 1]);
    set_ne(&mut lines, "prefill.ffn_moe_weights-5", 0, [1, 3, 3, 1]);
    set_ne(
        &mut lines,
        &format!("prefill.{GLM_LAST}-5"),
        0,
        [1, 3, 3, 1],
    );
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        false,
        &["prefill.ffn_moe_topk-5/0 picks 3 experts a position, an earlier row 2"],
    );
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// A picks row is `[n_used, positions]` and a weights row of the same
/// `n_used` is `[n_used, positions]` or `[1, n_used, positions]`; any other
/// ne is refused naming the row and its ne.
#[test]
fn a_row_of_another_shape_is_refused() -> Result<(), RefError> {
    let dir = set_dir("routes-shape")?;
    let whole = route_set(&dir, &[3, 2], &[2], GLM_LAST);
    let mut lines = whole.clone();
    set_ne(&mut lines, "prefill.ffn_moe_topk-2", 0, [N_USED, 3, 2, 1]);
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        false,
        &["prefill.ffn_moe_topk-2/0 has ne [2, 3, 2, 1], want [n_used, positions, 1, 1]"],
    );

    let mut lines = whole.clone();
    set_ne(&mut lines, "prefill.ffn_moe_topk-2", 1, [N_USED, 0, 1, 1]);
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        false,
        &["prefill.ffn_moe_topk-2/1 has ne [2, 0, 1, 1], want [n_used, positions, 1, 1]"],
    );

    let mut lines = whole;
    set_ne(&mut lines, "prefill.ffn_moe_weights-2", 1, [4, 2, 1, 1]);
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        false,
        &[
            "prefill.ffn_moe_weights-2/1 has ne [4, 2, 1, 1], want [2, positions, 1, 1] or [1, 2, positions, 1]",
        ],
    );
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// The ids of the picks twin are expert numbers: a negative one is refused
/// naming the row and the prefill position.
#[test]
fn a_pick_that_is_no_expert_number_is_refused() -> Result<(), RefError> {
    let dir = set_dir("routes-negative")?;
    let mut lines = route_set(&dir, &[3, 2], &[2], GLM_LAST);
    let name = "prefill.ffn_moe_topk-2";
    let mut ids: Vec<i32> = (0..3 * N_USED).map(|i| i as i32).collect();
    ids[2 * N_USED + 1] = -1;
    *line_at(
        &mut lines,
        &format!("int\t{name}\t0\ttensor\ti32\ti32\tlogical\t"),
    ) = int_row(&dir, name, 0, true, &ids);
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        false,
        &[
            &format!("{name}/0 position 2 picks expert -1, not a u32"),
            &dir.display().to_string(),
        ],
    );
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// A position's picks are distinct: an expert picked twice is refused
/// naming the row and the prefill position (here the second ubatch's first,
/// position 3).
#[test]
fn a_pick_repeated_within_a_position_is_refused() -> Result<(), RefError> {
    let dir = set_dir("routes-twice")?;
    let mut lines = route_set(&dir, &[3, 2], &[2], GLM_LAST);
    let name = "prefill.ffn_moe_topk-2";
    *line_at(
        &mut lines,
        &format!("int\t{name}\t1\ttensor\ti32\ti32\tlogical\t"),
    ) = int_row(&dir, name, 1, true, &[4, 5, 6, 6]);
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        false,
        &[&format!("{name}/1 position 4 picks expert 6 twice")],
    );
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// The picks twin holds one id per cell of its row.
#[test]
fn the_picks_twin_must_hold_a_pick_for_every_cell_of_its_row() -> Result<(), RefError> {
    let dir = set_dir("routes-twinlen")?;
    let mut lines = route_set(&dir, &[3, 2], &[2], GLM_LAST);
    let name = "prefill.ffn_moe_topk-2";
    *line_at(
        &mut lines,
        &format!("int\t{name}\t0\ttensor\ti32\ti32\tlogical\t"),
    ) = int_row(&dir, name, 0, true, &[0, 1, 2, 3]);
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        false,
        &[&format!("{name}/0 has 6 picks, its integer twin 4")],
    );
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// A reader's own refusal names the set and the row: a picks row with no
/// logical integer twin is `Missing`, a weights file with a NaN or an
/// infinity is `Malformed`, in either weights stem.
#[test]
fn a_reader_refusal_is_named_by_set_and_row() -> Result<(), RefError> {
    let dir = set_dir("routes-reader")?;
    let whole = route_set(&dir, &[3, 2], &[2, 5], GLM_LAST);
    let mut lines = whole.clone();
    lines.retain(|l| {
        !l.starts_with("int\tprefill.ffn_moe_topk-2\t0\t") || !l.contains("\tlogical\t")
    });
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        true,
        &[
            "prefill.ffn_moe_topk-2/0",
            "no logical integer twin",
            &dir.display().to_string(),
        ],
    );

    for (stem, bad) in [("ffn_moe_weights", f32::NAN), (GLM_LAST, f32::INFINITY)] {
        let whole = route_set(&dir, &[3, 2], &[2, 5], GLM_LAST);
        let name = format!("prefill.{stem}-5");
        let file = dump_file_name(&name, 1, RowKind::Tensor, Layout::Flat, FileElem::F32);
        let mut vals = vec![1.0; 2 * N_USED];
        vals[3] = bad;
        f32_file(&dir.join(&file), &vals);
        let e = refusal(&dir, &whole)?;
        assert_refusal(
            &e,
            false,
            &[
                &format!("{name}/1"),
                "non-finite value at index 3",
                &file,
                &dir.display().to_string(),
            ],
        );
    }
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// The picks are below the model's expert count when the caller says it:
/// the first pick at or past it is refused naming its layer and position.
#[test]
fn check_experts_names_the_first_pick_past_the_models_count() -> Result<(), RefError> {
    let dir = set_dir("routes-experts")?;
    let lines = route_set(&dir, &[3, 2], &[2, 5], GLM_LAST);
    let routes = routes_of(&dir, &lines, GLM_LAST)?;
    routes.check_experts(7)?;
    let e = routes.check_experts(6).expect_err("picks up to 6");
    assert_refusal(
        &e,
        false,
        &[
            "layer 2 position 3 picks expert 6, the model has 6",
            &dir.display().to_string(),
        ],
    );
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}

/// A row the dumper does not write as a routing node is no route row: the
/// whole name is a stem, a dash and digits. The step's own `ffn_moe_topk-2`,
/// a name with a suffix and a name with no layer change nothing; a layer too
/// large for a `u32` is refused naming the row.
#[test]
fn only_the_whole_name_of_a_routing_node_is_a_route_row() -> Result<(), RefError> {
    let dir = set_dir("routes-decoys")?;
    let mut lines = route_set(&dir, &[3, 2], &[2], GLM_LAST);
    let clean = routes_of(&dir, &lines, GLM_LAST)?;
    for name in [
        "ffn_moe_topk-2",
        "prefill.ffn_moe_weights-2 (view)",
        "prefill.ffn_moe_weights-",
        "prefill.ffn_moe_weights-x",
    ] {
        lines.push(format!(
            "tensor\t{name}\t0\tf32\t1\t1\t1\t1\t4\t0.000000\tNONE\t1\t0\t-\t-"
        ));
    }
    assert_eq!(routes_of(&dir, &lines, GLM_LAST)?, clean);

    lines.push("tensor\tprefill.ffn_moe_weights-99999999999\t0\tf32\t1\t1\t1\t1\t4\t0.000000\tNONE\t1\t0\t-\t-".to_string());
    let e = refusal(&dir, &lines)?;
    assert_refusal(
        &e,
        false,
        &["prefill.ffn_moe_weights-99999999999: layer 99999999999"],
    );
    std::fs::remove_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(())
}
