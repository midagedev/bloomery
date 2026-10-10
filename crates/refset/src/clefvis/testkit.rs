//! Builders of scratch sets for the clefvis readers' tests: each kind of set in a scratch directory, of any seat's
//! [`Profile`], and the header lines the identity checks read.

use super::{EMBD, INP_RAW, Profile, TapScope, tap_names};
use crate::ik::{FileElem, Layout, RowKind, dump_file_name};
use std::path::{Path, PathBuf};

/// A fresh directory for one test.
pub(crate) fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("bloomery-clefvis-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    dir
}

pub(crate) fn write(path: &Path, bytes: &[u8]) {
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// The header fields the identity checks read.
pub(crate) struct Head<'a> {
    pub kind: &'a str,
    pub model: &'a str,
    pub build: &'a str,
    pub arch: &'a str,
    pub mmproj: &'a str,
    pub mmproj_sha: &'a str,
    pub complete: bool,
}

impl Head<'_> {
    pub(crate) fn lines(&self) -> Vec<String> {
        vec![
            "# dump_mtmd test".to_string(),
            format!("# model\t{}", self.model),
            format!("# build\t{}", self.build),
            format!("# arch\t{}", self.arch),
            format!("# clefvis\t{}", self.kind),
            format!("# mmproj\t{}\tsha256\t{}", self.mmproj, self.mmproj_sha),
            "# device\tcuda\tthreads\t4\tcard\tNVIDIA RTX A6000\tggml\tCUDA0 NVIDIA RTX A6000"
                .to_string(),
        ]
    }
}

pub(crate) const IMAGE_COLUMNS: &str =
    "# image columns\tname png_sha256 rgb8_sha256 w h best_w best_h nx ny n_tokens n_pos";
/// 96x64 planned 128x96: a 4x3 grid of 12 tokens.
pub(crate) const IMAGE: &str = "# image\tg\taa\tbb\t96\t64\t128\t96\t4\t3\t12\t4";
/// 64x32: a 2x1 grid of 2 tokens, the image of the prompt sets.
pub(crate) const IMAGE_2X1: &str = "# image\tg\taa\tbb\t64\t32\t64\t32\t2\t1\t2\t2";

/// The tensor row and f32 file of shape `ne` filled with `v`.
pub(crate) fn put(dir: &Path, rows: &mut Vec<String>, name: &str, ne: [u64; 4], v: f32) {
    let n: u64 = ne.iter().product();
    let bytes: Vec<u8> = (0..n).flat_map(|_| v.to_le_bytes()).collect();
    let file = dump_file_name(name, 0, RowKind::Tensor, Layout::Flat, FileElem::F32);
    write(&dir.join(file), &bytes);
    rows.push(format!(
        "tensor\t{name}\t0\tf32\t{}\t{}\t{}\t{}\t{}\t{:.6}\tADD\t1\t0\t-\t-",
        ne[0],
        ne[1],
        ne[2],
        ne[3],
        4 * n,
        f64::from(v) * n as f64
    ));
}

/// The i32 row `name` `[v.len(), 1, 1, 1]` with its f32 file and its lossless i32 twin.
pub(crate) fn put_ints(dir: &Path, rows: &mut Vec<String>, name: &str, v: &[i32]) {
    let n = v.len();
    let f32s: Vec<u8> = v.iter().flat_map(|&p| (p as f32).to_le_bytes()).collect();
    let i32s: Vec<u8> = v.iter().flat_map(|p| p.to_le_bytes()).collect();
    let stem = |e| dump_file_name(name, 0, RowKind::Tensor, Layout::Flat, e);
    write(&dir.join(stem(FileElem::F32)), &f32s);
    write(&dir.join(stem(FileElem::I32)), &i32s);
    let sum: i64 = v.iter().map(|&p| i64::from(p)).sum();
    let absmax = v.iter().map(|p| p.unsigned_abs()).max().unwrap_or(0);
    rows.push(format!(
        "tensor\t{name}\t0\ti32\t{n}\t1\t1\t1\t{}\t{sum}.000000\tINPUT\t1\t0\t-\t-",
        4 * n
    ));
    rows.push(format!(
        "int\t{name}\t0\ttensor\ti32\ti32\tflat\t{n}\t{}\t{sum}\t{absmax}\t{}",
        4 * n,
        stem(FileElem::I32)
    ));
}

pub(crate) fn finish(dir: &Path, head: &Head, extra: &[String], rows: &[String]) {
    let mut l = head.lines();
    l.extend(extra.iter().cloned());
    l.push("# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1".to_string());
    l.push(
        "# int\tname\toccurrence\tof\ttype\ttwin\tlayout\tcount\tbytes\tsum\tabsmax\tfile"
            .to_string(),
    );
    l.extend(rows.iter().cloned());
    if head.complete {
        let written = rows.iter().filter(|r| r.starts_with("tensor\t")).count();
        l.push(format!("# complete\t{written}\t0"));
    }
    write(&dir.join("MANIFEST.tsv"), (l.join("\n") + "\n").as_bytes());
}

/// A preproc set of one image (128x96, 4x3 tokens).
pub(crate) fn preproc(dir: &Path, head: &Head, image: &str) {
    let mut rows = Vec::new();
    put(
        dir,
        &mut rows,
        &format!("g/{INP_RAW}"),
        [128, 96, 3, 1],
        0.5,
    );
    finish(
        dir,
        head,
        &[IMAGE_COLUMNS.to_string(), image.to_string()],
        &rows,
    );
}

/// A taps set of the one image, tapped to `scope`, its embeddings as wide as the seat's text model.
pub(crate) fn taps(dir: &Path, head: &Head, scope: TapScope, profile: &Profile) {
    let n_embd = profile.n_embd;
    let mut rows = Vec::new();
    let nodes = match scope {
        TapScope::Full => tap_names(),
        TapScope::Final => vec![super::POST_LN.to_string()],
        TapScope::EmbdOnly => Vec::new(),
    };
    for n in nodes {
        let ne = if n.starts_with("Qcur_rope") {
            [72, 16, 48, 1]
        } else {
            [1152, 48, 1, 1]
        };
        put(dir, &mut rows, &format!("g/{n}"), ne, 1.0);
    }
    put(
        dir,
        &mut rows,
        &format!("g/{EMBD}"),
        [n_embd as u64, 12, 1, 1],
        2.0,
    );
    let how = match scope {
        TapScope::Full => "full",
        TapScope::Final => "final",
        TapScope::EmbdOnly => "inp_raw only",
    };
    let effect = format!(
        "# tap_effect\tg\tembd values {}\tdiffering 0\tmax_abs_diff 0\ttaps {how}",
        n_embd * 12
    );
    let extra = [IMAGE_COLUMNS.to_string(), IMAGE.to_string(), effect];
    finish(dir, head, &extra, &rows);
}

/// The positions of the prompt below as `[3, 6]`: two text rows, a 2x1 image at position 2,
/// two text rows after the image's `max(nx, ny)` = 2 positions.
pub(crate) const POS: [i32; 18] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 4, 4, 5, 5, 5];

/// The header lines of a prompt of 6 ids: text, text, a span of 2 image rows, text, text.
fn prompt_lines(pad_id: u32, with_image: bool) -> Vec<String> {
    let mut extra = vec![
        "# tokens_count\t6".to_string(),
        format!("# image_pad_id\t{pad_id}"),
        "# span columns\tindex image at len nx ny n_pos start_pos end_pos".to_string(),
        "# span\t0\tg\t2\t2\t2\t1\t2\t2\t4".to_string(),
    ];
    if with_image {
        extra.push(IMAGE_COLUMNS.to_string());
        extra.push(IMAGE_2X1.to_string());
    }
    extra
}

/// A prompt set of 6 ids: text, text, a span of 2 image rows, text, text.
pub(crate) fn prompt(dir: &Path, head: &Head, with_image: bool, pad_id: u32, profile: &Profile) {
    let mut rows = Vec::new();
    put(
        dir,
        &mut rows,
        "result_norm",
        [profile.n_embd as u64, 6, 1, 1],
        1.5,
    );
    let f32s: Vec<u8> = POS.iter().flat_map(|&p| (p as f32).to_le_bytes()).collect();
    let i32s: Vec<u8> = POS.iter().flat_map(|p| p.to_le_bytes()).collect();
    let stem = |e| dump_file_name("mrope_pos", 0, RowKind::Tensor, Layout::Flat, e);
    write(&dir.join(stem(FileElem::F32)), &f32s);
    write(&dir.join(stem(FileElem::I32)), &i32s);
    let sum: i32 = POS.iter().sum();
    rows.push(format!(
        "tensor\tmrope_pos\t0\ti32\t3\t6\t1\t1\t72\t{sum}.000000\tINPUT\t1\t0\t-\t-"
    ));
    rows.push(format!(
        "int\tmrope_pos\t0\ttensor\ti32\ti32\tflat\t18\t72\t{sum}\t5\t{}",
        stem(FileElem::I32)
    ));
    finish(dir, head, &prompt_lines(pad_id, with_image), &rows);
}

/// The ids of the 4x3-image prompt of the chat and decode sets: text, text, twelve image-pad ids, text, text. The
/// image holds 12 tokens and advances the decoder 4 positions, so the prompt's last id sits at position 8.
pub(crate) fn chat_ids_of(pad_id: u32) -> Vec<i32> {
    let mut ids = vec![11, 12];
    ids.extend(std::iter::repeat_n(pad_id as i32, 12));
    ids.extend([13, 14]);
    ids
}

/// The header lines of that prompt.
fn big_prompt_lines(pad_id: u32) -> Vec<String> {
    vec![
        "# tokens_count\t16".to_string(),
        format!("# image_pad_id\t{pad_id}"),
        "# span columns\tindex image at len nx ny n_pos start_pos end_pos".to_string(),
        "# span\t0\tg\t2\t12\t4\t3\t4\t2\t6".to_string(),
        IMAGE_COLUMNS.to_string(),
        IMAGE.to_string(),
    ]
}

/// A chat-ids set of [`chat_ids_of`], its ids as `ids`; `chat` and `prompt` are its `# chat` and `# prompt` lines.
pub(crate) fn chat_ids(
    dir: &Path,
    head: &Head,
    ids: &[i32],
    chat: &str,
    prompt: &str,
    pad_id: u32,
) {
    let mut rows = Vec::new();
    put_ints(dir, &mut rows, "ids", ids);
    let mut extra = big_prompt_lines(pad_id);
    extra[0] = format!("# tokens_count\t{}", ids.len());
    extra.push(chat.to_string());
    extra.push(prompt.to_string());
    extra.push("# prompt_sha256\t00".to_string());
    finish(dir, head, &extra, &rows);
}

/// The `# chat` line of a request that opens the next turn, with thinking.
pub(crate) const CHAT: &str = "# chat\ttemplate_sha256 aa\tuse_jinja 1\tenable_thinking 1\treasoning_format deepseek\tadd_generation_prompt 1\tcontinue_final_message none";

/// The position the first decoded token sits at after [`chat_ids_of`]: 16 ids, the image 8 positions short.
pub(crate) const DECODE_START: i64 = 8;

/// A decode set after that prompt: one step a value of `n_past`, the first token at `start`.
pub(crate) fn decode(
    dir: &Path,
    head: &Head,
    start: i64,
    n_past: &[i32],
    profile: &Profile,
    pad_id: u32,
) {
    let steps = n_past.len();
    let mut rows = Vec::new();
    put_ints(dir, &mut rows, "ids", &vec![7; steps]);
    put(
        dir,
        &mut rows,
        "result_norm",
        [profile.n_embd as u64, steps as u64, 1, 1],
        1.5,
    );
    put_ints(dir, &mut rows, "n_past", n_past);
    let mut extra = big_prompt_lines(pad_id);
    extra.push(format!(
        "# decode\tsteps\t{steps}\tn_past_start\t{start}\tsampler\tgreedy"
    ));
    finish(dir, head, &extra, &rows);
}
