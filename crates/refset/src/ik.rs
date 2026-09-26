//! ik's node dumps (`tools/ref/dump_ref.cpp`): a set's `MANIFEST.tsv`, its
//! header lines and rows, and the files the rows name.
//!
//! A row's fields are read by the names its kind's column line gives them —
//! `# kind\t…` for `tensor` rows, `# input\t…`, `# int\t…` — so a set written
//! before the v2 columns (`contig logical src0 src1`) parses to `None` there,
//! and a column a later writer adds is ignored. `skip` and `skip-input` rows
//! have no column line; they are counted, never read.

use crate::RefError;
use crate::columns::Columns;
use crate::family::Family;
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

/// Which kind of manifest row a file belongs to: a graph node (`tensor`) or
/// a graph leaf the host fills before the graph runs (`input` — token ids,
/// positions, masks, engram row ids). The two are separate occurrence
/// namespaces, and an input's files carry `.input` in their names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RowKind {
    Tensor,
    Input,
}

impl RowKind {
    /// The manifest's spelling: `tensor` or `input`.
    pub fn as_str(self) -> &'static str {
        match self {
            RowKind::Tensor => "tensor",
            RowKind::Input => "input",
        }
    }
}

/// The element order of a set file: `Flat` is the tensor's memory read
/// contiguously from its data pointer (the plain file), `Logical` its
/// elements in ggml index order through its own strides (the twin every
/// view or non-contiguous row also gets).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    Flat,
    Logical,
}

impl Layout {
    /// The manifest's spelling: `flat` or `logical`.
    pub fn as_str(self) -> &'static str {
        match self {
            Layout::Flat => "flat",
            Layout::Logical => "logical",
        }
    }
}

/// What a set file holds per element: the f32 conversion every row has, or
/// an integer twin's lossless width (i8, i16 and i32 widen to `I32`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileElem {
    F32,
    I32,
    I64,
}

impl FileElem {
    /// The file-name extension, also the manifest's spelling of a twin.
    pub fn ext(self) -> &'static str {
        match self {
            FileElem::F32 => "f32",
            FileElem::I32 => "i32",
            FileElem::I64 => "i64",
        }
    }

    /// Bytes per element.
    pub fn width(self) -> u64 {
        match self {
            FileElem::F32 | FileElem::I32 => 4,
            FileElem::I64 => 8,
        }
    }
}

/// A manifest name as the stem of its files: every `/`, `\` and space
/// becomes `_`, which keeps the name one path element. This is `safe_name`
/// in `tools/ref/dump_ref.cpp` and `dump_draft.cpp`. The manifest keeps the
/// raw name, so a file name formed from a row goes through
/// [`dump_file_name`], which calls this.
pub fn dump_stem(name: &str) -> String {
    name.replace(['/', '\\', ' '], "_")
}

/// The name the dumper gives a set file:
/// `<stem>.<occurrence>[.input][.logical].<ext>` — `.input` on a graph
/// input's files, `.logical` on a logical twin, `ext` from `elem` (`f32`, or
/// an integer twin's `i32`/`i64`), and the stem from [`dump_stem`]. Every set
/// file name this crate reads is formed here.
pub fn dump_file_name(
    name: &str,
    occurrence: u32,
    of: RowKind,
    layout: Layout,
    elem: FileElem,
) -> String {
    let input = match of {
        RowKind::Tensor => "",
        RowKind::Input => ".input",
    };
    let logical = match layout {
        Layout::Flat => "",
        Layout::Logical => ".logical",
    };
    format!(
        "{}.{occurrence}{input}{logical}.{}",
        dump_stem(name),
        elem.ext()
    )
}

/// One `tensor` or `input` row of a dump set's MANIFEST.tsv. `ne` is
/// ne0..ne3 as written: ne0 is the contiguous dimension (values of an
/// activation, rows of a MUL_MAT output); `sum` is the dumper's element
/// sum, usable to prove a VIEW row carries the same bytes as its base.
/// The four trailing fields are the v2 columns (`contig`, `logical`,
/// `src0`, `src1`); `None` on a set written before they existed.
#[derive(Debug, Clone)]
pub struct RefRow {
    pub kind: RowKind,
    pub name: String,
    pub occurrence: u32,
    pub ty: String,
    pub ne: [u64; 4],
    pub bytes: u64,
    pub sum: f64,
    pub op: String,
    pub contig: Option<u8>,
    pub logical: Option<u8>,
    pub src0: Option<String>,
    pub src1: Option<String>,
}

impl RefRow {
    /// Product of the four ne counts (elements, = bytes/4 for an f32 row).
    pub fn count(&self) -> u64 {
        self.ne.iter().product()
    }

    /// The plain file's name ([`dump_file_name`]): the flat f32 read.
    pub fn file_name(&self) -> String {
        dump_file_name(
            &self.name,
            self.occurrence,
            self.kind,
            Layout::Flat,
            FileElem::F32,
        )
    }

    /// The logical twin's file name, a file that exists when `logical` is
    /// `Some(1)`.
    pub fn logical_file_name(&self) -> String {
        dump_file_name(
            &self.name,
            self.occurrence,
            self.kind,
            Layout::Logical,
            FileElem::F32,
        )
    }

    /// Prove this row's type, dims and op — the chain check every consumer
    /// runs before trusting a tensor (`op` = what produced it). `op = "in"`
    /// only checks type and dims (an input's op varies). `what` names the
    /// call site in the error.
    pub fn expect(&self, what: &str, ty: &str, ne: [u64; 4], op: &str) -> Result<(), RefError> {
        if self.ty != ty || self.ne != ne || (op != "in" && self.op != op) {
            return Err(RefError::malformed(
                format!("expect: {what}"),
                format!(
                    "{}/{} is {} {:?} op {}, want {ty} {ne:?} op {op}",
                    self.name, self.occurrence, self.ty, self.ne, self.op
                ),
            ));
        }
        Ok(())
    }
}

/// One `int` row: the lossless twin of an integer tensor's f32 file of the
/// same layout, raw little-endian in the same element order. `sum` is the
/// exact element sum (wrapping 64-bit, printed signed), `absmax` the largest
/// magnitude, `file` the name the dumper wrote.
#[derive(Debug, Clone)]
pub struct IntRow {
    pub name: String,
    pub occurrence: u32,
    /// The kind of the row this file twins.
    pub of: RowKind,
    /// The twinned tensor's ggml type (`i8`, `i16`, `i32` or `i64`).
    pub ty: String,
    /// `I32` (i8, i16 and i32 widened) or `I64`; the parser admits no `F32`.
    pub twin: FileElem,
    pub layout: Layout,
    pub count: u64,
    pub bytes: u64,
    pub sum: i64,
    pub absmax: u64,
    pub file: String,
}

/// `int <of> <name>/<occurrence> <layout> (<file>)` — how errors name a row.
impl fmt::Display for IntRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "int {} {}/{} {} ({})",
            self.of.as_str(),
            self.name,
            self.occurrence,
            self.layout.as_str(),
            self.file
        )
    }
}

/// A dump set's whole MANIFEST.tsv: its header lines and every row the
/// dumper writes, by kind, with an index over the rows. A gate reads it once
/// per set and takes rows out of it.
#[derive(Debug, Clone)]
pub struct RefManifest {
    /// The set's directory; every file a row names resolves under it.
    pub dir: PathBuf,
    /// `# arch`, the model's `general.architecture`. `None` on a set dumped
    /// before the dumper wrote the line.
    pub arch: Option<String>,
    /// `# build`, the ik build that wrote the set.
    pub build: Option<String>,
    /// `# complete <written> <skipped>`: the node rows written and skipped.
    /// `None` when the trailer is missing — the dump that wrote the set died.
    pub complete: Option<(u64, u64)>,
    /// The other header lines.
    pub header: RefHeader,
    /// `tensor` rows (graph nodes), in file order.
    pub tensors: Vec<RefRow>,
    /// `input` rows (graph leaves), in file order.
    pub inputs: Vec<RefRow>,
    /// `int` rows, one per integer twin file, in file order.
    pub ints: Vec<IntRow>,
    /// `skip` rows: nodes written to no file (quantized, unhandled type,
    /// unallocated).
    pub skipped_nodes: u64,
    /// `skip-input` rows: the same for graph inputs.
    pub skipped_inputs: u64,
    index: RowIndex,
}

/// The header lines of a set that [`RefManifest`] does not hold as its own
/// fields (`# arch`, `# build`, the `# complete` trailer). A line the set
/// does not carry leaves its field `None`.
#[derive(Debug, Clone, Default)]
pub struct RefHeader {
    /// `# tokens`: the ids of the whole sequence the set was dumped for.
    pub tokens: Option<Vec<u32>>,
    /// `# tokens_file`: the file the dumper read the tokens from.
    pub tokens_file: Option<String>,
    /// `# tokens_file_sha256`: that file's digest.
    pub tokens_file_sha256: Option<String>,
    /// `# tokens_count`: how many of the file's tokens the dumper took.
    pub tokens_count: Option<u64>,
    /// `# model_file`: the name of the model file the set was dumped from.
    pub model_file: Option<String>,
    /// `# flags`: the dumper's command line.
    pub flags: Option<String>,
    /// `-c` in `# flags`: the context the dumper ran at.
    pub ctx: Option<u64>,
    /// `-t` in `# flags`: the threads the dumper ran with.
    pub threads: Option<u64>,
    /// `# prefill`: in a decode-step set, the tokens run before the step.
    pub prefill: Option<u32>,
    /// `# decode_pos`: in a decode-step set, the step's position.
    pub decode_pos: Option<u32>,
    /// `# state_inputs`: in a decode-step set, the dumper's note that the
    /// persistent leaves the step reads (caches, compressor states) are
    /// input rows, written at their first reader.
    pub state_inputs: Option<String>,
    /// `# fused_idx_topk`: whether the indexer's scores and top-k ran as one
    /// op (`1`) or as nodes of their own (`0`).
    pub fused_idx_topk: Option<bool>,
    /// Every other `#` line, verbatim and in file order: the dumper's title,
    /// `# model`, the column lines, and any line this parser does not know.
    pub other: Vec<String>,
}

impl RefHeader {
    /// `# model`: the path of the model file the set was dumped from, one of
    /// the lines kept verbatim in [`RefHeader::other`].
    #[must_use]
    pub fn model(&self) -> Option<&str> {
        self.other.iter().find_map(|l| l.strip_prefix("# model\t"))
    }
}

/// Where each row sits in its vector, keyed the way readers look rows up:
/// by name, then `(kind, occurrence)`, and for an `int` row also its layout.
/// Built once by [`RefManifest::read`] over the rows it parsed; of a
/// repeated key it keeps the first row, the one a scan in file order finds.
/// A lookup checks that the row it lands on carries the key, so an answer
/// always comes from the rows themselves.
#[derive(Debug, Clone, Default)]
struct RowIndex {
    rows: HashMap<String, Vec<(RowKind, u32, usize)>>,
    ints: HashMap<String, Vec<(RowKind, u32, Layout, usize)>>,
    /// Tensor and input rows whose key an earlier row already carries.
    duplicates: usize,
    /// Per input row, the position in `tensors` of the first tensor row the
    /// file holds after it; `None` when no tensor row follows.
    touchers: Vec<Option<usize>>,
}

impl RowIndex {
    fn over(
        tensors: &[RefRow],
        inputs: &[RefRow],
        ints: &[IntRow],
        touchers: Vec<Option<usize>>,
    ) -> RowIndex {
        let mut ix = RowIndex {
            touchers,
            ..RowIndex::default()
        };
        for (at, r) in tensors.iter().enumerate().chain(inputs.iter().enumerate()) {
            let keys = ix.rows.entry(r.name.clone()).or_default();
            if keys
                .iter()
                .any(|&(k, o, _)| k == r.kind && o == r.occurrence)
            {
                ix.duplicates += 1;
            } else {
                keys.push((r.kind, r.occurrence, at));
            }
        }
        for (at, r) in ints.iter().enumerate() {
            let keys = ix.ints.entry(r.name.clone()).or_default();
            if !keys
                .iter()
                .any(|&(of, o, l, _)| of == r.of && o == r.occurrence && l == r.layout)
            {
                keys.push((r.of, r.occurrence, r.layout, at));
            }
        }
        ix
    }
}

/// The column lines of a set, one per row kind that has one.
#[derive(Default)]
struct RowColumns {
    tensor: Option<Columns>,
    input: Option<Columns>,
    int: Option<Columns>,
}

impl RowColumns {
    /// Record `line` if it is a column line (`# kind\t…`, `# input\t…`,
    /// `# int\t…`); a kind's second column line is an error.
    fn take(&mut self, line: &str, at: At) -> Result<(), RefError> {
        let Some((key, names)) = line.strip_prefix("# ").and_then(|h| h.split_once('\t')) else {
            return Ok(());
        };
        let slot = match key {
            "kind" => &mut self.tensor,
            "input" => &mut self.input,
            "int" => &mut self.int,
            _ => return Ok(()),
        };
        if slot.is_some() {
            return Err(RefError::malformed(
                at,
                format!("a second # {key} column line"),
            ));
        }
        *slot = Some(Columns::new(key, names.split('\t')));
        Ok(())
    }

    /// The columns of row kind `kind`, the word its rows open with; a row
    /// before its kind's column line is an error.
    fn of(&self, kind: &str, at: At) -> Result<&Columns, RefError> {
        let (cols, line) = match kind {
            "tensor" => (&self.tensor, "# kind"),
            "input" => (&self.input, "# input"),
            _ => (&self.int, "# int"),
        };
        cols.as_ref().ok_or_else(|| {
            RefError::malformed(at, format!("a {kind} row before any {line} column line"))
        })
    }
}

impl RefManifest {
    /// Parse `dir/MANIFEST.tsv`. Header lines start with `#` ([`RefHeader`]
    /// and this struct's own `arch`, `build`, `complete`); data rows are
    /// tab-separated, the kind first, and read by the names of their kind's
    /// column line: `tensor` rows by `# kind` (`name occurrence type ne0 ne1
    /// ne2 ne3 bytes sum op`, and on a v2 set `contig logical src0 src1`),
    /// `input` rows by `# input`, `int` rows by `# int` (`name occurrence of
    /// type twin layout count bytes sum absmax file`); `skip`/`skip-input`
    /// rows are `name occurrence type reason`. Tensor names may contain
    /// spaces, so fields are split on tabs only. A row of any other kind, a
    /// row of another width than its column line or before it, or a field
    /// that does not parse is `Malformed` naming its line, and so is a
    /// manifest with no `tensor` row. The rows are indexed here, once, with
    /// each input row's first toucher ([`first_touched_by`](Self::first_touched_by)).
    pub fn read(dir: &Path) -> Result<RefManifest, RefError> {
        let path = dir.join("MANIFEST.tsv");
        let text = std::fs::read_to_string(&path).map_err(|e| {
            RefError::missing(
                &path,
                format!("ref_manifest: cannot read {}: {e}", path.display()),
            )
        })?;
        let mut man = RefManifest {
            dir: dir.to_path_buf(),
            arch: None,
            build: None,
            complete: None,
            header: RefHeader::default(),
            tensors: Vec::new(),
            inputs: Vec::new(),
            ints: Vec::new(),
            skipped_nodes: 0,
            skipped_inputs: 0,
            index: RowIndex::default(),
        };
        let mut cols = RowColumns::default();
        // Per input row, the tensor row that follows it in the file; the
        // inputs no tensor row has followed yet.
        let (mut touchers, mut pending) = (Vec::new(), Vec::new());
        for (i, line) in text.lines().enumerate() {
            let at = At(&path, i + 1);
            if line.starts_with('#') {
                cols.take(line, at)?;
                man.header_line(line, at)?;
            } else if !line.is_empty() {
                let kind = line.split('\t').next().unwrap_or("");
                match kind {
                    "tensor" => {
                        let row = parse_ref_row(RowKind::Tensor, cols.of(kind, at)?, line, &at)?;
                        for p in pending.drain(..) {
                            touchers[p] = Some(man.tensors.len());
                        }
                        man.tensors.push(row);
                    }
                    "input" => {
                        man.inputs.push(parse_ref_row(
                            RowKind::Input,
                            cols.of(kind, at)?,
                            line,
                            &at,
                        )?);
                        pending.push(touchers.len());
                        touchers.push(None);
                    }
                    "int" => man.ints.push(parse_int_row(cols.of(kind, at)?, line, &at)?),
                    "skip" | "skip-input" if line.split('\t').count() != 5 => {
                        return Err(RefError::malformed(
                            at,
                            format!("{kind} row has {} fields, want 5", line.split('\t').count()),
                        ));
                    }
                    "skip" => man.skipped_nodes += 1,
                    "skip-input" => man.skipped_inputs += 1,
                    k => return Err(RefError::malformed(at, format!("unknown row kind {k:?}"))),
                }
            }
        }
        if man.tensors.is_empty() {
            return Err(RefError::malformed(
                format!("ref_manifest: {}", path.display()),
                "no tensor rows",
            ));
        }
        man.index = RowIndex::over(&man.tensors, &man.inputs, &man.ints, touchers);
        Ok(man)
    }

    /// Read the set at `dir` and check it against `family`'s row
    /// ([`check_family`](Self::check_family)).
    pub fn open(dir: &Path, family: &Family) -> Result<RefManifest, RefError> {
        let man = RefManifest::read(dir)?;
        man.check_family(family)?;
        Ok(man)
    }

    /// This set against its family's row: the completion trailer
    /// ([`RefError::Unfinished`]), then the model file its `# model` line
    /// states ([`RefError::Stale`]), its `# arch` and its `# build`
    /// ([`RefError::Foreign`]).
    pub fn check_family(&self, family: &Family) -> Result<(), RefError> {
        if self.complete.is_none() {
            return Err(RefError::Unfinished {
                set: self.dir.display().to_string(),
            });
        }
        family.check_file(&self.dir, "# model", self.header.model())?;
        family.check_arch(&self.dir, self.arch.as_deref())?;
        family.check_build(&self.dir, self.build.as_deref())
    }

    /// One `#` line. A `# <key>\t<value>` line of a key this parser knows
    /// fills its field — a key given twice, or a value that does not parse,
    /// is an error naming the line — and every other line is kept verbatim
    /// in `header.other`.
    fn header_line(&mut self, line: &str, at: At) -> Result<(), RefError> {
        let Some((key, v)) = line.strip_prefix("# ").and_then(|h| h.split_once('\t')) else {
            self.header.other.push(line.to_string());
            return Ok(());
        };
        let bad = |e: &dyn fmt::Display| RefError::malformed(at, format!("{line:?}: {e}"));
        let text = || v.to_string();
        let h = &mut self.header;
        match key {
            "arch" => once(&mut self.arch, text(), key, at),
            "build" => once(&mut self.build, text(), key, at),
            "complete" => {
                let (w, s) = v.split_once('\t').ok_or_else(|| bad(&"not two counts"))?;
                once(
                    &mut self.complete,
                    (parse_u64(w, at)?, parse_u64(s, at)?),
                    key,
                    at,
                )
            }
            "tokens" => {
                let ids = v
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<Vec<u32>, _>>()
                    .map_err(|e| bad(&e))?;
                once(&mut h.tokens, ids, key, at)
            }
            "tokens_file" => once(&mut h.tokens_file, text(), key, at),
            "tokens_file_sha256" => once(&mut h.tokens_file_sha256, text(), key, at),
            "tokens_count" => once(&mut h.tokens_count, parse_u64(v, at)?, key, at),
            "model_file" => once(&mut h.model_file, text(), key, at),
            "flags" => {
                h.ctx = flag_value(v, "-c").map_err(|e| bad(&e))?;
                h.threads = flag_value(v, "-t").map_err(|e| bad(&e))?;
                once(&mut h.flags, text(), key, at)
            }
            "prefill" => once(&mut h.prefill, parse_narrow(v, at)?, key, at),
            "decode_pos" => once(&mut h.decode_pos, parse_narrow(v, at)?, key, at),
            "state_inputs" => once(&mut h.state_inputs, text(), key, at),
            "fused_idx_topk" => {
                let fused = match v {
                    "0" => false,
                    "1" => true,
                    _ => return Err(bad(&"want 0 or 1")),
                };
                once(&mut h.fused_idx_topk, fused, key, at)
            }
            _ => {
                h.other.push(line.to_string());
                Ok(())
            }
        }
    }

    /// Where the row of kind `kind` keyed `name`/`occurrence` sits among the
    /// rows of its kind (`tensors` or `inputs`, file order), through the
    /// index — the row a scan of the file finds first — or `None` if the set
    /// has none.
    pub fn position(&self, kind: RowKind, name: &str, occurrence: u32) -> Option<usize> {
        let &(_, _, at) = self
            .index
            .rows
            .get(name)?
            .iter()
            .find(|&&(k, o, _)| k == kind && o == occurrence)?;
        self.rows_of(kind)
            .get(at)
            .filter(|r| r.name == name && r.occurrence == occurrence)
            .map(|_| at)
    }

    /// The row of kind `kind` keyed `name`/`occurrence`, through the index —
    /// the row a scan of the file finds first — or `None` if the set has
    /// none.
    pub fn find(&self, kind: RowKind, name: &str, occurrence: u32) -> Option<&RefRow> {
        let at = self.position(kind, name, occurrence)?;
        self.rows_of(kind).get(at)
    }

    /// The first row of kind `kind` named `name`, whatever its occurrence —
    /// the row a scan of the file finds first. The dumper counts a name's
    /// skipped nodes in its occurrences, so that row need not be occurrence 0.
    pub fn first_named(&self, kind: RowKind, name: &str) -> Option<&RefRow> {
        let at = self
            .index
            .rows
            .get(name)?
            .iter()
            .filter(|&&(k, ..)| k == kind)
            .map(|&(.., at)| at)
            .min()?;
        self.rows_of(kind).get(at).filter(|r| r.name == name)
    }

    /// The one row of kind `kind` whose name opens with `prefix`, whatever
    /// its occurrence: the row of a tensor the graph names after the last of
    /// several callers (`dsv4_raw_mask_padded-<layer>`), which a reader cannot
    /// name in advance. No such row, or more than one, is an error naming the
    /// rows found and the set.
    pub fn only_with_prefix(&self, kind: RowKind, prefix: &str) -> Result<&RefRow, RefError> {
        let found: Vec<&RefRow> = self
            .rows_of(kind)
            .iter()
            .filter(|r| r.name.starts_with(prefix))
            .collect();
        match found[..] {
            [r] => Ok(r),
            _ => Err(RefError::missing(
                &self.dir,
                format!(
                    "ref_manifest: want one {} row named {prefix}…, {} has {:?}",
                    kind.as_str(),
                    self.dir.join("MANIFEST.tsv").display(),
                    found
                        .iter()
                        .map(|r| format!("{}/{}", r.name, r.occurrence))
                        .collect::<Vec<_>>()
                ),
            )),
        }
    }

    /// The `tensor` row `name`/`occurrence` ([`find`](Self::find)); a set
    /// without it is an error naming the set.
    pub fn tensor(&self, name: &str, occurrence: u32) -> Result<&RefRow, RefError> {
        self.row(RowKind::Tensor, name, occurrence)
    }

    /// The `tensor` row `name`/`occurrence` with its position in `tensors`,
    /// where a walk back through the rows before it starts
    /// ([`last_before`](Self::last_before)); a set without it is an error
    /// naming the set.
    pub fn tensor_at(&self, name: &str, occurrence: u32) -> Result<(usize, &RefRow), RefError> {
        let at = self
            .position(RowKind::Tensor, name, occurrence)
            .ok_or_else(|| self.missing(RowKind::Tensor, name, occurrence))?;
        Ok((at, &self.tensors[at]))
    }

    /// The `input` row `name`/`occurrence` ([`find`](Self::find)); a set
    /// without it is an error naming the set.
    pub fn input(&self, name: &str, occurrence: u32) -> Result<&RefRow, RefError> {
        self.row(RowKind::Input, name, occurrence)
    }

    /// The node a reader at `tensors[at]` reads as `name`, one of its source
    /// columns: the last tensor row of that name before it — manifest order
    /// is execution order — with its position. An empty column is an error.
    pub fn last_before(&self, at: usize, name: Option<&str>) -> Result<(usize, &RefRow), RefError> {
        let name = name.ok_or_else(|| {
            RefError::missing(&self.dir, "a reader row has no src column".to_string())
        })?;
        self.tensors[..at]
            .iter()
            .enumerate()
            .rev()
            .find(|(_, r)| r.name == name)
            .ok_or_else(|| {
                RefError::missing(
                    &self.dir,
                    format!("no node {name:?} before manifest row {at}"),
                )
            })
    }

    /// The input rows tensor row `at` touches first, in file order. The
    /// dumper writes a graph input right before the node that first touches
    /// it — reads it, or writes into it as a `SET_ROWS` destination, which no
    /// source column names — so an input's first toucher is the tensor row
    /// that follows it in the file.
    pub fn first_touched_by(&self, at: usize) -> Vec<&RefRow> {
        self.inputs
            .iter()
            .zip(&self.index.touchers)
            .filter(|&(_, &t)| t == Some(at))
            .map(|(r, _)| r)
            .collect()
    }

    fn rows_of(&self, kind: RowKind) -> &[RefRow] {
        match kind {
            RowKind::Tensor => &self.tensors,
            RowKind::Input => &self.inputs,
        }
    }

    fn row(&self, kind: RowKind, name: &str, occurrence: u32) -> Result<&RefRow, RefError> {
        self.find(kind, name, occurrence)
            .ok_or_else(|| self.missing(kind, name, occurrence))
    }

    /// The error a lookup of a row the set does not hold returns.
    fn missing(&self, kind: RowKind, name: &str, occurrence: u32) -> RefError {
        RefError::missing(
            &self.dir,
            format!(
                "ref_manifest: {} {name}/{occurrence} not in {}",
                kind.as_str(),
                self.dir.join("MANIFEST.tsv").display()
            ),
        )
    }

    /// Tensor and input rows whose `(kind, name, occurrence)` an earlier row
    /// already carries; a lookup never lands on one of them.
    pub fn duplicate_keys(&self) -> usize {
        self.index.duplicates
    }

    /// The step the set holds, as its first position, its tokens and the
    /// tokens before it: in a batch set every token of `# tokens` from
    /// position 0; in a decode-step set the last token, at `# decode_pos`,
    /// after the rest. A decode step runs right after its prefill, so
    /// `# prefill`, where the set has it, is the step's position. A set with
    /// no `# tokens`, a `# decode_pos` that is not its last token, or a
    /// `# prefill` that differs from it is an error naming the set.
    pub fn step(&self) -> Result<(u32, &[u32], &[u32]), RefError> {
        let path = self.dir.join("MANIFEST.tsv");
        let h = &self.header;
        let tokens = h.tokens.as_deref().ok_or_else(|| {
            RefError::missing(&path, format!("{} has no # tokens line", path.display()))
        })?;
        if let (Some(p), Some(n)) = (h.decode_pos, h.prefill)
            && p != n
        {
            return Err(RefError::malformed(
                path.display(),
                format!("# decode_pos {p} does not follow # prefill {n}"),
            ));
        }
        match h.decode_pos {
            None => Ok((0, tokens, &[])),
            Some(p) if p as usize + 1 == tokens.len() => {
                let at = p as usize;
                Ok((p, &tokens[at..], &tokens[..at]))
            }
            Some(p) => Err(RefError::malformed(
                path.display(),
                format!(
                    "# decode_pos {p} is not the last of the {} tokens",
                    tokens.len()
                ),
            )),
        }
    }
}

/// Fill a header field once: a second line of the same key is an error.
fn once<T>(slot: &mut Option<T>, v: T, key: &str, at: At) -> Result<(), RefError> {
    if slot.replace(v).is_some() {
        return Err(RefError::malformed(at, format!("a second # {key} line")));
    }
    Ok(())
}

/// A manifest line's position, `ref_manifest: <path>:<line>`, formatted only
/// into an error.
#[derive(Clone, Copy)]
struct At<'a>(&'a Path, usize);

impl fmt::Display for At<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ref_manifest: {}:{}", self.0.display(), self.1)
    }
}

/// The number after the first `flag` in a `# flags` command line; `None` when
/// the line has no such flag.
fn flag_value(line: &str, flag: &str) -> Result<Option<u64>, String> {
    let mut args = line.split_whitespace();
    if !args.by_ref().any(|a| a == flag) {
        return Ok(None);
    }
    let v = args.next().unwrap_or("");
    v.parse()
        .map(Some)
        .map_err(|e| format!("{flag} {v:?}: {e}"))
}

fn parse_u64(s: &str, at: At) -> Result<u64, RefError> {
    s.parse::<u64>()
        .map_err(|e| RefError::malformed(at, format!("{s:?}: {e}")))
}

fn parse_narrow<T: TryFrom<u64>>(s: &str, at: At) -> Result<T, RefError> {
    T::try_from(parse_u64(s, at)?)
        .map_err(|_| RefError::malformed(at, format!("{s:?} is out of range")))
}

/// A `tensor` or `input` row, by its kind's column names. A draft set's rows
/// ([`crate::dsref`]) carry these columns and more; the rest are its own.
pub(crate) fn parse_ref_row(
    kind: RowKind,
    cols: &Columns,
    line: &str,
    at: &dyn fmt::Display,
) -> Result<RefRow, RefError> {
    let r = cols.row(line, at)?;
    Ok(RefRow {
        kind,
        name: r.text("name")?.to_string(),
        occurrence: r.parse("occurrence")?,
        ty: r.text("type")?.to_string(),
        ne: [
            r.parse("ne0")?,
            r.parse("ne1")?,
            r.parse("ne2")?,
            r.parse("ne3")?,
        ],
        bytes: r.parse("bytes")?,
        sum: r.parse("sum")?,
        op: r.text("op")?.to_string(),
        contig: r.parse_opt("contig")?,
        logical: r.parse_opt("logical")?,
        src0: r.opt("src0").map(str::to_string),
        src1: r.opt("src1").map(str::to_string),
    })
}

/// An `int` row, by the `# int` line's column names.
pub(crate) fn parse_int_row(
    cols: &Columns,
    line: &str,
    at: &dyn fmt::Display,
) -> Result<IntRow, RefError> {
    let r = cols.row(line, at)?;
    let bad = |col: &str, v: &str| RefError::malformed(at, format!("int row {col} {v:?}"));
    let of = match r.text("of")? {
        "tensor" => RowKind::Tensor,
        "input" => RowKind::Input,
        v => return Err(bad("of", v)),
    };
    let twin = match r.text("twin")? {
        "i32" => FileElem::I32,
        "i64" => FileElem::I64,
        v => return Err(bad("twin", v)),
    };
    let layout = match r.text("layout")? {
        "flat" => Layout::Flat,
        "logical" => Layout::Logical,
        v => return Err(bad("layout", v)),
    };
    Ok(IntRow {
        name: r.text("name")?.to_string(),
        occurrence: r.parse("occurrence")?,
        of,
        ty: r.text("type")?.to_string(),
        twin,
        layout,
        count: r.parse("count")?,
        bytes: r.parse("bytes")?,
        sum: r.parse("sum")?,
        absmax: r.parse("absmax")?,
        file: r.text("file")?.to_string(),
    })
}

/// The `tensor` rows of the set at `dir` ([`RefManifest::read`]).
pub fn ref_manifest_in(dir: &Path) -> Result<Vec<RefRow>, RefError> {
    Ok(RefManifest::read(dir)?.tensors)
}

/// The manifest row for `(name, occurrence)` in `man`, a scan of the slice
/// naming `dir` in the error: a gate holding a [`RefManifest`] looks rows up
/// through its index ([`RefManifest::tensor`]).
pub fn find_ref_row_in<'a>(
    dir: &Path,
    man: &'a [RefRow],
    name: &str,
    occurrence: u32,
) -> Result<&'a RefRow, RefError> {
    man.iter()
        .find(|r| r.name == name && r.occurrence == occurrence)
        .ok_or_else(|| {
            RefError::missing(
                dir,
                format!(
                    "find_ref_row: {}/{} not in {}",
                    name,
                    occurrence,
                    dir.join("MANIFEST.tsv").display()
                ),
            )
        })
}

/// Load one manifest row's f32 file of the set at `dir`, checking it against
/// its own row: the type must be f32, the row's byte count must equal
/// 4·ne0·ne1·ne2·ne3 and the file's length, and every value must be finite.
/// Every error names the offending path.
pub fn ref_tensor_of_in(dir: &Path, row: &RefRow) -> Result<Vec<f32>, RefError> {
    let path = dir.join(row.file_name());
    let bad = |what: String| RefError::malformed("ref_tensor_of", what);
    if row.ty != "f32" {
        return Err(bad(format!(
            "{} has type {}, want f32",
            path.display(),
            row.ty
        )));
    }
    let expect = 4_u64
        .checked_mul(row.count())
        .ok_or_else(|| bad("element count overflows".to_string()))?;
    if row.bytes != expect {
        return Err(bad(format!(
            "{} manifest bytes {} != 4*count {}",
            path.display(),
            row.bytes,
            expect
        )));
    }
    let raw = std::fs::read(&path).map_err(|e| {
        RefError::missing(
            &path,
            format!("ref_tensor_of: cannot read {}: {e}", path.display()),
        )
    })?;
    if raw.len() as u64 != row.bytes {
        return Err(bad(format!(
            "{} is {} bytes, manifest says {}",
            path.display(),
            raw.len(),
            row.bytes
        )));
    }
    let vals: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    if let Some(i) = vals.iter().position(|v| !v.is_finite()) {
        return Err(bad(format!(
            "non-finite value at index {i} of {}",
            path.display()
        )));
    }
    Ok(vals)
}

/// Tensor row `(name, occ)` of the held manifest `man`, through its index,
/// and the row's f32 file ([`ref_tensor_of_in`]).
pub fn load_ref_in(
    man: &RefManifest,
    name: &str,
    occ: u32,
) -> Result<(RefRow, Vec<f32>), RefError> {
    let row = man.tensor(name, occ)?;
    Ok((row.clone(), ref_tensor_of_in(&man.dir, row)?))
}

/// Tensor row `(name, occ)` of the held manifest `man`, through its index,
/// and the row's LOGICAL elements ([`ref_tensor_logical_in`]).
pub fn load_ref_logical_in(
    man: &RefManifest,
    name: &str,
    occ: u32,
) -> Result<(RefRow, Vec<f32>), RefError> {
    let row = man.tensor(name, occ)?;
    Ok((row.clone(), ref_tensor_logical_in(&man.dir, row)?))
}

/// The LOGICAL elements, in ggml index order, of row `row` of the set at
/// `dir`: the `.logical.f32` twin when the dump wrote one, else the plain
/// file when the row is provably not a flat VIEW read (a contiguous tensor's
/// plain file IS its logical order; a pre-v2 manifest without the `contig`
/// column is accepted for non-VIEW rows only). A VIEW row without a logical
/// twin is an error, never a silent flat read — that flat read is a
/// different tensor than the one being asked for.
pub fn ref_tensor_logical_in(dir: &Path, row: &RefRow) -> Result<Vec<f32>, RefError> {
    let bad = |what: String| RefError::malformed("ref_tensor_logical_in", what);
    if row.logical == Some(1) {
        if row.ty != "f32" {
            return Err(bad(format!("{} has type {}, want f32", row.name, row.ty)));
        }
        let path = dir.join(row.logical_file_name());
        let expect = 4_u64
            .checked_mul(row.count())
            .ok_or_else(|| bad("element count overflows".to_string()))?;
        let raw = std::fs::read(&path).map_err(|e| {
            RefError::missing(
                &path,
                format!("ref_tensor_logical_in: cannot read {}: {e}", path.display()),
            )
        })?;
        if raw.len() as u64 != expect {
            return Err(bad(format!(
                "{} is {} bytes, want {} (4*count)",
                path.display(),
                raw.len(),
                expect
            )));
        }
        let vals: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        if let Some(i) = vals.iter().position(|v| !v.is_finite()) {
            return Err(bad(format!(
                "non-finite value at index {i} of {}",
                path.display()
            )));
        }
        return Ok(vals);
    }
    if !plain_is_logical(row) {
        return Err(bad(format!(
            "{} is a view/non-contiguous row with no logical twin in {} — the plain file is a \
             flat read, not the tensor",
            row.name,
            dir.display()
        )));
    }
    ref_tensor_of_in(dir, row)
}

/// Whether a row's plain file is its logical order, so a whole-tensor read
/// of it is the tensor: a v2 manifest's `contig` column decides; on a pre-v2
/// manifest only the op can rule out a flat VIEW read.
pub fn plain_is_logical(row: &RefRow) -> bool {
    match (row.contig, row.op.as_str()) {
        (Some(c), _) => c == 1,
        (None, "VIEW") => false,
        (None, _) => true,
    }
}

/// The `int` row twinning `(name, occurrence)` of kind `of` in `layout`,
/// through the manifest's index.
pub fn find_int_row<'a>(
    man: &'a RefManifest,
    name: &str,
    occurrence: u32,
    of: RowKind,
    layout: Layout,
) -> Result<&'a IntRow, RefError> {
    man.index
        .ints
        .get(name)
        .and_then(|keys| {
            keys.iter()
                .find(|&&(k, o, l, _)| k == of && o == occurrence && l == layout)
        })
        .and_then(|&(.., at)| man.ints.get(at))
        .filter(|r| {
            r.name == name && r.occurrence == occurrence && r.of == of && r.layout == layout
        })
        .ok_or_else(|| {
            RefError::missing(
                &man.dir,
                format!(
                    "find_int_row: no {} integer twin of {} {name}/{occurrence} in {}",
                    layout.as_str(),
                    of.as_str(),
                    man.dir.join("MANIFEST.tsv").display()
                ),
            )
        })
}

/// The exact integers of int row `row` of the set at `dir`, widened to
/// i64, read from its `file` after proving that file is the one the row
/// describes: `file` is the name [`dump_file_name`] gives the row (the
/// dumper and this crate share one naming rule), `bytes` is `count`
/// elements of the twin's width and the file's length, and the `sum`
/// (wrapping) and `absmax` recomputed from the values equal the row's. A
/// wrong file, offset or width fails here, naming the row.
pub fn ref_ints_of_in(dir: &Path, row: &IntRow) -> Result<Vec<i64>, RefError> {
    let bad = |what: String| RefError::malformed(format!("ref_ints: {row}"), what);
    let want = dump_file_name(&row.name, row.occurrence, row.of, row.layout, row.twin);
    if row.file != want {
        return Err(bad(format!("the dumper names this row's file {want}")));
    }
    let expect = row
        .count
        .checked_mul(row.twin.width())
        .ok_or_else(|| bad("element count overflows".to_string()))?;
    if row.bytes != expect {
        return Err(bad(format!(
            "bytes {} != count {} x {}",
            row.bytes,
            row.count,
            row.twin.width()
        )));
    }
    let path = dir.join(&row.file);
    let raw = std::fs::read(&path).map_err(|e| {
        RefError::missing(
            &path,
            format!("ref_ints: {row}: cannot read {}: {e}", path.display()),
        )
    })?;
    if raw.len() as u64 != row.bytes {
        return Err(bad(format!(
            "{} is {} bytes, the row says {}",
            path.display(),
            raw.len(),
            row.bytes
        )));
    }
    let vals: Vec<i64> = match row.twin {
        FileElem::I32 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i64::from(i32::from_le_bytes(*c)))
            .collect(),
        FileElem::I64 => raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect(),
        FileElem::F32 => return Err(bad("not an integer twin".to_string())),
    };
    let sum = vals
        .iter()
        .fold(0u64, |a, &v| a.wrapping_add(v.cast_unsigned()))
        .cast_signed();
    let absmax = vals.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
    if sum != row.sum || absmax != row.absmax {
        return Err(bad(format!(
            "the file sums to {sum} with absmax {absmax}, the row says {} and {}",
            row.sum, row.absmax
        )));
    }
    Ok(vals)
}

/// The exact integers of `(name, occurrence)`'s twin of kind `of` in
/// `layout`: [`find_int_row`] in `man`, then [`ref_ints_of_in`].
pub fn ref_ints(
    man: &RefManifest,
    name: &str,
    occurrence: u32,
    of: RowKind,
    layout: Layout,
) -> Result<Vec<i64>, RefError> {
    ref_ints_of_in(&man.dir, find_int_row(man, name, occurrence, of, layout)?)
}

/// Rows `idx` of f16 row `row` of the set at `dir` as f16 bits, row after
/// row, `ne[0]` values a row. The dump widened each half to f32 exactly, so
/// rounding back with `gguf::quant::f32_to_f16_bits` — the CPU oracle's own
/// rounding, not a second transcription — recovers ik's bits; a value that
/// does not widen back to itself is an error naming the row. Only the rows
/// asked for are read from the file. The rows are read from the plain file,
/// so it must be the tensor: a row that is a view or not contiguous
/// ([`plain_is_logical`]) is an error.
pub fn widened_f16_bits_in(dir: &Path, row: &RefRow, idx: &[u32]) -> Result<Vec<u16>, RefError> {
    use gguf::quant::{f32_to_f16_bits, half_to_f32};
    use std::io::{Read, Seek, SeekFrom};

    let path = dir.join(row.file_name());
    let what = || {
        format!(
            "{} {}/{} ({})",
            row.kind.as_str(),
            row.name,
            row.occurrence,
            path.display()
        )
    };
    let bad = |detail: String| RefError::malformed("widened_f16_bits", detail);
    let count = row.count();
    if row.ty != "f16" || 4_u64.checked_mul(count) != Some(row.bytes) {
        return Err(bad(format!(
            "{} is {} with {} bytes for {count} values, want f16 widened to f32",
            what(),
            row.ty,
            row.bytes
        )));
    }
    if !plain_is_logical(row) {
        return Err(bad(format!(
            "{} is a view/non-contiguous row — its plain file is a flat read, not the tensor",
            what()
        )));
    }
    let io =
        |e: std::io::Error| RefError::missing(&path, format!("widened_f16_bits: {}: {e}", what()));
    let mut file = std::fs::File::open(&path).map_err(io)?;
    let len = file.metadata().map_err(io)?.len();
    if len != row.bytes {
        return Err(bad(format!(
            "{} is {len} bytes, the row says {}",
            what(),
            row.bytes
        )));
    }
    let width = row.ne[0];
    let line_len =
        usize::try_from(width).map_err(|e| bad(format!("{}: ne0 {width}: {e}", what())))?;
    let mut line = vec![0u8; 4 * line_len];
    let mut out = Vec::with_capacity(idx.len() * line_len);
    for &r in idx {
        let first = u64::from(r) * width;
        if first + width > count {
            return Err(bad(format!(
                "{}: row {r} is past its {count} values",
                what()
            )));
        }
        file.seek(SeekFrom::Start(4 * first)).map_err(io)?;
        file.read_exact(&mut line).map_err(io)?;
        for c in line.as_chunks::<4>().0 {
            let v = f32::from_le_bytes(*c);
            let h = f32_to_f16_bits(v);
            if half_to_f32(h).to_bits() != v.to_bits() {
                return Err(bad(format!(
                    "{} holds {v} in row {r}, not a widened f16",
                    what()
                )));
            }
            out.push(h);
        }
    }
    Ok(out)
}

/// Every row of f16 row `row` of the set at `dir`, as f16 bits:
/// [`widened_f16_bits_in`] over rows `0 .. count / ne[0]`, which refuses a
/// row whose plain file is not the tensor.
pub fn widened_f16_rows_in(dir: &Path, row: &RefRow) -> Result<Vec<u16>, RefError> {
    let rows = row.count().checked_div(row.ne[0]).unwrap_or(0);
    let idx: Vec<u32> = (0..u32::try_from(rows).map_err(|e| {
        RefError::malformed(
            "widened_f16_rows",
            format!("{}/{}: {rows} rows: {e}", row.name, row.occurrence),
        )
    })?)
        .collect();
    widened_f16_bits_in(dir, row, &idx)
}

/// The f16 bits of mask row `row` of the set at `dir`: an `f16` row, its
/// plain file its logical order ([`plain_is_logical`]), whose f32 file holds
/// every value exactly `0.0` (a cell the query sees) or `-inf` (one it does
/// not), each mapped to that value's f16 bits (`0x0000`, `0xfc00`). Anything
/// else — another type, a view or non-contiguous row, a byte count that is
/// not four a value, any other value, `-0.0` and NaN included — is an error
/// naming the row, and for a value, the first offending one and its index.
pub fn mask_bits_in(dir: &Path, row: &RefRow) -> Result<Vec<u16>, RefError> {
    use gguf::quant::f32_to_f16_bits;

    const ZERO: u32 = 0x0000_0000;
    const MINUS_INF: u32 = 0xff80_0000;
    let path = dir.join(row.file_name());
    let what = || {
        format!(
            "{} {}/{} ({})",
            row.kind.as_str(),
            row.name,
            row.occurrence,
            path.display()
        )
    };
    let bad = |detail: String| RefError::malformed("mask_bits", detail);
    if row.ty != "f16" {
        return Err(bad(format!("{} is {}, want an f16 mask", what(), row.ty)));
    }
    if !plain_is_logical(row) {
        return Err(bad(format!(
            "{} is a view/non-contiguous row — its plain file is a flat read, not the mask",
            what()
        )));
    }
    let raw = std::fs::read(&path)
        .map_err(|e| RefError::missing(&path, format!("mask_bits: {}: {e}", what())))?;
    let want = 4_u64.checked_mul(row.count());
    if want != Some(row.bytes) || want != Some(raw.len() as u64) {
        return Err(bad(format!(
            "{} is {} bytes and its row says {}, want 4 x {} values",
            what(),
            raw.len(),
            row.bytes,
            row.count()
        )));
    }
    let visible = f32_to_f16_bits(f32::from_bits(ZERO));
    let hidden = f32_to_f16_bits(f32::from_bits(MINUS_INF));
    raw.as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .map(|(i, b)| match u32::from_le_bytes(*b) {
            ZERO => Ok(visible),
            MINUS_INF => Ok(hidden),
            v => Err(bad(format!(
                "{} holds {} at {i}, neither 0 nor -inf",
                what(),
                f32::from_bits(v)
            ))),
        })
        .collect()
}

/// The topk row's ids from its LOGICAL twin: `ffn_moe_topk-L` is an i32
/// VIEW of the argsort output whose plain file is the flat parent read
/// (token 0's ranking only), so the ids of every token live in the logical
/// twin, read exact from its `.logical.i32` integer twin ([`ref_ints`]).
/// For a set whose manifest has no `int` rows at all (dumped before integer
/// twins existed) it reads the `.logical.f32` twin, each id cast to f32 by
/// the dumper. Accepted only when the row carries a logical twin and every
/// id is an integer in `0..n_expert`, which rules out any stride or cast
/// mix-up; the manifest's element-sum column describes the plain file and
/// is not checked here. `row` is a row of `man`.
pub fn topk_ids_logical_within(
    man: &RefManifest,
    row: &RefRow,
    n_expert: u32,
) -> Result<Vec<i32>, RefError> {
    let bad = |what: String| RefError::malformed("topk_ids_logical_within", what);
    if row.logical != Some(1) {
        return Err(bad(format!(
            "{} has no logical twin — the ids need a v2 dump set",
            row.name
        )));
    }
    if !man.ints.is_empty() {
        let ids = ref_ints(man, &row.name, row.occurrence, row.kind, Layout::Logical)?;
        if ids.len() as u64 != row.count() {
            return Err(bad(format!(
                "{}/{} has {} elements, its integer twin {}",
                row.name,
                row.occurrence,
                row.count(),
                ids.len()
            )));
        }
        return ids
            .iter()
            .map(|&v| match i32::try_from(v) {
                Ok(id) if u32::try_from(id).is_ok_and(|e| e < n_expert) => Ok(id),
                _ => Err(bad(format!(
                    "{}/{} holds id {v}, outside 0..{n_expert}",
                    row.name, row.occurrence
                ))),
            })
            .collect();
    }
    let path = man.dir.join(row.logical_file_name());
    let raw = std::fs::read(&path).map_err(|e| {
        RefError::missing(
            &path,
            format!(
                "topk_ids_logical_within: cannot read {}: {e}",
                path.display()
            ),
        )
    })?;
    let expect = 4_u64
        .checked_mul(row.count())
        .ok_or_else(|| bad("topk element count overflows".to_string()))?;
    if raw.len() as u64 != expect {
        return Err(bad(format!(
            "{} is {} bytes, want {expect} (4*count)",
            path.display(),
            raw.len()
        )));
    }
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|c| {
            let v = f32::from_le_bytes(*c);
            if v.fract() == 0.0 && (0.0..f64::from(n_expert)).contains(&f64::from(v)) {
                Ok(v as i32)
            } else {
                Err(bad(format!(
                    "{} holds non-integral id {v} — not ids cast to f32",
                    path.display()
                )))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
