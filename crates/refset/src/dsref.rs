//! ik's DSpark draft sets (`tools/ref/dump_draft.cpp`): every node of every
//! graph the draft computes while the target decodes a fixed prompt with the
//! draft on, block by block.
//!
//! The manifest is a node dump's ([`crate::ik`]) with four more columns on
//! every `tensor`, `input` and `int` row — `block`, `row`, `accepted`, `graph`
//! (`kv`, the feature-to-KV graph, or `block`, the block pass) — and three
//! more row kinds, each with its column line: `draft` (the block's proposals),
//! `verify` (what the target made of the block), `plain` (a step decoded
//! without the draft). A file is named
//! `<b<block> | w>.[kv.]<stem>.<occurrence>[.input][.logical].<ext>`: `w` for
//! the prompt warmup, block −1.

use crate::RefError;
use crate::columns::Columns;
use crate::family::Family;
use crate::ik::{
    FileElem, IntRow, Layout, RefRow, RowKind, dump_file_name, parse_int_row, parse_ref_row,
};
use std::fmt;
use std::ops::Deref;
use std::path::{Path, PathBuf};

/// Which of a block's graphs a row belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Graph {
    /// The feature-to-KV graph: the committed target rows into the draft's
    /// KV ring.
    Kv,
    /// The block pass.
    Block,
}

impl Graph {
    fn parse(v: &str, at: &dyn fmt::Display) -> Result<Graph, RefError> {
        match v {
            "kv" => Ok(Graph::Kv),
            "block" => Ok(Graph::Block),
            v => Err(RefError::malformed(
                at,
                format!("graph {v:?}, want kv or block"),
            )),
        }
    }
}

/// A `tensor` or `input` row of a draft set: a node dump's row
/// ([`RefRow`], which it dereferences to) in a block and a graph.
#[derive(Debug, Clone)]
pub struct DsRow {
    pub row: RefRow,
    /// The block the row was dumped in; −1 is the prompt warmup.
    pub block: i32,
    pub graph: Graph,
}

impl Deref for DsRow {
    type Target = RefRow;
    fn deref(&self) -> &RefRow {
        &self.row
    }
}

/// An `int` row of a draft set, in a block and a graph.
#[derive(Debug, Clone)]
pub struct DsInt {
    pub int: IntRow,
    pub block: i32,
    pub graph: Graph,
}

impl Deref for DsInt {
    type Target = IntRow;
    fn deref(&self) -> &IntRow {
        &self.int
    }
}

/// A `draft` row: row `row` of block `block`'s last proposals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Draft {
    pub block: i32,
    pub row: usize,
    pub token: u32,
}

/// A `verify` row: what the target made of a block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verify {
    pub block: i32,
    /// The target position the round started at.
    pub pos: u64,
    /// The token the block starts from.
    pub id_last: u32,
    /// Whether `id_last` came from the previous round's carry (`1`).
    pub carry: u32,
    /// The tokens ik proposed and accepted, its own metrics.
    pub drafted: u64,
    pub accepted: usize,
    /// The target tokens the round committed.
    pub target: Vec<u32>,
}

impl Verify {
    /// The target's token for each draft row, in order.
    #[must_use]
    pub fn targets(&self) -> &[u32] {
        let skip = usize::from(self.carry == 0).min(self.target.len());
        &self.target[skip..]
    }
}

/// A `plain` row: a step decoded without the draft.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plain {
    pub pos: u64,
    pub token: u32,
}

/// A draft set's whole manifest.
#[derive(Debug, Clone)]
pub struct Dsref {
    /// The set's directory; every file a row names resolves under it.
    pub dir: PathBuf,
    /// `# model`: the target file the set was dumped from.
    pub model: Option<String>,
    /// `# draft_model`: the draft file.
    pub draft_model: Option<String>,
    /// `# build`: the draft tree, its base build and its patch.
    pub build: Option<String>,
    /// `# arch`: the target's architecture.
    pub arch: Option<String>,
    /// `# complete <written> <skipped>`; `None` when the dump died.
    pub complete: Option<(u64, u64)>,
    /// `tensor` and `input` rows, in file order.
    pub rows: Vec<DsRow>,
    /// `int` rows, in file order.
    pub ints: Vec<DsInt>,
    pub drafts: Vec<Draft>,
    pub verify: Vec<Verify>,
    pub plain: Vec<Plain>,
}

/// The column lines of a draft set.
#[derive(Default)]
struct SetColumns {
    tensor: Option<Columns>,
    input: Option<Columns>,
    int: Option<Columns>,
    draft: Option<Columns>,
    verify: Option<Columns>,
    plain: Option<Columns>,
}

impl SetColumns {
    /// The slot of column line `key`, `None` for a line that is not one.
    fn slot(&mut self, key: &str) -> Option<&mut Option<Columns>> {
        Some(match key {
            "kind" => &mut self.tensor,
            "input" => &mut self.input,
            "int" => &mut self.int,
            "draft" => &mut self.draft,
            "verify" => &mut self.verify,
            "plain" => &mut self.plain,
            _ => return None,
        })
    }

    fn of(&self, kind: &str, at: &dyn fmt::Display) -> Result<&Columns, RefError> {
        let cols = match kind {
            "tensor" => &self.tensor,
            "input" => &self.input,
            "int" => &self.int,
            "draft" => &self.draft,
            "verify" => &self.verify,
            _ => &self.plain,
        };
        cols.as_ref()
            .ok_or_else(|| RefError::malformed(at, format!("a {kind} row before its column line")))
    }
}

/// A draft manifest line's position, formatted only into an error.
struct At<'a>(&'a Path, usize);

impl fmt::Display for At<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dsref: {}:{}", self.0.display(), self.1)
    }
}

impl Dsref {
    /// Parse `dir/MANIFEST.tsv`, every row by its kind's column line. A row
    /// of an unknown kind, of another width than its line or before it, a
    /// field that does not parse, or a header value twice is `Malformed`.
    pub fn read(dir: &Path) -> Result<Dsref, RefError> {
        let path = dir.join("MANIFEST.tsv");
        let text = std::fs::read_to_string(&path).map_err(|e| {
            RefError::missing(&path, format!("dsref: cannot read {}: {e}", path.display()))
        })?;
        let mut set = Dsref {
            dir: dir.to_path_buf(),
            model: None,
            draft_model: None,
            build: None,
            arch: None,
            complete: None,
            rows: Vec::new(),
            ints: Vec::new(),
            drafts: Vec::new(),
            verify: Vec::new(),
            plain: Vec::new(),
        };
        let mut cols = SetColumns::default();
        for (i, line) in text.lines().enumerate() {
            let at = At(&path, i + 1);
            if let Some(h) = line.strip_prefix('#') {
                set.header_line(h, &mut cols, &at)?;
            } else if !line.is_empty() {
                set.data_line(line, &cols, &at)?;
            }
        }
        Ok(set)
    }

    /// Read the set at `dir` and check it against `family`'s row
    /// ([`check_family`](Self::check_family)).
    pub fn open(dir: &Path, family: &Family) -> Result<Dsref, RefError> {
        let set = Dsref::read(dir)?;
        set.check_family(family)?;
        Ok(set)
    }

    /// This set against its family's row: the completion trailer
    /// ([`RefError::Unfinished`]), the target and the draft files it states
    /// ([`RefError::Stale`]), then its `# arch` and `# build`
    /// ([`RefError::Foreign`]).
    pub fn check_family(&self, family: &Family) -> Result<(), RefError> {
        if self.complete.is_none() {
            return Err(RefError::Unfinished {
                set: self.dir.display().to_string(),
            });
        }
        family.check_file(&self.dir, "# model", self.model.as_deref())?;
        family.check_draft(&self.dir, self.draft_model.as_deref())?;
        family.check_arch(&self.dir, self.arch.as_deref())?;
        family.check_build(&self.dir, self.build.as_deref())
    }

    /// One `#` line, its leading `#` taken off.
    fn header_line(
        &mut self,
        h: &str,
        cols: &mut SetColumns,
        at: &dyn fmt::Display,
    ) -> Result<(), RefError> {
        let Some((key, v)) = h.strip_prefix(' ').and_then(|h| h.split_once('\t')) else {
            return Ok(());
        };
        if let Some(slot) = cols.slot(key) {
            if slot.is_some() {
                return Err(RefError::malformed(
                    at,
                    format!("a second # {key} column line"),
                ));
            }
            *slot = Some(Columns::new(key, v.split('\t')));
            return Ok(());
        }
        let once = |slot: &mut Option<String>| {
            if slot.replace(v.to_string()).is_some() {
                return Err(RefError::malformed(at, format!("a second # {key} line")));
            }
            Ok(())
        };
        match key {
            "model" => once(&mut self.model),
            "draft_model" => once(&mut self.draft_model),
            "build" => once(&mut self.build),
            "arch" => once(&mut self.arch),
            "complete" => {
                let (w, s) = v.split_once('\t').ok_or_else(|| {
                    RefError::malformed(at, format!("complete {v:?}: not two counts"))
                })?;
                let counts = (
                    crate::columns::parse_field(w, "complete", at)?,
                    crate::columns::parse_field(s, "complete", at)?,
                );
                if self.complete.replace(counts).is_some() {
                    return Err(RefError::malformed(at, "a second # complete line"));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// One data row.
    fn data_line(
        &mut self,
        line: &str,
        cols: &SetColumns,
        at: &dyn fmt::Display,
    ) -> Result<(), RefError> {
        let kind = line.split('\t').next().unwrap_or("");
        match kind {
            "tensor" | "input" => {
                let c = cols.of(kind, at)?;
                let row = parse_ref_row(
                    if kind == "tensor" {
                        RowKind::Tensor
                    } else {
                        RowKind::Input
                    },
                    c,
                    line,
                    at,
                )?;
                let (block, graph) = place(c, line, at)?;
                self.rows.push(DsRow { row, block, graph });
            }
            "int" => {
                let c = cols.of(kind, at)?;
                let int = parse_int_row(c, line, at)?;
                let (block, graph) = place(c, line, at)?;
                self.ints.push(DsInt { int, block, graph });
            }
            "draft" => {
                let r = cols.of(kind, at)?.row(line, at)?;
                self.drafts.push(Draft {
                    block: r.parse("block")?,
                    row: r.parse("row")?,
                    token: r.parse("token")?,
                });
            }
            "verify" => {
                let r = cols.of(kind, at)?.row(line, at)?;
                let target = r.text("target")?;
                self.verify.push(Verify {
                    block: r.parse("block")?,
                    pos: r.parse("pos")?,
                    id_last: r.parse("id_last")?,
                    carry: r.parse("carry")?,
                    drafted: r.parse("drafted")?,
                    accepted: r.parse("accepted")?,
                    target: target
                        .split(',')
                        .map(|t| crate::columns::parse_field(t, "target", at))
                        .collect::<Result<_, _>>()?,
                });
            }
            "plain" => {
                let r = cols.of(kind, at)?.row(line, at)?;
                self.plain.push(Plain {
                    pos: r.parse("pos")?,
                    token: r.parse("token")?,
                });
            }
            "skip" | "skip-input" => {}
            k => return Err(RefError::malformed(at, format!("unknown row kind {k:?}"))),
        }
        Ok(())
    }

    /// The `tensor` and `input` rows of block `b`'s graph `graph`, in file
    /// order.
    #[must_use]
    pub fn in_block(&self, b: i32, graph: Graph) -> Vec<&DsRow> {
        self.rows
            .iter()
            .filter(|r| r.block == b && r.graph == graph)
            .collect()
    }

    /// The row `name`/`occurrence` of block `b`'s graph `graph`, whatever
    /// its kind; a set without it is an error naming the block.
    pub fn find(
        &self,
        b: i32,
        graph: Graph,
        name: &str,
        occurrence: u32,
    ) -> Result<&DsRow, RefError> {
        self.in_block(b, graph)
            .into_iter()
            .find(|r| r.name == name && r.occurrence == occurrence)
            .ok_or_else(|| {
                self.no_row(format!(
                    "the set has no {} row {name:?}#{occurrence} in block {b}",
                    graph_name(graph)
                ))
            })
    }

    /// Block `b`'s block-pass node of `op` whose sources match (`""` matches
    /// any), the first in graph order.
    pub fn find_op(&self, b: i32, op: &str, src0: &str, src1: &str) -> Result<&DsRow, RefError> {
        let src =
            |col: &Option<String>, want: &str| want.is_empty() || col.as_deref() == Some(want);
        self.in_block(b, Graph::Block)
            .into_iter()
            .find(|r| {
                r.kind == RowKind::Tensor && r.op == op && src(&r.src0, src0) && src(&r.src1, src1)
            })
            .ok_or_else(|| {
                self.no_row(format!(
                    "block {b} has no {op} node over {src0:?}, {src1:?}"
                ))
            })
    }

    /// The `i`-th block-pass node of `op` in block `b`, in graph order.
    pub fn nth_op(&self, b: i32, op: &str, i: usize) -> Result<&DsRow, RefError> {
        self.in_block(b, Graph::Block)
            .into_iter()
            .filter(|r| r.kind == RowKind::Tensor && r.op == op)
            .nth(i)
            .ok_or_else(|| self.no_row(format!("block {b} has fewer than {} {op} nodes", i + 1)))
    }

    /// The `int` row of block `b`'s graph `graph` twinning `name`/occurrence
    /// in `layout`.
    pub fn int(
        &self,
        b: i32,
        graph: Graph,
        name: &str,
        occurrence: u32,
        layout: Layout,
    ) -> Result<&DsInt, RefError> {
        self.ints
            .iter()
            .find(|r| {
                r.block == b
                    && r.graph == graph
                    && r.name == name
                    && r.occurrence == occurrence
                    && r.layout == layout
            })
            .ok_or_else(|| {
                self.no_row(format!(
                    "the set has no {} int row {name:?}#{occurrence} in block {b}",
                    layout.as_str()
                ))
            })
    }

    /// The name the dumper gives `r`'s file of `layout` and `elem`.
    #[must_use]
    pub fn file_name(r: &DsRow, layout: Layout, elem: FileElem) -> String {
        let block = if r.block < 0 {
            "w".to_string()
        } else {
            format!("b{}", r.block)
        };
        let kv = match r.graph {
            Graph::Kv => "kv.",
            Graph::Block => "",
        };
        format!(
            "{block}.{kv}{}",
            dump_file_name(&r.name, r.occurrence, r.kind, layout, elem)
        )
    }

    /// `r`'s plain f32 file: as many values as its `ne` has elements.
    pub fn f32s(&self, r: &DsRow) -> Result<Vec<f32>, RefError> {
        self.f32_file(r, Layout::Flat)
    }

    /// `r`'s elements in logical order: its `.logical.f32` twin when the
    /// row has one, else its plain file.
    pub fn logical_f32s(&self, r: &DsRow) -> Result<Vec<f32>, RefError> {
        let layout = if r.logical == Some(1) {
            Layout::Logical
        } else {
            Layout::Flat
        };
        self.f32_file(r, layout)
    }

    fn f32_file(&self, r: &DsRow, layout: Layout) -> Result<Vec<f32>, RefError> {
        let p = self.dir.join(Dsref::file_name(r, layout, FileElem::F32));
        let v: Vec<f32> = self
            .bytes(&p)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        self.count(&p, v.len() as u64, r.count(), r.ne)?;
        Ok(v)
    }

    /// `r`'s `.i32` integer twin in `layout`: as many values as its `ne`
    /// has elements.
    pub fn i32s(&self, r: &DsRow, layout: Layout) -> Result<Vec<i32>, RefError> {
        let p = self.dir.join(Dsref::file_name(r, layout, FileElem::I32));
        let v = i32s_of(&self.bytes(&p)?);
        self.count(&p, v.len() as u64, r.count(), r.ne)?;
        Ok(v)
    }

    /// The values of int row `r`'s file: as many as its `count`.
    pub fn int_values(&self, r: &DsInt) -> Result<Vec<i32>, RefError> {
        let p = self.dir.join(&r.file);
        let v = i32s_of(&self.bytes(&p)?);
        if v.len() as u64 != r.count {
            return Err(RefError::malformed(
                p.display(),
                format!("{} ids, manifest {}", v.len(), r.count),
            ));
        }
        Ok(v)
    }

    /// The `verify` row of block `b`.
    pub fn verify_of(&self, b: i32) -> Result<&Verify, RefError> {
        self.verify
            .iter()
            .find(|v| v.block == b)
            .ok_or_else(|| self.no_row(format!("the set has no verify line for block {b}")))
    }

    /// The blocks the target verified, in file order.
    #[must_use]
    pub fn blocks(&self) -> Vec<i32> {
        self.verify.iter().map(|v| v.block).collect()
    }

    /// Block `b`'s proposals, row after row.
    #[must_use]
    pub fn drafts_of(&self, b: i32) -> Vec<u32> {
        self.drafts
            .iter()
            .filter(|d| d.block == b)
            .map(|d| d.token)
            .collect()
    }

    fn bytes(&self, p: &Path) -> Result<Vec<u8>, RefError> {
        std::fs::read(p).map_err(|e| RefError::missing(p, format!("{}: {e}", p.display())))
    }

    fn count(&self, p: &Path, got: u64, want: u64, ne: [u64; 4]) -> Result<(), RefError> {
        if got != want {
            return Err(RefError::malformed(
                p.display(),
                format!("{got} values, manifest ne {ne:?}"),
            ));
        }
        Ok(())
    }

    fn no_row(&self, what: String) -> RefError {
        RefError::missing(&self.dir, what)
    }
}

fn graph_name(g: Graph) -> &'static str {
    match g {
        Graph::Kv => "kv",
        Graph::Block => "block",
    }
}

/// A row's `block` and `graph` columns.
fn place(c: &Columns, line: &str, at: &dyn fmt::Display) -> Result<(i32, Graph), RefError> {
    let r = c.row(line, at)?;
    Ok((r.parse("block")?, Graph::parse(r.text("graph")?, at)?))
}

fn i32s_of(b: &[u8]) -> Vec<i32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect()
}

#[cfg(test)]
mod tests;
