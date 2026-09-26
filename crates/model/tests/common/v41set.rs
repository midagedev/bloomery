//! Read a V4.1 oracle set that `tools/ref/dump.sh` wrote for the file the tree
//! runs, through the reference-set reader (`refset::ik`): the node dumps'
//! family check (the completion trailer, the model file, the architecture, the
//! ik tree), then the file a split opens — the set's `# model` path and its
//! `# model_file` name — and its contiguous f32 nodes and the logical twins of
//! its integer nodes. The set is produced by the lead and only read here; an
//! absent or stale set stops the gate by name.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use gguf::Split;
use refset::arch::deepseek41::IK;
use refset::ik::{FileElem, Layout, RefManifest, RowKind, dump_file_name};

pub struct Set {
    name: &'static str,
    dir: PathBuf,
    pub shapes: HashMap<String, [usize; 3]>,
    logical: HashMap<String, String>,
}

impl Set {
    /// The set `name` of the V4.1 file `split` opens.
    pub fn open(split: &Split, name: &'static str) -> Set {
        let man = RefManifest::open(&IK.path(name), &IK).unwrap_or_else(|e| {
            panic!(
                "{e}. The lead produces the set (tools/ref/dump.sh). \
                 Do not run it yourself and do not skip this test."
            )
        });
        // Both V4.1 files name their shards alike: the path tells them apart.
        let shard = split.shard_path(0);
        assert_eq!(
            man.header.model().map(str::to_string),
            shard.map(|p| p.to_string_lossy().into_owned()),
            "{name} was dumped from another file"
        );
        assert_eq!(
            man.header.model_file,
            shard
                .and_then(Path::file_name)
                .map(|n| n.to_string_lossy().into_owned()),
            "{name} was dumped from another file"
        );
        let mut shapes = HashMap::new();
        for r in man
            .tensors
            .iter()
            .filter(|r| r.occurrence == 0 && r.contig == Some(1))
        {
            let ne = |i: usize| {
                usize::try_from(r.ne[i]).unwrap_or_else(|e| panic!("{name}: {}: {e}", r.name))
            };
            shapes.insert(r.name.clone(), [ne(0), ne(1), ne(2)]);
        }
        let mut logical = HashMap::new();
        for r in man
            .ints
            .iter()
            .filter(|r| r.occurrence == 0 && r.layout == Layout::Logical)
        {
            logical.insert(r.name.clone(), r.file.clone());
        }
        Set {
            name,
            dir: man.dir,
            shapes,
            logical,
        }
    }

    fn bytes(&self, file: &str) -> Vec<u8> {
        let path = self.dir.join(file);
        std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    /// A contiguous f32 node of shape `ne` (`[ne0, ne1, ne2]`).
    pub fn f32s(&self, node: &str, ne: [usize; 3]) -> Vec<f32> {
        let got = self
            .shapes
            .get(node)
            .unwrap_or_else(|| panic!("{}: no contiguous node {node}", self.name));
        assert_eq!(*got, ne, "{}: {node}'s shape", self.name);
        let b = self.bytes(&dump_file_name(
            node,
            0,
            RowKind::Tensor,
            Layout::Flat,
            FileElem::F32,
        ));
        assert_eq!(
            b.len(),
            4 * ne.iter().product::<usize>(),
            "{node}: file length"
        );
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    /// The logical twin of an integer node: `count` values in logical order.
    pub fn i32s_logical(&self, node: &str, count: usize) -> Vec<i32> {
        let file = self
            .logical
            .get(node)
            .unwrap_or_else(|| panic!("{}: no logical twin of {node}", self.name));
        let b = self.bytes(file);
        assert_eq!(b.len(), 4 * count, "{node}: logical twin length");
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect()
    }
}
