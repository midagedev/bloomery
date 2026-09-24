//! Read a V4.1 oracle set that `tools/ref/dump.sh` wrote for the file the tree
//! runs: its header checked (the ik tree, the architecture, the completion
//! trailer, the model file), its contiguous f32 nodes and the logical twins
//! of its integer nodes. The set is produced by the lead and only read here;
//! an absent or stale set stops the gate by name.

use std::collections::HashMap;
use std::path::PathBuf;

use gguf::Split;

pub struct Set {
    name: &'static str,
    dir: PathBuf,
    pub shapes: HashMap<String, [usize; 3]>,
    logical: HashMap<String, String>,
}

impl Set {
    /// The set `name` of the V4.1 file `split` opens, dumped by ik tree `build`.
    pub fn open(split: &Split, name: &'static str, build: &str) -> Set {
        let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
        let dir = PathBuf::from(base).join(gguf::v41::set(name));
        let path = dir.join("MANIFEST.tsv");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "no oracle manifest at {} ({e}). The lead produces it (tools/ref/dump.sh). \
                 Do not run it yourself and do not skip this test.",
                path.display()
            )
        });
        let mut header = HashMap::new();
        let mut shapes = HashMap::new();
        let mut logical = HashMap::new();
        for line in text.lines() {
            if let Some(h) = line.strip_prefix("# ") {
                if let Some((k, v)) = h.split_once('\t') {
                    header.insert(k.to_string(), v.to_string());
                }
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            match f[0] {
                "tensor" if f[2] == "0" && f[11] == "1" => {
                    let ne = |i: usize| {
                        f[i].parse::<usize>()
                            .unwrap_or_else(|e| panic!("{line}: {e}"))
                    };
                    shapes.insert(f[1].to_string(), [ne(4), ne(5), ne(6)]);
                }
                "int" if f[2] == "0" && f[6] == "logical" => {
                    logical.insert(f[1].to_string(), f[11].to_string());
                }
                _ => {}
            }
        }
        let get = |k: &str| header.get(k).map(String::as_str);
        assert_eq!(
            get("build"),
            Some(build),
            "{name} was dumped from another ik tree"
        );
        assert_eq!(get("arch"), Some("deepseek41"), "{name} is not a V4.1 set");
        assert!(
            header.contains_key("complete"),
            "{name}: the manifest has no completion trailer — the dump that wrote it did not finish"
        );
        // Both V4.1 files name their shards alike: the path tells them apart.
        assert_eq!(
            get("model").map(str::to_string),
            split
                .shard_path(0)
                .map(|p| p.to_string_lossy().into_owned()),
            "{name} was dumped from another file"
        );
        Set {
            name,
            dir,
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
        let b = self.bytes(&format!("{node}.0.f32"));
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
