//! The stage-1 oracle set's tensors, read through its manifest (`manifest`). Every
//! module's gate imports this; no module writes its own copy, because three readers
//! would disagree about what the manifest means before they disagreed about numbers.

use std::collections::HashMap;
use std::path::PathBuf;

/// One manifest row: the tensor's name and occurrence, its type, its shape,
/// the byte count of its file, and the op that produced it.
#[derive(Clone, Debug)]
pub struct RefTensor {
    pub name: String,
    pub occurrence: u32,
    pub ty: String,
    pub ne: [i64; 4],
    pub bytes: u64,
    pub op: String,
}

pub struct Oracle {
    dir: PathBuf,
    by_key: HashMap<(String, u32), RefTensor>,
}

impl Oracle {
    /// Loads `$BLOOMERY_DATA/ref/MANIFEST.tsv` ([`super::manifest::read`]). A
    /// row of a type the dumper does not widen to f32, with no op, or whose
    /// byte count is not its `ne` in f32 is a named panic.
    pub fn open() -> Oracle {
        let (dir, text) = super::manifest::read();
        let mut by_key = HashMap::new();
        for line in text.lines() {
            if !line.starts_with("tensor\t") {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            let field = |i: usize| {
                f.get(i)
                    .unwrap_or_else(|| panic!("oracle manifest row {line:?} has no column {i}"))
            };
            let int = |i: usize| -> i64 {
                field(i)
                    .parse()
                    .unwrap_or_else(|e| panic!("oracle manifest row {line:?} column {i}: {e}"))
            };
            let t = RefTensor {
                name: field(1).to_string(),
                occurrence: u32::try_from(int(2)).unwrap(),
                ty: field(3).to_string(),
                ne: [int(4), int(5), int(6), int(7)],
                bytes: u64::try_from(int(8)).unwrap(),
                op: field(10).to_string(),
            };
            // The dumper widens every one of these to raw f32; another type's
            // bytes would be read as floats they are not.
            assert!(
                matches!(t.ty.as_str(), "f32" | "f16" | "i32") && !t.op.is_empty(),
                "oracle manifest row {line:?}: type {:?} op {:?} (want f32, f16 or i32, and an op)",
                t.ty,
                t.op
            );
            let f32s: i64 = t.ne.iter().product();
            assert!(
                u64::try_from(4 * f32s) == Ok(t.bytes),
                "oracle manifest row {line:?}: ne {:?} is {f32s} f32, the row says {} bytes",
                t.ne,
                t.bytes
            );
            by_key.insert((t.name.clone(), t.occurrence), t);
        }
        assert!(!by_key.is_empty(), "oracle manifest has no tensor rows");
        Oracle { dir, by_key }
    }

    pub fn info(&self, name: &str, occurrence: u32) -> &RefTensor {
        self.by_key
            .get(&(name.to_string(), occurrence))
            .unwrap_or_else(|| {
                panic!(
                    "oracle has no tensor {name}#{occurrence}. Names come from the manifest, \
                 and a name can occur more than once in one graph — pass the occurrence."
                )
            })
    }

    /// The reference values, flattened in ggml order (ne0 contiguous). A
    /// file whose length is not its row's byte count (its `ne` in f32, which
    /// `open` checked) is a named panic.
    pub fn load(&self, name: &str, occurrence: u32) -> (Vec<f32>, &RefTensor) {
        let info = self.info(name, occurrence);
        let safe: String = name
            .chars()
            .map(|c| {
                if c == '/' || c == '\\' || c == ' ' {
                    '_'
                } else {
                    c
                }
            })
            .collect();
        let path = self.dir.join(format!("{safe}.{occurrence}.f32"));
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("oracle file {} is missing ({e})", path.display()));
        let (words, tail) = bytes.as_chunks::<4>();
        if !tail.is_empty() || u64::try_from(bytes.len()) != Ok(info.bytes) {
            panic!(
                "oracle file {} holds {} bytes; the manifest's {name}#{occurrence} is ne {:?}, \
                 {} bytes. A dump cut short or written for another shape is not a \
                 reference; the lead regenerates it: `just dump-ref`.",
                path.display(),
                bytes.len(),
                info.ne,
                info.bytes
            );
        }
        let vals: Vec<f32> = words.iter().map(|c| f32::from_le_bytes(*c)).collect();
        (vals, info)
    }
}
