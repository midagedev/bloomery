//! Read the stage-1 oracle set that `tools/ref/dump.sh` wrote. Every module's gate imports
//! this; no module writes its own copy, because three readers would disagree about what the
//! manifest means before they disagreed about numbers.
//!
//! The reference set is produced by the lead and only read here. If it is absent this
//! stops with an error naming the command — it never falls back to computing something,
//! because a gate that quietly measures nothing stays green (measured 2026-09-19: a runner
//! kept pointing at a stale binary through every green gate).

use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct RefTensor {
    pub name: String,
    pub occurrence: u32,
    pub ty: String,
    pub ne: [i64; 4],
    pub sum: f64,
    pub op: String,
}

pub struct Oracle {
    dir: PathBuf,
    pub tokens: Vec<i32>,
    pub model: String,
    by_key: HashMap<(String, u32), RefTensor>,
}

impl Oracle {
    /// Loads `$MULLE_DATA/ref/MANIFEST.tsv`.
    pub fn open() -> Oracle {
        let base = std::env::var("MULLE_DATA").unwrap_or_else(|_| "/root/mulle-data".into());
        let dir = PathBuf::from(base).join("ref");
        let manifest = dir.join("MANIFEST.tsv");
        let text = std::fs::read_to_string(&manifest).unwrap_or_else(|e| {
            panic!(
                "no oracle at {} ({e}). The lead produces it: `just dump-ref`. \
                 Do not run it yourself and do not skip this test.",
                manifest.display()
            )
        });
        let mut by_key = HashMap::new();
        let mut tokens = Vec::new();
        let mut model = String::new();
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("# tokens\t") {
                tokens = rest
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                continue;
            }
            if let Some(rest) = line.strip_prefix("# model\t") {
                model = rest.trim().to_string();
                continue;
            }
            if !line.starts_with("tensor\t") {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            let t = RefTensor {
                name: f[1].to_string(),
                occurrence: f[2].parse().unwrap(),
                ty: f[3].to_string(),
                ne: [
                    f[4].parse().unwrap(),
                    f[5].parse().unwrap(),
                    f[6].parse().unwrap(),
                    f[7].parse().unwrap(),
                ],
                sum: f[9].parse().unwrap(),
                op: f[10].to_string(),
            };
            by_key.insert((t.name.clone(), t.occurrence), t);
        }
        assert!(!by_key.is_empty(), "oracle manifest has no tensor rows");
        Oracle {
            dir,
            tokens,
            model,
            by_key,
        }
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

    /// The reference values, flattened in ggml order (ne0 contiguous).
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
        let vals: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        (vals, info)
    }
}

/// Elementwise comparison that reports WHERE it failed, not just that it did. A gate that
/// prints only a max is a gate you cannot act on.
pub fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: length {} vs reference {}",
        got.len(),
        want.len()
    );
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        worst <= tol,
        "{what}: max |diff| = {worst:e} at index {at} (got {}, reference {}); gate is {tol:e}",
        got[at],
        want[at]
    );
    eprintln!("{what:38} max|diff| = {worst:e}   ok (gate {tol:e})");
}

/// Routing decisions are integers and get no tolerance. A wrong expert choice inside a 1e-3
/// numeric gate is invisible, and it is the failure that matters most in the MoE block.
pub fn assert_exact_i32(got: &[i32], want_f32: &[f32], what: &str) {
    assert_eq!(
        got.len(),
        want_f32.len(),
        "{what}: length {} vs reference {}",
        got.len(),
        want_f32.len()
    );
    for (i, (&g, &w)) in got.iter().zip(want_f32).enumerate() {
        let w = w as i32;
        assert_eq!(g, w, "{what}: index {i} chose {g}, reference chose {w}");
    }
    eprintln!("{what:38} exact match on {} ids   ok", got.len());
}
