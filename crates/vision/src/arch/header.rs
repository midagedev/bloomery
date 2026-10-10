//! Reading an encoder file's header: the checks every projector type's hyperparameters share.
//!
//! Each projector module (`deepseek41v`, `qwen3vl`) declares its own table of keys and values as
//! data and calls these helpers, so the wording of a refusal, the numeric types a key may hold and
//! the order of the checks are written once. A refusal is a [`VisionError::Metadata`] naming the
//! key, the value the file holds and the value the module runs; a key that is absent is refused by
//! name as well, never defaulted.

use gguf::{Gguf, Value};

use crate::VisionError;

/// The `general.architecture` of every encoder file.
pub(crate) const ARCHITECTURE: &str = "clip";
pub(crate) const KEY_PROJECTOR: &str = "clip.projector_type";
pub(crate) const KEY_HAS_VISION: &str = "clip.has_vision_encoder";
pub(crate) const KEY_EPS: &str = "clip.vision.attention.layer_norm_epsilon";
pub(crate) const KEY_MEAN: &str = "clip.vision.image_mean";
pub(crate) const KEY_STD: &str = "clip.vision.image_std";

/// Where the keys come from: the file, or a table in the tests.
pub(crate) trait Meta {
    fn value(&self, key: &str) -> Option<&Value>;
}

impl Meta for Gguf {
    fn value(&self, key: &str) -> Option<&Value> {
        Gguf::value(self, key)
    }
}

pub(crate) fn refusal(key: &str, detail: impl Into<String>) -> VisionError {
    VisionError::Metadata {
        key: key.to_string(),
        detail: detail.into(),
    }
}

pub(crate) fn need<'a>(m: &'a impl Meta, key: &str) -> Result<&'a Value, VisionError> {
    m.value(key).ok_or_else(|| refusal(key, "is absent"))
}

pub(crate) fn need_f32(m: &impl Meta, key: &str) -> Result<f32, VisionError> {
    let v = need(m, key)?;
    match v {
        Value::F32(x) => Ok(*x),
        _ => Err(refusal(key, format!("is {v:?}, not an f32"))),
    }
}

/// The file is an encoder file: its architecture is `clip`.
pub(crate) fn check_architecture(gguf: &Gguf) -> Result<(), VisionError> {
    match gguf.architecture() {
        Some(ARCHITECTURE) => Ok(()),
        other => Err(refusal(
            "architecture",
            format!("is {other:?}; an encoder file is \"{ARCHITECTURE}\""),
        )),
    }
}

/// The file names `projector` as its projector type.
pub(crate) fn check_projector(m: &impl Meta, projector: &str) -> Result<(), VisionError> {
    match need(m, KEY_PROJECTOR)?.as_str() {
        Some(p) if p == projector => Ok(()),
        Some(other) => Err(refusal(
            KEY_PROJECTOR,
            format!("is \"{other}\"; this module reads \"{projector}\""),
        )),
        None => Err(refusal(KEY_PROJECTOR, "is not a string")),
    }
}

/// A boolean key that must be true; `why` says what needs it.
pub(crate) fn check_true(m: &impl Meta, key: &str, why: &str) -> Result<(), VisionError> {
    match need(m, key)?.as_bool() {
        Some(true) => Ok(()),
        other => Err(refusal(key, format!("is {other:?}; {why} need true"))),
    }
}

/// Count keys: `(key, the value the module runs, where that value comes from)` each. A count
/// that is not a non-negative integer, or holds another value, is refused.
pub(crate) fn counts<const N: usize>(
    m: &impl Meta,
    projector: &str,
    table: [(&str, u64, &str); N],
) -> Result<[usize; N], VisionError> {
    let mut out = [0usize; N];
    for (slot, (key, want, from)) in out.iter_mut().zip(table) {
        let v = need(m, key)?;
        let got = v
            .as_unsigned()
            .ok_or_else(|| refusal(key, format!("is {v:?}, not a count")))?;
        if got != want {
            return Err(refusal(
                key,
                format!("is {got}; {projector} runs {want} ({from})"),
            ));
        }
        *slot =
            usize::try_from(got).map_err(|_| refusal(key, format!("{got} does not fit usize")))?;
    }
    Ok(out)
}

/// A count key that may hold any positive multiple of `unit`: the width a kernel's tile takes.
/// `why` is the refusal's lead-in, ending where "a positive multiple of `unit`" continues it.
pub(crate) fn count_multiple(
    m: &impl Meta,
    key: &str,
    unit: u64,
    why: &str,
) -> Result<usize, VisionError> {
    let v = need(m, key)?;
    let got = v
        .as_unsigned()
        .ok_or_else(|| refusal(key, format!("is {v:?}, not a count")))?;
    if got == 0 || !got.is_multiple_of(unit) {
        return Err(refusal(
            key,
            format!("is {got}; {why} a positive multiple of {unit}"),
        ));
    }
    usize::try_from(got).map_err(|_| refusal(key, format!("{got} does not fit usize")))
}

/// The layer-norm epsilon, an f32 key that must equal `want`; `source` names what uses it.
pub(crate) fn check_eps(m: &impl Meta, want: f32, source: &str) -> Result<f32, VisionError> {
    let eps = need_f32(m, KEY_EPS)?;
    if eps != want {
        return Err(refusal(
            KEY_EPS,
            format!("is {eps:e}; {source} uses {want:e}"),
        ));
    }
    Ok(eps)
}

/// The per-channel normalization keys: both must hold `[0.5, 0.5, 0.5]`, the `(x - 0.5) / 0.5` of
/// the preprocessing; `why` says what runs that rule.
pub(crate) fn check_mean_std(m: &impl Meta, why: &str) -> Result<(), VisionError> {
    for key in [KEY_MEAN, KEY_STD] {
        let v = need(m, key)?;
        let per_channel = match v {
            Value::Array(a) => a.iter().map(Value::as_f32).collect::<Option<Vec<f32>>>(),
            _ => None,
        };
        if per_channel.as_deref() != Some(&[0.5f32; 3]) {
            return Err(refusal(key, format!("is {v:?}; {why}")));
        }
    }
    Ok(())
}

/// A key table and the harness the projector modules' refusal tests share.
#[cfg(test)]
pub(crate) mod testing {
    use gguf::Value;

    use super::Meta;
    use crate::VisionError;

    /// A header as a key table.
    pub(crate) struct Table(pub(crate) Vec<(&'static str, Value)>);

    impl Meta for Table {
        fn value(&self, key: &str) -> Option<&Value> {
            self.0.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
        }
    }

    /// A GGUF file of a table's keys that is deleted when dropped.
    pub(crate) struct TempFile(pub(crate) std::path::PathBuf);

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    impl Table {
        /// The table as a file with no tensor, under `general.architecture` `arch`.
        pub(crate) fn write(&self, arch: &str) -> TempFile {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let mut kvs = vec![(
                gguf::GENERAL_ARCHITECTURE.to_string(),
                Value::String(arch.into()),
            )];
            kvs.extend(self.0.iter().map(|(k, v)| ((*k).to_string(), v.clone())));
            let layout = gguf::write::Layout::new(&kvs, Vec::new()).expect("layout");
            let path = std::env::temp_dir().join(format!(
                "bloomery-vision-{}-{}.gguf",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let file = std::fs::File::create(&path).expect("create");
            gguf::write::Writer::new(file, layout)
                .and_then(gguf::write::Writer::finish)
                .expect("write");
            TempFile(path)
        }

        /// The table with `key` set to `v`, added when absent.
        pub(crate) fn with(mut self, key: &'static str, v: Value) -> Table {
            match self.0.iter_mut().find(|(k, _)| *k == key) {
                Some(slot) => slot.1 = v,
                None => self.0.push((key, v)),
            }
            self
        }

        /// The table without `key`.
        pub(crate) fn without(self, key: &str) -> Table {
            Table(self.0.into_iter().filter(|(k, _)| *k != key).collect())
        }
    }

    /// Every case must be refused with an error that starts with its text; the failures are
    /// listed together.
    pub(crate) fn expect_refusals<T>(
        cases: Vec<(Table, &str)>,
        read: impl Fn(&Table) -> Result<T, VisionError>,
    ) {
        let wrong: Vec<String> = cases
            .into_iter()
            .enumerate()
            .filter_map(|(i, (t, want))| match read(&t) {
                Ok(_) => Some(format!("case {i} read; expected \"{want}\"")),
                Err(e) if e.to_string().starts_with(want) => None,
                Err(e) => Some(format!("case {i}: {e}")),
            })
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }
}
