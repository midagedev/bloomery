//! The host union's tap on the routed down's input rows: per down stack and expert, the column sums of
//! squares of the SwiGLU rows its down reads and how many rows that is, which an importance collector adds
//! to the card's own sums for the picks the card never holds as f32 rows.
//!
//! A union call that is handed a tap ([`crate::moe::HostLayer::experts_union_tapped_into`]) sums each
//! distinct expert's rows in one worker, in column order, in f32 (the card's kernel adds its picks in f32
//! the same way) and adds that partial to the expert's running sums here in f64, once a call: one lock an
//! (expert, call), never one a row, and no two workers share an expert, so no lock is ever waited on. The
//! tap knows no architecture: a stack is its tensor's name.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, PoisonError};

use crate::ModelError;

/// One expert's running totals: the rows seen and the column sums of squares over them, empty until the first
/// row.
#[derive(Default)]
struct ExpertSums {
    rows: u64,
    sums: Vec<f64>,
}

/// One down stack's experts, each behind its own lock.
pub struct DownStack {
    ff: usize,
    experts: Box<[Mutex<ExpertSums>]>,
}

impl DownStack {
    /// Add `rows` rows of one call to expert `id`: `sum2` the column sums of squares of those rows, `ff`
    /// values.
    ///
    /// # Panics
    /// An `id` past the stack's experts or a `sum2` of another width than the stack's: the union call checked
    /// both before any pass ran.
    pub fn add(&self, id: u32, rows: usize, sum2: &[f32]) {
        assert_eq!(
            sum2.len(),
            self.ff,
            "down tap: a row sum of {} values for a stack of rows of {}",
            sum2.len(),
            self.ff
        );
        let mut e = self
            .experts
            .get(id as usize)
            .unwrap_or_else(|| {
                panic!(
                    "down tap: expert {id} of a stack of {} experts",
                    self.experts.len()
                )
            })
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if e.sums.is_empty() {
            e.sums.resize(self.ff, 0.0);
        }
        e.rows += rows as u64;
        for (s, &v) in e.sums.iter_mut().zip(sum2) {
            *s += f64::from(v);
        }
    }
}

/// One stack's totals as they stand: `n_expert` matrices of `ff` columns, expert `e`'s sums at
/// `sums[e·ff..(e + 1)·ff]` (zeros for an expert no row reached) and its rows at `rows[e]`.
pub struct DownRows {
    pub ff: usize,
    pub n_expert: usize,
    pub sums: Vec<f64>,
    pub rows: Vec<u64>,
}

/// The taps of a pass: every down stack a union call fed, by tensor name.
#[derive(Default)]
pub struct DownTap {
    stacks: Mutex<BTreeMap<String, Arc<DownStack>>>,
}

impl DownTap {
    /// An empty tap.
    #[must_use]
    pub const fn new() -> DownTap {
        DownTap {
            stacks: Mutex::new(BTreeMap::new()),
        }
    }

    /// Stack `name` of `n_expert` experts over rows of `ff` values, made at its first call; a later call that
    /// names another shape is refused by name. One lock a union call.
    pub fn stack(
        &self,
        name: &str,
        n_expert: usize,
        ff: usize,
    ) -> Result<Arc<DownStack>, ModelError> {
        let mut stacks = self.stacks.lock().unwrap_or_else(PoisonError::into_inner);
        let stack = stacks.entry(name.to_string()).or_insert_with(|| {
            Arc::new(DownStack {
                ff,
                experts: (0..n_expert).map(|_| Mutex::default()).collect(),
            })
        });
        if (stack.ff, stack.experts.len()) != (ff, n_expert) {
            return Err(ModelError::Metadata {
                key: name.to_string(),
                detail: format!(
                    "down tap: fed as {n_expert} experts of rows of {ff}, tapped as {} of {}",
                    stack.experts.len(),
                    stack.ff
                ),
            });
        }
        Ok(Arc::clone(stack))
    }

    /// The stack names tapped so far, sorted.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        let stacks = self.stacks.lock().unwrap_or_else(PoisonError::into_inner);
        stacks.keys().cloned().collect()
    }

    /// Whether any call fed `name`.
    #[must_use]
    pub fn holds(&self, name: &str) -> bool {
        let stacks = self.stacks.lock().unwrap_or_else(PoisonError::into_inner);
        stacks.contains_key(name)
    }

    /// `name`'s totals as they stand, read between calls, or `None` for a stack no call fed.
    #[must_use]
    pub fn rows_of(&self, name: &str) -> Option<DownRows> {
        let stack = {
            let stacks = self.stacks.lock().unwrap_or_else(PoisonError::into_inner);
            Arc::clone(stacks.get(name)?)
        };
        let n_expert = stack.experts.len();
        let mut out = DownRows {
            ff: stack.ff,
            n_expert,
            sums: vec![0.0; n_expert * stack.ff],
            rows: vec![0; n_expert],
        };
        for (e, cell) in stack.experts.iter().enumerate() {
            let cell = cell.lock().unwrap_or_else(PoisonError::into_inner);
            out.rows[e] = cell.rows;
            if !cell.sums.is_empty() {
                out.sums[e * stack.ff..(e + 1) * stack.ff].copy_from_slice(&cell.sums);
            }
        }
        Some(out)
    }

    /// Drop every stack: the next call starts a fresh pass.
    pub fn clear(&self) {
        self.stacks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tap_adds_calls_by_expert_and_refuses_another_shape() {
        let tap = DownTap::new();
        let name = "stack";
        let s = tap.stack(name, 3, 2).unwrap();
        s.add(2, 4, &[1.0, 2.0]);
        s.add(2, 1, &[0.5, 0.25]);
        s.add(0, 7, &[3.0, 0.0]);
        let r = tap.rows_of(name).unwrap();
        assert_eq!((r.ff, r.n_expert), (2, 3));
        assert_eq!(r.rows, [7, 0, 5]);
        assert_eq!(r.sums, [3.0, 0.0, 0.0, 0.0, 1.5, 2.25]);
        assert!(tap.stack(name, 3, 4).is_err());
        assert!(tap.stack(name, 2, 2).is_err());
        assert_eq!(tap.names(), [name]);
        tap.clear();
        assert!(!tap.holds(name));
        assert!(tap.rows_of(name).is_none());
    }
}
