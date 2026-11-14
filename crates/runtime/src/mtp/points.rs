//! The hidden rows a host keeps beside its checkpoints, so the MTP draft
//! rejoins a sequence that is cut back to a point: [`PointRows`].
//!
//! A draft layer's row at position `q` reads the token at `q` and the
//! target's final hidden row at `q − 1`. A cut back to the point at `pos`
//! leaves the target's recurrent state at `pos` and no arena holding the row
//! at `pos − 1`, so the point carries that one row beside its state; the host
//! writes it to the step arena at the cut ([`super::Host::rejoins`]) and the
//! window's [`super::MtpDraft::rejoin`] walks it with the next call's first
//! id. The rows follow the host's ledger of points: a row stands exactly
//! while its point does.

/// The hidden row beside each point of a host's checkpoints, by position.
#[derive(Clone, Debug, Default)]
pub struct PointRows {
    /// Ascending by position, one row each.
    rows: Vec<(u32, Vec<f32>)>,
}

impl PointRows {
    /// No point holds a row.
    #[must_use]
    pub fn new() -> PointRows {
        PointRows::default()
    }

    /// The row beside the point at `at`: the target's final hidden row at
    /// position `at − 1`, replacing the one already there.
    ///
    /// # Panics
    ///
    /// `at` 0, by name: nothing precedes position 0, so no point there
    /// carries a row.
    pub fn put(&mut self, at: u32, row: &[f32]) {
        assert!(
            at > 0,
            "a hidden row beside the point at position 0: nothing precedes it"
        );
        match self.rows.binary_search_by_key(&at, |&(p, _)| p) {
            Ok(i) => {
                self.rows[i].1.clear();
                self.rows[i].1.extend_from_slice(row);
            }
            Err(i) => self.rows.insert(i, (at, row.to_vec())),
        }
    }

    /// The row beside the point at `at`, if it holds one.
    #[must_use]
    pub fn get(&self, at: u32) -> Option<&[f32]> {
        let i = self.rows.binary_search_by_key(&at, |&(p, _)| p).ok()?;
        Some(&self.rows[i].1)
    }

    /// Only the rows of the points in `points` (ascending: the ledger's
    /// positions) stay; a point the ledger evicted, dropped with a cut or
    /// cleared leaves its row behind otherwise.
    pub fn retain_points(&mut self, points: &[u32]) {
        self.rows.retain(|&(p, _)| points.binary_search(&p).is_ok());
    }

    /// No row stays.
    pub fn clear(&mut self) {
        self.rows.clear();
    }

    /// The rows kept.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether no row is kept.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::PointRows;

    /// A row is kept beside its point whatever the order the points were
    /// taken in, and a second put at a position replaces the first.
    ///
    /// Mutant: the rows kept in the order put (a lookup by position misses),
    /// a second put that keeps the first row or appends beside it.
    #[test]
    fn a_row_stands_beside_its_point_and_a_put_replaces() {
        let mut r = PointRows::new();
        assert!(r.is_empty());
        r.put(30, &[3.0, 3.5]);
        r.put(10, &[1.0, 1.5]);
        r.put(20, &[2.0, 2.5]);
        assert_eq!(r.len(), 3);
        assert_eq!(r.get(10), Some(&[1.0, 1.5][..]));
        assert_eq!(r.get(20), Some(&[2.0, 2.5][..]));
        assert_eq!(r.get(30), Some(&[3.0, 3.5][..]));
        assert_eq!(r.get(25), None);
        r.put(20, &[9.0, 9.5]);
        assert_eq!(r.len(), 3);
        assert_eq!(r.get(20), Some(&[9.0, 9.5][..]));
    }

    /// The ledger's positions decide which rows stand: a point it dropped
    /// loses its row, a point it holds keeps it, and a position it holds with
    /// no row stays without one.
    ///
    /// Mutant: a retain that keeps every row, drops every row, or keeps the
    /// rows below the highest position instead of the listed ones.
    #[test]
    fn the_ledgers_points_decide_which_rows_stand() {
        let mut r = PointRows::new();
        for at in [8, 16, 24, 32] {
            r.put(at, &[at as f32]);
        }
        r.retain_points(&[8, 24, 40]);
        assert_eq!(r.len(), 2);
        assert_eq!(r.get(8), Some(&[8.0][..]));
        assert_eq!(r.get(16), None);
        assert_eq!(r.get(24), Some(&[24.0][..]));
        assert_eq!(r.get(32), None);
        assert_eq!(r.get(40), None);
        r.clear();
        assert!(r.is_empty());
        assert_eq!(r.get(8), None);
    }

    /// Position 0 has no row before it: a put there is the caller's breach,
    /// refused by name.
    #[test]
    #[should_panic(expected = "nothing precedes it")]
    fn a_row_beside_position_zero_is_refused_by_name() {
        PointRows::new().put(0, &[0.0]);
    }
}
