//! The residency profile's counting side ([`model::placement::profile`], the
//! format's owner): a running machine counts its own router picks, each id
//! of each kept row of every pass it ends ([`super::SwapMachine::end_pass`]),
//! undecayed, beside the rule's counts, and adds them to the model file's
//! profile every [`FLUSH_PASSES`] passes and when it stops. Rows a pass
//! rejects count nothing, as they move nothing in the rule.

use std::ops::Range;

use model::placement::profile::{self, FLUSH_PASSES, ProfileTarget};
use runtime::swaprule::KeptRows;

use super::Tally;
use crate::GpuError;

/// A machine's counts since its last addition to the profile file.
pub(super) struct ProfileTally {
    target: ProfileTarget,
    /// The machine's map layers, in the model's numbering.
    layers: Range<usize>,
    /// How many of them, from the first, the file holds: a draft's
    /// next-token layer, numbered after the model's, counts nothing (the
    /// draft's plan takes no seed).
    counted: usize,
    /// The file's shape: the model's layers × experts.
    delta: Vec<u32>,
    passes: u64,
    /// An addition failed: named on stderr once, and none after it.
    failed: bool,
}

impl ProfileTally {
    /// The tally of a machine over `layers` of `experts` experts a layer
    /// for the file `target`; another expert count is refused by name.
    pub(super) fn new(
        target: ProfileTarget,
        layers: Range<usize>,
        experts: usize,
    ) -> Result<ProfileTally, GpuError> {
        if experts != target.experts {
            return Err(GpuError::shape(
                "SwapMachine::new",
                format!(
                    "a machine over layers {layers:?} of {experts} experts for the profile {} \
                     of {} layers of {}",
                    target.path.display(),
                    target.layers,
                    target.experts
                ),
            ));
        }
        Ok(ProfileTally {
            delta: vec![0; target.layers * target.experts],
            counted: target.layers.saturating_sub(layers.start).min(layers.len()),
            target,
            layers,
            passes: 0,
            failed: false,
        })
    }

    /// The ended pass's `kept` rows of `t`, every id of every routed layer;
    /// every [`FLUSH_PASSES`]th pass adds the counts to the file. `t` holds
    /// what [`super::SwapMachine::end_pass`] checked: each kept row's routed
    /// layers noted whole.
    pub(super) fn fold(&mut self, t: &Tally, kept: KeptRows) {
        let n = self.layers.len();
        let experts = self.target.experts;
        for row in (0..t.max_rows).filter(|&r| kept.keeps(r)) {
            for l in 0..self.counted {
                let cell = row * n + l;
                if t.filled[cell] == 0 {
                    continue;
                }
                let base = (self.layers.start + l) * experts;
                for &id in &t.ids[cell * t.top_k..(cell + 1) * t.top_k] {
                    let c = &mut self.delta[base + id as usize];
                    *c = c.saturating_add(1);
                }
            }
        }
        self.passes += 1;
        if self.passes.is_multiple_of(FLUSH_PASSES) {
            self.flush();
        }
    }

    /// Add the counts since the last addition to the file
    /// ([`profile::add_to_file`]), then start them over. A failed addition
    /// is named on stderr, and the machine adds nothing after it; serving
    /// goes on.
    pub(super) fn flush(&mut self) {
        if self.failed || self.delta.iter().all(|&c| c == 0) {
            return;
        }
        match profile::add_to_file(&self.target, &self.delta) {
            Ok(_) => self.delta.fill(0),
            Err(e) => {
                self.failed = true;
                eprintln!("residency profile: {e}; this run adds no more counts to it");
            }
        }
    }
}
