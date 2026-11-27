//! The step profile of an Inkling 2 token: every group of the walk on a
//! command buffer of its own, waited, its GPU interval read, one sample a
//! group in the token's order.
//!
//! The walk encodes a token as groups (the dispatches between two barriers,
//! [`super::count`]), and a group's own code is straight-line, so the profile
//! runs the walk once a group: each round's encoder holds every group but one
//! back from Metal ([`crate::encode::ComputeEncoder::probe_only`]), the host
//! side of the walk (windows, checks, launch words) runs whole, and the
//! command buffer carries the one group. Each group runs once, in the
//! token's order, so the token's state ends where a plain step leaves it.
//! The tags the walk puts on its encoder name each group; the paper's group
//! list ([`super::count::visit_groups`]) over the checked file's visits is
//! held against them, so a group the walk gains or loses is a named refusal
//! here and a red clause in the e2e gate.
//!
//! The groups' GPU times are the profile's own, not the plain step's: each
//! command buffer starts and drains alone, so their sum reads above the real
//! step's GPU span by what a barrier overlaps and below its wall by what the
//! host-fed gap holds.

use bloomery_device::Elems;

use super::check::Checked;
use super::count::{self, Kind, head_groups};
use super::ops::Walk;
use super::session::{SessionError, refuse};
use crate::encode::EncodeCount;
use crate::inkling2_geom::{HEAD, KV_HEADS};
use crate::session::clock::STAMP_WAIT;
use crate::stream::GpuInterval;

/// The bytes one group of a token moves, by what moves them: the weight
/// bytes its launches read and the cache bytes the flash reads. The
/// activations the groups read and write are not counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupBytes {
    pub weights: u64,
    pub state: u64,
}

impl GroupBytes {
    /// The group's bytes altogether.
    #[must_use]
    pub const fn total(self) -> u64 {
        self.weights + self.state
    }
}

/// One group of a profiled token: what it runs, the visit it runs in (none
/// for the head), the dispatches it holds, the GPU seconds its own command
/// buffer ran (`GPUEndTime` − `GPUStartTime`) and the bytes it moved.
#[derive(Clone, Copy, Debug)]
pub struct ProfileGroup {
    pub kind: Kind,
    pub visit: Option<usize>,
    pub dispatches: u32,
    pub gpu_s: f64,
    pub bytes: GroupBytes,
}

/// A group of the token as the paper names it: the kind, the dispatches and
/// the visit.
type Paper = (Kind, u32, Option<usize>);

/// The token's groups over the checked file's visits, as the paper counts
/// them, in walk order.
fn paper_groups(c: &Checked) -> Vec<Paper> {
    let mut out = Vec::new();
    for (vi, v) in c.visits.iter().enumerate() {
        let visit = count::Visit {
            windowed: v.windowed,
            message: v.message,
            boundary: v.boundary,
            pass_end: v.pass_end,
            first: v.layer == 0,
            layer: v.layer,
        };
        out.extend(
            count::visit_groups(&visit)
                .iter()
                .map(|g| (g.kind, g.dispatches, Some(vi))),
        );
    }
    out.extend(head_groups().iter().map(|g| (g.kind, g.dispatches, None)));
    out
}

/// The bytes a group of `kind` in visit `visit` (none: the head) of the token
/// at `pos` moves. `first` is the first group of its kind in its visit: the
/// flash's walk carries the cache bytes and the fold none, the head's logits
/// carry the table and its pick none.
fn bytes_of(c: &Checked, kind: Kind, visit: Option<usize>, pos: u32, first: bool) -> GroupBytes {
    let weights = |w: usize| GroupBytes {
        weights: w as u64,
        state: 0,
    };
    let Some(vi) = visit else {
        return match kind {
            Kind::Head if first => weights(c.head.bytes_of(c.cut)),
            _ => GroupBytes::default(),
        };
    };
    let v = &c.visits[vi];
    let p = &c.layers[v.layer];
    match kind {
        Kind::Proj => {
            let sites = p.qrg.iter().chain([&p.sink, &p.k, &p.v]);
            let taps = p.k_taps.iter().chain(&p.v_taps);
            weights(sites.chain(taps).map(super::check::Site::bytes).sum())
        }
        Kind::Wo => weights(p.wo.bytes()),
        Kind::GateUp => weights(c.top_k * p.gate_up.expert_bytes()),
        Kind::Down => weights(c.top_k * p.down.expert_bytes()),
        Kind::Flash if first => {
            let live = pos as usize + 1;
            let keys = if v.windowed { live.min(c.window) } else { live };
            // K and V rows of the live keys, f16 a value.
            GroupBytes {
                weights: 0,
                state: (keys * 2 * KV_HEADS * HEAD * 2) as u64,
            }
        }
        _ => GroupBytes::default(),
    }
}

impl Walk<'_> {
    /// The token `id` at position `pos`, every group on a command buffer of
    /// its own, waited; one sample a group appended to `out` (cleared first,
    /// its capacity kept). Returns the pick and what the token encoded.
    ///
    /// # Errors
    /// A walk whose groups are not the paper's (by group), a command buffer
    /// that failed, or the device's.
    pub(super) fn token_profile(
        &mut self,
        id: u32,
        pos: u32,
        out: &mut Vec<ProfileGroup>,
    ) -> Result<(u32, EncodeCount), SessionError> {
        let paper = paper_groups(self.c);
        let h0 = self.embedded(id)?;
        self.a.scratch.rows.write_at(Elems(0), &h0)?;
        self.a.scratch.moe.zero();
        self.a.scratch.message.zero();
        out.clear();
        let stream = self.stream;
        let mut count = EncodeCount::default();
        let mut round = 0;
        while round < paper.len().max(1) {
            let mut rec = stream.begin()?;
            let mut enc = rec.token_encoder()?;
            enc.probe_only(round);
            self.encode_visits(&mut enc, pos)?;
            if round == 0 {
                count = enc.count();
                let seen = enc.probe_log();
                if seen.len() != paper.len() {
                    return Err(refuse(format!(
                        "the walk encoded {} groups; the paper's token has {}",
                        seen.len(),
                        paper.len()
                    )));
                }
                for (i, (g, (kind, dispatches, visit))) in seen.iter().zip(&paper).enumerate() {
                    if Kind::from_tag(g.tag) != Some(*kind) || g.dispatches != *dispatches {
                        return Err(refuse(format!(
                            "group {i} (visit {visit:?}): the walk ran tag {} of {} dispatches; \
                             the paper has {} of {dispatches}",
                            g.tag,
                            g.dispatches,
                            kind.name()
                        )));
                    }
                }
            } else if enc.count() != count {
                return Err(refuse(format!(
                    "round {round} encoded {:?}; round 0 encoded {count:?}",
                    enc.count()
                )));
            }
            enc.end();
            let stamps = rec.stamps();
            let signalled = stream.submit_signalled(rec)?;
            stream.settle(signalled)?;
            let gpu_s = match stamps.interval(STAMP_WAIT)? {
                GpuInterval::Ran(s) => s,
                GpuInterval::StillRunning => {
                    return Err(refuse(format!(
                        "group {round}: its buffer still runs after its settle"
                    )));
                }
            };
            let (kind, dispatches, visit) = paper[round];
            let first = out
                .last()
                .is_none_or(|g| g.kind != kind || g.visit != visit);
            out.push(ProfileGroup {
                kind,
                visit,
                dispatches,
                gpu_s,
                bytes: bytes_of(self.c, kind, visit, pos, first),
            });
            round += 1;
        }
        let mut pick = [0u32];
        self.a
            .scratch
            .step_id
            .pages()
            .read_at(Elems(0), &mut pick)?;
        Ok((pick[0], count))
    }
}
