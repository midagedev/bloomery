//! A rank's side of a tensor-parallel load's sums ([`RankJoin`]): the outputs a rank computes as
//! its part of a sum over the ranks, carried to every other rank and added there in rank order
//! (`bloomery_gpu::host::Gathers::enqueue_sum`), so every rank holds the same bits.
//!
//! Three outputs of a V4.1 step are such parts: the attention sub-layer's after its output
//! projection (the rank's heads and output groups), the MoE sub-layer's after its combine (the
//! rank's part of every expert's inner width) and an engram site's `engram_wkv` over the rank's
//! columns of the site's rows. The trunk's two run on one slot, each a post, a wait and a sum in
//! turn. Each engram site has a slot of its own: its part depends on the token alone, so it is
//! posted early ([`RankJoin::post`]) and summed where the site's layer reads it
//! ([`RankJoin::sum`]); with a slot a site, a rank rewrites a site's set only after its sum of
//! the same site's other set, which waited for every other rank's part of it, which each rank
//! posted after its own sum of the set before. One row a step.

use std::sync::Arc;

use bloomery_gpu::host::{GatherSet, GatherShape, Gathers};
use bloomery_gpu::{Gpu, GpuError, MappedHost, WindowMut, mem_batch};
use cuda_core::{CudaContext, DeviceBuffer, sys};

const WHAT: &str = "deepseek41 RankJoin";

/// A slot of the page: the trunk's, or an engram site's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinSlot {
    Trunk,
    Site(usize),
}

/// One slot's operations: per set the post and the wait, the set the next post takes, and the set
/// posted and not yet summed.
struct SlotOps {
    width: usize,
    post: [Vec<sys::CUstreamBatchMemOpParams>; Gathers::SETS],
    wait: [Vec<sys::CUstreamBatchMemOpParams>; Gathers::SETS],
    next: usize,
    posted: Option<usize>,
}

/// One rank's page of a load's sums and the operations on it: see the module comment.
pub struct RankJoin {
    layout: Gathers,
    page: Arc<MappedHost>,
    rank: usize,
    slots: Vec<SlotOps>,
}

impl RankJoin {
    /// Rank `rank` of `ranks`' page in `ctx`: the trunk's slot of `n_embd` values and one slot of
    /// `site_width` values for each of `sites` engram sites. Fewer than two ranks, a rank past
    /// them and an empty width are refused by name ([`Gathers::new`]).
    pub fn new(
        ctx: &Arc<CudaContext>,
        (rank, ranks): (usize, usize),
        n_embd: usize,
        (sites, site_width): (usize, usize),
    ) -> Result<RankJoin, GpuError> {
        let widths: Vec<usize> = std::iter::once(n_embd)
            .chain(std::iter::repeat_n(site_width, sites))
            .collect();
        let shapes: Vec<GatherShape> = widths
            .iter()
            .map(|&width| GatherShape { rows: 1, width })
            .collect();
        let layout = Gathers::new(0, ranks, rank, &shapes)?;
        let page = MappedHost::new(ctx, layout.page_bytes(), "deepseek41 tp sum page")?;
        let dev = page.dev_at(0);
        let slots = widths
            .iter()
            .enumerate()
            .map(|(slot, &width)| {
                let at = |set| GatherSet { dev, slot, set };
                SlotOps {
                    width,
                    post: std::array::from_fn(|set| layout.post(at(set)).to_vec()),
                    wait: std::array::from_fn(|set| layout.wait(at(set))),
                    next: 0,
                    posted: None,
                }
            })
            .collect();
        Ok(RankJoin {
            layout,
            page: Arc::new(page),
            rank,
            slots,
        })
    }

    /// The page's layout and the page, for the link that joins it to the other ranks'.
    #[must_use]
    pub fn link(&self) -> (&Gathers, &Arc<MappedHost>) {
        (&self.layout, &self.page)
    }

    fn slot(&self, slot: JoinSlot) -> Result<usize, GpuError> {
        let i = match slot {
            JoinSlot::Trunk => 0,
            JoinSlot::Site(s) => s + 1,
        };
        if i < self.slots.len() {
            Ok(i)
        } else {
            Err(GpuError::shape(
                WHAT,
                format!(
                    "{slot:?}: the page has {} engram slots",
                    self.slots.len() - 1
                ),
            ))
        }
    }

    /// Enqueue the post of `part`, the rank's part of `slot`'s sum: the part copied into the
    /// rank's own place of the slot's next set, then that set's post. A part of another width, and
    /// a post while the slot's last one is not summed, are refused by name.
    pub fn post(
        &mut self,
        gpu: &Gpu,
        slot: JoinSlot,
        part: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let i = self.slot(slot)?;
        let rank = self.rank;
        let dev = self.page.dev_at(0);
        let s = &mut self.slots[i];
        if part.len() != s.width || s.posted.is_some() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{slot:?}: a part of {} values, posted {:?}; the slot takes {} and one post a \
                     sum",
                    part.len(),
                    s.posted,
                    s.width
                ),
            ));
        }
        let set = s.next;
        let own = dev + self.layout.part(i, set, rank).start as u64;
        // SAFETY: the part's values lie in the rank's own page, mapped in this context, which the
        // join holds until every launch it enqueued has run (its owner's drop).
        let mut place = unsafe { WindowMut::<f32>::from_raw(own, s.width, gpu.context()) };
        place.copy_from_device_async(part, gpu.stream())?;
        // SAFETY: every counter the operations name lies in the rank's page, held as above.
        unsafe { mem_batch(gpu.stream(), &mut s.post[set], "deepseek41 tp post")? };
        s.posted = Some(set);
        s.next = (set + 1) % Gathers::SETS;
        Ok(())
    }

    /// Enqueue the wait for every other rank's part of `slot`'s last post and the parts' sum in
    /// rank order into `out`. A slot with no post to sum, and an `out` of another width, are
    /// refused by name.
    pub fn sum(
        &mut self,
        gpu: &Gpu,
        slot: JoinSlot,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let i = self.slot(slot)?;
        let dev = self.page.dev_at(0);
        let s = &mut self.slots[i];
        let set = match s.posted {
            Some(set) if out.len() == s.width => set,
            posted => {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "{slot:?}: a sum into {} values, posted {posted:?}; the slot takes {}",
                        out.len(),
                        s.width
                    ),
                ));
            }
        };
        // SAFETY: every counter the operations name lies in the rank's page, held as above.
        unsafe { mem_batch(gpu.stream(), &mut s.wait[set], "deepseek41 tp wait")? };
        s.posted = None;
        let at = GatherSet { dev, slot: i, set };
        // SAFETY: `dev` is the rank's page of this layout, mapped in this context and held as
        // above.
        unsafe {
            self.layout
                .enqueue_sum(gpu.elem(), gpu.stream(), at, 1, out)
        }
    }

    /// [`RankJoin::post`] of `out` then [`RankJoin::sum`] into it.
    pub fn join(
        &mut self,
        gpu: &Gpu,
        slot: JoinSlot,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.post(gpu, slot, out)?;
        self.sum(gpu, slot, out)
    }
}
