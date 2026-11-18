//! The body's rooms: every sparse layer's cache is one ([`bloomery_gpu::RoomTensor`]), backed page
//! by page as far as the sequence reaches, on the card or, under the plan's `BLOOMERY_KV_HOST`
//! word, on the host. The plan counts them at its floor ([`model::arch::hy4::place`]); the body
//! fits them through the rows a launch writes ([`Body::fit`]) before the launch runs, never inside
//! a capture, and a reset gives the pages back.
//!
//! What is no room stays flat and counted at the context: the pooled keys, their scores, the index
//! ring, the list and the window layers' rings. The body holds no residency machine, so a fit that
//! runs out of card bytes is the room's named refusal ([`bloomery_gpu::fit_rooms`]), and no page
//! is shared between two sequences' stores.

use bloomery_gpu::{GpuError, KvRoomBody, RoomTensor, fit_rooms};

use crate::body::{Body, Store, WHAT};

impl Body {
    /// Every sparse layer's cache room, layer by layer.
    fn rooms_mut(&mut self) -> impl Iterator<Item = &mut RoomTensor<u16>> {
        self.stores.iter_mut().filter_map(|store| match store {
            Store::Dsa(d) => Some(&mut d.cache),
            Store::Swa { .. } => None,
        })
    }

    /// The rooms fitted through `pos + rows` positions before a launch writes rows from `pos`
    /// ([`fit_rooms`]). Never inside a capture.
    pub(crate) fn fit(&mut self, pos: usize, rows: usize) -> Result<(), GpuError> {
        fit_rooms(self, pos, rows)
    }

    /// Every cache room backed through `positions` positions at once: the pages a flat store of
    /// that many positions would hold, which a gate's baseline reads to prove the lazily fitted
    /// rooms hold the same rows. Never inside a capture.
    ///
    /// # Errors
    ///
    /// The room's named refusal ([`GpuError::NoRoom`]).
    pub fn fit_through(&mut self, positions: usize) -> Result<(), GpuError> {
        self.fit(0, positions)
    }

    /// The bytes the cache rooms back with memory of their own, on the card and on the host: what
    /// the rows the live sequence reached hold, by pages.
    #[must_use]
    pub fn room_mapped(&self) -> (usize, usize) {
        self.stores
            .iter()
            .filter_map(|store| match store {
                Store::Dsa(d) => Some(d.cache.mapped()),
                Store::Swa { .. } => None,
            })
            .fold((0, 0), |(c, h), (rc, rh)| (c + rc, h + rh))
    }

    /// Every cache room's pages given back, the sequence's end: after the device has finished
    /// every call that could read one.
    pub(crate) fn settle_rooms(&mut self) -> Result<(), GpuError> {
        self.rooms_mut().try_for_each(RoomTensor::settle)
    }
}

/// The rooms [`fit_rooms`] runs: every sparse layer's cache. The body holds no residency machine
/// ([`KvRoomBody::free_expert_bytes`]'s default refuses by name) and shares no page between two
/// sequences, so a fit only backs pages.
impl KvRoomBody for Body {
    const WHAT: &'static str = WHAT;

    fn fit_each(&mut self, _pos: usize, through: usize) -> Result<(), GpuError> {
        self.rooms_mut().try_for_each(|room| room.fit(through))
    }

    fn asks(&self, through: usize) -> usize {
        self.stores
            .iter()
            .map(|store| match store {
                Store::Dsa(d) => d.cache.asks(through),
                Store::Swa { .. } => 0,
            })
            .sum()
    }
}
