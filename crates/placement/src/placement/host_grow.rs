//! The host K/V's growth from its floor to the cap, and the host experts that yield to it: the
//! common owner, pure. A plan that reads its K/V at a floor ([`Plan::kv_floor`]) whose host
//! rooms grow as positions arrive (the ring shadows, a host-backed room's pages:
//! `BLOOMERY_KV_HOST` other than `off`) charges the host for them at the floor, and proves the
//! cap reachable by the NVMe tier's split at the cap ([`expert_nvme_tier`]), refused by name as
//! any split is. The experts that split pages and the room between the floor and the cap holds
//! are kept on the host at load ([`restore_steps`]); as the K/V's host pages grow past what the
//! plan charged, they yield in the order the split pages them ([`HostGrow::due`]) — the id tail
//! of each routed layer, its coldest under the plan's rank, the layers kept at one share of
//! their host-served ids — so a load with every yield done holds the static plan at the cap.
//!
//! The plan's host need is the cap's whatever the floor: the experts kept at load count in it,
//! and the K/V's growth they will yield to does not, so the headroom a churn pool budgets from
//! is the static plan's. Nothing here touches a device or a mapping: the host residency
//! machine (`bloomery_gpu::host::residency`) drops the pages a yield names, and the K/V rooms'
//! fit (`bloomery_gpu::room::host_grow`) asks it to.
//!
//! [`expert_nvme_tier`]: super::expert_nvme_tier

use std::fmt;

/// One expert the load keeps on the host past the static plan at the cap and gives up to the
/// K/V's growth: its layer, its id and its host bytes over the layer's routed stacks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Yield {
    pub layer: usize,
    pub id: u32,
    pub bytes: u64,
}

/// What holds the room the host K/V grows into from its floor to the cap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GrowHold {
    /// No growth past a floor: the plan has no floor, or its K/V's host rooms are not rooms
    /// that grow (`BLOOMERY_KV_HOST=off`, where the ring shadows may be one page-locked
    /// allocation of the cap), or the model's loads yield nothing ([`super::KvBytes::host_yields`]);
    /// the K/V is charged at the cap.
    #[default]
    Cap,
    /// The room holds the K/V at the cap beside every host expert: the split paged nothing.
    Room,
    /// Host experts the split pages at the cap are kept at load and yield to the growth.
    Yields,
    /// The split's RAM arena serves the paged ids through a host segment the NVMe tier and the
    /// host leg fix at load, so no id can move to it later: the K/V is charged at the cap, and
    /// the plan prints the growth `static` for the reason `arena`.
    Arena,
}

impl GrowHold {
    /// The word a plan record prints.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            GrowHold::Cap => "cap",
            GrowHold::Room => "room",
            GrowHold::Yields => "yields",
            GrowHold::Arena => "static",
        }
    }

    /// Why a plan with a floor charges its host K/V at the cap all the same, the word a plan
    /// record prints beside `static`; `None` for a growth the plan holds.
    #[must_use]
    pub fn why_static(self) -> Option<&'static str> {
        match self {
            GrowHold::Arena => Some("arena"),
            GrowHold::Cap | GrowHold::Room | GrowHold::Yields => None,
        }
    }
}

/// A plan's host K/V from its floor to the cap ([`super::HostTotals::grow`]): the bytes charged
/// at the floor, the growth to the cap, and the experts kept on the host that yield to it, in
/// yield order. Empty ([`GrowHold::Cap`]) where the K/V is charged at the cap.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HostGrow {
    /// The positions the K/V is charged at on the host; 0 under [`GrowHold::Cap`].
    pub floor: u64,
    /// The host K/V bytes at the floor, every resident sequence's: the ring shadows' rows and
    /// a host-backed room's pages the floor reaches.
    pub floor_bytes: u64,
    /// The host K/V bytes from the floor to the cap, every resident sequence's: inside
    /// [`super::HostTotals::shadow_bytes`] and [`super::HostTotals::kv_host_bytes`], which
    /// stay the cap's.
    pub grow_bytes: u64,
    /// The experts the load keeps on the host that the static plan pages at the cap, the first
    /// to yield first; their bytes are in [`super::HostTotals::expert_bytes`].
    pub yields: Vec<Yield>,
    /// The bytes of [`HostGrow::yields`].
    pub yield_bytes: u64,
    pub hold: GrowHold,
}

/// How far a load's yields have gone: the count of [`HostGrow::yields`] given up, first to
/// last, and their bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct YieldCursor {
    pub done: usize,
    pub bytes: u64,
}

/// The yields a growth step asks for ([`HostGrow::due`]): `yields[from..to]` of
/// [`HostGrow::yields`], `bytes` of them, for a growth `shortfall` bytes past what the plan
/// charged and the yields done so far hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct YieldAsk {
    pub from: usize,
    pub to: usize,
    pub bytes: u64,
    pub shortfall: u64,
}

/// A growth no yield left can make room for ([`HostGrow::due`]): the host already holds `held`
/// bytes of K/V, past the `allowed` the plan charged and every yield holds, before a fit that
/// asks `asks` bytes more; `left` bytes of yields were not yet done.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "the host K/V holds {held} B, past the {allowed} B the plan charged and every yield holds \
     ({left} B of host experts were left to yield), before a fit that asks {asks} B more: the \
     K/V grows past the cap the plan proved"
)]
pub struct GrowShort {
    pub held: u64,
    pub asks: u64,
    pub allowed: u64,
    pub left: u64,
}

impl HostGrow {
    /// A growth charged at `floor` positions: `floor_bytes` there, `grow_bytes` more at the
    /// cap, no yield yet ([`GrowHold::Room`] until the split says otherwise).
    #[must_use]
    pub fn at_floor(floor: u64, floor_bytes: u64, grow_bytes: u64) -> HostGrow {
        HostGrow {
            floor,
            floor_bytes,
            grow_bytes,
            yields: Vec::new(),
            yield_bytes: 0,
            hold: GrowHold::Room,
        }
    }

    /// Whether the K/V is charged at a floor ([`GrowHold::Cap`] is not).
    #[must_use]
    pub fn armed(&self) -> bool {
        self.hold != GrowHold::Cap
    }

    /// Pages of a host-backed room counted at the cap (`cap`) and at the floor (`floor`): the
    /// floor's into the charge, the rest into the growth. Nothing on a growth not armed.
    pub fn add_pages(&mut self, cap: u64, floor: u64) {
        if self.armed() {
            self.floor_bytes = self.floor_bytes.saturating_add(floor);
            self.grow_bytes = self.grow_bytes.saturating_add(cap.saturating_sub(floor));
        }
    }

    /// The host K/V bytes the plan's host need counts: the floor's and the growth the yields
    /// do not cover.
    #[must_use]
    pub fn charged(&self) -> u64 {
        self.floor_bytes
            .saturating_add(self.grow_bytes.saturating_sub(self.yield_bytes))
    }

    /// The host K/V bytes at the cap.
    #[must_use]
    pub fn cap_bytes(&self) -> u64 {
        self.floor_bytes.saturating_add(self.grow_bytes)
    }

    /// The yields a fit asks for that backs `asks` bytes of new host pages beside the `held`
    /// bytes of host K/V the process holds, with `at` yields done: none for a fit that backs
    /// no new host page, or while `held + asks` is within what the plan charged and the done
    /// yields hold; else the fewest next yields whose bytes reach the shortfall plus `page`
    /// (the room's next page, so the next fit does not ask at once), or every one left when
    /// they do not.
    ///
    /// Refused by name ([`GrowShort`]) only when the bytes the host already holds pass what
    /// the plan charged and every yield holds: a fit's asks count every new page a host-capable
    /// room may back, and a spill room's lands on the card while it has room, and a room's
    /// last page rounds the plan's exact rows up, so a fit those leave past the books goes on
    /// as the static plan's would.
    pub fn due(
        &self,
        (held, asks): (u64, u64),
        at: YieldCursor,
        page: u64,
    ) -> Result<Option<YieldAsk>, GrowShort> {
        let allowed = self.charged().saturating_add(at.bytes);
        let backed = held.saturating_add(asks);
        let Some(shortfall) = backed.checked_sub(allowed).filter(|&s| s > 0 && asks > 0) else {
            return Ok(None);
        };
        let want = shortfall.saturating_add(page);
        let rest = self.yields.get(at.done..).unwrap_or(&[]);
        let mut bytes = 0u64;
        let mut to = at.done;
        for y in rest {
            if bytes >= want {
                break;
            }
            bytes = bytes.saturating_add(y.bytes);
            to += 1;
        }
        if bytes < shortfall && held > allowed.saturating_add(bytes) {
            return Err(GrowShort {
                held,
                asks,
                allowed: allowed.saturating_add(bytes),
                left: bytes,
            });
        }
        Ok((to > at.done).then_some(YieldAsk {
            from: at.done,
            to,
            bytes,
            shortfall,
        }))
    }
}

impl fmt::Display for HostGrow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(why) = self.hold.why_static() {
            return write!(
                f,
                "host K/V {} B charged at the cap past its floor of {} positions: growth static \
                 ({why})",
                self.cap_bytes(),
                self.floor
            );
        }
        write!(
            f,
            "host K/V {} B at the floor of {} positions, {} B more to the cap: {} host experts of \
             {} B yield to it ({})",
            self.floor_bytes,
            self.floor,
            self.grow_bytes,
            self.yields.len(),
            self.yield_bytes,
            self.hold.name()
        )
    }
}

/// One routed layer the split paged as [`restore_steps`] reads it: the ids it serves on the
/// host leg (`served`, resident and paged), the ones the split keeps resident (`kept`), the
/// paged ones (`paged`, at most `served − kept`) and one expert's bytes over the layer's routed
/// stacks (`unit`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrowLayer {
    pub served: u64,
    pub kept: u64,
    pub paged: u64,
    pub unit: u64,
}

/// The order the load keeps paged experts on the host in, within `budget` bytes, one layer's
/// index into `layers` a step: each step takes the next paged expert of the layer whose kept
/// share of its served ids is the least (the lower index among equal shares), skipping a
/// layer with none left or whose expert passes what the budget leaves, until none fits. The
/// layers stay at one share of their host-served ids, as the split's own proportional keep
/// holds them; the reverse is the yield order. Each layer's steps take its paged ids from the
/// one the split would keep first as the room grows.
#[must_use]
pub fn restore_steps(layers: &[GrowLayer], budget: u64) -> Vec<usize> {
    let mut taken = vec![0u64; layers.len()];
    let mut left = budget;
    let mut steps = Vec::new();
    loop {
        let mut best: Option<usize> = None;
        for (k, l) in layers.iter().enumerate() {
            if taken[k] >= l.paged || l.unit > left || l.served == 0 {
                continue;
            }
            let less = best.is_none_or(|b| {
                let (kb, lb) = (layers[b].kept + taken[b], layers[b].served);
                u128::from(l.kept + taken[k]) * u128::from(lb)
                    < u128::from(kb) * u128::from(l.served)
            });
            if less {
                best = Some(k);
            }
        }
        let Some(k) = best else {
            return steps;
        };
        taken[k] += 1;
        left -= layers[k].unit;
        steps.push(k);
    }
}

/// The yields of the restores `steps` (each an index into `paged`) over each layer's paged ids
/// `paged` (`(layer, its paged ids in the order a larger room keeps them, one expert's bytes)`),
/// with how many of each layer's paged ids the restores keep: the last kept yields first, so a
/// layer gives up the back of what it kept before the front, and every yield done is the split
/// the restores started from. A step past its layer's paged ids is a caller's defect, refused
/// by name.
#[must_use]
pub fn yields_of(steps: &[usize], paged: &[(usize, &[u32], u64)]) -> (Vec<Yield>, Vec<usize>) {
    let mut taken = vec![0usize; paged.len()];
    let mut yields = Vec::with_capacity(steps.len());
    for &k in steps {
        let (layer, ids, bytes) = paged[k];
        let id = *ids.get(taken[k]).unwrap_or_else(|| {
            panic!(
                "host_grow::yields_of: restore {} of layer {layer}, which pages {}",
                taken[k] + 1,
                ids.len()
            )
        });
        taken[k] += 1;
        yields.push(Yield { layer, id, bytes });
    }
    yields.reverse();
    (yields, taken)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The yields run the restores backwards: the last kept first, each layer's later paged id
    /// before its earlier one, and the restores' counts per layer. FAIL-first: the yields kept
    /// in restore order give layer 3's id 5 first.
    #[test]
    fn the_yields_are_the_restores_backwards() {
        let (l3, l4): (&[u32], &[u32]) = (&[5, 6, 7], &[2, 3]);
        let (yields, taken) = yields_of(&[0, 1, 0], &[(3, l3, 10), (4, l4, 20)]);
        let got: Vec<(usize, u32, u64)> = yields.iter().map(|y| (y.layer, y.id, y.bytes)).collect();
        assert_eq!(got, [(3, 6, 10), (4, 2, 20), (3, 5, 10)]);
        assert_eq!(taken, [2, 1]);
    }

    fn grow(charged_floor: u64, grow_bytes: u64, units: &[u64]) -> HostGrow {
        let yields: Vec<Yield> = units
            .iter()
            .enumerate()
            .map(|(i, &bytes)| Yield {
                layer: 0,
                id: 100 - i as u32,
                bytes,
            })
            .collect();
        HostGrow {
            floor: 8,
            floor_bytes: charged_floor,
            grow_bytes,
            yield_bytes: units.iter().sum(),
            yields,
            hold: GrowHold::Yields,
        }
    }

    /// Within what the plan charged nothing yields; past it the fewest next yields reach the
    /// shortfall plus a page, in list order; the cursor carries the bytes already yielded.
    /// FAIL-first: a `due` that asks for the shortfall alone yields one expert at 1,400 B, not
    /// two.
    #[test]
    fn a_growth_step_yields_the_shortfall_plus_a_page_in_order() {
        // 1,000 B at the floor, 1,000 B of growth of which 900 B are three yields of 300 B:
        // the plan charges 1,100 B. The host holds 1,000 B; each fit asks the rest.
        let g = grow(1_000, 1_000, &[300, 300, 300]);
        assert_eq!(g.charged(), 1_100);
        let start = YieldCursor::default();
        assert_eq!(
            g.due((1_000, 100), start, 50),
            Ok(None),
            "within the charge"
        );
        // 1,150 B backed: 50 B short, 100 B wanted with the page: one yield of 300 B.
        assert_eq!(
            g.due((1_000, 150), start, 50),
            Ok(Some(YieldAsk {
                from: 0,
                to: 1,
                bytes: 300,
                shortfall: 50
            }))
        );
        // 1,400 B backed: 300 B short, 350 B wanted: two yields.
        assert_eq!(
            g.due((1_000, 400), start, 50),
            Ok(Some(YieldAsk {
                from: 0,
                to: 2,
                bytes: 600,
                shortfall: 300
            }))
        );
        // One yield done holds 300 B more: 1,400 B is 0 B short of 1,400 B.
        let one = YieldCursor {
            done: 1,
            bytes: 300,
        };
        assert_eq!(g.due((1_000, 400), one, 50), Ok(None));
        assert_eq!(
            g.due((1_000, 401), one, 50)
                .map(|a| a.map(|a| (a.from, a.to))),
            Ok(Some((1, 2)))
        );
    }

    /// At the cap every yield is done and nothing is short; the last yields are taken short
    /// of the page when the shortfall alone fits them, and every one left when they do not
    /// reach it; a fit whose asks pass the books while the host holds no byte past them goes
    /// on (a spill page may land on the card, a room's last page rounds up), a fit that backs
    /// no new host page never yields, and only a host that already holds past every yield is
    /// refused by name with what was left. FAIL-first: refusing on `held + asks` refuses the
    /// fit at (cap, 1) with every yield done; dropping the no-asks clause yields nothing but
    /// refuses the decode step at (cap + 1, 0).
    #[test]
    fn the_cap_is_reached_with_every_yield_and_past_it_is_refused() {
        let g = grow(1_000, 1_000, &[300, 300, 300]);
        let cap = g.cap_bytes();
        assert_eq!(cap, 2_000);
        let two = YieldCursor {
            done: 2,
            bytes: 600,
        };
        assert_eq!(
            g.due((cap - 100, 100), two, 500),
            Ok(Some(YieldAsk {
                from: 2,
                to: 3,
                bytes: 300,
                shortfall: 300
            })),
            "the last yield, short of the page"
        );
        assert_eq!(
            g.due((1_900, 200), two, 500),
            Ok(Some(YieldAsk {
                from: 2,
                to: 3,
                bytes: 300,
                shortfall: 400
            })),
            "every yield left, short of the shortfall, the host within the books with them"
        );
        let all = YieldCursor {
            done: 3,
            bytes: 900,
        };
        assert_eq!(
            g.due((cap - 100, 100), all, 500),
            Ok(None),
            "the cap holds with every yield"
        );
        assert_eq!(
            g.due((cap, 1), all, 500),
            Ok(None),
            "asks past the books, the host at them"
        );
        assert_eq!(
            g.due((cap + 1, 0), all, 500),
            Ok(None),
            "a fit that backs nothing"
        );
        let past = g.due((cap + 1, 1), all, 500).expect_err("past the cap");
        assert_eq!(
            past,
            GrowShort {
                held: cap + 1,
                asks: 1,
                allowed: cap,
                left: 0
            }
        );
        assert!(past.to_string().contains("past the cap the plan proved"));
        assert_eq!(
            g.due((cap + 1, 1), two, 500),
            Err(GrowShort {
                held: cap + 1,
                asks: 1,
                allowed: cap,
                left: 300
            }),
            "short with the yields left: none is made"
        );
    }

    /// A growth charged at the cap for an arena prints `static` and says why, and charges the
    /// cap with no yield; a held growth names no reason. FAIL-first: `Arena` named `arena`
    /// prints `grow=arena`, and with no reason.
    #[test]
    fn an_arena_growth_is_static_and_says_why() {
        let mut g = HostGrow::at_floor(8, 1_000, 1_000);
        g.hold = GrowHold::Arena;
        assert!(g.armed());
        assert_eq!(
            (g.hold.name(), g.hold.why_static()),
            ("static", Some("arena"))
        );
        assert_eq!(g.charged(), g.cap_bytes(), "the cap's charge");
        assert_eq!(
            g.due((1_999, 1), YieldCursor::default(), 50),
            Ok(None),
            "within the cap: nothing to yield"
        );
        assert!(g.to_string().contains("growth static (arena)"), "{g}");
        for held in [GrowHold::Cap, GrowHold::Room, GrowHold::Yields] {
            assert_eq!(held.why_static(), None, "{held:?}");
        }
    }

    /// The steps keep the layers at one share of their served ids: a layer that keeps less
    /// takes first, equal shares go to the lower index, a layer is done at its paged count and
    /// one whose expert passes the budget left is skipped. FAIL-first: picking the least kept
    /// count instead of the least share sends the first step to layer 0.
    #[test]
    fn restores_keep_the_layers_at_one_share() {
        let layers = [
            GrowLayer {
                served: 10,
                kept: 4,
                paged: 6,
                unit: 10,
            },
            GrowLayer {
                served: 20,
                kept: 6,
                paged: 14,
                unit: 10,
            },
        ];
        // 0.4 against 0.3: layer 1 first, then 0.4 against 0.35, 0.4 against 0.4 (layer 0, the
        // lower index), ...
        let steps = restore_steps(&layers, 60);
        assert_eq!(steps, vec![1, 1, 0, 1, 1, 0]);
        // Every paged expert fits: the steps end at the paged counts.
        let all = restore_steps(&layers, u64::MAX);
        assert_eq!(all.iter().filter(|&&k| k == 0).count(), 6);
        assert_eq!(all.iter().filter(|&&k| k == 1).count(), 14);
        // A layer whose expert passes the budget left is skipped for one that fits.
        let wide = [
            GrowLayer {
                served: 10,
                kept: 0,
                paged: 10,
                unit: 25,
            },
            GrowLayer {
                served: 10,
                kept: 5,
                paged: 5,
                unit: 10,
            },
        ];
        assert_eq!(restore_steps(&wide, 35), vec![0, 1]);
    }
}
