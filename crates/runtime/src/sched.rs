//! The schedules' overlap: which of a layer program's parts runs for which
//! unit and layer, in what order, and through which port. One type,
//! [`Overlap`], names every schedule's point — the one-token step, the pair
//! pass, the prompt call's group of batches — and [`order`] is its sequence,
//! a pure function of the point and the layer count. [`walk`] runs that
//! sequence over a [`LayerProgram`] and its [`Port`]; the program enqueues,
//! the port exchanges with the host, and neither decides an order.
//! [`slot_lanes`] is the point a pass of resident slots takes, and which
//! slots ride which lane: no body chooses its lanes itself.
//!
//! A unit is what runs one layer at a time through the chain: a row of the
//! step or of the pair pass, a batch of a prompt group. A layer's program
//! has up to three parts:
//!
//! - the front, up to the handoff of its host leg (the whole layer when it
//!   has none);
//! - the shadow, card work that runs while the host serves the leg;
//! - the back, from the join on.
//!
//! The two ports differ in who serves a leg and so in where the back sits:
//!
//! - [`PortKind::Step`]: the leg is served by the host pool reacting to the
//!   card's go, and the back is a wait on the stream. Round `r`, per unit:
//!   the back of layer `r − 1`, then the front and the shadow of layer `r`;
//!   every unit's begin first, each unit's end right after its last back. One
//!   unit is the step, two are the pair pass: rows one layer apart, so the
//!   host serves one row's layer while the card runs the other's.
//! - [`PortKind::Batch`]: the calling thread serves each leg ([`Item::Serve`])
//!   and returns once its sums are on their way back. Items `x = (layer,
//!   unit)` layer by layer: the front of item 0, then per item its shadow, its
//!   serve and its back, with the next item's front enqueued ahead of the
//!   serve when the group holds two units or more (the next item is another
//!   unit's, so nothing it reads waits on this serve) and after the back
//!   otherwise (it reads this back's output). A batch has no end: the call's
//!   head follows its last group.
//!
//! Either way, while the host serves item `x` the card holds `x`'s shadow
//! ahead of `x`'s back.

use std::fmt;

/// How a schedule hands a layer's host leg over and takes it back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortKind {
    /// Inside the captured chain: the card raises a mapped word, the host
    /// pool serves on seeing it, the card's back waits on the answer.
    Step,
    /// Eager: the calling thread serves each leg once the card's copies
    /// landed, and the card's back follows the upload in stream order.
    Batch,
}

/// One schedule's point: `units` units of `cols` columns each, exchanged
/// through a port of kind `port`. The one-token step is `(1, 1, Step)`, the
/// pair pass `(2, 1, Step)`, a prompt group of `g` batches of `T` positions
/// `(g, T, Batch)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Overlap {
    pub units: usize,
    pub cols: usize,
    pub port: PortKind,
}

/// A unit and a layer, both counted from 0 within the walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct At {
    pub unit: usize,
    pub layer: usize,
}

/// One entry of a walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Item {
    /// The unit's start, before its first layer.
    Begin(usize),
    Front(At),
    Shadow(At),
    /// The port serves the host leg ([`PortKind::Batch`] only).
    Serve(At),
    Back(At),
    /// The unit's end, after its last layer ([`PortKind::Step`] only).
    End(usize),
}

/// The walk of `o` over `layers` layers, in enqueue order (the module
/// comment's rule). Allocation-free: each entry is computed from its index.
pub fn order(o: Overlap, layers: usize) -> impl Iterator<Item = Item> {
    (0..).map_while(move |k| nth(o, layers, k))
}

/// Entry `k` of [`order`], `None` past the last.
fn nth(o: Overlap, layers: usize, k: usize) -> Option<Item> {
    let u = o.units;
    if k < u {
        return Some(Item::Begin(k));
    }
    let k = k - u;
    match o.port {
        PortKind::Step => step_nth(u, layers, k),
        PortKind::Batch => batch_nth(u, layers, k),
    }
}

/// Entry `k` after the begins of a step walk. Per unit, round 0 holds the
/// front and the shadow of layer 0, rounds `1 .. layers` the back of the layer
/// before and the front and the shadow of their own, round `layers` the back
/// of the last layer and the end.
fn step_nth(u: usize, layers: usize, k: usize) -> Option<Item> {
    use Item::{Back, End, Front, Shadow};
    let at = |unit, layer| At { unit, layer };
    if layers == 0 {
        return (k < u).then_some(End(k));
    }
    if k < 2 * u {
        let a = at(k / 2, 0);
        return Some(if k.is_multiple_of(2) {
            Front(a)
        } else {
            Shadow(a)
        });
    }
    let k = k - 2 * u;
    let middle = 3 * u * (layers - 1);
    if k < middle {
        let (r, j) = (1 + k / (3 * u), k % (3 * u));
        let unit = j / 3;
        return Some(match j % 3 {
            0 => Back(at(unit, r - 1)),
            1 => Front(at(unit, r)),
            _ => Shadow(at(unit, r)),
        });
    }
    let k = k - middle;
    (k < 2 * u).then(|| {
        let unit = k / 2;
        if k.is_multiple_of(2) {
            Back(at(unit, layers - 1))
        } else {
            End(unit)
        }
    })
}

/// Entry `k` after the begins of a batch walk: the front of item 0, then four
/// entries per item — three for the last, which has no next front.
fn batch_nth(u: usize, layers: usize, k: usize) -> Option<Item> {
    use Item::{Back, Front, Serve, Shadow};
    let n = u * layers;
    let at = |x: usize| At {
        unit: x % u,
        layer: x / u,
    };
    if n == 0 {
        return None;
    }
    if k == 0 {
        return Some(Front(at(0)));
    }
    let (x, j) = ((k - 1) / 4, (k - 1) % 4);
    if x >= n {
        return None;
    }
    let next = (x + 1 < n).then(|| Front(at(x + 1)));
    let a = at(x);
    let ahead = u >= 2;
    let items = if ahead {
        [Some(Shadow(a)), next, Some(Serve(a)), Some(Back(a))]
    } else {
        [Some(Shadow(a)), Some(Serve(a)), Some(Back(a)), next]
    };
    items.into_iter().flatten().nth(j)
}

/// Why a walk did not start, or a pass of resident slots has no point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refused {
    /// The overlap names one port kind and the port is another.
    Port { overlap: PortKind, port: PortKind },
    /// An overlap of no unit or no column.
    Empty { units: usize, cols: usize },
    /// A serve reached a port whose legs the host serves on its own.
    Serve(At),
    /// A pass of resident slots with no slot in it ([`slot_lanes`]).
    NoSlots,
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::Port { overlap, port } => {
                write!(
                    f,
                    "an overlap through a {overlap:?} port walked on a {port:?} port"
                )
            }
            Refused::Empty { units, cols } => {
                write!(
                    f,
                    "an overlap of {units} units of {cols} columns: nothing to walk"
                )
            }
            Refused::Serve(at) => write!(
                f,
                "a serve of unit {} layer {} on a port whose host serves its legs on its own",
                at.unit, at.layer
            ),
            Refused::NoSlots => write!(f, "a pass of 0 resident slots: no lane to lay out"),
        }
    }
}

/// A schedule's exchange with the host tier.
pub trait Port {
    type Error;

    /// The schedule this port is the exchange of.
    const KIND: PortKind;

    /// A walk of `o` begins: the port refuses by name a point it cannot lay
    /// out.
    fn open(&mut self, o: Overlap) -> Result<(), Self::Error>;

    /// Serve `at`'s host leg; [`PortKind::Batch`] only.
    fn serve(&mut self, at: At) -> Result<(), Self::Error> {
        Err(Self::refused(Refused::Serve(at)))
    }

    /// `why` as this port's error.
    fn refused(why: Refused) -> Self::Error;
}

/// A walk's error: its program's port's.
pub type Error<P> = <<P as LayerProgram>::Port as Port>::Error;

/// A layer kind's program over the units of a walk: each part enqueues its
/// launches and nothing else — no synchronization, no allocation, no host
/// round trip beyond what its port does — so the same parts run eager and
/// under capture.
pub trait LayerProgram {
    type Port: Port;

    /// Unit `unit`'s start, before its first layer.
    fn begin(&mut self, unit: usize) -> Result<(), Error<Self>> {
        let _ = unit;
        Ok(())
    }

    /// Whether `at`'s layer has a host leg: the walk serves only those. A
    /// layer with none (a dense layer ahead of the routed run) downloads
    /// nothing, so a serve of it would meet another layer's exchange.
    fn host_leg(&self, at: At) -> bool {
        let _ = at;
        true
    }

    /// `at`'s front: the whole layer when it has no host leg.
    fn front(&mut self, port: &mut Self::Port, at: At) -> Result<(), Error<Self>>;

    /// `at`'s card work under its host leg.
    fn shadow(&mut self, port: &mut Self::Port, at: At) -> Result<(), Error<Self>> {
        let _ = (port, at);
        Ok(())
    }

    /// `at`'s back, from the join on.
    fn back(&mut self, port: &mut Self::Port, at: At) -> Result<(), Error<Self>> {
        let _ = (port, at);
        Ok(())
    }

    /// Unit `unit`'s end, after its last layer.
    fn end(&mut self, unit: usize) -> Result<(), Error<Self>> {
        let _ = unit;
        Ok(())
    }
}

/// Run `prog` over `layers` layers in the order of `o` ([`order`]), `port`
/// opened first and serving the [`Item::Serve`] entries of the layers with a
/// host leg ([`LayerProgram::host_leg`]).
pub fn walk<P: LayerProgram>(
    o: Overlap,
    layers: usize,
    port: &mut P::Port,
    prog: &mut P,
) -> Result<(), Error<P>> {
    let kind = <P::Port as Port>::KIND;
    if o.port != kind {
        return Err(<P::Port as Port>::refused(Refused::Port {
            overlap: o.port,
            port: kind,
        }));
    }
    if o.units == 0 || o.cols == 0 {
        return Err(<P::Port as Port>::refused(Refused::Empty {
            units: o.units,
            cols: o.cols,
        }));
    }
    port.open(o)?;
    for item in order(o, layers) {
        match item {
            Item::Begin(u) => prog.begin(u)?,
            Item::Front(at) => prog.front(port, at)?,
            Item::Shadow(at) => prog.shadow(port, at)?,
            Item::Serve(at) if prog.host_leg(at) => port.serve(at)?,
            Item::Serve(_) => {}
            Item::Back(at) => prog.back(port, at)?,
            Item::End(u) => prog.end(u)?,
        }
    }
    Ok(())
}

/// The point a pass of resident slots takes ([`slot_lanes`]): its overlap,
/// and the busy slots lane 0 takes, the first `lane0` in slot order; lane 1
/// takes the rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lanes {
    pub overlap: Overlap,
    pub lane0: usize,
}

/// The point a pass of `slots` resident slots takes on a step port: the
/// step for one slot; else two lanes one layer apart (the host serves one
/// lane's layer while the card runs the other's), lane 0 the first
/// ⌈slots/2⌉ slots, lane 1 the rest. Refused by name at 0.
///
/// The overlap's `cols` is lane 0's width: for an odd count lane 1 holds one
/// slot fewer. From three slots on the point is two lanes of two columns or
/// more: a step port that serves one column a row refuses it.
pub fn slot_lanes(slots: usize) -> Result<Lanes, Refused> {
    if slots == 0 {
        return Err(Refused::NoSlots);
    }
    let lane0 = slots.div_ceil(2);
    Ok(Lanes {
        overlap: Overlap {
            units: slots.min(2),
            cols: lane0,
            port: PortKind::Step,
        },
        lane0,
    })
}

#[cfg(test)]
mod tests {
    use super::Item::{Back, Begin, End, Front, Serve, Shadow};
    use super::*;

    fn at(unit: usize, layer: usize) -> At {
        At { unit, layer }
    }

    fn step(units: usize) -> Overlap {
        Overlap {
            units,
            cols: 1,
            port: PortKind::Step,
        }
    }

    fn group(g: usize) -> Overlap {
        Overlap {
            units: g,
            cols: 512,
            port: PortKind::Batch,
        }
    }

    fn walked(o: Overlap, layers: usize) -> Vec<Item> {
        order(o, layers).collect()
    }

    /// The one-token step as nested loops, the shape V4.1's chain body
    /// enqueued it in before [`walk`]: the row's start, then per layer its
    /// front (the engram step, the attention, the MoE sub-layer up to its go),
    /// its shadow and its back (the wait and the join), then the head.
    fn today_step(layers: usize) -> Vec<Item> {
        let mut v = vec![Begin(0)];
        for i in 0..layers {
            v.extend([Front(at(0, i)), Shadow(at(0, i)), Back(at(0, i))]);
        }
        v.push(End(0));
        v
    }

    /// The pair pass as nested loops, over `rows` rows: every row's start,
    /// then per layer and row the back of the layer before, the front and the
    /// shadow; last, per row, its last back and its head. V4.1 runs two rows;
    /// more are the same rule.
    fn today_pair(layers: usize, rows: usize) -> Vec<Item> {
        let mut v: Vec<Item> = (0..rows).map(Begin).collect();
        let mut last = None;
        for i in 0..layers {
            for row in 0..rows {
                if let Some(j) = last {
                    v.push(Back(at(row, j)));
                }
                v.extend([Front(at(row, i)), Shadow(at(row, i))]);
            }
            last = Some(i);
        }
        let j = last.expect("the pair pass refuses a chain of no layer");
        for row in 0..rows {
            v.extend([Back(at(row, j)), End(row)]);
        }
        v
    }

    /// A prompt group as nested loops, the shape V4.1's prompt call enqueues
    /// it in, over `g` batches: every batch's start, the route of the first
    /// layer-batch (its front), then per layer-batch its shadow, the next
    /// route when `ahead`, the serve, the post (its back), and the next route
    /// when not.
    fn today_group(layers: usize, g: usize) -> Vec<Item> {
        let mut v: Vec<Item> = (0..g).map(Begin).collect();
        let items: Vec<(usize, usize)> = (0..layers)
            .flat_map(|i| (0..g).map(move |j| (i, j)))
            .collect();
        let route = |&(i, j): &(usize, usize)| Front(at(j, i));
        let ahead = g >= 2;
        if let Some(x0) = items.first() {
            v.push(route(x0));
        }
        for (x, &(i, j)) in items.iter().enumerate() {
            v.push(Shadow(at(j, i)));
            if ahead && let Some(n) = items.get(x + 1) {
                v.push(route(n));
            }
            v.push(Serve(at(j, i)));
            v.push(Back(at(j, i)));
            if !ahead && let Some(n) = items.get(x + 1) {
                v.push(route(n));
            }
        }
        v
    }

    /// T-a on two layers, written out: the four walks today's code runs.
    #[test]
    fn the_four_walks_on_two_layers() {
        let (a, b) = (at(0, 0), at(0, 1));
        assert_eq!(
            walked(step(1), 2),
            [
                Begin(0),
                Front(a),
                Shadow(a),
                Back(a),
                Front(b),
                Shadow(b),
                Back(b),
                End(0)
            ]
        );
        let (a0, a1, b0, b1) = (at(0, 0), at(1, 0), at(0, 1), at(1, 1));
        assert_eq!(
            walked(step(2), 2),
            [
                Begin(0),
                Begin(1),
                Front(a0),
                Shadow(a0),
                Front(a1),
                Shadow(a1),
                Back(a0),
                Front(b0),
                Shadow(b0),
                Back(a1),
                Front(b1),
                Shadow(b1),
                Back(b0),
                End(0),
                Back(b1),
                End(1)
            ]
        );
        assert_eq!(
            walked(group(1), 2),
            [
                Begin(0),
                Front(a),
                Shadow(a),
                Serve(a),
                Back(a),
                Front(b),
                Shadow(b),
                Serve(b),
                Back(b)
            ]
        );
        assert_eq!(
            walked(group(2), 2),
            [
                Begin(0),
                Begin(1),
                Front(a0),
                Shadow(a0),
                Front(a1),
                Serve(a0),
                Back(a0),
                Shadow(a1),
                Front(b0),
                Serve(a1),
                Back(a1),
                Shadow(b0),
                Front(b1),
                Serve(b0),
                Back(b0),
                Shadow(b1),
                Serve(b1),
                Back(b1)
            ]
        );
    }

    /// T-a over every shape: the transcriptions of today's loops against
    /// [`order`], for V4.1's 40 layers and the ones around it, the step, the
    /// pair and wider row groups, and prompt groups up to `GROUP_MAX`.
    #[test]
    fn order_is_todays_walks() {
        for layers in 0..=43 {
            assert_eq!(
                walked(step(1), layers),
                today_step(layers),
                "step, {layers} layers"
            );
            for g in 1..=8 {
                assert_eq!(
                    walked(group(g), layers),
                    today_group(layers, g),
                    "group of {g}, {layers} layers"
                );
                if layers > 0 {
                    assert_eq!(
                        walked(step(g), layers),
                        today_pair(layers, g),
                        "{g} rows, {layers} layers"
                    );
                }
            }
        }
    }

    /// A program and a port that record what the walk asks of them.
    #[derive(Default)]
    struct Rec {
        seen: Vec<Item>,
        opened: Option<Overlap>,
    }

    struct RecPort<const B: bool>(Rec);

    impl<const B: bool> Port for RecPort<B> {
        type Error = Refused;
        const KIND: PortKind = if B { PortKind::Batch } else { PortKind::Step };

        fn open(&mut self, o: Overlap) -> Result<(), Refused> {
            self.0.opened = Some(o);
            Ok(())
        }

        fn serve(&mut self, at: At) -> Result<(), Refused> {
            if B {
                self.0.seen.push(Serve(at));
                Ok(())
            } else {
                Err(Refused::Serve(at))
            }
        }

        fn refused(why: Refused) -> Refused {
            why
        }
    }

    struct Prog<const B: bool>;

    impl<const B: bool> LayerProgram for Prog<B> {
        type Port = RecPort<B>;

        fn begin(&mut self, unit: usize) -> Result<(), Refused> {
            let _ = unit;
            Ok(())
        }

        fn front(&mut self, port: &mut RecPort<B>, at: At) -> Result<(), Refused> {
            port.0.seen.push(Front(at));
            Ok(())
        }

        fn shadow(&mut self, port: &mut RecPort<B>, at: At) -> Result<(), Refused> {
            port.0.seen.push(Shadow(at));
            Ok(())
        }

        fn back(&mut self, port: &mut RecPort<B>, at: At) -> Result<(), Refused> {
            port.0.seen.push(Back(at));
            Ok(())
        }
    }

    /// The walk calls each part in [`order`]'s order and opens the port with
    /// the overlap first.
    #[test]
    fn walk_runs_the_order() {
        let layers_parts = |v: Vec<Item>| -> Vec<Item> {
            v.into_iter()
                .filter(|i| !matches!(i, Begin(_) | End(_)))
                .collect()
        };
        for units in 1..=3 {
            let mut port = RecPort::<false>(Rec::default());
            walk(step(units), 5, &mut port, &mut Prog::<false>).unwrap();
            assert_eq!(port.0.opened, Some(step(units)));
            assert_eq!(port.0.seen, layers_parts(walked(step(units), 5)));
            let mut port = RecPort::<true>(Rec::default());
            walk(group(units), 5, &mut port, &mut Prog::<true>).unwrap();
            assert_eq!(port.0.opened, Some(group(units)));
            assert_eq!(port.0.seen, layers_parts(walked(group(units), 5)));
        }
    }

    /// A program whose first layers have no host leg: the walk passes their
    /// serves by and runs every other part as [`order`] lays it out.
    #[test]
    fn walk_serves_host_legs_only() {
        struct Lead(usize);
        impl LayerProgram for Lead {
            type Port = RecPort<true>;
            fn host_leg(&self, at: At) -> bool {
                at.layer >= self.0
            }
            fn front(&mut self, port: &mut RecPort<true>, at: At) -> Result<(), Refused> {
                port.0.seen.push(Front(at));
                Ok(())
            }
        }
        for units in 1..=3 {
            let mut port = RecPort::<true>(Rec::default());
            walk(group(units), 5, &mut port, &mut Lead(2)).unwrap();
            let want: Vec<Item> = walked(group(units), 5)
                .into_iter()
                .filter(|i| match i {
                    Front(_) => true,
                    Serve(a) => a.layer >= 2,
                    _ => false,
                })
                .collect();
            assert_eq!(port.0.seen, want);
        }
    }

    /// A walk on the other kind of port, or of nothing, is refused before the
    /// port opens; a step port refuses a serve.
    #[test]
    fn walks_refused_by_name() {
        let mut port = RecPort::<false>(Rec::default());
        assert_eq!(
            walk(group(2), 3, &mut port, &mut Prog::<false>),
            Err(Refused::Port {
                overlap: PortKind::Batch,
                port: PortKind::Step
            })
        );
        assert_eq!(
            walk(step(0), 3, &mut port, &mut Prog::<false>),
            Err(Refused::Empty { units: 0, cols: 1 })
        );
        assert_eq!(port.0.opened, None);
        assert_eq!(port.serve(at(0, 1)), Err(Refused::Serve(at(0, 1))));
    }

    /// A pass of resident slots: one slot is the step and two the pair,
    /// today's walks; more are two lanes, lane 0 the first ⌈n/2⌉ slots and
    /// as wide, lane 1 the rest, one slot fewer for an odd count, never
    /// none. No slot is refused by name.
    #[test]
    fn slot_lanes_split_the_busy_slots() {
        assert_eq!(slot_lanes(0), Err(Refused::NoSlots));
        assert!(
            Refused::NoSlots.to_string().contains("0 resident slots"),
            "{}",
            Refused::NoSlots
        );
        assert_eq!(slot_lanes(1).map(|l| l.overlap), Ok(step(1)));
        assert_eq!(slot_lanes(2).map(|l| l.overlap), Ok(step(2)));
        for (n, units, cols, lane0) in [
            (1, 1, 1, 1),
            (2, 2, 1, 1),
            (3, 2, 2, 2),
            (4, 2, 2, 2),
            (8, 2, 4, 4),
        ] {
            let want = Lanes {
                overlap: Overlap {
                    units,
                    cols,
                    port: PortKind::Step,
                },
                lane0,
            };
            assert_eq!(slot_lanes(n), Ok(want), "{n} slots");
        }
        for n in 2..=17 {
            let l = slot_lanes(n).expect("a pass of slots");
            let lane1 = n - l.lane0;
            assert_eq!(l.overlap.units, 2, "{n} slots");
            assert_eq!(l.overlap.cols, l.lane0, "{n} slots");
            assert!((1..=l.lane0).contains(&lane1), "{n} slots: lane 1 {lane1}");
            assert_eq!(l.lane0 - lane1, n % 2, "{n} slots");
        }
    }
}
