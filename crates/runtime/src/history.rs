//! What the drafts that read no weights share: a context of ids (the prompt,
//! then every id the passes kept) that follows its target by position, and a
//! proposal made from it alone. A seat's slot table drives any of them through
//! this one surface, so a new history draft is a type and its rule, never a
//! copy of the table.

use crate::{Lookup, Recall};

/// A draft of the token history: its context, kept in step with the target's
/// positions, and the ids the context says come next.
pub trait HistoryDraft {
    /// The draft's name as `/props` and `BLOOMERY_DRAFT` spell it.
    fn kind(&self) -> &'static str;

    /// The context so far: the prompt, then every kept id.
    fn context(&self) -> &[u32];

    /// Append `id` to the context.
    fn push(&mut self, id: u32);

    /// An empty context again, the tables' memory kept.
    fn reset(&mut self);

    /// The context `ctx`, the tables rebuilt over it.
    fn rebuild(&mut self, ctx: &[u32]);

    /// The context made what it holds of a target's positions before `pos`,
    /// then `last`, the id at `pos`.
    fn stand(&mut self, pos: usize, last: u32);

    /// The ids the context proposes to follow it, written to the front of
    /// `out` and counted: 0 is no proposal and leaves `out` alone, and a
    /// proposal never holds more ids than `out` has room for.
    fn propose_ids(&self, out: &mut [u32]) -> usize;
}

impl HistoryDraft for Lookup {
    fn kind(&self) -> &'static str {
        "lookup"
    }

    fn context(&self) -> &[u32] {
        Lookup::context(self)
    }

    fn push(&mut self, id: u32) {
        Lookup::push(self, id);
    }

    fn reset(&mut self) {
        Lookup::reset(self);
    }

    fn rebuild(&mut self, ctx: &[u32]) {
        Lookup::rebuild(self, ctx);
    }

    fn stand(&mut self, pos: usize, last: u32) {
        Lookup::stand(self, pos, last);
    }

    fn propose_ids(&self, out: &mut [u32]) -> usize {
        match (out.first_mut(), Lookup::propose(self)) {
            (Some(room), Some(id)) => {
                *room = id;
                1
            }
            _ => 0,
        }
    }
}

impl<const W: usize> HistoryDraft for Recall<W> {
    fn kind(&self) -> &'static str {
        "recall"
    }

    fn context(&self) -> &[u32] {
        Recall::context(self)
    }

    fn push(&mut self, id: u32) {
        Recall::push(self, id);
    }

    fn reset(&mut self) {
        Recall::reset(self);
    }

    fn rebuild(&mut self, ctx: &[u32]) {
        Recall::rebuild(self, ctx);
    }

    fn stand(&mut self, pos: usize, last: u32) {
        Recall::stand(self, pos, last);
    }

    fn propose_ids(&self, out: &mut [u32]) -> usize {
        let n = out.len().min(W);
        match Recall::propose(self, &mut out[..n]) {
            Some(_) => n,
            None => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HistoryDraft;
    use crate::{Lookup, Recall};

    /// Both drafts behind the one surface propose what their own types do:
    /// the lookup one id, the recall draft up to its width and the room, none
    /// with no match or no room, and `out` untouched then; and `stand` follows
    /// a target that cut back and stepped on.
    #[test]
    fn both_history_drafts_propose_through_one_surface() {
        let ctx = [5, 6, 9, 1, 5, 6, 9];
        let mut lookup = Lookup::new();
        let mut recall = Recall::<3>::new(2).unwrap();
        let drafts: [&mut dyn HistoryDraft; 2] = [&mut lookup, &mut recall];
        let want = [(1, "lookup"), (3, "recall")];
        for (d, (most, kind)) in drafts.into_iter().zip(want) {
            assert_eq!(d.kind(), kind);
            d.rebuild(&ctx);
            assert_eq!(d.context(), ctx);
            let mut out = [u32::MAX; 8];
            // The tail (6, 9) recurs: it was followed by 1, then 5, 6.
            assert_eq!(d.propose_ids(&mut out), most, "{kind}");
            assert_eq!(out[..most.min(3)], [1, 5, 6][..most.min(3)], "{kind}");
            assert!(
                out[most..].iter().all(|&id| id == u32::MAX),
                "{kind}: past it"
            );
            let mut none: [u32; 0] = [];
            assert_eq!(d.propose_ids(&mut none), 0, "{kind}: no room");
            let mut two = [u32::MAX; 2];
            assert_eq!(d.propose_ids(&mut two), most.min(2), "{kind}: room of 2");
            // A cut back to 3 ids and a step on: the context follows by position.
            d.stand(3, 1);
            assert_eq!(d.context(), [5, 6, 9, 1], "{kind}: stood at 3");
            d.reset();
            assert!(d.context().is_empty(), "{kind}: reset");
            let mut room = [u32::MAX; 4];
            assert_eq!(d.propose_ids(&mut room), 0, "{kind}: nothing after a reset");
            assert_eq!(room, [u32::MAX; 4]);
        }
    }
}
