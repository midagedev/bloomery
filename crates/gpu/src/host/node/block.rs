//! A prompt batch's block on a tier node ([`super`]): the block open, which
//! makes the node's block staging for blocks of up to a batch port's tokens,
//! and one block request a tiered layer-batch, which carries the block's
//! q8_1 planes (the q3 plane and its scales, the form the tier's image
//! carries, [`super::super::tier::TierAct::Q8`]) and each slot's tier place,
//! and is answered with the rows the node's experts wrote, packed: the tier
//! slots' rows alone, in slot order, after their count.

use super::super::tier::BlockRows;
use super::wire::{In, MAX_FRAME, Out};

// A reply's rows go as the node's card wrote them and land as node 0's
// stage card reads them, both in the wire's byte order.
const _: () = assert!(cfg!(target_endian = "little"));

/// What a node holds for a prompt batch's blocks, as its block open's reply
/// names it: the device bytes of its staging and of the architecture's block
/// scratch, where the architecture leaves a block's rows ([`BlockRows`]), and
/// the block's columns at most.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeBlock {
    pub staging: usize,
    pub scratch: usize,
    pub rows: BlockRows,
    pub cap: usize,
}

/// The bytes of a block request's body for `cols` columns of `n_used` slots
/// with `q3` and `d8` plane bytes a column: the layer and columns, the
/// places, the planes, each list behind its length word.
fn request_bytes(cols: usize, n_used: usize, (q3, d8): (usize, usize)) -> Option<usize> {
    let places = cols.checked_mul(n_used)?.checked_mul(4)?;
    let planes = cols.checked_mul(q3.checked_add(d8)?)?;
    places.checked_add(planes)?.checked_add(5 * 8)
}

/// The bytes of a block reply's body when every one of the block's
/// `cols · n_used` slots is the node's: the count, the rows' length word and
/// the rows of `hidden` f32.
fn reply_bytes(cols: usize, n_used: usize, hidden: usize) -> Option<usize> {
    cols.checked_mul(n_used)?
        .checked_mul(hidden)?
        .checked_mul(4)?
        .checked_add(2 * 8)
}

/// Refuse by name blocks of up to `cap` columns whose request or largest
/// reply, a frame's tag before its body, would pass [`MAX_FRAME`]: the
/// request's `planes` (q3 and d8 bytes a column) and `n_used` places a
/// column, the reply's rows of `hidden` f32 for every slot.
pub(super) fn check_frames(
    cap: usize,
    n_used: usize,
    hidden: usize,
    planes: (usize, usize),
) -> Result<(), String> {
    let fits = |n: Option<usize>| {
        n.and_then(|n| n.checked_add(1))
            .is_some_and(|n| n <= MAX_FRAME)
    };
    let (req, rep) = (
        request_bytes(cap, n_used, planes),
        reply_bytes(cap, n_used, hidden),
    );
    if fits(req) && fits(rep) {
        Ok(())
    } else {
        Err(format!(
            "blocks of {cap} columns of {n_used} slots of {hidden} values: a request of {req:?} \
             bytes or a reply of up to {rep:?} bytes passes a frame's {MAX_FRAME}"
        ))
    }
}

fn rows_byte(rows: BlockRows) -> u8 {
    match rows {
        BlockRows::Staged => 0,
        BlockRows::Packed => 1,
    }
}

fn rows_of(b: u8) -> Result<BlockRows, String> {
    match b {
        0 => Ok(BlockRows::Staged),
        1 => Ok(BlockRows::Packed),
        _ => Err(format!("a block-rows kind {b}, 0 (staged) or 1 (packed)")),
    }
}

/// The block open's reply: a flag, then for a node that runs blocks its
/// [`NodeBlock`].
pub(super) fn write_open_reply(o: &mut Out, block: Option<NodeBlock>) {
    match block {
        None => {
            o.u8(0);
        }
        Some(b) => {
            o.u8(1)
                .len(b.staging)
                .len(b.scratch)
                .u8(rows_byte(b.rows))
                .len(b.cap);
        }
    }
}

/// [`write_open_reply`]'s payload `body` for an open of blocks of `cap`
/// columns: a node that runs them must hold exactly `cap`.
pub(super) fn read_open_reply(body: &[u8], cap: usize) -> Result<Option<NodeBlock>, String> {
    let mut i = In::new(body);
    let block = if i.flag("runs blocks")? {
        let b = NodeBlock {
            staging: i.len("staging bytes")?,
            scratch: i.len("scratch bytes")?,
            rows: rows_of(i.u8("block rows")?)?,
            cap: i.len("block columns")?,
        };
        if b.cap != cap {
            return Err(format!(
                "a block open of {cap} columns answered for {} columns",
                b.cap
            ));
        }
        Some(b)
    } else {
        None
    };
    i.end()?;
    Ok(block)
}

/// A block request as the node reads it: layer `layer`'s block of `cols`
/// columns, each of its `cols · n_used` slots' tier place (`places`), and
/// the columns' q3 plane and scales, the stage card's bytes.
pub(super) struct BlockRequest<'a> {
    pub layer: usize,
    pub cols: usize,
    pub places: Vec<u32>,
    pub q3: &'a [u8],
    pub d8: &'a [u8],
}

/// Write a block request's payload ([`BlockRequest`]).
pub(super) fn write_request(
    o: &mut Out,
    (layer, cols): (usize, usize),
    places: &[u32],
    q3: &[u8],
    d8: &[u8],
) {
    o.len(layer).len(cols).u32s(places).bytes(q3).bytes(d8);
}

impl<'a> BlockRequest<'a> {
    /// The request in `i`, read to its end; its shape is checked against
    /// the node's staging by the node.
    pub(super) fn read(i: &mut In<'a>) -> Result<BlockRequest<'a>, String> {
        let r = BlockRequest {
            layer: i.len("layer")?,
            cols: i.len("columns")?,
            places: i.u32s("places")?,
            q3: i.bytes("q3 plane")?,
            d8: i.bytes("q3 scales")?,
        };
        i.end()?;
        Ok(r)
    }
}

/// The head of a block's reply: `k` rows of `hidden` f32 follow, written
/// with [`Out::raw`].
pub(super) fn write_reply_head(o: &mut Out, k: usize, hidden: usize) {
    o.len(k).len(k * hidden * 4);
}

/// The rows of a block reply's payload `body` for a block of `slots` slots
/// of rows of `hidden` f32, which sent `sent` tier slots: refused by name
/// for a count past the block's slots or other than the tier slots it sent,
/// and for rows of another length.
pub(super) fn read_reply(
    body: &[u8],
    slots: usize,
    hidden: usize,
    sent: usize,
) -> Result<&[u8], String> {
    let mut i = In::new(body);
    let k = i.len("rows")?;
    if k > slots {
        return Err(format!(
            "a block reply of {k} rows, past the block's {slots} slots"
        ));
    }
    if k != sent {
        return Err(format!(
            "a block reply of {k} rows where the block sent {sent} tier slots"
        ));
    }
    let rows = i.bytes("row values")?;
    if Some(rows.len()) != k.checked_mul(hidden).and_then(|n| n.checked_mul(4)) {
        return Err(format!(
            "a block reply of {k} rows of {hidden} values in {} bytes",
            rows.len()
        ));
    }
    i.end()?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::{
        BlockRequest, NodeBlock, check_frames, read_open_reply, read_reply, write_open_reply,
        write_reply_head, write_request,
    };
    use crate::host::node::wire::{In, MAX_FRAME, Out};
    use crate::host::tier::BlockRows;

    /// V4.1's block shapes: 7,168 values a column (28 super-blocks: a q3
    /// plane of 64 · 14 u64 and 56 f32 scales a column), 6 slots a token.
    const HIDDEN: usize = 7168;
    const PLANES: (usize, usize) = (64 * 14 * 8, 56 * 4);

    /// A block request reads back what was written; a reply's rows read
    /// back for the tier slots the block sent, and a count past the block's
    /// slots, a count other than the tier slots sent and rows of another
    /// length are refused by name.
    #[test]
    fn a_block_request_and_its_reply_read_back_as_written() {
        let places = [
            0,
            u32::MAX,
            3,
            u32::MAX,
            1,
            2,
            u32::MAX,
            4,
            5,
            6,
            7,
            u32::MAX,
        ];
        let (q3, d8) = (vec![7u8; 2 * PLANES.0], vec![9u8; 2 * PLANES.1]);
        let mut o = Out::default();
        write_request(&mut o, (5, 2), &places, &q3, &d8);
        let r = BlockRequest::read(&mut In::new(o.done())).expect("the request");
        assert_eq!((r.layer, r.cols), (5, 2));
        assert_eq!(r.places, places);
        assert!(r.q3 == q3.as_slice() && r.d8 == d8.as_slice());
        let mut cut = o.done().to_vec();
        cut.pop();
        assert!(BlockRequest::read(&mut In::new(&cut)).is_err());

        let hidden = 4;
        let rows: Vec<u8> = (0..3 * hidden * 4).map(|b| b as u8).collect();
        let reply = |k: usize, rows: &[u8]| {
            let mut o = Out::default();
            write_reply_head(&mut o, k, hidden);
            o.raw(rows);
            o.done().to_vec()
        };
        let ok = reply(3, &rows);
        assert_eq!(read_reply(&ok, 12, hidden, 3).expect("the rows"), rows);
        let past = reply(13, &[]);
        let e = read_reply(&past, 12, hidden, 13).unwrap_err();
        assert!(e.contains("past the block's 12 slots"), "{e}");
        let e = read_reply(&ok, 12, hidden, 2).unwrap_err();
        assert!(e.contains("where the block sent 2 tier slots"), "{e}");
        let mut short = Out::default();
        short.len(3).bytes(&rows[4..]);
        let e = read_reply(short.done(), 12, hidden, 3).unwrap_err();
        assert!(e.contains("of 3 rows of 4 values"), "{e}");
    }

    /// The block open's reply reads back a node that runs blocks and one
    /// that does not; an answer for other columns is refused. V4.1's blocks
    /// of 512 columns fit a frame and blocks past it are refused by name.
    #[test]
    fn a_block_open_reads_back_and_its_frames_fit() {
        let b = NodeBlock {
            staging: 1 << 30,
            scratch: 12345,
            rows: BlockRows::Packed,
            cap: 512,
        };
        let mut o = Out::default();
        write_open_reply(&mut o, Some(b));
        assert_eq!(read_open_reply(o.done(), 512), Ok(Some(b)));
        assert!(read_open_reply(o.done(), 256).is_err());
        let mut none = Out::default();
        write_open_reply(&mut none, None);
        assert_eq!(read_open_reply(none.done(), 512), Ok(None));

        assert!(check_frames(512, 6, HIDDEN, PLANES).is_ok());
        let past = MAX_FRAME / (6 * HIDDEN * 4) + 1;
        let e = check_frames(past, 6, HIDDEN, PLANES).unwrap_err();
        assert!(e.contains("passes a frame's"), "{e}");
        assert!(check_frames(usize::MAX, 6, HIDDEN, PLANES).is_err());
    }
}
