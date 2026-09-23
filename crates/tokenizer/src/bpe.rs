//! Byte-level BPE on one pre-tokenizer word, as `llm_tokenizer_bpe_session`
//! runs it.
//!
//! Every byte of the word starts as one symbol (written, in the vocabulary,
//! as its GPT-2 alphabet character). The adjacent pair with the lowest merge
//! rank merges first, ties going to the leftmost pair; a queued pair whose
//! sides have changed since it was queued is skipped. A finished symbol
//! whose text is not a token falls back to the one-byte tokens of its text's
//! bytes, and bytes with no such token are dropped.
//!
//! Strings are interned so the merge loop compares integers: token texts
//! first, so a token's intern id is its token id, then any merge side or
//! merge result that is not a token. An interned id names one exact string,
//! so comparing ids is comparing the reference's strings.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::unicode::{byte_to_cpt_table, encode_cpt};

const NONE: u32 = u32::MAX;

pub(crate) struct Bpe {
    n_tokens: u32,
    /// The intern id of each byte's one-character string, or `NONE`.
    byte_iid: [u32; 256],
    /// Each byte's GPT-2 alphabet character, UTF-8.
    byte_text: [Vec<u8>; 256],
    /// The token whose text is exactly this one byte, or `NONE`.
    one_byte_token: [u32; 256],
    /// `(left, right)` intern ids -> `(rank, merged intern id)`; the first
    /// listing of a pair keeps its rank.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Merges whose sides or result are not tokens (reported, not needed).
    pub(crate) merges_off_vocab: usize,
}

struct Interner {
    ids: HashMap<Vec<u8>, u32>,
}

impl Interner {
    fn get_or_add(&mut self, s: &[u8]) -> u32 {
        if let Some(&id) = self.ids.get(s) {
            return id;
        }
        let id = u32::try_from(self.ids.len()).expect("fewer than 2^32 distinct strings");
        self.ids.insert(s.to_vec(), id);
        id
    }
}

impl Bpe {
    pub(crate) fn new(texts: &[String], merges: &[String]) -> Bpe {
        let n_tokens = u32::try_from(texts.len()).expect("the vocabulary loader bounds the count");
        let mut interner = Interner {
            ids: HashMap::with_capacity(texts.len() + merges.len()),
        };
        for t in texts {
            interner.get_or_add(t.as_bytes());
        }

        let mut off_vocab = 0;
        let mut table = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            let m = m.as_bytes();
            // The reference splits at the first space after the first byte;
            // a merge with no such space is the pair ("", "").
            let (first, second) = match m.iter().skip(1).position(|&b| b == b' ') {
                Some(p) => (&m[..p + 1], &m[p + 2..]),
                None => (&m[..0], &m[..0]),
            };
            let merged = [first, second].concat();
            let (l, r, j) = (
                interner.get_or_add(first),
                interner.get_or_add(second),
                interner.get_or_add(&merged),
            );
            if l >= n_tokens || r >= n_tokens || j >= n_tokens {
                off_vocab += 1;
            }
            let rank = u32::try_from(rank).expect("fewer than 2^32 merges");
            table.entry((l, r)).or_insert((rank, j));
        }

        let cpts = byte_to_cpt_table();
        let byte_text: [Vec<u8>; 256] = std::array::from_fn(|b| {
            let mut s = Vec::with_capacity(2);
            encode_cpt(cpts[b], &mut s);
            s
        });
        let byte_iid =
            std::array::from_fn(|b| interner.ids.get(&byte_text[b]).copied().unwrap_or(NONE));
        let one_byte_token = std::array::from_fn(|b| {
            let id = interner.ids.get(&[b as u8][..]).copied().unwrap_or(NONE);
            if id < n_tokens { id } else { NONE }
        });
        Bpe {
            n_tokens,
            byte_iid,
            byte_text,
            one_byte_token,
            merges: table,
            merges_off_vocab: off_vocab,
        }
    }

    /// Append the tokens of one word (raw bytes, before the GPT-2 byte
    /// encoding) to `out`.
    pub(crate) fn word(&self, word: &[u8], s: &mut Scratch, out: &mut Vec<u32>) {
        let n = word.len();
        s.syms.clear();
        s.syms.extend(word.iter().enumerate().map(|(i, &b)| Sym {
            len: 1,
            iid: self.byte_iid[b as usize],
            prev: if i == 0 { NONE } else { i as u32 - 1 },
            next: if i + 1 == n { NONE } else { i as u32 + 1 },
        }));
        s.heap.clear();
        for i in 1..n as u32 {
            self.push(&s.syms, &mut s.heap, i - 1, i);
        }

        while let Some(Reverse((_, left, right, size, merged))) = s.heap.pop() {
            let (l, r) = (s.syms[left as usize], s.syms[right as usize]);
            if l.len == 0 || r.len == 0 || l.len + r.len != size {
                continue;
            }
            let next = r.next;
            s.syms[left as usize] = Sym {
                len: size,
                iid: merged,
                next,
                ..l
            };
            s.syms[right as usize].len = 0;
            if next != NONE {
                s.syms[next as usize].prev = left;
            }
            if l.prev != NONE {
                self.push(&s.syms, &mut s.heap, l.prev, left);
            }
            if next != NONE {
                self.push(&s.syms, &mut s.heap, left, next);
            }
        }

        // A live symbol starts at the byte whose index it has.
        for (start, sym) in s.syms.iter().enumerate() {
            let len = sym.len as usize;
            if len == 0 {
                continue;
            }
            if sym.iid < self.n_tokens {
                out.push(sym.iid);
                continue;
            }
            for &b in &word[start..start + len] {
                for &e in &self.byte_text[b as usize] {
                    let t = self.one_byte_token[e as usize];
                    if t != NONE {
                        out.push(t);
                    }
                }
            }
        }
    }

    fn push(&self, syms: &[Sym], heap: &mut Heap, left: u32, right: u32) {
        let (l, r) = (syms[left as usize], syms[right as usize]);
        if l.iid == NONE || r.iid == NONE {
            return;
        }
        if let Some(&(rank, merged)) = self.merges.get(&(l.iid, r.iid)) {
            heap.push(Reverse((rank, left, right, l.len + r.len, merged)));
        }
    }
}

#[derive(Clone, Copy)]
struct Sym {
    /// Bytes this symbol covers; 0 once merged into its left neighbour.
    len: u32,
    iid: u32,
    prev: u32,
    next: u32,
}

/// `(rank, left, right, size at queueing, merged intern id)`, smallest first.
type Heap = BinaryHeap<Reverse<(u32, u32, u32, u32, u32)>>;

/// Reusable buffers for [`Bpe::word`].
#[derive(Default)]
pub(crate) struct Scratch {
    syms: Vec<Sym>,
    heap: Heap,
}
