//! One sequence's context window, walked through the engine's hash.

use engram::{EngramError, Hash};

/// The last `n_gram` mapped tokens of one sequence, newest first.
///
/// Before the sequence starts every slot is the pad value, and a slot stays pad
/// until a real token has shifted into it. That covers a **contiguous** stream,
/// which is the only shape this crate is fed. The port's rule is wider: it
/// blocks from the first unavailable position outward, so a window with a hole
/// in the middle pads everything older than the hole. A caller that can produce
/// such a window — a batch over several sequences, or a cache whose cells were
/// evicted — needs that rule added here, not assumed.
pub struct Context<'a> {
    hash: &'a Hash,
    window: Vec<u64>,
}

impl<'a> Context<'a> {
    /// A window at a sequence start: every slot pad.
    pub fn new(hash: &'a Hash) -> Context<'a> {
        Context {
            hash,
            window: vec![hash.pad_id(); hash.n_gram()],
        }
    }

    /// Back to a sequence start.
    pub fn reset(&mut self) {
        let pad = self.hash.pad_id();
        self.window.fill(pad);
    }

    /// Advance by one token. Older slots shift back by one; the oldest falls
    /// off the end. No allocation. A token past the hash's map is not a token
    /// of the model: it is refused by name ([`Hash::map_token`]) and the
    /// window stays as it was.
    pub fn push(&mut self, token: u32) -> Result<(), EngramError> {
        let mapped = self.hash.map_token(token)?;
        self.window.rotate_right(1);
        self.window[0] = mapped;
        Ok(())
    }

    /// The mapped window, `window()[s]` being the token `s` positions back.
    pub fn window(&self) -> &[u64] {
        &self.window
    }

    /// The hash this window maps its tokens through.
    pub fn hash(&self) -> &'a Hash {
        self.hash
    }
}
