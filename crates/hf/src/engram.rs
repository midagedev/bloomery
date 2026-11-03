//! DeepSeek-V4.1's Engram hash constants as its reference derives them (`inference/engram.py`
//! of the checkpoint folder), from `config.json` and `tokenizer.json`: no folder stores them.
//!
//! - The multipliers of a site are `rng.integers(0, bound, 4) * 2 + 1` of numpy's
//!   `default_rng(10007 * layer)`, `bound = (i64::MAX / compressed vocabulary) / 2`
//!   ([`multipliers`], numpy's `SeedSequence`, `PCG64` and Lemire's bounded draw ported).
//! - The primes are the bucket sizes: each n-gram order's heads take the next primes past
//!   `engram_vocab_size - 1` no earlier bucket took, so every bucket of every site has its own
//!   ([`primes`]); a site's offsets are its buckets' prefix sums.
//! - The token map sends each token id to the id of its decoded text normalized (NFKC, NFD,
//!   accents stripped, lower case, whitespace runs one space, stripped), so tokens that
//!   normalize alike share an id ([`token_map`]); a token whose bytes are not whole UTF-8 keeps
//!   its spelling as its key.

use std::collections::HashMap;

use unicode_normalization_alignments::UnicodeNormalization;
use unicode_normalization_alignments::char::is_combining_mark;

/// numpy's `SeedSequence` of the one-word entropy `seed` (`spawn_key` empty, pool of four
/// words), and the four 64-bit words its `generate_state(4, uint64)` gives.
fn seed_sequence(seed: u32) -> [u64; 4] {
    const INIT_A: u32 = 0x43b0_d7e5;
    const MULT_A: u32 = 0x931e_8875;
    const INIT_B: u32 = 0x8b51_f9dd;
    const MULT_B: u32 = 0x58f3_8ded;
    const MIX_MULT_L: u32 = 0xca01_f9dd;
    const MIX_MULT_R: u32 = 0x4973_f715;
    let mut hash_const = INIT_A;
    let mut hashmix = |value: u32| {
        let mut v = value ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_A);
        v = v.wrapping_mul(hash_const);
        v ^ (v >> 16)
    };
    let mix = |x: u32, y: u32| {
        let r = MIX_MULT_L
            .wrapping_mul(x)
            .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
        r ^ (r >> 16)
    };
    let mut pool = [0u32; 4];
    for (i, word) in pool.iter_mut().enumerate() {
        *word = hashmix(if i == 0 { seed } else { 0 });
    }
    for src in 0..4 {
        for dst in 0..4 {
            if src != dst {
                pool[dst] = mix(pool[dst], hashmix(pool[src]));
            }
        }
    }
    let mut hash_const = INIT_B;
    let words: Vec<u32> = (0..8)
        .map(|i| {
            let mut v = pool[i % 4] ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            v = v.wrapping_mul(hash_const);
            v ^ (v >> 16)
        })
        .collect();
    std::array::from_fn(|i| u64::from(words[2 * i]) | (u64::from(words[2 * i + 1]) << 32))
}

/// numpy's `PCG64` (XSL-RR over a 128-bit LCG).
struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    const MULT: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

    /// `PCG64(SeedSequence(seed))`: the state and stream of `generate_state`'s four words.
    fn new(seed: u32) -> Pcg64 {
        let w = seed_sequence(seed);
        let state = (u128::from(w[0]) << 64) | u128::from(w[1]);
        let inc = (u128::from(w[2]) << 64) | u128::from(w[3]);
        let mut rng = Pcg64 {
            state: 0,
            inc: (inc << 1) | 1,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(state);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(Self::MULT).wrapping_add(self.inc);
    }

    fn next_u64(&mut self) -> u64 {
        self.step();
        let rot = (self.state >> 122) as u32;
        (((self.state >> 64) as u64) ^ (self.state as u64)).rotate_right(rot)
    }

    /// `Generator.integers(0, high)` for one int64 of a range past 32 bits: Lemire's
    /// multiply-and-reject over 64-bit draws (numpy's `bounded_lemire_uint64`).
    fn below(&mut self, high: u64) -> u64 {
        let mut m = u128::from(self.next_u64()) * u128::from(high);
        if (m as u64) < high {
            let threshold = high.wrapping_neg() % high;
            while (m as u64) < threshold {
                m = u128::from(self.next_u64()) * u128::from(high);
            }
        }
        (m >> 64) as u64
    }
}

/// Each site's `n_gram` hash multipliers, site after site: the odd `2x + 1` of numpy's
/// `default_rng(10007 * layer).integers(0, (i64::MAX / vocab) / 2, n_gram)`. A compressed
/// vocabulary of no ids or of 2^30 ids or more (whose range numpy draws in 32 bits) and a seed
/// past u32 are refused.
pub fn multipliers(layer_ids: &[u64], n_gram: usize, vocab: u64) -> Result<Vec<u64>, String> {
    let high = (i64::MAX as u64)
        .checked_div(vocab)
        .map(|b| (b / 2).max(1))
        .filter(|&b| b > u64::from(u32::MAX) + 1)
        .ok_or_else(|| format!("a compressed vocabulary of {vocab} ids is not 1 to below 2^30"))?;
    let mut out = Vec::with_capacity(layer_ids.len() * n_gram);
    for &layer in layer_ids {
        let seed = layer
            .checked_mul(10007)
            .and_then(|s| u32::try_from(s).ok())
            .ok_or_else(|| format!("layer {layer}'s seed passes u32"))?;
        let mut rng = Pcg64::new(seed);
        out.extend((0..n_gram).map(|_| rng.below(high) * 2 + 1));
    }
    Ok(out)
}

fn is_prime(n: u64) -> bool {
    n >= 2
        && (2..)
            .take_while(|d| d * d <= n)
            .all(|d| !n.is_multiple_of(d))
}

/// The most buckets [`primes`] finds: far past any table's (24 a site in V4.1).
pub const MAX_BUCKETS: usize = 1 << 16;

/// Each of `sites` sites' bucket sizes, `orders` n-gram orders of `heads` heads in that order:
/// each order starts past `vocab - 1` and takes the next primes no bucket took before it. More
/// than [`MAX_BUCKETS`] buckets in all are refused.
pub fn primes(
    sites: usize,
    orders: usize,
    heads: usize,
    vocab: u64,
) -> Result<Vec<Vec<u64>>, String> {
    sites
        .checked_mul(orders)
        .and_then(|n| n.checked_mul(heads))
        .filter(|&n| n <= MAX_BUCKETS)
        .ok_or_else(|| {
            format!("{sites} sites of {orders} orders of {heads} heads pass {MAX_BUCKETS} buckets")
        })?;
    let mut taken = std::collections::BTreeSet::new();
    Ok((0..sites)
        .map(|_| {
            let mut site = Vec::with_capacity(orders * heads);
            for _ in 0..orders {
                let mut at = vocab.saturating_sub(1);
                for _ in 0..heads {
                    at = (at + 1..)
                        .find(|&c| is_prime(c) && !taken.contains(&c))
                        .expect("a prime past every u64 taken");
                    taken.insert(at);
                    site.push(at);
                }
            }
            site
        })
        .collect())
}

/// Each bucket's first row in its site's table: the prefix sums of `sizes` from 0.
#[must_use]
pub fn offsets(sizes: &[u64]) -> Vec<u64> {
    sizes
        .iter()
        .scan(0u64, |at, &s| {
            let first = *at;
            *at += s;
            Some(first)
        })
        .collect()
}

/// The text Hugging Face's ByteLevel decoder gives the one token `token`: the bytes `byte` gives
/// its codepoints ([`tokenizer::byte_decoder`]), or the token's own UTF-8 when a codepoint lies
/// outside the alphabet, read lossily.
fn decoded(token: &str, byte: &[Option<u8>]) -> String {
    let bytes: Option<Vec<u8>> = token
        .chars()
        .map(|c| byte.get(c as usize).copied().flatten())
        .collect();
    match bytes {
        Some(b) => String::from_utf8_lossy(&b).into_owned(),
        None => token.to_string(),
    }
}

/// The reference's normalizer: NFKC, NFD, combining marks dropped, lower case, every run of
/// spaces, tabs and line ends one space, then stripped of whitespace, but a text that is one
/// space stays one.
fn normalized(text: &str) -> String {
    let nfkc: String = text.nfkc().map(|(c, _)| c).collect();
    let lower: String = nfkc
        .nfd()
        .map(|(c, _)| c)
        .filter(|&c| !is_combining_mark(c))
        .flat_map(char::to_lowercase)
        .collect();
    let mut out = String::with_capacity(lower.len());
    for c in lower.chars() {
        if matches!(c, ' ' | '\t' | '\r' | '\n') {
            if !out.ends_with(' ') {
                out.push(' ');
            }
        } else {
            out.push(c);
        }
    }
    if out == " " {
        return out;
    }
    out.trim_matches(char::is_whitespace).to_string()
}

/// The compressed id of each token of `tokens` (a token's spelling by id: its added text or its
/// vocabulary string, empty for an id with none) and how many ids there are: a token's key is
/// its decoded text normalized, the decoded text when that is empty, and its spelling when the
/// decoded text holds U+FFFD; keys take ids in order of first sight.
#[must_use]
pub fn token_map(tokens: &[&str]) -> (Vec<u32>, u32) {
    let byte = tokenizer::byte_decoder();
    let mut ids: HashMap<String, u32> = HashMap::new();
    let map = tokens
        .iter()
        .map(|&token| {
            let text = decoded(token, &byte);
            let key = if text.contains('\u{FFFD}') {
                token.to_string()
            } else {
                Some(normalized(&text))
                    .filter(|n| !n.is_empty())
                    .unwrap_or(text)
            };
            let next = ids.len() as u32;
            *ids.entry(key).or_insert(next)
        })
        .collect();
    (map, ids.len() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// numpy 2.5's `default_rng(10007 * layer).integers(0, 46539438283891, 4) * 2 + 1` for
    /// DeepSeek-V4.1-Flash's two sites (compressed vocabulary 99,092).
    #[test]
    fn multipliers_are_numpy_s_draws() {
        assert_eq!(
            multipliers(&[1, 14], 4, 99_092).unwrap(),
            [
                76_632_096_046_245,
                4_839_876_093_313,
                35_959_672_319_349,
                73_987_337_458_391,
                67_716_810_739_261,
                51_510_806_800_915,
                30_921_347_202_721,
                82_619_226_485_591
            ]
        );
        assert!(multipliers(&[1], 4, u64::from(u32::MAX)).is_err());
    }

    /// Each order restarts past the vocabulary and skips the primes taken; the two real sites'
    /// 24 buckets sum to their tables' rows.
    #[test]
    fn primes_never_repeat_and_fill_the_tables() {
        assert_eq!(primes(1, 2, 2, 10).unwrap(), [vec![11, 13, 17, 19]]);
        assert!(primes(1, 2, 1 << 32, 10).is_err());
        let p = primes(2, 3, 8, 16_000_000).unwrap();
        let rows: Vec<u64> = p.iter().map(|s| s.iter().sum()).collect();
        assert_eq!(rows, [384_006_168, 384_016_682]);
        assert_eq!(offsets(&[3, 5, 7]), [0, 3, 8]);
    }

    /// Tokens that normalize alike share an id; an unpaired UTF-8 byte keys by its spelling; one
    /// space stays a token of its own, apart from the empty text.
    #[test]
    fn tokens_share_an_id_by_their_normalized_text() {
        // "Ġ" is the space byte, "Ã©" the two bytes of "é", "Ã" one byte of it alone.
        let (map, n) = token_map(&["The", "ĠThe", "the", "THE", "Ã©", "e", "Ã", "Ġ", "", "<x>"]);
        assert_eq!(map, [0, 0, 0, 0, 1, 1, 2, 3, 4, 5]);
        assert_eq!(n, 6);
    }
}
