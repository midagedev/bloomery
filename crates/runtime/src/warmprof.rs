//! The warm section: one stream's own routing, as a value with a byte codec.
//!
//! Two parts, both indexed `layer * experts + id`: `long`, how many times each
//! routed expert was chosen over every position the stream covers, and
//! `recent`, the stream's heat as [`crate::swaprule::SwapRule::stream_close`]
//! hands it over and [`crate::swaprule::SwapRule::stream_open`] takes it back,
//! empty when the stream ran no decode. The section holds nothing that belongs
//! to the card as a whole.
//!
//! The encoding is little-endian: the magic `BWRM`, a `u16` version, `layers`,
//! `experts`, `long.len()` and `recent.len()` as `u32`, then `long` as `u32`
//! cells and `recent` as the bits of its `f64` cells. [`WarmSection::decode`]
//! takes the shape the caller expects and refuses by name anything else.

use std::fmt;

use crate::swaprule::valid_heat_cell;

const MAGIC: [u8; 4] = *b"BWRM";
const VERSION: u16 = 1;
/// Magic, version, then `layers`, `experts`, `long.len()` and `recent.len()`.
const HEADER_BYTES: usize = MAGIC.len() + size_of::<u16>() + 4 * size_of::<u32>();

/// One stream's routing over `layers × experts` cells.
#[derive(Clone, Debug, PartialEq)]
pub struct WarmSection {
    pub layers: u32,
    pub experts: u32,
    /// Counts per (layer, expert) over every position the stream covers:
    /// `layers * experts` long.
    pub long: Vec<u32>,
    /// The stream's heat: `layers * experts` long, or empty when the stream
    /// ran no decode.
    pub recent: Vec<f64>,
}

/// A byte string [`WarmSection::decode`] refuses, by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarmError {
    /// Fewer bytes than the header, or than the body the header counts.
    Truncated { len: usize, need: u64 },
    /// More bytes than the body the header counts.
    TooLong { len: usize, want: u64 },
    /// Not a warm section.
    BadMagic { got: [u8; 4] },
    /// A version this reader does not know.
    Version { got: u16 },
    /// A section of another shape than the one asked.
    Shape {
        layers: u32,
        experts: u32,
        want_layers: u32,
        want_experts: u32,
    },
    /// A `long` that is not `layers * experts` long.
    LongLength { len: u32, want: u64 },
    /// A `recent` that is neither empty nor `layers * experts` long.
    RecentLength { len: u32, want: u64 },
    /// A `recent` cell that is not finite or is negative: no fold writes one.
    RecentCell { index: usize },
}

impl fmt::Display for WarmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            WarmError::Truncated { len, need } => {
                write!(
                    f,
                    "warm section of {len} bytes, short of the {need} it needs"
                )
            }
            WarmError::TooLong { len, want } => {
                write!(
                    f,
                    "warm section of {len} bytes, past the {want} its header counts"
                )
            }
            WarmError::BadMagic { got } => {
                write!(f, "warm section magic {got:02x?}, not {MAGIC:02x?}")
            }
            WarmError::Version { got } => {
                write!(f, "warm section version {got}, this reader knows {VERSION}")
            }
            WarmError::Shape {
                layers,
                experts,
                want_layers,
                want_experts,
            } => write!(
                f,
                "warm section of {layers} layers × {experts} experts, not the \
                 {want_layers} × {want_experts} asked"
            ),
            WarmError::LongLength { len, want } => {
                write!(
                    f,
                    "warm section holds {len} long counts, not layers × experts = {want}"
                )
            }
            WarmError::RecentLength { len, want } => write!(
                f,
                "warm section holds {len} recent cells, neither none nor layers × experts = {want}"
            ),
            WarmError::RecentCell { index } => write!(
                f,
                "warm section recent cell {index} is not finite or is negative: no fold writes one"
            ),
        }
    }
}

impl std::error::Error for WarmError {}

impl WarmSection {
    /// The encoded length in bytes: what [`WarmSection::encode`] appends.
    #[must_use]
    pub fn n_bytes(&self) -> usize {
        HEADER_BYTES + size_of::<u32>() * self.long.len() + size_of::<f64>() * self.recent.len()
    }

    /// Append the encoding to `out`. The lengths written are the vectors' own:
    /// a section whose vectors do not fit its shape encodes as it is and is
    /// refused by [`WarmSection::decode`].
    pub fn encode(&self, out: &mut Vec<u8>) {
        let count = |len: usize| {
            u32::try_from(len).expect(
                "a section holds layers × experts cells, both u32 and far under 2^32 for any model",
            )
        };
        out.reserve(self.n_bytes());
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        for n in [
            self.layers,
            self.experts,
            count(self.long.len()),
            count(self.recent.len()),
        ] {
            out.extend_from_slice(&n.to_le_bytes());
        }
        for &c in &self.long {
            out.extend_from_slice(&c.to_le_bytes());
        }
        for &c in &self.recent {
            out.extend_from_slice(&c.to_bits().to_le_bytes());
        }
    }

    /// The section `bytes` encode, of shape `layers × experts`. Refused by
    /// name: a bad magic, another version, a truncated or long body, another
    /// shape, a `long` or `recent` length that does not fit the shape (`long`
    /// must be full, `recent` full or empty), and a `recent` cell that is not
    /// finite or is negative.
    pub fn decode(bytes: &[u8], layers: u32, experts: u32) -> Result<WarmSection, WarmError> {
        if bytes.len() < HEADER_BYTES {
            return Err(WarmError::Truncated {
                len: bytes.len(),
                need: HEADER_BYTES as u64,
            });
        }
        let got: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if got != MAGIC {
            return Err(WarmError::BadMagic { got });
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != VERSION {
            return Err(WarmError::Version { got: version });
        }
        let word = |at: usize| {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        let (have_layers, have_experts) = (word(6), word(10));
        if (have_layers, have_experts) != (layers, experts) {
            return Err(WarmError::Shape {
                layers: have_layers,
                experts: have_experts,
                want_layers: layers,
                want_experts: experts,
            });
        }
        let (long_len, recent_len) = (word(14), word(18));
        let cells = u64::from(layers) * u64::from(experts);
        if u64::from(long_len) != cells {
            return Err(WarmError::LongLength {
                len: long_len,
                want: cells,
            });
        }
        if recent_len != 0 && u64::from(recent_len) != cells {
            return Err(WarmError::RecentLength {
                len: recent_len,
                want: cells,
            });
        }
        // The lengths are checked against the shape, so the body's size is the
        // header's claim: compared with the bytes before anything is allocated.
        let body = u128::from(long_len) * size_of::<u32>() as u128
            + u128::from(recent_len) * size_of::<f64>() as u128;
        let want = HEADER_BYTES as u128 + body;
        let want_bytes = u64::try_from(want).unwrap_or(u64::MAX);
        if (bytes.len() as u128) < want {
            return Err(WarmError::Truncated {
                len: bytes.len(),
                need: want_bytes,
            });
        }
        if (bytes.len() as u128) > want {
            return Err(WarmError::TooLong {
                len: bytes.len(),
                want: want_bytes,
            });
        }
        let (long_bytes, recent_bytes) =
            bytes[HEADER_BYTES..].split_at(long_len as usize * size_of::<u32>());
        let long = long_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| u32::from_le_bytes(c))
            .collect();
        let recent: Vec<f64> = recent_bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&c| f64::from_bits(u64::from_le_bytes(c)))
            .collect();
        if let Some(index) = recent.iter().position(|&c| !valid_heat_cell(c)) {
            return Err(WarmError::RecentCell { index });
        }
        Ok(WarmSection {
            layers,
            experts,
            long,
            recent,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{HEADER_BYTES, WarmError, WarmSection};

    // The section's codec and its refusals. Header offsets: magic 0, version
    // 4, layers 6, experts 10, `long.len()` 14, `recent.len()` 18.

    /// 3 layers × 4 experts: `long` counts, `recent` with a subnormal, a zero
    /// and a fraction a decimal cannot hold.
    fn section() -> WarmSection {
        let recent = (0..12)
            .map(|i| match i {
                0 => 0.0,
                1 => f64::from_bits(1),
                2 => 0.1 + 0.2,
                _ => f64::from(i) * 0.9_f64.powi(i),
            })
            .collect();
        WarmSection {
            layers: 3,
            experts: 4,
            long: (0..12).map(|i| i * 7).collect(),
            recent,
        }
    }

    fn encoded(s: &WarmSection) -> Vec<u8> {
        let mut out = Vec::new();
        s.encode(&mut out);
        out
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|c| c.to_bits()).collect()
    }

    /// Put `value` (little-endian) at header offset `at`.
    fn patch(bytes: &mut [u8], at: usize, value: &[u8]) {
        bytes[at..at + value.len()].copy_from_slice(value);
    }

    /// An encoding round-trips bit for bit, `recent` empty or full, and
    /// `n_bytes` is the encoded length.
    #[test]
    fn a_section_round_trips_bit_for_bit() {
        let s = section();
        let bytes = encoded(&s);
        assert_eq!(bytes.len(), s.n_bytes());
        assert_eq!(bytes.len(), HEADER_BYTES + 12 * 4 + 12 * 8);
        let back = WarmSection::decode(&bytes, 3, 4).unwrap();
        assert_eq!((back.layers, back.experts), (3, 4));
        assert_eq!(back.long, s.long);
        assert_eq!(bits(&back.recent), bits(&s.recent));
        assert_eq!(back.recent[1], f64::from_bits(1), "the subnormal survives");
        assert_eq!(
            back.recent[0].to_bits(),
            0,
            "the zero keeps its sign bit clear"
        );

        let none = WarmSection {
            recent: Vec::new(),
            ..s
        };
        let bytes = encoded(&none);
        assert_eq!(bytes.len(), none.n_bytes());
        assert_eq!(WarmSection::decode(&bytes, 3, 4).unwrap(), none);
    }

    /// `encode` appends: bytes already in `out` stay.
    #[test]
    fn encode_appends_to_the_buffer() {
        let s = section();
        let mut out = vec![9, 9];
        s.encode(&mut out);
        assert_eq!(out[..2], [9, 9]);
        assert_eq!(out[2..], encoded(&s));
    }

    /// Another magic is refused whole.
    #[test]
    fn a_bad_magic_is_refused() {
        let mut bytes = encoded(&section());
        patch(&mut bytes, 0, b"BWRX");
        assert_eq!(
            WarmSection::decode(&bytes, 3, 4),
            Err(WarmError::BadMagic { got: *b"BWRX" })
        );
    }

    /// Another version is refused, the one after this as well as 0.
    #[test]
    fn another_version_is_refused() {
        for v in [0u16, 2] {
            let mut bytes = encoded(&section());
            patch(&mut bytes, 4, &v.to_le_bytes());
            assert_eq!(
                WarmSection::decode(&bytes, 3, 4),
                Err(WarmError::Version { got: v })
            );
        }
    }

    /// A body short by one byte, a header cut short, and no bytes are each
    /// refused as truncated, with what they needed.
    #[test]
    fn a_truncated_body_is_refused() {
        let bytes = encoded(&section());
        assert_eq!(
            WarmSection::decode(&bytes[..bytes.len() - 1], 3, 4),
            Err(WarmError::Truncated {
                len: bytes.len() - 1,
                need: bytes.len() as u64
            })
        );
        assert_eq!(
            WarmSection::decode(&bytes[..HEADER_BYTES - 1], 3, 4),
            Err(WarmError::Truncated {
                len: HEADER_BYTES - 1,
                need: HEADER_BYTES as u64
            })
        );
        assert_eq!(
            WarmSection::decode(&[], 3, 4),
            Err(WarmError::Truncated {
                len: 0,
                need: HEADER_BYTES as u64
            })
        );
    }

    /// A byte past the body is refused.
    #[test]
    fn a_long_body_is_refused() {
        let mut bytes = encoded(&section());
        let want = bytes.len() as u64;
        bytes.push(0);
        assert_eq!(
            WarmSection::decode(&bytes, 3, 4),
            Err(WarmError::TooLong {
                len: bytes.len(),
                want
            })
        );
    }

    /// A section of another shape than the one asked is refused, one with the
    /// same cell count too (2 × 6 for 3 × 4), which only the shape check
    /// tells apart. Mutant: `decode` without the shape check.
    #[test]
    fn another_shape_is_refused() {
        let bytes = encoded(&section());
        for (layers, experts) in [(2, 6), (4, 4), (3, 5)] {
            assert_eq!(
                WarmSection::decode(&bytes, layers, experts),
                Err(WarmError::Shape {
                    layers: 3,
                    experts: 4,
                    want_layers: layers,
                    want_experts: experts
                })
            );
        }
    }

    /// A `long` that is not full, short, empty or long, is refused.
    #[test]
    fn a_long_that_is_not_full_is_refused() {
        for len in [0u32, 11, 13] {
            let mut bytes = encoded(&section());
            patch(&mut bytes, 14, &len.to_le_bytes());
            assert_eq!(
                WarmSection::decode(&bytes, 3, 4),
                Err(WarmError::LongLength { len, want: 12 })
            );
        }
    }

    /// A `recent` that is neither empty nor full is refused.
    #[test]
    fn a_recent_that_is_neither_empty_nor_full_is_refused() {
        for len in [1u32, 11, 13] {
            let mut bytes = encoded(&section());
            patch(&mut bytes, 18, &len.to_le_bytes());
            assert_eq!(
                WarmSection::decode(&bytes, 3, 4),
                Err(WarmError::RecentLength { len, want: 12 })
            );
        }
    }

    /// A `recent` cell that is NaN, infinite or negative is refused with its
    /// index; the same cell positive decodes.
    #[test]
    fn a_recent_cell_no_fold_writes_is_refused() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let mut s = section();
            s.recent[5] = bad;
            assert_eq!(
                WarmSection::decode(&encoded(&s), 3, 4),
                Err(WarmError::RecentCell { index: 5 }),
                "cell {bad:?}"
            );
        }
        let mut s = section();
        s.recent[5] = 1.0;
        assert!(WarmSection::decode(&encoded(&s), 3, 4).is_ok());
    }

    /// A header that claims more bytes than it was given is refused as
    /// truncated before anything is allocated: 2^32 - 1 layers of one expert
    /// is a body of some 51 GB.
    #[test]
    fn a_huge_claim_is_refused_before_it_allocates() {
        let mut bytes = encoded(&section());
        patch(&mut bytes, 6, &u32::MAX.to_le_bytes());
        patch(&mut bytes, 10, &1u32.to_le_bytes());
        patch(&mut bytes, 14, &u32::MAX.to_le_bytes());
        patch(&mut bytes, 18, &u32::MAX.to_le_bytes());
        let need = HEADER_BYTES as u64 + u64::from(u32::MAX) * 12;
        assert_eq!(
            WarmSection::decode(&bytes, u32::MAX, 1),
            Err(WarmError::Truncated {
                len: bytes.len(),
                need
            })
        );
    }
}
