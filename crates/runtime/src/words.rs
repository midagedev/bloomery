//! The resident word layout of a flat weight stream: the one rule that says
//! how many little-endian u32 words a byte run takes on the card, shared by
//! the loader's staging (the gpu crate's upload path) and the packings the
//! gates pin against it. Host-only arithmetic — no reader and no kernel
//! lives here, only the count both sides must agree on for a streamed upload
//! to hold the bytes a scalar packing would.

/// Words of the little-endian u32 stream a byte run `len` bytes long takes
/// as one flat weight stream of `rows` rows: the words of the bytes alone
/// (`ceil(len / 4)` — a partial final word zero-pads), rounded up to a whole
/// number of words per row, so the stream is `rows` rows of one word count
/// and a row's bytes are never padded inside it. `None` when `rows` is zero
/// or the rounded count passes u64.
///
/// The device bytes of such a stream on a little-endian host are the run's
/// bytes verbatim followed by a zero tail to `4 * words`: staging that
/// copies the bytes and leaves the tail zero holds the same words as
/// packing the bytes into u32s and resizing.
#[must_use]
pub fn stream_words(len: u64, rows: u64) -> Option<u64> {
    if rows == 0 {
        return None;
    }
    len.div_ceil(4).div_ceil(rows).checked_mul(rows)
}

#[cfg(test)]
mod tests {
    use super::stream_words;

    /// The scalar packing the gate binaries carry as `bytes_to_words`:
    /// every four bytes one little-endian word, a partial final word
    /// zero-padded. This copy pins the same rule here, where it can be
    /// compared against the staging layout natively.
    fn words_of(b: &[u8]) -> Vec<u32> {
        b.chunks(4)
            .map(|c| {
                let mut w = [0u8; 4];
                w[..c.len()].copy_from_slice(c);
                u32::from_le_bytes(w)
            })
            .collect()
    }

    /// The packing the loader's staging replaces — `words_of` of the bytes,
    /// zero-resized to a whole number of words per row by the loader's own
    /// resize expression — against the bytes the staged stream holds: the
    /// bytes verbatim, a zero tail to [`stream_words`]'s count, read back as
    /// words. Equal streams mean an upload that copies the raw bytes into a
    /// zeroed buffer lands the words a scalar packing would. The two counts
    /// are computed by different expressions on purpose: the resize's own
    /// arithmetic here, [`stream_words`] on the staged side — deriving both
    /// from [`stream_words`] would compare the function with itself and pass
    /// under any rule. Lengths around the word and row boundaries: the empty
    /// run, a byte short of a word, a word, a byte past it, a word short of
    /// a round four-thousand words and a round one, and rows whose tails pad
    /// inside the last word, to a whole word, and to whole words per row.
    #[test]
    fn a_zero_tail_stream_holds_the_scalar_packing() {
        let packed = |bytes: &[u8], rows: u64| {
            let mut words = words_of(bytes);
            words.resize(words.len().div_ceil(rows as usize) * rows as usize, 0);
            words
        };
        let staged = |bytes: &[u8], rows: u64| {
            let words = usize::try_from(stream_words(bytes.len() as u64, rows).unwrap())
                .expect("a test length fits usize");
            let mut staged = bytes.to_vec();
            staged.resize(words * 4, 0);
            staged
                .chunks(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        for (len, rows) in [
            (0, 1),
            (1, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (3, 4),
            (4095, 1),
            (4096, 1),
            (4099, 7),
            (110, 1),
            (220, 2),
            (8192, 3),
        ] {
            let bytes: Vec<u8> = (0..len).map(|i| (i * 37 % 251) as u8).collect();
            assert_eq!(
                packed(&bytes, rows),
                staged(&bytes, rows),
                "{len} bytes in {rows} rows"
            );
        }
    }

    /// The count refuses a zero row count (the divisor of the per-row
    /// rounding) and an empty run takes no words in one row.
    #[test]
    fn the_count_refuses_zero_rows_and_empties() {
        assert_eq!(stream_words(0, 1), Some(0));
        assert_eq!(stream_words(0, 4), Some(0));
        assert_eq!(stream_words(7, 0), None);
    }
}
