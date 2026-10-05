//! An 8-bit RGB image, and the PNG and JPEG decodes that produce one.
//!
//! The reference opens an image with Pillow and calls `convert("RGB")` before anything else. For
//! the PNG kinds decoded here that conversion is exact and needs no colour arithmetic: RGB is the
//! identity, RGBA drops the alpha (Pillow's RGBA→RGB does not composite), greyscale repeats the
//! grey, and a palette is expanded to its colours. A 16-bit or other PNG is refused by name rather
//! than converted by a rule that might differ from Pillow's. The oracle set is RGB PNGs only, so
//! the gates prove the RGB arm; the other arms follow Pillow's documented conversions unmeasured.
//!
//! A JPEG is decoded by zune-jpeg, YCbCr to RGB and greyscale repeated, the two kinds Pillow opens
//! as `RGB` and `L`; CMYK, YCCK, RGB-coded and other component layouts are refused by name. Pillow
//! decodes with libjpeg-turbo, whose IDCT, chroma upsampling and colour conversion round differently
//! from zune-jpeg's, so a JPEG's pixels are not bit-identical to the reference's: `tests/jpeg.rs`
//! holds the difference to the frontier measured on the fixtures `tools/ref/vision/dump-jpeg.py`
//! wrote. A truncated JPEG is an error, as it is in Pillow; so is a scan with a code no table
//! holds, where Pillow returns an image. A scan an EOI marker ends early still decodes, its missing
//! blocks filled, as in Pillow: zune-jpeg does not report it.

use std::io::Cursor;

use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

use crate::VisionError;

/// What an image file is, by its first bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileKind {
    Png,
    Jpeg,
    Webp,
    Gif,
    /// None of the above.
    Unknown,
}

impl FileKind {
    /// The kind of a file from its signature: PNG's eight bytes, JPEG's SOI and the marker after
    /// it, `RIFF....WEBP`, `GIF87a`/`GIF89a`.
    #[must_use]
    pub fn sniff(bytes: &[u8]) -> FileKind {
        if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
            FileKind::Png
        } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
            FileKind::Jpeg
        } else if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
            FileKind::Webp
        } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
            FileKind::Gif
        } else {
            FileKind::Unknown
        }
    }

    /// The format's name, for errors.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            FileKind::Png => "PNG",
            FileKind::Jpeg => "JPEG",
            FileKind::Webp => "WebP",
            FileKind::Gif => "GIF",
            FileKind::Unknown => "unknown",
        }
    }
}

/// Row-major 8-bit RGB, three bytes per pixel, no padding between rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb8 {
    pub width: usize,
    pub height: usize,
    /// `height * width * 3` bytes: R, G, B of pixel (y, x) at `(y * width + x) * 3`.
    pub data: Vec<u8>,
}

impl Rgb8 {
    /// An image of one colour.
    #[must_use]
    pub fn filled(width: usize, height: usize, rgb: [u8; 3]) -> Rgb8 {
        let mut data = Vec::with_capacity(width * height * 3);
        for _ in 0..width * height {
            data.extend_from_slice(&rgb);
        }
        Rgb8 {
            width,
            height,
            data,
        }
    }

    /// Decode a PNG or JPEG file's bytes to RGB, the decoder chosen by the file's signature; any
    /// other file is refused by its name.
    pub fn from_bytes(bytes: &[u8]) -> Result<Rgb8, VisionError> {
        match FileKind::sniff(bytes) {
            FileKind::Png => Rgb8::from_png(bytes),
            FileKind::Jpeg => Rgb8::from_jpeg(bytes),
            other => Err(VisionError::Format(format!(
                "a {} file; only PNG and JPEG decode here",
                other.name()
            ))),
        }
    }

    /// Decode a JPEG file's bytes to RGB, as the reference's `Image.open(...).convert("RGB")` up to
    /// the decoder's rounding (module doc).
    pub fn from_jpeg(bytes: &[u8]) -> Result<Rgb8, VisionError> {
        // The decoder finishes a baseline file whose EOI marker, or the scan bytes before it, are
        // missing; Pillow calls that file truncated.
        if !reaches_eoi(bytes) {
            return Err(VisionError::Decode(
                "the JPEG ends before its EOI marker: the file is truncated".into(),
            ));
        }
        // Strict: the default decodes on past a code no Huffman table holds, the blocks after it
        // grey, and skips bytes it does not expect.
        let options = DecoderOptions::default()
            .set_strict_mode(true)
            .jpeg_set_out_colorspace(ColorSpace::RGB);
        let mut decoder = JpegDecoder::new_with_options(ZCursor::new(bytes), options);
        let decode = |e: zune_jpeg::errors::DecodeErrors| VisionError::Decode(e.to_string());
        decoder.decode_headers().map_err(decode)?;
        let Some(coded) = decoder.input_colorspace() else {
            return Err(VisionError::Decode(
                "the JPEG's headers name no colour space".into(),
            ));
        };
        let grey = match coded {
            ColorSpace::YCbCr => false,
            ColorSpace::Luma => true,
            other => {
                return Err(VisionError::Format(format!(
                    "a JPEG coded as {other:?}; only YCbCr and greyscale convert to RGB here"
                )));
            }
        };
        if grey {
            decoder.set_options(options.jpeg_set_out_colorspace(ColorSpace::Luma));
        }
        let (width, height) = decoder
            .dimensions()
            .ok_or_else(|| VisionError::Decode("the JPEG's headers name no frame size".into()))?;
        let pixels = decoder.decode().map_err(decode)?;
        let data = if grey {
            pixels.iter().flat_map(|&g| [g, g, g]).collect()
        } else {
            pixels
        };
        if data.len() != width * height * 3 {
            return Err(VisionError::Decode(format!(
                "{} RGB bytes for a {width}x{height} frame",
                data.len()
            )));
        }
        Ok(Rgb8 {
            width,
            height,
            data,
        })
    }

    /// Decode a PNG file's bytes to RGB, as the reference's `Image.open(...).convert("RGB")`.
    pub fn from_png(bytes: &[u8]) -> Result<Rgb8, VisionError> {
        let mut decoder = png::Decoder::new(Cursor::new(bytes));
        // Palette indices become their colours and 1/2/4-bit greys become 8-bit greys; 16-bit
        // samples stay 16-bit and are refused below.
        decoder.set_transformations(png::Transformations::EXPAND);
        let mut reader = decoder
            .read_info()
            .map_err(|e| VisionError::Decode(e.to_string()))?;
        let size = reader
            .output_buffer_size()
            .ok_or_else(|| VisionError::Decode("the PNG's frame size overflows".into()))?;
        let mut buf = vec![0u8; size];
        let info = reader
            .next_frame(&mut buf)
            .map_err(|e| VisionError::Decode(e.to_string()))?;
        let (width, height) = (info.width as usize, info.height as usize);
        if info.bit_depth != png::BitDepth::Eight {
            return Err(VisionError::Format(format!(
                "a {:?}-bit PNG; only 8-bit samples convert to RGB here",
                info.bit_depth
            )));
        }
        let rows = buf.chunks_exact(info.line_size).take(height);
        let mut data = Vec::with_capacity(width * height * 3);
        match info.color_type {
            png::ColorType::Rgb => rows.for_each(|r| data.extend_from_slice(&r[..width * 3])),
            png::ColorType::Rgba => rows.for_each(|r| {
                r[..width * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .for_each(|p| data.extend_from_slice(&p[..3]));
            }),
            png::ColorType::Grayscale => rows.for_each(|r| {
                r[..width]
                    .iter()
                    .for_each(|&g| data.extend_from_slice(&[g, g, g]));
            }),
            png::ColorType::GrayscaleAlpha => rows.for_each(|r| {
                r[..width * 2]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .for_each(|p| data.extend_from_slice(&[p[0], p[0], p[0]]));
            }),
            other => {
                return Err(VisionError::Format(format!(
                    "a {other:?} PNG after palette expansion"
                )));
            }
        }
        if data.len() != width * height * 3 {
            return Err(VisionError::Decode(format!(
                "{} RGB bytes for a {width}x{height} frame",
                data.len()
            )));
        }
        Ok(Rgb8 {
            width,
            height,
            data,
        })
    }
}

/// Whether a JPEG's marker sequence reaches an EOI. Segments are walked by their lengths; any other
/// byte that is not `0xFF` is skipped (a scan's entropy-coded bytes, or stray bytes libjpeg skips
/// too), as are a stuffed `FF 00`, fill `FF`s and the markers that carry no length. Whatever follows
/// the first EOI is not read.
fn reaches_eoi(bytes: &[u8]) -> bool {
    let mut i = 2;
    while i + 1 < bytes.len() {
        if bytes[i] != 0xFF {
            i += 1;
            continue;
        }
        match bytes[i + 1] {
            0xD9 => return true,
            0xFF => i += 1,
            0x00 | 0x01 | 0xD0..=0xD8 => i += 2,
            _ => {
                let Some(&[hi, lo]) = bytes.get(i + 2..i + 4) else {
                    return false;
                };
                i += 2 + usize::from(u16::from_be_bytes([hi, lo]));
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{FileKind, Rgb8};
    use crate::VisionError;

    #[test]
    fn file_kinds_by_signature() {
        let webp = b"RIFF\x24\0\0\0WEBPVP8 ";
        for (bytes, kind) in [
            (&b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR"[..], FileKind::Png),
            (&[0xFF, 0xD8, 0xFF, 0xE0][..], FileKind::Jpeg),
            (&webp[..], FileKind::Webp),
            (b"GIF89a\x01\0\x01\0", FileKind::Gif),
            (b"GIF87a\x01\0\x01\0", FileKind::Gif),
            (b"BM\x3a\0\0\0", FileKind::Unknown),
            (b"RIFF\x24\0\0\0WAVEfmt ", FileKind::Unknown),
            (&[0xFF, 0xD8][..], FileKind::Unknown),
            (b"", FileKind::Unknown),
        ] {
            assert_eq!(FileKind::sniff(bytes), kind, "{bytes:02x?}");
        }
        for (bytes, name) in [
            (&webp[..], "WebP"),
            (b"GIF89a\x01\0\x01\0", "GIF"),
            (b"BM\x3a\0", "unknown"),
        ] {
            match Rgb8::from_bytes(bytes) {
                Err(VisionError::Format(msg)) => {
                    assert!(msg.contains(&format!("a {name} file")), "{msg}");
                }
                other => panic!("{name}: {other:?}"),
            }
        }
    }
}
