//! An 8-bit RGB image, and the PNG decode that produces one.
//!
//! The reference opens an image with Pillow and calls `convert("RGB")` before anything else. For
//! the PNG kinds decoded here that conversion is exact and needs no colour arithmetic: RGB is the
//! identity, RGBA drops the alpha (Pillow's RGBA→RGB does not composite), greyscale repeats the
//! grey, and a palette is expanded to its colours. A 16-bit or other PNG is refused by name rather
//! than converted by a rule that might differ from Pillow's. The oracle set is RGB PNGs only, so
//! the gates prove the RGB arm; the other arms follow Pillow's documented conversions unmeasured.

use std::io::Cursor;

use crate::VisionError;

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
