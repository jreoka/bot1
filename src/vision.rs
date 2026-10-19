//! What the bot sees, and the coarse fingerprint it reduces that to.
//! Ported from `capture.py`'s `gray_signature` and `novelty.py`'s
//! `_signature_key`.
//!
//! The fingerprint is the single most consequential choice in the whole reward:
//! it decides what counts as "a new situation". 256 cells at 16 levels is fine
//! enough that walking from one end of a corridor to the other is a hundred new
//! states rather than one, and coarse enough that a flickering texture is not.

/// One frame stack: `(channels, height, width)`, row-major, in `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

impl Observation {
    pub fn zeros(channels: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            height,
            width,
            data: vec![0.0; channels * height * width],
        }
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (self.channels, self.height, self.width)
    }

    pub fn channel(&self, index: usize) -> &[f32] {
        let plane = self.height * self.width;
        &self.data[index * plane..(index + 1) * plane]
    }

    pub fn channel_mut(&mut self, index: usize) -> &mut [f32] {
        let plane = self.height * self.width;
        &mut self.data[index * plane..(index + 1) * plane]
    }

    /// Mean of the first `count` channels, as one `(height, width)` plane.
    ///
    /// The session fingerprints this rather than a derived channel, so the
    /// reward sees the picture and not the motion channel it computed from it.
    pub fn mean_of_channels(&self, count: usize) -> Vec<f32> {
        let count = count.max(1).min(self.channels);
        let plane = self.height * self.width;
        let mut out = vec![0.0f32; plane];
        for channel in 0..count {
            let source = self.channel(channel);
            for (target, value) in out.iter_mut().zip(source) {
                *target += value;
            }
        }
        let scale = 1.0 / count as f32;
        for value in out.iter_mut() {
            *value *= scale;
        }
        out
    }
}

/// A small colour picture of one frame: RGB in `[0, 1]`, row-major.
///
/// Only ever drawn. The model gets grey and has to keep getting grey - the
/// weights were trained on luminance - but a person reading a dashboard does not:
/// a corridor, a health bar and a hazard are told apart by hue long before they
/// are told apart by brightness, and the grey stack has already thrown that away.
/// So this is a second, much smaller picture of the same frame, made for the eye.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Picture {
    pub width: usize,
    pub height: usize,
    /// `width * height * 3` values in `[0, 1]`, RGB order.
    pub rgb: Vec<f32>,
}

impl Picture {
    pub fn new(width: usize, height: usize, rgb: Vec<f32>) -> Self {
        Self { width, height, rgb }
    }

    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0 || self.rgb.len() < self.width * self.height * 3
    }

    /// One pixel. Out-of-range reads come back black rather than panicking: a
    /// malformed picture must not stop a run.
    pub fn pixel(&self, row: usize, column: usize) -> [f32; 3] {
        if row >= self.height || column >= self.width {
            return [0.0, 0.0, 0.0];
        }
        let index = (row * self.width + column) * 3;
        match self.rgb.get(index..index + 3) {
            Some([r, g, b]) => [*r, *g, *b],
            _ => [0.0, 0.0, 0.0],
        }
    }

    /// Rec. 601 luminance, which is what the shade ramp and the model both use.
    pub fn luma(&self, row: usize, column: usize) -> f32 {
        let [red, green, blue] = self.pixel(row, column);
        0.299 * red + 0.587 * green + 0.114 * blue
    }

    /// A grey plane drawn as a colour picture, so the renderer has one resize.
    ///
    /// A frame the environment cannot produce in colour still gets drawn: three
    /// equal channels read as grey through the same code path.
    pub fn from_grey(plane: &[f32], height: usize, width: usize) -> Self {
        if height == 0 || width == 0 || plane.len() < height * width {
            return Self::default();
        }
        let mut rgb = Vec::with_capacity(height * width * 3);
        for value in plane.iter().take(height * width) {
            rgb.push(*value);
            rgb.push(*value);
            rgb.push(*value);
        }
        Self::new(width, height, rgb)
    }

    /// One channel as a `(height, width)` plane, for the area resizer.
    fn plane(&self, channel: usize) -> Vec<f32> {
        self.rgb.iter().skip(channel).step_by(3).copied().collect()
    }

    /// Area-averaged to `rows x columns` - the size the panel has room to draw.
    pub fn resized(&self, rows: usize, columns: usize) -> Self {
        if rows == 0 || columns == 0 || self.is_empty() {
            return Self::default();
        }
        if rows == self.height && columns == self.width {
            return self.clone();
        }
        let mut planes: Vec<Vec<f32>> = (0..3)
            .map(|channel| area_resize(&self.plane(channel), self.height, self.width, rows, columns))
            .collect();
        let mut rgb = Vec::with_capacity(rows * columns * 3);
        for index in 0..rows * columns {
            for plane in planes.iter_mut() {
                rgb.push(plane[index]);
            }
        }
        Self::new(columns, rows, rgb)
    }

    /// A captured BGRA frame, averaged down to `rows x columns`.
    ///
    /// Each cell averages a small kernel centred on it rather than every pixel it
    /// covers: a 32x32 thumbnail of a 1920x1080 frame would otherwise read two
    /// million pixels to produce a thousand, twenty times a second, and none of
    /// that fidelity survives being drawn at 32 columns wide. `size / 4` capped at
    /// eight keeps a thin feature - a health bar, a crosshair - from disappearing
    /// between samples.
    pub fn from_bgra(
        bgra: &[u8],
        width: usize,
        height: usize,
        columns: usize,
        rows: usize,
    ) -> Self {
        if width == 0 || height == 0 || columns == 0 || rows == 0 || bgra.len() < width * height * 4
        {
            return Self::default();
        }
        let column_step = (width / columns).max(1);
        let row_step = (height / rows).max(1);
        let kernel_columns = (column_step / 4).clamp(1, 8);
        let kernel_rows = (row_step / 4).clamp(1, 8);
        let mut rgb = vec![0.0f32; rows * columns * 3];
        for out_row in 0..rows {
            let middle_row = out_row * row_step + row_step / 2;
            let first_row = middle_row.saturating_sub(kernel_rows / 2).min(height - 1);
            let last_row = (first_row + kernel_rows).min(height);
            for out_column in 0..columns {
                let middle_column = out_column * column_step + column_step / 2;
                let first_column = middle_column
                    .saturating_sub(kernel_columns / 2)
                    .min(width - 1);
                let last_column = (first_column + kernel_columns).min(width);
                let mut sums = [0.0f32; 3];
                let mut count = 0.0f32;
                for row in first_row..last_row {
                    for column in first_column..last_column {
                        let pixel = (row * width + column) * 4;
                        sums[0] += bgra[pixel + 2] as f32;
                        sums[1] += bgra[pixel + 1] as f32;
                        sums[2] += bgra[pixel] as f32;
                        count += 1.0;
                    }
                }
                let scale = if count > 0.0 { 1.0 / (count * 255.0) } else { 0.0 };
                let cell = (out_row * columns + out_column) * 3;
                for channel in 0..3 {
                    rgb[cell + channel] = (sums[channel] * scale).clamp(0.0, 1.0);
                }
            }
        }
        Self::new(columns, rows, rgb)
    }
}

/// Number of cells in the coarse fingerprint.
pub const GRID: usize = 256;
/// Levels each cell is quantised into.
pub const LEVELS: usize = 16;

/// Tiny fingerprint of a frame, used by novelty and reset detection.
///
/// Deliberately blunt: a sharp fingerprint makes every frame look like a new
/// situation, which turns count-based novelty into a flat per-step bonus and
/// makes reset detection fire constantly.
///
/// The Python resizes with `cv2.INTER_AREA`; this is the same area average
/// written out. `cv2` does it in fixed-point arithmetic, so the two can differ
/// by one part in 255 on a boundary pixel - which is not nothing for a
/// quantised fingerprint, but it changes only which side of a cell boundary a
/// frame lands on, never whether the fingerprint is meaningful. A run trains
/// from its own fingerprints either way.
pub fn gray_signature(frame: &[f32], height: usize, width: usize, size: usize) -> Vec<f32> {
    // The Python detects the range rather than assuming it: a fingerprint of
    // all zeros makes every frame look identical, which silently removes the
    // entire episodic reward.
    let scale = if frame.iter().copied().fold(0.0f32, f32::max) > 1.5 {
        255.0
    } else {
        1.0
    };
    let resized = area_resize(frame, height, width, size, size);
    resized.into_iter().map(|value| value / scale).collect()
}

/// Area-average resize, which is what `cv2.INTER_AREA` means when shrinking.
pub fn area_resize(
    frame: &[f32],
    height: usize,
    width: usize,
    out_height: usize,
    out_width: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_height * out_width];
    let row_scale = height as f32 / out_height as f32;
    let column_scale = width as f32 / out_width as f32;
    for out_row in 0..out_height {
        let top = out_row as f32 * row_scale;
        let bottom = (out_row + 1) as f32 * row_scale;
        let first_row = top.floor() as usize;
        let last_row = (bottom.ceil() as usize).min(height);
        for out_column in 0..out_width {
            let left = out_column as f32 * column_scale;
            let right = (out_column + 1) as f32 * column_scale;
            let first_column = left.floor() as usize;
            let last_column = (right.ceil() as usize).min(width);
            let mut total = 0.0f32;
            let mut weight = 0.0f32;
            for row in first_row..last_row {
                let row_overlap = (bottom.min(row as f32 + 1.0) - top.max(row as f32)).max(0.0);
                if row_overlap <= 0.0 {
                    continue;
                }
                for column in first_column..last_column {
                    let column_overlap =
                        (right.min(column as f32 + 1.0) - left.max(column as f32)).max(0.0);
                    if column_overlap <= 0.0 {
                        continue;
                    }
                    let area = row_overlap * column_overlap;
                    total += frame[row * width + column] * area;
                    weight += area;
                }
            }
            out[out_row * out_width + out_column] = if weight > 0.0 {
                total / weight
            } else {
                0.0
            };
        }
    }
    out
}

/// Hash a frame fingerprint to an integer.
///
/// The Python uses the built-in `hash` of the packed bits, which is salted per
/// process, so its keys are not stable across runs either - only equality
/// within a run matters, and this is the same thing with a fixed hash so a
/// Rust run is reproducible.
pub fn signature_key(signature: &[f32]) -> u64 {
    let columns = (signature.len() / GRID).max(1);
    let mut cells = [0u8; GRID];
    for (cell, slot) in cells.iter_mut().enumerate() {
        let start = cell * columns;
        let end = ((cell + 1) * columns).min(signature.len());
        if start >= end {
            continue;
        }
        let mean: f32 = signature[start..end].iter().sum::<f32>() / (end - start) as f32;
        let quantised = ((mean * LEVELS as f32) as i64).clamp(0, LEVELS as i64 - 1) as u8;
        *slot = quantised;
    }
    // Four bits per cell, most significant first, packed the way `np.packbits`
    // packs them - so the bytes hashed here are the bytes the Python hashes.
    let mut packed = [0u8; GRID / 2];
    for (index, cell) in cells.iter().enumerate() {
        let byte = index / 2;
        if index % 2 == 0 {
            packed[byte] |= cell << 4;
        } else {
            packed[byte] |= cell;
        }
    }
    fnv1a(&packed)
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    // A final mix, so that fingerprints differing in the last byte do not land
    // in adjacent buckets of any table that indexes by the low bits.
    hash ^= hash >> 32;
    hash = hash.wrapping_mul(0xd6e8feb86659fd93);
    hash ^ (hash >> 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_resize_averages_the_block() {
        // 4x4 all ones, halved: every output pixel is exactly 1.
        let frame = vec![1.0f32; 16];
        let out = area_resize(&frame, 4, 4, 2, 2);
        assert_eq!(out, vec![1.0, 1.0, 1.0, 1.0]);
        // A single bright pixel spreads over the cells that cover it.
        let mut frame = vec![0.0f32; 16];
        frame[0] = 4.0;
        let out = area_resize(&frame, 4, 4, 2, 2);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn identical_frames_hash_the_same_and_different_frames_do_not() {
        let frame = vec![0.5f32; 48 * 48];
        let first = gray_signature(&frame, 48, 48, 32);
        let second = gray_signature(&frame, 48, 48, 32);
        assert_eq!(signature_key(&first), signature_key(&second));

        let mut other = frame.clone();
        for value in other.iter_mut().take(48 * 48 / 2) {
            *value = 0.0;
        }
        let different = gray_signature(&other, 48, 48, 32);
        assert_ne!(signature_key(&first), signature_key(&different));
    }

    #[test]
    fn a_bgra_frame_becomes_a_thumbnail_in_the_right_colours() {
        // 4x2, left half pure blue and right half pure red, in BGRA order:
        // dropping an alpha-free buffer into the wrong channels is exactly the bug
        // that makes a picture look haunted instead of wrong.
        let mut bgra = Vec::new();
        for _ in 0..2 {
            for column in 0..4 {
                if column < 2 {
                    bgra.extend([255, 0, 0, 255]);
                } else {
                    bgra.extend([0, 0, 255, 255]);
                }
            }
        }
        let picture = Picture::from_bgra(&bgra, 4, 2, 2, 1);
        assert_eq!((picture.width, picture.height), (2, 1));
        assert_eq!(picture.pixel(0, 0), [0.0, 0.0, 1.0]);
        assert_eq!(picture.pixel(0, 1), [1.0, 0.0, 0.0]);
        // A buffer that is not the shape it claims to be is empty, not a panic.
        assert!(Picture::from_bgra(&bgra[..4], 4, 2, 2, 1).is_empty());
        assert!(Picture::from_bgra(&bgra, 0, 2, 2, 1).is_empty());
    }

    #[test]
    fn a_thumbnail_keeps_its_colours_when_it_is_resized() {
        // Two pixels: red then blue, stretched to four columns.
        let picture = Picture::new(2, 1, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let resized = picture.resized(1, 4);
        assert_eq!(resized.width, 4);
        // The left half stays red and the right half stays blue: a resize that
        // averaged the channels together would grey the picture out.
        assert!(resized.pixel(0, 0)[0] > 0.9 && resized.pixel(0, 0)[2] < 0.1);
        assert!(resized.pixel(0, 3)[2] > 0.9 && resized.pixel(0, 3)[0] < 0.1);
    }

    #[test]
    fn a_grey_plane_becomes_a_picture_with_three_equal_channels() {
        let picture = Picture::from_grey(&[0.0, 0.5, 1.0, 0.25], 2, 2);
        assert_eq!(picture.pixel(0, 1), [0.5, 0.5, 0.5]);
        assert!((picture.luma(1, 0) - 1.0).abs() < 1e-6);
        assert!(Picture::from_grey(&[0.0], 4, 4).is_empty());
    }

    #[test]
    fn a_fingerprint_is_the_size_the_quantiser_expects() {
        let frame = vec![0.25f32; 32 * 32];
        let signature = gray_signature(&frame, 32, 32, 32);
        assert_eq!(signature.len(), 1024);
        // Two cells per byte, so 256 cells at four bits each.
        assert_eq!(GRID / 2, 128);
    }
}
