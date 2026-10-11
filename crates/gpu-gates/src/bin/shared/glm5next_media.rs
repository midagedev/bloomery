//! The image-row clause of GLM-5.3-Flash's prompt batch (`gate_glm5next_e2e --only media`): a media
//! prompt call (`bloomery_gpu_glm5next::prefill_media`) fed the oracle's own image rows, so the
//! injection is proven apart from our tower.
//!
//! The oracle is mainline's mtmd on the respelled fixture copy (`refset::arch::glm5next::vis`): set B's
//! `embd` rows of each image (the tower's output, f32) rounded to bf16 as the card tower writes them, the
//! chat ids of the prompts c1-c3, and set C's `result_norm` of the CUDA run and of its CPU twin. The
//! engine's head holds the call's last position, so each prompt is held on that row:
//! - the row lies within `TWIN` times the twin's own distance from the CUDA row (the Qwen seats' rule: the
//!   twin is the floor any two realizations of the rule lie apart), and
//! - it is nearer the CUDA row than the control call is: the same ids through the plain prompt, whose image
//!   positions then read the `<|image|>` token's embedding, so a band that anything passes is caught;
//! - a media call that is refused by name moves nothing: a span of another width, an id that is not the
//!   image token, a load not set for media.
//!
//! The serve seat's media call is [`seat`]'s (`gate_glm5next_e2e --only media-seat`, behind the `vision`
//! feature): our tower's rows of procedural images, through the seat's own steps, held to the body's
//! media call on the same rows.

use bloomery_gpu::GpuError;
use bloomery_gpu::media::MediaSpan;
use bloomery_gpu_gates::{GateError, verdict};
use bloomery_gpu_glm5next::{Glm5nextModel, prefill, prefill_media, set_prefill_media};
use refset::arch::glm5next::vis::{CHATIDS, HIDDEN, PROFILE, TAPS};
use refset::clefvis::{ClefvisSet, EMBD};
use std::ops::Range;

/// PIN(2026-10-11): the engine's last-row distance from the CUDA row is held within `TWIN` times the CPU
/// twin's. Fixed before the engine met the sets, from the one measurement of this fixture's text path: on the
/// five ids, mainline CPU against CUDA `result_norm` is 0.155 and ik against mainline CPU 0.262 (ratio 1.69),
/// both dominated by near-tie router flips (the fixture's weights are seeded noise), so an engine that rounds
/// as ik does sits about 1.7 twin from CUDA. At 2.5 that estimate is 0.68 of the band; the bf16 rounding of
/// the image rows adds a term no set measures (the seat has no bf16-rows control), so the box round may
/// move this constant once with its attribution.
const TWIN: f64 = 2.5;

/// The cases, in the order of [`HIDDEN`]'s CUDA sets (the twin of case `n` is set `n + 3`).
const CASES: [&str; 3] = ["c1", "c2", "c3"];

/// One prompt of the oracle: its ids, its image spans with their bf16 rows, and the last position's
/// `result_norm` row of the CUDA run and of the CPU twin.
struct Case {
    ids: Vec<u32>,
    spans: Vec<(Range<usize>, Vec<u16>)>,
    cuda: Vec<f32>,
    twin: Vec<f32>,
}

impl Case {
    /// Prompt `n` of set C: the chat ids, each span an image-token run holding its image's `embd` rows from
    /// the tower set, and the last row of the two `result_norm` arrays.
    fn load(n: usize, name: &'static str, tower: &ClefvisSet) -> Result<Case, GateError> {
        let width = PROFILE.n_embd;
        let open =
            |fam: &'static refset::family::Family, set: &str| ClefvisSet::open(&fam.path(set), fam);
        let hidden = open(&HIDDEN, HIDDEN.sets[n])?;
        let twin = open(&HIDDEN, HIDDEN.sets[n + 3])?;
        let chat = open(&CHATIDS, CHATIDS.sets[n])?;
        let ids = chat.ids()?;
        if hidden.n_ids()? != ids.len() {
            return Err(format!(
                "{name}: the hidden set holds {} ids, the chat ids {}",
                hidden.n_ids()?,
                ids.len()
            )
            .into());
        }
        let mut spans = Vec::new();
        for s in &hidden.spans {
            let at = s.at..s.at + s.len;
            let pad = Some(PROFILE.image_pad_id);
            if ids
                .get(at.clone())
                .is_none_or(|run| run.iter().any(|&i| Some(i) != pad))
            {
                return Err(format!("{name}: span {} is not an image-token run", s.index).into());
            }
            let (_, rows) = tower.tap(&s.image, EMBD)?;
            if rows.len() != s.len * width {
                return Err(format!("{name}: {} tower rows for {}", rows.len(), s.image).into());
            }
            let rows = rows
                .iter()
                .map(|&x| cuda_device::convert::f32_to_bf16_rne(x))
                .collect();
            spans.push((at, rows));
        }
        let last = |v: Vec<f32>| -> Result<Vec<f32>, GateError> {
            if v.len() != ids.len() * width {
                return Err(format!("{name}: result_norm rows are not the ids' count").into());
            }
            Ok(v[v.len() - width..].to_vec())
        };
        Ok(Case {
            cuda: last(hidden.result_norm()?)?,
            twin: last(twin.result_norm()?)?,
            ids,
            spans,
        })
    }

    /// The call's spans.
    fn media(&self) -> Vec<MediaSpan<'_>> {
        self.spans
            .iter()
            .map(|(at, rows)| MediaSpan {
                at: at.clone(),
                rows,
            })
            .collect()
    }
}

/// `‖a − b‖ / ‖b‖`.
fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for (x, y) in a.iter().zip(b) {
        num += f64::from(x - y).powi(2);
        den += f64::from(*y).powi(2);
    }
    (num / den).sqrt()
}

/// Whether `r` is a refusal of `m` naming `what`, with the model where it stood.
fn refused(m: &Glm5nextModel, r: Result<u32, GpuError>, what: &str, from: u32) -> bool {
    let named = matches!(&r, Err(e) if e.to_string().contains(what));
    let stood = m.pos() == from;
    println!(
        "  refusal {what:?}: {} {}",
        if named { "named" } else { "NOT NAMED" },
        if stood { "unmoved" } else { "MOVED" }
    );
    named && stood
}

/// (im): the media call on prompts c1-c3 and the refusals, on a model open on the batch feed with room for
/// the longest prompt.
pub fn media_clause(m: &mut Glm5nextModel) -> Result<bool, GateError> {
    let tower = ClefvisSet::open(&TAPS.path(TAPS.sets[0]), &TAPS)?;
    let mut ok = true;
    for (n, name) in CASES.into_iter().enumerate() {
        let case = Case::load(n, name, &tower)?;
        set_prefill_media(m, Some(PROFILE.image_pad_id))?;
        m.reset()?;
        prefill_media(m, &case.ids, &case.media())?;
        let media = rel_l2(&m.head_normed()?, &case.cuda);
        m.reset()?;
        prefill(m, &case.ids)?;
        let control = rel_l2(&m.head_normed()?, &case.cuda);
        let twin = rel_l2(&case.twin, &case.cuda);
        let pass = media <= TWIN * twin && media < control;
        println!(
            "(im) {name}: {} ids, {} image positions; last-row rel L2 from CUDA: media {media:.4}, twin \
             {twin:.4} (band {TWIN} x = {:.4}), control {control:.4} {}",
            case.ids.len(),
            case.spans.iter().map(|(at, _)| at.len()).sum::<usize>(),
            TWIN * twin,
            verdict(pass)
        );
        ok &= pass;
    }
    let case = Case::load(0, CASES[0], &tower)?;
    let span = case
        .spans
        .first()
        .ok_or("the first prompt holds no image")?;
    let width = PROFILE.n_embd;
    m.reset()?;
    let from = m.pos();
    let narrow = [MediaSpan {
        at: span.0.clone(),
        rows: &span.1[..span.1.len() - width / 2],
    }];
    let r = prefill_media(m, &case.ids, &narrow);
    ok &= refused(m, r, "the call holds a span whole or not at all", from);
    let mut ids = case.ids.clone();
    ids[span.0.start] = ids[0];
    let r = prefill_media(m, &ids, &case.media());
    ok &= refused(m, r, "not the image token", from);
    set_prefill_media(m, None)?;
    let r = prefill_media(m, &case.ids, &case.media());
    ok &= refused(m, r, "not set for media", from);
    set_prefill_media(m, Some(PROFILE.image_pad_id))?;
    Ok(ok)
}

/// (sm): the serve seat's media call (`bloomery-serve --mmproj` on a GLM file) against the body's own,
/// on our tower's rows of procedural images. The seat's steps are the resident encoder's
/// `merged_rows` (the one call the Qwen3 and GLM seats make for their rows), then
/// `Session::prompt_media` over the spans the rows make (`serve_seats::glm`'s `prefill_media`); the
/// body's call is `prefill_media` with spans put together from the images' known sizes. The numbers' quality is the tower's gate's and (im)'s, not this
/// one's: any image will do.
#[cfg(feature = "vision")]
pub mod seat {
    use std::path::Path;

    use app::Session;
    use bloomery_gpu::media::MediaSpan;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::vision_seat::{
        self, Flags, Image, Opened, PleImage, Resident, TextFacts,
    };
    use bloomery_gpu_gates::{GateError, verdict};
    use bloomery_gpu_glm5next::{Body, prefill, prefill_media, set_prefill_media};
    use runtime::Want;
    use vision::Prepared;
    use vision::Rgb8;
    use vision::arch::glm5v::{
        self,
        media::{BEGIN_OF_IMAGE_ID, END_OF_IMAGE_ID, IMAGE_TOKEN_ID},
    };

    /// The tower's cap in merged tokens: above both images' grids.
    const CAP: usize = 256;

    /// A smaller cap: the tower of the refusal clause holds fewer tokens than the first image's grid.
    const SMALL_CAP: usize = 64;

    /// The two images, `(width, height)` in pixels: 224 x 336 is a grid of 8 x 12 merged tokens
    /// (28-pixel cells), 448 x 112 one of 16 x 4 — another aspect, and under [`SMALL_CAP`] only the
    /// second.
    const SIZES: [(usize, usize); 2] = [(224, 336), (448, 112)];

    /// The tokens each image takes: the grids above.
    const TOKENS: [usize; 2] = [96, 64];

    /// Ordinary text ids around the spans.
    const TEXT: [u32; 6] = [785, 6722, 315, 9621, 374, 1879];

    /// An image of `w` x `h` whose every cell differs: channel `c` of pixel `(x, y)` a mix of its
    /// position and `salt`.
    fn procedural(w: usize, h: usize, salt: usize) -> Rgb8 {
        let mut image = Rgb8::filled(w, h, [0, 0, 0]);
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    image.data[(y * w + x) * 3 + c] =
                        ((x * (c + 3) + y * (7 - c) + salt * 41 + x * y / 17) % 256) as u8;
                }
            }
        }
        image
    }

    /// The seat's flags for `mmproj` at a cap of `cap` merged tokens.
    fn flags(mmproj: &Path, cap: usize) -> Result<Flags, GateError> {
        let (file, max) = (mmproj.display().to_string(), cap.to_string());
        let args = ["--mmproj", &file, "--image-max-tokens", &max];
        let mut it = args.into_iter();
        let mut flags = Flags::default();
        while let Some(flag) = it.next() {
            flags.take(flag, &mut it)?;
        }
        Ok(flags)
    }

    /// The encoder file seated as the seat seats it ([`vision_seat::open_seat`]), held against the
    /// text model's width and vocabulary `vocab`.
    fn opened(
        mmproj: &Path,
        cap: usize,
        n_embd: usize,
        vocab: &dyn Fn(u32) -> Option<String>,
    ) -> Result<Opened, GateError> {
        let flags = flags(mmproj, cap)?;
        let facts = TextFacts {
            n_embd,
            imrope: None,
            ple: PleImage::Absent,
        };
        vision_seat::open_seat(
            &flags,
            flags.mmproj()?,
            Place::A,
            &[glm5v::PROJECTOR_TYPE],
            &facts,
            vocab,
        )?
        .ok_or_else(|| "--mmproj opened no encoder".into())
    }

    /// The call's ids: text, then each image as the server expands its placeholder (`<|begin_of_image|>`,
    /// `span_len` image tokens, `<|end_of_image|>`) with text after it; and each span's first position.
    fn call(images: &[Prepared]) -> (Vec<u32>, Vec<usize>) {
        let mut ids = TEXT.to_vec();
        let mut at = Vec::new();
        for (i, p) in images.iter().enumerate() {
            ids.push(BEGIN_OF_IMAGE_ID);
            at.push(ids.len());
            ids.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, p.span_len));
            ids.push(END_OF_IMAGE_ID);
            ids.extend_from_slice(&TEXT[i..i + 3]);
        }
        (ids, at)
    }

    /// The bf16 value of `bits` as f32.
    fn bf16(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    /// `‖a − b‖ / ‖b‖`.
    fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
        let (mut num, mut den) = (0.0_f64, 0.0_f64);
        for (x, y) in a.iter().zip(b) {
            num += f64::from(x - y).powi(2);
            den += f64::from(*y).powi(2);
        }
        (num / den).sqrt()
    }

    /// (sm): on the session `s` (the batch feed, room for the call), the encoder file `mmproj` and
    /// the text model's width `n_embd` and vocabulary `vocab`.
    pub fn clauses(
        s: &mut Session<Body>,
        mmproj: &Path,
        n_embd: usize,
        vocab: &dyn Fn(u32) -> Option<String>,
    ) -> Result<bool, GateError> {
        let at = opened(mmproj, CAP, n_embd, vocab)?;
        println!(
            "(sm) encoder file {}: seat {} on {}, cap {} tokens, card figure {} B",
            at.path.display(),
            at.seat.word(),
            at.seat.name,
            at.cap,
            at.card.total()
        );
        set_prefill_media(s.model_mut(), Some(at.media.image_token()))?;
        let ctx = s.model().gpu().context().clone();
        let mut tower = Resident::open(&ctx, &at)?;
        let images: Vec<Prepared> = SIZES
            .iter()
            .enumerate()
            .map(|(i, &(w, h))| at.media.prepare(&procedural(w, h, i)))
            .collect::<Result<_, _>>()?;
        let tokens: Vec<usize> = images.iter().map(|p| p.span_len).collect();
        let mut ok = tokens == TOKENS;
        println!(
            "(sm) the images' spans: {tokens:?} tokens (want {TOKENS:?}, the size rule's grids of \
             {SIZES:?} pixels) {}",
            verdict(ok)
        );
        let (ids, starts) = call(&images);

        // The tower refuses patches planned past its cap by name, and the model's card is bound
        // back after the refusal: a request cannot reach it (the size rule shrinks every canvas
        // into the cap the seat's limits name), so this holds the tower's own check.
        let small = opened(mmproj, SMALL_CAP, n_embd, vocab)?;
        let mut narrow = Resident::open(&ctx, &small)?;
        let from = s.model().pos();
        let refused = narrow.merged_rows(&ctx, [Image::of(starts[0], &images[0])]);
        let named = matches!(
            &refused,
            Err(e) if e.to_string().contains("does not fit the tower's buffers")
        );
        let stood = s.model().pos() == from;
        println!(
            "(sm) a {}-token grid at a tower of {SMALL_CAP} tokens: {} {}, model {} {}",
            tokens[0],
            if named {
                "refused by name"
            } else {
                "NOT REFUSED BY NAME"
            },
            verdict(named),
            if stood { "unmoved" } else { "MOVED" },
            verdict(stood)
        );
        ok &= named && stood;
        drop(narrow);

        // The seat's steps: both images encoded on the resident encoder in one call, read back as bf16
        // rows, one a cell of each grid, then one prompt call over the spans those rows make.
        let rows = tower.merged_rows(
            &ctx,
            images
                .iter()
                .zip(&starts)
                .map(|(p, &start)| Image::of(start, p)),
        )?;
        let finite = rows.rows.iter().all(|&b| bf16(b).is_finite());
        let alive = rows.rows.iter().any(|&b| bf16(b) != 0.0);
        println!(
            "(sm) tower rows: {} values ({} positions of {n_embd}), finite {finite}, not all zero \
             {alive} {}",
            rows.rows.len(),
            rows.rows.len() / n_embd,
            verdict(finite && alive && rows.rows.len() == tokens.iter().sum::<usize>() * n_embd)
        );
        ok &= finite && alive && rows.rows.len() == tokens.iter().sum::<usize>() * n_embd;
        s.model_mut().reset()?;
        let seat_next = s
            .prompt_media(&ids, &rows.media_spans(), Want::Argmax)?
            .argmax();
        let seat_row = s.model().head_normed()?;

        // The body's call on the same rows, its spans put together from the images' known sizes.
        let mut first = 0;
        let spans: Vec<MediaSpan<'_>> = tokens
            .iter()
            .zip(&starts)
            .map(|(&n, &start)| {
                let span = MediaSpan {
                    at: start..start + n,
                    rows: &rows.rows[first * n_embd..(first + n) * n_embd],
                };
                first += n;
                span
            })
            .collect();
        s.model_mut().reset()?;
        let body_next = prefill_media(s.model_mut(), &ids, &spans)?;
        let body_row = s.model().head_normed()?;
        let same_bits = seat_row.len() == body_row.len()
            && seat_row
                .iter()
                .zip(&body_row)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        let pass = seat_next == body_next && same_bits;
        println!(
            "(sm) the seat's media call against the body's on the same rows: argmax {seat_next} vs \
             {body_next}, last hidden row bit-identical {same_bits} {}",
            verdict(pass)
        );
        ok &= pass;

        // The control: the same ids with no rows, whose image positions read the image token's own
        // embedding, lies away from the media call, so a call that dropped the rows is not this one.
        s.model_mut().reset()?;
        prefill(s.model_mut(), &ids)?;
        let away = rel_l2(&s.model().head_normed()?, &seat_row);
        let pass = away > 0.0;
        println!(
            "(sm) control (the image tokens' own embeddings): last row {away:.4} rel L2 from the media \
             call {}",
            verdict(pass)
        );
        ok &= pass;
        Ok(ok)
    }
}
