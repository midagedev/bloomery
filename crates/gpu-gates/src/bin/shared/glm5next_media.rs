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
