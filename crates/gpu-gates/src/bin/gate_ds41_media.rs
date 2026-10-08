//! GPU gate for the V4.1 text-side image injection (`body::prefill_media`,
//! `Session::prompt_media`) on the gate placement (`crate::gate_card::plan_gate`: the 3090's
//! bytes in the real tier, the largest visible card under the header's card budget in the
//! fixture tier): a prompt whose media
//! span carries bf16 rows takes those rows at its positions, picks its
//! experts there with `exp_probs_b_vl`, and hashes its engram n-grams as
//! DEAD — each clause against a rule the gate computes itself, on the
//! official vision oracle set's aligner rows (`refset` family
//! `deepseek41v`, the set `gate-vision` reads) and the learned delimiter
//! rows of the mmproj file that set names.
//!
//! The feed: one image of the set (its `image` row's grid), its `aligner`
//! tap's rows in reading order, and `v.token_embd.img_start`,
//! `v.image_newline`, `v.token_embd.img_end` read from the mmproj by name —
//! the reference's splice (`model.py:1235-1239`): a Start row, each grid
//! row's aligner rows and a NewLine row, an End row, every position of the
//! span carrying the image token. Three layouts over `$BLOOMERY_DATA/
//! engram/corpus-prose.ids`: the span mid-prompt, the span crossing the
//! batch boundary `T_MAX`, and the span ending the prompt.
//!
//! - **(i) Embedding.** A media call's front seam, read after the
//!   broadcast: every media position's four streams and its layer-0 input
//!   equal the widened bf16 row (`f32::from_bits(bits << 16)`) value for
//!   value, and its text positions equal a call of the same ids with no
//!   span — the control observed as a media call of no spans, so its front
//!   seam fires too — bit for bit.
//! - **(ii) Router.** Every route seam's block: the card's routed ids
//!   equal the host port of the pick rule (`router.rs`'s `select`: six
//!   rounds of argmax over the card's own scores plus a bias, ties to the
//!   larger id), run with `bias_vl` at media positions and the text `bias`
//!   elsewhere, per layer. The coverage assertion: the gate refuses itself
//!   unless the two biases' picks differ at `>= 1` image position and
//!   `>= 1` separator position (Start, NewLine, End) of the spans — a run
//!   where no checked position distinguishes them proves nothing.
//! - **(iii) Engram.** At each engram site layer (the file's
//!   `engram.layer_ids`), the streams the engram seam shows after the step
//!   equal, at the media positions that seam and the attention seam before
//!   it both claim, what that attention seam showed — `out = x + value·gate`
//!   over the zero rows the call reads: `x` again, the one difference a
//!   `-0` widened to `+0`, counted and named. The blocked lookback window
//!   itself is the plan gate's (`gate-ds41-plan`, `plan_dead_into`).
//! - **(iv) Split invariance.** The same ids and span fed as one call and
//!   as two calls cut in the text before the span — the mid layout, and the
//!   crossing layout whose span straddles `T_MAX` inside the one call and
//!   inside the second of its two calls — leave the same state: every
//!   layer's window ring, compressor state, compressed rows and index keys
//!   hashed equal, the same cuts granted, and `STEPS` decode steps on the
//!   same fed ids yielding the same greedy ids.
//! - **(v) Decode.** After a prompt the span ends, the first three steps'
//!   engram windows (`Body::step_window`) are dead exactly where their
//!   lookback reaches into the span — three entries at the first step, one
//!   fewer each step after, none from the fourth on, the live entries the
//!   tokens those positions hold — and a snapshot taken after two steps and
//!   resumed, like the same state as bytes through `save_state`/
//!   `restore_state`, decodes the remaining steps to the uninterrupted
//!   run's ids.
//! - **(vi) Refusals.** By name: media under the steps feed (the engine's
//!   `prefill_media` and `Session::prompt_media` both), a span the call
//!   does not hold whole, and a cut inside a span.
//!
//! Every clause is self-consistency — the engine against a rule the gate computes from the
//! model's own tensors, over the vision set's rows as inputs — so the fixture tier runs all of
//! them; (ii)'s coverage needs the file's `exp_probs_b_vl` to pick differently from
//! `exp_probs_b` somewhere, which the fixture's biases (0 and a uniform ±0.1) do.
//!
//! Text-only prompts are `gate-gpu-ds41-prefill`'s (bit for bit); the
//! kernels are untouched (ptx-scan against the base). The FAIL-first
//! mutants — the rows staged one position off, the media broadcast reading
//! `token_embd`, every position picking with the text bias, the span's
//! Start and End separators left on the text bias, real engram rows hashed
//! at the media positions, a state save that drops the spans, a refusal
//! removed — are source edits this gate goes red on, one clause each.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_ds41_media: built without the `deepseek41` feature; see `just gate-gpu-ds41-media`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_ds41_media", gate::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_tier.rs"]
#[allow(
    dead_code,
    reason = "the gate reads the path and the clause tags; the triangle's facts serve the prefill gate"
)]
mod ds41_tier;

#[cfg(feature = "deepseek41")]
mod gate {
    use app::Session;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{DeviceTensor, Gpu, GpuError};
    use bloomery_gpu_deepseek41::body::{
        self, BatchSeam, BatchSeamKind, Deepseek41Model, MediaKind, MediaSpan, PrefillMode,
    };
    use bloomery_gpu_deepseek41::router::{N_EXPERT, N_USED};
    use bloomery_gpu_deepseek41::span::span;
    use bloomery_gpu_gates::ds41_media;
    use bloomery_gpu_gates::tier;
    use bloomery_gpu_gates::{
        Fnv1a64, GateError, checks_failed, data_dir, prose_ids, split_f32, verdict,
    };
    use bloomery_levers::{
        CARD_BUDGET, CARD_DONTNEED, CED, ENGRAM_HELPER, HOST_LOCK, HOST_POPULATE, PREFILL_GROUP,
        R8, STEP_STATS,
    };
    use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};
    use gguf::Split;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::names;
    use model::placement::workstation;
    use refset::arch::deepseek41v::{VISION, VISION_SET};
    use refset::vision::VisionSet;
    use runtime::Want;

    const NAME: &str = "gate_ds41_media";
    /// The image whose aligner rows the gate feeds: the set's full-tap image.
    const IMAGE: &str = "grad-448";
    /// Decode steps the split-invariance clauses and the snapshot clauses run.
    const STEPS: usize = 24;
    /// The decode step the snapshot clauses take their state after.
    const SNAP_AT: usize = 2;
    /// The dead-window clause's steps: the three that see the span, and one
    /// past them.
    const DEAD_STEPS: usize = 4;

    // ------------------------------------------------------------ the feed

    /// One image's span as the gate feeds it: every position's bf16 row
    /// (`n_embd` values) and its kind, the reference's splice.
    struct Feed {
        rows: Vec<u16>,
        kinds: Vec<MediaKind>,
        /// The image token every span position carries.
        token: u32,
    }

    impl Feed {
        /// The span, placed at `at` of a prompt's ids.
        fn span(&self, at: usize) -> MediaSpan<'_> {
            MediaSpan {
                at: at..at + self.kinds.len(),
                rows: &self.rows,
                kinds: &self.kinds,
            }
        }

        fn len(&self) -> usize {
            self.kinds.len()
        }
    }

    /// The vision oracle set's image as the feed, with the model's `n_embd`.
    fn read_feed(hp: &Hparams) -> Result<(Feed, usize), GateError> {
        let dir = match std::env::var("BLOOMERY_VISION_SET") {
            Ok(set) => data_dir().join("ref-vision").join(set),
            Err(_) => VISION.path(VISION_SET),
        };
        let set = VisionSet::read(&dir)?;
        if !set.complete {
            return Err(format!("{}: the set is not complete", dir.display()).into());
        }
        let img = set
            .images
            .iter()
            .find(|i| i.stem() == IMAGE)
            .ok_or_else(|| {
                format!(
                    "{}: no image {IMAGE:?} of {}",
                    dir.display(),
                    set.images.len()
                )
            })?;
        let n_embd = hp.n_embd;
        let (w, h) = (img.n_llm_w, img.n_llm_h);
        if img.n_tokens != 2 + h * (w + 1) {
            return Err(format!(
                "{}: {} tokens of a {h}x{w} grid, want {}",
                img.name,
                img.n_tokens,
                2 + h * (w + 1)
            )
            .into());
        }
        let want = format!("{IMAGE}.aligner.bf16");
        let file = set
            .files
            .iter()
            .find(|f| f.name == want)
            .ok_or_else(|| format!("{}: no aligner rows {want:?}", dir.display()))?;
        let [rows, cols] = file.shape[..] else {
            return Err(format!("{want}: shaped {:?}, want [rows, {n_embd}]", file.shape).into());
        };
        if cols != n_embd || rows != h * w {
            return Err(format!(
                "{want}: {rows} rows of {cols}, want the image's {h}x{w} = {} rows of the \
                 model's n_embd {n_embd}",
                h * w
            )
            .into());
        }
        let aligner = read_bf16(&dir.join(&file.name))?;
        // The span itself — the learned delimiter rows of the mmproj the set
        // names and the splice over the aligner rows — is the one library
        // owner `bloomery_gpu_gates::ds41_media` (this gate and the serve
        // seat's `--mmproj` path share it).
        let mmproj = gguf::Gguf::open(&set.mmproj)?;
        let d = ds41_media::delims(&mmproj, n_embd)?;
        let (rows, kinds) = ds41_media::span_rows(&aligner, (h, w), n_embd, &d)?;
        Ok((
            Feed {
                rows,
                kinds,
                token: set.image_token_id,
            },
            n_embd,
        ))
    }

    /// One file of little-endian bf16 values.
    fn read_bf16(path: &std::path::Path) -> Result<Vec<u16>, GateError> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(b.as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect())
    }

    /// One layout: its ids, where the span sits, and each position's kind
    /// (`None` off the span).
    struct Layout {
        ids: Vec<u32>,
        at: usize,
        kinds: Vec<Option<MediaKind>>,
    }

    impl Layout {
        /// `pre` text ids, the span, `post` text ids.
        fn new(feed: &Feed, text: &[u32], pre: usize, post: usize) -> Layout {
            let span = feed.len();
            let mut ids = Vec::with_capacity(pre + span + post);
            ids.extend_from_slice(&text[..pre]);
            ids.extend(std::iter::repeat_n(feed.token, span));
            ids.extend_from_slice(&text[pre..pre + post]);
            let mut kinds = vec![None; ids.len()];
            for (k, kind) in feed.kinds.iter().enumerate() {
                kinds[pre + k] = Some(*kind);
            }
            Layout {
                ids,
                at: pre,
                kinds,
            }
        }

        fn positions(&self) -> usize {
            self.ids.len()
        }

        /// The span as one call of the whole layout takes it.
        fn span<'a>(&self, feed: &'a Feed) -> MediaSpan<'a> {
            feed.span(self.at)
        }

        /// The span as a call of `len` ids from `at` takes it: `None` when
        /// the call holds none of it. The layouts never split their span;
        /// [`run_once`] refuses an arrangement that does.
        fn span_from<'a>(&self, feed: &'a Feed, at: usize, len: usize) -> Option<MediaSpan<'a>> {
            (self.at >= at && self.at + feed.len() <= at + len).then(|| feed.span(self.at - at))
        }
    }

    /// bf16 widened to f32, the broadcast's own rule (exact).
    fn bf16_f32(b: u16) -> f32 {
        f32::from_bits(u32::from(b) << 16)
    }

    /// The card's pick rule on host scores (`router.rs`'s `select`): six
    /// rounds of argmax over `score + bias` among the experts no earlier
    /// round took, ties to the larger id, the rounds' winners in order.
    fn select_n(scores: &[f32], bias: &[f32]) -> [u32; N_USED] {
        let mut taken = [false; N_EXPERT];
        let mut out = [0u32; N_USED];
        for slot in &mut out {
            let (mut bv, mut bi) = (f32::NEG_INFINITY, 0usize);
            for (e, (&s, &b)) in scores.iter().zip(bias).enumerate() {
                if taken[e] {
                    continue;
                }
                let v = s + b;
                if v > bv || (v == bv && e > bi) {
                    bv = v;
                    bi = e;
                }
            }
            taken[bi] = true;
            *slot = bi as u32;
        }
        out
    }

    // ------------------------------------------------------------ observing

    /// One batch's front seam: its first position, its tokens, and its
    /// streams and layer-0 input as the broadcast left them.
    type Front = (u32, usize, Vec<f32>, Vec<f32>);

    /// What one observed layout run collected: the media call's fronts, the
    /// no-span control's, the media positions' streams at the engram site
    /// layers' engram seams and at the layer before each site's attention
    /// and post-join seams, the layers a route seam fired at, and the router
    /// clause's counters.
    #[derive(Default)]
    struct Seen {
        fronts: Vec<Front>,
        control: Vec<Front>,
        engram_streams: Vec<(usize, Vec<usize>, Vec<Vec<f32>>)>,
        ffn_streams: Vec<(usize, Vec<usize>, Vec<Vec<f32>>)>,
        attn_streams: Vec<(usize, Vec<usize>, Vec<Vec<f32>>)>,
        routed_layers: Vec<usize>,
        routes_checked: usize,
        routes_media: usize,
        route_mismatch: Vec<String>,
        cover_image: usize,
        cover_sep: usize,
    }

    /// The per-position media facts of a layout: whether a position is
    /// media, and whether it is an image position (a separator otherwise).
    struct MediaOf {
        is: Vec<bool>,
        image: Vec<bool>,
    }

    impl MediaOf {
        fn new(layout: &Layout, feed: &Feed) -> MediaOf {
            let mut of = MediaOf {
                is: vec![false; layout.positions()],
                image: vec![false; layout.positions()],
            };
            for (k, kind) in feed.kinds.iter().enumerate() {
                let p = layout.at + k;
                of.is[p] = true;
                of.image[p] = *kind == MediaKind::Image;
            }
            of
        }
    }

    /// Copy a device buffer to the host.
    fn host<T: DeviceCopy + Default + Clone>(
        stream: &CudaStream,
        from: &DeviceBuffer<T>,
    ) -> Result<Vec<T>, GpuError> {
        let mut out = vec![T::default(); from.len()];
        from.copy_to_host(stream, &mut out)?;
        Ok(out)
    }

    /// One layout observed twice: the no-span control call and the media
    /// call, their seams read — fronts stored, routes checked against the
    /// host pick inline, the site layers' media streams stored.
    #[allow(
        clippy::too_many_arguments,
        reason = "a layout's data and the rules its seams are read against (rust-quality R8)"
    )]
    fn observed(
        m: &mut Deepseek41Model,
        layout: &Layout,
        feed: &Feed,
        of: &MediaOf,
        n_embd: usize,
        text_bias: &[Vec<f32>],
        vl_bias: &[Vec<f32>],
        sites: &[usize],
    ) -> Result<Seen, GateError> {
        let mut seen = Seen::default();
        m.reset()?;
        {
            let mut fronts = Vec::new();
            let mut obs = |gpu: &Gpu, seam: BatchSeam<'_>| -> Result<(), GpuError> {
                if seam.kind != BatchSeamKind::Front {
                    return Ok(());
                }
                let stream = gpu.stream();
                stream.synchronize()?;
                let mut streams = vec![0.0f32; seam.streams.len()];
                seam.streams.copy_to_host(stream, &mut streams)?;
                let mut fold = Vec::new();
                if let Some(f) = seam.fold {
                    fold = vec![0.0f32; f.len()];
                    f.copy_to_host(stream, &mut fold)?;
                }
                fronts.push((seam.first, seam.tokens, streams, fold));
                Ok(())
            };
            body::prefill_media_observed(m, &layout.ids, &[], &mut obs)?;
            seen.control = fronts;
        }
        m.reset()?;
        let mut obs = |gpu: &Gpu, seam: BatchSeam<'_>| -> Result<(), GpuError> {
            let stream = gpu.stream();
            stream.synchronize()?;
            match seam.kind {
                BatchSeamKind::Front => {
                    let streams = host(stream, seam.streams)?;
                    let fold = match seam.fold {
                        Some(f) => host(stream, f)?,
                        None => Vec::new(),
                    };
                    seen.fronts.push((seam.first, seam.tokens, streams, fold));
                }
                BatchSeamKind::Attn | BatchSeamKind::Engram | BatchSeamKind::Ffn => {
                    // The site layers' engram seams, and the two seams around
                    // the layer before each — its attention's output and its
                    // post-join streams — of which the comparison picks what
                    // the engram step read: the join is the only writer
                    // between them.
                    let hold = match seam.kind {
                        BatchSeamKind::Engram => sites.contains(&seam.layer),
                        _ => sites.contains(&(seam.layer + 1)),
                    };
                    if !hold {
                        return Ok(());
                    }
                    let s4 = 4 * n_embd;
                    let base = seam.first as usize - seam.at;
                    let streams = host(stream, seam.streams)?;
                    let mut ps = Vec::new();
                    let mut vs = Vec::new();
                    for t in seam.at..seam.tokens {
                        let p = base + t;
                        if p < of.is.len() && of.is[p] {
                            ps.push(p);
                            vs.push(streams[t * s4..(t + 1) * s4].to_vec());
                        }
                    }
                    match seam.kind {
                        BatchSeamKind::Engram => seen.engram_streams.push((seam.layer, ps, vs)),
                        BatchSeamKind::Ffn => seen.ffn_streams.push((seam.layer, ps, vs)),
                        _ => seen.attn_streams.push((seam.layer, ps, vs)),
                    }
                }
                BatchSeamKind::Route => {
                    let tap = seam.route.as_ref().ok_or(GpuError::State {
                        what: NAME,
                        missing: "the route seam's tap",
                    })?;
                    let (at, t) = (seam.at, seam.tokens);
                    let probs = host(stream, tap.probs)?;
                    let ids = host(stream, tap.ids)?;
                    if probs.len() < t * N_EXPERT || ids.len() < (at + t) * N_USED {
                        return Err(GpuError::State {
                            what: NAME,
                            missing: "the route tap's buffers",
                        });
                    }
                    let base = seam.first as usize - at;
                    if !seen.routed_layers.contains(&seam.layer) {
                        seen.routed_layers.push(seam.layer);
                    }
                    let (bias, vl) = match (text_bias.get(seam.layer), vl_bias.get(seam.layer)) {
                        (Some(b), Some(v)) => (&b[..], &v[..]),
                        _ => {
                            return Err(GpuError::State {
                                what: NAME,
                                missing: "the layer's biases",
                            });
                        }
                    };
                    for k in 0..t {
                        let p = base + at + k;
                        let scores = &probs[k * N_EXPERT..(k + 1) * N_EXPERT];
                        let got = &ids[(at + k) * N_USED..(at + k + 1) * N_USED];
                        let media = p < of.is.len() && of.is[p];
                        let with_bias = select_n(scores, bias);
                        let want = if media {
                            select_n(scores, vl)
                        } else {
                            with_bias
                        };
                        seen.routes_checked += 1;
                        if media {
                            seen.routes_media += 1;
                            if want != with_bias {
                                if of.image[p] {
                                    seen.cover_image += 1;
                                } else {
                                    seen.cover_sep += 1;
                                }
                            }
                        }
                        if got != want && seen.route_mismatch.len() < 4 {
                            seen.route_mismatch.push(format!(
                                "layer {} position {} (got {:?}, want {:?})",
                                seam.layer, p, got, want
                            ));
                        }
                    }
                }
            }
            Ok(())
        };
        let span = layout
            .span_from(feed, 0, layout.positions())
            .ok_or(format!("{NAME}: the layout does not hold its span whole"))?;
        body::prefill_media_observed(m, &layout.ids, &[span], &mut obs)?;
        Ok(seen)
    }

    // ------------------------------------------------------------ the state

    /// Every layer's window ring, compressor state, compressed rows and
    /// index keys, hashed: what two arrangements of one layout must leave
    /// equal.
    struct Digests {
        ring: Vec<u64>,
        state: Vec<Option<u64>>,
        rows: Vec<Option<u64>>,
        keys: Vec<Option<u64>>,
    }

    /// The state's digest at `p` positions: the rings and states whole, the
    /// compressed rows and index keys of the `⌊p / ratio⌋` rows held.
    fn digest(m: &mut Deepseek41Model, hp: &Hparams, p: usize) -> Result<Digests, GateError> {
        let (gpu, _, b) = m.body_parts(NAME)?;
        let stream = gpu.stream();
        let hash = |v: Vec<u32>| Fnv1a64::default().u32s(&v).value();
        let u16s = |t: &DeviceTensor<u16>, rows: usize| -> Result<u64, GateError> {
            let n = rows * t.cols();
            let w = span(NAME, t.buf(), 0, n)?;
            let mut host = vec![0u16; n];
            w.copy_to_host(stream, &mut host)?;
            Ok(hash(host.iter().map(|x| u32::from(*x)).collect()))
        };
        let f32s = |t: &DeviceTensor<f32>| -> Result<u64, GateError> {
            let n = t.rows() * t.cols();
            let w = span(NAME, t.buf(), 0, n)?;
            let mut host = vec![0f32; n];
            w.copy_to_host(stream, &mut host)?;
            Ok(hash(host.iter().map(|x| x.to_bits()).collect()))
        };
        let mut d = Digests {
            ring: Vec::new(),
            state: Vec::new(),
            rows: Vec::new(),
            keys: Vec::new(),
        };
        for l in b.layers() {
            let st = b
                .state_mut(l)
                .ok_or_else(|| format!("{NAME}: no state for layer {l}"))?;
            d.ring.push(u16s(st.ring, st.ring.rows())?);
            d.state.push(match (st.values, st.scores) {
                (Some(v), Some(sc)) => Some(f32s(v)? ^ f32s(sc)?),
                _ => None,
            });
            // A layer that holds rows or keys and reads ratio 0 would hash
            // zero rows on each side and pass on an unwritten cache.
            let held = if st.rows.is_some() || st.keys.is_some() {
                let kind = hp.layers.get(l).ok_or_else(|| {
                    format!("{NAME}: layer {l} holds rows or keys and has no layer entry")
                })?;
                let ratio = usize::try_from(kind.ratio())?;
                if ratio == 0 {
                    return Err(format!("{NAME}: layer {l} holds rows and reads ratio 0").into());
                }
                p / ratio
            } else {
                0
            };
            d.rows.push(match st.rows {
                Some(t) => Some(u16s(t, held)?),
                None => None,
            });
            d.keys.push(match st.keys {
                Some(t) => Some(u16s(t, held)?),
                None => None,
            });
        }
        Ok(d)
    }

    /// One arrangement of a layout: its calls fed, then the state digest,
    /// the cuts granted, and `STEPS` greedy ids over the same fed ids.
    struct Run {
        digests: Digests,
        ids: Vec<u32>,
        keeps: [usize; 3],
    }

    /// Feed `layout` as calls of `calls` tokens each (their lengths sum to
    /// the layout's, the span whole inside one of them), then read its
    /// state and decode `STEPS` steps on `fed`.
    fn run_once(
        m: &mut Deepseek41Model,
        hp: &Hparams,
        layout: &Layout,
        feed: &Feed,
        calls: &[usize],
        fed: &[u32],
    ) -> Result<Run, GateError> {
        m.reset()?;
        let (mut held, mut at) = (0, 0);
        for &len in calls {
            let ids = &layout.ids[at..at + len];
            match layout.span_from(feed, at, len) {
                Some(span) => {
                    body::prefill_media(m, ids, &[span])?;
                    held += 1;
                }
                None => {
                    body::prefill_media(m, ids, &[])?;
                }
            }
            at += len;
        }
        if held != 1 || at != layout.positions() {
            return Err(format!(
                "{NAME}: the calls {calls:?} hold the span {held} times over {} positions",
                layout.positions()
            )
            .into());
        }
        let p = layout.positions();
        let digests = digest(m, hp, p)?;
        let keeps =
            [p / 2, p.saturating_sub(1), p].map(|k| m.body(NAME).map_or(0, |b| b.keep_point(k)));
        let mut ids = Vec::with_capacity(STEPS);
        for &id in &fed[..STEPS] {
            ids.push(m.step(&[id])?);
        }
        Ok(Run {
            digests,
            ids,
            keeps,
        })
    }

    /// Whether `r` is a refusal by name whose words hold `says`.
    fn refused<T>(r: Result<T, GpuError>, says: &str) -> bool {
        match r {
            Err(e) => e.to_string().contains(says),
            Ok(_) => false,
        }
    }

    /// The token position `q` holds: the layout's below its end, the fed
    /// ids above it.
    fn token_at(layout: &Layout, fed: &[u32], q: usize) -> u32 {
        if q < layout.positions() {
            layout.ids[q]
        } else {
            fed[q - layout.positions()]
        }
    }

    /// The ending layout's prompt and `SNAP_AT` steps, then `take` — a
    /// snapshot and its resume, or the byte form's round trip — then the
    /// remaining steps' greedy ids.
    fn resumed<F>(
        m: &mut Deepseek41Model,
        end: &Layout,
        feed: &Feed,
        fed: &[u32],
        take: F,
    ) -> Result<Vec<u32>, GateError>
    where
        F: FnOnce(&mut Deepseek41Model) -> Result<(), GateError>,
    {
        m.reset()?;
        body::prefill_media(m, &end.ids, &[end.span(feed)])?;
        for &id in &fed[..SNAP_AT] {
            m.step(&[id])?;
        }
        take(m)?;
        let mut ids = Vec::with_capacity(STEPS - SNAP_AT);
        for &id in &fed[SNAP_AT..STEPS] {
            ids.push(m.step(&[id])?);
        }
        Ok(ids)
    }

    // ------------------------------------------------------------ the gate

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[
            CED,
            PREFILL_GROUP,
            ENGRAM_HELPER,
            STEP_STATS,
            CARD_BUDGET,
            HOST_POPULATE,
            HOST_LOCK,
            CARD_DONTNEED,
            R8,
        ])?;
        crate::gate_card::init()?;
        let mut cfg = body::OpenCfg::from_levers(&levers)?;
        let path = crate::ds41_tier::model_path()?;
        let head = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        cfg.place = tier::plan_levers(&head, &levers, 0)?;
        let hp = Hparams::read(&head)?;
        drop(head);
        let sites = hp
            .engram
            .as_ref()
            .map(|e| e.layer_ids.clone())
            .ok_or("the model file has no engram sites")?;
        let ngram = hp.engram.as_ref().map_or(0, |e| e.max_ngram);
        let (feed, n_embd) = read_feed(&hp)?;
        let file = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let t = std::time::Instant::now();
        let mut m = body::open(
            file,
            crate::gate_card::plan_gate,
            usize::try_from(workstation::CTX_MAX)?,
            &cfg,
        )?;
        m.set_mode(StepMode::Graph);
        body::prepare_prefill(&mut m)?;
        body::prepare_media(&mut m)?;
        m.capture_step()?;
        let split = Split::open(&path)?;
        let mut text_bias = Vec::with_capacity(hp.n_layer);
        let mut vl_bias = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            text_bias.push(split_f32(&split, &names::exp_probs_b(l), N_EXPERT)?);
            vl_bias.push(split_f32(&split, &names::exp_probs_b_vl(l), N_EXPERT)?);
        }
        let corpus = prose_ids("engram", 1600)?;
        // The layouts: the span mid-prompt, crossing T_MAX, ending the
        // prompt.
        let mid = Layout::new(&feed, &corpus, 58, 58);
        let cross = Layout::new(&feed, &corpus, 450, 66);
        let end = Layout::new(&feed, &corpus, 50, 0);
        let span_len = feed.len();
        if !(mid.at + span_len <= body::T_MAX
            && cross.at < body::T_MAX
            && cross.at + span_len > body::T_MAX
            && end.at + span_len == end.positions())
        {
            return Err(format!(
                "the layouts do not hold their premises: the mid span ends inside one batch ({} \
                 <= {}), the crossing span straddles it ({} < {} < {}), the ending span closes \
                 the prompt",
                mid.at + span_len,
                body::T_MAX,
                cross.at,
                body::T_MAX,
                cross.at + span_len
            )
            .into());
        }
        println!(
            "{NAME}: loaded in {:.1} s; span {span_len} positions of {n_embd} bf16 (token {}), \
             engram sites {sites:?}, ngram {ngram}; layouts mid {} (span {}..{}), cross {} (span \
             {}..{}), end {} (span {}..{}); T_MAX {}",
            t.elapsed().as_secs_f64(),
            feed.token,
            mid.positions(),
            mid.at,
            mid.at + span_len,
            cross.positions(),
            cross.at,
            cross.at + span_len,
            end.positions(),
            end.at,
            end.at + span_len,
            body::T_MAX,
        );
        let mut ok = true;

        // (i), (ii), (iii): each layout observed against its no-span control.
        let mut cover = (0usize, 0usize);
        let mut seens = Vec::new();
        for layout in [&mid, &cross, &end] {
            let of = MediaOf::new(layout, &feed);
            seens.push(observed(
                &mut m, layout, &feed, &of, n_embd, &text_bias, &vl_bias, &sites,
            )?);
        }
        crate::ds41_tier::sc("(i) embedding: the media positions hold the widened row")?;
        crate::ds41_tier::sc("(ii) router: every route seam's ids are the host pick's")?;
        crate::ds41_tier::sc(
            "(ii) coverage: the two biases' picks differ at an image and a separator",
        )?;
        for (what, layout, seen) in [
            ("mid", &mid, &seens[0]),
            ("cross", &cross, &seens[1]),
            ("end", &end, &seens[2]),
        ] {
            // (i): the media positions hold the widened row, the text
            // positions what the no-span call wrote.
            let s4 = 4 * n_embd;
            let (mut media_n, mut media_bad, mut text_n, mut text_bad, mut first_bad) =
                (0usize, 0usize, 0usize, 0usize, String::new());
            for (first, tokens, streams, fold) in &seen.fronts {
                let base = *first as usize;
                let control = seen
                    .control
                    .iter()
                    .find(|(f, ..)| f == first)
                    .ok_or(format!("{NAME}: no control front for the batch at {first}"))?;
                for t in 0..*tokens {
                    let p = base + t;
                    if p >= layout.positions() {
                        continue;
                    }
                    let media = layout.kinds[p].is_some();
                    let s = &streams[t * s4..(t + 1) * s4];
                    let f = &fold[t * n_embd..(t + 1) * n_embd];
                    if media {
                        media_n += 1;
                        let at = (p - layout.at) * n_embd;
                        let row = &feed.rows[at..at + n_embd];
                        let wide: Vec<u32> = row.iter().map(|b| bf16_f32(*b).to_bits()).collect();
                        let bad = (0..4).any(|hs| {
                            s[hs * n_embd..(hs + 1) * n_embd]
                                .iter()
                                .zip(&wide)
                                .any(|(a, b)| a.to_bits() != *b)
                        }) || f.iter().zip(&wide).any(|(a, b)| a.to_bits() != *b);
                        if bad {
                            media_bad += 1;
                            if first_bad.is_empty() {
                                first_bad = format!("media position {p}");
                            }
                        }
                    } else {
                        text_n += 1;
                        let bad = s != &control.2[t * s4..(t + 1) * s4]
                            || f != &control.3[t * n_embd..(t + 1) * n_embd];
                        if bad {
                            text_bad += 1;
                            if first_bad.is_empty() {
                                first_bad = format!("text position {p}");
                            }
                        }
                    }
                }
            }
            let pass = media_bad == 0 && text_bad == 0 && media_n > 0 && text_n > 0;
            ok &= pass;
            println!(
                "{NAME}: (i) {what}: {media_n} media positions hold their widened row and \
                 {text_n} text positions the no-span call's values ({} bad{}): {}",
                media_bad + text_bad,
                if first_bad.is_empty() {
                    String::new()
                } else {
                    format!(", first at {first_bad}")
                },
                verdict(pass)
            );
            // (ii): every route seam's ids against the host pick.
            cover.0 += seen.cover_image;
            cover.1 += seen.cover_sep;
            let pass = seen.route_mismatch.is_empty() && seen.routes_media > 0;
            ok &= pass;
            println!(
                "{NAME}: (ii) {what}: {} routed tokens of them {} media match the host pick \
                 (bias_vl at media, bias elsewhere){}: {}",
                seen.routes_checked,
                seen.routes_media,
                seen.route_mismatch
                    .first()
                    .map_or(String::new(), |f| format!("; first off at {f}")),
                verdict(pass)
            );
        }
        let pass = cover.0 >= 1 && cover.1 >= 1;
        ok &= pass;
        println!(
            "{NAME}: (ii) coverage: pick(bias_vl) differs from pick(bias) at {} image and {} \
             separator positions (want >= 1 each): {}",
            cover.0,
            cover.1,
            verdict(pass)
        );
        crate::ds41_tier::sc("(iii) engram: the media streams pass the engram step unchanged")?;
        for site in &sites {
            // (iii): the engram seam's media streams equal what the engram
            // step read — the layer before's post-join streams where it runs
            // a block (its route seam fired), its attention's output where
            // it runs none — a -0 widened to +0 apart.
            let (mut compared, mut bad, mut minus_zero, mut first_bad) = (0, 0, 0, String::new());
            for seen in &seens {
                let after = seen
                    .engram_streams
                    .iter()
                    .filter(|(l, ..)| l == site)
                    .flat_map(|(_, ps, vs)| ps.iter().zip(vs).map(|(p, v)| (*p, v)));
                let routed = seen.routed_layers.contains(&site.saturating_sub(1));
                let from = if routed {
                    &seen.ffn_streams
                } else {
                    &seen.attn_streams
                };
                let before: Vec<(usize, &Vec<f32>)> = from
                    .iter()
                    .filter(|(l, ..)| l + 1 == *site)
                    .flat_map(|(_, ps, vs)| ps.iter().zip(vs).map(|(p, v)| (*p, v)))
                    .collect();
                for (p, got) in after {
                    let Some((_, want)) = before.iter().find(|(q, _)| *q == p) else {
                        continue;
                    };
                    compared += 1;
                    for (a, b) in got.iter().zip(*want) {
                        if a.to_bits() == b.to_bits() {
                            continue;
                        }
                        if *a == *b && a == &0.0 {
                            minus_zero += 1;
                        } else {
                            bad += 1;
                            if first_bad.is_empty() {
                                first_bad = format!("position {p}");
                            }
                        }
                    }
                }
            }
            let pass = bad == 0 && compared > 0;
            ok &= pass;
            println!(
                "{NAME}: (iii) layer {site}: {compared} media positions' streams unchanged by \
                 the engram step ({} values differ, {minus_zero} -0 widened to +0{}): {}",
                bad,
                if first_bad.is_empty() {
                    String::new()
                } else {
                    format!(", first at {first_bad}")
                },
                verdict(pass)
            );
        }

        // (iv): split invariance. The cuts sit in the text before the span.
        crate::ds41_tier::sc("(iv) split invariance: one call = two calls cut before the span")?;
        for (what, layout, cut) in [("mid", &mid, 30usize), ("cross", &cross, 100usize)] {
            let p = layout.positions();
            let fed = &corpus[p..p + STEPS];
            let one = run_once(&mut m, &hp, layout, &feed, &[p], fed)?;
            let two = run_once(&mut m, &hp, layout, &feed, &[cut, p - cut], fed)?;
            let field = |name: &str, same: bool, n: usize, of: usize| {
                format!("{name} {n}/{of}{}", if same { "" } else { " DIFFER" })
            };
            // The granted cuts are the arrangements' own (a cut lands on a
            // call's boundary, so the two-call run keeps where it cut): the
            // state and the ids are the claim, the cuts the record.
            let pass = one.digests.ring == two.digests.ring
                && one.digests.state == two.digests.state
                && one.digests.rows == two.digests.rows
                && one.digests.keys == two.digests.keys
                && one.ids == two.ids;
            ok &= pass;
            println!(
                "{NAME}: (iv) {what}: one call of {p} = {cut} + {} (the span at {}..{}): {} | {} \
                 | {} | {} | cuts granted {:?}/{:?} (the calls' own) | {STEPS} decode ids {}: {}",
                p - cut,
                layout.at,
                layout.at + span_len,
                field(
                    "ring",
                    one.digests.ring == two.digests.ring,
                    one.digests
                        .ring
                        .iter()
                        .zip(&two.digests.ring)
                        .filter(|(a, b)| a == b)
                        .count(),
                    one.digests.ring.len()
                ),
                field(
                    "state",
                    one.digests.state == two.digests.state,
                    one.digests
                        .state
                        .iter()
                        .zip(&two.digests.state)
                        .filter(|(a, b)| a == b)
                        .count(),
                    one.digests.state.len()
                ),
                field(
                    "rows",
                    one.digests.rows == two.digests.rows,
                    one.digests
                        .rows
                        .iter()
                        .zip(&two.digests.rows)
                        .filter(|(a, b)| a == b)
                        .count(),
                    one.digests.rows.len()
                ),
                field(
                    "keys",
                    one.digests.keys == two.digests.keys,
                    one.digests
                        .keys
                        .iter()
                        .zip(&two.digests.keys)
                        .filter(|(a, b)| a == b)
                        .count(),
                    one.digests.keys.len()
                ),
                one.keeps,
                two.keeps,
                if one.ids == two.ids {
                    "identical".to_string()
                } else {
                    format!(
                        "differ at {:?}",
                        one.ids.iter().zip(&two.ids).position(|(a, b)| a != b)
                    )
                },
                verdict(pass),
            );
        }

        // (v): the dead lookback after an image-ending prompt, then the
        // snapshot clauses over the same uninterrupted run.
        let p = end.positions();
        let (s0, e0) = (end.at, end.at + span_len);
        let fed = &corpus[p..p + STEPS];
        crate::ds41_tier::sc("(v) decode: the dead lookback after an image-ending prompt")?;
        let straight = run_once(&mut m, &hp, &end, &feed, &[p], fed)?;
        m.reset()?;
        body::prefill_media(&mut m, &end.ids, &[end.span(&feed)])?;
        let mut dead = Vec::new();
        let mut dead_ok = true;
        for k in 0..DEAD_STEPS {
            let id = fed[k];
            m.step(&[id])?;
            let window = m.body(NAME)?.step_window().to_vec();
            let pos = p + k;
            let want: Vec<Option<u32>> = (0..ngram)
                .map(|back| {
                    let q = pos - back;
                    if (s0..e0).contains(&q) {
                        None
                    } else {
                        Some(token_at(&end, fed, q))
                    }
                })
                .collect();
            if window != want {
                dead_ok = false;
            }
            dead.push(format!(
                "{}/{}",
                window.iter().filter(|e| e.is_none()).count(),
                ngram
            ));
        }
        ok &= dead_ok;
        println!(
            "{NAME}: (v) the {DEAD_STEPS} steps after the span-ending prompt see dead/tokens \
             {dead:?} (the span {s0}..{e0} in every lookback that reaches it): {}",
            verdict(dead_ok)
        );
        // The snapshot and the byte form, each resumed after step SNAP_AT.
        crate::ds41_tier::sc("(v) snapshot: resume and save/restore give the uninterrupted ids")?;
        let snap = resumed(&mut m, &end, &feed, fed, |m| {
            let s = body::snapshot(m)?;
            body::resume(m, &s)?;
            Ok(())
        })?;
        let byte = resumed(&mut m, &end, &feed, fed, |m| {
            let mut bytes = Vec::new();
            body::save_state(m, &mut bytes)?;
            m.reset()?;
            let mut input = bytes.as_slice();
            body::restore_state(m, &mut input)?;
            Ok(())
        })?;
        let want = &straight.ids[SNAP_AT..];
        let pass = snap == want && byte == want;
        ok &= pass;
        println!(
            "{NAME}: (v) snapshot/resume and save_state/restore_state after step {SNAP_AT}: {} \
             and {} of the uninterrupted {} ids identical: {}",
            snap.len(),
            byte.len(),
            want.len(),
            verdict(pass)
        );

        // (vi): the refusals, by name. The model stands at the resumed state
        // of the ending layout, its span in the history.
        crate::ds41_tier::sc("(vi) refusals: a cut inside a span, a torn span, media under steps")?;
        let cut_ok = refused(m.rollback(u32::try_from(s0 + 40)?), "inside the media span");
        ok &= cut_ok;
        println!(
            "{NAME}: (vi) a cut to {} inside the span {s0}..{e0} refused by name: {}",
            s0 + 40,
            verdict(cut_ok)
        );
        let torn = MediaSpan {
            at: mid.at..mid.at + span_len,
            rows: &feed.rows,
            kinds: &feed.kinds,
        };
        let whole_ok = refused(
            body::prefill_media(&mut m, &mid.ids[..mid.at + 4], &[torn]),
            "a span whole or not at all",
        );
        ok &= whole_ok;
        println!(
            "{NAME}: (vi) a span the call does not hold whole refused by name: {}",
            verdict(whole_ok)
        );
        m.body_parts(NAME)?.2.set_prefill_mode(PrefillMode::Steps);
        let engine_steps = body::prefill_media(&mut m, &mid.ids, &[mid.span(&feed)]);
        let mut s = Session::from_model(m, u32::try_from(workstation::CTX_MAX)?);
        let session_steps = s.prompt_media(&mid.ids, &[mid.span(&feed)], Want::Argmax);
        let steps_ok = refused(
            engine_steps,
            "media in a prompt under BLOOMERY_PREFILL=steps",
        ) && matches!(
            &session_steps,
            Err(app::SessionError::Refused(msg)) if msg.contains("BLOOMERY_PREFILL=steps")
        );
        ok &= steps_ok;
        println!(
            "{NAME}: (vi) media under the steps feed refused by name (engine and session): {}",
            verdict(steps_ok)
        );
        let m = s.model_mut();
        m.body_parts(NAME)?.2.set_prefill_mode(PrefillMode::Batch);
        m.reset()?;

        println!("{NAME}: {}", tier::tally_line());
        if ok {
            println!("gate-gpu-ds41-media: PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
