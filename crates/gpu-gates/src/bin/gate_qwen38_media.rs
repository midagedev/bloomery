//! The image-input gate of Qwen3.8-Flash-Next (`qwen4exp`, `Body38`): a media prompt call
//! (`MediaBody::prefill_media`) against the oracle sets of mainline's mtmd on the same file
//! (`refset::arch::qwen4exp::vis`, `just dump-ref-qvis qwen4exp`): set B's tower rows (rounded to bf16, the
//! rows the C″ runs were fed) as the spans' rows, set C's three prompts as the oracle of the positions, set
//! F's two greedy decodes as the oracle of the ids, and set E as the oracle of the prompt the server renders.
//! The text file is the UD-Q4_K_XL split set, loaded through the planner (`shared/qwen38_open.rs`: the routed
//! experts on the card's id prefix, the host beyond it) on the gate card's bytes less the tower's reserve and
//! the rotation tables'. The gate holds the real file only (`tools/ref/real-only.sh`). Every clause is
//! FAIL-first by a mutant of its own, one line each:
//!
//! - (A) injection: each unit's embedding rows as layer 0 reads them (`set_media_tap`): the rows at image
//!   positions are the widened bf16 rows bit for bit, and the text rows equal the same ids' rows of a call
//!   with no span (its span positions carry a filler id: the image placeholder is the PLE hash's inside spans
//!   only). The units are the walk's, so a span a unit boundary cuts is read in each of its units (c2's second
//!   image crosses two). MUTANT: the media rows copied one position late (the unit's feed position).
//! - (B) positions and the table: our (t, h, w) of every row equal set C's dumped integer arrays, and the
//!   card's rotation table (`rope_rows`) equals the plain table (the text load's) gathered by them through the
//!   interleaved sector rule written out here, bit for bit; the same table equals the host's full rebuild
//!   (`MropeSeq::rows_into`) from the sequence's record. MUTANT: `SeqRope::push_images` returns at its first
//!   line.
//! - (C) identity: Qwen3.6's projector (a narrower aligner) is refused by name against this text file through
//!   `vision_seat::open_seat` with the file's facts and the vocabulary's tokens, and Qwen3.8's own is accepted
//!   at its digest. MUTANT: the width comparison removed in `Projector::check_text`.
//! - (D) text equality and refusals on the media load: a text request through the media call with no span and
//!   through the plain ubatch call equals the text load's, token and logits bit for bit (the plain call also
//!   every position's final streams), fresh and after image calls and a `reset`; a call that starts strictly
//!   inside a held image (a media call, a media call with a span after the image's end, the plain prompt on the
//!   ubatch and the pass paths, a decode step) is refused by name with the position, the table and the record as
//!   they were; a cut back to an image's start then text equals the same text from a fresh run split the same
//!   way, and a cut inside an image is refused before anything moves; a media call on a capturing stream is
//!   refused by name before anything moves; the text load refuses an image span, a record of images and the tap
//!   by name, and `enable_media` after a call or twice; a step graph captured on the media load before any
//!   media call and replayed after an image call equals the eager step. MUTANTS, one per sub-clause: the media
//!   load's table built one row late (the table and the no-span call); the call-start refusal
//!   (`refuse_inside_image`) removed; `rollback_on` leaving the table uncut; the capturing check removed from
//!   the media call.
//! - (E) decode: after prompts c1 and c2, the first token and every greedy step equal set F's ids on the prefix
//!   where mainline's CUDA run and its CPU twin take the same ids (the line says how long), or parting at a
//!   near tie (our logit lead over set F's token below `NEAR_TIE`), each step fed set F's own id; our (t, h, w) of
//!   each decoded row and the table row the step reads equal the position set F
//!   dumped (`n_past`). MUTANT: the decode step reads the shared table (`enqueue_chain` keeps `table:
//!   &rope.table`). The hidden band of set C is not here: `Body38`'s head mix is the final norm and no host
//!   rule of it norms the unit sink's streams, so a band per region has no instrument yet.
//! - (F) the tower rows through the seat's own encoder path: `vision_seat::Resident::open` and `encode` of
//!   set C's images, each against set B's `embd` within `K_TAP` times the largest of the CPU twin's distance,
//!   the bf16 rounding floor and the network's sensitivity to a nudged input (`shared/vision_taps.rs`'s
//!   `band_check`, as `gate_qwen3vl_tower` holds a tap). MUTANT: `merged_cells` rotates the rows by one cell.
//! - (G) one answer through serve: `bloomery-serve --model qwen38 -m <file> --mmproj <projector>` on the gate
//!   placement renders c1's request (an `image_url` data URI of set C's image and a short question) to the
//!   prompt set E holds for it (its image run cut to the one placeholder the template writes), evaluates the
//!   set's count, and answers greedily with ids equal to set F's over the prefix its CPU twin agrees on, or
//!   parting at a near tie (`NEAR_TIE`); the oracle is mainline's own ids, the cheapest honest one, as
//!   `gate_qwen3_serve` holds its seat. The same request to a server with no `--mmproj` is the server's 400.
//!   MUTANT: the span's rows zeroed in the seat's `prefill_media`.
//!
//! The serve prompt cache and slot file are the seat gates'; the slot clauses (a pass of an image slot and a
//! text slot, a parked sequence) are not here.

#[cfg(all(feature = "vision", feature = "deepseek41"))]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen38_media", gate::run())
}

#[cfg(all(feature = "vision", feature = "deepseek41"))]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(all(feature = "vision", feature = "deepseek41"))]
#[path = "shared/qwen38_open.rs"]
#[allow(
    dead_code,
    reason = "the gate prints the plan's counts and opens the body; the other facts serve the e2e gates"
)]
mod qwen38_open;

#[cfg(all(feature = "vision", feature = "deepseek41"))]
#[path = "shared/vision_taps.rs"]
#[allow(
    dead_code,
    reason = "the gate takes the band check, the nudge and their distances; the rest serves the tower gates"
)]
mod vision_taps;

#[cfg(all(feature = "vision", feature = "deepseek41"))]
mod gate {
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use crate::gate_card;
    use crate::qwen38_open::{self, Open38};
    use crate::vision_taps::{K_TAP, band_check, dist_f32, floor, nudge, sens};
    use bloomery_gpu::arch::qwen3moe::{Body38, Prompt38, Qwen38Model, TargetRows};
    use bloomery_gpu::host::served::prompt_pass;
    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu::kv_format::KvQ8;
    use bloomery_gpu::media::{ImageShape, ImageSpan, MediaBody, MediaSpan};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{GpuError, capture_template};
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::serve_client::{
        self, Served, answered, check, curl, greedy_chat, json_of, serve_cmd, tokenized,
    };
    use bloomery_gpu_gates::vision_seat::{self, Image, Opened, PleImage, TextFacts};
    use bloomery_gpu_gates::{
        GateError, checks_failed, equal, first_diff, oracle_row, ref_model_path, verdict,
    };
    use bloomery_gpu_vision::f32_bf16;
    use bloomery_levers::HostCfg;
    use gguf::Split;
    use model::arch::qwen35moe::place::Experts;
    use model::placement::PlanLevers;
    use model::placement::workstation::{CardSpec, PLAN_ROOM};
    use refset::arch::qwen4exp::MODEL;
    use refset::arch::qwen4exp::vis::{
        CHATIDS, DECODE, HIDDEN, MMPROJ_BF16, MMPROJ_BF16_SHA256, TAPS,
    };
    use refset::arch::qwen35moe::vis::MMPROJ_BF16 as MMPROJ_QWEN36;
    use refset::clefvis::{ClefvisSet, EMBD, IMAGE_PAD_ID, Span};
    use serde_json::{Value, json};
    use tokenizer::Tokenizer;
    use vision::mrope::{MropeSeq, SectionOrder};
    use vision::{EncoderInput, Rgb8};

    /// The rotation table's values a position, and the text width.
    const ROT: usize = 64;
    const WIDTH: usize = 2560;
    /// The load's cache rows: set C's longest prompt and set F's steps fit.
    const CTX: usize = 4096;
    /// The walk's ubatch: c2's second image crosses two unit boundaries at it.
    const UBATCH: usize = 512;
    /// The file's `rope.dimension_sections`: the sectors the written-out rule reads.
    const SECTIONS: [u32; 4] = [11, 11, 10, 0];
    /// The id a span's positions carry in a call with no span: any id the PLE hash takes.
    const FILLER: u32 = 1000;
    /// The ids of the text request that leaves the walk's second unit a short tail.
    const SHORT_TEXT: usize = 517;
    /// The fewest ids a prompt's CUDA and CPU oracle runs must agree on for set F to hold anything: a shorter
    /// prefix says too little of the image to be a clause.
    const MIN_PREFIX: usize = 8;
    /// The widest gap between the top two logprobs at which an id of the oracle's that the server parts from
    /// counts as a near tie, in nats; the decode clause reads its flips' logit leads against it too.
    // PIN(2026-10-18): `gate_qwen3_serve`'s F_NEAR_TIE (its PIN of 2026-10-12), taken for the same served-answer-to-mainline comparison on this seat and not yet read on this seat's own spread [provisional].
    const NEAR_TIE: f64 = 0.5;
    /// The server's start: polls a few seconds apart, as long as a cold load of the file takes.
    const POLLS: usize = 240;
    const POLL: Duration = Duration::from_secs(5);
    /// The chat requests set E was dumped from, one JSON line a request, and the images they name.
    const REQUESTS: &str = "../../tools/ref/qvis/requests.jsonl";
    const IMAGE_DIR: &str = "../../tools/ref/clefvis/images";
    /// What the text load names when it refuses a media call, a record of images, the tap or a second
    /// enabling: the error names the media or the call that enables it.
    const TEXT_LOAD: &[&str] = &["without media", "with media", "enable_media"];
    /// What a server with no `--mmproj` answers an image part with.
    const NO_IMAGE: &str = "image input is not supported: this server reads text only";

    // ------------------------------------------------------------ files

    /// The file's width, rotary dimension count, sections and PLE image id against the constants the
    /// clauses are written with: a file of another shape fails here by name.
    fn check_shape(split: &Split) -> Result<(), GateError> {
        let get = |k: &str| {
            split
                .value(&split.arch_key(k))
                .and_then(gguf::Value::as_unsigned)
        };
        let (width, rot, ple) = (
            get("embedding_length"),
            get("rope.dimension_count"),
            get("ple.image_token_id"),
        );
        let sections: Option<Vec<u64>> = split
            .arch_get_arr("rope.dimension_sections")
            .map(|a| a.iter().filter_map(gguf::Value::as_unsigned).collect());
        let want_sections: Vec<u64> = SECTIONS.iter().map(|&s| u64::from(s)).collect();
        if width != u64::try_from(WIDTH).ok()
            || rot != u64::try_from(ROT).ok()
            || sections.as_deref() != Some(want_sections.as_slice())
            || ple != Some(u64::from(IMAGE_PAD_ID))
        {
            return Err(format!(
                "{MODEL}: width {width:?}, rotary dims {rot:?}, sections {sections:?}, PLE image id \
                 {ple:?}; the clauses are written for {WIDTH}, {ROT}, {SECTIONS:?} and \
                 {IMAGE_PAD_ID}"
            )
            .into());
        }
        Ok(())
    }

    /// The PNG of set image `name`, held to the digest the set recorded for it.
    fn png_of(set: &ClefvisSet, name: &str) -> Result<Vec<u8>, GateError> {
        let image = set.image(name)?;
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(IMAGE_DIR)
            .join(format!("{name}.png"));
        if gguf::digest::sha256_file(&path)? != image.png_sha256 {
            return Err(format!(
                "{} is not the image the sets were dumped from ({})",
                path.display(),
                image.png_sha256
            )
            .into());
        }
        Ok(std::fs::read(&path)?)
    }

    /// A set C prompt, its span rows and its positions.
    struct Case {
        name: &'static str,
        /// The sha256 of the tokens file the set was dumped from.
        tokens_sha: String,
        ids: Vec<u32>,
        spans: Vec<Span>,
        /// Each span's bf16 rows: set B's tower rows of the span's image, rounded to nearest even.
        rows: Vec<Vec<u16>>,
        /// Set C's (t, y, x) per sequence index.
        pos: Vec<[i32; 3]>,
    }

    impl Case {
        /// Prompt `n` (0-based) of set C: its ids read from the tokens file the set names, held to the
        /// header's digest and count, the image-pad run of each span checked and no placeholder left
        /// outside a span (the PLE hash refuses one), and the rows of each span's image from the tower
        /// set `tower`.
        fn load(n: usize, name: &'static str, tower: &ClefvisSet) -> Result<Case, GateError> {
            let hidden = ClefvisSet::open(&HIDDEN.path(HIDDEN.sets[n]), &HIDDEN)?;
            let h = &hidden.man.header;
            let file = h
                .tokens_file
                .as_deref()
                .ok_or_else(|| format!("{name}: no `# tokens_file` line"))?;
            let want = h
                .tokens_file_sha256
                .as_deref()
                .ok_or_else(|| format!("{name}: no `# tokens_file_sha256` line"))?;
            let got = gguf::digest::sha256_file(Path::new(file))?;
            let count = hidden.n_ids()?;
            if got != want || h.tokens_count != Some(u64::try_from(count)?) {
                return Err(format!(
                    "{name}: the set was dumped from {want} ({:?} ids); {file} is {got}",
                    h.tokens_count
                )
                .into());
            }
            let ids = refset::tokenizer::read_ids(Path::new(file), count)?;
            let in_span = |p: usize| {
                hidden
                    .spans
                    .iter()
                    .any(|s| (s.at..s.at + s.len).contains(&p))
            };
            if let Some(p) = (0..ids.len()).find(|&p| ids[p] == IMAGE_PAD_ID && !in_span(p)) {
                return Err(format!("{name}: the placeholder at {p} is outside every span").into());
            }
            let mut rows = Vec::new();
            for s in &hidden.spans {
                if s.nx * s.ny != s.len
                    || ids[s.at..s.at + s.len].iter().any(|&i| i != IMAGE_PAD_ID)
                {
                    return Err(format!("{name}: span {} is not an image-pad run", s.index).into());
                }
                let (_, tower_rows) = tower.tap(&s.image, EMBD)?;
                if tower_rows.len() != s.len * WIDTH {
                    return Err(
                        format!("{name}: {} tower rows for {}", tower_rows.len(), s.image).into(),
                    );
                }
                rows.push(tower_rows.iter().map(|&x| f32_bf16(x)).collect());
            }
            let pos = hidden.positions()?;
            if pos.len() != ids.len() {
                return Err(format!("{name}: the set's positions are not the ids' count").into());
            }
            Ok(Case {
                name,
                tokens_sha: got,
                ids,
                spans: hidden.spans.clone(),
                rows,
                pos,
            })
        }

        /// The call's spans, for a call whose ids start at id `from` of the prompt.
        fn image_spans(&self, from: usize) -> Vec<ImageSpan<'_>> {
            self.spans
                .iter()
                .zip(&self.rows)
                .map(|(s, rows)| ImageSpan {
                    span: MediaSpan {
                        at: s.at - from..s.at + s.len - from,
                        rows,
                    },
                    shape: ImageShape::Grid(s.nx, s.ny),
                })
                .collect()
        }

        /// The ids with every span's positions a filler id: a prompt the text paths take.
        fn text_ids(&self) -> Vec<u32> {
            let mut v = self.ids.clone();
            for s in &self.spans {
                v[s.at..s.at + s.len].fill(FILLER);
            }
            v
        }

        fn n_image(&self) -> usize {
            self.spans.iter().map(|s| s.len).sum()
        }

        /// The sequence's record of the prompt's images.
        fn record(&self) -> Result<MropeSeq, GateError> {
            let mut rec = MropeSeq::new();
            for s in &self.spans {
                rec.push(s.at, s.nx, s.ny).map_err(|e| e.to_string())?;
            }
            Ok(rec)
        }
    }

    // ------------------------------------------------------------ helpers

    /// Where a model stands: its position, the rotation table its rope launches read and the images
    /// its record holds, each `(first row, nx, ny)`.
    type Stand<'a> = (usize, &'a [f32], &'a [(usize, usize, usize)]);

    fn spans_of(m: &MropeSeq) -> Vec<(usize, usize, usize)> {
        m.spans().collect()
    }

    /// Whether `r` is an error whose text holds one of `wants`, with one line.
    fn refused<T>(what: &str, r: Result<T, GpuError>, wants: &[&str]) -> bool {
        let named = wants.join("\" or \"");
        match r {
            Err(e) if wants.iter().any(|w| e.to_string().contains(w)) => {
                println!("    {what}: refused (\"{named}\") ok");
                true
            }
            Err(e) => {
                println!("    {what}: refused, but not by \"{named}\": {e}  FAIL");
                false
            }
            Ok(_) => {
                println!("    {what}: accepted  FAIL");
                false
            }
        }
    }

    // ------------------------------------------------------------ the rotation table

    /// The (t, h, w) of sequence row `r` of prompt `c` as set C feeds it: the dumped triple inside the
    /// prompt, the text position after it (the last row's position, advanced) beyond.
    fn feed_position(c: &Case, r: usize) -> (usize, usize, usize) {
        let n = c.ids.len();
        if r < n {
            let p = c.pos[r];
            return (p[0] as usize, p[1] as usize, p[2] as usize);
        }
        let saved = (n - 1) - c.pos[n - 1][0] as usize;
        (r - saved, r - saved, r - saved)
    }

    // ------------------------------------------------------------ calls

    fn prompt_media(
        m: &mut Qwen38Model,
        ids: &[u32],
        spans: &[ImageSpan<'_>],
    ) -> Result<u32, GpuError> {
        prompt_pass(m, |m| <Body38 as MediaBody>::prefill_media(m, ids, spans))
    }

    /// A call's token after it and the logits of its last position.
    struct Tail {
        next: u32,
        logits: Vec<f32>,
    }

    /// `ids` through the media call with no span, from where the model stands.
    fn via_media(m: &mut Qwen38Model, ids: &[u32]) -> Result<Tail, GateError> {
        let next = prompt_media(m, ids, &[])?;
        Ok(Tail {
            next,
            logits: m.logits()?,
        })
    }

    /// `ids` through the plain ubatch call from a reset: the call's [`Tail`] and every position's final
    /// streams, read from the unit sink.
    fn via_plain(m: &mut Qwen38Model, ids: &[u32]) -> Result<(Tail, Vec<f32>), GateError> {
        m.reset()?;
        let wide = m.body("gate_qwen38_media")?.mtp_tap_width();
        let mut streams = vec![0.0f32; ids.len() * wide];
        let next = {
            let mut sink = |mm: &mut Qwen38Model,
                            walk: TargetRows,
                            first: u32,
                            rows: usize|
             -> Result<(), GpuError> {
                let f = first as usize;
                let v = mm.target_streams(walk, rows)?;
                streams[f * wide..(f + rows) * wide].copy_from_slice(&v);
                Ok(())
            };
            m.prompt38_with(ids, Prompt38::Gemm, Some(&mut sink))?
        };
        let tail = Tail {
            next,
            logits: m.logits()?,
        };
        Ok((tail, streams))
    }

    /// `a` and `b` the same call's token and logits, bit for bit.
    fn same_tail(what: &str, a: &Tail, b: &Tail) -> bool {
        let ids = a.next == b.next;
        println!(
            "    {what}: token {} against {} {}",
            a.next,
            b.next,
            verdict(ids)
        );
        equal(&format!("{what}: logits"), &a.logits, &b.logits) && ids
    }

    /// A text request: ids a text load takes.
    struct Req {
        name: String,
        ids: Vec<u32>,
    }

    /// The text load's answer to one request: its call's tail and every position's final streams.
    struct Reference {
        tail: Tail,
        streams: Vec<f32>,
    }

    // ------------------------------------------------------------ the load

    /// What every load of the gate is planned and opened by: the gate card's bytes less the tower's reserve
    /// and the rotation tables', one sequence, the routed experts on the card's id prefix.
    struct Load {
        card: CardSpec,
        host: HostCfg,
        plan_levers: PlanLevers,
    }

    impl Load {
        /// The file planned and opened, a media load (`enable_media`) when `media`.
        fn open(&self, media: bool) -> Result<Qwen38Model, GateError> {
            let file = qwen38_open::open_split(Path::new(MODEL))?;
            let t = Instant::now();
            let planned = Open38 {
                card: self.card,
                ctx: CTX,
                ub: UBATCH,
                experts: Experts::Card,
                slots: 1,
                plan_levers: &self.plan_levers,
                host: self.host,
                // PIN(2026-10-28): the plan read `MemAvailable`, which another process's load
                // moves; the room is the workstation's own.
                room: Some(PLAN_ROOM),
                residency: Residency::Off,
                kv: KvQ8::F16,
            }
            .plan(file)?;
            let p = planned.facts();
            println!(
                "plan card={} ctx_max={} ubatch={UBATCH} host_experts={} card_experts={} \
                 card_budget={} media={media}",
                self.card.name,
                p.ctx_max,
                p.host_experts,
                p.card_experts,
                self.plan_levers.card_budget_bytes
            );
            let mut model = planned.open()?.model;
            println!(
                "load resident_bytes={} ctx={CTX} layers={} in {:.1} s (runtime value)",
                model.resident_bytes(),
                model.layers().len(),
                t.elapsed().as_secs_f64()
            );
            if media {
                model.enable_media()?;
            }
            Ok(model)
        }
    }

    // ------------------------------------------------------------ (C) identity

    /// (C): Qwen3.6's projector refused by name against this text file and Qwen3.8's accepted at its
    /// digest, both through the seat's own check. The accepted file, opened and seated, is the one the
    /// tower clause loads.
    fn clause_identity(path: &Path) -> Result<(bool, Opened), GateError> {
        let tok = Tokenizer::from_gguf(path)?;
        let token = |id: u32| tok.text(id).map(str::to_owned);
        let facts = TextFacts {
            n_embd: WIDTH,
            imrope: Some(SECTIONS),
            ple: PleImage::Id(IMAGE_PAD_ID),
        };
        let takes = [vision::arch::qwen3vl::PROJECTOR_TYPE];
        let flags = vision_seat::Flags::default();
        let want = "the encoder's aligner rows are";
        let refused_36 = match vision_seat::open_seat(
            &flags,
            Some(Path::new(MMPROJ_QWEN36)),
            Place::Gate,
            &takes,
            &facts,
            &token,
        ) {
            Err(e) if e.to_string().contains(want) => {
                println!("    Qwen3.6's projector: refused (\"{want}\") ok: {e}");
                true
            }
            Err(e) => {
                println!("    Qwen3.6's projector: refused, but not by \"{want}\": {e}  FAIL");
                false
            }
            Ok(_) => {
                println!("    Qwen3.6's projector: accepted  FAIL");
                false
            }
        };
        let got = gguf::digest::sha256_file(Path::new(MMPROJ_BF16))?;
        if got != MMPROJ_BF16_SHA256 {
            return Err(format!(
                "{MMPROJ_BF16} is {got}; the Qwen3.8 sets were dumped with {MMPROJ_BF16_SHA256}"
            )
            .into());
        }
        let opened = vision_seat::open_seat(
            &flags,
            Some(Path::new(MMPROJ_BF16)),
            Place::Gate,
            &takes,
            &facts,
            &token,
        )?
        .ok_or("the seat opened no encoder for --mmproj")?;
        let accepted = opened.path == Path::new(MMPROJ_BF16) && opened.cap > 0;
        println!(
            "    Qwen3.8's projector: accepted {accepted}; the tower's card figure {} bytes, at most {} \
             tokens an image",
            opened.card.total(),
            opened.cap
        );
        let ok = refused_36 && accepted;
        println!("  (C) identity  {}", verdict(ok));
        Ok((ok, opened))
    }

    // ------------------------------------------------------------ (F) the tower rows

    /// (F): each of `images`, prepared by the model's own prepare and encoded on the resident tower the
    /// seat opens beside the model, against set B's `embd` within [`K_TAP`] times the largest of the CPU
    /// twin's distance, the bf16 rounding floor and the network's sensitivity to a nudged input.
    fn clause_tower(
        m: &Qwen38Model,
        opened: &Opened,
        (tower, twin): (&ClefvisSet, &ClefvisSet),
        images: &[String],
    ) -> Result<bool, GateError> {
        let ctx = m.gpu().context();
        let mut resident = vision_seat::Resident::open(ctx, opened)?;
        let mut ok = resident.n_embd() == WIDTH;
        println!(
            "  (F) the resident tower writes rows {} wide  {}",
            resident.n_embd(),
            verdict(ok)
        );
        for name in images {
            let rgb = Rgb8::from_png(&png_of(tower, name)?)?;
            let prepared = opened.media.prepare(&rgb)?;
            let info = tower.image(name)?;
            let encoded = resident.encode(ctx, std::iter::once(Image::of(0, &prepared)))?;
            let ours = &encoded.rows().rows;
            let shaped = prepared.span_len == info.n_tokens && ours.len() == info.n_tokens * WIDTH;
            println!(
                "  (F) {name}: span {} (set B {}), {} values {}",
                prepared.span_len,
                info.n_tokens,
                ours.len(),
                verdict(shaped)
            );
            let mut noisy = prepared.clone();
            match &mut noisy.input {
                EncoderInput::Grid(patches) => nudge(&mut patches.bf16),
                EncoderInput::Tiles(_) => {
                    return Err(format!("{name}: the model prepared tiles, not a grid").into());
                }
            }
            let nudged = resident.encode(ctx, std::iter::once(Image::of(0, &noisy)))?;
            let (_, refv) = tower.tap(name, EMBD)?;
            let (_, twin_v) = twin.tap(name, EMBD)?;
            ok &= shaped
                && band_check(
                    K_TAP,
                    name,
                    "embd",
                    ours,
                    &refv,
                    dist_f32(&twin_v, &refv).rms_rel,
                    floor(&refv),
                    Some(sens(&nudged.rows().rows, ours)),
                );
        }
        println!("  (F) tower rows  {}", verdict(ok));
        Ok(ok)
    }

    // ------------------------------------------------------------ (A) and (B)

    /// (A): the unit rows of an image call against the widened rows and a no-span call's rows.
    fn clause_inject(c: &Case, tap: &[(usize, Vec<f32>)], text: &[(usize, Vec<f32>)]) -> bool {
        let n = c.ids.len();
        let want: Vec<usize> = (0..n).step_by(UBATCH).collect();
        let firsts: Vec<usize> = tap.iter().map(|u| u.0).collect();
        let shaped = firsts == want
            && text.iter().map(|u| u.0).eq(firsts.iter().copied())
            && tap
                .iter()
                .all(|(f, r)| r.len() == (n.min(f + UBATCH) - f) * WIDTH);
        let (mut img, mut img_bad, mut txt, mut txt_bad, mut first_bad) = (0, 0, 0, 0, None);
        for (u, (first, rows)) in tap.iter().enumerate() {
            for (i, row) in rows.as_chunks::<WIDTH>().0.iter().enumerate() {
                let p = first + i;
                let ok = if let Some(j) = c
                    .spans
                    .iter()
                    .position(|s| (s.at..s.at + s.len).contains(&p))
                {
                    img += 1;
                    let from = (p - c.spans[j].at) * WIDTH;
                    let widened: Vec<f32> = c.rows[j][from..from + WIDTH]
                        .iter()
                        .map(|&b| f32::from_bits(u32::from(b) << 16))
                        .collect();
                    let ok = first_diff(row, &widened).is_none();
                    img_bad += usize::from(!ok);
                    ok
                } else {
                    txt += 1;
                    let ok = text
                        .get(u)
                        .and_then(|t| t.1.get(i * WIDTH..(i + 1) * WIDTH))
                        .is_some_and(|t| first_diff(row, t).is_none());
                    txt_bad += usize::from(!ok);
                    ok
                };
                if !ok && first_bad.is_none() {
                    first_bad = Some(p);
                }
            }
        }
        let ok =
            shaped && img == c.n_image() && txt == n - c.n_image() && img_bad == 0 && txt_bad == 0;
        println!(
            "  (A) {}: {} units at {firsts:?} (want {want:?}); {img} image rows, {img_bad} off the widened \
             bf16 row; {txt} text rows, {txt_bad} off the no-span call's; first bad {first_bad:?}  {}",
            c.name,
            tap.len(),
            verdict(ok)
        );
        ok
    }

    /// (B) of the sequence the call left on `m`, and the host rebuild of its table.
    fn clause_table(m: &Qwen38Model, c: &Case, plain: &[f32]) -> Result<bool, GateError> {
        let n = c.ids.len();
        let saved = m.rope_saved()?;
        let wrong = (0..n).find(|&r| {
            let (t, h, w) = saved.pos(r);
            [t as i32, h as i32, w as i32] != c.pos[r]
        });
        let want_spans: Vec<_> = c.spans.iter().map(|s| (s.at, s.nx, s.ny)).collect();
        let ok_pos = wrong.is_none() && spans_of(&saved) == want_spans;
        println!(
            "  (B) {}: the (t, h, w) of {n} rows against set C's, first differing row {wrong:?}, record \
             {:?}  {}",
            c.name,
            spans_of(&saved),
            verdict(ok_pos)
        );
        let table = m.rope_rows()?;
        let rows = table.len() / ROT;
        let bad = (0..rows).find(|&r| {
            first_diff(
                &table[r * ROT..(r + 1) * ROT],
                &oracle_row::<ROT>(plain, feed_position(c, r)),
            )
            .is_some()
        });
        let ok_table = table.len() == plain.len() && table.len() % ROT == 0 && bad.is_none();
        println!(
            "  (B) {}: {rows} table rows against the plain table gathered by set C's positions, first \
             differing row {bad:?}  {}",
            c.name,
            verdict(ok_table)
        );
        let mut full = vec![0.0f32; table.len()];
        saved
            .rows_into(plain, ROT, SECTIONS, SectionOrder::Tyxz, 0..rows, &mut full)
            .map_err(|e| e.to_string())?;
        let ok_host = equal(
            &format!("(B) {}: the card table vs the host rebuild", c.name),
            &table,
            &full,
        );
        Ok(ok_pos && ok_table && ok_host)
    }

    /// One prompt through the media call: (A) and (B).
    fn run_case(m: &mut Qwen38Model, c: &Case, plain: &[f32]) -> Result<bool, GateError> {
        println!(
            " {} ({} ids, {} images):",
            c.name,
            c.ids.len(),
            c.spans.len()
        );
        m.reset()?;
        let mut ok = equal(
            "(D) the table after a reset vs the text load's",
            &m.rope_rows()?,
            plain,
        );
        m.set_media_tap(true)?;
        prompt_media(m, &c.text_ids(), &[])?;
        let text = m.take_media_tap()?;
        m.reset()?;
        prompt_media(m, &c.ids, &c.image_spans(0))?;
        let tap = m.take_media_tap()?;
        m.set_media_tap(false)?;
        ok &= clause_inject(c, &tap, &text);
        ok &= clause_table(m, c, plain)?;
        Ok(ok)
    }

    // ------------------------------------------------------------ (D)

    /// The requests on the media load, through the media call with no span and the plain ubatch call,
    /// against the text load's answers.
    fn clause_text(
        m: &mut Qwen38Model,
        plain: &[f32],
        reqs: &[Req],
        refs: &[Reference],
        when: &str,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        m.reset()?;
        ok &= equal(
            &format!("(D) {when}: the table vs the text load's"),
            &m.rope_rows()?,
            plain,
        );
        for (r, want) in reqs.iter().zip(refs) {
            m.reset()?;
            let media = via_media(m, &r.ids)?;
            ok &= same_tail(
                &format!("(D) {when} {}: no-span media call vs the text load", r.name),
                &media,
                &want.tail,
            );
            let (direct, streams) = via_plain(m, &r.ids)?;
            ok &= same_tail(
                &format!("(D) {when} {}: plain call vs the text load", r.name),
                &direct,
                &want.tail,
            );
            ok &= equal(
                &format!("(D) {when} {}: every position's final streams", r.name),
                &streams,
                &want.streams,
            );
        }
        println!("  (D) text on the media load, {when}  {}", verdict(ok));
        Ok(ok)
    }

    /// A media call on a capturing stream.
    fn clause_capture(m: &mut Qwen38Model, c: &Case, plain: &[f32]) -> Result<bool, GateError> {
        m.reset()?;
        let stream = m.gpu().stream_handle();
        let end = c.spans[0].at + c.spans[0].len + 16;
        let ids = &c.ids[..end];
        let spans = c.image_spans(0);
        let r = capture_template(&stream, |_| prompt_media(m, ids, &spans).map(|_| ()));
        let mut ok = refused("(D) a media call while capturing", r, &["eager"]);
        let still = m.pos() == 0 && m.rope_saved()?.is_empty();
        let table = m.rope_rows()?;
        ok &= still && first_diff(&table, plain).is_none();
        println!(
            "  (D) refused before anything moved: position {}, record empty {}  {}",
            m.pos(),
            still,
            verdict(ok)
        );
        Ok(ok)
    }

    /// One refused call of the call-start clause: refused by name, and the position, the table and the
    /// record as they were.
    fn stood(
        m: &Qwen38Model,
        what: &str,
        r: Result<(), GpuError>,
        (at, table, held): Stand<'_>,
    ) -> Result<bool, GateError> {
        let kept = m.pos() as usize == at
            && first_diff(&m.rope_rows()?, table).is_none()
            && spans_of(&m.rope_saved()?) == held;
        let named = refused(what, r, &["inside the image"]);
        println!(
            "    {what}: position, table and record as they were {}",
            verdict(kept)
        );
        Ok(named && kept)
    }

    /// A model that stands at position 100 with c1's image record restored ahead of it: every call that
    /// starts there is refused, and the position, the table and the record stay.
    fn clause_inside(m: &mut Qwen38Model, c: &Case) -> Result<bool, GateError> {
        const AT: usize = 100;
        let image = &c.spans[0];
        if !(image.at < AT && AT < image.at + image.len) {
            return Err(format!(
                "{}: position {AT} is not inside the image at {}..{}",
                c.name,
                image.at,
                image.at + image.len
            )
            .into());
        }
        let text = c.text_ids();
        m.reset()?;
        prompt_media(m, &text[..AT], &[])?;
        m.rope_restore(c.record()?)?;
        let (table, held) = (m.rope_rows()?, spans_of(&m.rope_saved()?));
        println!(
            "  call start: position {AT} inside the image at {}..{} (the record restored ahead of the state)",
            image.at,
            image.at + image.len
        );
        // A span after the image's end: its positions hold the placeholder, the rest of the call text.
        let after = (image.at + image.len).saturating_sub(AT) + 4;
        let mut late_ids = text[AT..(AT + after + 20).min(text.len())].to_vec();
        if late_ids.len() < after + 4 {
            return Err(format!(
                "{}: too few ids after position {AT} for a late span",
                c.name
            )
            .into());
        }
        late_ids[after..after + 4].fill(IMAGE_PAD_ID);
        let late = ImageSpan {
            span: MediaSpan {
                at: after..after + 4,
                rows: &c.rows[0][..4 * WIDTH],
            },
            shape: ImageShape::Grid(2, 2),
        };
        let stand = (AT, table.as_slice(), held.as_slice());
        let mut ok = true;
        let r = prompt_media(m, &text[AT..AT + 60], &[]).map(|_| ());
        ok &= stood(m, "a text media call", r, stand)?;
        let r = prompt_media(m, &late_ids, &[late]).map(|_| ());
        ok &= stood(
            m,
            "a media call with a span after the image's end",
            r,
            stand,
        )?;
        let r = m.prompt38(&text[AT..AT + 60], Prompt38::Gemm).map(|_| ());
        ok &= stood(m, "a plain ubatch prompt", r, stand)?;
        let r = m.prompt38(&text[AT..AT + 60], Prompt38::Pass).map(|_| ());
        ok &= stood(m, "a plain pass prompt", r, stand)?;
        let r = m.step(&[text[AT]]).map(|_| ());
        ok &= stood(m, "a decode step", r, stand)?;
        m.reset()?;
        println!("  call start  {}", verdict(ok));
        Ok(ok)
    }

    /// The cut back to an image's start and its refusal inside an image. The calls that take the
    /// checkpoints are the prompt calls, with the load's checkpoints on: a call that continues a sequence
    /// takes one at its start and one at its end.
    fn clause_cut(
        m: &mut Qwen38Model,
        c1: &Case,
        c2: &Case,
        plain: &[f32],
    ) -> Result<bool, GateError> {
        m.body_parts("gate_qwen38_media")?.2.set_checkpoints(true);
        let at = c1.spans[0].at;
        let after = at + c1.spans[0].len;
        let text = c1.text_ids();
        let before = &text[..at];
        // The cut run: the text before the image, the image and the text after it, a cut back to the
        // image's start, the text after the image again. The reference: the text before the image, then the
        // same text.
        m.reset()?;
        prompt_media(m, before, &[])?;
        prompt_media(m, &c1.ids[at..], &c1.image_spans(at))?;
        m.rollback(u32::try_from(at)?)?;
        let cut_state = m.pos() as usize == at && m.rope_saved()?.is_empty();
        let cut_table = m.rope_rows()?;
        let cut = via_media(m, &c1.ids[after..])?;
        m.reset()?;
        prompt_media(m, before, &[])?;
        let fresh = via_media(m, &c1.ids[after..])?;
        println!(
            "  (D) the cut: after it the sequence stands at {at} with no image {}",
            verdict(cut_state)
        );
        let mut ok = cut_state;
        ok &= equal("(D) the cut table vs the text load's", &cut_table, plain);
        ok &= same_tail(
            "(D) the text after the cut vs the same text from a fresh run",
            &cut,
            &fresh,
        );
        // A cut inside an image is refused before anything moves.
        m.reset()?;
        prompt_media(m, &c2.ids, &c2.image_spans(0))?;
        let (pos, table, held) = (m.pos(), m.rope_rows()?, spans_of(&m.rope_saved()?));
        let inside = c2.spans[1].at + c2.spans[1].len / 2;
        let r = m.rollback(u32::try_from(inside)?);
        let named = refused(
            &format!("(D) a cut to {inside} inside c2's second image"),
            r,
            &["inside the image"],
        );
        let kept = m.pos() == pos
            && first_diff(&m.rope_rows()?, &table).is_none()
            && spans_of(&m.rope_saved()?) == held;
        println!(
            "    the position, table and record as they were {}",
            verdict(kept)
        );
        ok &= named && kept;
        m.body_parts("gate_qwen38_media")?.2.set_checkpoints(false);
        m.reset()?;
        println!("  (D) the cut  {}", verdict(ok));
        Ok(ok)
    }

    /// The load half of the replay clause: a step graph captured after `enable_media` and before any media
    /// call, on a text prompt.
    fn capture_at_load(m: &mut Qwen38Model, c: &Case) -> Result<usize, GateError> {
        m.reset()?;
        let t = prompt_media(m, &c.text_ids()[..c.spans[0].at], &[])?;
        m.step(&[t])?;
        let nodes = m.step_graph_nodes()?.len();
        m.reset()?;
        println!("  a step graph of {nodes} nodes captured at load, before any media call");
        Ok(nodes)
    }

    /// The replay half: after an image call the captured graph's replay equals the eager step.
    fn clause_replay(m: &mut Qwen38Model, c: &Case, nodes: usize) -> Result<bool, GateError> {
        const STEPS: usize = 8;
        let run = |m: &mut Qwen38Model, mode: StepMode| -> Result<Vec<Vec<f32>>, GateError> {
            m.set_mode(mode);
            m.reset()?;
            let mut tok = prompt_media(m, &c.ids, &c.image_spans(0))?;
            let mut logits = Vec::new();
            for _ in 0..STEPS {
                tok = m.step(&[tok])?;
                logits.push(m.logits()?);
            }
            Ok(logits)
        };
        let graph = run(m, StepMode::Graph)?;
        let same_graph = m.step_graph_nodes()?.len() == nodes && nodes > 0;
        let eager = run(m, StepMode::Eager)?;
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let mut ok = same_graph;
        println!(
            "  (D) the replayed graph is the one captured at load ({nodes} nodes) {}",
            verdict(same_graph)
        );
        for (i, (g, e)) in graph.iter().zip(&eager).enumerate() {
            ok &= equal(&format!("(D) step {i}: replay vs eager logits"), g, e);
        }
        println!("  (D) the replay  {}", verdict(ok));
        Ok(ok)
    }

    // ------------------------------------------------------------ (E) the decode after an image

    /// Set F of prompt `k`: mainline's greedy decode steps after the prompt, held on the prefix where its
    /// CPU twin takes the same ids.
    struct Decode {
        /// The token each step took (`ids[0]` is the greedy token after the prompt).
        ids: Vec<u32>,
        /// How many of them the CUDA run and its CPU twin take alike.
        agree: usize,
        /// The position after each step's token, and the position of the first one.
        n_past: Vec<i32>,
        n_past_start: i64,
    }

    impl Decode {
        /// Set F of prompt `k`, held to the same prompt as `c` (its tokens file's digest, its id count and
        /// its spans); the prefix the twin agrees on must hold [`MIN_PREFIX`] ids at least.
        fn load(k: usize, c: &Case) -> Result<Decode, GateError> {
            let open = |set: &str| ClefvisSet::open(&DECODE.path(set), &DECODE);
            let (f, twin) = (open(DECODE.sets[k])?, open(DECODE.sets[k + 2])?);
            if f.man.header.tokens_file_sha256.as_deref() != Some(c.tokens_sha.as_str())
                || f.n_ids()? != c.ids.len()
                || f.spans != c.spans
            {
                return Err(format!("{}: set F is not of the prompt of set C", c.name).into());
            }
            let line = f.decode()?;
            let (steps, n_past_start) = (line.steps, line.n_past_start);
            let ids = f.decode_ids()?;
            let n_past = f.decode_n_past()?;
            if ids.len() != steps || n_past.len() != steps {
                return Err(format!("{}: set F's arrays are not {steps} steps", c.name).into());
            }
            let twin_ids = twin.decode_ids()?;
            let agree = ids
                .iter()
                .zip(&twin_ids)
                .take_while(|(a, b)| a == b)
                .count();
            println!(
                "set F {}: {steps} CUDA ids, the CPU twin agrees on the first {agree}",
                c.name
            );
            if agree < MIN_PREFIX {
                return Err(format!(
                    "{}: set F's CUDA and CPU ids agree on {agree} ids, fewer than {MIN_PREFIX}: no \
                     prefix to hold anything to",
                    c.name
                )
                .into());
            }
            Ok(Decode {
                ids,
                agree,
                n_past,
                n_past_start,
            })
        }
    }

    /// (E): after prompt `c`'s image call, set F's steps through the decode path, each fed set F's own id:
    /// the first token and every next token equal set F's on the prefix the twin agrees on, our (t, h, w) of
    /// each decoded row and the table row the step reads equal the position set F dumped (`n_past`, the
    /// position after the token).
    fn clause_decode(
        m: &mut Qwen38Model,
        c: &Case,
        f: &Decode,
        plain: &[f32],
    ) -> Result<bool, GateError> {
        let (n, steps) = (c.ids.len(), f.ids.len());
        println!(" {} decode ({steps} steps after {n} ids):", c.name);
        m.reset()?;
        let first = prompt_media(m, &c.ids, &c.image_spans(0))?;
        let (mut next, mut gaps) = (Vec::with_capacity(steps), Vec::new());
        for (i, &tok) in f.ids.iter().enumerate() {
            let got = m.step(&[tok])?;
            next.push(got);
            // The logit lead of our token over set F's, on a step where they differ: a flip within
            // `NEAR_TIE` is a tie, as set F's own CPU twin takes other ids than its CUDA run.
            if let Some(&want) = f.ids.get(i + 1).filter(|&&w| i + 1 < f.agree && w != got) {
                let logits = m.logits()?;
                gaps.push((i, f64::from(logits[got as usize] - logits[want as usize])));
            }
        }
        let flips: Vec<usize> = gaps.iter().map(|g| g.0).collect();
        let ok_ids = first == f.ids[0] && gaps.iter().all(|g| g.1 < NEAR_TIE);
        println!(
            "  (E) ids: the token after the prompt {first} (set F {}), the next tokens of steps 0..{} \
             differ from set F's at steps {flips:?} by logit leads {:?} (a near tie is below {NEAR_TIE}) \
             (held on the first {} of {steps})  {}",
            f.ids[0],
            f.agree - 1,
            gaps.iter().map(|g| g.1).collect::<Vec<_>>(),
            f.agree,
            verdict(ok_ids)
        );

        let saved = m.rope_saved()?;
        let table = m.rope_rows()?;
        let at = |i: usize| i64::from(f.n_past[i]) - 1;
        let bad_pos = (0..steps).find(|&i| {
            let (t, h, w) = saved.pos(n + i);
            [t as i64, h as i64, w as i64] != [at(i); 3]
        });
        let bad_row = (0..steps).find(|&i| {
            let p = at(i) as usize;
            first_diff(
                &table[(n + i) * ROT..(n + i + 1) * ROT],
                &oracle_row::<ROT>(plain, (p, p, p)),
            )
            .is_some()
        });
        let ends = at(0) == f.n_past_start && m.pos() as usize == n + steps;
        let ok_pos = bad_pos.is_none() && bad_row.is_none() && ends;
        println!(
            "  (E) positions: the (t, h, w) of rows {n}..{} against set F's n_past, first differing step \
             {bad_pos:?}; the table rows they read, first differing step {bad_row:?}; first position {} \
             (set F {}), the model stands at {}  {}",
            n + steps,
            saved.pos(n).0,
            f.n_past_start,
            m.pos(),
            verdict(ok_pos)
        );
        Ok(ok_ids && ok_pos)
    }

    // ------------------------------------------------------------ (G) one answer through serve

    /// Request c1 of set E (`tools/ref/qvis/requests.jsonl`) with its image as a data URI and the thinking
    /// switch the set's chat line records, and the set's ids with the image run cut to the one placeholder
    /// the template writes.
    struct Chat {
        body: Value,
        /// The ids set E holds for the request, and the same with each span cut to one id.
        ids: Vec<u32>,
        collapsed: Vec<u32>,
    }

    impl Chat {
        /// Request c1 against set E's c1 and set C's c1 `case`: the same prompt, or a named error before
        /// any load.
        fn load(case: &Case) -> Result<Chat, GateError> {
            let set = ClefvisSet::open(&CHATIDS.path(CHATIDS.sets[0]), &CHATIDS)?;
            let ids = set.ids()?;
            if ids != case.ids || set.spans != case.spans {
                return Err(format!(
                    "set E's c1 is not set C's {}: the oracles are of two prompts",
                    case.name
                )
                .into());
            }
            let text =
                std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(REQUESTS))?;
            let line = text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(serde_json::from_str::<Value>)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .find(|v| v["id"] == "c1")
                .ok_or("no request c1 in the requests file")?;
            let mut messages = line["messages"].clone();
            for m in messages.as_array_mut().into_iter().flatten() {
                for part in m["content"].as_array_mut().into_iter().flatten() {
                    if part["type"] != "image_url" {
                        continue;
                    }
                    let name = part["image_url"]["url"]
                        .as_str()
                        .ok_or("an image part with no name")?
                        .to_owned();
                    let png = png_of(&set, &name)?;
                    part["image_url"]["url"] = json!(format!(
                        "data:image/png;base64,{}",
                        serve_client::base64(&png)
                    ));
                }
            }
            let body = json!({
                "messages": messages,
                "chat_template_kwargs": { "enable_thinking": set.chat()?.enable_thinking },
            });
            let mut collapsed = Vec::new();
            let mut at = 0;
            for s in &set.spans {
                let run = ids
                    .get(s.at..s.at + s.len)
                    .filter(|r| !r.is_empty() && r.iter().all(|&i| i == r[0]))
                    .ok_or_else(|| format!("span {} is not one id repeated", s.index))?;
                collapsed.extend_from_slice(&ids[at..s.at]);
                collapsed.push(run[0]);
                at = s.at + s.len;
            }
            collapsed.extend_from_slice(&ids[at..]);
            Ok(Chat {
                body,
                ids,
                collapsed,
            })
        }
    }

    /// A server on the Qwen3.8 seat of `bloomery-serve` with `extra` flags, its stderr and stdout under
    /// `dir`, its draft and residency off; its address once it listens, else the failure.
    fn start(dir: &Path, path: &Path, extra: &[&str]) -> Result<(Served, String), GateError> {
        std::fs::create_dir_all(dir)?;
        let model = path.to_str().ok_or("the model path is not UTF-8")?;
        let mut cmd = serve_cmd()?;
        cmd.env(bloomery_levers::DRAFT, "off")
            .env(bloomery_levers::RESIDENCY, "off")
            .env_remove(bloomery_levers::MTP_WIDTH);
        let mut args = vec![
            "--model",
            "qwen38",
            "--port",
            "0",
            "--place",
            "gate",
            "--ctx-size",
            "4096",
            "--parallel",
            "1",
            "-m",
            model,
        ];
        args.extend_from_slice(extra);
        let mut served = Served::spawn_cmd(cmd, &args, dir)?;
        let addr = served.address(&dir.join("server.err"), POLLS, POLL)?;
        Ok((served, addr))
    }

    /// (G): the server with `--mmproj` against set E's prompt and set F's ids, then the server without it.
    fn clause_serve(dir: &Path, path: &Path, chat: &Chat, f: &Decode) -> Result<bool, GateError> {
        let mut ok = true;
        let with = dir.join("with-mmproj");
        let (mut served, addr) = match start(&with, path, &["--mmproj", MMPROJ_BF16]) {
            Ok(s) => s,
            Err(e) => {
                println!("the image server never listened: {e}");
                check(&mut ok, "mmproj_server_listens", false);
                return Ok(ok);
            }
        };
        let url = |p: &str| format!("http://{addr}{p}");
        let (st, body) = curl(&url("/props"), None, false)?;
        let props = json_of("/props", st, &body)?;
        println!("props modalities {}", props["modalities"]);
        check(
            &mut ok,
            "mmproj_props_vision_true",
            props["modalities"]["vision"] == json!(true),
        );

        // The prompt the request renders to is set E's, its image one placeholder the server expands to
        // the span, and the count it evaluates is the set's.
        let (st, text) = curl(&url("/apply-template"), Some(&chat.body), false)?;
        let prompt = json_of("/apply-template", st, &text)?["prompt"]
            .as_str()
            .ok_or("/apply-template: no prompt")?
            .to_owned();
        let rendered = tokenized(&url, &prompt)?;
        let ask = greedy_chat(&chat.body, f.ids.len());
        let (st, text) = curl(&url("/v1/chat/completions"), Some(&ask), false)?;
        std::fs::write(with.join("chat.json"), &text)?;
        let reply = answered(st, &text)?;
        println!(
            "G: the rendered prompt {} ids (set E {} collapsed); the server evaluated {} (set E {})",
            rendered.len(),
            chat.collapsed.len(),
            reply.prompt,
            chat.ids.len()
        );
        check(
            &mut ok,
            "mmproj_prompt_ids_are_set_e",
            rendered == chat.collapsed && reply.prompt == chat.ids.len() as u64,
        );

        // The answer's ids are set F's over the prefix its CPU twin agrees on, or part at a near tie.
        let (upto, gap, follows) = reply.follows(&f.ids, f.agree, NEAR_TIE);
        println!(
            "G: {} ids answered {:?}; set F's {:?}; agreed up to step {upto} of the {} the twin holds, gap \
             at the first difference {gap:?} (near tie below {NEAR_TIE})",
            reply.ids.len(),
            &reply.ids[..reply.ids.len().min(f.agree)],
            &f.ids[..f.agree],
            f.agree
        );
        check(&mut ok, "mmproj_answer_follows_set_f", follows);
        println!("the image server stopped: {}", served.stop()?);

        // The same request to a server with no --mmproj is the server's own 400.
        let (mut plain, addr) = match start(&dir.join("without-mmproj"), path, &[]) {
            Ok(s) => s,
            Err(e) => {
                println!("the text server never listened: {e}");
                check(&mut ok, "no_mmproj_server_listens", false);
                return Ok(ok);
            }
        };
        let (st, body) = curl(
            &format!("http://{addr}/v1/chat/completions"),
            Some(&chat.body),
            false,
        )?;
        println!("an image on the seat with no --mmproj: {st} {body}");
        check(
            &mut ok,
            "image_without_mmproj_is_refused",
            st == 400 && body.contains(NO_IMAGE),
        );
        println!("the text server stopped: {}", plain.stop()?);
        println!("  (G) serve  {}", verdict(ok));
        Ok(ok)
    }

    // ------------------------------------------------------------ the run

    /// `--dir <dir>`: where the servers' stderr and the chat answer go.
    fn dir_arg() -> Result<PathBuf, GateError> {
        let mut it = std::env::args().skip(1);
        match (it.next().as_deref(), it.next(), it.next()) {
            (Some("--dir"), Some(dir), None) => Ok(PathBuf::from(dir)),
            _ => Err("usage: gate_qwen38_media --dir <dir>".into()),
        }
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[bloomery_gpu::arch::qwen3moe::ubatch::UBATCH_ENV])?;
        let dir = dir_arg()?;
        let path = ref_model_path()?;
        if path != Path::new(MODEL) {
            return Err(format!(
                "{} is not {MODEL}, the file the Qwen3.8 image sets were dumped from",
                path.display()
            )
            .into());
        }
        println!(
            "gate_qwen38_media: Qwen3.8-Flash-Next at {}",
            path.display()
        );
        let split = qwen38_open::open_split(&path)?;
        check_shape(&split)?;
        drop(split);
        let tower = ClefvisSet::open(&TAPS.path(TAPS.sets[0]), &TAPS)?;
        let twin = ClefvisSet::open(&TAPS.path(TAPS.sets[1]), &TAPS)?;
        let cases = [
            Case::load(0, "c1", &tower)?,
            Case::load(1, "c2", &tower)?,
            Case::load(2, "c3", &tower)?,
        ];
        let c1 = &cases[0];
        let decodes = [Decode::load(0, &cases[0])?, Decode::load(1, &cases[1])?];
        let chat = Chat::load(c1)?;
        let c3 = &cases[2];
        let tail = &c3.ids[c3.spans.last().map_or(0, |s| s.at + s.len)..];
        if tail.len() < SHORT_TEXT {
            return Err(format!(
                "c3 holds {} ids after its image, fewer than the {SHORT_TEXT} the short request takes",
                tail.len()
            )
            .into());
        }
        let reqs = [
            Req {
                name: format!("{} ids", tail.len()),
                ids: tail.to_vec(),
            },
            Req {
                name: format!("{SHORT_TEXT} ids (a short tail past one unit)"),
                ids: tail[..SHORT_TEXT].to_vec(),
            },
        ];
        let mut images: Vec<String> = cases
            .iter()
            .flat_map(|c| c.spans.iter().map(|s| s.image.clone()))
            .collect();
        images.sort();
        images.dedup();
        let mut ok = true;

        println!("(C) identity: the encoder file against the text file's facts");
        let (ok_c, opened) = clause_identity(&path)?;
        ok &= ok_c;

        // The tower's bytes and the rotation tables' are the card's beside the plan: the plan is made on
        // the card less them.
        let card = gate_card::init()?;
        let tables = (CTX * ROT * size_of::<f32>()) as u64;
        let budget = card
            .usable_bytes()
            .checked_sub(opened.card.total() + tables)
            .ok_or("the tower and the tables are more than the card")?;
        let load = Load {
            card,
            host: levers.host(),
            plan_levers: PlanLevers {
                card_budget_bytes: Some(budget).into(),
            },
        };

        println!("the text load (no media): references and refusals");
        let mut text = load.open(false)?;
        let plain = text.rope_rows()?;
        let mut refs = Vec::new();
        for r in &reqs {
            let (call, streams) = via_plain(&mut text, &r.ids)?;
            refs.push(Reference {
                tail: call,
                streams,
            });
        }
        let none = !text.body("gate_qwen38_media")?.takes_media();
        println!("    the text load takes no media {}", verdict(none));
        ok &= none;
        ok &= refused(
            "an image span on a text load",
            prompt_media(&mut text, &c1.ids, &c1.image_spans(0)),
            TEXT_LOAD,
        );
        ok &= refused(
            "a record of images on a text load",
            text.rope_restore(c1.record()?),
            TEXT_LOAD,
        );
        ok &= refused(
            "the media tap on a text load",
            text.set_media_tap(true),
            TEXT_LOAD,
        );
        ok &= refused("enable_media after a call", text.enable_media(), TEXT_LOAD);
        drop(text);

        println!("the media load (enable_media)");
        let mut m = load.open(true)?;
        let takes = m.body("gate_qwen38_media")?.takes_media();
        println!("    the media load takes media {}", verdict(takes));
        ok &= takes;
        ok &= refused("enable_media twice", m.enable_media(), TEXT_LOAD);
        let nodes = capture_at_load(&mut m, c1)?;
        ok &= clause_tower(&m, &opened, (&tower, &twin), &images)?;
        ok &= clause_text(&mut m, &plain, &reqs, &refs, "fresh")?;
        println!("(A) and (B) per set C prompt");
        for c in &cases {
            ok &= run_case(&mut m, c, &plain)?;
        }
        ok &= clause_text(
            &mut m,
            &plain,
            &reqs,
            &refs,
            "after image calls and a reset",
        )?;
        ok &= clause_capture(&mut m, c1, &plain)?;
        ok &= clause_inside(&mut m, c1)?;
        ok &= clause_cut(&mut m, c1, &cases[1], &plain)?;
        ok &= clause_replay(&mut m, c1, nodes)?;
        println!("(E) the decode on set F");
        for (k, f) in decodes.iter().enumerate() {
            ok &= clause_decode(&mut m, &cases[k], f, &plain)?;
        }
        drop(m);

        println!("(G) one answer through serve");
        ok &= clause_serve(&dir, &path, &chat, &decodes[0])?;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
