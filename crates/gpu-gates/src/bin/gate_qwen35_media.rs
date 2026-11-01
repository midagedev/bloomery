//! The image-row gate of Body35, the body Clef-Flash and Qwen3.6 run: a media prompt call
//! (`MediaBody::prefill_hidden_media` and `prefill_media`) against the oracle sets of mainline's mtmd on the same
//! file: set B's tower rows (rounded to bf16, the rows mainline's C″ run was fed) as the spans' rows, and the three
//! prompts of set C (the CUDA runs, their CPU twins and the C″ runs) as the oracle of the positions and the hidden
//! states. Two seats, one harness, the recipe's `BLOOMERY_MODEL` picking the seat ([`gate::seat_of`]): the
//! `qwen35` profile's Clef-Flash at bartowski's Q8_0 (`refset::arch::qwen35::clefvis`; `gate-gpu-qwen35-media`) and
//! the `qwen35moe` profile's Qwen3.6-35B-A3B at lmstudio's Q4_K_M (`refset::arch::qwen35moe::vis`;
//! `gate-gpu-qwen35moe-media`). Every clause is FAIL-first by a mutant of its own.
//!
//! - (i) injection: each unit's embedding rows as layer 0 reads them (`set_media_tap`): the rows at image
//!   positions are the widened bf16 rows bit for bit and the text rows equal the same ids' rows of a call with no
//!   span; the units are the Wide plan's, so a span a unit boundary cuts is read in each of its units (c2's second
//!   image crosses two), and the call at one unit is bit for bit the cut call's hidden states.
//! - (ii) positions: the sequence's (t, h, w) of every row equal set C's dumped integer arrays.
//! - (iii) the table: the card's rotation table, `ctx` rows, equals the plain table (a text load's) gathered by
//!   (ii)'s positions through the interleaved sector rule written out here, bit for bit.
//! - S1: the same table equals the host's full rebuild (`MropeSeq::rows_into`) from the sequence's record.
//! - (iv) and S3: a text request on the media load, through the media call with no span and through the plain
//!   Wide call, equals the text load's, bit for bit; fresh, and after an image call and a `reset`.
//! - (v) a media call on a capturing stream is refused by name before anything moves.
//! - the call-start clause: a call whose first position is strictly inside a held image (a prompt, a media call
//!   with its span after the image's end, a plain prompt, a decode step, a pass of two rows) is refused by name
//!   with the position, the table and the record as they were; so is a pass of slots in which one stands there,
//!   the live slot or a parked one. The state is built by `rope_restore` ahead of the position.
//! - S5: a cut back to an image's start, then text, equals the same text from a fresh run split the same way; a
//!   cut inside an image (a checkpoint exists there) is refused before the stores move.
//! - S6: one pass of an image slot and a text slot gives each row the logits of its slot stepped alone.
//! - S7: the sequence's record saved, the model reset and the record restored leaves the table as it was; a
//!   sequence parked in a slot and resumed continues as an uninterrupted run, and the slot that ran meanwhile
//!   reads its own table.
//! - S8: a step graph captured at load, before any media call, replayed after an image call equals the eager
//!   step: the table is allocated at open.
//! - the hidden band of set C: per prompt, the image positions and the text positions after the image apart, the
//!   median, the 90th percentile and each quarter's median against `max(TWIN × mainline's CPU twin, K × C″)`.
//!   The prefix before the first image and ours against C″ are printed.
//!
//! Qwen3.6's seat adds set F, where the seat's family table has it (Clef-Flash has none):
//!
//! - S2 on set F: after prompt c1's and c2's image calls, our (t, h, w) of each of the 32 decoded rows and the
//!   rotation-table row the step reads equal the position set F dumped (`n_past`, the position after the token),
//!   and the first position is its `n_past_start`.
//! - S4: the first token after the prompt and every step's next token equal set F's ids (mainline's CPU twin
//!   takes all 32 on both prompts), and each step's `result_norm` is inside `DECODE_K` times the twin's distance
//!   from mainline's CUDA run per statistic (the median, the 90th percentile and each quarter's median over the
//!   32 steps), and the steps' logits equal the same ids' through passes of two rows bit for bit (the band is
//!   coarse: a decode that reads the shared table instead of the sequence's moves a row by a few hundredths, under
//!   the twin's own distance from mainline). A decode step's hidden is its last layer's output, tapped on the eager
//!   step and normed on the host.
//! - S8 (the replayed graph equals the eager step) runs on both seats; Qwen3.6's seat leaves Body35's slot and
//!   cache clauses to the dense seat (the code is the backbone's own, which they do not touch).
//!
//! The serve prompt cache and slot file are R4a's.

#[cfg(feature = "vision")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen35_media", gate::run())
}

#[cfg(feature = "vision")]
#[path = "shared/qwen35_open.rs"]
#[allow(
    dead_code,
    reason = "the gate opens the dense seat through open_with (the media load) and the text load through it too; `open` serves the other bins"
)]
mod qwen35_open;

#[cfg(feature = "vision")]
#[path = "shared/qwen3moe_place.rs"]
#[allow(
    dead_code,
    reason = "the gate opens a qwen35moe file whole through the planner's verdict; the rest of the planner serves the CLI and the seats"
)]
mod q3place;

#[cfg(feature = "vision")]
mod gate {
    use super::q3place::{self, Unplaced};
    use super::qwen35_open;
    use bloomery_gpu::arch::qwen3moe::{Body35, Open35, PrefillPath, Qwen35moeModel};
    use bloomery_gpu::host::PassKind;
    use bloomery_gpu::kv_format::KvQ8;
    use bloomery_gpu::media::{ImageSpan, MediaBody, MediaSpan};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::weights::DevWeight;
    use bloomery_gpu::{Gpu, GpuError, capture_template};
    use bloomery_gpu_gates::{
        GateError, checks_failed, equal, first_diff, oracle_row, quantile, ref_model_path, verdict,
    };
    use bloomery_gpu_vision::f32_bf16;
    use gguf::Split;
    use refset::arch::qwen35::MODEL_FLASH_Q8;
    use refset::arch::qwen35::clefvis as clef;
    use refset::arch::qwen35moe::{MODEL as MODEL_Q36, vis as q36};
    use refset::clefvis::{ClefvisSet, EMBD, IMAGE_PAD_ID, Span};
    use refset::family::Family;
    use runtime::swaprule::KeptRows;
    use std::path::Path;
    use vision::mrope::{MropeSeq, SectionOrder};

    /// The rotation table's values a row and the sets' cache rows; the text width is the seat's.
    const ROT: usize = 64;
    const CTX: usize = 4096;
    /// The dumps' ubatch: c1 is two units, c2 three (its second image, rows 339..1107, crosses two
    /// boundaries), c3 five.
    const UBATCH: usize = 512;
    /// The file's `rope.dimension_sections`: the sectors the written-out rule reads.
    const SECTIONS: [u32; 4] = [11, 11, 10, 0];

    /// PIN(2026-10-11): fixed before the engine met the sets. A band is `max(TWIN × the CPU twin's distance from
    /// mainline's CUDA rows, K × the C″ run's distance from them)`, statistic by statistic (median, 90th
    /// percentile, each quarter's median), per prompt and region. Our rows are mainline's own tower rows rounded
    /// to bf16, so the engine sits from mainline's CUDA run about √(twin² + C″²): the twin is the floor any two
    /// realizations of the rule lie apart (a Q8_0 backbone's 8-bit activations, f32 order), C″ is what the bf16
    /// rounding of the rows does to mainline itself. Read from the sets before any run: the post-image quarters
    /// of c1, c2 and c3 read twin 0.017-0.031 and C″ 0.010-0.017, the image positions twin 0.065-0.21 (median)
    /// and C″ 0.032-0.041. At K = 2 the quadrature estimate sits at 0.73-0.78 of the band in every statistic
    /// read (c3 post-image median 0.016 against 0.022, c1 image median 0.072 against 0.084); K = 1.5 would put it
    /// at the band's edge. A position fault (M-RoPE off) moves the post-image rows to order one, 10 times the
    /// band.
    const K: f64 = 2.0;
    const TWIN: f64 = 1.3;

    /// PIN(2026-10-11): the decode region's band, fixed before the engine met set F. Set F has no C″ run (the
    /// bf16 rows' effect on a decode step), so §3.8's `max(twin, C″)` is the twin alone, scaled by K: the engine
    /// sits from mainline's CUDA decode about √(twin² + c″²), and on set C's post-image rows c″ is 0.8-1.34 of the
    /// twin (Qwen3.6, c1-c3), so the estimate is 1.28-1.67 twin against a band of 2 twin (at most 0.84 of it); the
    /// twin alone at TWIN = 1.3 would put it at about 1.29 of the band. The twin's decode rows read 0.04-0.20 a
    /// step, quarter medians 0.058-0.092 (c1, c2). The ids: mainline's CPU twin takes the same 32 ids as its CUDA
    /// run on both prompts, so ours must take all 32.
    const DECODE_K: f64 = K;

    /// What a seat names: the model file Body35 runs, the oracle families its sets are dumped into, and what its
    /// sets can close. The recipe's `BLOOMERY_MODEL` picks it: the architecture of the profile's reference file
    /// ([`seat_of`]).
    struct Seat {
        name: &'static str,
        /// The model file the sets were dumped from; the one the gate loads.
        path: &'static str,
        /// The text width: a row of a span, of a set's hidden rows and of the cache.
        width: usize,
        /// The decode flash pass: the tensor-core one when `true` (the pass a qwen35moe load runs; the dense
        /// seat holds the scalar one, whose gate it is).
        mma: bool,
        /// Set B's family (the tower's rows are `sets[0]`), set C's (the CUDA runs `sets[0..3]`, their CPU twins
        /// `sets[3..6]`), set C″'s, and set F's when the seat has them (the CUDA runs `sets[0..2]`, the twins
        /// `sets[2..4]`).
        taps: &'static Family,
        hidden: &'static Family,
        bf16rows: &'static Family,
        decode: Option<&'static Family>,
        /// The prompt of set C that holds the 517-id text request (a tail under the Wide floor).
        long: usize,
        /// Whether the seat also runs Body35's slot and cache clauses (S3's after-image arm, (v), the call-start
        /// clause, S5-S7 and S6): code the backbone does not touch, closed on the dense seat.
        body35_tail: bool,
        open: fn(&Seat, bool) -> Result<Qwen35moeModel, GateError>,
    }

    static CLEF: Seat = Seat {
        name: "Clef-Flash (Q8_0)",
        path: MODEL_FLASH_Q8,
        width: 4096,
        mma: false,
        taps: &clef::TAPS,
        hidden: &clef::HIDDEN,
        bf16rows: &clef::BF16ROWS,
        decode: None,
        long: 0,
        body35_tail: true,
        open: open_dense,
    };

    static QWEN36: Seat = Seat {
        name: "Qwen3.6-35B-A3B (Q4_K_M)",
        path: MODEL_Q36,
        width: 2048,
        mma: true,
        taps: &q36::TAPS,
        hidden: &q36::HIDDEN,
        bf16rows: &q36::BF16ROWS,
        decode: Some(&q36::DECODE),
        long: 2,
        body35_tail: false,
        open: open_moe,
    };

    /// The seat of the profile the recipe picked: `$BLOOMERY_REF_MODEL` (the profile's file, `tools/box.sh`
    /// exports it) is a Qwen3.5 dense body for the `qwen35` profile and a `qwen35moe` file for the `qwen35moe`
    /// one. Any other architecture is refused by name; so is a `qwen35moe` profile file that is not the file
    /// the seat's sets were dumped from.
    fn seat_of(profile_file: &Path) -> Result<&'static Seat, GateError> {
        let split = Split::open(profile_file)
            .map_err(|e| format!("open {}: {e}", profile_file.display()))?;
        match split.architecture() {
            Some(a) if model::arch::is_qwen35_body(a) => Ok(&CLEF),
            Some("qwen35moe") if profile_file == Path::new(MODEL_Q36) => Ok(&QWEN36),
            Some("qwen35moe") => Err(format!(
                "{} is a qwen35moe file, and the seat's sets were dumped from {MODEL_Q36}",
                profile_file.display()
            )
            .into()),
            other => Err(format!(
                "{} is {other:?}: this gate runs the qwen35 profile's Clef-Flash seat and the qwen35moe \
                 profile's Qwen3.6 seat",
                profile_file.display()
            )
            .into()),
        }
    }

    /// The file's width, rotary dimension count and sections against the constants the clauses are written
    /// with: a file of another shape fails here by name. Returns the file's RMS epsilon (a decode step's
    /// hidden row is normed on the host with it).
    fn check_shape(seat: &Seat, split: &Split) -> Result<f32, GateError> {
        let get = |k: &str| {
            split
                .value(&split.arch_key(k))
                .and_then(gguf::Value::as_unsigned)
        };
        let (width, rot) = (get("embedding_length"), get("rope.dimension_count"));
        let sections: Option<Vec<u64>> = split
            .arch_get_arr("rope.dimension_sections")
            .map(|a| a.iter().filter_map(gguf::Value::as_unsigned).collect());
        let want_sections: Vec<u64> = SECTIONS.iter().map(|&s| u64::from(s)).collect();
        if width != u64::try_from(seat.width).ok()
            || rot != u64::try_from(ROT).ok()
            || sections.as_deref() != Some(want_sections.as_slice())
        {
            return Err(format!(
                "{}: width {width:?}, rotary dims {rot:?}, sections {sections:?}; the clauses are written for \
                 {}, {ROT} and {SECTIONS:?}",
                seat.path, seat.width
            )
            .into());
        }
        split
            .arch_get_f32("attention.layer_norm_rms_epsilon")
            .ok_or_else(|| format!("{}: no attention.layer_norm_rms_epsilon", seat.path).into())
    }

    /// The dense seat's load: [`qwen35_open::open_with`], whole on `a`'s card.
    fn open_dense(seat: &Seat, media: bool) -> Result<Qwen35moeModel, GateError> {
        Ok(qwen35_open::open_with(Path::new(seat.path), CTX, UBATCH, media)?.0)
    }

    /// The `qwen35moe` seat's load: whole on `a`'s card when the whole-fit verdict takes it (its line is on
    /// stderr), else refused by name with the verdict: a placed load runs its experts on the host, which is
    /// another set of bits than the card's, and the media call on one is the serve seat's.
    fn open_moe(seat: &Seat, media: bool) -> Result<Qwen35moeModel, GateError> {
        let o = Open35 {
            ctx: CTX,
            mma: seat.mma,
            ubatch: UBATCH,
            kv: KvQ8::F16,
            media,
        };
        let file = Split::open(seat.path).map_err(|e| format!("open {}: {e}", seat.path))?;
        match q3place::unplaced_qwen35(&file, &o)? {
            Unplaced::Whole(card) => {
                Ok(Qwen35moeModel::open(Gpu::open_card(card.name, card.device)?, file, o)?)
            }
            Unplaced::Placed(_) => Err(format!(
                "{}: the whole load does not fit the gate's card (the verdict line above); this gate holds the \
                 whole load only",
                seat.path
            )
            .into()),
        }
    }

    /// A set C prompt, its three oracles and the span rows.
    struct Case {
        name: &'static str,
        /// The text width of the seat.
        width: usize,
        /// The sha256 of the tokens file the set was dumped from.
        tokens_sha: String,
        ids: Vec<u32>,
        spans: Vec<Span>,
        /// Each span's bf16 rows: set B's tower rows of the span's image, rounded to nearest even.
        rows: Vec<Vec<u16>>,
        /// Set C's (t, y, x) per sequence index.
        pos: Vec<[i32; 3]>,
        cuda: Vec<f32>,
        twin: Vec<f32>,
        c2p: Vec<f32>,
    }

    impl Case {
        /// Prompt `n` (0-based) of the seat's set C: its ids read from the tokens file the set names, held to the
        /// header's digest and count, the image-pad run of each span checked, and the rows of each span's image
        /// from the tower set `tower`.
        fn load(
            seat: &Seat,
            n: usize,
            name: &'static str,
            tower: &ClefvisSet,
        ) -> Result<Case, GateError> {
            let width = seat.width;
            let open = |fam: &'static Family, set: &str| ClefvisSet::open(&fam.path(set), fam);
            let hidden = open(seat.hidden, seat.hidden.sets[n])?;
            let twin = open(seat.hidden, seat.hidden.sets[n + 3])?;
            let c2p = open(seat.bf16rows, seat.bf16rows.sets[n])?;
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
            let ids = qwen35_open::read_ids(Path::new(file), Some(count))?;
            let mut rows = Vec::new();
            for s in &hidden.spans {
                if s.nx * s.ny != s.len
                    || ids[s.at..s.at + s.len].iter().any(|&i| i != IMAGE_PAD_ID)
                {
                    return Err(format!("{name}: span {} is not an image-pad run", s.index).into());
                }
                let (_, tower_rows) = tower.tap(&s.image, EMBD)?;
                if tower_rows.len() != s.len * width {
                    return Err(
                        format!("{name}: {} tower rows for {}", tower_rows.len(), s.image).into(),
                    );
                }
                rows.push(tower_rows.iter().map(|&x| f32_bf16(x)).collect());
            }
            let pos = hidden.positions()?;
            let (cuda, twin, c2p) = (
                hidden.result_norm()?,
                twin.result_norm()?,
                c2p.result_norm()?,
            );
            if pos.len() != ids.len() || cuda.len() != ids.len() * width {
                return Err(format!("{name}: the set's rows are not the ids' count").into());
            }
            Ok(Case {
                name,
                width,
                tokens_sha: got,
                ids,
                spans: hidden.spans.clone(),
                rows,
                pos,
                cuda,
                twin,
                c2p,
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
                    shape: bloomery_gpu::media::ImageShape::Grid(s.nx, s.ny),
                })
                .collect()
        }

        fn is_image(&self, p: usize) -> bool {
            self.spans.iter().any(|s| (s.at..s.at + s.len).contains(&p))
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

    fn spans_of(m: &MropeSeq) -> Vec<(usize, usize, usize)> {
        m.spans().collect()
    }

    /// Whether `r` is an error whose text holds `want`, with one line.
    fn refused<T>(what: &str, r: Result<T, GpuError>, want: &str) -> bool {
        match r {
            Err(e) if e.to_string().contains(want) => {
                println!("    {what}: refused ({want}) ok");
                true
            }
            Err(e) => {
                println!("    {what}: refused, but not by \"{want}\": {e}  FAIL");
                false
            }
            Ok(_) => {
                println!("    {what}: accepted  FAIL");
                false
            }
        }
    }

    /// `‖a − b‖ / ‖b‖` in f64, a NaN reading infinite.
    fn distance(a: &[f32], b: &[f32]) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            let e = f64::from(x) - f64::from(y);
            num += e * e;
            den += f64::from(y).powi(2);
        }
        let rel = (num / den.max(f64::MIN_POSITIVE)).sqrt();
        if rel.is_nan() { f64::INFINITY } else { rel }
    }

    /// A region's statistics: the median, the 90th percentile and each quarter's median of the positions'
    /// distances, in position order.
    #[derive(Clone, Copy)]
    struct Stats {
        med: f64,
        p90: f64,
        q: [f64; 4],
    }

    impl Stats {
        fn of(width: usize, a: &[f32], b: &[f32], positions: &[usize]) -> Stats {
            let rel: Vec<f64> = positions
                .iter()
                .map(|&p| {
                    distance(
                        &a[p * width..(p + 1) * width],
                        &b[p * width..(p + 1) * width],
                    )
                })
                .collect();
            let n = rel.len();
            let quarter = |i: usize| quantile(&rel[i * n / 4..(i + 1) * n / 4], 0.5);
            Stats {
                med: quantile(&rel, 0.5),
                p90: quantile(&rel, 0.9),
                q: [quarter(0), quarter(1), quarter(2), quarter(3)],
            }
        }

        fn all(&self) -> [f64; 6] {
            [
                self.med, self.p90, self.q[0], self.q[1], self.q[2], self.q[3],
            ]
        }

        fn show(&self) -> String {
            format!(
                "med {:.4} p90 {:.4} q [{:.4} {:.4} {:.4} {:.4}]",
                self.med, self.p90, self.q[0], self.q[1], self.q[2], self.q[3]
            )
        }
    }

    // ------------------------------------------------------------ the rotation table

    /// The (t, h, w) of sequence row `r` of prompt `c` as set C feeds it: the dumped triple inside the prompt,
    /// the text position after it (the last row's position, advanced) beyond.
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

    fn hidden_media(
        m: &mut Qwen35moeModel,
        ids: &[u32],
        spans: &[ImageSpan<'_>],
    ) -> Result<Vec<f32>, GpuError> {
        <Body35 as MediaBody>::prefill_hidden_media(m, ids, spans)
    }

    fn prompt_media(
        m: &mut Qwen35moeModel,
        ids: &[u32],
        spans: &[ImageSpan<'_>],
    ) -> Result<u32, GpuError> {
        <Body35 as MediaBody>::prefill_media(m, ids, spans)
    }

    /// A text request: the first `n` ids of Clef's c1 prompt, as text.
    struct Req {
        name: String,
        ids: Vec<u32>,
    }

    // ------------------------------------------------------------ the clauses

    /// (i): the unit rows of an image call against the widened rows and a no-span call's rows.
    fn clause_i(
        c: &Case,
        tap: &[(usize, Vec<f32>)],
        text: &[(usize, Vec<f32>)],
        ubatch: usize,
    ) -> bool {
        let n = c.ids.len();
        let want: Vec<usize> = (0..n).step_by(ubatch).collect();
        let firsts: Vec<usize> = tap.iter().map(|u| u.0).collect();
        let shaped = firsts == want
            && text.iter().map(|u| u.0).eq(firsts.iter().copied())
            && tap
                .iter()
                .all(|(f, r)| r.len() == (n.min(f + ubatch) - f) * c.width);
        let (mut img, mut img_bad, mut txt, mut txt_bad, mut first_bad) = (0, 0, 0, 0, None);
        for (u, (first, rows)) in tap.iter().enumerate() {
            for (i, row) in rows.chunks_exact(c.width).enumerate() {
                let p = first + i;
                let ok = if let Some(j) = c
                    .spans
                    .iter()
                    .position(|s| (s.at..s.at + s.len).contains(&p))
                {
                    img += 1;
                    let from = (p - c.spans[j].at) * c.width;
                    let widened: Vec<f32> = c.rows[j][from..from + c.width]
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
                        .and_then(|t| t.1.get(i * c.width..(i + 1) * c.width))
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
            "  (i) {}: {} units at {firsts:?} (want {want:?}); {img} image rows, {img_bad} off the widened \
             bf16 row; {txt} text rows, {txt_bad} off the no-span call's; first bad {first_bad:?}  {}",
            c.name,
            tap.len(),
            verdict(ok)
        );
        ok
    }

    /// (ii), (iii) and S1 of the sequence the call left on `m`.
    fn clause_table(m: &Qwen35moeModel, c: &Case, plain: &[f32]) -> Result<bool, GateError> {
        let n = c.ids.len();
        let saved = m.rope_saved()?;
        let wrong = (0..n).find(|&r| {
            let (t, h, w) = saved.pos(r);
            [t as i32, h as i32, w as i32] != c.pos[r]
        });
        let want_spans: Vec<_> = c.spans.iter().map(|s| (s.at, s.nx, s.ny)).collect();
        let ok_ii = wrong.is_none() && spans_of(&saved) == want_spans;
        println!(
            "  (ii) {}: the (t, h, w) of {n} rows against set C's, first differing row {wrong:?}, record {:?}  {}",
            c.name,
            spans_of(&saved),
            verdict(ok_ii)
        );
        let table = m.rope_rows()?;
        let bad = (0..CTX).find(|&r| {
            first_diff(
                &table[r * ROT..(r + 1) * ROT],
                &oracle_row::<ROT>(plain, feed_position(c, r)),
            )
            .is_some()
        });
        let ok_iii = table.len() == CTX * ROT && bad.is_none();
        println!(
            "  (iii) {}: {CTX} table rows against the plain table gathered by set C's positions, first \
             differing row {bad:?}  {}",
            c.name,
            verdict(ok_iii)
        );
        let mut full = vec![0.0f32; CTX * ROT];
        saved
            .rows_into(plain, ROT, SECTIONS, SectionOrder::Tyxz, 0..CTX, &mut full)
            .map_err(|e| e.to_string())?;
        let ok_s1 = equal(
            &format!("S1 {}: the card table vs the host rebuild", c.name),
            &table,
            &full,
        );
        Ok(ok_ii && ok_iii && ok_s1)
    }

    /// The hidden band of one prompt.
    fn clause_band(c: &Case, ours: &[f32]) -> bool {
        let n = c.ids.len();
        let first = c.spans[0].at;
        let last = c.spans.last().map_or(0, |s| s.at + s.len);
        let pre: Vec<usize> = (0..first).collect();
        let image: Vec<usize> = (0..n).filter(|&p| c.is_image(p)).collect();
        let post: Vec<usize> = (last..n).collect();
        let mut ok = true;
        let s_pre = Stats::of(c.width, ours, &c.cuda, &pre);
        println!(
            "  band {}: before the first image ({} rows, not banded): {}",
            c.name,
            pre.len(),
            s_pre.show()
        );
        for (region, positions) in [("image", &image), ("post-image", &post)] {
            let o = Stats::of(c.width, ours, &c.cuda, positions);
            let t = Stats::of(c.width, &c.twin, &c.cuda, positions);
            let e = Stats::of(c.width, &c.c2p, &c.cuda, positions);
            let vs_c2p = Stats::of(c.width, ours, &c.c2p, positions);
            let band = Stats {
                med: (TWIN * t.med).max(K * e.med),
                p90: (TWIN * t.p90).max(K * e.p90),
                q: std::array::from_fn(|i| (TWIN * t.q[i]).max(K * e.q[i])),
            };
            let within = o.all().iter().zip(band.all()).all(|(a, b)| *a <= b);
            println!(
                "  band {} {region} ({} rows): ours {}",
                c.name,
                positions.len(),
                o.show()
            );
            println!(
                "    twin {}  C″ {}  ours vs C″ {}",
                t.show(),
                e.show(),
                vs_c2p.show()
            );
            println!("    band {}  {}", band.show(), verdict(within));
            ok &= within;
        }
        ok
    }

    /// One prompt through the media call: (i), (ii), (iii), S1 and the band. `uncut` also runs it at one unit
    /// and holds it to the cut call's bits.
    fn run_case(
        m: &mut Qwen35moeModel,
        c: &Case,
        plain: &[f32],
        uncut: bool,
    ) -> Result<bool, GateError> {
        println!(
            " {} ({} ids, {} images):",
            c.name,
            c.ids.len(),
            c.spans.len()
        );
        m.reset()?;
        let mut ok = equal(
            "S3 the table after a reset vs the text load's",
            &m.rope_rows()?,
            plain,
        );
        m.set_media_tap(true)?;
        hidden_media(m, &c.ids, &[])?;
        let text = m.take_media_tap()?;
        m.reset()?;
        let ours = hidden_media(m, &c.ids, &c.image_spans(0))?;
        let tap = m.take_media_tap()?;
        m.set_media_tap(false)?;
        ok &= clause_i(c, &tap, &text, UBATCH);
        ok &= clause_table(m, c, plain)?;
        ok &= clause_band(c, &ours);
        if uncut {
            m.set_ubatch(CTX)?;
            m.reset()?;
            m.set_media_tap(true)?;
            let one = hidden_media(m, &c.ids, &c.image_spans(0))?;
            let tap_one = m.take_media_tap()?;
            m.set_media_tap(false)?;
            m.set_ubatch(UBATCH)?;
            ok &= equal(
                &format!("{} at one unit vs {UBATCH}-row units", c.name),
                &one,
                &ours,
            );
            let one_unit = tap_one.len() == 1 && tap_one[0].0 == 0;
            println!(
                "    the one-unit call tapped {} unit(s) {}",
                tap_one.len(),
                verdict(one_unit)
            );
            ok &= one_unit;
        }
        Ok(ok)
    }

    /// (iv) and S3: the requests on the media load, through the media call with no span and the plain Wide
    /// call, against the text load's rows.
    fn clause_text(
        m: &mut Qwen35moeModel,
        plain: &[f32],
        reqs: &[Req],
        refs: &[Vec<f32>],
        when: &str,
    ) -> Result<bool, GateError> {
        let mut ok = true;
        m.reset()?;
        let table = m.rope_rows()?;
        ok &= equal(
            &format!("S3 {when}: the table vs the text load's"),
            &table,
            plain,
        );
        for (r, want) in reqs.iter().zip(refs) {
            m.reset()?;
            let via_media = hidden_media(m, &r.ids, &[])?;
            m.reset()?;
            let direct = m.prefill_hidden(&r.ids, PrefillPath::Wide)?;
            ok &= equal(
                &format!(
                    "(iv) {when} {}: no-span media call vs the text load",
                    r.name
                ),
                &via_media,
                want,
            );
            ok &= equal(
                &format!("(iv) {when} {}: plain Wide call vs the text load", r.name),
                &direct,
                want,
            );
        }
        println!("  (iv) and S3 {when}  {}", verdict(ok));
        Ok(ok)
    }

    /// (v): a media call on a capturing stream.
    fn clause_capture(m: &mut Qwen35moeModel, c: &Case, plain: &[f32]) -> Result<bool, GateError> {
        m.reset()?;
        let stream = m.gpu().stream_handle();
        let ids = &c.ids[..c.spans[0].at + c.spans[0].len + 16];
        let spans = c.image_spans(0);
        let r = capture_template(&stream, |_| prompt_media(m, ids, &spans).map(|_| ()));
        let mut ok = refused("(v) a media call while capturing", r, "an eager call");
        let still = m.pos() == 0 && m.rope_saved()?.is_empty();
        let table = m.rope_rows()?;
        ok &= still && first_diff(&table, plain).is_none();
        println!(
            "  (v) refused before anything moved: position {}, record empty {}  {}",
            m.pos(),
            still,
            verdict(ok)
        );
        Ok(ok)
    }

    /// A model that stands at position 100 with c1's image record restored ahead of it: every call that starts
    /// there is refused, and the position, the table and the record stay.
    fn clause_inside(m: &mut Qwen35moeModel, c: &Case) -> Result<bool, GateError> {
        const AT: usize = 100;
        m.reset()?;
        hidden_media(m, &c.ids[..AT], &[])?;
        m.rope_restore(c.record()?)?;
        let (table, held) = (m.rope_rows()?, spans_of(&m.rope_saved()?));
        let image = &c.spans[0];
        println!(
            "  call start: position {AT} inside the image at {}..{} (the record restored ahead of the state)",
            image.at,
            image.at + image.len
        );
        let after = (image.at + image.len).saturating_sub(AT) + 4;
        let late = ImageSpan {
            span: MediaSpan {
                at: after..after + 4,
                rows: &c.rows[0][..4 * c.width],
            },
            shape: bloomery_gpu::media::ImageShape::Grid(2, 2),
        };
        let want = "inside the image";
        let mut ok = true;
        let mut check = |m: &mut Qwen35moeModel,
                         what: &str,
                         r: Result<(), GpuError>|
         -> Result<(), GateError> {
            let kept = m.pos() as usize == AT
                && first_diff(&m.rope_rows()?, &table).is_none()
                && spans_of(&m.rope_saved()?) == held;
            let refused = refused(what, r, want);
            println!(
                "    {what}: position, table and record as they were {}",
                verdict(kept)
            );
            ok &= refused && kept;
            Ok(())
        };
        let r = hidden_media(m, &c.ids[AT..AT + 60], &[]).map(|_| ());
        check(m, "a text media call", r)?;
        let r = hidden_media(m, &c.ids[AT..AT + after + 20], &[late]).map(|_| ());
        check(m, "a media call with a span after the image's end", r)?;
        let r = prompt_media(m, &c.ids[AT..AT + 60], &[]).map(|_| ());
        check(m, "a media prompt for the next token", r)?;
        let r = m
            .prefill_hidden(&c.ids[AT..AT + 60], PrefillPath::Wide)
            .map(|_| ());
        check(m, "a plain hidden prompt", r)?;
        let r = m
            .prefill_with(&c.ids[AT..AT + 60], PrefillPath::Wide)
            .map(|_| ());
        check(m, "a plain prompt", r)?;
        let r = m.step(&[c.ids[AT]]).map(|_| ());
        check(m, "a decode step", r)?;
        let r = m.step_rows([c.ids[AT], c.ids[AT + 1]]).map(|_| ());
        check(m, "a pass of two rows", r)?;
        m.reset()?;
        println!("  call start  {}", verdict(ok));
        Ok(ok)
    }

    /// S5 and its refusal. The calls that take the checkpoints are the prompt calls (`prefill_with`): a hidden
    /// call takes none.
    fn clause_cut(
        m: &mut Qwen35moeModel,
        c1: &Case,
        c2: &Case,
        plain: &[f32],
    ) -> Result<bool, GateError> {
        m.body_parts("gate_qwen35_media")?.2.set_checkpoints(true);
        let at = c1.spans[0].at;
        let after = at + c1.spans[0].len;
        // The cut run: the text before the image, the image and the text after it, a cut back to the image's
        // start, the text after the image again. The reference: the text before the image, then the same text.
        m.reset()?;
        prompt_media(m, &c1.ids[..at], &[])?;
        prompt_media(m, &c1.ids[at..], &c1.image_spans(at))?;
        m.rollback(u32::try_from(at)?)?;
        let cut_state = m.pos() as usize == at && m.rope_saved()?.is_empty();
        let cut_table = m.rope_rows()?;
        let cut = hidden_media(m, &c1.ids[after..], &[])?;
        m.reset()?;
        prompt_media(m, &c1.ids[..at], &[])?;
        let fresh = hidden_media(m, &c1.ids[after..], &[])?;
        let mut ok = cut_state;
        println!(
            "  S5: after the cut the sequence stands at {at} with no image {}",
            verdict(cut_state)
        );
        ok &= equal("S5 the cut table vs the text load's", &cut_table, plain);
        ok &= equal(
            "S5 the text after the cut vs the same text from a fresh run",
            &cut,
            &fresh,
        );
        // A cut inside an image (checkpoints stand at each ubatch multiple) is refused before anything moves.
        m.reset()?;
        prompt_media(m, &c2.ids, &c2.image_spans(0))?;
        let (pos, table, held) = (m.pos(), m.rope_rows()?, spans_of(&m.rope_saved()?));
        for inside in [UBATCH, 2 * UBATCH] {
            let r = m.rollback(u32::try_from(inside)?);
            let refused = refused(
                &format!("S5 a cut to {inside} inside c2's second image"),
                r,
                "inside the image",
            );
            let kept = m.pos() == pos
                && first_diff(&m.rope_rows()?, &table).is_none()
                && spans_of(&m.rope_saved()?) == held;
            println!(
                "    the position, table and record as they were {}",
                verdict(kept)
            );
            ok &= refused && kept;
        }
        m.body_parts("gate_qwen35_media")?.2.set_checkpoints(false);
        m.reset()?;
        println!("  S5  {}", verdict(ok));
        Ok(ok)
    }

    /// S7, the record: saved, reset, restored.
    fn clause_record(m: &mut Qwen35moeModel, c: &Case, plain: &[f32]) -> Result<bool, GateError> {
        m.reset()?;
        hidden_media(m, &c.ids, &c.image_spans(0))?;
        let (before, saved) = (m.rope_rows()?, m.rope_saved()?);
        m.reset()?;
        let mut ok = equal(
            "S7 reset: the table vs the text load's",
            &m.rope_rows()?,
            plain,
        );
        m.rope_restore(saved.clone())?;
        ok &= equal(
            "S7 restore: the table vs before the reset",
            &m.rope_rows()?,
            &before,
        );
        let same = spans_of(&m.rope_saved()?) == spans_of(&saved) && !saved.is_empty();
        println!(
            "    S7 the restored record is the saved one: {}",
            verdict(same)
        );
        ok &= same;
        m.reset()?;
        println!("  S7 the record  {}", verdict(ok));
        Ok(ok)
    }

    /// S8, the load half: a step graph captured before any media call, on a text prompt.
    fn capture_at_load(m: &mut Qwen35moeModel, c: &Case) -> Result<usize, GateError> {
        m.reset()?;
        let t = prompt_media(m, &c.ids[..c.spans[0].at], &[])?;
        m.step(&[t])?;
        let nodes = m.step_graph_nodes()?.len();
        m.reset()?;
        println!("  S8 a step graph of {nodes} nodes captured at load, before any media call");
        Ok(nodes)
    }

    /// S8, the replay half: after an image call the captured graph's replay equals the eager step.
    fn clause_replay(m: &mut Qwen35moeModel, c: &Case, nodes: usize) -> Result<bool, GateError> {
        const STEPS: usize = 8;
        let run = |m: &mut Qwen35moeModel, mode: StepMode| -> Result<Vec<Vec<f32>>, GateError> {
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
            "  S8 the replayed graph is the one captured at load ({nodes} nodes) {}",
            verdict(same_graph)
        );
        for (i, (g, e)) in graph.iter().zip(&eager).enumerate() {
            ok &= equal(&format!("S8 step {i}: replay vs eager logits"), g, e);
        }
        println!("  S8  {}", verdict(ok));
        Ok(ok)
    }

    /// One slot's greedy steps alone, by `step` on the selected slot (its own table, not the pass's): each step's
    /// logits.
    fn solo(
        m: &mut Qwen35moeModel,
        slot: usize,
        first: u32,
        steps: usize,
    ) -> Result<Vec<Vec<f32>>, GateError> {
        m.select_slot(slot)?;
        let (mut tok, mut out) = (first, Vec::new());
        for _ in 0..steps {
            tok = m.step(&[tok])?;
            out.push(m.logits()?);
        }
        Ok(out)
    }

    /// A pass of slots in which one stands inside an image (position 100, c1's record restored ahead of it) is
    /// refused, the slot being the live one or a parked one: no slot's position, table or record moves.
    fn clause_inside_slots(
        m: &mut Qwen35moeModel,
        c: &Case,
        text: &[u32],
    ) -> Result<bool, GateError> {
        const AT: usize = 100;
        let image = &c.spans[0];
        println!(
            "  call start, slot pass: a slot at position {AT} inside the image at {}..{}",
            image.at,
            image.at + image.len
        );
        type Seen = Vec<(u32, Vec<f32>, Vec<(usize, usize, usize)>)>;
        let seen = |m: &mut Qwen35moeModel| -> Result<Seen, GateError> {
            let mut out = Vec::new();
            for s in [0, 1] {
                m.select_slot(s)?;
                out.push((m.pos(), m.rope_rows()?, spans_of(&m.rope_saved()?)));
            }
            Ok(out)
        };
        let mut ok = true;
        for (inside, what) in [(1, "the parked slot"), (0, "the live slot")] {
            let other = 1 - inside;
            m.select_slot(other)?;
            m.reset()?;
            let next = prompt_media(m, text, &[])?;
            m.select_slot(inside)?;
            m.reset()?;
            hidden_media(m, &c.ids[..AT], &[])?;
            m.rope_restore(c.record()?)?;
            let before = seen(m)?;
            let r = m
                .step_slots(&[(other, &[next]), (inside, &[c.ids[AT]])])
                .map(|_| ());
            let got = refused(
                &format!("a pass with {what} inside the image"),
                r,
                "inside the image",
            );
            let after = seen(m)?;
            let kept = before
                .iter()
                .zip(&after)
                .all(|(b, a)| b.0 == a.0 && first_diff(&b.1, &a.1).is_none() && b.2 == a.2);
            println!(
                "    {what}: positions, tables and records as they were {}",
                verdict(kept)
            );
            ok &= got && kept;
        }
        for s in [1, 0] {
            m.select_slot(s)?;
            m.reset()?;
        }
        println!("  call start, slot pass  {}", verdict(ok));
        Ok(ok)
    }

    /// S6 and the park half of S7: two slots, slot 0 holding c1's image and slot 1 a text.
    fn clause_slots(m: &mut Qwen35moeModel, c: &Case) -> Result<bool, GateError> {
        const STEPS: usize = 4;
        let at = c.spans[0].at;
        let after = at + c.spans[0].len;
        let text: Vec<u32> = c.ids[..200].to_vec();
        let cont: Vec<u32> = c.ids[after..after + 64].to_vec();
        m.add_slots(2)?;
        let fresh = |m: &mut Qwen35moeModel| -> Result<(u32, u32), GateError> {
            m.select_slot(1)?;
            m.reset()?;
            let b = prompt_media(m, &text, &[])?;
            m.select_slot(0)?;
            m.reset()?;
            let a = prompt_media(m, &c.ids, &c.image_spans(0))?;
            Ok((a, b))
        };
        // The references: each slot stepped alone by `step`, which reads the live sequence's table and not the pass's.
        let (ta, tb) = fresh(m)?;
        let solo_a = solo(m, 0, ta, STEPS)?;
        fresh(m)?;
        let solo_b = solo(m, 1, tb, STEPS)?;
        // One pass of both.
        let (mut ta, mut tb) = fresh(m)?;
        let mut ok = true;
        for i in 0..STEPS {
            let out = m.step_slots(&[(0, &[ta]), (1, &[tb])])?;
            let logits = m.slots_logits()?;
            ok &= equal(
                &format!("S6 pass {i}: the image slot's logits vs stepped alone"),
                &logits[0],
                &solo_a[i],
            );
            ok &= equal(
                &format!("S6 pass {i}: the text slot's logits vs stepped alone"),
                &logits[1],
                &solo_b[i],
            );
            (ta, tb) = (out.ids[0], out.ids[1]);
        }
        println!("  S6  {}", verdict(ok));
        // S7, parked: the text slot's own call, and the image slot's continuation after it.
        m.select_slot(1)?;
        m.reset()?;
        let ref_b = hidden_media(m, &text, &[])?;
        m.select_slot(0)?;
        m.reset()?;
        prompt_media(m, &c.ids, &c.image_spans(0))?;
        let ref_cont = hidden_media(m, &cont, &[])?;
        m.reset()?;
        prompt_media(m, &c.ids, &c.image_spans(0))?;
        m.select_slot(1)?;
        m.reset()?;
        let park_b = hidden_media(m, &text, &[])?;
        m.select_slot(0)?;
        let park_cont = hidden_media(m, &cont, &[])?;
        let mut parked = equal(
            "S7 parked: the text slot's hidden rows vs its own run",
            &park_b,
            &ref_b,
        );
        parked &= equal(
            "S7 parked: the image slot's continuation vs the uninterrupted run",
            &park_cont,
            &ref_cont,
        );
        println!("  S7 parked  {}", verdict(parked));
        for s in [1, 0] {
            m.select_slot(s)?;
            m.reset()?;
        }
        let inside = clause_inside_slots(m, c, &text)?;
        Ok(ok && parked && inside)
    }

    // ------------------------------------------------------------ set F: the decode after an image

    /// Set F of prompt `k`: mainline's greedy decode steps after the prompt, its CPU twin's rows beside them.
    struct Decode {
        /// The token each step took (`ids[0]` is the greedy token after the prompt).
        ids: Vec<u32>,
        /// `result_norm` of each step's row, and the CPU twin's.
        rows: Vec<f32>,
        twin: Vec<f32>,
        /// The position after each step's token, and the position of the first one.
        n_past: Vec<i32>,
        n_past_start: i64,
    }

    impl Decode {
        /// Set F of prompt `k` of the seat, held to the same prompt as `c` (its tokens file's digest, its id
        /// count and its spans) and to the twin taking the same ids as the CUDA run: the ids clause's premise.
        fn load(seat: &Seat, k: usize, c: &Case) -> Result<Decode, GateError> {
            let fam = seat
                .decode
                .ok_or_else(|| format!("{} has no decode sets", seat.name))?;
            let open = |set: &str| ClefvisSet::open(&fam.path(set), fam);
            let (f, twin) = (open(fam.sets[k])?, open(fam.sets[k + 2])?);
            if f.man.header.tokens_file_sha256.as_deref() != Some(c.tokens_sha.as_str())
                || f.n_ids()? != c.ids.len()
                || f.spans != c.spans
            {
                return Err(format!("{}: set F is not of the prompt of set C", c.name).into());
            }
            let line = f.decode()?;
            let (steps, n_past_start) = (line.steps, line.n_past_start);
            let d = Decode {
                ids: f.decode_ids()?,
                rows: f.decode_result_norm()?,
                twin: twin.decode_result_norm()?,
                n_past: f.decode_n_past()?,
                n_past_start,
            };
            if d.ids.len() != steps
                || d.n_past.len() != steps
                || d.rows.len() != steps * c.width
                || d.twin.len() != d.rows.len()
            {
                return Err(format!("{}: set F's arrays are not {steps} steps", c.name).into());
            }
            if twin.decode_ids()? != d.ids {
                return Err(format!(
                    "{}: set F's CPU twin takes other ids than its CUDA run; the ids clause is written for a twin that takes all {steps}",
                    c.name
                )
                .into());
            }
            Ok(d)
        }
    }

    /// The rows of `x` (`gain.len()` values each) RMS-normed by `gain`, in f64 then rounded: a decode step's
    /// `result_norm` from its last layer's output (the card's own norm differs from this in f32 order, about
    /// 1e-7 of a row).
    fn normed(x: &[f32], gain: &[f32], eps: f32) -> Vec<f32> {
        x.chunks(gain.len())
            .flat_map(|r| {
                let ms = r.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / r.len() as f64;
                let s = 1.0 / (ms + f64::from(eps)).sqrt();
                r.iter()
                    .zip(gain)
                    .map(move |(&v, &g)| (f64::from(v) * s * f64::from(g)) as f32)
            })
            .collect()
    }

    /// The final norm's gain, read back from the card.
    fn head_gain(m: &mut Qwen35moeModel) -> Result<Vec<f32>, GateError> {
        let (gpu, w, _) = m.body_parts("gate_qwen35_media")?;
        let Some(DevWeight::F32 { w: gain, .. }) = w.get("output_norm.weight") else {
            return Err("output_norm.weight is not resident as F32".into());
        };
        Ok(gain.buf().to_host_vec(gpu.stream())?)
    }

    /// S2 on set F and S4: after prompt `c`'s image call, set F's steps through the decode path. S2: our
    /// (t, h, w) of each decoded row, and the table row the step reads, equal the position set F dumped
    /// (`n_past`, the position after the token). S4: the first token and every step's next token equal set F's
    /// ids, every step's `result_norm` (the last layer's output, tapped on the eager step, normed here) is
    /// inside `DECODE_K` times the CPU twin's distance from mainline's CUDA run, statistic by statistic, and the
    /// steps' logits equal the same ids' through passes of two rows bit for bit.
    fn clause_decode(
        m: &mut Qwen35moeModel,
        c: &Case,
        f: &Decode,
        plain: &[f32],
        gain: &[f32],
        eps: f32,
    ) -> Result<bool, GateError> {
        let (n, steps) = (c.ids.len(), f.ids.len());
        println!(" {} decode ({} steps after {n} ids):", c.name, steps);
        m.reset()?;
        let first = prompt_media(m, &c.ids, &c.image_spans(0))?;
        m.set_layer_taps(true)?;
        let (mut rows, mut next, mut logits) = (
            Vec::with_capacity(steps * c.width),
            Vec::with_capacity(steps),
            Vec::with_capacity(steps),
        );
        for &tok in &f.ids {
            next.push(m.step(&[tok])?);
            logits.push(m.logits()?);
            let taps = m.layer_taps()?;
            let last = taps.last().ok_or("layer taps hold no layer")?;
            rows.extend(normed(last, gain, eps));
        }
        m.set_layer_taps(false)?;

        let wrong: Vec<usize> = (0..steps)
            .filter(|&i| next[i] != f.ids.get(i + 1).copied().unwrap_or(next[i]))
            .collect();
        let ok_ids = first == f.ids[0] && wrong.is_empty();
        println!(
            "  S4 ids: the token after the prompt {first} (set F {}), the {} steps' next tokens differ from set F's at steps {wrong:?}  {}",
            f.ids[0],
            steps - 1,
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
            "  S2 decode: the (t, h, w) of rows {n}..{} against set F's n_past, first differing step {bad_pos:?}; the table rows they read, first differing step {bad_row:?}; first position {} (set F {}), the model stands at {} {}",
            n + steps,
            saved.pos(n).0,
            f.n_past_start,
            m.pos(),
            verdict(ok_pos)
        );

        // The route: the same ids from the same state through a pass of two rows (the pass's launches read the
        // sequence's table where the decode chain's own enqueue does), logits bit for bit. No noise floor: the
        // band below cannot see a decode that reads the shared table, an error of a few hundredths of a row.
        m.reset()?;
        prompt_media(m, &c.ids, &c.image_spans(0))?;
        let mut route_bad = Vec::new();
        for (j, pair) in f.ids.as_chunks::<2>().0.iter().enumerate() {
            m.step_rows([pair[0], pair[1]])?;
            m.keep_rows(KeptRows::prefix(2), PassKind::Pair)?;
            let got = m.rows_logits::<2>()?;
            for (r, row) in got.iter().enumerate() {
                if first_diff(row, &logits[2 * j + r]).is_some() {
                    route_bad.push(2 * j + r);
                }
            }
        }
        m.set_mode(StepMode::Graph);
        let ok_route = route_bad.is_empty() && steps % 2 == 0;
        println!(
            "  S4 route: {steps} decode steps' logits against the same ids through passes of two rows, {} differ bit for bit (first {:?})  {}",
            route_bad.len(),
            route_bad.first(),
            verdict(ok_route)
        );

        let all: Vec<usize> = (0..steps).collect();
        let ours = Stats::of(c.width, &rows, &f.rows, &all);
        let twin = Stats::of(c.width, &f.twin, &f.rows, &all);
        let band = Stats {
            med: DECODE_K * twin.med,
            p90: DECODE_K * twin.p90,
            q: std::array::from_fn(|i| DECODE_K * twin.q[i]),
        };
        let within = ours.all().iter().zip(band.all()).all(|(a, b)| *a <= b);
        println!("  S4 band {} decode: ours {}", c.name, ours.show());
        println!("    twin {}", twin.show());
        println!("    band {}  {}", band.show(), verdict(within));
        Ok(ok_ids && ok_pos && ok_route && within)
    }

    // ------------------------------------------------------------ the run

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let seat = seat_of(&ref_model_path()?)?;
        let path = Path::new(seat.path);
        println!("gate_qwen35_media: {} at {}", seat.name, path.display());
        let eps = check_shape(
            seat,
            &Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?,
        )?;
        let tower = ClefvisSet::open(&seat.taps.path(seat.taps.sets[0]), seat.taps)?;
        let cases = [
            Case::load(seat, 0, "c1", &tower)?,
            Case::load(seat, 1, "c2", &tower)?,
            Case::load(seat, 2, "c3", &tower)?,
        ];
        let c1 = &cases[0];
        let reqs = [
            Req {
                name: format!("{} ids", c1.ids.len()),
                ids: c1.ids.clone(),
            },
            Req {
                name: "517 ids (a tail under the Wide floor)".into(),
                ids: cases[seat.long].ids[..517].to_vec(),
            },
        ];
        let mut ok = true;

        println!("the text load (media: false): references and refusals");
        let mut text = (seat.open)(seat, false)?;
        let plain = text.rope_rows()?;
        let mut refs = Vec::new();
        for r in &reqs {
            text.reset()?;
            refs.push(text.prefill_hidden(&r.ids, PrefillPath::Wide)?);
        }
        ok &= refused(
            "an image span on a text load",
            hidden_media(&mut text, &c1.ids, &c1.image_spans(0)),
            "without media",
        );
        ok &= refused(
            "a record of images on a text load",
            text.rope_restore(c1.record()?),
            "without media",
        );
        ok &= refused(
            "the media tap on a text load",
            text.set_media_tap(true),
            "Open35::media",
        );
        drop(text);

        println!("the media load (media: true)");
        let mut m = (seat.open)(seat, true)?;
        let nodes = capture_at_load(&mut m, c1)?;
        ok &= clause_text(&mut m, &plain, &reqs, &refs, "fresh")?;
        println!("(i) (ii) (iii) S1 and the band, per set C prompt (K = {K}, twin x {TWIN})");
        for (i, c) in cases.iter().enumerate() {
            ok &= run_case(&mut m, c, &plain, i == 1)?;
        }
        ok &= clause_text(
            &mut m,
            &plain,
            &reqs,
            &refs,
            "after image calls and a reset",
        )?;
        if seat.body35_tail {
            ok &= clause_capture(&mut m, c1, &plain)?;
            ok &= clause_inside(&mut m, c1)?;
            ok &= clause_cut(&mut m, c1, &cases[1], &plain)?;
            ok &= clause_record(&mut m, &cases[1], &plain)?;
        }
        ok &= clause_replay(&mut m, c1, nodes)?;
        if seat.body35_tail {
            ok &= clause_slots(&mut m, c1)?;
        }
        if seat.decode.is_some() {
            println!("S2 and S4 on set F (K = {DECODE_K} of the twin)");
            let gain = head_gain(&mut m)?;
            for (k, c) in cases.iter().take(2).enumerate() {
                let f = Decode::load(seat, k, c)?;
                ok &= clause_decode(&mut m, c, &f, &plain, &gain, eps)?;
            }
        }
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
