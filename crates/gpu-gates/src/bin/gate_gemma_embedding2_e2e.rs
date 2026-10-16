//! GPU gate for EmbeddingGemma 2's encoder (`bloomery-gpu-gemma-embedding2`)
//! end to end on the Q8_0 file the sets are dumped from ([`MODEL`], held to
//! its sha256), against llama.cpp mainline's text set dumped on the card
//! ([`TEXT`]) and its CPU twin ([`TEXT_CPU`]), both read through
//! [`refset::emb2::Emb2Set::open`]: each case the set names, its ids read from the
//! tokenizer oracle's gemma4 set ([`IDS_SET`]) and held to the set's count
//! and digest, through one pass at the load's context with its taps read
//! back.
//!
//! A row's band is [`TWIN`] times the CPU twin's own distance from the card
//! set, and at least [`FLOOR`]: the twin is the floor any two realizations of
//! the graph lie apart (the Qwen seats' rule, `shared/glm5next_media.rs`). A
//! row is within its band only when its distance and the twin's are finite.
//!
//! 1. (t) Per case: `inp_scaled` bit for bit (the host's token rows times
//!    `√hidden`, the one f32 multiply `ggml_scale` makes); `inp_per_layer`,
//!    every layer's `l_out`, `result_norm`, the token rows against `embd`
//!    and their mean ([`Pass::pooled`]) against `pooled`, each within its
//!    band. A case longer than a sliding layer's reach plus one catches a
//!    dropped window; a set with no such case fails by name. The host test
//!    `key_bounds_are_the_symmetric_mask` holds the exact key edges; every
//!    case holds the global layers.
//! 2. (d) The longest case run again with no taps is bit for bit the first
//!    run's rows.
//! 3. (i) Image-only and `mix_free-448x448` inputs assembled by the serving helper,
//!    pooled and normalized against the server card set and its CPU twin.
//! 4. (r) Refused by name, the load untouched after each: a pass of no ids,
//!    of one id past the load's context and of an id past the vocabulary,
//!    and a load at one position past [`CARD_CTX`].

#[cfg(feature = "gemma-embedding2")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_gemma_embedding2_e2e", gate::run())
}

#[cfg(feature = "gemma-embedding2")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[path = "shared/emb2_image.rs"]
mod image;

#[cfg(feature = "gemma-embedding2")]
mod gate {
    use std::path::Path;
    use std::time::Instant;

    use crate::e2e::{clause, elapsed, refused_by, rel, same_bits, tap};
    use crate::image::open;
    use bloomery_gpu::Gpu;
    use bloomery_gpu_gates::{
        GateError, RefManifest, checks_failed, data_dir, ref_model_path, sha256, verdict,
    };
    use bloomery_gpu_gemma_embedding2::{Encoder, Pass, card_bytes};
    use gguf::Split;
    use model::arch::gemma_embedding2::hparams::{CARD_CTX, Hparams};
    use refset::arch::gemma_embedding2::{
        IDS_SET, MODEL, MODEL_SHA256, REL_FLOOR as FLOOR, TEXT, TEXT_CPU, TWIN,
    };
    use refset::ik::RefHeader;
    use refset::tokenizer::read_ids;

    /// One case of the set: its name and its ids.
    struct Case {
        id: String,
        ids: Vec<u32>,
    }

    fn check_case_identities(card: &RefHeader, cpu: &RefHeader) -> Result<(), GateError> {
        if !card
            .other
            .iter()
            .filter(|line| line.starts_with("# case\t"))
            .eq(cpu.other.iter().filter(|line| line.starts_with("# case\t")))
        {
            return Err("CPU twin case identities differ from the card set".into());
        }
        Ok(())
    }

    /// The cases `man`'s `# case <id> <n> <path> <sha256>` lines name, each
    /// one's ids read from [`IDS_SET`] and refused by name when their count
    /// or their file's digest is not the line's.
    fn cases(man: &RefManifest) -> Result<Vec<Case>, GateError> {
        let dir = data_dir().join(IDS_SET);
        let mut out = Vec::new();
        for line in &man.header.other {
            let Some(rest) = line.strip_prefix("# case\t") else {
                continue;
            };
            let fields: Vec<&str> = rest.split('\t').collect();
            let [id, n, _, sha] = fields[..] else {
                return Err(
                    format!("{}: a case line of another shape: {line:?}", TEXT.name).into(),
                );
            };
            let path = dir.join(format!("{id}.ids"));
            let want = n
                .parse::<usize>()
                .map_err(|e| format!("{}: case {id}: token count {n:?}: {e}", TEXT.name))?;
            let ids = read_ids(&path, want)?;
            let digest = sha256(&path)?;
            if digest != sha {
                return Err(format!(
                    "{}: {} ids of digest {digest}; the set's case line says {n} of {sha}",
                    path.display(),
                    ids.len()
                )
                .into());
            }
            out.push(Case {
                id: id.to_string(),
                ids,
            });
        }
        if out.is_empty() {
            return Err(format!("{}: no case line", TEXT.name).into());
        }
        Ok(out)
    }

    /// One row held to the card set: its distance, the twin's and the band.
    struct Held {
        r: f64,
        twin: f64,
        band: f64,
    }

    impl Held {
        fn of(ours: &[f32], card: &[f32], twin: &[f32]) -> Held {
            let (r, twin) = (rel(ours, card), rel(twin, card));
            Held {
                r,
                twin,
                band: (TWIN * twin).max(FLOOR),
            }
        }

        /// Within the band, both distances finite: `rel` makes a NaN or a
        /// length mismatch infinite, and an infinite twin an infinite band.
        fn ok(&self) -> bool {
            self.r.is_finite() && self.twin.is_finite() && self.r <= self.band
        }

        fn line(&self, what: &str) -> String {
            format!(
                "{what} {:.3e} (twin {:.3e}, band {:.3e})",
                self.r, self.twin, self.band
            )
        }
    }

    /// Clause (t) for one case: its pass with taps, held tap by tap. The
    /// pass's rows, for (d).
    fn case(
        enc: &mut Encoder,
        c: &Case,
        man: &RefManifest,
        cpu: &RefManifest,
        hp: &Hparams,
    ) -> Result<(bool, Pass), GateError> {
        let t0 = Instant::now();
        let pass = enc.pass(&c.ids, true)?;
        let taps = pass.taps.as_ref().ok_or("a pass with taps returned none")?;
        assert_eq!(taps.l_out.len(), hp.n_layer, "missing l_out taps");
        let row = |m: &RefManifest, name: &str| tap(m, &format!("{}/{name}", c.id));
        let held = |name: &str, ours: &[f32]| -> Result<Held, GateError> {
            Ok(Held::of(ours, &row(man, name)?, &row(cpu, name)?))
        };
        let mut ok = true;
        let mut lines = Vec::new();
        let exact = same_bits(&taps.inp_scaled, &row(man, "inp_scaled")?);
        lines.push(format!("inp_scaled bits {}", verdict(exact)));
        ok &= exact;
        let h = held("inp_per_layer", &taps.inp_per_layer)?;
        lines.push(h.line("inp_per_layer"));
        ok &= h.ok();
        // The worst layer by its distance over its band.
        let mut worst = (0.0f64, String::new());
        for (l, ours) in taps.l_out.iter().enumerate() {
            let h = held(&format!("l_out-{l}"), ours)?;
            if !h.ok() {
                println!(
                    "(t) {}: {} past its band {}",
                    c.id,
                    h.line(&format!("layer {l} l_out")),
                    verdict(false)
                );
                ok = false;
            }
            if h.r / h.band >= worst.0 {
                worst = (h.r / h.band, h.line(&format!("l_out worst layer {l}")));
            }
        }
        lines.push(worst.1);
        for (what, ours) in [
            ("result_norm", &taps.result_norm),
            ("embd", &pass.rows),
            ("pooled", &pass.pooled),
        ] {
            let h = held(what, ours)?;
            lines.push(h.line(what));
            ok &= h.ok();
        }
        println!(
            "(t) {}: {} ids, {}; {:.1} ms (runtime value) {}",
            c.id,
            c.ids.len(),
            lines.join("; "),
            t0.elapsed().as_secs_f64() * 1e3,
            verdict(ok)
        );
        Ok((ok, pass))
    }

    /// A case long enough for a dropped window to leave the tap bands:
    /// some row's keys stop short of the input's ends. The host test holds
    /// the exact edges. No sliding layer passes; a set with no such case fails.
    fn window(reach: &[Option<u32>], cases: &[Case]) -> bool {
        let Some(reach) = reach.iter().flatten().copied().max().map(|r| r as usize) else {
            return true;
        };
        let longest = cases.iter().map(|c| c.ids.len()).max().unwrap_or(0);
        let held = longest > reach + 1;
        println!(
            "(t) the window: the longest case holds {longest} ids, a sliding layer reaches {reach} \
             on each side {}",
            verdict(held)
        );
        held
    }

    /// Clause (r): each refusal by its words, and a pass after them the
    /// first one's bits.
    fn refusals(
        enc: &mut Encoder,
        c: &Case,
        rows: &[f32],
        vocab: usize,
    ) -> Result<bool, GateError> {
        const PASS: &str = "Encoder::pass";
        let past = vec![c.ids[0]; enc.ctx() + 1];
        let mut bad = c.ids.clone();
        bad[0] = u32::try_from(vocab)?;
        let checks = [
            (
                "no ids",
                refused_by(&enc.pass(&[], false), PASS, "0 positions"),
            ),
            (
                "one past the context",
                refused_by(&enc.pass(&past, false), PASS, "a pass takes 1 to"),
            ),
            (
                "an id past the vocabulary",
                refused_by(&enc.pass(&bad, false), PASS, "past the vocabulary"),
            ),
        ];
        let mut ok = true;
        for (what, refused) in checks {
            println!("(r) {what}: refused by name {}", verdict(refused));
            ok &= refused;
        }
        enc.pass(&vec![c.ids[0]; enc.ctx()], false)?;
        let again = enc.pass(&c.ids, false)?;
        let same = same_bits(&again.rows, rows);
        println!(
            "(r) {}: after the refusals the rows bit for bit {}",
            c.id,
            verdict(same)
        );
        Ok(ok && same)
    }

    fn images(enc: &mut Encoder, model: &Path) -> Result<bool, GateError> {
        use crate::image;
        use bloomery_gpu::media::MediaSpan;
        use bloomery_gpu_gates::bind::Vocab;
        use bloomery_gpu_vision::gemma4v::Encoder as Tower;
        use refset::arch::gemma_embedding2::{SERVER, SERVER_CPU, TOWER, TOWER_CPU, TOWER_IMAGES};
        use serve::embeddings::{InputPart, assemble};
        use vision::arch::{TokenLimits, gemma4v};

        let card = image::open(&TOWER)?;
        let cpu = image::open(&TOWER_CPU)?;
        let server = image::open(&SERVER)?;
        let server_cpu = image::open(&SERVER_CPU)?;
        let file = gguf::Gguf::open(image::projector(model)?)?;
        let hp = gemma4v::Hparams::read(&file)?;
        let limits = TokenLimits::new(70, 1120)?;
        let rule = hp.size_rule(limits)?;
        let stream = enc.context().new_stream_with_priority(0)?;
        let mut tower = Tower::load(enc.context(), &stream, &file, limits)?;
        let tok = Vocab::new(tokenizer::Tokenizer::from_gguf(model)?)?;
        let mut ok = true;
        for name in TOWER_IMAGES {
            let (rgb, digest) = image::image(&card, &cpu, name)?;
            let p = gemma4v::preprocess(&rgb, &rule)?;
            let tokens = p.plan.n_llm_h * p.plan.n_llm_w;
            let rows = tower.encode(&p)?.rows.buf().to_host_vec(&stream)?;
            for mixed in [false, true] {
                if mixed && *name != "free-448x448" {
                    continue;
                }
                let mut parts = Vec::new();
                if mixed {
                    parts.push(InputPart::Text("A picture of coloured rings and ramps: "));
                }
                parts.push(InputPart::Image {
                    begin: gemma4v::IMAGE_BEGIN,
                    end: gemma4v::IMAGE_END,
                    positions: tokens,
                });
                let assembled = assemble(parts.into_iter().map(Ok), &tok, CARD_CTX)?;
                let ids = &assembled.ids;
                let span = assembled
                    .spans
                    .first()
                    .ok_or("image assembly returned no span")?;
                let key = format!("{}_{name}", if mixed { "mix" } else { "img" });
                // The server pair must have consumed these same bytes and position count.
                if image::image(&server, &server_cpu, &key)?.1 != digest {
                    return Err(format!("{key}: server/tower image sha256 mismatch").into());
                }
                for man in [&server, &server_cpu] {
                    let prefix = format!("# image\t{key}\t");
                    let expected = format!("prompt_tokens {}", ids.len());
                    if !man
                        .header
                        .other
                        .iter()
                        .any(|l| l.starts_with(&prefix) && l.split('\t').any(|f| f == expected))
                    {
                        return Err(format!(
                            "{key}: server position count differs from {}",
                            ids.len()
                        )
                        .into());
                    }
                }
                let pass = enc.pass_media(
                    ids,
                    &[MediaSpan {
                        at: span.clone(),
                        rows: &rows,
                    }],
                    false,
                )?;
                let norm = serve::embed::normalize(&pass.pooled, serve::embed::Norm::default())?;
                for (node, ours) in [("raw", &pass.pooled), ("embd", &norm)] {
                    let key = format!("{key}/{node}");
                    let expected = image::row(&server, &key, [768, 1, 1, 1])?;
                    let twin = image::row(&server_cpu, &key, [768, 1, 1, 1])?;
                    ok &= image::compare(
                        &format!("(i) {key} pass_media {} positions", ids.len()),
                        ours,
                        &expected,
                        &twin,
                    );
                }
            }
        }
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let args: Vec<String> = std::env::args().skip(1).collect();
        if !args.is_empty() {
            return Err(format!("no arguments; got {args:?}").into());
        }
        let model = ref_model_path()?;
        let digest = sha256(&model)?;
        if digest != MODEL_SHA256 {
            return Err(format!(
                "{MODEL}: sha256 {digest}, the sets are dumped from {MODEL_SHA256}"
            )
            .into());
        }
        let split = Split::open(&model).map_err(|e| format!("open {}: {e}", model.display()))?;
        let hp = Hparams::read(&split).map_err(|e| format!("{MODEL}: {e}"))?;
        let read = model::arch::spec(&split).map_err(|e| format!("{MODEL}: {e}"))?;
        let encoder = read
            .spec
            .encoder
            .as_ref()
            .ok_or("gemma-embedding2 has no encoder description")?;
        let ctx = CARD_CTX;
        let bytes = card_bytes(&split, ctx)?;
        let man = open(&TEXT)?;
        let cpu = open(&TEXT_CPU)?;
        check_case_identities(&man.header, &cpu.header)?;
        let cases = cases(&man)?;
        let gpu = Gpu::new()?;
        println!(
            "gate_gemma_embedding2_e2e: device {}: {MODEL} at {ctx} positions, card figure {bytes} B; \
             {} cases; band {TWIN} x the CPU twin, at least {FLOOR:e}",
            gpu.device_name()?,
            cases.len()
        );
        let mut ok = window(&encoder.reach, &cases);
        let t = Instant::now();
        let mut enc = Encoder::open(split, ctx, gpu)?;
        elapsed("(open)", &t);
        let mut longest: Option<(usize, Pass)> = None;
        for (i, c) in cases.iter().enumerate() {
            let r = case(&mut enc, c, &man, &cpu, &hp).map(|(pass_ok, pass)| {
                if longest
                    .as_ref()
                    .is_none_or(|(j, _)| c.ids.len() > cases[*j].ids.len())
                {
                    longest = Some((i, pass));
                }
                pass_ok
            });
            ok &= clause("(t)", &c.id, r);
        }
        if let Some((i, first)) = longest {
            let c = &cases[i];
            ok &= clause(
                "(d)",
                &c.id,
                enc.pass(&c.ids, false)
                    .map_err(GateError::from)
                    .map(|again| {
                        let same = same_bits(&again.rows, &first.rows);
                        println!(
                            "(d) {}: {} ids, a rerun's rows bit for bit {}",
                            c.id,
                            c.ids.len(),
                            verdict(same)
                        );
                        same
                    }),
            );
            ok &= clause(
                "(r)",
                "refusals",
                refusals(&mut enc, c, &first.rows, hp.n_vocab),
            );
        }
        ok &= clause("(i)", "image inputs", images(&mut enc, &model));
        let past = Split::open(&model).map_err(|e| format!("open {}: {e}", model.display()))?;
        let refused = refused_by(
            &Encoder::open(past, ctx + 1, Gpu::new()?),
            "Encoder::open",
            "a context of",
        );
        println!(
            "(r) a load at {} positions: refused by name {}",
            ctx + 1,
            verdict(refused)
        );
        ok &= refused;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }

    #[cfg(test)]
    mod tests {
        use super::{Case, RefHeader, check_case_identities, window};

        #[test]
        fn a_window_case_reaches_past_the_longest_sliding_layer() {
            let mut cases = [Case {
                id: "window".to_owned(),
                ids: vec![0; 3],
            }];
            assert!(window(&[None], &[]));
            assert!(!window(&[Some(1), None, Some(2)], &cases));
            cases[0].ids.push(0);
            assert!(window(&[Some(1), None, Some(2)], &cases));
        }

        #[test]
        fn a_cpu_case_of_another_digest_is_refused_by_name() {
            let card = RefHeader {
                other: vec![format!(
                    "# case\tcases/01\t2\tcases/01.ids\t{}",
                    "a".repeat(64)
                )],
                ..RefHeader::default()
            };
            let mut cpu = card.clone();
            check_case_identities(&card, &cpu).unwrap();
            cpu.other[0] = cpu.other[0].replace(&"a".repeat(64), &"b".repeat(64));
            let err = check_case_identities(&card, &cpu).unwrap_err().to_string();
            assert_eq!(err, "CPU twin case identities differ from the card set");
        }
    }
}
