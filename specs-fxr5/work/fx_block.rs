    // ------------------------------- the fixture tier's clauses against ik

    /// The fixture-oracle clauses a run declares, held by `tier::expect_fixture_oracle` at the
    /// gate's end in both tiers: `ik_argmax_fx` at the three places the last argmax is compared
    /// ((g)'s free arm, (g)'s forced arm and (k)(iii)), (cfx) and (crx) on the batch set, and one
    /// (tfx) per (step set, feed) the run takes ([`FX_STEP_SETS`]: the two 4-token sets fed by
    /// steps, D1K and D3K fed by ubatches with ik's prefill routes planted, D3K fed by steps).
    const FX_ARGMAX: usize = 3;
    const FX_FREE: usize = 2;
    const FX_STEP_SETS: usize = 5;
    const FX_ORACLE: usize = FX_ARGMAX + FX_FREE + FX_STEP_SETS;

    /// A site that reaches the stream through a sigmoid: its slope is at most ¼, squared.
    const MULT_SIGMOID: f64 = 1.0 / 16.0;

    /// A gate through silu: the relative error of silu(g) is at most `1 + g(1 − σ(g))` times the
    /// gate's own, 1.28 at g = 1.28, squared.
    const MULT_SILU: f64 = 1.64;

    /// The gated-residual mixes' down and up (`hc_*_down` and `hc_*_up`, Q8_0 sites) reach the
    /// stream through σ, whose slope is at most ¼: they add under 4 % to a band
    /// ([`gemm_band`]'s derivation).
    const HC_SHARE: f64 = 1.04;

    /// One projection's term in a band: the site's tensor and file type, its two engines' joint
    /// worst-crest rounding `q` (`act_rule::site_rel(ty, arm).joint()`: ik's activation form and
    /// ours, independent) and the multiplicity of `q²` in the stream the site joins.
    struct Term {
        site: String,
        ty: GgmlType,
        q: f64,
        mult: f64,
    }

    impl Term {
        /// The site `tensor` of `file`, read under our launch `arm`. A type no launch reads is
        /// refused by name.
        fn of(file: &Split, tensor: String, arm: Arm, mult: f64) -> Result<Term, GateError> {
            let ty = file
                .find(&tensor)
                .map(|(_, t)| t.ty)
                .ok_or_else(|| format!("ik_bands: {tensor} is not in the file"))?;
            let rel = site_rel(ty, arm).ok_or_else(|| {
                format!(
                    "ik_bands: {tensor} is {ty:?}, which no launch of this program reads as \
                     {arm:?} (act_rule::site_rel)"
                )
            })?;
            Ok(Term {
                site: tensor,
                ty,
                q: rel.joint(),
                mult,
            })
        }

        /// The variance this site adds, in units of a relative error squared.
        fn var(&self) -> f64 {
            self.mult * self.q * self.q
        }

        fn text(&self) -> String {
            format!(
                "{} {:?} q {:.3e} x{:.4}",
                self.site.trim_start_matches("blk."),
                self.ty,
                self.q,
                self.mult
            )
        }
    }

    /// The bands of a fixture clause against ik's sets, per layer and at the head: the relative
    /// distance our value may sit from ik's where both engines took the same experts. Counted as
    /// the GLM gate's `gemm_bands` counts, over every projection site, since against ik every
    /// site differs (each engine rounds its activation, each its own way): a site's `q` is its
    /// two roundings' `√(q_ik² + q_ours²)` at the worst crest ([`site_rel`]); a site's variance is
    /// `mult · q²`; the variances of every site a stream has gained add
    /// (outputs of one input through different weights' rows are independent); a band is
    /// [`HC_SHARE`] times the root of the sum. `router[l]` is the stream the layer's router reads
    /// (the stream before the layer, the PLE site's terms on its layer and the layer's mixer);
    /// `stream[l]` the layer's output; `head` the logits.
    struct IkBands {
        router: Vec<f64>,
        stream: Vec<f64>,
        head: f64,
    }

    /// [`IkBands`] with our dense projections read as `arm` (`Gemv` for the steps, `Gemm` for the
    /// ubatch walk), the routed experts as `Arm::Expert` on both (the host tier's experts round
    /// as ik's do and the card's as the q8_1 forms do; the joint is the conservative of the two),
    /// from the types the file names. Every term is printed.
    ///
    /// Sites per layer, with the multiplicity of each in units of its own `q²` (the GLM gate's
    /// counts: a projection once, a sigmoid's ¼ slope squared, silu's 1.28 log-slope squared):
    /// - a delta layer: `attn_qkv` 3 (q, k and v enter the state bilinearly), `attn_gate` (the
    ///   gated norm's silu gate) 1.64, β and α 1/16 each (a sigmoid or softplus decay, slope at most
    ///   ¼), `ssm_out` 1;
    /// - a selecting layer: `attn_q` 1 + 1/16 (its rows beside each head's sigmoid gate), `attn_k`
    ///   1, `attn_v` 1, `attn_output` 1; the indexer's projections 0 (they reach the output only
    ///   by which pools are read, a selection, never a distance);
    /// - the block: the shared expert's gate 1.64, up 1 and down 1, and the routed experts' the
    ///   same (`ffn_gate_exps` through silu 1.64, `ffn_up_exps` 1, `ffn_down_exps` 1); the router
    ///   and the shared gate's `ffn_gate_inp*` are F32, which neither engine rounds;
    /// - the PLE site's `ple_key` and `ple_value`, 1 each, on its layer;
    /// - the head's `output`, 1.
    fn ik_bands(arm: Arm) -> Result<IkBands, GateError> {
        let file = open_split()?;
        let s = shape();
        println!(
            "ik_bands {arm:?}: band = {HC_SHARE} · √(Σ mult · q²), q = site_rel(ty, arm).joint() \
             at the worst crest (ik's activation form and ours, independent); the head adds \
             `output`"
        );
        let (mut c, mut router, mut stream) = (0.0f64, Vec::new(), Vec::new());
        for l in 0..n_layer() {
            let qsa = s.qsa.contains(&l);
            let mut terms = Vec::new();
            if s.ple_layer == Some(l) {
                for name in [names::ple_key(l), names::ple_value(l)] {
                    terms.push(Term::of(&file, name, arm, 1.0)?);
                }
            }
            let ple_var: f64 = terms.iter().map(Term::var).sum();
            let mixer: Vec<Term> = if qsa {
                vec![
                    Term::of(&file, names::attn_q(l), arm, 1.0 + MULT_SIGMOID)?,
                    Term::of(&file, names::attn_k(l), arm, 1.0)?,
                    Term::of(&file, names::attn_v(l), arm, 1.0)?,
                    Term::of(&file, names::attn_output(l), arm, 1.0)?,
                ]
            } else {
                vec![
                    Term::of(&file, names::attn_qkv(l), arm, 3.0)?,
                    Term::of(&file, names::attn_gate(l), arm, MULT_SILU)?,
                    Term::of(&file, names::ssm_beta(l), arm, MULT_SIGMOID)?,
                    Term::of(&file, names::ssm_alpha(l), arm, MULT_SIGMOID)?,
                    Term::of(&file, names::ssm_out(l), arm, 1.0)?,
                ]
            };
            let block = vec![
                Term::of(&file, names::ffn_gate_inp(l), arm, 1.0)?,
                Term::of(&file, names::ffn_gate_inp_shexp(l), arm, MULT_SIGMOID)?,
                Term::of(&file, names::ffn_gate_shexp(l), arm, MULT_SILU)?,
                Term::of(&file, names::ffn_up_shexp(l), arm, 1.0)?,
                Term::of(&file, names::ffn_down_shexp(l), arm, 1.0)?,
                Term::of(&file, names::ffn_gate_exps(l), Arm::Expert, MULT_SILU)?,
                Term::of(&file, names::ffn_up_exps(l), Arm::Expert, 1.0)?,
                Term::of(&file, names::ffn_down_exps(l), Arm::Expert, 1.0)?,
            ];
            let (mixer_var, block_var): (f64, f64) = (
                mixer.iter().map(Term::var).sum(),
                block.iter().map(Term::var).sum(),
            );
            c += ple_var;
            router.push(HC_SHARE * (c + mixer_var).sqrt());
            c += mixer_var + block_var;
            stream.push(HC_SHARE * c.sqrt());
            println!(
                "ik_bands {arm:?} layer {l} ({}): router {:.4e} stream {:.4e}; {}",
                if qsa { "selecting" } else { "delta" },
                router[l],
                stream[l],
                terms
                    .iter()
                    .chain(&mixer)
                    .chain(&block)
                    .map(Term::text)
                    .collect::<Vec<_>>()
                    .join("; ")
            );
        }
        let head = Term::of(&file, names::output(), arm, 1.0)?;
        let head_band = HC_SHARE * (c + head.var()).sqrt();
        println!(
            "ik_bands {arm:?} head {head_band:.4e} = {HC_SHARE} · √({c:.4e} the layers + {:.4e} {})",
            head.var(),
            head.text()
        );
        Ok(IkBands {
            router,
            stream,
            head: head_band,
        })
    }

    /// [`ik_bands`] of `arm` (`Gemv` or `Gemm`), computed and printed on the first call.
    fn fx_bands(arm: Arm) -> Result<&'static IkBands, GateError> {
        static GEMV: std::sync::OnceLock<IkBands> = std::sync::OnceLock::new();
        static GEMM: std::sync::OnceLock<IkBands> = std::sync::OnceLock::new();
        let cell = match arm {
            Arm::Gemv => &GEMV,
            Arm::Gemm => &GEMM,
            Arm::Expert => return Err("ik_bands: a dense arm is Gemv or Gemm".into()),
        };
        if let Some(b) = cell.get() {
            return Ok(b);
        }
        let b = ik_bands(arm)?;
        Ok(cell.get_or_init(|| b))
    }

    /// The fixture set `set` of a fixture-oracle clause, opened through the fixture family's
    /// check (`tier::fixture_set`: missing, stale, foreign or unfinished ends the gate by name)
    /// and then its reader.
    fn fx_set(clause: &str, set: &str) -> Result<RefManifest, GateError> {
        let path = tier::fixture_set(clause, &fixture::IK, set)?;
        Ok(RefManifest::open(&path, &fixture::IK)?)
    }

    /// The fixture's `ik_argmax`: our last argmax `top` is ik's, or ik's own logit at it lies
    /// within the head's tie cap of ik's top (`flip::head_tie` at the ubatch walk's head band) —
    /// never the strict equality, which on the fixture's near-flat logits changes on any rounding
    /// change. Runs only in the fixture tier; prints its line and returns its verdict (true where
    /// the tier leaves the clause).
    fn ik_argmax_fx(vocab: usize, clause: &str, top: u32) -> Result<bool, GateError> {
        let name = format!("{clause} against ik's fixture set, as a tie band");
        if !tier::run_clause(&name, Tag::FixtureOracle)? {
            return Ok(true);
        }
        let ik = ik_last(&fx_set(&name, fixture::BATCH)?, vocab)?;
        let band = fx_bands(Arm::Gemm)?.head;
        let ok = flip::head_tie(&ik, top, band);
        println!(
            "{name}: ours {top}, ik's {}; ik's logit at ours {:.3e} below its top (tie cap {:.3e} \
             at head band {band:.3e}) {}",
            argmax(&ik),
            flip::head_gap(&ik, top),
            flip::head_cap(band, &ik),
            verdict(ok)
        );
        Ok(ok)
    }

    /// ik's picks of token `t` as the plant takes them.
    fn ik_picks(ik: &IkRoute, t: usize) -> Vec<u32> {
        ik.at(t).1.iter().map(|&e| e as u32).collect()
    }

    /// Whether a distance is inside its band; NaN is not (`d <= band` is false for it).
    fn within(d: f64, band: f64) -> bool {
        d <= band
    }

    /// How far below the lowest picked logit the plant puts every other expert: `e^-30` of the
    /// picks' probabilities, which the routing launch's renormalization over the picks does not
    /// see.
    const PLANT_FILLER: f32 = 30.0;

    /// The route one position and layer is planted with. `plant_ubatch_routes` plants router
    /// logits (the routing launch re-derives the ids and weights from them), and a dump of ik's
    /// prefill routes holds the picks and the final weights, so the logits are made from those:
    /// a pick's is `ln` of its final weight — the launch's softmax over the picks, renormalized,
    /// then gives the weight back (the weights sum to 1, to f32 rounding) — and every other
    /// expert's is [`PLANT_FILLER`] below the lowest pick's, so the launch picks exactly ik's set.
    /// The picks are distinct experts of the model, and every weight finite and positive: a
    /// route that is not is a named error. `at` is the (layer, position) it names.
    fn planted_route(
        picks: &[u32],
        weights: &[f32],
        at: (usize, usize),
    ) -> Result<RouteTap, GateError> {
        let bad = |what: &str| -> GateError {
            format!("plant: layer {} position {}: {what}", at.0, at.1).into()
        };
        if picks.len() != N_USED || weights.len() != N_USED {
            return Err(bad(&format!(
                "{} picks and {} weights, a position takes {N_USED}",
                picks.len(),
                weights.len()
            )));
        }
        let mut logits = vec![f32::NAN; N_EXPERT];
        let mut lowest = f32::INFINITY;
        for (&e, &w) in picks.iter().zip(weights) {
            if e as usize >= N_EXPERT {
                return Err(bad(&format!("expert {e}, the model has {N_EXPERT}")));
            }
            if !(w.is_finite() && w > 0.0) {
                return Err(bad(&format!("expert {e} has weight {w}")));
            }
            if !logits[e as usize].is_nan() {
                return Err(bad(&format!("expert {e} picked twice")));
            }
            logits[e as usize] = w.ln();
            lowest = lowest.min(logits[e as usize]);
        }
        for x in &mut logits {
            if x.is_nan() {
                *x = lowest - PLANT_FILLER;
            }
        }
        Ok(RouteTap {
            logits,
            ids: picks.to_vec(),
        })
    }

    /// ik's prefill routes of a set dumped with `--prefill-routes`, as the plant takes them,
    /// `[position][layer]`: read through `RefManifest::prefill_routes` (the picks, the gathered
    /// weights and the final weights, `ffn_moe_weights_norm`, as dumped), bounded by the model's
    /// expert count.
    fn prefill_plant(man: &RefManifest, positions: usize) -> Result<Vec<Vec<RouteTap>>, GateError> {
        let routes =
            man.prefill_routes("ffn_moe_topk", "ffn_moe_weights", "ffn_moe_weights_norm")?;
        routes.check_experts(u32::try_from(N_EXPERT)?)?;
        let layers_ok = routes.layers.len() == n_layer()
            && routes
                .layers
                .iter()
                .enumerate()
                .all(|(i, l)| l.layer as usize == i);
        if routes.n_used != N_USED || routes.positions != positions || !layers_ok {
            return Err(format!(
                "plant: the set's prefill routes hold {} layers of {} picks over {} positions, \
                 the run's prefill is {positions} positions of {} layers of {N_USED}",
                routes.layers.len(),
                routes.n_used,
                routes.positions,
                n_layer()
            )
            .into());
        }
        (0..positions)
            .map(|p| {
                routes
                    .layers
                    .iter()
                    .enumerate()
                    .map(|(l, r)| {
                        let at = p * N_USED..(p + 1) * N_USED;
                        planted_route(&r.picks[at.clone()], &r.last[at], (l, p))
                    })
                    .collect()
            })
            .collect()
    }

    /// The flips of `eager`'s routes against ik's `ik_routes` (one per layer), and for each the
    /// pair cap and margin bound at its layer's router band.
    fn fx_flips(
        eager_routes: &[Vec<RouteTap>],
        ik_routes: &[IkRoute],
    ) -> Result<Vec<Flip>, GateError> {
        let mut flips = Vec::new();
        for (l, ik) in ik_routes.iter().enumerate() {
            if ik.tokens() != eager_routes.len() {
                return Err(format!(
                    "layer {l}: ik routes {} tokens, the run {}",
                    ik.tokens(),
                    eager_routes.len()
                )
                .into());
            }
            for (t, routes) in eager_routes.iter().enumerate() {
                let ours = routes
                    .get(l)
                    .ok_or_else(|| format!("position {t}: no route tap for layer {l}"))?;
                flips.extend(flip_at(l, t, ours, ik));
            }
        }
        Ok(flips)
    }

    /// Whether a flip lies on an earlier flip's path: a flip at an earlier layer at this token or
    /// before.
    fn after_flip(f: &Flip, flips: &[Flip]) -> bool {
        flips
            .iter()
            .any(|g| g.layer < f.layer && g.token <= f.token)
    }

    /// The line of a flip no rule holds: its pairs, the cap at its layer and why it is not
    /// held.
    fn unheld_flip_line(arm: &str, f: &Flip, ik_routes: &[IkRoute], bands: &IkBands) -> String {
        let (row, _) = ik_routes[f.layer].at(f.token);
        let cap = flip::flip_cap(bands.router[f.layer], row);
        let pairs: Vec<String> = f
            .pairs
            .iter()
            .map(|(a, b, gap, err)| format!("{a}<-{b} gap {gap:.3e} err {err:.3e}"))
            .collect();
        format!(
            "{arm} flip layer={} token={}: ik margin {:.3e}; {} (cap {cap:.3e}): on a path no \
             band covers, not held",
            f.layer,
            f.token,
            f.margin,
            pairs.join(", ")
        )
    }

    /// Each flip, printed, and how many are held and not allowed. A flip with no earlier flip on
    /// its path reads a router input that carries rounding alone, so it is allowed only within
    /// the pair cap ([`flip::flip_cap`]) and, at its gap, the margin bound ([`flip::margin_cap`]),
    /// both at its layer's router band; past an earlier flip the input also carries that flip's
    /// other experts, which no band covers, so it is printed and counted.
    fn judge_fx_flips(
        arm: &str,
        flips: &[Flip],
        ik_routes: &[IkRoute],
        bands: &IkBands,
    ) -> (usize, usize) {
        let (mut first, mut refused) = (0usize, 0usize);
        for f in flips {
            if after_flip(f, flips) {
                println!("{}", unheld_flip_line(arm, f, ik_routes, bands));
                continue;
            }
            let (row, _) = ik_routes[f.layer].at(f.token);
            let band = bands.router[f.layer];
            let (cap, bound) = (flip::flip_cap(band, row), flip::margin_cap(band, row));
            let widest = f
                .pairs
                .iter()
                .map(|p| p.2)
                .fold(f64::NEG_INFINITY, f64::max);
            let bad = !f.allowed(cap) || widest > bound || widest.is_nan();
            first += 1;
            refused += usize::from(bad);
            println!(
                "{}; first on its path: widest gap {widest:.3e} (margin bound {bound:.3e}) {}",
                f.line(arm, cap),
                if bad {
                    "FAIL: not allowed at its layer's band"
                } else {
                    "within"
                }
            );
        }
        (first, refused)
    }

    /// (cfx): (c) free on the fixture, against ik's batch set dumped on it. Each layer's output
    /// off every flip's path within [`IkBands::stream`] of the layer ([`ik_bands`] with our
    /// dense projections on the gemv: the eager steps), the outputs past a flip printed and
    /// counted; each flip with no earlier flip on its path allowed under the pair cap and the
    /// margin bound at its layer's router band ([`judge_fx_flips`]); the last argmax a tie
    /// ([`flip::head_tie`]). Returns the verdict and by position the first layer a flip lies on
    /// the path of (`n_layer` where none does; empty where the tier leaves the clause), which
    /// (tfx)'s 4-token sets read as their band's layer.
    fn free_fx(eager: &Run, vocab: usize) -> Result<(bool, Vec<usize>), GateError> {
        const CLAUSE: &str = "(cfx) free: each layer's output off every flip's path, each flip and \
                              the last argmax against ik's fixture set";
        if !tier::run_clause(CLAUSE, Tag::FixtureOracle)? {
            return Ok((true, Vec::new()));
        }
        let man = fx_set(CLAUSE, fixture::BATCH)?;
        let bands = fx_bands(Arm::Gemv)?;
        let table = layer_table(&man, &eager.taps, STREAMS * HIDDEN, n_layer())?;
        let ik_routes = (0..n_layer())
            .map(|l| IkRoute::read(&man, l))
            .collect::<Result<Vec<_>, _>>()?;
        let flips = fx_flips(&eager.routes, &ik_routes)?;
        let (first, refused) = judge_fx_flips("fx free", &flips, &ik_routes, bands);
        let firsts = first_flip_layers(&flips, eager.taps.len(), n_layer());
        let (mut worst, mut past, mut held) = ((0.0f64, 0usize, 0usize), Vec::new(), 0usize);
        let mut exempt = 0usize;
        for (l, row) in table.iter().enumerate() {
            let mut cells = Vec::new();
            for (t, e) in row.iter().enumerate() {
                let Some(e) = *e else {
                    cells.push("-".to_string());
                    continue;
                };
                let frac = e / bands.stream[l];
                if on_path(&flips, l, t, false) {
                    exempt += 1;
                    cells.push(format!("{frac:.3}*"));
                    continue;
                }
                held += 1;
                if frac > worst.0 {
                    worst = (frac, l, t);
                }
                if !within(e, bands.stream[l]) {
                    past.push((l, t));
                }
                cells.push(format!("{frac:.3}"));
            }
            println!(
                "fx free table layer={l} l_out_rel over its band {:.3e} by position {} (* past a \
                 flip: printed)",
                bands.stream[l],
                cells.join(" ")
            );
        }
        let layers_ok = past.is_empty() && held > 0;
        let ik = ik_last(&man, vocab)?;
        let ours = eager.logits.last().ok_or("no logits")?;
        let top = argmax(ours);
        let tie = flip::head_tie(&ik, top, bands.head);
        let ok = layers_ok && refused == 0 && tie;
        println!(
            "fx free: {} tokens, {} flips ({first} first on their path, {refused} of those not \
             allowed); {held} (layer, position) outputs off every flip's path, worst {:.3} of its \
             band at layer {} position {}, past it at {past:?}, {exempt} past a flip printed; \
             first layer a flip lies on the path of, by position {firsts:?}; last argmax ours={top} \
             ik={}, ik's logit at ours {:.3e} below its top (tie cap {:.3e} at head band \
             {:.3e}); logits_rel {:.3e} (printed) {}",
            eager.tokens.len(),
            flips.len(),
            worst.0,
            worst.1,
            worst.2,
            argmax(&ik),
            flip::head_gap(&ik, top),
            flip::head_cap(bands.head, &ik),
            bands.head,
            rel(ours, &ik),
            verdict(ok)
        );
        Ok((ok, firsts))
    }

    /// (crx): the ubatch walk of the batch set with ik's own routing planted at every layer and
    /// position — ik's `ffn_moe_topk-L` and final weights `ffn_moe_weights_norm-L`
    /// ([`planted_route`]) — so no flip can happen and the error model alone separates us from
    /// ik. Held, against ik's batch set dumped on the fixture, at [`ik_bands`] with our dense
    /// projections on the GEMM: every planted id taken (as sets, ik's order is not the launch's);
    /// at every layer our router's logits, which read the stream after that layer's mixer and so
    /// carry every layer before it, within the layer's router band ([`IkBands::router`], the
    /// ubatch walk taps no layer's output); the last layer's output within its band; the last
    /// logits within the head band; the last argmax a tie ([`flip::head_tie`]).
    fn forced_fx(m: &mut Qwen38Model, toks: &[u32]) -> Result<bool, GateError> {
        const CLAUSE: &str = "(crx) ik-routed: ik's picks and final weights planted, each layer's \
                              router logits, the last layer's output, the logits and the argmax \
                              against ik's fixture set";
        if !tier::run_clause(CLAUSE, Tag::FixtureOracle)? {
            return Ok(true);
        }
        let man = fx_set(CLAUSE, fixture::BATCH)?;
        let bands = fx_bands(Arm::Gemm)?;
        let n = toks.len();
        let mut ik_routes = Vec::with_capacity(n_layer());
        let mut plant: Vec<Vec<RouteTap>> = (0..n).map(|_| Vec::with_capacity(n_layer())).collect();
        for l in 0..n_layer() {
            let ik = IkRoute::read(&man, l)?;
            let w = tap(&man, &format!("ffn_moe_weights_norm-{l}"))?;
            if ik.tokens() != n || w.len() != n * N_USED {
                return Err(format!(
                    "layer {l}: ik routes {} tokens and {} final weights, the run {n} tokens of \
                     {N_USED}",
                    ik.tokens(),
                    w.len()
                )
                .into());
            }
            for (t, row) in plant.iter_mut().enumerate() {
                row.push(planted_route(
                    &ik_picks(&ik, t),
                    &w[t * N_USED..(t + 1) * N_USED],
                    (l, t),
                )?);
            }
            ik_routes.push(ik);
        }
        let walk = gemm_walk(m, toks, Some(&plant));
        gemm_taps_off(m)?;
        let (run, routes) = walk?;
        let streams = m.target_streams(TargetRows::Ubatch, n)?;
        let mut taken = 0usize;
        let mut other = Vec::new();
        for (t, ours) in routes.iter().enumerate() {
            for (l, o) in ours.iter().enumerate() {
                let (mut got, mut want) = (o.ids.clone(), ik_picks(&ik_routes[l], t));
                got.sort_unstable();
                want.sort_unstable();
                if got == want {
                    taken += 1;
                } else {
                    other.push((l, t));
                }
            }
        }
        let cells = n * n_layer();
        let ids_ok = routes.len() == n && taken == cells;
        println!(
            "fx ik-routed: ik's picks taken at {taken} of {cells} (layer, position) cells, others \
             {:?} {}",
            &other[..other.len().min(8)],
            verdict(ids_ok)
        );
        let mut ok = ids_ok;
        let mut past = Vec::new();
        for l in 0..n_layer() {
            let worst = (0..n)
                .map(|t| {
                    routes
                        .get(t)
                        .and_then(|r| r.get(l))
                        .map_or(f64::INFINITY, |o| rel(&o.logits, ik_routes[l].at(t).0))
                })
                .fold(0.0f64, f64::max);
            let band = bands.router[l];
            if !within(worst, band) {
                past.push(l);
            }
            println!(
                "fx ik-routed: layer {l} router logits' distance from ik's, worst over positions \
                 {worst:.3e} (band {band:.3e}, {:.3} of it)",
                worst / band
            );
        }
        let routers_ok = past.is_empty();
        ok &= routers_ok;
        println!(
            "fx ik-routed: every layer's router logits within its band, past it at {past:?} {}",
            verdict(routers_ok)
        );
        let last = n_layer() - 1;
        let width = STREAMS * HIDDEN;
        let ik_out = tap(&man, &format!("l_out-{last}"))?;
        let out = (0..n)
            .map(|t| {
                match (
                    streams.get(t * width..(t + 1) * width),
                    ik_out.get(t * width..(t + 1) * width),
                ) {
                    (Some(a), Some(b)) => rel(a, b),
                    _ => f64::INFINITY,
                }
            })
            .fold(0.0f64, f64::max);
        let out_ok = out <= bands.stream[last];
        ok &= out_ok;
        println!(
            "fx ik-routed: layer {last} output's distance from ik's, worst over positions \
             {out:.3e} (band {:.3e}, {:.3} of it) {}",
            bands.stream[last],
            out / bands.stream[last],
            verdict(out_ok)
        );
        let vocab = m.body("forced_fx")?.vocab();
        let ik = ik_last(&man, vocab)?;
        let ours = run.logits.first().ok_or("no logits")?;
        let top = argmax(ours);
        let logits_rel = rel(ours, &ik);
        let logits_ok = logits_rel <= bands.head;
        let tie = flip::head_tie(&ik, top, bands.head);
        ok &= logits_ok && tie;
        println!(
            "fx ik-routed: last logits' distance from ik's {logits_rel:.3e} (head band {:.3e}, \
             {:.3} of it); last argmax ours={top} (returned {}) ik={}, ik's logit at ours {:.3e} \
             below its top (tie cap {:.3e}) {}",
            bands.head,
            logits_rel / bands.head,
            run.tokens[0],
            argmax(&ik),
            flip::head_gap(&ik, top),
            flip::head_cap(bands.head, &ik),
            verdict(logits_ok && tie && run.tokens[0] == top)
        );
        ok &= run.tokens[0] == top;
        Ok(ok)
    }

    /// How a (tfx) step set's prefill is fed.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum FxFeed {
        /// One captured step a position, from a reset: our routing.
        Steps,
        /// Ubatches from a reset with ik's prefill routes planted ([`prefill_plant`]).
        PlantedUbatches,
    }

    impl FxFeed {
        fn name(self) -> &'static str {
            match self {
                FxFeed::Steps => "steps",
                FxFeed::PlantedUbatches => "ubatches with ik's prefill routes planted",
            }
        }

        /// Our dense projections' launch arm: the ubatch walk's stores carry the GEMM's rounding.
        fn arm(self) -> Arm {
            match self {
                FxFeed::Steps => Arm::Gemv,
                FxFeed::PlantedUbatches => Arm::Gemm,
            }
        }
    }

    /// (tfx): step set `set` of the fixture, its prefill fed by `feed` from a reset, then the
    /// step eagerly with the taps armed, against ik's set dumped on the fixture. Held:
    /// - each layer's output below the first layer a flip lies on the path of
    ///   ([`IkBands::stream`] of the layer): the step's own flips, found against ik's step
    ///   routes in the set, and the prefill's. The prefill carries none where ik's routes are
    ///   planted; for the 4-token sets fed by steps (`band`: the batch set's tokens and the first
    ///   flip layer of its last position, (cfx)'s) it carries the batch set's; for a long set fed
    ///   by steps it carries a flip at every layer from 1 on at some position (a layer-0 flip
    ///   alters only layer 0's output at its position: layer 0's stores read the embeddings),
    ///   so layer 0, whose delta state is a long recurrence over the embeddings, is the one
    ///   held;
    /// - the last argmax a tie ([`flip::head_tie`]), always: its cap is wide against a flip's
    ///   distance;
    /// - the last logits within the head band where no flip lies on the path of the last layer;
    ///   printed otherwise.
    ///
    /// D3K's step reads a selection (751 pools past 512): ik keeps up to three keys of the 513th
    /// pool that ours does not (the difference q38sel measured), a few keys among 2,051 in one
    /// softmax, far inside the band.
    fn step_fx(
        m: &mut Qwen38Model,
        set: &str,
        feed: FxFeed,
        band: Option<(&[u32], usize)>,
    ) -> Result<bool, GateError> {
        let clause = format!(
            "(tfx) step set {set} fed by {}: layer outputs below the first flip, logits and argmax \
             against ik's fixture set",
            feed.name()
        );
        if !tier::run_clause(&clause, Tag::FixtureOracle)? {
            return Ok(true);
        }
        tier::fixture_set(&clause, &fixture::IK, set)?;
        let (man, pos, tok, prefill, first_batch) = set_open((set, &fixture::IK), band)?;
        let bands = fx_bands(feed.arm())?;
        fresh(m)?;
        let t = Instant::now();
        match feed {
            FxFeed::Steps => {
                if !prefill.is_empty() {
                    m.prompt38(&prefill, Prompt38::Step)?;
                }
            }
            FxFeed::PlantedUbatches => {
                let plant = prefill_plant(&man, prefill.len())?;
                {
                    let (gpu, _, b) = m.body_parts("step_fx")?;
                    b.plant_ubatch_routes(gpu, Some((0, &plant)))?;
                }
                let walked = m.prompt38(&prefill, Prompt38::Gemm);
                gemm_taps_off(m)?;
                walked?;
            }
        }
        let fed = t.elapsed().as_secs_f64();
        let r = run_last(m, tok)?;
        let ik_routes = (0..n_layer())
            .map(|l| IkRoute::read(&man, l))
            .collect::<Result<Vec<_>, _>>()?;
        let flips = fx_flips(&r.routes, &ik_routes)?;
        let step_first = flips.iter().map(|f| f.layer).min().unwrap_or(n_layer());
        let prefill_first = match (feed, band) {
            (FxFeed::PlantedUbatches, _) => n_layer(),
            (FxFeed::Steps, Some(_)) => first_batch,
            (FxFeed::Steps, None) => 1.min(n_layer()),
        };
        let held = prefill_first.min(step_first);
        let arm = format!("step {set}");
        let refused = if feed == FxFeed::PlantedUbatches {
            judge_fx_flips(&arm, &flips, &ik_routes, bands).1
        } else {
            for f in &flips {
                println!("{}", unheld_flip_line(&arm, f, &ik_routes, bands));
            }
            0
        };
        let rels = layer_rels(&man, &r.taps, STREAMS * HIDDEN, n_layer())?;
        let mut past = Vec::new();
        for (l, &(e, _)) in rels.iter().enumerate() {
            let on = l < held;
            if on && !within(e, bands.stream[l]) {
                past.push(l);
            }
            println!(
                "step {set} layer={l} l_out_rel={e:.3e} (band {:.3e}, {:.3} of it{})",
                bands.stream[l],
                e / bands.stream[l],
                if on { ", held" } else { ", on a flip's path: printed" }
            );
        }
        let layers_ok = past.is_empty() && refused == 0;
        let vocab = m.body("step_fx")?.vocab();
        let ik = ik_last(&man, vocab)?;
        let ours = r.logits.last().ok_or("no logits")?;
        let top = argmax(ours);
        let tie = flip::head_tie(&ik, top, bands.head);
        let logits_rel = rel(ours, &ik);
        let logits_ok = held < n_layer() || logits_rel <= bands.head;
        let ok = layers_ok && tie && logits_ok;
        println!(
            "step {set}: position {pos} after {} fed by {} ({fed:.1} s, runtime value); {} flips at \
             the step, first on its path at layer {step_first} (the prefill's {prefill_first}), \
             layers 0..{held} held, past the band at {past:?}; argmax ours={top} ik={}, ik's logit at \
             ours {:.3e} below its top (tie cap {:.3e} at head band {:.3e}); logits_rel \
             {logits_rel:.3e} ({}) {}",
            prefill.len(),
            feed.name(),
            flips.len(),
            argmax(&ik),
            flip::head_gap(&ik, top),
            flip::head_cap(bands.head, &ik),
            bands.head,
            if held < n_layer() {
                "printed: a flip lies on the path"
            } else {
                "held"
            },
            verdict(ok)
        );
        Ok(ok)
    }

