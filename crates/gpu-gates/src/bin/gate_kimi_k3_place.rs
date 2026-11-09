//! GPU gate for Kimi K3's placed load (`--place bp`): the stage card holds the dense trunk and each
//! routed layer's first experts, the second card is the expert tier (`LegTier`, in process or
//! served by a loopback `tier_server` through [`bloomery_gpu::arch::kimi_k3::NODE_FAMILY`]), the
//! embedding is a host row, and a plan's share the cards cannot hold is the host tier's (SiTU
//! combined by `model::ops::ExpertForm::Situ`). The fixture tier only: the plans are found by
//! card budgets over a fixture's experts, and the real file's run is
//! `generate_kimi_k3 --place bp`'s.
//!
//! The plan is made on the gate's `bp` machine (the A6000 the stage, the 3090 the tier, each
//! counting its uploads at [`GRANULE`]: the fixture's experts, 38 KB each, fit the shared granules
//! the dense trunk leaves open at the card's 2 MiB, so only a finer granule gives a card budget a
//! window; every child the gate runs gets the same `--granule`). The window is found by bisection
//! from the header's budget: from the first budget that plans (the stage's floor, every expert on
//! the tier) to the first that leaves the tier idle (every expert on the stage); the plans between
//! move one expert at a time from the tier to the stage, and the gate takes the first budget with
//! experts on both cards and none on the host.
//!
//! Clauses, each decided by `tier::sc` (every one runs in the tier the gate has):
//!
//! - `precondition tier`: the plan holds experts on both cards and none on the host, and the load
//!   holds what the plan says per layer (`Placed::n_card`, `Placed::tier_k`).
//! - `placed ids`: the placed load's greedy ids after the prompt equal the whole-card body's
//!   (`GpuModel::open`) on the same ids. The routed sum is `(card + tier) + host` where the whole
//!   card sums every slot in one pass, so the rows may differ by rounding (printed as a relative
//!   L2, not pinned) and the clause is on the argmax, as the e2e gate holds the free chain.
//! - `placed ids, stage without experts`: the same on the first budget of the window, whose plan
//!   holds every expert on the tier card and none on the stage or the host, the real file's shape
//!   (every layer a hosted layer with no card expert).
//! - `placed eager`: the same load stepped eagerly after a reset reads the graph run's ids and
//!   logits bits.
//! - `precondition sent`: every tier layer was sent a routed slot, so the tier's legs ran.
//! - `node ids`: the tier served by a loopback `tier_server` on the 3090 alone
//!   (`BLOOMERY_TIER_NODES`) through `generate_kimi_k3 --place bp` reads the in-process tier's ids
//!   and the same logits bits (an FNV-1a over the last row's bits).
//! - `reserve`: a plan made with a tier reserve of 0 where the load makes tier places is refused
//!   by name at the load (`tier places`).
//! - `host head`: a plan whose stage card carries no head (its head tensors left on the host, read
//!   in place) loads, and fed the whole-card body's ids each step's logits row is within
//!   [`HOST_HEAD_REL_BAND`] relative L2 of the same plan's load with the head on the card: the
//!   head's bytes leave the stage card's budget, so the experts split alike and the head is the
//!   one difference (the host's activation format and dot against the card's q8_1 gemv).
//! - `host head fault`: a fault planted on the card is the host head's readback error, not a token.
//! - `host head refused`: a plan with no head card whose head tensors are not left on the host
//!   is refused by name at the load.
//! - `host share`: a plan whose tier card is budgeted under what the experts need (per-card
//!   budgets: the stage at the window's first, the tier below) leaves experts on the host; the
//!   load holds the plan's counts per layer, and fed the whole-card body's ids each step's logits
//!   row is within [`HOST_REL_BAND`] relative L2 of the whole-card row (the host's SiTU against
//!   the card's rule). The free-running ids are not the clause: a flat fixture row flips its
//!   argmax on a rounding, and the rows after that are another history's.
//! - `host layer without tier`: a plan where a layer holds host experts and no tier expert is
//!   refused by name at the load (the host handoff's entries are not built for K3's slot count).
//! - `node family`: a node that serves only another family refuses the open by name.
//! - `plan records`: `generate_kimi_k3 --plan --place bp` prints the `plan` record of the placed
//!   machine and no host expert (the gate's plan holds none).
//!
//! The placed prompt pass (`GpuModel::open_placed_pass`), each clause per pinned [`PASS_SHAPES`]
//! (16 rows, the latent mixers' inner run at 8 and at 1), on a plan whose prompt reserve is the
//! pass's bytes, whose tier batch holds its rows and whose per-card budgets add each to the
//! gate's, over [`PASS_PROMPT`] ids: two passes, the first across the delta layers' chunks. Each
//! compares the same load stepped after a reset, so the expert split is one and the bits are the
//! clause:
//!
//! - `pass state`: the prompt as passes, then [`STEPS`] greedy ids, reads the stepped prompt's
//!   ids and every logits row's bits: each row is the step's and the state the passes leave is
//!   the steps'. Mutant: the close's add over every row but the pass's last (that row's sum and
//!   every later row off).
//! - `pass batches`: the tier's hits over the prompt are the steps' (each row's slots sent once)
//!   and its batches served one per tier layer per pass. The NVMe tier's one read of an expert a
//!   pass is the real file's (the fixture pages nothing): `generate_kimi_k3 --walk pass` on it.
//!   Mutant: a pass per row (`pass state` stays green, `pass batches` red).
//! - `pass score`: `GpuModel::score_placed` over the prompt and the greedy ids but the last reads
//!   the stepped rows' bits at each of the last [`STEPS`] rows (the card head, a row at a time).
//!   Mutant: the window one row early.
//! - `pass refused`: a shape out of range at the load, `prompt_placed` on a load without the
//!   pass, `score_placed` past the pass's rows and a plan whose prompt reserve does not hold the
//!   pass are refused by name. Mutant: the load's reserve check without the pass's bytes.
//! - `pass score, host head`: the same on a plan whose head the host reads, its rows through the
//!   host's projection of up to `HOST_HEAD_ROWS` rows at once (one `matmul_q` of [`STEPS`]
//!   columns against the step's one-column calls). Mutant: the projection's second row a copy of
//!   its first.
//! - `pass cli`: `generate_kimi_k3 --walk pass --plan` prints the `prompt_pass` record of the
//!   widest shape at the gate's budget for it, and the run's ids and last logits row are the
//!   same plan's stepped load's; `--score --walk pass` over the stepped ids prints a line a
//!   completion position whose top id is the stepped load's argmax there.
//!
//! - `pass past top-k`: the fixture made a K3.1 file at a top-k of 16 (seeded indexer tensors on
//!   its latent layer), at a context of 64 (the load selects): a prompt of 24 ids in two calls, the
//!   first of [`DSA_SPLIT`] ids, so the second's first pass runs rows on both sides of the top-k,
//!   then [`STEPS`] greedy ids, reads the same ids stepped from a reset: the token, the position,
//!   every logits row's bits (after the prompt and after each id fed back, so the key rows, the
//!   caches and the delta stores the passes leave are the steps'). Once on the whole-card body
//!   (`GpuModel::prompt_rows`, passes of eight) and once per pinned [`PASS_SHAPES`] on the placed
//!   pass load through `Batch::prompt`. Mutant: the pass attends every key past top-k (the dense
//!   rows on a load that selects): the rows past it read other rows than the step's.
//!
//! - `room`: the stores' rooms of that whole-card K3.1 body (a latent layer's cached rows and
//!   index key rows, the buffers `Hparams::pos_bufs` declares) map no page on a fresh load, the
//!   pages of the rows the prompt and the steps reached on the card alone (`room_mapped`: some,
//!   none on the host), the same pages after a first reset (a settle keeps the pages the
//!   sequence it ends reached, `Room::settle`; V4.1's `Seq::clear` and Qwen3.8's reset are the
//!   same call), and none after a second reset (it ends an empty sequence). Mutants
//!   (`.mutants/kvc-k31-*.patch`): `reset` leaves the pages mapped (the second reset keeps them), the passes and the step fit no row (the pages never map, the rows
//!   read the spare).
//!
//! - `kvspill`: the plan of the K3.1 fixture under `BLOOMERY_KV_HOST=spill` with a floor counts
//!   the host-able cached rows' pages past the floor against its host room, the key rows never,
//!   as the stage card's host page rounds them: latent layers x (pages of the cached run at the
//!   context - pages at the floor) x the page [derived here from the header's widths, not read
//!   from the plan's owner]. Mutants (`.mutants/kvc-k31-*.patch`): the host count left out of
//!   the plan, the floor's pages not taken off.
//!
//! The prompt is the fixed walk of `e2e::walk` through the vocabulary: [`PROMPT`] ids, [`STEPS`]
//! greedy ids.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    if std::env::args().nth(1).as_deref() == Some(gate::SERVE_OTHER) {
        return bloomery_gpu_gates::exit_with("gate_kimi_k3_place node", gate::serve_other());
    }
    bloomery_gpu_gates::exit_with("gate_kimi_k3_place", gate::run())
}

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the shared harness serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/node_proc.rs"]
mod node_proc;

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/tier_twocard.rs"]
mod tier_twocard;

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the shared window finder serves several gates; each takes the helpers it reads"
)]
#[path = "shared/kimi_k3_budget.rs"]
mod kimi_k3_budget;

#[cfg(feature = "gpu")]
mod gate {
    use std::net::TcpListener;
    use std::process::Command;
    use std::sync::Arc;

    use app::Prompt;
    use bloomery_gpu::arch::kimi_k3::{Body, CHUNK, Pass, PassShape, SHAPES, pass_bytes};
    use bloomery_gpu::host::node;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Fault, FaultSite, Gpu, GpuError, GpuModel};
    use bloomery_gpu_gates::generate::{Place, stage_host_page};
    use bloomery_gpu_gates::tier::{self, Tier};
    use bloomery_gpu_gates::{
        GateError, beside, checks_failed, die_with_parent, fnv_row, open_split, ref_model_path,
    };
    use bloomery_levers::{
        CARD_BUDGET, CARD_DONTNEED, CardBudget, HOST_LOCK, HOST_POPULATE, HostCfg, KvHost,
        TIER_NODES,
    };
    use gguf::Split;
    use model::arch::Arch;
    use model::arch::kimi_k3::place::{PlanInputs, tier_batch, tier_batch_at};
    use model::placement::{Device, Machine, Plan, PlanLevers, Role, workstation};

    use crate::e2e;
    use crate::kimi_k3_budget::{
        Budget, Cut, GRANULE, bp_machine, bp_machine_at, cut, layer_splits, levers_of, pick_budget,
        pick_tier_only, split_table, totals, under,
    };
    use crate::node_proc::{LISTENING, tier_server, tier_server_command};
    use crate::tier_twocard::{Sent, sent_between, sent_every, tier_counts};

    /// The gate's name: the recipe's and the refusals'.
    const NAME: &str = "gate_kimi_k3_place";
    /// The recipe that runs it.
    const RECIPE: &str = "gate-gpu-kimik3-place";
    /// The argument that makes this binary a tier node serving another family.
    pub const SERVE_OTHER: &str = "--serve-other-family";

    /// Cache rows: past the prompt and the generated ids.
    const CTX: usize = 128;
    /// Prompt positions, one decode step each.
    const PROMPT: usize = 6;
    /// Greedy ids after the prompt.
    const STEPS: usize = 8;
    /// Prompt positions of the pass clauses: two passes of [`PASS_SHAPES`]' rows, the first
    /// across the delta layers' chunks and the second shorter than one.
    const PASS_PROMPT: usize = 2 * CHUNK + 3;
    /// The pass shapes the pass clauses pin: 16 rows, the latent mixers' inner run at
    /// [`CHUNK`] rows and at one.
    const PASS_SHAPES: [PassShape; 2] = [
        PassShape {
            rows: 16,
            mla_rows: CHUNK,
        },
        PassShape {
            rows: 16,
            mla_rows: 1,
        },
    ];

    /// The worst step's logits row's relative L2 distance of a load with host experts from the
    /// whole-card body's, both fed the same ids. The experts are the one difference: the host
    /// rounds both inputs of a routed expert in q8_K blocks of 256 values, the card in q8_1 blocks
    /// of 128, and `gate_hybrid`'s error model prices each rounding as `e = crest / (127 √12)` of
    /// an int8 block of crest `amax / rms`:
    /// - the latent row x (256 values, Gaussian, crest 3.0 over 256 and 2.8 over 128): `e` 6.9e-3
    ///   and 6.4e-3, the gate and the up each carrying both;
    /// - the SiTU product h (256 values, a heavier tail, crest 6.4 and 5.5 on simulated Gaussian
    ///   rows): `e` 1.4e-2 and 1.3e-2, the down carrying both;
    /// - the weights add no term: both paths read the same codes and sub-scales of IQ1_S and
    ///   IQ2_XXS, the block dot is an exact i32 sum, and only the f32 order of the scale products
    ///   differs (about 1e-7).
    ///
    /// One host expert then differs by `r² = 2 (e₂₅₆(x)² + e₁₂₈(x)²) + e₂₅₆(h)² + e₁₂₈(h)²`, so
    /// `r` = 2.3e-2. The plan holds 18 of its 24 routed experts on the host (a factor √(18/24) =
    /// 0.87 on a layer's branch) over 3 routed layers (√3, independent): 2.3e-2 · 0.87 · 1.73 =
    /// 3.5e-2, with the routed branch taken as the whole row. That sum is the rounding floor. It
    /// leaves out what the row passes on the way to the logits (the routed norm, the latent up,
    /// the cross-layer mix, the head) and the router's top-2 pick on rows that now differ, and
    /// the worst observed step is above it: the band is twice the observed step, not a multiple
    /// of the floor.
    /// PIN(2026-11-06): twice the worst measured step (8.16e-2 of 1.8e-2 to 8.2e-2 over the eight
    /// steps); the swapped betas read 5.8e-1.
    const HOST_REL_BAND: f64 = 1.6e-1;

    /// The worst step's logits row's relative L2 distance of a load whose head the host reads from
    /// the same plan's load with the head on the card, both fed the same ids over the same expert
    /// split: only the final norm's and the projection's activation format and dot order differ
    /// (the host's q8_2 and fused dot against the card's q8_1 gemv).
    /// PIN(2026-11-06): twice the worst measured step (7.73e-3, inside the 3e-3 to 2e-2 derived
    /// from two int8 activation roundings of one normed row); the final norm skipped reads 5.3e-1.
    const HOST_HEAD_REL_BAND: f64 = 1.6e-2;

    /// What a run read: the greedy ids and the last logits row.
    struct Run {
        ids: Vec<u32>,
        /// The last logits row.
        logits: Vec<f32>,
        /// The logits row after the prompt and after each id fed back: `ids.len()` rows.
        rows: Vec<Vec<f32>>,
    }

    impl Run {
        fn same_bits(&self, other: &Run) -> bool {
            self.ids == other.ids && fnv_row(&self.logits) == fnv_row(&other.logits)
        }

        /// The same ids and every logits row's bits.
        fn same_rows(&self, other: &Run) -> bool {
            self.ids == other.ids
                && self.rows.len() == other.rows.len()
                && self
                    .rows
                    .iter()
                    .zip(&other.rows)
                    .all(|(a, b)| fnv_row(a) == fnv_row(b))
        }
    }

    /// The prompt, then [`STEPS`] ids greedy, the last logits row and every step's.
    fn greedy(m: &mut GpuModel<Body>, prompt: &[u32]) -> Result<Run, GateError> {
        let mut ids = vec![Body::prompt(m, prompt)?];
        let mut rows = vec![m.logits()?];
        while ids.len() < STEPS {
            let last = ids[ids.len() - 1];
            ids.push(m.step(&[last])?);
            rows.push(m.logits()?);
        }
        Ok(Run {
            ids,
            logits: m.logits()?,
            rows,
        })
    }

    /// The prompt, then `ids` fed back one a step (all but the last): the logits row after the
    /// prompt and after each id, the rows a free-running load would read had it chosen `ids`.
    fn forced(
        m: &mut GpuModel<Body>,
        prompt: &[u32],
        ids: &[u32],
    ) -> Result<Vec<Vec<f32>>, GateError> {
        Body::prompt(m, prompt)?;
        let mut rows = vec![m.logits()?];
        for &id in &ids[..ids.len() - 1] {
            m.step(&[id])?;
            rows.push(m.logits()?);
        }
        Ok(rows)
    }

    /// The relative L2 distance of row `b` from row `a`.
    fn rel(a: &[f32], b: &[f32]) -> f64 {
        let (mut num, mut den) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            let (x, y) = (f64::from(*x), f64::from(*y));
            num += (x - y) * (x - y);
            den += x * x;
        }
        (num / den).sqrt()
    }

    /// One clause's verdict line.
    fn line(ok: bool, clause: &str, detail: &str) -> bool {
        println!("{} {clause}: {detail}", if ok { "ok" } else { "FAIL" });
        ok
    }

    /// A clause group's verdict: an error it returns is a `FAIL` line with the message, so the
    /// clauses after it still run and the gate ends by its tally.
    fn guarded(clause: &str, run: Result<bool, GateError>) -> bool {
        run.unwrap_or_else(|e| line(false, clause, &format!("the clause ended in an error: {e}")))
    }

    /// The prompt as placed passes (`pass`) or stepped, then [`STEPS`] ids greedy, every step's
    /// logits row, and the tier's counters over the prompt alone.
    fn walked(
        m: &mut GpuModel<Body>,
        prompt: &[u32],
        pass: bool,
    ) -> Result<(Run, Option<Sent>), GateError> {
        let counts = |m: &GpuModel<Body>| -> Result<Option<Sent>, GateError> {
            Ok(tier_counts(
                m.body(NAME)?.placed().ok_or("not a placed load")?.hybrid(),
            ))
        };
        let before = counts(m)?;
        let first = if pass {
            m.prompt_placed(prompt)?
        } else {
            m.step(prompt)?
        };
        let sent = sent_between(before, counts(m)?);
        let mut ids = vec![first];
        let mut rows = vec![m.logits()?];
        while ids.len() < STEPS {
            let last = ids[ids.len() - 1];
            ids.push(m.step(&[last])?);
            rows.push(m.logits()?);
        }
        Ok((
            Run {
                ids,
                logits: m.logits()?,
                rows,
            },
            sent,
        ))
    }

    /// A plan that leaves experts on the host: its levers and the experts each tier holds.
    struct HostShare {
        levers: PlanLevers,
        stage: u64,
        tier: u64,
        host: u64,
    }

    /// The kind of host share a plan makes.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Share {
        /// Every layer with host experts holds tier experts too.
        Tiered,
        /// Some layer holds host experts and no tier expert.
        Tierless,
    }

    /// The card budgets of a plan that leaves experts on the host: the stage card at `stage`, the
    /// tier card at the first budget above its floor that gives it some experts and the host the
    /// rest, and the split that makes, of the kind `share` names. None when no budget does.
    fn host_share_plan(
        inputs: &PlanInputs,
        stage: u64,
        share: Share,
    ) -> Result<Option<HostShare>, GateError> {
        let machine = bp_machine(inputs)?;
        let (Some(s), Some(t)) = (
            machine.cards.first().and_then(|c| c.device),
            machine.tiers.first().and_then(|c| c.device),
        ) else {
            return Err(
                "the gate's cards carry no census device, which a per-card budget names".into(),
            );
        };
        let of =
            |tier: u64| levers_of(CardBudget::Per(vec![(s.ordinal, stage), (t.ordinal, tier)]));
        if matches!(cut(inputs, &machine, CTX as u64, &of(stage)), Cut::Short(_)) {
            return Ok(None);
        }
        let (mut short, mut plans) = (0u64, stage);
        while plans - short > 1 {
            let mid = short + (plans - short) / 2;
            if matches!(cut(inputs, &machine, CTX as u64, &of(mid)), Cut::Short(_)) {
                short = mid;
            } else {
                plans = mid;
            }
        }
        let step = ((stage - plans) / 512).max(1);
        let mut b = plans;
        while b < stage {
            if let Ok(plan) = inputs.plan(&machine, CTX as u64, &of(b)) {
                let splits = layer_splits(&plan);
                let (stage_n, tier, host) = totals(&splits);
                let kind = if splits.iter().any(|&(_, _, t, h)| h > 0 && t == 0) {
                    Share::Tierless
                } else {
                    Share::Tiered
                };
                if tier > 0 && host > 0 && kind == share {
                    return Ok(Some(HostShare {
                        levers: of(b),
                        stage: stage_n,
                        tier,
                        host,
                    }));
                }
            }
            b += step;
        }
        Ok(None)
    }

    /// The whole-card body's run: the reference.
    fn whole(prompt: &[u32]) -> Result<Run, GateError> {
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let mut m = GpuModel::<Body>::open(Gpu::new()?, file, CTX)?;
        m.set_mode(StepMode::Graph);
        greedy(&mut m, prompt)
    }

    /// `generate_kimi_k3` over `prompt` on `--place bp` with the given environment: its status,
    /// stdout and stderr.
    fn generate(
        prompt: &[u32],
        env: &[(&str, String)],
        plan_only: bool,
    ) -> Result<(bool, String, String), GateError> {
        generate_with(prompt, env, plan_only, &[])
    }

    /// [`generate`] with the flags `extra`.
    fn generate_with(
        prompt: &[u32],
        env: &[(&str, String)],
        plan_only: bool,
        extra: &[&str],
    ) -> Result<(bool, String, String), GateError> {
        let bin = beside("generate_kimi_k3")?;
        let ids = prompt
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let mut cmd = Command::new(&bin);
        cmd.args(["--tokens", &ids, "-n", &STEPS.to_string()])
            .args(["--ctx", &CTX.to_string(), "--place", "bp"])
            .args(["--granule", &GRANULE.to_string()])
            .args(extra);
        if plan_only {
            cmd.arg("--plan");
        }
        for (k, v) in env {
            cmd.env(k, v);
        }
        die_with_parent(&mut cmd);
        let out = cmd
            .output()
            .map_err(|e| format!("{}: {e}", bin.display()))?;
        Ok((
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        ))
    }

    /// [`generate`] with `--walk pass`.
    fn generate_walk(
        prompt: &[u32],
        env: &[(&str, String)],
        plan_only: bool,
    ) -> Result<(bool, String, String), GateError> {
        generate_with(prompt, env, plan_only, &["--walk", "pass"])
    }

    /// The ids and the logits hash `generate_kimi_k3` printed.
    fn printed(stdout: &str) -> Result<(Vec<u32>, u64), GateError> {
        let ids = stdout
            .lines()
            .find_map(|l| l.strip_prefix("ids ["))
            .and_then(|l| l.strip_suffix(']'))
            .ok_or_else(|| format!("generate_kimi_k3 printed no ids line:\n{stdout}"))?
            .split(',')
            .map(|t| t.trim().parse::<u32>().map_err(GateError::from))
            .collect::<Result<Vec<_>, _>>()?;
        let hash = stdout
            .lines()
            .find_map(|l| l.strip_prefix("logits fnv1a64="))
            .and_then(|l| l.split_whitespace().next())
            .ok_or_else(|| format!("generate_kimi_k3 printed no logits line:\n{stdout}"))?;
        Ok((ids, u64::from_str_radix(hash, 16)?))
    }

    /// `CUDA_VISIBLE_DEVICES` of a node: the 3090 alone, by the UUID the driver reports for it.
    fn node_card() -> Result<String, GateError> {
        let census = bloomery_gpu::census()?;
        let device = census
            .iter()
            .find(|d| d.name.contains(workstation::RTX_3090.name))
            .ok_or_else(|| {
                format!(
                    "the node takes the {} and the census has none: {}",
                    workstation::RTX_3090.name,
                    census
                        .iter()
                        .map(|d| d.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;
        Ok(bloomery_gpu_gates::gpu_census::gpu_uuid(&device.uuid))
    }

    /// The error text of a load that must be refused; `None` if it opened.
    fn refused(r: Result<GpuModel<Body>, bloomery_gpu::GpuError>) -> Option<String> {
        r.err().map(|e| e.to_string())
    }

    /// This binary as a tier node serving the Qwen3.8 family alone over the file: what a K3
    /// stage's open is refused by. Prints the address it listens on.
    pub fn serve_other() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let path = ref_model_path()?;
        let file =
            Arc::new(Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?);
        let listener = TcpListener::bind("127.0.0.1:0")?;
        println!("{LISTENING}{}", listener.local_addr()?);
        std::io::Write::flush(&mut std::io::stdout())?;
        node::serve(
            &listener,
            &file,
            &[bloomery_gpu::arch::qwen3moe::NODE_FAMILY],
            |words| println!("{words}"),
        )?;
        Ok(())
    }

    pub fn run() -> Result<(), GateError> {
        if Tier::from_env()? != Tier::Fixture {
            return Err(format!(
                "{NAME}: the gate's plans split the fixture's experts by card budget, so it runs \
                 the fixture tier: BLOOMERY_TIER=fixture just {RECIPE}; the real file's run is \
                 generate_kimi_k3 --place bp"
            )
            .into());
        }
        let levers =
            bloomery_levers::at_main(&tier::acts_on(&[HOST_POPULATE, HOST_LOCK, CARD_DONTNEED])?)?;
        let split = open_split(Arch::KimiK3, RECIPE)?;
        let inputs = PlanInputs::read_without_chat(&split)?;
        let header = tier::plan_levers(&split, &levers, 0)?
            .card_budget_bytes
            .every()
            .ok_or("the fixture tier plans under the header's one budget")?;
        drop(split);
        if PROMPT + STEPS > CTX {
            return Err(format!("{PROMPT} + {STEPS} positions, the caches hold {CTX}").into());
        }
        let prompt = e2e::walk(PROMPT, inputs.hp.n_vocab);
        let table = split_table(&inputs, &bp_machine(&inputs)?, CTX as u64, header)?;
        let budget = pick_budget(&table)?;
        let host = levers.host();
        let want = whole(&prompt)?;
        println!("whole-card ids {:?}", want.ids);
        let mut pass = true;
        pass &= placed(&inputs, budget, host, (&prompt, &want))?;
        pass &= tier_only(&inputs, &table, host, (&prompt, &want))?;
        pass &= refusals(&inputs, budget, host)?;
        pass &= host_head(&inputs, (budget, host), (&prompt, &want))?;
        pass &= host_share(&inputs, &table, (budget, host), (&prompt, &want))?;
        pass &= host_without_tier(&inputs, &table, (budget, host))?;
        pass &= node_family_and_records(&prompt, budget)?;
        pass &= guarded("pass", placed_pass(&inputs, budget, host));
        pass &= guarded("pass cli", pass_cli(&inputs, budget, host));
        pass &= guarded("pass past top-k", pass_past_top_k(&inputs, header, host));
        println!("{NAME}: {}", tier::tally_line());
        if !pass {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} the placed load's greedy ids equal the whole-card body's, eager reads \
             the graph's bits, the loopback node's tier reads the in-process tier's bits, a plan's \
             host share reads the whole-card logits rows within the band, the placed pass reads \
             the steps' bits, and the load refuses a wrong tier reserve, a node of another family \
             and a pass the plan does not hold by name"
        );
        Ok(())
    }

    /// The placed load against the whole-card body, its eager run, and the tier on a node.
    fn placed(
        inputs: &PlanInputs,
        budget: u64,
        host: HostCfg,
        (prompt, want): (&[u32], &Run),
    ) -> Result<bool, GateError> {
        let machine = bp_machine(inputs)?;
        let plan = inputs.plan(&machine, CTX as u64, &under(budget))?;
        let splits = layer_splits(&plan);
        let (stage, tier_n, host_n) = totals(&splits);
        println!(
            "plan under a card budget of {budget} B at ctx {CTX}: stage {stage} experts, tier \
             {tier_n}, host {host_n}"
        );
        for (l, s, t, h) in &splits {
            println!("  layer {l}: stage {s}, tier {t}, host {h}");
        }
        let tier_layers: Vec<usize> = splits.iter().filter(|o| o.2 > 0).map(|o| o.0).collect();
        let mut pass = true;
        tier::sc("precondition tier: the plan holds experts on both cards and none on the host")?;

        let t = std::time::Instant::now();
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let mut m = GpuModel::<Body>::open_placed(file, &plan, inputs, 0, host)?;
        m.set_mode(StepMode::Graph);
        println!(
            "placed load in {:.1} s: resident_bytes={} layers={}",
            t.elapsed().as_secs_f64(),
            m.resident_bytes(),
            m.layers().len()
        );
        {
            let p = m
                .body(NAME)?
                .placed()
                .ok_or("the load is not a placed load")?;
            let wrong: Vec<_> = splits
                .iter()
                .filter(|&&(l, s, t, _)| p.n_card(l) as u64 != s || p.tier_k(l) as u64 != t)
                .map(|&(l, ..)| (l, p.n_card(l), p.tier_k(l)))
                .collect();
            pass &= line(
                wrong.is_empty() && !tier_layers.is_empty() && host_n == 0,
                "precondition tier",
                &format!(
                    "the tier holds experts on {} layers, the host none, and the load holds the \
                     plan's counts (layers off the plan: {wrong:?})",
                    tier_layers.len()
                ),
            );
        }

        let got = greedy(&mut m, prompt)?;
        tier::sc("placed ids: the placed load's greedy ids are the whole-card body's")?;
        pass &= line(
            got.ids == want.ids,
            "placed ids",
            &format!(
                "placed {:?}, whole-card {:?}, last logits row relative L2 {:.2e}",
                got.ids,
                want.ids,
                rel(&want.logits, &got.logits)
            ),
        );
        let sent = tier_counts(m.body(NAME)?.placed().ok_or("not placed")?.hybrid());
        tier::sc("precondition sent: every tier layer was sent a routed slot")?;
        pass &= sent_every("placed", sent.as_ref(), &tier_layers, false);
        pass &= stage_hit(&splits, inputs.hp.experts.n_used, sent.as_ref());

        m.reset()?;
        m.set_mode(StepMode::Eager);
        let eager = greedy(&mut m, prompt)?;
        tier::sc("placed eager: the eager run reads the graph run's bits")?;
        pass &= line(
            eager.same_bits(&got),
            "placed eager",
            &format!("eager {:?}, graph {:?}", eager.ids, got.ids),
        );
        drop(m);

        pass &= node(prompt, budget, &got)?;
        Ok(pass)
    }

    /// The stage card's own experts were routed slots: the run evaluated [`PROMPT`] + [`STEPS`] - 1
    /// positions of `n_used` slots a layer, the tier took `hits` of them, and the rest went to the
    /// layer's stage experts, so some layer holding stage experts must show a remainder; a run whose
    /// slots all went to the tier leaves the card sum unread.
    fn stage_hit(splits: &[(usize, u64, u64, u64)], n_used: usize, sent: Option<&Sent>) -> bool {
        let Some(sent) = sent else {
            return false;
        };
        let slots = ((PROMPT + STEPS - 1) * n_used) as u64;
        let stage_hits: Vec<(usize, u64)> = splits
            .iter()
            .filter(|o| o.1 > 0 && o.2 > 0)
            .map(|&(l, ..)| {
                let tier = l
                    .checked_sub(sent.first)
                    .and_then(|i| sent.hits.get(i))
                    .copied()
                    .unwrap_or(0);
                (l, slots.saturating_sub(tier))
            })
            .collect();
        line(
            stage_hits.iter().any(|&(_, n)| n > 0),
            "precondition stage",
            &format!(
                "slots routed to the stage card's experts per layer {stage_hits:?} of {slots}; \
                 the card sum and the tier sum both ran"
            ),
        )
    }

    /// The placed load of a plan with no expert on the stage: its greedy ids are the whole-card
    /// body's, as on the gate's own plan.
    fn tier_only(
        inputs: &PlanInputs,
        table: &[Budget],
        host: HostCfg,
        (prompt, want): (&[u32], &Run),
    ) -> Result<bool, GateError> {
        let clause = "placed ids, stage without experts";
        tier::sc("placed ids, stage without experts: the ids are the whole-card body's")?;
        let Some(budget) = pick_tier_only(table) else {
            return Ok(line(
                false,
                clause,
                "no budget up to the header's puts every expert on the tier card: the clause has \
                 no input",
            ));
        };
        let machine = bp_machine(inputs)?;
        let plan = inputs.plan(&machine, CTX as u64, &under(budget))?;
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let mut m = GpuModel::<Body>::open_placed(file, &plan, inputs, 0, host)?;
        m.set_mode(StepMode::Graph);
        let on_stage: usize = {
            let p = m.body(NAME)?.placed().ok_or("not a placed load")?;
            (0..inputs.hp.n_layer).map(|l| p.n_card(l)).sum()
        };
        let got = greedy(&mut m, prompt)?;
        Ok(line(
            on_stage == 0 && got.ids == want.ids,
            clause,
            &format!(
                "stage holds {on_stage} experts; placed {:?}, whole-card {:?}, last logits row \
                 relative L2 {:.2e}",
                got.ids,
                want.ids,
                rel(&want.logits, &got.logits)
            ),
        ))
    }

    /// The tier on a loopback `tier_server` on the 3090 alone, through `generate_kimi_k3`.
    fn node(prompt: &[u32], budget: u64, in_process: &Run) -> Result<bool, GateError> {
        let mut cmd = tier_server_command(None)?;
        cmd.env("CUDA_VISIBLE_DEVICES", node_card()?);
        let (_server, addr) = tier_server(cmd, "node")?;
        let (ok, stdout, stderr) = generate(
            prompt,
            &[
                (TIER_NODES, addr.to_string()),
                (CARD_BUDGET, budget.to_string()),
            ],
            false,
        )?;
        tier::sc("node ids: the loopback node's tier reads the in-process tier's bits")?;
        if !ok {
            println!("generate_kimi_k3 on the node tier failed:\n{stdout}\n{stderr}");
            return Ok(line(false, "node ids", "generate_kimi_k3 ended nonzero"));
        }
        let (ids, hash) = printed(&stdout)?;
        Ok(line(
            ids == in_process.ids && hash == fnv_row(&in_process.logits),
            "node ids",
            &format!(
                "node {ids:?} logits fnv1a64={hash:016x}, in-process {:?} fnv1a64={:016x}",
                in_process.ids,
                fnv_row(&in_process.logits)
            ),
        ))
    }

    /// The loads the body refuses by name.
    fn refusals(inputs: &PlanInputs, budget: u64, host: HostCfg) -> Result<bool, GateError> {
        let machine = bp_machine(inputs)?;

        // A tier reserve of 0 where the load makes tier places.
        let wrong = inputs.clone().with_tier_reserve(0);
        let plan = wrong.plan(&machine, CTX as u64, &under(budget))?;
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let text = refused(GpuModel::<Body>::open_placed(file, &plan, &wrong, 0, host));
        tier::sc("reserve: a tier reserve of 0 is refused by name at the load")?;
        let mut pass = line(
            text.as_deref()
                .is_some_and(|t| t.contains("tier places") && t.contains("the plan reserves 0")),
            "reserve",
            &format!("the load of a plan with a tier reserve of 0: {text:?}"),
        );

        // A plan with no head card whose head tensors are on no host: neither a card head nor a
        // host head stage exists. The plan is made headless and its head rows moved to the card.
        let mut headless = bp_machine(inputs)?;
        for card in &mut headless.cards {
            card.head = false;
        }
        let mut plan = inputs.plan(
            &headless,
            CTX as u64,
            &under(budget - card_head_bytes(inputs, budget)?),
        )?;
        let model = plan.model;
        for row in &mut plan.rows {
            if model.tensors[row.tensor].role == Role::Head {
                for segment in &mut row.segments {
                    segment.device = Device::Card(0);
                }
            }
        }
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let text = refused(GpuModel::<Body>::open_placed(file, &plan, inputs, 0, host));
        tier::sc(
            "host head refused: a plan with no head card and no host head is refused by name",
        )?;
        pass &= line(
            text.as_deref()
                .is_some_and(|t| t.contains("carries no head") && t.contains("output.weight")),
            "host head refused",
            &format!("the load of a plan with no head card and its head on a card: {text:?}"),
        );
        Ok(pass)
    }

    /// The bytes the plan at card budget `budget` puts on the stage card for the head tensors.
    fn card_head_bytes(inputs: &PlanInputs, budget: u64) -> Result<u64, GateError> {
        let machine = bp_machine(inputs)?;
        let plan = inputs.plan(&machine, CTX as u64, &under(budget))?;
        Ok(head_bytes(&plan))
    }

    /// The bytes the plan puts on cards for the head tensors.
    fn head_bytes(plan: &Plan<'_>) -> u64 {
        plan.rows
            .iter()
            .filter(|r| plan.model.tensors[r.tensor].role == Role::Head)
            .flat_map(|r| &r.segments)
            .filter(|s| matches!(s.device, Device::Card(_)))
            .map(|s| s.resident_bytes)
            .sum()
    }

    /// A plan whose stage card carries no head, its head tensors left on the host: the load
    /// opens, and fed the whole-card body's ids each step's logits row is within
    /// [`HOST_HEAD_REL_BAND`] of the same plan's load with the head on the card, which holds the
    /// same expert split (the head's bytes leave the stage card's budget). A fault planted on
    /// the card is then the host head's readback error.
    fn host_head(
        inputs: &PlanInputs,
        (budget, host): (u64, HostCfg),
        (prompt, want): (&[u32], &Run),
    ) -> Result<bool, GateError> {
        let machine = bp_machine(inputs)?;
        let with = inputs.plan(&machine, CTX as u64, &under(budget))?;
        let mut headless = bp_machine(inputs)?;
        for card in &mut headless.cards {
            card.head = false;
        }
        let without = inputs.plan(&headless, CTX as u64, &under(budget - head_bytes(&with)))?;
        let on_host = without
            .rows
            .iter()
            .filter(|r| without.model.tensors[r.tensor].role == Role::Head)
            .all(|r| matches!(r.segments.as_slice(), [s] if s.device == Device::Host));
        let same = layer_splits(&with) == layer_splits(&without);
        tier::sc("host head: the host head's logits rows read the card head's")?;
        let mut pass = line(
            on_host && same && head_bytes(&without) == 0,
            "precondition host head",
            &format!(
                "the headless plan leaves the head tensors on the host ({on_host}) and splits the \
                 experts as the plan with the card head does ({same})"
            ),
        );
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let mut m = GpuModel::<Body>::open_placed(file, &with, inputs, 0, host)?;
        m.set_mode(StepMode::Graph);
        let have = forced(&mut m, prompt, &want.ids)?;
        drop(m);
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let mut m = GpuModel::<Body>::open_placed(file, &without, inputs, 0, host)?;
        m.set_mode(StepMode::Graph);
        let got = forced(&mut m, prompt, &want.ids)?;
        let dists: Vec<f64> = have.iter().zip(&got).map(|(w, g)| rel(w, g)).collect();
        let shown: Vec<String> = dists.iter().map(|d| format!("{d:.2e}")).collect();
        let (at, worst) = dists
            .iter()
            .copied()
            .enumerate()
            .fold((0, 0.0), |b, (i, d)| if d > b.1 { (i, d) } else { b });
        pass &= line(
            got.len() == have.len() && worst <= HOST_HEAD_REL_BAND,
            "host head",
            &format!(
                "fed the whole-card ids {:?}, each step's logits row relative L2 of the card \
                 head's {shown:?}, worst {worst:.3e} at step {at}, band {HOST_HEAD_REL_BAND:.1e}",
                want.ids
            ),
        );

        tier::sc("host head fault: a planted fault is the host head's readback error")?;
        m.plant_fault(Fault::at(3, FaultSite::CachePos))?;
        let r = m.step(&[want.ids[0]]);
        pass &= line(
            matches!(r, Err(GpuError::Fault { .. })),
            "host head fault",
            &format!(
                "a fault planted before a step: {:?}",
                r.map_err(|e| e.to_string())
            ),
        );
        Ok(pass)
    }

    /// A plan that leaves experts on the host (the tier card's budget squeezed under what the
    /// experts need): the load holds the plan's counts per layer, and fed the whole-card body's
    /// ids each step's logits row is within [`HOST_REL_BAND`] of the whole-card row. The load's
    /// own greedy ids and its last row alone are not compared.
    fn host_share(
        inputs: &PlanInputs,
        table: &[Budget],
        (budget, host): (u64, HostCfg),
        (prompt, want): (&[u32], &Run),
    ) -> Result<bool, GateError> {
        let clause = "host share";
        tier::sc("host share: the host tier's experts read the whole-card body's logits rows")?;
        let Some(HostShare {
            levers,
            stage,
            tier: held,
            host: left,
        }) = host_share_plan(
            inputs,
            pick_tier_only(table).unwrap_or(budget),
            Share::Tiered,
        )?
        else {
            return Ok(line(
                false,
                clause,
                "no tier budget leaves an expert on the host: the clause has no input",
            ));
        };
        let machine = bp_machine(inputs)?;
        let plan = inputs.plan(&machine, CTX as u64, &levers)?;
        let splits = layer_splits(&plan);
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let mut m = GpuModel::<Body>::open_placed(file, &plan, inputs, 0, host)?;
        m.set_mode(StepMode::Graph);
        let wrong: Vec<_> = {
            let p = m.body(NAME)?.placed().ok_or("not a placed load")?;
            splits
                .iter()
                .filter(|&&(l, s, t, _)| p.n_card(l) as u64 != s || p.tier_k(l) as u64 != t)
                .map(|&(l, ..)| (l, p.n_card(l), p.tier_k(l)))
                .collect()
        };
        let rows = forced(&mut m, prompt, &want.ids)?;
        let dists: Vec<f64> = want
            .rows
            .iter()
            .zip(&rows)
            .map(|(w, g)| rel(w, g))
            .collect();
        let shown: Vec<String> = dists.iter().map(|d| format!("{d:.2e}")).collect();
        let (at, worst) = dists
            .iter()
            .copied()
            .enumerate()
            .fold((0, 0.0f64), |a, (i, d)| if d > a.1 { (i, d) } else { a });
        Ok(line(
            wrong.is_empty() && rows.len() == want.rows.len() && worst <= HOST_REL_BAND,
            clause,
            &format!(
                "per-card budgets {} hold stage {stage}, tier {held}, host {left} experts (layers \
                 off the plan: {wrong:?}); fed the whole-card ids {:?}, each step's logits row \
                 relative L2 {shown:?}, worst {worst:.3e} at step {at}, band {HOST_REL_BAND:.1e}",
                levers.card_budget_bytes, want.ids
            ),
        ))
    }

    /// A plan where some layer holds host experts and no tier expert: the load refuses it by name,
    /// since the host handoff's entries are built for other slot counts than K3's.
    fn host_without_tier(
        inputs: &PlanInputs,
        table: &[Budget],
        (budget, host): (u64, HostCfg),
    ) -> Result<bool, GateError> {
        let clause = "host layer without tier";
        tier::sc("host layer without tier: a host share with no tier expert is refused by name")?;
        let Some(HostShare {
            levers,
            stage,
            tier: held,
            host: left,
        }) = host_share_plan(
            inputs,
            pick_tier_only(table).unwrap_or(budget),
            Share::Tierless,
        )?
        else {
            return Ok(line(
                false,
                clause,
                "no tier budget leaves a layer with host experts and none on the tier: the clause \
                 has no input",
            ));
        };
        let machine = bp_machine(inputs)?;
        let plan = inputs.plan(&machine, CTX as u64, &levers)?;
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let text = refused(GpuModel::<Body>::open_placed(file, &plan, inputs, 0, host));
        Ok(line(
            text.as_deref()
                .is_some_and(|t| t.contains("keep experts on the host and none on a tier card")),
            clause,
            &format!(
                "per-card budgets {} hold stage {stage}, tier {held}, host {left} experts; its \
                 load: {text:?}",
                levers.card_budget_bytes
            ),
        ))
    }

    /// The node of another family, and the plan records.
    fn node_family_and_records(prompt: &[u32], budget: u64) -> Result<bool, GateError> {
        let mut pass = true;
        let mut cmd = Command::new(std::env::current_exe()?);
        cmd.arg(SERVE_OTHER)
            .env("CUDA_VISIBLE_DEVICES", node_card()?)
            .env_remove(TIER_NODES)
            .env_remove(CARD_BUDGET)
            .env_remove(HOST_POPULATE)
            .env_remove(HOST_LOCK)
            .env_remove(CARD_DONTNEED);
        let (_other, addr) = tier_server(cmd, "other")?;
        let (ok, _, stderr) = generate(
            prompt,
            &[
                (TIER_NODES, addr.to_string()),
                (CARD_BUDGET, budget.to_string()),
            ],
            false,
        )?;
        tier::sc("node family: a node that serves another family refuses the open by name")?;
        pass &= line(
            !ok && stderr.contains("a tier of the family \"kimi-k3\"")
                && stderr.contains("this server opens"),
            "node family",
            &format!(
                "generate_kimi_k3 against a node of another family: ok={ok}, stderr {stderr:?}"
            ),
        );

        let (ok, _, stderr) = generate(prompt, &[(CARD_BUDGET, budget.to_string())], true)?;
        let plan = stderr.lines().find(|l| l.starts_with("plan "));
        tier::sc("plan records: --plan --place bp prints the plan record of the placed machine")?;
        pass &= line(
            ok && plan.is_some_and(|l| l.contains("place=bp") && l.contains("host_experts=0")),
            "plan records",
            &format!("generate_kimi_k3 --plan --place bp: ok={ok}, record {plan:?}"),
        );
        Ok(pass)
    }

    /// The pass's plan at `shape` over the gate's `budget`: the inputs with the pass's bytes as
    /// the prompt reserve and the arena at its floor, the machine with the tier's batch at the
    /// pass's rows, and per-card budgets, the stage card's raised by the pass's bytes and the
    /// tier card's by its batch's growth, so each card's experts keep their room.
    /// With `headless`, no card carries the head and the stage card's budget gives up the head's
    /// bytes, so the experts split as with the head on the card.
    fn pass_plan_inputs(
        inputs: &PlanInputs,
        budget: u64,
        shape: PassShape,
        headless: bool,
    ) -> Result<(PlanInputs, Machine, PlanLevers), GateError> {
        let reserve = pass_bytes(&inputs.hp, CTX, shape)?;
        let grown = tier_batch_at(&inputs.hp, shape.rows as u64)?.card()?
            - tier_batch(&inputs.hp)?.card()?;
        let pass = inputs
            .clone()
            .with_prompt_reserve(reserve)
            .with_pass_arena();
        let mut machine = bp_machine_at(inputs, Some(shape.rows as u64))?;
        let head = if headless {
            machine.cards.iter_mut().for_each(|c| c.head = false);
            card_head_bytes(inputs, budget)?
        } else {
            0
        };
        let (Some(s), Some(t)) = (
            machine.cards.first().and_then(|c| c.device),
            machine.tiers.first().and_then(|c| c.device),
        ) else {
            return Err(
                "the gate's cards carry no census device, which a per-card budget names".into(),
            );
        };
        let levers = levers_of(CardBudget::Per(vec![
            (s.ordinal, budget + reserve - head),
            (t.ordinal, budget + grown),
        ]));
        Ok((pass, machine, levers))
    }

    /// The placed pass at each of [`PASS_SHAPES`] against the same load stepped, and its
    /// refusals.
    fn placed_pass(inputs: &PlanInputs, budget: u64, host: HostCfg) -> Result<bool, GateError> {
        let prompt = e2e::walk(PASS_PROMPT, inputs.hp.n_vocab);
        let mut pass = true;
        for shape in PASS_SHAPES {
            let (pass_inputs, machine, at) = pass_plan_inputs(inputs, budget, shape, false)?;
            let plan = pass_inputs.plan(&machine, CTX as u64, &at)?;
            let splits = layer_splits(&plan);
            let (stage, tier_n, host_n) = totals(&splits);
            let tier_layers = splits.iter().filter(|o| o.2 > 0).count() as u64;
            let label = format!("{} rows, latent {}", shape.rows, shape.mla_rows);
            println!(
                "pass plan ({label}) under card budgets {}: stage {stage} experts, tier {tier_n}, \
                 host {host_n}, reserve {} B",
                at.card_budget_bytes, pass_inputs.prompt_reserve
            );
            let file = open_split(Arch::KimiK3, RECIPE)?;
            let mut m =
                GpuModel::<Body>::open_placed_pass(file, &plan, &pass_inputs, 0, host, shape)?;
            m.set_mode(StepMode::Graph);
            let (got, sent) = walked(&mut m, &prompt, true)?;
            m.reset()?;
            let (want, stepped) = walked(&mut m, &prompt, false)?;
            tier::sc("pass state: the passes read the stepped prompt's ids and rows")?;
            pass &= line(
                got.same_rows(&want),
                "pass state",
                &format!(
                    "{label}: passes {:?}, steps {:?}, rows equal {} of {}",
                    got.ids,
                    want.ids,
                    got.rows
                        .iter()
                        .zip(&want.rows)
                        .filter(|(a, b)| fnv_row(a) == fnv_row(b))
                        .count(),
                    want.rows.len()
                ),
            );

            let passes = PASS_PROMPT.div_ceil(shape.rows) as u64;
            tier::sc("pass batches: the tier's hits are the steps', one batch a layer a pass")?;
            let (hits, steps_hits, served) = match (&sent, &stepped) {
                (Some(p), Some(s)) => (p.hits.clone(), s.hits.clone(), p.stats.served),
                _ => (Vec::new(), vec![1], 0),
            };
            pass &= line(
                tier_layers > 0 && hits == steps_hits && served == passes * tier_layers,
                "pass batches",
                &format!(
                    "{label}: tier hits over the prompt {hits:?} (steps {steps_hits:?}); batches \
                     served {served}, {passes} passes of {tier_layers} tier layers"
                ),
            );

            m.reset()?;
            let ids: Vec<u32> = prompt
                .iter()
                .chain(&want.ids[..STEPS - 1])
                .copied()
                .collect();
            let logits = m.score_placed(&ids, STEPS)?;
            let n_vocab = inputs.hp.n_vocab;
            let same = logits.len() == STEPS * n_vocab
                && (0..STEPS).all(|t| {
                    fnv_row(&logits[t * n_vocab..(t + 1) * n_vocab]) == fnv_row(&want.rows[t])
                });
            tier::sc("pass score: the last rows' logits are the steps' rows")?;
            pass &= line(
                same,
                "pass score",
                &format!(
                    "{label}: {} ids as passes, the last {STEPS} rows' logits against the stepped \
                     rows: {same}",
                    ids.len()
                ),
            );
            let past = m
                .score_placed(&ids, shape.rows + 1)
                .err()
                .map(|e| e.to_string());
            drop(m);
            tier::sc("pass refused: rows past the pass are refused by name")?;
            pass &= line(
                past.as_deref()
                    .is_some_and(|t| t.contains(&format!("the pass's {} rows", shape.rows))),
                "pass refused",
                &format!("{label}: score_placed of {} rows: {past:?}", shape.rows + 1),
            );
        }

        pass &= host_head_pass(inputs, budget, host, &prompt)?;

        let shape = PASS_SHAPES[0];
        let (pass_inputs, machine, at) = pass_plan_inputs(inputs, budget, shape, false)?;
        let plan = pass_inputs.plan(&machine, CTX as u64, &at)?;
        let wide = PassShape {
            rows: model::moe::UNION_MAX_COLS + 1,
            mla_rows: 1,
        };
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let text = refused(GpuModel::<Body>::open_placed_pass(
            file,
            &plan,
            &pass_inputs,
            0,
            host,
            wide,
        ));
        tier::sc("pass refused: a shape out of range at the load")?;
        pass &= line(
            text.as_deref()
                .is_some_and(|t| t.contains(&format!("a pass of {} rows", wide.rows))),
            "pass refused",
            &format!("a pass of {} rows: {text:?}", wide.rows),
        );

        // The pass's machine and tier batch with no prompt reserve: the stage card's budget is
        // the gate's alone, so the experts split as with the reserve.
        let short = inputs.clone().with_pass_arena();
        let grown = tier_batch_at(&inputs.hp, shape.rows as u64)?.card()?
            - tier_batch(&inputs.hp)?.card()?;
        let (Some(s), Some(t)) = (
            machine.cards.first().and_then(|c| c.device),
            machine.tiers.first().and_then(|c| c.device),
        ) else {
            return Err("the gate's cards carry no census device".into());
        };
        let at = levers_of(CardBudget::Per(vec![
            (s.ordinal, budget),
            (t.ordinal, budget + grown),
        ]));
        let plan = short.plan(&machine, CTX as u64, &at)?;
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let text = refused(GpuModel::<Body>::open_placed_pass(
            file, &plan, &short, 0, host, shape,
        ));
        tier::sc("pass refused: a plan that does not reserve the pass")?;
        pass &= line(
            text.as_deref()
                .is_some_and(|t| t.contains("tier places and placed pass")),
            "pass refused",
            &format!("a plan with no prompt reserve: {text:?}"),
        );

        // The step's plan: the tier's batch one token, no pass anywhere.
        let step_machine = bp_machine(inputs)?;
        let step = short.plan(
            &step_machine,
            CTX as u64,
            &levers_of(CardBudget::Every(budget)),
        )?;
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let mut m = GpuModel::<Body>::open_placed(file, &step, &short, 0, host)?;
        let text = m.prompt_placed(&prompt).err().map(|e| e.to_string());
        tier::sc("pass refused: prompt_placed on a load without the pass")?;
        pass &= line(
            text.as_deref()
                .is_some_and(|t| t.contains("a load made without the placed pass")),
            "pass refused",
            &format!("prompt_placed on a load without the pass: {text:?}"),
        );
        Ok(pass)
    }

    /// `pass score, host head`: a plan whose head the host reads, scored as passes against the
    /// same load stepped.
    fn host_head_pass(
        inputs: &PlanInputs,
        budget: u64,
        host: HostCfg,
        prompt: &[u32],
    ) -> Result<bool, GateError> {
        let shape = PASS_SHAPES[1];
        let (pass_inputs, machine, at) = pass_plan_inputs(inputs, budget, shape, true)?;
        let plan = pass_inputs.plan(&machine, CTX as u64, &at)?;
        let on_host = plan
            .rows
            .iter()
            .filter(|r| plan.model.tensors[r.tensor].role == Role::Head)
            .all(|r| matches!(r.segments.as_slice(), [s] if s.device == Device::Host));
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let mut m = GpuModel::<Body>::open_placed_pass(file, &plan, &pass_inputs, 0, host, shape)?;
        m.set_mode(StepMode::Graph);
        let (want, _) = walked(&mut m, prompt, false)?;
        m.reset()?;
        let ids: Vec<u32> = prompt
            .iter()
            .chain(&want.ids[..STEPS - 1])
            .copied()
            .collect();
        let logits = m.score_placed(&ids, STEPS)?;
        let n_vocab = inputs.hp.n_vocab;
        let equal = (0..STEPS)
            .filter(|&t| {
                logits.get(t * n_vocab..(t + 1) * n_vocab).map(fnv_row)
                    == Some(fnv_row(&want.rows[t]))
            })
            .count();
        tier::sc("pass score, host head: the host's rows projection reads the steps' rows")?;
        Ok(line(
            on_host && equal == STEPS,
            "pass score, host head",
            &format!(
                "the plan leaves the head on the host ({on_host}); the last {STEPS} rows' logits \
                 equal to the stepped rows: {equal} of {STEPS}"
            ),
        ))
    }

    /// `generate_kimi_k3 --score FILE --walk pass` over `env`: its status, stdout and stderr.
    fn score_walk(
        file: &std::path::Path,
        env: &[(&str, String)],
    ) -> Result<(bool, String, String), GateError> {
        let bin = beside("generate_kimi_k3")?;
        let mut cmd = Command::new(&bin);
        cmd.arg("--score")
            .arg(file)
            .args(["--walk", "pass", "--ctx", &CTX.to_string(), "--place", "bp"])
            .args(["--granule", &GRANULE.to_string()]);
        for (k, v) in env {
            cmd.env(k, v);
        }
        die_with_parent(&mut cmd);
        let out = cmd
            .output()
            .map_err(|e| format!("{}: {e}", bin.display()))?;
        Ok((
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        ))
    }

    /// The first `top` id of a `--score` line: the row's argmax.
    fn top_id(line: &str) -> Option<u32> {
        let at = line.find("\"top\":[[")? + "\"top\":[[".len();
        let digits: String = line[at..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        digits.parse().ok()
    }

    /// `generate_kimi_k3 --walk pass` at a budget that holds its widest shape at [`CTX`]: the
    /// `prompt_pass` record names that shape, and the run reads the same plan's load stepped.
    /// The derived K3.1 file's indexer top-k: the prompt of [`pass_past_top_k`] crosses it.
    const DSA_TOP_K: usize = 16;
    /// Its cache rows: past the top-k, so the load selects, and at most the attention's one
    /// segment (`flash::seg_keys`), so the list's gathered rows and the cache take the same flash
    /// form and the selection below the top-k is the dense walk's bits.
    const DSA_CTX: usize = 64;
    /// The ids of [`pass_past_top_k`]'s first call: the second call's first pass starts here and
    /// runs past the top-k.
    const DSA_SPLIT: usize = 5;

    /// The K3 fixture as a K3.1 file: every key under `kimi-k31.` for `kimi-k3.`, the indexer's
    /// keys at a top-k of [`DSA_TOP_K`] on the latent layers, and their five tensors each (the
    /// matrices Q8_0 of seeded values, the key norm's gain 1 and bias 0, F32), every other tensor
    /// the fixture's bytes. Written under the temp directory; the path.
    fn k31_fixture(inputs: &PlanInputs) -> Result<std::path::PathBuf, GateError> {
        use bloomery_gpu_gates::{GateRng, write_gguf};
        use gguf::arch::kimi_k3::{ARCH as K3, ARCH_K31 as ARCH};
        use gguf::quant::Q8Block;
        use gguf::{GENERAL_ARCHITECTURE, GgmlType, Value};
        use model::arch::kimi_k3::hparams::{INDEX_DIM, INDEX_HEADS, Kind};

        let k3_prefix = format!("{K3}.");
        let split = open_split(Arch::KimiK3, RECIPE)?;
        let hp = &inputs.hp;
        // One file: the fixture's split keys count its own tensors, not the indexer's added here.
        let mut kvs: Vec<(String, Value)> = split
            .iter_kv()
            .filter(|(k, _)| !k.starts_with("split."))
            .map(|(k, v)| match (k, k.strip_prefix(k3_prefix.as_str())) {
                (k, _) if k == GENERAL_ARCHITECTURE => (k.to_string(), Value::String(ARCH.into())),
                (_, Some(rest)) => (format!("{ARCH}.{rest}"), v.clone()),
                _ => (k.to_string(), v.clone()),
            })
            .collect();
        let latent: Vec<bool> = hp.kinds.iter().map(|k| *k == Kind::Mla).collect();
        for (suffix, v) in [
            (
                "attention.indexer.head_count",
                Value::U32(u32::try_from(INDEX_HEADS)?),
            ),
            (
                "attention.indexer.key_length",
                Value::U32(u32::try_from(INDEX_DIM)?),
            ),
            (
                "attention.indexer.top_k",
                Value::U32(u32::try_from(DSA_TOP_K)?),
            ),
            (
                "attention.indexer.types",
                Value::Array(latent.iter().map(|&b| Value::Bool(b)).collect()),
            ),
            ("attention.indexer.rope_dimension_count", Value::U32(0)),
            ("attention.layer_norm_epsilon", Value::F32(1e-5)),
        ] {
            kvs.push((format!("{ARCH}.{suffix}"), v));
        }
        let mut decls: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)> = split
            .iter_tensors()
            .map(|(_, t)| {
                Ok((
                    t.name.clone(),
                    t.dims.clone(),
                    t.ty,
                    split.data(&t.name)?.to_vec(),
                ))
            })
            .collect::<Result<_, GateError>>()?;
        let mut r = GateRng(0x6b33_3164_7361_0001);
        let mut q8 = |k: usize, n: usize| -> Result<Vec<u8>, GateError> {
            let vals: Vec<f32> = (0..k * n).map(|_| r.uni(-0.05, 0.05)).collect();
            let mut out = Vec::with_capacity(k * n / 32 * 34);
            for c in vals.as_chunks::<32>().0 {
                let b = Q8Block::quantize_ref(c).ok_or("a non-finite Q8_0 block")?;
                out.extend_from_slice(&b.d.to_le_bytes());
                out.extend(b.q.iter().map(|&q| q.to_le_bytes()[0]));
            }
            Ok(out)
        };
        let f32s = |v: f32| -> Vec<u8> { (0..INDEX_DIM).flat_map(|_| v.to_le_bytes()).collect() };
        let (embd, q_lora, d) = (hp.n_embd, hp.mla.q_lora, INDEX_DIM);
        for l in (0..hp.n_layer).filter(|&l| latent[l]) {
            let blk = |stem: &str| format!("blk.{l}.indexer.{stem}");
            // A matrix's dims are `[in, out]`, the ggml order: rows of `in` values, `out` of them.
            for (stem, k, n) in [
                ("attn_q_b.weight", q_lora, INDEX_HEADS * d),
                ("attn_k.weight", embd, d),
                ("proj.weight", embd, INDEX_HEADS),
            ] {
                let dims = vec![k as u64, n as u64];
                decls.push((blk(stem), dims, GgmlType::Q8_0, q8(k, n)?));
            }
            for (stem, v) in [("k_norm.weight", 1.0), ("k_norm.bias", 0.0)] {
                decls.push((blk(stem), vec![d as u64], GgmlType::F32, f32s(v)));
            }
        }
        let dir = std::env::temp_dir().join(format!("gate-kimi-k31-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("kimi-k31-fixture.gguf");
        write_gguf(&path, &kvs, &decls)?;
        println!("pass past top-k: model={}", path.display());
        Ok(path)
    }

    /// `pass past top-k` (the module doc): the K3.1 fixture written, its arms run, the fixture
    /// removed.
    fn pass_past_top_k(inputs: &PlanInputs, header: u64, host: HostCfg) -> Result<bool, GateError> {
        let path = k31_fixture(inputs)?;
        let ran = past_top_k_arms(&path, header, host);
        let _ = std::fs::remove_dir_all(path.parent().ok_or("the fixture's directory")?);
        ran
    }

    /// The greedy ids from `first`, the token after a prompt: [`STEPS`] ids and the logits row
    /// after the prompt and after each id fed back.
    fn greedy_from(m: &mut GpuModel<Body>, first: u32) -> Result<Run, GateError> {
        let mut ids = vec![first];
        let mut rows = vec![m.logits()?];
        while ids.len() < STEPS {
            let last = ids[ids.len() - 1];
            ids.push(m.step(&[last])?);
            rows.push(m.logits()?);
        }
        Ok(Run {
            ids,
            logits: m.logits()?,
            rows,
        })
    }

    /// Whether `m`'s load selects at [`DSA_CTX`] over the K3.1 fixture's top-k.
    fn selects(m: &GpuModel<Body>) -> Result<bool, GateError> {
        Ok(m.body(NAME)?
            .indexer()
            .is_some_and(|ix| DSA_CTX > ix.top_k && ix.top_k == DSA_TOP_K))
    }

    /// One arm's verdict: the run `got` at `got_pos` against the stepped run `want` at `want_pos`.
    fn past_top_k_line(
        arm: &str,
        selects: bool,
        (got, got_pos): (&Run, u32),
        (want, want_pos): (&Run, u32),
    ) -> Result<bool, GateError> {
        let clause = format!("pass past top-k, {arm}");
        tier::sc(&format!(
            "{clause}: a prompt across the top-k reads the stepped token, position and logits rows"
        ))?;
        Ok(line(
            selects && got_pos == want_pos && got.same_rows(want),
            &clause,
            &format!(
                "top-k {DSA_TOP_K}, ctx {DSA_CTX}, the load selects: {selects}; {DSA_SPLIT} ids, \
                 then {} more: token {} at position {got_pos}, ids {:?}; stepped: token {} at \
                 position {want_pos}, ids {:?}; every logits row's bits equal: {}",
                DSA_TOP_K + 8 - DSA_SPLIT,
                got.ids[0],
                got.ids,
                want.ids[0],
                want.ids,
                got.same_rows(want)
            ),
        ))
    }

    /// Clause `room` (the module doc): the stores' rooms of the whole-card K3.1 body back no
    /// page before a row is written (`fresh`), the pages of the rows a prompt and its steps
    /// reached on the card alone after them (`used`: some, none on the host, the load having no
    /// spill), the same pages after a reset (`kept`: a settle keeps the pages the sequence it ends
    /// reached), and none after a second reset (`settled`, which ends an empty sequence).
    fn room_line(
        fresh: (usize, usize),
        used: (usize, usize),
        kept: (usize, usize),
        settled: (usize, usize),
    ) -> Result<bool, GateError> {
        tier::sc("room: the stores' pages are backed as far as the positions written reach")?;
        Ok(line(
            fresh == (0, 0) && used.0 > 0 && used.1 == 0 && kept == used && settled == (0, 0),
            "room",
            &format!(
                "mapped (card, host) bytes: fresh {fresh:?}, after the prompt and {STEPS} steps \
                 {used:?}, after a reset {kept:?}, after a second reset {settled:?}"
            ),
        ))
    }

    /// Clause `kvspill` (the module doc): the plan's host bytes under `spill` with a floor of
    /// [`KV_FLOOR`] positions at a context of [`KV_CTX`] on the stage card's host page, against
    /// the value derived from the cached row's width (a latent layer's `kv_lora` + rope
    /// values, f16) and the page; `off` counts none. The machine is the `bp` pair's stage card
    /// alone: the stores and their host pages are the stage's, a tier card holds none, and at the
    /// whole card's room (no card budget here) the stage holds every routed expert of the small
    /// file, which leaves a tier card idle and the plan refused (`IdleTier`).
    fn kv_spill(inputs: &PlanInputs) -> Result<bool, GateError> {
        const KV_CTX: u64 = 262_144;
        const KV_FLOOR: u64 = 8_192;
        tier::sc("kvspill: the plan counts the pages past the floor against its host room")?;
        let page = stage_host_page(Place::Bp.on_host()?.card_specs()?[0])?;
        let mut inputs = inputs.clone();
        inputs.kv_host_page = Some(page);
        let mut machine = bp_machine(&inputs)?;
        machine.tiers.clear();
        machine.nodes.truncate(1);
        let held = |host: KvHost| -> Result<u64, GateError> {
            let levers = PlanLevers {
                kv_floor: Some(KV_FLOOR),
                kv_host: host,
                ..PlanLevers::default()
            };
            Ok(inputs.plan(&machine, KV_CTX, &levers)?.host.kv_host_bytes)
        };
        let latent = inputs
            .hp
            .kinds
            .iter()
            .filter(|k| **k == model::arch::kimi_k3::hparams::Kind::Mla)
            .count() as u64;
        let row = 2 * (inputs.hp.mla.kv_lora + inputs.hp.mla.pe) as u64;
        let pages = |positions: u64| (positions * row).div_ceil(page);
        let want = latent * (pages(KV_CTX) - pages(KV_FLOOR)) * page;
        let (spill, off) = (held(KvHost::Spill)?, held(KvHost::Off)?);
        Ok(line(
            spill == want && off == 0,
            "kvspill",
            &format!(
                "host bytes under spill {spill}, derived {want} ({latent} latent layers, row \
                 {row} B, page {page} B); under off {off}"
            ),
        ))
    }

    /// The arms of `pass past top-k` over the K3.1 fixture at `path`: the whole-card body, then
    /// the placed pass load at each of [`PASS_SHAPES`], its card budget the K3.1 file's own pick
    /// under the header's budget `header` (the plan counts the indexer's tensors as dense).
    fn past_top_k_arms(
        path: &std::path::Path,
        header: u64,
        host: HostCfg,
    ) -> Result<bool, GateError> {
        let inputs31 = PlanInputs::read_without_chat(&Split::open(path)?)?;
        let prompt = e2e::walk(DSA_TOP_K + 8, inputs31.hp.n_vocab);
        let (head, rest) = prompt.split_at(DSA_SPLIT);
        let mut pass = true;

        let mut m = GpuModel::<Body>::open(Gpu::new()?, Split::open(path)?, DSA_CTX)?;
        let mut rows = {
            let (gpu, _, body) = m.body_parts(NAME)?;
            Pass::new(gpu, body)?
        };
        let on = selects(&m)?;
        let fresh = m.body(NAME)?.room_mapped();
        m.prompt_rows(&mut rows, head)?;
        let first = m.prompt_rows(&mut rows, rest)?;
        let got_pos = m.pos();
        let got = greedy_from(&mut m, first)?;
        let used = m.body(NAME)?.room_mapped();
        m.reset()?;
        let kept = m.body(NAME)?.room_mapped();
        m.reset()?;
        let settled = m.body(NAME)?.room_mapped();
        let first = m.step(&prompt)?;
        let want_pos = m.pos();
        let want = greedy_from(&mut m, first)?;
        drop((m, rows));
        pass &= past_top_k_line("whole card", on, (&got, got_pos), (&want, want_pos))?;
        pass &= room_line(fresh, used, kept, settled)?;
        pass &= kv_spill(&inputs31)?;

        let table = split_table(&inputs31, &bp_machine(&inputs31)?, DSA_CTX as u64, header)?;
        let budget = pick_budget(&table)?;
        let latent = inputs31
            .hp
            .kinds
            .iter()
            .filter(|k| **k == model::arch::kimi_k3::hparams::Kind::Mla)
            .count() as u64;
        // The stage card also holds the latent layers' key rows, beside the plan.
        let extra = latent * (DSA_CTX as u64) * 256 + (1 << 20);
        for shape in PASS_SHAPES {
            let (pass_inputs, machine, mut at) = pass_plan_inputs(&inputs31, budget, shape, false)?;
            // Kept out of the expert budget, as the prompt reserve is: the split stays the pick's.
            let reserve = pass_inputs.prompt_reserve + extra;
            let pass_inputs = pass_inputs.with_prompt_reserve(reserve);
            if let CardBudget::Per(cards) = &mut at.card_budget_bytes
                && let Some(stage) = cards.first_mut()
            {
                stage.1 += extra;
            }
            let plan = pass_inputs.plan(&machine, DSA_CTX as u64, &at)?;
            let split = Split::open(path)?;
            let mut m =
                GpuModel::<Body>::open_placed_pass(split, &plan, &pass_inputs, 0, host, shape)?;
            m.set_mode(StepMode::Graph);
            let on = selects(&m)?;
            Body::prompt(&mut m, head)?;
            let first = Body::prompt(&mut m, rest)?;
            let got_pos = m.pos();
            let got = greedy_from(&mut m, first)?;
            m.reset()?;
            let first = m.step(&prompt)?;
            let want_pos = m.pos();
            let want = greedy_from(&mut m, first)?;
            drop(m);
            let arm = format!("placed rows={} mla_rows={}", shape.rows, shape.mla_rows);
            pass &= past_top_k_line(&arm, on, (&got, got_pos), (&want, want_pos))?;
        }
        Ok(pass)
    }

    fn pass_cli(inputs: &PlanInputs, budget: u64, host: HostCfg) -> Result<bool, GateError> {
        let shape = SHAPES
            .iter()
            .copied()
            .find(|s| s.rows <= CTX)
            .ok_or("no pass shape at the gate's context")?;
        let (pass_inputs, machine, at) = pass_plan_inputs(inputs, budget, shape, false)?;
        let prompt = e2e::walk(PASS_PROMPT, inputs.hp.n_vocab);
        let env = [(CARD_BUDGET, at.card_budget_bytes.to_string())];
        let mut pass = true;
        let (ok, _, stderr) = generate_walk(&prompt, &env, true)?;
        let record = stderr.lines().find(|l| l.starts_with("prompt pass "));
        let want = format!("rows={} mla_rows={} ", shape.rows, shape.mla_rows);
        tier::sc("pass cli: --walk pass --plan prints the widest shape's prompt_pass record")?;
        pass &= line(
            ok && record.is_some_and(|l| l.contains(&want) && !l.contains("wider=")),
            "pass cli",
            &format!("generate_kimi_k3 --walk pass --plan: ok={ok}, record {record:?}"),
        );

        let plan = pass_inputs.plan(&machine, CTX as u64, &at)?;
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let mut m = GpuModel::<Body>::open_placed_pass(file, &plan, &pass_inputs, 0, host, shape)?;
        m.set_mode(StepMode::Graph);
        let (stepped, _) = walked(&mut m, &prompt, false)?;
        drop(m);
        let (ok, stdout, stderr) = generate_walk(&prompt, &env, false)?;
        let read = if ok { printed(&stdout).ok() } else { None };
        tier::sc("pass cli: --walk pass reads the same plan's stepped load")?;
        pass &= line(
            read.as_ref().is_some_and(|(ids, hash)| {
                *ids == stepped.ids && *hash == fnv_row(&stepped.logits)
            }),
            "pass cli",
            &format!(
                "generate_kimi_k3 --walk pass: ok={ok}, read {read:?}, stepped {:?} {:016x}{}",
                stepped.ids,
                fnv_row(&stepped.logits),
                if ok {
                    String::new()
                } else {
                    format!(", stderr {stderr:?}")
                }
            ),
        );

        let list = |ids: &[u32]| ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        let file = std::env::temp_dir().join(format!("{NAME}-score-{}.tsv", std::process::id()));
        std::fs::write(
            &file,
            format!("pass\t{}\t{}\t\n", list(&prompt), list(&stepped.ids)),
        )?;
        let (ok, stdout, stderr) = score_walk(&file, &env)?;
        std::fs::remove_file(&file)?;
        let tops: Vec<Option<u32>> = stdout
            .lines()
            .filter(|l| l.starts_with('{'))
            .map(top_id)
            .collect();
        let want: Vec<Option<u32>> = stepped.ids.iter().copied().map(Some).collect();
        tier::sc("pass cli: --score --walk pass reads the stepped load's argmax at each position")?;
        pass &= line(
            ok && tops == want,
            "pass cli",
            &format!(
                "generate_kimi_k3 --score --walk pass: ok={ok}, top ids {tops:?}, stepped {:?}{}",
                stepped.ids,
                if ok {
                    String::new()
                } else {
                    format!(", stderr {stderr:?}")
                }
            ),
        );
        Ok(pass)
    }
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_kimi_k3_place: built without the `gpu` feature");
    std::process::exit(2);
}
