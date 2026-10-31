# Placement census — every family that places, what it accepts, what it reserves

The evidence table behind [`place-any-card.md`](place-any-card.md). Read-only: written from the tree at int039
(`9c90b168d`) and round muse2card's uncommitted tree (`/Users/hckim/repo/bloomery-muse2card`, 2026-10-23 ~19:50 KST).
Nothing here was run on a card. The NVMe tier rows include the current common host-run loaders.
A number marked [derived] is arithmetic over the pins and constants cited beside it.

Terms. The **stage** is device 0 of the plan's card list (every layer, the head, KV, then experts in its budget). A
**tier** is a later card holding routed experts only (`Machine::all_cards` = stage cards ++ tiers,
`crates/placement/src/placement.rs:350`; `Device::Card(i)` indexes that list, `:498`). A **named reserve** is an entry in
`Card::reserves` (`placement.rs:228`) or `Host::reserves` (`:309`), taken out of that device's budget by
`Card::set_aside_bytes` (`:255`). The **scalar reserve** is the unnamed `reserve: u64` argument of
`plan_routed_reserving` (`:2268`), `plan_host_routed_reserving` (`:2308`) and `nodes::loaded_routed`, which `plan_rule`
(`:2341`) hands to the stage *and every tier* Fill and, on a unified pool, counts once per card (`:2472-2477`). A
**post-plan buffer** is device memory a body allocates after the load from the card's free room
(`Gpu::alloc_room`, `largest_fit` with `FIT_RESERVE` = 256 MiB, `crates/gpu/src/model/fit.rs:19,27`), which no plan
term counts; it lives in the plan's 1 GiB `MARGIN` (`workstation.rs:112`).

## 1. Headline

| Machine shape | Families that accept it | Families that refuse it |
|---|---|---|
| one card | all 11 | — |
| stage + one tier card | deepseek41, glm5next, qwen4exp, qwen3moe/qwen35moe (plan only), solar2, musespark (muse2card) | mimo2, minimaxm3, raon1 (missing pieces); nemotron_h (architecture); nemotron_h_moe (missing piece) |
| stage + N ≥ 2 tier cards | the planner holds N (`plan_tiers`, `workstation.rs:509`) | every body: `SERVED_TIERS = 1` (`crates/gpu/src/host/mod.rs:160`, "N >= 2 lands with V4.1's `_tiers` handoff"); the slot map holds 8 (`crates/gpu/src/host/slots.rs:17`) |
| layer cut (`plan_b`, `CUT`) | deepseek41 only, and only in `gate_load_v41` / `stage-gpu-load-v41` (`crates/gpu-gates/src/bin/gate_load_v41.rs:248`) | the rest never ask |
| tensor parallel | glm5next (`crates/model/src/arch/glm5next/tp.rs`, `crates/gpu-glm5next/src/tp/`) | the rest never ask |
| N nodes (Spark) | deepseek41, glm5next, qwen4exp (`nodes::loaded_routed`) | the rest never ask |
| NVMe expert tier | deepseek41, glm5next, qwen4exp, nemotron_h_moe; minimaxm3, musespark and solar2 through `place_hybrid::PlanInputs::with_nvme_tier` and the common host run | `place_hybrid` loaders that do not opt in, by name (`PlaceError::NoNvmeTier`) |
| unified memory (one pool) | deepseek41, glm5next, qwen4exp | the rest never ask |

Refusal causes, by class:

| Class | Where | Families |
|---|---|---|
| architecture fact | no routed experts; the body runs every layer on one card (`crates/model/src/arch/hybrid.rs:480`) | nemotron_h |
| missing tier leg (no `LegFamily` adapter, no `tier_reserve`) | `place_hybrid.rs:236-245` `PlanInputs::reserve_on` → `PlaceError::Tier` | mimo2, minimaxm3, raon1; musespark until muse2card lands; nemotron_h_moe via `hybrid.rs:564` |
| missing quant/activation binding on the common card leg | `accepts.rs` `leg_arms`/`routed_rule` data | raon1 (no Q6_K down: `resident_down_runs`); nemotron_h_moe (ungated relu² up/down) |
| missing runtime piece (plan and leg exist) | `serve_seats/qwen3.rs:1014-1019`: "the body runs no pass of several slots beside an expert tier card" | qwen3moe/qwen35moe seat |
| missing chooser (bp served, unset never offers it) | `Place::choose_untiered` (`crates/gpu-gates/src/generate.rs:565`) | solar2 (`serve_seats/solar2.rs:42`), and every `Tiered` seat until muse2card's `TIER_RULE` hook lands |
| missing reserve owner (bytes placed outside the plan or on the wrong card) | the scalar reserve; `TierDraft.on`; post-plan buffers; off-plan seats | deepseek41 (DSpark under `a`/`gate`), glm5next, solar2, qwen3.6 DSpark, musespark, mimo2, nemotron_h_moe, every vision/embedding seat |
| missing NVMe runtime binding | `PlaceError::NoNvmeTier` | `place_hybrid` loaders without `with_nvme_tier`; minimaxm3, musespark and solar2 opt in |

The lead's list of tier-refusing models was minimaxm3, musespark, mimo2, raon1, nemotron_h, solar2. **solar2 does not
refuse**: its description carries `with_tier_reserve` (`crates/model/src/arch/solar2/place.rs:190-194`), its test plans
`plan_bp` (`:253-290`), and its seat serves `--place bp` (`serve_seats/solar2.rs:38-46`); only its unset rule is
untiered. nemotron_h_moe belongs on the list (`hybrid.rs:564`). And `crates/gpu-gates/src/bin/shared/routed_legs/*` are
the adapters of the `gate_routed_legs` plan-legality gate (`crates/gpu-gates/src/bin/gate_routed_legs.rs:13-17`), not
tier legs: the tier leg is already common (`crates/gpu/src/host/leg_tier.rs` `LegTier` over
`crates/gpu/src/card_leg.rs`), each family giving a `LegFamily` adapter of ~80 lines.

## 2. Reserves: where each one lands and whether the plan counts it

| Reserve | Family | Device it lands on | Counted? | Where |
|---|---|---|---|---|
| DSpark draft, `--place bp` | deepseek41 | the tier card (named `DRAFT_RESERVE`) | yes, on that card | `workstation.rs:185 tier_card`, `:332`, `:487 plan_bp` (`TierDraft{on:0}`) |
| DSpark draft, `--place a` on two cards | deepseek41 | the 3090, **outside the plan** | no | `generate.rs:2550 draft_spec` (unset: the fewest-bytes visible card); `Place::machine` refuses draft bytes with no tier (`generate.rs:749-759`), so `ds41_dspark.rs:76 draft_reserve` returns `None` |
| DSpark draft, `gate` or one card | deepseek41 | the stage card itself | no (8.6 GB against the 1 GiB margin) | `generate.rs:2541-2548` doc: "the stage itself under `gate`" |
| tier prompt batch | deepseek41, glm5next, qwen4exp, qwen3moe, solar2, musespark | tier card (named `TIER_BATCH_RESERVE`) + host rows | yes | `workstation.rs:337,356,465 tier_leg_batch_bytes` |
| prompt front + tiles + units | glm5next | **every card** (scalar), tier included | yes, twice: on the stage and again on the tier | `glm5next/place.rs:343 prompt_reserve_bytes`; `:670 loaded_routed(.., front + beside, ..)`; `:1044 plan_routed_reserving` |
| NextN draft card bytes + arena (16 MiB) | glm5next | every card (scalar) | yes, on every card | `glm5next/place.rs:736 NEXTN_ARENA_BYTES`, `:1044` |
| NextN draft layer's host experts | glm5next | host, beside the plan | checked, not planned | `record.rs` `residency_host_beside` (`ChurnPool::check_beside`) |
| MTP draft | qwen4exp | one card: scalar on the card; tier machine: named `MTP_RESERVE` on the stage | yes | `qwen35moe/place.rs:744-800 plan_mtp_with`, `:1255` |
| checkpoints, point rows | qwen4exp | host (named) | yes | `workstation.rs:342,347`; `serve_seats/qwen38.rs:612-630` |
| resident slots, sequence tables | qwen3moe, qwen4exp media | stage (named) | yes | `shared/qwen3moe_place.rs:202,282`; `crates/gpu-gates/src/image_plan.rs:18` |
| prompt reserve + tier places | solar2 | **every card** (scalar), tier included | yes, twice under bp | `solar2/place.rs:190-194`; pin `crates/model/tests/solar_open2_meta.rs:271` (558,471,268 B) |
| tier places | musespark (muse2card) | every card (scalar) | yes, twice under bp (≤ 0.2 MB [derived]) | muse2card `musespark/place.rs` `tier_places_bytes` |
| prompt-pass buffers | musespark | stage, post-plan (`choose_pass_rows`) | **no** (~0.6 GB, the lead's measurement) | `crates/gpu/src/arch/musespark/body.rs:854`, `body/prefill.rs:176-205` |
| prompt batch | mimo2 | stage, post-plan | **no** ("the load's plan reserves none") | `crates/gpu-mimo2/src/body.rs:1320`, `prefill.rs:425-435,549-584` |
| prompt batch | minimaxm3 | stage, inside the plan's scratch | yes (`with_prompt_reserve(0)`, the batch in `scratch_bytes`, held by `refuse_scratch_past`) | `minimaxm3/place.rs:43`; `crates/gpu-minimaxm3/src/body.rs:587,664`; the seat's doc says the opposite (`serve_seats/minimaxm3.rs:42-44`) |
| prefill buffers | nemotron_h_moe | stage, post-plan | **no** | `crates/gpu/src/arch/nemotron_h_moe/body.rs:346` |
| DSpark draft | qwen3.6 (qwen35moe on the qwen3 seat) | stage, whole-card load only, from free room | **no** | `serve_seats/qwen3.rs:686`; `bin/generate_qwen3moe.rs:2292-2305` |
| dflash2 draft | raon1 | stage (named `DRAFT_RESERVE`) | yes | `serve_seats/raon1.rs:666-670` |
| image encoder / embedding model, a card off the plan in view | every vision/embedding seat | the smallest card **outside the plan** | **no** (reserve 0, `seat=off-plan`) | `crates/gpu-gates/src/vision_seat.rs:659-690 seat_of`; `side_seat.rs:102-114` |
| image encoder / embedding model, no card off the plan | every vision/embedding seat | stage (named, `beside::reserve_on`) | yes | `crates/gpu-gates/src/beside.rs:41` |

Two consequences the table shows and no record prints today:
- **The scalar reserve taxes the tier.** Solar under `bp` [derived]: (558,471,268 + 12 × 512 × 8) / 8,847,360 B an
  expert ≈ 63 experts the 3090 does not hold. GLM loses its front + NextN + beside bytes on the 3090 the same way
  (bytes not computed here; the stage-1 FAIL-first prints them). Qwen3.8 already avoids it by naming `MTP_RESERVE` on
  tier machines (`qwen35moe/place.rs:744-800`: "which the tier's budget does not see"). On a pool the scalar is counted
  once per card (`placement.rs:2472-2477`).
- **The 3090 under `a` carries what nothing sums.** The DSpark draft (8,629,780,480 B [derived from the pins below])
  and an off-plan image tower can both land on it, and no plan or check adds them.

The `plan` record (`crates/gpu-gates/src/record.rs:3392`) prints the stage card's experts, the host, `n_l`, budget,
paged/arena and the device list; it prints no reserve and no tier expert count.

## 3. Per family

Pins used below (`crates/model/tests/placement.rs`): V4.1 plan (a) A6000 2,668 experts at 44,750,684,160 B, so
16,773,120 B a card expert; `bp` tier 1,392 experts without the draft (reserve 152,511,496 B, the tier batch) and 881
with it (8,782,291,976 B), so the draft holds 8,629,780,480 B; plan (b) A6000 2,814, 3090 1,265; `gate` 3090 1,146
(`docs/HARDWARE.md`).

### deepseek41 (DeepSeek V4.1-Flash)
- Shapes: one card (`plan_a` `workstation.rs:290`, `plan_gate` `:647`); stage + tier (`plan_bp` `:487`, `plan_tiers`
  `:509` for any resolved pair); layer cut (`plan_b` `:317`, gate only); nodes (`plan_node_tiers` `:541`,
  `deepseek41/place.rs:217`); NVMe (`:228`); pool. Own `PlanInputs` with no scalar reserve (`nodes::loaded_routed(..,
  0, ..)`, `deepseek41/place.rs:217`).
- Tier leg: its own `Ds41Tier` (`crates/gpu-deepseek41/src/chain/ffn/tier.rs:816`), binding `DS41_LEG`; refuses N ≥ 2
  tiers before upload (`crates/gpu-deepseek41/src/body.rs:2367`).
- Refusals: draft bytes with no tier (`generate.rs:756`, "no tier card holds a draft reserve; the draft's card is
  outside the plan"); `plan_a_draft` → `DraftOffPlan` (`workstation.rs:300`, `placement.rs:1213`) because `TierDraft.on`
  is a tier-list index (0 = "the first tier" and, with no tier, "off the plan").
- Reserves: §2 rows 1-4, beside.
- Chooser: `Place::choose` with `TIER_RULE` break-even `None` (`shared/ds41_place.rs:30-48`) → unset is `a`
  (README: "`--place bp` is not faster than one card"; `docs/cards/v41bp-ab.card`).
- Seat: `BLOOMERY_DSPARK_CARD` may name only the tier under `bp`, any visible card otherwise (`generate.rs:2537-2548`).

### glm5next (GLM-5.3-Flash)
- Shapes: one card; stage + tier; tp; nodes (`glm5next/place.rs:645-707`); NVMe (`:686`, `:1062`); pool.
- Tier leg: common, `impl LegFamily for Glm` (`crates/gpu-glm5next/src/tier.rs:39`), binding `planes_leg_*`; a form the
  tier cannot run is refused (`refuse_off_tier`, `glm5next/place.rs:435`).
- Reserves: prompt front, NextN and beside as the scalar on every card (§2); tier batch named (`tier_batch` `:166`).
- Chooser: `TIER_RULE` break-even 526 (`shared/glm5next_place.rs:25-33`, `docs/cards/glmbp-ab.card:42`); seat `glm.rs:886`.
  The gate pins the 3090's `bp` experts at 1503..=1529 (`bin/gate_glm5next_twocard.rs:236`).

### qwen4exp (Qwen3.8-Flash-Next; `qwen35moe/place.rs`)
- Shapes: one card (`placed_on`, `qwen35moe/place.rs:1249`); stage + exactly one tier (`machine_bp_on` `:1357-1374`;
  `shared/qwen38_place.rs:55-71`); nodes (`:479`); NVMe (`:494`, `:687`); pool and larger cards
  (`crates/placement/src/placement/qwen38_cards.rs`).
- Tier leg: its own `Tier38` (`crates/gpu/src/arch/qwen3moe/tier38.rs:561`).
- Refusals: a host expert rule on a tier machine (`TierHost`), a tier with no leg (`Tier`), an MTP reserve of other
  bytes (`DraftReserve`) — `qwen35moe/place.rs:234-267`.
- Reserves: MTP two ways (scalar on one card, named on the stage of a tier machine); checkpoints and point rows on the
  host; beside and tables on the stage; tier batch on the tier (`leg_tier_batch` `:1266`).
- Chooser: `TIER_RULE` break-even 1,152 (`shared/qwen38_place.rs:19-41`, `docs/cards/q38bpbug-ab.card:76`).

### qwen3moe / qwen35moe (Qwen3-30B-A3B, Qwen3.6-35B-A3B; `qwen3moe/place.rs`)
- Shapes: one card (`placed_with_arena`); stage + one tier in the plan (`plan_tiers` + `stage_placed`,
  `qwen3moe/place.rs` doc 1-25). No NVMe, nodes or pool term in this file.
- Tier leg: common, `impl LegFamily` for Qwen3 and Qwen35 (`crates/gpu/src/arch/qwen3moe/tier_leg.rs:124,187`).
- Reserves: resident slots and tables on the stage (named); Qwen3.6's DSpark from free room after a whole-card load
  (not counted).
- Chooser: `TIER_RULE` break-even `None` (`shared/qwen3moe_place.rs:108-131`); the serve seat uses
  `choose_untiered` (`serve_seats/qwen3.rs:64`) and refuses a tier (`:1014-1019`); README: "`--place bp` is refused by
  name on the `nemotron` seat, Qwen3.6 and Qwen3-30B".

### solar2 (Solar-Open2; `place_hybrid`)
- Shapes: one card; stage + tier (above). NVMe refused (`place_hybrid.rs:318`).
- Tier leg: common, `impl LegFamily for Solar2` (`crates/gpu-solar2/src/tier.rs:41`), binding `card_gate_up_runs` /
  `card_down_runs`.
- Reserves: prompt reserve + tier places as the scalar (≈ 63 experts off the 3090 under `bp` [derived]); tier batch
  named (`solar2/place.rs:52`). The seat's doc says the prompt batch comes from free bytes (`serve_seats/solar2.rs:34-37`)
  while the description reserves it at `PROMPT_ROWS` (`solar2/place.rs:190`): one of the two is stale.
- Chooser: `choose_untiered` ("no sitting has shown the tier not slower", `serve_seats/solar2.rs:42`); seat refuses
  more tier cards than the host serves (`refuse_tier_count`).

### musespark (Muse Spark Open; `place_hybrid`)
- Base: one card only; tier refused (`PlaceError::Tier`), seat `choose_untiered` (`serve_seats/musespark.rs:49`).
- muse2card (in flight): `with_tier_reserve(tier_places_bytes)`, `tier_batch` through `tier_leg_batch_bytes`,
  `pass_rows_max` as the one formula of the pass rows, a tier body over `crate::host::leg_tier`
  (`crates/gpu/src/arch/musespark/body/tier.rs`, IQ2_XXS gate·up with SiLU and IQ2_XXS down in the latent), and in the
  common seat a `Tiered::TIER_RULE` (default `TierRule::UNTIERED`) and `Tiered::machine` hook
  (`serve_seats/tiered.rs`); Muse's `TIER_RULE` is break-even `Some(1)` with basis `docs/cards/musespark-2card.card`.
- Reserves: tier places (scalar, ≤ 0.2 MB on each card [derived: 4 × (62 × 256 + (1 + rows) × 8) B, rows ≤ 4,096]);
  the prompt pass post-plan (~0.6 GB, uncounted).

### mimo2 (MiMo-V2.6-Flash; `place_hybrid`)
- One card; tier refused (`mimo2/place.rs:127-130`); seat `machine_untiered` (`serve_seats/mimo2.rs:54`).
- Needs: `tier_reserve`, a `LegFamily` adapter, `tier_batch`, a `TIER_RULE`; binding data exists
  (`card_gate_up_runs`/`card_down_runs`). Prompt batch post-plan (uncounted).

### minimaxm3 (MiniMax-M3.1-Flash; `place_hybrid`)
- One card; tier refused (`minimaxm3/place.rs:164-167`); NVMe refused; seat untiered (`serve_seats/minimaxm3.rs:50`).
- Needs: as mimo2, with the NVFP4 planes binding (`planes_leg_*`) on the tier. Prompt batch counted in scratch.

### raon1 (`place_hybrid`)
- One card; tier refused (`raon1/place.rs:20-46`); "No host tier serves a Raon-1 replay" (`crates/gpu/src/arch/raon1/rows.rs:206`);
  seat `choose_untiered` (`serve_seats/raon1.rs:649`).
- Needs: as mimo2, plus a Q6_K down form on the card leg (`resident_down_runs` lists none). Its dflash2 draft is the
  one draft today already named on the stage (`serve_seats/raon1.rs:666-670`).

### nemotron_h (Nemotron-H, dense hybrid)
- One card (whole or placed, `serve_seats/nemotron.rs:362-390`); a tier refused by an architecture fact
  (`hybrid.rs:480`: no routed experts for a tier to hold). Correct as is.

### nemotron_h_moe (Nemotron-H MoE)
- One card; tier refused by the shared hybrid rule (`hybrid.rs:564`) though the model has routed experts (latent,
  IQ2_XXS up/down, relu², no gate). Needs an ungated activation arm on the common leg as data, a `LegFamily`, a
  tier reserve. Prefill buffers post-plan (`nemotron_h_moe/body.rs:346`).

## 4. Where the places are called

`Place::machine` callers (one dispatch site per seat or gate): `bloomery_chat.rs`, `gate_deepseek41_twocard.rs`,
`gate_ds41_callstream.rs`, `gate_ds41_nvtier.rs`, `gate_ds41_serve.rs` (3), `gate_mimo2_serve.rs`,
`gate_solar2_twocard.rs`, `generate_ds41.rs`, `shared/ds41_place.rs`, `routed_legs/deepseek41.rs`,
`serve_seats/{ds41 (2), hybrid, minimaxm3, musespark, tiered}.rs`, `residency38.rs`; `machine_placed` /
`machine_untiered`: `gate_nemotron_serve.rs`, `gate_qwen38_nodes.rs`, `serve_seats/{mimo2, nemotron, raon1}.rs`;
qwen38's own `machine_bp38` (`shared/qwen38_place.rs:55`). `beside::reserve_on` callers: `serve_seats/ds41.rs:648,910`,
`serve_seats/{qwen38.rs:612, raon1.rs:671, nemotron.rs:378,383, decide.rs:221,593}`, `shared/ds41_place.rs:108`,
`shared/qwen38_place.rs:102`, `shared/qwen3moe_place.rs:288,877`, and `serve_seats/tiered.rs` in muse2card's tree.

Gate reach (`python3 tools/recipes.py reach --brief`, build-free): `crates/placement/src/placement.rs` and
`placement/workstation.rs` 205 of 210 gates; `crates/gpu-gates/src/record.rs` 70; `generate.rs` 69;
`crates/model/src/arch/place_hybrid.rs` 67; `glm5next/place.rs` 59; `beside.rs` 37;
`crates/gpu/src/arch/musespark/body/prefill.rs` 19; `shared/ds41_dspark.rs` 23 bins read it as their own.
