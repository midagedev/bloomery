# Adaptive picks: where else choosing by the workload pays

A design survey (2026-10-11, Mac only: no box run, no code change). The question: after the per-script MTP head
decision, where else does one static choice, made for every workload, leave speed on the table, and would a pick from
the request, or from counts the engine already keeps, take it back?

Every number is cited, or marked `[derived]` with its terms. Paths are this repo's unless prefixed `rig-log`
(`~/repo/rig-log`).

## 0. Result

| Rank | Candidate | Fact the pick reads | Step a user sees, A6000 | 12 GB card + host | DGX Spark | Verdict |
|---|---|---|---|---|---|---|
| 1 | Hybrid prompt path by prompt length (§3.4) | P (request) + lane rate (load probe) | 0 (file fits whole) | TTFT 10.1 s → 0.6–1.8 s at P 1464 `[derived]` | 0 (one pool) | **open**, the triage's hybrid-prompt design round (`q35hybpp`) |
| 2 | KV format by planned depth, Qwen3 (§3.3) | planned ctx (config), depth (run-time) | decode +4..+5 % @4K, +14..+17 % @16K, +25..+28 % @32K `[derived]`; lower edge 0 | ×1.88 context on a qwen3moe file | same % as A6000 `[derived]` | **open** `q3kvdepth` (one depth A/B) |
| 3 | V4.1 lookup draft on by default behind the chooser (§3.2) | run-time count (the chooser's walls and kept rows) | threads and code 0..+10 %, prose ~0 `[derived, loose upper bound]` | not derived | not derived | **open** `v41lookup` (one A/B) |
| 4 | Width-chooser state keyed by the head class (§3.1) | request fact (the per-script head) | 0..+2.5 % on a request after a class switch, 0 otherwise `[derived]` | same | same | **open**, folded into the per-script head round |

Counts: **4 open, 6 idea ledger, 7 model says 0, 1 owned elsewhere** (§2). For 0.3.1, ranks 3 and 4 fit: each is S
once one term is read. Rank 2 fits only if the Qwen3 e2e count pin takes a q8_0 arm (§3.3.5). Rank 1 is M–L, so its design round can open now, but it lands after 0.3.1 unless the design
comes out small.

## 1. What a new pick is measured against

The engine already picks by workload in more places than its static defaults suggest. A new pick gains only what the
existing adapter loses, either in its convergence transient or by mixing workloads. Every row below is priced against
the adapter, not against the static default it replaced. Rows priced against "static" would all come out overstated.

| Existing adaptive pick | Fact | Where |
|---|---|---|
| Draft width per window, and the gate (draft on/off) | run-time: median pass walls by width, the draft's `p_i` × an online calibration `g_i` | `crates/runtime/src/width.rs:1-66`, `Rule::DEFAULT` `:173-181` (margin 1.03, 8 readings, calibration step 0.05, probe 64/256, shadow every 16th closed pass) |
| The gate's mean profile | run-time EMA: step `max(1/seen, 0.05)`, about a 20-window memory | `width.rs:780-781` |
| Expert residency | run-time: routed counts, every 4 kept rows, decay 0.9 | `crates/runtime/src/swaprule.rs:57-68` |
| Expert stream and admission in a prompt unit | request (the ubatch width) + load probe (the lane rate) | `crates/runtime/src/xsplit.rs:1-12` (model-agnostic by its header), `splitwalk.rs:133` `pays` |
| V4.1 host stream in a prompt call | request: P ≥ 128 | `crates/gpu-deepseek41/src/body/prefill.rs:151-155` |
| Wide walk vs passes | request: P ≥ pass rows + 1 | `crates/gpu/src/arch/qwen3moe/body38.rs:318`, `crates/app/src/arch/qwen3moe/mod.rs:92` |
| Keep a prefix, or reset with the draft on | request: `max_tokens` against k* = R·Δ·ρ | `crates/gpu-gates/src/bin/shared/serve_seats/qwen38.rs:2252`, rig-log `docs/bloomery/plan.md:120-129` |
| Logits readback only for sampled, banned or `logprobs` requests | request | `crates/serve/src/genloop.rs:1561`, `crates/gpu-gates/src/bind.rs:2042-2045` |
| Ubatch arena rows ≤ ctx | config | `crates/gpu/src/arch/qwen3moe/body35.rs:148-153`, `body38.rs:1014` |

The binding lesson is the retired router-frequency expert list (offline ranking), retired because its list was learned in-sample (`crates/levers/src/registry.rs:230-243`).
It applies to existing defaults too, and two of them fail it today. The residency rule's constants (§3.8) were chosen by
replay on the 09-23 router traces, and the table that reports them was computed on those same traces (rig-log
`log/2026-09-28.md:136-150`). `NOMINAL_REPLY = 277` (`qwen38.rs:398`) is the mean reply of the same Korean prompts the
shipped head list was chosen on (§3.11).

## 2. Candidate table

| # | Where (static choice today) | Fact (kind) | Term moved | Prediction | Verdict |
|---|---|---|---|---|---|
| 3.1 | `width.rs:448-500`: one chooser state per slot (`drafted.rs:100-107`) or per process (`ds41.rs:1120`) across requests; `drafted.rs:351-358` restarts the draft, not the chooser | request: head class | windows priced on the old class's profile and costs | 0..+2.5 % after a switch `[derived]` | open (head round) |
| 3.2 | `registry.rs:334` `DRAFT` unset on V4.1 = plain | run-time: the chooser | positions per pass | threads/code ≤ +10 %, prose ~0 `[derived]` | open `v41lookup` |
| 3.3 | `registry.rs:769` `QWEN3_KV` unset = f16 | config/run-time: depth | KV bytes a step | +4..+28 % by depth `[derived]` | open `q3kvdepth` |
| 3.4 | `body35.rs` `placed_open`: a placed load runs a prompt as units of 512 rows; the 8-row reading below is the walk before that | request: P; probe: lane rate | host-union ms → PCIe + card | 12 GB: TTFT ÷ 5–15 `[derived]` | open `q35hybpp` (triage item) |
| 3.5 | `width.rs:622` `round_ran`: a multi-slot round verifies every slot's whole width | run-time: live slots, each slot's profile | rows a round | 0 on matched slots; mixed pairs unknown | ledger |
| 3.6 | Qwen3.8/GLM: MTP is the only proposer | run-time: n-gram hit beside MTP | positions per window | ≤ +11.4 % ceiling (Qwen3.8) `[derived]` | ledger |
| 3.7 | every process starts residency from the id-prefix seed | the user's own routing, persisted | card hit rate in the transient | one-shot runs only; cross-topic carry unknown | ledger (Mac replay) |
| 3.8 | `swaprule.rs:57-68` `mid` constants from an in-sample replay | held-out traces | hit rate | unknown | ledger |
| 3.9 | `body38.rs:1014` arena = min(4096, ctx) for the process | run-time: observed prompt lengths | card expert bytes vs pp | unknown | ledger |
| 3.10 | `lookup.rs:85` lookup reports p = 1 for every match length | run-time: hit rate per n | per-window profile | ≤ the lookup gain | ledger (with 3.2) |
| 3.11 | `qwen38.rs:398` `NOMINAL_REPLY = 277` (in-sample) | run-time: running mean of replies | keep/reset at k* | ~0 (second order) | model says 0 |
| 3.12 | `prefill.rs:155` V4.1 `STREAM_MIN_P = 128`, not priced by xsplit | request + probe | prompts under 128 | ≤ 0.13 s TTFT `[derived]` | model says 0 |
| 3.13 | `GEMM_FROM` = pass rows + 1 (three families) | request | bytes a prompt | 0 by construction | model says 0 |
| 3.14 | `crates/gpu/src/cores.rs:395-404` Q4_K gemv at K = 512, half the lanes idle | shape (static) | lane idle | 0 on A6000 (`docs/research/sparkdec.md` §4.3) | model says 0 |
| 3.15 | `genloop.rs:291-297` sampled requests draft on qwen38/glm | request: temperature | positions per pass | 0 beyond the per-slot chooser | model says 0 |
| 3.16 | `crates/gpu/src/host/nvtier.rs:99-102` 16 fill threads | drive (not workload) | — | 0 for workload | model says 0 |
| 3.17 | `registry.rs:808` `HOST_LANES` ccd, `:791` `LANE_PREFETCH` off | machine (not workload) | — | 0 for workload | model says 0 |
| 3.18 | `ds41.rs:246`, `qwen38.rs:1291`, `glm.rs:784`, `qwen3.rs:945`: 2 slots by default; `crates/runtime/src/seqstate.rs:41` 4 GiB of checkpoints a slot | run-time: concurrent requests | context a slot, host bytes | capacity: the 3060's 6K-prompt refusal (rig-log `docs/bloomery/plan-triage.md:92-94`) | owned elsewhere (KV pool work) |

**Lever registry sweep** (`crates/levers/src/registry.rs`, parsed rows). These defaults depend on the workload:
`DRAFT`, `MTP_HEAD_ROWS`, `MTP_WIDTH` (already adaptive), `RESIDENCY` (already adaptive), `HOSTSTREAM`, `XSTREAM` and
`SPLIT_WALK` (already prompt-width rules), `QWEN3_KV` (depth), `GEN_SLOTS` (a generator lever; the serve seats read
`--parallel`), and `PREFILL_GROUP` (prompt length, below). These are machine or file facts with no workload term:
`THREADS`, `SPIN`, `PIN_MAIN`, `HOST_LANES`, `LANE_PREFETCH`, `R8`, `HOST_POPULATE`, `HOST_LOCK`, `CARD_DONTNEED`,
`CARD_BUDGET`, `HOST_ROOM`, `NVTIER_BYTES`, `NVTIER_READ`, `QWEN38_EXPERTS` and `CED`. Diagnostics carry no speed
term. `PREFILL_GROUP` (`:176`, 2) matters only for a prompt of more than one batch, and a group past 2 takes card
margin. With no measured gain past 2, the model gives it no workload term.

## 3. Candidates

### 3.1 Width-chooser state keyed by the head class (completes the per-script head)

1. **Where.** The chooser keeps its gate, its calibration `gains`, its mean profile and its cost medians in one
   `Choosing` (`crates/runtime/src/width.rs:448-500`). That is one per slot on Qwen3.8 and GLM
   (`crates/gpu-gates/src/bin/shared/serve_seats/drafted.rs:100-107`) and one per process on V4.1
   (`serve_seats/ds41.rs:1022`, `:1120`). A request's reset prints the tally and restarts the inner draft only
   (`drafted.rs:351-358`), so all of that state crosses requests.
2. **Fact.** The head class the per-script pick chose for this request. This is a request fact.
3. **Decomposition.** The per-window profile is `p_i · g_i` (`width.rs:10-18`). The `p_i` term is instant. The
   `g_i` term and the gate's mean profile step 0.05 per window (`width.rs:780-781`), an e-fold of about 20 windows. A
   reduced head renormalizes the draft's probability over its rows, so `g_i` differs by head. It also reads fewer
   head bytes per walk, so the cost medians differ too. The full Q8_0 head is 675.4 MB a walk, 3 walks a window
   (rig-log `docs/bloomery/plan-triage.md:2199`). A reply of 277 tokens at 3.59 positions a window (rig-log
   `log/2026-09-30.md:181`, Qwen3.8 (a), prose) is about 77 windows `[derived]`, and 20–40 of them run on the previous
   class's state after a switch. If a mis-rated window picks one width off the optimum and loses 0–10 % of its rate
   (assumed; the `mtp width` record's cost medians by width fix it), a request after a switch loses 0..2.5 %
   `[derived from the assumption]`. A stream of one class loses 0. On Qwen3.8 the gate stays open
   in both classes (drafted 79.33 against plain 57.88 tok/s at P 4096, `log/2026-09-30.md:177-179`), so the
   closed-gate lag (first probe after 64 passes) does not arise there.
4. **Timeline.** The width sets the rows of a pass: the card leg plus the host union. It is on the critical path,
   but the term is second order around the optimum.
5. **Cost and risk.** S, in `runtime::width`, the owner all three families already use: the state becomes a small
   map keyed by the head, switched at `begin`. Tokens do not change, only the rows a pass runs. The exception is a
   running residency, where acceptance moves the passes, the flips, and the target's bits at a near tie (the
   `MTP_HEAD_ROWS` row, `registry.rs:389-415`). The pin is a `width.rs` unit test on the mock clock with two classes
   alternating. FAIL-first: without the key, the second class's first window takes the first class's width.
6. **Verdict.** Open, inside the per-script head round, as one more field in the common owner the user already
   named. It does not justify a separate round.

### 3.2 V4.1: the lookup draft on by default, behind the chooser

1. **Where.** `DRAFT` unset on `generate_ds41` and `bloomery-serve-ds41` is the plain path
   (`crates/levers/src/registry.rs:334-345`). The lookup draft proposes one id (`crates/runtime/src/lookup.rs:85`),
   because V4.1's pass holds two rows (`crates/gpu-deepseek41/src/body/slots.rs:35`).
2. **Fact.** A run-time count: the chooser's own pass walls and kept rows (`MTP_WIDTH` unset is `cost`,
   `registry.rs:455-475`) decide per window and gate the draft off where it does not pay.
3. **Decomposition.** Open gain per window = (1 + a) / (c₂/c₁) − 1, where a is the acceptance and c₂/c₁ the
   two-row pass's cost over a plain step. The break-even recorded on 09-24 was a = 0.34 (rig-log `log/2026-09-24.md:43`,
   so c₂/c₁ = 1.34 then). The predictability of the corpus text itself (`log/2026-09-24.md:44-50`) is code 0.595,
   threads 0.477, prose 0.382 and korean 0.288. That is the predictability of the text, not the engine's acceptance on
   model output. Taking it as a: code +19 %, threads +10 %, prose +3 %, korean gate closed (0) `[derived, loose upper bound]`.
   An engine A/B at fixed width (sitting 9, 09-24) was reported far under this bound on code; its row is not in
   rig-log `log/2026-09-24.md`, so this doc does not use its numbers. The predicted band is therefore threads and
   code 0..+10 %, prose ~0.
   A closed gate costs one probe verify per 256 passes at 0.34 extra (+0.13 %) `[derived]`. The shadows are a
   host-side hash lookup. The term derivation cannot fix is **c₂/c₁ today**: residency has since moved V4.1's
   host leg, and nothing recorded re-reads the pair ratio under the serving default (`mid-p0-s1`, `--place a`).
4. **Timeline.** V4.1 decode under `--place a` is bound by the host union. The second row's routed experts overlap
   the first row's only partly, so c₂/c₁ is a host-DRAM term on the critical path.
5. **Cost and risk.** S: the unset word in `bloomery_levers` for the ds41 seats. The chooser already wraps
   `Lookup` (`ds41.rs:1022`). The greedy ids are the plain run's, except under the running residency (as in 3.1).
   Under the default two slots (`ds41.rs:246`), a round of two holds no draft row, so lookup acts only while one slot
   runs alone. DSpark is not this row: it holds 8.5 GB resident and its own card (`BLOOMERY_DSPARK_CARD`).
6. **Verdict.** Open `v41lookup`: one lease A/B, same binary, unset against `lookup`, on code, threads and prose
   prompts from tapes not in the 09-23/09-24 corpora. The card's term is threads (+2..+10 %); a lower edge inside
   the ruler needs six rounds and an A/A arm. Flip the unset word where the threads row clears it and prose stays
   inside the ruler.

### 3.3 KV format by planned depth (Qwen3, dense attention)

1. **Where.** `QWEN3_KV` unset is f16 (`registry.rs:769-789`). The q8_0 form is two planes at 17/16 B a value
   against f16's 2 B. It is read once at load, so a request cannot pick it.
2. **Fact.** The context the server plans (`--ctx-size`, or the automatic answer) and the depth the session reaches.
   These are config and run-time facts, not per request.
3. **Decomposition.** Qwen3's cache per position is 48 layers × 4 KV heads × 128 × 2 planes × 2 B = 98,304 B in f16
   (dims in `docs/research/modelspec-design.md:455-461`) and 52,224 B in q8_0. At ~700 GB/s each decode step saves
   46,080 B × D: 0.27 ms at D 4,096, 1.08 ms at 16,384 and 2.16 ms at 32,768 `[derived]`. The f16 KV read alone takes
   0.58, 2.30 and 4.60 ms. The shallow step is not measured on the A6000: 5.2–6.3 ms (`[derived]`; the 3090's 193.9
   tok/s scaled to 768 GB/s is ~5.6 ms, `docs/research/sparkdec.md` §4.2). As tok/s, t_f16 / t_q8 − 1, that gives
   +4..+5 % at 4K, +14..+17 % at 16K and +25..+28 % at 32K `[derived]`. The one term derivation cannot fix is **whether the q8_0 flash reads at the f16 flash's byte
   rate**. If the dequant binds, the lower edge is 0. Capacity: the seat's automatic context reaches ×1.88 (the row's
   doc).
4. **Timeline.** A whole-card load, so the step is card DRAM. KV and weights are the same resource, and the term is
   on the critical path. On GB10 both terms are bytes from one pool, so the same percentages hold `[derived]`. On a
   qwen35moe file (Qwen3.6: 40 layers, 2 KV heads × 256, full attention on a fraction of layers) the cache term is
   about a fifth of Qwen3's or less `[derived, if one layer in four is full attention]`, so it moves little there.
5. **Cost and risk.** S for the pick itself: the seat's `KvQ8` pick reads the planned ctx, or the default flips
   outright if q8_0 is never slower. Tokens change, because a q8_0 run's bits are its own. Under "performance first,
   accuracy opt-in" the default may take it, and σ against `exact_ref` stays a diagnostic. The flash gate pins the
   q8_0 read path (`crates/gpu-gates/src/bin/gate_qwen3moe_flash.rs:95-98`). This pass found no q8_0 arm of the e2e
   count pin. If there is none, a default flip also needs that arm and its reference run (S–M). Qwen3.8 supports
   q8_0 for K and V; its raw and pooled index keys stay f16. Unset, a checkpoint folder's cache is f16 like a GGUF
   file's (`crates/gpu/src/kv_format.rs`).
6. **Verdict.** Open `q3kvdepth`: one depth A/B (`tools/ref/depth-decode.sh`, `BLOOMERY_DEPTHS="1024 4096 16384"`,
   arms `QWEN3_KV=f16|q8_0`). If q8_0 is never slower, flip the default outright. If it loses shallow and wins deep,
   pick by the planned ctx at the measured crossover.

### 3.4 Hybrid prompt path by prompt length (qwen35moe placed loads)

1. **Where.** ~~A placed qwen35moe load runs every prompt as passes of at most 8 rows through the host tier's batch
   port, whatever its length.~~ A placed qwen35moe load now walks a prompt in units of `PLACED_ROWS` = 512 rows on
   the wide arm, each unit reading a host expert once (`crates/gpu/src/arch/qwen3moe/body35.rs` `placed_open`,
   `placed.rs`). `--prefill pass` uses passes of at most 8 ids; `auto` also uses a pass for a prompt or final tail of at most 8 ids. The
   decomposition below prices the old 8-row walk; for units of 512 the host union's compute is the wall, and
   `docs/cards/q35pfa.card` derives 325–510 tok/s at the 3060 shape, unmeasured there.
2. **Fact.** The prompt length P (request) and the lane's copy rate (the load's probe), priced by
   `runtime::xsplit`. No model or card name enters that rule (`crates/runtime/src/xsplit.rs:1-5`).
3. **Decomposition.** Measured on the 3060 test PC (RTX 3060 12 GB, WSL2): a 1,464-token prompt took 10.1 s TTFT, 145
   tok/s (rig-log `docs/bloomery/plan-triage.md:90-96`). That is 183 passes, each reading its rows' union of host
   experts. Streaming the host experts to the card once per ubatch costs B_h / R_pcie. Assume B_h = 9–12 GB of host
   experts (a ~21 GB Q4_K_M file less a 12 GB card's share) and R_pcie = 12–22 GB/s (PCIe 4.0 x16 under WSL2, not
   measured). That is 0.4–1.0 s a ubatch, overlapped layer by layer with 0.3–0.8 s of card compute. TTFT comes to
   0.6–1.8 s, or 800–2,400 tok/s `[derived]`, against 145 measured and the reference's ~550, self-reported on the
   same class of machine (triage `:95-96`). Break-even P* is where streaming bytes equal the pass walk:
   B_h / R_pcie = P × 6.9 ms, so P* ≈ 60–150 `[derived]`. Passes stay below P*. ik's fixed rule for this file is
   batch × 8 ≥ 32 × 256, that is batch ≥ 1,024 (§4). Our P* is lower only because our 8-row pass is slow, not
   because the cut is the same one. xsplit prices the cut per expert.
4. **Timeline.** Today the host CPU's union is the whole critical path. The stream moves the term onto PCIe and the
   card's SMs, which overlap one layer under the next. The A6000 term is 0 (Qwen3.6 fits whole), and so is GB10's
   (one pool, nothing to stream).
5. **Cost and risk.** M–L: an ubatch walk on the placed qwen35moe body, plus the xsplit call and the
   `host::xstream` ring that Qwen3.8 owns today. That is a common-owner lift, not a copy. The arena on 12 GB at U
   1,024 is about 0.7 GB (Qwen3.8's 2.84 GB at 4,096, triage `:1622`, scaled `[derived]`). The prompt's bits become
   the band's, as `XSTREAM`'s doc says. The pins are the placed qwen35moe prompt gate plus a band gate.
6. **Verdict.** Open, as the hybrid-prompt design round the triage already names (`q35hybpp`). This survey adds one
   constraint: the pick is P ≥ P* from the request, priced by xsplit from the probe, never a fixed batch threshold.

### 3.5 Width in multi-slot rounds

1. **Where.** A pass of several slots' windows runs the draft's whole width and is only counted
   (`width.rs:622` `round_ran`). Two slots are the default on qwen3, qwen38, glm and ds41 (§2, row 3.18).
2. **Fact.** Run-time: the live slots and each slot's own profile.
3. **Decomposition.** On Qwen3.8 (a) at P 4096, a window takes 3.59 / 79.33 = 45.3 ms against a plain step of 17.28 ms
   (`log/2026-09-30.md:177-181`). The three draft rows and walks then cost about 28 ms, ~9.3 ms a row solo
   `[derived]`. Two slots of 4 rows fill `MAX_PASS_ROWS` = 8 (`crates/gpu/src/model.rs:277`). A slot whose head
   misses its script, or that samples at a high temperature, keeps perhaps 0.3–1 of its 3 draft rows (assumed;
   `mtp width` records per slot fix it), and cutting them saves their share of the union. **The term:** the 8-row pass wall by rows (the `slots round` record under
   `BLOOMERY_STEP_STATS`). mistral.rs makes the same cut by count: DFlash depth 3 past 8 live sequences
   (`mistralrs-core/src/speculative/dflash.rs:50-56`).
4. **Timeline.** At P 4096 the host leg sits under the card's shadow on Qwen3.8 (`log/2026-09-30.md:181`), so a row
   that adds host experts may cost the card leg only.
5. **Cost and risk.** S–M in `runtime::width` (a per-slot cut within the round). Rows change, tokens do not, with
   the residency exception.
6. **Verdict.** Ledger. The deciding term is the two-slot pass wall by rows, Qwen3.8 (a). Predicted 0 on matched
   slots. Mixed pairs are not predicted until that term is read.

### 3.6 n-gram lookup beside MTP (Qwen3.8, GLM)

1. **Where.** The MTP draft is the only proposer on Qwen3.8 and GLM. ik runs staged drafts and an n-gram map beside
   them (§4).
2. **Fact.** Run-time: an n-gram hit at the window's position.
3. **Decomposition.** A Qwen3.8 window holds 4 positions and keeps 3.59 on prose (`log/2026-09-30.md:181`). A
   second proposer can add at most 0.41 a window, +11.4 % `[derived ceiling]`, and only where MTP misses and the text
   repeats. **The term:** P(MTP miss ∧ lookup hit) on held-out code and thread outputs. On the Mac it can be read
   from `mtp window` records (`BLOOMERY_MTP_WINDOWS`, `registry.rs:438-453`) and `tools/ref/draft-accept.py`.
4. **Timeline.** The proposal is host-side and the pass is unchanged, so the gain is pure numerator.
5. **Cost and risk.** M: two proposers behind one chooser. Tokens are unchanged, with the residency exception.
6. **Verdict.** Ledger (term above).

### 3.7 Residency map carried across process starts

1. **Where.** Each process starts from the id-prefix seed; `POST /residency/reset` returns to it
   (`registry.rs:568-700`).
2. **Fact.** The user's own routing, saved at exit and loaded at start: run-time counts, not our corpora.
3. **Decomposition.** The replay (rig-log `log/2026-09-28.md:136-150`, V4.1) gives card hits of 18.7 % on a fixed
   prefix, 39.9 % per request (d 8) and 77.8 % continuous. The log's rate: 26.7 → ~31 tok/s over a request's first
   96 tokens and ~50 tok/s steady `[derived there]`. A long-lived server reaches steady state once, so only a one-shot
   CLI run spends its life in the transient. **The term:** cross-topic carry, a map learned on one corpus and
   replayed on another (prose → code, code → korean). This is a Mac replay of the existing traces.
4. **Timeline.** Card hits move the host union, which is on V4.1's critical path.
5. **Cost and risk.** S–M (a slot-file field next to `--slot-save-path`). Bits move at near ties, as any residency
   does.
6. **Verdict.** Ledger (Mac-only term).

### 3.8 The residency rule's constants, held out

1. **Where.** `swaprule.rs:57-68` `mid`: every 4 kept rows, cap 24, margin 3, min count 2, decay 0.9, one spare.
   They were chosen by replay of the 09-23 traces and reported on those traces (rig-log `log/2026-09-28.md:139`).
2. **Fact.** Not a new pick, but the binding lesson applied to an existing default.
3. **Decomposition.** Six constants are far less able to overfit than a ranked list, and the replay's prose, code
   and korean rows sit within 3 points of each other (77.8 / 81.0 / 79.5 %). The 09-28 caveat stands: the traces are
   not the engine's chat routing. **The term:** the hit rate of `mid` against the best constants on a held-out trace
   set, namely serve route traces of real chat (`BLOOMERY_ROUTE_TRACE` on a serve seat).
4. Timeline: as 3.7.
5. Cost and risk: Mac replay only.
6. **Verdict.** Ledger.

### 3.9 Ubatch arena by observed prompt lengths

1. **Where.** The arena holds min(4096, ctx) rows (`body35.rs:148-153`; `body38.rs:1014`). That is about 2.84 GB a
   card at 4,096 (rig-log `docs/bloomery/plan-triage.md:1622`), held through decode.
2. **Fact.** Run-time: the prompt lengths a session actually feeds. Under the prompt cache most turns are short.
3. **Decomposition.** Shrinking to 1,024 frees about 2.1 GB of card for experts `[derived]`, but each extra ubatch
   re-reads every host expert, so pp at 4,096 drops. **The term:** Qwen3.8's decode slope per card GB (one
   `BLOOMERY_CARD_BUDGET` −2 GB arm).
4. **Timeline.** Decode is on the card side at P 4096 (`log/2026-09-30.md:181`), so card experts move the card leg
   only through hits.
5. **Cost and risk.** M (re-plan at run time).
6. **Verdict.** Ledger.

### 3.10 Lookup confidence by match length

`lookup.rs:85` reports p = 1 whatever n matched (3, 2 or 1), so the chooser's per-window profile is flat for lookup.
ik's n-gram map caps its draft by the count each key earned (`common/ngram-map.cpp:380`). A hit rate kept online per n
(a run-time count, so no held-out set is needed) would let the chooser verify a 3-gram match and skip a 1-gram one.
The gain is bounded by 3.2's lookup gain, and it moves rows, not tokens. **Ledger**, with the same A/B as 3.2 (one
more arm).

### 3.11–3.17 Model says 0

- **3.11 `NOMINAL_REPLY` as a running mean.** k* = R·Δ·ρ is linear in R (rig-log `docs/bloomery/plan.md:120-127`).
  An error in R misroutes only prefixes between k*(R_true) and k*(277), and each misrouted one costs its distance
  from the break-even, which is second order. The constant's doc should still say that its prompts are the ones the
  head list was chosen on.
- **3.12 `STREAM_MIN_P` priced by xsplit.** Below 128 positions the gain at the boundary is +9 % `[derived]`
  (`prefill.rs:151-154`). At V4.1's ~90 tok/s prompt rate (rig-log `docs/bloomery/plan.md:13`, 91.2 at 512) that is
  ≤ 0.13 s of a ≤ 1.4 s TTFT `[derived]`. It is worth doing as a common-code lift, not for speed.
- **3.13 `GEMM_FROM`.** A wide unit reads each routed expert once per ubatch, and passes read it once per 8 rows, so
  the threshold already sits at the byte crossover. The bits fix it there too.
- **3.14 K = 512 lane-idle.** A shape fact. The model gives 0 on the A6000 and at most −0.09 ms on GB10, under the
  ruler, so the lever is not opened on either card (`docs/research/sparkdec.md` §4.3, lever 1d).
- **3.15 Sampled requests and the draft.** The per-slot chooser already sees a sampled slot's lower acceptance and
  narrows or closes. ik records the temperature and drafts anyway (`common/speculative.cpp:1953`).
- **3.16 NVMe fill threads.** The code's own doc puts 16 at the drive's knee (`nvtier.rs:99-102`), a drive fact.
- **3.17 `HOST_LANES`, `LANE_PREFETCH`.** Machine facts. `LANE_PREFETCH` already acts only on prompt calls wider than
  8 columns. Its own card (`docs/cards/glmprefetch-ab.card`) predicts 0..+5.7 % and has no result line yet.

### 3.18 Owned elsewhere: slots and default context

Two slots by default, with 4 GiB of checkpoints a slot (`crates/runtime/src/seqstate.rs:41`,
`crates/gpu-gates/src/residency38.rs:52-84`), halved the 3060's context to 2,048 a slot and refused a ~6K prompt
(rig-log `docs/bloomery/plan-triage.md:92-94`). A context sized to concurrent use is a run-time pick, and the KV pool
work owns it. This survey opens no round on it.

## 4. Outside engines

exllamav3 is not on this Mac (its source is on the box only), so it was not read. Every line below was opened in
`~/repo/ik_llama.cpp` or `~/repo/mistral.rs`. No web sources were used.

- **ik, draft length by run time.** `common/spec-tuner.cpp:347`
  `coord.current_idx = coord.select_epsilon_greedy(epsilon);` is an ε-greedy (0.15) search over n_max, p_min, n_min
  and the suffix match length, rewarded by step tok/s. It is opt-in (`--spec-autotune`, `common/common.cpp:1594`).
  The per-token cutoff is `common/speculative.cpp:608` `if (cur_p->data[0].p < params.p_min) {`, default 0.75. The
  n-gram map caps a draft by the accepted count its key earned (`common/ngram-map.cpp:380`). The server clamps a
  slot's draft to its context and remaining tokens, and drafts nothing under n_min
  (`examples/server/server-context.cpp:673-684`).
- **ik, prompt length.** Host-resident weights go to the card only for a large enough batch. For MoE that is
  `ggml/src/ggml-cuda.cu:5244`
  `bool should_offload = batch_size*n_experts_active >= min_batch_size*n_experts_tot;`, with
  `GGML_CUDA_MIN_BATCH_OFFLOAD` 32 by default. On a 256-expert, top-8 file that is a batch of ≥ 1,024 tokens. Kernel
  picks: MMVQ at ≤ 8 columns, MMQ below 64.
- **ik, serving.** It picks a slot by prompt similarity (0.1 by default, `examples/server/server-context.cpp:1046`).
  `only_active_experts` copies only the routed experts of the batch (`ggml/src/ggml-backend.cpp:2053`). It has no
  expert cache that adapts to routing.
- **mistral.rs.** DFlash depth falls to 3 past 8 live sequences
  (`mistralrs-core/src/speculative/dflash.rs:55`
  `if max_n == 0 || !adaptive || batch.max(live_sequences) <= ADAPT_FULL_DEPTH_MAX_BATCH {`). It does not feed
  acceptance back (`speculative/target.rs:200` is a no-op). It has no n-gram draft. Its prefill chunk is the token
  budget divided by the batch (`paged_attention/scheduler.rs:178`).
- **Where the three diverge.** ik searches its draft parameters by exploration and pays the ε. Our chooser models
  E(k)/cost(k) from measured walls and the draft's probabilities, and is on by default. Our open gaps against ik
  are a per-key (per-n) confidence for lookup (3.10) and the prompt-length offload on qwen35moe placed loads (3.4).
  xsplit is finer than ik's single batch threshold: it prices per expert. mistral.rs's count-based depth cut is
  3.5's idea.

## 5. Ranking and 0.3.1

Ranked by the step a real user sees:

1. **3.4 `q35hybpp`.** The largest term, but only on the 12 GB class: TTFT ÷ 5–15 at P 1,464 `[derived]`. Design
   now, land after 0.3.1 unless the design is small.
2. **3.3 `q3kvdepth`.** +14..+17 % at 16K on Qwen3 `[derived]` and ×1.88 context. One A/B, then S (S–M if the e2e
   pin needs a q8_0 arm). Fits 0.3.1 only in the S case.
3. **3.2 `v41lookup`.** Threads and code ≤ +10 %, prose ~0 `[derived, loose upper bound]`. One A/B, then S. Fits 0.3.1.
4. **3.1 Chooser keyed by head.** 0..+2.5 % `[derived]`. Rides the per-script head round. Fits 0.3.1.

## 6. What this survey did not cover

- The GPU flash decode's split by depth, the vision and media paths, and V4.1's CED and engram (no workload pick
  looked plausible in a skim).
- MiMo's seat and the CPU engine's flash levers (`FLASH_SEGMENTS`, `KV_PREFETCH_ROWS`).
- Thread counts per phase (prompt vs decode). It has no measured row to price against.
- DSpark against lookup per request. That needs both drafts resident, and DSpark's 8.5 GB rules it out on one card.
- Acceptance as a function of temperature (E29, unmeasured).
- Multi-node Spark.
