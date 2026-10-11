# warmprof: the warm profile keyed by the KV pool's prefix entry (design note for 0.3.1)

A design note (2026-10-12; Mac only, no card, no box, no code change). It takes the user's five decisions (relayed by the
lead, 2026-10-11) as constraints and prices them against the code at `b00ff641` and the numbers already measured. It
builds on `headpick-warmup.md` §4.1, §5.6 and §6 and does not repeat them.

Every number is either `[measured]` with its source and conditions, a code constant with `path:line`, `[derived]`
(arithmetic on cited terms; the script is Appendix A, run with `python3 -I`), `[assumption]`, or `[unmeasured]`.

## 0. Result

1. **The section.** One `WarmSection` per prefix entry, an immutable `Arc` carried in the seat's saved state
   (`Cached<S, D>`, `crates/gpu-gates/src/bind.rs:2288`), counted in its `n_bytes` (`:2329`), gone when the entry goes.
   It holds the stream's own routing and nothing card-wide: (a) `long`, u32 counts per (layer, expert) over every
   position the entry covers; (b) `recent`, the stream's own decayed rule vector. Size: 101–192 KiB [derived].
2. **What restores it.** Only a full hit restores it: the request keeps every position the entry holds. A cut hit or a
   miss falls back to the global profile. On a cut, `prompt_cache_key` may stand in for "the same conversation" (§5).
3. **The blend.** `SwapRule` keeps one heat vector per live stream instead of one `counts` vector. The card-wide counts
   are `C = Σ ω_s·h_s`, summed in slot order, with `ω_s = min(1, B_s / (M·r_s/R))`. With one stream, or with every
   stream saturated, `ω = 1.0` exactly, and the rule is today's bit for bit. A stream that ends leaves the sum.
4. **The movers.** Nothing new moves on the request path before the first token. A tail of 32 ids or more opens a
   call, and the call's pick sees (a) plus the tail's counts. This differs from a miss on purpose: a miss picks from
   each unit's own counts (`swaprule.rs:1125-1127`), the split walk from its first unit's. A shorter tail has no call, so it
   gets `SwapMachine::open` (`SwapRule::open`, which has no production caller today), capped by a break-even. When
   another stream is live, only the rule's own async flips move the card. The tier refills on a new reader kind,
   `Kind::Warm`, at the lowest rank, with a quota for each stream's share.
5. **Where it pays.** It pays mainly on the paged NVMe tier. Say a conversation returns after another one has evicted
   its arena ids. Without a section, that hit pays up to 5.2–5.7 s of chain fills over its first 96 tokens. A miss
   re-prefills in 2.45 s at P = 4096 and its keep set warms the arena. So a hit can lose to a miss [derived upper
   bound]. The card part's ceiling is 0.09–0.14 s per returning 96-token turn [derived from q38seed-ab].
6. **Head and hints.** The MTP head gets no section; its rows refold at admission (headpick §5.6, §6). `user` and
   `id_slot` are ignored.
7. **Decided (unitier, 2026-10-12, on the user's rule to decide by their tendencies).**
   - **Restore on a full hit only.** A cut hit's section describes the abandoned branch, and the request re-prefills
     that tail, so its call pick already sees the tail's own counts. A cut goes to the global profile and prints
     `warm why=cut`. W6 (a matching `prompt_cache_key` on a cut) stays optional.
   - **One definition of expected tokens:** `E_s = min(B_s, L̄)`. `B_s` is the stream's remaining token budget
     (`max_tokens`, n_predict). `L̄` is the server's running mean of completed turn lengths. Both the rule's blend and
     the tier quota read this one value:
     - the rule's weight is `ω_s = E_s / max_t E_t`, so one stream, or equal streams, keep today's rule bit for bit;
     - the tier quota is `q_s = ⌊S · E_s / Σ E⌋`.
     - The saturation weight `min(1, B_s / (M·r_s/R))` of §3 is replaced by this one.

0.3.1 rounds (§7): **W1** per-stream rule (runtime, pure) → **W2** machine side (tally rows to streams,
`SwapMachine::open`, the keep rule) → **W3** section in the seats and the serve loop (save, restore, `n_bytes`, lever,
record) → **W4** tier (`Kind::Warm`, quota, claims) → **W5** the sitting (TTFT and first-96-step counters,
interleaved conversations). W6 (`prompt_cache_key` on cut hits) and W7 (persisting the global profile to disk) are
optional.

## 1. Paper decomposition and resource timeline

### 1.1 The terms

| Term | Value | Source |
|---|---|---|
| the rule's clock and decay | every 4 kept rows, ×0.9, cap 24 flips a plan, 1 spare a layer | `SwapParams::mid` `crates/runtime/src/swaprule.rs:60-70`, run by every seat through `Residency::params` `crates/gpu/src/host/swap.rs:247-256` |
| counts' half-life / mean memory `M` | 26.3 / 40 kept rows; flips ≤ 6 a kept row | [derived] |
| prompt rows in the counts | none: a call keeps 0 rows | `body38.rs:3297` `keep_rows(KeptRows::prefix(0), PassKind::Prompt)`; `swaprule.rs:1241-1246` |
| a call keeps its placement | only when the rule holds no decode history | `body38.rs:3280-3297`, `host/mod.rs:1364-1369`, `swaprule.rs:1247-1249` (`counts > 0`) |
| a call opens at all | ≥ 32 ids (`STREAM_FLOOR`) | `crates/gpu/src/arch/qwen3moe/wide38.rs:2278`, `body38.rs:1884-1885`; V4.1 `crates/gpu-deepseek41/src/body/prefill.rs:162` |
| H2D (A6000) | 26.28 GB/s | `pcie-a6000.card`, cited by headpick §5.6 |
| drive | 2.51 / 6.77 GB/s at QD 1 / 16 | [measured, qdsweep 10-11], tierengine.md §4 |
| one chain fill / host pick | 1.0–1.1 ms / 23.6 µs | [measured, R1 probe; H8 replay], tierengine.md §4 |
| cold arena, Qwen3.8 r4b shape | 54 fills a token (5,194 / 96 steps); warm ≈ 0–1 | [measured, R1 probe] and [derived], tierengine.md §4 |
| card warm start, Qwen3.8 | decode with MTP 82.48 → 91.24 tok/s (P 512), 70.91 → 79.33 (P 4096); residency only 54.17 → 57.18 (interval touches 1); host slots a step 181 → 136, 180 → 77 | [measured] rig-log `log/2026-09-30.md#q38seed-ab`: A6000, plan a, prose, ctx 4352, 4/6 rounds; the call's placement kept vs the seed start |

**Family shapes:**
- **Qwen3.8**:
  - 48 routed layers [derived: 10,164 paged ids ≈ 212 a layer, tierengine.md §4];
  - 512 experts (`crates/models/src/shape.rs:105-110` `experts() = 32·per_lane`, the `_512` row at `:194`);
  - top-k 10 (tierengine.md §3 (c));
  - 3.134 MB an expert (tierengine.md §4).
- **V4.1**:
  - 40 routed layers [derived: 1,146 + 14,214 = 15,360 routed experts / 384, `docs/HARDWARE.md:9`];
  - 384 experts and top 6 (`crates/gpu-deepseek41/src/router.rs:75-78`);
  - ≈ 15.2 MB an expert [derived: 334 MB / 22, rig-log `plan-triage.md:1514`].
- **GLM-5.3-Flash**:
  - ≤ 45 routed layers (45 trunk layers, `docs/research/modelspec-design.md:50`);
  - 288 experts and top 8 (`shape.rs:221`, per_lane 9);
  - expert bytes not read here.

### 1.2 The bytes of a section, per family [derived]

| Family | (a) `long` u32 | (b) `recent` f32 | Section | Against the entry it rides in | Movers it feeds |
|---|---|---|---|---|---|
| Qwen3.8 | 96.0 KiB | 96.0 KiB | 192 KiB | KV attention rows 28,416 B a position (tierengine.md §3 (l)): 14.5 MB at P = 512 → 1.35 %, 116.4 MB at 4096 → 0.17 % (GDN checkpoints not counted, so a ceiling) | rule, card movers, tier |
| V4.1 | 60.0 KiB | 60.0 KiB | 120 KiB | entry bytes not read here | rule, card movers; tier only once r7 lands (tierengine.md §2) |
| GLM-5.3-Flash | ≤ 50.6 KiB | ≤ 50.6 KiB | ≤ 101 KiB | not read here | rule and card where `card_experts > 0` (0 on the 3090, so residency is off there, tierengine.md §2); tier on r7 |
| MiMo | n/a | n/a | n/a | — | residency unwired (tierengine.md §2) |

The section is under 1.4 % of the entry at P ≥ 512 on Qwen3.8, so it does not move the cache budget (`cache_ram`).
The head adds nothing (§2).

### 1.3 Timeline of a full hit (Qwen3.8, A6000, paged r4b shape) and of a miss

Per admission. A hit with a tail of `t` ids is priced at the P = 512 rate, which is optimistic for short calls
[assumption].

| Resource | Full hit with restore | Miss (no entry, or a cut without a key) |
|---|---|---|
| host CPU | (b) into the stream's vector: 24,576 cells, ~25 µs [derived, 1 ns a cell assumed]. Tier claims: ~1 µs an id [assumption] under the books lock. `open`'s ranking: 48 × 512 log 512 compares, 0.2–0.5 ms, after the first token [derived]. The blend recomputes once per plan: N × 25 µs every 4 kept rows ≈ 0.02 % of a step [derived] | none new |
| card SMs | the tail's prefill: 31 / 62 / 250 ms at t = 32 / 64 / 256 [derived]; nothing new (no kernel) | the whole prefill: 0.499 s at P = 512, 2.448 s at 4,096 (README.md:132, 0.2.8, A6000 plan a); 1.950 s with the split walk (commit `275074de`) |
| PCIe H2D | the KV copy-in, 0.55 / 4.43 ms at P = 512 / 4096 [derived], plus the card movers' flips at 0.119 ms each. They share the link, so the costs **add** | the call's flips, inside its layer batches |
| host DRAM | arena writes of the restored ids, 3.134 MB each | the keep set's fills |
| NVMe | `Kind::Warm`: 0.463 ms an id at QD 16 [derived], at most 2.40 s for a fully cold set of 5,194 ids | the keep set's prompt fills, hidden by the prefill (R4b) |
| wall | KV copy-in + the tail's prefill. The warm reads run beside the decode and the request never waits for them | the prefill |

**Flow first.**
- The restore adds nothing to the first-token time by construction:
  - the host work before the first step is ~25 µs;
  - no flip is issued before the call's own layer shadow;
  - `open` runs at the quiet boundary after the call, so its copies delay the second token, not the first;
  - the tier reads are not waited for.
- What hides, by tail:

  | Tail `t` (ids) | Warm ids the drive reads during the tail's prefill [derived] | Flips a layer the call's shadow holds [derived] |
  |---|---|---|
  | 32 | 67 | 5.4 |
  | 256 | 539 | 43 |

- The rest of a cold tier set, up to 2.40 s, runs beside the first decode steps. It shares the drive with the
  decode's chain fills, so it is ordered hottest first.
- V4.1 is host-bound (`docs/research/adaptive-picks.md:128`).
  - A flip costs 0.578 ms of H2D [derived], and a tier id costs 2.24 ms of drive [derived].
  - Its call pick already moves 22 experts a layer (334 MB) inside one 86–114 ms layer batch (`plan-triage.md:1514`).
  - So on V4.1 the card part rides the call and has no timeline of its own.

### 1.4 Restore against the first token, P = 512 and 4096

| | P = 512 | P = 4096 |
|---|---|---|
| miss TTFT, Qwen3.8 (README.md:132) | 0.499 s | 2.448 s (1.950 split walk) |
| miss TTFT, V4.1 | 2.573 s (3090, 199 tok/s, README.md:118) | 11.43 s (A6000, 358.4 tok/s, `bloomery-bench.md:16`, provisional) |
| miss TTFT, GLM-5.3-Flash | 2.860 s (3090, 179 tok/s, README.md:117) | 44.5 s (A6000, 91.98 tok/s, rig-log `log/2026-10-01.md`, `f826ae60`) |
| hit TTFT, Qwen3.8, t = 64 | 0.55 ms KV + 62 ms | 4.43 ms KV + 62 ms |
| restore's own cost before the first token | ~25 µs host | ~25 µs host |
| card gain after it, Qwen3.8 (ceiling) | ≤ 0.112 s over 96 tokens (MTP); ≤ 0.093 s residency only | ≤ 0.144 s |
| tier gain after it, Qwen3.8 paged (ceiling) | ≤ 5.2–5.7 s of chain fills over 96 tokens | same |

The restore is free against the first token. Its gain comes after the first token and is front-loaded.
- **Card:** at most 9.6 % (P 512) and 10.6 % (P 4096) of a 96-token turn [derived: 0.112 / 1.164 s, 0.144 / 1.354 s].
  That ceiling is q38seed-ab's own speedup, from the seed. A return starts from B's set, which sits between the seed
  and A's own set, so the expected share is lower (§8 Q1).
- **Tier:** this is the case that decides. A paged hit without the section can be slower than a miss.

## 2. The data model

**Owner: one place.**
- The section type is `runtime::warm::WarmSection`. It is pure, sits beside `swaprule`, and has no device type. It is
  stored as `warm: Option<Arc<WarmSection>>` in `Cached<S, D>` (`bind.rs:2288`), which is every seat's `Saved`, so
  all four seats share one owner.
- It is taken on the engine thread at the snapshot, as `draft` is (`bind.rs:2290-2291`). It is restored in
  `put_back`, the one whole-state restore order (commit `c7685588`), after `unpark`.
- `n_bytes` returns `body.bytes() + warm.bytes()`; today it counts the body only (`bind.rs:2329-2331`, headpick §5.6).
- The entry is `serve::promptcache::Entry` (`crates/serve/src/promptcache.rs:27`: `held`, `state`, `pending`). The
  section is inside `state`, so the entry and its section are one identity.
- `take` leaves the entry in the cache (`:98-106`), so two branches can restore one section. That is why the section
  is an immutable `Arc` and the restore copies (b) into the stream's own vector. A shared mutable vector would blend
  the two branches.

**Fields.**

| Field | Type | Content | Why it is the stream's own |
|---|---|---|---|
| `shape` | layers, experts, top-k, model key (headpick §6's `ModelKey`) | refuses a section of another shape by name (`WarmError::Shape`) | — |
| `positions` | u32 | the positions (a) covers = the entry's `n_tokens` | the full-hit test |
| `long` (a) | u32 [layers × experts] | every prompt unit's per-layer counts, as the call already builds them on the host for its pick (`count_routed`, `host/mod.rs:1140-1152`), plus every kept decode row of this stream; undecayed, summed across turns | the same ids give the same counts; it is what a re-prefill's call would count |
| `recent` (b) | f32 [layers × experts] | the stream's own rule vector `h_s` at the snapshot, decay applied | restoring it is "the stream that never stopped" for the rule (headpick §4.1's `kv_cont`, carried to the rule) |
| `key` | u64 | a hash of the request's `prompt_cache_key`, 0 when absent (§5) | — |

**What it does not hold, and why.**
- The literal card set and arena set. Both are card-wide and mix every live stream's ids. A snapshot of them is the
  cross-stream blend that headpick §4.1 measured as a loss for the head.
- Ticks. The books' tick is one global counter (`crates/gpu/src/host/nvtier.rs:430-431`), so a stored tick means
  nothing in another epoch.
- The card and the tier targets are therefore derived at restore from (a) and (b) against the live blend.
- Head rows: refolded at admission (headpick §5.6 pick; decision 2).

**Against headpick.**
- **Agree:** the head gets no section and refolds at admission (§5.6, §6). The card-wide state is the only state that
  blends.
- **Differ:** the residency part is carried, not refolded. (b) decays on the global plan clock, so it depends on
  other streams' rows and is not a function of A's ids. (a) is a function of A's ids, but rebuilding it from the ids
  would cost a re-prefill's routing.
- §5.6's "carry vs refold" pick therefore goes to carry here.

**The size bound** is the table in §1.2. It is fixed per model and independent of P, so it is at most 192 KiB an entry.
- u32 does not saturate at 128K positions: 128K × 10 picks fit u32 with room to spare [derived]; u16 would saturate
  around 6.5K positions for a 1 %-hot expert [derived].

**The global profile.**
- Decision 1's "global profile" is card-wide only (RESD and WIDT in headpick §6). The head has none. This removes the
  one disagreement headpick §4.1 and §7 raised ("a miss starts from the base" for the head).
- Nothing persists today (headpick §6). So in 0.3.1 the global profile is the live card-wide state as it stands: the
  blend of the live streams and the seed under it.
- A miss starts its stream with `h = 0`, and its own call picks from its prompt. That pick is kept when no other stream
  is live (§3).
- W7 (optional) persists a slow card-wide vector `G` as headpick §6's `RESD` section. A miss would then start
  `h = G`, scaled to the stationary mass (360 a layer on Qwen3.8 [derived]), instead of 0.

## 3. The blend

- **Per-stream heat.**
  - `SwapRule` holds one `counts` vector (`swaprule.rs:458`), and the tally has no map from row to stream
    (`Tally`, `swap.rs:363-385`).
  - W1 replaces `counts` with `heat: Vec<StreamHeat>`, one vector per live stream, keyed by the seat's slot index.
  - `end_pass` takes the row→stream map that the seat already knows (each slot's rows in the pass) and folds each
    kept row into its own stream's vector.
  - `plan` decays every vector by 0.9, as it decays `counts` today (`:957`).
- **The card-wide counts.** At each plan, `C = Σ_s ω_s·h_s` over the live streams in ascending slot index. The tie
  order is fixed by that index, never by admission order. `make_pairs`, `open` and `call_pick` read `C` and keep
  their own ties (count descending, seed rank ascending, id ascending; `swaprule.rs:1059-1060`).
- **The weight is expected tokens over the rule's memory** (superseded by §0 item 7: `ω_s = E_s / max_t E_t`; the derivation below is kept for the record):

  `ω_s = min(1, B_s / (M · r_s / R))`

  | Symbol | Meaning |
  |---|---|
  | `B_s` | the stream's remaining kept-row budget: `n_predict` minus the rows kept so far when `n_predict ≥ 0` (the field taken from `n_predict`, `max_completion_tokens`, `max_tokens`, `api.rs:1478-1482`), else the context room left |
  | `M` | 40 kept rows, the counts' mean memory [derived] |
  | `r_s` | the stream's measured kept rows a step (its draft acceptance; 1 without a draft) |
  | `R` | the sum of `r_s` over the live streams |

  - `M·r_s/R` is the share of the card's 40-row memory that this stream's own rows fill.
  - A stream that will outlive that share weighs exactly 1.0. A stream in its last rows weighs in proportion.
  - Examples [derived]: B = 20 rows alone → 0.5; B = 20 with an equal second stream → 1.0; B = 20 at r = 2.4 beside a
    plain stream → 0.71.
  - The rate is needed only to turn a token budget into a share of the clock. The clock counts kept rows
    (`swaprule.rs:4-8`), so `h_s` is already rate-weighted, and weighting by raw `max_tokens` would count the rate
    twice.
  - The EOS stop probability is `[unmeasured]` and left out, so `ω` is an upper bound.
- **What the weight does in practice.**
  - `ω < 1` only in a stream's last `M·r_s/R` rows of an explicit budget.
  - Clients that send a large `max_tokens` and stop on EOS see `ω = 1`.
  - So the blend's main effects are its two edges: a restored stream enters with (b), and a stream that ends leaves.
- **A stream ending removes its weight.**
  - Its vector is dropped from the sum. `C` is recomputed, never decremented, so no float residue is left.
  - On a save, (b) is that vector, so the section holds it after the stream has left `C`.
  - **This changes today's sequential behaviour.** Today an ended stream's counts linger, because no plan runs
    between requests, so `has_history` stays true forever after the first decode. Every later call then hands its
    placement back. A session reset leaves the residency where use took it
    (`serve_seats/glm.rs:1468-1469`); only the explicit residency reset action moves it back (`api.rs:2860`).
  - W2's keep rule becomes: **a call keeps its placement when no other live stream holds rows.** The admitted
    stream's restored (b) is a prior, not another stream's history.
  - The measured direction is q38seed-ab, where the kept placement beat the seed start by +10.6–11.9 % decode with MTP.
    Against "the previous conversation's set" it is unmeasured (§8 Q4).
- **Equivalences.**
  - With one stream and no prior: `C = 1.0·h_1 = h_1` and `h_1` evolves exactly like today's `counts`, so the rule is
    bit-identical. This is gate G1.
  - With N ≥ 2 streams at `ω = 1`: `Σ h_s` and today's single vector differ by float rounding, because 0.9·(a+b) is
    not 0.9a + 0.9b. That is a behaviour change by design, and G2 pins it on a replay.

## 4. Eviction and the swap cost

**Eviction of a section is eviction of its entry.** There is one rule, the cache's, which has three doors:
- `place` (`promptcache.rs:204-260`): first the entries a newer state keeps all of, then the oldest for the budget,
  never one whose copy still runs;
- `evict_oldest_landed` (`:179`) on a staging `NoRoom`;
- `remove` (`:137`), for a state the engine refused.

The `Arc` drops with the entry.
- **On the card and the tier it triggers nothing:** 0 bytes moved, 0 reads. It frees the section's 101–192 KiB of host
  memory [derived].
- A conversation that returns after its entry left is a KV miss. Its full prefill re-warms the card through its call
  and the arena through the R4b keep set. So losing the section costs nothing beyond the KV miss itself, and the
  section's eviction needs no price of its own.

**The swap cost that matters is the restore's displacement of live streams.**

| Mover | What it costs a live stream B | The guard |
|---|---|---|
| rule flips toward A's (b) | B loses the picks of a displaced card expert, 23.6 µs each, until the rule flips it back | A enters `C` with `ω_A`, and an admit needs margin 3 over the victim's blended count (`SwapParams::mid`), so B's hot experts are displaced only by heavier blended heat |
| `SwapMachine::open` (tail < 32) | the same, all at once | it runs only when no other stream is live |
| the call pick (tail ≥ 32) | the same | the call keeps its placement only when no other stream is live; otherwise it hands it back (today's rule, narrowed) |
| `Kind::Warm` fills | one chain fill, 1.0–1.1 ms, for each hot arena id of B's it evicts | with another stream live, claims are `Claim::Transient`: a miss enters at the cold end, the next victim (`crates/model/src/moe.rs:2018-2021`), so B's hot slots are untouched. Alone, they are `Claim::Keep`. Per-layer quota `q_A = ⌊S · E_A / Σ E⌋`, where `E = min(B, L̄)` and `L̄` is the server's running mean of completed turn lengths |

**Break-even: when a restore is refused** [derived].
- **Cut hit.** It keeps fewer positions than the entry holds. (b) is then the abandoned branch's tail (the last ~26–40
  rows), and (a) counts positions the request will re-prefill. Refused, unless the key matches (§5). Record:
  `warm why=cut`.
- **Card pair through `open`.** A pair (admit a, evict v) is worth it when `Δf · T · 23.6 µs > 0.119 ms` (one flip on
  the wall), that is `Δf · T > 5.10` picks.
  - `Δf` is the difference of the two experts' pick rates per row in (a)/`positions`.
  - `T = min(E_A, D/6)` is how long the rule alone would take to make that flip at 6 flips a row.
  - `m` is the number of pairs that pass, at most the cap the second token can carry. Tie order as `open`'s.
- **Tier id beside the decode.** The id is warmed when `p_use = 1 − (1 − f)^E_A > 0.463 / 1.05 = 0.44`, where
  `f = long[id] / positions`. Below that, the drive time it takes from the decode's chain fills costs more than the
  fill it saves [derived, assuming each warm read delays a chain read by its own drive time]. Inside the hidden window
  (§1.3: the tail's prefill time / 0.463 ms) every ranked id is warmed, whatever its `p_use`.
- **Already there.** Ids already filled are hits in the claim: no read, ~1 µs each [assumption]. A single-user
  server whose arena still holds A's ids pays only the claims, so it needs no special case.
- **Lever off** (`BLOOMERY_WARM=off`), or a shape mismatch: refused by name and recorded.

## 5. Hints

| Field | Its meaning, from source | Can it name a prefix entry? | Rule |
|---|---|---|---|
| OpenAI `prompt_cache_key` | "Used by OpenAI to cache responses for similar requests to optimize your cache hit rates. Replaces the `user` field." (openai-python `src/openai/types/chat/completion_create_params.py:192-197`, main, fetched 2026-10-12). Guide: "use a stable prompt_cache_key for requests that share a reusable prefix to help route related requests to the same cache"; "Keys influence routing; they do not pin requests to a machine or guarantee a cache hit." (https://developers.openai.com/api/docs/guides/prompt-caching) | **A group of entries, not one.** The ids already name the entry exactly | Stored as `key` (u64 hash). **Never acts on a miss** (decision 1). On a **cut hit** whose entry carries the same non-zero key, (a) is restored and (b) is not (the client says it is the same conversation: an edit or a regenerate). With no key, or another key, the cut restores nothing. A tie among full hits (`best`: "of equals, the most recent", `promptcache.rs:80-81`) is left alone: both branches carry the same key, so it carries no information there |
| OpenAI `user` | "This field is being replaced by `safety_identifier` and `prompt_cache_key`. … A stable identifier for your end-users." (same file `:368-375`) | **No: a person, not a prefix.** One person runs unrelated conversations | Ignored, as today (`crates/serve/src/responses.rs:55-57`; unknown fields on the chat routes, `api.rs:1493`). A per-person profile would be a new scope, not one of the five decisions |
| llama.cpp `id_slot` | "Assign the completion task to an specific slot. If is -1 the task will be assigned to a Idle slot." (`tools/server/README.md:585`, `36a73916e`); read at `server-context.cpp:4674`; wraps modulo the slot count (`:1653-1655`); picks the slot before the similarity rule (`:1686-1690`) | **No: a container.** A slot holds whatever request ran last; the ids in it are the entry | Ignored for warm. Today our server does not read it at all (unknown, `api.rs:1493`). If a later round honours it as a slot pin, an out-of-range value should be a named 400, not llama.cpp's wrap |
| llama.cpp `cache_prompt` | "Re-use KV cache from a previous request if possible." (`server-schema.cpp:31-32`) | — | Already honoured (`api.rs:1483-1484`). `false` resets the slot, so no entry is hit and no section is restored |

The OpenAI API reference page (`platform.openai.com/docs/api-reference/chat/create`) returned 403 to the fetch. The SDK
docstrings above are generated from that reference.

## 6. Gates the implementation needs

Each clause gets FAIL-first by its own mutant.

| # | Clause | Tier | FAIL-first mutant |
|---|---|---|---|
| G1 | one stream, no prior: `SwapRule` after any history equals today's single-vector rule (`PartialEq`, `swaprule.rs:480-498`), flips and counts bit for bit, over the existing replay vectors | runtime unit (`just mac-test`) | the fold adds 1.0 to the wrong stream; the decay skips one vector |
| G2 | N streams at `ω = 1`: the flips equal a pure per-stream simulation in slot order; against today's summed vector the differing flips are counted and are near ties only (`PIN(...)`, derived from the replay) | runtime unit | sum in admission order (two admission orders give different flips) |
| G3 | `save → restore` of (b) is bit-identical, and the restored stream's next plan equals the stream that never stopped, on a fixed id stream | runtime unit + `gate-swap` | restore (b) into the blended `C` instead of `h_s` |
| G4 | a stream's end leaves `C = Σ` of the rest exactly; `has_history` false when no other stream is live | runtime unit | decrement instead of recompute (float residue ≠ 0) |
| G5 | `ω` table: the saturated set gives exactly 1.0; the examples of §3 | runtime unit | weight by raw `max_tokens` |
| G6 | full hit: the pick's input is (a) plus the tail's per-unit counts (a chosen difference from a miss, which picks per unit). (a) equals the sum over every unit of a re-prefill's per-layer counts (`count_routed`, `host/mod.rs:1140-1152`) plus the kept decode rows, within the routing-agreement band (integer counts, equal where the routes agree) | `gate-gpu-qwen38-serve` kp clauses, fixture tier | a section taken from another entry |
| G7 | refusals by name: shape, cut without a key, lever off; each prints a `warm` record with `why` | serve unit + seat gate | a cut hit restores (b) |
| G8 | tier: `Keep` alone, `Transient` with another live stream; the quota per layer; a decode miss on an id a warm job holds `FILLING` joins that fill (one read counted) | `gate-nvtier` | `Keep` with another stream live (B's hot slot evicted, counted) |
| G9 | `n_bytes` = body + section; eviction drops the `Arc` (strong count 0) and moves nothing | `promptcache` unit + `bind.rs` unit | `n_bytes` without the section |
| G10 | `open`: at most `m` pairs, live at once, never while a pass has rows (`RowsPending`, `swaprule.rs:264`), never with another stream live; the card's words agree with the ledger (`swap.rs` `agree`) | `gate-swap` | `open` with another stream live |

**The ruler.**
- A residency-on decode A/B cannot hold "same greedy tokens": a late flip decides card or host compute, and that moves
  the target's bits at near ties (headpick §6, `registry.rs:410-413`).
- The predicted card gain is +2–6 % (§8 Q1), a fraction of q38seed-ab's +10.6 %. The residency pair's sd is 2.6 %,
  and the 95 % interval is ±2.63 % at 10 rounds (tierengine.md §4). A tok/s A/B would sit about 1–2 sd out, which
  is marginal, so the counters decide.
- The warm A/B therefore runs **on a fixed id stream** (teacher-forced ids, so every arm routes the same prompt and
  the same continuation) and judges three things: engine counters over the first 96 steps (host slots a step from the
  `nvtier` and residency records, fills a step, flips); the first-token time; and the first-96-step wall.
- The pattern is conversations A, B, A, with A's return a full hit. The arms are `BLOOMERY_WARM=off|on`, plus an A/A
  pair.
- The registry row `BLOOMERY_WARM` (`parsed`, `on|off`, default `on` after W5) and the `warm` record kind
  (`record.rs`: `why`, `positions`, `ids_queued`, `ids_hit`, `pairs`, `flips`, `host_us`) come in W3.

## 7. Rounds for 0.3.1

**Attach table first.** One row per common owner; a per-model line outside the last column is a copy to lift.

| Owner (one place) | What warmprof adds there | Qwen3.8 | V4.1 | GLM-5.3-Flash | MiMo | Forced per model |
|---|---|---|---|---|---|---|
| prefix entry: `serve::promptcache` (`promptcache.rs:27`) + `Cached<S, D>` (`bind.rs:2288`) | `warm` field, `n_bytes`, the full-hit test (`kept == positions`), the key hash | ✓ (`serve_seats/qwen38.rs` via `put_back`) | ✓ (`serve_seats/ds41.rs`) | ✓ (`serve_seats/glm.rs`) | ✓ (`mimo2.rs`), section empty | the shape key only |
| residency rule: `runtime::swaprule` | per-stream heat, `C` blend, `ω`, the new keep rule | ✓ | ✓ | ✓ where `card_experts > 0` | unwired: none | none |
| residency machine: `gpu::host::swap` | row→stream map in `Tally` and `end_pass`, `SwapMachine::open`, `has_history` → other-stream rows | `body38.rs:1456` | `residency_glue.attach_tier` | `glue.attach_tier` | none | none |
| call pick: `SwapRule::call_pick` + each body's `stream_begin` | (a) added to the call's counts on a full hit (a chosen difference from a miss's per-unit pick); the per-layer admit bound from the tail's layer time | `body38.rs:3311` | `ds41` prefill stream | GLM call pick not built (`plan-triage.md:1514`) | — | none |
| NvTier books + reader: `gpu::host::nvtier` | `Kind::Warm` (rank 2, after `Prompt` 0 and `Victim` 1, `reader.rs:35-55`), quota, `Keep`/`Transient` | ✓ (paged loads) | after r7 | after r7 | — | part sizes only |
| seats: `bind.rs` `put_back` / `snapshot_begin` | take (a)+(b) at the snapshot, restore after `unpark` | ✓ | ✓ | ✓ | ✓ | none |
| serve loop: `genloop.rs` `reuse` (`:540-575`) | `B_s`, `r_s`, `L̄`, key hash to the seat; no new field | common | common | common | common | none |
| levers + records | `BLOOMERY_WARM`, `warm` kind, `records-refresh` | common | common | common | common | none |

**Rounds.**
- Each wall is an [estimate], to price against `gate-times.tsv` medians before the run.
- Each list starts from `just affected` and narrows only with the scan proof.

| Round | Content | Change class and proof | Gates | Predicted wall |
|---|---|---|---|---|
| **W1** | `SwapRule` per-stream heat, blend, `ω`, keep rule's predicate; `runtime::warm::WarmSection` type and codec (pure) | behaviour-preserving at N = 1 (G1 bit-identical); a new behaviour at N ≥ 2 (G2 pin); host only, `ptx-scan --no-jit` identical | G1–G5 on the Mac (`just mac-test`), then the runtime lib gates and `gate-swap` on the box, fixture tier | Mac ~1 h of tests; box ~15–25 min |
| **W2** | `Tally` row→stream map, `end_pass` per stream, `SwapMachine::open` (the machine's `relayout` generalised: `reset` = relayout to the seed), `has_history` → other-stream rows; the call's admit bound | host dispatch path, no kernel: `just narrow` with covering scan pairs; G10 | `gate-swap`, `gate-xstream`, the residency gates of the three seats, `gate-gpu-lib` | ~45–75 min (residency gates on two families) |
| **W3** | section taken in `snapshot_begin`, restored in `put_back`; `n_bytes`; full-hit test and key hash in `genloop`; lever row; `warm` record | add (no engine caller until on); G6, G7, G9 | `gate-gpu-qwen38-serve`, `gate-ds41-serve`, the GLM serve gate, `gate-levers`, `records-refresh`, serve unit tests | ~40–60 min |
| **W4** | `Kind::Warm`, quota, claim kinds, the FILLING join | host path on the tier; G8 | `gate-nvtier` (fixture + one real), `gate-gpu-qwen38-serve` paged clause | ~30–45 min |
| **W5** | the sitting: A, B, A on a fixed id stream, `BLOOMERY_WARM=off|on` + A/A, Qwen3.8 paged r4b shape and plan a, P 512 and 4096 | prediction card per term (§8) | — | ~40 min, one lease |
| W6 (opt.) | `prompt_cache_key` on cut hits (§5) | add; G7's key clause | serve unit + kp clause | ~20 min |
| W7 (opt.) | persisting `G` as headpick §6's `.bprof` `RESD` | new file owner (`crates/models/src/profile.rs`) | headpick §6's refusals | ~1 h + gates |

Order: W1 → W2 → W3, with W4 in parallel to W3 (disjoint files: `nvtier` against `bind.rs`/`genloop`) → W5. This is a
train: W1, W2 and W4 share `crates/gpu`.

## 8. Not measured, and open questions

Each question names the one box run that settles it and its expected value. All of them fit in one W5 lease.

| # | Question | Run | Expected |
|---|---|---|---|
| Q1 | card value for a returning conversation: its decode starts from B's set, not the seed | W5 counters: host slots a step over A's first 96 steps, warm off / on | off between 136 and 181 (q38seed's two arms bound it), on near 136 at P 512; decode +2–6 % [assumption: half the ceiling]. Under 1 % → card movers go behind the tier |
| Q2 | tier fills for A after B on the paged shape | W5: fills over A's first 96 steps | off 1,000–5,194 (≤ the cold probe; shared experts lower it); on ≤ 100 inside the hidden window. Off < 500 → the tier part is worth < 0.5 s and W4 drops to optional |
| Q3 | do (a)'s counts agree with a re-prefill's call counts (prefill kernels against decode kernels near router ties)? | functional, no lease: G6's diff, cell by cell | ≥ 99 % of cells equal [assumption]. Under 95 % → G6's band widens with a dated PIN |
| Q4 | the sequential case: the call keeps its placement once ended streams leave `C` (today it hands it back) | W5 extra arm: two unrelated conversations in sequence, keep rule old / new | decode over the second's first 96 steps ≥ old at the ruler; + up to q38seed's +10.6 % on a topic change [assumption] |
| Q5 | chain-fill latency with `Kind::Warm` reads in flight (it sets the 0.44 break-even) | W5: chain fill ms with warm reads queued | 1.05 → 1.2–1.6 ms [assumption]; above 2 ms → warm reads beside the decode capped at QD 4 |
| Q6 | `L̄`, the mean completed-turn length | none: serve logs from the dogfood pass | 200–800 kept rows [assumption] |

Decided, with no run needed (§0 item 7): full-hit-only restore, and one definition of expected tokens (§0
item 7).

Not measured here, and outside any run above:
- The EOS stop probability in `ω`.
- GLM's and V4.1's expert bytes on their tiers: they come with r7.
- MiMo: unwired.
- The real-chat locality of (a) across turns. headpick §8's sets are concatenated documents, so their restore gains
  are a floor.

## Appendix A: the paper terms (`warmterms.py`, run with `python3 -I`)

```python
#!/usr/bin/env python3
# warmprof paper terms: every input is a cited measured number or a code constant; every output is [derived].
import math
EVERY, DECAY, CAP = 4, 0.9, 24            # SwapParams::mid (swaprule.rs:60-70), run by every seat (swap.rs:247-256)
PCIE = 26.28e9                             # A6000 H2D, pcie-a6000.card (headpick-warmup §5.6)
DRIVE16, FILL_CHAIN = 6.77e9, 1.05e-3      # QD16 rate (qdsweep, tierengine §4); one chain fill 1.0-1.1 ms (R1 probe)
PICK = 23.6e-6                             # host time a pick (H8 replay, tierengine §4)
FAM = {  # layers, experts, top-k, expert bytes (None = not read here), kv bytes a position (None = not read)
    "qwen38": (48, 512, 10, 3.134e6, 28416),
    "v41":    (40, 384, 6, 334e6 / 22, None),
    "glm":    (45, 288, 8, None, None),
}
half = math.log(0.5) / math.log(DECAY) * EVERY
mem = EVERY / (1 - DECAY)
print(f"counts half-life {half:.1f} kept rows; mean memory {mem:.0f} kept rows; flips <= {CAP/EVERY:.0f} a kept row")
for f, (L, E, k, xb, kvp) in FAM.items():
    sec = L * E * 4
    print(f"{f}: section (a) u32 {sec} B = {sec/1024:.1f} KiB; (a)+(b) {2*sec/1024:.1f} KiB; stationary mass a layer {DECAY*EVERY*k/(1-DECAY):.0f}")
    if xb:
        print(f"  flip {xb/PCIE*1e3:.3f} ms on H2D; drive {xb/DRIVE16*1e3:.3f} ms an id at QD16")
    if kvp:
        for P in (512, 4096):
            kv = P * kvp
            print(f"  P={P}: KV attention rows {kv/1e6:.1f} MB, copy-in {kv/PCIE*1e3:.2f} ms; section/entry {2*sec/kv*100:.2f} %")
# first-token time (README 0.2.8 rows, Qwen3.8 UD-Q4_K_XL, A6000 plan a; V4.1/GLM bench rows)
for name, P, rate in (("qwen38", 512, 1026), ("qwen38", 4096, 1673), ("qwen38 split walk", 4096, 2100.4),
                      ("v41 3090", 512, 199), ("v41 A6000", 4096, 358.4), ("glm 3090", 512, 179), ("glm A6000", 4096, 91.98)):
    print(f"TTFT miss {name} P={P}: {P/rate:.3f} s")
for t in (32, 64, 256):
    print(f"qwen38 hit tail {t}: {t/1026*1e3:.0f} ms at the P=512 rate; hides {t/1026/(3.134e6/DRIVE16):.0f} warm ids and "
          f"{t/1026/48/(3.16e6/PCIE):.1f} flips a layer")
# tier: cold arena after another conversation (upper bound: the R1 probe's cold arena, 5,194 fills / 96 steps)
cold = 5194
print(f"tier cold upper bound: {cold*1.0e-3:.2f}-{cold*1.1e-3:.2f} s chain fills over 96 steps; restore drive {cold*3.134e6/DRIVE16:.2f} s")
print(f"tier break-even beside the decode: p_use > {3.134e6/DRIVE16/FILL_CHAIN:.2f}")
# card: q38seed-ab ceiling (rig-log 09-30, A6000 plan a, MTP on, 96 steps): 82.48->91.24 (P512), 70.91->79.33 (P4096)
for P, a, b in ((512, 82.48, 91.24), (4096, 70.91, 79.33), (512, 54.17, 57.18)):
    print(f"card ceiling P={P}: {96/a:.3f} -> {96/b:.3f} s over 96 tokens, saves {96/a-96/b:.3f} s")
print(f"card stand-alone open break-even: df*T > {3.16e6/PCIE/PICK:.2f} picks a pair")
# blend weight: w = min(1, B / (M * r / R)); saturated equal streams -> 1.0 exactly
def w(B, r, R): return min(1.0, B / (mem * r / R))
print("weights:", [round(w(B, r, R), 3) for B, r, R in ((4096, 1, 1), (20, 1, 1), (20, 1, 2), (20, 2.4, 3.4), (10, 1, 4))])
assert w(4096, 1, 2) == 1.0 and w(10**6, 2.4, 3.4) == 1.0
```

Its output (2026-10-12, the Mac), the lines the decisions read; the rest are the numbers of §1.3–§1.4:

```
counts half-life 26.3 kept rows; mean memory 40 kept rows; flips <= 6 a kept row
qwen38: section (a) u32 98304 B = 96.0 KiB; (a)+(b) 192.0 KiB; stationary mass a layer 360
  flip 0.119 ms on H2D; drive 0.463 ms an id at QD16
tier cold upper bound: 5.19-5.71 s chain fills over 96 steps; restore drive 2.40 s
tier break-even beside the decode: p_use > 0.44
card ceiling P=4096: 1.354 -> 1.210 s over 96 tokens, saves 0.144 s
card stand-alone open break-even: df*T > 5.10 picks a pair
weights: [1.0, 0.5, 1.0, 0.708, 1.0]
```
