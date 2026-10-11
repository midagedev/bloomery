# headpick: one common row-list draft head with a per-request list pick

A design (2026-10-11, Mac and read-only box fetches: no run, no code change) for 0.3.1. Qwen3.8's MTP head scores a list
of vocabulary ids read in place from the target's `output`. 0.3.0 ships one list for every request (cjk-76000). This
document designs (a) a per-request pick between two lists, (b) the one common owner of lists, maps, pick and pin that
Qwen3.8 MTP, GLM NextN and V4.1 DSpark attach to, and (c) the width chooser's state per head class.

Every claim carries `path:line` (this tree, main `2ce989d7`) or `[derived]` with its terms. `D` is the 65,536-row
kohangul list, `C` is cjk-76000, `S`/`δ` are defined in §1. Numbers marked `[counted]` are counts of ids in the
checked-in held-out files against the two lists (formula in §1.1); they reproduce the q38cjk card's own q values.

## 0. Result

| # | Question | Answer |
|---|---|---|
| 1 | Pick signal | The prompt's ids against the two lists, not code points: `S = #(ids in C∖D) − #(ids in D∖C)`. Pick `C` iff `S ≥ max(3, 0.00245·P)`; else `D`. The 0.00245 is the break-even difference of the two lists' miss rates (§1.2). |
| 1 | Boundary | `C` beats `D` when the text's `q_D − q_C > 0.00245` [derived]. For zh/ja text in an en or code background that is a CJK share of **1.7 to 3.5 %** of the ids. Korean is a tie (§1.3). |
| 1 | Generated tokens | **Yes, one way.** A prompt-only pick has a −55..−59 % tail (default list on a CJK reply) against a −1.6..−2.8 % cost of the other mistake. Upgrade `D→C` when ≥ 5 of the last 16 kept target tokens are outside `D`; never downgrade. On the held-out sequences it fires in a median 6 to 8 tokens on zh/ja and in none of 300-token en/code replies. |
| 2 | Slots | The head is **per slot on the host, one per model on the device.** One map per list lives in the existing 248,320-word map buffer; the walk key carries the list. No union list: the draft chains run one slot at a time. |
| 3 | Swap | 0 bytes copied, a key change; one graph capture per (slot, walk key) on first use. A copy-into-place swap is invalid: the grid and `n` are baked too. |
| 4 | Residency | Counts are the kept tokens' routing, list-independent. The plan clock counts kept rows but plans fall at pass boundaries and flips go live in passes, so a list moves the flip timing, not the counts. The tokens-identical gate runs with the residency off. |
| 5 | Second list | `D` restored byte for byte (sha `b9deced6…` checked). No re-rank: the headroom is ≤ +0.1 % on en (§5). |
| 7 | Gain per model | Qwen3.8: +1.6..+2.8 % on en/code against cjk-only. GLM: +0.0..+0.6 %, centre +0.2 % [derived], under the A/B ruler. V4.1 DSpark: ≤ 0 % at the Qwen profile's sensitivity [derived], no speed case. The attach is for the common owner, not for speed. |
| 8 | Chooser state | **Required with the upgrade:** a class change reopens a closed chooser (otherwise a gate closed before a `D`-zh request never reopens within the reply). **Optional:** restore the class's own profile, derived 0..+0.3 % (ceiling +2 %), not +2.5 %. Keyed by the class the head owner hands out; only the profile is keyed. Gate: mock-clock unit tests, no timed card. |

Migration: Qwen3.8 first, in three landings (§7.4). GLM and DSpark attach with zero lists first.

## 0.1 L2 recut: one list plus warm spare rows (decided 2026-10-11)

The user delegated the two questions `headpick-warmup.md` raised to the lead, who decided on 2026-10-11. This section
supersedes the parts of §0-§9 named below. §1-§5 stay as the derivation that led to the decision.

**The head is one list plus warm spare rows** (`headpick-warmup.md` §3.5, arm `cblind`). `C` (cjk-76000) stays the one
shipped list. Its 2,048 highest non-control ids (below the control block, id ≥ 248,044, which is protected by
definition) are a writable spare region per slot, filled from the slot's accepted ids. Every held-out set reaches
≥ 0.991 of its best static list after 256 ids, and zh/ja/ko reach 0.998-1.01 from id 0. That is within the 4-round
ruler of the pick of §1 `[sim]`, with one list instead of per-language lists. This is the user's own idea ("적당히 웜업").
- **A miss starts from the cold base.** A session's spare rows are a function of its ids, so a KV-prefix hit refolds the
  prefix's ids into the slot's map. That is exact (79,200 of 79,200 windows, `headpick-warmup.md` §4.1) and stores
  nothing. A miss starts from `C` as shipped. A blended card-wide head profile loses to the cold base (ja −1.8 to −2.2
  points `[sim]`), so the profile file holds card-wide residency and width state only, and the head has no section in it
  (`headpick-warmup.md` §6).
- **Rows never blend across slots.** Each slot has its own map of `R` words, swapped with its graphs (`DraftSeq38`).
- **What it replaces:**
  - the pick rule (§1.1-§1.3, G1, G3's pick) and the `D→C` upgrade (§1.4, G2);
  - the second list `D` (§5) as a shipped default;
  - the chooser's class state (§8, G10): with one list there is no class change. A large burst of spare-row writes is
    not a class.
  - `BLOOMERY_MTP_HEAD_ROWS` set to a path stays "one list, as given": no spare rows, as §9 says for a given list.
- **Cost:** a dirty span of ≤ 8 KB a change window, written by an async host→card copy on the stream between passes.
  It adds 0 kernels and 0 graph nodes; the walk's map address, `rows`, grid and argmax `n` are fixed by the fixed `R`
  (`headpick-warmup.md` §5.1-§5.2).
- **G8 exception (dated 2026-10-11):** G8 pinned `plan.map_bytes` to the single-list load's. One map per slot makes it
  `max(vocab, slots × R)` words: 608 KB at the default 2 slots, 1.2 MB at 4 (`headpick-warmup.md` §5.3). The plan's
  shape (which rows, the grid, `n`) does not change, only the spare rows' contents. G8 becomes "the plan's
  `map_bytes` and the open-time `got == want` equal `max(vocab, slots × R)` words, the same at every point of a run".
  `gate_qwen4exp_mtp.rs` (d) and (r), which pin the map as the list's ids, compare the protected prefix (`R − 2,048`
  rows) after the first write: a coverage change with this dated reason, not a relaxation.
- **Gates of the recut** (they replace G1, G2, G3's pick and G10; G4-G7 and G9 stay, with G6's ratio clause measured per
  slot's spare region):
  - **S1:** the fold is pure, in `crates/models/src/headpick.rs` unit tests (`just mac-test`):
    - the same ids folded live or at once give the same set;
    - a control id is never written;
    - the eviction keeps `R`;
    - FAIL-first mutants: a fold that writes a control id, and a fold whose order depends on time.
  - **S2:** two slots, zh-chat on slot 0 and en-prose on slot 1, run under `BLOOMERY_MTP_WIDTH=fixed`. Each slot's spare
    rows hold only its own ids. FAIL-first mutant: one map shared by both slots.
  - **S3:** a KV-prefix hit refolds the prefix, so turn 2's first window equals the unbroken stream's.
- **Migration:**
  - L0 and L1 are unchanged: the host reader and one device map (§7.4).
  - L2 becomes the spare rows: the per-slot map, the fold, the refold at admission, S1-S3 and the G8 exception.
  - The timed claim is one `noninf` card: spare rows against `C` alone, same binary, predicted +0.6 to +2.9 % on en/code
    `[derived]`: `C` sits at 0.972-0.985 of the best list there (§3.1 of `headpick-warmup.md`) and the spare rows at
    ≥ 0.991 after 256 ids (its §3.5). The zh/ja/ko rows are predicted at 0.0 %.

## 1. The pick rule and its boundary

### 1.1 The model the rule is derived from

`crates/models/tests/head_coverage.rs:59-78` (`break_even`): a window walks the head `WALKS = 3` times, each reading
every row of the list once (`:40`); a row is 2,720 B (`:43`); the head reads at 701 GB/s (`:47`, the top of the
measured 579-701); the window is 37.43 ms (`:50`); the kept windows of q38head2-ab at P 512 are `KEPT = [0, 3, 2, 21]`
(`:53`, `docs/cards/q38head2-ab.card:29-31`). A target token outside the list ends the window's accepted prefix, so
`AL(q) = 1 + Σ_{j=1..3} A_j (1−q)^j` with `A = (26, 23, 21)/26` (`docs/cards/q38cjk-ab.card:7-9`).

A text's tok/s under a list of `r` rows is `AL(q_list) / T_list`, `T_list = 37.43 ms − 3·(248,320 − r)·2,720 B/701 GB/s`.
Counting the 11 held-out sets' ids (`crates/models/tests/data/head-holdout/*.ids`) against `C`
(`crates/model/data/mtp-head-rows-qwen38-cjk-76000.txt`) and `D` (`git show
347a86e5:crates/model/data/mtp-head-rows-qwen38-kohangul-65536.txt`, 65,536 ids; sha `b9deced66fc00185…`
`[counted]`) gives, with tok/s of `C` against `D` on that text:

| set | q_D | q_C | δ = q_D − q_C | AL_D | AL_C | tok/s C vs D |
|---|---|---|---|---|---|---|
| zh-wiki | .6692 | .0332 | +.6360 | 1.457 | 3.524 | +141.0 % |
| zh-chat | .5984 | .0327 | +.5657 | 1.597 | 3.526 | +120.1 % |
| ja-wiki | .5901 | .0064 | +.5838 | 1.614 | 3.659 | +125.9 % |
| ja-chat | .6006 | .0059 | +.5946 | 1.592 | 3.662 | +129.2 % |
| ko-wiki | .0195 | .0068 | +.0127 | 3.592 | 3.657 | +1.45 % |
| ko-chat | .0029 | .0031 | −.0002 | 3.677 | 3.676 | −0.37 % |
| en-prose | .0091 | .0178 | −.0088 | 3.646 | 3.601 | −1.57 % |
| en-math | .0071 | .0191 | −.0120 | 3.655 | 3.594 | −2.01 % |
| code-rust | .0132 | .0270 | −.0138 | 3.624 | 3.555 | −2.26 % |
| code-python | .0152 | .0321 | −.0168 | 3.614 | 3.529 | −2.68 % |
| code-c | .0154 | .0332 | −.0178 | 3.613 | 3.524 | −2.81 % |

The q_D and q_C columns reproduce the q38cjk card's table (`q38cjk-ab.card:10-13`); the en-prose row reproduces its
−1.2 % acceptance plus −0.35 % head bytes (the `C` window is 0.122 ms longer, 0.345 %: 10,464 rows × 2,720 B × 3 /
701 GB/s [derived]). Break-evens against the full head: `C` .0391 (the gate's pin, `head_coverage.rs:289-297`), `D`
.0415 [derived, same function at 65,536 rows]. Every `D` miss rate above is under .0415, so `D` is a valid second
list on en, code and Korean by the gate's own test.

### 1.2 The signal: the ids' share inside each list

`δ(text) = q_D − q_C = share(ids in C∖D) − share(ids in D∖C)`: the two lists' miss rates differ by exactly what lies
in one list and not the other (|C∖D| = 30,171 ids, |D∖C| = 19,707, |C∩D| = 45,829 `[counted]`). `C` beats `D`
iff `AL(q_C)/T_C > AL(q_D)/T_D`, i.e. `AL(q_C)/AL(q_D) > 1 + 0.345 %`. With `(dAL/dq)/AL = −1.406..−1.410` over
q in [0, .02] `[derived]` this is `δ > 0.00245`, and the threshold is the same (0.00243..0.00245) from q_C = 0 to
.05, so it is one constant for the model, not one per text. It is a break-even of the same function the gate pins.

The pick therefore reads **the prompt's ids against the two lists**: one bitmap lookup per prompt id, `P × ~1 ns`, no
tokenizer table, no Unicode ranges, no vocabulary knowledge. A CJK code-point share (needs the text, and is blind to
which tokens each list actually holds) and a Hangul share (Hangul is in both lists: every one of the 6,807 Hangul
tokens is in `D`, `q38head2-ab.card:8-10`) were rejected: the id count is in the break-even's own units and fits the
common owner with no per-model branch (§7.2).

The statistic is a count so that template tokens (in `C∩D`) cannot dilute it: `S = #(prompt ids in C∖D) − #(prompt ids
in D∖C)`, and **pick `C` iff `S ≥ max(τ, δ*·P)`**, `δ* = 0.00245`, `τ = 3`. `δ*·P` is the break-even scaled to the
prompt (`ceil(0.00245 · 4,096)` = 11 at P = 4,096); `τ` keeps a 1-2 id fluctuation of a short English prompt from flipping the pick. On chunks of
`P` ids of the held-out sets (400 random chunks each, `[counted]`; share that picks `C`):

| P | zh/ja chunks | ko-wiki / ko-chat | en-prose, en-math, rust, python | code-c |
|---|---|---|---|---|
| 16 | .91-.97 | .03 / .00 | 0.00 | 0.00 |
| 64 | .97-1.00 | .14 / .01 | 0.00 | 0.00 |
| 256 | .98-1.00 | .22 / .05 | 0.00 | .03 |
| 1,024 | 1.00 | .40 / .22 | 0.00 | .02 |
| 4,096 | 1.00 | .69 / .10 | 0.00 | 0.00 |

(τ = 3. τ = 1, 2 and 4 give the same picture within ±.03 on zh/ja and ≤ .03 on en/code.) Short zh/ja prompts the rule
misses (P = 16: 4-9 %) start on `D` and are caught by the upgrade (§1.4).

### 1.3 The boundary in CJK share

Mixing a CJK set into an en/code set with share `s` of the ids, `C` ties `D` at: zh-chat `s* =` .0195 (en-prose), .0250
(en-math), .0279 (rust), .0331 (python), .0346 (c); zh-wiki .0174-.0309; ja-chat .0186-.0330 `[counted]`. So a prompt
with **more than ~2 to 3.5 % of its ids** in Chinese or Japanese text is a `C` prompt; a 3,000-token code file with a
dozen Chinese comment tokens is not. Korean is a tie by construction (ko-wiki +1.45 %, ko-chat −0.37 % above; the
mixture crossing is at s* = .52-.66 for ko-wiki into en/code): the rule sends Korean prompts to whichever list their
ids favour and neither choice costs more than ~1.5 %. At `s = 0.5` the table above (zh-chat into en-prose) gives
`C` +48 %; at 0.05 +2.5 %; at 0.01 −0.8 %.

### 1.4 Mixed prompts and the generated tokens: a one-way upgrade

The prompt is a proxy for the reply's script, and the two mistakes are not symmetric `[counted]`:

| mistake | tok/s |
|---|---|
| `D` on a CJK reply | −54.6 % (zh-chat), −58.5 % (zh-wiki), −56.4 % (ja-chat), −55.7 % (ja-wiki) vs `C` |
| `C` on a non-CJK reply | −1.57 % (en-prose), −2.01 % (en-math), −2.26 % (rust), −2.68 % (python), −2.81 % (c) vs `D` |

The first costs ~20-35 times the second, so a prompt-only pick pays off only if fewer than ~3-5 % of its `D` picks meet
a CJK reply (`1.6/(55+1.6)` to `2.8/(57+2.8)`). "English prompt asking for a Chinese answer", code with Korean or
Chinese comments, a translation task and a chat that turns to Chinese on turn 3 all exceed that. **The pick must
therefore also read the generated tokens**: the verify gives the target's kept ids every window
(`crates/app/src/mtp.rs:489` `record(p0, rows, out, accepted)` for a window, `:838` `stepped(.., next)` for a plain
step), and a bitmap lookup per kept id is a host cost with no device term.

Rule: while the request runs on `D`, keep the last 16 kept target ids; **upgrade to `C` when ≥ 5 of them are outside
`D`**; never downgrade within a sequence (the downgrade saves 1.6-2.8 % on a rare case and costs a switch). On the
held-out sequences, random start offsets, 300 trials per set `[counted]`:

| window | zh-wiki | zh-chat | ja-wiki | ja-chat | ko-wiki / ko-chat | en-prose, en-math, rust, python | code-c |
|---|---|---|---|---|---|---|---|
| fires within 300 tokens | .993 | 1.000 | .997 | 1.000 | .080 / .027 | 0.000 | 0.000 |
| median tokens to fire | 6 | 7 | 8 | 8 | - | - | - |
| p95 tokens to fire (2000) | 17 | 42 | 22 | 17 | - | - | - |

(n = 16, k = 5. n = 8, k = 3 fires 2-3 tokens sooner but also in code-c 6 % and python 1 % of 300-token replies;
n = 32, k = 6 fires 1-2 tokens later and still fires in code-c 3 % of them.) The upgrade's cost on a CJK reply that started on `D`: each token before it runs
at `T_D/AL_D = 35.30/1.60 = 22.1 ms` instead of `T_C/AL_C = 35.42/3.53 = 10.0 ms` (zh-chat) [derived], 12.1 ms lost
a token: a median 7 tokens is 0.085 s, p95 42 tokens 0.51 s, against the same reply wholly on `D` (+120 % wall). A
`C` pick on a Korean prompt that the upgrade never touches costs ≤ 1.5 %; a false upgrade on en/code is 0 in the 1,500
300-token trials above and, if it happened, costs the −1.6..−2.8 % of the rest of that reply.

**Where the detector sits: where every committed id passes.** A kept target id reaches the draft by three hooks: `accept`/`record`
for a verified window (`out`), `stepped` for the caller's own plain step, and `held` for the chooser's plain pass, whose
default is `stepped` (`runtime/src/speculative.rs:130`; the chooser forwards its own plain turn to `held`, `width.rs:1040-1060`).
`MtpDraft` overrides `record` and `stepped` and inherits `held`, so the detector goes in those two and sees all of them
(`mtp.rs:489`, `:838`); a round's skipping slot is "told nothing" (`mtp.rs:600-602`) but a skipping draft drafts nothing, so no
head is read. The case that needs the care is a **closed chooser**: after a low-acceptance request the gate is closed and
its plain passes go through `held`, and the chooser reopens only by a shadow every 16th closed pass whose `learn` steps the
mean 0.05 a shadow (`width.rs:844-865`, `:779-786`), an e-fold of ~20 shadows = 320 passes, or by probes at 64, 128, 256 passes.
A `D`-zh reply (AL 1.6 gives 45 tok/s against the plain 57.9, `adaptive-picks.md` §3.1) keeps such a gate closed for the whole
reply, and a long English system prompt with a Chinese question picks `D` by construction. So the upgrade alone is not
enough: **a class change must reopen the chooser** (§8, required, not optional). Without it that path runs plain steps
at ≈ 42 % below the `C` rate for the reply.

The upgrade is sticky for the **sequence**: a held sequence's next prompt call (`MtpDraft::prompt`,
`crates/app/src/mtp.rs:732`, a call that continues a held sequence carries ids from the held position on, `:722-731`) keeps the class and the
pick restarts on `restart`/reset (`mtp.rs:466`, `drafted.rs:351-358`).

### 1.5 Where the pick runs

At the top of `MtpDraft::prompt` (`mtp.rs:732`): the prompt call's draft walks are store walks, which take no head
(`mtp38.rs:1525-1545`, `store_walk` at `:~2205` has no head argument), and the first headed walk is the first chain
after `begin` (`mtp.rs:761`). The pick sets the draft's head selector before any headed walk. Today `MtpDraft::open`
fixes the head once (`mtp.rs:414` `B::head(m)`, "Load-time only", `:242`).

## 2. Slots: per slot on the host, one map table per model on the device

- Host: `SlotDrafts::open` makes one `MtpDraft` per resident slot, each with its own head and its own width chooser
  (`crates/gpu-gates/src/bin/shared/serve_seats/drafted.rs:100-107`; `MtpDraft.head` is a field, `mtp.rs:363`,
  `Card` copies it per call `:378-386`).
- Chains are per slot: `pass_slots` selects each window's slot and proposes on it before the one batched verify
  (`mtp.rs:622` `select_slot`, `:625` `Draft::propose`, `:648` `verify_slots`); `pass_rows_one_pass` takes the
  windows out of the table and runs them through it (`rounds.rs:198-330`). Only the verify is batched.
- Device: `Mtp38.head: HeadMap` is one buffer for the model (`mtp38.rs:166-170`, `:177`); the captured walks and the
  store are per sequence (`DraftSeq38`, `:208-213`, swapped by `swap_slot` `:561-567`, called on `select_slot`
  `body38.rs:4228`), and a walk bakes the map's address (`mtp38.rs:2134-2186`, `Some(&hm.ids)`).
- V4.1 DSpark is single slot: `--parallel > 1` with `BLOOMERY_DRAFT=dspark` is refused (`docs/serving.md:171-172`); its
  chooser is the seat's one `Choosing` (`serve_seats/ds41.rs:1019-1030`, `behind` `:1120`).

So the pick is per slot (a host value in `MtpDraft`) and the device must hold all lists the slots may use at once.
Options, priced per slot-window at 3 walks of `rows × 2,720 B` (Qwen3.8):

| option | bytes a window (slot on D / on C) | launches | resident | verdict |
|---|---|---|---|---|
| **per-slot list, maps resident** | 534.8 MB / 620.2 MB (3 × 65,536 × 2,720 B; 3 × 76,000 × 2,720 B) | 0 extra | 0 (§3) | recommended |
| union list (`C∪D` = 95,707 ids [counted]) | 781.0 MB on every slot (+160.8 MB on a `C` slot, +246.1 MB on a `D` slot: +0.23 ms / +0.35 ms at 701 GB/s) | 0 | 0 (76,000 + 65,536 + 95,707 = 237,243 words ≤ 248,320 in the one buffer) | only valid if one walk served several slots, which no seat does |
| the batch takes the larger list | 620.2 MB on every slot (en/code slots lose 1.6-2.8 %) | 0 | 0 | no |

The union and larger-list options solve a problem the code does not have: the draft's walks read the head once per
slot walk, never once for a batch. If a batched draft walk (one walk over several slots' rows) lands, the head would
be read once for all of them and the union list becomes the correct shape for it; the owner's map table (§3) holds
a union as one more list, with no change to its shape.

## 3. Map swap cost

What reaches the kernel: `ids: &DeviceBuffer<u32>` and `rows` of `q8_0_gemv_ids`/`_mcol` (`q8f32.rs:1902-1978`), the
same `map` and `n` of `argmax_p_rows_fault` (`gpu/src/mtp.rs:463-510`), and `q6k_ids` for a Q6_K output
(`q6k_ids.rs:591-600`). Three things are baked into a captured walk: **the map's address, `rows` as the kernel
parameter, and the grid `rows.div_ceil(8)` (`q8f32.rs:1944`)**; the argmax's `n` too (`mtp.rs:505`). Copying a new
list into the same buffer is therefore **not** a valid swap: 65,536 and 76,000 rows launch different grids.

The walk is already keyed: `MtpKey { m, own, head }` (`mtp38.rs:909-913`), captured lazily per sequence on first use
(`:1662-1683`, `Graph::capture`), `head: MtpHead::{Full, Rows}` (`:762-769`). The design makes `head` carry the list:
`HeadSel::Rows(list)`. Then:

- **Bytes.** 0 copied. Both lists live in the model's one existing map buffer of `vocab` words (`mtp38.rs:303-330`:
  "one `u32` a vocabulary id whatever the head"): `C` at word offset 0 (76,000 words), `D` at 76,000 (65,536 words),
  141,536 ≤ 248,320. Each list is a non-owning window (`param_view`, `scratch.rs:1571-1584`, the device-pointer view
  the walk already uses for its inbox; the kernels only check `rows ≤ ids.len()`, `q8f32.rs:1902-1978`, and
  `map.len() ≥ n`, `mtp.rs:471-476`). The buffer stays 248,320 × 4 B = 993,280 B, so the plan's `map_bytes` and
  the open-time `got == want` check (`mtp38.rs:355-372`) and the cards' `card_bytes` pins do not move, and the lever
  text "the load's plan is the full head's, byte for byte" (`registry.rs:404-406`) stays true. The owner refuses
  by name a list set whose rows sum past the vocabulary (DSpark's 129,280 cannot hold two 65,536 lists in one
  buffer; its owner sizes the buffer `max(vocab, Σ rows)` and `plan.map_bytes` states it once).
- **When.** Between requests (the pick) and between windows (the upgrade): a chain starts with a refresh walk, a
  `MtpFeed::Rows` walk that writes the own row (`mtp38.rs:1448-1452`, `forget` leaves the store; the module doc
  "a chain's refresh walk rewrites it before any own walk of the chain does", `:~200-206`); the store walk takes no
  head, so a head change between two chains touches no state a walk reads. Never inside a chain: its own walks feed
  on the head's argmax.
- **Graph capture.** One capture per (sequence, key) on first use. Keys per list per sequence: refresh walks
  `m ∈ 1..=MTP_GRAPH_ROWS = VERIFY_ROWS` (`mtp38.rs:729`, `scratch38.rs:60`) with `own = false`, and `m = 1` with
  `own = true`: at most `VERIFY_ROWS + 1` keys, each ~36-37 nodes (`walk_launches`, `:751-758`: 2 + ⌈4m/8⌉ + 11 +
  15 + 5 + 2 + [Q6_K 1]). With `s` slots and two lists at most `2·s·(VERIFY_ROWS + 1)` captured walks. The capture
  time is **not measured here**; it is paid once per (slot, list) for the process and the first window after an
  upgrade is the one term to read (§9). `set_taps` drops all captures (`mtp38.rs:1422`), the only other path.
- **Rebuilt-per-request alternative** (one map, content rewritten at each pick): invalid for the reasons above and it
  would also rewrite a map under a captured graph that other slots replay.
- **Header-in-device alternative** (one graph for all lists: kernels read `rows` from a map header word and exit
  early): removes the per-key capture and the extra keys, at the price of editing `q8_0_gemv_ids`, `_mcol`,
  `q6k_ids` and `argmax_p_rows_fault`. Not proposed unless the first-window capture measures above one window.

## 4. Residency coupling

- The residency counts the **kept rows' routing only** (`crates/gpu/src/host/swap.rs:148`, "folded into the rule for
  the pass's kept rows only, so a rejected row leaves no trace"). A kept row is the target's own token at its
  position, bit for bit wherever it sits in a pass (`rounds.rs:177`, "each slot's rows are bit for bit its own"),
  so **which experts are hot does not depend on the list**.
- The rule's clock is kept rows, not passes (`crates/runtime/src/swaprule.rs:3-6`, `registry.rs:593`), but two things
  are in passes: a plan fires at the first pass boundary whose ended passes have kept ≥ `every = 4` rows (a 3-row
  pass plans at kept rows 6, 9, 12, 18, ..., `swaprule.rs:1610-1621`), and a flip goes live `delay` passes after
  its boundary (`swaprule.rs:14-16`). A list that changes acceptance (zh: 1.6 → 3.5 kept per window) moves the plan
  stamps by up to a pass and the flips' token-time delay by `delay × E`; the copies' wall time per pass is the
  link's, so a higher-acceptance list gives each flip less token distance and the same wall.
- The target's bits move only at a near tie, when a flip changes which side computes an expert (the lever's own
  statement, `registry.rs:410-413`, "a running residency counts by pass ... moves the passes, the experts' flips
  with them, and the target's bits at a near tie"; q38head2-ab.card:32-35 measured it: the same prompt part from the
  full head's tokens at generated token 102 under a list, and equal under `BLOOMERY_RESIDENCY=off`). The
  `registry.rs:410` wording is stale against `swaprule.rs:3-6` (counts by kept row, flips by pass).
- State persists across requests: only an explicit call resets it (`registry.rs:602-603`). Two lists therefore make
  **the placement at a given token index depend on the request mix through the pick** (a zh request on `C` runs
  ~2.2x fewer passes per token than the same text on `D`, so its flips land at other token distances), while **the
  counts depend on the mix only through the tokens**, as they already do. Greedy output across lists is therefore
  equal only with the residency off (`BLOOMERY_RESIDENCY=off`); the gates in §6 run there, and no gate claims
  bit-equality of a running residency across picks.

## 5. The second shipped list

`D` is `git show 347a86e5:crates/model/data/mtp-head-rows-qwen38-kohangul-65536.txt`, sha `b9deced66fc00185620bb0a8cdc591240e6801e298bdc76438490eed3e426bdc`
(the sha the lead gave, verified here), header `vocab=248320 rows=65536 vocab_sha256=b7f4906b…`, the same tokenizer
digest as `C`. It is restored byte for byte as `crates/model/data/mtp-head-rows-qwen38-kohangul-65536.txt`.

**No re-rank is needed.** Any 65,536-row list's gain over `D` is bounded by its miss rate times `(dAL/dq)/AL ≈ 1.41`:
≤ +1.3 % on en-prose (q .0091), +1.9..+2.2 % on code, +0.4 % on ko-chat, +2.7 % on ko-wiki [derived], and only a
list that holds exactly the held-out ids reaches the bound. The one 65,536-row alternative on record, the q38head list,
covered corpus-prose.ids at .99268 against `D`'s .99197 (`q38head2-ab.card:10-12`): +0.0007 coverage is +0.1 % tok/s
[derived], below the A/B ruler, and it covered Korean chat at .6887 against `D`'s .9861. No candidate beats `D` on en,
code and ko together; a future one is tested by its q on the 11 sets against `D`'s table in §1.1 under the same pick.
Do not re-rank for 0.3.1.

**One reader, two pinned lists.** Today `head_list.rs` holds the one list as constants and `shipped_rows` compares the
text against `SHIPPED_SHA256` (`head_list.rs:31-34`, `:149-166`; the reader is `parse_head_rows`/`read_head_rows`/
`vocab_sha256` in `qwen35moe/place.rs:1989-2210`). The cut is data, not a second reader:

```rust
pub struct ListDef { pub name: &'static str, pub text: &'static str, pub sha256: &'static str }
pub const DEFAULT: ListDef = ListDef { name: "kohangul-65536", text: include_str!(".../kohangul-65536.txt"), sha256: "b9deced6…" };
pub const WIDE:    ListDef = ListDef { name: "cjk-76000",      text: include_str!(".../cjk-76000.txt"),      sha256: "e20c786d…" };
pub fn list_rows(def: &ListDef, vocab: u32, digest: &[u8; 32]) -> Result<HeadPick, ShippedError>;  // shipped_rows with def.sha256
```

`list_rows` is `shipped_rows` with the pin read from the definition; the three tests at `head_list.rs:199-256` run
once per definition. The pair is read by one function (`head_rows_of`, `head_list.rs:123-142`), whose `None` arm
returns both lists of a target whose tokenizer digest matches (else the full head, `HeadWhy::OtherTokenizer`, as
today).

## 6. Gates

Every clause below names its file and FAIL-first mutant. All run with `BLOOMERY_RESIDENCY=off` where they compare
tokens (§4).

| # | Clause | File | FAIL-first mutant |
|---|---|---|---|
| G1 | Pick rule on fixed id vectors built from the two real lists: an all-`C∖D` prompt picks `C`; an English prompt (ids in `D∩C` plus `D∖C`) picks `D`; `S = 2` picks `D`, `S = 3` picks `C` at P = 16; at P = 4,096 `S = ceil(δ*·P) − 1 = 10` picks `D` and `S = 11` picks `C` (the test computes the boundary from the pinned `δ*`, not a literal); the empty prompt picks `D`; two lists of different digests are refused by name | `crates/models/src/headpick.rs` unit tests (pure, `just mac-test`) | (a) rule always `D`: zh case red; (b) `>=` to `>` on τ: boundary red; (c) roles of the two bitmaps swapped: zh and en cases red; (d) `δ*·P` term dropped: long-English-with-20-zh-ids case red |
| G2 | Upgrade detector: 4 misses in 16 do not fire, the 5th does; a miss outside the window of 16 does not count; never downgrades; reset by `restart`; **it sees every committed id**: a draft behind a chooser forced closed before a `D` request, whose reply is Chinese, still upgrades, and the chooser reopens (G10) | same file, and the closed-chooser case on the mock clock in `crates/runtime/src/width.rs` tests | (a) k = 4: the 4-miss case red; (b) a downgrade arm: never-downgrade case red; (c) window not reset by `restart`: reset case red; (d) the detector wired to `record` only, or to `record` and a `stepped` the chooser never calls while closed: the closed-chooser case red |
| G3 | Pick scope: run the rule on each held-out set (fixed 256-id strided chunks, majority pick) and assert the picked list's miss rate on the set is under **that list's own** break-even | `crates/models/tests/head_coverage.rs` (the one test that replaces `every_set_misses_less_than_the_break_even` for two lists) | rule forced to `D`: zh/ja red (q .59-.67 ≥ .0415) |
| G4 | Each list's coverage and leakage gate: `shipped_list()` reads each of the two files (it asserts exactly one file today, `head_coverage.rs:171-195`; it becomes "exactly these two"); `break_even_is_the_derived_one` pins both (`C` .0391 as today `:292-297`; `D` .0415 [derived] with the same ±5e-4); `held_out_text_is_not_a_build_input` runs per list against that list's **own** build record (`C`: `build-sources.txt`; `D`: a new `build-sources-kohangul.txt` naming q38head2-ab's inputs, `q38head2-ab.card:5-12`, the ten `learn` prompts of `tools/ref/data/d2-prompts-ko.tsv` among them: the existing record does not name those) | `head_coverage.rs`, `crates/models/tests/data/head-holdout/` | a held-out item overlapping `D`'s build record: leakage red; `D`'s pin off by 1e-3: pin red |
| G5 | Tokens identical: greedy ids of one prompt set under {full head, `D`, `C`} are equal, residency off | `gate_qwen4exp_mtp.rs` (map uses `:3732`, `:4828`) | a map that aliases the target's buffers: red. **This cannot catch a wrong-list routing bug**: tokens are equal under any list by construction (the head changes the draft only, `registry.rs:406-410`) |
| G6 | Per-slot routing that sees acceptance: two slots through `pass_rows_one_pass`, a zh-chat prompt on slot 0 and an en-prose prompt on slot 1, greedy, residency off; each slot's pick record names its list; run under `BLOOMERY_MTP_WIDTH=fixed` and `BLOOMERY_MTP_WINDOWS=1` (as q38cjk-ab, `q38cjk-ab.card:19-22`: under the cost chooser the width is cut or the gate closes on `D`-zh and E(4) is not measured), slot 0's E(4) under the pick is **≥ 1.5x** its E(4) forced to `D` (derived zh-chat ratio 3.526/1.597 = 2.21, the q38cjk card predicts 2.0-2.5x; a wrong-list slot gives ≈ 1.0x, so 1.5x sits well between and does not ride the band's lower edge) | `gate_qwen4exp_mtp.rs` and `serve_seats` | model-level "last list set wins" state (the list stored on `Mtp38` instead of the slot): slot 0 runs `D` or slot 1 runs `C`, E(4) clause red |
| G7 | Swap: one slot runs en → zh → en with the upgrade; after each change the tokens equal the plain run's and the first headed walk's key carries the new list (`Mtp38::graph_nodes()` names it, `mtp38.rs:1481`) | same | key without the list: the post-switch walk replays the old list's baked grid, E(4) clause red (same fixed-width, 1.5x condition) |
| G8 | (Amended 2026-10-11, §0.1: `max(vocab, slots × R)` words with one map per slot.) Plan invariance: `plan.map_bytes` and the open-time `got == want` equal the single-list load's with two lists; a list set whose rows sum past the vocabulary is refused by name | `mtp38.rs:355-372` clause in `gate_qwen4exp_mtp.rs`; the owner's unit test | overlap allowed: refusal red |
| G9 | Records: the per-request pick record (list, rows, `S`, `P`, class, `from`: `pick` / `upgrade` / `set`) is a `Kind` in `record.rs` (the load's `MTP_HEAD38` at `:1198-1212` is one record per load); `just records-refresh` | `crates/gpu-gates/src/bin/shared/record.rs` area, `tools/bloomery/schema/` | a missing field panics by name (the `Record` machinery) |
| G10 | Chooser class state (§8): (1) a chooser forced closed, then a class change: the next pass drafts; (2) two classes alternating on the mock clock: the second class's first window uses its own profile | `crates/runtime/src/width.rs` tests | (1) without the reopen the closed gate persists for the reply; (2) without the key the second class's first window takes the first class's width |

The timed claim of this change is one same-lease card per AGENTS (a change on the dispatch path), and it is a `noninf` card,
not an `ab` card (a predicted 0.0 % is inside any ruler and the lease refuses it): `prose:512` and `prose:4096`, the pick
against `BLOOMERY_MTP_HEAD_ROWS=<D path>`, same binary, predicted **0.0 %** (the pick sends en to `D`: the identical
dispatch after a P-id bitmap pass of ~4 µs [derived]), margin 1.2 % (the paired ruler at 8 rounds, sd 1.0 %, `q38cjk-ab.card:44-46`).
The pick-against-`C` effect on prose, +1.57 % [derived, §1.1], is **q38cjk-ab's measurement read the other way** (its unset
arm is `C`, its list arm is `D`, the pick's choice on prose) and opens no sitting of its own; the zh/ja effect is functional
(G6). The one term the derivation cannot close is the first window after an upgrade (graph capture time), read once from
G7's run.

## 7. One common owner

### 7.1 What the owner is

The owner is the row-list head: list data format and reader, pin check, the pick rule, the upgrade detector, the
map table, the walk key and the head kernels' call. Three layers, each in the lowest crate its consumers share
(dependency edges from the crates' `Cargo.toml`: `models → gguf`; `runtime → models`; `model → gguf, levers, models,
runtime`; `gpu → model, runtime, levers`; `gpu-deepseek41 → gpu, model`; `gpu-glm5next → gpu, gpu-deepseek41, model,
models`; `app → gpu, model, runtime`, the arch crates optional):

| layer | crate / module | holds |
|---|---|---|
| host data and rules | `crates/models` (new `src/headpick.rs`; `HeadRows` already lives here, `models/src/lib.rs:707`) | `HeadSet` (the lists of a draft), the reader (`parse_head_rows`, `HEAD_ROWS_FORMAT`, `HeadRowsError`, `vocab_sha256`, moved from `qwen35moe/place.rs:1989-2210`), `HeadSel`, the pick rule, the upgrade detector, `break_even` (moved out of the gate so the pin and the rule share one function) |
| device maps and call | `crates/gpu` (new `src/head_rows.rs`) | `RowMaps`: the model's one map buffer with the lists at word offsets and their windows; `project_rows` choosing `q8_0_gemv_ids`/`_mcol` or `q6k_ids` by the output's form and then `argmax_p_rows_fault`; the key type |
| per-request glue | `crates/app` (`mtp.rs`, `arch/deepseek41/draft.rs`) and `crates/runtime` (`speculative.rs` `Draft::class`, `width.rs`) | `HeadPicker` held by `MtpDraft` and `CardDraft`, the width chooser's class key |

`models` is reachable from `gpu-glm5next` (direct), `gpu` and `gpu-deepseek41` (through `model`), `app`, with no
cycle: `models` depends only on `gguf`. `HeadSel` is a plain `Copy` enum (`Full`, `Rows(u8)`) in `models`, so
`runtime` can carry a class as a `u8` without importing it. The `levers` glue (`BLOOMERY_MTP_HEAD_ROWS`) stays in
`model` (`levers` is a leaf crate): `head_rows_of` keeps its place beside each family's list data.

### 7.2 Attach table

| item of the owner | Qwen3.8 MTP | GLM NextN | V4.1 DSpark | architecture fact that forces what stays model-side |
|---|---|---|---|---|
| list data and pins | own (`head_list.rs`: `C`, `D`) | own, new (GLM tokenizer, §7.5) | own, new | each tokenizer has its own ids; lists are data in the family's module |
| reader and refusals | owner | owner | owner | none: the format is tokenizer-agnostic (`draft-vocab.py:1-60`) |
| pick (`S`, `τ`, `δ*`) | owner | owner | owner | `δ*` is derived from a family's `KEPT`, window, row bytes and walks, which are family data passed in (`head_coverage.rs:36-53`); the function is the owner's |
| upgrade detector | owner, fed by `MtpDraft::record`/`stepped` (`mtp.rs:489`, `:838`; `held` defaults to `stepped`, `speculative.rs:130`) | same (`MtpDraft<Body>`, shared) | fed by `CardDraft::accept`/`stepped` (`draft.rs:319`, `:331`) | where a kept id is observed: the body's draft adapter, one line |
| map table | owner (`RowMaps`) held by the model beside the target's `Weights` | owner | owner | the map indexes the target's `output`, which `GpuModel` owns (`gpu/src/model.rs:484-520`) |
| head kernel call | owner's `project_rows` | owner for `Rows`; `Head::enqueue` stays for `Full` until it migrates | owner's `q6k_ids` leg; Markov loop stays | the output's form (Q8_0, Q6_K) is read from the weight, as `head.rs` already does (`OutW`/`head_out_w`, `crates/gpu/src/head.rs:53-100`) |
| walk key | `MtpKey.head: HeadSel` (`mtp38.rs:909-913`) | none: the NextN walk is eager, no capture (`nextn.rs:22-31`, `:1243`) | `(width, list)` graphs captured at load (`draft/mod.rs` "Submission") | whether a body captures walks |
| record fields | owner's `Kind` | same | same | none |
| model-specific | the walk's layers and `head_x` | the host union leg inside the walk; ids-only readback (`PROBS = false`, `glm5next/mtp.rs:70-86`) | block pass, Markov loop (`markov.rs:288-302`), `ds41_markov` over listed rows | DSpark proposes a block in one pass then a serial bigram loop; GLM runs host experts mid-walk |

### 7.3 No per-model branch, flag or trait method: the check

Places where the models differ, and where each sits:

1. **Tokenizer lists and digests**: data constants in each family's module; the owner's functions take `ListDef`s.
2. **Output form** (Q8_0 for Qwen3.8, Q6_K for V4.1, GLM Q6_K or Q8_0 unread): `project_rows` matches the weight's
   form (as `HeadOut` does at `mtp38.rs:152-160`), the same dispatch for all.
3. **Acceptance constants** (`KEPT`, window, walks, row bytes): the owner's `break_even`/`δ*` take a `Family` value
   from each model's side, no `if model`.
4. **Whether a walk is captured**: the key is the owner's; an eager body never builds one.
5. **ids-only readback (GLM)**: the common argmax already writes `p` beside the ids (`mtp.rs:379-390`,
   `ArgmaxPArgs`); GLM ignores `p`. It is a layout fact (`2m + 2` words), so GLM's `Rows` leg reads that buffer; no
   flag.
6. **DSpark's Markov loop**: stays the model's kernels; it needs a list variant of `ds41_markov` that reads
   `w2[map[r]]` and an argmax that maps the row to its id.

**One place that cannot sit on the model side as the code stands, and the cut.** `MtpBody` has `type Head` and
`fn head(m)` (`mtp.rs:234-243`), both per-model items that return what "the load opened" (Qwen: `MtpHead`
`mtp38.rs:762`; GLM: `NextnHead::Full`, `nextn.rs:116-118`). Keeping them lets the owner grow an enum and a
constructor per model. The cut: the maps belong to the **model**, not to a body, because they index the target's
`output` (`GpuModel` owns the `Weights`); `HeadSel` is then the one common `Head` type and `fn head(m)` becomes
`GpuModel::heads()`, an accessor on the common struct. `MtpBody::Head` is removed from the trait, so the trait loses
two per-model items. `MtpDraft::open` reads the maps' lists from the model (`mtp.rs:408-425`). The map buffer is declared after `graphs`, `slot_graphs` and
`parked` and before the body (`gpu/src/model.rs:484-497`'s own convention): a capture addresses it and must drop first.

### 7.4 Migration order: Qwen3.8 first, three landings

Not all three at once. The reasons are the files, the gates and the half-moved risk.

| landing | content | files | class (AGENTS table) and proof | gates `just affected` selects |
|---|---|---|---|---|
| L0 move, host | reader, `HeadRows`→`HeadSet`, `break_even` to `models::headpick`; `head_list.rs` onto `ListDef` with one list | `models/src/{lib,headpick}.rs`, `model/src/arch/qwen35moe/{place,head_list}.rs`, `head_coverage.rs`, `gpu-gates` call sites (`qwen38.rs:342`, `:1370`; `generate_qwen3moe.rs:398`, `:991`, `:2228`) | move: `ptx-scan` identical (`--no-jit` suffices); no timed A/B | crates `models`→ its dependants: wide; narrowed by `just narrow` to the host-path gates plus `gate-models` and the recipe checks |
| L1 move, device | `RowMaps`/`HeadSel` in `gpu`; `Mtp38` onto it with **one** list; `MtpKey.head: HeadSel`; `MtpBody::Head` removed | `gpu/src/head_rows.rs`, `mtp38.rs`, `body38.rs`, `app/src/mtp.rs`, `app/src/arch/{qwen3moe,glm5next}/mtp.rs` (GLM keeps `Full` only), and the `head_map()` consumers (`gate_qwen4exp_mtp.rs:3732`, `:4828`, `generate_qwen3moe.rs:2299`, `mtp38.rs:454`) | move: `ptx-scan` identical; structural lines (graph node count per key, eager = replay) | `gate-gpu-qwen4exp-mtp`, `gate-gpu-lib` (never dropped), the app and serve gates |
| L2 feature | (recut 2026-10-11, §0.1: one list plus warm spare rows per slot, the fold and the refold, S1-S3, the G8 exception) second list, pick, upgrade, chooser class state (§8), records, lever text | `head_list.rs`, `headpick.rs`, `mtp.rs`, `runtime/src/{width,speculative}.rs`, `record.rs`, `registry.rs:389-415`, cards `q38cjk-ab.card:47-50` conditions (`head=rows=76000 shipped` becomes the `from=pick` word) | behaviour: G1-G9, one A/B (§6) | the L1 set plus `gate-runtime`, `gate-levers`, `check-recipes`, `records-refresh` |
| later: GLM (list build + attach), DSpark (list + kernels) | §7.5, §7.6 | | add: ptx-scan = base plus exactly the new entries (DSpark) | |

Why this order: L0 and L1 are moves that change no byte of the Qwen3.8 binary, so they land first and cheap
(`ptx-scan` equal to base) and every later landing starts on the final owner. The half-moved risk is a second copy of
the reader or of the key: L0 moves the reader once, and the owner has exactly one consumer until L2, which is the
intended intermediate state. GLM and DSpark can attach with **zero lists** (the full head through the owner,
`HeadSel::Full`), which is a no-op for their kernels, so their attach does not wait for a list; building their
lists is gain-gated (§7.5, §7.6). Doing all three at once would put a DSpark kernel add (an `add` class landing,
`gate-ptx-spill` pins) and a GLM tokenizer build in the same train as the Qwen3.8 key change, and a red in either
would hold the 0.3.1 feature.

### 7.5 The gain, per model, with the resource timeline

**Qwen3.8 (the decided case).** Per window the card runs the chain (a refresh walk and 2 own walks, one readback)
then the verify, in series: the chain feeds the verify, which feeds the next chain; inside the verify the card and
the host experts overlap. The head is in the chain, **on the critical path**: card HBM only, no PCIe, no NVMe.
Per window `[derived]` at 579-701 GB/s (`head_coverage.rs:45-47`): `D` 3 × 178.26 MB = 534.8 MB, 0.76-0.92 ms (2.2 %
of the 35.3 ms window); `C` 620.2 MB, 0.88-1.07 ms; the full head was 2.026 GB, 2.9-3.5 ms (`q38head2-ab.card:14-17`).
So the head is a bytes term on the serial chain, not under a shadow, and the table in §1.1 is the gain: +1.6..+2.8 %
on en/code against cjk-only, 0 against the kohangul list of before.

**(a) GLM MTP** (`crates/app/src/arch/glm5next/mtp.rs:29-39` returns `NextnHead::Full`). Terms from
`docs/cards/glmwire-mtp.card:1-10`: plain step S1 = 47.7 ms, V2/S1 = 1.34-1.52 (`:16` correction), chain D =
2.4-2.7 ms (card share 1.29-1.42 ms including the shared head, host union 0.9-1.0 ms), α = 0.684, so a window is
V2 + D = 66.3-75.2 ms [derived]. GLM's head is `bloomery_gpu::head::Head` over the target's `output`, vocab
151,552 × n_embd 4,096 (`glm5next/place.rs:1253-1256`); the file's `output` type was not read (the box's models are not
touched in a design round), so both forms the head supports (`crates/gpu/src/head.rs:53-100`) are bracketed: Q6_K 3,360 B a row =
509.2 MB, 0.73-0.88 ms; Q8_0 4,352 B a row = 659.6 MB, 0.94-1.14 ms.

Timeline of one window: card (verify dense) ‖ host (union experts) → card chain part A → **host union leg
0.9-1.0 ms** → card part B ending in the head → readback → next verify. The head is after the host leg and before
the readback: serial, not shadowed (`nextn.rs:22-31`). A 65,536-row list saves (151,552 − 65,536) rows: 289.0 MB
(Q6_K) or 374.4 MB (Q8_0), 0.41-0.50 or 0.53-0.65 ms, **0.55-0.98 %** of the 66.3-75.2 ms window. The acceptance cost is
`α q/(1 + α(1−q)) ≈ 0.406 q` for a width-1 proposal: 0.37-0.61 % at q = .009-.015 (Qwen's `D` rates as the stand-in,
no GLM list exists). Net **+0.0..+0.6 %, centre +0.2 %**, inside the ruler at any round count that fits a sitting.
The chain-overlap lever on the card (`glmwire-mtp.card:25-26`, "the next window's row 0 under the chain") would put the
chain's card leg under the verify's row 0 and take the head term to 0. Verdict: attach for the common owner, make no
GLM speed claim.

GLM's list: `tools/ref/draft-vocab.py` is tokenizer-agnostic (target first shard in, digest and eos/pad from the
header, `draft-vocab.py:1-60`) and needs, for GLM ids: corpora re-tokenized with the GLM tokenizer (the
`bloomery-tokenize` path the q38 build names, `build-sources.txt:11-12`), ik greedy continuations
(`--greedy`, `argmax_ref.cpp --gen`; `docs/cards/glmmtpref-dump.card` is the existing GLM MTP dump), the 11 held-out
sets re-tokenized (11 × ~40,000 ids), and a `head_coverage` instance for GLM's digest with the family's own model (a
width-1 `KEPT`/α, window 66.4-75 ms, row bytes by the output form). GLM serves Chinese and English alike, so it needs
the same two-class structure (a default and a CJK-wide list). Effort M; the gate is the coverage gate, per family
(`Family` data in the model's module, the function in the owner).

**(b) V4.1 DSpark** (`crates/gpu-deepseek41/src/draft/head.rs:1-20`). Correction to the premise: `DraftHead` does
**not** gemv the projection once per row; `enqueue_logits` is three launches over the block's `m` rows (rms_norm,
q8_1 quantize, one `q6k_gemv` of `M ≤ 8` columns, `draft/head.rs:97-115`, `gpu/src/lib.rs:2871-2916` and the kernel doc
above `q6k_gemv`), reading the projection once: 129,280 × 4,200 B = **543.0 MB** (n_embd 5120, Q6_K; `v41-b5-plan-report.md:384`,
`ds41_meta.rs:157`). The Markov loop is `m` serial steps of two launches each (`draft/head.rs:134-157`, `markov.rs:288-302`):
step `c` adds `w2[v] · w1[prev]` for every `v` (bf16 `[rank 256, vocab]`, 512 B a `v`, **66.2 MB a step**,
`markov.rs:1-30`) and then re-runs `argmax_rows_fault` over all `m` rows (`:301`). For m = 3: 543.0 + 198.6 =
741.6 MB, 1.06-1.28 ms at 579-701 GB/s [derived; the second card's rate is not measured here]; m = 5 (the file's
`block_size`, `gate_dspark_read.rs:54`): 874.0 MB, 1.25-1.51 ms. A 65,536-row list (50.7 % of the rows) with the
Markov add restricted to listed rows reads 275.2 + 3 × 33.6 = 375.9 MB: saves 365.7 MB, **0.52-0.63 ms a window**.

The window is not a measured number in the cards I read. ik's draft pass is a flat 10.32 ms and its k = 3 window
112.4 ms (`rig-log log/2026-09-19.md:128`: 10.32 + 50.21 + 3 × 17.28), ours is shorter; for T in 60-112 ms the saving
is 0.5-1.0 % and the acceptance cost at q ≈ .01 is ~1.3 % (Qwen's `(dAL/dq)/AL`, no DSpark profile available), net
**−0.9..−0.4 %** [derived; ≈ 0 only if the list's q is ≤ .004]. The proposal is on the critical path (`propose` returns once its ids are on the host, the
verify needs them: `draft/mod.rs` contract steps 3-5), so it is not shadowed, only small. Model says ≈ 0; attach,
build no V4.1 list for speed.

Attach cost in files, lines: `draft/head.rs:97-157` (a `Rows` leg calling the owner's `q6k_ids` and a listed Markov step; the
Markov loop stays), `markov.rs:288-302` plus the `ds41_markov` kernel (a rows variant reading `w2[map[r]]`, ~30 kernel
lines and its launcher) and an argmax with a map (the owner's `argmax_p_rows_fault` has one but writes `2m + 2`
words, DSpark's `tok` holds `m + 2`, `draft/head.rs:46-49`), `block.rs:970-977` (the head's call), `draft/mod.rs`
"Submission" (graphs per width captured at load, `1..=MAX_WIDTH = 5`: with two lists 10), `draft/load.rs` (the lists),
`app/src/arch/deepseek41/draft.rs:281-336` (the pick hook and the observer), `serve_seats/ds41.rs:1019-1030` (the list
load), a V4.1 `head_list` module, `gate_dspark_*` and `gate-ptx-spill` rows. About 8 files and 250-350 lines with one
kernel variant (an `add` class landing). The Markov loop stays the model's.

### 7.6 How the reference engines shrink a draft head

Read on the box, read-only (a Haiku 5.5 sub-round, `BLOOMERY_REMOTE=~/repo/bloomery` because this tree's remote dir
did not exist; nothing synced); three claims re-read by me are marked.

| engine | reduced head or vocabulary for a draft | per-request pick |
|---|---|---|
| ik_llama.cpp (`/home/user/ik_llama.cpp`, oracle) | none. The MTP/DFlash/DSpark graphs project with `n_vocab` rows of `model.output` (`src/graphs/build_qwen4exp.cpp:696`, `build_deepseek4.cpp:1674`, `build_dflash.cpp:618-619`); DSpark is a full-vocab bigram bias, `markov_w2 {markov_rank, n_vocab}` (`src/llama-load-tensors.cpp:2672-2673`); Gemma4's centroid tensors are loaded but unused: "kept in the GGUF for future use", full-vocab matmul (`src/graphs/build_gemma4.cpp:867-872`, re-read); EAGLE3 is a stub (`common/speculative.cpp:677-697`) | none; the only map is a text-string vocab map for an incompatible external draft, built per load (`common/common.cpp:1512-1517`) |
| exllamav3 (`/home/user/exllamav3-src`) | none. MTP heads borrow the target's `lm_head`, n_vocab rows (`exllamav3/architecture/glm_moe_dsa_mtp.py:204-206`, `qwen4_exp_mtp.py:150-151`, `glm5_next_mtp.py:202-204`); DSpark head full vocab, `markov_w2` out = vocab (`deepseek_v4_mtp.py:307-317`, `:186-188`); DFlash-Laguna asserts `draft_vocab_size == vocab_size`, "a reduced draft vocabulary is not supported" (`dflash_laguna.py:67-70`, re-read) | none |
| mistral.rs (`/home/user/mistral.rs`) | **Gemma4 assistant only**: a checkpoint-trained centroid head, `masked_embedding.centroids` (hidden → 2,048 centroids) and `token_ordering [vocab]`; per step the top 32 centroids pick `32 × V/2048` ids, only those rows of the assistant's embedding are dotted, the rest set to `min − 1` (`mistralrs-core/src/vision_models/gemma4/mtp.rs:777-846`, `:804-824` re-read); every other draft (Qwen3.5 MTP, DFlash) is the full head | **per step, from the hidden state**, not per request; no `d2t`/`t2d`/FR-Spec/EAGLE-3 anywhere |

Where we diverge: nobody ships a frequency-list head; ours is a static list chosen from the target's own tokens and
corpora, picked per request, and it needs no trained table. mistral.rs's per-step centroid pick is a learned,
state-dependent reduction (V/64 rows a step at its defaults [derived from `mtp.rs:806`], ~10.5 MB a step against our
178 MB at 65,536 rows) but needs centroids trained into the checkpoint; none of Qwen3.8, GLM or V4.1 carries them, so
it cannot be adopted without training. Idea-ledger line: "centroid head from k-means of the target's `output` rows:
head bytes −94 % at an unmeasured q; needs a held-out sweep and a derivation of q(centroid top-k) before any round."

## 8. The MTP width chooser's state per head class

Facts: the chooser is `runtime::width::Choosing`, one per slot on Qwen3.8 and GLM (`drafted.rs:100-107`) and one for
the single-slot V4.1 seat (`ds41.rs:1019-1030`, `behind` `:1120`). A request's reset prints the tally and restarts the
inner draft; the chooser's state crosses requests (`drafted.rs:351-358`; `width.rs:448-500`).

Update rule and warm-up (`width.rs`): the acceptance profile is `a_i = mean_i · g_i`; `mean_i` is a running mean with
step `max(1/seen, 0.05)` (`:779-786`, an e-fold of 20 windows once `seen ≥ 20`, 95 % in ~60), `g_i` moves 0.05 of the
way toward each tested position's hit (`:807-822`, `Rule::calibrate`); costs are medians of the last 8 readings per width,
decisions need 4 (`Rule::DEFAULT`, `:172-180`); a width moves only past `margin = 1.03` (`:743-766`).

**Re-derivation.** What differs between the two head classes?
- *Cost.* The medians differ by the head bytes only: 0.122 ms of a 35.3 ms window, **0.345 %**, under the margin
  (3 %) and the 8-reading median's own scatter. Not worth keying.
- *Profile.* `p_i` is the draft's softmax over the list's rows, so a larger list moves it by the mass of the extra
  rows (≤ the miss-rate difference, ≤ 0.9 % on en); the calibration `g_i` absorbs the rest. The acceptance profiles
  of the classes' own texts differ little in the model: AL 3.52-3.66 (`C` on zh/ja) against 3.53-3.65 (`D` on en and code),
  a ≤ 4 % spread [derived, §1.1]. Content differences (zh vs en replies) are real but are a property of the text, which
  the chooser re-learns in ~20-60 windows either way and the head class only loosely predicts.
- *Rate flips.* A stale profile changes the chosen width only past the 3 % margin, and the loss of running the next
  best width is a second-order term around the optimum (`adaptive-picks.md` §3.1 item 4); on Qwen3.8 the gate sits at
  k = 3 with the widths within the margin (`draftgate-noninf.card:3-6`).

**Two parts, one required.** (1) *Required with the upgrade (§1.4):* a class change reopens the chooser. Switching the class
makes the chooser take that class's profile; a class it has not seen starts fresh (`seen = 0`: the running mean steps `1/seen`,
so the first windows set it, `width.rs:779-786`) and the gate is set open (`width = D::WIDTH`, `since_probe = 0`,
`closed_period` as after a close), so the next pass drafts and learns instead of waiting for a shadow or a probe. Without
this a gate closed before a `D`-zh request never reopens within the reply (§1.4). (2) *Optional, the survey's item:* restore
the class's own profile when its class returns. That part is the one worth 0 to +0.3 %. The loss of a mis-rated window is bounded by the spread, ≤ ~4 % of rate, and only on the 26-52 % of a reply's windows
that run on the other class's state (20-40 of ~77, `adaptive-picks.md` §3.1 item 3): a **ceiling of +2 %** after a class
switch (not +2.5 %, which assumed a 0-10 % mis-rating loss against the 3 % margin the rule itself enforces), and
**0..+0.3 %** expected [derived]; 0 on a stream of one class. It is under the A/B ruler (±1.0 % at four rounds), so
it has **no timed card**; the pin is a mock-clock unit test.

**Where the state lives.** Keyed by the class the head owner hands out, inside the chooser (the owner of the state,
`runtime::width`), and **only the profile**: `gains`, `mean`, `seen` become `Vec<Profile>` indexed by a `u8` class;
the cost medians, the gate and the probe clocks stay global (cost is class-invariant within the margin). The chooser
learns the class through the `Draft` trait, one defaulted method (`fn class(&self) -> u8 { 0 }` beside
`unproposed`/`before_plain`, `runtime/src/speculative.rs:105`, `:140`), read at the top of `propose` (`width.rs:925`) and at `begin` (`:912`). `MtpDraft`
and `CardDraft` return their `HeadSel`'s class; the lookup draft keeps the default 0. This meets both shapes without a
model branch: the per-slot choosers of Qwen3.8 and GLM and the per-process chooser of V4.1 each hold the map for the
classes they have seen, switched at `begin`. The default method is a common trait item with one meaning ("the
head class of this request"), not one per model. A class change is read at the top of `Choosing::propose` (`width.rs:925`), not only at `begin`, because the upgrade changes the
class mid-request. Verdict: S; part (1) rides L2 with the upgrade (G2, G10), part (2) is the same code and costs nothing
extra; no timed claim.

## 9. Records, levers and cards that change

- Lever `BLOOMERY_MTP_HEAD_ROWS` (`registry.rs:389-415`): unset = the per-request pick between the two shipped lists on a
  matching tokenizer, else the full head (today's `OtherTokenizer`); `full` = the full head; a path = **one** list for
  every request (no pick: a given list means the user chose it). The doc text, `tools/levers-direct.txt` and
  `just check-levers` move with it.
- The load prints the pick once (`qwen38.rs:1366-1370`, `MTP_HEAD38` `record.rs:1198`). A per-request pick needs a
  per-request record (G9) and `just records-refresh`; the cards whose conditions read `head=rows=76000 shipped`
  (`q38cjk-ab.card:47-50`) change.
- `docs/cards/q38head2-ab.card:35` and `registry.rs:410` say the residency "counts by pass"; `swaprule.rs:3-6` counts
  kept rows and only the plan stamps and flip delays are pass-based (§4).
- Read once at the first run, not derivable: the first headed window after an upgrade (graph capture ms; §3).

## 10. What was not measured

- GLM's `output` type (bracketed Q6_K/Q8_0), V4.1's window on our engine, the second card's head rate, the capture
  time of a walk key, DSpark's and GLM's list miss rates (Qwen3.8's `D` rates stand in).
- The q values are on held-out sets of text, not of prompt/reply pairs; the pick's table (§1.2) takes chunks of
  those sets as prompts. A translation prompt (English text, CJK reply) is the case only the upgrade covers, and the
  held-out sets hold none.
- The box reads ran read-only through `BLOOMERY_REMOTE=~/repo/bloomery` (the round's own remote dir did not exist, rc
  66) and synced nothing; the model files were not opened.

## Appendix A: reproduction of the counted numbers

All counted numbers are arithmetic on checked-in files; nothing here ran on a card. Python 3, `python3 -I`.

```python
W = "<repo root>"; SETS = ["zh-wiki","zh-chat","ja-wiki","ja-chat","ko-wiki","ko-chat","en-prose","en-math","code-rust","code-python","code-c"]
def ids_of(path):                       # one id a line, '#' lines are headers
    return [int(l) for l in map(str.strip, open(path)) if l and not l.startswith("#")]
C = set(ids_of(f"{W}/crates/model/data/mtp-head-rows-qwen38-cjk-76000.txt"))          # 76,000
D = set(ids_of("kohangul-65536.txt"))   # git show 347a86e5:crates/model/data/mtp-head-rows-qwen38-kohangul-65536.txt
KEPT, VOCAB, WALKS, ROW, GBS, WIN = [0,3,2,21], 248320, 3.0, 2720.0, 701.0, 37.43   # head_coverage.rs:36-53
def al(q):                              # head_coverage.rs:59-78
    w = sum(KEPT); return 1 + sum(sum(KEPT[j:]) / w * (1 - q) ** j for j in range(1, len(KEPT)))
def saved_ms(rows): return WALKS * (VOCAB - rows) * ROW / (GBS * 1e9) * 1e3
T_C, T_D = WIN - saved_ms(76000), WIN - saved_ms(65536)                                # 35.424, 35.302 ms
for s in SETS:
    ids = ids_of(f"{W}/crates/models/tests/data/head-holdout/{s}.ids")
    qd = sum(i not in D for i in ids) / len(ids); qc = sum(i not in C for i in ids) / len(ids)
    print(s, qd, qc, qd - qc, 100 * ((al(qc) / T_C) / (al(qd) / T_D) - 1))             # the §1.1 table
# pick statistic of a prompt chunk:   S = sum(i in C and i not in D for i in ch) - sum(i in D and i not in C for i in ch)
# upgrade: first t at which >= k of the last n ids are not in D (notin = [0 if i in D else 1 ...]), n = 16, k = 5
```

The tie `δ*` solves `AL(q_C)/T_C = AL(q_C + δ)/T_D` by bisection (0.00243 at q_C = .05 to 0.00245 at 0); the chunk and trigger tables
draw 400 chunks and 300 start offsets a set with `random.seed(7)`. The scratch scripts that produced them were
`model.py` and `rule.py` in the round's scratch directory; this appendix is their content.
