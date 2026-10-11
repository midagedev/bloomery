# headpick-warmup: a warm-up draft head and a saved profile against the two static head lists

A Mac simulation (2026-10-11; no card, no box, no code change) that prices the user's proposal against `headpick.md`'s
two static lists. The proposal: one per-slot row set of a fixed row count R, started from one language-neutral base,
filled from the prompt's ids and the accepted ids, its map rewritten in place so the captured walk stays; and a second
arm, a saved profile that the next start begins from.

Every number is one of: `[sim]` (a count over the checked-in held-out sets through the model of `head_coverage.rs`,
script in Appendix A), `[derived]` (arithmetic on the cited terms), `[extrapolated]` (a fit carried past the data),
or `[unmeasured]` (an assumption named as one). Nothing here ran on a card. `D` is the 65,536-row kohangul list, `C` is
cjk-76000, as in `headpick.md`.

## 0. Result

| # | Question | Answer |
|---|---|---|
| 1 | Does one warm-up head replace the lists? | **No: verdict (b).** With a language-neutral base (lowest ids, en+code ranking, or the ids both lists hold) the next-id miss on zh/ja/ko stays far above the lists': tok/s against the best static list, worst set of the zh/ja class, is 0.51-0.63 at t=0, 0.57-0.69 at 1,024 ids seen and **0.66-0.77 at 4,096** (ko, `lowid` and `ec`: .52, .63, .75; `both` holds Korean at parity) `[sim]`. Parity needs a miss under the best list's + .007 (1 % tok/s = .007 of q `[derived]`); the `lowid` arm still misses .19-.33 at 4,096 ids. |
| 2 | Does the saved profile rescue it? | **No: not (c).** A profile built from 20,000 ids of the same language, started on the other half, lifts t=0 to 0.79-0.90 on zh/ja/ko (cold 0.52-0.59) `[sim]`; the fit to parity is 0.6-4 million ids `[extrapolated]`. On a base that already covers the language (`cblind`) the profile adds +0.3 to +0.8 points on en/code at t=0 and within +-0.2 on zh/ja/ko, at the seed spread (0.6 points). |
| 3 | What does work | **The shipped list plus spare rows, not a neutral base.** `C` with 2,048 spare rows (its highest ids below the control block; ids >= 248,044 stay protected) made writable and filled from the accepted ids (arm `cblind`): every one of the 11 sets is at 0.991 or more of its best static list by **256 ids seen**, zh/ja/ko at 0.998-1.01 from id 0, en/code 0.983-0.992 at id 0 `[sim]`. Against the headpick rule (pick + upgrade) it is a trade inside about 1 point from 256 ids on, a tie at the 4-round ruler (+-1.0 %) but not inside the seed spread: the pick is ahead on en/code (+0.8 and +0.9 points on en-prose and code-python at 256 ids), `cblind` is ahead on zh/ja in the first window (+4-5 points at id 0, while the pick starts on `D`). It needs one list, no pick, no upgrade, no class state and one captured-walk key per (m, own) per slot (§5, §7). |
| 4 | Fill policy | Prompt+accepted beats prompt-only by 0.5-0.7 points on `cblind` and 4-8 points on the neutral arm at 256-1,024 ids. LRU and LFU agree to three digits wherever the writable region does not overflow, which is every spec-grid arm (F >= 10,464) and `cblind` (<= ~660 ids written in 20,000); the neutral arm at F = 1,024-2,048 overflows and the two differ by 1-7 points with no winner. Base size 32K/48K/64K changes nothing for the same reason (§3.6). |
| 5 | Cost a step | Ids changed per accepted token: .002-.033 on `cblind`, .19-.51 on zh/ja/ko under a neutral base `[sim]`. One async host-to-card copy of the dirty tail (<= 8 KB at F = 2,048) in the windows that wrote, **no kernel, no graph node**: <= 0.06 % of the 35.3 ms window in a window that wrote, <= 0.007 % averaged `[derived]`, with the copy's fixed cost assumed <= 20 us `[unmeasured]` (§5.1). |
| 6 | Captured graph | **Kept**: a fixed R fixes the map's address, `rows`, the grid and the argmax `n`; only the map's contents move, between chains, stream-ordered (§5.2). |
| 7 | Per-slot map | A writable copy per slot in `DraftSeq38`, swapped with `graphs` (`mtp38.rs:561-566`). It **contradicts headpick G8** (plan invariance): `plan.map_bytes` becomes slots x R x 4 B (§5.3). A map shared by two slots breaks no token; it costs 5.8-7.6 % tok/s only on a neutral base with a small writable region (§5.4). |
| 8 | Profile owner | **Nothing persists today, and no saved state reaches disk** (the prompt cache and the park table are host RAM). The file is **card-wide only** (residency counts, width chooser): one file, one reader, `crates/models/src/profile.rs`, model identity checked, refusals by name (§6). **The head has no section**: its set is per session and a function of the ids, rebuilt at admission by the prompt fold (§4.1, §5.6, §6). |
| 9 | KV-prefix restore (turn 2 restores turn 1's row set; added at the lead's request) | **A restore is the stream that never stopped, exactly** (79,200 of 79,200 windows equal; the row set after a turn equals the same ids folded in order, 19,800 of 19,800 `[sim]`): the head's set is a function of the ids so far, so the restore is the admission's prompt fold and needs no stored copy. Ratio to the best list at turn-2 t = 0 after T1 = 256 / 1,024 / 4,096 ids, class minimum: neutral base, zh/ja cold start .499 -> restore .528 / .568 / .657, ko .522 -> .569 / .632 / .747 (never parity); `cblind`, code .983 -> .991 / .996 / .999, en .989 -> .991 / .994 / .994, zh/ja/ko +0.0 to +0.1, and the gain is gone within about 256 turn-2 ids (+0.8 on code, +0.3 on en averaged over a 512-id reply) `[sim]`. **A blended card-wide profile as the start of a miss loses to the cold base on `cblind`: ja -1.8 to -2.2 points, ko -0.6, en +0.3, code +0.9** (§4.1): it measures "rows never blend across streams" and contradicts "a miss starts from the global profile" for the head. |

## 1. Decomposition on paper, and the resource timeline

**Terms.** For an arm with R fixed rows, `tok/s = AL(q) / T(R)` (`head_coverage.rs:59-78`, `AL(q) = 1 + sum_j A_j (1-q)^j`,
`T(R) = 37.43 ms - 3 (248,320 - R) 2,720 B / 701 GB/s`). R is fixed, so `T` is fixed: the warm-up moves tok/s only through
q(t), the share of the next ids outside the row set after t ids seen. Near the optimum `(dAL/dq) / AL = -1.41` `[derived]`,
so 1 % tok/s is .007 of q. q(t) splits into the base's miss (known from the lists and the held-out files:
the counts of Appendix C, e.g. the lowest-id rows miss .70 of zh-wiki at R = 76,000) and the unseen mass after t ids,
which the Good-Turing estimate puts at `N1(t)/t` (the non-base ids seen exactly once in the first t, over t).

**Prediction (written down before the simulation ran; the predicted columns of §3.7 are its numbers).** Neutral base: q(4,096) = .15-.30 on zh/ja/ko, so tok/s
0.68-0.81 of the best list; no parity inside 4K ids; parity needs `N1(t)/t <= ~.04`, a t well past 20,000 ids. En/code reach
parity inside 256 ids. The one term the derivation cannot fix is whether the held-out streams behave like exchangeable draws
(Good-Turing assumes it); the simulation reads that term.

**Resource timeline, per window (35.3 ms, one slot, Qwen3.8, A6000)** `[derived]` from `headpick.md` §7.5 and the cited terms:

| resource | work of the window | ms busy | note |
|---|---|---|---|
| card SMs / HBM | verify (dense + routed), chain (refresh walk + 2 own walks) | ~35 | the head reads R x 2,720 B x 3 walks: 534.8 MB at 65,536 rows, 620.2 MB at 76,000 (0.88-1.07 ms), **on the serial chain** |
| host CPU | map update: <= 4 kept ids, a bitmap test and an LRU touch each | < 0.001 | `[unmeasured]`: ~0.1 us an id; prompt ingestion P x 0.1 us = 0.4 ms at P = 4,096, once |
| PCIe | the dirty tail of the map: <= F x 4 B = 8 KB at F = 2,048 (0.31 us at the measured 26.28 GB/s, `pcie-a6000.card:2`); a full map 304 KB = 11.6 us | < 0.012 | only in windows that wrote (<= 12 % of windows on `cblind`) |
| host DRAM | the shadow map and its recency list, 0.35 MB a slot | 0 | |
| NVMe | the profile write, 0.4 MB every N kept tokens, background thread | 0 | not on the step |

The wall is the card's chain plus verify, as today. The map update is serial with them but three orders of magnitude under
the window, so no flow lever (asynchrony, bulk, SIMD, SIMT) is worth a line: a term of 0.06 % is below the paired ruler's
floor (+-0.3 % at 32 rounds, `AGENTS.md`). The design is judged by ownership and graph validity, not by speed.

## 2. The simulation

Script: Appendix A (`python3 -I`, standard library only). Inputs: the 11 held-out sets
(`crates/models/tests/data/head-holdout/*.ids`, 37,624-40,000 ids each) and the two lists.

- **The lists carry no rank.** `tools/ref/draft-vocab.py` writes ascending ids and `HeadRowsError::Unsorted`
  (`parse_head_rows` in `crates/models/src/headpick.rs`) refuses any other order, so "ranked high in both lists (the larger of the two ranks)" cannot be
  built. The neutral bases are therefore: `lowid` (the lowest ids: byte-level BPE numbers tokens in merge order, so this is a
  prior from no data, `draft-vocab.py` docstring); `ec` (ranked by id count in the en+code halves only, then the lowest ids,
  cross-fitted: a base used on half h is trained on the other half); `both` (the ids both lists hold, 45,829, ascending, then
  the rest of the union).
- **The row set.** R rows. The first B ranks of the base are protected; the other R - B rows start as the base's own ranks
  B..R-1 (so t = 0 is the neutral R-row list whatever B is) and are overwritten, lowest-ranked first, by seen ids that are not
  rows; after that the least-recently (or least-frequently) used seen row goes. R = 65,536 and 76,000; B = 32,768, 49,152
  (48K) and 65,536 (64K; = R is the static neutral list).
- **A trial.** A random start offset in a set (200 per half, 400 per set, seed 7), then 4,352 ids walked in windows of
  4 kept ids (`DELTA_LANES`): a window's ids are in the row set from the next window on. At t in {0, 64, 256, 1,024, 4,096}
  ids seen the miss is read over the next 256 ids (64 is in the json) two ways: **online** (prompt+accepted: the set keeps
  updating) and **prompt-only** (the set freezes at t). The ids before t are the prompt.
- **Best static** = the better of `C` and `D` on that set and window (an oracle pick). The **pick** column is the headpick
  rule exactly: `S >= max(3, .00245 P)` on the prompt, upgrade `D -> C` at 5 of the last 16 kept ids outside `D`, sticky,
  its tok/s the time-average over positions of each list's `T / AL`.
- **tok/s** = `AL(q)/T(R)` of the window's mean q (`[sim]` inputs, `[derived]` model). Only ratios against the best static
  list are claimed.
- **Instruments.** (1) Exact identities (smoke test in the session): a static arm's miss equals the count of ids outside its
  rows; the fill-only arm's miss equals the count of first-seen ids at the 4-id delay; the set size stays R under eviction.
  (2) Seed spread: three offset seeds, the largest spread of any ratio over sets and t: **0.006** (cjk-76000 + spare rows),
  .007 (`both`), .016 (`lowid`, small windows). A ratio is claimed to +-0.6 points. (3) The held-out sets are concatenated
  documents (articles, replies), not prompt/reply pairs: a prompt of t ids spans unrelated documents of one language. Real chats
  carry topic locality the sets do not.

## 3. Miss rate to tok/s

### 3.1 The static lists and the headpick rule

| set | q C | q D | best static | C vs best | D vs best | pick+upgrade vs best at t=0 / 64 / 256 | prompt-only pick at t=64 / 256 |
|---|---|---|---|---|---|---|---|
| zh-wiki | 0.0375 | 0.6794 | C | 1.000 | 0.412 | 0.962/1.001/1.001 | 0.994/0.990 |
| zh-chat | 0.0357 | 0.5794 | C | 1.000 | 0.468 | 0.961/0.999/1.000 | 0.975/1.000 |
| ja-wiki | 0.0064 | 0.5813 | C | 1.000 | 0.448 | 0.956/1.000/1.000 | 0.997/0.998 |
| ja-chat | 0.0060 | 0.5938 | C | 1.000 | 0.440 | 0.953/1.000/1.000 | 1.000/1.000 |
| ko-wiki | 0.0056 | 0.0201 | C | 1.000 | 0.983 | 0.995/0.999/1.001 | 0.997/1.000 |
| ko-chat | 0.0033 | 0.0033 | D | 0.997 | 1.000 | 1.000/1.000/1.000 | 1.000/1.000 |
| en-prose | 0.0183 | 0.0099 | D | 0.985 | 1.000 | 1.000/1.000/1.000 | 1.000/1.000 |
| en-math | 0.0187 | 0.0071 | D | 0.980 | 1.000 | 1.000/1.000/1.000 | 1.000/1.000 |
| code-rust | 0.0256 | 0.0121 | D | 0.978 | 1.000 | 1.000/1.000/1.000 | 1.000/1.000 |
| code-python | 0.0322 | 0.0147 | D | 0.972 | 1.000 | 1.000/1.000/1.000 | 1.000/1.000 |
| code-c | 0.0339 | 0.0163 | D | 0.972 | 1.000 | 1.000/1.000/1.000 | 1.000/1.000 |

(window 256 after t = 0; the pick column's t = 0 is an empty prompt: it starts on `D` and upgrades in a median ~7 ids.)

### 3.2 The arms against the best static list: minimum over the sets of a class, at t = 0 / 256 / 1,024 / 4,096

| arm | en | code | zh/ja | ko |
|---|---|---|---|---|
| pick+upgrade (headpick 1.2 + 1.4) | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0.953/1.000/1.000/1.000 | 0.995/1.000/0.999/0.999 |
| static cjk-76000 | 0.980/0.981/0.979/0.978 | 0.972/0.974/0.972/0.971 | 1.000/1.000/1.000/1.000 | 0.997/0.996/0.997/0.997 |
| static kohangul-65536 | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0.412/0.408/0.404/0.409 | 0.983/0.984/0.987/0.981 |
| lowid R=76000 B=48K, online | 0.990/0.994/0.997/0.999 | 1.001/1.004/1.008/1.010 | 0.509/0.526/0.569/0.659 | 0.519/0.567/0.630/0.749 |
| lowid R=76000 B=48K, prompt-only | 0.979/0.989/0.993/0.996 | 0.996/1.001/1.005/1.007 | 0.393/0.470/0.517/0.617 | 0.426/0.484/0.558/0.696 |
| lowid R=65536 B=32K, online | 0.981/0.988/0.992/0.996 | 0.997/1.002/1.007/1.010 | 0.509/0.527/0.570/0.660 | 0.513/0.562/0.627/0.749 |
| ec R=65536 B=32K, online | 0.996/0.998/0.999/1.001 | 1.009/1.011/1.011/1.013 | 0.510/0.527/0.570/0.660 | 0.513/0.562/0.627/0.749 |
| both R=76000 B=48K, online | 0.999/1.000/1.001/1.001 | 1.002/1.005/1.007/1.009 | 0.625/0.656/0.687/0.767 | 0.999/0.999/0.999/1.000 |
| cjk-76000 + 2048 spare rows (highest ids), online | 0.989/0.992/0.993/0.994 | 0.983/0.991/0.996/1.000 | 1.001/1.001/1.001/1.000 | 0.998/0.998/0.998/0.998 |
| cjk-76000 + 2048 spare rows, prompt-only | 0.980/0.985/0.988/0.992 | 0.972/0.985/0.991/0.996 | 1.000/1.000/1.000/1.000 | 0.997/0.997/0.997/0.997 |
| cjk-76000 + 6000 spare rows, online | 0.989/0.992/0.993/0.994 | 0.983/0.991/0.996/1.000 | 1.001/1.001/1.001/1.000 | 0.998/0.998/0.998/0.998 |
| cjk-76000 + fill ranked by held-out halves, B=48K, online | 0.989/0.992/0.993/0.994 | 0.983/0.991/0.996/1.000 | 1.001/1.001/1.001/1.001 | 0.998/0.998/0.998/0.998 |
| cjk-65536 + fill ranked by held-out halves, B=56K, online | 0.992/0.995/0.996/0.997 | 0.986/0.994/0.999/1.003 | 0.940/0.944/0.946/0.950 | 0.941/0.947/0.951/0.955 |
| kohangul-65536 + 2048 spare rows, online (pick + fill) | 1.002/1.003/1.004/1.005 | 1.005/1.009/1.010/1.013 | 0.514/0.542/0.579/0.665 | 0.989/0.993/0.996/0.989 |

`cblind` = the arm described in §3.5; `chyb` = the same with the writable rows chosen by the held-out halves' counts (it trains
on every set's other half, against "do not train on what you test on", so it is a comparison, not the claim); `lowid`,
`ec`, `both` = the neutral bases of §2. The neutral arms are the proposal as the user stated it.

### 3.3 Ids seen to parity (every set of the class at >= 0.99 of its best static list, online)

| arm | en | code | zh/ja | ko |
|---|---|---|---|---|
| pick+upgrade | 0 | 0 | 64 | 0 |
| lowid R=76000 B=48K | 64 | 0 | not by 4096 (min at 4096 0.659; long-run 16384 0.818) | not by 4096 (min at 4096 0.749; long-run 16384 0.851) |
| ec R=65536 B=32K | 0 | 0 | not by 4096 (min at 4096 0.660; long-run 16384 0.820) | not by 4096 (min at 4096 0.749; long-run 16384 0.852) |
| both R=76000 B=48K | 0 | 0 | not by 4096 (min at 4096 0.767; long-run 16384 0.873) | 0 |
| cjk-76000 + 2048 spare rows | 64 | 256 | 0 | 0 |
| kohangul-65536 + 2048 spare rows | 0 | 0 | not by 4096 (min at 4096 0.665; long-run 16384 0.707) | 64 |

### 3.4 Per set: the neutral `lowid` base (R = 76,000, B = 48K), then the `both` base

`lowid`:

| set | t=0 | t=64 | t=256 | t=1024 | t=4096 | long 8192 | long 16384 |
|---|---|---|---|---|---|---|---|
| zh-wiki | 0.534 | 0.570 | 0.615 | 0.676 | 0.756 | 0.766 | 0.917 |
| zh-chat | 0.509 | 0.521 | 0.526 | 0.569 | 0.659 | 0.653 | 0.818 |
| ja-wiki | 0.559 | 0.587 | 0.630 | 0.699 | 0.756 | 0.866 | 0.851 |
| ja-chat | 0.529 | 0.545 | 0.563 | 0.601 | 0.697 | 0.756 | 0.853 |
| ko-wiki | 0.592 | 0.624 | 0.670 | 0.708 | 0.769 | 0.884 | 0.891 |
| ko-chat | 0.519 | 0.542 | 0.567 | 0.630 | 0.749 | 0.830 | 0.851 |
| en-prose | 0.990 | 0.992 | 0.994 | 0.997 | 0.999 | 1.002 | 1.002 |
| en-math | 0.997 | 0.999 | 1.000 | 1.003 | 1.004 | 1.000 | 1.007 |
| code-rust | 1.001 | 1.003 | 1.004 | 1.008 | 1.010 | 1.012 | 1.010 |
| code-python | 1.008 | 1.009 | 1.011 | 1.010 | 1.012 | 1.009 | 1.015 |
| code-c | 1.015 | 1.015 | 1.016 | 1.015 | 1.017 | 1.020 | 1.013 |

`both`:

| set | t=0 | t=64 | t=256 | t=1024 | t=4096 | long 8192 | long 16384 |
|---|---|---|---|---|---|---|---|
| zh-wiki | 0.891 | 0.898 | 0.909 | 0.921 | 0.933 | 0.934 | 0.992 |
| zh-chat | 0.839 | 0.841 | 0.838 | 0.842 | 0.844 | 0.812 | 0.881 |
| ja-wiki | 0.710 | 0.730 | 0.758 | 0.798 | 0.835 | 0.924 | 0.873 |
| ja-chat | 0.625 | 0.639 | 0.656 | 0.687 | 0.767 | 0.813 | 0.886 |
| ko-wiki | 1.002 | 1.002 | 1.003 | 1.004 | 1.002 | 1.004 | 1.001 |
| ko-chat | 0.999 | 0.999 | 0.999 | 0.999 | 1.000 | 1.000 | 0.997 |
| en-prose | 0.999 | 1.000 | 1.000 | 1.001 | 1.001 | 1.004 | 1.004 |
| en-math | 1.000 | 1.000 | 1.000 | 1.001 | 1.002 | 0.998 | 1.007 |
| code-rust | 1.004 | 1.006 | 1.007 | 1.009 | 1.010 | 1.011 | 1.010 |
| code-python | 1.002 | 1.003 | 1.005 | 1.007 | 1.009 | 1.006 | 1.015 |
| code-c | 1.007 | 1.009 | 1.011 | 1.012 | 1.014 | 1.015 | 1.009 |

(`long` = the same arm walked from id 0 through a whole half, two paths a set, a 512-id window at 8,192 and 16,384 ids:
two paths, so +-0.05.) The neutral arm does not reach parity on zh/ja/ko inside the 16K ids of the long run.

### 3.5 Per set: the shipped list plus spare rows (`cblind`)


`cblind` is `C` with F = 2,048 writable rows: the highest ids of `C` below the tokenizer's control block. At load the map is
today's ascending map; an accepted id that is not a row replaces the highest writable row not yet touched (then the least
recently used), so the writable rows are one contiguous span of the ascending map. **The control block is never writable**:
`CONTROL_FROM` = 248,044 in the script, the nine ids that both lists hold above 248,000 (the end and pad ids the builder
always keeps, `build-sources.txt` keeps 248045 by hand, and the chat-template and think-tag ids). A real chat would otherwise
evict `<|im_end|>` before its first reply ends; the held-out sets are raw text with no template, so the simulation never
touches a control id and **cannot price special-token traffic**. The protection is by definition, and a gate clause
carries it (§7). In the design the set is "the F highest ids of `C` that are not the tokenizer's control or user-defined
tokens (`token_type`) and not a keep id", read from the GGUF at load. The arm uses no data beyond the list itself, which is
why it, and not `chyb`, is the claimed arm: its writable ids run from ~237,000 to 248,042, and the control matches the
data-ranked arm `chyb` within 0.001 at every t up to 4,096 and 0.002 in the long run.

Online:

| set | t=0 | t=64 | t=256 | t=1024 | t=4096 | long 8192 | long 16384 |
|---|---|---|---|---|---|---|---|
| zh-wiki | 1.013 | 1.016 | 1.018 | 1.019 | 1.015 | 1.038 | 1.041 |
| zh-chat | 1.004 | 1.004 | 1.004 | 1.004 | 1.004 | 1.004 | 1.020 |
| ja-wiki | 1.002 | 1.002 | 1.003 | 1.002 | 1.003 | 0.988 | 0.999 |
| ja-chat | 1.001 | 1.001 | 1.001 | 1.001 | 1.000 | 1.000 | 1.002 |
| ko-wiki | 1.002 | 1.003 | 1.003 | 1.004 | 1.002 | 1.004 | 1.002 |
| ko-chat | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 1.000 | 0.997 |
| en-prose | 0.989 | 0.990 | 0.992 | 0.993 | 0.994 | 0.997 | 0.997 |
| en-math | 0.991 | 0.992 | 0.993 | 0.995 | 0.997 | 0.996 | 1.002 |
| code-rust | 0.992 | 0.996 | 0.999 | 1.003 | 1.005 | 1.007 | 1.007 |
| code-python | 0.983 | 0.986 | 0.991 | 0.996 | 1.000 | 0.993 | 1.009 |
| code-c | 0.992 | 0.996 | 1.001 | 1.004 | 1.009 | 1.015 | 0.991 |

Prompt-only (the set frozen at t):

| set | t=0 | t=64 | t=256 | t=1024 | t=4096 |
|---|---|---|---|---|---|
| zh-wiki | 1.000 | 1.007 | 1.012 | 1.013 | 1.012 |
| zh-chat | 1.000 | 1.001 | 1.001 | 1.001 | 1.001 |
| ja-wiki | 1.000 | 1.001 | 1.001 | 1.002 | 1.002 |
| ja-chat | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ko-wiki | 1.000 | 1.001 | 1.002 | 1.002 | 1.002 |
| ko-chat | 0.997 | 0.997 | 0.997 | 0.997 | 0.997 |
| en-prose | 0.985 | 0.987 | 0.989 | 0.991 | 0.992 |
| en-math | 0.980 | 0.983 | 0.985 | 0.988 | 0.992 |
| code-rust | 0.978 | 0.987 | 0.993 | 0.999 | 1.000 |
| code-python | 0.972 | 0.978 | 0.985 | 0.991 | 0.996 |
| code-c | 0.972 | 0.981 | 0.989 | 0.996 | 1.003 |

Same with `D` as the base (the pick's other list, R = 65,536, 2,048 spare rows; the ratio is against the best static list, so
zh/ja are not its sets):

| set | t=0 | t=64 | t=256 | t=1024 | t=4096 | long 8192 | long 16384 |
|---|---|---|---|---|---|---|---|
| zh-wiki | 0.549 | 0.583 | 0.624 | 0.684 | 0.763 | 0.767 | 0.885 |
| zh-chat | 0.530 | 0.539 | 0.542 | 0.579 | 0.665 | 0.604 | 0.707 |
| ja-wiki | 0.554 | 0.582 | 0.626 | 0.697 | 0.757 | 0.865 | 0.806 |
| ja-chat | 0.514 | 0.532 | 0.554 | 0.597 | 0.700 | 0.740 | 0.752 |
| ko-wiki | 0.989 | 0.991 | 0.993 | 0.996 | 0.989 | 0.999 | 1.000 |
| ko-chat | 1.001 | 1.001 | 1.001 | 1.001 | 1.001 | 0.995 | 0.994 |
| en-prose | 1.002 | 1.003 | 1.003 | 1.004 | 1.005 | 1.007 | 1.007 |
| en-math | 1.003 | 1.003 | 1.004 | 1.005 | 1.006 | 1.002 | 1.010 |
| code-rust | 1.007 | 1.010 | 1.010 | 1.012 | 1.013 | 1.014 | 1.013 |
| code-python | 1.005 | 1.007 | 1.009 | 1.010 | 1.013 | 1.010 | 1.018 |
| code-c | 1.010 | 1.012 | 1.014 | 1.015 | 1.018 | 1.019 | 1.012 |

### 3.6 Fill policy and writable-region size

The spec grid (F = 10,464 to 43,232) never overflowed within 4,096 ids nor within 20,000 on a half: B = 32K, 48K and 64K give
the same numbers to three digits, and so do LRU and LFU, for `lowid`, `both` and `chyb`. `cblind` at F = 2,048 does not
overflow either: it writes .002-.033 ids a token, at most ~660 ids in 20,000 tokens, fewer than its rows. So "LRU = LFU" is a
property of the data and the sizes, not a finding about the policies. The neutral arm at F = 2,048 and 1,024 (R = 65,536) does
overflow:

| F | policy | zh/ja min at t=4096 | ko min at t=4096 | zh/ja min long 16384 | ko min long 16384 |
|---|---|---|---|---|---|
| 2048 | lru | 0.660 | 0.749 | 0.703 | 0.806 |
| 2048 | lfu | 0.660 | 0.749 | 0.727 | 0.811 |
| 1024 | lru | 0.613 | 0.717 | 0.625 | 0.700 |
| 1024 | lfu | 0.620 | 0.720 | 0.552 | 0.656 |

LFU is better on zh/ja at F = 2,048 and worse at F = 1,024 (long run, two paths: +-0.05); no winner. LRU is the simpler state to
keep and persist (an order, no counts).

### 3.7 The prediction against the simulation (the term that was wrong)

| set | predicted miss at t=0 | simulated, set frozen at t=0 | predicted miss after 4096 ids (Good-Turing, B=32K) | simulated, online, ids 4096..4352 | simulated, prompt-only (frozen at 4096) |
|---|---|---|---|---|---|
| zh-wiki | 0.702 | 0.714 | 0.180 | 0.226 | 0.267 |
| zh-chat | 0.644 | 0.622 | 0.300 | 0.326 | 0.373 |
| ja-wiki | 0.582 | 0.578 | 0.165 | 0.204 | 0.241 |
| ja-chat | 0.571 | 0.563 | 0.230 | 0.261 | 0.307 |
| ko-wiki | 0.508 | 0.518 | 0.150 | 0.191 | 0.231 |
| ko-chat | 0.609 | 0.611 | 0.180 | 0.205 | 0.256 |
| en-prose | 0.021 | 0.022 | 0.027 | 0.007 | 0.009 |
| en-math | 0.010 | 0.011 | 0.009 | 0.002 | 0.004 |
| code-rust | 0.016 | 0.013 | 0.011 | 0.004 | 0.006 |
| code-python | 0.009 | 0.009 | 0.017 | 0.003 | 0.004 |
| code-c | 0.005 | 0.004 | 0.006 | 0.002 | 0.003 |

The t = 0 column agrees (the coverage counts predicted it). On zh/ja/ko the Good-Turing estimate was low by .03-.05 in the
online column and by .07-.09 against the prompt-only column (exact first-seen miss, e.g. zh-chat .373 against GT .298): the
held-out streams are not exchangeable draws; documents bring their own words, so the unseen mass is higher than `N1/t`. The
verdict direction held (predicted 0.68-0.81 of the best list at 4,096, measured 0.66-0.75 for the worst set of each class on `lowid`).
The en/code prediction column used B = 32K as the threshold while the simulated rows reach 76,000; with the threshold at
76,000 the same estimator gives .005 / .002 / .002 / .003 / .001, next to the simulated .007 / .002 / .004 / .003 / .002.

## 4. The saved profile

A profile is the writable region after a half has run through the arm (20,000 ids, F rows in recency order), started on the
other half; and the same in the other direction (each row below averages both directions, 50 trials each). `n` is the ids
the profile was built from.

**Neutral base (`lowid`, R = 76,000, B = 48K):**

| set | q cold | q n=1K | q n=5K | q n=20K | ratio cold | ratio n=20K | ratio n=20K at t=4096 | exponent | ids to ratio .99 (extrapolated) |
|---|---|---|---|---|---|---|---|---|---|
| zh-wiki | 0.4600 | 0.3870 | 0.3179 | 0.2044 | 0.550 | 0.789 | 0.863 | 0.32 | 2367K |
| zh-chat | 0.4979 | 0.4189 | 0.2959 | 0.1771 | 0.520 | 0.818 | 0.836 | 0.37 | 924K |
| ja-wiki | 0.4241 | 0.3587 | 0.2611 | 0.1414 | 0.553 | 0.826 | 0.880 | 0.44 | 4018K |
| ja-chat | 0.4571 | 0.3798 | 0.2484 | 0.1054 | 0.528 | 0.869 | 0.888 | 0.62 | 583K |
| ko-wiki | 0.3778 | 0.3219 | 0.2109 | 0.0942 | 0.589 | 0.882 | 0.910 | 0.58 | 621K |
| ko-chat | 0.4626 | 0.3566 | 0.1955 | 0.0793 | 0.520 | 0.895 | 0.917 | 0.65 | 677K |
| en-prose | 0.0157 | 0.0148 | 0.0134 | 0.0100 | 0.988 | 0.996 | 1.003 | 0.21 | reached |
| en-math | 0.0068 | 0.0037 | 0.0037 | 0.0029 | 0.997 | 1.002 | 1.005 | 0.17 | reached |
| code-rust | 0.0091 | 0.0083 | 0.0071 | 0.0044 | 1.001 | 1.007 | 1.013 | 0.35 | reached |
| code-python | 0.0063 | 0.0059 | 0.0057 | 0.0052 | 1.008 | 1.010 | 1.012 | 0.08 | reached |
| code-c | 0.0027 | 0.0027 | 0.0027 | 0.0023 | 1.016 | 1.016 | 1.018 | 0.12 | reached |

**Shipped list plus spare rows (`cblind`):**

| set | q cold | q n=1K | q n=5K | q n=20K | ratio cold | ratio n=20K | ratio n=20K at t=4096 | exponent | ids to ratio .99 (extrapolated) |
|---|---|---|---|---|---|---|---|---|---|
| zh-wiki | 0.0263 | 0.0263 | 0.0263 | 0.0261 | 1.016 | 1.016 | 1.018 | 0.00 | reached |
| zh-chat | 0.0327 | 0.0324 | 0.0321 | 0.0313 | 1.004 | 1.006 | 1.008 | 0.02 | reached |
| ja-wiki | 0.0045 | 0.0045 | 0.0048 | 0.0055 | 1.003 | 1.001 | 1.002 | -0.11 | reached |
| ja-chat | 0.0056 | 0.0057 | 0.0062 | 0.0074 | 1.000 | 0.998 | 1.000 | -0.13 | reached |
| ko-wiki | 0.0041 | 0.0041 | 0.0041 | 0.0037 | 1.002 | 1.003 | 1.003 | 0.08 | reached |
| ko-chat | 0.0025 | 0.0025 | 0.0025 | 0.0023 | 0.998 | 0.998 | 0.999 | 0.04 | reached |
| en-prose | 0.0156 | 0.0155 | 0.0150 | 0.0138 | 0.989 | 0.991 | 0.995 | 0.06 | reached |
| en-math | 0.0119 | 0.0113 | 0.0109 | 0.0082 | 0.990 | 0.995 | 1.000 | 0.21 | reached |
| code-rust | 0.0154 | 0.0147 | 0.0140 | 0.0096 | 0.992 | 1.000 | 1.010 | 0.27 | reached |
| code-python | 0.0240 | 0.0230 | 0.0220 | 0.0181 | 0.984 | 0.992 | 1.003 | 0.14 | reached |
| code-c | 0.0190 | 0.0188 | 0.0180 | 0.0138 | 0.993 | 1.000 | 1.011 | 0.19 | reached |

Per direction (n = 20,000; the table rows above average these two; 50 trials a cell; the ratio is against the both-halves best static list):

`lowid`:

| set | built on -> tested on | q cold | ratio cold | ratio n=20K at t=0 | t=64 | t=256 | t=1024 | t=4096 |
|---|---|---|---|---|---|---|---|---|
| zh-wiki | half 0 -> half 1 | 0.4474 | 0.559 | 0.777 | 0.789 | 0.807 | 0.778 | 0.819 |
| zh-wiki | half 1 -> half 0 | 0.4727 | 0.540 | 0.801 | 0.816 | 0.835 | 0.873 | 0.910 |
| zh-chat | half 0 -> half 1 | 0.4909 | 0.526 | 0.821 | 0.823 | 0.815 | 0.827 | 0.838 |
| zh-chat | half 1 -> half 0 | 0.5049 | 0.515 | 0.815 | 0.820 | 0.816 | 0.821 | 0.833 |
| ja-wiki | half 0 -> half 1 | 0.4188 | 0.557 | 0.829 | 0.842 | 0.858 | 0.879 | 0.888 |
| ja-wiki | half 1 -> half 0 | 0.4294 | 0.549 | 0.823 | 0.833 | 0.847 | 0.865 | 0.871 |
| ja-chat | half 0 -> half 1 | 0.4629 | 0.524 | 0.874 | 0.882 | 0.883 | 0.885 | 0.894 |
| ja-chat | half 1 -> half 0 | 0.4514 | 0.532 | 0.863 | 0.867 | 0.869 | 0.875 | 0.883 |
| ko-wiki | half 0 -> half 1 | 0.3788 | 0.589 | 0.869 | 0.882 | 0.888 | 0.882 | 0.890 |
| ko-wiki | half 1 -> half 0 | 0.3768 | 0.590 | 0.896 | 0.907 | 0.919 | 0.926 | 0.931 |
| ko-chat | half 0 -> half 1 | 0.4514 | 0.528 | 0.901 | 0.905 | 0.909 | 0.917 | 0.924 |
| ko-chat | half 1 -> half 0 | 0.4738 | 0.512 | 0.889 | 0.897 | 0.901 | 0.908 | 0.910 |
| en-prose | half 0 -> half 1 | 0.0165 | 0.987 | 0.995 | 0.996 | 0.999 | 1.001 | 1.001 |
| en-prose | half 1 -> half 0 | 0.0149 | 0.989 | 0.997 | 0.999 | 1.000 | 1.002 | 1.004 |
| en-math | half 0 -> half 1 | 0.0065 | 0.997 | 1.003 | 1.004 | 1.003 | 1.003 | 1.005 |
| en-math | half 1 -> half 0 | 0.0071 | 0.997 | 1.002 | 1.002 | 1.002 | 1.004 | 1.006 |
| code-rust | half 0 -> half 1 | 0.0085 | 1.002 | 1.008 | 1.009 | 1.010 | 1.012 | 1.012 |
| code-rust | half 1 -> half 0 | 0.0097 | 1.000 | 1.007 | 1.008 | 1.010 | 1.013 | 1.013 |
| code-python | half 0 -> half 1 | 0.0064 | 1.008 | 1.010 | 1.011 | 1.011 | 1.011 | 1.012 |
| code-python | half 1 -> half 0 | 0.0063 | 1.009 | 1.010 | 1.011 | 1.013 | 1.012 | 1.013 |
| code-c | half 0 -> half 1 | 0.0031 | 1.015 | 1.016 | 1.015 | 1.016 | 1.015 | 1.017 |
| code-c | half 1 -> half 0 | 0.0023 | 1.016 | 1.017 | 1.017 | 1.017 | 1.017 | 1.019 |

`cblind`:

| set | built on -> tested on | q cold | ratio cold | ratio n=20K at t=0 | t=64 | t=256 | t=1024 | t=4096 |
|---|---|---|---|---|---|---|---|---|
| zh-wiki | half 0 -> half 1 | 0.0305 | 1.010 | 1.010 | 1.012 | 1.009 | 1.009 | 1.006 |
| zh-wiki | half 1 -> half 0 | 0.0220 | 1.022 | 1.022 | 1.025 | 1.028 | 1.035 | 1.029 |
| zh-chat | half 0 -> half 1 | 0.0336 | 1.003 | 1.006 | 1.008 | 1.005 | 1.009 | 1.013 |
| zh-chat | half 1 -> half 0 | 0.0318 | 1.006 | 1.007 | 1.012 | 1.008 | 1.007 | 1.002 |
| ja-wiki | half 0 -> half 1 | 0.0063 | 1.000 | 0.999 | 0.999 | 1.001 | 0.996 | 1.001 |
| ja-wiki | half 1 -> half 0 | 0.0027 | 1.005 | 1.003 | 1.004 | 1.003 | 1.003 | 1.004 |
| ja-chat | half 0 -> half 1 | 0.0061 | 1.000 | 0.998 | 0.998 | 1.001 | 0.998 | 1.002 |
| ja-chat | half 1 -> half 0 | 0.0052 | 1.001 | 0.998 | 0.998 | 1.000 | 0.998 | 0.998 |
| ko-wiki | half 0 -> half 1 | 0.0032 | 1.003 | 1.005 | 1.005 | 1.004 | 1.004 | 1.002 |
| ko-wiki | half 1 -> half 0 | 0.0050 | 1.001 | 1.001 | 1.002 | 1.006 | 1.005 | 1.003 |
| ko-chat | half 0 -> half 1 | 0.0018 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |
| ko-chat | half 1 -> half 0 | 0.0031 | 0.997 | 0.997 | 0.997 | 0.997 | 0.998 | 0.999 |
| en-prose | half 0 -> half 1 | 0.0147 | 0.990 | 0.993 | 0.994 | 0.997 | 0.993 | 0.995 |
| en-prose | half 1 -> half 0 | 0.0165 | 0.987 | 0.989 | 0.990 | 0.993 | 0.994 | 0.995 |
| en-math | half 0 -> half 1 | 0.0111 | 0.991 | 0.996 | 0.997 | 0.997 | 0.998 | 1.000 |
| en-math | half 1 -> half 0 | 0.0127 | 0.989 | 0.994 | 0.994 | 0.996 | 0.998 | 1.000 |
| code-rust | half 0 -> half 1 | 0.0150 | 0.993 | 1.001 | 1.004 | 1.008 | 1.007 | 1.007 |
| code-rust | half 1 -> half 0 | 0.0158 | 0.991 | 0.999 | 1.002 | 1.006 | 1.010 | 1.013 |
| code-python | half 0 -> half 1 | 0.0233 | 0.985 | 0.995 | 0.997 | 0.997 | 1.000 | 1.005 |
| code-python | half 1 -> half 0 | 0.0247 | 0.983 | 0.989 | 0.994 | 0.997 | 1.000 | 1.001 |
| code-c | half 0 -> half 1 | 0.0231 | 0.987 | 0.998 | 1.000 | 1.004 | 1.008 | 1.007 |
| code-c | half 1 -> half 0 | 0.0148 | 0.999 | 1.002 | 1.006 | 1.007 | 1.012 | 1.014 |

- On the neutral base the profile is the only thing that moves zh/ja/ko, and it moves them slowly: q falls as n^-0.3 to
  n^-0.65 (the exponent column, a 3-point fit over 5K, 10K and 20K ids). Reaching 0.99 of the best list is 0.6-4 million
  same-language ids `[extrapolated]`; the headline stays the measured 0.79-0.90 at n = 20,000, t = 0.
- On `cblind` it adds +0.3 to +0.8 points on en/code at t = 0, 0 on the rest: at or just above the 0.6-point seed spread, and
  only in the first 256 ids (the arm is at >= .991 there anyway), so no case for a profile there.
- **Stale and cross-language profiles.** A profile of one set tested on every other set at 256 ids (25 trials per half):
  `cblind`: worst loss -0.013 (profile of zh-chat on ja-chat); best cross-gain 0.003 (profile of code-rust on en-prose). `lowid`: worst loss -0.018 (profile of zh-chat on ko-wiki); best cross-gain 0.249 (profile of ko-wiki on ko-chat). A wrong-language profile costs at most 1.3 points (zh-chat
  profile on ja-chat) on `cblind` and 1.8 on the neutral base. A cell here is 50 trials against the cold row's 400, so its noise
  is about twice the 0.6-point seed spread: a stale profile is not distinguishable from no loss on the arm that works. The best
  gain is a same-language profile (ko-wiki on ko-chat, +25 points on the neutral base).

### 4.1 The KV-prefix restore: turn 2 starts from the row set turn 1 left

Added at the lead's request (the user's rule: the warm row state is keyed by the KV cache's prefix, per session; a request
that hits a cached prefix restores that session's row set; a miss starts from the global profile; per-slot rows never blend
across streams, only card-wide state does).

**Paper first.** The row set is a fold of the id sequence in order: every touch reads the set and one id, and the BLOCK window
moves *when* a write lands, not the order. So `state = fold(start, ids)`, and a restore after T1 ids is the stream that never
stopped: `miss_restore(T1, t2) = miss_continuous(T1 + t2)`. The grid of §3 already holds that at T1 = 256, 1,024 and 4,096
(t2 = 0), so the restore's numbers were predicted from the existing tables before anything ran (Appendix C.3). The run is for
the three things the grid does not hold: (a) the restore against a *miss* (a start from the base, from a same-language profile,
or from a card-wide profile) on identical turn-2 windows; (b) whether the fold is pure, which decides whether any copy needs to
be stored; (c) the regime of a card-wide (blended) head profile, which the user's rule excludes and the numbers can test: the
11 training halves hold 16,762-17,241 distinct ids outside the neutral base against 25,830-25,878 free filler rows, and
1,883-2,163 ids outside `C` against 1,184-1,230 free spare rows (`kvcount.py`, Appendix C.2).

**Method.** A turn is a run of consecutive ids of one held-out set: turn 1 is T1 ids, turn 2 the next 1,280. 100 trials a
cell (50 a half, seed 7), cross-fitted (a profile or blend is built on the half the trial does not use), and **every arm is
paired on the same offsets and the same turn-2 windows**, which also carry the two static lists. Arms: *cold start* (turn 2
from the base: a new conversation), *profile start* (from the same-set profile of §4, 20,000 ids: **an upper bound, not a
realistic start, because a miss does not know the request's language**), *blend start* (from a card-wide profile: all 11 sets'
other halves folded round-robin in chunks of 256 ids, which is what "the global profile" is when traffic mixes languages), and
the three *restore* arms (the session began from each of them, turn 1 ran, turn 2 restores). `t2` is turn-2 ids seen when the
miss is read; a new user message at the head of turn 2 moves the first draft from t2 = 0 to t2 = its length (columns 64 and
256). The turns are consecutive chunks of concatenated documents, so a real two-turn chat shares more topic than these do:
the restore gains below are a floor for it.

**The fold is pure.** The two identity lines `kvreport` prints: `kv_cont`: 79,200 of 79,200 (trial, T1, t2, start) windows have exactly the miss count of the same
window of the stream that never stopped. `kv_pure`: 19,800 of 19,800 (state, T1) pairs have the row set after turn 1 (walked
live in windows of 4 kept ids) equal, rows and LRU order, to the same ids folded in order in one go, which is what an
admission that folds its prompt builds. The cell counts are for the two arms and 11 sets; both are equalities, not tolerances.

**Ratio to the best list at turn-2 t2 = 0, T1 = 256 / 1,024 / 4,096, minimum over the sets of a class (`[sim]`):**

Neutral base (`lowid`, R = 76,000, B = 48K):

| arm | en | code | zh/ja | ko |
|---|---|---|---|---|
| cold start: a new conversation, from the base | 0.986/0.991/0.993 | 1.003/1.004/1.000 | 0.499/0.500/0.497 | 0.522/0.524/0.514 |
| same-set profile start (upper bound: an oracle on the language) | 0.995/0.999/1.000 | 1.010/1.009/1.009 | 0.787/0.759/0.761 | 0.887/0.879/0.881 |
| blended card-wide profile start (11 sets) | 0.985/0.988/0.989 | 1.002/1.000/1.000 | 0.841/0.849/0.846 | 0.937/0.944/0.937 |
| restore, session began cold | 0.992/0.999/1.001 | 1.007/1.009/1.010 | 0.528/0.568/0.657 | 0.569/0.632/0.747 |
| restore, session began from the same-set profile (upper bound) | 0.997/1.003/1.003 | 1.012/1.012/1.012 | 0.814/0.819/0.835 | 0.901/0.904/0.910 |
| restore, session began from the blended profile | 0.989/0.993/0.995 | 1.004/1.005/1.006 | 0.847/0.856/0.863 | 0.942/0.951/0.950 |

Shipped list plus spare rows (`cblind`, R = 76,000, F = 2,048):

| arm | en | code | zh/ja | ko |
|---|---|---|---|---|
| cold start: a new conversation, from the base | 0.989/0.990/0.989 | 0.983/0.984/0.983 | 1.000/1.001/1.001 | 0.997/0.998/0.999 |
| same-set profile start (upper bound: an oracle on the language) | 0.991/0.993/0.992 | 0.993/0.992/0.992 | 0.999/0.999/0.998 | 0.997/0.998/0.999 |
| blended card-wide profile start (11 sets) | 0.992/0.994/0.992 | 0.993/0.992/0.992 | 0.982/0.981/0.979 | 0.991/0.991/0.993 |
| restore, session began cold | 0.991/0.994/0.994 | 0.991/0.996/0.999 | 1.001/1.001/1.000 | 0.998/0.998/0.999 |
| restore, session began from the same-set profile (upper bound) | 0.993/0.996/0.996 | 0.998/1.000/1.003 | 0.999/0.999/0.999 | 0.998/0.998/1.000 |
| restore, session began from the blended profile | 0.993/0.997/0.996 | 0.998/1.000/1.003 | 0.983/0.983/0.982 | 0.992/0.992/0.994 |

How fast the restore's lead goes (`cblind`, T1 = 1,024, turn-2 t2 = 0 / 64 / 256 / 1,024):

| arm | en | code | zh/ja | ko |
|---|---|---|---|---|
| cold start: a new conversation, from the base | 0.990/0.992/0.991/0.994 | 0.984/0.987/0.991/0.994 | 1.001/1.001/1.001/1.000 | 0.998/0.999/0.999/0.999 |
| same-set profile start (upper bound: an oracle on the language) | 0.993/0.995/0.994/0.995 | 0.992/0.995/0.998/0.999 | 0.999/0.999/0.999/0.998 | 0.998/0.999/1.000/0.999 |
| blended card-wide profile start (11 sets) | 0.994/0.995/0.994/0.996 | 0.992/0.995/0.998/0.999 | 0.981/0.982/0.983/0.984 | 0.991/0.993/0.992/0.993 |
| restore, session began cold | 0.994/0.994/0.992/0.996 | 0.996/0.996/0.995/0.997 | 1.001/1.001/1.001/1.000 | 0.998/0.999/0.999/0.999 |
| restore, session began from the same-set profile (upper bound) | 0.996/0.997/0.995/0.998 | 1.000/1.001/1.001/1.001 | 0.999/0.999/0.999/0.998 | 0.998/1.000/1.000/0.999 |
| restore, session began from the blended profile | 0.997/0.997/0.995/0.997 | 1.000/1.001/1.000/1.001 | 0.983/0.983/0.984/0.984 | 0.992/0.993/0.993/0.993 |

The next-256-id miss rate q at turn-2 t = 0 (T1 = 1,024; the lead's quantity; q C and q D are the two static lists on the same windows):

`cblind`:

| set | q C | q D | cold start | profile start | blend start | restore (cold) | restore (profile) | restore (blend) |
|---|---|---|---|---|---|---|---|---|
| zh-wiki | 0.0356 | 0.7009 | 0.0266 | 0.0262 | 0.0271 | 0.0220 | 0.0217 | 0.0222 |
| zh-chat | 0.0326 | 0.5944 | 0.0298 | 0.0288 | 0.0291 | 0.0292 | 0.0282 | 0.0285 |
| ja-wiki | 0.0086 | 0.5795 | 0.0067 | 0.0077 | 0.0195 | 0.0059 | 0.0067 | 0.0150 |
| ja-chat | 0.0072 | 0.5974 | 0.0065 | 0.0082 | 0.0205 | 0.0065 | 0.0079 | 0.0196 |
| ko-wiki | 0.0057 | 0.0205 | 0.0045 | 0.0041 | 0.0078 | 0.0038 | 0.0034 | 0.0066 |
| ko-chat | 0.0030 | 0.0030 | 0.0021 | 0.0019 | 0.0066 | 0.0019 | 0.0017 | 0.0061 |
| en-prose | 0.0203 | 0.0112 | 0.0157 | 0.0137 | 0.0132 | 0.0131 | 0.0113 | 0.0111 |
| en-math | 0.0202 | 0.0077 | 0.0116 | 0.0081 | 0.0080 | 0.0093 | 0.0067 | 0.0068 |
| code-rust | 0.0245 | 0.0135 | 0.0143 | 0.0079 | 0.0080 | 0.0075 | 0.0040 | 0.0039 |
| code-python | 0.0305 | 0.0148 | 0.0239 | 0.0178 | 0.0180 | 0.0154 | 0.0121 | 0.0124 |
| code-c | 0.0326 | 0.0140 | 0.0182 | 0.0130 | 0.0137 | 0.0086 | 0.0057 | 0.0059 |

`lowid`:

| set | q C | q D | cold start | profile start | blend start | restore (cold) | restore (profile) | restore (blend) |
|---|---|---|---|---|---|---|---|---|
| zh-wiki | 0.0356 | 0.7009 | 0.4992 | 0.2291 | 0.1497 | 0.3205 | 0.1758 | 0.1185 |
| zh-chat | 0.0326 | 0.5944 | 0.5235 | 0.1761 | 0.1475 | 0.4308 | 0.1691 | 0.1421 |
| ja-wiki | 0.0086 | 0.5795 | 0.4257 | 0.1396 | 0.0586 | 0.2652 | 0.1026 | 0.0454 |
| ja-chat | 0.0072 | 0.5974 | 0.4580 | 0.1047 | 0.0634 | 0.3650 | 0.0968 | 0.0588 |
| ko-wiki | 0.0057 | 0.0205 | 0.3670 | 0.0965 | 0.0462 | 0.2517 | 0.0772 | 0.0365 |
| ko-chat | 0.0030 | 0.0030 | 0.4563 | 0.0745 | 0.0414 | 0.3235 | 0.0657 | 0.0362 |
| en-prose | 0.0203 | 0.0112 | 0.0149 | 0.0096 | 0.0173 | 0.0095 | 0.0068 | 0.0134 |
| en-math | 0.0202 | 0.0077 | 0.0068 | 0.0029 | 0.0057 | 0.0034 | 0.0026 | 0.0048 |
| code-rust | 0.0245 | 0.0135 | 0.0083 | 0.0030 | 0.0062 | 0.0044 | 0.0016 | 0.0039 |
| code-python | 0.0305 | 0.0148 | 0.0072 | 0.0059 | 0.0124 | 0.0045 | 0.0037 | 0.0090 |
| code-c | 0.0326 | 0.0140 | 0.0031 | 0.0023 | 0.0063 | 0.0017 | 0.0011 | 0.0030 |

Restore minus its own cold start, points of tok/s at t2 = 0, T1 = 256 / 1,024 / 4,096 (`cblind`; the three code sets agree in sign and size):

| set | restore(cold) - cold start, T1 = 256 / 1024 / 4096 | restore(profile) - profile start | restore(blend) - blend start |
|---|---|---|---|
| zh-wiki | +0.4/+0.7/+0.7 | +0.4/+0.7/+0.7 | +0.4/+0.7/+0.8 |
| zh-chat | +0.1/+0.1/+0.1 | +0.1/+0.1/+0.1 | +0.1/+0.1/+0.1 |
| ja-wiki | +0.0/+0.1/+0.1 | +0.1/+0.1/+0.1 | +0.4/+0.6/+0.7 |
| ja-chat | +0.0/+0.0/-0.1 | +0.0/+0.1/+0.1 | +0.1/+0.1/+0.3 |
| ko-wiki | +0.1/+0.1/+0.1 | +0.1/+0.1/+0.1 | +0.1/+0.2/+0.2 |
| ko-chat | +0.0/+0.0/+0.0 | +0.0/+0.0/+0.0 | +0.1/+0.1/+0.1 |
| en-prose | +0.2/+0.4/+0.5 | +0.2/+0.3/+0.4 | +0.2/+0.3/+0.4 |
| en-math | +0.1/+0.3/+0.7 | +0.0/+0.2/+0.3 | +0.0/+0.2/+0.3 |
| code-rust | +0.6/+0.9/+1.1 | +0.4/+0.6/+0.5 | +0.4/+0.6/+0.5 |
| code-python | +0.7/+1.2/+1.7 | +0.5/+0.8/+1.1 | +0.5/+0.8/+1.1 |
| code-c | +0.8/+1.4/+1.7 | +0.5/+1.0/+1.3 | +0.5/+1.1/+1.3 |

**Prediction against the run** (`kvpredict.txt`, Appendix C.3; the restore columns were read off the existing tables):

| quantity | predicted | measured | |
|---|---|---|---|
| `lowid` restore, zh/ja, T1 = 256 / 1,024 / 4,096 | .526 / .569 / .659 | .528 / .568 / .657 | ok |
| `lowid` restore, ko | .567 / .630 / .749 | .569 / .632 / .747 | ok |
| `lowid` restore, en / code | .994 / .997 / .999 and 1.004 / 1.008 / 1.010 | .992 / .999 / 1.001 and 1.007 / 1.009 / 1.010 | ok |
| `cblind` restore, en / code | .992 / .993 / .994 and .991 / .996 / 1.000 | .991 / .994 / .994 and .991 / .996 / .999 | ok |
| `cblind` restore, zh/ja / ko | 1.001 and .998 | 1.001 / 1.001 / 1.000 and .998 / .998 / .999 | ok |
| same-set profile start, `lowid`, t2 = 0, zh/ja / ko | .777 / .869 | .787 / .759 / .761 and .887 / .879 / .881 | ok (cell noise ~.03) |
| restore from the profile, `lowid`, zh/ja | .807 / .778 / .819 | .814 / .819 / .835 | ok; the predicted .778 was a non-monotone 50-trial cell |
| **blend start, `lowid`, zh/ja / ko** | **= the same-set profile: .777 / .869** | **.841 / .849 / .846 and .937 / .944 / .937** | **missed by +6 to +9 points; en -0.1 to -0.4 vs cold** |
| **blend start, `cblind`** | **gain <= the same-set profile's; 0 on zh/ja/ko** | **zh/ja .982 / .981 / .979, ko .991 / .991 / .993, en +0.3, code +0.9 (cold start: zh/ja 1.000, ko .998, en .989, code .983)** | **missed: zh/ja -1.8 to -2.2, ko -0.6** |

The two misses name their wrong terms. **`lowid` blend:** the count showed the union fits the filler rows, so I took the blend to
equal each set's own profile. Two terms were not counted. The blend is a *superset* of the set's own ids and the CJK sets share
ids (Han between zh and ja, Han and Hangul with ko): zh-wiki's q falls from .229 with its own profile to .150 with the blend.
And "nothing is evicted" is not "nothing is displaced": every id the blend adds takes a filler row, top rank first, so the
17K ids the blend holds overwrote the filler rows from id 75,999 down, and the en and code ids in that range, which the
set's own small profile left standing, stopped being rows (en-prose q .0096 with its own profile, .0173 with the blend, .0149
cold). **`cblind` blend:** I took the overflow (0.7-0.9K ids) to be an LRU turnover of harmless spare rows. The 2,048 writable
rows are the highest non-control ids of `C` and they are not spare for CJK: they carry **4.3-4.9 % of the tokens of ja**
(536 and 632 distinct ids), 1.0-1.3 % of ko, 0-0.3 % of zh, and nothing of en and code (`kvtail.py`, Appendix C.2). A blend that
holds 1.9-2.2K ids missed by `C` (mostly en and code) overwrites all of them, and a ja session then has to win those rows
back. Alone, a session never pays this: its own seen ids are protected, and it overwrites only as many tail rows as it has
distinct misses (<= ~660 in 20,000 ids).

**What the numbers say.**

- *Restore vs a cold start.* On `cblind` the restore is worth +0.2 / +0.4 / +0.5 points on en and +0.8 / +1.2 / +1.6 on code at
  t2 = 0 for T1 = 256 / 1,024 / 4,096, and +0.0 to +0.1 on zh/ja/ko (they start at parity). The lead is gone within about 256
  turn-2 ids (code, T1 = 1,024: +1.2, +0.9, +0.4, +0.3 at t2 = 0 / 64 / 256 / 1,024), so over a 512-id reply it averages **+0.8
  on code and +0.3 on en** `[derived]` from the trajectory table: inside the 4-round ruler (+-1.0 %). On the neutral base it is
  +3 / +7 / +16 points on zh/ja and +5 / +11 / +23 on ko at t2 = 0, and still .66 and .75 at T1 = 4,096: a restore does not
  make the neutral base reach the lists (verdict (b) stands).
- *Restore vs the global profile.* The same-set profile start (20K ids of the same language, an oracle) is above a cold-begun
  restore on the neutral base at every T1 <= 4,096 (.76-.79 against .53-.66 on zh/ja), and a restore that continues from that
  profile adds +0.5 to +9.1 points on the CJK sets (zh-wiki +3.3 / +6.0 / +9.1, ja-wiki +2.8 / +4.5 / +5.8, the other four +0.5 to
  +2.9). On `cblind` the arms sit within 0.3 points of each other on zh/ja/ko, except the blend's loss, and within 2 points on
  en/code (code: .983 cold start to 1.003 restore from the profile).
- *A miss should start from the base.* On `cblind` the blended start is **below the cold start by 1.8 / 2.0 / 2.2 points on zh/ja
  and 0.6 / 0.7 / 0.6 on ko** and above it by +0.3 on en and +0.9 on code: it trades CJK for en/code, outside the ruler on ja.
  On the neutral base it is the best start on CJK (.84-.85 on zh/ja, .94 on ko) and still not a list; there it works as a
  17K-id CJK list built from usage, which the shipped list already is, built offline from far more data. So the data
  supports "per-session rows, restore by prefix" and "a miss starts from the base", and **does not support a card-wide head
  profile**. This contradicts the lead's "a miss starts from the global profile" if it means a head profile; if it means the
  card-wide file (residency, width), nothing here contradicts it.
- *A restore is not a separate mechanism.* The restored set is the fold of the conversation's ids, and the conversation's ids
  are the prompt the next request carries. A KV hit and a KV miss that re-prefills the history both hand the head the same
  ids, so both arrive at the same set. The restore differs from a miss only where the miss does not carry the history, which
  is a different conversation (the cold-start arm), and from a fold only in the host time it saves (§5.6).

## 5. Cost side

### 5.1 What a step writes, and how it reaches the card

Map writes per kept token (window 256 after t, online; a write = one id taking a row):

| set | cjk-76000 + 2048 spare rows, t=0 / 256 / 4096 | lowid R=76000 B=48K, t=0 / 256 / 4096 |
|---|---|---|
| zh-wiki | 0.028/0.023/0.019 | 0.480/0.378/0.226 |
| zh-chat | 0.033/0.033/0.030 | 0.514/0.490/0.325 |
| ja-wiki | 0.005/0.005/0.004 | 0.416/0.331/0.204 |
| ja-chat | 0.005/0.005/0.006 | 0.455/0.410/0.260 |
| ko-wiki | 0.004/0.004/0.004 | 0.374/0.288/0.191 |
| ko-chat | 0.002/0.002/0.002 | 0.464/0.401/0.205 |
| en-prose | 0.015/0.013/0.010 | 0.015/0.011/0.007 |
| en-math | 0.011/0.009/0.007 | 0.007/0.004/0.002 |
| code-rust | 0.015/0.011/0.007 | 0.009/0.007/0.004 |
| code-python | 0.025/0.020/0.012 | 0.007/0.006/0.003 |
| code-c | 0.019/0.013/0.008 | 0.003/0.003/0.002 |

A window keeps ~3.5 tokens (AL at q ~ .03) and at most 4 (`DELTA_LANES`), so `cblind` writes in <= 12 % of windows
(.033 x 3.5) and never more than 4 words; the neutral base writes ~1 id a window early on.
**Recommended path: one asynchronous host-to-card copy of the dirty span before the chain, nothing else.** The writable rows
are one contiguous span of F rows, so the dirty span is inside F x 4 B (8 KB at F = 2,048); the pinned shadow map and a copied-event are the
`Inbox` pattern (`scratch.rs:76-130`: pinned host words, a device twin, an event that keeps the host side unwritten while the
copy still reads it). The host updates after the verify readback (`MtpDraft::record`, `mtp.rs:489`, and `stepped`, `:838`),
which is a stream sync, so the previous copy has long finished and `host_mut` never waits. Prompt ingestion runs the prompt's
ids through the same touch (once, outside the step) and copies the span once before the first chain.

**Per policy.** Prompt-only writes once, at the end of the prompt: one copy of the dirty span (<= F words; the whole map,
304 KB, if the span is unknown), then 0 per step. Prompt+accepted adds the per-window copy above (<= 4 words changed). LRU and
LFU write the same words unless the region overflows, where only the victim row differs.

| path | bytes a changed window | launches a changed window | extra graph nodes | note |
|---|---|---|---|---|
| tail copy (recommended) | <= 8 KB (0.31 us at 26.28 GB/s `[measured link]`) | 1 async copy, fixed cost `[unmeasured]`, assumed <= 20 us | 0 | no kernel, graph untouched; <= 0.057 % of the window in a changed window, <= 0.007 % averaged at 12 % changed windows `[derived]` |
| whole map | 304 KB (11.6 us) | 1 | 0 | simpler, same order |
| (id, row) pairs in the existing inbox copy, applied by a captured scatter | 36 B | 0 | +1 per refresh-walk key (0.852 us, `c_node`, `sparkdec.md:196`) | a new kernel (an `add` entry for ptx-scan and `gate-ptx-spill`) for a term already at 0: not worth it |

### 5.2 Does a fixed-count map keep the captured graph

**Yes.** A captured walk bakes four things (`headpick.md` §3 read against this tree): the map's address (`Some(&hm.ids)`,
`mtp38.rs:2134-2139`), `rows` as the kernel parameter (`q8f32.rs:1944-1955`), the grid `rows.div_ceil(8)`
(`q8f32.rs:1944`), and the argmax's `n` (`gpu/src/mtp.rs:463-510`). A fixed R fixes all four; the walk's key
`MtpKey { m, own, head }` (`mtp38.rs:909-913`) is unchanged, so the captures stay at 5 per slot (`m` = 1..4 refresh, `m` = 1
own), where headpick's two lists double them. Conditions: (a) every map word is always a valid id < vocab (the spare rows hold
the list's own highest ids from load: no empty rows); (b) ids are unique (a host bitmap); (c) the copy is enqueued on the
engine stream after the previous chain's readback and before the next chain's first replay (the host writes between `record`
and the next `propose`); (d) the per-slot map buffer is never reallocated.

### 5.3 Where the per-slot map lives against headpick §2 and G8

Today: one `HeadMap` per model, a vocab-word buffer (`mtp38.rs:166-170`, built at `:302-330`), and `swap_slot` swaps
store, graphs, last and held only (`:561-566`). A writable map per slot is a buffer of R words in `DraftSeq38`, swapped with
`graphs` (a slot's captures bake that slot's own map address). Device bytes: slots x R x 4 B = 608 KB for the default 2 slots
at R = 76,000, 1.2 MB at 4. **Conflict with headpick:** §2 puts both static lists in the one 248,320-word buffer
(141,536 words) and G8 pins `plan.map_bytes` and the open-time `got == want` check to the single-list load's. Two writable
76,000-row maps (152,000 words) do not fit beside the two lists (106,784 free); and one list per slot needs
`plan.map_bytes = slots x R x 4` (the owner states it once, `max(vocab, slots x R)` words). So either headpick keeps two static
lists and no warm map, or it takes one list per slot and G8 changes. The host side per slot is a bitmap (31 KB), the rows
(304 KB) and the recency links (F x 8 B): 0.35 MB, in `MtpDraft`, which is already per slot (its `head` field, `mtp.rs:363`).

### 5.4 What breaks if one map is shared by two slots

- **Tokens: nothing.** The head is draft-only; the target's verify decides every token. The tokens-identical gate (G5) cannot see
  a wrong map at all, as headpick already says: the clause that sees it is an acceptance one (E(4) per slot).
- **Acceptance:** one writable region fed by both slots holds the union of two languages' novel ids. On `cblind` the
  union is small and the cost is ~0 (zh-chat + ko-chat .0285 -> .0285; ja-wiki + zh-wiki .0027 -> .0042, -0.2 points of tok/s).
  On a neutral base with F = 2,048 it is real: zh-chat + ko-chat at 4,096 ids q .318 -> .360 and .209 -> .265 =
  **-5.8 % and -7.6 % tok/s** `[derived]` from the table below.
- **Ordering:** `pass_slots` runs each slot's chain before the one batched verify, all on one stream, and updates come after
  the verify, so a shared map is rewritten between passes, never under a replay. A future pipelined chain (the GLM chain-overlap
  lever) that reads the map while a verify's `record` writes it would score the wrong rows with no error: the copy then needs an
  event against the readers.
- **State:** one host warm set for the model, so the saved profile has one writer and no merge.

| base | F | policy | pair | slot A q split -> shared at t=4096 | slot B q split -> shared at t=4096 |
|---|---|---|---|---|---|
| cblind | 2048 | lru | ja-wiki + en-prose | 0.0027 -> 0.0039 | 0.0107 -> 0.0107 |
| cblind | 2048 | lru | en-prose + code-rust | 0.0107 -> 0.0107 | 0.0064 -> 0.0063 |
| cblind | 2048 | lru | zh-wiki + ja-wiki | 0.0178 -> 0.0178 | 0.0027 -> 0.0044 |
| cblind | 2048 | lru | zh-chat + ko-chat | 0.0285 -> 0.0285 | 0.0025 -> 0.0025 |
| lowid | 26848 | lru | en-prose + code-rust | 0.0063 -> 0.0062 | 0.0036 -> 0.0035 |
| lowid | 26848 | lru | ja-wiki + en-prose | 0.2133 -> 0.2133 | 0.0063 -> 0.0070 |
| lowid | 26848 | lru | zh-wiki + ja-wiki | 0.2205 -> 0.2039 | 0.2133 -> 0.1947 |
| lowid | 26848 | lru | zh-chat + ko-chat | 0.3169 -> 0.3169 | 0.2075 -> 0.2075 |
| chyb | 2048 | lru | zh-chat + ko-chat | 0.0285 -> 0.0285 | 0.0025 -> 0.0025 |
| chyb | 2048 | lru | ja-wiki + en-prose | 0.0032 -> 0.0036 | 0.0107 -> 0.0107 |
| lowid | 2048 | lru | ja-wiki + en-prose | 0.2151 -> 0.2151 | 0.0101 -> 0.0107 |
| chyb | 2048 | lru | en-prose + code-rust | 0.0107 -> 0.0107 | 0.0064 -> 0.0063 |
| chyb | 2048 | lru | zh-wiki + ja-wiki | 0.0178 -> 0.0178 | 0.0032 -> 0.0037 |
| lowid | 2048 | lru | zh-chat + ko-chat | 0.3178 -> 0.3597 | 0.2094 -> 0.2650 |
| lowid | 2048 | lru | en-prose + code-rust | 0.0101 -> 0.0099 | 0.0059 -> 0.0057 |
| lowid | 2048 | lru | zh-wiki + ja-wiki | 0.2208 -> 0.2103 | 0.2151 -> 0.2042 |

### 5.5 Gates and tie order that the writable tail touches

- Argmax ties go to the lower row (`gpu/src/mtp.rs:19`). With an ascending list that is the lower id, as in the full head;
  a rewritten tail breaks the equality for exact f32 ties between tail rows only, and only changes a draft proposal.
- `gate_qwen4exp_mtp.rs` (d) `head_rows` (`:3727-3748`) pins the map as the list's ids then zeros and `:4822-4836` (r) pins
  `map == ids`: both hold at load and must read the shadow after the first write, or compare the protected prefix only. That is
  a coverage change with a dated reason, not a relaxation.

### 5.6 The restore at admission: where the set comes from, and what it costs

Per request that hits a cached prefix (once per request, not per step), resource timeline `[derived]`:

| resource | work | busy | note |
|---|---|---|---|
| host CPU | refold the hit's ids into the slot's map: P x ~0.1 us | 0.4 ms at P = 4,096; 6.6 ms at 64K | `[unmeasured]` per-id cost, the term of §1; a stored copy instead is a memcpy of the tail, microseconds |
| PCIe | the dirty tail of the map to the card, once: F x 4 B | 8 KB at F = 2,048: 0.31 us at 26.28 GB/s; a full 76,000-row map 304 KB: 11.6 us | `pcie-a6000.card`; the same copy §5.1 prices |
| card SMs | none | 0 | no kernel, the captured walk is kept (§5.2) |
| host DRAM | a stored copy beside the entry: F x 4 B plus the recency links | 8 KB + links at F = 2,048; 107 KB + links at the neutral base's F = 26,848 | `Saved::n_bytes` counts the body only (`bind.rs:2324-2331`), so a stored copy must be added to it |
| NVMe | none | 0 | no saved state reaches disk (below) |

The wall is the KV copy-in or the prefill of the uncached tail, whichever the request waits on. The fold runs on the host
beside them and is hidden while it is shorter: 0.1 us x 26.28 GB/s = 2.6 KB, so a state of 2.6 KB per position or more hides
the fold under its own copy-in `[derived]`.

**Where the key and the data are.** (Paths: `qwen38.rs` and `drafted.rs` are under `crates/gpu-gates/src/bin/shared/serve_seats/`,
`bind.rs` is `crates/gpu-gates/src/bind.rs`, `mtp.rs` is `crates/app/src/mtp.rs`.) The prompt cache keys a saved state by the ids it covers (`serve::promptcache`, `Held`) and
keeps it in host RAM under the `cache_ram` budget (`engine.rs:46`); the park table (`swap.rs`) parks a slot's state the same
way when another request takes the engine. Neither writes a file (`File::create`, `fs::write` and `OpenOptions` do not occur in
`promptcache.rs`, `swap.rs`, `engine.rs`, `sched.rs` or `gpu/src/checkpoint/saved.rs`). The draft's side of a saved state is
`Cached::draft` (`bind.rs:2288-2293`) = `Option<ParkedDraft<TargetRows>>` (`qwen38.rs:2420`), taken by `SlotDrafts::park`
(`drafted.rs:364`) at `snapshot` / `snapshot_begin` (`qwen38.rs:2738`, `:2760`), put back by `unpark` (`qwen38.rs:2440`) when
`resume` puts the whole state back, and holding `app::mtp::Parked` (`mtp.rs:333`: the waiting rows, the last unit, the skip
why). The head's set could ride there as one more field.

**Two ways to restore, and the pick.**

1. *Carry* the set in `Parked` as an immutable `Arc<RowSet>` and copy it into the slot's map in `unpark`. Zero host time at
   admission, the exact set of the saved position even if the start state moved since. It needs: `n_bytes` to count it; the copy
   on `unpark`, because the cache keeps a state after a take so that two requests that branch from one saved state both find it
   (`promptcache.rs` module doc), and a shared mutable set would blend them; and an exception for a cut hit, below.
2. *Refold* the request's own prompt prefix (the first `at` ids, `at` being what `resume` / `resume_prefix` returns,
   `qwen38.rs:2769-2800`) into a fresh map from the base. Same code as the fold a miss runs on its whole prompt. No stored
   state, no `n_bytes` term, no aliasing rule; exact at a cut hit; the park table's `Park::Ids` engines, which re-prefill from ids,
   take it unchanged. It costs the host time of the table above.

**Pick: refold**, with the carried copy left as an optimisation to add only if the refold shows above its shadow (the
2.6 KB-per-position threshold, measured when the head's fold is timed). Reason: `resume_prefix` returns a cut `at` below the
state's positions and turns the draft off from there today (`DRAFT_OFF_WHY`, `qwen38.rs:2784-2800`), and a carried set is the
set of the *full* saved positions, wrong at a cut; the fold to `at` is exact (`kv_pure`). The restore therefore matters for
full hits only while the draft is off on cuts, and a refold serves both.

## 6. The profile's owner

The rule (the user, via the lead): **per-slot rows never blend across streams; only card-wide state does.** So the profile
file is card-wide, and the head's row set is per session, keyed by the KV cache's prefix.

**What persists today: nothing.** Searched: `crates/gpu-gates/src/residency38.rs`, `crates/runtime/src/`,
`crates/placement/src/`, `crates/gpu/src/host/` (no `fs::write`, `File::create` or `OpenOptions` in any of them; the repo-wide
list is the diagnostic route trace `host/route_trace.rs` (opt-in, a router set on disk), the gates' temp files, and the GGUF
writers), and `crates/serve/src/{promptcache,swap,engine,sched}.rs` and `crates/gpu/src/checkpoint/saved.rs` (no saved state
reaches disk: §5.6). `residency38.rs`'s "checkpoints" are host RAM reservations for recurrent state. The residency rule starts
every process from the id-prefix seed (`swaprule.rs:1-30`; `adaptive-picks.md` ledger 3.7 lists "the user's own routing,
persisted" as an unopened idea), the state of a running residency resets only by an explicit call (`registry.rs:602`), the width
chooser's profile crosses requests inside the process only (`width.rs:448-500`), and no placement file is read at load
(`registry.rs:230-243` refuses the retired name). The only per-model derived file is the r8 weight sidecar
(`docs/BUILD.md:141`), beside the model directory and not run state. The `tierlift` and `tiercut` worktrees add no state file
(their diffs: `placed.rs`/`model.rs`, a gate and the justfile); unitier's own uncommitted work, if any, is not visible from here.

**The split.**

| state | scope | lifetime | owner | on disk |
|---|---|---|---|---|
| residency counts (`RESD`) | card-wide | the model, across processes | `runtime::swaprule` | yes |
| width chooser (`WIDT`) | card-wide | the model, across processes | `runtime::width` | yes |
| head row set | per session, keyed by the KV prefix (the prompt's ids) | the conversation | the slot's head (`MtpDraft`, already per slot) | **no** |

The head needs no section because its set is a function of the ids (`kv_pure`, `kv_cont`; §4.1): with the start fixed at the
base, the KV prefix's id list is the key and the data at once. At admission the seat folds the request's prompt prefix into the
slot's own map (§5.6, refold); nothing is stored beside the cache entry, so there is no head state for a profile file, a disk
tier of the prompt cache or a second process to carry, and nothing to blend: a fold writes only its slot's map. If the prompt
cache ever gains a disk tier, entries hold no head state to write. A card-wide head section would be the blended profile of
§4.1, which loses to the cold base on ja (-1.8 to -2.2 points on `cblind`), so it is not designed.

**The card-wide file: one file, one reader, a section per owner.** Owner module: `crates/models/src/profile.rs`, the frame only
(header, model key, sections, reader, writer, refusals; no state of its own). `models` depends only on `gguf`, and `runtime`,
`placement`, `model`, `gpu` and `app` all reach it; `placement` does not depend on `runtime` (`crates/placement/Cargo.toml`:
gguf, levers, models), so a residency owner that lands in `placement` could not read a frame owned by `runtime` without a new
edge. Each state keeps its own codec beside it (`runtime::swaprule`, `runtime::width`) and hands the frame bytes; the frame has
no per-model branch.

| field | value |
|---|---|
| name and place | `<state dir>/<model key, 16 hex>.bprof`; state dir `$XDG_STATE_HOME/bloomery` (else `~/.local/state/bloomery`), mode 0600 (`RESD` holds which experts the user's prompts routed to); lever `BLOOMERY_PROFILE=<dir>\|off`, a registry row |
| header | magic `BPROF1`, `u32` format version, the model key, the section count |
| model key | `arch`, `n_layer`, `n_expert`, `vocab_sha256` (the tokenizer digest, `models::headpick::vocab_sha256`), and `tensor_table_sha256`: SHA-256 over every tensor's name, type, dims and size from the GGUF header (to write in `crates/gguf`; header only, no weight read). A path is not an identity: the file moves |
| section | `tag` (4 bytes: `RESD`, `WIDT`), `u32` version, `u32` length, payload, CRC32 of the payload; an unknown tag is carried through a rewrite untouched; every section in the file is card-wide, and a session-scoped tag is refused by the writer |
| `RESD` payload | per layer x expert `u16` counts (the rule's decayed counts): layers x experts x 2 B, tens of KB `[derived]`; codec in `runtime::swaprule` |
| `WIDT` payload | the chooser's per-width gains, mean, seen: < 1 KB; codec in `runtime::width` |
| write points | graceful exit (the session's drop, and a signal handler the server must install) and every N = 65,536 kept tokens (15-20 min at 55-70 tok/s `[derived]`); a snapshot copied on the step thread (microseconds), written by a background thread to `<file>.tmp`, `fsync`, `rename`; never inside a window |
| read | `Profile::open(path, &ModelKey)` at load, one function; <1 ms |
| refusals, by name | `Missing` (cold start, a `profile none` record), `Torn` (no trailer), `FormatVersion`, `OtherModel { field, want, got }`, `SectionVersion { tag }`, `SectionCrc { tag }`. A refusal is printed and the run starts cold; a profile is never used on another model |
| determinism | `RESD` moves the target's bits at near ties (`registry.rs:410-413`), so it needs its own lever and gate; a profile may not make a greedy run depend on history silently. The head moves only the draft: tokens are identical by construction, with or without a restore |

The verdict (§7) does not need the file for the head. The file is worth building for residency, and it carries no head state.

## 7. Verdict

**(b): the lists stay; warm-up is a later step, and the later step is a fill on the shipped list, not a neutral base.**

The numbers that decide it (tok/s against the best static list per set, `[sim]`, seed spread 0.006):

- Neutral bases, minimum over zh/ja at 256 / 1,024 / 4,096 ids seen: lowid .53 / .57 / .66, ec .53 / .57 / .66, both .66 / .69 /
  .77; ko: .57 / .63 / .75 for lowid and ec, .999 for `both` (the ids both lists hold include Hangul). Long-run lowid
  .82-.92 at 16,384 ids. (a) fails on the 6 zh/ja/ko sets under `lowid` and `ec`, and on the 4 zh/ja sets under `both`, for every B and policy. By set at 4,096 ids (lowid): zh-wiki .76, zh-chat .66, ja-wiki .76, ja-chat .70, ko-wiki .77, ko-chat .75.
- (c) fails: the 20,000-id profile gives .79-.90 at t = 0 on those sets; the fit to parity is 0.6-4 million ids.
- With `C` as the base and 2,048 writable rows (`cblind`): all 11 sets >= .991 at 256 ids, N = 256; the same arm with the
  rows chosen by data is within 0.001.
- Against the headpick rule it is a trade, not a tie inside the seed spread: the rule is >= .999 from 64 ids on every set
  but .953-.962 on zh/ja at id 0 (the empty prompt sits on `D`); `cblind` is .983-.992 on en/code at id 0, .991-1.001 by 256
  and .994 on en-prose at 4,096 (the 0.345 % of the larger head and the misses `C` lacks), so the pick leads en/code by up to
  0.9 points at 256 ids and `cblind` leads zh/ja by 4-5 points at id 0. Both are inside the 4-round ruler (+-1.0 %) from 256
  ids on. Pick + fill on `D` (`dblind`) beats both on en/code by +0.2 to +1.8 points and costs both mechanisms.
- Restore and the start of a miss (§4.1): a restore is the fold of the conversation's ids, exact (79,200 of 79,200 windows).
  On `cblind` it is worth +0.8 (code) and +0.3 (en) points averaged over a 512-id reply and 0 on zh/ja/ko, inside the 4-round
  ruler; on the neutral base it is +3 to +23 points at t2 = 0 and zh/ja stay at .53-.66 for T1 <= 4,096, so (b) stands. A blended
  card-wide head profile as a miss's start costs ja 1.8-2.2 points on `cblind` and ko 0.6: **a miss starts from the base, and the
  head has no card-wide section**. That is the one place the data and the lead's stated design disagree ("a miss starts from the
  global profile"); it needs the user's re-decision if the global profile was meant to include a head part.

**What it means for headpick L2.** L0 and L1 stand (move-only, `HeadSel`, one owner). Before L2 builds a second list, a pick,
an upgrade and a chooser class state (G1-G10), consider one list (`C`) with 2,048 writable rows: the list file and
its pin are unchanged (the span is the ascending list's own highest non-control ids), `D` and its restore are not needed, the walk key
keeps its shape, and the closed-chooser reopen of §1.4 does not arise (no class change). It recovers the en/code 1.5-2.9 %
that 0.3.0's `C` gives up (static `C` .971-.985 of the best list -> .991-1.001 by 256 ids). It costs a per-slot writable map (§5.3, which moves G8), a tail
copy per changed window (§5.1) and a new gate pair (host: the fill policy by id vectors; device: the map equals its shadow after
N windows, FAIL-first on a stale copy; a control-id clause, FAIL-first on a span that includes the control block: after a
chat-template prompt `<|im_end|>` is still a row; plus an acceptance clause per slot, since tokens cannot see a wrong map). It is the lead's
call; the simulation prices it at a trade with the pick, inside the 4-round ruler from 256 ids on, and it is the simpler of the two.

## 8. What was not measured, and open questions

- Real chats: the sets are concatenated documents; topic locality would speed the fill (and a translation prompt, English
  text with a Chinese reply, is again the case only an upgrade or a fill covers: `cblind` has `C` for it).
- The victims: `cblind` evicted at most the highest ~660 ids of `C` in 20,000 ids; a long session evicts further into `C`'s tail
  and the cost is then the use-mass of those ids, unmeasured beyond that. Real ranks from the builder (it drops them: the list is
  written ascending) would choose them better.
- The host cost of a touch (~0.1 us an id), the fixed cost of one async copy, and the capture time of a walk key are
  `[unmeasured]`; each sits under any ruler.
- Qwen3.8 only. GLM and DSpark have no list; the common owner's attach is headpick §7.
- The `ec`/`both` bases are the two readable surrogates of "ranked high in both lists"; with ranks (a builder change,
  `draft-vocab.py`) a rank-based neutral base is a fifth candidate the sets could price in minutes.

- Turns are consecutive chunks of concatenated documents, so a real two-turn chat has more topic locality than the restore
  arm models: its gains on the neutral base are a floor. A restore at a cut hit is not simulated (the draft is off there
  today, `qwen38.rs:2784-2800`); the refold of §5.6 is exact there by `kv_pure`.
- The blend is one order (round-robin in chunks of 256 ids over all 11 sets' other halves). A card whose traffic is one
  language blends less; en/code-only traffic gives a blend that costs ja its tail rows (§4.1). Not simulated: a blend that
  excludes the test language, or a different chunk size.
- `cblind`'s spare rows (the highest non-control ids of `C`) are the most ja-heavy rows `C` has: 4.3-4.9 % of ja's tokens. A
  session that switches into ja after en or code has overwritten some of them is not simulated; a rank-based choice (the
  builder drops the ranks, `draft-vocab.py:26-30`) would take the least-used rows instead.
- The fold's host cost per id (~0.1 us) and the 2.6 KB-per-position threshold at which a state's copy-in hides it are
  `[derived]` / `[unmeasured]`; the head's fold has not been timed.

## Appendix A: the simulation script

`HW_ROOT=<repo> python3 -I headwarm.py <step> <out_dir> [configs] [json name]`, steps `cold static profile cross shared long
kvturn kvreport report`. `configs` is `;`-joined JSON arrays `["kind", R, B, "lru"|"lfu"]`. Seeds: `HW_SEED=8` or `9`.

```python
#!/usr/bin/env python3
"""headwarm: a warm-up draft head and a saved profile against the two static head lists. Mac simulation, standard
library only; run as `python3 -I headwarm.py <step> [out_dir]` with step one of cold static profile cross shared long kvturn kvreport.

Row set: R rows fixed. The first B ranks of a neutral ranking are the protected base; the other R-B rows are mutable,
at t=0 the ranking's own ranks B..R-1 ("filler"). A seen id that is not in the set takes the lowest-ranked filler row,
then (no filler left) the LRU/LFU seen row; the base is never evicted. The map is fed in windows of BLOCK kept ids.
"""
import sys, os, random, heapq, collections, json, multiprocessing as mp

W = os.environ.get("HW_ROOT", ".")   # a bloomery checkout that has commit 347a86e5
SETS = ["zh-wiki", "zh-chat", "ja-wiki", "ja-chat", "ko-wiki", "ko-chat", "en-prose", "en-math",
        "code-rust", "code-python", "code-c"]
ENCODE = ["en-prose", "en-math", "code-rust", "code-python", "code-c"]
CLASS = {"en": ["en-prose", "en-math"], "code": ["code-rust", "code-python", "code-c"],
         "zh/ja": ["zh-wiki", "zh-chat", "ja-wiki", "ja-chat"], "ko": ["ko-wiki", "ko-chat"]}
VOCAB = 248320
CONTROL_FROM = 248044          # the nine ids both lists hold above 248,000 (end, pad, chat-template, think tags): never writable
BLOCK = 4                      # runtime::stores::DELTA_LANES = kept ids a window feeds back
TS = (0, 64, 256, 1024, 4096)  # tokens seen
WN = (64, 256)                 # the miss is read over the next WN ids
TSET = frozenset(TS)
LMAX = max(TS) + max(WN)       # ids a trial walks
K = 200                        # trials per half and set
SEED = os.environ.get("HW_SEED", "7")
# head_coverage.rs:36-78 (the model every ratio goes through)
KEPT = [0.0, 3.0, 2.0, 21.0]; WALKS = 3.0; ROW = 2720.0; GBS = 701.0; WINDOW = 37.43


def al(q):
    w = sum(KEPT)
    return 1 + sum(sum(KEPT[j:]) / w * (1 - q) ** j for j in range(1, len(KEPT)))


def window_ms(rows):
    return WINDOW - WALKS * (VOCAB - rows) * ROW / (GBS * 1e9) * 1e3


def toks(q, rows):             # tokens a ms of the model's window; only ratios are claimed
    return al(q) / window_ms(rows)


def ids_of(lines):
    return [int(l) for l in (x.strip() for x in lines) if l and not l.startswith("#")]


def init():
    import subprocess
    global CL, DL, Cs, Ds, CmD, DmC, S, HALF, _RK, _ST0
    CL = ids_of(open(f"{W}/crates/model/data/mtp-head-rows-qwen38-cjk-76000.txt"))
    DL = ids_of(subprocess.check_output(["git", "-C", W, "show", "347a86e5:crates/model/data/"
                "mtp-head-rows-qwen38-kohangul-65536.txt"]).decode().splitlines())
    Cs, Ds = set(CL), set(DL)
    CmD, DmC = Cs - Ds, Ds - Cs
    S = {s: ids_of(open(f"{W}/crates/models/tests/data/head-holdout/{s}.ids")) for s in SETS}
    HALF = {s: len(v) // 2 for s, v in S.items()}
    _RK, _ST0 = {}, {}


def half_ids(s, h):
    return S[s][:HALF[s]] if h == 0 else S[s][HALF[s]:]


def ranking(kind, g):
    """The neutral ranking's first 76,000 ids; `g` is the half whose ids may train it (never the half it is tested on)."""
    key = (kind, g if kind in ("ec", "chyb") else None)
    if key in _RK:
        return _RK[key]
    if kind == "lowid":                       # BPE merge order: the tokenizer's own prior, no data
        r = list(range(76000))
    elif kind == "both":                      # in both lists (ascending id: the files carry no rank), then the rest of the union
        inter = sorted(Cs & Ds)
        r = (inter + sorted((Cs | Ds) - (Cs & Ds)))[:76000]
    elif kind == "ec":                        # en + code halves by count, then the lowest ids
        cnt = collections.Counter()
        for s in ENCODE:
            cnt.update(half_ids(s, g))
        top = sorted(cnt, key=lambda i: (-cnt[i], i)); used = set(top)
        r = (top + [i for i in range(76000 + len(top)) if i not in used])[:76000]
    elif kind == "chyb":                      # cjk-76000's own rows, ranked by their count in every set's half g
        cnt = collections.Counter()
        for s in SETS:
            cnt.update(half_ids(s, g))
        r = sorted(CL, key=lambda i: (-cnt[i], i))
    elif kind == "dblind":                    # no data: kohangul-65536 in id order, control block first, so the writable rows are the highest ids below it
        r = sorted(i for i in DL if i >= CONTROL_FROM) + sorted(i for i in DL if i < CONTROL_FROM)
    elif kind == "cblind":                    # no data: cjk-76000 in id order, control block first, so the writable rows are the highest ids below it
        r = sorted(i for i in CL if i >= CONTROL_FROM) + sorted(i for i in CL if i < CONTROL_FROM)
    else:
        raise ValueError(kind)
    _RK[key] = r
    return r


def state0(kind, g, R, B):
    key = (kind, g if kind in ("ec", "chyb") else None, R, B)
    if key not in _ST0:
        r = ranking(kind, g); st = bytearray(VOCAB)
        for i in range(B):
            st[r[i]] = 1
        for i in range(B, R):
            st[r[i]] = 2
        _ST0[key] = st
    return _ST0[key]


class Warm:
    """The mutable row set. st[id]: 0 not a row, 1 base, 2 filler (a row nothing has asked for), 3 seen."""
    def __init__(self, st0, rank, B, R, mode):
        self.st, self.rank, self.B, self.fp, self.mode = bytearray(st0), rank, B, R - 1, mode
        self.od, self.cnt, self.heap, self.last, self.stamp, self.changes = collections.OrderedDict(), {}, [], {}, 0, 0

    def clone(self):
        w = Warm.__new__(Warm)
        w.st, w.rank, w.B, w.fp, w.mode = bytearray(self.st), self.rank, self.B, self.fp, self.mode
        w.od, w.cnt, w.heap = self.od.copy(), dict(self.cnt), list(self.heap)
        w.last, w.stamp, w.changes = dict(self.last), self.stamp, 0
        return w

    def _add(self, x):
        if self.mode == "lru":
            self.od[x] = None
        else:
            self.stamp += 1; self.cnt[x] = 1; self.last[x] = self.stamp
            heapq.heappush(self.heap, (1, self.stamp, x))

    def _victim(self):
        if self.mode == "lru":
            return self.od.popitem(last=False)[0] if self.od else -1
        while self.heap:
            c, sp, i = heapq.heappop(self.heap)
            if self.st[i] == 3 and self.cnt.get(i) == c and self.last.get(i) == sp:
                del self.cnt[i], self.last[i]
                return i
        return -1

    def touch(self, x):
        st = self.st; s = st[x]
        if s == 1:
            return
        if s == 3:
            if self.mode == "lru":
                self.od.move_to_end(x)
            else:
                self.stamp += 1; c = self.cnt[x] + 1; self.cnt[x] = c; self.last[x] = self.stamp
                heapq.heappush(self.heap, (c, self.stamp, x))
            return
        if s == 2:                             # a filler row asked for: the row already holds the id, no map write
            st[x] = 3; self._add(x)
            return
        fp, rank, B = self.fp, self.rank, self.B
        while fp >= B and st[rank[fp]] != 2:
            fp -= 1
        self.fp = fp
        if fp >= B:
            st[rank[fp]] = 0; self.fp = fp - 1
        else:
            v = self._victim()
            if v < 0:
                return                         # B == R: no mutable row
            st[v] = 0
        st[x] = 3; self._add(x); self.changes += 1


def run(w, ids, o, L=LMAX, frozen=True):
    """Walk ids[o:o+L] through `w`: the miss bit of every position (the row set before the window's update), the
    frozen-set miss counts at each t in TS (prompt-only: the set stops at t), the changes counter at every position."""
    st, touch = w.st, w.touch
    miss, chg, froz, pend = bytearray(L), [0] * (L + 1), {}, []
    for j in range(L):
        if frozen and j in TSET:
            froz[j] = tuple(sum(1 for x in ids[o + j:o + j + n] if st[x] == 0) for n in WN)
        x = ids[o + j]
        if st[x] == 0:
            miss[j] = 1
        chg[j] = w.changes
        pend.append(x)
        if len(pend) == BLOCK:
            for y in pend:
                touch(y)
            pend = []
    chg[L] = w.changes
    return miss, froz, chg


def offsets(s, h, k=K, span=LMAX):
    rng = random.Random(f"{s}/{h}/{SEED}")
    return [rng.randrange(h * HALF[s], (h + 1) * HALF[s] - span + 1) for _ in range(k)]


def zero():
    return {f"{t}/{n}": [0, 0, 0] for t in TS for n in WN}   # online miss, frozen miss, map writes


def acc(res, miss, froz, chg):
    for t in TS:
        for n in WN:
            a = res[f"{t}/{n}"]
            a[0] += sum(miss[t:t + n]); a[1] += froz[t][WN.index(n)]; a[2] += chg[t + n] - chg[t]


def task_cold(a):
    cfg, s = a; kind, R, B, mode = cfg
    res = zero()
    for h in (0, 1):
        g = 1 - h
        w0 = Warm(state0(kind, g, R, B), ranking(kind, g), B, R, mode)
        for o in offsets(s, h):
            acc(res, *run(w0.clone(), S[s], o))
    return ("cold", cfg, s, res)


def pick_c(prompt):
    if not prompt:
        return False
    sc = sum(1 for x in prompt if x in CmD) - sum(1 for x in prompt if x in DmC)
    return sc >= max(3, 0.00245 * len(prompt))


def task_static(s):
    """The two lists and the headpick rule on the same trials. `pick` is the rule of headpick.md 1.2 plus the 1.4
    upgrade, `pickonly` the prompt-only pick; each reads [misses on C, positions on C, misses on D, positions on D]."""
    keys = [f"{t}/{n}" for t in TS for n in WN]
    out = {"C": dict.fromkeys(keys, 0), "D": dict.fromkeys(keys, 0)}
    for nm in ("pick", "pickonly"):
        out[nm] = {k: [0, 0, 0, 0] for k in keys}
    for h in (0, 1):
        for o in offsets(s, h):
            ids = S[s]
            for t in TS:
                useC0 = pick_c(ids[o:o + t])
                for name, up in (("pick", True), ("pickonly", False)):
                    useC, last, rows, blk = useC0, collections.deque(maxlen=16), [], []
                    for j in range(t, t + max(WN)):
                        x = ids[o + j]
                        rows.append((useC, 0 if x in (Cs if useC else Ds) else 1))
                        blk.append(x)
                        if len(blk) == BLOCK:
                            for y in blk:
                                last.append(y not in Ds)
                                if up and not useC and sum(last) >= 5:
                                    useC = True
                            blk = []
                    for n in WN:
                        a = out[name][f"{t}/{n}"]
                        for c, m in rows[:n]:
                            a[0 if c else 2] += m; a[1 if c else 3] += 1
                for n in WN:
                    seg = ids[o + t:o + t + n]
                    out["C"][f"{t}/{n}"] += sum(1 for x in seg if x not in Cs)
                    out["D"][f"{t}/{n}"] += sum(1 for x in seg if x not in Ds)
    return ("static", s, out)


NTRAIN = (0, 1000, 2500, 5000, 10000, None)   # ids the profile was built from; None = the whole half


def task_profile(a):
    cfg, s, kp = a; kind, R, B, mode = cfg
    out = {}
    for h in (0, 1):                           # tested on half h, built on half g: out[h][n]
        g = 1 - h
        w = Warm(state0(kind, g, R, B), ranking(kind, g), B, R, mode)
        tr = half_ids(s, g); done = 0
        for n in NTRAIN:
            m = len(tr) if n is None else n
            for x in tr[done:m]:
                w.touch(x)
            done = m
            res = zero()
            for o in offsets(s, h, kp):
                acc(res, *run(w.clone(), S[s], o))
            out.setdefault(str(h), {})[str(n)] = res
    return ("profile", cfg, s, out)


def task_cross(a):
    cfg, xs, kp = a; kind, R, B, mode = cfg
    out = {}
    for g in (0, 1):                           # profile built on half g of set xs, tested on half 1-g of every set
        h = 1 - g
        w = Warm(state0(kind, g, R, B), ranking(kind, g), B, R, mode)
        for x in half_ids(xs, g):
            w.touch(x)
        for ys in SETS:
            res = out.setdefault(ys, zero())
            r2 = zero()
            for o in offsets(ys, h, kp):
                acc(r2, *run(w.clone(), S[ys], o))
            for k2, v in r2.items():
                for i in range(3):
                    res[k2][i] += v[i]
    return ("cross", cfg, xs, out)


def task_shared(a):
    cfg, pair, kp = a; kind, R, B, mode = cfg
    sa, sb = pair; L = LMAX
    out = {"shared": {sa: zero(), sb: zero()}, "split": {sa: zero(), sb: zero()}}
    for h in (0, 1):
        g = 1 - h
        oa, ob = offsets(sa, h, kp), offsets(sb, h, kp)
        for i in range(kp):
            base = Warm(state0(kind, g, R, B), ranking(kind, g), B, R, mode)
            for mode2 in ("split", "shared"):
                wa = base.clone(); wb = wa if mode2 == "shared" else base.clone()
                mm = {sa: bytearray(L), sb: bytearray(L)}
                for j in range(0, L, BLOCK):   # the two slots' windows alternate; one map under "shared"
                    for nm, w, o in ((sa, wa, oa[i]), (sb, wb, ob[i])):
                        blk = S[nm][o + j:o + j + BLOCK]
                        for q, x in enumerate(blk):
                            if w.st[x] == 0:
                                mm[nm][j + q] = 1
                        for x in blk:
                            w.touch(x)
                for nm in (sa, sb):
                    for t in TS:
                        for n in WN:
                            out[mode2][nm][f"{t}/{n}"][0] += sum(mm[nm][t:t + n])
    return ("shared", cfg, pair, out)


KT1 = (256, 1024, 4096)        # ids of turn 1 (prompt + reply) before turn 2 restores its row set
KT2 = (0, 64, 256, 1024)       # ids of turn 2 seen when the miss is read over the next KN ids
KN = 256
KL2 = max(KT2) + KN
KSPAN = max(KT1) + KL2         # ids a two-turn trial walks
KSTART = ("base", "prof", "blend")   # what a session's first turn starts from: the base, the same-set profile, the card-wide blend
_BLEND = {}


def blend_profile(kind, g, R, B, mode):
    """The card-wide profile: the row set after every set's half g went through it, round-robin in chunks of 256 ids."""
    key = (kind, g if kind in ("ec", "chyb") else None, g, R, B, mode)
    if key not in _BLEND:
        w = Warm(state0(kind, g, R, B), ranking(kind, g), B, R, mode)
        tr = {s: half_ids(s, g) for s in SETS}; pos = dict.fromkeys(SETS, 0); live = True
        while live:
            live = False
            for s in SETS:
                c = tr[s][pos[s]:pos[s] + 256]
                if c:
                    live = True
                    for x in c:
                        w.touch(x)
                pos[s] += 256
        _BLEND[key] = w
    return _BLEND[key]


def task_kvturn(a):
    """Two turns of one set: turn 1 is ids[o:o+T1], turn 2 the next KL2 ids. `restore`: turn 2 starts from the row set turn 1
    left (a KV-prefix hit); `fresh`: turn 2 starts from the session's start state (a miss). The miss of both is counted on the
    same turn-2 windows as the two static lists. `pure`: the row set after turn 1, walked live in windows of BLOCK ids, against
    the same ids folded in order in one go (what an admission that refolds the cached ids would build)."""
    cfg, s, kp = a; kind, R, B, mode = cfg
    ids = S[s]
    res = {nm: {str(T1): {str(t2): [0, 0] for t2 in KT2} for T1 in KT1} for nm in KSTART}
    stat = {str(T1): {str(t2): [0, 0] for t2 in KT2} for T1 in KT1}
    pure, cont = [0, 0], [0, 0]
    for h in (0, 1):
        g = 1 - h
        base = Warm(state0(kind, g, R, B), ranking(kind, g), B, R, mode)
        prof = base.clone()
        for x in half_ids(s, g):
            prof.touch(x)
        starts = {"base": base, "prof": prof, "blend": blend_profile(kind, g, R, B, mode)}
        for o in offsets(s, h, kp, KSPAN):
            for T1 in KT1:
                o2 = o + T1
                for t2 in KT2:
                    seg = ids[o2 + t2:o2 + t2 + KN]
                    stat[str(T1)][str(t2)][0] += sum(1 for x in seg if x not in Cs)
                    stat[str(T1)][str(t2)][1] += sum(1 for x in seg if x not in Ds)
                for nm, w0 in starts.items():
                    w = w0.clone()
                    run(w, ids, o, L=T1, frozen=False)                       # turn 1, live
                    ref = w0.clone()
                    for x in ids[o:o + T1]:
                        ref.touch(x)                                          # the same ids folded in order
                    pure[1] += 1
                    pure[0] += int(ref.st == w.st and list(ref.od) == list(w.od))
                    mr, _, _ = run(w, ids, o2, L=KL2, frozen=False)           # KV hit: turn 2 on turn 1's set
                    mf, _, _ = run(w0.clone(), ids, o2, L=KL2, frozen=False)  # KV miss: turn 2 on the start state
                    mc, _, _ = run(w0.clone(), ids, o, L=T1 + KL2, frozen=False)  # the stream that never stopped
                    for t2 in KT2:
                        cont[1] += 1; cont[0] += int(sum(mc[T1 + t2:T1 + t2 + KN]) == sum(mr[t2:t2 + KN]))
                        c = res[nm][str(T1)][str(t2)]
                        c[0] += sum(mr[t2:t2 + KN]); c[1] += sum(mf[t2:t2 + KN])
    return ("kvturn", cfg, s, {"res": res, "stat": stat, "pure": pure, "cont": cont, "trials": 2 * kp})

LONGT = (0, 1024, 4096, 8192, 12288, 16384)


def task_long(a):
    cfg, s = a; kind, R, B, mode = cfg
    out = {str(t): 0 for t in LONGT}; cnt = 0
    for h in (0, 1):
        g = 1 - h
        w = Warm(state0(kind, g, R, B), ranking(kind, g), B, R, mode)
        L = HALF[s] - HALF[s] % BLOCK
        miss, _, _ = run(w, S[s], h * HALF[s], L=L, frozen=False)
        for t in LONGT:
            out[str(t)] += sum(miss[t:t + 512])
        cnt += 1
    return ("long", cfg, s, out)


def configs():
    c = []
    for kind in ("lowid", "ec", "both"):
        for R in (65536, 76000):
            for B in (32768, 49152, 65536):
                if B <= R:
                    c += [(kind, R, B, m) for m in (("lru", "lfu") if B < R else ("lru",))]
    c += [("chyb", 76000, B, m) for B in (32768, 49152, 65536) for m in ("lru", "lfu")]
    return c


def pool_run(func, tasks, out):
    ctx = mp.get_context("fork")
    with ctx.Pool(10) as p:
        res = list(p.imap_unordered(func, tasks, chunksize=1))
    json.dump(res, open(out, "w"))
    return res


# ---------------------------------------------------------------- report: the tables of the doc, from the json files
def _load(outd, name):
    return json.load(open(f"{outd}/{name}.json"))


def _idx(rows, with_cfg=True):
    return {((tuple(r[1]), r[2]) if with_cfg else r[1]): r[-1] for r in rows}


def report(outd):
    os.makedirs(f"{outd}/tables", exist_ok=True)
    NTR = 2 * K
    cold, stat = _idx(_load(outd, "cold")), {r[1]: r[2] for r in _load(outd, "static")}
    for nm in ("coldx", "coldc65", "coldcb", "colddb"):
        cold.update(_idx(_load(outd, nm)))
    lng = _idx(_load(outd, "long")); lng.update(_idx(_load(outd, "longx")))
    for nm in ("longc65", "longcb", "longdb"):
        lng.update(_idx(_load(outd, nm)))
    T = {}

    def best(s, t, n=256):
        return max(toks(stat[s]["C"][f"{t}/{n}"] / (NTR * n), 76000), toks(stat[s]["D"][f"{t}/{n}"] / (NTR * n), 65536))

    def pick_tokms(a):
        nC, nD = a[1], a[3]; tot = nC + nD; tau = 0.0
        if nC:
            tau += nC / tot * window_ms(76000) / al(a[0] / nC)
        if nD:
            tau += nD / tot * window_ms(65536) / al(a[2] / nD)
        return 1 / tau

    def r_static(name, s, t, n=256):
        if name == "C":
            return toks(stat[s]["C"][f"{t}/{n}"] / (NTR * n), 76000) / best(s, t, n)
        if name == "D":
            return toks(stat[s]["D"][f"{t}/{n}"] / (NTR * n), 65536) / best(s, t, n)
        return pick_tokms(stat[s][name][f"{t}/{n}"]) / best(s, t, n)

    def r_warm(cfg, s, t, which=0, n=256):
        return toks(cold[(cfg, s)][f"{t}/{n}"][which] / (NTR * n), cfg[1]) / best(s, t, n)

    def cell(vals):
        return "/".join("%.3f" % v for v in vals)

    # T1: the static lists and the pick, q and ratio at t=0 (window 256)
    L = ["| set | q C | q D | best static | C vs best | D vs best | pick+upgrade vs best at t=0 / 64 / 256 | prompt-only pick at t=64 / 256 |", "|---|---|---|---|---|---|---|---|"]
    for sname in SETS:
        qc, qd = stat[sname]["C"]["0/256"] / (NTR * 256), stat[sname]["D"]["0/256"] / (NTR * 256)
        b = "C" if toks(qc, 76000) >= toks(qd, 65536) else "D"
        L.append("| %s | %.4f | %.4f | %s | %.3f | %.3f | %s | %s |" % (sname, qc, qd, b, r_static("C", sname, 0), r_static("D", sname, 0),
                 cell([r_static("pick", sname, t) for t in (0, 64, 256)]), cell([r_static("pickonly", sname, t) for t in (64, 256)])))
    T["static"] = "\n".join(L)

    # T2: families against the best static, the minimum over the sets of a class, t = 0 / 256 / 1K / 4K (online, window 256)
    fam = [("pick+upgrade (headpick 1.2 + 1.4)", None, "pick"), ("static cjk-76000", None, "C"), ("static kohangul-65536", None, "D"),
           ("lowid R=76000 B=48K, online", ("lowid", 76000, 49152, "lru"), 0), ("lowid R=76000 B=48K, prompt-only", ("lowid", 76000, 49152, "lru"), 1),
           ("lowid R=65536 B=32K, online", ("lowid", 65536, 32768, "lru"), 0), ("ec R=65536 B=32K, online", ("ec", 65536, 32768, "lru"), 0),
           ("both R=76000 B=48K, online", ("both", 76000, 49152, "lru"), 0),
           ("cjk-76000 + 2048 spare rows (highest ids), online", ("cblind", 76000, 73952, "lru"), 0),
           ("cjk-76000 + 2048 spare rows, prompt-only", ("cblind", 76000, 73952, "lru"), 1),
           ("cjk-76000 + 6000 spare rows, online", ("cblind", 76000, 70000, "lru"), 0),
           ("cjk-76000 + fill ranked by held-out halves, B=48K, online", ("chyb", 76000, 49152, "lru"), 0),
           ("cjk-65536 + fill ranked by held-out halves, B=56K, online", ("chyb", 65536, 57344, "lru"), 0),
           ("kohangul-65536 + 2048 spare rows, online (pick + fill)", ("dblind", 65536, 63488, "lru"), 0)]
    L = ["| arm | " + " | ".join(CLASS) + " |", "|---|" + "---|" * len(CLASS)]
    for label, cfg, w in fam:
        cells = []
        for c, ss in CLASS.items():
            vs = []
            for t in (0, 256, 1024, 4096):
                if cfg is None:
                    v = min(r_static(w, x, t) for x in ss)
                else:
                    v = min(r_warm(cfg, x, t, w) for x in ss)
                vs.append(v)
            cells.append(cell(vs))
        L.append("| %s | %s |" % (label, " | ".join(cells)))
    T["families"] = "\n".join(L)

    # T3: every set, four arms, online ratio at t=0 / 64 / 256 / 1K / 4K
    def per_set(cfg, which=0):
        longcols = which == 0                 # the long run is online: no prompt-only column of it
        L = ["| set | " + " | ".join("t=%d" % t for t in TS) + (" | long 8192 | long 16384 |" if longcols else " |"),
             "|---|" + "---|" * (len(TS) + (2 if longcols else 0))]
        for sname in SETS:
            row = " | ".join("%.3f" % r_warm(cfg, sname, t, which) for t in TS)
            if longcols:
                row += " | " + " | ".join("%.3f" % (toks(lng[(cfg, sname)][str(t)] / (2 * 512), cfg[1]) / best(sname, 0)) for t in (8192, 16384))
            L.append("| %s | %s |" % (sname, row))
        return "\n".join(L)
    T["set_lowid"] = per_set(("lowid", 76000, 49152, "lru"))
    T["set_both"] = per_set(("both", 76000, 49152, "lru"))
    T["set_cblind"] = per_set(("cblind", 76000, 73952, "lru"))
    T["set_cblind_frozen"] = per_set(("cblind", 76000, 73952, "lru"), 1)
    T["set_dblind"] = per_set(("dblind", 65536, 63488, "lru"))


    # T3b: ids seen to parity: the first t at which every set of the class reaches 0.99 of its best static (online, window 256)
    arms = [("pick+upgrade", None), ("lowid R=76000 B=48K", ("lowid", 76000, 49152, "lru")), ("ec R=65536 B=32K", ("ec", 65536, 32768, "lru")),
            ("both R=76000 B=48K", ("both", 76000, 49152, "lru")), ("cjk-76000 + 2048 spare rows", ("cblind", 76000, 73952, "lru")),
            ("kohangul-65536 + 2048 spare rows", ("dblind", 65536, 63488, "lru"))]
    L = ["| arm | " + " | ".join(CLASS) + " |", "|---|" + "---|" * len(CLASS)]
    for label, cfg in arms:
        cells = []
        for c, ss in CLASS.items():
            hit = None
            for t in TS:
                v = min((r_static("pick", x, t) if cfg is None else r_warm(cfg, x, t)) for x in ss)
                if v >= 0.99:
                    hit = t; break
            if hit is not None:
                cells.append("%d" % hit)
            else:
                lg = min(toks(lng[(cfg, x)]["16384"] / 1024, cfg[1]) / best(x, 0) for x in ss) if cfg else float("nan")
                cells.append("not by 4096 (min at 4096 %.3f; long-run 16384 %.3f)" % (min(r_warm(cfg, x, 4096) for x in ss), lg))
        L.append("| %s | %s |" % (label, " | ".join(cells)))
    T["parity"] = "\n".join(L)

    # T4: map writes per kept token (window 256 after t)
    L = ["| set | cjk-76000 + 2048 spare rows, t=0 / 256 / 4096 | lowid R=76000 B=48K, t=0 / 256 / 4096 |", "|---|---|---|"]
    for sname in SETS:
        a = [cold[(("cblind", 76000, 73952, "lru"), sname)][f"{t}/256"][2] / (NTR * 256) for t in (0, 256, 4096)]
        b = [cold[(("lowid", 76000, 49152, "lru"), sname)][f"{t}/256"][2] / (NTR * 256) for t in (0, 256, 4096)]
        L.append("| %s | %s | %s |" % (sname, "/".join("%.3f" % v for v in a), "/".join("%.3f" % v for v in b)))
    T["writes"] = "\n".join(L)

    # T5: the saved profile, q and ratio at t=0 against the ids it was built from, and the extrapolation
    import math
    prof_dir = {(tuple(r[1]), r[2]): r[3] for r in _load(outd, "profile")}
    prof = {}
    for k_, v_ in prof_dir.items():            # both directions summed: 100 trials a cell
        tot = {}
        for h_ in v_:
            for n_, res_ in v_[h_].items():
                a_ = tot.setdefault(n_, zero())
                for kk, vv in res_.items():
                    for i_ in range(3):
                        a_[kk][i_] += vv[i_]
        prof[k_] = tot
    for cfg in (("lowid", 76000, 49152, "lru"), ("cblind", 76000, 73952, "lru")):
        L = ["| set | built on -> tested on | q cold | ratio cold | ratio n=20K at t=0 | t=64 | t=256 | t=1024 | t=4096 |", "|---|---|---|---|---|---|---|---|---|"]
        for sname in SETS:
            for h_, lab in (("1", "half 0 -> half 1"), ("0", "half 1 -> half 0")):
                P_ = prof_dir[(cfg, sname)][h_]
                qq = lambda n_, t_: P_[n_][f"{t_}/256"][0] / (50 * 256)
                L.append("| %s | %s | %.4f | %.3f | %s |" % (sname, lab, qq("0", 0), toks(qq("0", 0), cfg[1]) / best(sname, 0),
                         " | ".join("%.3f" % (toks(qq("None", t_), cfg[1]) / best(sname, t_)) for t_ in (0, 64, 256, 1024, 4096))))
        T["profile_dir_" + cfg[0]] = "\n".join(L)
    KP = 50; NP = 2 * KP
    for cfg in (("lowid", 76000, 49152, "lru"), ("cblind", 76000, 73952, "lru")):
        L = ["| set | q cold | q n=1K | q n=5K | q n=20K | ratio cold | ratio n=20K | ratio n=20K at t=4096 | exponent | ids to ratio .99 (extrapolated) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for sname in SETS:
            P = prof[(cfg, sname)]
            q = {n: P[n]["0/256"][0] / (NP * 256) for n in ("0", "1000", "5000", "10000", "None")}
            xs = [math.log(5000), math.log(10000), math.log(20000)]
            ys = [math.log(max(q["5000"], 1e-6)), math.log(max(q["10000"], 1e-6)), math.log(max(q["None"], 1e-6))]
            mx, my = sum(xs) / 3, sum(ys) / 3
            sl = -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            lo, hi = 0.0, 1.0                      # q* at which the ratio is 0.99
            for _ in range(50):
                mid = (lo + hi) / 2
                if toks(mid, cfg[1]) / best(sname, 0) > 0.99:
                    lo = mid
                else:
                    hi = mid
            qstar = lo
            n_star = "reached" if q["None"] <= qstar else ("%.0fK" % (20 * (q["None"] / qstar) ** (1 / max(sl, 0.05))) if sl > 0.05 else "no decay")
            L.append("| %s | %.4f | %.4f | %.4f | %.4f | %.3f | %.3f | %.3f | %.2f | %s |" % (sname, q["0"], q["1000"], q["5000"], q["None"],
                     toks(q["0"], cfg[1]) / best(sname, 0), toks(q["None"], cfg[1]) / best(sname, 0),
                     toks(P["None"]["4096/256"][0] / (NP * 256), cfg[1]) / best(sname, 4096), sl, n_star))
        T["profile_" + cfg[0]] = "\n".join(L)

    # T6: stale profile, the worst loss against the cold start, per base
    cross = _load(outd, "cross")
    for cfg in (("lowid", 76000, 49152, "lru"), ("cblind", 76000, 73952, "lru")):
        X = {r[2]: r[3] for r in cross if tuple(r[1]) == cfg}
        worst = (0.0, "", "")
        for xs_ in SETS:
            for ys_ in SETS:
                v = toks(X[xs_][ys_]["256/256"][0] / (2 * 25 * 256), cfg[1]) / best(ys_, 256) - r_warm(cfg, ys_, 256)
                if v < worst[0]:
                    worst = (v, xs_, ys_)
        gain = max(((toks(X[a][b]["256/256"][0] / (2 * 25 * 256), cfg[1]) / best(b, 256) - r_warm(cfg, b, 256)), a, b) for a in SETS for b in SETS if a != b)
        T["stale_" + cfg[0]] = "worst loss %.3f (profile of %s on %s); best cross-gain %.3f (profile of %s on %s)" % (worst[0], worst[1], worst[2], gain[0], gain[1], gain[2])

    # T7: seed spread of the ratio (seeds 7, 8, 9), the ruler of these tables
    sp = []
    for cfg in (("cblind", 76000, 73952, "lru"), ("chyb", 76000, 49152, "lru"), ("lowid", 76000, 49152, "lru"), ("both", 76000, 49152, "lru"), ("lowid", 65536, 32768, "lru"), ("chyb", 65536, 57344, "lru")):
        mx = 0.0
        for sname in SETS:
            for t in TS:
                vs = [r_warm(cfg, sname, t)]
                for sd in ("s8", "s9"):
                    c2, s2 = _idx(_load(f"{outd}/{sd}", "cold")), {r[1]: r[2] for r in _load(f"{outd}/{sd}", "static")}
                    b2 = max(toks(s2[sname]["C"][f"{t}/256"] / (NTR * 256), 76000), toks(s2[sname]["D"][f"{t}/256"] / (NTR * 256), 65536))
                    vs.append(toks(c2[(cfg, sname)][f"{t}/256"][0] / (NTR * 256), cfg[1]) / b2)
                mx = max(mx, max(vs) - min(vs))
        sp.append("%s: %.4f" % (":".join(map(str, cfg)), mx))
    T["seeds"] = "; ".join(sp)

    # T8: eviction at a small fill region (lowid R=65536) and the instrument check against Good-Turing
    L = ["| F | policy | zh/ja min at t=4096 | ko min at t=4096 | zh/ja min long 16384 | ko min long 16384 |", "|---|---|---|---|---|---|"]
    for B in (63488, 64512):
        for m in ("lru", "lfu"):
            cfg = ("lowid", 65536, B, m)
            z = [r_warm(cfg, x, 4096) for x in CLASS["zh/ja"]]; k = [r_warm(cfg, x, 4096) for x in CLASS["ko"]]
            zl = [toks(lng[(cfg, x)]["16384"] / 1024, 65536) / best(x, 0) for x in CLASS["zh/ja"]]; kl = [toks(lng[(cfg, x)]["16384"] / 1024, 65536) / best(x, 0) for x in CLASS["ko"]]
            L.append("| %d | %s | %.3f | %.3f | %.3f | %.3f |" % (65536 - B, m, min(z), min(k), min(zl), min(kl)))
    T["evict"] = "\n".join(L)

    sh = _load(outd, "shared"); sh2 = _load(outd, "shared2")
    L = ["| base | F | policy | pair | slot A q split -> shared at t=4096 | slot B q split -> shared at t=4096 |", "|---|---|---|---|---|---|"]
    for _, c, pair, o in sh + sh2:
        if c[3] != "lru":
            continue
        a, b = pair
        L.append("| %s | %d | %s | %s + %s | %.4f -> %.4f | %.4f -> %.4f |" % (c[0], c[1] - c[2], c[3], a, b,
                 o["split"][a]["4096/256"][0] / (2 * 40 * 256), o["shared"][a]["4096/256"][0] / (2 * 40 * 256),
                 o["split"][b]["4096/256"][0] / (2 * 40 * 256), o["shared"][b]["4096/256"][0] / (2 * 40 * 256)))
    T["shared"] = "\n".join(L)
    # T9: the paper prediction against the simulation (lowid R=76000 B=32K): the frozen set at t=0 and the online miss in the 256 ids after t=4096
    PRED0 = {"zh-wiki": .702, "zh-chat": .644, "ja-wiki": .582, "ja-chat": .571, "ko-wiki": .508, "ko-chat": .609, "en-prose": .021, "en-math": .010,
             "code-rust": .016, "code-python": .009, "code-c": .005}
    PRED4K = {"zh-wiki": .18, "zh-chat": .30, "ja-wiki": .165, "ja-chat": .23, "ko-wiki": .15, "ko-chat": .18, "en-prose": .027, "en-math": .009,
              "code-rust": .011, "code-python": .017, "code-c": .006}
    cfg = ("lowid", 76000, 32768, "lru")
    L = ["| set | predicted miss at t=0 | simulated, set frozen at t=0 | predicted miss after 4096 ids (Good-Turing, B=32K) | simulated, online, ids 4096..4352 | simulated, prompt-only (frozen at 4096) |", "|---|---|---|---|---|---|"]
    for sname in SETS:
        r0, r4 = cold[(cfg, sname)]["0/256"], cold[(cfg, sname)]["4096/256"]
        L.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f |" % (sname, PRED0[sname], r0[1] / (NTR * 256), PRED4K[sname], r4[0] / (NTR * 256), r4[1] / (NTR * 256)))
    T["prediction"] = "\n".join(L)

    for k_, v in T.items():
        open(f"{outd}/tables/{k_}.md", "w").write(v + "\n")


def report_kv(outd):
    """The KV-prefix restore arm: tables from kvturn.json."""
    rows = _load(outd, "kvturn")
    D = {(tuple(r[1]), r[2]): r[3] for r in rows}
    os.makedirs(f"{outd}/tables", exist_ok=True)
    T = {}

    def pooled(cfg, ss, nm, which, T1s, t2):
        m = n = cC = cD = 0
        for s in ss:
            d = D[(cfg, s)]; npos = d["trials"] * KN
            for T1 in T1s:
                m += d["res"][nm][str(T1)][str(t2)][which]
                cC += d["stat"][str(T1)][str(t2)][0]; cD += d["stat"][str(T1)][str(t2)][1]; n += npos
        return m, n, cC, cD

    def ratio(cfg, s, nm, which, T1s, t2):
        m, n, cC, cD = pooled(cfg, [s], nm, which, T1s, t2)
        return toks(m / n, cfg[1]) / max(toks(cC / n, 76000), toks(cD / n, 65536))

    ARMS = [("cold start: a new conversation, from the base", "base", 1), ("same-set profile start (upper bound: an oracle on the language)", "prof", 1),
            ("blended card-wide profile start (11 sets)", "blend", 1),
            ("restore, session began cold", "base", 0), ("restore, session began from the same-set profile (upper bound)", "prof", 0),
            ("restore, session began from the blended profile", "blend", 0)]
    for cfg in (("lowid", 76000, 49152, "lru"), ("cblind", 76000, 73952, "lru")):
        L = ["| arm | " + " | ".join(CLASS) + " |", "|---|" + "---|" * len(CLASS)]
        for lab, nm, which in ARMS:
            cells = []
            for c, ss in CLASS.items():
                vs = []
                for T1 in (256, 1024, 4096):
                    vs.append(min(ratio(cfg, s, nm, which, (T1,), 0) for s in ss))
                cells.append("/".join("%.3f" % v for v in vs))
            L.append("| %s | %s |" % (lab, " | ".join(cells)))
        T["kv_t0_" + cfg[0]] = "\n".join(L)
        L = ["| arm | " + " | ".join(CLASS) + " |", "|---|" + "---|" * len(CLASS)]    # T1 = 1024, t2 = 0 / 64 / 256 / 1024
        for lab, nm, which in ARMS:
            cells = []
            for c, ss in CLASS.items():
                cells.append("/".join("%.3f" % min(ratio(cfg, s, nm, which, (1024,), t2) for s in ss) for t2 in KT2))
            L.append("| %s | %s |" % (lab, " | ".join(cells)))
        T["kv_traj_" + cfg[0]] = "\n".join(L)
        L = ["| set | cold start | profile start | blend start | restore (cold) | restore (profile) | restore (blend) |", "|---|---|---|---|---|---|---|"]
        for s in SETS:                                                          # T1 = 1024, t2 = 0
            L.append("| %s | %s |" % (s, " | ".join("%.3f" % ratio(cfg, s, nm, which, (1024,), 0) for _, nm, which in ARMS)))
        T["kv_set_" + cfg[0]] = "\n".join(L)
        L = ["| set | q C | q D | " + " | ".join(("cold start", "profile start", "blend start", "restore (cold)", "restore (profile)", "restore (blend)")) + " |",
             "|---|---|---|---|---|---|---|---|---|"]
        for s in SETS:                                                          # T1 = 1024, t2 = 0: the next-256-id miss rate q
            m0, n0, cC, cD = pooled(cfg, [s], "base", 0, (1024,), 0)
            L.append("| %s | %.4f | %.4f | %s |" % (s, cC / n0, cD / n0, " | ".join("%.4f" % (pooled(cfg, [s], nm, which, (1024,), 0)[0] / n0) for _, nm, which in ARMS)))
        T["kv_q_" + cfg[0]] = "\n".join(L)
        # the per-set restore gain over its own cold start, T1 = 256 / 1024 / 4096, in points of tok/s
        L = ["| set | restore(cold) - cold start, T1 = 256 / 1024 / 4096 | restore(profile) - profile start | restore(blend) - blend start |", "|---|---|---|---|"]
        for s in SETS:
            def gain(nm):
                return "/".join("%+.1f" % (100 * (ratio(cfg, s, nm, 0, (T1,), 0) - ratio(cfg, s, nm, 1, (T1,), 0))) for T1 in (256, 1024, 4096))
            L.append("| %s | %s | %s | %s |" % (s, gain("base"), gain("prof"), gain("blend")))
        T["kv_gain_" + cfg[0]] = "\n".join(L)
    pe = sum(d["pure"][0] for d in D.values()); pn = sum(d["pure"][1] for d in D.values())
    ce = sum(d["cont"][0] for d in D.values()); cn = sum(d["cont"][1] for d in D.values())
    T["kv_cont"] = "%d of %d (trial, T1, t2, start) windows: the miss count of turn 2 on the restored set equals the same window of the stream that never stopped" % (ce, cn)
    T["kv_pure"] = "%d of %d (state, T1) pairs: the row set after turn 1 walked live equals the same ids folded in order (st bytes and LRU order)" % (pe, pn)
    for k_, v in T.items():
        open(f"{outd}/tables/{k_}.md", "w").write(v + "\n")


if __name__ == "__main__":
    step, outd = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else ".")
    init()
    cfs = [tuple(json.loads(x)) for x in sys.argv[3].split(";")] if len(sys.argv) > 3 else configs()
    name = sys.argv[4] if len(sys.argv) > 4 else step
    if step == "cold":
        pool_run(task_cold, [(c, s) for c in cfs for s in SETS], f"{outd}/{name}.json")
    elif step == "static":
        pool_run(task_static, SETS, f"{outd}/static.json")
    elif step == "profile":
        cf = cfs
        pool_run(task_profile, [(c, s, 50) for c in cf for s in SETS], f"{outd}/profile.json")
    elif step == "cross":
        cf = cfs
        pool_run(task_cross, [(c, x, 25) for c in cf for x in SETS], f"{outd}/cross.json")
    elif step == "shared":
        cf = cfs
        pairs = [("zh-chat", "ko-chat"), ("ja-wiki", "en-prose"), ("en-prose", "code-rust"), ("zh-wiki", "ja-wiki")]
        pool_run(task_shared, [(c, p, 40) for c in cf for p in pairs], f"{outd}/shared.json")
    elif step == "kvturn":
        pool_run(task_kvturn, [(c, s, 50) for c in cfs for s in SETS], f"{outd}/kvturn.json")
    elif step == "kvreport":
        report_kv(outd)
    elif step == "long":
        pool_run(task_long, [(c, s) for c in cfs for s in SETS], f"{outd}/{name}.json")
    elif step == "report":
        report(outd)
    else:
        raise SystemExit(f"unknown step {step}")
```

## Appendix B: commands that produced the tables

```sh
# every step reads and writes json under $OUT; HW_ROOT is a checkout holding commit 347a86e5 (the kohangul list)
S=headwarm.py; OUT=out; mkdir -p $OUT
python3 -I $S static  $OUT                                   # the two lists and the pick, same trials
python3 -I $S cold    $OUT                                   # 39 spec-grid arms x 11 sets (lowid, ec, both, chyb) -> cold.json
python3 -I $S long    $OUT                                   # the same arms along a whole half -> long.json
STRESS='["lowid",65536,64512,"lru"];["lowid",65536,64512,"lfu"];["lowid",65536,63488,"lru"];["lowid",65536,63488,"lfu"];["chyb",76000,75488,"lru"];["chyb",76000,75488,"lfu"];["chyb",76000,73952,"lru"];["chyb",76000,73952,"lfu"];["chyb",76000,70000,"lru"];["chyb",76000,70000,"lfu"]'
python3 -I $S cold $OUT "$STRESS" coldx;  python3 -I $S long $OUT "$STRESS" longx          # small writable regions
CH65='["chyb",65536,57344,"lru"];["chyb",65536,49152,"lru"];["chyb",65536,32768,"lru"];["chyb",65536,57344,"lfu"]'
python3 -I $S cold $OUT "$CH65" coldc65;  python3 -I $S long $OUT "$CH65" longc65          # cjk-76000 trimmed to 65,536
CBL='["cblind",76000,73952,"lru"];["cblind",76000,70000,"lru"]'
python3 -I $S cold $OUT "$CBL" coldcb;    python3 -I $S long $OUT "$CBL" longcb            # the claimed arm (no data) and a 6,000-row control
DBL='["dblind",65536,63488,"lru"];["dblind",65536,57344,"lru"]'
python3 -I $S cold $OUT "$DBL" colddb;    python3 -I $S long $OUT "$DBL" longdb            # pick + fill on the kohangul list
PR='["lowid",76000,49152,"lru"];["cblind",76000,73952,"lru"]'
python3 -I $S profile $OUT "$PR"; python3 -I $S cross $OUT "$PR"; python3 -I $S shared $OUT "$PR"
SH='["chyb",76000,73952,"lru"];["chyb",76000,73952,"lfu"];["lowid",65536,63488,"lru"];["lowid",65536,63488,"lfu"]'
python3 -I $S shared $OUT "$SH"; mv $OUT/shared.json $OUT/shared2.json; python3 -I $S shared $OUT "$PR"   # shared2.json = F=2048 stress
SEEDCF='["cblind",76000,73952,"lru"];["chyb",76000,49152,"lru"];["lowid",76000,49152,"lru"];["both",76000,49152,"lru"];["lowid",65536,32768,"lru"];["chyb",65536,57344,"lru"]'
for sd in 8 9; do mkdir -p $OUT/s$sd; HW_SEED=$sd python3 -I $S static $OUT/s$sd; HW_SEED=$sd python3 -I $S cold $OUT/s$sd "$SEEDCF" cold; done
python3 -I $S kvturn $OUT "$PR"                               # the KV-prefix restore arm (4.1): lowid and cblind x 11 sets -> kvturn.json
python3 -I $S report $OUT; python3 -I $S kvreport $OUT       # tables under $OUT/tables (the kv_* tables are kvreport's)
python3 -I kvcount.py; python3 -I kvtail.py                  # the paper count of 4.1 and the use-mass of the writable tail (Appendix C.2)
```

## Appendix C: the paper terms of §1 (run before the simulation)

```python
#!/usr/bin/env python3
"""The paper terms of headpick-warmup.md section 1: what a row set of the lowest ids covers, and the Good-Turing estimate
N1(t)/t of the ids outside the base that a prompt of t ids has not shown yet. `python3 -I paper.py` (HW_ROOT as in headwarm.py)."""
import os, random, subprocess
W = os.environ.get("HW_ROOT", ".")
SETS = ["zh-wiki", "zh-chat", "ja-wiki", "ja-chat", "ko-wiki", "ko-chat", "en-prose", "en-math", "code-rust", "code-python", "code-c"]
ids_of = lambda lines: [int(l) for l in (x.strip() for x in lines) if l and not l.startswith("#")]
Cs = set(ids_of(open(f"{W}/crates/model/data/mtp-head-rows-qwen38-cjk-76000.txt")))
Ds = set(ids_of(subprocess.check_output(["git", "-C", W, "show", "347a86e5:crates/model/data/mtp-head-rows-qwen38-kohangul-65536.txt"]).decode().splitlines()))
S = {s: ids_of(open(f"{W}/crates/models/tests/data/head-holdout/{s}.ids")) for s in SETS}
print("C&D", len(Cs & Ds), "C-D", len(Cs - Ds), "D-C", len(Ds - Cs), "| D holds ids 0..52380:", all(i in Ds for i in range(52381)))
print("%-12s %6s | miss of the lowest R ids, R = 32768 49152 65536 76000 | q_C q_D" % ("set", "types"))
for s in SETS:
    ids = S[s]; n = len(ids)
    print("%-12s %6d | %s | %.4f %.4f" % (s, len(set(ids)), " ".join("%.4f" % (sum(i >= R for i in ids) / n) for R in (32768, 49152, 65536, 76000)),
          sum(i not in Cs for i in ids) / n, sum(i not in Ds for i in ids) / n))
random.seed(7)
print("\nGood-Turing q(t) = N1_nonbase(t)/t, base = the lowest B ids, 100 offsets: t = 64 256 1024 4096")
for s in SETS:
    ids = S[s]; n = len(ids); out = []
    for B in (32768, 49152):
        row = []
        for t in (64, 256, 1024, 4096):
            acc = 0.0
            for _ in range(100):
                o = random.randrange(0, n - t - 1); cnt = {}
                for x in ids[o:o + t]:
                    if x >= B:
                        cnt[x] = cnt.get(x, 0) + 1
                acc += sum(1 for v in cnt.values() if v == 1) / t
            row.append(acc / 100)
        out.append(row)
    print("%-12s B32K: %s  B48K: %s" % (s, " ".join("%.3f" % x for x in out[0]), " ".join("%.3f" % x for x in out[1])))
```

Its output (`python3 -I paper.py`, the numbers behind the predicted columns of §3.7):

```
C&D 45829 C-D 30171 D-C 19707 | D holds ids 0..52380: True
set           types | miss of the lowest R ids, R = 32768 49152 65536 76000 | q_C q_D
zh-wiki        7101 | 0.7119 0.7068 0.7028 0.7017 | 0.0332 0.6692
zh-chat        9719 | 0.6526 0.6483 0.6461 0.6437 | 0.0327 0.5984
ja-wiki        5659 | 0.6488 0.6270 0.6004 0.5824 | 0.0064 0.5901
ja-chat        6406 | 0.6532 0.6255 0.5947 0.5709 | 0.0059 0.6006
ko-wiki        4877 | 0.6306 0.5914 0.5210 0.5075 | 0.0068 0.0195
ko-chat        4632 | 0.7367 0.6917 0.6247 0.6093 | 0.0031 0.0029
en-prose       5107 | 0.0987 0.0595 0.0358 0.0205 | 0.0178 0.0091
en-math        2119 | 0.0712 0.0334 0.0137 0.0099 | 0.0191 0.0071
code-rust      2591 | 0.0761 0.0482 0.0245 0.0162 | 0.0270 0.0132
code-python    3534 | 0.0574 0.0326 0.0189 0.0089 | 0.0321 0.0152
code-c         1782 | 0.0452 0.0261 0.0103 0.0049 | 0.0332 0.0154

Good-Turing q(t) = N1_nonbase(t)/t, base = the lowest B ids, 100 offsets: t = 64 256 1024 4096
zh-wiki      B32K: 0.511 0.372 0.261 0.183  B48K: 0.490 0.369 0.264 0.177
zh-chat      B32K: 0.479 0.449 0.395 0.304  B48K: 0.538 0.456 0.386 0.299
ja-wiki      B32K: 0.495 0.347 0.260 0.165  B48K: 0.493 0.364 0.248 0.166
ja-chat      B32K: 0.530 0.434 0.344 0.230  B48K: 0.498 0.418 0.335 0.225
ko-wiki      B32K: 0.472 0.354 0.234 0.152  B48K: 0.444 0.321 0.233 0.146
ko-chat      B32K: 0.562 0.434 0.312 0.177  B48K: 0.542 0.410 0.305 0.175
en-prose     B32K: 0.077 0.046 0.036 0.027  B48K: 0.040 0.030 0.021 0.015
en-math      B32K: 0.034 0.021 0.012 0.009  B48K: 0.020 0.012 0.006 0.005
code-rust    B32K: 0.055 0.033 0.022 0.011  B48K: 0.030 0.020 0.014 0.006
code-python  B32K: 0.049 0.037 0.027 0.017  B48K: 0.024 0.020 0.015 0.010
code-c       B32K: 0.020 0.016 0.009 0.006  B48K: 0.012 0.009 0.005 0.003
```

### C.2 The counts behind the blended-profile prediction and the tail's use-mass (`kvcount.py`, `kvtail.py`)

`kvcount.py` (run before the restore arm) compares the distinct ids the 11 training halves hold outside the base with the free
writable rows; `kvtail.py` (run after it) measures the share of each set's tokens that sit on `cblind`'s 2,048 writable rows.
Both `import headwarm as H` from the directory of Appendix A's script.

```python
# paper term for the blended global profile: how many distinct ids outside the base do the 11 training halves hold, against the
# writable rows. If the union fits, nothing is evicted and the blend equals every set's own profile; if not, LRU keeps the newest.
import sys
sys.path.insert(0, ".")
import headwarm as H
H.init()
for g in (0, 1):
    for cfg in (("lowid", 76000, 49152), ("cblind", 76000, 73952)):
        kind, R, B = cfg
        st0 = H.state0(kind, g, R, B)
        seen = set()
        per = {}
        for s in H.SETS:
            u = {x for x in H.half_ids(s, g) if st0[x] != 1}     # the base is protected: never a write
            per[s] = u
            seen |= u
        asked_filler = {x for x in seen if st0[x] == 2}
        out = {x for x in seen if st0[x] == 0}
        F = R - B
        free = F - len(asked_filler)
        print("g=%d %s F=%d union non-base=%d  of which filler rows asked for=%d, misses (take a free filler row)=%d, free filler=%d  overflow=%d"
              % (g, kind, F, len(seen), len(asked_filler), len(out), free, max(0, len(out) - free)))
        if g == 0:
            print("   per-set misses:", {s: sum(1 for x in per[s] if st0[x] == 0) for s in H.SETS})
```

Output of `kvcount.py`:

```
g=0 lowid F=26848 union non-base=17732  of which filler rows asked for=970, misses (take a free filler row)=16762, free filler=25878  overflow=0
   per-set misses: {'zh-wiki': 3513, 'zh-chat': 5287, 'ja-wiki': 3453, 'ja-chat': 3967, 'ko-wiki': 2577, 'ko-chat': 3082, 'en-prose': 120, 'en-math': 42, 'code-rust': 64, 'code-python': 66, 'code-c': 28}
g=0 cblind F=2048 union non-base=2747  of which filler rows asked for=864, misses (take a free filler row)=1883, free filler=1184  overflow=699
   per-set misses: {'zh-wiki': 268, 'zh-chat': 601, 'ja-wiki': 58, 'ja-chat': 96, 'ko-wiki': 76, 'ko-chat': 43, 'en-prose': 200, 'en-math': 126, 'code-rust': 115, 'code-python': 220, 'code-c': 114}
g=1 lowid F=26848 union non-base=18259  of which filler rows asked for=1018, misses (take a free filler row)=17241, free filler=25830  overflow=0
g=1 cblind F=2048 union non-base=2981  of which filler rows asked for=818, misses (take a free filler row)=2163, free filler=1230  overflow=933
```

```python
# the use-mass of cblind's writable tail: which share of each set's tokens sit on the 2,048 rows a blended profile can overwrite
import sys
sys.path.insert(0, ".")
import headwarm as H
H.init()
r = H.ranking("cblind", 0)
tail = set(r[73952:76000])
print("tail ids: min %d max %d  (control block from %d is outside: %s)" % (min(tail), max(tail), H.CONTROL_FROM, all(i < H.CONTROL_FROM for i in tail)))
for s in H.SETS:
    ids = H.S[s]
    print("%-12s share of tokens on the tail rows: %.4f   distinct tail ids used: %d" % (s, sum(1 for x in ids if x in tail) / len(ids), len({x for x in ids if x in tail})))
```

Output of `kvtail.py`:

```
tail ids: min 237230 max 248042  (control block from 248044 is outside: True)
zh-wiki      share of tokens on the tail rows: 0.0034   distinct tail ids used: 38
zh-chat      share of tokens on the tail rows: 0.0000   distinct tail ids used: 0
ja-wiki      share of tokens on the tail rows: 0.0494   distinct tail ids used: 536
ja-chat      share of tokens on the tail rows: 0.0425   distinct tail ids used: 632
ko-wiki      share of tokens on the tail rows: 0.0100   distinct tail ids used: 178
ko-chat      share of tokens on the tail rows: 0.0129   distinct tail ids used: 210
en-prose     share of tokens on the tail rows: 0.0000   distinct tail ids used: 0
en-math      share of tokens on the tail rows: 0.0000   distinct tail ids used: 0
code-rust    share of tokens on the tail rows: 0.0000   distinct tail ids used: 0
code-python  share of tokens on the tail rows: 0.0000   distinct tail ids used: 0
code-c       share of tokens on the tail rows: 0.0000   distinct tail ids used: 0
```

### C.3 The prediction card of the restore arm (`kvpredict.txt`, written before `kvturn` ran)

```
kvturn prediction card (written before the kvturn step ran; Mac simulation, no lease)

The one question: the miss rate of turn 2 at tokens-seen t2 = 0 when turn 2 restores turn 1's row set (KV-prefix hit), next to
(a) a cold neutral start and (b) the global-profile start, as tok/s against the best static list (ratio, class minimum over the sets).

Derivation (no run needed for the t2 = 0 column):
  the row set is a fold of the id sequence in order: state(restore after T1 ids) = fold(start, turn-1 ids) = the state of an
  uninterrupted stream at t = T1. So  ratio_restore(T1, t2 = 0, start)  =  ratio_cold(t = T1, start)  of the existing grid
  (cold.json / profile.json columns t = 256, 1024, 4096 are T1 = 256, 1024, 4096). Turn 2 sits at "consecutive ids of the same
  set", so the grid's trials are exactly the two-turn trials.

Predicted ratio at t2 = 0 (class minimum over the sets), read from the existing tables (families.md, profile_dir_*.md):
                         neutral base (lowid R=76000 B=48K)                 cjk-76000 + 2048 spare rows (cblind)
                         cold start | restore T1=256 / 1024 / 4096          cold start | restore T1=256 / 1024 / 4096
  en                     .990       | .994 / .997 / .999                     .989       | .992 / .993 / .994
  code                   1.001      | 1.004 / 1.008 / 1.010                  .983       | .991 / .996 / 1.000
  zh/ja                  .509       | .526 / .569 / .659                     1.001      | 1.001 / 1.001 / 1.000
  ko                     .519       | .567 / .630 / .749                     .998       | .998 / .998 / .998
  global-profile start (same set, whole other half, n = 20K ids; min over sets and both directions) at t2 = 0:
  lowid:  en .995  code 1.007  zh/ja .777  ko .869        cblind:  en .989  code .989  zh/ja .998  ko .997
  restore of a session that STARTED from that profile (profile -> T1 -> restore), read at t = 256 / 1024 / 4096 of profile.json:
  lowid:  zh/ja .807 / .778 / .819   ko .888 / .882 / .890   (within the 0.016 seed spread of the profile start: no gain predicted)
  cblind: en .993 / .993 / .995   code .997 / 1.000 / 1.001   zh/ja 1.000 / .996 / .998   ko .997 / .998 / .999
So the ordering predicted on the neutral base, zh/ja/ko, at every T1 <= 4096: cold < restore(cold-started) < global profile
(same-set, 20K ids) <= restore(profile-started); the restore passes the 20K-id profile only past T1 ~ 16K (long run 16,384: .82-.92).
On cblind all arms sit inside ~1 point of each other: the restore adds what the fill adds (en/code +0.4 to +1.6 points over the cold start
by T1 = 1024), nothing on zh/ja/ko.

The blended (card-wide) profile is the one term the table cannot give; the count (kvcount.out) fixes its regime:
  lowid: the 11 training halves hold 16,762-17,241 distinct ids outside the base+filler against 25,830-25,878 free filler rows:
         NO eviction, so the blend holds every set's ids at once and equals each set's own profile (predicted ratio: the
         same-set profile's, +-0.02 seed spread) -- the blend costs nothing on the neutral base.
  cblind: 1,883-2,163 missed ids against 1,184-1,230 free rows: 0.7-0.9K ids overflow, LRU keeps the newest; predicted blend gain over
         the cold start <= the same-set profile's (+0 to +0.8 points on en/code at t2 = 0), 0 on zh/ja/ko.

Decision table (what each outcome does to the design):
  - restore == grid reading to the count (identity holds) and refold-from-ids == stored state in every trial: the head row set needs
    no stored copy; the prompt cache's ids are the key AND the data (fold on admission). If the identity fails by more than 0
    counts, state is order-dependent in a way the id list does not carry and the set must be stored with the saved state.
  - blended profile >= same-set profile - 0.02 on the neutral base: a card-wide head section is not harmful there, but it is a 17K-id
    list, i.e. what the shipped list already is; no head section in the card-wide file. If the blend loses more than 0.03 to the
    same-set profile on some class, the user's "rows never blend across streams" is a measured rule, not a preference.
```
