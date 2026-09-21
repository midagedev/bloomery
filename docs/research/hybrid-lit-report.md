# CPU–GPU co-execution for MoE decode: a formal model, the literature, and what survives at batch 1

Investigation round, 2026-09-21. Subject: the *academic* literature and the theory of hybrid
CPU/GPU MoE inference, applied to DeepSeek-V4.1-Flash on one Threadripper 5975WX + RTX 3090/A6000
box. A sibling round reads engine source; this one does not.

Throughout, every claim carries a provenance tag:

- **[M-box]** measured on our machine and recorded in our docs (`bloomery/docs/*`, rig-log).
- **[M-paper]** measured by a paper's authors; I read the number in the paper's text/table (cited with location).
- **[P]** proven/derived analytically by the cited authors.
- **[D]** derived here by me from [M-box] numbers, by arithmetic only.
- **[I]** my inference/judgement — not measured, not proven.

---

## 1. Abstract

A decode step of DeepSeek-V4.1-Flash on this box is a chain of 40 two-processor fork-join gadgets, not
a set of independent jobs, so flow-shop scheduling does not apply and the critical path is the sum of
the chain. With our measured numbers — host term 22.6 ms, GPU term 33.9 ms, ratio r = 0.666 — the
work bound gives a maximum overlap speedup of 1 + min(r, 1/r) = **1.67×**, but that is a *throughput*
number requiring ≥ 2 independent sequences. For a single stream with today's block-granular placement
the exact ceiling is **1.04×**; splitting each expert's rows across both processors as a divisible
load raises it to **1.15×** at our VRAM capacity (1.37× uncapped). On-demand expert transfer loses to
host compute by exactly β_host/β_PCIe = **5.6×**, and only wins above ≈ **800 concurrent tokens** —
which is why MoE-Lightning and MoE-Lens (batch 20k–200k) stream weights while Fiddler, KTransformers
and we compute on the host at batch 1. Expert-ID prediction cannot overlap *compute*, only prefetch,
and our weights are already in DRAM. KTransformers independently measures our central obstacle
(shared experts are 18 % of GPU time) and buys past it with an output-changing Expert Deferral
(+33 %, ≤ 0.5 % accuracy). Speculative verify batches cannot amortise a 384-of-6 router:
E[distinct experts] = 23.5 at k = 4.

---

## 2. A formal model of the hybrid decode step, and its bounds

### 2.1 The machine and the model, as numbers

All of the following are [M-box] unless marked. Sources are `bloomery/docs/roofline.md`,
`docs/plan.md`, `docs/research/v41-ops.md`, `docs/research/v41-ports.md`.

| quantity | symbol | value | provenance |
|---|---|---:|---|
| host read bandwidth, 32 threads | β_h | 147.7 GB/s | [M-box] rig-log; contested against 215.6, WKS-35 open |
| GPU *effective* bandwidth over a whole decode pass | β_g | 247 GB/s | [D] roofline.md:77 — 8.366 GB ÷ 33.9 ms; folds in attention math, launches, sync |
| 3090 STREAM-style ceiling | — | ~760 GB/s | [D] 81 % of 936 spec; never measured on this box |
| PCIe 4.0 x16 H2D, pinned | β_link | 26.2 GB/s | [M-box] v41-ports.md:38 |
| PCIe 4.0 x16 H2D, pageable | — | 14.4 GB/s | [M-box] v41-ports.md:38 |
| PCIe/launch latency α | α | **not measured** | see §6-H5 |
| model dim | d | 5120 | [M-box] v41-ops.md (`wkv: 5120→512`) |
| layers, all MoE | L | 40 | [M-box] v41-ops.md:27 |
| routed experts / top-k / shared | N / k / — | 384 / 6 / 1 | [M-box] v41-ops.md:27 |
| routed-expert bytes per token | B_E | 3.7656 GiB = 4.044 GB | [M-box] roofline.md:20, re-counted by `gguf-inventory` |
| dense bytes per token (attn+shared+router+head, embd excluded) | B_D | 7.1320 GiB = 7.658 GB | [M-box] roofline.md:20,27 |
| engram rows per token | — | 48 rows × 272 B = 12.75 KiB | [D] v41-ops.md:63-65 |
| total bytes per token | B | 10.8976 GiB = 11.701 GB | [M-box] roofline.md:23 |
| 2026-09-16 placement: host blocks / GPU blocks | L_h / L_g | 33 / 7 | [M-box] roofline.md:50 |
| host routed bytes | C_B | 3.336 GB | [M-box] roofline.md:55 |
| GPU routed bytes | — | 0.708 GB | [M-box] roofline.md:56 |
| measured pass time, draft OFF | T_pass | 56.5 ms (17.7 tok/s) | [M-box] roofline.md:74, `configs/v41-serve.sh:63` |
| same, WKS-36 arm "no draft" | — | 50.21 ms | [M-box] roofline.md:209 |

Two derived quantities do the work in everything below:

- **Host term** `C = C_B / β_h = 3.336 / 147.7 = 22.58 ms` [D] — 40 % of the pass (roofline.md:83).
- **GPU term** `G = T_pass − C = 33.9 ms` [D], which is where β_g = 247 GB/s comes from. G is
  therefore *defined* to make the serial decomposition exact; it is not an independent measurement.
  Everything that depends on G inherits that circularity, and I flag it each time.

**Per-layer constants** [D], used throughout:

| per layer | host block (33 of them) | GPU block (7 of them) |
|---|---:|---:|
| routed-expert bytes | 0.1011 GB | 0.1011 GB |
| host expert time `c` | 0.1011/147.7 = **0.684 ms** | 0 |
| GPU dense time `D` | 7.658/40/247 = **0.775 ms** | 0.775 ms |
| GPU expert time | 0 | 0.1011/247 = 0.409 ms |
| **layer total (serial)** | **1.459 ms** | 1.184 ms |

Check: 33 × 1.459 + 7 × 1.184 = 48.15 + 8.29 = **56.4 ms** ≈ T_pass. The decomposition closes.

> **Correction to the round's own spec.** The spec quotes "~0.20 ms per layer per token" for host
> experts. That figure is from ik on **Qwen3.6** (`plan.md:247`), a much smaller expert. For V4.1 the
> host cost is **0.684 ms/layer** [D]. Using 0.20 × 40 = 8 ms would understate the host term by 2.8×
> and would make every conclusion below wrong in the optimistic direction.

**Per-expert size** [D]: `B_E / (L · k) = 4.044e9 / 240 = 16.85 MB` per (layer, expert), gate+up+down
at the serving quantization.

### 2.2 The decode step as a DAG over two processors and a link

Let one decode step for one token be a DAG `Γ = (V, E)`. Vertices are typed by processor:
`g` (GPU), `h` (host), `x` (link transfer). For layer ℓ ∈ {0..39} the sub-DAG is, reading
`v41-ops.md:38-70` for V4.1's actual shape:

```
  s_ℓ  : hyper-connection fold  (4 streams × d → d)                     [g]
  n_ℓ  : RMSNorm + quantize                                             [g]
  a_ℓ  : attention  = {q low-rank + norm, wkv 5120→512, rope,
                       indexer (layers 2,8,14,20,24,28,32,36),
                       sparse attn over ≤640 rows, inverse-rope,
                       grouped wo_a·wo_b}                               [g]
  r_ℓ  : router gemv 5120→384, √softplus + bias, top-6, renorm, ×1.5    [g]
  e_ℓ,S: shared expert  (SwiGLU, clamp ±10)                             [g]
  e_ℓ,j: routed expert j ∈ top-6,  partitioned  J_g ⊎ J_h               [g] / [h]
  m_ℓ  : weighted combine + hyper-connection residual mix (Sinkhorn 20) [g]
  X↓_ℓ : GPU→host activation                                           [x]
  X↑_ℓ : host→GPU partial sums                                          [x]
```

with edges `s_ℓ → n_ℓ → a_ℓ → (n'_ℓ) → r_ℓ → {e_ℓ,·} → m_ℓ → s_{ℓ+1}`, plus
`r_ℓ → X↓_ℓ → {e_ℓ,j : j ∈ J_h} → X↑_ℓ → m_ℓ`, and two engram sites at ℓ = 1, 14 that add to
the stream before attention.

Costs. Processor speeds are **roofline terms** (Williams, Waterman & Patterson, CACM 2009) —
at batch 1 every gemv is a gemv with arithmetic intensity 4.65 flop/byte for Q3_K [D]
(roofline.md:116), far below both machine balance points (host 24.9 flop/byte at 147.7 GB/s;
3090 46.8 at 760) — so **cost = bytes touched ÷ bandwidth** on both sides and neither side is
compute-bound at m = 1. (This is the one place a classical model applies without qualification.)

Link cost uses the **α–β (postal/Hockney) model** (Hockney, *Parallel Computing* 20(3), 1994):
`t(n) = α + n/β_link`. For the finer question of *who pays* — the host CPU cannot do expert
arithmetic while it is posting a descriptor — the **LogP model** (Culler et al., PPoPP '93)
separates latency `L`, per-message overhead `o`, gap `g`. Here `n` is tiny and the relevant
term is `o` and `L`, not `n/β_link`:

- `X↓_ℓ`: one activation vector, d = 5120. f32 20.48 KB → 0.78 µs of wire time; q8 ≈ 5.6 KB
  → 0.21 µs [D].
- `X↑_ℓ`: one combined d-vector back, same order.
- So per host layer the *bytes* cost ≈ 1.6 µs, and the per-crossing constant 2(α+o) dominates.
  Over 33 host layers: **33 × 2 × α**. At a plausible α+o ∈ [5, 10] µs [I], that is 0.33–0.66 ms,
  i.e. 0.6–1.2 % of the pass. **α is not measured on this box** — H5 in §6.

> **Finding 1 [D].** In this regime the link is *not* a bandwidth problem and *not* (at these α)
> a latency problem either. The host boundary costs ~1 % of the step. Every paper whose headline
> is "we moved less over PCIe" is optimising a term that is already ~1 % for us. What costs 40 %
> is that the host *reads its own DRAM slowly*.

### 2.3 Makespan lower bounds

Write per layer: `c_ℓ` (host expert time), `D_ℓ` (GPU work on the chain before/after the expert
stage), `s_ℓ` (GPU work that is *concurrent-eligible* with the host experts — the shared expert,
plus any GPU-resident routed experts of the *same* layer).

Because `s_ℓ → n_ℓ → a_ℓ → r_ℓ → e_ℓ → m_ℓ → s_{ℓ+1}` is a **chain**, the DAG is a chain of
2-processor "fork-join" gadgets. Three bounds, in increasing strength:

**(B1) Work bound (per processor).**
`T ≥ max(Σ_ℓ c_ℓ, Σ_ℓ (D_ℓ + s_ℓ))` — each processor must at least do its own work.
For us: `max(22.58, 33.9) = 33.9 ms` [D]. This is the bound people quote as "1.67×"
(56.5/33.9 = 1.667, roofline.md:149).

**(B2) Critical path.** The chain forbids overlapping layer ℓ's host work with layer ℓ′ ≠ ℓ's GPU
work *for the same token*. Hence

```
  CP = Σ_ℓ [ D_ℓ + max(c_ℓ, s_ℓ) ]                                        (1)
```

and `T ≥ CP`. **(B2) is the real single-token bound; (B1) is not attainable for one sequence.**

**(B3) With the link.** Add `2(α+o)` per host layer, inside the `max` if the host cannot post and
compute simultaneously, outside if it can:
`CP' = Σ_ℓ [ D_ℓ + 2(α+o)·1{c_ℓ>0} + max(c_ℓ, s_ℓ) ]`.

### 2.4 The Amdahl-type bound and the ratio r

Define `r = C/G = Σc_ℓ / Σ(D_ℓ+s_ℓ)`. For the **work bound (B1)** the speedup of perfect
two-processor overlap over the fully serial schedule is

```
  S_work(r) = (C+G)/max(C,G) = (1+r)/max(1,r) = 1 + min(r, 1/r)           (2)
```

[P, trivially] — a strictly concave, symmetric function peaking at **S = 2 when r = 1**, falling to
1 as r → 0 or ∞. This *is* Amdahl's law (Amdahl, AFIPS 1967) with the serial fraction being
whichever processor is idle: `S = 1/(1 − φ)` where `φ = min(C,G)/(C+G)` is the overlappable
fraction of serial time. For us **r = 22.58/33.9 = 0.666**, so

> **S_work ≤ 1.666.** [D] And by (2), *no rebalancing of the current placement can exceed 2×*,
> ever, however clever the schedule.

Gustafson's law (CACM 1988) does **not** apply and I do not invoke it: our problem size per step is
fixed by the model, not scaled with processors.

**Rebalancing to r = 1.** r is a *choice*: it is set by how many expert bytes live on the host.
Let `x` GB of routed-expert bytes be host-resident per token (0 ≤ x ≤ B_E = 4.044). Then
`C(x) = x/β_h`, `G(x) = (B_D + B_E − x)/β_g`, and S_work is maximised where they meet:
`x* = β_h(B_D+B_E)/(β_h+β_g) = 147.7 × 11.702 / 394.7 = 4.379 GB` [D] — which exceeds B_E.
So **within this model the balanced point is unreachable from below: the optimum under the work
bound is to put *every* routed expert on the host**, giving
`C = 4.044/147.7 = 27.38 ms`, `G = 7.658/247 = 31.00 ms`, `r = 0.883`, `S_work = 1.883`,
makespan `31.0 ms` = **1.82× the present 56.5 ms** [D].

That is a counter-intuitive and falsifiable consequence: *under a throughput regime, moving experts
off the GPU makes the machine faster*, because it buys GPU time, which is the scarce processor.
(It is also sensitive to β_g = 247 being derived; see §5.4.)

### 2.5 Plugging in (B2): what a single stream can actually get

The whole question is the size of `s_ℓ` — GPU work of the *same layer* that does not depend on the
host experts. From `v41-ops.md`, exactly two things qualify:

1. **The shared expert** `e_ℓ,S`. Same SwiGLU shape as a routed expert [I, from v41-ops.md:35 "expert
   SwiGLU shape" shared with V2-Lite and "shared expert added without a weight"] ⇒ ≈16.85 MB ⇒
   `16.85 MB / 247 GB/s = 0.068 ms` [D].
2. **GPU-resident routed experts of the same layer** — zero in the current *block-granular*
   placement, because a block is either all-host or all-GPU (roofline.md:50). This is the single
   most consequential accident of the current layout, and §2.6 is about undoing it.

So today, with block-granular placement:

```
  CP = 33·[0.775 + max(0.684, 0.068)] + 7·[0.775 + max(0, 0.409)]
     = 33·1.459 + 7·1.184  = 56.4 ms          — i.e. the overlap buys nothing,
```
because the shared expert is already inside the 0.775 ms dense term. Counting it out of the dense
term instead: `33·[0.707 + 0.684] + 7·1.184 = 45.9 + 8.3 = 54.2 ms` → **1.04×** [D].

> **Finding 2 [D].** For one sequence, with block-granular placement, *intra-layer* CPU/GPU overlap
> is worth **≈ 4 %**, not 67 %. The 1.67× in `roofline.md:149` is the work bound (B1); the doc
> already says (roofline.md:150-154) it needs ≥ 2 independent sequences and is throughput, not
> latency. This report's contribution is to say *how much* the latency-side number actually is,
> and that it is 0.04, not 0.67.

### 2.6 The exact relaxation nobody needs prediction for: splitting an expert's rows

`s_ℓ = 0.068 ms` only because placement is block-granular. But an expert's `down`/`gate`/`up`
matrices are row-partitionable: rows `[0,m)` on the host, `[m, n)` on the GPU, partial products
summed. This is **exact up to floating-point summation order** (a band gate, not a bit gate — same
class as our own `q_nope2` cell split, `plan.md:117`), needs no prediction, and makes the expert
stage a **divisible load** (Bharadwaj, Ghose, Mani & Robertazzi, *Scheduling Divisible Loads in
Parallel and Distributed Systems*, IEEE CS Press 1996 [P] — optimal single-installment split of a
divisible load across heterogeneous processors equalises finishing times).

Per layer, let `E = 0.1011 GB` of routed bytes be split `x` host / `E−x` GPU. Equalising:

```
  x* = E·β_h/(β_h+β_g) = 0.1011 × 147.7/394.7 = 0.0378 GB   (37.4 % host)
  t*  = x*/β_h = 0.2561 ms        (vs 0.684 all-host, 0.409 all-GPU)        (3)
```

Unconstrained, `CP = 40 × (0.775 + 0.256) = 41.3 ms` → **1.37×**, exact, single stream [D].

**But capacity binds.** Total routed-expert bytes in the file = `B_E × N/k = 3.7656 GiB × 64
= 241 GiB` [D] (consistent with `plan.md:55`: 249 GiB of weights = 241 experts + 7.13 dense +
1.23 embd). VRAM = 24 + 48 = 72 GB = 67 GiB; subtract dense ~8.4 GiB and workspace/KV ~3 GiB [I]
⇒ ≈ 55.6 GiB for experts ⇒ **γ = 55.6/241 = 23.1 %** of expert bytes can be GPU-resident [D].
With routing near-uniform (§3), γ is also the per-token GPU share. Then per layer
`x = 0.769·E = 0.0777 GB → 0.526 ms` host, `0.0234 GB → 0.095 ms` GPU, and

```
  CP = 40 × [0.707 + max(0.526, 0.095 + 0.068)] = 40 × 1.233 = 49.3 ms  → 1.15×    (4)
```

> **Finding 3 [D].** The **exact single-stream ceiling** for this machine, at 147.7 GB/s host and
> 72 GB of VRAM, is **≈ 1.15×** (49.3 ms vs 56.5 ms), obtained by row-splitting every expert across
> both processors at the capacity-limited ratio. It is capacity, not scheduling, that stops us at
> 1.15 instead of 1.37. Freeing VRAM (a smaller engram footprint, a cheaper KV, offloading dense
> head weights) converts directly into speedup along (3)–(4).

And the two-stream number, for completeness: with ≥ 2 independent sequences, (B1) applies and the
optimum is the *all-host* placement of §2.4 — **31.0 ms/token, 1.82× throughput** [D], at the cost
of latency per stream.

### 2.7 Why exact cross-layer pipelining is impossible, and what staleness buys

The residual dependency: `e_ℓ` needs `x_ℓ`, which needs `m_{ℓ-1}`, which needs `e_{ℓ-1}` — including
the host-resident subset. So for a single sequence there is **no** schedule in which the host
computes layer ℓ+1's experts while the GPU computes layer ℓ's anything. Formally, the chain DAG's
critical path equals (1) and no reordering shortens it. Consequently:

- **Two-machine flow-shop scheduling (Johnson's rule, *Naval Res. Log. Q.* 1954) does not apply** and
  I do not cite it as if it did: Johnson assumes *n independent jobs* each visiting both machines.
  Our layers are a chain, not independent jobs. The moment there are ≥ 2 sequences, Johnson-type
  two-machine reasoning *does* apply to the token-pair schedule, and the bound it gives is (B1).
- **Communication-delay scheduling** (Papadimitriou & Yannakakis, *SIAM J. Comput.* 19(2), 1990 [P] —
  P|prec, c|C_max is NP-hard, and they give a 2-approximation for unit-time/unit-delay) is the right
  classical frame for (B3), but with a chain the problem is trivial and the theory adds nothing
  beyond "the chain is the answer". I mention it to say it does not help, not to name-drop it.

**What relaxation buys, precisely.** Three distinct relaxations, with different exactness:

| relaxation | what it lets you start early | exact? | gain in this model |
|---|---|---|---|
| R1. Predict expert **IDs** for layer ℓ+1 | *prefetching weights* only (into VRAM, or into L3/L2) | **exact** | 0 bandwidth. Only helps if the host is latency- or capacity-bound, not bandwidth-bound. Our host is bandwidth-bound ⇒ **≈ 0**; a wrong prefetch *costs* bandwidth. |
| R2. Compute e_ℓ from a **stale/predicted** x̂_ℓ | the host expert compute itself | **output-changing** | up to full overlap: CP → Σ D_ℓ = 31.0 ms → **1.82×**. The entire gap between 1.15× and 1.82× is paid for in output fidelity. |
| R3. Hoist **input-independent** sub-work | see below | **exact** | small but free |

R3 deserves a line because V4.1 uniquely offers two instances:

- **Hyper-connections are consumed one sublayer late** (`v41-ops.md:48`: "믹스는 한 서브층 늦게
  소비된다"; V4 consumed in the same sublayer). The `comb` matrix for sublayer s+1 is built from
  sublayer s's input by a 24-dim gemv + sigmoid + 20 Sinkhorn iterations. Sinkhorn-20 on a 4×4
  matrix is latency, not bytes. **This lag lets the mix-matrix computation be hoisted one sublayer
  off the critical path** [I, read from v41-ops.md:46-49]. It does **not** relax the *data* path: the
  residual streams still must be updated by sublayer s's output. So it is a real but small exact win,
  and it is *not* the cross-layer overlap the spec hoped for. I checked the doc specifically for
  this and the answer is: the lag is on the mixing coefficients, not on the activations.
- **engram row IDs for token t+1 are computable the moment token t is sampled** (`v41-ops.md:69-70`:
  only the previous 3 tokens are needed). So the 48 scattered 272 B NVMe reads have an entire step of
  slack and belong off the critical path by construction. Our ports' own numbers bound the prize:
  major faults 41–62/token → 13.6 tok/s, with prefetch 1–11 faults → 18.4 tok/s, fully cached
  18.8–19.0 [M-box, v41-ports.md:12-13] ⇒ **≈ 19.2 ms/token, ≈ 380 µs per major fault** [D].
  That is a *larger* term than everything in §2.5-2.6 and it is already solved in principle.

### 2.8 Break-even: transfer the expert, or compute it on the host?

This is the single question on which the literature splits, and in this model it has a one-line
answer. Per (layer, expert) of size `w = 16.85 MB`:

```
  transfer-and-compute-on-GPU:  w/β_link + w/β_g = 0.643 + 0.068 = 0.711 ms
  compute-on-host:              w/β_h            = 0.114 ms
  ratio  ≈  β_h/β_link = 147.7/26.2 = 5.64×                                  (5)
```

> **Finding 4 [D].** At batch 1, **on-demand expert transfer is 5.6× worse than host compute on this
> box, and the factor is exactly `β_h/β_link` — independent of expert size, quantization, and model.**
> It does not improve with batch: per-step transfer and per-step host compute both amortise the same
> way over tokens that share an expert, and §3 shows tokens share almost nothing.

Transfer only wins when the transferred expert is **cached and reused across steps**. Payback:
a promotion costs `w/β_link = 0.643 ms`; each subsequent *hit* saves `w/β_h − w/β_g = 0.114 −
0.068 = 0.046 ms`; so a promoted expert must be hit **14 times** to pay back [D]. At a per-step
activation probability of `k/N = 6/384 = 1.5625 %`, that is an expected residency of
`14/0.015625 ≈ 900 tokens` [D] before break-even. (With β_g at a more honest 700 GB/s the numbers
are 7 hits, ≈ 450 tokens.) This is why every expert-cache design in the literature has an
*admission* policy rather than promoting on first miss — and it is exactly what our own port
measured: admission on the **second miss inside a 64-step window**, promotions 160 → 113 per step
(`v41-ports.md:36`), with the added [M-box] lesson that pageable-memory promotion serialises against
the critical-path input copy in the driver staging lock and cost ~27 ms/step until it was pinned.

### 2.9 Summary of §2 — the numbers the rest of the report is judged against

**Every bound below is keyed to β_h, which is unresolved (WKS-35) and which §5.5-1 argues is closer
to 215–233 than to 147.7.** I give both branches; treat the pair as the honest answer until WKS-35
closes.

| bound | **β_h = 147.7** (r = 0.666) | **β_h = 233** (r = 0.40) | regime | exact? |
|---|---:|---:|---|---|
| present serial pass | 56.5 ms (17.7 tok/s) | 50.2 ms | 1 stream, block-granular | — |
| intra-layer overlap, block-granular (B2) | 54.2 ms, **1.04×** | ~49.3 ms, **1.02×** | 1 stream | exact |
| + row-split experts, capacity-limited γ=23 % (4) | 49.3 ms, **1.15×** | ~47.4 ms, **1.06×** | 1 stream | exact (fp order) |
| + row-split, capacity-unconstrained (3) | 41.3 ms, **1.37×** | ~44 ms, **1.14×** | 1 stream | exact (fp order) |
| work bound `1+min(r,1/r)`, present placement (B1) | 33.9 ms, **1.67×** | 35.9 ms, **1.40×** | ≥ 2 streams | exact |
| work bound, all-experts-on-host | 31.0 ms, **1.82×** | ~33 ms, **1.52×** | ≥ 2 streams | exact |
| absolute ceiling of 2-processor overlap, any r | **2.00×** | **2.00×** | any | — |
| stale-input cross-layer pipelining (R2) | 1.82× | 1.52× | 1 stream | **output-changing** |
| engram fault removal (already planned) | ~19.2 ms/token | ~19.2 ms/token | 1 stream | exact |

The β_h = 233 column is [D] and coarser (it re-uses the same per-layer structure with C = 14.3 ms,
G = 35.9 ms); its purpose is to show that **the ordering of the levers does not change, only their
size** — and that the single-stream prize shrinks toward 1.05× if the host is faster than we thought.

Answer to the spec's question about the user's intuition — *"batching/chunking GPU and CPU work
together must raise speed"*: **yes, and by at most 1 + min(r, 1/r) = 1.67× at today's placement,
2.00× at any placement; but for a single stream the exact, prediction-free ceiling is 1.15×, and
the 1.67 is a throughput number requiring ≥ 2 independent sequences.**

---

## 3. Placement and caching: the optimisation problem, and what is measured about routing

### 3.1 The static problem is a knapsack that collapses to "sort by p"

Let `S` be the set of (layer, expert) pairs, `|S| = L·N = 40 × 384 = 15 360`. Each has size
`w = 16.85 MB` (identical — MoE experts within a model are the same shape) and an activation
probability `p_{ℓ,e}` per decode step. Let `x_{ℓ,e} ∈ {0,1}` be "GPU-resident". Expected per-token
cost is

```
  minimise  Σ_{ℓ,e} p_{ℓ,e} · w · [ x·(1/β_g) + (1−x)·(1/β_h) ]
  s.t.      Σ x_{ℓ,e} · w ≤ C_VRAM                                         (6)
```

This is a 0/1 knapsack, but with **identical item sizes** it is not NP-hard at all: the greedy
solution — take the `⌊C_VRAM/w⌋` pairs with the largest `p_{ℓ,e}` — is exactly optimal [P, standard].
Saving per promoted expert is `p·w·(1/β_h − 1/β_g) = p × 0.046 ms` [D].

Two consequences follow immediately and are worth stating because they contradict a lot of
systems-paper machinery:

- **The problem is only interesting if `p` is skewed.** If `p_{ℓ,e} = k/N` for all e (perfectly
  balanced routing), every feasible set is optimal and the expected hit rate equals the cache
  fraction γ exactly. Then §2.6's *row-splitting* — which is size-proportional, not
  identity-based — strictly dominates any identity-based placement, because it achieves γ-proportional
  benefit **on every expert** instead of on a γ-fraction of them, and it never mispredicts.
- **Capacity is in bytes of *weights*, not bytes per token.** Our γ = 23.1 % [D] of 241 GiB.

### 3.2 What changes with temporal locality — and what the classical theory actually says

If the reference stream were i.i.d. (**Independent Reference Model**, Coffman & Denning,
*Operating Systems Theory*, Prentice-Hall 1973), then the optimal *online* policy is the static
one: keep the most-frequent items; LRU cannot beat it, and paging theory's worst cases are
irrelevant. The relevant classical facts, applied only where they apply:

- **Belady's MIN** (Belady, *IBM Systems J.* 5(2), 1966 [P]): the offline optimum evicts the item
  whose next reference is farthest away. It is the upper bound any online policy is measured against,
  and papers that report only "we beat LRU" without a Belady line are reporting an unbounded quantity.
- **LRU is exactly k-competitive** for a cache of k items (Sleator & Tarjan, *CACM* 28(2), 1985 [P]),
  and no deterministic online algorithm is better. **Randomised marking is 2H_k-competitive**
  (Fiat, Karp, Luby, McGeoch, Sleator & Young, *J. Algorithms* 12(4), 1991 [P]).
  With k ≈ 3 500 resident experts these ratios are astronomically loose and say nothing useful:
  *competitive analysis is the wrong tool here and I do not use it beyond noting that it is.*
  The right tool is the measured hit-rate curve.

So the entire question is empirical: **how skewed and how temporally correlated is `p` for a
384-expert, top-6, shared-expert router?**

### 3.3 What the literature measures about MoE routing statistics

| source | model (experts / top-k / layers) | what is measured | number | read level |
|---|---|---|---|---|
| Liang et al., *Not All Models Suit Expert Offloading*, ICLR 2026 / arXiv 2505.16056 | 22 MoE models | SRP, SCH (segment-level routing consistency & best achievable cache hit) | correlation of SCH with LRU/LFU/Fixed rises with segment length m: m=4 → LRU 81.2, LFU 77.4, Fixed 76.3; m=256 → 97.5 / 99.2 / 97.9 (their Table 3) | **full text read** (HTML) |
| same | same | turning point of the SCH(ρ) curve | "cache size approximately 2× the number of active experts"; "ρ=2 can balance cache effectiveness and efficiency" (their §5, Fig. 5) | full text; **per-model SCH values are only in Fig. 5, not tabulated** — I could not extract numbers |
| same | grouping result | which architectures cache well | "Models that apply MoE on every layer and do not use shared experts exhibit the highest local routing consistency" | full text |
| *Cacheable by Design?* (pre-registered negative result), arXiv 2608.18261 | Qwen3-30B-A3B: 128 experts, top-8, 48 layers | expert cache hit rate @ 13.4 % of experts resident, layer 24, code domain | **LRU 65.9 %, static-pin 59.2 %, Belady 79.1 %** (their Table 3 / Fig. 3) | full text read |
| same | same | adjacent-token reuse | "Expert reuse between adjacent tokens is **2.0× chance**" (Table 3) | full text |
| same | same | skew | "95 % of traffic flows through **52.5 %** of experts" (Table 3) | full text |
| same | same | domain structure | per-domain expert-set Jaccard: code vs others **0.11–0.16**; prose/math/medical mutually **0.33–0.42** | full text |
| Zhong et al., **HybriMoE**, DAC 2025 / arXiv 2504.05897 | Mixtral-8x7B (8/2), DeepSeek-V2-Lite (64/6, 2 shared), Qwen2-57B-A14B (64/8, 1 shared) | hit rate at **25 % cache capacity**, LRU → their MRS score-based policy | Mixtral **30.2 → 36.2 %**; DeepSeek-V2-Lite **47.7 → 52.7 %**; Qwen2 **45.0 → 52.8 %**; gap narrows at 75 % | full text (HTML) |
| Yi et al., **ReMoE**, arXiv 2605.27081 | DeepSeek-V2-Lite; Qwen1.5-MoE-A2.7B | expert overlap gain from router fine-tuning with a temporal-locality loss | **+26.4 %** and **+27.2 %** expert overlap | abstract/secondary only — **not obtained** in full |
| *Cacheable by Design?* | 137M/340M MoE (16 experts, top-2) trained from scratch | cost of *training* for locality | λ=0.02: −15 % misses, **+0.3 % PPL**; λ=0.05: −59–60 % misses, **+2.1–3.1 % PPL**; domain confinement 0.991 static-pin hit at **+3.1 % PPL**; at 340M the tax *rose* to +2.5 % | full text |
| **SpecMD**, arXiv 2602.03921 | OLMoE (64/8), Mixtral (8/2), Qwen1.5-MoE (60/4), Phi-3.5-MoE (16/2) | prefetch-policy comparison on an A100 with cache software-limited to 1/5/25 % | "Least-Stale" cuts *collision misses* up to **85×** vs LRU @1 %; **34.7 % TTFT** reduction on OLMoE @5 % | full text; per-method prediction accuracy NOT reported |

**A caveat I must record.** The architecture table I extracted from arXiv 2505.16056's HTML lists
Mixtral-8x7B as "4 experts, top-1" and Qwen3 as "16 experts, top-1", which contradicts the known
configurations (Mixtral is 8 experts top-2). Either the HTML table is an *expert-group–normalised*
presentation or my extraction mis-registered columns. **I therefore use only that paper's
qualitative conclusions and its Table 3 correlations, and treat its per-model architecture column as
unverified.** Its most important sentence for us — "models that … do **not** use shared experts
exhibit the highest local routing consistency" — classifies DeepSeek-style routers (shared expert
present) on the *low*-consistency side.

### 3.4 How the achievable hit rate scales for a 384-expert, top-6 router — our case

Nothing in the literature I obtained measures a 256–384-expert router's cache curve. The closest is
Qwen3-30B-A3B at 128 experts / top-8 (2608.18261). Therefore this subsection is **[D] and [I]**, and
the measurement that closes it is H2 in §6.

Two reference curves bracket the answer:

- **Uniform IRM floor**: `hit(γ) = γ`. At γ = 23.1 % ⇒ 23.1 % [D].
- **Qwen3-30B-A3B empirical**: LRU 65.9 % at γ = 13.4 %, Belady 79.1 % [M-paper]. If V4.1 behaved the
  same, γ = 23 % would give ~70–80 %.

The floor is much more likely to be right for V4.1, for three independent reasons:

1. **The router is trained to be flat.** DeepSeek-V3's auxiliary-loss-free load balancing adjusts a
   per-expert *selection bias* precisely so that expert load equalises without an auxiliary loss
   (DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv 2412.19437) — and V4.1's router carries exactly
   that shape: "`√softplus` score plus a **selection-only bias**, top-6, renormalise, ×1.5"
   (`v41-ops.md:27` [M-box]). A bias tuned for balance is a bias tuned *against* the skew that
   identity-based caching monetises. [I]
2. **Shared experts absorb the common-mode signal.** 2505.16056's grouping says exactly this: shared
   experts lower local routing consistency, because the "always useful" computation has been factored
   out of the routed set, leaving the routed set to carry the *differentiating* — hence
   token-dependent, hence less cacheable — part. [M-paper, qualitative]
3. **N/k is 64 here.** Chance overlap between two tokens is `k²/N = 36/384 = 0.094 experts` [D]
   (this is the "0.09" in `roofline.md:220`, and it is a *derived* independence value, not a
   measurement — I correct the doc on that point). Even the *measured* 2.0×-chance reuse of a
   128-expert model would give only 0.19 of 6 experts, i.e. 3 % sharing. Locality that is real but
   2× chance is worth almost nothing when chance is 1.6 %.

**Our own bound on sharing, and its confound.** `roofline.md:214-217` [M-box] fits
`ms/pass(k) = 10.32 + 50.21 + 17.28·k` with residual ±0.36 ms, and 17.28/50.21 = 34.43 % against a
routed byte share of 34.56 %.

**The doc reads that agreement as "sharing ≈ 0". That inference is only valid if host and GPU read at
the same effective bandwidth** — a *byte* share equals a *time* share only under uniform speed. Under
this report's (147.7, 247) the routed term's time share would be 25.45/50.21 = **50.7 %**, not 34.4 %.
So the WKS-36 slope constrains a *product* of two unknowns, and it is at least as much a
**bandwidth** measurement as a sharing measurement — see §5.5-1, where it forces β_h into [202, 231]
and fits the whole pass on a single 233 GB/s. Under the uniform-233 reading the slope does bound
sharing at `s ≲ 2 %`, i.e. below the 1.6 % chance level; under (147.7, 247) it bounds nothing about
sharing at all. **Both branches are open until H0.**

> **And even the favourable branch is confounded.** `roofline.md:195` records that ik's MoE decode is
> *slower* at batch 2 than batch 1, i.e. the engine may re-read expert bytes per token whether or not
> they are shared. The slope test cannot separate "no sharing" from "sharing not exploited". The
> counter-evidence is that the *dense* term in the same fit amortises perfectly, so the harness is
> capable of amortising — but dense and expert dispatch are different code paths. **H2 in §6 closes
> this with an offline count of distinct router IDs, which needs no engine change and no bandwidth
> assumption at all.**

### 3.5 Verify batches: the expert-count explosion

For a verify batch of k tokens, under independence the expected number of **distinct** experts
touched per layer is

```
  E[distinct] = N · (1 − (1 − k_top/N)^k) = 384 · (1 − (378/384)^k)                (7)
```

[D]: k=1 → 6.0; k=2 → 11.9; **k=4 → 23.4**; **k=8 → 45.5**; k=16 → 85.5.
Compare Mixtral (N=8, top-2): k=1 → 2.0; k=4 → 5.5; k=8 → 7.2 — saturating at 8, i.e. **at k=8 a
Mixtral verify batch reads 3.6× the expert bytes of a single token while ours reads 7.6×** [D].

> **Finding 5 [D].** Every speculative-decoding-with-offloading result obtained on Mixtral-class
> routers (8 experts, top-2) gets most of its benefit from a byte-amortisation that a 384-expert
> top-6 router structurally cannot provide. Expression (7) is the single number to extract from any
> such paper before believing its transfer to us. This is the same fact `roofline.md:219-222`
> measured end-to-end on our box, arrived at from the router's combinatorics instead.

Corollary for placement: since a verify batch of k=4 touches ~23.5 of 384 experts per layer
(6.1 % of experts), and GPU capacity γ = 23 %, an *identity*-based cache is not made more effective
by speculation — the extra experts are drawn from the same flat distribution. Row-splitting (§2.6),
by contrast, is completely indifferent to k.

### 3.6 Where the caching literature is actually right for us

One place, and it is not the expert cache: **`p` is not flat across *layers* if measured per byte of
benefit — but it is flat enough that our binding decision is the capacity split, not the identity
choice.** The literature's genuinely transferable results are the negative ones:

- Training the router for locality costs perplexity roughly linearly in the locality gained, and the
  {≥30 % miss reduction, ≤1 % PPL} corner is empirically unreachable at small scale, with the tax
  *increasing* with model size (2608.18261 [M-paper]). We are not going to fine-tune V4.1's router.
- Score-based caching (HybriMoE's MRS) beats LRU by **6–8 pp at 25 % capacity and the gap closes at
  75 %** [M-paper] — at our γ = 23 % this is the honest size of the prize from a *better* policy:
  single-digit percentage points of hit rate, i.e. `0.08 × 22.6 ms ≈ 1.8 ms` of the 56.5 ms pass,
  **≈ 3 %** [D].

---

## 4. Taxonomy and survey

Read-level key: **F** = full text read (HTML or PDF pages rendered), quoting section/table;
**A** = abstract or secondary source only; **B** = bibliographic citation only (see §7).

### 4.a Offloading with on-demand weight transfer — the class our numbers rule out

The defining move: expert weights live in DRAM (or NVMe) and are *copied to the GPU* when routed.
Per §2.8 this costs `β_h/β_link = 5.6×` more than computing on the host, at batch 1, on this box.
I therefore survey this class for *what it measures*, not as a candidate design.

| paper | venue/year | hardware | model (exp/top-k) | batch | baseline | headline | exact? | read |
|---|---|---|---|---|---|---|---|---|
| Eliseev & Mazur, *Fast Inference of MoE LMs with Offloading*, arXiv 2312.17238 | preprint 2023 | T4 16 GB (Colab), RTX 3060 12 GB, RTX 3080M, A100-80 | Mixtral-8x7B (8/2) | 1 | naive offloading (HF accelerate) 0.661–1.392 tok/s | **2.092 / 2.278 / 2.655 / 3.061 tok/s** with 2-bit experts (their Table 2) | **yes** — "speculative loading does not change the final model predictions" | F |
| same | | | | | | LRU hit ratio vs cache size k in Fig. 2 (left); speculative loading = run layer ℓ+1's gate on layer ℓ's hidden state, recall in Fig. 2 (right) at 1/2/10 layers ahead | | F |
| Xue et al., **MoE-Infinity**, arXiv 2401.14361 | preprint 2024 | — | — | — | — | activation-aware expert offloading | — | **B** |
| Song et al., **ProMoE**, arXiv 2410.22134 | preprint 2024 | — | — | — | — | proactive caching | — | **B** |
| Tang et al., **HOBBIT**, arXiv 2411.01433 | preprint 2024 | — | — | — | — | mixed-precision expert offloading (multiple precisions per expert) | changes precision per expert | **B** |
| Yi et al., **EdgeMoE**, IEEE TMC 2025 | journal 2025 | — | — | — | — | per-expert precision chosen offline by importance | output-changing | **B** |
| He et al., **ExpertFlow**, arXiv 2410.17954 | preprint 2024 | — | — | — | — | expert activation + token allocation | — | **B** |

**Why the class does not transfer to us** [D]: (i) the 5.6× penalty of (5); (ii) their baselines are weak
— Eliseev & Mazur's is HF `accelerate`, and a 2–4× over naive offloading says nothing against a host
that computes; (iii) all of the obtained numbers are Mixtral (8 experts, top-2), where an LRU cache of a
few experts covers a large share of traffic and (7) saturates at 8; at 384/6 neither holds.
The one genuinely reusable artefact is Eliseev & Mazur's **speculative expert loading** — running the
*next* layer's gate on the *current* layer's hidden state — because it is output-preserving. In our
regime it prefetches weights we already have in DRAM, so per §2.7-R1 its value is ≈ 0.

### 4.b CPU co-computation — the class we are in

| paper | venue/year | hardware | model (exp/top-k) | batch | baseline | headline | exact? | read |
|---|---|---|---|---|---|---|---|---|
| Kamahori, Gu, Zhu & Kasikci, **Fiddler**, **ICLR 2025** (arXiv 2402.07033) | peer-reviewed | Quadro RTX 6000 24 GB + Xeon Gold 6126 48c, PCIe3 x16 32 GB/s; and RTX 6000 Ada 48 GB + Xeon Platinum 8480+ 112c, PCIe4 x16 64 GB/s | Mixtral-8x7B (8/2), 16-bit | 1 (and beam 4–16) | DeepSpeed-MII 0.2.3, Mixtral-offloading, llama.cpp b2956 | **1.26×** single-batch (Fig. 4); 1.30× long prefill (Fig. 5); 11.57× beam search (Fig. 6) | **yes** — "without modifying the model structure or accuracy" | F |
| same — the decision rule | | | | | | `if cpu_lat(s) > gpu_lat(s) + trans_lat() then GPU else CPU`, cpu_lat linear in token count s, gpu_lat constant; "weight transfer latency is 2–5× longer than actual computation time" | | F |
| Chen et al., **KTransformers**, **SOSP '25** (ACM DOI 10.1145/3731569.3764843) | peer-reviewed | dual-socket Xeon Platinum 8452Y (36c each), **1 TB DDR5**, intra-socket **220 GB/s**, cross-socket **125 GB/s** (Intel MLC), A100-40 and RTX 4080-16, **PCIe 4.0, 32 GB/s theoretical** (§6.1) | DS-3 671B (**256/top-8**, 58 MoE layers, Int4), DS-2 236B (160/top-6, 59 layers, Int8), Qwen2-57B-A14B (64/top-8, 28 layers, Int8) | **1** — "All experiments use a batch size of 1, representing the most typical local deployment scenario" (§6.1) | Fiddler and llama.cpp *extended with expert-level offloading* — i.e. **strong** | decode **2.42–4.09× over Fiddler**, **1.25–1.76× over llama.cpp** (BF16); **1.77–1.93×** over llama.cpp quantized (§6.2) | **yes** for the kernel/scheduling part | **F (all 16 pages)** |
| same — Expert Deferral | | | | | | defer low-scoring routed experts' outputs to layer k+2; CPU util **74 → 100 %**, GPU **28 → 37 %**, **+33 %** decode on DS-3, up to **1.45×**; overall **1.66–2.56×** over llama.cpp (§6.3) | **no** — output-changing | F |
| same — the σ problem | | | | | | "in DeepSeek-V3, **shared experts account for only 18 % of the GPU execution time**, yielding limited performance benefits from overlapping execution and causing frequent idle periods" (§4.1) | | F |
| Zhong et al., **HybriMoE**, **DAC 2025** (arXiv 2504.05897) | peer-reviewed | Xeon Gold 5220R **restricted to 10 cores**, RTX A6000, Marlin 4-bit | Mixtral-8x7B (8/2), **DeepSeek-V2-Lite (64/6, 2 shared)**, Qwen2-57B-A14B (64/8, 1 shared) | batch 1 (implied; not stated) | llama.cpp, AdapMoE, KTransformers | **1.33×** prefill, **1.70×** decode over KTransformers | not stated | F |
| same — its scheduler | | | | | | minimise `max(CPU_TIME(cpu_expert), GPU_TIME(gpu_expert))` (their Eq. 2). **No expert is split across processors**: "the CPU computes the cached expert E while the GPU processes the uncached expert C" | | F |
| Liu et al., **HeteGen**, **MLSys 2024** (arXiv 2403.01164) | peer-reviewed | Xeon @2.30 GHz + **A10 24 GB**, PCIe ~30 GB/s | **OPT-6.7B…30B — dense, not MoE**, precision not stated | **1**, prefill 512 / decode 64 | DeepSpeed Inference, HF Accelerate, FlexGen | up to **+317 %** over FlexGen (OPT-30B) | not stated | F |
| same — the split | | | | | | **splits a single linear layer's weight dimensions** between CPU compute and GPU transfer, ratio `α ≈ T'_CPU/(T'_CPU + T'_COM)` (their Eq. 7), from `α = V_GPU·V_COM/(V_CPU·V_GPU + V_CPU·V_COM + V_COM·V_GPU)` (Eq. 5). Overlap is **within a layer** | | F |
| Yuan, Ma & Talati, **MoE-Lens**, arXiv 2504.09345 | preprint 2025 | A40/L4/A100 + large-RAM host | Mixtral-8x7B etc. | **20 000 – 200 000** (Fig. 4) | MoE-Lightning | **4.6× avg, up to 25.5×**; model predicts to 94 % | — | F (pp. 1–6) |
| same — what the CPU does | | | | | | **the CPU computes *attention*, not experts**: "The GPU handles compute-intensive GEMM operations, while the CPU processes the relatively light-weight attention mechanism" (Abstract); "All attention computations are offloaded to the CPU" (§6.1) | | F |
| Cao et al., **MoE-Lightning**, **ASPLOS 2025** (arXiv 2411.11217) | peer-reviewed | 1×T4 16 GB / 1×L4 24 GB / 2×T4 / 4×T4, Xeon 24–32c, 192–416 GB | Mixtral-8x7B, 8x22B, DBRX (16 exp) | large; **"we are focusing on off-line, batch-processing workloads"** | FlexGen (±CPU attention), DeepSpeed ZeRO-Inference 0.14.3 | up to **10.3×** over offloading baselines; vs FlexGen on MTBench S1 **1.16×** (their Table 4) | — | F |
| same — HRM | | | | | | `P_x^i = min(P_peak^i, B_peak^i·I_x^i, B_peak^(j,i)·I_x^j)` (Eq. 7); balance point `B_peak^i·I_x^i = B_peak^(j,i)·I_x^j` (Eq. 11). **"When I is less than P1's corresponding I, there is no benefit in swapping the data to GPU for computation … it is more beneficial to have a static weights placement strategy."** | | F |
| Song et al., **PowerInfer**, SOSP '24 | peer-reviewed | — | dense ReLU LLMs, hot/cold *neurons* | — | — | GPU keeps hot neurons, CPU computes cold | — | **B** |
| Xue et al., **PowerInfer-2**, arXiv 2406.06282 | preprint | smartphone | — | — | — | — | — | **B** |
| **CoX-MoE**, arXiv 2605.17889 | preprint 2026 | AMX-enabled Xeon + NVIDIA GPU | not extracted | not extracted | FlexGen, MoE-Infinity, HOBBIT | coalesced expert execution | — | **A** — PDF text not extractable; see §7 |

**The load-bearing observation of this subsection.** Fiddler, KTransformers and HybriMoE all place
whole experts on one side or the other. HeteGen splits *a weight matrix's dimensions* at batch 1 and
derives the split ratio in closed form — **but on dense models, and its "GPU share" is a share that is
*transferred over PCIe*, not one that is resident.** I found **no paper that applies a
HeteGen-style α-split to MoE experts with GPU-*resident* weights.** That is precisely §2.6, and it is
the strongest exact lever our model identifies. [I]

### 4.c Expert prediction / pre-gating

| paper | venue/year | method | reported accuracy | quality cost | relevance to us | read |
|---|---|---|---|---|---|---|
| Hwang et al., **Pre-gated MoE**, **ISCA 2024** (DOI 10.1109/ISCA57975.2024.00078, pp. 1018–1031) | peer-reviewed | replaces the gate with a *pre-gate* trained to decide layer ℓ+1's experts at layer ℓ, so weights migrate one layer early | not obtained | **retrained router → output-changing by construction** | R1 only: prefetch, not compute. Our weights are already in DRAM ⇒ ≈ 0 | **B** |
| Eliseev & Mazur (above) | preprint | run layer ℓ+1's *existing* gate on layer ℓ's hidden state | recall curves, Fig. 2 (right), 1/2/10 layers ahead | **none — exact** | same: ≈ 0 for us | F |
| Zhong et al., **AdapMoE**, ICCAD 2024 (arXiv 2408.10284) | peer-reviewed | adaptive sensitivity-based gating + prefetch using cross-layer activation similarity | not obtained | adaptive gating changes k ⇒ output-changing | — | **B** |
| **SiDA-MoE** | — | data-aware hash predicting activation, parallel prefetch | not obtained | — | — | **B** |
| *Accurate Expert Predictions in MoE Inference via Cross-Layer Gate*, arXiv 2502.12224 | preprint 2025 | layer-level expert predictor from inter-layer residuals | **54–67 % top-2 accuracy, 90.3–95.5 % hit-1** on Mixtral-8x7B / 8x22B | exact (prefetch only) | Mixtral-class; a top-2 predictor does not extrapolate to top-6-of-384 | **A** |
| **SpecMD**, arXiv 2602.03921 | preprint 2026 | study of speculative expert prefetching policies; "Least-Stale" | up to **85×** fewer collision misses than LRU at 1 % cache; **34.7 % TTFT** cut on OLMoE at 5 % | exact | A100, cache-limited; TTFT = prefill, not decode | F |

> **Finding 6 [I], and it is the crux.** Prediction of expert *identities* can only ever buy a
> *prefetch*. Prefetching cannot start the *computation*, because `e_ℓ(x_ℓ)` needs `x_ℓ`, which does
> not exist until layer ℓ−1's combine and layer ℓ's attention have run. Therefore **class (c) cannot
> produce cross-layer compute overlap.** Any paper that claims it either (i) is in the
> transfer regime, where the prefetch *is* the expensive thing, or (ii) computes on a stale input and
> is output-changing. KTransformers' Expert Deferral is the honest version of (ii) and says so.

### 4.d Skipping / deferral / merging / approximation — quantified quality cost

| method | source | what changes | measured quality cost | speed | read |
|---|---|---|---|---|---|
| **Expert Deferral** | KTransformers §4.1, §6.3, Fig. 13(b), Table 2 | routed experts' outputs delivered to layer k+2 instead of k+1 | DS-3 on LiveBench: **−0.5 % avg at 6 deferred**, −6.7 % at 8. HumanEval/MBPP/GSM8K/StrategyQA DS-3 (2+6): 83.0/70.2/95.2/82.9 vs (8+0) 83.0/71.2/94.8/83.0 (Table 2) | **+33 %** decode DS-3; up to 1.45× | F |
| **Expert Skipping** (same experiment) | KTransformers Fig. 13(a) | lowest-scoring experts discarded | **−13.3 % avg at 6 skipped, −88.7 % at 8** | similar | F |
| Lu et al., *Not All Experts are Equal: Expert Pruning and Skipping*, **ACL 2024**, 6159–6172 | | permanent pruning | not obtained | — | **B** |
| Huang et al., **MC-MoE**, arXiv 2410.06270 | | mixture-of-precisions | not obtained | — | **B** |
| Yue et al., **Ada-K routing**, ICLR 2025 | | RL-trained per-token k | not obtained | — | **B** |
| Yi et al., **ReMoE**, arXiv 2605.27081 | | router fine-tuned for temporal locality | +26.4 % expert overlap (DeepSeek-V2-Lite) | — | **A** |
| *Cacheable by Design?*, arXiv 2608.18261 | | locality loss during training | λ=0.05: −59–60 % misses at **+2.1–3.1 % PPL**; tax **grows** with model size | — | F |

> The Deferral-vs-Skipping pair is the single most useful quantitative result in this class:
> **at the same number of affected experts, deferring costs 0.5 % and skipping costs 13.3 %**
> [M-paper, KTransformers Fig. 13]. Residual networks tolerate *delay* far better than *deletion*.

### 4.e Batching, scheduling, and speculative decoding with offload

| paper | venue/year | hardware | model (exp/top-k) | batch | headline | exact? | read |
|---|---|---|---|---|---|---|---|
| Svirschevski et al., **SpecExec**, **NeurIPS 2024** | peer-reviewed | consumer GPU + RAM offload | **dense** 50B+ (Llama-class) | draft trees, up to 20 tok/iteration | **4–6 tok/s** at 4-bit, 2–3 at 16-bit; **up to 15×** over standard offloading | lossless SD | **A** |
| **SpecOffload**, arXiv 2505.10259 | preprint 2025 | RTX 4090 24 GB + i9-10980XE/256 GB (PCIe3) and EPYC 7542/448 GB (PCIe4) | Mixtral-8x7B, 8x22B; draft Mistral-7B; bf16 | prefill 80, **decode 192**, draft 8 | **2.54–4.71×** over HF Accelerate / DeepSpeed-FastGen / FlexGen / Fiddler | not stated | F |
| same — placement | | | | | **CPU runs the target's *attention*; GPU runs the draft model and the target's FFN** | | F |
| **Dovetail**, arXiv 2412.18934 | preprint 2024/25 | RTX 2080S 8 GB / 3090 24 GB / GTX 1050M, Xeon Silver 4214R 24c or i5-9300H 4c | **dense** LLaMA2-Chat 7B/13B, Vicuna-13B | γ=16 (server) / 7 (PC); acceptance τ **4.53–6.26** (Table 6) | 10.14× (HumanEval, 13B, 3090); 3.78× vs SpecExec 2.98× on 2080S | **lossless** | F |
| same — placement | | | | | **draft on GPU, target on CPU**; "the primary bottleneck shifts to parallel verification" | | F |
| Cao et al., **MoE-Lightning** (above) | ASPLOS 2025 | T4/L4 | Mixtral/DBRX | large, offline | CGOPipe + HRM; 10.3× | — | F |
| Yuan et al., **MoE-Lens** (above) | preprint | A40/L4/A100 | Mixtral | **20k–200k** | 4.6× avg over MoE-Lightning | — | F |
| **LayerScope**, arXiv 2509.23638 | preprint 2025 | "legacy servers" | — | multi-batch | predictive cross-layer scheduling | — | **B** |
| **Klotski**, **MoE-Gen**, **NEO**, **FastDecode** | — | — | — | — | — | — | **not obtained**, §7 |

**The question the spec asks — what happens to CPU-side expert cost when a verify batch of 4–8 tokens
activates 24–48 distinct experts — is answered by no paper I obtained.** Every speculative-decoding
result above is either on a **dense** model (SpecExec, Dovetail) or puts the *attention* on the CPU and
keeps the FFN/experts on the GPU (SpecOffload, MoE-Lens, MoE-Lightning). **Nobody measures a verify
batch against host-resident experts of a high-expert-count router.** Our `roofline.md` WKS-36 arm
(slope 17.28 ms/token = 34.43 % of the pass = the routed byte share to 0.13 pp) appears to be the only
such measurement in existence, and expression (7) explains it. [I]

### 4.f System mechanisms, where papers measure them

| mechanism | measurement | source | read |
|---|---|---|---|
| kernel-launch overhead in hybrid MoE decode | Fiddler: **~7 000 CUDA launches per decoded token**, avg 16 µs, **73 % of GPU execution time**. llama.cpp: ~3 000 launches/token, 5 µs, **21 %** | KTransformers §2.3, Fig. 4 | F |
| CUDA Graph with host work in the graph | `submit` and `sync` hidden inside `cudaLaunchHostFunc`, whole decode path for one token in **one** CUDA graph ⇒ **1.23×** decode | KTransformers §3.3, §6.4 | F |
| NUMA | single MoE layer decode of DS-3 under Fiddler: **6.9 ms on one socket, 5.8 ms across both (16 %)**; with NUMA-aware *tensor* parallelism (every expert's matrix partitioned across sockets) **up to 1.63×** decode | KTransformers §2.3, §3.3, §6.4 | F |
| AMX vs AVX-512 at low arithmetic intensity | AMX peak 5.4 TFLOPS vs AVX-512 1.8 TFLOPS on DS-3 MoE layers, but only ~7 % of AMX's 73.7 TFLOPS theoretical; **AVX-512 beats AMX when ≤ 4 tokens are assigned to each expert** and gives **2.22×** decode vs 1.45× for AMX | KTransformers §2.2, §3.2, Fig. 3, Fig. 7, §6.4 | F |
| dynamic work scheduling | **1.83×** in prefill, "minimal" in decode (decode tasks are already balanced) | KTransformers §6.4 | F |
| pinned vs pageable promotion staging | promotion copies from **pageable** memory serialise against the critical-path input copy in the driver staging lock: **~27 ms/step** lost until pinned. Measured PCIe: pinned HtoD **26.2 GB/s** (26.15 even with the compute stream saturated), pageable **14.4** | our own port, `v41-ports.md:36-38` | **[M-box]** |
| mmap major faults on an NVMe-resident table | 41–62 faults/token → 13.6 tok/s; with `WILLNEED` prefetch 1–11 faults → 18.4; fully cached 18.8–19.0 ⇒ **≈ 380 µs per major fault** [D] | `v41-ports.md:12-13` | **[M-box]** |

> **Finding 7 [I].** Two of these are load-bearing for us and neither is about bandwidth.
> (i) **AVX-512 beats AMX below ~4 tokens/expert** — KTransformers' own measurement — which says the
> matrix-extension hardware our box *lacks* would not have helped at batch 1 anyway. What our AVX2-only
> host loses versus their AVX-512 host is the vector width on the *low*-ARI kernel, not AMX.
> (ii) **CUDA-graph capture across a host boundary** is worth 1.23× on its own, and our own port
> measured **−13.8 % TG when graph capture breaks** (`v41-ports.md:34`). Independent agreement,
> opposite sign, same mechanism.

---

## 5. Critical synthesis

### 5.1 "Transfer the expert" vs "compute on the host" is not a disagreement — it is one break-even seen from two batch sizes

Fiddler states the rule explicitly: `if cpu_lat(s) > gpu_lat(s) + trans_lat() then GPU else CPU`,
with `cpu_lat` **linear in the number of tokens s assigned to that expert** and `trans_lat` constant
[M-paper, Fiddler §3]. Every paper in §4 sits on one side of that inequality and none contradicts
another:

- Fiddler, KTransformers, HybriMoE run at **batch 1** ⇒ s = 1 ⇒ **compute on the host**.
- MoE-Lightning (offline batching) and MoE-Lens (**batch 20 000–200 000**) run at huge s ⇒ the GPU's
  compute wins and weights stream over PCIe; MoE-Lens does not even put the *experts* on the CPU, it
  puts the *attention* there [M-paper, MoE-Lens Abstract, §6.1].

We can locate the crossover for our machine exactly. The host is bandwidth-bound until
`k* ≈ 2.2` tokens per weight pass [M-box, `roofline.md:130-133`], after which its time grows ∝ s.
Setting host time `(s/k*)·w/β_h` equal to transfer time `w/β_link`:

```
  s* = k* · β_h/β_link = 2.2 × 5.64 = 12.4 tokens per expert                     (8)
```

and since a batch of B tokens through a top-k-of-N router gives each activated expert
`B·k/N` tokens on average,

```
  B* = s* · N/k = 12.4 × 64 = ≈ 794 concurrent tokens                            (9)
```

> **Finding 8 [D].** On this box, on-demand expert transfer only starts to beat host compute at
> around **800 concurrent tokens**. MoE-Lightning and MoE-Lens run at 20k–200k; Fiddler,
> KTransformers and we run at 1. The literature is consistent; it is the *reader* who imports the
> wrong half. Formula (9) is the test to apply to any offloading paper before believing it here.

### 5.2 What evaporates at batch 1, on AVX2, or against a strong baseline

- **10.3× (MoE-Lightning), 25.5× (MoE-Lens), 317 % (HeteGen), 15× (SpecExec), 11.57× (Fiddler beam
  search)** are all against offloading or naive baselines (HF `accelerate`, DeepSpeed ZeRO-Inference,
  FlexGen) and/or at large batch. The number to carry away from this literature is the one measured at
  batch 1 against a *tuned* baseline: **KTransformers' 1.25–1.76× over an expert-offloading-extended
  llama.cpp** [M-paper, §6.2], and **Fiddler's 1.26×** over three real systems [M-paper, Fig. 4].
  That is the realistic scale of the whole hybrid-scheduling art at batch 1 — and it brackets my
  §2.6 exact bound of **1.15×** and §2.4's two-stream **1.67–1.82×** very comfortably.
- **AVX2-only costs us less than it looks.** KTransformers' own microbenchmark says **AVX-512 beats
  AMX whenever ≤ 4 tokens are assigned to each expert**, and their decode gain comes from the AVX-512
  kernel (2.22×), not AMX (1.45×) [M-paper, §6.4, Fig. 3, Fig. 7]. More importantly, by (7) a
  384-of-6 router gives each activated expert **≈ 1.02 tokens even in a k=4 verify batch** [D], so the
  routed-expert path never gains arithmetic intensity from batching at all: its ARI stays at
  4.65 flop/byte [M-box, `roofline.md:116`] regardless of k. **Therefore `k*` — and with it the whole
  AVX2-vs-AVX-512-vs-AMX question — is irrelevant to the routed-expert term on this model.** Our own
  kernels already measure at 95–99 % of ik's on AVX2 (`plan.md:115,118`), and `plan.md:9` records that
  two very different CPU kernels land within 3 % — the term is bandwidth, as roofline predicted.
  **Recommendation: do not spend a round chasing the ISA gap.** [I]
- **Expert caching's remaining prize is single-digit percent.** HybriMoE's MRS beats LRU by 6–8 pp at
  25 % capacity and the gap closes by 75 % [M-paper] ⇒ ≈ 1.8 ms of 56.5 [D].
- **Prediction/pre-gating buys ≈ 0** (Finding 6), because our weights are already in DRAM.

### 5.3 The open problem our setting exposes

Three candidates; one is genuinely open.

1. **Exact cross-layer overlap under hyper-connections — closed, negatively.** I read
   `v41-ops.md:46-49` specifically for this. V4.1's hyper-connection lag is on the *mixing
   coefficients* (`comb`, built by a 24-dim gemv + sigmoid + Sinkhorn-20 and consumed one sublayer
   late), not on the activations. The residual **data** path is still a chain. What the lag does buy
   is hoisting the Sinkhorn work — latency, not bytes — off the critical path (§2.7-R3). **Not the
   open problem.**
2. **Engram/NVMe in the critical path — closed, positively.** Next-token row IDs are computable at
   sampling time [M-box, `v41-ops.md:69-70`], so the 48 scattered reads have a full step of slack.
   Our own ports already measure the prize at ≈ 19.2 ms/token, ≈ 380 µs per major fault [D from
   `v41-ports.md:12-13`]. Engineering, not research.
3. **Speculative verify batches against host-resident experts of a high-expert-count router —
   genuinely open.** No paper I obtained measures it: speculative-decoding-with-offload work is on
   **dense** models (SpecExec, Dovetail) or keeps experts on the GPU (SpecOffload, MoE-Lens,
   MoE-Lightning). Expression (7) says the host term scales **linearly and un-amortisably** with
   verify width, so for us

   ```
     pass(k tokens) = k·C + G = 22.6k + 33.9 ms        [D]
     per-token, all accepted:  22.6 + 33.9/k  →  22.6 ms  as k → ∞
     ceiling of speculation alone:  56.5/22.6 = 2.50×
   ```

   — the *dense* half amortises, the routed half never does. At the measured acceptance
   (τ = 2.408 tokens at n_max=3, `roofline.md:212`) the realised figure is 112.20/2.408 = 46.6 ms/token
   = **1.078×** [D]. The open research question is whether *anything* can make a verify batch share
   expert reads in a 384-of-6 router — the only candidates are output-changing (force the k draft
   tokens through a common expert set; nobody has published this) — and it is exactly the question our
   box is instrumented to answer.

**A fourth, which I think is the real one** [I]: **nobody has applied divisible-load splitting to MoE
experts with GPU-resident weights.** HeteGen derives the α-split at batch 1 but for dense layers and
against a *transfer* cost (its Eq. 5/7 has `V_COM` in it); HybriMoE explicitly refuses to split
(`CPU computes the cached expert, GPU processes the uncached expert`); KTransformers partitions
experts across *NUMA sockets* (1.63×) but not across the CPU/GPU boundary. Our §2.6 is the missing
cell of that table, it is exact, and it is the only lever in this report that is simultaneously
prediction-free, batch-independent, and worth more than 3 %.

### 5.4 Is the user's intuition right, and by how much?

*"Batching/chunking GPU and CPU work together must raise speed."* **Yes, and here is the ladder,
with our r = C/G = 0.666:**

| claim | value | requires | exact? |
|---|---:|---|---|
| any two-processor overlap, ever | ≤ 2.00× | r = 1 | — |
| work bound at today's placement, `1 + min(r, 1/r)` | ≤ **1.67×** | **≥ 2 independent sequences** | exact |
| same, rebalanced to all-experts-on-host | ≤ **1.82×** | ≥ 2 sequences + placement change | exact |
| **single stream, row-split experts, capacity-limited** | **1.15×** | nothing but engineering | **exact** |
| single stream, row-split, if VRAM were free | 1.37× | more VRAM | exact |
| single stream, block-granular (today's shape) | 1.04× | — | exact |
| single stream, stale-input (Expert Deferral) | → 1.82× | accuracy cost | **no** |

So the intuition is right in direction and, for a single stream, worth about **15 %** — not 67 %.
The 67 % is real but it is *throughput* and needs a second sequence, which `roofline.md:150-154`
already says. KTransformers arrives at the same place from measurement: they needed to *break* the
dependency (Expert Deferral) to get CPU utilisation from 74 % to 100 %, and they report shared experts
are only 18 % of GPU time [M-paper] — our σ is ~8 % [D], so we have *less* intra-layer slack than
DeepSeek-V3 does, not more.

### 5.5 Threats to this report's own numbers

1. **β_h = 147.7 is contradicted by a measurement already in our repo. This is the most important
   line in the report.** [D] WKS-36 fits a marginal cost of **17.28 ms per extra verify token**
   (`roofline.md:214`). By (7), the marginal token's experts are ~98 % fresh, so that 17.28 ms buys
   one token's worth of routed reads. But this model says one token's routed reads cost
   `3.336/β_h + 0.708/β_g = 22.58 + 2.87 = 25.45 ms` at (147.7, 247). **The marginal token is
   measured cheaper than the model says a single token's routed bytes can possibly be.**
   Inverting the slope for β_h:

   | assumed β_g (GB/s) | 247 | 400 | 700 | 936 |
   |---|---:|---:|---:|---:|
   | **implied β_h (GB/s)** | **231** | **215** | **205** | **202** |

   Every branch lands on the **215.6** side of the WKS-35 fork, none near 147.7.
   Pushed further: the doc's own celebrated coincidence — slope share 34.43 % ≈ routed byte share
   34.56 % (`roofline.md:216-217`) — is *only* expected when host and GPU read at the same effective
   speed (otherwise the time share would be 25.45/50.21 = 50.7 %, not 34.4 %). Taking that at face
   value gives a **one-parameter fit**: `4.044 GB / 17.28 ms = 234.0 GB/s` for the routed term, and
   `11.701 GB / 50.21 ms = 233.0 GB/s` for the whole pass — **agreement to 0.4 %**, with
   dense = 7.658/233 = 32.87 ms and 32.87 + 17.28 = 50.15 ms against a measured 50.21 ms.
   *Under that reading the host/GPU decomposition is not identifiable from the pass time at all*:
   `roofline.md:77` derived β_g = 247 by **assuming** β_h = 147.7, and the same data is fit at least
   as well by a single 233 GB/s.
   *Consequence:* C = 3.336/233 = **14.3 ms (28 % of the pass, not 40 %)**, G = 35.9 ms,
   **r = 0.40**, work bound **1.40×**, and §2.6's exact single-stream ceiling falls from 1.15× to
   roughly **1.06×** [D].
   *The caveat that could save 147.7:* `roofline.md:238` records that the 50.21 ms arms and the
   56.5 ms pair are **different runners**. If WKS-36 did not run the 33-host/7-GPU placement, the
   slope is not comparable and this collapses to "check the config". **That check is a file read, not
   a lease — see H4.**
2. **β_g = 247 GB/s is defined by subtraction** (`roofline.md:77`), so `G`, `r`, σ, and every row of
   §2.9 inherit it, and by (1) above it may be an artefact of the β_h assumption. If the cards
   actually deliver 500–700 GB/s effective and the missing time is launch/sync rather than bytes,
   then G's *bandwidth* part shrinks but its *serial* part does not, r rises, and §2.6's optimal split
   moves toward the host. H3.
3. **σ assumes the shared expert is the same size as a routed expert.** Inferred from
   `v41-ops.md:35`, not measured from the GGUF. A one-line `gguf-inventory` query settles it.
4. **The whole §2 assumes today's execution is fully serial.** If ik already overlaps some host and
   GPU work, the 56.5 ms is not `C + G` and the decomposition is wrong in the conservative direction.
5. **Our s ≤ 0.4 % sharing bound is confounded** by ik's non-batching MoE decode (§3.4). H2.

---

## 6. Hypotheses and proposed measurements

Ranked by (expected information) ÷ (cost to measure). Each is falsifiable on this box.

**H0 — WKS-36's slope already answers WKS-35, and the answer is not 147.7. (Cost: one file read.)**
*Hypothesis:* WKS-36 ran the same 33-host/7-GPU placement as the 56.5 ms pass. If so, its
17.28 ms/token slope forces `β_h ∈ [202, 231]` for any β_g (§5.5-1), and the whole-pass fit closes to
0.4 % on a **single** 233 GB/s — which would mean `roofline.md:77`'s β_g = 247 is an artefact of
assuming β_h = 147.7.
*Measurement:* read the WKS-36 runner config (`rig-log log/2026-09-19-verification-does-not-amortise.md`
and the `-ot` / block-placement flags it used) and confirm the placement and the model file.
*Kill condition:* a different placement or a different quantization ⇒ the slope is not comparable,
this collapses, and WKS-35 stays open.
*Why first:* it is free, and it re-scales every other number in this report.

**H1 — Row-split every expert across host and GPU at the capacity-limited ratio.**
*Hypothesis:* a single decode step falls from 56.5 ms to **49.3 ± 3 ms (1.15×)**, and to 41.3 ms
(1.37×) if VRAM for experts is raised to 37 % of expert bytes.
*Predicted by:* (3), (4). *Exact?* **Exact up to floating-point summation order** — gate it as a band
(the same class as our `q_nope2` cell split, `plan.md:117`), not as bit-identity.
*Measurement:* in bloomery's B2 tier, give the expert stage a split knob `x ∈ [0,1]` (host row
fraction) and sweep 0, 0.4, 0.6, 0.77, 1.0 on a quiet-machine lease; plot step ms against
`max(x·E/β_h, (1−x)·E/β_g)` + fixed dense. The model is confirmed if the minimum lands at
`x* = β_h/(β_h+β_g)` clamped by capacity, and killed if the curve is flat (⇒ the boundary cost, not
bandwidth, dominates — go to H5).
*Supported by:* HeteGen Eq. 5/7 (MLSys 2024, dense, batch 1); contradicted-by-omission by HybriMoE
(DAC'25, Eq. 2, refuses to split) and KTransformers (partitions across NUMA sockets, not across
CPU/GPU).

**H2 — V4.1's routing is flat and has no exploitable temporal locality.**
*Hypothesis:* for V4.1-Flash, (a) the cache hit rate at cache fraction γ is `≤ γ + 0.10`, i.e.
≤ 33 % at γ = 23 %, far below Qwen3-30B-A3B's 65.9 % at 13.4 %; (b) adjacent-token expert overlap
is ≤ 1.5× chance, i.e. ≤ 0.14 of 6 experts; (c) distinct experts per layer for a k=4 window is
23.5 ± 2.
*Predicted by:* §3.4, (7). *Exact?* n/a — pure offline analysis, **no engine change**.
*Measurement:* dump top-6 router IDs per layer per token for a few thousand tokens across ≥ 3 domains
(the oracle-v3 tap already exists — `v41-ops.md:84` says router top-6 IDs are integer-gateable), then
compute offline: `p_{ℓ,e}` histograms, Belady / LRU / LFU / static-greedy hit rate vs γ, adjacent-token
Jaccard, and (7)'s distinct count for k = 1..8.
*Why first-class:* this single dump **decides classes (a) and (c) permanently**, removes the confound
in §3.4, and costs no engine work. If (a) is refuted and the curve looks like Qwen3's, then
identity-based caching is back on the table and H1 drops in priority.
*Supported by:* 2608.18261 (Table 3), 2505.16056 (ICLR 2026), HybriMoE (DAC'25 hit-rate table).

**H3 — β_g is not 247 GB/s.**
*Hypothesis:* a STREAM-style device read on the 3090 and A6000 gives 600–780 GB/s, so the 33.9 ms
"GPU term" is mostly **not** bytes, and `G_bytes ≈ 8.366/0.70 = 12 ms` with ~22 ms of
launch/sync/attention-math.
*Consequence if true:* r rises toward 22.6/12 ≈ 1.9 (host becomes the bottleneck), the work bound
becomes `1 + 1/r = 1.53×`, and §2.6's optimal split moves further toward the GPU. Everything in §2.9
must be recomputed. *Cost:* one lease, two kernels. **Do this before H1.**
*Named as an open item by our own doc:* `roofline.md:94`.

**H4 — β_h is 147.7, not 215.6.** (rig-log WKS-35, already queued, `roofline.md:93`.) If 215.6,
C = 15.5 ms, r = 0.46, and the single-stream row-split ceiling drops from 1.15× to ≈ 1.09× [D].
**This report now predicts the outcome: H0's inversion of the WKS-36 slope puts β_h in [202, 231]
for any β_g, so I expect WKS-35 to come back on the 215.6 side or higher.** If the lease instead
returns 147.7 on the same working set that WKS-36 used, then the two measurements are inconsistent
and the instrument is the suspect — which is §8-rule 7 of the operating manual (의심 계측기), not a
model failure.

**H5 — The host boundary costs < 1 ms per token.**
*Hypothesis:* `2(α+o) ≤ 10 µs` per host layer ⇒ ≤ 0.66 ms over 33 layers (< 1.2 % of the pass).
*If refuted* (e.g. 50 µs ⇒ 3.3 ms ⇒ 6 %), the fix is known and measured by others: put the whole
decode path in **one CUDA graph with `cudaLaunchHostFunc`** for submit/sync — KTransformers reports
**1.23×** decode from exactly this [M-paper, §3.3, §6.4], and our own port measured **−13.8 % TG when
graph capture breaks** [M-box, `v41-ports.md:34`].
*Measurement:* a ping-pong microbenchmark (GPU→host 20 KB, host callback, host→GPU 20 KB) on the
compute stream, p50/p99; then the same inside a captured graph.

**H6 — VRAM freed converts linearly into speed.** At the (4) operating point,
`d(step)/dγ = −E_total/β_h = −4.044/147.7 = −27.4 ms per unit γ` ⇒ **−0.27 ms per +1 pp of γ** [D].
So 5 GB of VRAM recovered (≈ +2 pp of γ) ≈ −0.6 ms/step. *Measurement:* the same sweep as H1 with the
capacity clamp moved. *Consequence:* keeping engram on NVMe (already the design, `plan.md:64`) and
quantizing KV (`v41-ports.md:48-52`) are speed levers, not just capacity levers.

**H7 — Two independent sequences reach 1.67×, and all-experts-on-host reaches 1.82×.**
*Exact.* Throughput only; no single-stream latency gain. *Measurement:* `plan.md`'s C3 row already
names it (2 and 4 concurrent streams, summed tok/s). Add the **placement sweep**: move the 7 GPU
blocks' experts to the host and re-measure — the model predicts total throughput *rises* even though
single-stream latency falls, which is the counter-intuitive, easily-falsified part.

**H8 — Expert Deferral gives us less than it gave DeepSeek-V3.**
*Hypothesis:* KTransformers measured +33 % decode on DS-3 with 6 deferred experts at ≤ 0.5 % average
accuracy cost [M-paper, §6.3, Fig. 13, Table 2]. Our intra-layer slack is smaller (σ ≈ 8 % vs their
shared-expert share of 18 %), so I predict **+10–20 %**, not +33 %.
*Exact?* **No — output-changing.** Gate it on our own eval, and note the paired control they ran:
at the same count, *skipping* cost 13.3 % and *deferring* cost 0.5 %.
*Priority:* after H1, because H1 is exact and may capture much of the same idle time.

**H9 — Engram prefetch at sampling time removes ≈ 19 ms/token.** Already in the plan (B3,
`plan.md:64`); the literature adds nothing. Record major-fault counts as telemetry
(`v41-ports.md:75`).

**H10 — There is no expert-sharing left to find in a verify batch.** By (7), k=4 touches 23.5 distinct
experts of 384. *Hypothesis:* the distinct count measured in H2 matches (7) within 5 %, and therefore
speculation's ceiling on this machine is `56.5/22.6 = 2.50×` at infinite width and **1.08×** at the
measured acceptance. *If refuted* — if real drafts concentrate routing — the ceiling rises and C4
becomes the top lever instead of H1.

---

## 7. Not obtained / could not verify

**Could not open the full text (cited from a bibliography or an abstract only; not used for any
quantitative claim):** MoE-Infinity (arXiv 2401.14361); ProMoE (2410.22134); HOBBIT (2411.01433);
EdgeMoE (IEEE TMC 2025); ExpertFlow (2410.17954); MC-MoE (2410.06270); SwapMoE; Fate; Pre-gated MoE
(ISCA 2024 — searched, PDF not parsed); SiDA-MoE; AdapMoE (ICCAD 2024, 2408.10284); Ada-K routing
(ICLR 2025); "Not All Experts are Equal" (ACL 2024); PowerInfer (SOSP '24) and PowerInfer-2
(2406.06282); TwinPilots; LIA; Klotski; MoE-Gen; NEO; FastDecode; LayerScope (2509.23638);
DALI (2602.03495); SAEM (2608.21614); MELINOE (2602.11192); Harvest (2602.00328); ReMoE (2605.27081);
Q-Infer (ACM DOI 10.1145/3764589); APEX (2506.03296). Several of these are 2026 preprints surfaced by
search whose abstracts looked directly relevant; they are the obvious next fetch batch.

**Opened but could not extract:** CoX-MoE (arXiv 2605.17889) — *Coalesced Expert Execution for
High-Throughput MoE Inference with AMX-Enabled CPU-GPU Co-Execution*, 2026. The PDF's text layer did
not parse and the HTML route was not tried. **This is the highest-value unread paper in the set**: its
title names exactly the mechanism §2.6 proposes. Recommend a follow-up.

**Extracted but distrusted:** the per-model architecture table in arXiv 2505.16056's HTML lists
Mixtral-8x7B as 4 experts / top-1, contradicting the known configuration; I used only that paper's
qualitative conclusions and Table 3 (§3.3). Its per-model SCH values exist only inside Figure 5 and
could not be read as numbers.

**Numbers I deliberately did not use:** KTransformers Figure 7's y-axis ("Latency per MoE layer (ms)",
≈ 75–125 ms for DS-3) is not reconcilable with their own §2.3 statement that a DS-3 MoE layer decodes
in 6.9 ms under Fiddler on one socket; I used the §2.3 numbers, which are self-consistent with their
220 GB/s and DS-3's per-layer BF16 expert bytes, and cite Figure 7 only for its qualitative crossover
(AVX-512 > AMX at ≤ 4 tokens/expert).

**Not measured anywhere, ours or theirs:** α (PCIe + launch + host-thread wake) on this box; GPU
effective read bandwidth on this box; the shared expert's exact byte size in our GGUF;
V4.1-Flash routing statistics of any kind.

---

## 8. Bibliography

**Peer-reviewed — systems**

1. Hongtao Chen, Weiyu Xie, Boxin Zhang, Jingqi Tang, Jiahao Wang, Jianwei Dong, Shaoyuan Chen, Ziwei
   Yuan, Chen Lin, Chengyu Qiu, Yuening Zhu, Qingliang Ou, Jiaqi Liao, Xianglin Chen, Zhiyuan Ai,
   Yongwei Wu, Mingxing Zhang. **KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid
   Inference for MoE Models.** *SOSP '25*, Seoul, Oct 13–16 2025, 16 pp.
   https://dl.acm.org/doi/10.1145/3731569.3764843 · PDF:
   https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf
   — **read in full (16 pp.)**
2. Keisuke Kamahori, Yile Gu, Kan Zhu, Baris Kasikci. **Fiddler: CPU-GPU Orchestration for Fast
   Inference of Mixture-of-Experts Models.** *ICLR 2025*; arXiv:2402.07033.
   https://arxiv.org/html/2402.07033v2 — **read (HTML)**
3. **HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference.**
   *DAC 2025* (IEEE Xplore 11133274, ACM DOI 10.1109/DAC63849.2025.11133274); arXiv:2504.05897.
   Group: PKU-SEC-Lab (Meng Li's group, per his publication page).
   https://arxiv.org/html/2504.05897v1 — **read (HTML); author list NOT verified by the fetch**
4. **MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs.** *ASPLOS 2025*
   (ACM DOI 10.1145/3669940.3707267); arXiv:2411.11217. Author page:
   https://pschafhalter.com/moe-lightning/ . https://arxiv.org/html/2411.11217v1 —
   **read (HTML); author list NOT verified by the fetch**
5. Ziming Liu et al. **HeteGen: Efficient Heterogeneous Parallel Inference for Large Language
   Models on Resource-Constrained Devices.** *MLSys 2024*
   (https://proceedings.mlsys.org/paper_files/paper/2024/hash/5431dca75a8d2abc1fb51e89e8324f10-Abstract-Conference.html);
   arXiv:2403.01164. https://arxiv.org/html/2403.01164v1 —
   **read (HTML); first author from the MLSys author page, full list NOT verified**
6. Ruslan Svirschevski, Avner May, Zhuoming Chen, Beidi Chen, Zhihao Jia, Max Ryabinin. **SpecExec:
   Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices.**
   *NeurIPS 2024*, 16342–16368. https://arxiv.org/pdf/2406.02532 — **abstract/secondary only**
7. Ranggi Hwang, Jianyu Wei, Shijie Cao, Changho Hwang, Xiaohu Tang, Ting Cao, Mao Yang.
   **Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable MoE Inference.** *ISCA 2024*,
   1018–1031. DOI 10.1109/ISCA57975.2024.00078 — **not obtained**
8. Yixin Song, Zeyu Mi, Haotong Xie, Haibo Chen. **PowerInfer: Fast Large Language Model Serving with
   a Consumer-grade GPU.** *SOSP '24*, 590–606. DOI 10.1145/3694715.3695964 — **not obtained**
9. Shuzhang Zhong, Ling Liang, Yuan Wang, Runsheng Wang, Ru Huang, Meng Li. **AdapMoE.** *ICCAD 2024*;
   arXiv:2408.10284 — **not obtained**
10. Xudong Lu et al. **Not All Experts are Equal: Efficient Expert Pruning and Skipping for MoE LLMs.**
    *ACL 2024*, 6159–6172 — **not obtained**

**Preprints**

11. Yichao Yuan, Lin Ma, Nishil Talati. **MoE-Lens: Towards the Hardware Limit of High-Throughput MoE
    LLM Serving Under Resource Constraints.** arXiv:2504.09345v1, 12 Apr 2025.
    https://arxiv.org/pdf/2504.09345 — **read pp. 1–6**
12. Artyom Eliseev, Denis Mazur. **Fast Inference of Mixture-of-Experts Language Models with
    Offloading.** arXiv:2312.17238. https://arxiv.org/html/2312.17238v1 — **read (HTML)**
13. Zixuan Liang et al. **Not All Models Suit Expert Offloading: On Local Routing Consistency of
    Mixture-of-Expert Models.** arXiv:2505.16056; accepted ICLR 2026.
    https://arxiv.org/html/2505.16056v2 — **read (HTML); architecture table distrusted, see §7**
14. **Cacheable by Design? Training Mixture-of-Experts Routers for Locality Against the Edge
    Memory-Bandwidth Wall — A Pre-Registered Negative Result, with a Systems Measurement Study.**
    arXiv:2608.18261. https://arxiv.org/html/2608.18261 — **read (HTML)**
15. **SpecMD: A Comprehensive Study On Speculative Expert Prefetching.** arXiv:2602.03921.
    https://arxiv.org/html/2602.03921 — **read (HTML)**
16. **SpecOffload: Unlocking Latent GPU Capacity for LLM Inference on Resource-Constrained Devices.**
    arXiv:2505.10259v3. https://arxiv.org/html/2505.10259v3 — **read (HTML)**
17. **Dovetail: A CPU/GPU Heterogeneous Speculative Decoding for LLM Inference.** arXiv:2412.18934v2.
    https://arxiv.org/html/2412.18934v2 — **read (HTML)**
18. **CoX-MoE: Coalesced Expert Execution for High-Throughput MoE Inference with AMX-Enabled CPU-GPU
    Co-Execution.** arXiv:2605.17889. https://arxiv.org/pdf/2605.17889 — **not extractable, see §7**
19. **Accurate Expert Predictions in MoE Inference via Cross-Layer Gate.** arXiv:2502.12224 —
    **abstract only**
20. DeepSeek-AI. **DeepSeek-V3 Technical Report.** arXiv:2412.19437 — **not obtained**; cited only for
    the existence of auxiliary-loss-free load balancing, flagged [I] in §3.4.

**Classical results used (and only where they apply)**

21. Samuel Williams, Andrew Waterman, David Patterson. **Roofline: An Insightful Visual Performance
    Model for Multicore Architectures.** *CACM* 52(4), 2009, 65–76. — processor speeds in §2.2.
22. Gene M. Amdahl. **Validity of the Single Processor Approach to Achieving Large Scale Computing
    Capabilities.** *AFIPS Spring Joint Computer Conf.*, 1967, 483–485. — bound (2).
23. Roger W. Hockney. **The Communication Challenge for MPP: Intel Paragon and Meiko CS-2.**
    *Parallel Computing* 20(3), 1994, 389–398. — the α–β link model in §2.2.
24. David Culler, Richard Karp, David Patterson, Abhijit Sahay, Klaus Erik Schauser, Eunice Santos,
    Ramesh Subramonian, Thorsten von Eicken. **LogP: Towards a Realistic Model of Parallel
    Computation.** *PPoPP '93*, 1–12. — the `o`/`L` decomposition in §2.2.
25. Veeravalli Bharadwaj, Debasish Ghose, Venkataraman Mani, Thomas G. Robertazzi. **Scheduling
    Divisible Loads in Parallel and Distributed Systems.** IEEE Computer Society Press, 1996. —
    equalising-finish-time split, (3).
26. László A. Belády. **A Study of Replacement Algorithms for a Virtual-Storage Computer.**
    *IBM Systems Journal* 5(2), 1966, 78–101. — the offline optimum in §3.2.
27. Daniel D. Sleator, Robert E. Tarjan. **Amortized Efficiency of List Update and Paging Rules.**
    *CACM* 28(2), 1985, 202–208. — LRU is k-competitive.
28. Amos Fiat, Richard M. Karp, Michael Luby, Lyle A. McGeoch, Daniel D. Sleator, Neal E. Young.
    **Competitive Paging Algorithms.** *J. Algorithms* 12(4), 1991, 685–699. — randomized marking.
29. Edward G. Coffman, Peter J. Denning. **Operating Systems Theory.** Prentice-Hall, 1973. —
    the Independent Reference Model in §3.2.
30. Christos H. Papadimitriou, Mihalis Yannakakis. **Towards an Architecture-Independent Analysis of
    Parallel Algorithms.** *SIAM J. Computing* 19(2), 1990, 322–328. — scheduling with communication
    delays; cited in §2.7 **to say it does not help on a chain**.

**Cited and explicitly *not* applied:** S. M. Johnson, *Optimal two- and three-stage production
schedules with setup times included*, Naval Res. Log. Q. 1(1), 1954 — two-machine flow-shop assumes
independent jobs; our layers are a chain (§2.7). John L. Gustafson, *Reevaluating Amdahl's Law*,
CACM 31(5), 1988 — no scaled-problem argument exists here (§2.4).

**Our own measurements (primary sources in this repo)**

`/Users/hckim/repo/bloomery/docs/roofline.md`, `docs/plan.md`, `docs/research/v41-ops.md`,
`docs/research/v41-ports.md`, `docs/research/fusion.md`; rig-log WKS-35, WKS-36,
`log/2026-09-19-verification-does-not-amortise.md`.

---

## 9. Improvement opportunities outside scope

Reported only; nothing was edited.

- `bloomery/docs/roofline.md:224` — "상한은 8배가 아니라 **1 / 0.654 = 1.53배**". From the same fit
  (`ms/pass(k) = 10.32 + 50.21 + 17.28k`), the k→∞ ceiling of amortising the dense half is
  `50.21/17.28 = 2.91×`, not 1.53×; `1/0.654 = 1.53` is what you get if the *routed* half were free.
  Worth re-deriving — the conclusion of the section does not change, but the number is quoted forward.
- `bloomery/docs/roofline.md:220` — "두 토큰이 겹칠 기대값이 0.09개" reads as measured; it is exactly
  `k²/N = 36/384 = 0.0938`, the independence value. Mark it 유도.
- **`bloomery/docs/roofline.md:216-217` + `:83` are mutually inconsistent and the file does not say so.**
  The 17.28 ms/token slope implies one token's routed reads cost 17.28 ms; the 147.7 GB/s host line
  implies 22.6 ms for the host part alone. Inverting the slope gives β_h ∈ [202, 231] for any β_g,
  and a single 233 GB/s fits both the slope (234) and the whole pass (233) to 0.4 %. Either the two
  runners differ in placement (line 238 warns they differ in other ways) or 147.7 is wrong. This is
  the highest-value line in this report's §9 — full derivation in its §5.5-1 and H0.
- `bloomery/docs/roofline.md:59-64` — the 500 / 700 GB/s GPU rows are now known to be optimistic
  (247 derived at line 77). Candidates for a 선 그음 so they are not quoted forward.
- `bloomery/docs/roofline.md:93` — the 147.7 vs 215.6 fork gates every number in that file and in
  this report; it deserves to be the top of the WKS queue, not a bullet.
- `bloomery/docs/plan.md:247` — "ik의 0.20 ms/층(Qwen3.6)" is the B2 reference, but V4.1's per-layer
  host floor is 0.684 ms. The gate line should name the model so the 3.4× is not read as a regression.
- `bloomery/docs/plan.md:63` — B2's end-number is the whole-pass 22.6 ms; adding the per-layer
  0.684 ms would let the gate fail at the first layer instead of at assembly.
- `bloomery/docs/research/v41-ports.md:12-13` — the fault numbers imply ≈ 380 µs per major fault;
  that constant is not written down anywhere and belongs in B3's telemetry contract
  (`plan.md:64`).
- `bloomery/docs/research/fusion.md:41` — "384 전문가 라우팅은 융합이 덜 중요해지는 곳" assumes the
  host boundary dominates. If §2.6's row-split lands, each layer gains *two* expert stages instead of
  one and the per-layer launch count rises — fusion's value in V4.1 may be larger, not smaller, than
  this line assumes.
- `bloomery/docs/plan.md:80` (C3) — the row says "처리량 1.67배" without naming the placement it
  assumes; §2.4 says the throughput optimum is a *different* placement (all experts on host, 1.82×)
  than the latency optimum. Worth two rows instead of one.

DONE-hybridlit
