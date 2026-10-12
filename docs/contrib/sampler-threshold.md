# Relative-weight sampler experiment

The experimental arm uses `w = exp(logit - max)` and combines top-k, top-p and
min-p cutoffs into one scalar floor. Eight byte-radix histogram passes locate
each count or cumulative-mass boundary. Filtering uses untempered weights, and
the top-p normalizer includes every top-k survivor before min-p. Survivors retain
the production ordering: descending logit, then ascending token id.

The existing benchmark chain supplies the CDF draw, with the production RNG
sequence in the experimental module. This avoids sending candidates through a
second full sampler/filter pass. Candidate vectors and sampled tokens are checked
against `Sampler::sample` outside both timers on every call. The production
sampler and its API are unchanged.

```sh
cargo run --release -p bloomery-sampler --example sample_bench -- --check-only
cargo run --release -p bloomery-sampler --example sample_bench -- --threshold-only
```

The original benchmark command still includes its warm/cold rows, pass breakdown
and before/after table, followed by the new comparison table. `--threshold-only`
skips those existing measurements. Each new cell rotates four synthetic rows,
warms up for 100 paired calls, then records 1,000 pairs with alternating arm order.
`paired_saved_us` is the median of baseline minus experimental time within each
pair; the independent arm medians need not subtract to that value.

## Semantic boundaries

A scalar probability cannot retain only part of an equal-probability group.
Four equal logits with `top_k=2` leave two candidates in the production sampler
and four in the scalar-cutoff arm. The same issue appears at a top-p boundary.
The engine's min-p comparison is `logit >= max + ln(min_p)` in f32; replacing it
with a comparison of f64 relative weights also changes some rounded boundaries.
The explicit check rows expose all three differences and print `false false`.
No fallback hides these counterexamples.

The pre-temperature, pre-min-p normalizer, nonfinite, all-NaN, greedy and empty
check rows agree. These examples and the timed row checks do not prove
equivalence for arbitrary inputs.

## Local CPU result, 2026-10-12

Measured on an Intel Core i5-12600K, Linux x86_64, pinned to logical CPU 0, with
the repository's `nightly-2026-08-28` and `RUSTFLAGS='-C target-cpu=native'`.
The base was `b8d1ff06e29819c857948ecd68a450b5226d80fe`. Both arms ran in the
same release binary under `tools/ref/lease-hold.sh`, using a local lock directory
and `docs/cards/contributor-sampler-threshold.card`. Builds and fuzzing had
finished. The ordinary serving processes remained present; the witness records
their activity. This table describes this contributor CPU, not a maintainer GPU
or end-to-end decode performance.

| V | Row | Filters | Baseline median us | Cutoff median us | Paired saved us |
|---:|---|---|---:|---:|---:|
| 151936 | peaked | T0.8 k40 p0.95 m0.05 | 47.4 | 1924.2 | -1875.2 |
| 151936 | peaked | top-k only | 47.9 | 1426.6 | -1376.4 |
| 151936 | peaked | top-p + min-p | 598.3 | 1478.0 | -876.6 |
| 151936 | peaked | min-p only | 127.3 | 542.0 | -414.1 |
| 151936 | flat | T0.8 k40 p0.95 m0.05 | 44.7 | 1908.8 | -1861.8 |
| 151936 | flat | top-k only | 39.5 | 1390.3 | -1346.0 |
| 151936 | flat | top-p + min-p | 970.1 | 1946.5 | -984.1 |
| 151936 | flat | min-p only | 449.9 | 914.1 | -461.8 |
| 154880 | peaked | T0.8 k40 p0.95 m0.05 | 45.7 | 1947.3 | -1899.7 |
| 154880 | peaked | top-k only | 47.2 | 1463.1 | -1413.3 |
| 154880 | peaked | top-p + min-p | 613.5 | 1517.7 | -902.5 |
| 154880 | peaked | min-p only | 128.7 | 550.9 | -422.5 |
| 154880 | flat | T0.8 k40 p0.95 m0.05 | 43.8 | 1941.5 | -1894.6 |
| 154880 | flat | top-k only | 40.2 | 1411.3 | -1369.4 |
| 154880 | flat | top-p + min-p | 994.5 | 1986.5 | -999.2 |
| 154880 | flat | min-p only | 462.2 | 929.5 | -470.1 |

All sixteen timed cells reported zero candidate and token differences across
their 1,100 paired calls. None saved the issue's required 300 us; this radix
implementation was slower by 414.1–1899.7 us per call at the paired median.
Computing relative weights for every logit and repeatedly scanning histograms
costs more here than the existing top-k scan and logit-space min-p filter [derived].

Record the negative result without moving this arm into the engine. Any future
implementation must address tie ordering and floating-point boundaries, then
establish its own semantic and timing evidence. This experiment does not rule
out a different implementation or performance on a different processor.
