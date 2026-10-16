# G1 — IQ2_XXS in the grouped int8 GEMM: FAIL-first mutants

Two one-line mutations of `crates/gpu/src/gemm/grouped.rs`, each a `.patch`
beside this note that `git apply` takes on this tree (validated with
`patch -p1 --dry-run`; `git apply --check` is hook-blocked for delegates).
Each turns the `iq2xxs_*` cases of `gate_gemm` red; the clean tree is green.

| mutant | file:line | change | gate clauses that go red |
|---|---|---|---|
| `scale-2ls` | `crates/gpu/src/gemm/grouped.rs`, `iq2xxs_lane`'s scale word | `((aux >> 28) << 1) \| 1` → `(aux >> 28) << 1` (the sub-block scale `2·ls + 1` taken as `2·ls`) | every `gemm case=iq2xxs_*` row: `worst_err_over_band` ≫ 1 with a `first_failure:` line (the f64 band) and `contract_bits_differ` > 0 with a `first_contract_bits_failure:` line (the contract's bits), verdict `FAIL`; the gate exits 1 (`checks_failed`) |
| `sign-63` | same fn, the `sgn` closure | `signs_of((aux >> (7 * g)) & 127)` → `& 63` (the sign index cut to six bits) | the same clauses |

Clauses that stay green under both (deterministic wrongness, not noise):
`decode_vs_ggml=same` (the host decode is not mutated), the rerun and the
graph replay (bit-identical to the equally wrong eager run), `iq_nan_d`
(a NaN `d` still makes every output NaN). `rows_diag` only grows (a
diagnostic, not a pin).

Suggested run (the gate through the box, the mutants through the runner):

    tools/mutant-run.sh --crate bloomery-gpu \
        --expect scale-2ls='first_failure|first_contract_bits_failure' \
        --expect sign-63='first_failure|first_contract_bits_failure' \
        mutants-g1 -- \
        BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'cargo oxide build --arch sm_86 \
          -- -p bloomery-gpu-gates --features gpu --release --bin gate_gemm && \
          BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh \
          gate_gemm --case iq2xxs'

(`--case iq2xxs` runs only the IQ2_XXS cases: `iq2xxs_gate_synth`,
`iq2xxs_ragged_rows_synth`, `iq2xxs_types`; the q38 and K-quant cases filter
out.) Move this dir out of the tree first if the runner's sync prefers it
outside.

## Host-side verification already run (no card)

The decode the kernel and the gate's reference use, transcribed into a
standalone binary against `gguf::quant::dequant_row` (bit-exact to ggml) on
the `synthetic` byte stream of seed 0x5f1, 64 super-blocks — the same
comparison `decode_matches` makes on the card, plus a lane-by-lane
cross-check of `iq2xxs_lane`'s mma fragment words against the value-order
decode:

    correct decode: decode_matches=true worst_err_over_band=0.000 (want true)
    mutant (a) scale 2·ls: decode_matches=false worst_err_over_band=4194304.000 (want false)
    mutant (b) sign & 63: decode_matches=false worst_err_over_band=4194304.000 (want false)
    scratch: ok — decode transcription matches ggml; both mutants break it

A first wiring of that harness applied mutation (b) only to a side check and
it survived — the lesson written into the table above: the mutation must sit
in the output path the reference compares against, which is where both
patches above sit.
