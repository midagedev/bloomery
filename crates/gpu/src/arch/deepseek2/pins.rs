//! The widths deepseek2's kernels are compiled for, held against the file's
//! hyperparameters at load. A value the kernels were not built and gated for is
//! refused with its key and value before anything is allocated — the launch
//! would refuse some of them too, but only at the first step, and some not at
//! all.

use crate::GpuError;
use crate::flash::{LATENT, MMA_ROWS, MMA_WIDTH};
use crate::router::{N_EXPERT, N_USED};
use model::arch::deepseek2::hparams::Hparams;

/// The file's full metadata key for a suffix (`Gguf::arch_key`).
pub(super) type Key<'a> = &'a dyn Fn(&str) -> String;

const WHAT: &str = "deepseek2 kernel pins";

/// The attention half's pins: the latent flash kernels' block width, the
/// cache row the tensor-core tile is sized for, and the head count the tile
/// and the half-split `wv_b` quantize were gated at. `kv_b_latent` is the
/// latent width read from `attn_kv_b` (`MlaParams::latent`); it must be the
/// key's.
pub(super) fn attention(hp: &Hparams, kv_b_latent: usize, key: Key<'_>) -> Result<(), GpuError> {
    let latent_key = key("attention.kv_lora_rank");
    if hp.kv_lora_rank != kv_b_latent {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "{latent_key} is {}, attn_kv_b's rows are {} wide",
                hp.kv_lora_rank, kv_b_latent
            ),
        ));
    }
    if hp.kv_lora_rank != LATENT {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "{latent_key} is {}; the latent flash kernels run one {LATENT}-thread block \
                 per query row",
                hp.kv_lora_rank
            ),
        ));
    }
    let row = hp.kv_lora_rank + hp.rope_dims;
    if row != MMA_WIDTH {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "{} is {}: the cache row kv_lora_rank + rope is {row}, the tensor-core flash \
                 tile is sized for {MMA_WIDTH}",
                key("rope.dimension_count"),
                hp.rope_dims
            ),
        ));
    }
    if hp.n_head != MMA_ROWS {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "{} is {}; the tensor-core flash tile holds {MMA_ROWS} query rows and the \
                 wv_b site quantizes two halves of {} heads, gated at {MMA_ROWS} heads only",
                key("attention.head_count"),
                hp.n_head,
                MMA_ROWS / 2
            ),
        ));
    }
    Ok(())
}

/// The routed half's pin: the router kernel ranks a compiled [`N_EXPERT`]
/// experts into a compiled [`N_USED`] slots.
pub(super) fn router(hp: &Hparams, key: Key<'_>) -> Result<(), GpuError> {
    let e = &hp.experts;
    if e.n_expert == N_EXPERT && e.n_used == N_USED {
        return Ok(());
    }
    Err(GpuError::shape(
        WHAT,
        format!(
            "{} is {}, {} is {}; the router kernel ranks {N_EXPERT} experts into {N_USED} slots",
            key("expert_count"),
            e.n_expert,
            key("expert_used_count"),
            e.n_used
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::{attention, router};
    use model::arch::deepseek2::hparams::{Experts, Gating, Hparams};

    fn key(suffix: &str) -> String {
        format!("deepseek2.{suffix}")
    }

    /// The V2-Lite file's hyperparameters.
    fn v2_lite() -> Hparams {
        Hparams {
            n_layer: 27,
            n_head: 16,
            key_length: 192,
            value_length: 128,
            kv_lora_rank: 512,
            rope_dims: 64,
            experts: Experts {
                n_expert: 64,
                n_used: 6,
                n_shared: 2,
                ff: 1408,
                scale: 1.0,
                dense_lead: 1,
                gating: Gating::Softmax,
            },
        }
    }

    /// V2-Lite passes both pins; each value the kernels were not built for —
    /// GLM-4.7-Flash's where it has one — is refused by key and value.
    #[test]
    fn pins_refuse_by_key_and_value() {
        let hp = v2_lite();
        attention(&hp, 512, &key).expect("V2-Lite's attention");
        router(&hp, &key).expect("V2-Lite's router");

        type Edit = fn(&mut Hparams);
        let attn_cases: [(Edit, usize, &str); 4] = [
            (
                |_| {},
                256,
                "deepseek2.attention.kv_lora_rank is 512, attn_kv_b's rows are 256 wide",
            ),
            (
                |h| h.kv_lora_rank = 256,
                256,
                "deepseek2.attention.kv_lora_rank is 256; the latent flash kernels run one \
                 512-thread block",
            ),
            (
                |h| h.rope_dims = 32,
                512,
                "deepseek2.rope.dimension_count is 32: the cache row kv_lora_rank + rope is 544",
            ),
            (
                |h| h.n_head = 20,
                512,
                "deepseek2.attention.head_count is 20; the tensor-core flash tile holds 16",
            ),
        ];
        let mut wrong = Vec::new();
        let mut judge = |case: String, r: Result<(), crate::GpuError>, want: &str| match r {
            Ok(()) => wrong.push(format!("{case} passed; expected \"{want}\"")),
            Err(e) if e.to_string().contains(want) => {}
            Err(e) => wrong.push(format!("{case}: {e}")),
        };
        for (i, (edit, latent, want)) in attn_cases.into_iter().enumerate() {
            let mut h = v2_lite();
            edit(&mut h);
            judge(
                format!("attention case {i}"),
                attention(&h, latent, &key),
                want,
            );
        }

        let router_cases: [(Edit, &str); 2] = [
            (
                |h| h.experts.n_used = 4,
                "deepseek2.expert_count is 64, deepseek2.expert_used_count is 4; the router \
                 kernel ranks 64 experts into 6 slots",
            ),
            (
                |h| h.experts.n_expert = 160,
                "deepseek2.expert_count is 160, deepseek2.expert_used_count is 6;",
            ),
        ];
        for (i, (edit, want)) in router_cases.into_iter().enumerate() {
            let mut h = v2_lite();
            edit(&mut h);
            judge(format!("router case {i}"), router(&h, &key), want);
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }
}
