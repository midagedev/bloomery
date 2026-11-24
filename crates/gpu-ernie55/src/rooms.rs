//! The parts' stores against the plan: the KV sizing the plan counted at its context (the common
//! hybrid layout, [`model::arch::ernie55::place::KvLayout`]: an HCA layer's window ring,
//! compressor state and pooled rows, a latent layer's cached rows and indexer keys) is the bytes
//! the mixers' caches hold at the load. The caches are whole buffers sized from the context at the
//! load, no rooms: a sequence's pages are all backed before its first step. The two sides count
//! the same stores: a buffer nothing reads back (the append's shadow row, the pool's pre-rope
//! rows) is scratch the layers share, not a layer's store.

use bloomery_gpu::GpuError;
use model::arch::ernie55::hparams::Hparams;
use model::arch::ernie55::place::{
    FLASH_SEG_KEYS, FLASH_SEGMENTS, HC_MAX_TOKENS, HC_PIECE, HCA_COLS, HIST_BINS, MLA_UNIT,
    PROMPT_ROWS,
};

use crate::attn_hca::HcaDims;
use crate::body::Parts;
use crate::placed::PASS_ROWS;

// The widths the plan's reserve formula repeats, held to the kernels' and the body's own.
const _: () = assert!(PASS_ROWS == PROMPT_ROWS);
const _: () = assert!(HC_PIECE == bloomery_gpu::hc::HC_PIECE);
const _: () = assert!(HC_MAX_TOKENS == bloomery_gpu::hc::HC_MAX_TOKENS);
const _: () = assert!(MLA_UNIT == bloomery_gpu::site::BY_NAME_COLS);
const _: () = assert!(HCA_COLS == HcaDims::COLS);
const _: () = assert!(FLASH_SEG_KEYS == bloomery_gpu::hca::SEG_KEYS);
const _: () = assert!(FLASH_SEGMENTS == bloomery_gpu::flash_gqa::SEGMENTS);
const _: () = assert!(HIST_BINS == bloomery_gpu::indexer::HIST_BINS);

/// What the check's refusal names.
const WHAT: &str = "ernie55 stores";

/// `parts`' caches against `planned`, the bytes the plan's KV layout counts for the stage card's
/// layers at its context: caches past the plan's count are refused by name (the card would hold
/// more than the plan left it room for).
pub(crate) fn check(parts: &Parts, hp: &Hparams, planned: u64) -> Result<(), GpuError> {
    held_against(
        parts.hca.cache_bytes() + parts.mla.cache_bytes(),
        hp.n_layer,
        planned,
    )
}

/// The device bytes `parts` hold beside their stores: the walk's streams, the mixers' and the
/// routed block's working buffers, the words and the head's row. A part that reports fewer bytes
/// than its stores is refused by name.
pub(crate) fn beside_stores(parts: &Parts) -> Result<u64, GpuError> {
    let whole = [
        parts.hca.bytes() as u64,
        parts.mla.bytes() as u64,
        parts.moe.bytes() as u64,
        parts.trunk.bytes() as u64,
        parts.head.bytes() as u64,
    ]
    .iter()
    .sum::<u64>();
    let stores = parts.hca.cache_bytes() + parts.mla.cache_bytes();
    whole.checked_sub(stores).ok_or_else(|| {
        GpuError::shape(
            WHAT,
            format!("the parts report {whole} device bytes, fewer than their stores' {stores}"),
        )
    })
}

/// `held` device bytes of `n_layer` layers' caches against the plan's `planned`.
fn held_against(held: u64, n_layer: usize, planned: u64) -> Result<(), GpuError> {
    if held > planned {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "the {n_layer} layers' caches hold {held} device bytes; the plan's layout counts \
                 {planned}"
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use model::arch::delta_mla::{Mla, Situ};
    use model::arch::ernie55::hparams::{Hc, Hca, Hparams, Indexer, Rope};
    use model::arch::ernie55::place::{
        MLA_UNIT, PassTerms, hca_partials_block, mla_scores_row, prompt_reserve_bytes,
    };
    use model::arch::ernie55::spec::spec_of;
    use model::arch::models::ChatSpec;
    use model::arch::noaux::Experts;
    use model::arch::place_hybrid::KvLayout;

    use super::held_against;
    use crate::attn_hca::HcaDims;
    use crate::attn_mla::Shape;
    use crate::placed::{HostedPass, PASS_ROWS};

    /// The real checkpoint's widths over the layers `ratios` name (0 a latent layer), the routed
    /// experts `expert_ff` wide.
    fn hparams(ratios: Vec<usize>, expert_ff: usize) -> Hparams {
        Hparams {
            n_layer: ratios.len(),
            n_embd: 4096,
            n_vocab: 201_216,
            n_ctx_train: 1_048_576,
            dense_lead: 1,
            dense_ff: 14_336,
            rms_eps: 1e-6,
            heads: 64,
            mla: Mla {
                q_lora: 1024,
                kv_lora: 512,
                head_k: 256,
                head_v: 256,
                pe: 64,
            },
            hca: Hca {
                head: 512,
                ratio: 128,
                window: 128,
                o_groups: 8,
                o_lora: 1024,
            },
            ratios,
            indexer: Indexer {
                heads: 64,
                d: 128,
                top_k: 2048,
            },
            vha_rank: 16,
            rope: Rope {
                base: 8_000_000.0,
                compress_base: 160_000.0,
                factor: 16.0,
                original_ctx: 65_536,
                beta_fast: 32.0,
                beta_slow: 1.0,
            },
            experts: Experts {
                n_expert: 512,
                n_used: 10,
                ff: expert_ff,
                n_shared: 1,
                shared_width: expert_ff,
                weights_scale: 3.0,
                weights_norm: true,
            },
            latent: 2048,
            situ: Situ {
                gate_beta: 4.0,
                value_beta: 25.0,
            },
            hc: Hc {
                streams: 4,
                eps: 1e-6,
                sinkhorn_iters: 20,
            },
            defaults: Vec::new(),
        }
    }

    /// The plan's count of `hp`'s layers at `ctx` and the bytes the mixers' caches hold there, the
    /// plan side as the family's description reads it (its index keys f32) and the body side by
    /// the dims the mixers are built from.
    fn counts(hp: &Hparams, ctx: usize) -> (u64, u64) {
        let chat = ChatSpec {
            pre: "ernie".to_string(),
            template: None,
            tools: None,
            reasoning: None,
        };
        let spec = spec_of(hp, chat).expect("the description reads");
        let kv = KvLayout::of(spec.layers.iter().map(|l| &l.mixer))
            .expect("the layers' stores are sized")
            .with_f32_index_keys();
        let planned = kv.bytes(0..hp.n_layer, ctx as u64);
        let hca = HcaDims::of(hp, ctx, PASS_ROWS).expect("the HCA dims");
        let mla = Shape::of(hp, ctx).expect("the latent shape");
        let latent = (0..hp.n_layer).filter(|&l| hp.latent_layer(l)).count();
        let held = (hp.n_layer - latent) * hca.cache_bytes() + latent * mla.cache_bytes();
        (planned, held as u64)
    }

    /// The e2e gate's fixture plan, three layers (two HCA, one latent) at its 4,096-position
    /// context, counts 8,192,000 B: each HCA layer's 131,072 B ring, 32,768 B of pooled rows and
    /// 524,288 B of compressor state (688,128 B), and the latent layer's 4,096 rows of 1,664 B (a
    /// 1,152 B f16 cache row and a 512 B f32 index key). The mixers' caches hold those bytes
    /// exactly, and the check accepts them. Mutants: the hold's state, ring or pool term left out
    /// (the plan counts less than the caches hold), the shadow row or the pre-rope rows made a
    /// layer's again (the caches hold more), the index keys planned at f16.
    #[test]
    fn the_fixture_plan_holds_the_caches_the_mixers_make() {
        let hp = hparams(vec![128, 128, 0], 512);
        let (planned, held) = counts(&hp, 4096);
        assert_eq!(planned, 2 * 688_128 + 4096 * 1664);
        assert_eq!(planned, 8_192_000);
        assert_eq!(held, planned);
        held_against(held, hp.n_layer, planned).expect("the check accepts the fixture plan");
        // The two write-only buffers one layer used to hold (a 1,024 B shadow row, 2,048 B of
        // pre-rope rows) are a layer's store no more.
        let refused = held_against(held + 3072, hp.n_layer, planned);
        assert!(
            refused.is_err_and(|e| e.to_string().contains("the plan's layout counts 8192000")),
            "a cache past the plan is refused by name"
        );
    }

    /// The same equality on the real checkpoint's 43 layers (37 HCA, 6 latent) at the contexts a
    /// serve plans, and at 64 positions, below the window: the ring is whole from the first
    /// position, so the plan counts its 128 rows there too.
    #[test]
    fn the_real_widths_hold_what_the_plan_counts() {
        let ratios = (0..43)
            .map(|l| if l % 7 == 0 && l > 0 { 0 } else { 128 })
            .collect();
        let hp = hparams(ratios, 2048);
        for ctx in [64, 4096, 32_768, 262_144] {
            let (planned, held) = counts(&hp, ctx);
            assert_eq!(held, planned, "{ctx} positions");
            held_against(held, hp.n_layer, planned).expect("the check accepts the plan");
        }
        let (planned, _) = counts(&hp, 262_144);
        assert_eq!(
            planned,
            37 * (2 * 128 * 512 + 2 * 2048 * 512 + 524_288) + 6 * 262_144 * 1664
        );
    }

    /// The plan's reserve of the two mixers is the bytes their working buffers take: at each
    /// context the HCA term and the flash partials of the pooled rows' segments (a block of
    /// positions each) are [`HcaDims::scratch_bytes`], and the latent term and the score rows (a
    /// row a position) are [`Shape::scratch_bytes`] of one span, the sinks and the staging
    /// included. The `placed` term is the hosted pass's buffers ([`HostedPass::bytes_at`]): the
    /// host's sums of a unit's rows and the places of its routed slots. The whole reserve is the
    /// parts' terms summed. Mutants: a term of the formula left out of either side, the shadow row
    /// or the pre-rope rows made a layer's again, the slots' places left out of the `placed` term.
    #[test]
    fn the_reserve_holds_the_mixers_working_buffers() {
        let ratios = (0..43)
            .map(|l| if l % 7 == 0 && l > 0 { 0 } else { 128 })
            .collect();
        let hp = hparams(ratios, 2048);
        let terms = PassTerms::of(&hp);
        let (partials, block) = hca_partials_block(&hp);
        for ctx in [64, 4096, 32_768, 262_144] {
            let hca = HcaDims::of(&hp, ctx, PASS_ROWS).expect("the HCA dims");
            let planned = terms.hca + partials * (ctx as u64).div_ceil(block);
            assert_eq!(
                hca.scratch_bytes() as u64,
                planned,
                "HCA at {ctx} positions"
            );
            let mla = Shape::of(&hp, ctx).expect("the latent shape");
            let planned = terms.mla + mla_scores_row() * ctx as u64;
            assert_eq!(
                mla.scratch_bytes(MLA_UNIT) as u64,
                planned,
                "latent at {ctx} positions"
            );
        }
        let made = HostedPass::bytes_at((PASS_ROWS, hp.latent, hp.experts.n_used));
        assert_eq!(
            made, 1_053_696,
            "the hosted pass's buffers at the real widths"
        );
        assert_eq!(terms.placed, made as u64);
        assert_eq!(
            prompt_reserve_bytes(&hp),
            terms.trunk + terms.hca + terms.mla + terms.moe + terms.head + terms.placed
        );
    }
}
