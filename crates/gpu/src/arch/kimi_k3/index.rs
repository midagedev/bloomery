//! The indexer's key rows of a K3.1 file's latent layers. A layer that runs the indexer keeps one
//! f16 key of the indexer's key width for each position, beside its cached row. The plan counts
//! that row from the description's selector (`model::arch::kimi_k3::place::KvLayout`), and the
//! placed load holds its stores to that count (`body::refuse_stores`), so this reading of the
//! header and the plan's are two readings of one fact. The body neither reads nor writes the rows:
//! the selected attention that fills them is not in it yet.

use model::arch::kimi_k3::hparams::Hparams;

/// The values of the index key row layer `l` keeps for each position: the indexer's key width on a
/// layer that runs the indexer ([`Hparams::dsa`]), `None` on a delta layer and on every layer of a
/// K3 file.
pub(super) fn width(hp: &Hparams, l: usize) -> Option<usize> {
    hp.indexer.as_ref().filter(|ix| ix.layers[l]).map(|ix| ix.d)
}

#[cfg(test)]
mod tests {
    use model::arch::delta_mla::{Mla, Situ};
    use model::arch::kda::Keys;
    use model::arch::kimi_k3::hparams::{Hparams, INDEX_DIM, INDEX_HEADS, Indexer, Kind};
    use model::arch::kimi_k3::place::KvLayout;
    use model::arch::kimi_k3::spec::spec_of;
    use model::arch::models::{ChatSpec, KdaGate};
    use model::arch::noaux::Experts;
    use model::placement::KvBytes;

    use super::width;

    /// Positions of the layout the rows are counted at.
    const CTX: u64 = 64;

    /// The kinds of the small header: delta layers with a latent layer at 2 and 5.
    const KINDS: [Kind; 6] = [
        Kind::Kda,
        Kind::Kda,
        Kind::Mla,
        Kind::Kda,
        Kind::Kda,
        Kind::Mla,
    ];

    /// The small header's hyperparameters, a K3.1 file's when `indexer` is set: the indexer runs on
    /// the latent layers and no other.
    fn header(indexer: bool) -> Hparams {
        Hparams {
            arch: if indexer { "kimi-k31" } else { "kimi-k3" },
            n_layer: KINDS.len(),
            n_embd: 512,
            n_head: 2,
            kda: Keys {
                conv: 4,
                head_dim: 128,
            },
            gate: KdaGate::Sigmoid { lower_bound: -5.0 },
            mla: Mla {
                q_lora: 256,
                kv_lora: 512,
                head_k: 192,
                head_v: 128,
                pe: 64,
            },
            rms_eps: 1e-5,
            situ: Situ {
                gate_beta: 4.0,
                value_beta: 4.0,
            },
            latent: 256,
            attn_res_block: 2,
            n_vocab: 2,
            n_ctx_train: 4096,
            dense_lead: 1,
            dense_ff: 768,
            experts: Experts {
                n_expert: 8,
                n_used: 2,
                ff: 256,
                n_shared: 2,
                shared_width: 512,
                weights_scale: 1.0,
                weights_norm: true,
            },
            kinds: KINDS.to_vec(),
            indexer: indexer.then(|| Indexer {
                heads: INDEX_HEADS,
                d: INDEX_DIM,
                top_k: 16,
                norm_eps: 1e-5,
                layers: KINDS.iter().map(|&k| k == Kind::Mla).collect(),
            }),
            defaults: Vec::new(),
        }
    }

    /// The plan's layout of `hp`, from its description.
    fn layout(hp: &Hparams) -> KvLayout {
        let chat = ChatSpec {
            pre: "kimi-k2".to_string(),
            template: None,
            tools: None,
            reasoning: None,
        };
        let spec = spec_of(hp, chat).expect("the small header describes");
        KvLayout::of(spec.layers.iter().map(|l| &l.mixer)).expect("the description lays out")
    }

    /// The body's rows a position on a latent layer, the cached row and the index key row, are the
    /// plan's bytes for that layer at [`CTX`] positions, f16 values each [derived from the
    /// header's widths, apart from the reader]: a K3.1 latent layer holds 512 + 64 cached values
    /// and 128 key values a position, 90,112 B at 64 positions, where a K3 latent layer holds
    /// 73,728 B; a delta layer keeps no key row. The load's `refuse_stores` holds the stores to
    /// the same count. Mutants: the width `None` on a latent layer (the plan's key row is not
    /// held), the width kept on a delta layer, the width of the index heads, the width kept in a
    /// K3 file.
    #[test]
    fn a_latent_layers_rows_are_the_plans_bytes() {
        let f16 = size_of::<u16>() as u64;
        for indexer in [false, true] {
            let hp = header(indexer);
            let kv = layout(&hp);
            for (l, kind) in KINDS.into_iter().enumerate() {
                let keys = width(&hp, l);
                if kind == Kind::Kda {
                    assert_eq!(
                        keys, None,
                        "indexer {indexer}: delta layer {l} keeps no key"
                    );
                    continue;
                }
                let rows = (hp.mla.row() + keys.unwrap_or(0)) as u64;
                assert_eq!(
                    kv.layer_bytes(l, CTX),
                    CTX * f16 * rows,
                    "indexer {indexer}: latent layer {l}"
                );
            }
        }
    }
}
