//! ernie55 meta gate, the seam of the two halves: the header the folder map writes from an ERNIE 5.5
//! Flash Preview checkpoint folder (`model_hf::folder::open`, the load half) read through the
//! description's reader (`model::arch::ernie55`, the description half), and the tokenizer the folder
//! opens with. The folder is `BLOOMERY_REF_MODEL`, the profile's `MODEL` (`tools/ref/models/ernie5_5.sh`):
//! the vendor's folder in the real tier, the cut `tools/fixture-cuts/ernie55.json` makes in the
//! fixture tier; unset, the vendor's folder at its box path. No tensor byte is read.
//!
//! Two clauses, both `hw_` and `#[ignore]`d (`gate-ernie55-meta` runs them):
//! [`hw_the_folder_header_plans_as_the_description_pinned`] and
//! [`hw_the_folders_tokenizer_opens_as_ernie5`].

use std::path::PathBuf;

use gguf::arch::ernie55::{ARCH, PRE};
use model::arch::ernie55::place::PlanInputs;
use model::placement::accepts::{Program, routed_rule};
use model::placement::workstation::{self, CardSpec, MIB};
use model::placement::{self, PlanLevers};
use model_hf::folder::{self, Opened};

/// The vendor's folder at its box path, where `BLOOMERY_REF_MODEL` names none.
const REAL_FOLDER: &str = "/models/ERNIE-5.5-Flash-Preview";

/// The layers of the vendor's folder, and of the cut of it (layers 0, 1 and 7).
const REAL_LAYERS: usize = 43;
const CUT_LAYERS: usize = 3;

/// The routed bytes of one expert: the gate, up and down matrices as MXFP4 codes with a scale each
/// 32 values, at the vendor's expert width 2,048 over the latent 2,048 (3 x (2,048 x 1,024 + 2,048 x 64)).
const REAL_EXPERT_BYTES: u64 = 6_684_672;
/// The cut's expert width is 512: 3 x (512 x 1,024 + 512 x 64).
const CUT_EXPERT_BYTES: u64 = 1_671_168;

fn dir() -> PathBuf {
    PathBuf::from(
        std::env::var("BLOOMERY_REF_MODEL")
            .ok()
            .filter(|d| !d.is_empty())
            .unwrap_or_else(|| REAL_FOLDER.to_string()),
    )
}

fn open() -> Opened {
    let dir = dir();
    folder::open(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
}

/// The card the plan is made on: an RTX 5090 as `nvidia-smi` reports it, its driver reserve the
/// figure the named cards carry [assumed: no 5090 row in `workstation`]. The description's own gate
/// plans on the same card.
const RTX_5090: CardSpec = CardSpec {
    name: "5090",
    total_bytes: 32_607 * MIB,
    driver_reserve_bytes: 550 * MIB,
    device: None,
    free_bytes: None,
    held_by: None,
    integrated: false,
    margin_bytes: None,
    granule_bytes: workstation::GRANULE,
};

/// What the plan of a folder at one context holds: the experts and dense bytes the card keeps, its
/// KV bytes, and the bytes of the routed experts left on the host.
#[derive(Debug, PartialEq, Eq)]
struct Planned {
    card_experts: u64,
    dense: u64,
    kv: u64,
    host_experts: u64,
}

fn planned(inputs: &PlanInputs, ctx: u64) -> Planned {
    inputs.check_ctx(ctx).expect("a trained context");
    let machine = workstation::plan_on(RTX_5090, inputs.model.layers);
    let plan = placement::plan_routed_reserving(
        &inputs.model,
        &machine,
        ctx,
        &inputs.kv,
        &PlanLevers::default(),
        routed_rule(Program::Ernie55Body),
        0,
    )
    .unwrap_or_else(|e| panic!("plan @{ctx}: {e}"));
    let bad = plan.violations();
    assert!(
        bad.is_empty(),
        "plan @{ctx} breaks its invariants: {}",
        placement::joined(&bad)
    );
    let card = &plan.cards[0];
    Planned {
        card_experts: card.experts,
        dense: card.dense_bytes,
        kv: card.kv_bytes,
        host_experts: plan.host.expert_bytes,
    }
}

/// The folder's header read by the description's reader and planned on a 5090 equals what the
/// description's own gate pinned from the checkpoint's published shapes (`plan_on_a_5090` in
/// `crates/model/tests/ernie55_desc.rs`: 3,347 card experts at 32,768 and 3,028 at 262,144), so
/// the map's keys, stems, dims and types and the reader's agree on the whole file. Every expert is
/// on the card or the host (the card's and the host's bytes make the file's routed bytes), and the
/// architecture and `tokenizer.ggml.pre` of the header are the loader's two constants.
///
/// PIN(2026-11-24): `dense` is 9,227,818,876 B; the description's gate printed 9,228,171,132 B
/// before the merge, from a header table typing the 43 `ffn_norm` vectors BF16 where the loader
/// writes them F32, 352,256 B (43 x 8,192) the plan books as rounding instead (dense plus rounding
/// stays 9,247,260,672 B at 32,768); the experts, KV and host bytes did not move. The cut's `dense`
/// and `kv` are this run's [measured]: its KV is one latent layer and two HCA layers of the real
/// folder's per-layer bytes (46,137,344 + 2 x 917,504).
///
/// FAIL-first: the architecture string of the map or the description changed, a header key the
/// map writes as another type, or a stem one side renames, refuses the header or moves a figure.
#[test]
#[ignore = "needs the folder at BLOOMERY_REF_MODEL (default /models/ERNIE-5.5-Flash-Preview)"]
fn hw_the_folder_header_plans_as_the_description_pinned() {
    let o = open();
    assert_eq!(o.header.architecture(), Some(ARCH));
    let inputs = PlanInputs::describe(&o.header).unwrap_or_else(|e| panic!("{e}"));
    let layers = inputs.hp.n_layer;
    let (expert_bytes, at_32k, at_256k) = match layers {
        REAL_LAYERS => (
            REAL_EXPERT_BYTES,
            Planned {
                card_experts: 3_347,
                dense: 9_227_818_876,
                kv: 310_771_712,
                host_experts: 121_373_589_504,
            },
            Some(Planned {
                card_experts: 3_028,
                dense: 9_227_818_876,
                kv: 2_316_435_456,
                host_experts: 123_505_999_872,
            }),
        ),
        CUT_LAYERS => (
            CUT_EXPERT_BYTES,
            Planned {
                card_experts: 1_024,
                dense: 1_345_061_308,
                kv: 47_972_352,
                host_experts: 0,
            },
            None,
        ),
        n => panic!("{}: a folder of {n} layers, no pin for it", dir().display()),
    };
    let routed = (layers as u64 - 1) * 512 * expert_bytes;
    for (ctx, want) in [(32_768, Some(at_32k)), (262_144, at_256k)] {
        let Some(want) = want else { continue };
        let got = planned(&inputs, ctx);
        assert_eq!(got, want, "plan @{ctx}, {layers} layers, 5090");
        assert_eq!(
            got.card_experts * expert_bytes + got.host_experts,
            routed,
            "every expert is on the card or the host @{ctx}"
        );
        println!(
            "ernie55 folder plan @{ctx}, {layers} layers, 5090: card experts {}, dense {} B, kv {} B, host experts {} B",
            got.card_experts, got.dense, got.kv, got.host_experts
        );
    }
}

/// The folder's tokenizer opens through the `ernie5` row of `hf::vocab::PRE` with ERNIE's own
/// pre-tokenizer and BOS rule: `tokenizer.ggml.pre` is the loader's `PRE`, the BOS is the template
/// token `tokenizer.json`'s post-processor prepends (`<|ERNIE_MASK|>`, id 2) and not the config's
/// bos id 1, it is added, the stops are the generation config's `[2, 8]`, and the vocabulary is
/// the config's 201,216. FAIL-first: the BOS rule taking the config id, or the row dropped, moves a line.
#[test]
#[ignore = "needs the folder at BLOOMERY_REF_MODEL (default /models/ERNIE-5.5-Flash-Preview)"]
fn hw_the_folders_tokenizer_opens_as_ernie5() {
    let o = open();
    let t = &o.tokenizer;
    assert_eq!(t.pre(), PRE);
    assert_eq!(
        t.specials().bos,
        Some(2),
        "the template's token, not config bos 1"
    );
    assert!(t.add_bos(), "the post-processor adds it");
    assert_eq!(t.eog(), [2, 8]);
    assert_eq!(t.n_vocab(), 201_216);
    let plain = t.encode("Hello world", false, false);
    let marked = t.encode("Hello world", true, false);
    assert!(!plain.is_empty());
    assert_eq!(marked[0], 2);
    assert_eq!(marked[1..], plain[..], "the BOS is the only difference");
    println!(
        "ernie55 tokenizer: pre {}, bos {:?}, eog {:?}, {} ids, \"Hello world\" -> {marked:?}",
        t.pre(),
        t.specials().bos,
        t.eog(),
        t.n_vocab()
    );
}
