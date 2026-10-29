//! The catalog's rows: data only, no per-model code. A revision is a commit read from a table the repo
//! already carries (`tools/census/stubs.tsv`, a registry fixture row, a profile or a header README) and a
//! row without one is [`FLOATING`](super::FLOATING) and unverified. A size is a decimal byte count: exact
//! where one of those tables holds the file, else the docs' figure at 0.1 GB, an estimate the listing
//! replaces.

use super::{Format, Kind, Model, Row, Status};

pub(super) static MODELS: &[Model] = &[
    Model {
        name: "deepseek-v4.1-flash",
        aliases: &["ds41"],
        seat: "ds41",
        arch: "deepseek41",
        rows: &[
            Row {
                variant: "Q3_K_M",
                kind: Kind::Gguf,
                repo: "vcruz305/DeepSeek-V4.1-Flash-GGUF",
                revision: "c4a085541cb53f67ee5e57b63d255e80cef286e7",
                format: Format::Gguf3,
                bytes: Some(347_271_472_288),
                status: Status::Verified {
                    evidence: "README Use table and docs/models.md recommend this file; revision and size from tools/census/stubs.tsv",
                },
            },
            Row {
                variant: "Q8_0",
                kind: Kind::Gguf,
                repo: "vcruz305/DeepSeek-V4.1-Flash-GGUF",
                revision: "c4a085541cb53f67ee5e57b63d255e80cef286e7",
                format: Format::Gguf8,
                bytes: Some(507_954_227_008),
                status: Status::Unverified {
                    why: "explicit only: revision and size from tools/census/stubs.tsv; no doc or gate names this file",
                },
            },
        ],
    },
    Model {
        name: "glm-5.3-flash",
        aliases: &["glm"],
        seat: "glm",
        arch: "glm5next",
        rows: &[
            Row {
                variant: "UD-Q4_K_XL",
                kind: Kind::Gguf,
                repo: "unsloth/GLM-5.3-Flash-GGUF",
                revision: "a38483c8cd5df544f53d70fb281afe97369d5ab6",
                format: Format::Gguf4,
                bytes: Some(199_707_321_472),
                status: Status::Verified {
                    evidence: "README Use table and docs/models.md recommend this file; revision and size from tools/census/stubs.tsv",
                },
            },
            Row {
                variant: "Q8_0",
                kind: Kind::Gguf,
                repo: "unsloth/GLM-5.3-Flash-GGUF",
                revision: "a38483c8cd5df544f53d70fb281afe97369d5ab6",
                format: Format::Gguf8,
                bytes: Some(340_981_966_208),
                status: Status::Unverified {
                    why: "explicit only: revision and size from tools/census/stubs.tsv; no doc or gate names this file",
                },
            },
        ],
    },
    Model {
        name: "qwen3.8-flash-next",
        aliases: &["qwen38"],
        seat: "qwen38",
        arch: "qwen4exp",
        rows: &[
            Row {
                variant: "UD-Q4_K_XL",
                kind: Kind::Gguf,
                repo: "unsloth/Qwen3.8-Flash-Next-GGUF",
                revision: "766911a6b7369840a91dbcd95f9f997acaab6cd6",
                format: Format::Gguf4,
                bytes: Some(111_334_654_784),
                status: Status::Verified {
                    evidence: "README Use table and docs/models.md recommend this file; revision and size from tools/census/stubs.tsv",
                },
            },
            Row {
                variant: "UD-Q3_K_XL",
                kind: Kind::Gguf,
                repo: "unsloth/Qwen3.8-Flash-Next-GGUF",
                revision: "766911a6b7369840a91dbcd95f9f997acaab6cd6",
                format: Format::Gguf3,
                bytes: Some(89_986_353_824),
                status: Status::Verified {
                    evidence: "docs/models.md says this file runs; revision and size from tools/census/stubs.tsv",
                },
            },
            Row {
                variant: "nvfp4",
                kind: Kind::Folder,
                repo: "nvidia/Qwen3.8-Flash-Next-NVFP4",
                revision: "fc694b54fb0174e0913e6adf86691ef85a4ead47",
                format: Format::Nvfp4,
                bytes: Some(132_700_000_000),
                status: Status::Verified {
                    evidence: "README Use table; gate-gpu-qwen4exp-nvfp4 runs this folder; revision from crates/hf/tests/fixtures/quant-configs/README.md, size from docs/contrib/dgx-spark.md (rounded to 0.1 GB)",
                },
            },
            Row {
                variant: "fp4-local",
                kind: Kind::Folder,
                repo: "nvidia/Qwen3.8-Flash-Next-FP4-Local",
                revision: "b9ce18a6887e18c089fdea8509a33e63429a3300",
                format: Format::Fp4Local,
                bytes: Some(102_300_000_000),
                status: Status::Verified {
                    evidence: "README Use table; revision from crates/hf/tests/fixtures/quant-configs/README.md, size from docs/contrib/dgx-spark.md (rounded to 0.1 GB)",
                },
            },
            Row {
                variant: "Q8_0",
                kind: Kind::Gguf,
                repo: "unsloth/Qwen3.8-Flash-Next-GGUF",
                revision: "766911a6b7369840a91dbcd95f9f997acaab6cd6",
                format: Format::Gguf8,
                bytes: Some(188_225_033_248),
                status: Status::Unverified {
                    why: "explicit only: revision and size from tools/census/stubs.tsv; no doc or gate names this file",
                },
            },
        ],
    },
    Model {
        name: "qwen3.8-27b",
        aliases: &[],
        seat: "qwen3",
        arch: "qwen35",
        rows: &[
            Row {
                variant: "Q4_K_M",
                kind: Kind::Gguf,
                repo: "lmstudio-community/Qwen3.8-27B-GGUF",
                revision: "5a7da681f60570ab5b439a587e912d2e5eddb582",
                format: Format::Gguf4,
                bytes: Some(16_810_714_336),
                status: Status::Verified {
                    evidence: "the qwen38d profile's file (tools/ref/models/qwen38d.sh); revision and size from tools/ref/dense-logits.py",
                },
            },
            Row {
                variant: "nvfp4",
                kind: Kind::Folder,
                repo: "nvidia/Qwen3.8-27B-NVFP4",
                revision: "482ca0f3832238542f8f5295dde86b5f22711d80",
                format: Format::Nvfp4,
                bytes: None,
                status: Status::Verified {
                    evidence: "the qwen3 seat serves this folder (the dense gate's NVFP4 cut); revision from crates/hf/tests/fixtures/checkpoint-headers/README.md; size not recorded",
                },
            },
        ],
    },
    Model {
        name: "qwen3.6-35b-a3b",
        aliases: &[],
        seat: "qwen3",
        arch: "qwen35moe",
        rows: &[Row {
            variant: "Q4_K_M",
            kind: Kind::Gguf,
            repo: "lmstudio-community/Qwen3.6-35B-A3B-GGUF",
            revision: "68a34855558af61cbef0324d31f411be8a506b08",
            format: Format::Gguf4,
            bytes: Some(21_166_757_728),
            status: Status::Verified {
                evidence: "docs/models.md recommends this file; revision from crates/model/src/arch/registry.rs, size from crates/hf/tests/fixtures/lmstudio-community-qwen3-6-35b-a3b-gguf.json",
            },
        }],
    },
    Model {
        name: "qwen3-30b-a3b-instruct-2507",
        aliases: &["qwen3"],
        seat: "qwen3",
        arch: "qwen3moe",
        rows: &[
            Row {
                variant: "Q4_K_M",
                kind: Kind::Gguf,
                repo: "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
                revision: "eea7b2be5805a5f151f8847ede8e5f9a9284bf77",
                format: Format::Gguf4,
                bytes: Some(18_556_686_752),
                status: Status::Verified {
                    evidence: "README Use table and docs/models.md recommend this file; revision and size from tools/census/stubs.tsv",
                },
            },
            Row {
                variant: "Q8_0",
                kind: Kind::Gguf,
                repo: "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
                revision: "eea7b2be5805a5f151f8847ede8e5f9a9284bf77",
                format: Format::Gguf8,
                bytes: Some(32_483_932_576),
                status: Status::Unverified {
                    why: "explicit only: revision and size from tools/census/stubs.tsv; no doc or gate names this file",
                },
            },
        ],
    },
    Model {
        name: "minicpm-v-4.7-35b-a3b",
        aliases: &[],
        seat: "qwen3",
        arch: "qwen35moe",
        rows: &[Row {
            variant: "Q4_K_M",
            kind: Kind::Gguf,
            repo: "openbmb/MiniCPM-V-4.7-35B-A3B-gguf",
            revision: "main",
            format: Format::Gguf4,
            bytes: None,
            status: Status::Unverified {
                why: "no revision is pinned in the repo's tables; docs/models.md names this file; size not recorded",
            },
        }],
    },
    Model {
        name: "sage-a1.5",
        aliases: &[],
        seat: "qwen3",
        arch: "qwen35moe",
        rows: &[Row {
            variant: "Q4_K_M",
            kind: Kind::Gguf,
            repo: "InternScience/Sage-A1.5-Q4_K_M-GGUF",
            revision: "main",
            format: Format::Gguf4,
            bytes: Some(21_700_000_000),
            status: Status::Unverified {
                why: "no revision is pinned in the repo's tables; docs/models.md names this file and its size (rounded to 0.1 GB)",
            },
        }],
    },
    Model {
        name: "mimo-v2.6-flash",
        aliases: &["mimo2"],
        seat: "mimo2",
        arch: "mimo2",
        rows: &[Row {
            variant: "MXFP4",
            kind: Kind::Gguf,
            repo: "ggml-org/MiMo-V2.6-Flash-MOPD-GGUF",
            revision: "e6d4bced739310a721eefad4c247ec7e0247c89d",
            format: Format::Mxfp4,
            bytes: Some(167_369_465_152),
            status: Status::Verified {
                evidence: "README Use table and docs/models.md recommend this file; revision and size from tools/census/stubs.tsv",
            },
        }],
    },
    Model {
        name: "minimax-m3.1-flash",
        aliases: &["minimaxm3"],
        seat: "minimaxm3",
        arch: "minimax-m3",
        rows: &[Row {
            variant: "nvfp4",
            kind: Kind::Folder,
            repo: "MiniMaxAI/MiniMax-M3.1-Flash",
            revision: "76faee1b4ac0f412e547781172e94548a6526a88",
            format: Format::Nvfp4,
            bytes: Some(250_100_000_000),
            status: Status::Verified {
                evidence: "docs/models.md lists this folder; revision from tools/fixture-cuts/minimaxm31.json, size from docs/models.md (rounded to 0.1 GB)",
            },
        }],
    },
    Model {
        name: "minimax-m3",
        aliases: &[],
        seat: "minimaxm3",
        arch: "minimax-m3",
        rows: &[Row {
            variant: "UD-IQ4_XS",
            kind: Kind::Gguf,
            repo: "unsloth/MiniMax-M3-GGUF",
            revision: "main",
            format: Format::Gguf4,
            bytes: Some(208_500_000_000),
            status: Status::Unverified {
                why: "docs/models.md says no gate runs on this file; no revision is pinned; size from docs/models.md (rounded to 0.1 GB)",
            },
        }],
    },
    Model {
        name: "muse-spark-open-816b-a42b",
        aliases: &["musespark"],
        seat: "musespark",
        arch: "muse-spark",
        rows: &[Row {
            variant: "UD-IQ2_XXS",
            kind: Kind::Gguf,
            repo: "unsloth/Muse-Spark-Open-816B-A42B-GGUF",
            revision: "c00670568b6c",
            format: Format::Gguf2,
            bytes: Some(225_600_000_000),
            status: Status::Unverified {
                why: "the revision is known only as the 12-character prefix in crates/model/src/arch/registry.rs; size from tools/ref/models/muse-spark.sh (rounded to 0.1 GB)",
            },
        }],
    },
    Model {
        name: "nemotron-3-nano-4b",
        aliases: &["nemotron"],
        seat: "nemotron",
        arch: "nemotron_h",
        rows: &[Row {
            variant: "Q4_K_M",
            kind: Kind::Gguf,
            repo: "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
            revision: "ba223d14e45525f7fae81db77ea8cabeb2fc6c25",
            format: Format::Gguf4,
            bytes: Some(2_837_072_864),
            status: Status::Verified {
                evidence: "docs/models.md lists this file; revision and size from crates/model/src/arch/registry.rs",
            },
        }],
    },
    Model {
        name: "nemotron-3.5-bolt-74b-a9b",
        aliases: &[],
        seat: "nemotron",
        arch: "nemotron_h_moe",
        rows: &[Row {
            variant: "24GB",
            kind: Kind::Gguf,
            repo: "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF",
            revision: "e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246",
            format: Format::Gguf2,
            bytes: None,
            status: Status::Unverified {
                why: "crates/model/src/arch/registry.rs: the maintainers' copy was not compared with this revision; size not recorded",
            },
        }],
    },
    Model {
        name: "nemotron-3.5-lightning-30b-a3b",
        aliases: &[],
        seat: "nemotron",
        arch: "nemotron_h_moe",
        rows: &[Row {
            variant: "UD-Q4_K_XL",
            kind: Kind::Gguf,
            repo: "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
            revision: "f2d3fe3694501008786e81e5f20360cbf715496a",
            format: Format::Gguf4,
            bytes: None,
            status: Status::Unverified {
                why: "revision from a header fixture (crates/hf/tests/fixtures/checkpoint-headers/README.md), the maintainers' copy was not compared with it; size not recorded",
            },
        }],
    },
    Model {
        name: "solar-open2-250b",
        aliases: &["solar2"],
        seat: "solar2",
        arch: "solar-open2",
        rows: &[Row {
            variant: "Q4_K_M",
            kind: Kind::Gguf,
            repo: "llamaraspberryrabbit/Solar-Open2-250B-GGUF",
            revision: "6e78cf22866264a9e3052c2f02ca056bf79c9ac9",
            format: Format::Gguf4,
            bytes: Some(151_800_000_000),
            status: Status::Verified {
                evidence: "README Use table and docs/models.md; revision from crates/model/src/arch/registry.rs, size from docs/models.md (rounded to 0.1 GB)",
            },
        }],
    },
];
