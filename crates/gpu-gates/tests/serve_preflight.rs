use std::process::Command;
use std::time::{Duration, Instant};

use bloomery_gpu_gates::serve_client::Served;

#[test]
fn invalid_arguments_end_before_the_driver_or_fetch() {
    for (name, tail, alias) in [
        ("plain", vec!["--ctx-size", "abc"], None),
        ("env-alias", vec!["--ctx-size", "abc"], Some("--levers")),
        (
            "list-alias",
            vec!["--alias", "--list-devices", "--ctx-size", "abc"],
            None,
        ),
        (
            "schema-alias",
            vec!["--alias", "--records-schema", "--ctx-size", "abc"],
            None,
        ),
        (
            "version-alias",
            vec!["--alias", "--version", "--ctx-size", "abc"],
            None,
        ),
    ] {
        refused(name, "qwen38", &tail, alias, "--ctx-size \"abc\"");
    }
}

#[test]
fn a_command_line_lever_alias_is_a_value() {
    refused(
        "lever-alias",
        "qwen38",
        &["--alias", "--levers", "--ctx-size", "abc"],
        None,
        "--ctx-size \"abc\"",
    );
}

#[test]
fn a_command_line_lever_alias_with_placement_is_a_value() {
    refused(
        "placed-lever-alias",
        "qwen38",
        &["--alias", "--levers", "-ngl", "all", "--ctx-size", "abc"],
        None,
        "--ctx-size \"abc\"",
    );
}

#[test]
fn nemotron_refuses_multiple_sequences_before_the_fetch() {
    refused(
        "nemotron-parallel",
        "nemotron",
        &["--parallel", "2"],
        None,
        "--parallel 2: both nemotron bodies hold one sequence",
    );
}

#[test]
fn a_slot_context_is_checked_before_the_fetch() {
    refused(
        "qwen3-floor",
        "qwen3",
        &["--ctx-size", "1"],
        None,
        "--parallel 1 of a --ctx of 1",
    );
}

#[test]
fn the_decision_seat_refuses_an_embedding_model_before_the_fetch() {
    refused(
        "decide-side",
        "decide",
        &["--embd-hf", "o/embedding"],
        None,
        "--embd-model: the decide seat answers its row's POST routes and no embeddings route",
    );
}

#[cfg(not(feature = "vision"))]
#[test]
fn a_build_without_vision_refuses_images_before_the_fetch() {
    for model in ["ds41", "qwen38", "qwen3", "glm", "decide"] {
        for (name, tail) in [
            ("projector", vec!["--mmproj", "p.gguf"]),
            ("limit", vec!["--image-max-tokens", "1024"]),
        ] {
            refused(
                &format!("{model}-{name}"),
                model,
                &tail,
                None,
                bloomery_gpu_gates::vision_seat::NO_TOWER,
            );
        }
    }
    for (name, tail) in [
        ("projector", vec!["--embd-mmproj", "p.gguf"]),
        ("limit", vec!["--embd-image-max-tokens", "1024"]),
    ] {
        let mut args = vec!["--embd-hf", "o/embedding"];
        args.extend(tail);
        refused(
            &format!("side-{name}"),
            "qwen38",
            &args,
            None,
            &format!(
                "embedding image flags: {}",
                bloomery_gpu_gates::vision_seat::NO_TOWER
            ),
        );
    }
}

#[test]
fn disabled_images_refuse_limits_before_the_fetch() {
    for model in ["ds41", "qwen38", "qwen3", "glm", "decide"] {
        refused(
            &format!("{model}-disabled"),
            model,
            &["--no-mmproj", "--image-max-tokens", "1024"],
            None,
            if cfg!(feature = "vision") {
                "--image-min-tokens/--image-max-tokens without --mmproj: no image input to bound"
            } else {
                bloomery_gpu_gates::vision_seat::NO_TOWER
            },
        );
    }
}

#[test]
fn image_limits_refuse_invalid_counts_before_the_fetch() {
    for model in ["ds41", "qwen38", "qwen3", "glm", "decide"] {
        for (name, tail) in [
            ("zero-min", vec!["--image-min-tokens", "0"]),
            ("zero-max", vec!["--image-max-tokens", "0"]),
            (
                "inverted",
                vec!["--image-min-tokens", "9", "--image-max-tokens", "8"],
            ),
        ] {
            refused(
                &format!("{model}-{name}"),
                model,
                &tail,
                None,
                "--image-min-tokens/--image-max-tokens: image size limits:",
            );
        }
    }
    refused(
        "side-inverted",
        "qwen38",
        &[
            "--embd-hf",
            "o/embedding",
            "--embd-image-min-tokens",
            "9",
            "--embd-image-max-tokens",
            "8",
        ],
        None,
        "embedding image flags: --image-min-tokens/--image-max-tokens: image size limits:",
    );
}

#[test]
#[ignore = "requires an NVIDIA CUDA 13 driver"]
fn hw_a_device_list_alias_reaches_the_selected_seat() {
    let path = std::env::temp_dir().join(format!("serve-alias-{}.gguf", std::process::id()));
    let layout = gguf::write::Layout::new(
        &[(
            gguf::GENERAL_ARCHITECTURE.into(),
            gguf::Value::String("qwen4exp".into()),
        )],
        Vec::new(),
    )
    .unwrap();
    std::fs::write(&path, layout.header_bytes()).unwrap();
    let model = path.to_str().unwrap();
    refusal(
        "selected-seat-device-alias",
        env!("CARGO_BIN_EXE_bloomery-serve"),
        &[
            "--model",
            "qwen38",
            "-m",
            model,
            "--alias",
            "--list-devices",
            "--embd-model",
            model,
        ],
        None,
        &format!("bloomery-serve: --embd-model {model}: a qwen4exp file"),
        false,
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn a_head_asks_the_decide_seat_for_its_usage() {
    for help in ["--help", "-h"] {
        refusal(
            &format!("head{help}"),
            env!("CARGO_BIN_EXE_bloomery-serve"),
            &["--head", "h", help],
            None,
            "bloomery-serve: usage: bloomery-serve [--model decide]",
            true,
        );
    }
}

#[test]
fn stand_alone_device_list_aliases_reach_the_parser() {
    for (name, bin) in [
        ("qwen38", env!("CARGO_BIN_EXE_bloomery-serve-qwen38")),
        ("ds41", env!("CARGO_BIN_EXE_bloomery-serve-ds41")),
    ] {
        refusal(
            &format!("{name}-device-alias"),
            bin,
            &["--alias", "--list-devices", "--ctx-size", "abc"],
            None,
            &format!("bloomery-serve-{name}: --ctx-size \"abc\""),
            true,
        );
    }
}

fn refused(name: &str, model: &str, tail: &[&str], alias: Option<&str>, expected: &str) {
    let mut args = vec!["--model", model, "--hf", "o/r"];
    args.extend_from_slice(tail);
    refusal(
        name,
        env!("CARGO_BIN_EXE_bloomery-serve"),
        &args,
        alias,
        &format!("bloomery-serve: {expected}"),
        true,
    );
}

fn refusal(
    name: &str,
    bin: &str,
    args: &[&str],
    alias: Option<&str>,
    expected: &str,
    before_model: bool,
) {
    let env: Vec<_> = alias
        .map(|value| ("LLAMA_ARG_ALIAS", value))
        .into_iter()
        .collect();
    let (dir, err) = startup(name, bin, args, &env);
    let message = if before_model {
        err.as_str()
    } else {
        err.lines().last().expect("the seat refused its side model")
    };
    assert!(message.starts_with(expected), "{args:?} {alias:?}: {err}");
    clean_refusal(&dir, &err, before_model);
    std::fs::remove_dir_all(dir).unwrap();
}

fn startup(
    name: &str,
    bin: &str,
    args: &[&str],
    env: &[(&str, &str)],
) -> (std::path::PathBuf, String) {
    let dir = std::env::temp_dir().join(format!(
        "bloomery-serve-preflight-{}-{name}",
        std::process::id()
    ));
    std::fs::create_dir(&dir).unwrap();
    let cache = dir.join("cache");
    std::fs::create_dir(&cache).unwrap();
    let mut cmd = Command::new(bin);
    cmd.env_clear()
        .env("HOME", &dir)
        .env("BLOOMERY_CACHE", &cache)
        .env("HF_HUB_OFFLINE", "1");
    cmd.envs(env.iter().copied());
    let mut server = Served::spawn_cmd(cmd, args, &dir).unwrap();
    let started = Instant::now();
    let status = loop {
        if let Some(status) = server.child.try_wait().unwrap() {
            break status;
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "startup did not end"
        );
        std::thread::sleep(Duration::from_millis(10));
    };
    let err = std::fs::read_to_string(dir.join("server.err")).unwrap();
    assert_eq!(status.code(), Some(1), "{args:?} {env:?}: {err}");
    drop(server);
    (dir, err)
}

#[test]
fn preflight_source_main() {
    checked_refusal(
        "source_main",
        &["--model", "qwen38", "--hf", "invalid"],
        &[],
        "a repo is <owner>/<name>",
    );
}

#[test]
fn preflight_source_side() {
    checked_refusal(
        "source_side",
        &["--model", "qwen38", "--hf", "o/r", "--embd-hf", "invalid"],
        &[],
        "embedding flags",
    );
}

#[test]
fn preflight_source_env() {
    checked_refusal(
        "source_env",
        &["--model", "qwen38"],
        &[("LLAMA_ARG_HF_REPO", "invalid")],
        "a repo is <owner>/<name>",
    );
}

#[test]
fn preflight_source_reference() {
    checked_refusal(
        "source_reference",
        &["--model", "qwen38", "--hf", "o/r"],
        &[("BLOOMERY_REF_MODEL", "other.gguf")],
        "the model is named twice",
    );
}

#[test]
fn preflight_ds41_draft() {
    checked_refusal(
        "ds41_draft",
        &["--model", "ds41", "--hf", "o/r"],
        &[("BLOOMERY_DRAFT", "mtp")],
        "BLOOMERY_DRAFT is lookup",
    );
}

#[test]
fn preflight_ds41_width() {
    checked_refusal(
        "ds41_width",
        &["--model", "ds41", "--hf", "o/r"],
        &[("BLOOMERY_MTP_WIDTH", "cost")],
        "BLOOMERY_DRAFT is unset",
    );
}

#[test]
fn preflight_ds41_slots() {
    checked_refusal(
        "ds41_slots",
        &["--model", "ds41", "--hf", "o/r", "--parallel", "2"],
        &[("BLOOMERY_DRAFT", "dspark")],
        "--parallel 2",
    );
}

#[test]
fn preflight_ds41_trace_feed() {
    checked_refusal(
        "ds41_trace_feed",
        &["--model", "ds41", "--hf", "o/r"],
        &[("BLOOMERY_ROUTE_TRACE", "trace")],
        "BLOOMERY_ROUTE_TRACE records the step feed",
    );
}

#[test]
fn preflight_ds41_trace_slots() {
    checked_refusal(
        "ds41_trace_slots",
        &["--model", "ds41", "--hf", "o/r", "--parallel", "2"],
        &[
            ("BLOOMERY_ROUTE_TRACE", "trace"),
            ("BLOOMERY_PREFILL", "steps"),
        ],
        "BLOOMERY_ROUTE_TRACE records one sequence",
    );
}

#[test]
fn preflight_qwen38_draft() {
    checked_refusal(
        "qwen38_draft",
        &["--model", "qwen38", "--hf", "o/r"],
        &[("BLOOMERY_DRAFT", "dspark")],
        "BLOOMERY_DRAFT=dspark",
    );
}

#[test]
fn preflight_qwen38_lookup_head() {
    checked_refusal(
        "qwen38_lookup_head",
        &["--model", "qwen38", "--hf", "o/r"],
        &[
            ("BLOOMERY_DRAFT", "lookup"),
            ("BLOOMERY_MTP_HEAD_ROWS", "full"),
        ],
        "BLOOMERY_MTP_HEAD_ROWS",
    );
}

#[test]
fn preflight_qwen38_off_width() {
    checked_refusal(
        "qwen38_off_width",
        &["--model", "qwen38", "--hf", "o/r"],
        &[("BLOOMERY_DRAFT", "off"), ("BLOOMERY_MTP_WIDTH", "cost")],
        "BLOOMERY_MTP_WIDTH",
    );
}

#[test]
fn preflight_qwen38_trace_feed() {
    checked_refusal(
        "qwen38_trace_feed",
        &["--model", "qwen38", "--hf", "o/r"],
        &[("BLOOMERY_ROUTE_TRACE", "trace")],
        "BLOOMERY_ROUTE_TRACE records one-row steps",
    );
}

#[test]
fn preflight_qwen38_mtp_steps() {
    checked_refusal(
        "qwen38_mtp_steps",
        &["--model", "qwen38", "--hf", "o/r"],
        &[("BLOOMERY_DRAFT", "mtp"), ("BLOOMERY_PREFILL", "steps")],
        "BLOOMERY_PREFILL=steps",
    );
}

#[test]
fn preflight_qwen38_mtp_ctx() {
    checked_refusal(
        "qwen38_mtp_ctx",
        &["--model", "qwen38", "--hf", "o/r", "--ctx-size", "1"],
        &[("BLOOMERY_DRAFT", "mtp")],
        "a slot's context",
    );
}

#[test]
fn preflight_glm_draft() {
    checked_refusal(
        "glm_draft",
        &["--model", "glm", "--hf", "o/r"],
        &[("BLOOMERY_DRAFT", "lookup")],
        "BLOOMERY_DRAFT=lookup",
    );
}

#[test]
fn preflight_glm_mtp_ctx() {
    checked_refusal(
        "glm_mtp_ctx",
        &[
            "--model",
            "glm",
            "--hf",
            "o/r",
            "--ctx-size",
            "2",
            "--parallel",
            "1",
        ],
        &[("BLOOMERY_DRAFT", "mtp")],
        "--ctx 2",
    );
}

#[test]
fn preflight_qwen3_off_width() {
    checked_refusal(
        "qwen3_off_width",
        &["--model", "qwen3", "--hf", "o/r"],
        &[("BLOOMERY_DRAFT", "off"), ("BLOOMERY_MTP_WIDTH", "cost")],
        "BLOOMERY_MTP_WIDTH",
    );
}

#[test]
fn preflight_qwen3_dspark_slots() {
    checked_refusal(
        "qwen3_dspark_slots",
        &["--model", "qwen3", "--hf", "o/r", "--parallel", "2"],
        &[("BLOOMERY_DRAFT", "dspark")],
        "--parallel 2",
    );
}

#[test]
fn preflight_qwen3_mtp_slots() {
    checked_refusal(
        "qwen3_mtp_slots",
        &["--model", "qwen3", "--hf", "o/r", "--parallel", "2"],
        &[("BLOOMERY_DRAFT", "mtp")],
        "BLOOMERY_DRAFT=mtp",
    );
}

#[test]
fn preflight_raon1_draft() {
    checked_refusal(
        "raon1_draft",
        &["--model", "raon1", "--hf", "o/r"],
        &[("BLOOMERY_DRAFT", "lookup")],
        "BLOOMERY_DRAFT=lookup",
    );
}

#[test]
fn preflight_raon1_width() {
    checked_refusal(
        "raon1_width",
        &["--model", "raon1", "--hf", "o/r"],
        &[("BLOOMERY_DRAFT", "off"), ("BLOOMERY_MTP_WIDTH", "cost")],
        "BLOOMERY_MTP_WIDTH",
    );
}

#[test]
fn preflight_mimo2_tier_slots() {
    checked_refusal(
        "mimo2_tier_slots",
        &[
            "--model",
            "mimo2",
            "--hf",
            "o/r",
            "--place",
            "bp",
            "--parallel",
            "2",
        ],
        &[],
        "--place bp with --parallel 2",
    );
}

fn checked_refusal(name: &str, args: &[&str], env: &[(&str, &str)], expected: &str) {
    let (dir, err) = startup(name, env!("CARGO_BIN_EXE_bloomery-serve"), args, env);
    assert!(err.contains(expected), "{err}");
    clean_refusal(&dir, &err, true);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
#[ignore = "requires an NVIDIA CUDA 13 driver"]
fn hw_preflight_normal_inputs_reach_the_fetch() {
    for model in [
        "ds41",
        "qwen38",
        "glm",
        "qwen3",
        "mimo2",
        "solar2",
        "minimaxm3",
        "nemotron",
        "raon1",
        "musespark",
        "flint",
        "decide",
    ] {
        let alias = !["qwen3", "decide"].contains(&model);
        for value in [
            "ordinary",
            "--cpu_moe",
            "--cache-ram",
            "-cram",
            "--place",
            "--model",
            "--hf",
            "--head",
            "--embd-hf",
        ] {
            if !alias && value != "ordinary" {
                continue;
            }
            let mut args = vec!["--model", model, "--hf", "o/r", "--ctx-size", "8192"];
            if alias {
                args.extend(["--alias", value]);
            }
            let (dir, err) = startup(
                &format!("normal-{model}-{value}"),
                env!("CARGO_BIN_EXE_bloomery-serve"),
                &args,
                &[],
            );
            assert!(err.contains("first start"), "{model} {value}: {err}");
            assert!(err.contains("HF_HUB_OFFLINE"), "{model} {value}: {err}");
            std::fs::remove_dir_all(dir).unwrap();
        }
    }
    let qwen38_window =
        (<bloomery_gpu::arch::qwen3moe::Body38 as app::mtp::MtpBody>::VERIFY_ROWS + 1).to_string();
    for (i, (model, tail, env)) in [
        ("ds41", &["--ctx-size", "2", "--parallel", "1"][..], &[][..]),
        ("ds41", &["--ctx-size", "4", "--parallel", "2"][..], &[][..]),
        (
            "qwen38",
            &["--ctx-size", qwen38_window.as_str(), "--parallel", "1"][..],
            &[("BLOOMERY_DRAFT", "mtp")][..],
        ),
        (
            "qwen38",
            &["--ctx-size", "8192"][..],
            &[("BLOOMERY_DRAFT", "lookup")][..],
        ),
        (
            "glm",
            &["--ctx-size", "3", "--parallel", "1"][..],
            &[("BLOOMERY_DRAFT", "mtp")][..],
        ),
        (
            "glm",
            &["--ctx-size", "1", "--parallel", "1"][..],
            &[("BLOOMERY_DRAFT", "off")][..],
        ),
        (
            "qwen3",
            &["--ctx-size", "8192", "--parallel", "1"][..],
            &[("BLOOMERY_DRAFT", "dspark")][..],
        ),
        (
            "qwen3",
            &["--ctx-size", "8192", "--parallel", "1"][..],
            &[("BLOOMERY_DRAFT", "mtp")][..],
        ),
        (
            "raon1",
            &["--ctx-size", "1", "--parallel", "1"][..],
            &[("BLOOMERY_DRAFT", "off")][..],
        ),
        (
            "mimo2",
            &["--ctx-size", "1", "--parallel", "1", "--place", "bp"][..],
            &[][..],
        ),
        (
            "mimo2",
            &["--ctx-size", "2", "--parallel", "2", "--place", "a"][..],
            &[][..],
        ),
        (
            "glm",
            &["--ctx-size", "8192", "--prefill", "steps"][..],
            &[("BLOOMERY_RESIDENCY", "off")][..],
        ),
        (
            "mimo2",
            &["--ctx-size", "8192", "--prefill", "steps"][..],
            &[("BLOOMERY_RESIDENCY", "off")][..],
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let mut args = vec!["--model", model, "--hf", "o/r"];
        args.extend_from_slice(tail);
        let (dir, err) = startup(
            &format!("boundary-{model}-{i}"),
            env!("CARGO_BIN_EXE_bloomery-serve"),
            &args,
            env,
        );
        assert!(
            err.contains("first start") && err.contains("HF_HUB_OFFLINE"),
            "{model} {tail:?} {env:?}: {err}"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[cfg(feature = "vision")]
    for model in ["ds41", "qwen38", "glm", "qwen3", "decide"] {
        for side in [false, true] {
            let mut args = vec!["--model", model, "--hf", "o/r:Q8_0"];
            if side {
                if model == "decide" {
                    continue;
                }
                args.extend(["--embd-hf", "o/side", "--embd-image-max-tokens", "1024"]);
            } else {
                args.extend(["--image-max-tokens", "1024"]);
            }
            let (dir, err) = startup(
                &format!("{model}-repo-limits-{side}"),
                env!("CARGO_BIN_EXE_bloomery-serve"),
                &args,
                &[],
            );
            assert!(
                err.contains("first start") && err.contains("HF_HUB_OFFLINE"),
                "{err}"
            );
            std::fs::remove_dir_all(dir).unwrap();
        }
    }
}

#[cfg(feature = "vision")]
#[test]
fn preflight_local_side_limits() {
    checked_refusal(
        "local-side-limits",
        &[
            "--model",
            "qwen38",
            "--hf",
            "o/r",
            "--embd-model",
            "side.gguf",
            "--embd-image-max-tokens",
            "1024",
        ],
        &[],
        "no image input to bound",
    );
}

#[cfg(feature = "vision")]
#[test]
fn preflight_explicit_projector_feed() {
    for model in ["ds41", "qwen38"] {
        checked_refusal(
            &format!("{model}-projector-feed"),
            &["--model", model, "--hf", "o/r", "--mmproj", "p.gguf"],
            &[("BLOOMERY_PREFILL", "steps")],
            "--mmproj",
        );
    }
}

#[test]
fn preflight_glm_off_width() {
    checked_refusal(
        "glm-off-width",
        &["--model", "glm", "--hf", "o/r"],
        &[("BLOOMERY_DRAFT", "off"), ("BLOOMERY_MTP_WIDTH", "cost")],
        "BLOOMERY_MTP_WIDTH",
    );
}

#[test]
fn preflight_residency_steps() {
    for model in ["glm", "mimo2"] {
        checked_refusal(
            &format!("{model}-residency-steps"),
            &["--model", model, "--hf", "o/r", "--prefill", "steps"],
            &[("BLOOMERY_RESIDENCY", "mid-p8-s2")],
            "is refused beside --prefill steps",
        );
    }
}

#[test]
fn preflight_stand_alone_lever_aliases() {
    for (model, bin) in [
        ("ds41", env!("CARGO_BIN_EXE_bloomery-serve-ds41")),
        ("qwen38", env!("CARGO_BIN_EXE_bloomery-serve-qwen38")),
    ] {
        refusal(
            &format!("{model}-lever-alias"),
            bin,
            &["--alias", "--levers", "--ctx-size", "abc"],
            None,
            &format!("bloomery-serve-{model}: --ctx-size \"abc\""),
            true,
        );
    }
}

fn clean_refusal(dir: &std::path::Path, err: &str, before_model: bool) {
    for diagnostic in ["CUDA", "libcuda", "hf offline", "hf file", "hf set"] {
        assert!(!err.contains(diagnostic), "{diagnostic}: {err}");
    }
    if before_model {
        assert!(!err.contains("first start"), "{err}");
    }
    assert_eq!(std::fs::read_dir(dir.join("cache")).unwrap().count(), 0);
    assert!(std::fs::read(dir.join("server.out")).unwrap().is_empty());
}

#[test]
fn preflight_missing_model_source() {
    for model in [
        "ds41",
        "qwen38",
        "glm",
        "qwen3",
        "mimo2",
        "solar2",
        "minimaxm3",
        "nemotron",
        "raon1",
        "musespark",
        "flint",
        "decide",
    ] {
        checked_refusal(
            &format!("{model}-missing-source"),
            &["--model", model],
            &[],
            "no model file",
        );
    }
}

#[test]
fn preflight_every_seat_checks_its_slot_context() {
    for model in [
        "ds41",
        "qwen38",
        "glm",
        "qwen3",
        "mimo2",
        "solar2",
        "minimaxm3",
        "nemotron",
        "raon1",
        "musespark",
        "flint",
        "decide",
    ] {
        checked_refusal(
            &format!("{model}-zero-context"),
            &["--model", model, "--hf", "o/r", "--ctx-size", "0"],
            &[],
            "--",
        );
    }
}

#[test]
fn preflight_template_file() {
    for model in [
        "ds41",
        "qwen38",
        "glm",
        "qwen3",
        "mimo2",
        "solar2",
        "minimaxm3",
        "nemotron",
        "raon1",
        "musespark",
        "flint",
    ] {
        checked_refusal(
            &format!("{model}-template"),
            &[
                "--model",
                model,
                "--hf",
                "o/r",
                "--chat-template-file",
                "absent-template.jinja",
            ],
            &[],
            "--chat-template-file absent-template.jinja",
        );
    }
}

#[test]
fn preflight_explicit_tier_capacity() {
    for model in ["nemotron", "raon1"] {
        checked_refusal(
            &format!("{model}-tiers"),
            &["--model", model, "--hf", "o/r", "--place", "bp"],
            &[],
            "expert tier cards",
        );
    }
}

#[cfg(feature = "vision")]
#[test]
fn preflight_local_primary_limits() {
    for model in ["ds41", "qwen38", "glm", "qwen3", "decide"] {
        checked_refusal(
            &format!("{model}-local-limits"),
            &[
                "--model",
                model,
                "-m",
                "absent.gguf",
                "--image-max-tokens",
                "1024",
            ],
            &[],
            "no image input to bound",
        );
    }
}

#[test]
fn preflight_each_common_context_caller_uses_the_split_owner() {
    for model in [
        "qwen38",
        "glm",
        "mimo2",
        "solar2",
        "minimaxm3",
        "nemotron",
        "raon1",
        "musespark",
        "flint",
    ] {
        checked_refusal(
            &format!("{model}-split-context"),
            &[
                "--model",
                model,
                "--hf",
                "o/r",
                "--ctx-size",
                "1",
                "--parallel",
                "2",
            ],
            &[],
            "--ctx-size 1 --parallel 2",
        );
    }
}

#[test]
fn preflight_hosted_pools_refuse_unhosted_words() {
    for model in ["glm", "solar2", "flint", "minimaxm3", "musespark"] {
        checked_refusal(
            &format!("{model}-unhosted"),
            &["--model", model, "--hf", "o/r"],
            &[("BLOOMERY_RESIDENCY", "mid-p0-s1-unhosted")],
            "this load hosts every churn pool",
        );
    }
}
