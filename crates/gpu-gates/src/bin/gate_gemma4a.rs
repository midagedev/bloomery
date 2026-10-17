//! The pinned Gemma4a tower's soft rows, chunk boundaries and shared media reserve.

#[allow(dead_code, reason = "shared e2e helpers serve several gates")]
#[path = "shared/e2e.rs"]
mod e2e;

use bloomery_gpu::Gpu;
use bloomery_gpu_gates::{GateError, sha256};
use bloomery_gpu_gemma4a::{Encoder, arch::gemma4a::attention};
use gguf::Gguf;
use refset::arch::gemma_embedding2::{
    AUDIO_TOWER, AUDIO_TOWER_CPU, MMPROJ_BF16, MMPROJ_BF16_SHA256, REL_FLOOR, TWIN,
};
use refset::ik::{RefManifest, ref_tensor_of_in};
use vision::arch::gemma4a::Mel;

fn compare(
    name: &str,
    got: &[f32],
    card: &RefManifest,
    cpu: &RefManifest,
) -> Result<(), GateError> {
    let want = e2e::tap(card, name)?;
    let twin = e2e::tap(cpu, name)?;
    assert_eq!(got.len(), want.len(), "{name}: width");
    assert_eq!(twin.len(), want.len(), "{name}: twin width");
    assert!(
        got.iter().chain(&want).chain(&twin).all(|v| v.is_finite()),
        "{name}: finite values"
    );
    let relative = e2e::rel(got, &want);
    let twin_relative = e2e::rel(&twin, &want);
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let twin_abs = twin
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let peak = want.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    // PIN(2026-10-16): the encoder families' CPU-twin rule, also for absolute distance.
    let relative_band = REL_FLOOR.max(TWIN * twin_relative);
    let absolute_band = (REL_FLOOR * f64::from(peak)).max(TWIN * f64::from(twin_abs));
    eprintln!(
        "{name}: relative={relative:.9e} band={relative_band:.9e} max_abs={max_abs:.9e} band={absolute_band:.9e} twin_relative={twin_relative:.9e}"
    );
    assert!(relative <= relative_band, "{name}: relative distance");
    assert!(
        f64::from(max_abs) <= absolute_band,
        "{name}: absolute distance"
    );
    Ok(())
}

fn main() -> Result<(), GateError> {
    let _levers = bloomery_levers::at_main(&[])?;
    assert_eq!(
        sha256(std::path::Path::new(MMPROJ_BF16))?,
        MMPROJ_BF16_SHA256,
        "audio projector identity"
    );
    let file = Gguf::open(MMPROJ_BF16)?;
    let gpu = Gpu::new()?;
    let mut encoder = Encoder::load(&gpu, &file)?;
    assert_eq!(
        encoder.weight_bytes(),
        1_222_510_336,
        "pinned projector expanded weights"
    );
    assert_eq!(
        encoder.scratch_bytes(),
        124_521_984,
        "tower-owned scratch only"
    );
    assert_eq!(
        encoder.reserve_bytes(),
        encoder.weight_bytes() + encoder.scratch_bytes()
    );
    let beside = bloomery_gpu_gates::beside::Beside {
        image: Some(encoder.reserve_bytes()),
        embedding: None,
    };
    assert_eq!(
        beside.named().collect::<Vec<_>>(),
        vec![(
            bloomery_gpu_gates::beside::IMAGE_ENCODER,
            encoder.reserve_bytes()
        )]
    );
    eprintln!(
        "audio weights={} scratch={} reserve={} B",
        encoder.weight_bytes(),
        encoder.scratch_bytes(),
        encoder.reserve_bytes()
    );
    for bad in [
        Mel {
            frames: 0,
            data: vec![],
        },
        Mel {
            frames: 3000,
            data: vec![0.0; 3000 * 128],
        },
        Mel {
            frames: 1,
            data: vec![0.0; 127],
        },
        Mel {
            frames: 1,
            data: vec![0.0; 129],
        },
        Mel {
            frames: 1,
            data: vec![f32::NAN; 128],
        },
        Mel {
            frames: 1,
            data: vec![f32::INFINITY; 128],
        },
        Mel {
            frames: 1,
            data: vec![f32::NEG_INFINITY; 128],
        },
    ] {
        let error = encoder.encode(&bad).err().expect("malformed mel accepted");
        assert!(
            error.to_string().contains("gemma4a mel"),
            "malformed mel: {error}"
        );
    }
    let overflow = Mel {
        frames: 1,
        data: vec![f32::MAX; 128],
    };
    let error = encoder
        .encode(&overflow)
        .err()
        .expect("arithmetic overflow accepted");
    assert!(
        matches!(
            error,
            bloomery_gpu::GpuError::Fault {
                what: "gemma4a output",
                ..
            }
        ),
        "overflow: {error}"
    );
    eprintln!("audio overflow refused: {error}");
    gpu.clear_fault()?;
    let dir = AUDIO_TOWER.path(AUDIO_TOWER.sets[0]);
    let man = RefManifest::open(&dir, &AUDIO_TOWER)?;
    let cpu_dir = AUDIO_TOWER_CPU.path(AUDIO_TOWER_CPU.sets[0]);
    let cpu = RefManifest::open(&cpu_dir, &AUDIO_TOWER_CPU)?;
    for (set, device) in [(&man, "gpu"), (&cpu, "cpu")] {
        assert!(
            set.header
                .other
                .iter()
                .any(|s| s == &format!("# device\t{device}")),
            "reference device"
        );
        assert!(
            set.header
                .other
                .iter()
                .any(|s| s == "# tapped-output\tbit-identical"),
            "reference clean/tapped equality"
        );
    }
    let cases = [
        ("minimum/0", 1),
        ("sine/0", 99),
        ("silence/0", 199),
        ("before-30s/0", 2999),
        ("after-30s/0", 2999),
        ("after-30s/1", 2),
    ];
    assert_eq!(
        man.tensors.len(),
        18 * cases.len(),
        "the full audio reference set"
    );
    assert_eq!(cpu.tensors.len(), man.tensors.len(), "the full CPU twin");
    for (name, frames) in cases {
        let row = man.tensor(&format!("{name}/mel"), 0)?;
        row.expect(name, "f32", [frames as u64, 128, 1, 1], "NONE")?;
        let mel = Mel {
            frames,
            data: ref_tensor_of_in(&dir, row)?,
        };
        assert!(
            e2e::same_bits(&mel.data, &e2e::tap(&cpu, &format!("{name}/mel"))?),
            "{name}: twin mel"
        );
        let (position, mask) = attention::inputs(mel.tokens());
        for (key, input) in [("pos_emb", position), ("kq_mask", mask)] {
            assert!(
                e2e::same_bits(&input, &e2e::tap(&man, &format!("{name}/{key}"))?),
                "{name}: {key} bits"
            );
        }
        let clean = encoder.encode(&mel)?;
        let mut taps = Vec::new();
        let tapped = encoder.encode_tapped(&mel, &mut |stage, width, values| {
            assert_eq!(values.len(), clean.tokens * width);
            assert!(values.iter().all(|v| v.is_finite()), "non-finite tap");
            taps.push((stage.to_owned(), width, values.to_vec()));
        })?;
        assert_eq!(taps.len(), 14, "SSCP, every block and output projection");
        assert!(
            e2e::same_bits(&clean.values, &tapped.values),
            "{name}: clean/tapped output"
        );
        for (stage, width, values) in taps {
            let key = format!("{name}/{stage}");
            man.tensor(&key, 0)?.expect(
                &key,
                "f32",
                [width as u64, clean.tokens as u64, 1, 1],
                "NONE",
            )?;
            compare(&key, &values, &man, &cpu)?;
        }
        let key = format!("{name}/embd");
        man.tensor(&key, 0)?
            .expect(name, "f32", [512, clean.tokens as u64, 1, 1], "NONE")?;
        compare(&key, &clean.values, &man, &cpu)?;
        eprintln!("{name}: tokens={} clean/tapped=bit-identical", clean.tokens);
    }
    eprintln!("gemma4a tower PASS");
    Ok(())
}
