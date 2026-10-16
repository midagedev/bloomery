use super::{
    ARCH, MMPROJ_BF16, MMPROJ_BF16_SHA256, MODEL, MODEL_SHA256, PROFILE, SERVER, SERVER_CPU, TEXT,
    TEXT_CPU, TOWER, TOWER_CPU, VIDEO, VIDEO_CPU,
};
use crate::RefError;
use crate::emb2::tests::{check, header, remove, set_dir};
use crate::emb2::{Kind, LCPP_BUILD};
use crate::family::Family;

#[test]
fn audio_mel_requires_its_pcm_identity_architecture_build_and_trailer() -> Result<(), RefError> {
    use super::{AUDIO_MEL, AUDIO_PCM};
    let dir = set_dir("mel")?;
    let own = vec![
        ("model".into(), AUDIO_PCM.into()),
        ("arch".into(), "gemma4a".into()),
        ("build".into(), LCPP_BUILD.into()),
    ];
    assert_eq!(check(&dir, &AUDIO_MEL, &own, true)?, AUDIO_PCM);
    for (i, value) in [(0, MODEL), (1, ARCH), (2, "other-build")] {
        let mut other = own.clone();
        other[i].1 = value.into();
        assert!(check(&dir, &AUDIO_MEL, &other, true).is_err());
    }
    assert!(matches!(
        check(&dir, &AUDIO_MEL, &own, false),
        Err(RefError::Unfinished { .. })
    ));
    remove(&dir)
}

const Q8_MMPROJ: &str = "/models/embeddinggemma-2/mmproj-embeddinggemma-2-Q8_0.gguf";
const Q8_MMPROJ_SHA256: &str = "c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f";

/// A kind of set, the family checking it, the header lines replaced and the
/// text the stale error names.
type FileCase<'a> = (Kind, &'a Family, Vec<(&'a str, &'a str)>, &'a str);

/// A kind of set, the family checking it, the header line replaced, whether
/// the trailer is written, and the text the error names.
type RefusedCase<'a> = (Kind, &'a Family, (&'a str, &'a str), bool, &'a str);

const ALL: [(Kind, &Family); 8] = [
    (Kind::Text, &TEXT),
    (Kind::TextCpu, &TEXT_CPU),
    (Kind::Tower, &TOWER),
    (Kind::TowerCpu, &TOWER_CPU),
    (Kind::Server, &SERVER),
    (Kind::ServerCpu, &SERVER_CPU),
    (Kind::Server, &VIDEO),
    (Kind::ServerCpu, &VIDEO_CPU),
];

/// Each family takes a set of its own kind and device, of the pinned files,
/// commit and architecture, and states the file the set names as `# model`:
/// the text file, or the projector for a tower set.
#[test]
fn each_family_takes_a_set_of_its_kind() -> Result<(), RefError> {
    let dir = set_dir("own")?;
    for (kind, family) in ALL {
        let want = if matches!(kind, Kind::Tower | Kind::TowerCpu) {
            MMPROJ_BF16
        } else {
            MODEL
        };
        assert_eq!(
            check(&dir, family, &header(&PROFILE, ARCH, kind, &[]), true)?,
            want,
            "{kind:?}"
        );
    }
    remove(&dir)
}

/// A set of another kind or device is foreign to the family and names the
/// line: the CPU twin's set at the card set's place and the other way round,
/// and the server's vectors at the text set's.
#[test]
fn a_set_of_another_kind_or_device_is_foreign() -> Result<(), RefError> {
    let dir = set_dir("kind")?;
    let cases: [(Kind, &Family, &str); 7] = [
        (Kind::TextCpu, &TEXT, "device \"cpu\""),
        (Kind::Text, &TEXT_CPU, "device \"cuda\""),
        (Kind::TowerCpu, &TOWER, "device \"cpu\""),
        (Kind::Tower, &TOWER_CPU, "device \"cuda\""),
        (Kind::Server, &TEXT, "emb2 \"server\""),
        (Kind::ServerCpu, &SERVER, "device \"cpu\""),
        (Kind::Server, &SERVER_CPU, "device \"cuda\""),
    ];
    for (kind, family, want) in cases {
        match check(&dir, family, &header(&PROFILE, ARCH, kind, &[]), true) {
            Err(e @ RefError::Foreign { .. }) => {
                assert!(e.to_string().contains(want), "{kind:?}: {e}")
            }
            r => panic!("a {kind:?} set as {}: {r:?}", family.name),
        }
    }
    remove(&dir)
}

/// A set of another file is stale and names it: the bf16 text file, a text
/// file of another sha256, the Q8_0 projector (as a tower set's `# model` and
/// as a server set's `# mmproj`), and the bf16 projector of another sha256.
#[test]
fn a_set_of_another_file_is_stale() -> Result<(), RefError> {
    let dir = set_dir("file")?;
    let bf16 = "/models/embeddinggemma-2/embeddinggemma-2-BF16.gguf";
    let other_sha = "0".repeat(64);
    let q8_line = format!("{Q8_MMPROJ}\tsha256\t{Q8_MMPROJ_SHA256}");
    let path_line = format!("{Q8_MMPROJ}\tsha256\t{MMPROJ_BF16_SHA256}");
    let sha_line = format!("{MMPROJ_BF16}\tsha256\t{other_sha}");
    let text_line = format!("{bf16}\tsha256\t{MODEL_SHA256}");
    let cases: Vec<FileCase> = vec![
        (Kind::Text, &TEXT, vec![("model", bf16)], bf16),
        (
            Kind::TextCpu,
            &TEXT_CPU,
            vec![("model_sha256", &other_sha)],
            &other_sha,
        ),
        (
            Kind::Tower,
            &TOWER,
            vec![("model", Q8_MMPROJ), ("mmproj", &q8_line)],
            Q8_MMPROJ,
        ),
        (
            Kind::TowerCpu,
            &TOWER_CPU,
            vec![("mmproj", &sha_line)],
            &other_sha,
        ),
        (Kind::Tower, &TOWER, vec![("text_model", &text_line)], bf16),
        (Kind::Server, &SERVER, vec![("mmproj", &q8_line)], Q8_MMPROJ),
        (
            Kind::Server,
            &SERVER,
            vec![("mmproj", &path_line)],
            Q8_MMPROJ,
        ),
    ];
    for (kind, family, with, want) in cases {
        match check(&dir, family, &header(&PROFILE, ARCH, kind, &with), true) {
            Err(e @ RefError::Stale { .. }) => {
                assert!(e.to_string().contains(want), "{kind:?} {with:?}: {e}")
            }
            r => panic!("{kind:?} {with:?}: {r:?}"),
        }
    }
    remove(&dir)
}

/// A set of another commit or architecture, one without its trailer, and one
/// without the line its kind is read from are refused by name.
#[test]
fn a_set_of_another_build_arch_or_shape_is_refused() -> Result<(), RefError> {
    let dir = set_dir("build")?;
    let cases: [RefusedCase; 6] = [
        (Kind::Text, &TEXT, ("build", "9cf967a0"), true, "build"),
        (Kind::Tower, &TOWER, ("arch", ARCH), true, "arch"),
        (
            Kind::Server,
            &SERVER,
            ("build", LCPP_BUILD),
            false,
            "no `# complete` trailer",
        ),
        (Kind::Text, &TEXT, ("emb2", ""), true, "no `# emb2` line"),
        (
            Kind::TowerCpu,
            &TOWER_CPU,
            ("device", "cpu"),
            true,
            "want cuda|cpu",
        ),
        (
            Kind::Server,
            &SERVER,
            ("mmproj", ""),
            true,
            "no `# mmproj` line",
        ),
    ];
    for (kind, family, with, complete, want) in cases {
        let e = check(
            &dir,
            family,
            &header(&PROFILE, ARCH, kind, &[with]),
            complete,
        )
        .expect_err("refused")
        .to_string();
        assert!(e.contains(want), "{kind:?} {with:?} {complete}: {e}");
    }
    remove(&dir)
}

/// A complete tower set needs a captured block, on either device.
#[test]
fn a_tower_without_block_taps_is_refused() -> Result<(), RefError> {
    let dir = set_dir("block-taps")?;
    for (kind, family) in [(Kind::Tower, &TOWER), (Kind::TowerCpu, &TOWER_CPU)] {
        check(&dir, family, &header(&PROFILE, ARCH, kind, &[]), true)?;
        let path = dir.join("MANIFEST.tsv");
        let manifest =
            std::fs::read_to_string(&path).map_err(|e| RefError::missing(&path, e.to_string()))?;
        std::fs::write(
            &path,
            manifest.replace("small/layer_out-0", "small/projected"),
        )
        .map_err(|e| RefError::missing(&path, e.to_string()))?;
        let e = family
            .check_set(&dir)
            .expect_err("no block tap")
            .to_string();
        assert!(e.contains("layer_out-0"), "{kind:?}: {e}");
    }
    remove(&dir)
}
