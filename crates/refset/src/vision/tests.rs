use super::VisionSet;
use crate::RefError;
use crate::family::{Family, Identity};
use std::path::Path;

static FAMILY: Family = Family {
    name: "test-vision",
    sets: &[],
    resolve: None,
    recipe: "",
    identity: Identity::Checkpoint { revision: "r1" },
    arch: None,
    build: None,
    runs: None,
    draft_runs: None,
    consumers: &[],
};

fn write_set(dir: &Path, checkpoint: &str, image: &str, complete: bool) {
    let mut lines = vec![
        "# oracle\tdump_vision.py".to_string(),
        format!("# checkpoint\t{checkpoint}\t/models/ckpt"),
        "# mmproj\t/models/m.gguf\tsha256\tabc".to_string(),
        "# image_token_id\t129264".to_string(),
        "# sensitivity columns\ttap max_rel rms_rel differ\tthe reference against itself".to_string(),
        "# sensitivity\tblk0\t4.926e-03\t8.718e-04\t1.116e-01".to_string(),
        "# image columns\tname sha256 w h best_w best_h n_vit_h n_vit_w n_llm_h n_llm_w n_tokens resized_w resized_h off_x off_y".to_string(),
        image.to_string(),
        "# file columns\tname kind dtype shape bytes md5".to_string(),
        "file\tg.blk0.bf16\tblk0\tbf16\t1521x1024\t3115008\tffff".to_string(),
    ];
    if complete {
        lines.push("# complete\t1\t1\t1".to_string());
    }
    std::fs::write(dir.join("MANIFEST.tsv"), lines.join("\n") + "\n")
        .unwrap_or_else(|e| panic!("{e}"));
    std::fs::write(
        dir.join("plans.tsv"),
        "# kind\tw\th\tn_llm_h\tn_llm_w\tbest_h\tbest_w\tn_tokens\tresized_w\tresized_h\toff_x\toff_y\n\
         image\t777\t513\t13\t19\t518\t784\t262\t784\t518\t0\t0\n",
    )
    .unwrap_or_else(|e| panic!("{e}"));
}

const IMAGE: &str = "image\tg.png\tfd13\t448\t448\t546\t546\t39\t39\t13\t13\t184\t546\t546\t0\t0";

/// The manifest reads by its column lines — images, files with their
/// shapes, the sensitivity rows — and `plans.tsv` by its `# kind` line. A
/// set of another checkpoint revision is `Stale`, one without its trailer
/// `Unfinished`, and a garbage count `Malformed` naming the field.
#[test]
fn a_vision_set_reads_by_its_column_lines() {
    let dir = std::env::temp_dir().join(format!("bloomery-vision-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
    write_set(&dir, "org/model@r1", IMAGE, true);
    let set = VisionSet::open(&dir, &FAMILY).unwrap_or_else(|e| panic!("{e}"));
    let img = &set.images[0];
    assert_eq!(
        (img.stem(), img.n_vit_h, img.n_vit_w, img.n_tokens),
        ("g", 39, 39, 184)
    );
    assert_eq!(
        (set.files[0].shape.as_slice(), set.files[0].bytes),
        (&[1521, 1024][..], 3115008)
    );
    assert_eq!(
        (set.sensitivity[0].tap.as_str(), set.sensitivity[0].rms_rel),
        ("blk0", 8.718e-04)
    );
    assert_eq!(
        (set.mmproj_sha256.as_str(), set.image_token_id),
        ("abc", 129264)
    );
    let plans = set.plans().unwrap_or_else(|e| panic!("{e}"));
    assert_eq!((plans[0].w, plans[0].best_w, plans[0].off_y), (777, 784, 0));

    write_set(&dir, "org/model@r2", IMAGE, true);
    match VisionSet::open(&dir, &FAMILY) {
        Err(RefError::Stale {
            dumped_from, runs, ..
        }) => {
            assert_eq!(
                (dumped_from.as_str(), runs.as_str()),
                ("org/model@r2", "r1")
            );
        }
        other => panic!("another revision: {other:?}"),
    }
    write_set(&dir, "org/model@r1", IMAGE, false);
    assert!(matches!(
        VisionSet::open(&dir, &FAMILY),
        Err(RefError::Unfinished { .. })
    ));
    write_set(
        &dir,
        "org/model@r1",
        &IMAGE.replace("\t39\t39\t", "\tx\t39\t"),
        true,
    );
    match VisionSet::read(&dir) {
        Err(RefError::Malformed { what, .. }) => {
            assert!(what.starts_with("n_vit_h \"x\""), "{what}")
        }
        other => panic!("a garbage count: {other:?}"),
    }
    std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
}
