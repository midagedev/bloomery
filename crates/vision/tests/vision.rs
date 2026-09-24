//! Vision host gates against the official V4.1 checkpoint's own preprocessing
//! (`tools/ref/vision/dump-vision.sh`, set `$BLOOMERY_DATA/ref-vision/deepseek41v`) and against the
//! encoder file that set names. `just gate-vision` runs them on the box.
//!
//! * plan: `plan_image_grid` and the pad geometry equal the reference's for every image and every
//!   size of the set's table, integer for integer;
//! * preprocess: the padded u8 image and the bf16 patch tensor equal the reference's, byte for byte,
//!   for every image — resampled or not;
//! * span: ids and types equal `image_token_types`;
//! * mmproj: the file's header reads to the hyperparameters and every tensor is named.

mod common;

use common::{Manifest, i32s, image_path, sha256_file, u16s};
use gguf::Gguf;
use vision::arch::deepseek41v::{self, Hparams, tensors};
use vision::grid::GridParams;
use vision::preprocess::{padded, to_patches};
use vision::resample::PadGeometry;
use vision::{GridPlan, Rgb8, image_span, plan_image_grid};

/// The reference config's resize parameters; the mmproj gate proves the file declares the same.
const V41: GridParams = GridParams {
    patch: 14,
    downsample: 3,
    max_tokens: 1024,
    min_pixels: 295_936,
};

fn decode(row: &common::ImageRow) -> Rgb8 {
    let path = image_path(&row.name);
    let got = sha256_file(&path);
    assert_eq!(
        got,
        row.sha256,
        "{}: the committed image is not the one the set was dumped from",
        path.display()
    );
    let bytes = std::fs::read(&path).expect("read image");
    let img = Rgb8::from_png(&bytes).unwrap_or_else(|e| panic!("{}: {e}", row.name));
    assert_eq!((img.width, img.height), (row.w, row.h), "{}", row.name);
    img
}

#[test]
#[ignore = "reads the oracle set on the box"]
fn hw_plan_matches_oracle() {
    let m = Manifest::load();
    let rows = m.plans();
    let mut wrong = Vec::new();
    println!("kind   size          n_llm   best        tokens  contain     offset  verdict");
    for r in &rows {
        let plan = plan_image_grid(r.w, r.h, &V41);
        let g = PadGeometry::of(r.w, r.h, plan.best_w, plan.best_h).expect("geometry");
        let got = [
            plan.n_llm_h,
            plan.n_llm_w,
            plan.best_h,
            plan.best_w,
            plan.n_tokens(),
            g.resized_w,
            g.resized_h,
            g.off_x,
            g.off_y,
        ];
        let ok = got == r.want;
        println!(
            "{:<6} {:>5}x{:<6} {:>3}x{:<4} {:>5}x{:<5} {:>6}  {:>5}x{:<5} {:>2},{:<3}  {}",
            r.kind,
            r.w,
            r.h,
            got[0],
            got[1],
            got[3],
            got[2],
            got[4],
            got[5],
            got[6],
            got[7],
            got[8],
            if ok { "same" } else { "DIFF" }
        );
        if !ok {
            wrong.push(format!(
                "{}x{}: ours {got:?}, reference {:?}",
                r.w, r.h, r.want
            ));
        }
    }
    let sizes = rows.iter().filter(|r| r.kind == "size").count();
    println!(
        "plans: {} rows ({} images, {sizes} sizes), {} differ",
        rows.len(),
        rows.len() - sizes,
        wrong.len()
    );
    assert!(sizes >= 20, "the set's size table has {sizes} rows");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
#[ignore = "reads the oracle set on the box"]
fn hw_preprocess_matches_oracle() {
    let m = Manifest::load();
    let mut wrong = Vec::new();
    println!(
        "image                resampled  rgb bytes  differ  max|d|  patches   differ  verdict"
    );
    for row in &m.images {
        let img = decode(row);
        let plan = plan_image_grid(img.width, img.height, &V41);
        assert_eq!(
            (plan.best_w, plan.best_h, plan.n_llm_h, plan.n_llm_w),
            (row.best_w, row.best_h, row.n_llm_h, row.n_llm_w),
            "{}",
            row.name
        );
        let ours = padded(&img, &plan).expect("pad");
        let theirs = m.read(&format!("{}.rgb.u8", row.stem()));
        let (rgb_diff, max_d) = ours
            .data
            .iter()
            .zip(&theirs)
            .filter(|(a, b)| a != b)
            .fold((0usize, 0u8), |(n, d), (a, b)| {
                (n + 1, d.max(a.abs_diff(*b)))
            });
        let rgb_diff = rgb_diff + ours.data.len().abs_diff(theirs.len());
        let patches = to_patches(&ours, plan, &V41).expect("patches");
        assert_eq!(
            (patches.n_vit_h, patches.n_vit_w),
            (row.n_vit_h, row.n_vit_w),
            "{}",
            row.name
        );
        let want = u16s(&m.read(&format!("{}.patches.bf16", row.stem())));
        let p_diff = patches
            .bf16
            .iter()
            .zip(&want)
            .filter(|(a, b)| a != b)
            .count()
            + patches.bf16.len().abs_diff(want.len());
        let resampled = (row.w, row.h) != (row.best_w, row.best_h);
        let ok = rgb_diff == 0 && p_diff == 0;
        println!(
            "{:<20} {:<10} {:>9}  {:>6}  {:>6}  {:>8}  {:>6}  {}",
            row.name,
            if resampled { "bicubic" } else { "no" },
            theirs.len(),
            rgb_diff,
            max_d,
            want.len(),
            p_diff,
            if ok { "bit-exact" } else { "DIFF" }
        );
        if !ok {
            wrong.push(format!(
                "{}: {rgb_diff} rgb bytes (max {max_d}), {p_diff} patch values differ",
                row.name
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
#[ignore = "reads the oracle set on the box"]
fn hw_span_matches_oracle() {
    let m = Manifest::load();
    assert_eq!(
        m.image_token_id,
        deepseek41v::IMAGE_TOKEN_ID,
        "the set's image_token_id"
    );
    for row in &m.images {
        let plan = GridPlan {
            n_llm_h: row.n_llm_h,
            n_llm_w: row.n_llm_w,
            best_h: row.best_h,
            best_w: row.best_w,
        };
        let span = image_span(&plan, deepseek41v::IMAGE_TOKEN_ID);
        let types: Vec<i32> = span.types.iter().map(|t| t.code()).collect();
        let ids: Vec<i32> = span
            .ids
            .iter()
            .map(|&i| i32::try_from(i).expect("id fits i32"))
            .collect();
        assert_eq!(types.len(), row.n_tokens, "{}", row.name);
        assert_eq!(
            types,
            i32s(&m.read(&format!("{}.types.i32", row.stem()))),
            "{} types",
            row.name
        );
        assert_eq!(
            ids,
            i32s(&m.read(&format!("{}.ids.i32", row.stem()))),
            "{} ids",
            row.name
        );
        println!(
            "{:<20} {:>4} positions ({}x{} grid): types and ids same",
            row.name,
            types.len(),
            row.n_llm_h,
            row.n_llm_w
        );
    }
}

#[test]
#[ignore = "reads the mmproj file on the box"]
fn hw_mmproj_tensors_named() {
    let m = Manifest::load();
    let sha = sha256_file(&m.mmproj);
    assert_eq!(
        sha,
        m.mmproj_sha256,
        "{}: not the file the set names",
        m.mmproj.display()
    );
    let g = Gguf::open(&m.mmproj).unwrap_or_else(|e| panic!("{}: {e}", m.mmproj.display()));
    let hp = Hparams::read(&g).unwrap_or_else(|e| panic!("{}: {e}", m.mmproj.display()));
    assert_eq!(hp.grid(), V41, "the file's resize parameters");
    let named = tensors::check(
        &hp,
        g.iter_tensors()
            .map(|t| (t.name.as_str(), t.dims.as_slice(), t.ty)),
    )
    .unwrap_or_else(|e| panic!("{}: {e}", m.mmproj.display()));
    println!("mmproj {} sha256 {sha}", m.mmproj.display());
    println!(
        "hparams: {} blocks, dim {}, {} heads, ff {}, patch {}, downsample {}, max {} tokens, min {} px, out {}, eps {:e}, theta {}",
        hp.n_layer,
        hp.dim,
        hp.n_head,
        hp.ff,
        hp.patch,
        hp.downsample,
        hp.max_tokens,
        hp.min_pixels,
        hp.out_dim,
        hp.eps,
        hp.rope_theta
    );
    println!("tensors: {named} of {} in the file named", g.tensor_count());
    assert_eq!(named, g.tensor_count());
}
