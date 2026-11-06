use super::{
    GridTables, IqKind, IqQuantError, KMAP_SIZE, QK, STATUS_NAN, imatrix_weights, quantize_row,
    stable_order,
};
use crate::quant::{GgmlType, dequant_row};

/// ggml's neighbour list of off-grid code word `i`, built the way
/// `iq2xs_init_impl` builds it: every grid point's (distance, index) pair
/// sorted, then the points up to the `nwant`-th distinct distance.
fn naive_list(t: &GridTables, nwant: usize, i: usize) -> Vec<u32> {
    let mut d: Vec<(i32, usize)> = t
        .codes
        .iter()
        .enumerate()
        .map(|(j, &c)| {
            let d2 = (0..8)
                .map(|k| {
                    let p = 2 * ((i >> (2 * k)) & 3) as i32 + 1;
                    let g = 2 * ((c >> (2 * k)) & 3) as i32 + 1;
                    (g - p) * (g - p)
                })
                .sum();
            (d2, j)
        })
        .collect();
    d.sort_unstable();
    let mut out = Vec::new();
    let mut d2 = d[0].0;
    let mut nhave = 1;
    for &(dj, j) in &d {
        if dj > d2 {
            if nhave == nwant {
                break;
            }
            d2 = dj;
            nhave += 1;
        }
        out.push(j as u32);
    }
    out
}

#[test]
fn tables_are_ggml_init() {
    for (kind, nwant, first) in [
        (IqKind::Iq2Xxs, 2, [0u32, 2, 5, 8, 10, 17, 20, 32]),
        (IqKind::Iq1S, 3, [0u32, 2, 5, 8, 10, 17, 21, 32]),
    ] {
        let t = GridTables::new(kind);
        assert_eq!(t.codes.len(), kind.grid_size());
        assert_eq!(&t.codes[..8], &first, "{kind:?}: kgrid's first codes");
        assert_eq!(t.map.len(), KMAP_SIZE);
        // Every grid point maps to itself; every other word to a list that
        // is the naive sort's.
        for (g, &c) in t.codes.iter().enumerate() {
            assert_eq!(t.map[c as usize], g as i32);
        }
        let mut checked = 0;
        for i in (0..KMAP_SIZE).step_by(if kind == IqKind::Iq1S { 7 } else { 1 }) {
            let m = t.map[i];
            if m >= 0 {
                continue;
            }
            let start = (-m - 1) as usize;
            let n = t.neighbours[start] as usize;
            assert_eq!(
                &t.neighbours[start + 1..=start + n],
                naive_list(&t, nwant, i).as_slice(),
                "{kind:?}: word {i}"
            );
            checked += 1;
        }
        assert!(checked > 1000, "{kind:?}: {checked} lists checked");
    }
}

#[test]
fn order_is_stable_on_ties() {
    let mut x = [0.0f32; 32];
    for (i, v) in x.iter_mut().enumerate() {
        *v = [1.0, -0.0, 0.0, -1.0, 1.0][i % 5];
    }
    let idx = stable_order(&x);
    for w in idx.windows(2) {
        let (a, b) = (w[0] as usize, w[1] as usize);
        assert!(x[a] < x[b] || (x[a] == x[b] && a < b), "{a} before {b}");
    }
}

/// A row of MXFP4-like values (few distinct levels a sub-block, so ties
/// everywhere) and positive importance.
fn row(n: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
    let mut s = seed;
    let mut next = move || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        s >> 8
    };
    let levels = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let x: Vec<f32> = (0..n)
        .map(|_| {
            let r = next();
            let v = levels[(r % 8) as usize] * 0.01;
            if r & 256 != 0 { -v } else { v }
        })
        .collect();
    let w: Vec<f32> = (0..n)
        .map(|_| 0.1 + (next() % 1000) as f32 / 100.0)
        .collect();
    (x, w)
}

#[test]
fn rows_quantize_and_read_back() {
    for (kind, ty, floor) in [
        (IqKind::Iq2Xxs, GgmlType::IQ2_XXS, 0.8),
        (IqKind::Iq1S, GgmlType::IQ1_S, 0.6),
    ] {
        let t = GridTables::new(kind);
        let n = 4 * QK;
        let (x, w) = row(n, 7);
        let q = quantize_row(kind, &x, &w, t.view()).unwrap();
        assert_eq!(q.len(), n / QK * kind.block_bytes());
        assert_eq!(q, quantize_row(kind, &x, &w, t.view()).unwrap());
        let mut y = vec![0.0f32; n];
        dequant_row(ty, &q, &mut y).unwrap();
        let dot: f64 = x
            .iter()
            .zip(&y)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let nx: f64 = x.iter().map(|a| f64::from(*a).powi(2)).sum();
        let ny: f64 = y.iter().map(|a| f64::from(*a).powi(2)).sum();
        let cos = dot / (nx.sqrt() * ny.sqrt());
        assert!(cos > floor, "{kind:?}: cosine {cos}");
    }
}

#[test]
fn refusals_are_named() {
    let t = GridTables::new(IqKind::Iq1S);
    let (mut x, w) = row(QK, 3);
    assert_eq!(
        quantize_row(IqKind::Iq1S, &x[..100], &w[..100], t.view()),
        Err(IqQuantError::RowLength { n: 100 })
    );
    x[40] = f32::NAN;
    assert_eq!(
        quantize_row(IqKind::Iq1S, &x, &w, t.view()),
        Err(IqQuantError::Status {
            block: 0,
            status: STATUS_NAN
        })
    );
}

#[test]
fn imatrix_weights_divide_by_counts() {
    use crate::write::{Layout, TensorDecl, Writer};
    use crate::{Gguf, Value};
    let sums: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let counts = [4.0f32, 0.0];
    let f32_bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let decl = |name: &str, n: u64| TensorDecl {
        name: name.to_string(),
        dims: vec![n],
        type_id: 0,
        nbytes: 4 * n,
    };
    let kvs = vec![(
        "general.type".to_string(),
        Value::String("imatrix".to_string()),
    )];
    let layout = Layout::new(&kvs, vec![decl("t.in_sum2", 8), decl("t.counts", 2)]).unwrap();
    let mut w = Writer::new(Vec::new(), layout).unwrap();
    w.tensor("t.in_sum2", &f32_bytes(&sums)).unwrap();
    w.tensor("t.counts", &f32_bytes(&counts)).unwrap();
    let bytes = w.finish().unwrap();
    let dir = std::env::temp_dir().join(format!("iq-quant-imatrix-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.gguf");
    std::fs::write(&path, bytes).unwrap();
    let m = Gguf::open(&path).unwrap();
    assert_eq!(
        imatrix_weights(&m, "t").unwrap(),
        vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]
    );
    assert!(imatrix_weights(&m, "u").is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}
