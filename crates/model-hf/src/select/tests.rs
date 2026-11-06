use gguf::write::TensorDecl;

use super::*;

fn planned(names: &[&str]) -> Vec<Planned<()>> {
    names
        .iter()
        .map(|n| match n.strip_suffix(".scale") {
            Some(of) => Planned {
                decl: TensorDecl {
                    name: (*n).into(),
                    dims: vec![1],
                    type_id: 0,
                    nbytes: 4,
                },
                make: Make::Scale { of: of.into() },
            },
            None => Planned {
                decl: TensorDecl {
                    name: (*n).into(),
                    dims: vec![1],
                    type_id: 0,
                    nbytes: 4,
                },
                make: Make::Constant(crate::folder::Constant {
                    dims: vec![1],
                    values: vec![0.0],
                }),
            },
        })
        .collect()
}

fn names(p: &[Planned<()>]) -> Vec<&str> {
    p.iter().map(|p| p.decl.name.as_str()).collect()
}

fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).into()).collect()
}

const ALL: [&str; 7] = [
    "token_embd.weight",
    "blk.0.attn_q.weight",
    "blk.3.indexer.attn_k.weight",
    "blk.3.indexer.proj.weight",
    "blk.3.indexer.proj.weight.scale",
    "blk.31.indexer.attn_k.weight",
    "output.weight",
];

/// A glob is `*` for any run, none included, and `?` for one character, over the whole name.
#[test]
fn a_glob_matches_the_whole_name() {
    for (pattern, name, want) in [
        ("blk.*.indexer.*", "blk.3.indexer.attn_k.weight", true),
        ("blk.*.indexer.*", "blk.31.indexer.proj.weight", true),
        ("blk.*.indexer.*", "blk.3.attn_q.weight", false),
        ("blk.?.attn_q.weight", "blk.3.attn_q.weight", true),
        ("blk.?.attn_q.weight", "blk.31.attn_q.weight", false),
        ("*weight", "token_embd.weight", true),
        ("*", "", true),
        ("token_embd", "token_embd.weight", false),
        ("a*b*c", "aXbYbZc", true),
        ("a*b*c", "aXbYbZ", false),
        ("a**", "a", true),
    ] {
        assert_eq!(glob(pattern, name), want, "{pattern} on {name}");
    }
}

/// The tensors dropped and the tensors kept by the same pattern are complements: their union is
/// the plan, their intersection empty, their counts add up, and a tensor's scale goes with it.
#[test]
fn drop_and_only_split_the_plan_into_complements() {
    let pattern = strings(&["blk.*.indexer.*"]);
    let dropped = Select::of(pattern.clone(), vec![])
        .unwrap()
        .apply(planned(&ALL))
        .unwrap();
    let kept = Select::of(vec![], pattern)
        .unwrap()
        .apply(planned(&ALL))
        .unwrap();
    assert_eq!(
        names(&dropped),
        ["token_embd.weight", "blk.0.attn_q.weight", "output.weight"]
    );
    assert_eq!(
        names(&kept),
        [
            "blk.3.indexer.attn_k.weight",
            "blk.3.indexer.proj.weight",
            "blk.3.indexer.proj.weight.scale",
            "blk.31.indexer.attn_k.weight",
        ]
    );
    assert_eq!(dropped.len() + kept.len(), ALL.len());
    let mut union: Vec<&str> = names(&dropped).into_iter().chain(names(&kept)).collect();
    union.sort_unstable();
    let mut all = ALL.to_vec();
    all.sort_unstable();
    assert_eq!(union, all);
}

/// A pattern that matches no tensor, one that matches only a scale, both flags, an empty pattern
/// and a selection that leaves nothing are each refused by name.
#[test]
fn a_selection_that_selects_nothing_is_refused() {
    let refused = |select: Result<Select, Error>| -> String {
        select
            .and_then(|s| s.apply(planned(&ALL)).map(|_| ()))
            .unwrap_err()
            .to_string()
    };
    let e = refused(Select::of(strings(&["blk.*.nothing.*"]), vec![]));
    assert!(e.contains("--drop") && e.contains("blk.*.nothing.*"), "{e}");
    let e = refused(Select::of(vec![], strings(&["blk.0.*", "output.bias"])));
    assert!(e.contains("--only") && e.contains("output.bias"), "{e}");
    let e = refused(Select::of(strings(&["*.scale"]), vec![]));
    assert!(e.contains("*.scale"), "{e}");
    let e = refused(Select::of(strings(&["a"]), strings(&["b"])));
    assert!(e.contains("both"), "{e}");
    let e = refused(Select::of(strings(&[""]), vec![]));
    assert!(e.contains("empty pattern"), "{e}");
    let e = refused(Select::of(strings(&["*"]), vec![]));
    assert!(e.contains("no tensor is left"), "{e}");
}
