//! Gate test for helper placement ([`threads::helper::spawn_helper`]): a
//! helper spawned from a caller pinned to one cpu does not share that cpu.
//! Linux only, like the pool (`just gate-threads`).

use std::sync::mpsc;

use threads::helper::{Placement, float_caller, helpers, mask, sibling_of, spawn_helper};

/// Set the calling thread's mask to `cpus`.
fn set_mask(cpus: &[usize]) {
    let cpus: Vec<u32> = cpus
        .iter()
        .map(|&c| u32::try_from(c).expect("a cpu id fits a u32"))
        .collect();
    os::set_mask(&cpus).unwrap_or_else(|e| panic!("the mask is set to {cpus:?}: {e}"));
}

/// The mask a helper named `name`, placed as `asked`, runs its body with when
/// spawned from a thread whose mask is `opener`; and the cpu it says it is
/// pinned to.
fn helper_mask(name: &str, opener: Vec<usize>, asked: Placement) -> (Vec<usize>, Option<usize>) {
    let name = name.to_owned();
    std::thread::spawn(move || {
        set_mask(&opener);
        let (tx, rx) = mpsc::channel();
        let (handle, pinned) = spawn_helper(&name, asked, move || {
            let _ = tx.send(mask().expect("the helper's own mask"));
        })
        .expect("the helper is placed");
        let seen = rx.recv().expect("the helper's body ran");
        handle.join().expect("the helper");
        (seen, pinned)
    })
    .join()
    .expect("the opener")
}

/// Where a helper lands: floating off a one-cpu mask inherited from its
/// opener (the mask widens past that cpu, so the helper does not share it,
/// and the list of helpers says so); in a wider inherited mask, which floats
/// already, left as it is; on the cpu asked for; and on the SMT sibling of
/// the cpu asked for, from an opener pinned to that cpu. A test thread whose own
/// mask holds one cpu cannot tell the float from the pin and skips by name.
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "the Linux fact it reads: a thread's own affinity mask"
)]
#[test]
fn a_helper_leaves_its_openers_one_cpu_mask() {
    let wide = mask().expect("the test thread's mask");
    if wide.len() < 2 {
        println!(
            "SKIP a_helper_leaves_its_openers_one_cpu_mask: the test thread's mask is {wide:?}"
        );
        return;
    }
    let (c, pair) = (wide[0], vec![wide[0], wide[1]]);

    let (seen, pinned) = helper_mask("helper-test-float", vec![c], Placement::Float);
    assert_eq!(pinned, None, "no cpu asked: the helper floats");
    assert!(
        seen.len() > 1,
        "a helper spawned from a thread pinned to cpu {c} still runs on {seen:?}"
    );
    let listed = helpers()
        .into_iter()
        .find(|h| h.name == "helper-test-float")
        .expect("the helper is listed");
    assert_eq!(
        (listed.asked, listed.pinned, listed.cpus),
        (Placement::Float, None, seen.len()),
        "the list names the helper's mask"
    );

    let (seen, pinned) = helper_mask("helper-test-pair", pair.clone(), Placement::Float);
    assert_eq!(pinned, None);
    assert_eq!(seen, pair, "a mask of two cpus floats already and is kept");

    let (seen, pinned) = helper_mask("helper-test-pin", wide.clone(), Placement::Pin(c));
    assert_eq!(pinned, Some(c), "cpu {c} asked for");
    assert_eq!(seen, vec![c]);

    match sibling_of(c) {
        Some(s) if wide.contains(&s) => {
            let (seen, pinned) = helper_mask("helper-test-sibling", vec![c], Placement::Sibling(c));
            assert_eq!(pinned, Some(s), "cpu {c}'s sibling {s} asked for");
            assert_eq!(seen, vec![s], "a helper beside a thread pinned to cpu {c}");
        }
        other => println!("SKIP the sibling case: cpu {c}'s sibling is {other:?}"),
    }
}

/// A scoped thread a caller spawned itself inherits its opener's one-cpu mask
/// and [`float_caller`] widens it: the way a load's own helpers, which borrow the
/// load's data and so cannot be [`spawn_helper`]'s `'static` bodies, stay off the
/// opener's cpu. A test thread whose own mask holds one cpu cannot tell the
/// float from the pin and skips by name.
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "the Linux fact it reads: a thread's own affinity mask"
)]
#[test]
fn a_scoped_thread_floats_off_its_openers_one_cpu_mask() {
    let wide = mask().expect("the test thread's mask");
    if wide.len() < 2 {
        println!(
            "SKIP a_scoped_thread_floats_off_its_openers_one_cpu_mask: the test thread's mask is {wide:?}"
        );
        return;
    }
    let c = wide[0];
    let (inherited, floated) = std::thread::spawn(move || {
        set_mask(&[c]);
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let inherited = mask().expect("the scoped thread's inherited mask");
                    float_caller().expect("the scoped thread floats");
                    (inherited, mask().expect("the scoped thread's mask"))
                })
                .join()
                .expect("the scoped thread")
        })
    })
    .join()
    .expect("the opener");
    assert_eq!(
        inherited,
        vec![c],
        "the scoped thread inherits cpu {c} alone"
    );
    assert!(
        floated.len() > 1,
        "float_caller left the scoped thread on {floated:?}"
    );
}
