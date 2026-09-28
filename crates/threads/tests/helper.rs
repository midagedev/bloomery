//! Gate test for helper placement ([`threads::helper::spawn_helper`]): a
//! helper spawned from a caller pinned to one cpu does not share that cpu.
//! Linux only, like the pool (`just gate-threads`).

use std::sync::mpsc;

use threads::helper::{helpers, mask, spawn_helper};

/// Set the calling thread's mask to `cpus`.
fn set_mask(cpus: &[usize]) {
    // SAFETY: `set` is a zeroed cpu_set_t only written through `CPU_SET` with
    // indices the test took from a mask of the same size, and
    // `sched_setaffinity` reads it with the matching size.
    let ok = unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for &c in cpus {
            libc::CPU_SET(c, &mut set);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    };
    assert!(ok, "sched_setaffinity to {cpus:?}");
}

/// The mask a helper named `name`, asked for `cpu`, runs its body with when
/// spawned from a thread whose mask is `opener`; and the cpu it says it is
/// pinned to.
fn helper_mask(name: &str, opener: Vec<usize>, cpu: Option<usize>) -> (Vec<usize>, Option<usize>) {
    let name = name.to_owned();
    std::thread::spawn(move || {
        set_mask(&opener);
        let (tx, rx) = mpsc::channel();
        let (handle, pinned) = spawn_helper(&name, cpu, move || {
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
/// already, left as it is; and on the cpu asked for. A test thread whose own
/// mask holds one cpu cannot tell the float from the pin and skips by name.
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

    let (seen, pinned) = helper_mask("helper-test-float", vec![c], None);
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
        (listed.pinned, listed.cpus),
        (None, seen.len()),
        "the list names the helper's mask"
    );

    let (seen, pinned) = helper_mask("helper-test-pair", pair.clone(), None);
    assert_eq!(pinned, None);
    assert_eq!(seen, pair, "a mask of two cpus floats already and is kept");

    let (seen, pinned) = helper_mask("helper-test-pin", wide.clone(), Some(c));
    assert_eq!(pinned, Some(c), "cpu {c} asked for");
    assert_eq!(seen, vec![c]);
}
