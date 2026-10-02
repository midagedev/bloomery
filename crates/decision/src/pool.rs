//! The head's resident worker threads: one per available CPU, started on first use, never pinned
//! (each inherits the mask of the thread that first calls in), parked on a queue between calls.
//! [`par_map`] hands a call's tasks to them and returns when every task has ended.
//!
//! A call from inside a task runs its tasks on that thread, one after another: a worker that
//! waited on the queue it serves would hold its place while the work it waits for sits behind it.

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

type Job = Box<dyn FnOnce() + Send + 'static>;

thread_local! {
    static IN_POOL: Cell<bool> = const { Cell::new(false) };
}

struct Pool {
    jobs: Mutex<Sender<Job>>,
    size: usize,
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let size = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        let (tx, rx) = channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        for i in 0..size {
            let rx: Arc<Mutex<Receiver<Job>>> = Arc::clone(&rx);
            std::thread::Builder::new()
                .name(format!("decision:{i}"))
                .spawn(move || {
                    IN_POOL.with(|f| f.set(true));
                    loop {
                        let job = rx.lock().expect("the pool's queue lock").recv();
                        match job {
                            Ok(job) => job(),
                            Err(_) => return,
                        }
                    }
                })
                .expect("a decision pool thread starts");
        }
        Pool {
            jobs: Mutex::new(tx),
            size,
        }
    })
}

/// The threads a call can use: the pool's size.
pub(crate) fn size() -> usize {
    pool().size
}

/// `f(0), …, f(n − 1)`, task 0 on the calling thread and the rest on the pool, in task order. A
/// task that panics makes this call panic once every task has ended.
pub(crate) fn par_map<T: Send>(n: usize, f: &(dyn Fn(usize) -> T + Sync)) -> Vec<T> {
    if n <= 1 || IN_POOL.with(Cell::get) {
        return (0..n).map(f).collect();
    }
    let slots: Vec<Mutex<Option<T>>> = (0..n).map(|_| Mutex::new(None)).collect();
    // The count lives in an Arc each job holds, so a job's last touch of it (the unlock after its
    // count-down) cannot meet memory this call has already given back.
    let left = Arc::new((Mutex::new(n), Condvar::new()));
    let failed = Mutex::new(false);
    let task = |i: usize| match catch_unwind(AssertUnwindSafe(|| f(i))) {
        Ok(v) => *slots[i].lock().expect("a result slot") = Some(v),
        Err(_) => *failed.lock().expect("the failure flag") = true,
    };
    let task: &(dyn Fn(usize) + Sync) = &task;
    let done = |left: &(Mutex<usize>, Condvar)| {
        let mut l = left.0.lock().expect("the task count");
        *l -= 1;
        if *l == 0 {
            left.1.notify_all();
        }
    };
    {
        let tx = pool().jobs.lock().expect("the pool's sender");
        for i in 1..n {
            let left = Arc::clone(&left);
            let job: Box<dyn FnOnce() + Send + '_> = Box::new(move || {
                task(i);
                done(&left);
            });
            // SAFETY: the job borrows `task` (and through it `f`, `slots` and `failed`), which
            // live until this function returns; it returns only after the wait below has seen
            // every job count itself down, which each does after its last use of the borrow (a
            // panic in `f` is caught inside `task`). So no job uses what it borrows after the
            // borrow ends, and erasing its lifetime to hand it to the queue is sound.
            let job: Job = unsafe { std::mem::transmute(job) };
            tx.send(job).expect("the pool's workers are running");
        }
    }
    task(0);
    done(&left);
    let mut l = left.0.lock().expect("the task count");
    while *l > 0 {
        l = left.1.wait(l).expect("the task count");
    }
    drop(l);
    assert!(
        !*failed.lock().expect("the failure flag"),
        "a decision pool task panicked"
    );
    slots
        .into_iter()
        .map(|s| {
            s.into_inner()
                .expect("a result slot")
                .expect("every task stored its result")
        })
        .collect()
}

/// `f(i, chunk)` for each `chunk`-long piece of `data` (the last may be shorter), over the pool.
pub(crate) fn par_chunks_mut(
    data: &mut [f32],
    chunk: usize,
    f: &(dyn Fn(usize, &mut [f32]) + Sync),
) {
    let parts: Vec<Mutex<Option<&mut [f32]>>> = data
        .chunks_mut(chunk)
        .map(|c| Mutex::new(Some(c)))
        .collect();
    par_map(parts.len(), &|i| {
        let c = parts[i]
            .lock()
            .expect("a chunk slot")
            .take()
            .expect("each chunk is taken once");
        f(i, c);
    });
}
