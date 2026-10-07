//! The expert stream's gate (`bloomery_gpu::host::xstream`): a synthetic
//! source, no model file, the stream's own contract against its ring.
//!
//! The model: two layers of 16 experts, top-2, the stage stacks holding
//! experts 0..8 at places 0..8 and the host the rest; each expert has three
//! parts of 2048, 2048 and 1024 bytes, words from a hash of its (layer, id,
//! part). The rule's costs are the gate's (a host column far dearer than the
//! copy and the card's column), so every routed host expert is in the rule's
//! set and the balance never cuts it: the stream is the rule's set cut at
//! the ring's half, in rank order (count descending, id ascending). Each
//! unit is 64 columns, 128 picks: the card experts 8 each, the host experts
//! 16, 14, 10, 8, 6, 4, 4, 2 (a second unit's set reversed).
//!
//! Clauses:
//! - probe: the stream starts with a ring of the half it was asked for (the
//!   card has room) and a pinned staging of the slots asked for, the lane's
//!   rate measured and positive. The probe streams layer 1's host experts
//!   into the first half, which the first stream (layer 0's) then takes.
//! - set: a unit's streamed experts are the rule's set ([`stream_tail`]) cut
//!   at the half in rank order, the exclusion set the same ids ascending, and
//!   the layer's record says so (tail, streamed, the union's columns).
//! - rows: the half's ring row holds slot `i` for the `i`-th streamed expert
//!   and HOST for every other; its union row the stage map's places with
//!   `n_card + i` for the `i`-th.
//! - landed: with every fill thread held [`LANE_HOLD`] a job, the stream's
//!   copies record one event a batch, a third of them ([`land_batch_size`]):
//!   a snapshot of the whole half behind the first batch's event alone reads
//!   that batch's slots the source's bytes and every later batch's the bytes
//!   the probe left there (mutants: a single event for the whole layer, the
//!   one-wait shape this gate replaces, or an event recorded before its
//!   batch's own copies), and one behind every batch's event reads each slot
//!   its expert's source bytes, part by part (mutant: a batch's wait
//!   dropped; a part copied 16 bytes off its place).
//! - reuse: with the engine stream held by a host flag over a copy of a
//!   half's slots and the read after it, the half's next stream (two later)
//!   is issued and the host waits [`HOST_WAIT`] before it lets the engine go: the
//!   held copy reads the first stream's bytes, not the next one's (its
//!   mutant: the copies do not wait for the half's read event).
//! - serve set: a layer's exclusion set stands after its stream's read (the
//!   batch order serves a unit's layer after its card route) for that
//!   unit's serve alone — another unit of the layer (one the walk takes no
//!   stream for) reads none — until the layer's next unit, which, streaming
//!   nothing, empties it, or the call's end, which empties every layer's
//!   (mutants: the read clears the set; the set read whatever the unit; a
//!   unit that streams nothing leaves the last one's; the end keeps them).
//! - end: a call's end with a stream no read has closed is refused by name,
//!   and one whose streams were all read returns the call's experts and
//!   bytes.
//! - backlog deadline: a lane whose one fill thread is held in its source
//!   past the deadline, fed more jobs than its staging slots, is a named
//!   error within the deadline and a half (its mutant: the wait without its
//!   deadline, which returns only when the source does).
//! - room: a ring the card's free bytes past `keep_free` cannot hold is
//!   refused by its own name (`XSTREAM_ROOM`), which a caller resolving an
//!   unset lever tells from every other refusal.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_xstream: built without the `gpu` feature; see `just gate-gpu-xstream`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_xstream", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    use bloomery_gpu::host::batch::BatchKey;
    use bloomery_gpu::host::slots::HOST;
    use bloomery_gpu::host::swap::{Piece, SwapSource, Transform};
    use bloomery_gpu::host::xstream::{
        Costs, PARTS, RingLayer, XCfg, XSTREAM_ROOM, XStream, land_batch_size, stream_tail,
    };
    use bloomery_gpu::{Gpu, GpuError, HostFlags};
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use cuda_core::{CudaStream, DeviceBuffer, sys};
    use runtime::xsplit::{Constants, Split};

    const L: usize = 2;
    const E: usize = 16;
    const TOP_K: usize = 2;
    const N_CARD: usize = 8;
    const COLS: usize = 64;
    const PART_BYTES: [usize; PARTS] = [2048, 2048, 1024];
    /// The ring's half the gate asks for: fewer slots than the host experts
    /// a unit routes, so the half cuts the set.
    const HALF: usize = 6;
    const DEADLINE: Duration = Duration::from_secs(1);
    /// How long `landed` holds each fill thread's job: longer than the
    /// engine stream's enqueue of the layer's snapshot, so a copy that did not
    /// wait for the landed event reads the ring before its bytes.
    const LANE_HOLD: Duration = Duration::from_millis(30);
    /// How long `reuse` keeps the engine stream held after the half's next
    /// stream is issued: past the copies' time, so copies that did not wait
    /// for the half's read event land under the held snapshot.
    const HOST_WAIT: Duration = Duration::from_millis(100);

    /// The gate's costs: a host column far dearer than a copy of the
    /// synthetic expert and its card columns, so neither the floor nor the
    /// balance cuts a unit's set before the half at any lane rate the probe
    /// measures above a megabyte a second (a copy of the 5,120-byte expert
    /// is latency, and its measured rate moves with the box's load).
    const COSTS: Costs = Costs {
        host_us_per_col: 10_000.0,
        host_us_fixed: 1.0,
        card_us_fixed: 13.0,
        card_us_per_col: 0.16,
    };

    /// The synthetic source: each part's bytes from a hash of (layer, id,
    /// part); `hold` makes its first call sleep [`HOLD`].
    struct Synth {
        bytes: Vec<Vec<u8>>,
        hold: AtomicBool,
    }

    const HOLD: Duration = Duration::from_secs(3);

    impl Synth {
        fn new() -> Synth {
            let mut bytes = Vec::with_capacity(L * E * PARTS);
            for l in 0..L {
                for id in 0..E {
                    for (p, &n) in PART_BYTES.iter().enumerate() {
                        let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ ((l * E + id) * PARTS + p) as u64;
                        bytes.push(
                            (0..n)
                                .map(|_| {
                                    h ^= h << 13;
                                    h ^= h >> 7;
                                    h ^= h << 17;
                                    (h >> 24) as u8
                                })
                                .collect(),
                        );
                    }
                }
            }
            Synth {
                bytes,
                hold: AtomicBool::new(false),
            }
        }

        fn part(&self, l: usize, id: u32, p: usize) -> &[u8] {
            &self.bytes[(l * E + id as usize) * PARTS + p]
        }
    }

    impl SwapSource for Synth {
        fn part_bytes(&self, layer: usize) -> &[usize] {
            if layer < L { &PART_BYTES } else { &[] }
        }

        fn source(&self, layer: usize, id: u32, part: usize) -> Result<Piece<'_>, GpuError> {
            if self.hold.swap(false, Ordering::AcqRel) {
                std::thread::sleep(HOLD);
            }
            Ok(Piece {
                bytes: self.part(layer, id, part),
                transform: Transform::Identity,
            })
        }

        fn dest(
            &self,
            layer: usize,
            _part: usize,
            _slot: u32,
        ) -> Result<sys::CUdeviceptr, GpuError> {
            Err(GpuError::Shape {
                what: "Synth::dest",
                detail: format!("layer {layer}: the stream copies into its own ring"),
            })
        }

        fn prepare_victim(&self, _layer: usize, _id: u32) -> Result<(), GpuError> {
            Ok(())
        }

        fn host_resident(&self, _layer: usize, _id: u32) -> Result<bool, GpuError> {
            Ok(true)
        }

        fn release_host(&self, _layer: usize, _id: u32) -> Result<u64, GpuError> {
            Ok(0)
        }
    }

    fn cfg(fill: usize, staging: usize) -> XCfg {
        XCfg {
            costs: COSTS,
            experts: E,
            top_k: TOP_K,
            max_half: HALF,
            keep_free: 0,
            min_half: 2,
            fill_threads: fill,
            staging_slots: staging,
            deadline: DEADLINE,
            // Layer 1's experts: the probe leaves the first half holding bytes
            // no layer-0 stream copies, so a slot read before its copy lands
            // reads them.
            probe: (N_CARD as u32..E as u32).map(|id| (1, id)).collect(),
        }
    }

    /// A unit's counts: the card experts 8 each, the host experts `host` in
    /// id order; 128 picks of 64 columns.
    fn counts(host: [u32; E - N_CARD]) -> Vec<u32> {
        let mut c = vec![8u32; N_CARD];
        c.extend_from_slice(&host);
        c
    }

    const UNIT_A: [u32; E - N_CARD] = [16, 14, 10, 8, 6, 4, 4, 2];

    /// A unit that routes no host expert: the card experts 16 each.
    fn unrouted() -> Vec<u32> {
        let mut c = vec![16u32; N_CARD];
        c.extend_from_slice(&[0; E - N_CARD]);
        c
    }
    const UNIT_B: [u32; E - N_CARD] = [2, 4, 4, 6, 8, 10, 14, 16];

    fn stage() -> Vec<u32> {
        (0..E as u32)
            .map(|id| if (id as usize) < N_CARD { id } else { HOST })
            .collect()
    }

    fn host() -> Vec<u32> {
        (N_CARD as u32..E as u32).collect()
    }

    /// The rule's set cut at the half, in rank order.
    fn want_set(x: &XStream, l: usize, c: &[u32]) -> Result<Vec<u32>, GateError> {
        let k: Constants = x.constants(l).ok_or("the stream's constants of a layer")?;
        let mut s = Split::default();
        stream_tail(c, &host(), &k, &mut s).map_err(|e| format!("the rule: {e}"))?;
        s.stream.truncate(x.half_slots());
        Ok(s.stream)
    }

    /// The gate's one unit of layer `l`: the walk's key of a 64-column unit.
    fn key(l: usize) -> BatchKey {
        BatchKey {
            layer: l,
            set: 0,
            at: 0,
            u: COLS,
        }
    }

    /// Stream layer `l` of a unit of `c` on `x`.
    fn unit(
        x: &mut XStream,
        stream: &CudaStream,
        l: usize,
        c: &[u32],
    ) -> Result<bloomery_gpu::host::xstream::XLayer, GateError> {
        Ok(x.layer(stream, key(l), COLS, c, (&host(), 0), (&stage(), N_CARD))?)
    }

    /// A copy on `stream` of slots `0..n` of `r`'s half, part by part, into
    /// `dst` (`n` experts' parts back to back, `PART_BYTES` each), enqueued
    /// in the stream's order.
    fn snapshot(
        stream: &CudaStream,
        r: &RingLayer,
        n: usize,
        dst: &DeviceBuffer<u8>,
    ) -> Result<(), GateError> {
        let mut at = 0u64;
        for i in 0..n {
            for (p, &b) in PART_BYTES.iter().enumerate() {
                // SAFETY: slot i of the half's part p holds `b` bytes at
                // `parts[p] + i·b` (the stream packs a layer's experts at
                // their part bytes); `dst` holds `n · Σ PART_BYTES` bytes and
                // `at` stays inside it; both outlive the copy (the caller
                // drains `stream` before either goes).
                let rc = unsafe {
                    sys::cuMemcpyDtoDAsync_v2(
                        dst.cu_deviceptr() + at,
                        r.parts[p] + (i * b) as u64,
                        b,
                        stream.cu_stream(),
                    )
                };
                if rc != sys::cudaError_enum_CUDA_SUCCESS {
                    return Err(format!("cuMemcpyDtoDAsync rc {rc}").into());
                }
                at += b as u64;
            }
        }
        Ok(())
    }

    /// Whether `got` holds, expert after expert of `ids`, the source's parts
    /// of layer `l`.
    fn bytes_match(src: &Synth, l: usize, ids: &[u32], got: &[u8]) -> bool {
        let mut at = 0;
        for &id in ids {
            for (p, &b) in PART_BYTES.iter().enumerate() {
                if got.get(at..at + b) != Some(src.part(l, id, p)) {
                    return false;
                }
                at += b;
            }
        }
        true
    }

    /// The `u32` words at mapped address `ptr`, `n` of them, read on `stream`.
    fn words(stream: &CudaStream, ptr: sys::CUdeviceptr, n: usize) -> Result<Vec<u32>, GateError> {
        let buf = DeviceBuffer::<u32>::zeroed(stream, n)?;
        // SAFETY: `ptr` is the stream's mapped row of `n` words, alive while
        // the stream lives; `buf` holds `n` words; the copy is drained below.
        let rc = unsafe {
            sys::cuMemcpyDtoDAsync_v2(buf.cu_deviceptr(), ptr, 4 * n, stream.cu_stream())
        };
        if rc != sys::cudaError_enum_CUDA_SUCCESS {
            return Err(format!("cuMemcpyDtoDAsync rc {rc}").into());
        }
        Ok(buf.to_host_vec(stream)?)
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let (ctx, stream) = (gpu.context(), gpu.stream());
        let src = Arc::new(Synth::new());
        let mut ok = true;

        // probe
        let mut x = XStream::new(
            ctx,
            stream,
            Arc::clone(&src) as Arc<dyn SwapSource>,
            L,
            cfg(2, 4),
        )?;
        let probe_ok = x.half_slots() == HALF && x.staging_slots() == 4 && x.lane_b_per_us() > 0.0;
        println!(
            "probe: half {} (asked {HALF}), staging {} (asked 4), lane {:.3} GB/s {}",
            x.half_slots(),
            x.staging_slots(),
            x.lane_b_per_us() / 1000.0,
            verdict(probe_ok)
        );
        ok &= probe_ok;

        // set, rows, landed (a batch a third of the stream)
        x.begin_call();
        x.delay_lane(LANE_HOLD);
        let ca = counts(UNIT_A);
        let want_a = want_set(&x, 0, &ca)?;
        let rec = unit(&mut x, stream, 0, &ca)?;
        let r = x
            .ring_layer(0)
            .ok_or("a ring view of a layer that streamed")?;
        let snap = DeviceBuffer::<u8>::zeroed(stream, HALF * PART_BYTES.iter().sum::<usize>())?;
        // The first batch's event alone: every later batch's copies are
        // still behind held fill-thread jobs, so their slots read the bytes
        // the probe left in the half — one event a batch is real.
        let per = land_batch_size(r.n);
        let one: usize = PART_BYTES.iter().sum();
        if let Some(e) = r.batches.first().and_then(|b| b.event.as_deref()) {
            stream.wait(e)?;
        }
        snapshot(stream, &r, r.n, &snap)?;
        let first = snap.to_host_vec(stream)?;
        let boundary_ok = r.batches.len() > 1
            && bytes_match(&src, 0, &want_a[..per], &first)
            && !bytes_match(&src, 0, &want_a[per..], &first[per * one..]);
        // Every batch's event: each slot reads its own expert's bytes.
        x.wait_layer(stream, 0)?;
        snapshot(stream, &r, r.n, &snap)?;
        let got = snap.to_host_vec(stream)?;
        x.delay_lane(Duration::ZERO);
        let landed_ok = boundary_ok && bytes_match(&src, 0, &want_a, &got);
        let mut sorted = want_a.clone();
        sorted.sort_unstable();
        let host_columns: u64 = host()
            .iter()
            .filter(|id| !want_a.contains(id))
            .map(|&id| u64::from(ca[id as usize]))
            .sum();
        let set_ok = want_a.len() == HALF
            && r.n == HALF
            && x.excluded(key(0)) == sorted.as_slice()
            && rec.tail == E - N_CARD
            && rec.streamed == HALF
            && rec.host_columns == host_columns;
        println!(
            "set: streamed {:?} (the rule's set cut at the half: {want_a:?}), excluded {:?}, \
             record tail {} streamed {} host_columns {} (want {host_columns}) {}",
            &want_a[..r.n.min(want_a.len())],
            x.excluded(key(0)),
            rec.tail,
            rec.streamed,
            rec.host_columns,
            verdict(set_ok)
        );
        ok &= set_ok;
        let ring_row = words(stream, r.ring_map, E)?;
        let union_row = words(stream, r.union_map, E)?;
        let rows_ok = (0..E as u32).all(|id| {
            let rank = want_a.iter().position(|&w| w == id);
            let ring_want = rank.map_or(HOST, |i| i as u32);
            let union_want = match rank {
                Some(i) => (N_CARD + i) as u32,
                None => stage()[id as usize],
            };
            ring_row[id as usize] == ring_want && union_row[id as usize] == union_want
        });
        println!(
            "rows: ring row {ring_row:?}, union row {union_row:?} {}",
            verdict(rows_ok)
        );
        ok &= rows_ok;
        println!(
            "landed: with every fill thread held {LANE_HOLD:?} a job, the first batch's event \
             ({} of {} slots) alone lands its own slots and leaves the later batches' uncopied \
             {}, and every batch's event lands each slot its expert's bytes {}",
            per,
            r.n,
            verdict(boundary_ok),
            verdict(bytes_match(&src, 0, &want_a, &got))
        );
        ok &= landed_ok;
        x.read(0, stream)?;
        let after_read = x.excluded(key(0)) == sorted.as_slice();
        // Another unit of the layer (a narrower one the walk takes no stream
        // for) reads no set.
        let other = BatchKey {
            u: COLS - 1,
            ..key(0)
        };
        let other_unit = x.excluded(other).is_empty();

        // reuse: half 0 held over a copy of its slots and its read.
        let flags = HostFlags::new(ctx, 1)?;
        flags.clear(0)?;
        let ca2 = counts(UNIT_A);
        let first = unit(&mut x, stream, 1, &ca2)?;
        // Layer 1 took half 1; layer 0's next stream takes half 0.
        x.read(1, stream)?;
        let cb = counts(UNIT_B);
        let rec0 = unit(&mut x, stream, 0, &cb)?;
        let r0 = x
            .ring_layer(0)
            .ok_or("a ring view of a layer that streamed")?;
        let want_b = want_set(&x, 0, &cb)?;
        flags.enqueue_wait(stream, 0)?;
        snapshot(stream, &r0, r0.n, &snap)?;
        x.read(0, stream)?;
        // Layer 1's next stream takes half 1, layer 0's after it half 0
        // again: its copies must wait for the held read.
        let _ = unit(&mut x, stream, 1, &cb)?;
        x.read(1, stream)?;
        let ca3 = counts(UNIT_A);
        let _ = unit(&mut x, stream, 0, &ca3)?;
        std::thread::sleep(HOST_WAIT);
        flags.raise(0)?;
        let held = snap.to_host_vec(stream)?;
        x.read(0, stream)?;
        let reuse_ok = first.streamed > 0
            && rec0.streamed > 0
            && want_b != want_a
            && bytes_match(&src, 0, &want_b, &held);
        println!(
            "reuse: the held copy of half 0 reads its stream's bytes ({want_b:?}), not the next \
             stream's ({want_a:?}) {}",
            verdict(reuse_ok)
        );
        ok &= reuse_ok;

        // end
        let cd = counts(UNIT_B);
        let _ = unit(&mut x, stream, 1, &cd)?;
        let refused = match x.end_call() {
            Err(e) => {
                let s = e.to_string();
                s.contains("layer 1") && s.contains("no read")
            }
            Ok(_) => false,
        };
        x.begin_call();
        let streamed =
            unit(&mut x, stream, 0, &ca)?.streamed + unit(&mut x, stream, 1, &cb)?.streamed;
        x.read(0, stream)?;
        x.read(1, stream)?;
        let none = unit(&mut x, stream, 0, &unrouted())?;
        x.read(0, stream)?;
        let emptied =
            none.streamed == 0 && x.excluded(key(0)).is_empty() && !x.excluded(key(1)).is_empty();
        let report = x.end_call()?;
        let closed = (0..L).all(|l| x.excluded(key(l)).is_empty());
        let serve_ok = after_read && other_unit && emptied && closed;
        println!(
            "serve set: after the read {after_read}; another unit of the layer reads none \
             {other_unit}; a unit that streams nothing empties its layer's set and leaves the \
             other's {emptied}; the call's end empties every set {closed} {}",
            verdict(serve_ok)
        );
        ok &= serve_ok;
        let bytes: u64 = PART_BYTES.iter().map(|&b| b as u64).sum::<u64>() * streamed as u64;
        let end_ok =
            refused && report.streamed == streamed && report.bytes == bytes && report.layers == 2;
        println!(
            "end: an unread stream refused by name {refused}; a read call's end streamed {} of \
             {streamed}, {} B of {bytes}, {} layers {}",
            report.streamed,
            report.bytes,
            report.layers,
            verdict(end_ok)
        );
        ok &= end_ok;
        drop(x);

        // backlog deadline
        let mut slow = XStream::new(
            ctx,
            stream,
            Arc::clone(&src) as Arc<dyn SwapSource>,
            L,
            cfg(1, 2),
        )?;
        slow.begin_call();
        src.hold.store(true, Ordering::Release);
        let t0 = Instant::now();
        let r = unit(&mut slow, stream, 0, &ca);
        let waited = t0.elapsed();
        let deadline_ok = match &r {
            Err(e) => e.to_string().contains("staging backlog") && waited < DEADLINE + DEADLINE / 2,
            Ok(_) => false,
        };
        println!(
            "backlog deadline: a held source past the {DEADLINE:?} deadline: {} after {waited:?} {}",
            match &r {
                Err(e) => format!("refused: {e}"),
                Ok(_) => "streamed".to_string(),
            },
            verdict(deadline_ok)
        );
        ok &= deadline_ok;
        drop(slow);

        // room
        let mut tight = cfg(2, 4);
        tight.keep_free = u64::MAX;
        let room_ok = matches!(
            XStream::new(
                ctx,
                stream,
                Arc::clone(&src) as Arc<dyn SwapSource>,
                L,
                tight
            ),
            Err(GpuError::Shape {
                what: XSTREAM_ROOM,
                ..
            })
        );
        println!(
            "room: a ring past the card's free bytes refused by {XSTREAM_ROOM:?} {}",
            verdict(room_ok)
        );
        ok &= room_ok;

        if ok {
            println!(
                "gate_xstream: PASS — the stream is the rule's set cut at the half in rank order, \
                 its rows and exclusion set say so, the engine stream reads a batch's slots only \
                 once that batch's copies have landed (a batch a third of them), a half's next \
                 stream waits for the half's read, an unread stream refuses the call's end, a \
                 stuck lane is a named error within its deadline, and a ring with no room is \
                 refused by its own name."
            );
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
