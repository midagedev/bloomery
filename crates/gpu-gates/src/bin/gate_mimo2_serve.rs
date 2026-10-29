//! `gate_mimo2_serve` — the MiMo seat of `bloomery-serve` against the session
//! it serves, on the model file of the mimo2 profile (`BLOOMERY_REF_MODEL`).
//!
//!     gate_mimo2_serve --dir <dir> [--after-refusals]
//!
//! Two clauses need different censuses, and every clause runs on the census
//! the recipe gives (both cards in view, `BLOOMERY_CARD=both`):
//!
//! - `refusals_before_any_load`'s `--place bp` half needs two visible cards.
//!
//! The seat's unset `--place` runs plan (b′) on a census of two cards (the
//! plan's tier holds past the seat's break-even, `TIER_RULE`) and `a` on one
//! ([`unset_place`]); the parent's clauses that leave `--place` unset hold
//! the placement it runs on the census they see.
//! - `server_ids_are_the_session_ids` and `ids_see_an_id_short` need one: the
//!   session opens on the gate card (`shared/gate_card.rs`).
//!
//! On a census of two or more this process runs `plan_only_names_its_default`
//! and `refusals_before_any_load` (both halves), then runs every later clause
//! in a child of itself (`--after-refusals`, logs in `<dir>/child/`) whose
//! `CUDA_VISIBLE_DEVICES` names the A6000 alone, taken by name from the census
//! and passed by the UUID the driver reports for it. A census with no A6000 is
//! a named error. The child prints one line saying the parent ran the two
//! clauses; its exit code and a `check <clause>: PASS` line for each clause it
//! owes are the verdict, and the final `PASS` needs this process's clauses and
//! the child's.
//!
//! On a census of one (a hand run) every clause runs in this process except
//! the `--place bp` half, which prints `deferred(two cards) …`, and the check
//! of the clause then names what ran (`--parallel 2 only`). A child started by
//! hand on a census of two prints `deferred(one card) …` for the two session
//! clauses. A `deferred(…)` line is a half that did not run, never a pass.
//!
//! The seat serves resident slots (`--parallel N`, two unset), a round of
//! several as one pass of the busy rows, the routed experts the plan puts on
//! the card and the rest on the host tier, the prompt fed in batches
//! (bit for bit the decode steps), and a later request keeps any prefix it
//! shares with what its slot holds. Every server is `bloomery-serve --model
//! mimo2 -m <file> …` beside this binary, `BLOOMERY_REF_MODEL` removed from its
//! environment and `BLOOMERY_RESIDENCY=off`, except in
//! `residency_default_is_the_machine`: the ids a server answers are compared
//! with a static session's and a one-slot server's, which the residency
//! machine's adapting split would move. Logs per server in `<dir>/<name>/` (`server.err`, the
//! responses).
//!
//! Without a load:
//!
//! - `host_room_halves_the_default`: the host room fits the default under
//!   its target as a short card does. Every `--plan` child whose output a clause compares
//!   with another child's runs under `BLOOMERY_HOST_ROOM` = `PLAN_ROOM`, this
//!   workstation's host room with nothing loaded, so no reading of the box's
//!   available bytes enters a comparison. At that room the default's `ctx`
//!   line and the plan's `ctx_max` equal `plan_only_names_its_default`'s, and
//!   no host fitted it. Under a room halfway between the host need of half
//!   the default and of the default (two plans made here at `PLAN_ROOM`,
//!   `HostNeed::of`) the line is `ctx rule=fitted` at a context from that
//!   half up to under the default whose host need the room holds,
//!   `host-short`, and the plan's `ctx_max` is that context (the host needs are
//!   those of the placement the unset flag runs, [`unset_place`]): the body serves no
//!   NVMe tier (`PlaceError::NoNvmeTier`), and a shorter context keeps more
//!   experts on the card. Under a room a GiB under the floor's need the
//!   process exits non-zero within [`REFUSE_WITHIN`] with the words of
//!   `ctx_ask::AskError::FloorShort` on a host, and no `load` record is
//!   printed. FAIL-first, by reading: on a seat that reads a host refusal as
//!   its own error the middle room refuses the load instead of fitting it,
//!   and the clause is red; on a seat whose plan reads no room
//!   `BLOOMERY_HOST_ROOM` moves nothing and both halves are red.
//! - `residency_default_is_the_machine`: `--plan` at [`CTX`] with no
//!   `BLOOMERY_RESIDENCY` prints the `residency lever` record at `mid-p0-s1`
//!   with why `place` ahead of the plan record, and the `residency host`
//!   record of the churn pool after it; at the largest total two slots hold
//!   (the KV planes take the card, the plan holds no expert to spare; read
//!   off the seat's refusal of a total past it) the common rule prints `off`
//!   and no pool, with why `no_card_experts` where the plan holds no card
//!   expert and `no_room` where the rung's leftover holds a few; under
//!   `BLOOMERY_RESIDENCY=off` it prints `off` with why `set` and no pool
//!   record.
//! - `plan_only_names_its_default`: `--plan` with no `--place`, `--ctx` or
//!   `--parallel` exits 0 and prints `place unset` (the word of
//!   [`unset_place`]: `bp` on two cards, `a` on one), the plan
//!   record and the `parallel` line, in that order, and no `load` or
//!   `listening` record. The `parallel` line is the default's one slot:
//!   `slots=1 slot_ctx=<ctx_max> total=<ctx_max> from=default pass=one`. The
//!   one `ctx` line holds its word (`ctx_ladder::default_line_held`: `target`
//!   or `fitted`, the trained context or a step of 1024 under it down to
//!   4096, the limit the trained context) and names the plan's `ctx_max`;
//!   the plan at it stands (`--parallel 1 --ctx <ctx_max> --plan` exits 0
//!   with the same `ctx_max`); and the step above it, when there is one,
//!   does not (`--parallel 1 --ctx <step above> --plan` is refused).
//!   FAIL-first: a seat that defaults to 4096 regardless names a context
//!   whose step above stands, and the hold is red; a seat on two default slots
//!   prints `slots=2`, and the `parallel` line hold is red.
//! - `refusals_before_any_load`: `--ctx 1 --parallel 2` — a split that
//!   leaves a slot no position — exits non-zero within [`REFUSE_WITHIN`],
//!   its stderr names the split and no `load` record is printed. On a
//!   census of two cards, `--place bp --parallel 2` exits non-zero the same
//!   way, its stderr holding the seat's refusal of a tier card beside several
//!   sequences ([`tier_slots_refusal`]: the body serves its tier beside one);
//!   on one card the `bp` half is a `deferred(two cards)` line. FAIL-first: a
//!   seat that takes a slot of no position is still running at the bound and
//!   the clause is red; a seat that leaves the refusal to the body plans and
//!   opens the load, whose refusal names no `--place`, and the clause is
//!   red.
//! - `slots_total_is_planned`: the plan counts every slot's positions. At
//!   the one-slot default context `c` (`--parallel 1 --plan`), `--parallel 2
//!   --ctx c --plan` exits 0 and its `parallel` line splits it (`slots=2
//!   slot_ctx=⌊c/2⌋ pass=one`); `--parallel 2 --ctx <the step above c>`, a total the
//!   one-slot plan refuses, exits non-zero within [`REFUSE_WITHIN`], its
//!   stderr names the set total's refusal (`ctx_ask::AskError::DoesNotFit`)
//!   and no `load` record is printed (a default that is the target has no
//!   step above and the second half is a named skip). FAIL-first: a seat whose plan counts one slot
//!   (`plan_with_slots` at one, over a slot's share) loads and is still
//!   running at the bound; the refusal half is red.
//!
//! One server as a user starts it, before the child:
//!
//! - `default_load_listens`: `--port 0` and no other flag, no lever pinned,
//!   on the census the recipe gives. It loads and listens, and its `load` and
//!   `listening` records name the `ctx` its one `ctx` line names: the default
//!   the plan chose is the context the load made. FAIL-first: a load scratch
//!   check that does not count the stage's position rows the plan counts (the
//!   rope tables at the default context) refuses the load, and the clause is
//!   red.
//!
//! One server, `--ctx` [`CTX`], `--parallel 1`, `--port 0`, its answers the one
//! slot's. It plans at the gate card's bytes: `BLOOMERY_CARD_BUDGET` is
//! [`gate_budget`] (`RTX_3090`'s usable bytes), so on the A6000 it holds the
//! card and host experts the session's gate plan holds (`shared/gate_card.rs`
//! plans the 3090's bytes on any one card). Every comparison of ids below is
//! between runs of one plan:
//!
//! - `load_and_listen`: the `load` record comes before the `listening`
//!   record; the load names `arch=mimo2`, `slots=1`, `layers` [`LAYERS`] and
//!   `ctx` [`CTX`], and the experts the `plan` record of the same run names:
//!   `card_experts` above 0 (the seat's one-slot plan puts experts on the
//!   card) and `host_experts`, which with the card's make [`ROUTED_EXPERTS`];
//!   the `listening` record names `ctx` [`CTX`] and `slots=1`. A server that
//!   never listens is this clause red, its stderr printed, and the clauses
//!   that need it skipped by name. FAIL-first: a load whose `card_experts`
//!   is not its plan's (0 against a plan that names experts, or one off) is
//!   red.
//! - `completion_ids`: `/completion` at temperature 0, [`N`] tokens, of the
//!   batch set's ids (`refset::arch::mimo2::BATCH`, read through its family),
//!   of [`PROSE`] (`/tokenize`) and of the chat turn the server's own
//!   template renders (`/apply-template`, `/tokenize`) each answer [`N`] ids
//!   (or a prefix ending at an end-of-generation id), each from an empty
//!   cache (`cache_n` 0, `cache_prompt: false`). FAIL-first: a server that
//!   answers fewer than [`N`] ids without an end-of-generation stop, or keeps
//!   a prefix under `cache_prompt: false`, is red.
//! - `chat_is_those_ids`: `/v1/chat/completions` of the same turn at
//!   temperature 0 and `max_tokens` [`N`] is a 200 whose `usage` counts the
//!   prompt's ids and the completion's tokens, and whose message text (the
//!   reasoning and the content) is inside the text of the chat run's ids
//!   (`/detokenize`). FAIL-first: a seat whose chat path renders another
//!   prompt than `/apply-template` counts other prompt tokens.
//! - the prefix clauses, the seat keeping any held position:
//!   `edit_resend_keeps_the_row_where_it_diverges` — the turn of [`EDIT_A`]
//!   answered, then resent with its user message changed to [`EDIT_B`], which
//!   diverges at `j` inside the prompt — keeps `j` (`cache_n`);
//!   `edit_resend_ids_are_a_fresh_runs` — its ids equal the same request with
//!   `cache_prompt: false`, bit for bit, since the batch feed writes the
//!   steps' bits whatever the cut; and `extension_keeps_every_held_position` — the turn resent
//!   with its reply and a later user turn (the ids the template renders past a
//!   reply, found by [`later_ids`]) — keeps every position the slot held.
//!   FAIL-first: the seat's former keep rule, every position or none, keeps 0
//!   on the edit and the first clause is red; shown by that rule restored in
//!   `app::arch::mimo2`'s `Keep for Body` (`keepable` = `if n >= pos { pos }
//!   else { 0 }`, `cut` refusing any other position by name).
//!
//! Then the server is stopped by its handle and waited for (its host set is
//! free before the next load), and one server of `--parallel 2` over `2 ·`
//! [`CTX`] positions (a slot of [`CTX`]) answers the same prompts together.
//! A second slot's stores take card room, so the one-slot server's budget
//! would plan it fewer card experts, and another card/host split answers
//! other ids. Its `BLOOMERY_CARD_BUDGET` is therefore the one-slot server's
//! plus one slot's stores: the room the experts are planned in is the same,
//! and so is the split:
//!
//! - `two_slots_decode_as_one_slot_runs`: its `plan` record names the
//!   one-slot server's `card_experts`, `host_experts` and `n_l` range (the
//!   same plan), its `load` record `slots=2` and `ctx` [`CTX`], its
//!   `listening` record `slots=2`, its `parallel` line the split with
//!   `pass=one`, and its
//!   resident bytes outside its scratch (the `load` record's `scratch_bytes`)
//!   exceed the one-slot server's by exactly one slot's stores (the plan's
//!   own count, `KvLayout::bytes` over the layers at [`CTX`]), and its
//!   scratch exceeds the one-slot server's: a load of two sequences also
//!   holds a pass's buffers and the card experts' buffers of two rows, which
//!   the plan's fixed scratch term covers and a load of one sequence does not
//!   allocate.
//!   Three rounds, each of two requests posted at once at
//!   `cache_prompt: false` (the second is admitted while the first holds its
//!   slot, on a body of any speed) — the batch prompt with the prose prompt,
//!   the prose prompt with the batch prompt, the chat prompt with the prose
//!   prompt — answer each prompt's ids bit for bit as the one-slot server
//!   answered them, the two requests of a round on different slots
//!   (`id_slot`), and the server's
//!   `swaps_total` is 0 or absent (the slots are resident, no state is
//!   parked). The one-slot server's ids are the session's
//!   (`server_ids_are_the_session_ids`), so this is the session's ids by
//!   transitivity, and a second server is the only added load. FAIL-first: a
//!   `swap_seq` that exchanges nothing leaves both slots on the live stores,
//!   the slots overwrite each other's rows and the ids move; a `seq_bytes`
//!   of 0 leaves the resident bytes short of the stores; a budget without
//!   the added stores plans another split and the plan line is red.
//!
//! One server more, `--ctx` [`CTX`], `--parallel 1`, `--cache-ram 1024`,
//! `--slot-save-path` (the clause's `erase`), its own load:
//!
//! - `prefix_c_restores_the_nearest_checkpoint_from_the_host_cache` and
//!   `prefix_c_ids_are_a_fresh_runs` ([`prefix_restore`]): prompt A of
//!   [`PREFIX_LEN`] ids, B another prompt that takes the slot (A's state is
//!   saved), then C sharing [`PREFIX_SHARED`] of A's ids. C's `cache_n` is
//!   that shared prefix — a window ring keeps it, there is no checkpoint to
//!   fall back to — and C's ids equal C fed fresh (`cache_prompt: false`).
//!   FAIL-first: a snapshot that copies no rows, or a resume that writes
//!   none back, moves C's ids; a seat that still refuses the snapshot keeps
//!   `cache_n` 0.
//!
//! Then one server more of the one-slot shape (`--parallel 1` at [`CTX`], the
//! one-slot server's budget), under `BLOOMERY_PREFILL_GROUP=1`:
//!
//! - `load_group_follows_the_lever`: the one-slot server's `load` record, run
//!   with the lever unset, holds `group` the row's default
//!   (`bloomery_levers::PREFILL_GROUP_DEFAULT`, 2); this server's holds 1, on
//!   the one-slot server's plan (the plan reserves the batch's shared buffers
//!   and first unit, not the units past it) and with fewer resident bytes (a group of one holds one unit of
//!   the batch's buffers, the default's group more, and the load makes them
//!   once). FAIL-first: a seat that pins the group to 1 records `group=1` for
//!   the default server and the clause is red; a seat that does not act on the
//!   lever refuses `BLOOMERY_PREFILL_GROUP=1` by name at `main`, never
//!   listens, and the clause is red.
//!
//! Then, with every server stopped:
//!
//! - `server_ids_are_the_session_ids`: the session opened in process at [`CTX`]
//!   on the gate card (`shared/mimo2_open.rs`) plans the split of the
//!   one-slot server's `plan` record (`card_experts`, `host_experts`, the
//!   `n_l` range: the server ran at the gate card's bytes, which the gate
//!   plan is), each of the three prompts fed as the server feeds it — the
//!   prompt less its last id, then a step of the last, then `N − 1` steps at
//!   argmax — answers the server's ids bit for bit; the batch prompt's first
//!   id is ik's argmax at the set's last position or a named tie inside the
//!   e2e gate's rule (`ik_last`, `tie_allowed`). FAIL-first: a session on
//!   another split (the gate plan's budget moved) is red on the plan line. On a census of two or more cards (a child started by
//!   hand) the two session clauses are a `deferred(one card)` line: the gate
//!   plan (`shared/gate_card.rs`) runs on one visible card.
//! - `ids_see_an_id_short`: the same session fed a prompt one id short does
//!   not answer the server's ids for at least one prompt, so the clause above
//!   can see a seat that feeds one id short. FAIL-first: a server fed one id
//!   short answers the session's short run, and `server_ids_are_the_session_ids`
//!   is red on the prompt whose ids move.

#[cfg(all(feature = "mimo2", feature = "glm5next"))]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_mimo2_serve", gate::run())
}

#[cfg(all(feature = "mimo2", feature = "glm5next"))]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(all(feature = "mimo2", feature = "glm5next"))]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(all(feature = "mimo2", feature = "glm5next"))]
#[path = "shared/mimo2_open.rs"]
mod mimo2_open;

#[cfg(all(feature = "mimo2", feature = "glm5next"))]
mod gate {
    use std::io::{BufRead, BufReader};
    use std::num::NonZeroUsize;
    use std::path::{Path, PathBuf};
    use std::process::ExitStatus;
    use std::time::{Duration, Instant};

    use bloomery_gpu_gates::ctx_ladder::{
        STEP, ctx_line, default_line_held, field, floor_of, short_of, step_above,
    };
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::record::{self, Log};
    use bloomery_gpu_gates::serve_client::{
        Answer, PrefixRestore, PrefixTurns, Served, agree, chat_is_those_ids, check, curl,
        edit_and_extension, greedy, held_ms, json_of, metric, no_residency_host, prefix_restore,
        rendered, residency_lever_is, serve_cmd, server_log, together, tokenized, utf8,
    };
    use bloomery_gpu_gates::slots_gate;
    use bloomery_gpu_gates::{GateError, RefManifest, checks_failed, data_dir, ref_model_path};
    use bloomery_levers::CARD_BUDGET;
    use gguf::Split;
    use model::arch::mimo2::place::PlanInputs;
    use model::placement::PlanLevers;
    use model::placement::workstation::{HostNeed, HostRead, PLAN_ROOM, RTX_3090};
    use refset::arch::mimo2::{BATCH, IK};
    use serde_json::{Value, json};

    use crate::e2e::ik_last;
    use crate::mimo2_open::{N_VOCAB, last_argmax, open};

    const USAGE: &str = "usage: gate_mimo2_serve --dir <dir> [--after-refusals]";

    /// The tokens each request makes.
    const N: usize = 16;

    /// Ids of prompt A on the prefix server, and how many of them C shares.
    /// A's written end after [`N`] tokens is `PREFIX_LEN + N - 1`, and the
    /// depth back to the shared prefix stays inside a window ring's cut.
    const PREFIX_LEN: usize = 64;
    const PREFIX_SHARED: usize = 32;

    /// The texts the prefix clause repeats. A and B differ at the first id;
    /// C's tail diverges from A at [`PREFIX_SHARED`].
    const PREFIX_A: &str = "The lighthouse keeper counted the steps. ";
    const PREFIX_B: &str = "A copper kettle whistled on the stove. ";
    const PREFIX_TAIL: &str = "Then the foghorn answered from the reef.";

    /// `--cache-ram` on the prefix server, in MiB: a set budget, so the
    /// default's holding rule does not turn the cache off.
    const PREFIX_RAM_MIB: &str = "1024";

    /// Cache rows of the loaded server and of the session it is held to: the
    /// 4,096-position set's step at position 4,096, with a margin of whole 64
    /// positions (`gate_mimo2_e2e`'s).
    const CTX: usize = 4160;

    /// The routed layers and experts a layer holds, from the header: the
    /// dense layer 0 and 47 routed layers of 256 experts each, the card's and
    /// the host's together: 47 · 256.
    const ROUTED_LAYERS: u64 = 47;
    const EXPERTS_PER_LAYER: u64 = 256;
    const ROUTED_EXPERTS: u64 = ROUTED_LAYERS * EXPERTS_PER_LAYER;

    /// The model's layers, the dense one included.
    const LAYERS: u64 = 48;

    /// The slots a seat with no `--parallel` and no `--ctx` serves.
    const DEFAULT_SLOTS: u64 = 1;

    /// The clauses the child of a census of two or more owes: each prints its
    /// own `check` line, and the parent's verdict needs every one.
    const CHILD_CLAUSES: [&str; 11] = [
        "load_and_listen",
        "completion_ids",
        "chat_is_those_ids",
        "edit_resend_keeps_the_row_where_it_diverges",
        "edit_resend_ids_are_a_fresh_runs",
        "extension_keeps_every_held_position",
        "two_slots_decode_as_one_slot_runs",
        "server_ids_are_the_session_ids",
        "ids_see_an_id_short",
        "prefix_c_restores_the_nearest_checkpoint_from_the_host_cache",
        "prefix_c_ids_are_a_fresh_runs",
    ];

    /// How long a refused server may take to exit: it refuses before any
    /// load.
    const REFUSE_WITHIN: Duration = Duration::from_secs(30);

    /// A `--plan` reads the file's header and exits; it opens no card.
    const PLAN_WITHIN: Duration = Duration::from_secs(120);

    /// The words the slot split's refusal names (`--ctx 1 --parallel 2`): the
    /// flags, then the split's one owner (`placement::slots::split_ctx`, through
    /// `bind::check_slot_ctx` before the driver is asked): the total, the slots,
    /// the share and the floor.
    // PIN(2026-10-30): the seat's flags are checked before the driver (b1b138ec8), so
    // the split owner's words refuse first; were `ctx_ask`'s "--ctx 1 over 2 slots
    // gives a slot no position" and "ask at least 2", and the clause was red on them.
    const SPLIT_REFUSAL: [&str; 2] = [
        "--ctx-size 1 --parallel 2",
        "1 positions across 2 slots give a slot 0, below the floor of 1",
    ];

    /// The words the refusal of a set total the card cannot hold names
    /// (`ctx_ask::AskError::DoesNotFit`): a slot's share against the card's
    /// bytes, which the stores of every slot are counted in.
    const PLAN_REFUSAL: [&str; 2] = ["positions a slot, and the card holds", "bytes that takes"];

    /// The words the refusal of an unset `--ctx` that holds not even the floor
    /// names (`ctx_ask::AskError::FloorShort`) on a host that cannot hold the
    /// experts the card leaves it.
    const FLOOR_HOST_REFUSAL: [&str; 2] = ["the least the default tries", "on a host that holds"];

    /// How far under the floor's host need the refused room sits.
    const SHORT_OF_NEED: u64 = 1 << 30;

    /// The card budget of the servers whose ids are compared: the gate
    /// card's usable bytes, the bytes `shared/gate_card.rs` plans the session
    /// on whichever card is in view, so a server on the A6000 plans the
    /// session's split.
    fn gate_budget() -> u64 {
        RTX_3090.usable_bytes()
    }

    /// The text the seat prints for a `--place bp --parallel 2` it refuses
    /// before the plan: the body serves its expert tier card beside one
    /// sequence (the seat's `TIER_SLOTS`).
    fn tier_slots_refusal() -> String {
        format!(
            "--place {} with --parallel 2: the mimo2 body serves its expert tier card beside one \
             sequence",
            Place::Bp.name()
        )
    }

    /// The placement the seat's unset `--place` runs on a census of `cards`:
    /// plan (b′) on two, whose tier holds past the seat's break-even, and
    /// `a` on one.
    fn unset_place(cards: usize) -> Place {
        if cards >= 2 { Place::Bp } else { Place::A }
    }

    /// The server loads the whole host set: up to this many polls a second
    /// from a cold page cache.
    const LISTEN_POLLS: usize = 900;

    /// The chat turn the file answers, as the user message of [`EDIT_A`].
    const EDIT_A: &str =
        "Name the three primary colors of light, and say in one sentence why a screen mixes them.";

    /// The edit clause's turn changed: the resend diverges inside the first
    /// turn's prompt, with the rest of the prompt after it.
    const EDIT_B: &str =
        "Name the three primary colors of ink, and say in one sentence why a page reflects them.";

    /// The raw prompt the file continues.
    const PROSE: &str = "The lighthouse keeper counted the steps as he climbed: one hundred and twelve, the same as every night, until the hundred and thirteenth";

    /// The extension clause's reply stand-in and later user turn: the
    /// template renders them in a conversation, and the text past the reply
    /// stand-in is the later turn's ids ([`later_ids`]).
    const MARK: &str = "QXZMARKQXZ";
    const LATER_USER: &str =
        "And which of the three does a screen show when it shows none of them?";

    struct Args {
        dir: PathBuf,
        /// The parent ran the two clauses before the server's.
        after_refusals: bool,
    }

    fn parse_args() -> Result<Args, GateError> {
        let mut dir = None;
        let mut after_refusals = false;
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            if flag == "--after-refusals" {
                after_refusals = true;
                continue;
            }
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
            match flag.as_str() {
                "--dir" => dir = Some(PathBuf::from(v)),
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        Ok(Args {
            dir: dir.ok_or(USAGE)?,
            after_refusals,
        })
    }

    /// The chat turn the file answers.
    fn messages() -> Value {
        json!([{ "role": "user", "content": EDIT_A }])
    }

    /// The levers a server this gate starts runs under, whatever the gate's
    /// own environment holds of them; an unset field leaves the lever as the
    /// environment has it.
    #[derive(Clone, Copy, Default)]
    struct Pins<'a> {
        /// `BLOOMERY_RESIDENCY`: every server but
        /// `residency_default_is_the_machine`'s runs it `off`, for the ids a
        /// server answers are compared with a static session's and a
        /// one-slot server's, which the residency machine's adapting split
        /// would move.
        residency: Option<&'a str>,
        /// `BLOOMERY_CARD_BUDGET` in bytes: the plan takes `min(usable,
        /// budget)` of the card.
        budget: Option<u64>,
        /// `BLOOMERY_HOST_ROOM` in bytes: the host room the plan reads. Set
        /// for `--plan` and refusal children only; a loaded server keeps the
        /// host's own reading, for the lever also replaces the load's
        /// host-need check.
        room: Option<u64>,
        /// `BLOOMERY_PREFILL_GROUP`: only `load_group_follows_the_lever`'s
        /// second server sets it; every other server runs it unset, the
        /// row's default, since this gate's own `at_main` refuses the lever
        /// in the environment its servers inherit.
        group: Option<&'a str>,
    }

    /// The residency machine `off`, nothing else pinned.
    const OFF: Pins<'static> = Pins {
        residency: Some("off"),
        budget: None,
        room: None,
        group: None,
    };

    /// `bloomery-serve --model mimo2 -m <file> <args>` beside this binary, its
    /// logs in `<dir>/<name>/`, with the machine `off` and
    /// `BLOOMERY_CARD_BUDGET` of `budget` bytes: the plan takes `min(usable,
    /// budget)` of the card, so a server on the A6000 plans a smaller card's
    /// room ([`gate_budget`]).
    fn spawn_budgeted(
        dir: &Path,
        name: &str,
        file: &Path,
        args: &[&str],
        budget: u64,
    ) -> Result<(Served, PathBuf), GateError> {
        let pins = Pins {
            budget: Some(budget),
            ..OFF
        };
        spawn_levered(dir, name, file, args, pins)
    }

    /// `bloomery-serve --model mimo2 -m <file> <args>` beside this binary, its
    /// logs in `<dir>/<name>/`, under `pins`.
    fn spawn_levered(
        dir: &Path,
        name: &str,
        file: &Path,
        args: &[&str],
        pins: Pins<'_>,
    ) -> Result<(Served, PathBuf), GateError> {
        let d = dir.join(name);
        std::fs::create_dir_all(&d)?;
        let mut all = vec!["--model", "mimo2", "-m", utf8(file)?];
        all.extend_from_slice(args);
        let mut cmd = serve_cmd()?;
        if let Some(word) = pins.residency {
            cmd.env(bloomery_levers::RESIDENCY, word);
        }
        if let Some(bytes) = pins.budget {
            cmd.env(bloomery_levers::CARD_BUDGET, bytes.to_string());
        }
        if let Some(bytes) = pins.room {
            cmd.env(bloomery_levers::HOST_ROOM, bytes.to_string());
        }
        if let Some(group) = pins.group {
            cmd.env(bloomery_levers::PREFILL_GROUP, group);
        }
        Ok((Served::spawn_cmd(cmd, &all, &d)?, d))
    }

    /// The card/host split a `plan` record names: the experts on each side
    /// and the range of experts a held layer keeps. Equal splits answer equal
    /// ids; the card budget is not part of it, two plans reach one split from
    /// different budgets.
    #[derive(Debug, PartialEq, Eq)]
    struct Planned {
        card: u64,
        host: u64,
        n_l: String,
    }

    impl Planned {
        fn of(plan: &record::Fields) -> Result<Planned, GateError> {
            Ok(Planned {
                card: plan.u64("card_experts")?,
                host: plan.u64("host_experts")?,
                n_l: plan.word("n_l")?.to_owned(),
            })
        }
    }

    /// A plan's `n_l` word (`record::plan`'s): the least and the greatest
    /// count among the layers that keep any expert on the card.
    fn n_l_range(n_l: &[u64]) -> String {
        let held = || n_l.iter().copied().filter(|&n| n > 0);
        format!(
            "{}..{}",
            held().min().unwrap_or(0),
            held().max().unwrap_or(0)
        )
    }

    /// A server of `args` under `pins` run to its exit within `within`: its
    /// status (`None`: still running at the bound, killed) and its stderr's
    /// path.
    fn run_pinned(
        dir: &Path,
        name: &str,
        file: &Path,
        args: &[&str],
        (within, pins): (Duration, Pins<'_>),
    ) -> Result<(Option<ExitStatus>, PathBuf), GateError> {
        let (mut s, d) = spawn_levered(dir, name, file, args, pins)?;
        let status = wait_exit(&mut s, within)?;
        Ok((status, d.join("server.err")))
    }

    /// `s`'s exit status; a server still running at `within` is stopped and
    /// gives `None`.
    fn wait_exit(s: &mut Served, within: Duration) -> Result<Option<ExitStatus>, GateError> {
        let t = Instant::now();
        loop {
            if let Some(st) = s.child.try_wait()? {
                return Ok(Some(st));
            }
            if t.elapsed() > within {
                let _ = s.stop()?;
                return Ok(None);
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    /// `--plan` of `args` at [`PLAN_ROOM`], this workstation's host room with
    /// nothing loaded, given: the plan the clauses compare with one another's
    /// is the same whatever the box's `MemAvailable` reads. Its exit status
    /// and the log of its stderr.
    fn plan_run(
        dir: &Path,
        name: &str,
        file: &Path,
        args: &[&str],
    ) -> Result<(Option<ExitStatus>, Log, String), GateError> {
        let pins = Pins {
            room: Some(PLAN_ROOM),
            ..OFF
        };
        plan_pinned(dir, name, file, args, pins)
    }

    /// `--plan` of `args` under `pins`.
    fn plan_pinned(
        dir: &Path,
        name: &str,
        file: &Path,
        args: &[&str],
        pins: Pins<'_>,
    ) -> Result<(Option<ExitStatus>, Log, String), GateError> {
        let mut all = args.to_vec();
        all.push("--plan");
        let (status, err) = run_pinned(dir, name, file, &all, (PLAN_WITHIN, pins))?;
        let text = std::fs::read_to_string(&err).unwrap_or_default();
        println!(
            "{name}: {} --plan: {}",
            args.join(" "),
            status.map_or("still running at the bound".to_owned(), |s| s.to_string())
        );
        Ok((
            status,
            server_log(&err, record::BLOOMERY_SERVE_MIMO2)?,
            text,
        ))
    }

    /// The file's trained context, `None` when it states none.
    fn trained_ctx(file: &Path) -> Result<Option<u64>, GateError> {
        let split = Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
        Ok(split.arch_get_u64("context_length"))
    }

    /// The cards in view: the driver's census, the cards the server this
    /// gate starts sees. The census holds one context on each; a plan only
    /// names a holder in its refusal text and reads the free bytes.
    fn visible_cards() -> Result<usize, GateError> {
        Ok(bloomery_gpu::census()?.len())
    }

    /// The `CUDA_VISIBLE_DEVICES` of the child that runs the clauses after
    /// the refusals: the A6000, found by its name in the census and named by
    /// the UUID the driver reports for it, the form `nvidia-smi` lists.
    ///
    /// # Errors
    /// A census with no A6000.
    fn a6000_entry() -> Result<String, GateError> {
        let census = bloomery_gpu::census()?;
        let device = census
            .iter()
            .find(|d| d.name.contains("A6000"))
            .ok_or_else(|| {
                format!(
                    "the child that runs the clauses after the refusals takes the A6000 and the \
                     census has none: {}",
                    census
                        .iter()
                        .map(|d| d.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;
        Ok(bloomery_gpu_gates::gpu_census::gpu_uuid(&device.uuid))
    }

    /// This binary again with `--after-refusals` and the A6000 alone in view
    /// ([`a6000_entry`]): the clauses after the refusals. Its output goes to
    /// this process's streams; its exit status is the verdict of the clauses
    /// it ran.
    fn child_clauses(dir: &Path, ok: &mut bool) -> Result<(), GateError> {
        let entry = a6000_entry()?;
        let child_dir = dir.join("child");
        println!(
            "child: the clauses after the refusals run in a child on the A6000 \
             (CUDA_VISIBLE_DEVICES {entry}), logs in {}",
            child_dir.display()
        );
        let mut child = std::process::Command::new(std::env::current_exe()?)
            .arg("--dir")
            .arg(&child_dir)
            .arg("--after-refusals")
            .env("CUDA_VISIBLE_DEVICES", &entry)
            .stdout(std::process::Stdio::piped())
            .spawn()?;
        let mut seen = Vec::new();
        let out = child
            .stdout
            .take()
            .ok_or("the child's stdout is not piped")?;
        for line in BufReader::new(out).lines() {
            let line = line?;
            println!("{line}");
            seen.push(line);
        }
        let status = child.wait()?;
        println!("child: {status}");
        // An exit 0 is not the verdict until every clause it owes printed its PASS: a
        // child that ran nothing exits 0 too.
        let missing: Vec<&str> = CHILD_CLAUSES
            .iter()
            .copied()
            .filter(|c| !seen.iter().any(|l| *l == format!("check {c}: PASS")))
            .collect();
        if !missing.is_empty() {
            println!("child: no `check <clause>: PASS` line for {missing:?}");
        }
        check(
            ok,
            "clauses_after_the_refusals_in_the_a6000_child",
            status.success() && missing.is_empty(),
        );
        Ok(())
    }

    /// What a `--plan` with no flag printed of the default context: the `ctx`
    /// line and the plan's `ctx_max` (a slot's share of the total).
    struct Defaulted {
        ctx_line: String,
        ctx_max: u64,
    }

    /// `plan_only_names_its_default` (the module header): the default it
    /// printed, `None` when it printed none.
    fn plan_only_names_its_default(
        dir: &Path,
        file: &Path,
        cards: usize,
        ok: &mut bool,
    ) -> Result<Option<Defaulted>, GateError> {
        let trained = trained_ctx(file)?;
        let (status, log, text) = plan_run(dir, "plan", file, &[])?;
        let place = log.first(&record::PLACE_UNSET)?;
        let plan = log.first(&record::PLAN)?;
        let parallel_at = text
            .lines()
            .position(|l| l.starts_with("parallel "))
            .map(|i| i + 1);
        let parallel_line = text.lines().find(|l| l.starts_with("parallel "));
        let in_order = matches!(
            (&place, &plan, parallel_at),
            (Some(a), Some(b), Some(c)) if a.at() < b.at() && b.at() < c
        );
        let unset = unset_place(cards).name();
        let word_a = match &place {
            Some(p) => p.word("place")? == unset,
            None => false,
        };
        let no_load = log.first(&record::LOAD_MIMO2)?.is_none()
            && log.first(&record::LISTENING_MIMO2)?.is_none();
        let exits = status.is_some_and(|s| s.success());
        println!(
            "plan only: exit 0 {exits}; place unset, plan, parallel in order {in_order}; place word \
             {unset} {word_a}; no load or listening record {no_load}"
        );
        let ctx_max = match &plan {
            Some(p) => Some(p.u64("ctx_max")?),
            None => None,
        };
        let mut default_ok = false;
        if let (Some(c), Some(limit)) = (ctx_max, trained) {
            // The default is one slot of `ctx_max` positions.
            let want = format!(
                "parallel rule=slots slots={DEFAULT_SLOTS} slot_ctx={c} total={c} from=default \
                 pass=one"
            );
            let line_ok = parallel_line == Some(want.as_str());
            println!("plan only: the parallel line {parallel_line:?} is {want:?} {line_ok}");
            // The `ctx` line holds its word and names the plan's context.
            let line = ctx_line(&text);
            let held = line.is_some_and(|l| {
                default_line_held(l, DEFAULT_SLOTS, (limit, "trained")) == Some(true)
                    && field(l, "ctx") == Some(c)
            });
            println!("plan only: the ctx line {line:?} holds its word at {c} {held}");
            // The plan at that context stands: asked for by flag it exits 0
            // and plans the same slot context.
            let c_arg = c.to_string();
            let (st, at_log, _) = plan_run(
                dir,
                "plan-at-default",
                file,
                &["--parallel", "1", "--ctx", &c_arg],
            )?;
            let stands = st.is_some_and(|s| s.success())
                && match at_log.first(&record::PLAN)? {
                    Some(p) => p.u64("ctx_max")? == c,
                    None => false,
                };
            // The step above it does not: the default fitted only under a
            // context the machine could not hold.
            let above_short = match step_above(limit, c) {
                None => {
                    println!("plan only: the default {c} is the target, none above it");
                    true
                }
                Some(above) => {
                    let above_arg = above.to_string();
                    let (st, _, _) = plan_run(
                        dir,
                        "plan-past-default",
                        file,
                        &["--parallel", "1", "--ctx", &above_arg],
                    )?;
                    let refused = st.is_some_and(|s| !s.success());
                    println!("plan only: --ctx {above}, the step above, is refused {refused}");
                    refused
                }
            };
            println!("plan only: {c} stands {stands}");
            default_ok = line_ok && held && stands && above_short;
        }
        check(
            ok,
            "plan_only_names_its_default",
            exits && in_order && word_a && no_load && default_ok,
        );
        Ok(ctx_max
            .zip(ctx_line(&text))
            .map(|(ctx_max, line)| Defaulted {
                ctx_line: line.to_owned(),
                ctx_max,
            }))
    }

    fn default_load_listens(dir: &Path, file: &Path) -> Result<bool, GateError> {
        let pins = Pins {
            residency: None,
            ..OFF
        };
        let (mut s, d) = spawn_levered(dir, "default", file, &["--port", "0"], pins)?;
        let err_log = d.join("server.err");
        let listened = s.address(&err_log, LISTEN_POLLS, Duration::from_secs(1));
        let text = std::fs::read_to_string(&err_log).unwrap_or_default();
        let Ok(addr) = listened else {
            println!("default: the server never listened; server stderr:\n{text}");
            return Ok(false);
        };
        let records = server_log(&err_log, record::BLOOMERY_SERVE_MIMO2)?;
        let load = records.one(&record::LOAD_MIMO2)?;
        let listen = records.one(&record::LISTENING_MIMO2)?;
        let named = ctx_line(&text).and_then(|l| field(l, "ctx"));
        println!(
            "default at {addr}: ctx line {:?}; load record {:?}; listening record {:?}; {}",
            ctx_line(&text),
            load.line(),
            listen.line(),
            s.stop()?
        );
        Ok(named.is_some() && named == Some(load.u64("ctx")?) && named == Some(listen.u64("ctx")?))
    }

    /// The host bytes the seat's plan of `file` needs at `slot_ctx` positions
    /// a slot, [`DEFAULT_SLOTS`] slots, on the placement a flagless `--plan`
    /// takes (`a`) at [`PLAN_ROOM`]: the plan the server prints, made here.
    fn default_plan_need(
        file: &Path,
        levers: &bloomery_levers::Levers,
        (slot_ctx, cards): (u64, usize),
    ) -> Result<u64, GateError> {
        let split = Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let mut inputs = PlanInputs::describe(&split)?;
        inputs.room = (PLAN_ROOM, HostRead::Given);
        let place = unset_place(cards).on_host()?;
        let batch = (!place.tier_cards().is_empty())
            .then(|| bloomery_gpu_mimo2::tier::tier_batch(&inputs))
            .transpose()?;
        let machine = place.machine(None, batch)?(inputs.model.layers);
        let slots = NonZeroUsize::new(usize::try_from(DEFAULT_SLOTS)?).ok_or("no default slot")?;
        let plan =
            inputs.plan_with_slots(&machine, slot_ctx, &PlanLevers::from_levers(levers)?, slots)?;
        Ok(HostNeed::of(&plan, 0).bytes())
    }

    /// `host_room_halves_the_default` (the module header); `base` is the
    /// default `plan_only_names_its_default` printed.
    fn host_room_halves_the_default(
        dir: &Path,
        file: &Path,
        (levers, cards): (&bloomery_levers::Levers, usize),
        base: Option<&Defaulted>,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        const NAME: &str = "host_room_halves_the_default";
        let Some(base) = base else {
            println!("{NAME}: no default to hold the rooms to");
            check(ok, NAME, false);
            return Ok(());
        };
        // (i) At the plan room the default is the one
        // `plan_only_names_its_default` printed, and no host fitted it.
        let (status, log, text) = plan_run(dir, "plan-room-default", file, &[])?;
        let line = ctx_line(&text);
        let ctx_max = match log.first(&record::PLAN)? {
            Some(p) => Some(p.u64("ctx_max")?),
            None => None,
        };
        let not_host = line.is_some_and(|l| short_of(l) != Some("host"));
        let same = status.is_some_and(|s| s.success())
            && line == Some(base.ctx_line.as_str())
            && ctx_max == Some(base.ctx_max);
        println!(
            "host room, at {PLAN_ROOM} B: the default {line:?} at ctx_max {ctx_max:?} is \
             plan_only's {:?} at {} {same}; no host fitted it {not_host}",
            base.ctx_line, base.ctx_max
        );
        let limit = trained_ctx(file)?.ok_or("the file states no context_length")?;
        let floor = floor_of(limit);
        let below = (base.ctx_max / 2 / STEP * STEP).max(floor);
        if below >= base.ctx_max {
            println!(
                "skip {NAME} (the fit and the refusal): the default {} is the floor, no shorter \
                 context for a room to fit to",
                base.ctx_max
            );
            check(ok, NAME, same && not_host);
            return Ok(());
        }
        // (ii) A room between the host need of half the default and the
        // default's: the default's experts pass the room, the half's do not,
        // so the default fits under it, at least at the half, short of the
        // host.
        let (need_below, need_default) = (
            default_plan_need(file, levers, (below, cards))?,
            default_plan_need(file, levers, (base.ctx_max, cards))?,
        );
        if need_default <= need_below {
            return Err(format!(
                "the default's plan needs {need_default} B of host, not more than the half \
                 below's {need_below} B: the model's host need no longer grows with the context"
            )
            .into());
        }
        let between = need_below + (need_default - need_below) / 2;
        let pins = |room| Pins {
            room: Some(room),
            ..OFF
        };
        let (status, log, text) = plan_pinned(dir, "plan-room-between", file, &[], pins(between))?;
        // PIN(2026-10-28): the default is fitted, not halved; the clause pins the largest context the room holds, not one context.
        let fitted_to = ctx_line(&text)
            .filter(|l| l.starts_with("ctx rule=fitted ") && short_of(l) == Some("host"))
            .and_then(|l| field(l, "ctx"))
            .filter(|&c| below <= c && c < base.ctx_max);
        let planned = match log.first(&record::PLAN)? {
            Some(p) => Some(p.u64("ctx_max")?),
            None => None,
        };
        let need_fitted = fitted_to
            .map(|c| default_plan_need(file, levers, (c, cards)))
            .transpose()?;
        let need_above = fitted_to
            .map(|c| default_plan_need(file, levers, (c + STEP, cards)))
            .transpose()?;
        let fitted = status.is_some_and(|s| s.success())
            && fitted_to.is_some()
            && planned == fitted_to
            && need_fitted.is_some_and(|n| n <= between)
            && need_above.is_some_and(|n| n > between);
        println!(
            "host room {between} B (the half's need {need_below} B, the default's \
             {need_default} B): the line {:?} fits to {fitted_to:?} from {below}, needing \
             {need_fitted:?} B, host-short {fitted}",
            ctx_line(&text)
        );
        if !fitted {
            println!("plan-room-between stderr:\n{text}");
        }
        // (iii) A room a GiB under the floor's need holds no context: refused
        // by name before any load.
        let short = default_plan_need(file, levers, (floor, cards))? - SHORT_OF_NEED;
        let refused = refusal_under(
            dir,
            file,
            "refuse-room",
            &["--plan"],
            (&[FLOOR_HOST_REFUSAL[0], FLOOR_HOST_REFUSAL[1]], pins(short)),
        )?;
        check(ok, NAME, same && not_host && fitted && refused);
        Ok(())
    }

    /// `residency_default_is_the_machine` (the module header): `--plan` at
    /// [`CTX`] with no residency word prints the `residency lever` record at
    /// the serving word and why `place`, ahead of the plan, and the `residency
    /// host` record of the churn pool after it; at the largest total two
    /// slots hold, whose KV planes leave the plan no card expert to spare —
    /// the `ask N or less` of the seat's refusal of the trained context a
    /// slot — the common rule turns it `off` with no pool, why `no_card_experts`
    /// where the plan holds no card expert and `no_room` where the leftover of
    /// the last rung holds a few; and under
    /// `BLOOMERY_RESIDENCY=off` it prints `off` and why `set`, and no pool.
    fn residency_default_is_the_machine(
        dir: &Path,
        file: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let ctx = CTX.to_string();
        let (status, log, _) = plan_pinned(
            dir,
            "plan-residency",
            file,
            &["--ctx", &ctx],
            Pins::default(),
        )?;
        let (lever, plan, host) = (
            log.first(&record::RESIDENCY_LEVER)?,
            log.first(&record::PLAN)?,
            log.first(&record::RESIDENCY_HOST)?,
        );
        let word = residency_lever_is(&log, "mid-p0-s1", "place")?;
        let in_order = matches!(
            (&lever, &plan, &host),
            (Some(l), Some(p), Some(h)) if l.at() < p.at() && p.at() < h.at()
        );
        // PIN(2026-10-14): was the seat's default context, whose KV left no expert
        // to spare at the old two default slots; the default is one slot on the
        // ladder now (ctx-first), which holds experts on the card.
        let trained = trained_ctx(file)?.ok_or("the file states no context_length")?;
        let past = (2 * trained).to_string();
        let (_, _, said) = plan_pinned(
            dir,
            "plan-residency-past",
            file,
            &["--parallel", "2", "--ctx", &past],
            Pins::default(),
        )?;
        let largest = said
            .split("; ask ")
            .nth(1)
            .and_then(|r| r.split(" or less").next())
            .and_then(|n| n.parse::<u64>().ok())
            .ok_or_else(|| format!("no `ask N or less` in the refusal of --ctx {past}: {said}"))?
            .to_string();
        let (_, wide, _) = plan_pinned(
            dir,
            "plan-residency-largest",
            file,
            &["--parallel", "2", "--ctx", &largest],
            Pins::default(),
        )?;
        // PIN(2026-10-24): was `no_card_experts` alone. The largest total is the greatest multiple of
        // a rung a slot holds, so the card keeps under a rung of two slots' KV, which is under two
        // experts a layer: no expert (`no_card_experts`), or a few the plan spreads over its layers,
        // whose fewest (1) leaves the word no room (`no_room`). The plan's card experts say which.
        let held = wide
            .first(&record::PLAN)?
            .ok_or("no plan record at the largest total")?
            .u64("card_experts")?;
        let why = if held == 0 {
            "no_card_experts"
        } else {
            "no_room"
        };
        let seen = match wide.first(&record::RESIDENCY_LEVER)? {
            Some(l) => format!("{} ({})", l.word("residency")?, l.word("why")?),
            None => "no residency lever record".to_owned(),
        };
        let no_room = residency_lever_is(&wide, "off", why)? && no_residency_host(&wide)?;
        let (_, off_log, _) = plan_pinned(dir, "plan-residency-off", file, &["--ctx", &ctx], OFF)?;
        let off = residency_lever_is(&off_log, "off", "set")? && no_residency_host(&off_log)?;
        let exits = status.is_some_and(|s| s.success());
        println!(
            "residency default: exit 0 {exits}; at ctx {ctx} the lever record is mid-p0-s1 (place), \
             then the plan, then the pool {word} {in_order}; at --ctx {largest} over two slots \
             the plan holds {held} card experts and the lever is {seen}: want off ({why}) with no \
             pool {no_room}; set off prints off (set) and no pool {off}"
        );
        check(
            ok,
            "residency_default_is_the_machine",
            exits && word && in_order && no_room && off,
        );
        Ok(())
    }

    /// A server of `args` refused before any load: it exits non-zero within
    /// [`REFUSE_WITHIN`], its stderr holds every word of `want` and it
    /// printed no `load` record. A server still running at the bound is
    /// killed and the refusal is red.
    fn refusal(
        dir: &Path,
        file: &Path,
        name: &str,
        args: &[&str],
        want: &[&str],
    ) -> Result<bool, GateError> {
        refusal_under(dir, file, name, args, (want, OFF))
    }

    /// [`refusal`] with the server under `pins`.
    fn refusal_under(
        dir: &Path,
        file: &Path,
        name: &str,
        args: &[&str],
        (want, pins): (&[&str], Pins<'_>),
    ) -> Result<bool, GateError> {
        let (status, err) = run_pinned(dir, name, file, args, (REFUSE_WITHIN, pins))?;
        let text = std::fs::read_to_string(&err).unwrap_or_default();
        let log = server_log(&err, record::BLOOMERY_SERVE_MIMO2)?;
        let no_load = log.first(&record::LOAD_MIMO2)?.is_none();
        let named = want.iter().all(|w| text.contains(w));
        let exited = status.is_some_and(|s| !s.success());
        println!(
            "{name}: exit non-zero within {REFUSE_WITHIN:?} {exited}; names {want:?} {named}; \
             no load record {no_load}"
        );
        if !(exited && named && no_load) {
            println!("{name} stderr:\n{text}");
        }
        Ok(exited && named && no_load)
    }

    /// `refusals_before_any_load` (the module header); `cards` the visible
    /// cards of the census.
    fn refusals_before_any_load(
        dir: &Path,
        file: &Path,
        cards: usize,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let mut pass = refusal(
            dir,
            file,
            "refuse-split",
            &["--ctx", "1", "--parallel", "2", "--port", "0"],
            &SPLIT_REFUSAL,
        )?;
        if cards >= 2 {
            let want = tier_slots_refusal();
            pass &= refusal(
                dir,
                file,
                "refuse-bp",
                &["--place", "bp", "--parallel", "2", "--port", "0"],
                &[&want],
            )?;
            check(ok, "refusals_before_any_load", pass);
        } else {
            println!(
                "deferred(two cards) refusals_before_any_load (--place bp --parallel 2): {cards} card \
                 in view; the tier-slots refusal needs a census of two"
            );
            check(ok, "refusals_before_any_load (the split only)", pass);
        }
        Ok(())
    }

    /// `slots_total_is_planned` (the module header).
    fn slots_total_is_planned(dir: &Path, file: &Path, ok: &mut bool) -> Result<(), GateError> {
        let trained = trained_ctx(file)?;
        let (_, log, _) = plan_run(dir, "plan-for-slots", file, &["--parallel", "1"])?;
        let Some(plan) = log.first(&record::PLAN)? else {
            println!("slots_total_is_planned: the one-slot plan printed no plan record");
            check(ok, "slots_total_is_planned", false);
            return Ok(());
        };
        let c = plan.u64("ctx_max")?;
        let c_arg = c.to_string();
        let (status, _, text) = plan_run(
            dir,
            "plan-two-slots",
            file,
            &["--parallel", "2", "--ctx", &c_arg],
        )?;
        // `pass=one`: the seat's round of several slots is one pass of the busy rows.
        let want = format!(
            "parallel rule=slots slots=2 slot_ctx={} total={} from=flag pass=one",
            c / 2,
            2 * (c / 2)
        );
        let split = status.is_some_and(|s| s.success()) && text.lines().any(|l| l == want);
        println!(
            "two slots over the one-slot default context {c}: the parallel line {want:?} {split}"
        );
        // The step above the default the one-slot plan refused; two slots
        // plan the same total, so the refusal is the total's, not a slot's
        // share.
        let above = trained.and_then(|t| step_above(t, c));
        let refused = if let Some(above) = above {
            let next = above.to_string();
            refusal(
                dir,
                file,
                "refuse-total",
                &["--parallel", "2", "--ctx", &next, "--port", "0"],
                &PLAN_REFUSAL,
            )?
        } else {
            println!(
                "skip slots_total_is_planned (refusal half): the one-slot default {c} is the \
                 target, no total the one-slot plan refused"
            );
            true
        };
        check(ok, "slots_total_is_planned", split && refused);
        Ok(())
    }

    /// The ids of the later user turn the template renders past a reply: the
    /// text after [`MARK`] in the conversation of [`EDIT_A`], a reply of
    /// [`MARK`] and [`LATER_USER`], with the generation prompt.
    fn later_ids(url: &dyn Fn(&str) -> String) -> Result<Vec<u32>, GateError> {
        let messages = json!([
            { "role": "user", "content": EDIT_A },
            { "role": "assistant", "content": MARK },
            { "role": "user", "content": LATER_USER },
        ]);
        let (st, body) = curl(
            &url("/apply-template"),
            Some(&json!({ "messages": messages })),
            false,
        )?;
        let text = json_of("/apply-template", st, &body)?["prompt"]
            .as_str()
            .ok_or("/apply-template: no prompt")?
            .to_owned();
        let at = text
            .find(MARK)
            .ok_or_else(|| format!("the template's rendering holds no {MARK}: {text:?}"))?;
        let ids = tokenized(url, &text[at + MARK.len()..])?;
        if ids.is_empty() {
            return Err("the template renders no ids past a reply".into());
        }
        Ok(ids)
    }

    /// What the completions recorded: each prompt and the server's [`N`] ids
    /// for it.
    struct Answers {
        batch: Answer,
        prose: Answer,
        chat: Answer,
        /// The one-slot server's `load` record's resident and scratch bytes.
        resident: u64,
        scratch: u64,
        /// The prompt group of its `load` record, the lever unset.
        group: u64,
        /// The split of the one-slot server's `plan` record.
        planned: Planned,
    }

    /// The server's clauses on the one that listens, `load_and_listen` to the
    /// prefix clauses: the three answers, `None` when the server never
    /// listened.
    fn served(
        dir: &Path,
        file: &Path,
        batch: Vec<u32>,
        ok: &mut bool,
    ) -> Result<Option<Answers>, GateError> {
        let ctx = CTX.to_string();
        let (mut s, d) = spawn_budgeted(
            dir,
            "serve",
            file,
            &["--ctx", &ctx, "--parallel", "1", "--port", "0"],
            gate_budget(),
        )?;
        let err_log = d.join("server.err");
        let addr = match s.address(&err_log, LISTEN_POLLS, Duration::from_secs(1)) {
            Ok(addr) => addr,
            Err(e) => {
                println!("the server never listened: {e}");
                println!(
                    "server stderr:\n{}",
                    std::fs::read_to_string(&err_log).unwrap_or_default()
                );
                check(ok, "load_and_listen", false);
                for name in [
                    "completion_ids",
                    "chat_is_those_ids",
                    "edit_resend_keeps_the_row_where_it_diverges",
                    "edit_resend_ids_are_a_fresh_runs",
                    "extension_keeps_every_held_position",
                    "load_group_follows_the_lever",
                    "server_ids_are_the_session_ids",
                    "ids_see_an_id_short",
                ] {
                    println!("skip {name}: the server never listened");
                }
                return Ok(None);
            }
        };
        let url = |p: &str| format!("http://{addr}{p}");
        let records = server_log(&err_log, record::BLOOMERY_SERVE_MIMO2)?;
        let load = records.one(&record::LOAD_MIMO2)?;
        let listen = records.one(&record::LISTENING_MIMO2)?;
        let plan = records.one(&record::PLAN)?;
        let planned = Planned::of(&plan)?;
        let loaded = load.word("arch")? == "mimo2"
            && load.u64("slots")? == 1
            && planned.card > 0
            && load.u64("card_experts")? == planned.card
            && load.u64("host_experts")? == planned.host
            && planned.card + planned.host == ROUTED_EXPERTS
            && load.u64("layers")? == LAYERS
            && load.u64("ctx")? == CTX as u64;
        let listening = listen.u64("ctx")? == CTX as u64 && listen.u64("slots")? == 1;
        println!(
            "server {} at {addr}; plan record {:?}; load record {:?}; listening record {:?}",
            file.display(),
            plan.line(),
            load.line(),
            listen.line()
        );
        check(
            ok,
            "load_and_listen",
            load.at() < listen.at() && loaded && listening,
        );

        // The three prompts and the server's ids for each.
        let prose = tokenized(&url, PROSE)?;
        let chat = rendered(&url, messages())?;
        println!(
            "prompts: batch {} ids, prose {} ids, chat {} ids",
            batch.len(),
            prose.len(),
            chat.len()
        );
        let (batch, _) = greedy(&url, batch, N, &d, "batch", false)?;
        let (prose, _) = greedy(&url, prose, N, &d, "prose", false)?;
        let (chat, chat_v) = greedy(&url, chat, N, &d, "completion", false)?;
        let answered = [&batch, &prose, &chat].iter().all(|a| {
            !a.tokens.is_empty() && (a.tokens.len() == N || a.stop == "eos") && a.cache_n == 0
        });
        check(ok, "completion_ids", answered);

        let chat_ok = chat_is_those_ids(&url, &d, messages(), N, (&chat, &chat_v))?;
        check(ok, "chat_is_those_ids", chat_ok);

        let later = later_ids(&url)?;
        edit_and_extension(
            &url,
            &d,
            ok,
            &PrefixTurns {
                a: EDIT_A,
                b: EDIT_B,
                n: N,
                min_after: 1,
            },
            &later,
        )?;
        println!("server stopped: {}", s.stop()?);
        Ok(Some(Answers {
            batch,
            prose,
            chat,
            resident: load.u64("resident_bytes")?,
            scratch: load.u64("scratch_bytes")?,
            group: load.u64("group")?,
            planned,
        }))
    }

    /// What a request of a round answered, the slot it ran on and about how long it held it.
    struct Ran {
        tokens: Vec<u32>,
        stop: String,
        slot: Option<u64>,
        /// The server's `prompt_ms` plus `predicted_ms` ([`held_ms`]): the window a request
        /// posted beside it has to arrive in to take the other slot.
        held_ms: f64,
    }

    /// A round's two requests at once on `addr` ([`together`]), each a greedy `/completion` of
    /// [`N`] ids at `cache_prompt: false`, its body kept in `d` under its name.
    fn round(
        addr: &str,
        d: &Path,
        first: (&str, &Answer),
        second: (&str, &Answer),
    ) -> Result<[Ran; 2], GateError> {
        together(
            addr,
            (first.0, &first.1.prompt),
            (second.0, &second.1.prompt),
            |url, name, prompt| {
                let (a, v) = greedy(url, prompt.to_vec(), N, d, name, false)?;
                Ok(Ran {
                    tokens: a.tokens,
                    stop: a.stop,
                    slot: v["id_slot"].as_u64(),
                    held_ms: held_ms(name, &v)?,
                })
            },
        )
    }

    /// `two_slots_decode_as_one_slot_runs` (the module header): a server of
    /// two slots of [`CTX`] positions answers `one`'s prompts together, bit
    /// for bit as the one-slot server answered them.
    fn two_slots_decode_as_one_slot_runs(
        dir: &Path,
        file: &Path,
        one: &Answers,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        const NAME: &str = "two_slots_decode_as_one_slot_runs";
        let total = (2 * CTX).to_string();
        // One more sequence's stores, by the plan's own layout.
        let split = Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let store_bytes = PlanInputs::describe(&split)?
            .kv
            .bytes(0..LAYERS as usize, CTX as u64);
        // The one-slot server's budget and the room the second slot's stores
        // take: the experts are planned in the same room, so on the same split.
        let (mut s, d) = spawn_budgeted(
            dir,
            "serve-two",
            file,
            &["--ctx", &total, "--parallel", "2", "--port", "0"],
            gate_budget() + store_bytes,
        )?;
        let err_log = d.join("server.err");
        let addr = match s.address(&err_log, LISTEN_POLLS, Duration::from_secs(1)) {
            Ok(addr) => addr,
            Err(e) => {
                println!("the two-slot server never listened: {e}");
                println!(
                    "server stderr:\n{}",
                    std::fs::read_to_string(&err_log).unwrap_or_default()
                );
                check(ok, NAME, false);
                return Ok(());
            }
        };
        let url = |p: &str| format!("http://{addr}{p}");
        let records = server_log(&err_log, record::BLOOMERY_SERVE_MIMO2)?;
        let load = records.one(&record::LOAD_MIMO2)?;
        let listen = records.one(&record::LISTENING_MIMO2)?;
        let planned = Planned::of(&records.one(&record::PLAN)?)?;
        let stderr = std::fs::read_to_string(&err_log).unwrap_or_default();
        // `pass=one` as at `slots_total_is_planned`.
        let line = format!(
            "parallel rule=slots slots=2 slot_ctx={CTX} total={} from=flag pass=one",
            2 * CTX
        );
        let (resident, scratch) = (load.u64("resident_bytes")?, load.u64("scratch_bytes")?);
        // Outside the scratch, the second slot adds its stores and nothing
        // else; inside it, a load of two sequences adds a pass's buffers and
        // the card experts' second row (gpu-mimo2 `Body::load`).
        let outside = |resident: u64, scratch: u64| i128::from(resident) - i128::from(scratch);
        let past_stores = outside(resident, scratch)
            - outside(one.resident, one.scratch)
            - i128::from(store_bytes);
        let pass_bytes = i128::from(scratch) - i128::from(one.scratch);
        let same_plan = planned == one.planned;
        let loaded = same_plan
            && load.u64("slots")? == 2
            && load.u64("ctx")? == CTX as u64
            && listen.u64("slots")? == 2
            && listen.u64("ctx")? == CTX as u64
            && stderr.lines().any(|l| l == line)
            && past_stores == 0
            && pass_bytes > 0;
        println!(
            "two-slot server at {addr}; plan {planned:?} against the one-slot server's {:?}: \
             {same_plan}; load record {:?}; listening record {:?}; resident {resident} B \
             (scratch {scratch} B) against the one-slot server's {} B (scratch {} B): outside \
             the scratch, past one slot's stores {store_bytes} B, {past_stores} B (want 0); the \
             scratch's pass buffers {pass_bytes} B (want more than 0): {loaded}",
            one.planned,
            load.line(),
            listen.line(),
            one.resident,
            one.scratch
        );
        // Each round's two requests, and the one-slot answers each is held to.
        let by = [
            ("batch", &one.batch),
            ("prose", &one.prose),
            ("chat", &one.chat),
        ];
        let get = |n: &str| by.iter().find(|(m, _)| *m == n).map(|&(_, a)| a);
        let rounds = [("batch", "prose"), ("prose", "batch"), ("chat", "prose")];
        let mut same = true;
        for (i, (f, g)) in rounds.into_iter().enumerate() {
            let (Some(fa), Some(ga)) = (get(f), get(g)) else {
                return Err(format!("round {i}: no one-slot answer for {f} or {g}").into());
            };
            let ran = round(
                &addr,
                &d,
                (&format!("round{i}-{f}"), fa),
                (&format!("round{i}-{g}"), ga),
            )?;
            let equal = [(&ran[0], fa), (&ran[1], ga)]
                .iter()
                .all(|(r, a)| !r.tokens.is_empty() && r.tokens == a.tokens && r.stop == a.stop);
            let apart = matches!((ran[0].slot, ran[1].slot), (Some(a), Some(b)) if a != b);
            println!(
                "round {i}: {f} on slot {:?} (held {:.1} ms) and {g} on slot {:?} (held {:.1} ms) \
                 in flight together; ids equal the one-slot server's {equal}; on different slots \
                 {apart}",
                ran[0].slot, ran[0].held_ms, ran[1].slot, ran[1].held_ms
            );
            same &= equal && apart;
        }
        // The slots are resident: no state was parked to give the engine
        // over.
        let swaps = metric(&url, "swaps_total")?;
        println!("swaps_total {swaps:?}");
        check(ok, NAME, loaded && same && swaps.is_none_or(|v| v == 0.0));
        println!("two-slot server stopped: {}", s.stop()?);
        Ok(())
    }

    /// The server's cut as the session runs it: the prompt less its last id,
    /// then a step of the last, then steps at argmax up to `n` ids; the first
    /// id's logits row alongside.
    fn session_ids(
        m: &mut bloomery_gpu_mimo2::Mimo2Model,
        prompt: &[u32],
        n: usize,
    ) -> Result<(Vec<u32>, Vec<f32>), GateError> {
        let (last, head) = prompt.split_last().ok_or("a prompt of no id")?;
        m.reset()?;
        if !head.is_empty() {
            m.step(head)?;
        }
        let mut ids = vec![m.step(&[*last])?];
        let first_logits = m.logits()?;
        slots_gate::greedy(m, &mut ids, n.saturating_sub(1))?;
        Ok((ids, first_logits))
    }

    /// `load_group_follows_the_lever` (the module header): the one-slot
    /// server's `load` record, the lever unset, holds the row's default group,
    /// and a server of the same shape under `BLOOMERY_PREFILL_GROUP=1` holds
    /// 1, on the same plan and with fewer resident bytes (a group of one
    /// holds one unit of the prompt batch's buffers, the default's group
    /// more).
    fn load_group_follows_the_lever(
        dir: &Path,
        file: &Path,
        one: &Answers,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        const NAME: &str = "load_group_follows_the_lever";
        let ctx = CTX.to_string();
        let pins = Pins {
            budget: Some(gate_budget()),
            group: Some("1"),
            ..OFF
        };
        let (mut s, d) = spawn_levered(
            dir,
            "serve-group-one",
            file,
            &["--ctx", &ctx, "--parallel", "1", "--port", "0"],
            pins,
        )?;
        let err_log = d.join("server.err");
        if let Err(e) = s.address(&err_log, LISTEN_POLLS, Duration::from_secs(1)) {
            println!("the group-one server never listened: {e}");
            println!(
                "server stderr:\n{}",
                std::fs::read_to_string(&err_log).unwrap_or_default()
            );
            check(ok, NAME, false);
            return Ok(());
        }
        let records = server_log(&err_log, record::BLOOMERY_SERVE_MIMO2)?;
        let load = records.one(&record::LOAD_MIMO2)?;
        let planned = Planned::of(&records.one(&record::PLAN)?)?;
        let default = bloomery_levers::PREFILL_GROUP_DEFAULT;
        let (group, resident) = (load.u64("group")?, load.u64("resident_bytes")?);
        let fewer = i128::from(one.resident) - i128::from(resident);
        println!(
            "load groups: the default server's {}, the lever's default {default}; the server under \
             {}=1 {group}; resident {} B against {resident} B: {fewer} B fewer at group 1 (want \
             more than 0); plan {planned:?} against {:?}",
            one.group,
            bloomery_levers::PREFILL_GROUP,
            one.resident,
            one.planned
        );
        println!("server stopped: {}", s.stop()?);
        check(
            ok,
            NAME,
            one.group == default && group == 1 && fewer > 0 && planned == one.planned,
        );
        Ok(())
    }

    /// `server_ids_are_the_session_ids` and `ids_see_an_id_short` (the module
    /// header), the server already stopped.
    fn server_ids_are_the_session_ids(
        file: &Path,
        levers: &bloomery_levers::Levers,
        a: &Answers,
        man: &RefManifest,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        crate::gate_card::init()?;
        let (mut session, opened) = open(file, levers, CTX)?;
        let m = session.model_mut();
        // The session's gate plan splits the experts as the one-slot server's
        // plan record does: a split the ids are not held to is no comparison.
        let card: u64 = opened.n_l.iter().sum();
        let host = ROUTED_EXPERTS.checked_sub(card).ok_or_else(|| {
            format!("the session's plan puts {card} card experts, past the file's {ROUTED_EXPERTS}")
        })?;
        let ours = Planned {
            card,
            host,
            n_l: n_l_range(&opened.n_l),
        };
        let mut same = ours == a.planned;
        println!(
            "session plan {ours:?} against the server's {:?}: {same}",
            a.planned
        );
        let mut moved = 0usize;
        for (name, ans) in [("batch", &a.batch), ("prose", &a.prose), ("chat", &a.chat)] {
            let (ours, first) = session_ids(m, &ans.prompt, N)?;
            let equal = agree(&ans.tokens, &ans.stop, &ours);
            println!(
                "{name}: server ids {:?} (stop {}), session ids {ours:?}: {}",
                ans.tokens,
                ans.stop,
                if equal { "equal" } else { "DIFFERENT" }
            );
            same &= equal;
            if name == "batch" {
                let ik = ik_last(man, N_VOCAB)?;
                let (top_ok, _) = last_argmax("batch first id", (&first, &ik), f64::INFINITY);
                same &= top_ok;
            }
            // The same session fed one id short: its ids must move for at
            // least one prompt, or the equality above cannot see a seat that
            // feeds one id short.
            if ans.prompt.len() > 1 {
                let (short, _) = session_ids(m, &ans.prompt[..ans.prompt.len() - 1], N)?;
                let differs = !agree(&ans.tokens, &ans.stop, &short);
                println!(
                    "{name}: the session fed one id short answers {short:?}: {}",
                    if differs { "moves the ids" } else { "same ids" }
                );
                moved += usize::from(differs);
            }
        }
        check(ok, "server_ids_are_the_session_ids", same);
        check(ok, "ids_see_an_id_short", moved > 0);
        Ok(())
    }

    /// The slot dropped and not saved (`POST /slots/0?action=erase`): the
    /// prompt cache keeps what it held.
    fn erase(url: &dyn Fn(&str) -> String) -> Result<(), GateError> {
        let (st, body) = curl(&url("/slots/0?action=erase"), Some(&json!({})), false)?;
        json_of("/slots/0?action=erase", st, &body)?;
        Ok(())
    }

    /// The shared-prefix clause ([`prefix_restore`]) on a server of its own
    /// (`--ctx` [`CTX`], `--parallel 1`, `--cache-ram` [`PREFIX_RAM_MIB`]).
    /// C's `cache_n` is [`PREFIX_SHARED`]: a window ring keeps that prefix,
    /// and the check name says "checkpoint" because the helper owns it.
    ///
    /// FAIL-first: a snapshot that copies no rows, or a resume that writes
    /// none back, moves C's ids (`prefix_c_ids_are_a_fresh_runs`); a seat
    /// that still refuses the snapshot keeps `cache_n` 0
    /// (`prefix_c_restores_the_nearest_checkpoint_from_the_host_cache`).
    fn prefix_cache(dir: &Path, file: &Path, ok: &mut bool) -> Result<(), GateError> {
        let ctx = CTX.to_string();
        // Every slot action needs a save directory, `erase` too (llama-server's order); the clause
        // saves nothing there, and the server's own log directory is made before it starts.
        let save = dir.join("prefix");
        let (mut s, d) = spawn_budgeted(
            dir,
            "prefix",
            file,
            &[
                "--ctx",
                &ctx,
                "--parallel",
                "1",
                "--port",
                "0",
                "--cache-ram",
                PREFIX_RAM_MIB,
                "--slot-save-path",
                utf8(&save)?,
            ],
            gate_budget(),
        )?;
        let err_log = d.join("server.err");
        let addr = match s.address(&err_log, LISTEN_POLLS, Duration::from_secs(1)) {
            Ok(addr) => addr,
            Err(e) => {
                println!("the prefix server never listened: {e}");
                println!(
                    "server stderr:\n{}",
                    std::fs::read_to_string(&err_log).unwrap_or_default()
                );
                check(
                    ok,
                    "prefix_c_restores_the_nearest_checkpoint_from_the_host_cache",
                    false,
                );
                check(ok, "prefix_c_ids_are_a_fresh_runs", false);
                return Ok(());
            }
        };
        let url = |p: &str| format!("http://{addr}{p}");
        let t = PrefixRestore {
            a: PREFIX_A,
            b: PREFIX_B,
            tail: PREFIX_TAIL,
            a_len: PREFIX_LEN,
            shared: PREFIX_SHARED,
            kept: PREFIX_SHARED as u64,
            n: N,
        };
        let restored = prefix_restore(&url, &d, ok, &t, &|| erase(&url));
        println!("prefix server stopped: {}", s.stop()?);
        restored
    }

    /// The clauses after the refusals, in this process: the server's, the
    /// two-slot server's against them, then the session's on a census of one.
    fn after_refusals(
        dir: &Path,
        file: &Path,
        levers: &bloomery_levers::Levers,
        cards: usize,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let man = RefManifest::open(&data_dir().join(BATCH), &IK)?;
        let (_, toks, _) = man.step()?;
        let batch = toks.to_vec();
        let answers = served(dir, file, batch, ok)?;
        if let Some(one) = &answers {
            two_slots_decode_as_one_slot_runs(dir, file, one, ok)?;
            load_group_follows_the_lever(dir, file, one, ok)?;
        }
        match answers {
            Some(answers) if cards == 1 => {
                server_ids_are_the_session_ids(file, levers, &answers, &man, ok)?;
            }
            Some(_) => println!(
                "deferred(one card) server_ids_are_the_session_ids, ids_see_an_id_short: {cards} \
                 cards in view; the gate plan runs on one (shared/gate_card.rs)"
            ),
            None => {}
        }
        prefix_cache(dir, file, ok)?;
        Ok(())
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[CARD_BUDGET])?;
        let a = parse_args()?;
        std::fs::create_dir_all(&a.dir)?;
        let file = ref_model_path()?;
        println!("== {}", file.display());
        let cards = visible_cards()?;
        let mut ok = true;

        if a.after_refusals {
            println!(
                "plan_only_names_its_default, host_room_halves_the_default, \
                 residency_default_is_the_machine, refusals_before_any_load, \
                 slots_total_is_planned, default_load_listens: run by the parent of this child"
            );
            after_refusals(&a.dir, &file, &levers, cards, &mut ok)?;
        } else {
            let base = plan_only_names_its_default(&a.dir, &file, cards, &mut ok)?;
            host_room_halves_the_default(&a.dir, &file, (&levers, cards), base.as_ref(), &mut ok)?;
            residency_default_is_the_machine(&a.dir, &file, &mut ok)?;
            refusals_before_any_load(&a.dir, &file, cards, &mut ok)?;
            slots_total_is_planned(&a.dir, &file, &mut ok)?;
            check(
                &mut ok,
                "default_load_listens",
                default_load_listens(&a.dir, &file)?,
            );
            if cards >= 2 {
                child_clauses(&a.dir, &mut ok)?;
            } else {
                after_refusals(&a.dir, &file, &levers, cards, &mut ok)?;
            }
        }
        if ok {
            println!("PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
