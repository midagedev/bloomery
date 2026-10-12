//! `gate_mimo2_serve` — the MiMo seat of `bloomery-serve` against the session
//! it serves, on the model file of the mimo2 profile (`BLOOMERY_REF_MODEL`).
//!
//!     gate_mimo2_serve --dir <dir> [--after-refusals]
//!
//! Two clauses need different censuses, and every clause runs on the census
//! the recipe gives (both cards in view, `BLOOMERY_CARD=both`):
//!
//! - `refusals_before_any_load`'s `--place bp` half needs two visible cards.
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
//! - `host_room_vetoes_the_default`: the default context is the card's and
//!   the host room only vetoes it. Every `--plan` child whose output a clause
//!   compares with another child's runs under `BLOOMERY_HOST_ROOM` =
//!   `PLAN_ROOM`, this workstation's host room with nothing loaded, so no
//!   reading of the box's available bytes enters a comparison. At that room
//!   the default's `ctx rule=` line and the plan's `ctx_max` equal
//!   `plan_only_names_its_default`'s, and the rule is `card` or `trained`.
//!   Under a room halfway between the host need of the floor's plan and of
//!   the default's (two plans made here at `PLAN_ROOM`, `HostNeed::of`) the
//!   line is `ctx rule=host` at the floor, a slot's share of [`CTX_FLOOR`],
//!   and the plan's `ctx_max` is that share. Under a room a GiB under the
//!   floor plan's need the process exits non-zero within [`REFUSE_WITHIN`],
//!   its stderr holds the words of `PlaceError::NoNvmeTier` and the room, and
//!   no `load` record is printed. FAIL-first, by reading: on a seat whose
//!   plan reads no room `BLOOMERY_HOST_ROOM` moves nothing, so all three
//!   runs print the box's own default; the `rule=host` half and the refusal
//!   are red.
//! - `residency_default_is_the_machine`: `--plan` at [`CTX`] with no
//!   `BLOOMERY_RESIDENCY` prints the `residency lever` record at `mid-p0-s1`
//!   with why `place` ahead of the plan record, and the `residency host`
//!   record of the churn pool after it; at the seat's default context (the
//!   KV planes take the card, the plan holds no expert to spare) the common
//!   rule prints `off` with why `no_room` and no pool; under
//!   `BLOOMERY_RESIDENCY=off` it prints `off` with why `set` and no pool
//!   record.
//! - `plan_only_names_its_default`: `--plan` with no `--place`, `--ctx` or
//!   `--parallel` exits 0 and prints `place unset` (the word `a`), the plan
//!   record and the `parallel` line, in that order, and no `load` or
//!   `listening` record. The `parallel` line is the default's two slots, each
//!   round one pass of the busy rows: `slots=2 slot_ctx=<ctx_max>
//!   total=2·ctx_max from=default pass=one`. The plan's `ctx_max` is a slot's
//!   share of the seat's default context, the total: at least 4096, at most
//!   the file's trained context, a multiple of 1024 or the trained context
//!   itself; the plan at that total stands (`--parallel 2 --ctx <total>
//!   --plan` exits 0 with the same `ctx_max`); and, unless it is the trained
//!   context, the next step of 1024 does not (`--parallel 2 --ctx <total +
//!   1024> --plan` is refused). FAIL-first: a seat that defaults to 4096
//!   regardless plans a total whose next step stands, and the largest-plan
//!   hold is red; a seat left on turns (`Pass::of(slots, false)`) prints
//!   `pass=turns`, and one on one slot prints `slots=1`, and the `parallel`
//!   line hold is red.
//! - `refusals_before_any_load`: `--ctx 1 --parallel 2` — a split that
//!   leaves a slot no position — exits non-zero within [`REFUSE_WITHIN`],
//!   its stderr names the split and no `load` record is printed. On a
//!   census of two cards, `--place bp` exits non-zero the same way, its stderr
//!   holding the text of `PlaceError::Tier` for the tier card the placement
//!   names ([`tier_refusal`]); on one card the `bp` half is a
//!   `deferred(two cards)` line. FAIL-first: a seat that takes a slot of no
//!   position is still running at the bound and the clause is red; a tier
//!   refusal that names another card or tier is red.
//! - `slots_total_is_planned`: the plan counts every slot's positions. At
//!   the one-slot default context `c` (`--parallel 1 --plan`), `--parallel 2
//!   --ctx c --plan` exits 0 and its `parallel` line splits it (`slots=2
//!   slot_ctx=⌊c/2⌋ pass=one`); `--parallel 2 --ctx c + [`CTX_STEP`]`, a total the
//!   one-slot plan refuses, exits non-zero within [`REFUSE_WITHIN`], its
//!   stderr names the plan's refusal and no `load` record is printed (a file
//!   whose default is its trained context has no step past it and the second
//!   half is a named skip). FAIL-first: a seat whose plan counts one slot
//!   (`plan_with_slots` at one, over a slot's share) loads and is still
//!   running at the bound; the refusal half is red.
//!
//! - `default_second_slot_never_costs_a_load`: the default's second slot is
//!   dropped, never the load. Two slots over a total hold the stores of one
//!   slot over it (`KvLayout::layer_bytes` is linear in the positions) and
//!   the plan counts nothing else a slot (`SeqTerms::beside` is 0; the pass
//!   buffers sit in the plan's fixed `scratch_bytes` term), so no budget
//!   refuses the default's two slots where one slot over the same total
//!   stands: the fall to one slot (`parallel` line `from=fit`,
//!   `bind::default_slots_fit`, whose unit test holds the branch that falls)
//!   has no load to show on this model, and the wiring is pinned where it
//!   ends in a refusal. `--plan` under `BLOOMERY_CARD_BUDGET`
//!   [`NO_ROOM_BUDGET`], below the card's floor at any slot count: with no
//!   `--parallel` it exits non-zero with the budget's refusal
//!   (`PlacementError::CardBudgetFloor`) and one stderr line saying the
//!   default's second slot did not fit and one slot is served; `--parallel 2`
//!   at the same budget exits non-zero with the same refusal and no such line
//!   (a set count is never lowered). FAIL-first: a seat that plans the
//!   default at two slots with no fall prints no such line, and the first
//!   half is red.
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
//!   Two rounds, each of two requests in flight together at
//!   `cache_prompt: false` — the batch prompt with the prose prompt, then the
//!   prose prompt with the batch prompt started first — answer each prompt's
//!   ids bit for bit as the one-slot server answered them, the two requests
//!   of a round on different slots (`id_slot`), and the server's
//!   `swaps_total` is 0 or absent (the slots are resident, no state is
//!   parked). The one-slot server's ids are the session's
//!   (`server_ids_are_the_session_ids`), so this is the session's ids by
//!   transitivity, and a second server is the only added load. FAIL-first: a
//!   `swap_seq` that exchanges nothing leaves both slots on the live stores,
//!   the slots overwrite each other's rows and the ids move; a `seq_bytes`
//!   of 0 leaves the resident bytes short of the stores; a budget without
//!   the added stores plans another split and the plan line is red.
//!
//! Then one server more of the one-slot shape (`--parallel 1` at [`CTX`], the
//! one-slot server's budget), under `BLOOMERY_PREFILL_GROUP=1`:
//!
//! - `load_group_follows_the_lever`: the one-slot server's `load` record, run
//!   with the lever unset, holds `group` the row's default
//!   (`bloomery_levers::PREFILL_GROUP_DEFAULT`, 2); this server's holds 1, on
//!   the one-slot server's plan (the plan reserves none of the prompt batch's
//!   units) and with fewer resident bytes (a group of one holds one unit of
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

    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::record::{self, Log};
    use bloomery_gpu_gates::serve_client::{
        Answer, PrefixTurns, Served, agree, chat_is_those_ids, check, curl, edit_and_extension,
        greedy, json_of, metric, rendered, serve_cmd, server_log, tokenized,
    };
    use bloomery_gpu_gates::{GateError, RefManifest, checks_failed, data_dir, ref_model_path};
    use bloomery_levers::CARD_BUDGET;
    use gguf::Split;
    use model::arch::mimo2::place::{PlaceError, PlanInputs};
    use model::placement::PlanLevers;
    use model::placement::workstation::{HostNeed, HostRead, PLAN_ROOM, RTX_3090};
    use refset::arch::mimo2::{BATCH, IK};
    use serde_json::{Value, json};

    use crate::e2e::ik_last;
    use crate::mimo2_open::{N_VOCAB, last_argmax, open};

    const USAGE: &str = "usage: gate_mimo2_serve --dir <dir> [--after-refusals]";

    /// The tokens each request makes.
    const N: usize = 16;

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

    /// The seat's context floor and the step its default context searches in.
    const CTX_FLOOR: u64 = 4096;
    const CTX_STEP: u64 = 1024;

    /// The slots a seat with no `--parallel` and no `--ctx` serves.
    const DEFAULT_SLOTS: u64 = 2;

    /// The clauses the child of a census of two or more owes: each prints its
    /// own `check` line, and the parent's verdict needs every one.
    const CHILD_CLAUSES: [&str; 9] = [
        "load_and_listen",
        "completion_ids",
        "chat_is_those_ids",
        "edit_resend_keeps_the_row_where_it_diverges",
        "edit_resend_ids_are_a_fresh_runs",
        "extension_keeps_every_held_position",
        "two_slots_decode_as_one_slot_runs",
        "server_ids_are_the_session_ids",
        "ids_see_an_id_short",
    ];

    /// How long a refused server may take to exit: it refuses before any
    /// load.
    const REFUSE_WITHIN: Duration = Duration::from_secs(30);

    /// A `--plan` reads the file's header and exits; it opens no card.
    const PLAN_WITHIN: Duration = Duration::from_secs(120);

    /// The words the slot split's refusal names (`--ctx 1 --parallel 2`): the
    /// flags, and the positions a slot is left with.
    const SPLIT_REFUSAL: [&str; 2] = [
        "--parallel 2 splits it to 0 positions a slot",
        "a slot holds at least one position",
    ];

    /// A card budget under which no plan stands: the card's floor (the dense
    /// tensors, the stores, the context, the scratch and the margin) passes
    /// it, at one slot as at two.
    const NO_ROOM_BUDGET: u64 = 1 << 30;

    /// The words the refusal of a card budget below the card's floor names
    /// (`PlacementError::CardBudgetFloor`).
    const BUDGET_REFUSAL: [&str; 2] = ["is below its floor", "KV"];

    /// The line the seat prints when the default's second slot does not fit
    /// and it serves one slot.
    const SECOND_SLOT_LINE: &str = "the default's second slot does not fit on --place";

    /// The words the plan's refusal of a total the card cannot hold names:
    /// the dense trunk's need, which the KV term counts every slot's stores
    /// in.
    const PLAN_REFUSAL: [&str; 2] = ["the plan's dense trunk alone needs", "KV"];

    /// The words the refusal of a host room that cannot hold the host experts
    /// names (`PlaceError::NoNvmeTier`): the experts paged to the NVMe tier,
    /// and that the engine serves none. The paged bytes and the need move with
    /// the room, so the gate holds the fixed words and the room it gave.
    const NO_TIER_REFUSAL: [&str; 2] = [
        "routed experts to the NVMe expert tier",
        "the MiMo engine serves no NVMe expert tier",
    ];

    /// How far under the floor plan's host need the refused room sits.
    const SHORT_OF_NEED: u64 = 1 << 30;

    /// The card budget of the servers whose ids are compared: the gate
    /// card's usable bytes, the bytes `shared/gate_card.rs` plans the session
    /// on whichever card is in view, so a server on the A6000 plans the
    /// session's split.
    fn gate_budget() -> u64 {
        RTX_3090.usable_bytes()
    }

    /// The text the seat prints for a `--place bp` it refuses: the flag's
    /// prefix and `PlaceError::Tier`, the refusal of a machine with an expert
    /// tier card (the program hangs no expert tier), for the first tier card
    /// the placement names, tier 0.
    ///
    /// # Errors
    /// No card of the census for `bp`, or a placement with no tier card.
    fn tier_refusal() -> Result<String, GateError> {
        let place = Place::Bp.on_host()?;
        let card = place
            .tier_cards()
            .first()
            .map(|c| (*c).to_owned())
            .ok_or("the bp placement names no tier card")?;
        Ok(format!(
            "--place {}: {}",
            Place::Bp.name(),
            PlaceError::Tier { card, tier: 0 }
        ))
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

    fn utf8(p: &Path) -> Result<&str, GateError> {
        Ok(p.to_str()
            .ok_or_else(|| format!("{} is not UTF-8", p.display()))?)
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

    /// The first `ctx rule=` line of a seat's stderr.
    fn ctx_line(text: &str) -> Option<&str> {
        text.lines().find(|l| l.starts_with("ctx rule="))
    }

    /// `plan_only_names_its_default` (the module header): the default it
    /// printed, `None` when it printed none.
    fn plan_only_names_its_default(
        dir: &Path,
        file: &Path,
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
        let word_a = match &place {
            Some(p) => p.word("place")? == "a",
            None => false,
        };
        let no_load = log.first(&record::LOAD_MIMO2)?.is_none()
            && log.first(&record::LISTENING_MIMO2)?.is_none();
        let exits = status.is_some_and(|s| s.success());
        println!(
            "plan only: exit 0 {exits}; place unset, plan, parallel in order {in_order}; place word a \
             {word_a}; no load or listening record {no_load}"
        );
        let ctx_max = match &plan {
            Some(p) => Some(p.u64("ctx_max")?),
            None => None,
        };
        let mut default_ok = false;
        if let Some(c) = ctx_max {
            // `ctx_max` is a slot's share of the default total.
            let total = DEFAULT_SLOTS * c;
            // The default is two slots, each round one pass of the busy rows.
            let want = format!(
                "parallel rule=slots slots={DEFAULT_SLOTS} slot_ctx={c} total={total} \
                 from=default pass=one"
            );
            let line_ok = parallel_line == Some(want.as_str());
            println!("plan only: the parallel line {parallel_line:?} is {want:?} {line_ok}");
            let cap = trained.unwrap_or(CTX_FLOOR);
            let grid = total.is_multiple_of(CTX_STEP) || Some(total) == trained;
            let bounded = total >= CTX_FLOOR && total <= cap.max(CTX_FLOOR);
            println!(
                "plan only: total {total} (ctx_max {c} a slot) of trained {trained:?}; at least \
                 {CTX_FLOOR} and at most the trained context {bounded}; a multiple of {CTX_STEP} \
                 or the trained context {grid}"
            );
            // The plan at that total stands: asked for by flag it exits 0
            // and plans the same slot context.
            let slots_arg = DEFAULT_SLOTS.to_string();
            let total_arg = total.to_string();
            let (st, at_log, _) = plan_run(
                dir,
                "plan-at-default",
                file,
                &["--parallel", &slots_arg, "--ctx", &total_arg],
            )?;
            let stands = st.is_some_and(|s| s.success())
                && match at_log.first(&record::PLAN)? {
                    Some(p) => p.u64("ctx_max")? == c,
                    None => false,
                };
            // Past it, the next step does not: the default is the largest
            // total whose plan stands, unless the file's trained context
            // ends the search.
            let largest = if trained.is_some_and(|t| total + CTX_STEP > t) {
                println!(
                    "plan only: no step of {CTX_STEP} fits under the trained context past the \
                     default"
                );
                true
            } else {
                let next = (total + CTX_STEP).to_string();
                let (st, _, _) = plan_run(
                    dir,
                    "plan-past-default",
                    file,
                    &["--parallel", &slots_arg, "--ctx", &next],
                )?;
                let refused = st.is_some_and(|s| !s.success());
                println!(
                    "plan only: total {total} stands {stands}; --ctx {next} is refused {refused}"
                );
                refused
            };
            default_ok = line_ok && bounded && grid && stands && largest;
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

    /// The host bytes the seat's plan of `file` needs at `slot_ctx` positions
    /// a slot, [`DEFAULT_SLOTS`] slots, on the placement a flagless `--plan`
    /// takes (`a`) at [`PLAN_ROOM`]: the plan the server prints, made here.
    fn default_plan_need(
        file: &Path,
        levers: &bloomery_levers::Levers,
        slot_ctx: u64,
    ) -> Result<u64, GateError> {
        let split = Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let mut inputs = PlanInputs::describe(&split)?;
        inputs.room = (PLAN_ROOM, HostRead::Given);
        let machine = Place::A.on_host()?.machine(None, None)?(inputs.model.layers);
        let slots = NonZeroUsize::new(usize::try_from(DEFAULT_SLOTS)?).ok_or("no default slot")?;
        let plan =
            inputs.plan_with_slots(&machine, slot_ctx, &PlanLevers::from_levers(levers)?, slots)?;
        Ok(HostNeed::of(&plan, 0).bytes())
    }

    /// `host_room_vetoes_the_default` (the module header); `base` is the
    /// default `plan_only_names_its_default` printed.
    fn host_room_vetoes_the_default(
        dir: &Path,
        file: &Path,
        levers: &bloomery_levers::Levers,
        base: Option<&Defaulted>,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        let Some(base) = base else {
            println!("host_room_vetoes_the_default: no default to hold the rooms to");
            check(ok, "host_room_vetoes_the_default", false);
            return Ok(());
        };
        // (i) At the plan room the default is the card's context, the one
        // `plan_only_names_its_default` printed.
        let (status, log, text) = plan_run(dir, "plan-room-default", file, &[])?;
        let line = ctx_line(&text);
        let ctx_max = match log.first(&record::PLAN)? {
            Some(p) => Some(p.u64("ctx_max")?),
            None => None,
        };
        let card_rule = line
            .is_some_and(|l| l.starts_with("ctx rule=card ") || l.starts_with("ctx rule=trained "));
        let same = status.is_some_and(|s| s.success())
            && line == Some(base.ctx_line.as_str())
            && ctx_max == Some(base.ctx_max);
        println!(
            "host room, at {PLAN_ROOM} B: the default {line:?} at ctx_max {ctx_max:?} is \
             plan_only's {:?} at {} {same}; it is the card's context {card_rule}",
            base.ctx_line, base.ctx_max
        );
        let floor_slot = CTX_FLOOR / DEFAULT_SLOTS;
        if base.ctx_max <= floor_slot {
            println!(
                "skip host_room_vetoes_the_default (the veto and the refusal): the default {} a \
                 slot is not past the floor's {floor_slot}, no deeper context for a room to veto",
                base.ctx_max
            );
            check(ok, "host_room_vetoes_the_default", same && card_rule);
            return Ok(());
        }
        // (ii) A room between the floor plan's host need and the default
        // plan's: the card's context needs more than the room holds, the
        // floor's does not.
        let (need_floor, need_default) = (
            default_plan_need(file, levers, floor_slot)?,
            default_plan_need(file, levers, base.ctx_max)?,
        );
        if need_default <= need_floor {
            return Err(format!(
                "the default's plan needs {need_default} B of host, not more than the floor's \
                 {need_floor} B: the model's host need no longer grows with the context"
            )
            .into());
        }
        let between = need_floor + (need_default - need_floor) / 2;
        let pins = |room| Pins {
            room: Some(room),
            ..OFF
        };
        let (status, log, text) = plan_pinned(dir, "plan-room-between", file, &[], pins(between))?;
        let want =
            format!("ctx rule=host ctx={floor_slot} slots={DEFAULT_SLOTS} total={CTX_FLOOR} ");
        let vetoed = status.is_some_and(|s| s.success())
            && ctx_line(&text).is_some_and(|l| l.starts_with(&want))
            && match log.first(&record::PLAN)? {
                Some(p) => p.u64("ctx_max")? == floor_slot,
                None => false,
            };
        println!(
            "host room {between} B (the floor's need {need_floor} B, the default's {need_default} \
             B): the line {:?} starts {want:?} at ctx_max {floor_slot} {vetoed}",
            ctx_line(&text)
        );
        if !vetoed {
            println!("plan-room-between stderr:\n{text}");
        }
        // (iii) A room a GiB under the floor plan's need cannot hold even the
        // floor: refused by name before any load.
        let short = need_floor - SHORT_OF_NEED;
        let reading = format!("a reading of {short} B");
        let refused = refusal_under(
            dir,
            file,
            "refuse-room",
            &["--plan"],
            (
                &[NO_TIER_REFUSAL[0], NO_TIER_REFUSAL[1], reading.as_str()],
                pins(short),
            ),
        )?;
        check(
            ok,
            "host_room_vetoes_the_default",
            same && card_rule && vetoed && refused,
        );
        Ok(())
    }

    /// `residency_default_is_the_machine` (the module header): `--plan` at
    /// [`CTX`] with no residency word prints the `residency lever` record at
    /// the serving word and why `place`, ahead of the plan, and the `residency
    /// host` record of the churn pool after it; at the seat's default context,
    /// whose KV planes leave the plan no card expert to spare, the common
    /// rule turns it `off` (why `no_room`) with no pool; and under
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
        let word = match &lever {
            Some(l) => l.word("residency")? == "mid-p0-s1" && l.word("why")? == "place",
            None => false,
        };
        let in_order = matches!(
            (&lever, &plan, &host),
            (Some(l), Some(p), Some(h)) if l.at() < p.at() && p.at() < h.at()
        );
        let (_, wide, _) = plan_pinned(
            dir,
            "plan-residency-default-ctx",
            file,
            &[],
            Pins::default(),
        )?;
        let no_room = match wide.first(&record::RESIDENCY_LEVER)? {
            Some(l) => l.word("residency")? == "off" && l.word("why")? == "no_room",
            None => false,
        } && wide.first(&record::RESIDENCY_HOST)?.is_none();
        let (_, off_log, _) = plan_pinned(dir, "plan-residency-off", file, &["--ctx", &ctx], OFF)?;
        let off = match off_log.first(&record::RESIDENCY_LEVER)? {
            Some(l) => l.word("residency")? == "off" && l.word("why")? == "set",
            None => false,
        } && off_log.first(&record::RESIDENCY_HOST)?.is_none();
        let exits = status.is_some_and(|s| s.success());
        println!(
            "residency default: exit 0 {exits}; at ctx {ctx} the lever record is mid-p0-s1 (place), \
             then the plan, then the pool {word} {in_order}; at the default context it is off \
             (no_room) with no pool {no_room}; set off prints off (set) and no pool {off}"
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
            let want = tier_refusal()?;
            pass &= refusal(
                dir,
                file,
                "refuse-bp",
                &["--place", "bp", "--port", "0"],
                &[&want],
            )?;
            check(ok, "refusals_before_any_load", pass);
        } else {
            println!(
                "deferred(two cards) refusals_before_any_load (--place bp): {cards} card in view; \
                 the tier-card refusal needs a census of two"
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
        // One step past the default the one-slot plan refuses; two slots plan
        // the same total, so the refusal is the total's, not a slot's share.
        let refused = if trained.is_some_and(|t| c >= t) {
            println!(
                "skip slots_total_is_planned (refusal half): the one-slot default {c} is the \
                 trained context, no step of {CTX_STEP} past it"
            );
            true
        } else {
            let next = (c + CTX_STEP).to_string();
            refusal(
                dir,
                file,
                "refuse-total",
                &["--parallel", "2", "--ctx", &next, "--port", "0"],
                &PLAN_REFUSAL,
            )?
        };
        check(ok, "slots_total_is_planned", split && refused);
        Ok(())
    }

    /// `default_second_slot_never_costs_a_load` (the module header).
    fn default_second_slot_never_costs_a_load(
        dir: &Path,
        file: &Path,
        ok: &mut bool,
    ) -> Result<(), GateError> {
        // `--plan` under [`NO_ROOM_BUDGET`]: whether it is refused with the
        // budget's words, and whether it says the second slot did not fit.
        let plan_at_no_room = |name: &str, args: &[&str]| -> Result<(bool, bool), GateError> {
            let mut all = args.to_vec();
            all.push("--plan");
            let pins = Pins {
                budget: Some(NO_ROOM_BUDGET),
                ..OFF
            };
            let (mut s, d) = spawn_levered(dir, name, file, &all, pins)?;
            let status = wait_exit(&mut s, PLAN_WITHIN)?;
            let text = std::fs::read_to_string(d.join("server.err")).unwrap_or_default();
            let refused = status.is_some_and(|st| !st.success())
                && BUDGET_REFUSAL.iter().all(|w| text.contains(w));
            let says = text.contains(SECOND_SLOT_LINE) && text.contains("serving one slot");
            println!(
                "{name}: {} --plan under {NO_ROOM_BUDGET} B: refused with the budget's words \
                 {refused}; says the second slot did not fit {says}",
                args.join(" ")
            );
            if !refused {
                println!("{name} stderr:\n{text}");
            }
            Ok((refused, says))
        };
        // The default tries two slots, falls to one and the one-slot plan
        // refuses; a set `--parallel 2` is refused at two with no fall.
        let (default_refused, default_says) = plan_at_no_room("no-room-default", &[])?;
        let (flag_refused, flag_says) =
            plan_at_no_room("no-room-parallel-2", &["--parallel", "2"])?;
        check(
            ok,
            "default_second_slot_never_costs_a_load",
            default_refused && default_says && flag_refused && !flag_says,
        );
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

    /// What a request of a round answered, and the slot it ran on.
    struct Ran {
        tokens: Vec<u32>,
        stop: String,
        slot: Option<u64>,
    }

    /// How long a round's first request is in flight before its second is
    /// posted: the engine thread admits it to the other slot.
    const POST_GAP: Duration = Duration::from_millis(300);

    /// A round of two requests in flight together on `addr`: `first` posted,
    /// `gap` later `second`, each a greedy `/completion` of [`N`] ids at
    /// `cache_prompt: false`, its body kept in `d` under its name.
    fn together(
        addr: &str,
        d: &Path,
        gap: Duration,
        first: (&str, &Answer),
        second: (&str, &Answer),
    ) -> Result<[Ran; 2], GateError> {
        let post = |(name, a): (&str, &Answer)| {
            let (addr, d, name, prompt) = (
                addr.to_owned(),
                d.to_path_buf(),
                name.to_owned(),
                a.prompt.clone(),
            );
            move || -> Result<Ran, String> {
                let url = |p: &str| format!("http://{addr}{p}");
                let (a, v) =
                    greedy(&url, prompt, N, &d, &name, false).map_err(|e| e.to_string())?;
                Ok(Ran {
                    tokens: a.tokens,
                    stop: a.stop,
                    slot: v["id_slot"].as_u64(),
                })
            }
        };
        std::thread::scope(|sc| {
            let a = sc.spawn(post(first));
            std::thread::sleep(gap);
            let b = sc.spawn(post(second));
            let a = a.join().map_err(|_| "a round's first request panicked")??;
            let b = b
                .join()
                .map_err(|_| "a round's second request panicked")??;
            Ok([a, b])
        })
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
        let rounds = [
            ("batch", "prose", POST_GAP),
            ("prose", "batch", POST_GAP),
            ("chat", "prose", Duration::ZERO),
        ];
        let mut same = true;
        for (i, (f, g, gap)) in rounds.into_iter().enumerate() {
            let (Some(fa), Some(ga)) = (get(f), get(g)) else {
                return Err(format!("round {i}: no one-slot answer for {f} or {g}").into());
            };
            let ran = together(
                &addr,
                &d,
                gap,
                (&format!("round{i}-{f}"), fa),
                (&format!("round{i}-{g}"), ga),
            )?;
            let equal = [(&ran[0], fa), (&ran[1], ga)]
                .iter()
                .all(|(r, a)| !r.tokens.is_empty() && r.tokens == a.tokens && r.stop == a.stop);
            let apart = matches!((ran[0].slot, ran[1].slot), (Some(a), Some(b)) if a != b);
            println!(
                "round {i}: {f} on slot {:?} and {g} on slot {:?} in flight together; ids equal \
                 the one-slot server's {equal}; on different slots {apart}",
                ran[0].slot, ran[1].slot
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
        while ids.len() < n {
            let prev = *ids.last().ok_or("no id")?;
            ids.push(m.step(&[prev])?);
        }
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
                "plan_only_names_its_default, host_room_vetoes_the_default, \
                 residency_default_is_the_machine, refusals_before_any_load, \
                 slots_total_is_planned, default_second_slot_never_costs_a_load: run by the \
                 parent of this child"
            );
            after_refusals(&a.dir, &file, &levers, cards, &mut ok)?;
        } else {
            let base = plan_only_names_its_default(&a.dir, &file, &mut ok)?;
            host_room_vetoes_the_default(&a.dir, &file, &levers, base.as_ref(), &mut ok)?;
            residency_default_is_the_machine(&a.dir, &file, &mut ok)?;
            refusals_before_any_load(&a.dir, &file, cards, &mut ok)?;
            slots_total_is_planned(&a.dir, &file, &mut ok)?;
            default_second_slot_never_costs_a_load(&a.dir, &file, &mut ok)?;
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
