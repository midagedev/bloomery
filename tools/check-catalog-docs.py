#!/usr/bin/env python3
"""The docs an agent or a user starts a model from, held to the code they repeat (Mac, text only, no build).

    tools/check-catalog-docs.py check [--root DIR]
    tools/check-catalog-docs.py --self-test

A start by name (`bloomery-serve --model NAME`) has one owner of "which file is this model": the catalog
(crates/hf/src/catalog.rs, committed as tools/catalog.tsv, read by tools/bloomery/catalog.py). The README's Use
table and docs/models.md used to be two more copies of it, and docs/agents.md repeats the server's exit table and
event names. This check holds each copy to its owner, so an edited quant or a renumbered code is red here instead of
wrong in a page.

  README Use table   each row's `bloomery-serve --hf <repo>:<quant>` is a verified row of the model its seat word names
                     (the model whose alias is the seat word). A seat word with no catalog model is OUTSIDE_SEATS.
  docs/models.md     each row of the Models table whose model, lower-cased with its spaces made `-`, is a catalog name
                     recommends a file of that model: the first link `[`VARIANT`](https://huggingface.co/OWNER/REPO)`
                     of its Recommended cell is a row's variant and repo, or, with no link, the cell holds a row's
                     variant. A model row with no catalog name is in OUTSIDE_MODELS, and a catalog model with no row
                     is in UNDOCUMENTED, each with its reason. The README's row of a seat is the file its model's
                     docs row recommends.
  docs/agents.md     the exit table (status, name, meaning, next) equals the codes of crates/serve/src/exit.rs; every
                     event name and the schema of crates/serve/src/event.rs, every stop cause and every loading
                     phase (crates/serve/src/listener.rs) is named; the three commands are there.
  README.md, llms.txt  both point at docs/agents.md; llms.txt names `--recommend --json`.

Exit: 0 every copy equals its owner; 1 a copy differs (every problem is printed first, one a line, with the file and
line); 65 a file this tool cannot read or parse, by name.
"""
import importlib.util
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_by_path(name, *rel):
    """A module of this tree, loaded by path (tools/ has no packages), leaving no bytecode behind."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *rel))
    mod = importlib.util.module_from_spec(spec)
    keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = keep
    return mod


catalog = load_by_path("bloomery_catalog", "tools", "bloomery", "catalog.py")
# The README's Use table has one reader, the tool that checks its rows against the hub.
hf_arch = load_by_path("hf_arch_check", "tools", "hf-arch-check.py")

README = "README.md"
MODELS_MD = "docs/models.md"
AGENTS_MD = "docs/agents.md"
LLMS_TXT = "llms.txt"
EXIT_RS = "crates/serve/src/exit.rs"
EVENT_RS = "crates/serve/src/event.rs"
LISTENER_RS = "crates/serve/src/listener.rs"

# Seat words of the README's Use table that name no catalog model, and why.
OUTSIDE_SEATS = {
    "decide": "decision models are not in the catalog yet (the first version holds chat models)",
}
# Rows of docs/models.md that name no catalog model, and why.
OUTSIDE_MODELS = {
    "ornith-1.5-35b-a3b": "a qwen3 seat file with no catalog row yet",
    "clef-flash": "a decision model: the catalog holds chat models",
    "lev": "a decision model: the catalog holds chat models",
    "raon-1-mini-preview": "the raon1 seat has no catalog row yet",
    "deepseek-v2-lite-chat": "the first model, still gated, with no catalog row",
}
# Catalog models with no row in docs/models.md, and why.
UNDOCUMENTED = {
    "qwen3.8-27b": "docs/models.md has no row for the dense Qwen3.8-27B yet",
}

HF_LINK = re.compile(r"\[`([^`]+)`\]\(https://huggingface\.co/([^)\s]+)\)")
HF_COMMAND = re.compile(r"`bloomery-serve --hf (\S+)`")


class Invalid(Exception):
    """A file this tool cannot read or parse: the message names it."""


def read_text(root, rel):
    try:
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise Invalid(f"{rel}: {e.strerror}")


def cells_of(line):
    body = line.strip()
    body = body[1:] if body.startswith("|") else body
    body = body[:-1] if body.endswith("|") else body
    return [c.strip() for c in re.split(r"(?<!\\)\|", body)]


def table_under(text, heading, rel):
    """(header cells, [(line number, cells)]) of the first table under the line `heading`; an absent heading or table
    is an Invalid."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.rstrip() == heading), None)
    if start is None:
        raise Invalid(f"{rel}: no `{heading}` heading")
    table = []
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            break
        if lines[i].lstrip().startswith("|"):
            table.append((i + 1, lines[i]))
        elif table:
            break
    if len(table) < 3 or not all(re.fullmatch(r":?-+:?", c) for c in cells_of(table[1][1])):
        raise Invalid(f"{rel}: the table under `{heading}` is not a header, a separator and rows")
    return cells_of(table[0][1]), [(no, cells_of(line)) for no, line in table[2:]]


def model_key(cell):
    """The catalog name a model cell of docs/models.md spells: lower case, spaces made `-`."""
    return cell.strip().lower().replace(" ", "-")


def recommended(model_rows, cell):
    """The rows of one model that a Recommended cell names, or the reason it names none. The first link of
    `[`VARIANT`](https://huggingface.co/OWNER/REPO)` is one row's variant and repo; a cell with no link holds the
    variants of the rows it names (case-insensitive, as a file name spells them)."""
    link = HF_LINK.search(cell)
    if link:
        hit = [r for r in model_rows if r.variant == link.group(1) and r.repo == link.group(2)]
        return hit, None if hit else (
            f"its first link names {link.group(2)} `{link.group(1)}`, no row of the catalog "
            f"({', '.join(f'{r.repo} `{r.variant}`' for r in model_rows)})")
    hit = [r for r in model_rows if r.variant.lower() in cell.lower()]
    return hit, None if hit else (
        f"it names none of the catalog's variants ({', '.join(f'`{r.variant}`' for r in model_rows)})")


def models_problems(docs_text, readme_text, rows):
    """The problems of docs/models.md's Models table and of the README's Use table against the catalog `rows`, and the
    rows each docs entry recommends, by model name."""
    out, docs_rows, by_name = [], {}, catalog.models(rows)
    _, table = table_under(docs_text, "## Models", MODELS_MD)
    seen = set()
    for no, cells in table:
        if len(cells) < 3:
            out.append(f"{MODELS_MD}:{no}: a Models row of {len(cells)} cells; want the model, its recommended file and how it runs")
            continue
        key = model_key(cells[0])
        seen.add(key)
        if key not in by_name:
            if key not in OUTSIDE_MODELS:
                out.append(f"{MODELS_MD}:{no}: `{cells[0]}` is no catalog model ({', '.join(by_name)}); add it to the catalog, "
                           "or to OUTSIDE_MODELS in tools/check-catalog-docs.py with its reason")
            continue
        hit, why = recommended(by_name[key], cells[1])
        if why:
            out.append(f"{MODELS_MD}:{no}: the file {cells[0]} recommends is not the catalog's: {why}")
        docs_rows[key] = hit
    for key in OUTSIDE_MODELS:
        if key in by_name:
            out.append(f"{MODELS_MD}: {key} is in the catalog now: take it out of OUTSIDE_MODELS in tools/check-catalog-docs.py")
        elif key not in seen:
            out.append(f"{MODELS_MD}: {key} is in OUTSIDE_MODELS and has no row; take it out of tools/check-catalog-docs.py")
    for key in by_name:
        if key not in seen and key not in UNDOCUMENTED:
            out.append(f"{MODELS_MD}: the catalog model {key} has no row; add one, or name it in UNDOCUMENTED in "
                       "tools/check-catalog-docs.py with the reason")
        if key in seen and key in UNDOCUMENTED:
            out.append(f"{MODELS_MD}: {key} has a row now: take it out of UNDOCUMENTED in tools/check-catalog-docs.py")
    try:
        use = hf_arch.use_rows(readme_text)
    except hf_arch.Refuse as e:
        raise Invalid(str(e))
    for seat, repo, quant, no in use:
        if seat in OUTSIDE_SEATS:
            if seat in {r.seat for r in rows}:
                out.append(f"{README}:{no}: the seat {seat} has catalog models now: take it out of OUTSIDE_SEATS in "
                           "tools/check-catalog-docs.py")
            continue
        try:
            model = catalog.default_model(rows, seat)
        except catalog.Invalid as e:
            out.append(f"{README}:{no}: the seat word {seat} names no catalog model ({e}); add it to the catalog, or "
                       "to OUTSIDE_SEATS in tools/check-catalog-docs.py with its reason")
            continue
        hit = [r for r in by_name[model] if r.repo == repo and r.quant == quant]
        verified = [r for r in hit if r.verified]
        spelled = f"{repo}:{quant}"
        if not hit:
            out.append(f"{README}:{no}: `bloomery-serve --hf {spelled}` is no row of {model}; its rows: "
                       f"{', '.join(r.hf for r in by_name[model])}")
        elif not verified:
            out.append(f"{README}:{no}: `bloomery-serve --hf {spelled}` is an unverified row of {model}: the Use table "
                       "names verified rows only")
        elif model in docs_rows and hit[0] not in docs_rows[model]:
            out.append(f"{README}:{no}: the Use table starts {model} from {spelled}, and {MODELS_MD} recommends "
                       f"{', '.join(r.hf for r in docs_rows[model]) or 'no catalog row'}")
    return out


def exit_codes(text):
    """[(status, name, hint, doc)] of the `codes!` table of crates/serve/src/exit.rs, in its order."""
    body = re.search(r"codes!\s*\{(.*?)\n\}", text, re.S)
    if not body:
        raise Invalid(f"{EXIT_RS}: no `codes! {{ ... }}` table")
    lit = r'"((?:[^"\\]|\\[\s\S])*)"'
    rows = re.findall(rf"\b[A-Z]\w*\s*=\s*(\d+),\s*{lit},\s*{lit},\s*{lit};", body.group(1))
    if not rows:
        raise Invalid(f"{EXIT_RS}: the `codes!` table holds no row this tool reads")
    return [(int(s), n, rust_str(h), rust_str(d)) for s, n, h, d in rows]


def rust_str(body):
    """The value of a Rust string literal's body as far as the table uses it: a backslash before a newline drops
    the newline and the next line's leading whitespace, and an escaped quote or backslash is the character."""
    return re.sub(r'\\(["\\])', r"\1", re.sub(r"\\\n\s*", "", body))


def exit_problems(agents_text, exit_text):
    out = []
    header, table = table_under(agents_text, "## Exit codes", AGENTS_MD)
    if header != ["Exit", "Code", "Meaning", "Next"]:
        out.append(f"{AGENTS_MD}: the exit table's header is {header}, not ['Exit', 'Code', 'Meaning', 'Next']")
        return out
    got = {}
    for no, cells in table:
        if len(cells) != 4 or not cells[0].isdigit():
            out.append(f"{AGENTS_MD}:{no}: an exit row is `| status | code | meaning | next |`, not {cells}")
            continue
        got[int(cells[0])] = (no, cells[1].strip("`"), cells[2], cells[3])
    codes = exit_codes(exit_text)
    for status, name, hint, doc in codes:
        want = (name, doc, hint or "-")
        if status not in got:
            out.append(f"{AGENTS_MD}: the exit table has no row for {status} `{name}` ({EXIT_RS})")
        elif got[status][1:] != want:
            no, *have = got[status]
            out.append(f"{AGENTS_MD}:{no}: the row of {status} reads {have}, {EXIT_RS} says {list(want)}")
    for status, (no, *_rest) in got.items():
        if status not in {c[0] for c in codes}:
            out.append(f"{AGENTS_MD}:{no}: the exit table lists {status}, which {EXIT_RS} does not")
    return out


def contract_problems(agents_text, event_text, listener_text):
    """Every event name, the schema, each stop cause and each loading phase the server prints is in docs/agents.md."""
    out = []
    schema = re.search(r'pub const SCHEMA: &str = "([^"]+)";', event_text)
    events = re.findall(r'put\("event",\s*json!\("(\w+)"\)\)', event_text)
    causes = re.findall(r'Cause::\w+ => "(\w+)"', event_text)
    phases = re.findall(r'Phase::\w+ => "(\w+)"', listener_text)
    if not schema or not events or not causes or not phases:
        raise Invalid(f"{EVENT_RS}, {LISTENER_RS}: no schema, event name, stop cause or phase this tool reads "
                      f"(schema {bool(schema)}, events {len(events)}, causes {len(causes)}, phases {len(phases)})")
    if schema.group(1) not in agents_text:
        out.append(f"{AGENTS_MD}: it does not name the event schema `{schema.group(1)}` ({EVENT_RS})")
    for e in events:
        if f'"event":"{e}"' not in agents_text:
            out.append(f'{AGENTS_MD}: it shows no `"event":"{e}"` line ({EVENT_RS})')
    for word, what, rs in [(c, "stop cause", EVENT_RS) for c in causes] + [(p, "phase", LISTENER_RS) for p in phases]:
        if f"`{word}`" not in agents_text:
            out.append(f"{AGENTS_MD}: it does not name the {what} `{word}` ({rs})")
    for needle in ("--recommend --json", "--dry-run --json", "--model NAME", '"status":"loading"'):
        if needle not in agents_text:
            out.append(f"{AGENTS_MD}: it does not show `{needle}`")
    return out


def pointer_problems(readme_text, llms_text):
    out = []
    if AGENTS_MD not in readme_text:
        out.append(f"{README}: it does not point at {AGENTS_MD}")
    if AGENTS_MD not in llms_text:
        out.append(f"{LLMS_TXT}: it does not point at {AGENTS_MD}")
    if "--recommend --json" not in llms_text:
        out.append(f"{LLMS_TXT}: it does not name `--recommend --json`")
    return out


def check(root):
    """Every problem of the docs under `root`, one string each."""
    try:
        rows = catalog.read_file(root)
    except catalog.Invalid as e:
        raise Invalid(str(e))
    text = {rel: read_text(root, rel) for rel in (README, MODELS_MD, AGENTS_MD, LLMS_TXT, EXIT_RS, EVENT_RS, LISTENER_RS)}
    return (models_problems(text[MODELS_MD], text[README], rows)
            + exit_problems(text[AGENTS_MD], text[EXIT_RS])
            + contract_problems(text[AGENTS_MD], text[EVENT_RS], text[LISTENER_RS])
            + pointer_problems(text[README], text[LLMS_TXT]))


def run_check(root):
    try:
        problems = check(root)
    except Invalid as e:
        print(f"check-catalog-docs: {e}", file=sys.stderr)
        return 65
    for p in problems:
        print(f"check-catalog-docs: {p}")
    if problems:
        print(f"check-catalog-docs: red, {len(problems)} problem(s)")
        return 1
    print("check-catalog-docs: README Use table, docs/models.md and docs/agents.md equal the catalog and the serve contract")
    return 0


FILES = (README, MODELS_MD, AGENTS_MD, LLMS_TXT, EXIT_RS, EVENT_RS, LISTENER_RS, catalog.CATALOG_TSV)


def self_test():
    ok = True

    def check_(name, cond, detail=""):
        nonlocal ok
        print(("ok " if cond else "FAIL ") + name + ("" if cond else f": {detail}"))
        ok = ok and bool(cond)

    paths = {"readme": README, "models": MODELS_MD, "agents": AGENTS_MD, "llms": LLMS_TXT, "exit": EXIT_RS,
             "event": EVENT_RS, "listener": LISTENER_RS, "catalog": catalog.CATALOG_TSV}
    with tempfile.TemporaryDirectory() as tmp:
        def tree(name, **edits):
            """A copy of this tree's checked files under `tmp/name`, each `file=(old, new[, n])` edit applied to every
            occurrence. An `old` that is not in the file exactly `n` times (default once) is the test's own fault, never
            a pass."""
            base = os.path.join(tmp, name)
            for rel in FILES:
                os.makedirs(os.path.dirname(os.path.join(base, rel)), exist_ok=True)
                shutil.copy(os.path.join(ROOT, rel), os.path.join(base, rel))
            for key, (old, new, *count) in edits.items():
                want = count[0] if count else 1
                path = os.path.join(base, paths[key])
                with open(path, encoding="utf-8") as f:
                    body = f.read()
                if body.count(old) != want:
                    raise AssertionError(f"the test's edit of {paths[key]}: {old!r} is in it {body.count(old)} times, not {want}")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(body.replace(old, new))
            return base

        def red(name, needle, **edits):
            try:
                problems = check(tree(name.replace(" ", "-")[:40], **edits))
            except (Invalid, AssertionError) as e:
                check_(name, False, f"{type(e).__name__}: {e}")
                return
            check_(name, any(needle in p for p in problems), "; ".join(problems) or "green")

        try:
            problems = check(ROOT)
        except Invalid as e:
            problems = [f"unreadable: {e}"]
        check_("the tree's own docs equal the catalog and the serve contract", not problems, "; ".join(problems))

        red("an edited quant in the README's Use table is red", "is no row of glm-5.3-flash",
            readme=("GLM-5.3-Flash-GGUF:UD-Q4_K_XL", "GLM-5.3-Flash-GGUF:UD-Q3_K_XL"))
        red("a README row of an unverified file is red", "an unverified row of qwen3-30b-a3b-instruct-2507",
            readme=("`bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M`",
                    "`bloomery-serve --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q8_0`"))
        red("a README row that docs/models.md does not recommend is red", "docs/models.md recommends",
            readme=("GLM-5.3-Flash-GGUF:UD-Q4_K_XL", "GLM-5.3-Flash-GGUF:Q8_0"),
            catalog=("340981966208\tfalse", "340981966208\ttrue"))
        red("a Use row of a seat word no model owns is red", "names no catalog model",
            readme=("| `glm` |", "| `flint` |"))
        red("an edited variant in docs/models.md is red", "recommends is not the catalog's",
            models=("[`UD-Q4_K_XL`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)",
                    "[`UD-Q3_K_XL`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)"))
        red("an edited repo in docs/models.md is red", "recommends is not the catalog's",
            models=("huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF", "huggingface.co/someone/DeepSeek-V4.1-Flash-GGUF"))
        red("an unlinked file that names no variant is red", "names none of the catalog's variants",
            models=("NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf", "NVIDIA-Nemotron-3.5-Bolt-74B-A9B.gguf"))
        red("a model row that is no catalog model is red", "is no catalog model", models=("| Sage-A1.5 |", "| Sage-A1.6 |"))
        red("a catalog model with no docs row is red", "catalog model solar-open2-250b has no row",
            models=("| Solar-Open2-250B |", "| Solar-Open2-250B-x |"))
        red("a renamed exit code in docs/agents.md is red", "the row of 83",
            agents=("| 83 | `port-in-use` |", "| 83 | `port-taken` |"))
        red("a renamed exit code in exit.rs is red", "the row of 78",
            exit=('Lever = 78, "lever"', 'Lever = 78, "levers"'))
        red("a code exit.rs gained is red", "has no row for 79",
            exit=("    Lever = 78,", '    Spare = 79, "spare", "none", "x";\n    Lever = 78,'))
        red("an event the docs do not show is red", '`"event":"stopped"`',
            agents=('{"event":"stopped"', '{"evt":"stopped"'))
        red("a stop cause the docs do not name is red", "stop cause `listener`", agents=("`listener`", "`socket`"))
        red("a loading phase the docs do not name is red", "phase `plan`", agents=("`plan`", "`planning`"))
        red("a README with no pointer is red", "README.md: it does not point at docs/agents.md",
            readme=("docs/agents.md", "docs/agent.md", 2))
        red("an llms.txt with no pointer is red", "llms.txt: it does not point at docs/agents.md",
            llms=("docs/agents.md", "docs/agent.md", 2))
        base = tree("missing-owner")
        os.remove(os.path.join(base, EXIT_RS))
        try:
            check(base)
            check_("an unreadable owner is a named refusal", False, "no refusal")
        except Invalid as e:
            check_("an unreadable owner is a named refusal", EXIT_RS in str(e), str(e))
    print("check-catalog-docs self-test: " + ("PASS" if ok else "FAIL"))
    return ok


def main(argv):
    if argv == ["--self-test"]:
        return 0 if self_test() else 1
    if argv[:1] == ["check"]:
        rest = argv[1:]
        if rest == []:
            return run_check(ROOT)
        if len(rest) == 2 and rest[0] == "--root":
            return run_check(rest[1])
    print("usage: check-catalog-docs.py check [--root DIR] | --self-test", file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
