#!/usr/bin/env python3
"""The tape of a headline row: toktape's record of the row's own request on the server that answered it, and its link.

    tools/release/tape.py check [--toktape-bin PATH]
    tools/release/tape.py record --out DIR --tag TEXT --ids FILE --url URL [--n N] [--timeout S] [--note TEXT]
                                 [--work DIR] [--toktape-bin PATH]
    tools/release/tape.py publish VERDICT [--toktape-bin PATH] [--dry-run] [--no-text]
    tools/release/tape.py --self-test

check prints `toktape <version> at <path>`. A toktape that is missing, does not run, prints no version or one older than
MIN_VERSION is a refusal (exit 2) naming it: tools/release/sitting.sh calls this before its hold, so a sitting that records
tapes never starts without a toktape that can.

record is the command tools/ref/depth-serve.sh runs as DEPTH_SERVE_TAPE on each round-1 server, after that server's timed
request: one `toktape record` of the arm's prompt on the server that just answered it. --ids is the first D ids of the
profile's prose corpus, the very ids the timed request sent, one a line (depth-serve.sh writes the file and names it
DEPTH_SERVE_TAPE_IDS; sitting.sh spells the flags out of the hook's environment). The server detokenizes them (/detokenize)
and the text goes to /completion verbatim (toktape's `--endpoint completion`), greedy (`--temp 0`, `top_k` 1), --n tokens
with `ignore_eos`, `cache_prompt` off: the timed request's shape. /completion gives a text prompt the tokenizer's special
tokens, so the text is the one the server's own /tokenize turns into the arm's ids again (`prompt exact`: the whole prompt's,
or the prompt without its first id where the server puts that one in front of a text prompt itself). A server that makes
other ids of both still gets the whole prompt's text (`prompt text`), and prompt_n, the count it processed, stands in the line
beside D. The tape is written under --out; the prompt, toktape's output and its summary are kept in --work (else a directory
removed at exit). The caller's environment is not read.

record's last line is the product, once, recorded or not:
  tape D=<D> | file <id>.toktape | toktape <version> | prompt exact|text | prompt_n <n> | cache_n <n> | pp <x> tok/s |
       tg <y> tok/s | predicted_n <n> | caveats <codes, or ->
  tape D=<D> | failed: <what went wrong>
pp and tg are the server's timings as toktape read them (prompt_per_second, predicted_per_second): the figures of a second
request on a warm server, a witness beside the row, not the row. The caveats are toktape's, by code. sitting.sh keeps the
line in the round-1 block of the cell's run, where publish reads it back.

publish is the lead's own command, run where the tapes are. For a green verdict whose header says tapes=on it uploads the
tape of every row (a cell's depth) of the round-1 blocks (`toktape publish <tape> --yes --title ... --note ... -o json`;
--no-text leaves the prompt and the answer home; --dry-run prints what would go up and uploads nothing) and prints, a row, a
line for the release notes and the README:
  cell <I> (<profile> place=<p> cards=<c>) D=<D> | tape pp <x> tok/s, tg <y> tok/s (prompt_n <n>, <n> tokens, toktape <v>) |
       [recorded on tape](https://tape.midagedev.com/r/<id>)
A row whose tape the sitting failed to record, that the verdict has no line of, or whose tape file is gone is a NOT PUBLISHED
line naming it. The tapes are <verdict>.d/tapes/, so the command reads the same on the box and on a copy fetched from it.
<verdict>.d/tapes/published.tsv is the ledger: a tape already in it is not uploaded again and its link is reprinted.
Nothing here publishes by itself: the sitting records, this command uploads, and no gate calls it. toktape's receipt carries
an anonymous upload's delete token: toktape files it on the publishing machine (`toktape publish --delete <id>` takes the
run down); this tool reads the receipt for the link and prints and writes nothing else of it.

Exit status: check 0; record 0, 1 the recording failed, 2 a named refusal; publish 0 every row up, 1 a row not published, 2 no
verdict at that path or no toktape; 64 a usage error, a verdict that is none, a red one or one that recorded no tapes.
--self-test runs every case against a stub tokenizer server in this process and a stand-in toktape, then each mutant of the
code and requires its case to fail. No card, no network.
"""
import argparse
import http.client
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

MIN_VERSION = (0, 3, 0)
MIN_WHY = ("the first release whose record takes --prompts, --host-label and -o json, and whose publish takes "
           "--yes, --title and --note")
VERSION = re.compile(r"^toktape\s+v?(\d+)\.(\d+)\.(\d+)(?:[-+][0-9A-Za-z.-]*)?$")
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
BLOCK = re.compile(r"^== round ([0-9]+) cell ([0-9]+) rc=")
CELL = re.compile(r"^cell ([0-9]+): ([A-Za-z0-9][A-Za-z0-9._-]*)(?:=\S+)? place=(\S+) cards=(\S+) ctx=\S+ D=([0-9]+(?:,[0-9]+)*) ")
TAPE_LINE = re.compile(r"^tape D=([0-9]+) \| (.*)$")
FAILED = re.compile(r"^failed: (.*)$")


class Refuse(Exception):
    """A named refusal: exit 2."""


class Verdict(Exception):
    """A verdict that is no input publish takes: exit 64."""


class Fail(Exception):
    """A recording or an upload that failed: exit 1, the line naming it."""


def head(text, n=200):
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 3] + "..."


def find_toktape(name):
    """(path, version text) of the toktape binary `name` names, else PATH's; Refuse when there is none, when it does not
    run, when `version` prints nothing to read or a version under MIN_VERSION."""
    path = name or shutil.which("toktape")
    if not path:
        raise Refuse("toktape is not on PATH (--toktape-bin names it)")
    try:
        r = subprocess.run([path, "version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        raise Refuse(f"toktape at {path} does not run: {e}")
    said = " ".join(r.stdout.split())
    if r.returncode != 0:
        raise Refuse(f"toktape at {path}: `version` exited {r.returncode}: {head(r.stderr or r.stdout)}")
    m = VERSION.match(said)
    if not m:
        raise Refuse(f"toktape at {path} prints {said!r} for `version`: no version to read")
    got = tuple(int(x) for x in m.groups())
    if got < MIN_VERSION:
        raise Refuse(f"toktape at {path} is {said[len('toktape '):]}; "
                     f"{'.'.join(str(x) for x in MIN_VERSION)} or newer is needed ({MIN_WHY})")
    return path, said[len("toktape "):]


def post(url, method, path, body, timeout):
    """(status, text) of one request to the server at `url`; Fail when it does not answer."""
    u = urllib.parse.urlsplit(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    try:
        conn.request(method, path, body=None if body is None else json.dumps(body).encode(),
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        return r.status, r.read().decode("utf-8", "replace")
    except (OSError, http.client.HTTPException) as e:
        raise Fail(f"{method} {path}: the server did not answer: {e}")
    finally:
        conn.close()


def run_toktape(cmd, timeout):
    """(rc, stdout, stderr) of one toktape call; Fail when it does not end within `timeout`."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise Fail(f"toktape did not end within {timeout:g} s")
    except (OSError, subprocess.SubprocessError) as e:
        raise Fail(f"toktape does not run: {e}")
    return r.returncode, r.stdout, r.stderr


def toktape_failure(out, err):
    """What a toktape call that failed said: its error object's message and hint, else the head of what it printed."""
    try:
        error = json.loads(out).get("error")
    except (ValueError, AttributeError):
        error = None
    if isinstance(error, dict) and error.get("message"):
        return error["message"] + (f"; {error['hint']}" if error.get("hint") else "")
    return head(err or out)


def read_ids(path):
    """The ids of the arm's prompt, one a line; Refuse when the file is missing, unreadable or holds a line that is no id."""
    try:
        lines = Path(path).read_text().splitlines()
    except (OSError, UnicodeError) as e:
        raise Refuse(f"cannot read the ids file {path}: {e}")
    ids = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        if not re.fullmatch(r"[0-9]+", line.strip()):
            raise Refuse(f"line {n} of {path} ('{line.strip()[:40]}') is no id: the file is one id a line")
        ids.append(int(line))
    if not ids:
        raise Refuse(f"the ids file {path} holds no id")
    return ids


def detokenize(url, ids, timeout):
    """The server's text of `ids` (/detokenize)."""
    status, text = post(url, "POST", "/detokenize", {"tokens": ids}, timeout)
    if status != 200:
        raise Fail(f"/detokenize answered HTTP {status}: {head(text)}")
    try:
        content = json.loads(text)["content"]
    except (ValueError, KeyError, TypeError):
        content = None
    if not isinstance(content, str):
        raise Fail(f"/detokenize's reply has no content: {head(text)}")
    return content


def tokenizes_to(url, text, timeout):
    """The ids the server makes of a /completion prompt `text`: /tokenize with the special tokens /completion adds."""
    status, reply = post(url, "POST", "/tokenize", {"content": text, "add_special": True}, timeout)
    if status != 200:
        raise Fail(f"/tokenize answered HTTP {status}: {head(reply)}")
    try:
        return [int(t) for t in json.loads(reply)["tokens"]]
    except (ValueError, KeyError, TypeError):
        raise Fail(f"/tokenize's reply has no tokens list: {head(reply)}")


def prompt_for(url, ids, timeout):
    """(text, exact): the text of `ids` to send as a /completion prompt. exact when the server's /tokenize, the special
    tokens /completion adds included, gives `ids` back for it: the whole prompt's text, or the text of the prompt without
    its first id where the server puts that one in front of a text prompt itself. A server that makes other ids of both
    gets the whole prompt's text, and exact is False."""
    full = detokenize(url, ids, timeout)
    if tokenizes_to(url, full, timeout) == ids:
        return full, True
    if len(ids) > 1:
        rest = detokenize(url, ids[1:], timeout)
        if tokenizes_to(url, rest, timeout) == ids:
            return rest, True
    return full, False


def product_line(summary, d, run_id, version, exact):
    """The product line of a recorded tape; Fail when the tape holds no figures of the row's request."""
    t = summary.get("timings")
    t = t if isinstance(t, dict) else {}
    streams = summary.get("aggregate")
    if isinstance(streams, dict) and streams.get("streams_failed"):
        raise Fail(f"{streams['streams_failed']} of the tape's stream(s) did not answer")
    try:
        prompt_n, cache_n, predicted_n = int(t["prompt_n"]), int(t.get("cache_n") or 0), int(t["predicted_n"])
        pp, tg = float(t["prompt_per_second"]), float(t["predicted_per_second"])
    except (KeyError, TypeError, ValueError):
        raise Fail("the tape's summary holds no timings to read")
    if prompt_n <= 0 or predicted_n <= 0:
        raise Fail(f"the tape holds no figures of a request: prompt_n {prompt_n}, predicted_n {predicted_n}")
    if cache_n:
        raise Fail(f"cache_n {cache_n}: the server restored a prefix of the prompt, so the tape's prefill is not the row's")
    caveats = [str(c["code"]) for c in (summary.get("caveats") or []) if isinstance(c, dict) and c.get("code")]
    return (f"tape D={d} | file {run_id}.toktape | toktape {summary.get('toktape_version') or version} | "
            f"prompt {'exact' if exact else 'text'} | prompt_n {prompt_n} | cache_n {cache_n} | pp {pp:.2f} tok/s | "
            f"tg {tg:.2f} tok/s | predicted_n {predicted_n} | caveats {','.join(caveats) or '-'}")


def record(a, ids, work):
    """One `toktape record` of the arm's prompt: the product line, or Fail/Refuse naming what went wrong."""
    d = len(ids)
    path, version = find_toktape(a.toktape_bin)
    text, exact = prompt_for(a.url, ids, a.timeout)
    tag = f"{a.tag}-D{d}"
    prompts = work / "prompt.jsonl"
    prompts.write_text(json.dumps({"name": tag, "prompt": text}, ensure_ascii=False) + "\n", encoding="utf-8")
    cmd = [path, "record", "--url", a.url, "--endpoint", "completion", "--temp", "0", "--param", "top_k=1",
           "--param", "ignore_eos=true", "--param", "cache_prompt=false", "-n", str(a.n), "--for", "0", "--wait", "60s",
           "--host-label", "", "--prompts", str(prompts), "--out", a.out, "--tag", tag]
    if a.note:
        cmd += ["--note", a.note]
    cmd += ["-o", "json", "--quiet"]
    rc, out, err = run_toktape(cmd, a.timeout)
    (work / "toktape.out").write_text(out, encoding="utf-8")
    (work / "toktape.err").write_text(err, encoding="utf-8")
    if rc != 0:
        raise Fail(f"toktape exit {rc}: {toktape_failure(out, err)}")
    try:
        summary = json.loads(out)
    except ValueError:
        summary = None
    if not isinstance(summary, dict):
        raise Fail(f"toktape's summary is no JSON object: {head(out)}")
    if summary.get("error"):
        raise Fail(toktape_failure(out, err))
    run_id = summary.get("id")
    if not isinstance(run_id, str) or not RUN_ID.match(run_id):
        raise Fail(f"toktape's summary names no run id: {head(out)}")
    if not (Path(a.out) / f"{run_id}.toktape").is_file():
        raise Fail(f"toktape named the run {run_id} but wrote no {Path(a.out) / (run_id + '.toktape')}")
    return product_line(summary, d, run_id, version, exact)


def cmd_check(a):
    try:
        path, version = find_toktape(a.toktape_bin)
    except Refuse as e:
        refuse(str(e))
    print(f"toktape {version} at {path}", flush=True)
    return 0


def cmd_record(a):
    try:
        ids = read_ids(a.ids)
    except Refuse as e:
        refuse(str(e))
    own = None
    rc = 0
    try:
        if a.work:
            work = Path(a.work)
        else:
            own = tempfile.TemporaryDirectory(prefix="tape-record.")
            work = Path(own.name)
        work.mkdir(parents=True, exist_ok=True)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        line = record(a, ids, work)
    except Fail as e:
        rc, line = 1, f"tape D={len(ids)} | failed: {e}"
    except Refuse as e:
        rc, line = 2, f"tape D={len(ids)} | failed: {e}"
    except OSError as e:
        rc, line = 2, f"tape D={len(ids)} | failed: cannot write {e.filename or a.out}: {e.strerror}"
    finally:
        if own is not None:
            own.cleanup()
    print(line, flush=True)
    return rc


def fields(rest):
    """`a x | b y z | c` -> {a: x, b: y z, c: ''}: each ` | ` field by its first word."""
    out = {}
    for part in rest.split(" | "):
        key, _, value = part.partition(" ")
        out[key] = value
    return out

class Tape:
    """One tape line of a verdict's round-1 block: `why` is None for a recorded tape, else what the sitting said of it."""

    def __init__(self, cell, d, f, why=None):
        self.cell, self.d, self.fields, self.why = cell, d, f, why


def read_verdict(path):
    """(header, cells, tapes) of the verdict at `path`: cells {idx: profile, place, cards, depths}, tapes {(idx, D): Tape}
    from the `tape D=` lines of each round-1 block. Refuse when the verdict is not there, Verdict when it is no verdict of
    tools/release/sitting.sh, is red or recorded no tapes."""
    try:
        text = Path(path).read_text(errors="replace")
    except OSError as e:
        raise Refuse(f"cannot read the verdict {path}: {e.strerror}")
    paragraphs = [p for p in re.split(r"\n[ \t]*\n", text.strip("\n")) if p.strip()]
    if not paragraphs or paragraphs[0].split("\n")[0] != "sitting-verdict 1":
        raise Verdict(f"{path} is no verdict of tools/release/sitting.sh (its first line is not 'sitting-verdict 1')")
    header = {}
    for line in paragraphs[0].split("\n")[1:]:
        if "=" in line:
            k, v = line.split("=", 1)
            header.setdefault(k, v)
    cells, tapes = {}, {}
    for para in paragraphs[1:]:
        lines = para.split("\n")
        block = BLOCK.match(lines[0])
        if not block:
            for line in lines:
                m = CELL.match(line)
                if m:
                    cells[int(m.group(1))] = {"profile": m.group(2), "place": m.group(3), "cards": m.group(4),
                                              "depths": [int(x) for x in m.group(5).split(",")]}
            continue
        if block.group(1) != "1":
            continue
        for line in lines[1:]:
            m = TAPE_LINE.match(line)
            if not m:
                continue
            d, rest = int(m.group(1)), m.group(2)
            f = fields(rest)
            name = f.get("file", "")
            if rest.startswith("file ") and name.endswith(".toktape") and RUN_ID.match(name[: -len(".toktape")]):
                tapes[(int(block.group(2)), d)] = Tape(int(block.group(2)), d, f)
            else:
                failed = FAILED.match(rest)
                tapes[(int(block.group(2)), d)] = Tape(int(block.group(2)), d, {},
                                                        failed.group(1) if failed else f"its line is malformed: {rest}")
    if header.get("tapes") != "on":
        raise Verdict(f"{path} recorded no tapes (tapes={header.get('tapes', 'none')}): a sitting records them at the "
                      "portable stage unless it is run --no-tape, and at the tree stage with --tape")
    if header.get("result") != "green":
        raise Verdict(f"{path} is result={header.get('result', 'none')}: tapes are published from a green verdict; --resume "
                      "the sitting's red runs and publish the merged verdict")
    return header, cells, tapes


def read_ledger(path):
    """{tape file: (id, url)} of the publishes recorded at `path`; a ledger that is not there is empty."""
    out = {}
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            f = line.split("\t")
            if len(f) >= 3:
                out[f[0]] = (f[1], f[2])
    return out


def lab_note(header, cell, tape):
    """(title, body) of one tape's note on its page: what the tape is, the row it stands beside and what it is not."""
    name = (header.get("version") if header.get("version") not in (None, "-") else header.get("tree")) or "?"
    title = f"bloomery {name}: cell {tape.cell} {cell['profile']} place={cell['place']} cards={cell['cards']}, a {tape.d}-token prompt"
    body = (f"Recorded by bloomery's release sitting (tools/release/sitting.sh, stage {header.get('stage', '?')}): the request "
            f"of row cell {tape.cell} D={tape.d} replayed on the server that answered it. The row's prompt, {tape.d} ids of the "
            f"{cell['profile']} profile's prose corpus, sent to /completion as text, greedy, {tape.fields['predicted_n']} tokens, "
            "cache_prompt off; the figures are the server's own timings.\n\n"
            "A witness beside the row, not the row: a second request on a warm server. The row is the mean of the sitting's "
            "rounds, in the sitting's verdict.\n\n"
            f"bloomery-serve sha256 {header.get('binary_sha256', '?')}; toktape {tape.fields['toktape']}.")
    return (title if len(title) <= 120 else title[:119] + "…"), body


def rate(text):
    """`194.44 tok/s` -> `194.44`: the tape line's figure without its unit."""
    return text.split(" ")[0]


def link_line(who, tape, url):
    """The line for the release notes and the README: the row, the tape's own figures, its link."""
    f = tape.fields
    count = f"prompt_n {f['prompt_n']}" + ("" if f["prompt_n"] == str(tape.d) else f", not the row's D {tape.d}")
    said = f"{count}, {f['predicted_n']} tokens, toktape {f['toktape']}"
    if f.get("caveats", "-") != "-":
        said += f"; caveats {f['caveats']}"
    return f"{who} | tape pp {rate(f['pp'])} tok/s, tg {rate(f['tg'])} tok/s ({said}) | [recorded on tape]({url})"


def receipt_link(out):
    """(id, url) of a receipt toktape printed for one upload; Fail when it names no link. The rest of the receipt, an
    anonymous upload's delete token among it, stays here."""
    try:
        receipt = json.loads(out)
    except ValueError:
        receipt = None
    url = receipt.get("url") if isinstance(receipt, dict) else None
    run_id = receipt.get("id") if isinstance(receipt, dict) else None
    if not isinstance(url, str) or not re.match(r"^https?://\S+$", url):
        raise Fail("toktape's receipt names no link")
    return run_id if isinstance(run_id, str) else "", url


def cmd_publish(a):
    try:
        header, cells, tapes = read_verdict(a.verdict)
    except Refuse as e:
        refuse(str(e))
    except Verdict as e:
        usage(str(e))
    try:
        path, version = find_toktape(a.toktape_bin)
    except Refuse as e:
        refuse(str(e))
    tapes_dir = Path(str(a.verdict) + ".d") / "tapes"
    ledger = tapes_dir / "published.tsv"
    done = read_ledger(ledger)
    name = (header.get("version") if header.get("version") not in (None, "-") else header.get("tree")) or "?"
    print(f"tapes: bloomery {name}, stage={header.get('stage', '?')}, toktape {version}, verdict {a.verdict}"
          + (" (a dry run: nothing is uploaded)" if a.dry_run else ""), flush=True)
    rows = [(idx, d) for idx in sorted(cells) for d in cells[idx]["depths"]]
    bad = 0
    for idx, d in rows:
        cell = cells[idx]
        who = f"cell {idx} ({cell['profile']} place={cell['place']} cards={cell['cards']}) D={d}"
        tape = tapes.get((idx, d))
        if tape is None or tape.why is not None:
            why = "the verdict has no tape line for this row" if tape is None else tape.why
            print(f"{who}: NOT PUBLISHED: the sitting recorded no tape for this row: {why}", flush=True)
            bad += 1
            continue
        tape_path = tapes_dir / tape.fields["file"]
        if not tape_path.is_file():
            print(f"{who}: NOT PUBLISHED: no tape at {tape_path}", flush=True)
            bad += 1
            continue
        if tape.fields["file"] in done:
            print(link_line(who, tape, done[tape.fields["file"]][1]), flush=True)
            continue
        cmd = [path, "publish", str(tape_path)]
        try:
            if a.dry_run:
                rc, out, err = run_toktape(cmd + ["--dry-run"], 120)
                if rc != 0:
                    raise Fail(f"toktape exit {rc}: {toktape_failure(out, err)}")
                print(f"{who}: would publish {tape_path.name}", flush=True)
                for line in out.splitlines():
                    print("    " + line, flush=True)
                continue
            title, body = lab_note(header, cell, tape)
            rc, out, err = run_toktape(cmd + ["--yes", "--title", title, "--note", body] + a.no_text + ["-o", "json"], 300)
            if rc != 0:
                raise Fail(f"toktape exit {rc}: {toktape_failure(out, err)}")
            run_id, url = receipt_link(out)
            with ledger.open("a", encoding="utf-8") as f:
                f.write(f"{tape.fields['file']}\t{run_id}\t{url}\t{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        except Fail as e:
            print(f"{who}: NOT PUBLISHED: {e}", flush=True)
            bad += 1
            continue
        except OSError as e:
            print(f"{who}: NOT RECORDED: it is up as {url}, but the ledger {ledger} could not be written "
                  f"({e.strerror}): a second publish would upload it again", flush=True)
            bad += 1
            continue
        done[tape.fields["file"]] = (run_id, url)
        print(link_line(who, tape, url), flush=True)
    print(f"tapes: {len(rows) - bad} of {len(rows)} {'to publish' if a.dry_run else 'published'} (ledger {ledger})", flush=True)
    return 1 if bad else 0


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"tape.py: {message}", file=sys.stderr)
        sys.exit(64)


def refuse(msg):
    print(f"tape.py: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def usage(msg):
    print(f"tape.py: {msg}", file=sys.stderr, flush=True)
    sys.exit(64)


def main(argv):
    if argv[:1] == ["check"]:
        ap = Parser(prog="tape.py check")
        ap.add_argument("--toktape-bin")
        return cmd_check(ap.parse_args(argv[1:]))
    if argv[:1] == ["record"]:
        ap = Parser(prog="tape.py record")
        ap.add_argument("--out", required=True)
        ap.add_argument("--tag", required=True)
        ap.add_argument("--ids", required=True)
        ap.add_argument("--url", required=True)
        ap.add_argument("--n", type=int, default=128)
        ap.add_argument("--timeout", type=float, default=285.0)
        ap.add_argument("--note", default="")
        ap.add_argument("--work")
        ap.add_argument("--toktape-bin")
        a = ap.parse_args(argv[1:])
        if not RUN_ID.match(a.tag):
            ap.error("--tag is letters, digits, dot, underscore and dash, starting with a letter or a digit")
        if not re.match(r"^https?://[^/\s]+(:[0-9]+)?/?$", a.url):
            ap.error("--url is http://host[:port]")
        if a.n < 1 or a.timeout <= 0:
            ap.error("--n is at least 1 token and --timeout is seconds above 0")
        return cmd_record(a)
    if argv[:1] == ["publish"]:
        ap = Parser(prog="tape.py publish")
        ap.add_argument("verdict")
        ap.add_argument("--toktape-bin")
        ap.add_argument("--dry-run", action="store_true")
        ap.add_argument("--no-text", action="store_true")
        a = ap.parse_args(argv[1:])
        a.no_text = ["--no-text"] if a.no_text else []
        return cmd_publish(a)
    ap = Parser(prog="tape.py")
    ap.error("a command is required: check, record or publish" if not argv else f"unknown command '{argv[0]}'")
    return 64


# ---- self-test ----

SCRATCH = []


def scratch(prefix):
    """A temporary directory the self-test removes when it ends."""
    d = Path(tempfile.mkdtemp(prefix=prefix))
    SCRATCH.append(d)
    return d


class Sink:
    """A list as a stream: what a `main` call printed, a line an element."""

    def __init__(self, lines):
        self.lines, self.partial = lines, ""

    def write(self, text):
        self.partial += text
        while "\n" in self.partial:
            line, self.partial = self.partial.split("\n", 1)
            self.lines.append(line)
        return len(text)

    def flush(self):
        pass

    def close(self):
        if self.partial:
            self.lines.append(self.partial)
            self.partial = ""


def call(main, argv):
    """(rc, stdout lines, stderr) of one `main` call with its output captured."""
    out, err = [], []
    stdout, stderr = sys.stdout, sys.stderr
    sinks = sys.stdout, sys.stderr = Sink(out), Sink(err)
    try:
        try:
            rc = main(argv)
        except SystemExit as e:
            rc = e.code
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        for sink in sinks:
            sink.close()
    return rc, out, "\n".join(err)


CASES = {}


def case(fn):
    CASES[fn.__name__.replace("_", "-")] = fn
    return fn


ENV_KEYS = ("TT_FAKE_LOG", "TT_FAKE_MODE", "TT_FAKE_VERSION", "TT_FAKE_VERSION_RC")


def clear_env():
    """Nothing a case set in this process's environment is left for the next one."""
    for k in ENV_KEYS:
        os.environ.pop(k, None)


# A stand-in for toktape. `version` prints TT_FAKE_VERSION (TT_FAKE_VERSION_RC ends it with that status). `record` reads
# its --prompts line, writes <out>/<id>.toktape and prints the summary. `publish` prints a dry run's listing, or a receipt
# with a delete token that must never reach this tool's output. TT_FAKE_MODE bends it: rc3 (record ends 3), noid (a summary
# with no id), slash (an id with a path separator), notape (no file written), cache (a prompt the cache served part of),
# caveat, failed (a stream that did not answer), zero (no figures), pubrc5 (publish ends 5), nourl (a receipt with no
# link). Every call's argv is appended to TT_FAKE_LOG.
FAKE_TOKTAPE = r"""#!@PY@
import gzip, json, os, sys
argv = sys.argv[1:]
if os.environ.get("TT_FAKE_LOG"):
    with open(os.environ["TT_FAKE_LOG"], "a") as f:
        f.write(json.dumps(argv) + "\n")
def opt(name):
    return argv[argv.index(name) + 1]
mode = os.environ.get("TT_FAKE_MODE", "")
if argv[:1] == ["version"]:
    print("toktape " + os.environ.get("TT_FAKE_VERSION", "v0.8.0"))
    sys.exit(int(os.environ.get("TT_FAKE_VERSION_RC") or 0))
if argv[:1] == ["record"]:
    if mode == "rc3":
        print(json.dumps({"error": {"code": "streams", "exit": 3, "message": "toktape: every stream failed",
                                    "hint": "read the server's log"}}))
        sys.exit(3)
    prompt = json.loads(open(opt("--prompts")).readline())["prompt"]
    run_id = {"noid": "", "slash": "../" + opt("--tag")}.get(mode, "run-" + opt("--tag"))
    summary = {"id": run_id, "toktape_version": "0.8.0",
               "timings": {"prompt_n": 0 if mode == "zero" else len(prompt.split()) + 1, "cache_n": 64 if mode == "cache" else 0,
                           "prompt_per_second": 0.0 if mode == "zero" else 1000.5,
                           "predicted_n": 0 if mode == "zero" else 128, "predicted_per_second": 0.0 if mode == "zero" else 40.0},
               "aggregate": {"streams_failed": 1 if mode == "failed" else 0},
               "caveats": [{"code": "machine_contended", "severity": "run"}] if mode == "caveat" else []}
    if mode != "notape":
        os.makedirs(opt("--out"), exist_ok=True)
        with open(os.path.join(opt("--out"), run_id + ".toktape"), "wb") as f:
            f.write(gzip.compress(json.dumps({"schema": 1, "summary": summary}).encode()))
    print(json.dumps(summary))
    sys.exit(0)
if argv[:1] == ["publish"]:
    tape = argv[1]
    if "--dry-run" in argv:
        print("would upload " + os.path.basename(tape) + " with its text")
        sys.exit(0)
    if mode == "pubrc5":
        print(json.dumps({"error": {"code": "publish", "exit": 5, "message": "toktape: the service refused the upload",
                                    "hint": "toktape publish <tape> --dry-run prints what would be sent"}}))
        sys.exit(5)
    run_id = "fake" + os.path.basename(tape)[:-len(".toktape")]
    print(json.dumps({"id": run_id, "url": "" if mode == "nourl" else "https://tape.midagedev.com/r/" + run_id,
                      "delete_token": "SECRET-DELETE-TOKEN"}))
    sys.exit(0)
sys.exit(1)
"""


def fake_toktape(d, mode="", version="v0.8.0", version_rc=0):
    """(binary, argv log) of a stand-in toktape in `d`, bent by `mode` and `version`; its knobs are in this process's
    environment, which its children inherit, until clear_env."""
    fake, log = d / "toktape", d / "argv"
    fake.write_text(FAKE_TOKTAPE.replace("@PY@", sys.executable))
    fake.chmod(0o755)
    os.environ.update({"TT_FAKE_LOG": str(log), "TT_FAKE_MODE": mode, "TT_FAKE_VERSION": version,
                       "TT_FAKE_VERSION_RC": str(version_rc)})
    return str(fake), log


class Tok:
    """A stub tokenizer server in this process: id 1 is a BOS whose piece is its own spelling, `<s>`; id n is the word w<n>.
    /detokenize joins the pieces; /tokenize reads the words back, the BOS first when add_special is set, as /completion
    adds it to a text prompt. `modes`: no-detok answers /detokenize with 404; down answers nothing."""

    def __init__(self, modes=()):
        outer = self
        self.modes = set(modes)
        self.calls = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def reply(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.calls.append((self.path, req))
                if "down" in outer.modes:
                    return
                if self.path == "/detokenize" and "no-detok" not in outer.modes:
                    return self.reply(200, {"content": " ".join("<s>" if t == 1 else f"w{t}" for t in req["tokens"])})
                if self.path == "/tokenize":
                    ids = [1] if req.get("add_special") else []
                    ids += [1 if w == "<s>" else int(w[1:]) for w in req["content"].split()]
                    return self.reply(200, {"tokens": ids})
                self.reply(404, {"error": {"code": 404}})

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def ids_file(d, ids):
    f = d / "ids"
    f.write_text("".join(f"{i}\n" for i in ids))
    return str(f)


def record_run(main, tok, ids, mode="", extra=()):
    """One `record` of `ids` against the stub server `tok` through the stand-in: (rc, product lines, toktape's argv, the
    tapes written, stderr, the work directory)."""
    d = scratch("tape-rec.")
    binary, log = fake_toktape(d, mode)
    out, work = d / "tapes", d / "work"
    argv = ["record", "--out", str(out), "--tag", "rel-0.3.16-glm-cell1", "--ids", ids_file(d, ids), "--url", tok.url,
            "--work", str(work), "--toktape-bin", binary, *extra]
    rc, lines, err = call(main, argv)
    clear_env()
    tapes = sorted(p.name for p in out.glob("*.toktape")) if out.is_dir() else []
    calls = [" ".join(json.loads(x)) for x in (log.read_text().splitlines() if log.exists() else [])]
    return rc, [x for x in lines if x.startswith("tape D=")], [c for c in calls if c.startswith("record ")], tapes, err, work


@case
def check_reads_the_version_and_names_what_it_refuses(main):
    p = []
    d = scratch("tape-check.")
    binary, _ = fake_toktape(d)
    rc, lines, err = call(main, ["check", "--toktape-bin", binary])
    if rc != 0 or lines != [f"toktape v0.8.0 at {binary}"] or err:
        p.append(f"exit {rc}, {lines}, {err!r}; want 0 and one line naming the version and the path")
    for what, kw, want in (("an old toktape", {"version": "v0.2.6"}, f"toktape at {binary} is v0.2.6; 0.3.0 or newer is needed"),
                           ("a build with no version", {"version": "dev"}, "prints 'toktape dev' for `version`: no version to read"),
                           ("a toktape that does not answer", {"version_rc": 7}, "`version` exited 7")):
        binary, _ = fake_toktape(d, **kw)
        rc, lines, err = call(main, ["check", "--toktape-bin", binary])
        if rc != 2 or lines or want not in err or "tape.py: " not in err:
            p.append(f"{what}: exit {rc}, {lines}, {err!r}; want 2, nothing on stdout and {want!r} on stderr")
    clear_env()
    rc, lines, err = call(main, ["check", "--toktape-bin", str(d / "no-such-toktape")])
    if rc != 2 or "does not run" not in err:
        p.append(f"a path that is no binary: exit {rc}, {err!r}; want 2 naming it")
    saved = os.environ.get("PATH")
    os.environ["PATH"] = str(d / "no-bin")
    try:
        rc, lines, err = call(main, ["check"])
    finally:
        os.environ["PATH"] = saved if saved is not None else ""
    if rc != 2 or "tape.py: toktape is not on PATH (--toktape-bin names it)" != err:
        p.append(f"no toktape on PATH: exit {rc}, {err!r}; want 2 naming it")
    return p


@case
def record_replays_the_rows_prompt_on_the_rows_request_shape(main):
    tok = Tok()
    try:
        rc, lines, calls, tapes, err, work = record_run(main, tok, [1, 2, 3, 4], extra=("--note", "cell 1"))
        sent = json.loads((work / "prompt.jsonl").read_text())
        again = record_run(main, tok, [1, 2, 3, 4], extra=("--n", "96"))
    finally:
        tok.close()
    p = []
    want = (f"tape D=4 | file {tapes[0] if tapes else '-'} | toktape 0.8.0 | prompt exact | prompt_n 4 | cache_n 0 | "
            "pp 1000.50 tok/s | tg 40.00 tok/s | predicted_n 128 | caveats -")
    if rc != 0 or lines != [want] or tapes != ["run-rel-0.3.16-glm-cell1-D4.toktape"] or err:
        p.append(f"exit {rc}, {lines}, tapes {tapes}, {err!r}; want 0, one product line {want!r} and the tape file")
    shape = (r"^record --url http://127\.0\.0\.1:\d+ --endpoint completion --temp 0 --param top_k=1 --param ignore_eos=true "
             r"--param cache_prompt=false -n 128 --for 0 --wait 60s --host-label  --prompts \S+ --out \S+ "
             r"--tag rel-0.3.16-glm-cell1-D4 --note cell 1 -o json --quiet$")
    if len(calls) != 1 or not re.match(shape, calls[0]):
        p.append(f"toktape's argv is not the timed request's shape on the row's prompt: {calls}")
    if sent != {"name": "rel-0.3.16-glm-cell1-D4", "prompt": "w2 w3 w4"}:
        p.append(f"the prompt sent is not the server's text of the ids: {sent}")
    if not (work / "toktape.out").is_file():
        p.append("toktape's summary is not kept in the work directory")
    rc, lines, calls, tapes, err, work = again
    if len(calls) != 1 or " -n 96 --for 0 " not in calls[0] or "file run-rel-0.3.16-glm-cell1-D4.toktape" not in (lines or [""])[0]:
        p.append(f"--n does not reach toktape: {calls}, {lines}")
    return p


@case
def record_sends_the_prompt_the_server_makes_the_ids_of(main):
    p = []
    tok = Tok()
    try:
        rc, lines, calls, tapes, err, work = record_run(main, tok, [1, 3, 4])
        sent = json.loads((work / "prompt.jsonl").read_text())["prompt"]
        if rc != 0 or "| prompt exact |" not in lines[0] or sent != "w3 w4":
            p.append(f"a BOS that spells itself: exit {rc}, {lines}, sent {sent!r}; want prompt exact and the text without it")
        rc, lines, calls, tapes, err, work = record_run(main, tok, [2, 3, 4])
        sent = json.loads((work / "prompt.jsonl").read_text())["prompt"]
        if rc != 0 or not re.search(r"\| prompt text \| prompt_n 4 \|", lines[0]) or sent != "w2 w3 w4":
            p.append(f"ids with no BOS: exit {rc}, {lines}, sent {sent!r}; want prompt text, prompt_n 4 (D is 3), the whole text")
    finally:
        tok.close()
    for modes, word in ((("no-detok",), "/detokenize answered HTTP 404"), (("down",), "the server did not answer")):
        tok = Tok(modes)
        try:
            rc, lines, calls, tapes, err, work = record_run(main, tok, [1, 2])
        finally:
            tok.close()
        if rc != 1 or len(lines) != 1 or "tape D=2 | failed: " not in lines[0] or word not in lines[0] or calls:
            p.append(f"{modes}: exit {rc}, {lines}, toktape calls {len(calls)}; want 1, one failure line naming {word!r}, no call")
    return p


@case
def record_names_what_went_wrong_in_one_line(main):
    tok = Tok()
    try:
        p = []
        for mode, why in (("rc3", "toktape exit 3: toktape: every stream failed; read the server's log"),
                          ("noid", "names no run id"), ("slash", "names no run id"), ("notape", "wrote no"),
                          ("cache", "cache_n 64: the server restored a prefix"), ("failed", "1 of the tape's stream(s) did not answer"),
                          ("zero", "holds no figures of a request: prompt_n 0, predicted_n 0")):
            rc, lines, calls, tapes, err, work = record_run(main, tok, [1, 2, 3], mode=mode)
            if rc != 1 or len(lines) != 1 or not lines[0].startswith("tape D=3 | failed: ") or why not in lines[0]:
                p.append(f"{mode}: exit {rc}, {lines}; want 1 and one line 'tape D=3 | failed: ...' holding {why!r}")
        rc, lines, calls, tapes, err, work = record_run(main, tok, [1, 2, 3], mode="caveat")
        if rc != 0 or not lines[0].endswith("| predicted_n 128 | caveats machine_contended"):
            p.append(f"the tape's caveats are not in the line by code: {lines}")
    finally:
        tok.close()
    return p


@case
def a_toktape_that_hangs_fails_by_name(main):
    d = scratch("tape-hang.")
    hang = d / "toktape"
    hang.write_text(f"#!{sys.executable}\nimport sys, time\n"
                    "print('toktape v0.8.0') if sys.argv[1:] == ['version'] else time.sleep(30)\n")
    hang.chmod(0o755)
    tok = Tok()
    try:
        rc, lines, err = call(main, ["record", "--out", str(d / "tapes"), "--tag", "t", "--ids", ids_file(d, [1, 2]),
                                     "--url", tok.url, "--work", str(d / "work"), "--toktape-bin", str(hang),
                                     "--timeout", "0.5"])
    finally:
        tok.close()
    if rc != 1 or lines != ["tape D=2 | failed: toktape did not end within 0.5 s"]:
        return [f"exit {rc}, {lines}, {err}; want 1 and the timeout named in the one line"]
    return []


@case
def record_refuses_what_it_cannot_read_by_name(main):
    p = []
    d = scratch("tape-refuse.")
    binary, _ = fake_toktape(d)
    tok = Tok()
    try:
        base = ["record", "--out", str(d / "tapes"), "--tag", "t", "--url", tok.url, "--work", str(d / "w")]
        f = d / "ids"
        f.write_text("1\nx2\n3\n")
        rc, lines, err = call(main, base + ["--ids", str(f), "--toktape-bin", binary])
        if rc != 2 or lines or f"tape.py: line 2 of {f} ('x2') is no id" not in err:
            p.append(f"an ids file with a line that is no id: exit {rc}, {lines}, {err!r}; want 2 naming the line")
        rc, lines, err = call(main, base + ["--ids", str(d / "none"), "--toktape-bin", binary])
        if rc != 2 or "cannot read the ids file" not in err:
            p.append(f"a missing ids file: exit {rc}, {err!r}; want 2 naming it")
        rc, lines, err = call(main, base + ["--ids", ids_file(d, [1]), "--toktape-bin", str(d / "no-such")])
        if rc != 2 or len(lines) != 1 or not lines[0].startswith(f"tape D=1 | failed: toktape at {d / 'no-such'} does not run: [Errno 2] No such file or directory"):
            p.append(f"a toktape that is not there: exit {rc}, {lines}; want 2 and the failure line naming it")
        binary, _ = fake_toktape(d, version="v0.2.6")
        rc, lines, err = call(main, base + ["--ids", ids_file(d, [1]), "--toktape-bin", binary])
        if rc != 2 or [x.split(";")[0] for x in lines] != ["tape D=1 | failed: toktape at " + str(binary) + " is v0.2.6"]:
            p.append(f"an old toktape: exit {rc}, {lines}; want 2 and the failure line naming the version")
    finally:
        tok.close()
    clear_env()
    return p


SHA = "0" * 64
CELLS = {1: ("glm", "a", "5090", (512, 4096)), 2: ("mimo2", "bp", "both", (512,))}
FILES = {(1, 512): "a-512.toktape", (1, 4096): "a-4096.toktape", (2, 512): "b-512.toktape"}
ROWS = tuple(FILES)


def who(cell, d):
    """How publish names a row."""
    return f"cell {cell} ({CELLS[cell][0]} place={CELLS[cell][1]} cards={CELLS[cell][2]}) D={d}"


def url_of(key):
    return "https://tape.midagedev.com/r/fake" + FILES[key][: -len(".toktape")]


def expect_line(key, url=None, count=None, caveats="-"):
    """The line publish prints for the row `key` = (cell, D): the row, the tape's figures and its link."""
    count = count or f"prompt_n {key[1]}"
    return (who(*key) + f" | tape pp 194.44 tok/s, tg 30.09 tok/s ({count}, 128 tokens, toktape v0.8.0"
            + (f"; caveats {caveats}" if caveats != "-" else "") + f") | [recorded on tape]({url or url_of(key)})")


def tape_line(key, said="file", prompt_n=None, caveats="-"):
    """The `tape D=` line of a round-1 block: a recorded tape, or what the sitting wrote when it recorded none."""
    if said != "file":
        return f"tape D={key[1]} | {said}"
    return (f"tape D={key[1]} | file {FILES[key]} | toktape v0.8.0 | prompt exact | prompt_n {prompt_n or key[1]} | cache_n 0 | "
            f"pp 194.44 tok/s | tg 30.09 tok/s | predicted_n 128 | caveats {caveats}")


def verdict_at(d, said=None, result="green", tapes_on=True, prompt_n=None, caveats="-", only=ROWS):
    """A green verdict of two cells at <d>/0.3.16.sitting in sitting.sh's format, its round-1 blocks holding a tape line a row
    in `only` and its tapes directory the files. `said` maps a row to what its tape line says after `tape D=<D> | `."""
    said = said or {}
    v = d / "0.3.16.sitting"
    lines = ["sitting-verdict 1", "stage=portable", f"result={result}", "version=0.3.16", "tree=bloomery",
             f"binary_sha256={SHA}", "tapes=" + ("on" if tapes_on else "off"), "toktape=v0.8.0", ""]
    for cell, (profile, place, cards, depths) in CELLS.items():
        lines.append(f"cell {cell}: {profile}{'=/m/m.gguf' if cell == 2 else ''} place={place} cards={cards} ctx=unset "
                     f"D={','.join(map(str, depths))} checks=- fold=off")
    for cell, (profile, place, cards, depths) in CELLS.items():
        lines += ["", f"== round 1 cell {cell} rc=0 result=green", "stamp finished=2026-11-23T00:00:00Z binary_sha256=" + SHA]
        for d_ in depths:
            key = (cell, d_)
            lines.append(f"ROW r1 bloomery@portable/place={place}/ctx=16384 D={d_} | tg_tok/s 30.09 | pp_tok/s 194.44 | load_s 5 | majflt 0")
            if key in only:
                lines += [tape_line(key, said.get(key, "file"), prompt_n, caveats),
                          f"TAPE r1 bloomery@portable/place={place}/ctx=16384 D={d_} rc=0"]
        lines += ["", f"== round 2 cell {cell} rc=0 result=green", "stamp finished=2026-11-23T00:10:00Z binary_sha256=" + SHA]
        lines += [f"ROW r2 bloomery@portable/place={place}/ctx=16384 D={d_} | tg_tok/s 30.09 | pp_tok/s 194.44 | load_s 5 | majflt 0"
                  for d_ in depths]
    v.write_text("\n".join(lines) + "\n")
    tapes = d / "0.3.16.sitting.d" / "tapes"
    tapes.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        (tapes / name).write_text("tape")
    return v


def publish_run(main, v, mode="", extra=(), version="v0.8.0"):
    """One `publish` of the verdict `v` through the stand-in: (rc, stdout lines, stderr, toktape's publish calls as argv)."""
    d = v.parent
    binary, log = fake_toktape(d, mode, version)
    log.unlink(missing_ok=True)
    rc, lines, err = call(main, ["publish", str(v), "--toktape-bin", binary, *extra])
    clear_env()
    calls = [json.loads(x) for x in (log.read_text().splitlines() if log.exists() else [])]
    return rc, lines, err, [c for c in calls if c[0] == "publish"]


def under(d):
    """Every byte under `d`, to look for one that must not be anywhere there."""
    return "".join(x.read_text(errors="replace") for x in d.rglob("*") if x.is_file())


def ledger_of(d):
    return d / "0.3.16.sitting.d" / "tapes" / "published.tsv"


@case
def publish_prints_a_link_a_row_and_uploads_once(main):
    p = []
    d = scratch("tape-pub.")
    v = verdict_at(d)
    rc, lines, err, calls = publish_run(main, v)
    want = [f"tapes: bloomery 0.3.16, stage=portable, toktape v0.8.0, verdict {v}", *[expect_line(k) for k in ROWS],
            f"tapes: 3 of 3 published (ledger {ledger_of(d)})"]
    if rc != 0 or lines != want or err:
        p.append(f"exit {rc}, {lines}, {err!r}; want 0, the head line, a link a row in the cells' order and the count")
    if [c[1].rsplit("/", 1)[1] for c in calls] != [FILES[k] for k in ROWS] or \
            not all(c[2] == "--yes" and c[3] == "--title" and c[5] == "--note" and c[-2:] == ["-o", "json"] for c in calls):
        p.append(f"toktape's publish calls are not each tape's path, --yes, a title, a note and -o json: {calls}")
    for c in calls:
        title, note = c[c.index("--title") + 1], c[c.index("--note") + 1]
        if not re.match(r"^bloomery 0\.3\.16: cell [12] (glm|mimo2) place=(a|bp) cards=(5090|both), a (512|4096)-token prompt$", title):
            p.append(f"the title names the release, the cell and the prompt: {title!r}")
        if "A witness beside the row, not the row" not in note or "cache_prompt off" not in note or SHA not in note:
            p.append("the note does not say the tape is a witness beside the row, its request's shape and the binary's sha256")
    kept = ledger_of(d).read_text().splitlines()
    if [k.split("\t")[:3] for k in kept] != [[FILES[k], "fake" + FILES[k][:-8], url_of(k)] for k in ROWS] or \
            not all(re.fullmatch(r"[^\t]+\t[^\t]+\t[^\t]+\t[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z", k) for k in kept):
        p.append(f"the ledger is not a line a tape with its file, id, link and the time: {kept}")
    rc, lines2, err, calls2 = publish_run(main, v)
    if rc != 0 or calls2 or lines2 != lines:
        p.append(f"a second publish: exit {rc}, uploads {len(calls2)}, {lines2}; want none and the same lines")
    return p


@case
def publish_carries_the_tape_s_own_facts(main):
    p = []
    d = scratch("tape-facts.")
    v = verdict_at(d, prompt_n=513, caveats="machine_contended,short_generation", only=((1, 512),))
    rc, lines, err, calls = publish_run(main, v)
    want = expect_line((1, 512), count="prompt_n 513, not the row's D 512", caveats="machine_contended,short_generation")
    if rc != 1 or lines[1] != want:
        p.append(f"exit {rc}, {lines}; want the row's line saying the count is not the row's D, and the caveats")
    d = scratch("tape-notext.")
    rc, lines, err, calls = publish_run(main, verdict_at(d), extra=("--no-text",))
    if rc != 0 or len(calls) != 3 or not all(c[-2:] == ["-o", "json"] and "--no-text" in c for c in calls):
        p.append(f"--no-text: exit {rc}, {len(calls)} calls; want each carrying it")
    d = scratch("tape-text.")
    rc, lines, err, calls = publish_run(main, verdict_at(d))
    if any("--no-text" in c for c in calls):
        p.append("--no-text reached toktape without being asked")
    return p


@case
def a_dry_run_uploads_nothing(main):
    d = scratch("tape-dry.")
    v = verdict_at(d)
    rc, lines, err, calls = publish_run(main, v, extra=("--dry-run",))
    p = []
    if rc != 0 or not lines[0].endswith(" (a dry run: nothing is uploaded)") or lines[-1] != f"tapes: 3 of 3 to publish (ledger {ledger_of(d)})":
        p.append(f"exit {rc}, {lines}, {err!r}; want 0, the head line saying it is a dry run, and the count")
    if lines[1:-1] != [x for k in ROWS for x in (who(*k) + f": would publish {FILES[k]}", f"    would upload {FILES[k]} with its text")]:
        p.append(f"a dry run does not print what would go up, a row: {lines}")
    if [c[2:] for c in calls] != [["--dry-run"]] * 3:
        p.append(f"a dry run's toktape calls are not `publish <tape> --dry-run`: {calls}")
    if ledger_of(d).exists():
        p.append("a dry run wrote the ledger")
    rc, lines, err, calls = publish_run(main, v)
    if len(calls) != 3:
        p.append("after a dry run the tapes are still to publish")
    return p


@case
def publish_never_prints_the_delete_token(main):
    p = []
    d = scratch("tape-secret.")
    rc, lines, err, calls = publish_run(main, verdict_at(d))
    if "SECRET-DELETE-TOKEN" in "\n".join(lines + [err]) + under(d / "0.3.16.sitting.d"):
        p.append("the delete token reached the output or a file under the sitting's directory")
    if rc != 0 or len(calls) != 3:
        p.append(f"exit {rc}, uploads {len(calls)}; want 0 and three")
    for mode, why in (("pubrc5", "toktape exit 5: toktape: the service refused the upload; toktape publish <tape> --dry-run prints what would be sent"),
                      ("nourl", "toktape's receipt names no link")):
        d = scratch("tape-secret.")
        rc, lines, err, calls = publish_run(main, verdict_at(d), mode=mode)
        if rc != 1 or lines[1:-1] != [who(*k) + ": NOT PUBLISHED: " + why for k in ROWS] or lines[-1] != f"tapes: 0 of 3 published (ledger {ledger_of(d)})":
            p.append(f"{mode}: exit {rc}, {lines}; want 1 and each row NOT PUBLISHED naming {why!r}")
        if ledger_of(d).exists():
            p.append(f"{mode}: a tape that did not go up is in the ledger")
        if "SECRET-DELETE-TOKEN" in "\n".join(lines + [err]) + under(d / "0.3.16.sitting.d"):
            p.append(f"{mode}: the delete token reached the output or the sitting's directory")
    return p


@case
def publish_names_each_row_without_a_tape(main):
    p = []
    d = scratch("tape-gap.")
    bad = (1, 4096)
    for said, only, why in (("failed: toktape exit 3: every stream failed", ROWS, "toktape exit 3: every stream failed"),
                            ("file", ROWS[:1] + ROWS[2:], "the verdict has no tape line for this row"),
                            ("file ../elsewhere.toktape | toktape v0.8.0", ROWS, "its line is malformed: file ../elsewhere.toktape | toktape v0.8.0")):
        v = verdict_at(d, said={bad: said}, only=only)
        rc, lines, err, calls = publish_run(main, v)
        got = [x for x in lines if "NOT PUBLISHED" in x]
        if rc != 1 or len(calls) != 2 or got != [who(*bad) + ": NOT PUBLISHED: the sitting recorded no tape for this row: " + why] or \
                lines[-1] != f"tapes: 2 of 3 published (ledger {ledger_of(d)})":
            p.append(f"{said[:24]!r}: exit {rc}, {lines}, uploads {len(calls)}; want 1, two uploads and the row's NOT PUBLISHED line")
        ledger_of(d).unlink(missing_ok=True)
    ledger_of(d).unlink(missing_ok=True)
    v = verdict_at(d)
    (d / "0.3.16.sitting.d" / "tapes" / FILES[(1, 512)]).unlink()
    rc, lines, err, calls = publish_run(main, v)
    if rc != 1 or not any(x == who(1, 512) + f": NOT PUBLISHED: no tape at {d / '0.3.16.sitting.d/tapes/a-512.toktape'}" for x in lines):
        p.append(f"a tape file that is gone: exit {rc}, {lines}; want 1 and the path named")
    if len(calls) != 2:
        p.append(f"a tape file that is gone: {len(calls)} uploads; want the other two")
    return p


@case
def publish_refuses_a_verdict_it_cannot_publish_by_name(main):
    p = []
    d = scratch("tape-refuses.")
    for made, why in ((lambda: verdict_at(d, result="red"), "is result=red: tapes are published from a green verdict; --resume the sitting's red runs and publish the merged verdict"),
                      (lambda: verdict_at(d, tapes_on=False), "recorded no tapes (tapes=off)")):
        v = made()
        rc, lines, err, calls = publish_run(main, v)
        if rc != 64 or lines or calls or not err.startswith(f"tape.py: {v} {why}"):
            p.append(f"{why[:22]}: exit {rc}, {lines}, {err!r}, uploads {calls}; want 64, nothing on stdout, the reason on stderr")
    v = d / "none.sitting"
    rc, lines, err, calls = publish_run(main, v)
    if rc != 2 or not err.startswith(f"tape.py: cannot read the verdict {v}: "):
        p.append(f"no verdict at the path: exit {rc}, {err!r}; want 2 naming it")
    v.write_text("nonsense\n")
    rc, lines, err, calls = publish_run(main, v)
    if rc != 64 or "is no verdict of tools/release/sitting.sh (its first line is not 'sitting-verdict 1')" not in err:
        p.append(f"a file that is no verdict: exit {rc}, {err!r}; want 64 naming it")
    v.write_text("sitting-verdict 1\nstage=portable\nresult=green\ntapes=on\n")
    rc, lines, err, calls = publish_run(main, v)
    if rc != 0 or lines[1:] != [f"tapes: 0 of 0 published (ledger {d / 'none.sitting.d/tapes/published.tsv'})"]:
        p.append(f"a verdict with no cell: exit {rc}, {lines}; want 0 and the count naming none")
    rc, lines, err, calls = publish_run(main, verdict_at(d), version="v0.1.0")
    if rc != 2 or calls or f"tape.py: toktape at {d / 'toktape'} is v0.1.0; 0.3.0 or newer is needed" not in err:
        p.append(f"an old toktape: exit {rc}, {err!r}; want 2 naming it")
    return p


@case
def usage_errors_exit_64(main):
    p = []
    d = scratch("tape-usage.")
    ok = ["--out", str(d / "t"), "--tag", "t", "--ids", ids_file(d, [1]), "--url", "http://127.0.0.1:1"]

    def without(flag):
        i = ok.index(flag)
        return ok[:i] + ok[i + 2:]

    def setting(flag, value):
        i = ok.index(flag)
        return ok[:i + 1] + [value] + ok[i + 2:]

    for argv in ([], ["bogus"], ["publish"], ["record", *without("--out")], ["record", *without("--ids")], ["record", *without("--url")],
                 ["record", *without("--tag")], ["record", *setting("--tag", "has space")], ["record", *setting("--url", "127.0.0.1:1")],
                 ["record", *ok, "--n", "0"], ["record", *ok, "--bogus"]):
        rc, lines, err = call(main, argv)
        if rc != 64 or lines or "tape.py: " not in err or "usage:" not in err:
            p.append(f"{argv}: exit {rc}, {lines}, {err!r}; want 64, nothing on stdout and the usage on stderr")
    cli = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve())], capture_output=True, text=True,
                         env={"PATH": os.environ.get("PATH", "")})
    if cli.returncode != 64 or "a command is required: check, record or publish" not in cli.stderr:
        p.append(f"the script with no command: exit {cli.returncode}, stderr {cli.stderr!r}")
    return p


@case
def record_reads_no_environment_the_caller_did_not_spell_out(main):
    """The server and the ids are flags: a variable of the sitting's hook that is set and not passed is not read."""
    p = []
    d = scratch("tape-env.")
    os.environ.update({"BLOOMERY_SESSION_URL": "http://127.0.0.1:1", "BLOOMERY_SESSION_DIR": str(d / "w"),
                       "DEPTH_SERVE_TAPE_IDS": ids_file(d, [1])})
    try:
        for what, argv in (("--url", ["--ids", ids_file(d, [1])]), ("--ids", ["--url", "http://127.0.0.1:1"])):
            rc, lines, err = call(main, ["record", "--out", str(d / "t"), "--tag", "t", *argv])
            if rc != 64 or lines:
                p.append(f"no {what} with the hook's variables set: exit {rc}, {lines}; want 64: the flags are required")
    finally:
        for k in ("BLOOMERY_SESSION_URL", "BLOOMERY_SESSION_DIR", "DEPTH_SERVE_TAPE_IDS"):
            os.environ.pop(k, None)
    return p


@case
def the_minimum_version_is_the_first_release_with_what_this_tool_sends(main):
    g = main.__globals__
    p = []
    if g["MIN_VERSION"] != (0, 3, 0):
        p.append(f"MIN_VERSION {g['MIN_VERSION']} is not 0.3.0")
    for said, ok in (("toktape v0.8.0", True), ("toktape 0.7.1", True), ("toktape v0.3.0-46-g5dd7301", True),
                     ("toktape dev", False), ("toktape v1", False), ("", False)):
        if bool(g["VERSION"].match(said)) != ok:
            p.append(f"the version line {said!r} reads {'a version' if g['VERSION'].match(said) else 'no version'}")
    return p


# Each mutant puts back what a tool that ignored one of the rules above would do; the case named beside it must fail on it.
# An anchor that no longer matches fails the self-test by name.
MUTANTS = (
    ("takes any toktape version", (("if got < MIN_VERSION:", "if False:", 1),), "check-reads-the-version-and-names-what-it-refuses"),
    ("reads a version line that names none",
     (("m = VERSION.match(said)", 'm = VERSION.match(said) or re.match("(?:(9))", "9")', 1),),
     "check-reads-the-version-and-names-what-it-refuses"),
    ("publishes from a red verdict", (('if header.get("result") != "green":', "if False:", 1),),
     "publish-refuses-a-verdict-it-cannot-publish-by-name"),
    ("publishes a verdict that recorded no tapes", (('if header.get("tapes") != "on":', "if False:", 1),),
     "publish-refuses-a-verdict-it-cannot-publish-by-name"),
    ("uploads without being told to", (('["--yes", "--title", title, "--note", body]', '["--title", title, "--note", body]', 1),),
     "publish-prints-a-link-a-row-and-uploads-once"),
    ("uploads a tape twice", (('if tape.fields["file"] in done:', "if False:", 1),), "publish-prints-a-link-a-row-and-uploads-once"),
    ("uploads in a dry run",
     (('rc, out, err = run_toktape(cmd + ["--dry-run"], 120)', 'rc, out, err = run_toktape(cmd + ["--yes", "-o", "json"], 120)', 1),),
     "a-dry-run-uploads-nothing"),
    ("prints a dry run as a publish", (("'to publish' if a.dry_run else 'published'", "'published'", 1),), "a-dry-run-uploads-nothing"),
    ("prints a row without a tape as published", (("if tape is None or tape.why is not None:", "if False:", 1),),
     "publish-names-each-row-without-a-tape"),
    ("takes a missing tape file for one", (("if not tape_path.is_file():", "if False:", 1),), "publish-names-each-row-without-a-tape"),
    ("takes a tape line of any shape",
     (('if rest.startswith("file ") and name.endswith(".toktape") and RUN_ID.match(name[: -len(".toktape")]):',
       'if rest.startswith("file ") or name.endswith(".toktape") or RUN_ID.match(name[: -len(".toktape")]):', 1),),
     "publish-names-each-row-without-a-tape"),
    ("passes a toktape that published nothing",
     (('            if rc != 0:\n                raise Fail(f"toktape exit {rc}: {toktape_failure(out, err)}")\n            run_id, url = receipt_link(out)',
       '            if False:\n                raise Fail(f"toktape exit {rc}: {toktape_failure(out, err)}")\n            run_id, url = receipt_link(out)', 1),),
     "publish-never-prints-the-delete-token"),
    ("prints the receipt whole", (('return run_id if isinstance(run_id, str) else "", url', 'return run_id if isinstance(run_id, str) else "", url + out', 1),),
     "publish-never-prints-the-delete-token"),
    ("hides the prompt count that is not the row's", (('f", not the row\'s D {tape.d}"', '""', 1),), "publish-carries-the-tape-s-own-facts"),
    ("drops --no-text", (('["--yes", "--title", title, "--note", body] + a.no_text + ["-o", "json"]',
                          '["--yes", "--title", title, "--note", body] + ["-o", "json"]', 1),), "publish-carries-the-tape-s-own-facts"),
    ("publishes a row the verdict names no tape of", (("tape = tapes.get((idx, d))", "tape = tapes.get((idx, d)) or Tape(idx, d, {'file': 'x'})", 1),),
     "publish-names-each-row-without-a-tape"),
    ("passes a toktape that recorded nothing",
     (('    (work / "toktape.err").write_text(err, encoding="utf-8")\n    if rc != 0:',
       '    (work / "toktape.err").write_text(err, encoding="utf-8")\n    if False:', 1),),
     "record-names-what-went-wrong-in-one-line"),
    ("trusts a summary that names no run", (("if not isinstance(run_id, str) or not RUN_ID.match(run_id):", "if False:", 1),),
     "record-names-what-went-wrong-in-one-line"),
    ("takes a cache-served prefill for the row's", (("if cache_n:", "if False:", 1),), "record-names-what-went-wrong-in-one-line"),
    ("takes a tape with no figures", (("if prompt_n <= 0 or predicted_n <= 0:", "if False:", 1),),
     "record-names-what-went-wrong-in-one-line"),
    ("takes a stream that did not answer", (('if isinstance(streams, dict) and streams.get("streams_failed"):', "if False:", 1),),
     "record-names-what-went-wrong-in-one-line"),
    ("prints no failure line", (('rc, line = 1, f"tape D={len(ids)} | failed: {e}"', "rc, line = 1, ''", 1),),
     "record-names-what-went-wrong-in-one-line"),
    ("sends any text as the prompt", (("if tokenizes_to(url, full, timeout) == ids:", "if True:", 1),),
     "record-sends-the-prompt-the-server-makes-the-ids-of"),
    ("sends the whole prompt when the server adds its first id",
     (("if tokenizes_to(url, rest, timeout) == ids:", "if False:", 1),), "record-sends-the-prompt-the-server-makes-the-ids-of"),
    ("sends the prompt as a chat turn", (('"--endpoint", "completion"', '"--endpoint", "chat"', 1),),
     "record-replays-the-rows-prompt-on-the-rows-request-shape"),
    ("lets the server restore the prompt's prefix", (('"--param", "cache_prompt=false", ', '', 1),),
     "record-replays-the-rows-prompt-on-the-rows-request-shape"),
    ("asks for the server's default sampling", (('"--temp", "0", ', '', 1),),
     "record-replays-the-rows-prompt-on-the-rows-request-shape"),
    ("lets the answer end before its tokens", (('"--param", "ignore_eos=true", ', '', 1),),
     "record-replays-the-rows-prompt-on-the-rows-request-shape"),
    ("prints toktape's caveats not at all", (("caveats {','.join(caveats) or '-'}", "caveats -", 1),),
     "record-names-what-went-wrong-in-one-line"),
    ("reads the ids from nowhere", (("ids = read_ids(a.ids)", "ids = [1, 2, 3]", 1),), "record-refuses-what-it-cannot-read-by-name"),
    ("prints no version of the tape", (("toktape {summary.get('toktape_version') or version}", "toktape -", 1),),
     "record-replays-the-rows-prompt-on-the-rows-request-shape"),
    ("reads the sitting's variables itself", (('ap.add_argument("--url", required=True)', 'ap.add_argument("--url", default=os.environ.get("BLOOMERY_SESSION_URL"))', 1),),
     "record-reads-no-environment-the-caller-did-not-spell-out"),
)


def mutant_main(pairs):
    text = Path(__file__).read_text()
    cut = text.index("# Each mutant puts back")
    src, rest = text[:cut], text[cut:]
    for old, new, count in pairs:
        if src.count(old) != count:
            raise AssertionError(f"mutant anchor {old!r} is in the code {src.count(old)} times, want {count}")
        src = src.replace(old, new)
    src += rest
    ns = {"__file__": str(Path(__file__).resolve()), "__name__": "tape_mutant"}
    exec(compile(src, "<mutant>", "exec"), ns)
    return ns["main"]


def run_cases(main, only=None):
    problems = {}
    for name, fn in CASES.items():
        if only and name not in only:
            continue
        clear_env()
        try:
            got = fn(main)
        except Exception as e:  # a case that raised is a case that failed: the mutant's crash is its red
            got = [f"raised {type(e).__name__}: {e}"]
        clear_env()
        if got:
            problems[name] = got
    return problems


def self_test():
    try:
        problems = run_cases(main)
        for name, p in problems.items():
            for x in p:
                print(f"FAIL {name}: {x}")
        if problems:
            print(f"tape.py self-test: {len(problems)} of {len(CASES)} cases FAILED")
            return 1
        bad = 0
        for what, pairs, name in MUTANTS:
            try:
                red = run_cases(mutant_main(pairs), only={name}).get(name)
            except AssertionError as e:
                red = [str(e)]
            print(f"mutant {what}: {name} {'red: ' + head(red[0], 110) if red else 'GREEN'}")
            bad += not red
        if bad:
            print(f"tape.py self-test: {bad} of {len(MUTANTS)} mutants left their case green")
            return 1
        print(f"tape.py self-test: {len(CASES)} cases ok, {len(MUTANTS)} mutants red")
        return 0
    finally:
        for d in SCRATCH:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    sys.exit(main(sys.argv[1:]))
