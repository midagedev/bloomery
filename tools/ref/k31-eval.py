#!/usr/bin/env python3
"""k31-eval.py — Kimi K3.1 quant quality against the released model, through OpenRouter.

  teacher  the released model's greedy answers and top-20 log-probabilities for the held-out prompts
           (tools/ref/k31-eval-prompts.jsonl), one JSON line per prompt, resumable, with a spend limit
  ids      the teacher file as the student's `generate_kimi_k3 --score` input (one line per prompt)
  score    per-position agreement, NLL difference and KL of a student run against the teacher

    python3 -I tools/ref/k31-eval.py teacher --k31-folder DIR --out teacher.jsonl [--budget-usd 2.50]
    python3 -I tools/ref/k31-eval.py ids     --teacher teacher.jsonl --out student-in.tsv
    just gen-kimi-k3 --score student-in.tsv > student.jsonl    (on the box)
    python3 -I tools/ref/k31-eval.py score   --teacher teacher.jsonl --student student.jsonl
    python3 -I tools/ref/k31-eval.py --self-test

The key (`OPENROUTER_API_KEY` in ~/.config/outsource/credentials) is read inside `teacher` and goes
nowhere but the request's Authorization header. The prompt is rendered by the K3.1 folder's own
`encoding_k3.py` in its no-thinking form (`thinking=False`: the assistant turn opens `response`, not
`think`) and encoded by the folder's `tiktoken.model`; a prompt whose token count differs from the
server's `usage.prompt_tokens` is refused by name (not scored), which is how the render is held equal
to the server's. The server's count leaves out the render's last three tokens, the channel opener
`<|open|>response<|sep|>` (measured: "Say hi." 21 against our 24; reasoning on, `think`: 88 against 91,
and with `reasoning.effort` low 89 against 92); every other token must agree. Reasoning is off in the
request (`reasoning: {"enabled": false}`).
"""

import argparse
import base64
import contextlib
import importlib.util
import io
import json
import math
import os
import re
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request

MODEL = "moonshotai/kimi-k3.1-pro"
PROVIDER = "Moonshot AI"
URL = "https://openrouter.ai/api/v1/chat/completions"
CREDENTIALS = "~/.config/outsource/credentials"
TOP = 20
FILLER_LP = -9000.0  # the API pads top_logprobs with -9999 entries: not a log-probability
RETRIES = 4
CATEGORIES = ("en", "ko", "code")
OPENER = "<|open|>response<|sep|>"  # the render's last tokens, which usage.prompt_tokens does not count
OPENER_TOKENS = 3


class EvalError(Exception):
    """A named refusal; main() prints it and exits 1."""


# ---------------------------------------------------------------- vocabulary and encoder

# \p{Han} by its script ranges (the regex's own table is not in the standard library).
_HAN = [
    (0x2E80, 0x2E99), (0x2E9B, 0x2EF3), (0x2F00, 0x2FD5), (0x3005, 0x3005), (0x3007, 0x3007),
    (0x3021, 0x3029), (0x3038, 0x303B), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFA6D),
    (0xFA70, 0xFAD9), (0x16FE2, 0x16FE3), (0x16FF0, 0x16FF1), (0x20000, 0x2A6DF),
    (0x2A700, 0x2EBEF), (0x2EBF0, 0x2EE5D), (0x2F800, 0x2FA1D), (0x30000, 0x3134A),
    (0x31350, 0x323AF),
]
# Unicode White_Space.
_WS = [(9, 13), (32, 32), (0x85, 0x85), (0xA0, 0xA0), (0x1680, 0x1680), (0x2000, 0x200A),
       (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000)]
_CONTRACTION = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"

_pretok_cache = []


def _cls(ranges, negate=False):
    """A character class over `ranges` (inclusive codepoint pairs), complemented over all of Unicode
    when `negate`."""
    if negate:
        inv, at = [], 0
        for lo, hi in sorted(ranges):
            if lo > at:
                inv.append((at, lo - 1))
            at = max(at, hi + 1)
        if at <= 0x10FFFF:
            inv.append((at, 0x10FFFF))
        ranges = inv
    return "[" + "".join(f"\\U{lo:08x}-\\U{hi:08x}" for lo, hi in ranges) + "]"


def _ranges(codepoints):
    out, start, prev = [], None, None
    for c in codepoints:
        if start is None:
            start = prev = c
        elif c == prev + 1:
            prev = c
        else:
            out.append((start, prev))
            start = prev = c
    if start is not None:
        out.append((start, prev))
    return out


def pretokenizer():
    """The K3 tiktoken split pattern (tokenization_kimi.py `pat_str`) as a standard-library regex:
    its `\\p{..}` classes are built from `unicodedata` categories and its `&&[^\\p{Han}]` set
    intersections by set arithmetic."""
    if _pretok_cache:
        return _pretok_cache[0]
    by_cat = {}
    for c in range(0x110000):
        by_cat.setdefault(unicodedata.category(chr(c)), []).append(c)
    han = {c for lo, hi in _HAN for c in range(lo, hi + 1)}
    upper = _ranges(sorted(set().union(*(by_cat[k] for k in ("Lu", "Lt", "Lm", "Lo", "Mn", "Mc", "Me"))) - han))
    lower = _ranges(sorted(set().union(*(by_cat[k] for k in ("Ll", "Lm", "Lo", "Mn", "Mc", "Me"))) - han))
    letters = [c for k in ("Lu", "Ll", "Lt", "Lm", "Lo") for c in by_cat[k]]
    numbers = [c for k in ("Nd", "Nl", "No") for c in by_cat[k]]
    ln = _ranges(sorted(set(letters) | set(numbers)))
    lnw = _ranges(sorted(set(letters) | set(numbers) | {c for lo, hi in _WS for c in range(lo, hi + 1)}))
    ws = _cls(_WS)
    notws = _cls(_WS, negate=True)
    U, L = _cls(upper), _cls(lower)
    lead = _cls(sorted(ln + [(10, 10), (13, 13)]), negate=True)
    punct = _cls(lnw, negate=True)
    alts = [
        _cls([(lo, hi) for lo, hi in _HAN]) + "+",
        f"{lead}?{U}*{L}+{_CONTRACTION}",
        f"{lead}?{U}+{L}*{_CONTRACTION}",
        _cls(_ranges(sorted(numbers))) + "{1,3}",
        f" ?{punct}+[\\r\\n]*",
        f"{ws}*[\\r\\n]+",
        f"{ws}+(?!{notws})",
        f"{ws}+",
    ]
    _pretok_cache.append(re.compile("|".join(alts)))
    return _pretok_cache[0]


class Vocab:
    """The K3.1 vocabulary: `ranks` bytes -> id (tiktoken.model), `special` text -> id."""

    def __init__(self, ranks, special):
        self.ranks = dict(ranks)
        self.special = dict(special)
        self.by_special_bytes = {t.encode(): i for t, i in self.special.items()}
        self._bpe_cache = {}
        self._special_re = (re.compile("|".join(re.escape(t) for t in sorted(self.special, key=len, reverse=True)))
                            if self.special else None)

    @classmethod
    def load(cls, folder):
        ranks = {}
        path = os.path.join(folder, "tiktoken.model")
        try:
            with open(path, "rb") as f:
                for n, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    tok, rank = line.split()
                    ranks[base64.b64decode(tok)] = int(rank)
        except (OSError, ValueError) as e:
            raise EvalError(f"vocabulary {path}: {e}") from e
        try:
            with open(os.path.join(folder, "tokenizer_config.json"), encoding="utf-8") as f:
                added = json.load(f)["added_tokens_decoder"]
        except (OSError, ValueError, KeyError) as e:
            raise EvalError(f"{folder}/tokenizer_config.json: added_tokens_decoder: {e}") from e
        return cls(ranks, {v["content"]: int(k) for k, v in added.items()})

    def id_of(self, raw, text=None):
        """The id of the token whose bytes are `raw`: one vocabulary entry or one special token's text,
        never a guess. A token the API gives no bytes for is a special token named by `text`, or an
        error."""
        if raw is None:
            if text in self.special:
                return self.special[text]
            raise EvalError(f"token {text!r} has no bytes and is no special token")
        i = self.ranks.get(bytes(raw))
        if i is None:
            i = self.by_special_bytes.get(bytes(raw))
        if i is None:
            raise EvalError(f"token bytes {list(raw)} ({bytes(raw)!r}) are not one vocabulary entry")
        return i

    def _bpe(self, piece):
        hit = self._bpe_cache.get(piece)
        if hit is not None:
            return hit
        whole = self.ranks.get(piece)
        if whole is not None:
            return [whole]
        parts = [piece[i:i + 1] for i in range(len(piece))]
        while len(parts) > 1:
            best, at = None, -1
            for i in range(len(parts) - 1):
                r = self.ranks.get(parts[i] + parts[i + 1])
                if r is not None and (best is None or r < best):
                    best, at = r, i
            if best is None:
                break
            parts[at:at + 2] = [parts[at] + parts[at + 1]]
        try:
            out = [self.ranks[p] for p in parts]
        except KeyError as e:
            raise EvalError(f"BPE piece {e.args[0]!r} of {piece!r} is not in the vocabulary") from e
        self._bpe_cache[piece] = out
        return out

    def encode_text(self, text):
        """Ordinary text: the split pattern, then byte-pair merging of each piece."""
        out, pos = [], 0
        for m in pretokenizer().finditer(text):
            if m.start() != pos:
                raise EvalError(f"the split pattern skips {text[pos:m.start()]!r}")
            out.extend(self._bpe(m.group().encode("utf-8")))
            pos = m.end()
        if pos != len(text):
            raise EvalError(f"the split pattern skips {text[pos:]!r}")
        return out

    def encode_segment(self, text, allow_special):
        """One `encoding_k3.EncodeSegment`: with `allow_special` its special-token texts are their ids."""
        if not allow_special or self._special_re is None:
            return self.encode_text(text)
        out, pos = [], 0
        for m in self._special_re.finditer(text):
            out.extend(self.encode_text(text[pos:m.start()]))
            out.append(self.special[m.group()])
            pos = m.end()
        out.extend(self.encode_text(text[pos:]))
        return out


def load_renderer(folder, vocab):
    """`render(messages) -> ids`: the folder's `encoding_k3.build_chat_segments` in the no-thinking
    form, each segment encoded by `vocab`."""
    path = os.path.join(folder, "encoding_k3.py")
    spec = importlib.util.spec_from_file_location("encoding_k3", path)
    if spec is None or spec.loader is None:
        raise EvalError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["encoding_k3"] = mod
    spec.loader.exec_module(mod)

    opener = vocab.encode_segment(OPENER, True)
    if len(opener) != OPENER_TOKENS:
        raise EvalError(f"the channel opener is {len(opener)} tokens, not {OPENER_TOKENS}")

    def render(messages):
        segs = mod.build_chat_segments(messages, add_generation_prompt=True, thinking=False)
        ids = []
        for s in segs:
            ids.extend(vocab.encode_segment(s.text, s.allow_special))
        if ids[-OPENER_TOKENS:] != opener:
            raise EvalError("the render does not end with the channel opener")
        return ids

    return render


# ---------------------------------------------------------------- teacher


def read_key():
    path = os.path.expanduser(CREDENTIALS)
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("'\"")
                    if key:
                        return key
    except OSError as e:
        raise EvalError(f"credentials {path}: {e.strerror}") from e
    raise EvalError(f"OPENROUTER_API_KEY is not in {path}")


def http_transport(key):
    """`post(body) -> (status, json)`; the key is in this closure's header and nowhere else."""

    def post(body):
        req = urllib.request.Request(
            URL, json.dumps(body).encode(),
            {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read())
            except ValueError:
                return e.code, {}
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            return 599, {"error": {"message": type(e).__name__}}

    return post


def request_body(messages):
    return {
        "model": MODEL,
        "messages": messages,
        "provider": {"order": [PROVIDER], "allow_fallbacks": False},
        "temperature": 0,
        "max_tokens": 256,
        "logprobs": True,
        "top_logprobs": TOP,
        "reasoning": {"enabled": False},
        "usage": {"include": True},
    }


def call(post, body, sleep):
    """One completion: 429 and 5xx (and a transport failure, 599) retried up to RETRIES times with a
    growing wait; any other status stops by name."""
    for attempt in range(RETRIES + 1):
        status, data = post(body)
        if status == 200:
            return data
        if status == 429 or status >= 500:
            if attempt == RETRIES:
                raise EvalError(f"HTTP {status} after {RETRIES} retries")
            sleep(2.0 * (attempt + 1))
            continue
        msg = (data.get("error") or {}).get("message", "") if isinstance(data, dict) else ""
        raise EvalError(f"HTTP {status} (not retried): {str(msg)[:200]}")
    raise EvalError("unreachable")


def teacher_record(prompt, data, vocab, render):
    """The stored record of one answered prompt; what cannot be held is listed in `refused`."""
    messages = [{"role": "user", "content": prompt["prompt"]}]
    usage = data.get("usage") or {}
    choice = (data.get("choices") or [{}])[0]
    rec = {
        "id": prompt["id"], "cat": prompt["cat"], "messages": messages,
        "prompt_tokens": usage.get("prompt_tokens"), "cost": usage.get("cost"),
        "provider": data.get("provider"), "finish_reason": choice.get("finish_reason"),
        "text": (choice.get("message") or {}).get("content"),
        "tokens": [], "refused": [], "cut": None,
    }
    if not isinstance(rec["cost"], (int, float)):
        raise EvalError(f"{prompt['id']}: the response has no usage.cost (the spend limit needs it)")
    if rec["provider"] != PROVIDER:
        raise EvalError(f"{prompt['id']}: served by {rec['provider']!r}, not {PROVIDER!r}")
    if (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"):
        rec["refused"].append("reasoning tokens were produced")
    try:
        rec["prompt_ids"] = render(messages)
        if len(rec["prompt_ids"]) - OPENER_TOKENS != rec["prompt_tokens"]:
            rec["refused"].append(
                f"render is {len(rec['prompt_ids'])} tokens ({OPENER_TOKENS} of them the channel opener "
                f"the server does not count), server prompt_tokens {rec['prompt_tokens']}")
    except EvalError as e:
        rec["prompt_ids"] = None
        rec["refused"].append(f"render: {e}")
    content = (choice.get("logprobs") or {}).get("content")
    if not content:
        rec["refused"].append("no logprobs content")
        return rec
    for pos, c in enumerate(content):
        try:
            top = [{"id": vocab.id_of(t.get("bytes"), t.get("token")), "lp": t["logprob"]}
                   for t in c.get("top_logprobs") or [] if t["logprob"] > FILLER_LP]
            rec["tokens"].append({"id": vocab.id_of(c.get("bytes"), c.get("token")), "token": c["token"],
                                  "lp": c["logprob"], "top": top})
        except EvalError as e:
            # A top-20 entry the API gives no bytes for (a partial UTF-8 piece) cannot be placed, and
            # dropping it would leave the set short of its likeliest token: the completion is cut
            # before this position (the positions before it are the teacher's own).
            if pos == 0:
                rec["refused"].append(f"position 0: {e}")
            else:
                rec["cut"] = f"{pos} of {len(content)} positions kept; position {pos}: {e}"
            break
    return rec


def read_jsonl(path):
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except OSError as e:
        raise EvalError(f"{path}: {e.strerror}") from e
    except ValueError as e:
        raise EvalError(f"{path}: not JSON lines: {e}") from e


def run_teacher(prompts, out, budget, post, vocab, render, sleep=time.sleep, log=print):
    """Answer each prompt not already in `out`, appending one line per prompt; stop cleanly when the
    recorded spend reaches `budget`. Returns the summary line's fields."""
    done = read_jsonl(out) if os.path.exists(out) else []
    seen = {r["id"] for r in done}
    spent = sum(r.get("cost") or 0 for r in done)
    stopped = None
    for p in prompts:
        if p["id"] in seen:
            continue
        if spent >= budget:
            stopped = f"budget ${budget:.2f} reached"
            break
        try:
            data = call(post, request_body([{"role": "user", "content": p["prompt"]}]), sleep)
        except EvalError as e:
            stopped = f"{p['id']}: {e}"
            break
        rec = teacher_record(p, data, vocab, render)
        spent += rec["cost"]
        done.append(rec)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
        if rec["refused"]:
            log(f"refused {p['id']}: {'; '.join(rec['refused'])}")
        if rec.get("cut"):
            log(f"cut {p['id']}: {rec['cut']}")
    ok = sum(1 for r in done if not r["refused"])
    return {"total": len(prompts), "done": len(done), "cost": spent, "ok": ok,
            "providers": sorted({r["provider"] for r in done}), "stopped": stopped}


def teacher_main(a):
    prompts = read_jsonl(a.prompts)
    vocab = Vocab.load(a.k31_folder)
    render = load_renderer(a.k31_folder, vocab)
    if a.limit:
        prompts = prompts[:a.limit]
    s = run_teacher(prompts, a.out, a.budget_usd, http_transport(read_key()), vocab, render)
    print(f"teacher: prompts done {s['done']}/{s['total']}, cost ${s['cost']:.4f}, "
          f"provider {','.join(s['providers']) or '-'}, prompt_tokens check passed {s['ok']}/{s['done']}"
          + (f", stopped: {s['stopped']}" if s["stopped"] else ""))
    return 0 if s["done"] == s["total"] else 2


def student_input(teacher, out):
    """One line per scorable prompt: key, prompt ids, completion ids, each position's teacher top ids."""
    n = 0
    lines = []
    for r in teacher:
        if r["refused"] or not r["tokens"]:
            continue
        extra = "|".join(",".join(str(t["id"]) for t in tok["top"]) for tok in r["tokens"])
        lines.append("\t".join([r["id"], ",".join(map(str, r["prompt_ids"])),
                                ",".join(str(t["id"]) for t in r["tokens"]), extra]))
        n += 1
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return n


# ---------------------------------------------------------------- score

_T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
      10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t95(df):
    """Two-sided 95 % Student t for `df`, taken from the table at the nearest df below (never smaller)."""
    below = [k for k in _T if k <= df]
    return _T[max(below)] if below else float("nan") if df < 1 else 1.96


def mean_ci(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, (t95(n - 1) if n - 1 <= 120 else 1.96) * sd / math.sqrt(n)


def kl_over(p_lp, q_lp):
    """KL(p || q) over the keys of `p_lp`, both renormalized to that set (`q_lp` covers every key)."""
    zp = math.log(sum(math.exp(v) for v in p_lp.values()))
    zq = math.log(sum(math.exp(q_lp[k]) for k in p_lp))
    return sum(math.exp(v - zp) * ((v - zp) - (q_lp[k] - zq)) for k, v in p_lp.items())


def position_metrics(tok, st):
    """The three measures of one position (`tok` the teacher's entry, `st` the student's line)."""
    s_top = {i: lp for i, lp in st["top"]}
    s_at = {i: lp for i, lp in st["at"]}
    s_all = {**s_top, **s_at}
    t_lp = {t["id"]: t["lp"] for t in tok["top"]}
    t_lp.setdefault(tok["id"], tok["lp"])
    missing = [i for i in t_lp if i not in s_all]
    if missing:
        raise EvalError(f"{st['key']} position {st['pos']}: no student log-probability for ids {missing[:5]}")
    if st["teacher"] != tok["id"]:
        raise EvalError(f"{st['key']} position {st['pos']}: student forced id {st['teacher']}, teacher's is {tok['id']}")
    floor = min(t_lp.values())  # the teacher's lowest listed log-probability bounds a token it left out
    union_t = dict(t_lp)
    for i in s_top:
        union_t.setdefault(i, floor)
    return {
        "agree": 1.0 if st["top"][0][0] == tok["id"] else 0.0,
        "nll_diff": tok["lp"] - st["lp_teacher"],
        "kl_top": kl_over(t_lp, s_all),
        "kl_union": kl_over(union_t, s_all),
    }


def score(teacher, student_lines):
    """Per-prompt means of every measure; returns (rows, missing keys)."""
    by_key = {}
    for s in student_lines:
        by_key.setdefault(s["key"], {})[s["pos"]] = s
    rows, missing = [], []
    for r in teacher:
        if r["refused"] or not r["tokens"]:
            continue
        got = by_key.get(r["id"])
        if not got or len(got) != len(r["tokens"]):
            missing.append(r["id"])
            continue
        ms = [position_metrics(tok, got[i]) for i, tok in enumerate(r["tokens"])]
        row = {k: sum(m[k] for m in ms) / len(ms) for k in ms[0]}
        row.update(id=r["id"], cat=r["cat"], positions=len(ms))
        rows.append(row)
    return rows, missing


def report(rows, missing, out=print):
    out("KL(top) = truncated KL(teacher||student) over the teacher's top-20 set, both renormalized to it.")
    out("KL(union) = the same over the union of both top-20 sets; the teacher's log-probability for a token "
        "only the student lists is taken as its own lowest listed one (an upper bound, derived).")
    out("Means are over prompts (each prompt's positions averaged first); ± is the 95 % Student-t interval.")
    out(f"{'set':8}{'prompts':>8}{'positions':>10}{'top1 agree':>18}{'dNLL (nats)':>18}{'KL top':>18}{'KL union':>18}")
    for name in (*CATEGORIES, "all"):
        sel = rows if name == "all" else [r for r in rows if r["cat"] == name]
        if not sel:
            continue
        cells = []
        for k in ("agree", "nll_diff", "kl_top", "kl_union"):
            m, ci = mean_ci([r[k] for r in sel])
            cells.append(f"{m:10.4f} ±{ci:6.4f}" if ci == ci else f"{m:10.4f}   n/a ")
        out(f"{name:8}{len(sel):8}{sum(r['positions'] for r in sel):10}" + "".join(f"{c:>18}" for c in cells))
    out("10 worst prompts by mean KL(top):")
    for r in sorted(rows, key=lambda r: -r["kl_top"])[:10]:
        out(f"  {r['id']:10} KL top {r['kl_top']:.4f}  union {r['kl_union']:.4f}  agree {r['agree']:.3f}  "
            f"dNLL {r['nll_diff']:+.4f}  positions {r['positions']}")
    if missing:
        out(f"{len(missing)} prompts with no (or partial) student lines, not scored: {', '.join(missing[:20])}")


def score_main(a):
    teacher = read_jsonl(a.teacher)
    with open(a.student, encoding="utf-8") as f:
        student = [json.loads(l) for l in f if l.startswith('{"key"')]
    rows, missing = score(teacher, student)
    if not rows:
        raise EvalError("no prompt has a complete student run")
    report(rows, missing)
    return 0


# ---------------------------------------------------------------- self-test


def _fake_vocab():
    ranks = {bytes([i]): i for i in range(256)}
    ranks[b"ab"] = 256
    ranks[b"abc"] = 257
    ranks[b" w"] = 258
    return Vocab(ranks, {"<|x|>": 300})


def self_test():
    # bytes -> id and its refusal
    v = _fake_vocab()
    assert v.id_of(b"a") == 97 and v.id_of([97, 98]) == 256 and v.id_of(b"<|x|>") == 300
    assert v.id_of(None, "<|x|>") == 300
    try:
        v.id_of(None, "nope")
    except EvalError as e:
        assert "no bytes" in str(e)
    else:
        raise AssertionError("bytes-less token mapped")
    for bad in (b"zz", b"", b"abcd"):
        try:
            v.id_of(bad)
        except EvalError as e:
            assert "not one vocabulary entry" in str(e)
        else:
            raise AssertionError(f"{bad!r} mapped")
    # split pattern and merging: letters with their leading space, digits in threes, Han alone
    assert [m.group() for m in pretokenizer().finditer("Hello world 12345")] == ["Hello", " world", " ", "123", "45"]
    assert [m.group() for m in pretokenizer().finditer("한국어 text中文x")] == ["한국어", " text", "中文", "x"]
    assert v.encode_text("abc") == [257] and v.encode_text("abab") == [256, 256]
    assert v.encode_segment("<|x|>ab<|x|>", True) == [300, 256, 300]
    assert v.encode_segment("<|x|>", False) == [60, 124, 120, 124, 62]

    # score math: p = (.5, .5), q = (.9, .1): KL = .5 ln(.5/.9) + .5 ln(.5/.1)
    kl = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(5)
    assert abs(kl_over({1: math.log(.5), 2: math.log(.5)}, {1: math.log(.9), 2: math.log(.1), 3: math.log(.7)}) - kl) < 1e-12
    assert abs(kl - 0.510826) < 1e-6
    tok = {"id": 1, "lp": math.log(.5), "top": [{"id": 1, "lp": math.log(.5)}, {"id": 2, "lp": math.log(.5)}]}
    st = {"key": "k", "pos": 0, "teacher": 1, "lp_teacher": math.log(.9),
          "top": [[1, math.log(.9)], [3, math.log(.05)]], "at": [[1, math.log(.9)], [2, math.log(.1)]]}
    m = position_metrics(tok, st)
    assert m["agree"] == 1.0 and abs(m["kl_top"] - kl) < 1e-12
    assert abs(m["nll_diff"] - (math.log(.5) - math.log(.9))) < 1e-12
    assert m["kl_union"] > 0
    st2 = dict(st, top=[[2, math.log(.9)]])
    assert position_metrics(tok, st2)["agree"] == 0.0
    m, ci = mean_ci([1.0, 2.0, 3.0])
    assert m == 2.0 and abs(ci - 4.303 / math.sqrt(3)) < 1e-9

    # teacher with a fake transport
    key = "sk-or-SELFTEST-KEY-12345"
    prompts = [{"id": f"en-{i}", "cat": "en", "prompt": f"p{i}"} for i in range(4)]
    render = lambda msgs: [1, 2, 3, 4, 5, 6]
    calls = {"n": 0, "script": []}

    def reply(cost=0.4, pt=3, provider=PROVIDER, bytes_=b"a"):
        return 200, {"provider": provider, "usage": {"prompt_tokens": pt, "cost": cost,
                                                     "completion_tokens_details": {"reasoning_tokens": 0}},
                     "choices": [{"finish_reason": "stop", "message": {"content": "a"},
                                  "logprobs": {"content": [{
                                      "token": "a", "bytes": list(bytes_), "logprob": -0.1,
                                      "top_logprobs": [{"token": "a", "bytes": list(bytes_), "logprob": -0.1},
                                                       {"token": "ab", "bytes": [97, 98], "logprob": -2.5},
                                                       {"token": "z", "bytes": [122], "logprob": -9999}]}]}}]}

    def post(body):
        calls["n"] += 1
        assert body["model"] == MODEL and body["reasoning"] == {"enabled": False}
        assert body["provider"] == {"order": [PROVIDER], "allow_fallbacks": False}
        assert body["temperature"] == 0 and body["max_tokens"] == 256 and body["top_logprobs"] == 20
        return calls["script"].pop(0) if calls["script"] else reply()

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "t.jsonl")
        logs = io.StringIO()
        with contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
            # budget stop: 0.4 a prompt against 0.9 stops after three calls (spend 1.2 >= 0.9) and keeps them
            s1 = run_teacher(prompts, out, 0.9, post, v, render, sleep=lambda s: None)
            assert (s1["done"], s1["stopped"]) == (3, "budget $0.90 reached") and calls["n"] == 3
            # resume: the three finished prompts are skipped, the fourth runs
            s2 = run_teacher(prompts, out, 5.0, post, v, render, sleep=lambda s: None)
            assert s2["done"] == 4 and calls["n"] == 4 and s2["stopped"] is None
            recs = read_jsonl(out)
            assert [r["id"] for r in recs] == [p["id"] for p in prompts]
            assert recs[0]["tokens"][0]["id"] == 97 and recs[0]["tokens"][0]["top"] == [
                {"id": 97, "lp": -0.1}, {"id": 256, "lp": -2.5}]  # the -9999 filler entry is dropped
            assert all(not r["refused"] for r in recs)
            # retry: 429, 503 then ok; a transport failure (599) too
            calls["script"] = [(429, {}), (503, {}), (599, {}), reply()]
            n0 = calls["n"]
            assert call(post, request_body([]), lambda s: None)["provider"] == PROVIDER and calls["n"] == n0 + 4
            # five 429s: stops by name after RETRIES retries
            calls["script"] = [(429, {})] * 5
            try:
                call(post, request_body([]), lambda s: None)
            except EvalError as e:
                assert "after 4 retries" in str(e)
            else:
                raise AssertionError("no stop")
            # another 4xx is not retried and stops the run, keeping what was written
            calls["script"] = [(400, {"error": {"message": "bad request"}})]
            out2 = os.path.join(d, "t2.jsonl")
            s3 = run_teacher(prompts, out2, 5.0, post, v, render, sleep=lambda s: None)
            assert s3["done"] == 0 and "HTTP 400 (not retried)" in s3["stopped"]
            # a render of another length, and a token that is no vocabulary entry, are refused and listed
            out3 = os.path.join(d, "t3.jsonl")
            calls["script"] = [reply(pt=9), reply(bytes_=b"zz")]
            s4 = run_teacher(prompts[:2], out3, 5.0, post, v, render, sleep=lambda s: None)
            r3 = read_jsonl(out3)
            assert s4["ok"] == 0 and "render is 6 tokens (3 of them the channel opener the server does not count), server prompt_tokens 9" in r3[0]["refused"][0]
            assert "not one vocabulary entry" in r3[1]["refused"][0]
            two = reply()[1]
            two["choices"][0]["logprobs"]["content"].append(
                {"token": "", "bytes": None, "logprob": -1.0, "top_logprobs": []})
            rec_cut = teacher_record(prompts[0], two, v, render)
            assert not rec_cut["refused"] and len(rec_cut["tokens"]) == 1 and "1 of 2 positions kept" in rec_cut["cut"]
            assert student_input(r3, os.path.join(d, "s.tsv")) == 0
            n_ok = student_input(recs, os.path.join(d, "s.tsv"))
            line = open(os.path.join(d, "s.tsv"), encoding="utf-8").readline().rstrip("\n").split("\t")
            assert n_ok == 4 and line == ["en-0", "1,2,3,4,5,6", "97", "97,256"]
            # a provider other than Moonshot AI stops by name
            calls["script"] = [reply(provider="Other")]
            try:
                run_teacher(prompts[:1], os.path.join(d, "t4.jsonl"), 5.0, post, v, render, sleep=lambda s: None)
            except EvalError as e:
                assert "not 'Moonshot AI'" in str(e)
            else:
                raise AssertionError("provider accepted")
        # the key is in no output and no stored line
        assert key not in logs.getvalue()
        for name in os.listdir(d):
            assert key not in open(os.path.join(d, name), encoding="utf-8").read()
        # key reading: a credentials file with the key, never echoed by an error
        home = os.path.join(d, "home")
        os.makedirs(home + "/.config/outsource")
        with open(home + "/.config/outsource/credentials", "w") as f:
            f.write(f"OTHER=1\nOPENROUTER_API_KEY={key}\n")
        old = os.environ.get("HOME")
        os.environ["HOME"] = home
        try:
            assert read_key() == key
            os.remove(home + "/.config/outsource/credentials")
            try:
                read_key()
            except EvalError as e:
                assert key not in str(e)
        finally:
            os.environ["HOME"] = old if old is not None else ""

        # score end to end on a two-position prompt
        teacher = [{"id": "ko-1", "cat": "ko", "refused": [], "prompt_ids": [1], "tokens": [
            {"id": 1, "lp": math.log(.5), "top": [{"id": 1, "lp": math.log(.5)}, {"id": 2, "lp": math.log(.5)}]},
            {"id": 2, "lp": math.log(.5), "top": [{"id": 1, "lp": math.log(.5)}, {"id": 2, "lp": math.log(.5)}]}]}]
        lines = [dict(st, key="ko-1", pos=0),
                 dict(st, key="ko-1", pos=1, teacher=2, lp_teacher=math.log(.1), top=[[1, math.log(.9)]])]
        rows, missing = score(teacher, lines)
        assert missing == [] and abs(rows[0]["kl_top"] - kl) < 1e-12 and rows[0]["agree"] == 0.5
        rows2, missing2 = score(teacher, lines[:1])
        assert rows2 == [] and missing2 == ["ko-1"]
        buf = []
        report(rows, [], out=buf.append)
        assert any(l.startswith("ko ") for l in buf) and any("worst prompts" in l for l in buf)
    print("k31-eval self-test: ok")


# ---------------------------------------------------------------- main


def main(argv):
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teacher")
    t.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "k31-eval-prompts.jsonl"))
    t.add_argument("--out", required=True)
    t.add_argument("--k31-folder", required=True)
    t.add_argument("--budget-usd", type=float, default=2.50)
    t.add_argument("--limit", type=int, default=0, help="the first N prompts only")
    i = sub.add_parser("ids")
    i.add_argument("--teacher", required=True)
    i.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--teacher", required=True)
    s.add_argument("--student", required=True)
    a = ap.parse_args(argv[1:])
    try:
        if a.cmd == "teacher":
            return teacher_main(a)
        if a.cmd == "ids":
            n = student_input(read_jsonl(a.teacher), a.out)
            print(f"ids: {n} prompts written to {a.out}")
            return 0
        return score_main(a)
    except EvalError as e:
        print(f"k31-eval: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
