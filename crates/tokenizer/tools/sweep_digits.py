#!/usr/bin/env python3
"""Codepoint sweep of the two digit classes ERNIE's tokenizer.json runs, read off tokenizers itself.

    sweep_digits.py
    sweep_digits.py --self-test

NUMERIC: the codepoints the `Digits(individual_digits=false)` step cuts out (Rust `char::is_numeric`).
DECIMAL: the codepoints `\\p{Nd}` matches (Oniguruma's table).
Prints each as ranges, one `first last` pair a line, and the counts.

`--self-test` runs the sweep and the range folding on a stand-in pre-tokenizer over a small codepoint
range (no `tokenizers` package).
"""
import sys


def sweep(pt, probe, codepoints):
    """The codepoints `c` of `codepoints` for which `pt` cuts `a{c}a` as `probe(c)`."""
    out = []
    for cp in codepoints:
        if 0xD800 <= cp < 0xE000:
            continue
        c = chr(cp)
        pieces = [p for p, _ in pt.pre_tokenize_str(f"a{c}a")]
        if pieces == probe(c):
            out.append(cp)
    return out


def ranges(cps):
    """`cps`, ascending, as `[first, last]` runs of consecutive codepoints."""
    runs = []
    for cp in cps:
        if runs and runs[-1][1] + 1 == cp:
            runs[-1][1] = cp
        else:
            runs.append([cp, cp])
    return runs


def main():
    import tokenizers
    from tokenizers import Regex, pre_tokenizers

    digits = pre_tokenizers.Digits(individual_digits=False)
    nd = pre_tokenizers.Split(Regex(r"\p{Nd}"), "isolated")
    probe = lambda c: ["a", c, "a"]
    everything = range(0x110000)
    numeric = sweep(digits, probe, everything)
    decimal = sweep(nd, probe, everything)
    print("tokenizers", tokenizers.__version__, file=sys.stderr)
    for name, cps in (("NUMERIC", numeric), ("DECIMAL", decimal)):
        r = ranges(cps)
        print(f"{name} {len(cps)} codepoints in {len(r)} runs")
        for a, b in r:
            print(f"{a:06X} {b:06X}")


class _Digits:
    """Cuts a text into its runs of ASCII digits and the rest, as `Digits(individual_digits=false)` does
    for the one class it knows."""

    def pre_tokenize_str(self, text):
        text.encode("utf-8")  # a lone surrogate cannot cross into the real library either
        pieces, cur = [], ""
        for ch in text:
            if cur and ch.isdigit() != cur[-1].isdigit():
                pieces.append(cur)
                cur = ""
            cur += ch
        pieces.append(cur)
        return [(p, (0, 0)) for p in pieces]


def self_test():
    assert ranges([]) == []
    assert ranges([5]) == [[5, 5]]
    assert ranges([1, 2, 3, 5, 7, 8]) == [[1, 3], [5, 5], [7, 8]]
    # Over the ASCII range the stand-in finds the ten digits, one run; the surrogates are skipped.
    got = sweep(_Digits(), lambda c: ["a", c, "a"], range(0x80))
    assert got == list(range(0x30, 0x3A)), got
    assert ranges(got) == [[0x30, 0x39]]
    assert sweep(_Digits(), lambda c: ["a", c, "a"], range(0xD7FF, 0xE001)) == [], "surrogates are skipped"
    try:
        _Digits().pre_tokenize_str("a\ud800a")
    except UnicodeEncodeError:
        pass
    else:
        raise SystemExit("the stand-in accepted a lone surrogate")
    print("sweep_digits: self-test ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit(__doc__)
