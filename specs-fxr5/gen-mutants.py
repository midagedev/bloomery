#!/usr/bin/env python3
"""Regenerate the four fxr5 mutant patches from the pristine engine sources into specs-fxr5/mutants-regen/,
then compare each with specs-fxr5/mutants/ byte for byte. Reads only; writes only under specs-fxr5/."""
import difflib, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "specs-fxr5" / "mutants-regen"
OUT.mkdir(exist_ok=True)


def read(rel):
    return (ROOT / rel).read_text().split("\n")


def unified(rel, old, new):
    return "".join(
        difflib.unified_diff(
            [l + "\n" for l in old], [l + "\n" for l in new], f"a/{rel}", f"b/{rel}", n=3
        )
    )


def line_of(lines, needle, nth=0):
    hits = [i for i, l in enumerate(lines) if needle in l]
    return hits[nth]


def one(lines, needle, repl):
    """Replace the one line containing needle."""
    i = line_of(lines, needle)
    assert sum(needle in l for l in lines) == 1, f"{needle!r} is not unique"
    out = list(lines)
    out[i] = lines[i].replace(needle, repl)
    return out


# m1: the PLE site's conv is not launched (steps and eager pass; ubatch walk)
p38 = "crates/gpu/src/arch/qwen3moe/program38.rs"
a = read(p38)
i = line_of(a, "ple_conv(c, pp, &mut self.seqs, s, (self.cur, m), sink)?;")
b = a[:i] + ["            if false {", "    " + a[i], "            }"] + a[i + 1 :]
m1 = unified(p38, a, b)

w38 = "crates/gpu/src/arch/qwen3moe/wide38.rs"
a = read(w38)
i = line_of(a, "c.k.ple.enqueue_conv(")
indent = len(a[i]) - len(a[i].lstrip())
j = i
while not (a[j].startswith(" " * indent + ")?;")):
    j += 1
b = a[:i] + [" " * indent + "if false {"] + ["    " + l if l else l for l in a[i : j + 1]] + [" " * indent + "}"] + a[j + 1 :]
m1 += unified(w38, a, b)

# m2: the ubatch walk's route-tap logits read from row (t + 1) mod m
b = one(a, "logits: lg[t * taps.width..][..geo::EXPERTS].to_vec(),",
        "logits: lg[((t + 1) % m) * taps.width..][..geo::EXPERTS].to_vec(),")
m2 = unified(w38, a, b)

# m3: layer 0's delta step reads its state as zero at each launch
flt = "crates/gpu/src/fault.rs"
f = read(flt)
i = line_of(f, "pub fn read(self) -> u32")
k = i
while f[k - 1].lstrip().startswith("///") or f[k - 1].lstrip().startswith("#["):
    k -= 1
g = f[:k] + ["    #[inline(always)]", "    pub fn layer_of(self) -> u32 {", "        self.layer", "    }", ""] + f[k:]
dl = "crates/gpu/src/linear/delta.rs"
d = read(dl)
hits = [i for i, l in enumerate(d) if "let mut s = unsafe { lane_keys::<false>(state_g + 4 * s_at as u64, j) };" in l]
assert len(hits) == 2, hits
i = hits[1]  # the second is delta_lanes_body's
d2 = d[: i + 1] + ["    if fault.layer_of() == 0 {", "        s = [0.0f32; KEYS_PER_LANE];", "    }"] + d[i + 1 :]
m3 = unified(flt, f, g) + unified(dl, d, d2)

# m4: mtp_input writes [h_s | e] in place of [e | h_s]
mt = "crates/gpu/src/mtp.rs"
a = read(mt)
b = one(a, "let base = t * STREAMS * pitch + i;", "let base = t * STREAMS * pitch + hid + i;")
b = one(b, "*out.get_unchecked_mut((t * STREAMS + s) * pitch + hid + i) = v;",
        "*out.get_unchecked_mut((t * STREAMS + s) * pitch + i) = v;")
m4 = unified(mt, a, b)

names = {
    "m1-ple-skipped.patch": m1,
    "m2-logits-tap-row.patch": m2,
    "m3-gdn-carry.patch": m3,
    "m4-eh-halves.patch": m4,
}
bad = 0
for n, text in names.items():
    (OUT / n).write_text(text)
    have = (ROOT / "specs-fxr5" / "mutants" / n).read_text()
    same = have == text
    bad += not same
    print(f"{n}: regenerated {len(text.splitlines())} lines, byte-equal to mutants/: {same}")
sys.exit(1 if bad else 0)
