#!/usr/bin/env bash
# 호스트 rustflags의 소유자는 둘이다: plain cargo는 .cargo/config.toml의 target.<triple>.rustflags를,
# cargo oxide는 .cargo/cuda-oxide.toml의 [target.<triple>] extra-rustflags를 읽는다(cargo-oxide가 내보내는
# CARGO_ENCODED_RUSTFLAGS가 config의 표를 가린다). 트리플마다 둘이 같지 않으면 oxide 바이너리(generate_ds41,
# GPU 게이트 전부)의 호스트 코드가 다른 CPU로 컴파일된다. 맥에서 돈다(파싱뿐, 빌드 없음).
#   tools/check-rustflags.sh [--self-test]
# 매 실행이 먼저 자기 시험(아래 cases: 옛 평면 목록, aarch64의 znver3, 한쪽 누락 등이 각각 빨강)을 조용히 돌린다;
# --self-test는 그 결과를 줄마다 찍고 끝난다.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - "$@" <<'PY'
import os, sys, tomllib

X86 = "x86_64-unknown-linux-gnu"
ARM = "aarch64-unknown-linux-gnu"
CPU_KEYS = ("target-cpu=", "target-feature=")
# aarch64 CPU names the aarch64 table may hold; an x86 name there is a warning on every crate and a SIGSEGV
# inside LLVM on some.
ARM_CPUS = ("native", "generic")
ARM_CPU_PREFIXES = ("cortex-", "neoverse-", "apple-")


def codegen(flags):
    """The -C options of one rustflags list, `-C x`, `-Cx`, `--codegen x` and `--codegen=x` alike."""
    out, it = [], iter(flags or [])
    for f in it:
        if f in ("-C", "--codegen"):
            out.append(next(it, ""))
        elif f.startswith("--codegen="):
            out.append(f[len("--codegen="):])
        elif f.startswith("-C"):
            out.append(f[2:])
    # rustc takes `target_cpu` for `target-cpu`: the key is read with either separator.
    keyed = (o.partition("=") for o in out)
    return [key.replace("_", "-") + sep + value for key, sep, value in keyed]


def check(cfg_text, ox_text):
    """The named failures of one pair of files, [] when they agree."""
    cfg = tomllib.loads(cfg_text)
    ox = tomllib.loads(ox_text)
    bad = []
    flat = ox.get("extra-rustflags")
    if flat is not None and any(f.startswith(CPU_KEYS) for f in codegen(flat)):
        bad.append(f"cuda-oxide.toml has a top-level extra-rustflags with a CPU flag, which reaches every triple: {flat!r}")
    plain_t = {t: v.get("rustflags") for t, v in cfg.get("target", {}).items() if "rustflags" in v}
    oxide_t = {t: v.get("extra-rustflags") for t, v in ox.get("target", {}).items()}
    for t in (X86, ARM):
        if t not in plain_t or t not in oxide_t:
            bad.append(f"{t}: a side is missing — config.toml target rustflags {plain_t.get(t)!r}, "
                       f"cuda-oxide.toml [target.{t}] extra-rustflags {oxide_t.get(t)!r}")
    for t in sorted(set(plain_t) | set(oxide_t)):
        if t in plain_t and t in oxide_t and plain_t[t] != oxide_t[t]:
            bad.append(f"{t}: the two owners disagree — config.toml {plain_t[t]!r} vs cuda-oxide.toml {oxide_t[t]!r}")
        elif (t in plain_t) != (t in oxide_t) and t not in (X86, ARM):
            bad.append(f"{t}: only one owner names this triple — config.toml {plain_t.get(t)!r}, cuda-oxide.toml {oxide_t.get(t)!r}")
    x86 = codegen(oxide_t.get(X86))
    if X86 in oxide_t and "target-cpu=znver3" not in x86:
        bad.append(f"{X86}: target-cpu=znver3 is not in the flags: {x86!r}")
    # znver3 turns on sse4a, and LLVM then lowers some byte shuffles to EXTRQ/INSERTQ, which have no VEX
    # form: inside AVX code each one takes a ymm spill fault, then fill faults, on Zen 3.
    if X86 in oxide_t and "target-feature=-sse4a" not in x86:
        bad.append(f"{X86}: target-feature=-sse4a is not in the flags: {x86!r} — LLVM would emit "
                   "legacy-encoded EXTRQ/INSERTQ inside AVX kernels")
    # aarch64 takes only an aarch64 CPU name and no target-feature: an x86 name there is a warning on every
    # crate and a SIGSEGV inside LLVM on some, and no aarch64 feature is set today.
    arm = codegen(oxide_t.get(ARM))
    foreign = [f for f in arm if f.startswith("target-feature=")
               or (f.startswith("target-cpu=")
                   and f[len("target-cpu="):] not in ARM_CPUS
                   and not f[len("target-cpu="):].startswith(ARM_CPU_PREFIXES))]
    if foreign:
        bad.append(f"{ARM}: CPU flags that are not an aarch64 CPU name: {foreign!r}")
    return bad


def self_test(verbose):
    good_cfg = (f'[target.{X86}]\nrustflags = ["-C", "target-cpu=znver3", "-C", "target-feature=-sse4a"]\n'
                f'[target.{ARM}]\nrustflags = ["-C", "target-cpu=native"]\n')
    good_ox = (f'[target.{X86}]\nextra-rustflags = ["-C", "target-cpu=znver3", "-C", "target-feature=-sse4a"]\n'
               f'[target.{ARM}]\nextra-rustflags = ["-C", "target-cpu=native"]\n')
    cases = [
        ("good", good_cfg, good_ox, None),
        ("flat oxide list (the old shape)", good_cfg,
         'extra-rustflags = ["-C", "target-cpu=znver3", "-C", "target-feature=-sse4a"]\n', "top-level extra-rustflags"),
        ("aarch64 inherits znver3", good_cfg.replace("target-cpu=native", "target-cpu=znver3"),
         good_ox.replace("target-cpu=native", "target-cpu=znver3"), "not an aarch64 CPU name"),
        ("aarch64 owners disagree", good_cfg, good_ox.replace('"target-cpu=native"', '"target-cpu=cortex-x925"'),
         "the two owners disagree"),
        ("aarch64 missing in oxide", good_cfg, good_ox.split(f"[target.{ARM}]")[0], "a side is missing"),
        ("x86 lost sse4a", good_cfg.replace(', "-C", "target-feature=-sse4a"', ""),
         good_ox.replace(', "-C", "target-feature=-sse4a"', ""), "target-feature=-sse4a is not in"),
        ("x86 lost znver3", good_cfg.replace("znver3", "x86-64-v3"), good_ox.replace("znver3", "x86-64-v3"),
         "target-cpu=znver3 is not in"),
        ("flat list joined (-Ctarget-cpu=)", good_cfg, 'extra-rustflags = ["-Ctarget-cpu=znver3"]\n' + good_ox,
         "top-level extra-rustflags"),
        ("aarch64 haswell", good_cfg.replace("target-cpu=native", "target-cpu=haswell"),
         good_ox.replace("target-cpu=native", "target-cpu=haswell"), "not an aarch64 CPU name"),
        ("aarch64 x86 feature", good_cfg.replace('"target-cpu=native"', '"target-cpu=native", "-C", "target-feature=+bmi2"'),
         good_ox.replace('"target-cpu=native"', '"target-cpu=native", "-C", "target-feature=+bmi2"'),
         "not an aarch64 CPU name"),
        ("aarch64 target_cpu spelling", good_cfg.replace('"target-cpu=native"', '"target_cpu=znver3"'),
         good_ox.replace('"target-cpu=native"', '"target_cpu=znver3"'), "not an aarch64 CPU name"),
        ("third triple on one side", good_cfg + '[target.riscv64gc-unknown-linux-gnu]\nrustflags = []\n', good_ox,
         "only one owner names this triple"),
    ]
    failed = 0
    for name, cfg, ox, want in cases:
        bad = check(cfg, ox)
        ok = (not bad) if want is None else any(want in b for b in bad)
        if verbose or not ok:
            print(f"check-rustflags self-test: {'ok ' if ok else 'BAD'} {name}: {bad or 'clean'}")
        failed += not ok
    return failed


if sys.argv[1:] == ["--self-test"]:
    sys.exit(1 if self_test(True) else 0)
elif sys.argv[1:]:
    print(f"check-rustflags: usage: tools/check-rustflags.sh [--self-test], got {sys.argv[1:]!r}")
    sys.exit(64)
if self_test(False):
    print("check-rustflags: the self-test is red — the check itself is wrong")
    sys.exit(1)
for f in (".cargo/config.toml", ".cargo/cuda-oxide.toml"):
    if not os.path.isfile(f):
        print(f"check-rustflags: {f} is missing — the oxide builds would compile their host code for a baseline CPU")
        sys.exit(1)
bad = check(open(".cargo/config.toml").read(), open(".cargo/cuda-oxide.toml").read())
for b in bad:
    print(f"check-rustflags: {b}")
if bad:
    sys.exit(1)
ox = tomllib.load(open(".cargo/cuda-oxide.toml", "rb"))
for t, v in sorted(ox["target"].items()):
    print(f"check-rustflags: ok — {t}: both owners carry {v['extra-rustflags']}")
PY
