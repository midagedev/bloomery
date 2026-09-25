#!/usr/bin/env bash
# 호스트 rustflags의 소유자는 둘이다: plain cargo는 .cargo/config.toml의 target.<triple>.rustflags를,
# cargo oxide는 .cargo/cuda-oxide.toml의 extra-rustflags를 읽는다(cargo-oxide가 내보내는
# CARGO_ENCODED_RUSTFLAGS가 config의 표를 가린다). 둘이 같지 않으면 oxide 바이너리(generate_ds41,
# GPU 게이트 전부)의 호스트 코드가 다른 CPU로 컴파일된다. 맥에서 돈다(파싱뿐, 빌드 없음).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
import sys, tomllib, os
for f in (".cargo/config.toml", ".cargo/cuda-oxide.toml"):
    if not os.path.isfile(f):
        print(f"check-rustflags: {f} is missing — the oxide builds would compile their host code for baseline x86-64")
        sys.exit(1)
cfg = tomllib.load(open(".cargo/config.toml", "rb"))
ox = tomllib.load(open(".cargo/cuda-oxide.toml", "rb"))
plain = cfg.get("target", {}).get("x86_64-unknown-linux-gnu", {}).get("rustflags")
oxide = ox.get("extra-rustflags")
if not plain or not oxide:
    print(f"check-rustflags: a side is missing — config.toml target rustflags {plain!r}, cuda-oxide.toml extra-rustflags {oxide!r}")
    sys.exit(1)
if plain != oxide:
    print(f"check-rustflags: the two owners disagree — config.toml {plain!r} vs cuda-oxide.toml {oxide!r}")
    sys.exit(1)
if "target-cpu=znver3" not in oxide:
    print(f"check-rustflags: target-cpu=znver3 is not in the flags: {oxide!r}")
    sys.exit(1)
print(f"check-rustflags: ok — both owners carry {oxide}")
PY
