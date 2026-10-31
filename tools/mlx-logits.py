#!/usr/bin/env python3
"""mlx-logits.py — mlx-lm's first-step logit rows of an MLX fixture folder, the Metal e2e pin.

    tools/mlx-logits.py dump <folder> <family> [--ids 0,1 | --ids-walk N:FIRST] [--positions P,P] [--out DIR]
    tools/mlx-logits.py layers <folder> <dir> [--split FIRST.gguf]
    tools/mlx-logits.py compare <ours-dir> <mlx-dir>
    tools/mlx-logits.py f64 <ours-dir> <mlx-dir> <split.gguf>
    tools/mlx-logits.py --self-test

`dump` loads `<folder>` with mlx-lm and runs one forward on token id 0 (more ids only when
`--ids` names them, one step each, teacher-forced). The forward is mlx-lm's modules in f32:
every quantized linear and embedding is replaced by its exact dequant (scales and biases
widened before `mx.dequantize`) and every bf16 parameter is widened, so the row carries the
model's arithmetic without bf16 activation rounding. The embedding is not floatized whole: only
the rows of the dumped ids are gathered from the quantized embedding and dequantized the same
way (the head stays whole in f32), and an id outside that set raises by name; a folder whose
head is tied to the embedding keeps the whole f32 embedding. A folder that does not floatize
fully is refused by name. It prints two blocks on stdout:

    ----- <family>.txt -----
    version 1
    family <family>
    ids <id>[,<id>...]
    steps <n>
    n <vocab>
    fnv1a <16 lowercase hex digits>

    ----- <family>.f32 -----
    <base64 of the f32 little-endian rows, steps * vocab values>

The checksum is FNV-1a 64-bit over those bytes (offset basis 14695981039346656037, prime
1099511628211), the same function the e2e gate recomputes. A flipped logit changes it.

`--ids-walk N:FIRST` is the deep sequence: N ids, FIRST and then `(i * 7919) % 90001 + 1000` for
i from 1, the walk `tools/metal-compare.sh` feeds its deep trace (ids from 1000 to 91000). `--positions
P,P,...` keeps only the rows of those steps (0-based, increasing, each below the id count): the f32
block then holds one row per position, the text adds a line `positions` and a line `argmax` (the
first index of the largest value of each kept row, in the same order), and the checksum is over those
rows. The gate's `m_logits_deep` reads such a pin from `pins/deep/`. `--out DIR` writes
`DIR/<family>.txt` and `DIR/<family>.f32` (the f32 bytes themselves, not base64) in place of the two
stdout blocks.

An LFM3 folder (`model_type` `lfm3`) loads through mlx-vlm's `lfm3` instead, since mlx-lm has none:
its language model alone runs, with the same floatize and the same two blocks. Before it loads, the
folder's `config.json` and safetensors headers are held to what the Rust folder map reads (MLX's
4-bit affine blocks, conv taps stored `[channels, kernel, 1]`, the tie flag against `lm_head`), so a
pin is never taken of a folder the map would refuse.

A Qwen3.6 fixture folder names each layer in `layer_types` and may still carry the source
model's `full_attention_interval`. mlx-lm builds the layer kind from that interval. When the
two disagree, the load uses the interval `layer_types` implies, the same one the split's
converter writes, so the forward is the folder's layers.

`--self-test` checks that checksum, a missing key, and that flipping the first logit moves
the max-abs distance, the loader chosen by `model_type`, the LFM3 folder's checks over a header-only
folder, and the logits taken from either module's output. No mlx and no model.

`layers` writes each layer's output residual and the final norm, plus the mixer inputs the
modules return, through mlx-lm's own modules. It then replaces each quantized linear and
embedding with its exact f32 dequant and widens every bf16 parameter (as `dump` does), and
writes that forward under `<dir>/f32` when a module was replaced. `compare` prints max abs
and rms of the f32 files both directories share. `f64` scores layer 0's input norm and its
in-projections from the split, in f64, against both dumps. None of these edit mlx-lm.
"""

from __future__ import annotations

import base64
import struct
import sys

USAGE = (
    "usage: tools/mlx-logits.py dump <folder> <family> [--ids 0,1 | --ids-walk N:FIRST]"
    " [--positions P,P] [--out DIR]"
    " | layers <folder> <dir> [--split FIRST.gguf]"
    " | compare <ours-dir> <mlx-dir>"
    " | f64 <ours-dir> <mlx-dir> <split.gguf> | --self-test"
)
OFFSET = 14695981039346656037
PRIME = 1099511628211
MASK = 0xFFFFFFFFFFFFFFFF


def fnv1a(data: bytes) -> int:
    h = OFFSET
    for b in data:
        h ^= b
        h = (h * PRIME) & MASK
    return h


def header(
    family: str,
    ids: list[int],
    steps: int,
    n: int,
    raw: bytes,
    positions: list[int] | None = None,
    argmax: list[int] | None = None,
) -> str:
    """The pin's text. Without `positions` the f32 block holds a row for each step; with them, one
    row for each position, in order, and `argmax` names each such row's first largest index."""
    if steps != len(ids):
        raise SystemExit(f"mlx-logits: steps {steps} and {len(ids)} ids differ")
    rows = steps if positions is None else len(positions)
    if positions is not None:
        check_positions(positions, steps)
    if argmax is not None and len(argmax) != rows:
        raise SystemExit(f"mlx-logits: {len(argmax)} argmax values for {rows} rows")
    if len(raw) != rows * n * 4:
        raise SystemExit(f"mlx-logits: {len(raw)} bytes is not {rows} * {n} f32s")
    ids_s = ",".join(str(i) for i in ids)
    text = (
        "version 1\n"
        f"family {family}\n"
        f"ids {ids_s}\n"
        f"steps {steps}\n"
        f"n {n}\n"
        f"fnv1a {fnv1a(raw):016x}\n"
    )
    if positions is not None:
        text += "positions " + ",".join(str(p) for p in positions) + "\n"
    if argmax is not None:
        text += "argmax " + ",".join(str(a) for a in argmax) + "\n"
    return text


def emit(family: str, text: str, raw: bytes, out: str | None = None) -> None:
    """The pin's two files: as stdout blocks (the f32 bytes in base64), or into the directory `out`."""
    if out is not None:
        import pathlib

        root = pathlib.Path(out)
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{family}.txt").write_text(text if text.endswith("\n") else text + "\n")
        (root / f"{family}.f32").write_bytes(raw)
        return
    print(f"----- {family}.txt -----")
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    print(f"----- {family}.f32 -----")
    print(base64.b64encode(raw).decode("ascii"))


WALK_STEP = 7919
WALK_MOD = 90001
WALK_LO = 1000


def walk_ids(count: int, first: int) -> list[int]:
    """`count` ids: `first`, then `(i * 7919) % 90001 + 1000` for i from 1, the walk
    tools/metal-compare.sh feeds its deep trace."""
    if count < 1:
        raise SystemExit(f"mlx-logits: a walk of {count} ids")
    return [first] + [(i * WALK_STEP) % WALK_MOD + WALK_LO for i in range(1, count)]


def parse_walk(text: str) -> list[int]:
    """`N:FIRST` as the ids of [`walk_ids`]."""
    count, sep, first = text.partition(":")
    if not sep or not count.isdigit() or not first.isdigit():
        raise SystemExit(f"mlx-logits: {text!r} is not N:FIRST, a count and a first id")
    return walk_ids(int(count), int(first))


def check_positions(positions: list[int], steps: int) -> None:
    """Positions are steps of the run: increasing, each below the step count, at least one."""
    if not positions:
        raise SystemExit("mlx-logits: the position list is empty")
    for a, b in zip(positions, positions[1:]):
        if b <= a:
            raise SystemExit(f"mlx-logits: positions {a},{b} are not increasing")
    if positions[-1] >= steps:
        raise SystemExit(f"mlx-logits: position {positions[-1]} is past the {steps} steps")


def parse_positions(text: str, steps: int) -> list[int]:
    out = []
    for part in text.split(","):
        if part == "" or not part.isdigit():
            raise SystemExit(f"mlx-logits: {text!r} is not a list of positions")
        out.append(int(part))
    check_positions(out, steps)
    return out


def parse_ids(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if part == "" or not part.isdigit():
            raise SystemExit(f"mlx-logits: {text!r} is not a list of token ids")
        out.append(int(part))
    if not out:
        raise SystemExit("mlx-logits: the id list is empty")
    return out


def max_abs(a: bytes, b: bytes) -> float:
    if len(a) != len(b) or len(a) % 4:
        raise SystemExit("mlx-logits: the two rows differ in length")
    worst = 0.0
    for i in range(0, len(a), 4):
        x = struct.unpack_from("<f", a, i)[0]
        y = struct.unpack_from("<f", b, i)[0]
        d = abs(x - y)
        if d != d:  # NaN
            return float("inf")
        if d > worst:
            worst = d
    return worst


def attention_interval(layer_types: list[str]) -> int | None:
    """mlx-lm's rule: layer i is full attention when `(i + 1) % interval == 0`.

    The same search the folder converter uses. None when no interval says the list.
    """
    for interval in range(1, len(layer_types) + 2):
        if all(
            ((i + 1) % interval == 0) == (kind == "full_attention")
            for i, kind in enumerate(layer_types)
        ):
            return interval
    return None


def load_folder(folder: str, load):
    """Load `folder`. A `layer_types` list that is not the file's interval is loaded at the
    interval the list implies, through a copy of the config beside symlinks of the weights.
    """
    import json
    import pathlib
    import tempfile

    root = pathlib.Path(folder)
    cfg = json.loads((root / "config.json").read_text())
    text = cfg.get("text_config")
    types = text.get("layer_types") if isinstance(text, dict) else None
    if not types:
        return load(folder)
    interval = attention_interval(types)
    if interval is None:
        raise SystemExit(
            "mlx-logits: layer_types is not one full-attention layer in every N, "
            "so mlx-lm cannot build it"
        )
    if text.get("full_attention_interval") == interval:
        return load(folder)
    print(
        "mlx-logits: full_attention_interval "
        f"{text.get('full_attention_interval')} disagrees with layer_types; "
        f"loading at {interval}",
        file=sys.stderr,
    )
    text["full_attention_interval"] = interval
    with tempfile.TemporaryDirectory() as tmp:
        view = pathlib.Path(tmp)
        for child in root.iterdir():
            if child.name == "config.json":
                continue
            (view / child.name).symlink_to(child)
        (view / "config.json").write_text(json.dumps(cfg))
        return load(str(view))


# The model types mlx-lm has no module for: the folder loads through mlx-vlm's module of the same
# name, and its language model alone runs.
VLM_TYPES = ("lfm3",)


def read_config(folder: str) -> dict:
    """`config.json` of `folder`, refused by name when it is not there or not an object."""
    import json
    import pathlib

    path = pathlib.Path(folder) / "config.json"
    if not path.is_file():
        raise SystemExit(f"mlx-logits: {path} is not there")
    cfg = json.loads(path.read_text())
    if not isinstance(cfg, dict):
        raise SystemExit(f"mlx-logits: {path} is not an object")
    return cfg


def model_type_of(folder: str) -> str:
    """`config.json`'s `model_type`, else its `text_config`'s; refused by name when it has none."""
    cfg = read_config(folder)
    text = cfg.get("text_config")
    kind = cfg.get("model_type") or (text.get("model_type") if isinstance(text, dict) else None)
    if not isinstance(kind, str) or not kind:
        raise SystemExit(f"mlx-logits: {folder}/config.json names no model_type")
    return kind


def loader_of(model_type: str) -> str:
    """Which library runs a folder of `model_type`: `mlx-vlm` for the types in `VLM_TYPES`."""
    return "mlx-vlm" if model_type in VLM_TYPES else "mlx-lm"


def safetensors_shapes(folder: str) -> dict:
    """Every tensor's shape in the folder's safetensors files, from their headers alone (the
    8-byte little-endian length, then the JSON): no weight is read."""
    import json
    import pathlib

    root = pathlib.Path(folder)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        names = sorted(p.name for p in root.glob("*.safetensors"))
    if not names:
        raise SystemExit(f"mlx-logits: {folder} holds no safetensors")
    shapes = {}
    for name in names:
        with open(root / name, "rb") as handle:
            (n,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(n))
        for key, entry in header.items():
            if key != "__metadata__":
                shapes[key] = list(entry["shape"])
    return shapes


def check_lfm3_folder(folder: str) -> None:
    """Hold an LFM3 folder to what the Rust folder map reads, by name: `model_type` `lfm3`, MLX's
    affine 4-bit blocks of group 64, every conv layer's taps stored `[channels, kernel, 1]` with
    `conv_L_cache` taps (mlx-vlm transposes only `[channels, 1, kernel]`), and the head tied to the
    embedding exactly when the folder holds no `lm_head`."""
    cfg = read_config(folder)
    if cfg.get("model_type") != "lfm3":
        raise SystemExit(f"mlx-logits: {folder}: model_type {cfg.get('model_type')!r} is not lfm3")
    quant = cfg.get("quantization")
    if not isinstance(quant, dict) or (
        quant.get("bits"),
        quant.get("group_size"),
        quant.get("mode", "affine"),
    ) != (4, 64, "affine"):
        raise SystemExit(
            f"mlx-logits: {folder}: quantization {quant!r} is not affine 4-bit of group 64"
        )
    kinds = cfg.get("layer_types")
    if not isinstance(kinds, list) or len(kinds) != cfg.get("num_hidden_layers"):
        raise SystemExit(f"mlx-logits: {folder}: layer_types does not list num_hidden_layers layers")
    hidden = cfg.get("hidden_size")
    taps = cfg.get("conv_L_cache")
    shapes = safetensors_shapes(folder)
    for layer, kind in enumerate(kinds):
        if kind != "conv":
            continue
        key = f"language_model.model.layers.{layer}.conv.conv.weight"
        if shapes.get(key) != [hidden, taps, 1]:
            raise SystemExit(
                f"mlx-logits: {key} is {shapes.get(key)}, not MLX's [channels, kernel, 1] "
                f"= [{hidden}, {taps}, 1]"
            )
    has_head = "language_model.lm_head.weight" in shapes
    tied = cfg.get("tie_word_embeddings")
    if not isinstance(tied, bool) or tied == has_head:
        raise SystemExit(
            f"mlx-logits: {folder}: tie_word_embeddings {tied!r} with "
            f"{'an' if has_head else 'no'} lm_head"
        )


def load_lfm3(folder: str):
    """The language model of an LFM3 folder through mlx-vlm's `lfm3` (mlx-lm has none): the
    folder is checked first ([`check_lfm3_folder`]), and the loaded module is the model's
    `language_model`, the same one the Rust side runs."""
    import pathlib

    import mlx_vlm
    from mlx_vlm.utils import load_model

    check_lfm3_folder(folder)
    model = load_model(pathlib.Path(folder), lazy=False)
    print(
        f"mlx-logits: {folder} loaded through mlx-vlm {getattr(mlx_vlm, '__version__', 'unknown')}'s lfm3",
        file=sys.stderr,
    )
    return model.language_model


def unwrap_logits(out):
    """The logits of a module's output: mlx-lm returns them (or a tuple that starts with them),
    mlx-vlm's language model an object that holds them as `logits`."""
    if isinstance(out, tuple):
        return out[0]
    return getattr(out, "logits", out)


def dump(
    folder: str,
    family: str,
    ids: list[int],
    positions: list[int] | None = None,
    out: str | None = None,
) -> None:
    import mlx.core as mx
    import numpy as np

    if loader_of(model_type_of(folder)) == "mlx-vlm":
        model = load_lfm3(folder)
        make_cache = model.make_cache
    else:
        from mlx_lm.models import cache as mlx_cache
        from mlx_lm.utils import load

        model, _tokenizer = load_folder(folder, load)

        def make_cache():
            return mlx_cache.make_prompt_cache(model)

    gather_embedding(model, ids)
    note, replaced = floatize(model)
    if replaced == 0 or "bf16 parameters left: 0" not in note:
        raise SystemExit(f"mlx-logits: {folder} does not run in f32:\n{note}")
    prompt_cache = make_cache()
    keep = None if positions is None else set(positions)
    rows = []
    argmax = []
    n = 0
    for step, token in enumerate(ids):
        y = mx.array([token])
        logits = unwrap_logits(model(y[None], cache=prompt_cache))
        logits = logits[:, -1, :]
        mx.eval(logits)
        row = np.ascontiguousarray(np.array(logits.astype(mx.float32)), dtype=np.float32)
        if not n:
            n = row.size
            if max(ids) >= n:
                raise SystemExit(f"mlx-logits: id {max(ids)} is outside the vocabulary of {n}")
        if not np.isfinite(row).all():
            raise SystemExit(f"mlx-logits: step {step} (id {token}) has a non-finite logit")
        if keep is None or step in keep:
            rows.append(row.tobytes())
            argmax.append(int(np.argmax(row)))
    raw = b"".join(rows)
    text = header(
        family, ids, len(ids), n, raw, positions, None if positions is None else argmax
    )
    emit(family, text, raw, out)
    if out is not None:
        print(f"mlx-logits: wrote {out}/{family}.txt and {out}/{family}.f32", file=sys.stderr)


def self_test() -> int:
    fails = 0

    def check(name: str, ok: bool) -> None:
        nonlocal fails
        if not ok:
            print(f"mlx-logits self-test FAIL: {name}", file=sys.stderr)
            fails += 1

    check("empty basis", fnv1a(b"") == OFFSET)
    one = struct.pack("<f", 1.0)
    h1 = fnv1a(one)
    # One more byte step from the basis, so a transcription of the prime fails this.
    h = OFFSET
    for b in one:
        h ^= b
        h = (h * PRIME) & MASK
    check("one f32", h1 == h and h1 != OFFSET)

    raw = struct.pack("<ff", 1.25, -3.5)
    text = header("qwen35moe", [0], 1, 2, raw)
    check("header keys", "fnv1a " in text and "ids 0\n" in text and text.startswith("version 1\n"))
    flipped = struct.pack("<ff", 1.25 + 1.0, -3.5)
    check("flip moves the checksum", fnv1a(raw) != fnv1a(flipped))
    check("flip max abs", max_abs(raw, flipped) == 1.0)
    check("equal max abs", max_abs(raw, raw) == 0.0)
    nan = struct.pack("<ff", float("nan"), 0.0)
    check("nan is not a distance", max_abs(raw, nan) == float("inf"))
    check(
        "interval alternating",
        attention_interval(["linear_attention", "full_attention"] * 2) == 2,
    )
    check(
        "interval every four",
        attention_interval(["linear_attention"] * 3 + ["full_attention"]) == 4,
    )
    check("interval all full", attention_interval(["full_attention"] * 3) == 1)
    check(
        "interval none",
        attention_interval(["linear_attention", "full_attention", "full_attention"]) is None,
    )
    try:
        header("qwen35moe", [0, 1], 1, 2, raw)
        check("steps match ids", False)
    except SystemExit:
        check("steps match ids", True)

    grouped = untile([0.0, 2.0, 1.0, 3.0], 4, 2, 1)
    check("untile restores grouped heads", grouped == [0.0, 1.0, 2.0, 3.0])
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        ours = root / "ours"
        mlx = root / "mlx"
        ours.mkdir()
        mlx.mkdir()
        save_f32(ours / "embed.f32", [1.0, 2.0])
        save_f32(mlx / "embed.f32", [1.0, 3.0])
        save_f32(ours / "L00.out.f32", [0.0, 0.0])
        save_f32(mlx / "L00.out.f32", [0.0, 0.0])
        (ours / "manifest.txt").write_text("embed 2\nL00.out 2\n")
        (mlx / "manifest.txt").write_text("embed 2\nL00.out 2\nnorm 2\n")
        save_f32(mlx / "norm.f32", [1.0, 1.0])
        (ours / "geom.txt").write_text(
            "value_heads 4\nkey_heads 2\nhead_cols 1\n"
        )
        lines, first = compare_dirs(ours, mlx)
        text = "\n".join(lines)
        check("compare names the first row", first == "embed" and "1.0000e+00" in text)
        check("compare keeps a later match", "L00.out" in text and "0.0000e+00" in text)
        check("compare lists an mlx-only file", "only mlx: norm" in text)

    block = struct.pack("<8I", 0x76543210, 0, 0, 0, 0, 0, 0, 0) + struct.pack("<HH", 0x3F80, 0)
    decoded = dequant_a4_block(block)
    check("a4 low nibble first", decoded[:8] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    check("a4 rest of a zero word", decoded[8:] == [0.0] * 56)
    biased = struct.pack("<8I", 0x2, 0, 0, 0, 0, 0, 0, 0) + struct.pack("<HH", 0x3F00, 0xBF80)
    biased_vals = dequant_a4_block(biased)
    check("a4 scale and bias", biased_vals[0] == 0.0 and biased_vals[1] == -1.0)
    normed = rmsnorm_list([3.0, 4.0], [1.0, 1.0], 0.0)
    scale = 12.5**-0.5
    check(
        "rmsnorm mean of squares",
        abs(normed[0] - 3.0 * scale) < 1e-12 and abs(normed[1] - 4.0 * scale) < 1e-12,
    )
    try:
        import numpy as np

        wide = dequant_a4(block)
        check(
            "a4 vector matches the block",
            len(wide) == 64 and max(abs(a - b) for a, b in zip(wide, decoded)) == 0.0,
        )
        rounded = round_bf16(np.array([1.0, 0.1], dtype=np.float32))
        check("bf16 keeps 1", float(rounded[0]) == 1.0)
        # the F16 pairs: scale 1.0 (0x3C00) and 0.5 (0x3800), bias -0.25 (0xB400)
        f16block = struct.pack("<8I", 0x2, 0, 0, 0, 0, 0, 0, 0) + struct.pack("<HH", 0x3C00, 0xB400)
        f16_vals = dequant_a4_block(f16block, "f16")
        check("a4h scale and bias", f16_vals[0] == 1.75 and f16_vals[1] == -0.25)
        wide_h = dequant_a4(f16block, "f16")
        check(
            "a4h vector matches the block",
            len(wide_h) == 64 and max(abs(a - b) for a, b in zip(wide_h, f16_vals)) == 0.0,
        )
        a8h = bytes(range(64)) + struct.pack("<HH", 0x3800, 0xB400)
        a8h_vals = dequant_a8(a8h, "f16")
        check(
            "a8h byte codes and pair",
            a8h_vals[0] == -0.25 and a8h_vals[1] == 0.25 and a8h_vals[63] == 31.25,
        )
    except ImportError:
        pass

    fails += self_test_lfm3()
    fails += self_test_deep()
    if fails:
        print(f"mlx-logits self-test: {fails} failed", file=sys.stderr)
        return 1
    print("mlx-logits self-test: ok")
    return 0


def self_test_deep() -> int:
    """The deep pin's options: the id walk, the kept positions, the header lines they add, and
    the directory writer. No mlx and no model."""
    import pathlib
    import tempfile

    fails = 0

    def check(name: str, ok: bool) -> None:
        nonlocal fails
        if not ok:
            print(f"mlx-logits self-test FAIL: {name}", file=sys.stderr)
            fails += 1

    def refused(call, needle: str) -> bool:
        try:
            call()
        except SystemExit as exc:
            return needle in str(exc)
        return False

    check("walk steps from its first id", walk_ids(3, 124894) == [124894, 8919, 16838])
    walk = parse_walk("600:124894")
    check(
        "walk of 600 stays within 1000 to 91000",
        len(walk) == 600 and walk[0] == 124894 and 1000 <= min(walk[1:]) and max(walk[1:]) <= 91000,
    )
    check("walk's last id", walk[599] == (599 * 7919) % 90001 + 1000)
    check("walk needs a first id", refused(lambda: parse_walk("600"), "N:FIRST"))
    check("walk needs a count", refused(lambda: parse_walk("x:5"), "N:FIRST"))
    check("walk of none is refused", refused(lambda: walk_ids(0, 5), "walk of 0 ids"))

    check("positions parse", parse_positions("255,256,599", 600) == [255, 256, 599])
    check("a position past the steps is refused", refused(lambda: parse_positions("600", 600), "past the 600"))
    check("positions must increase", refused(lambda: parse_positions("256,255", 600), "not increasing"))
    check("a repeated position is refused", refused(lambda: parse_positions("5,5", 600), "not increasing"))
    check("an empty position is refused", refused(lambda: parse_positions("5,", 600), "list of positions"))

    raw = struct.pack("<6f", 0.0, 1.0, 2.0, 5.0, 4.0, 3.0)
    ids = [10, 11, 12, 13]
    text = header("lfm3", ids, 4, 3, raw, [1, 3], [2, 0])
    check("positions line", "steps 4\nn 3\n" in text and "positions 1,3\n" in text)
    check("argmax line", text.endswith("argmax 2,0\n"))
    check("checksum over the kept rows", f"fnv1a {fnv1a(raw):016x}\n" in text)
    check(
        "rows of the wrong count are refused",
        refused(lambda: header("lfm3", ids, 4, 3, raw, [1, 2, 3]), "bytes is not 3 * 3"),
    )
    check(
        "argmax of the wrong count is refused",
        refused(lambda: header("lfm3", ids, 4, 3, raw, [1, 3], [2]), "1 argmax values for 2 rows"),
    )
    check("no positions keeps the plain header", "positions" not in header("lfm3", [0], 1, 3, raw[:12]))

    with tempfile.TemporaryDirectory() as tmp:
        emit("lfm3", text, raw, tmp)
        root = pathlib.Path(tmp)
        check("the directory writer's text", (root / "lfm3.txt").read_text() == text)
        check("the directory writer's bytes", (root / "lfm3.f32").read_bytes() == raw)
    return fails


def self_test_lfm3() -> int:
    """The LFM3 path's checks that need no mlx and no weights: the loader by `model_type`, the
    folder's checks over header-only folders, and the logits taken from either output shape."""
    import json
    import pathlib
    import tempfile

    fails = 0

    def check(name: str, ok: bool) -> None:
        nonlocal fails
        if not ok:
            print(f"mlx-logits self-test FAIL: {name}", file=sys.stderr)
            fails += 1

    def refused(call, needle: str) -> bool:
        try:
            call()
        except SystemExit as exc:
            return needle in str(exc)
        return False

    check("lfm3 runs through mlx-vlm", loader_of("lfm3") == "mlx-vlm")
    check("qwen runs through mlx-lm", loader_of("qwen3_5_moe") == "mlx-lm")

    class Out:
        logits = [1.0]

    check("logits of an object", unwrap_logits(Out()) == [1.0])
    check("logits of a tuple", unwrap_logits(([2.0], None)) == [2.0])
    check("logits of an array", unwrap_logits([3.0]) == [3.0])

    def header_only(path, entries: dict) -> None:
        body = json.dumps(
            {name: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]} for name, shape in entries.items()}
        ).encode()
        path.write_bytes(struct.pack("<Q", len(body)) + body)

    layers = ["conv", "full_attention"]
    config = {
        "model_type": "lfm3",
        "hidden_size": 4,
        "conv_L_cache": 3,
        "num_hidden_layers": 2,
        "layer_types": layers,
        "tie_word_embeddings": True,
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    }
    conv = "language_model.model.layers.0.conv.conv.weight"
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)

        def folder(cfg: dict, tensors: dict) -> str:
            (root / "config.json").write_text(json.dumps(cfg))
            header_only(root / "model.safetensors", tensors)
            return str(root)

        check("model type of a config", model_type_of(folder(config, {conv: [4, 3, 1]})) == "lfm3")
        check("a taps layout of MLX's passes", not refused(lambda: check_lfm3_folder(str(root)), ""))
        check(
            "taps of HF's layout are refused",
            refused(
                lambda: check_lfm3_folder(folder(config, {conv: [4, 1, 3]})),
                "[channels, kernel, 1]",
            ),
        )
        check(
            "taps of another width are refused",
            refused(
                lambda: check_lfm3_folder(folder(config, {conv: [4, 7, 1]})),
                "[channels, kernel, 1]",
            ),
        )
        check(
            "a head under a tied flag is refused",
            refused(
                lambda: check_lfm3_folder(
                    folder(config, {conv: [4, 3, 1], "language_model.lm_head.weight": [8, 2]})
                ),
                "tie_word_embeddings",
            ),
        )
        untied = dict(config, tie_word_embeddings=False)
        check(
            "no head under an untied flag is refused",
            refused(
                lambda: check_lfm3_folder(folder(untied, {conv: [4, 3, 1]})),
                "tie_word_embeddings",
            ),
        )
        check(
            "an untied folder with its head passes",
            not refused(
                lambda: check_lfm3_folder(
                    folder(untied, {conv: [4, 3, 1], "language_model.lm_head.weight": [8, 2]})
                ),
                "",
            ),
        )
        eight = dict(config, quantization={"group_size": 64, "bits": 8, "mode": "affine"})
        check(
            "8-bit blocks are refused",
            refused(lambda: check_lfm3_folder(folder(eight, {conv: [4, 3, 1]})), "affine 4-bit"),
        )
        other = dict(config, model_type="lfm2")
        check(
            "another model type is refused",
            refused(lambda: check_lfm3_folder(folder(other, {conv: [4, 3, 1]})), "is not lfm3"),
        )
        bare = {k: v for k, v in config.items() if k != "model_type"}
        check(
            "a config with no model type is refused by name",
            refused(lambda: model_type_of(folder(bare, {conv: [4, 3, 1]})), "no model_type"),
        )
    return fails


def save_f32(path, values) -> None:
    import array

    raw = array.array("f", values)
    if sys.byteorder != "little":
        raw.byteswap()
    path.write_bytes(raw.tobytes())


def load_f32(path):
    import array

    data = path.read_bytes()
    if len(data) % 4:
        raise SystemExit(f"mlx-logits: {path} is {len(data)} bytes, not whole f32s")
    raw = array.array("f")
    raw.frombytes(data)
    if sys.byteorder != "little":
        raw.byteswap()
    return raw


def read_manifest(directory):
    path = directory / "manifest.txt"
    if not path.is_file():
        raise SystemExit(f"mlx-logits: {path} is not there")
    rows = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, n = line.split()
        rows.append((name, int(n)))
    return rows


def read_geom(directory) -> dict[str, str]:
    path = directory / "geom.txt"
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.split(None, 1)
        out[key] = value
    return out


def distance(a, b):
    """(max abs, rms, count over 0.05). A NaN is an infinite distance."""
    if len(a) != len(b) or len(a) == 0:
        raise SystemExit("mlx-logits: the two rows differ in length")
    worst = 0.0
    acc = 0.0
    over = 0
    for x, y in zip(a, b):
        d = abs(x - y)
        if d != d:
            return float("inf"), float("inf"), -1
        if d > worst:
            worst = d
        acc += d * d
        if d > 0.05:
            over += 1
    return worst, (acc / len(a)) ** 0.5, over


def untile(values, n_v: int, n_k: int, head: int):
    """Grouped order of a buffer whose V heads were stored tiled.

    The split writes grouped head h at place `(h % per) * n_k + h / per`.
    """
    if n_k == 0 or n_v % n_k or head == 0 or len(values) != n_v * head:
        raise SystemExit(
            f"mlx-logits: {len(values)} values are not {n_v} heads of {head} over {n_k} key heads"
        )
    per = n_v // n_k
    out = [0.0] * len(values)
    for h in range(n_v):
        place = (h % per) * n_k + h // per
        src = place * head
        dst = h * head
        out[dst : dst + head] = values[src : src + head]
    return out


def write_bucket(directory, bucket: dict) -> None:
    import pathlib

    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, values in bucket.items():
        save_f32(directory / f"{name}.f32", values)
        lines.append(f"{name} {len(values)}\n")
    (directory / "manifest.txt").write_text("".join(lines))


def flat(y):
    """The last token of `y`, as a contiguous f32 list."""
    import mlx.core as mx
    import numpy as np

    if isinstance(y, tuple):
        y = y[0]
    y = y.astype(mx.float32)
    mx.eval(y)
    arr = np.array(y, copy=False)
    row = np.ascontiguousarray(arr.reshape(-1, arr.shape[-1])[-1], dtype=np.float32)
    return row.tolist()


def text_and_inner(model):
    text = model.language_model if hasattr(model, "language_model") else model
    inner = text.model if hasattr(text, "model") else text
    return text, inner


def conv_silu(la, qkv):
    """SiLU of the module's own conv1d on one token, the zero state a first step has."""
    import mlx.core as mx
    import mlx.nn as nn

    qkv = qkv.reshape(1, 1, -1)
    state = mx.zeros((1, la.conv_kernel_size - 1, la.conv_dim), dtype=qkv.dtype)
    conv_in = mx.concatenate([state, qkv], axis=1)
    return flat(nn.silu(la.conv1d(conv_in)))


def forward_layers(model, token: int) -> dict:
    """One token through the loaded modules, the residual after each layer and the norm."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    text, inner = text_and_inner(model)
    cache = text.make_cache()
    h = inner.embed_tokens(mx.array([[token]]))
    bucket = {"embed": flat(h)}
    fa = getattr(inner, "fa_idx", None)
    ssm = getattr(inner, "ssm_idx", None)
    fa_mask = create_attention_mask(h, cache[fa]) if fa is not None else None
    ssm_mask = create_ssm_mask(h, cache[ssm]) if ssm is not None else None
    for i, layer in enumerate(inner.layers):
        mask = ssm_mask if layer.is_linear else fa_mask
        xn = layer.input_layernorm(h)
        bucket[f"L{i:02}.in_norm"] = flat(xn)
        if layer.is_linear:
            la = layer.linear_attn
            qkv = la.in_proj_qkv(xn)
            bucket[f"L{i:02}.qkv"] = flat(qkv)
            bucket[f"L{i:02}.z"] = flat(la.in_proj_z(xn))
            bucket[f"L{i:02}.b"] = flat(la.in_proj_b(xn))
            bucket[f"L{i:02}.a"] = flat(la.in_proj_a(xn))
            bucket[f"L{i:02}.conv"] = conv_silu(la, qkv)
            mixer = la(xn, mask=mask, cache=cache[i])
        else:
            attn = layer.self_attn
            bucket[f"L{i:02}.q"] = flat(attn.q_proj(xn))
            bucket[f"L{i:02}.k"] = flat(attn.k_proj(xn))
            bucket[f"L{i:02}.v"] = flat(attn.v_proj(xn))
            mixer = attn(xn, mask=mask, cache=cache[i])
        bucket[f"L{i:02}.mixer"] = flat(mixer)
        mid = h + mixer
        bucket[f"L{i:02}.mix"] = flat(mid)
        post = layer.post_attention_layernorm(mid)
        bucket[f"L{i:02}.post_norm"] = flat(post)
        mlp = layer.mlp(post)
        bucket[f"L{i:02}.mlp"] = flat(mlp)
        h = mid + mlp
        bucket[f"L{i:02}.out"] = flat(h)
    normed = inner.norm(h)
    bucket["norm"] = flat(normed)
    if text.args.tie_word_embeddings:
        logits = inner.embed_tokens.as_linear(normed)
    else:
        logits = text.lm_head(normed)
    bucket["logits"] = flat(logits)
    return bucket


def official_logits(model, token: int):
    """The same call `dump` makes, so a manual walk that disagrees is named."""
    import mlx.core as mx
    from mlx_lm.models import cache as mlx_cache

    prompt_cache = mlx_cache.make_prompt_cache(model)
    logits = model(mx.array([token])[None], cache=prompt_cache)
    if isinstance(logits, tuple):
        logits = logits[0]
    return flat(logits[:, -1, :])


def iter_safetensors(folder: str):
    import json
    import pathlib

    import mlx.core as mx

    root = pathlib.Path(folder)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    elif (root / "model.safetensors").is_file():
        names = ["model.safetensors"]
    else:
        names = sorted(p.name for p in root.glob("*.safetensors"))
    if not names:
        raise SystemExit(f"mlx-logits: {folder} holds no safetensors")
    for name in names:
        yield mx.load(str(root / name))


def raw_facts(folder: str) -> dict:
    """Conv shapes and the norm vectors as stored, before mlx-lm's sanitize."""
    import mlx.core as mx
    import numpy as np

    conv = []
    norms = {}
    dtypes = {}
    for shard in iter_safetensors(folder):
        for key, value in shard.items():
            if "conv1d.weight" in key:
                conv.append((key, tuple(int(n) for n in value.shape)))
            if key.endswith("norm.weight") or key.endswith("layernorm.weight"):
                arr = np.array(value.astype(mx.float32))
                norms[key] = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
                dtypes[key] = str(value.dtype)
        del shard
    return {"conv": conv, "norms": norms, "dtypes": dtypes}


def loaded_params(model) -> dict:
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    out = {}
    dtypes = {}
    for key, value in tree_flatten(model.parameters()):
        if key.endswith("norm.weight") or key.endswith("layernorm.weight"):
            arr = np.array(value.astype(mx.float32))
            out[key] = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
            dtypes[key] = str(value.dtype)
    return {"norms": out, "dtypes": dtypes}


def suffix_match(keys: list[str], suffix: str) -> str | None:
    hits = [k for k in keys if k.endswith(suffix)]
    if len(hits) == 1:
        return hits[0]
    return None


def gguf_f32_tensors(first: str) -> dict:
    """F32 tensors of a GGUF split, by name. Other types are skipped."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parent / "ref" / "gguf-ranges.py"
    spec = importlib.util.spec_from_file_location("gguf_ranges", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"mlx-logits: {path} does not load")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    meta, _, _, _, _ = mod.header(first)
    shards = mod.shard_paths(first, meta)
    found = {}
    for shard in shards:
        _meta, tensors, start, _size, _align = mod.header(shard)
        with open(shard, "rb") as handle:
            for name, ty, dims, offset in tensors:
                if ty != 0:
                    continue
                if "norm" not in name:
                    continue
                n = 1
                for dim in dims:
                    n *= dim
                handle.seek(start + offset)
                raw = handle.read(n * 4)
                if len(raw) != n * 4:
                    raise SystemExit(f"mlx-logits: {name} in {shard} is short")
                found[name] = load_f32_bytes(raw)
    return found


def load_f32_bytes(raw: bytes):
    import array

    out = array.array("f")
    out.frombytes(raw)
    if sys.byteorder != "little":
        out.byteswap()
    return out


def mean_of(values) -> float:
    return sum(values) / len(values) if values else float("nan")


def weight_report(raw: dict, model, split: str | None) -> str:
    """How the folder's norms, the loaded ones, and the split's F32 norms sit."""
    import pathlib

    loaded = loaded_params(model)
    lines = ["conv shapes as stored:"]
    for key, shape in raw["conv"]:
        lines.append(f"  {key} {shape}")
    if not raw["conv"]:
        lines.append("  none")
    try:
        import mlx_lm.models.qwen3_5 as qwen

        src = pathlib.Path(qwen.__file__).read_text().splitlines()
        lines.append(f"sanitize file: {qwen.__file__}")
        for i, line in enumerate(src, 1):
            if "unsanitized" in line or "+ 1" in line:
                lines.append(f"  {i}: {line.strip()}")
    except Exception as exc:
        lines.append(f"sanitize file: not read ({exc})")
    split_norms = {}
    if split:
        split_norms = gguf_f32_tensors(split)
        lines.append(f"split f32 norms: {len(split_norms)} in {split}")
    lines.append("norm  raw_mean  loaded_mean  split_mean  loaded-raw  split-loaded")
    suffixes = sorted({k.split("layers.")[-1] if "layers." in k else k for k in raw["norms"]})
    # Compare by the shared tail, one line a stored tensor.
    for raw_key, raw_vals in sorted(raw["norms"].items()):
        tail = raw_key.split("language_model.")[-1]
        loaded_key = suffix_match(list(loaded["norms"]), tail)
        loaded_vals = loaded["norms"].get(loaded_key) if loaded_key else None
        gguf_name = gguf_name_of(tail)
        split_vals = split_norms.get(gguf_name) if gguf_name else None
        raw_m = mean_of(raw_vals.tolist())
        load_m = mean_of(loaded_vals.tolist()) if loaded_vals is not None else float("nan")
        split_m = mean_of(split_vals) if split_vals is not None else float("nan")
        if loaded_vals is not None and len(loaded_vals) == len(raw_vals):
            delta_load, _, _ = distance(raw_vals.tolist(), loaded_vals.tolist())
        else:
            delta_load = float("nan")
        if split_vals is not None and loaded_vals is not None and len(split_vals) == len(loaded_vals):
            delta_split, _, _ = distance(list(loaded_vals), list(split_vals))
        else:
            delta_split = float("nan")
        lines.append(
            f"  {tail} raw {raw_m:.6g} dtype {raw['dtypes'].get(raw_key)}"
            f" loaded {load_m:.6g} dtype {loaded['dtypes'].get(loaded_key)}"
            f" split {gguf_name} {split_m:.6g}"
            f" |loaded-raw| {delta_load:.6g} |split-loaded| {delta_split:.6g}"
        )
    # Keep suffixes referenced so a rename of the tail rule stays visible.
    lines.append(f"tails {len(suffixes)}")
    return "\n".join(lines) + "\n"


def gguf_name_of(tail: str) -> str | None:
    """The split's name of an MLX norm tail, when the map has one."""
    import re

    if tail.endswith("model.norm.weight") or tail == "norm.weight":
        if "layers." not in tail:
            return "output_norm.weight"
    m = re.search(r"layers\.(\d+)\.input_layernorm\.weight$", tail)
    if m:
        return f"blk.{m.group(1)}.attn_norm.weight"
    m = re.search(r"layers\.(\d+)\.post_attention_layernorm\.weight$", tail)
    if m:
        return f"blk.{m.group(1)}.post_attention_norm.weight"
    m = re.search(r"layers\.(\d+)\.linear_attn\.norm\.weight$", tail)
    if m:
        return f"blk.{m.group(1)}.ssm_norm.weight"
    m = re.search(r"layers\.(\d+)\.self_attn\.q_norm\.weight$", tail)
    if m:
        return f"blk.{m.group(1)}.attn_q_norm.weight"
    m = re.search(r"layers\.(\d+)\.self_attn\.k_norm\.weight$", tail)
    if m:
        return f"blk.{m.group(1)}.attn_k_norm.weight"
    return None


def dequantize_f32(mod, weight, scales, biases):
    """`weight`, `scales` and `biases` (the whole of the quantized module `mod`, or the same rows
    of each) as exact f32 values.

    The scales and biases widen first: dequantized in their stored bf16, every weight would round
    to bf16 once more. Each value is its own group's `scale * code + bias`, so a row's values do
    not depend on which other rows are dequantized with it.
    """
    import mlx.core as mx

    return mx.dequantize(
        weight,
        scales=scales.astype(mx.float32),
        biases=None if biases is None else biases.astype(mx.float32),
        group_size=mod.group_size,
        bits=mod.bits,
        mode=getattr(mod, "mode", "affine"),
    ).astype(mx.float32)


def gather_embedding(model, ids: list[int]) -> bool:
    """Replace the quantized embedding by a module that holds only the f32 rows of `ids`.

    The rows are `dequantize_f32` of the same rows of the quantized embedding, the values
    `floatize` would give the whole table, and the module returns them where the embedding
    would; an id it was not given raises by name, and so does one beyond the table. A folder whose
    head is tied to the embedding (no `lm_head` module) or whose embedding is not one quantized
    `embed_tokens` module is left to `floatize` whole. Returns whether it replaced the embedding.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten

    leaves = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    embeds = [(path, mod) for path, mod in leaves if path.split(".")[-1] == "embed_tokens"]
    has_head = any(path.split(".")[-1] == "lm_head" for path, _ in leaves)
    if len(embeds) != 1 or not has_head or type(embeds[0][1]).__name__ != "QuantizedEmbedding":
        print(
            "mlx-logits: embedding floatized whole (tied head, or not one quantized embed_tokens)",
            file=sys.stderr,
        )
        return False
    path, mod = embeds[0]
    rows = sorted(set(ids))
    if rows[-1] >= mod.weight.shape[0]:
        raise SystemExit(
            f"mlx-logits: id {rows[-1]} is outside the embedding's {mod.weight.shape[0]} rows"
        )
    at = mx.array(rows, dtype=mx.int32)
    values = dequantize_f32(
        mod,
        mx.take(mod.weight, at, axis=0),
        mx.take(mod.scales, at, axis=0),
        None if mod.get("biases", None) is None else mx.take(mod.biases, at, axis=0),
    )
    mx.eval(values)

    class Index:
        """The position of each gathered id; a plain object, so the module keeps it out of its
        parameter tree."""

        def __init__(self):
            self.at = {i: k for k, i in enumerate(rows)}

    class GatheredEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.rows = values
            self.index = Index()

        def __call__(self, x):
            flat_ids = x.reshape(-1).tolist()
            missing = sorted({i for i in flat_ids if i not in self.index.at})
            if missing:
                raise SystemExit(
                    f"mlx-logits: the embedding holds the rows of ids {rows}; "
                    f"the model asked for {missing}"
                )
            take = mx.array([self.index.at[i] for i in flat_ids], dtype=mx.int32)
            return mx.take(self.rows, take, axis=0).reshape(*x.shape, self.rows.shape[-1])

    model.update_modules(tree_unflatten([(path, GatheredEmbedding())]))
    print(
        f"mlx-logits: embedding {path}: {len(rows)} rows of {mod.weight.shape[0]} gathered "
        f"({mod.bits}-bit, group {mod.group_size}); the head stays whole in f32",
        file=sys.stderr,
    )
    return True


def floatize(model) -> tuple[str, int]:
    """Replace quantized linears and embeddings with f32 copies, and widen bf16 parameters.

    The leaves come from the module's own `leaf_modules`: a quantized module is a dict
    entry, which a walk of `__dict__` does not see. The note names what was replaced
    and what bf16 parameter is left.
    """
    import inspect

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_map_with_path

    lines = []
    try:
        src_lines, start = inspect.getsourcelines(nn.QuantizedLinear.__call__)
        src_file = inspect.getfile(nn.QuantizedLinear)
        for i, line in enumerate(src_lines, start):
            if "quantized_matmul" in line:
                lines.append(f"quant call: {src_file}:{i}: {line.strip()}")
    except Exception as exc:
        lines.append(f"quant call: {exc}")

    replaced = 0
    skipped = []

    def swap(_path, mod):
        nonlocal replaced
        if not isinstance(mod, nn.Module):
            return mod
        name = type(mod).__name__
        if name not in ("QuantizedLinear", "QuantizedEmbedding"):
            return mod
        try:
            weight = dequantize_f32(mod, mod.weight, mod.scales, mod.get("biases", None))
        except Exception as exc:
            skipped.append(f"{name}: {exc}")
            return mod
        if weight.ndim != 2:
            skipped.append(f"{name}: dequantized ndim {weight.ndim}")
            return mod
        if name == "QuantizedLinear":
            made = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        else:
            made = nn.Embedding(weight.shape[0], weight.shape[1])
        made.weight = weight
        replaced += 1
        return made

    leaves = model.leaf_modules()
    model.update_modules(tree_map_with_path(swap, leaves, is_leaf=nn.Module.is_module))

    def widen(value):
        if isinstance(value, mx.array) and value.dtype == mx.bfloat16:
            return value.astype(mx.float32)
        return value

    model.apply(widen)
    left = [
        key
        for key, value in tree_flatten(model.parameters())
        if getattr(value, "dtype", None) == mx.bfloat16
    ]
    lines.append(
        f"floatize: replaced {replaced}; bf16 parameters left: {len(left)}"
        + ("" if not left else " " + ",".join(left[:8]))
    )
    if skipped:
        lines.append("floatize skipped: " + "; ".join(skipped[:8]))
    return "\n".join(lines), replaced


def compare_dirs(ours, mlx) -> tuple[list[str], str | None]:
    import pathlib

    ours = pathlib.Path(ours)
    mlx = pathlib.Path(mlx)
    ours_rows = dict(read_manifest(ours))
    mlx_rows = dict(read_manifest(mlx))
    geom = read_geom(ours)
    n_v = int(geom["value_heads"]) if "value_heads" in geom else 0
    n_k = int(geom["key_heads"]) if geom.get("key_heads", "none").isdigit() else 0
    head = int(geom["head_cols"]) if "head_cols" in geom else 0
    lines = [f"{'name':<22} {'n':>8} {'max_abs':>12} {'rms':>12} {'over':>8}"]
    first = None
    shared = [name for name in ours_rows if name in mlx_rows]
    for name in shared:
        a = load_f32(ours / f"{name}.f32")
        b = load_f32(mlx / f"{name}.f32")
        if len(a) != len(b):
            lines.append(f"{name:<22} lengths {len(a)} and {len(b)}")
            if first is None:
                first = name
            continue
        worst, rms, over = distance(a, b)
        lines.append(f"{name:<22} {len(a):8d} {worst:12.4e} {rms:12.4e} {over:8d}")
        # qkv, z, a and b are tiled on our side, so a raw gap there is not the residual's.
        residual = name in ("embed", "norm", "logits") or name.endswith((".mix", ".out"))
        if first is None and residual and worst > 1e-2:
            first = name
        extra = untiled_line(name, a, b, n_v, n_k, head)
        if extra is not None:
            lines.append(extra)
    only_ours = [name for name in ours_rows if name not in mlx_rows]
    only_mlx = [name for name in mlx_rows if name not in ours_rows]
    lines.append(f"shared {len(shared)}; only ours {len(only_ours)}; only mlx {len(only_mlx)}")
    if only_mlx:
        lines.append("only mlx: " + " ".join(only_mlx))
    if only_ours:
        lines.append("only ours: " + " ".join(only_ours))
    lines.append(f"first above 1e-2: {first if first else 'none'}")
    return lines, first


def untiled_line(name: str, ours, mlx, n_v: int, n_k: int, head: int) -> str | None:
    """The V-head tail in grouped order, when the split stored it tiled."""
    if n_v == 0 or n_k == 0 or head == 0:
        return None
    if name.endswith(".z"):
        span, start = head, 0
    elif name.endswith(".b") or name.endswith(".a"):
        span, start = 1, 0
    elif name.endswith(".qkv"):
        qk = 2 * n_k * head
        if len(ours) <= qk:
            return None
        worst, rms, over = distance(ours[:qk], mlx[:qk])
        grouped = untile(ours[qk:], n_v, n_k, head)
        vw, vr, vo = distance(grouped, mlx[qk:])
        return (
            f"{name + '.qk':<22} {qk:8d} {worst:12.4e} {rms:12.4e} {over:8d}\n"
            f"{name + '.v':<22} {len(grouped):8d} {vw:12.4e} {vr:12.4e} {vo:8d}"
        )
    else:
        return None
    if len(ours) != n_v * span:
        return None
    grouped = untile(ours[start:], n_v, n_k, span)
    worst, rms, over = distance(grouped, mlx)
    return f"{name + '.grouped':<22} {len(grouped):8d} {worst:12.4e} {rms:12.4e} {over:8d}"


def bf16_bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def f16_bits_to_float(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def pair_bits_to_float(bits: int, pair: str) -> float:
    """A scale or bias word widened to f64: bf16 bits (`MLX_A4`/`MLX_A8`) or f16
    bits (`MLX_A4H`/`MLX_A8H`)."""
    if pair == "f16":
        return f16_bits_to_float(bits)
    return bf16_bits_to_float(bits)


def dequant_a4_block(block: bytes, pair: str = "bf16") -> list[float]:
    """One MLX_A4 group (an MLX_A4H one at `pair="f16"`): value i is `s·q + b`, q the
    low-nibble-first code."""
    if len(block) != 36:
        raise SystemExit(f"mlx-logits: an A4 group is 36 bytes, not {len(block)}")
    scale = pair_bits_to_float(struct.unpack_from("<H", block, 32)[0], pair)
    bias = pair_bits_to_float(struct.unpack_from("<H", block, 34)[0], pair)
    out = []
    for w in range(8):
        word = struct.unpack_from("<I", block, w * 4)[0]
        for j in range(8):
            out.append(scale * ((word >> (4 * j)) & 0xF) + bias)
    return out


def rmsnorm_list(values, gain, eps: float) -> list[float]:
    """`y = (x · gain) / sqrt(mean(x²) + eps)`, the mat-vec's norm on load."""
    if len(values) != len(gain) or len(values) == 0:
        raise SystemExit("mlx-logits: a norm's gain is a different length than its input")
    mean = sum(v * v for v in values) / len(values)
    scale = (mean + eps) ** -0.5
    return [v * g * scale for v, g in zip(values, gain)]


def pair_widen(bits, pair: str):
    """A u16 array of scale or bias words widened to f64, bf16 or f16."""
    import numpy as np

    if pair == "f16":
        return np.ascontiguousarray(bits, dtype=np.uint16).view("<f2").astype(np.float64)
    u = (np.asarray(bits, dtype=np.uint32) << np.uint32(16)).astype("<u4")
    return np.frombuffer(u.tobytes(), dtype="<f4").reshape(np.asarray(bits).shape).astype(np.float64)


def dequant_a4(raw: bytes, pair: str = "bf16"):
    """Every MLX_A4 group in `raw` (an MLX_A4H one at `pair="f16"`), the same values
    `dequant_a4_block` returns."""
    import numpy as np

    if len(raw) % 36:
        raise SystemExit(f"mlx-logits: {len(raw)} bytes are not whole A4 groups")
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 36)
    words = np.ascontiguousarray(blocks[:, :32]).view(np.uint32).reshape(-1, 8)
    scale = np.ascontiguousarray(blocks[:, 32:34]).view(np.uint16).reshape(-1)
    bias = np.ascontiguousarray(blocks[:, 34:36]).view(np.uint16).reshape(-1)
    codes = np.empty((blocks.shape[0], 64), dtype=np.float64)
    for j in range(8):
        word = words[:, j]
        for k in range(8):
            codes[:, j * 8 + k] = (word >> np.uint32(4 * k)) & np.uint32(0xF)
    return (codes * pair_widen(scale, pair)[:, None] + pair_widen(bias, pair)[:, None]).reshape(-1)


def dequant_a8(raw: bytes, pair: str = "bf16"):
    """MLX_A8 groups of 64 (MLX_A8H at `pair="f16"`): byte i is the code, then the scale
    and bias words of the pair."""
    import numpy as np

    if len(raw) % 68:
        raise SystemExit(f"mlx-logits: {len(raw)} bytes are not whole A8 groups")
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 68)
    codes = blocks[:, :64].astype(np.float64)
    scale = np.ascontiguousarray(blocks[:, 64:66]).view(np.uint16).reshape(-1)
    bias = np.ascontiguousarray(blocks[:, 66:68]).view(np.uint16).reshape(-1)
    return (codes * pair_widen(scale, pair)[:, None] + pair_widen(bias, pair)[:, None]).reshape(-1)


def round_bf16(values):
    """f32 rounded to bf16 and widened back, nearest even."""
    import numpy as np

    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).copy()
    low = bits & np.uint32(0xFFFF)
    hi = bits >> np.uint32(16)
    up = (low > np.uint32(0x8000)) | ((low == np.uint32(0x8000)) & ((hi & np.uint32(1)) == 1))
    hi = (hi + up.astype(np.uint32)).astype(np.uint16)
    return ((hi.astype(np.uint32) << np.uint32(16)).view(np.float32)).astype(np.float64)


def gguf_tensors(path: str):
    """(data_start, {name: (type, dims, offset)}) of one GGUF file."""
    import importlib.util
    import pathlib

    src = pathlib.Path(__file__).resolve().parent / "ref" / "gguf-ranges.py"
    spec = importlib.util.spec_from_file_location("gguf_ranges", src)
    if spec is None or spec.loader is None:
        raise SystemExit(f"mlx-logits: {src} does not load")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _meta, tensors, start, _size, _align = mod.header(path)
    return start, {name: (ty, dims, offset) for name, ty, dims, offset in tensors}


def read_tensor(path: str, index, start: int, name: str):
    """The tensor's type, dims, and f64 values. F32, BF16 and the four MLX affine kinds."""
    import numpy as np

    if name not in index:
        near = [key for key in index if name.split(".")[-1] in key][:8]
        raise SystemExit(f"mlx-logits: {path} has no {name} (near: {near})")
    ty, dims, offset = index[name]
    n = 1
    for dim in dims:
        n *= int(dim)
    if ty == 0:
        nbytes = n * 4
    elif ty == 30:
        nbytes = n * 2
    elif ty in (2004, 2006):
        nbytes = n // 64 * 36
    elif ty in (2005, 2007):
        nbytes = n // 64 * 68
    else:
        raise SystemExit(f"mlx-logits: {name} has type id {ty}")
    with open(path, "rb") as handle:
        handle.seek(start + offset)
        raw = handle.read(nbytes)
    if len(raw) != nbytes:
        raise SystemExit(f"mlx-logits: {name} is short")
    if ty == 0:
        values = np.frombuffer(raw, dtype="<f4").astype(np.float64).copy()
    elif ty == 30:
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        values = (bits << np.uint32(16)).view(np.float32).astype(np.float64).copy()
    elif ty in (2004, 2006):
        values = dequant_a4(raw, "f16" if ty == 2006 else "bf16")
    else:
        values = dequant_a8(raw, "f16" if ty == 2007 else "bf16")
    return ty, [int(d) for d in dims], values


def report_distance(label: str, got, ref) -> None:
    if len(got) != len(ref):
        print(f"{label}: lengths {len(got)} and {len(ref)}")
        return
    worst, rms, over = distance(got, ref)
    print(f"{label}: max {worst:.4e} rms {rms:.4e} over {over}")


def project(values, dims, x):
    """`W @ x` with GGUF's first dim contiguous, in f64 and in f32."""
    import numpy as np

    cols, rows = int(dims[0]), int(dims[1])
    if cols * rows != len(values):
        raise SystemExit(f"mlx-logits: {dims} does not hold {len(values)} values")
    if cols != len(x):
        raise SystemExit(f"mlx-logits: cols {cols} against an input of {len(x)}")
    weight = np.asarray(values, dtype=np.float64).reshape(rows, cols)
    vec = np.asarray(x, dtype=np.float64)
    wide = weight @ vec
    narrow = (weight.astype(np.float32) @ vec.astype(np.float32)).astype(np.float64)
    return weight, wide, narrow


def grouped_of(values, kind: str, n_v: int, n_k: int, head: int):
    """The mlx order of a buffer the split stored with its V heads tiled."""
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    if kind == "qkv":
        qk = 2 * n_k * head
        tail = untile(values[qk:].tolist(), n_v, n_k, head)
        return values[:qk].tolist() + tail
    span = 1 if kind in ("a", "b") else head
    return untile(values.tolist(), n_v, n_k, span)


def f64_main(argv: list[str]) -> int:
    """Layer 0's norm and in-projections in f64, from the split, against both dumps."""
    import numpy as np

    if len(argv) != 3:
        print(USAGE, file=sys.stderr)
        return 64
    ours_dir, mlx_dir, split = argv
    import pathlib

    ours_dir = pathlib.Path(ours_dir)
    mlx_dir = pathlib.Path(mlx_dir)
    geom = read_geom(ours_dir)
    if geom.get("kind00") not in (None, "delta"):
        raise SystemExit(f"mlx-logits: layer 0 is {geom.get('kind00')}, not a delta layer")
    eps = float(geom["eps"])
    hidden = int(geom["hidden"])
    n_v = int(geom["value_heads"])
    n_k = int(geom["key_heads"])
    head = int(geom["head_cols"])
    ours_embed = load_f32(ours_dir / "embed.f32")
    mlx_embed = load_f32(mlx_dir / "embed.f32")
    mlx_norm = load_f32(mlx_dir / "L00.in_norm.f32")
    if len(ours_embed) != hidden or len(mlx_embed) != hidden:
        raise SystemExit(f"mlx-logits: embed is not hidden {hidden}")
    start, index = gguf_tensors(split)
    _ty, gain_dims, gain = read_tensor(split, index, start, "blk.0.attn_norm.weight")
    if int(np.prod(gain_dims)) != hidden:
        raise SystemExit(f"mlx-logits: attn_norm dims {gain_dims} against hidden {hidden}")
    ours_x = np.asarray(rmsnorm_list(ours_embed, gain, eps), dtype=np.float64)
    mlx_x = np.asarray(rmsnorm_list(mlx_embed, gain, eps), dtype=np.float64)
    print(f"eps {eps:.8e} gain mean {float(np.mean(gain)):.6g} gain max {float(np.max(np.abs(gain))):.6g}")
    report_distance("norm f64(mlx embed) vs mlx in_norm", mlx_x, mlx_norm)
    report_distance("norm f64(our embed) vs mlx in_norm", ours_x, mlx_norm)
    jobs = (
        ("qkv", "blk.0.attn_qkv.weight", "L00.qkv"),
        ("z", "blk.0.attn_gate.weight", "L00.z"),
        ("a", "blk.0.ssm_alpha.weight", "L00.a"),
        ("b", "blk.0.ssm_beta.weight", "L00.b"),
    )
    names = {0: "F32", 30: "BF16", 2004: "MLX_A4", 2005: "MLX_A8",
             2006: "MLX_A4H", 2007: "MLX_A8H"}
    for kind, tensor, dump in jobs:
        ty, dims, values = read_tensor(split, index, start, tensor)
        ours = load_f32(ours_dir / f"{dump}.f32")
        mlx = load_f32(mlx_dir / f"{dump}.f32")
        print(f"{tensor} {names.get(ty, ty)} dims {dims} -> {dump} {len(ours)}")
        if len(dims) != 2:
            print(f"  rank {len(dims)}, skipped")
            continue
        orders = [("ggml", dims)]
        if dims[0] != dims[1]:
            orders.append(("swapped", [dims[1], dims[0]]))
        scored = []
        for label, order in orders:
            if int(order[0]) != hidden:
                continue
            _weight, wide, narrow = project(values, order, ours_x)
            worst, _rms, _over = distance(narrow, ours)
            scored.append((worst, label, order))
        if not scored:
            print(f"  no orientation has cols {hidden}")
            continue
        scored.sort(key=lambda item: item[0])
        print("  " + ", ".join(f"{label} f32-vs-ours {worst:.4e}" for worst, label, _o in scored))
        _worst, label, order = scored[0]
        weight, wide, narrow = project(values, order, ours_x)
        _w, mlx_wide, mlx_narrow = project(values, order, mlx_x)
        report_distance(f"  {label} f32 acc, our input, vs ours", narrow, ours)
        report_distance(f"  {label} f64 acc, our input, vs ours", wide, ours)
        ours_g = grouped_of(narrow, kind, n_v, n_k, head)
        mlx_g = grouped_of(mlx_narrow, kind, n_v, n_k, head)
        report_distance(f"  {label} f32 acc, mlx input, vs mlx", mlx_g, mlx)
        rounded_w = round_bf16(weight)
        rounded_x = round_bf16(mlx_x)
        rounded = (rounded_w.astype(np.float32) @ rounded_x.astype(np.float32)).astype(np.float64)
        report_distance(
            f"  {label} bf16 w+x, mlx input, vs mlx",
            grouped_of(rounded, kind, n_v, n_k, head),
            mlx,
        )
    score_ffn(ours_dir, mlx_dir, split, start, index, eps, hidden)
    return 0


def read_span(path: str, start: int, offset: int, ty: int, begin: int, count: int):
    """`count` values of a tensor, starting at value `begin`. The span is whole groups."""
    import numpy as np

    if ty in (2004, 2006):
        group, nbytes_g = 64, 36
    elif ty in (2005, 2007):
        group, nbytes_g = 64, 68
    elif ty == 0:
        group, nbytes_g = 1, 4
    elif ty == 30:
        group, nbytes_g = 1, 2
    else:
        raise SystemExit(f"mlx-logits: type id {ty} has no span reader")
    if begin % group or count % group:
        raise SystemExit(f"mlx-logits: a span of {count} at {begin} is not whole groups of {group}")
    byte_off = begin // group * nbytes_g
    nbytes = count // group * nbytes_g
    with open(path, "rb") as handle:
        handle.seek(start + offset + byte_off)
        raw = handle.read(nbytes)
    if len(raw) != nbytes:
        raise SystemExit("mlx-logits: a tensor span is short")
    if ty in (2004, 2006):
        return dequant_a4(raw, "f16" if ty == 2006 else "bf16")
    if ty in (2005, 2007):
        return dequant_a8(raw, "f16" if ty == 2007 else "bf16")
    if ty == 0:
        return np.frombuffer(raw, dtype="<f4").astype(np.float64).copy()
    bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (bits << np.uint32(16)).view(np.float32).astype(np.float64).copy()


def silu(values):
    import numpy as np

    v = np.asarray(values, dtype=np.float64)
    return v / (1.0 + np.exp(-v))


def score_ffn(ours_dir, mlx_dir, split: str, start: int, index, eps: float, hidden: int) -> None:
    """Layer 3's MLP in f64 from the split: the norm, the router, then the picked experts."""
    import numpy as np

    f32_dir = mlx_dir / "f32"
    if not (f32_dir / "L03.out.f32").is_file():
        print("ffn: no f32 dump, skipped")
        return
    print("ffn tensors:")
    for name in sorted(index):
        if name.startswith("blk.3.ffn") or name.startswith("blk.3.post_attention"):
            ty, dims, _off = index[name]
            print(f"  {name} type {ty} dims {dims}")
    ours_mix = np.asarray(load_f32(ours_dir / "L03.mix.f32"), dtype=np.float64)
    ours_out = np.asarray(load_f32(ours_dir / "L03.out.f32"), dtype=np.float64)
    f32_mix = np.asarray(load_f32(f32_dir / "L03.mix.f32"), dtype=np.float64)
    f32_out = np.asarray(load_f32(f32_dir / "L03.out.f32"), dtype=np.float64)
    nat_mix = np.asarray(load_f32(mlx_dir / "L03.mix.f32"), dtype=np.float64)
    nat_out = np.asarray(load_f32(mlx_dir / "L03.out.f32"), dtype=np.float64)
    ours_mlp = ours_out - ours_mix
    f32_mlp = f32_out - f32_mix
    nat_mlp = nat_out - nat_mix
    for label, vec in (("ours", ours_mlp), ("f32", f32_mlp), ("native", nat_mlp)):
        print(f"  L03 mlp {label}: max {np.max(np.abs(vec)):.4e} rms {np.sqrt(np.mean(vec * vec)):.4e}")
    report_distance("  mlp ours vs f32", ours_mlp, f32_mlp)
    report_distance("  mlp ours vs native", ours_mlp, nat_mlp)
    report_distance("  mlp f32 vs native", f32_mlp, nat_mlp)

    _ty, gain_dims, gain = read_tensor(split, index, start, "blk.3.post_attention_norm.weight")
    if int(np.prod(gain_dims)) != hidden:
        raise SystemExit(f"mlx-logits: post norm dims {gain_dims}")
    ours_hat = np.asarray(rmsnorm_list(ours_mix, gain, eps), dtype=np.float64)
    f32_hat = np.asarray(rmsnorm_list(f32_mix, gain, eps), dtype=np.float64)
    report_distance("  post f64(our mix) vs f32 post_norm", ours_hat, load_f32(f32_dir / "L03.post_norm.f32"))
    report_distance("  post f64(f32 mix) vs f32 post_norm", f32_hat, load_f32(f32_dir / "L03.post_norm.f32"))
    report_distance(
        "  post f64(native mix) vs native post_norm",
        np.asarray(rmsnorm_list(nat_mix, gain, eps)),
        load_f32(mlx_dir / "L03.post_norm.f32"),
    )

    our_logits = np.asarray(load_f32(ours_dir / "L03.router.f32"), dtype=np.float64)
    our_weights = np.asarray(load_f32(ours_dir / "L03.weights.f32"), dtype=np.float64)
    routed_name = "blk.3.ffn_gate_inp.weight"
    ty, dims, values = read_tensor(split, index, start, routed_name)
    print(f"  router {ty} dims {dims} logits {len(our_logits)} weights {len(our_weights)}")
    _w, _wide, narrow = project(values, dims, ours_hat)
    shared_name = "blk.3.ffn_gate_inp_shexp.weight"
    shared = shared_name in index
    if shared:
        _sty, sdims, svalues = read_tensor(split, index, start, shared_name)
        srow = np.asarray(svalues, dtype=np.float64).reshape(-1)
        if srow.shape != (hidden,):
            raise SystemExit(f"mlx-logits: shared gate {sdims}")
        narrow = np.concatenate([narrow, [float(np.dot(srow, ours_hat))]])
    report_distance("  router f32 acc vs ours", narrow, our_logits)

    n_routed = int(dims[1])
    top_k = len(our_weights) - (1 if shared else 0)
    if top_k < 1 or n_routed < top_k:
        raise SystemExit(f"mlx-logits: top_k {top_k} over {n_routed} routed rows")
    picked = np.argsort(-our_logits[:n_routed])[:top_k]
    chosen = our_logits[picked]
    gap = float(chosen[-1] - np.sort(our_logits[:n_routed])[-(top_k + 1)])
    shift = chosen - np.max(chosen)
    soft = np.exp(shift)
    soft = soft / np.sum(soft)
    report_distance("  router softmax vs our weights", soft, our_weights[:top_k])
    print(f"  picks {picked.tolist()} margin {gap:.4e}")

    def mlp_of(xhat, ids, weights, shared_w: float, round_weights: bool):
        acc = np.zeros(hidden, dtype=np.float64)
        for slot, expert in enumerate(ids):
            gate = one_expert(split, start, index, "blk.3.ffn_gate_exps.weight", int(expert))
            up = one_expert(split, start, index, "blk.3.ffn_up_exps.weight", int(expert))
            down = one_expert(split, start, index, "blk.3.ffn_down_exps.weight", int(expert))
            if round_weights:
                gate, up, down = round_bf16(gate), round_bf16(up), round_bf16(down)
            hidden_up = silu(gate @ xhat) * (up @ xhat)
            acc += float(weights[slot]) * (down @ hidden_up)
        if shared:
            gate = one_matrix(split, start, index, "blk.3.ffn_gate_shexp.weight")
            up = one_matrix(split, start, index, "blk.3.ffn_up_shexp.weight")
            down = one_matrix(split, start, index, "blk.3.ffn_down_shexp.weight")
            if round_weights:
                gate, up, down = round_bf16(gate), round_bf16(up), round_bf16(down)
            hidden_up = silu(gate @ xhat) * (up @ xhat)
            acc += shared_w * (down @ hidden_up)
        return acc

    shared_w = float(our_weights[top_k]) if shared else 0.0
    report_distance(
        "  mlp f64 exact vs ours",
        mlp_of(ours_hat, picked, our_weights, shared_w, False),
        ours_mlp,
    )
    # mlx-lm softmaxes every expert, then keeps the top scores. norm_topk_prob
    # renormalizes those scores; our kernel always does.
    full = np.exp(our_logits[:n_routed] - np.max(our_logits[:n_routed]))
    full = full / np.sum(full)
    bare = full[picked]
    print(f"  top-k mass {float(np.sum(bare)):.6g}")
    # The f32 forward's own picks, from the f64 router of its mix.
    _w, _wide, f32_logits = project(values, dims, f32_hat)
    if shared:
        f32_logits = np.concatenate([f32_logits, [float(np.dot(srow, f32_hat))]])
    f32_picked = np.argsort(-f32_logits[:n_routed])[:top_k]
    f32_full = np.exp(f32_logits[:n_routed] - np.max(f32_logits[:n_routed]))
    f32_full = f32_full / np.sum(f32_full)
    f32_bare = f32_full[f32_picked]
    f32_soft = f32_bare / np.sum(f32_bare)
    f32_shared = float(1.0 / (1.0 + np.exp(-f32_logits[-1]))) if shared else 0.0
    same = sorted(picked.tolist()) == sorted(f32_picked.tolist())
    print(f"  f32 picks {f32_picked.tolist()} same set {same} mass {float(np.sum(f32_bare)):.6g}")
    report_distance(
        "  mlp f64 renorm, f32 input, vs f32",
        mlp_of(f32_hat, f32_picked, f32_soft, f32_shared, False),
        f32_mlp,
    )
    report_distance(
        "  mlp f64 bare softmax, f32 input, vs f32",
        mlp_of(f32_hat, f32_picked, f32_bare, f32_shared, False),
        f32_mlp,
    )
    report_distance(
        "  mlp f64 bare softmax bf16 w, f32 input, vs f32",
        mlp_of(f32_hat, f32_picked, f32_bare, f32_shared, True),
        f32_mlp,
    )


def one_expert(path: str, start: int, index, name: str, expert: int):
    """One expert of a stack whose slowest GGUF dim is the expert."""
    import numpy as np

    ty, dims, offset = index[name]
    if len(dims) != 3:
        raise SystemExit(f"mlx-logits: {name} dims {dims}")
    cols, rows, nexp = (int(d) for d in dims)
    if expert < 0 or expert >= nexp:
        raise SystemExit(f"mlx-logits: expert {expert} of {name} ({nexp})")
    per = cols * rows
    values = read_span(path, start, offset, ty, expert * per, per)
    return np.asarray(values, dtype=np.float64).reshape(rows, cols)


def one_matrix(path: str, start: int, index, name: str):
    """A rank-2 weight, rows of the contiguous GGUF dim."""
    import numpy as np

    ty, dims, offset = index[name]
    if len(dims) != 2:
        raise SystemExit(f"mlx-logits: {name} dims {dims}")
    cols, rows = int(dims[0]), int(dims[1])
    values = read_span(path, start, offset, ty, 0, cols * rows)
    return np.asarray(values, dtype=np.float64).reshape(rows, cols)


def layers_main(argv: list[str]) -> int:
    import pathlib

    if len(argv) < 2:
        print(USAGE, file=sys.stderr)
        return 64
    folder, directory = argv[0], argv[1]
    split = None
    rest = argv[2:]
    while rest:
        if rest[0] == "--split" and len(rest) >= 2:
            split = rest[1]
            rest = rest[2:]
        else:
            print(USAGE, file=sys.stderr)
            return 64
    out = pathlib.Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    raw = raw_facts(folder)
    model = _load_model(folder)
    report = weight_report(raw, model, split)
    (out / "weights.txt").write_text(report)
    print(report, flush=True)
    walked = forward_layers(model, 0)
    official = official_logits(model, 0)
    worst, rms, over = distance(walked["logits"], official)
    print(
        f"walk vs model(): max abs {worst:.4e} rms {rms:.4e} over {over}",
        flush=True,
    )
    write_bucket(out, walked)
    try:
        note, replaced = floatize(model)
        print(note, flush=True)
        (out / "floatize.txt").write_text(note + "\n")
        if replaced == 0:
            print("f32 forward skipped: no quantized module was replaced", flush=True)
        else:
            f32_bucket = forward_layers(model, 0)
            write_bucket(out / "f32", f32_bucket)
            fw, fr, fo = distance(walked["logits"], f32_bucket["logits"])
            print(f"native vs f32 logits: max abs {fw:.4e} rms {fr:.4e} over {fo}", flush=True)
            for name in walked:
                if name not in ("embed", "norm", "logits") and not name.endswith(".out"):
                    continue
                dw, dr, do = distance(walked[name], f32_bucket[name])
                print(f"native vs f32 {name}: max abs {dw:.4e} rms {dr:.4e} over {do}", flush=True)
    except Exception as exc:
        print(f"f32 forward: {exc}", flush=True)
        (out / "floatize.txt").write_text(f"f32 forward: {exc}\n")
    print(f"layers: {len(walked)} files in {out}")
    return 0


def _load_model(folder: str):
    from mlx_lm.utils import load

    model, _tokenizer = load_folder(folder, load)
    return model


def compare_main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(USAGE, file=sys.stderr)
        return 64
    lines, _first = compare_dirs(argv[0], argv[1])
    print("\n".join(lines))
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        return self_test()
    if argv and argv[0] == "layers":
        return layers_main(argv[1:])
    if argv and argv[0] == "f64":
        return f64_main(argv[1:])
    if argv and argv[0] == "compare":
        return compare_main(argv[1:])
    if not argv or argv[0] != "dump" or len(argv) < 3:
        print(USAGE, file=sys.stderr)
        return 64
    folder, family = argv[1], argv[2]
    ids = None
    positions_text = None
    out = None
    rest = argv[3:]
    while rest:
        if rest[0] in ("--ids", "--ids-walk") and len(rest) >= 2:
            if ids is not None:
                print("mlx-logits: --ids and --ids-walk name the ids once", file=sys.stderr)
                return 64
            ids = parse_ids(rest[1]) if rest[0] == "--ids" else parse_walk(rest[1])
            rest = rest[2:]
        elif rest[0] == "--positions" and len(rest) >= 2:
            positions_text = rest[1]
            rest = rest[2:]
        elif rest[0] == "--out" and len(rest) >= 2:
            out = rest[1]
            rest = rest[2:]
        else:
            print(USAGE, file=sys.stderr)
            return 64
    if ids is None:
        ids = [0]
    positions = None if positions_text is None else parse_positions(positions_text, len(ids))
    if not family or "/" in family or family.startswith("."):
        print(f"mlx-logits: {family!r} is not a family name", file=sys.stderr)
        return 64
    dump(folder, family, ids, positions, out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
