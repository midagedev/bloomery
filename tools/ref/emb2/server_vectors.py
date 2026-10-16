#!/usr/bin/env python3
"""Collect llama-server's EmbeddingGemma vectors into an oracle set in dump_ref's set format: the end-to-end answer
bloomery's /v1/embeddings is compared against.

  server_vectors.py --url <http://host:port> --pid <server pid> --model <gguf> --model-sha256 <hex> --arch <architecture>
                    --card <name> --build-record <file> --ids-dir <dir> --case <name> ... --out <dir>
                    [--mmproj <gguf> --mmproj-sha256 <hex> --image <name>=<png> ...] [--mix-image <name>]
                    [--server-flags <text>]

The server is the caller's (tools/ref/emb2/emb2.sh starts it): llama-server --embeddings with the file's mean pooling
and a ubatch that holds the longest case. Before any request and after the last, the pid the caller started (--pid) must
be alive and the server at --url must answer /props with the model it was given and a build_info of the commit
BLOOMERY_REF_BUILD names; whether that pid is the one answering is not checked here (emb2.sh refuses a port already
taken). The model and optional projector's sha256 are computed here and must be the ones given.
Per text case (<ids-dir>/<case>.txt and its ids, the profile's tokenizer set), two /v1/embeddings requests, both
encoding_format base64 so the f32 bytes are the server's own:
  <case>/embd    the default answer: the mean-pooled vector, L2 normalized (embd_normalize 2)          [768, 1]
  <case>/raw     the same request with embd_normalize -1: the pooled vector before normalization       [768, 1]
  <case>/mrl<k>  derived, not the server's: the first k values of raw, L2 normalized in f64 and rounded to f32, for
                 each Matryoshka width k the model card names (128, 256, 512); llama-server has no `dimensions` field
A case that is not UTF-8 cannot be a JSON string: it gets a `# skipped` line naming it and no rows. Each other case's ids
are checked against /tokenize (add_special, parse_special: what the embedding handler does) and the oracle's ids; a
`# tokenize` line records the comparison and the usage line the server's prompt_tokens, and a difference stops the run.
Per --image, the request is one {"content": [{"type": "image_url", ...}]} object (a data: URL), rows `img_<name>/embd`
and `img_<name>/raw`; --mix-image adds one request with a text part before that image (`mix_<name>`).
The header carries what the node-dump reader and the server family check: `# model`, `# arch`, `# build`
(BLOOMERY_REF_BUILD), `# emb2 server`, `# device` (cuda and the card), `# model_sha256`, and, when a projector is used,
`# mmproj <path> sha256 <hex>`;
then the server's flags and build, the build record's lines, and the `# complete` trailer, written last through a .partial
name. Python's standard library only.

Compare a running bloomery HTTP server with those saved vectors (no new oracle or threshold):
  server_vectors.py compare --url <http://host:port> --set <server-set> [--ids-dir <dir> --case <name> ...]
                            [--image <name>=<png> ...] [--mix-image <name>]
Prints cosine and maximum absolute difference for each selected raw and normalized vector; image inputs use the
collector's exact content-part construction. A successful command means the comparisons completed, not that a band passed.
"""

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
import urllib.request
from pathlib import Path

MRL = (128, 256, 512)
MIX_TEXT = "A picture of coloured rings and ramps: "


def image_entries(images: list[str], mix: str | None):
    """The same image-only and text-first inputs for collecting and comparing HTTP vectors."""
    found = False
    for item in images:
        name, png = item.split("=", 1)
        data = Path(png).read_bytes()
        url = "data:image/png;base64," + base64.b64encode(data).decode()
        part = {"type": "image_url", "image_url": {"url": url}}
        yield f"img_{name}", {"content": [part]}, png, data
        if name == mix:
            found = True
            text = {"type": "text", "text": MIX_TEXT}
            yield f"mix_{name}", {"content": [text, part]}, png, data
    if mix and not found:
        raise SystemExit(f"server_vectors: --mix-image {mix} names no --image")


def metrics(ours: list[float], reference: list[float]) -> tuple[float, float]:
    """Cosine and maximum absolute difference, refusing empty, non-finite or unequal vectors."""
    if not ours or len(ours) != len(reference) or not all(math.isfinite(x) for x in ours + reference):
        raise ValueError("empty, non-finite or unequal vector extents")
    dot = sum(x * y for x, y in zip(ours, reference))
    norm = math.sqrt(sum(x * x for x in ours)) * math.sqrt(sum(y * y for y in reference))
    if norm == 0:
        raise ValueError("cosine is undefined for a zero vector")
    return dot / norm, max(abs(x - y) for x, y in zip(ours, reference))


def compare_main(argv: list[str]) -> None:
    """Compare a running HTTP server with saved llama-server raw and normalized vectors."""
    ap = argparse.ArgumentParser(description=compare_main.__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--set", required=True, type=Path)
    ap.add_argument("--ids-dir", type=Path)
    ap.add_argument("--case", action="append", default=[])
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--mix-image")
    a = ap.parse_args(argv)
    if not a.case and not a.image:
        ap.error("at least one --case or --image is required")
    if a.case and a.ids_dir is None:
        ap.error("--case requires --ids-dir")
    spec = importlib.util.spec_from_file_location("manifest", Path(__file__).resolve().parents[2] / "bloomery/manifest.py")
    manifest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manifest)
    m = manifest.read(str(a.set))
    if m.complete is None or m.header.get("emb2") != "server" or m.header.get("arch") != "gemma-embedding2":
        raise SystemExit(f"server_vectors: {a.set}: not a complete EmbeddingGemma 2 server set")
    rows = {}
    for row in m.rows("tensor"):
        name = row["name"]
        if name in rows:
            raise SystemExit(f"server_vectors: {name}: duplicate tensor")
        rows[name] = row
    entries = []
    for name in a.case:
        entries.append((name, (a.ids_dir / f"{name}.txt").read_text(encoding="utf-8")))
    entries.extend((name, entry) for name, entry, _, _ in image_entries(a.image, a.mix_image))
    for name, entry in entries:
        for node, normalization in (("raw", -1), ("embd", 2)):
            key = f"{name}/{node}"
            if key not in rows:
                raise SystemExit(f"server_vectors: {a.set}: no tensor {key}")
            row = rows[key]
            if row["type"] != "f32" or row.int("occurrence") != 0 or any(row.int(f"ne{i}") != 1 for i in (1, 2, 3)):
                raise SystemExit(f"server_vectors: {key}: not one F32 vector")
            path = Path(m.dir) / manifest.dump_file_name(key, 0, "output", "contiguous", "f32")
            data = path.read_bytes()
            if len(data) != row.int("bytes") or len(data) != row.int("ne0") * 4:
                raise SystemExit(f"server_vectors: {key}: tensor byte count differs")
            reference = list(struct.unpack(f"<{len(data) // 4}f", data))
            ours, count = embed(a.url, entry, normalization)
            try:
                cosine, maximum = metrics(ours, reference)
            except ValueError as e:
                raise SystemExit(f"server_vectors: {key}: {e}") from e
            print(f"{key}: prompt_tokens {count} cosine {cosine:.9e} max_abs_diff {maximum:.9e}")



def post(url: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def embed(url: str, entry, normalize: int) -> tuple[list[float], int]:
    body = {"input": [entry], "encoding_format": "base64", "embd_normalize": normalize}
    r = post(url, "/v1/embeddings", body)
    raw = base64.b64decode(r["data"][0]["embedding"])
    return list(struct.unpack(f"<{len(raw) // 4}f", raw)), r["usage"]["prompt_tokens"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class SetWriter:
    def __init__(self, out: Path):
        self.out, self.header, self.rows = out, [], []
        out.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, xs: list[float], op: str) -> None:
        stem = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        data = struct.pack(f"<{len(xs)}f", *xs)
        if not all(math.isfinite(x) for x in xs):
            raise SystemExit(f"server_vectors: {name} holds a value that is not finite")
        (self.out / f"{stem}.0.f32").write_bytes(data)
        total = sum(struct.unpack(f"<{len(xs)}f", data))
        self.rows.append(f"tensor\t{name}\t0\tf32\t{len(xs)}\t1\t1\t1\t{len(data)}\t{total:.6f}\t{op}\t1\t0\t-\t-")

    def finish(self) -> None:
        cols = "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1"
        lines = self.header + [cols] + self.rows + [f"# complete\t{len(self.rows)}\t0"]
        partial = self.out / "MANIFEST.tsv.partial"
        partial.write_text("\n".join(lines) + "\n")
        partial.rename(self.out / "MANIFEST.tsv")


def pooled_rows(w: SetWriter, url: str, name: str, entry, mrl: bool) -> int:
    norm, n = embed(url, entry, 2)
    raw, n_raw = embed(url, entry, -1)
    if n != n_raw:
        raise SystemExit(f"server_vectors: {name}: prompt_tokens {n} and {n_raw} for one input")
    w.write(f"{name}/embd", norm, "SERVER")
    w.write(f"{name}/raw", raw, "SERVER")
    if mrl:
        for k in MRL:
            cut = raw[:k]
            s = math.sqrt(sum(x * x for x in cut))
            w.write(f"{name}/mrl{k}", [x / s for x in cut], "DERIVED")
    return n


def check_server(a: argparse.Namespace, build: str) -> dict:
    """The caller's pid is alive and the server at the url runs the given model, built at the commit `build` names."""
    try:
        os.kill(a.pid, 0)
    except OSError as e:
        raise SystemExit(f"server_vectors: the server's pid {a.pid} is not running ({e}): another server answers") from e
    props = json.loads(urllib.request.urlopen(a.url + "/props", timeout=60).read())
    if props.get("model_path") != a.model:
        raise SystemExit(f"server_vectors: the server runs {props.get('model_path')!r}, not {a.model}")
    info = props.get("build_info", "")
    commit = info.rsplit("-", 1)[-1]
    n = min(len(commit), len(build))
    if n < 7 or commit[:n] != build[:n]:
        raise SystemExit(f"server_vectors: the server's build_info is {info!r}, not the commit {build}")
    return props


def main() -> None:
    if sys.argv[1:2] == ["compare"]:
        compare_main(sys.argv[2:])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--pid", required=True, type=int)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-sha256", required=True)
    ap.add_argument("--mmproj")
    ap.add_argument("--mmproj-sha256")
    ap.add_argument("--arch", choices=("gemma-embedding", "gemma-embedding2"), required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--build-record", required=True, type=Path)
    ap.add_argument("--ids-dir", required=True, type=Path)
    ap.add_argument("--case", action="append", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--mix-image")
    ap.add_argument("--server-flags", default="")
    a = ap.parse_args()

    if a.cpu and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("server_vectors: --cpu requires CUDA_VISIBLE_DEVICES empty")
    build = os.environ.get("BLOOMERY_REF_BUILD", "")
    if not build:
        raise SystemExit("server_vectors: BLOOMERY_REF_BUILD names no commit")
    if bool(a.mmproj) != bool(a.mmproj_sha256) or (a.image and not a.mmproj):
        raise SystemExit("server_vectors: images need a projector path and sha256")
    files = [(a.model, a.model_sha256)]
    if a.mmproj:
        files.append((a.mmproj, a.mmproj_sha256))
    for path, want in files:
        got = sha256(Path(path))
        if got != want:
            raise SystemExit(f"server_vectors: {path} has sha256 {got}, want {want}")
    props = check_server(a, build)
    w = SetWriter(a.out)
    w.header += [
        "# server_vectors: llama.cpp mainline llama-server /v1/embeddings, raw f32, little-endian",
        f"# model\t{a.model}",
        f"# build\t{build}",
        f"# arch\t{a.arch}",
        "# emb2\tserver",
        f"# device\t{'cpu' if a.cpu else 'cuda'}\tthreads\t-\tcard\t{a.card}",
        f"# model_sha256\t{a.model_sha256}",
        f"# server_flags\t{a.server_flags}",
        f"# server_build\t{props['build_info']}",
    ]
    if a.mmproj:
        w.header.append(f"# mmproj\t{a.mmproj}\tsha256\t{a.mmproj_sha256}")
    w.header += [f"# build_record\t{line}" for line in a.build_record.read_text().splitlines() if line]
    for name in a.case:
        data = (a.ids_dir / f"{name}.txt").read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            w.header.append(f"# skipped\t{name}\tnot UTF-8, so not a JSON string")
            continue
        want = [int(x) for x in (a.ids_dir / f"{name}.ids").read_text().split()]
        got = post(a.url, "/tokenize", {"content": text, "add_special": True, "parse_special": True})["tokens"]
        n = pooled_rows(w, a.url, name, text, True)
        same = "equal" if got == want else f"differs server {len(got)} ids, oracle {len(want)}"
        w.header.append(f"# tokenize\t{name}\t{same}\tprompt_tokens {n}")
        if got != want or n != len(want):
            raise SystemExit(f"server_vectors: {name}: the server's ids are not the oracle's ({same}, prompt_tokens {n})")

    for name, entry, png, data in image_entries(a.image, a.mix_image):
        n = pooled_rows(w, a.url, name, entry, False)
        suffix = "\ttext first" if name.startswith("mix_") else ""
        w.header.append(f"# image\t{name}\t{png}\t{hashlib.sha256(data).hexdigest()}\tprompt_tokens {n}{suffix}")
    check_server(a, build)
    w.finish()
    print(f"server_vectors: {len(w.rows)} row(s) into {a.out}")


if __name__ == "__main__":
    main()
