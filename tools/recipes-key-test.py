#!/usr/bin/env python3
"""Consumer regressions for reached recipe keys, using offline Cargo workspaces.

Run: python3 tools/recipes-key-test.py --self-test
Against another implementation: add --source-root /path/to/checkout.
RECIPES_KEY_TEST_FILTER selects unittest names by substring (comma-separated).
No fixture compiles Rust or runs a box, contacts a registry, or needs a GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


SOURCE_ROOT = Path(__file__).resolve().parent.parent


class Workspace:
    """A real checkout, metadata, CLI and private cache; every key is a new process."""

    def __init__(self, case: unittest.TestCase, files: dict[str, str], *, git: bool = True):
        self.case = case
        self.tmp = tempfile.TemporaryDirectory(prefix="recipes-key-cli-")
        case.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "checkout"
        self.home = self.base / "home"
        self.root.mkdir()
        self.home.mkdir()
        old_home = Path.home()
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("BLOOMERY_", "CARGO_", "RUSTFLAGS", "RUSTDOCFLAGS"))}
        self.env.update(HOME=str(self.home), XDG_CACHE_HOME=str(self.home / ".cache"),
                        CARGO_HOME=os.environ.get("CARGO_HOME", str(old_home / ".cargo")),
                        RUSTUP_HOME=os.environ.get("RUSTUP_HOME", str(old_home / ".rustup")),
                        CARGO_NET_OFFLINE="true", PYTHONDONTWRITEBYTECODE="1",
                        GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull,
                        GIT_AUTHOR_NAME="test", GIT_AUTHOR_EMAIL="test@example.invalid",
                        GIT_COMMITTER_NAME="test", GIT_COMMITTER_EMAIL="test@example.invalid")
        for rel, text in files.items():
            self.write(rel, text)
        # Copy production code, not an echo of the desired key or a patched resolver.
        self.copy("tools/recipes.py")
        helpers = SOURCE_ROOT / "tools/bloomery"
        for src in helpers.rglob("*.py"):
            self.copy(str(src.relative_to(SOURCE_ROOT)))
        self.copy("tools/fixture-dirs.tsv")
        for rel in ("tools/box.sh", "tools/ref/ref-paths.sh",
                    "tools/ref/models/deepseek41.sh", "tools/ref/models/deepseek2.sh"):
            self.write(rel, "#!/bin/sh\n")
        self.manifest = self.base / "manifest.txt"
        self.manifest.write_text("# box-manifest 1 host=selftest data=/tmp/recipes-key-data\n"
                                 "env\tbloomery-env.sh\t" + "a" * 64 + "\n"
                                 "os\tkernel\tselftest\n"
                                 "tool\trustc\tselftest-offline\n# end\n")
        self.ledger = self.base / "ledger.tsv"
        self.parts_dir = self.base / "parts"
        self.run("cargo", "generate-lockfile", "--offline", "-q")
        if git:
            self.run("git", "init", "-q", "-b", "main")
            self.run("git", "add", "-A")
            self.run("git", "commit", "-q", "-m", "fixture")

    def copy(self, rel: str) -> None:
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_ROOT / rel, dst)

    def write(self, rel: str, text: str) -> None:
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)

    def run(self, *args: str) -> subprocess.CompletedProcess:
        done = subprocess.run(args, cwd=self.root, env=self.env, text=True,
                              capture_output=True, timeout=120)
        self.case.assertEqual(done.returncode, 0,
                              f"{args!r}\nstdout: {done.stdout}\nstderr: {done.stderr}")
        return done

    def keys(self, *items: str, ledger: bool = False) -> dict[str, tuple[str, str, str, list[str]]]:
        args = [sys.executable, "tools/recipes.py", "key", "--manifest", str(self.manifest),
                "--parts-dir", str(self.parts_dir)]
        if ledger:
            args += ["--ledger", str(self.ledger)]
        done = self.run(*args, *items)
        rows = done.stdout.splitlines()
        self.case.assertEqual(len(rows), len(items), done.stdout)
        result = {}
        for index, (line, item) in enumerate(zip(rows, items)):
            fields = line.split("\t", 3)
            self.case.assertEqual(len(fields), 4, line)
            key, got_item, status, detail = fields
            self.case.assertEqual(got_item, item)
            parts = []
            if status != "error":
                self.case.assertRegex(key, r"^[0-9a-f]{64}$")
                parts = (self.parts_dir / f"{index}.parts").read_text().splitlines()
                self.case.assertEqual(hashlib.sha256(("\n".join(parts) + "\n").encode()).hexdigest(), key)
                self.case.assertTrue(parts[0].startswith("v\tbloomery-gate-key "))
            else:
                self.case.assertEqual(key, "-")
            result[item] = key, status, detail, parts
        return result

    def green(self, *items: str) -> dict[str, tuple[str, str, str, list[str]]]:
        rows = self.keys(*items)
        lines = []
        for item, (key, status, _, parts) in rows.items():
            self.case.assertEqual(status, "key", rows[item])
            name = re.split(r"[@:]", item, maxsplit=1)[0]
            lines.append(f"{key}\t{name}\t{item}\tfixture\t2026-10-21T00:00:00Z\t/private/fixture\n")
            dst = Path(str(self.ledger) + ".parts") / f"{key}.parts"
            dst.parent.mkdir(exist_ok=True)
            dst.write_text("\n".join(parts) + "\n")
        self.ledger.write_text("".join(lines))
        self.assert_status(self.keys(*items, ledger=True), {item: "skip" for item in items})
        return rows

    def assert_status(self, rows, expected: dict[str, str]) -> None:
        self.case.assertEqual({item: row[1] for item, row in rows.items()}, expected)

    @contextlib.contextmanager
    def changed(self, rel: str, text: str, *, restore_mtime: bool = False):
        path = self.root / rel
        before = path.read_bytes()
        stat = path.stat()
        path.write_text(text)
        if restore_mtime:
            self.case.assertEqual(path.stat().st_size, stat.st_size)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        try:
            yield
        finally:
            path.write_bytes(before)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def package(name: str, extra: str = "") -> str:
    return f'[package]\nname = "{name}"\nversion = "0.1.0"\nedition = "2021"\n' + extra


def runtime_files() -> dict[str, str]:
    return {
        "Cargo.toml": '[workspace]\nresolver = "2"\nmembers = ["crates/app", "crates/mid", "crates/dep"]\n',
        "justfile": (
            "gate-called:\n    ./tools/box.sh 'cargo run -p app --bin called'\n"
            "gate-linked:\n    ./tools/box.sh 'cargo run -p app --bin linked'\n"
            "gate-build:\n    ./tools/box.sh 'cargo build -p app --bin called'\n"
            "gate-wrap: gate-build\n    ./tools/box.sh 'echo wrapped'\n"
            "gate-only: gate-build\n"
            "gate-feature:\n    ./tools/box.sh 'cargo run -p app --bin called --features extra'\n"
            "gate-test:\n    ./tools/box.sh 'cargo test -p app --lib'\n"
        ),
        "crates/app/Cargo.toml": package("app", '[features]\nextra = []\n[dependencies]\nmid = { path = "../mid" }\ndep = { path = "../dep" }\n'),
        "crates/app/src/bin/called.rs": (
            '#[cfg(feature = "extra")]\nmod extra;\n'
            'fn main() {\n    let _token = dep::Token;\n    mid::run();\n'
            '    #[cfg(feature = "extra")]\n    extra::run();\n}\n'),
        "crates/app/src/bin/extra/mod.rs": "pub fn run() { dep::feature::run(); }\n",
        "crates/app/src/bin/linked.rs": "fn main() { println!(\"linked\"); }\n",
        "crates/app/src/lib.rs": '#[cfg(test)]\nmod tests {\n    #[test]\n    fn check() { dep::testonly::run(); }\n}\n',
        "crates/mid/Cargo.toml": package("mid", '[dependencies]\ndep = { path = "../dep" }\n'),
        "crates/mid/src/lib.rs": "pub fn run() { dep::used::run(); }\n",
        "crates/dep/Cargo.toml": package("dep"),
        "crates/dep/src/lib.rs": "pub mod used;\npub mod unused;\npub mod feature;\npub mod testonly;\npub struct Token;\n",
        "crates/dep/src/used.rs": 'pub fn run() { println!("used1"); }\n',
        "crates/dep/src/unused.rs": 'pub fn ignored() { println!("unused1"); }\n',
        "crates/dep/src/feature.rs": 'pub fn run() { println!("feature1"); }\n',
        "crates/dep/src/testonly.rs": 'pub fn run() { println!("test1"); }\n',
    }


def macro_files() -> dict[str, str]:
    files = runtime_files()
    files["Cargo.toml"] = files["Cargo.toml"].replace('"crates/dep"]', '"crates/dep", "crates/macros"]')
    files.update({
        "crates/macros/Cargo.toml": package("macros", '[lib]\nproc-macro = true\n'),
        "crates/macros/src/lib.rs": r'''extern crate proc_macro;
mod helper;
use proc_macro::TokenStream;
#[proc_macro_attribute]
pub fn keep(_: TokenStream, item: TokenStream) -> TokenStream { helper::observe(); item }
#[proc_macro_derive(Generated)]
pub fn generated(_: TokenStream) -> TokenStream { helper::observe(); TokenStream::new() }
#[proc_macro_attribute]
pub fn startup(_: TokenStream, mut item: TokenStream) -> TokenStream {
    item.extend(r#"#[used] #[link_section = ".init_array"]
static START: extern "C" fn() = {
    extern "C" fn launch() { initialize(); }
    launch
};"#.parse::<TokenStream>().unwrap());
    item
}
#[proc_macro_attribute]
pub fn inject(_: TokenStream, _: TokenStream) -> TokenStream {
    "pub fn run() { crate::hidden::run(); }".parse().unwrap()
}
''',
        "crates/macros/src/helper.rs": 'pub fn observe() { let _ = include_str!("../macro-input.txt"); }\n',
        "crates/macros/macro-input.txt": "macro1\n",
    })
    return files


def source_macro_files(kind: str) -> dict[str, str]:
    files = runtime_files()
    files["crates/dep/src/lib.rs"] += "mod hidden;\nmod startup;\n"
    files["crates/dep/src/hidden.rs"] = 'pub fn run() { println!("source-macro1"); }\n'
    if kind in ("exported-startup", "exported-expression", "exported-import", "exported-proc-startup"):
        files["Cargo.toml"] = files["Cargo.toml"].replace('"crates/dep"]', '"crates/dep", "crates/macros"]')
        files["crates/dep/Cargo.toml"] += '[dependencies]\nmacros = { path = "../macros" }\n'
        files["crates/macros/Cargo.toml"] = package("macros")
        files["crates/macros/src/lib.rs"] = r'''#[macro_export]
macro_rules! boot {
    () => {
        #[used]
        #[link_section = ".init_array"]
        static START: extern "C" fn() = {
            extern "C" fn launch() { crate::hidden::run(); }
            launch
        };
    };
}
'''
        files["crates/dep/src/startup.rs"] = "macros::boot!();\n"
        if kind in ("exported-expression", "exported-import"):
            files["crates/macros/src/lib.rs"] = '#[macro_export]\nmacro_rules! boot { () => { crate::hidden::run() }; }\n'
            if kind == "exported-import":
                files["crates/macros/src/lib.rs"] = '#[macro_export]\nmacro_rules! boot { () => {{ use crate::hidden::run; run(); }}; }\n'
            files["crates/dep/src/startup.rs"] = "pub fn run() { macros::boot!(); }\n"
            files["crates/dep/src/used.rs"] = "pub fn run() { crate::startup::run(); }\n"
        elif kind == "exported-proc-startup":
            files["Cargo.toml"] = files["Cargo.toml"].replace('"crates/macros"]', '"crates/macros", "crates/procgen"]')
            files["crates/dep/Cargo.toml"] += 'procgen = { path = "../procgen" }\n'
            files["crates/macros/src/lib.rs"] = '#[macro_export]\nmacro_rules! boot { () => { procgen::boot!(); }; }\n'
            files["crates/procgen/Cargo.toml"] = package("procgen") + "[lib]\nproc-macro = true\n"
            files["crates/procgen/src/lib.rs"] = r'''extern crate proc_macro;
#[proc_macro]
pub fn boot(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    r#"#[used] #[link_section = ".init_array"]
       static START: extern "C" fn() = {
           extern "C" fn launch() { crate::hidden::run(); }
           launch
       };"#.parse().unwrap()
}
'''
    elif kind == "local-relative-import":
        files["crates/dep/src/lib.rs"] = ('macro_rules! boot { () => {{ use super::hidden::run; run(); }}; }\n'
                                         + files["crates/dep/src/lib.rs"])
        files["crates/dep/src/startup.rs"] = "pub fn run() { boot!(); }\n"
        files["crates/dep/src/used.rs"] = "pub fn run() { crate::startup::run(); }\n"
    else:
        source = ('macro_rules! boot { () => { impl Drop for crate::Token {\n'
                  '    fn drop(&mut self) { crate::hidden::run(); }\n'
                  '} }; }\nboot!();\n')
        files["crates/dep/src/startup.rs"] = source.replace("boot", "println") if kind == "shadowed-builtin" else source
    return files


class RecipeKeyCLI(unittest.TestCase):
    def assert_file(self, row, rel: str, present: bool = True) -> None:
        paths = {p.split("\t")[1] for p in row[3] if p.startswith(("file\t", "link\t", "manifest\t"))}
        self.assertEqual(rel in paths, present, f"{rel}: {row}")

    def test_called_versus_linked_transitive_and_ledger(self):
        ws = Workspace(self, runtime_files())
        items = ("gate-called", "gate-linked")
        baseline = ws.green(*items)
        self.assert_file(baseline[items[0]], "crates/dep/src/used.rs")
        self.assert_file(baseline[items[1]], "crates/dep/src/used.rs", False)
        self.assert_file(baseline[items[0]], "crates/dep/src/unused.rs", False)
        for rel, expected in (("crates/dep/src/unused.rs", {i: "skip" for i in items}),
                              ("crates/dep/src/used.rs", {items[0]: "run", items[1]: "skip"})):
            with self.subTest(rel=rel), ws.changed(rel, (ws.root / rel).read_text().replace("1", "2"), restore_mtime=True):
                rows = ws.keys(*items, ledger=True)
                ws.assert_status(rows, expected)
                for item in items:
                    self.assertEqual(rows[item][0] == baseline[item][0], expected[item] == "skip")

    def test_profiles_and_reached_sources_invalidate_their_consumers(self):
        files = runtime_files()
        files["justfile"] += (
            "gate-profiled:\n"
            "    BLOOMERY_MODEL=${BLOOMERY_MODEL:-qwen35moe} ./tools/box.sh 'cargo run -p app --bin called'\n"
            "    BLOOMERY_MODEL=deepseek2 ./tools/box.sh 'cargo run -p app --bin linked'\n"
            "gate-other-profile:\n"
            "    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'cargo run -p app --bin linked'\n"
        )
        for override, selected in ((None, "qwen35moe"), ("glm5next", "glm5next")):
            with self.subTest(inherited_profile=override):
                ws = Workspace(self, files)
                for profile in ("deepseek2", "qwen3moe", "qwen35moe", "glm5next"):
                    ws.write(f"tools/ref/models/{profile}.sh", f"PROFILE_VALUE={profile}-1\n")
                if override is not None:
                    ws.env["BLOOMERY_MODEL"] = override
                items = ("gate-profiled", "gate-other-profile")
                ws.green(*items)
                mutations = (
                    (f"tools/ref/models/{selected}.sh", ("run", "skip")),
                    ("tools/ref/models/deepseek2.sh", ("run", "skip")),
                    ("tools/ref/models/qwen3moe.sh", ("skip", "run")),
                    ("crates/dep/src/used.rs", ("run", "skip")),
                    ("crates/dep/src/unused.rs", ("skip", "skip")),
                )
                for rel, verdicts in mutations:
                    changed = (ws.root / rel).read_text().replace("1", "2")
                    with self.subTest(input=rel), ws.changed(rel, changed, restore_mtime=True):
                        ws.assert_status(ws.keys(*items, ledger=True), dict(zip(items, verdicts)))

    def test_recipe_dependencies_and_dependency_only_wrapper(self):
        ws = Workspace(self, runtime_files())
        items = ("gate-build", "gate-wrap", "gate-only")
        rows = ws.green(*items)
        for row in rows.values():
            self.assert_file(row, "crates/app/src/bin/called.rs")
            self.assert_file(row, "crates/dep/src/used.rs")
            self.assert_file(row, "crates/dep/src/unused.rs", False)
        for rel, status in (("crates/dep/src/used.rs", "run"), ("crates/dep/src/unused.rs", "skip"),
                            ("crates/app/src/bin/called.rs", "run")):
            with self.subTest(rel=rel), ws.changed(rel, (ws.root / rel).read_text() + "// changed\n"):
                ws.assert_status(ws.keys(*items, ledger=True), {i: status for i in items})

    def test_new_edge_in_previously_excluded_source(self):
        ws = Workspace(self, runtime_files())
        baseline = ws.green("gate-called")
        rel = "crates/dep/src/unused.rs"
        self.assert_file(baseline["gate-called"], rel, False)
        with ws.changed(rel, 'impl Drop for crate::Token {\n    fn drop(&mut self) { crate::used::run(); }\n}\n'):
            rows = ws.keys("gate-called", ledger=True)
            ws.assert_status(rows, {"gate-called": "run"})
            self.assert_file(rows["gate-called"], rel)

    def test_nonstandard_drop_impl_keeps_its_transitive_helper(self):
        files = runtime_files()
        files["crates/dep/src/lib.rs"] += '#[path = "drop.rust"] mod destructor;\nmod hidden;\n'
        files["crates/dep/src/drop.rust"] = 'impl Drop for crate::Token { fn drop(&mut self) { crate::hidden::run(); } }\n'
        files["crates/dep/src/hidden.rs"] = 'pub fn run() { println!("drop1"); }\npub fn alt() { println!("alt1"); }\n'
        ws = Workspace(self, files)
        items = ("gate-called", "gate-linked")
        baseline = ws.green(*items)
        for rel in ("crates/dep/src/drop.rust", "crates/dep/src/hidden.rs"):
            self.assert_file(baseline["gate-called"], rel)
            self.assert_file(baseline["gate-linked"], rel, False)
            changed = files[rel].replace("hidden::run()", "hidden::alt()") if rel.endswith(".rust") else files[rel].replace("drop1", "drop2")
            with self.subTest(rel=rel), ws.changed(rel, changed, restore_mtime=True):
                ws.assert_status(ws.keys(*items, ledger=True), {"gate-called": "run", "gate-linked": "skip"})

    def test_absolute_and_raw_paths_invalidate_called_inputs(self):
        for call, module in (("::dep::used::run()", "used"), ("dep::r#type::run()", "type")):
            with self.subTest(call=call):
                files = runtime_files()
                files["crates/mid/src/lib.rs"] = f"pub fn run() {{ {call}; }}\n"
                if module == "type":
                    files["crates/dep/src/lib.rs"] += "pub mod r#type;\n"
                    files["crates/dep/src/type.rs"] = 'pub fn run() { println!("raw1"); }\n'
                ws = Workspace(self, files)
                ws.green("gate-called", "gate-linked")
                rel = f"crates/dep/src/{module}.rs"
                with ws.changed(rel, files[rel].replace("1", "2"), restore_mtime=True):
                    ws.assert_status(ws.keys("gate-called", "gate-linked", ledger=True),
                                     {"gate-called": "run", "gate-linked": "skip"})

    def test_nonstandard_target_root_keeps_called_inputs(self):
        files = runtime_files()
        del files["crates/app/src/bin/called.rs"]
        files["crates/app/Cargo.toml"] += '\n[[bin]]\nname = "called"\npath = "src/main.rust"\n'
        files["crates/app/src/main.rust"] = "fn main() { dep::used::run(); }\n"
        ws = Workspace(self, files)
        ws.green("gate-called", "gate-linked")
        rel = "crates/dep/src/used.rs"
        with ws.changed(rel, files[rel].replace("used1", "used2"), restore_mtime=True):
            ws.assert_status(ws.keys("gate-called", "gate-linked", ledger=True),
                             {"gate-called": "run", "gate-linked": "skip"})

    def test_known_source_macro_startup_and_implicit_impl_inputs(self):
        for kind in ("exported-startup", "exported-expression", "exported-import", "exported-proc-startup",
                     "local-relative-import", "local-drop", "shadowed-builtin"):
            with self.subTest(expansion=kind):
                files = source_macro_files(kind)
                ws = Workspace(self, files)
                baseline = ws.green("gate-called", "gate-linked")
                rel = "crates/dep/src/hidden.rs"
                self.assert_file(baseline["gate-called"], rel)
                self.assert_file(baseline["gate-called"], "crates/dep/src/startup.rs")
                if kind.startswith("exported-"):
                    self.assert_file(baseline["gate-called"], "crates/macros/src/lib.rs")
                if kind in ("local-drop", "exported-expression"):
                    self.assert_file(baseline["gate-called"], "crates/dep/src/unused.rs", False)
                with ws.changed(rel, files[rel].replace("source-macro1", "source-macro2"), restore_mtime=True):
                    ws.assert_status(ws.keys("gate-called", "gate-linked", ledger=True),
                                     {"gate-called": "run", "gate-linked": "skip"})

    def test_parameterized_source_macro_keeps_argument_bound_calls(self):
        for parameter, name in (("module", "call"), ("모듈", "call"), ("crate\u0301x", "call"), ("module", "실행")):
            with self.subTest(parameter=parameter, name=name):
                files = runtime_files()
                files["crates/app/src/bin/called.rs"] = "fn main() { dep::run(); }\n"
                files["crates/dep/src/lib.rs"] += (
                    "mod hidden;\n"
                    f"macro_rules! {name} {{ (${parameter}:ident) => {{ ${parameter}::run() }}; }}\n"
                    f"pub fn run() {{ {name}!(hidden); }}\n"
                )
                files["crates/dep/src/hidden.rs"] = 'pub fn run() { println!("hidden1"); }\n'
                ws = Workspace(self, files)
                ws.green("gate-called", "gate-linked")
                rel = "crates/dep/src/hidden.rs"
                with ws.changed(rel, files[rel].replace("hidden1", "hidden2"), restore_mtime=True):
                    rows = ws.keys("gate-called", "gate-linked", ledger=True)
                    ws.assert_status(rows, {"gate-called": "run", "gate-linked": "skip"})
                    self.assert_file(rows["gate-called"], rel)

    def test_features_and_test_roots(self):
        ws = Workspace(self, runtime_files())
        items = ("gate-called", "gate-feature", "gate-test")
        baseline = ws.green(*items)
        self.assertEqual(len({row[0] for row in baseline.values()}), len(items))
        for rel, affected in (("crates/app/src/bin/extra/mod.rs", {"gate-feature"}),
                              ("crates/dep/src/testonly.rs", {"gate-test"})):
            with self.subTest(rel=rel), ws.changed(rel, (ws.root / rel).read_text() + "// changed\n"):
                ws.assert_status(ws.keys(*items, ledger=True),
                                 {i: "run" if i in affected else "skip" for i in items})

    def test_includes_and_literal_directory_children(self):
        files = runtime_files()
        files.update({
            "crates/dep/src/used.rs": (
                'pub fn run() {\n    println!("{}", include_str!("../data/message.txt"));\n'
                '    for p in std::fs::read_dir("tools/key-data").unwrap() { println!("{:?}", p); }\n}\n'),
            "crates/dep/data/message.txt": "included\n",
            "tools/key-data/first.txt": "child\n",
        })
        ws = Workspace(self, files)
        baseline = ws.green("gate-called", "gate-linked")
        for rel in ("crates/dep/data/message.txt", "tools/key-data/first.txt"):
            self.assert_file(baseline["gate-called"], rel)
            self.assert_file(baseline["gate-linked"], rel, False)
            with self.subTest(rel=rel), ws.changed(rel, "changed\n"):
                ws.assert_status(ws.keys("gate-called", "gate-linked", ledger=True),
                                 {"gate-called": "run", "gate-linked": "skip"})
        ws.write("tools/key-data/new.txt", "new child\n")
        ws.assert_status(ws.keys("gate-called", "gate-linked", ledger=True),
                         {"gate-called": "run", "gate-linked": "skip"})

    def test_directory_reads_refuse_enumeration_and_stat_errors(self):
        files = runtime_files()
        files["crates/dep/src/used.rs"] = (
            'pub fn run() { let _ = std::fs::read_dir("tools/key-data").unwrap(); }\n'
        )
        files["tools/key-data/child.txt"] = "held\n"
        ws = Workspace(self, files)
        baseline = ws.green("gate-called")
        self.assert_file(baseline["gate-called"], "tools/key-data/child.txt")
        # Inject at the OS boundary in a child: chmod refusals depend on the caller's privilege.
        driver = r'''
import importlib.util
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / "tools"))
spec = importlib.util.spec_from_file_location("loaded_recipes", "tools/recipes.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
mode = sys.argv[1]
target = str(Path.cwd() / "tools/key-data")
if mode == "stat":
    target += "/child.txt"
    original = os.stat
    def failing(path, *args, **kwargs):
        if isinstance(path, (str, bytes, os.PathLike)) and os.path.abspath(os.fsdecode(path)) == target:
            raise OSError(5, "directory child stat failed", target)
        return original(path, *args, **kwargs)
    os.stat = failing
else:
    original = os.scandir
    def failing(path):
        if os.path.abspath(os.fsdecode(path)) == target:
            raise PermissionError(13, "directory enumeration denied", target)
        return original(path)
    os.scandir = failing
raise SystemExit(module.main(["key", "--manifest", sys.argv[2], "--ledger", sys.argv[3], "gate-called"]))
'''
        for mode in ("enumeration", "stat"):
            with self.subTest(operation=mode):
                observed = ws.run(sys.executable, "-c", driver, mode, str(ws.manifest), str(ws.ledger))
                fields = observed.stdout.strip().split("\t", 3)
                self.assertEqual(fields[:3], ["-", "gate-called", "error"], observed.stdout + observed.stderr)
                self.assertIn("tools/key-data", fields[3])

    def test_missing_linked_source_and_include_are_named_errors(self):
        ws = Workspace(self, runtime_files())
        ws.green("gate-called", "gate-linked")
        rel = "crates/dep/src/unused.rs"
        (ws.root / rel).unlink()
        rows = ws.keys("gate-called", "gate-linked", ledger=True)
        ws.assert_status(rows, {"gate-called": "error", "gate-linked": "error"})
        for row in rows.values():
            self.assertIn("unused", row[2])
        ws.write(rel, "pub fn ignored() {}\n")
        with ws.changed("crates/dep/src/used.rs", 'pub fn run() { let _ = include_str!("../data/missing.txt"); }\n'):
            row = ws.keys("gate-called", ledger=True)["gate-called"]
            self.assertEqual(row[1], "error", row)
            self.assertIn("missing.txt", row[2])

    def test_non_git_checkout_content_invalidation(self):
        ws = Workspace(self, runtime_files(), git=False)
        ws.green("gate-called", "gate-linked")
        self.assertFalse((ws.root / ".git").exists())
        rel = "crates/dep/src/used.rs"
        with ws.changed(rel, (ws.root / rel).read_text().replace("used1", "used2"), restore_mtime=True):
            ws.assert_status(ws.keys("gate-called", "gate-linked", ledger=True),
                             {"gate-called": "run", "gate-linked": "skip"})

    def test_malformed_and_foreign_cache_recompute_safely(self):
        ws = Workspace(self, runtime_files())
        baseline = ws.green("gate-called", "gate-linked")
        cache = ws.home / ".cache"
        entries = [p for p in cache.rglob("*") if p.is_file()]
        self.assertTrue(entries, "actual key CLI must persist structural Reach state in private HOME")
        foreign_files = runtime_files()
        foreign_files["crates/mid/src/lib.rs"] = "pub fn run() { dep::unused::ignored(); }\n"
        foreign = Workspace(self, foreign_files)
        foreign.green("gate-called")
        foreign_entries = [p for p in (foreign.home / ".cache").rglob("*") if p.is_file()]
        self.assertTrue(foreign_entries, "second checkout must produce its own cache")
        foreign_payload = max((p.read_bytes() for p in foreign_entries), key=len)
        for label, payload in (("truncated", b'{"truncated":'),
                               ("foreign-version", b'{"version":"foreign","graph":{},"identity":"foreign"}\n'),
                               ("foreign-checkout", foreign_payload)):
            with self.subTest(cache=label):
                for path in entries:
                    path.write_bytes(payload)
                rows = ws.keys("gate-called", "gate-linked", ledger=True)
                ws.assert_status(rows, {"gate-called": "skip", "gate-linked": "skip"})
                for item in rows:
                    self.assertEqual((rows[item][0], rows[item][3]), (baseline[item][0], baseline[item][3]))
                rel = "crates/dep/src/used.rs"
                with ws.changed(rel, (ws.root / rel).read_text().replace("used1", "used2"), restore_mtime=True):
                    ws.assert_status(ws.keys("gate-called", "gate-linked", ledger=True),
                                     {"gate-called": "run", "gate-linked": "skip"})
                entries = [p for p in cache.rglob("*") if p.is_file()]

    def test_real_build_script_helper_input_and_workspace_build_dependency(self):
        files = runtime_files()
        files["Cargo.toml"] = files["Cargo.toml"].replace('"crates/dep"]', '"crates/dep", "crates/builddep"]')
        files["crates/dep/Cargo.toml"] += '[build-dependencies]\nbuilddep = { path = "../builddep" }\n'
        del files["crates/app/src/lib.rs"]  # exercise an actual bin-only package's build script
        files.update({
            "crates/dep/build.rs": 'mod build_helper;\nfn main() { build_helper::run(); builddep::run(); }\n',
            "crates/dep/build_helper.rs": 'pub fn run() { println!("cargo:rustc-env=KEY_INPUT={}", include_str!("build-input.txt").trim()); }\n',
            "crates/dep/build-input.txt": "value1\n",
            "crates/builddep/Cargo.toml": package("builddep"),
            "crates/builddep/src/lib.rs": 'mod helper;\npub fn run() { helper::run(); }\n',
            "crates/builddep/src/helper.rs": 'pub fn run() { println!("cargo:rustc-cfg=key_fixture"); }\n',
            # A bin-only package executes its build script too.
            "crates/app/build.rs": 'fn main() { println!("cargo:rustc-cfg=app_fixture"); }\n',
        })
        ws = Workspace(self, files)
        items = ("gate-called", "gate-linked")
        baseline = ws.green(*items)
        for rel in ("crates/dep/build.rs", "crates/dep/build_helper.rs", "crates/dep/build-input.txt",
                    "crates/builddep/src/lib.rs", "crates/builddep/src/helper.rs", "crates/app/build.rs"):
            for row in baseline.values():
                self.assert_file(row, rel)
            suffix = "// changed\n" if rel.endswith(".rs") else "changed\n"
            with self.subTest(rel=rel), ws.changed(rel, (ws.root / rel).read_text() + suffix):
                ws.assert_status(ws.keys(*items, ledger=True), {i: "run" for i in items})

    def test_real_attribute_derive_and_proc_macro_helper(self):
        files = macro_files()
        files["crates/app/Cargo.toml"] += 'macros = { path = "../macros" }\n'
        files["crates/app/src/bin/called.rs"] = (
            '#[macros::keep]\n#[derive(macros::Generated)]\nstruct Probe;\n'
            'fn main() { let _ = Probe; mid::run(); }\n')
        ws = Workspace(self, files)
        ws.green("gate-linked")
        with ws.changed("crates/macros/src/helper.rs", "pub fn observe() { panic!(); }\n"):
            ws.assert_status(ws.keys("gate-linked", ledger=True), {"gate-linked": "run"})
        baseline = ws.green("gate-called")
        for rel in ("crates/macros/src/lib.rs", "crates/macros/src/helper.rs", "crates/macros/macro-input.txt"):
            self.assert_file(baseline["gate-called"], rel)
            suffix = "// changed\n" if rel.endswith(".rs") else "changed\n"
            with self.subTest(rel=rel), ws.changed(rel, (ws.root / rel).read_text() + suffix):
                ws.assert_status(ws.keys("gate-called", ledger=True), {"gate-called": "run"})

    def test_script_read_keeps_data_also_named_by_unreached_rust(self):
        files = runtime_files()
        files["crates/dep/src/unused.rs"] = 'pub fn ignored() { let _ = "crates/dep/data/shared.txt"; }\n'
        files["crates/dep/data/shared.txt"] = "shared1\n"
        files["tools/read-shared.sh"] = "#!/bin/sh\ncat crates/dep/data/shared.txt\n"
        files["justfile"] = files["justfile"].replace(
            "gate-called:\n", "gate-called:\n    ./tools/box.sh 'bash tools/read-shared.sh'\n")
        ws = Workspace(self, files)
        baseline = ws.green("gate-called", "gate-linked")
        self.assert_file(baseline["gate-called"], "crates/dep/data/shared.txt")
        with ws.changed("crates/dep/data/shared.txt", "shared2\n", restore_mtime=True):
            ws.assert_status(ws.keys("gate-called", "gate-linked", ledger=True),
                             {"gate-called": "run", "gate-linked": "skip"})

    def test_procedural_startup_in_unreached_module_and_conditional_attribute(self):
        for kind in ("startup", "nonstandard-module", "conditional"):
            with self.subTest(expansion=kind):
                files = macro_files()
                files["crates/dep/Cargo.toml"] += '[dependencies]\nmacros = { path = "../macros" }\n'
                files["crates/dep/src/lib.rs"] += "pub mod hidden;\npub mod startup;\n"
                files["crates/dep/src/hidden.rs"] = 'pub fn run() { println!("hidden1"); }\n'
                if kind in ("startup", "nonstandard-module"):
                    source = "startup.rust" if kind == "nonstandard-module" else "startup.rs"
                    if kind == "nonstandard-module":
                        files["crates/dep/src/lib.rs"] = files["crates/dep/src/lib.rs"].replace(
                            "pub mod startup;", '#[path = "startup.rust"] pub mod startup;')
                    files["crates/dep/src/" + source] = '#[macros::startup]\nfn initialize() { crate::hidden::run(); }\n'
                else:
                    files["crates/dep/Cargo.toml"] += '[features]\ndefault = ["instrumented"]\ninstrumented = []\n'
                    files["crates/dep/src/startup.rs"] = ""
                    files["crates/dep/src/used.rs"] = '#[cfg_attr(feature = "instrumented", macros::inject)]\npub fn run() {}\n'
                ws = Workspace(self, files)
                baseline = ws.green("gate-called")
                self.assert_file(baseline["gate-called"], "crates/dep/src/hidden.rs")
                with ws.changed("crates/dep/src/hidden.rs", 'pub fn run() { println!("hidden2"); }\n', restore_mtime=True):
                    ws.assert_status(ws.keys("gate-called", ledger=True), {"gate-called": "run"})

    def test_negated_conditions_enum_globs_and_c_abi_callbacks_do_not_activate_unreached_code(self):
        files = runtime_files()
        files["crates/dep/src/used.rs"] = (
            'enum Choice { A }\npub fn run() {\n    use Choice::*;\n    let _ = A;\n    if !(false) { println!("used1"); }\n'
            '    while !(true) {}\n    let _callback: extern "C" fn() = callback;\n}\n'
            'extern "C" fn callback() {}\n')
        ws = Workspace(self, files)
        baseline = ws.green("gate-called")
        rel = "crates/dep/src/unused.rs"
        self.assert_file(baseline["gate-called"], rel, False)
        with ws.changed(rel, files[rel].replace("unused1", "unused2")):
            ws.assert_status(ws.keys("gate-called", ledger=True), {"gate-called": "skip"})

    def test_function_macro_startup_keeps_owner_and_forward_not_sibling_code(self):
        files = macro_files()
        files["Cargo.toml"] = files["Cargo.toml"].replace('"crates/macros"]', '"crates/macros", "crates/sibling"]')
        files["crates/app/Cargo.toml"] += 'sibling = { path = "../sibling" }\n'
        files["crates/mid/Cargo.toml"] += 'macros = { path = "../macros" }\n'
        files["crates/mid/src/lib.rs"] += "mod hidden;\nmod startup;\n"
        files["crates/mid/src/hidden.rs"] = 'pub fn run() { println!("hidden1"); }\n'
        files["crates/mid/src/startup.rs"] = "macros::boot!();\n"
        files["crates/macros/src/lib.rs"] += (
            '#[proc_macro]\npub fn boot(_: TokenStream) -> TokenStream {\n'
            '    startup(TokenStream::new(), "fn initialize() { crate::hidden::run(); }".parse().unwrap())\n}\n')
        files["crates/sibling/Cargo.toml"] = package("sibling")
        files["crates/sibling/src/lib.rs"] = 'pub fn unused() { println!("sibling1"); }\n'
        ws = Workspace(self, files)
        items = ("gate-called", "gate-linked")
        baseline = ws.green(*items)
        for rel in ("crates/mid/src/hidden.rs", "crates/dep/src/unused.rs"):
            self.assert_file(baseline["gate-called"], rel)
            with self.subTest(rel=rel), ws.changed(rel, files[rel] + "// edited\n"):
                ws.assert_status(ws.keys(*items, ledger=True), {"gate-called": "run", "gate-linked": "skip"})
        rel = "crates/sibling/src/lib.rs"
        self.assert_file(baseline["gate-called"], rel, False)
        with ws.changed(rel, files[rel].replace("sibling1", "sibling2")):
            ws.assert_status(ws.keys(*items, ledger=True), {item: "skip" for item in items})

    def test_unused_import_keeps_linked_crate_startup_inputs(self):
        for imported in ("extern crate sibling as _;", "use sibling as _;", "use sibling::Marker as _;",
                         "use sibling::Marker;", "use sibling::{Marker as _};", "use sibling::*;"):
            with self.subTest(imported=imported):
                files = macro_files()
                files["Cargo.toml"] = files["Cargo.toml"].replace('"crates/macros"]', '"crates/macros", "crates/sibling"]')
                files["crates/app/Cargo.toml"] += 'sibling = { path = "../sibling" }\n'
                files["crates/app/src/bin/called.rs"] = imported + "\n" + files["crates/app/src/bin/called.rs"]
                files["crates/sibling/Cargo.toml"] = package("sibling", '[dependencies]\nmacros = { path = "../macros" }\n')
                files["crates/sibling/src/lib.rs"] = "pub struct Marker;\nmod hidden;\nmod startup;\n"
                files["crates/sibling/src/hidden.rs"] = 'pub fn run() { println!("side1"); }\n'
                files["crates/sibling/src/startup.rs"] = '#[macros::startup]\nfn initialize() { crate::hidden::run(); }\n'
                ws = Workspace(self, files)
                baseline = ws.green("gate-called", "gate-linked")
                rel = "crates/sibling/src/hidden.rs"
                self.assert_file(baseline["gate-called"], rel)
                self.assert_file(baseline["gate-linked"], rel, False)
                with ws.changed(rel, files[rel].replace("side1", "side2"), restore_mtime=True):
                    ws.assert_status(ws.keys("gate-called", "gate-linked", ledger=True),
                                     {"gate-called": "run", "gate-linked": "skip"})

    def test_crlf_and_bare_cr_sources_remain_valid_and_content_keyed(self):
        for newline in ("\r\n", "\r"):
            with self.subTest(newline=repr(newline)):
                files = {path: text.replace("\n", newline) if path.endswith(".rs") else text
                         for path, text in runtime_files().items()}
                ws = Workspace(self, files)
                ws.green("gate-called", "gate-linked")
                rel = "crates/dep/src/used.rs"
                with ws.changed(rel, files[rel].replace("used1", "used2"), restore_mtime=True):
                    ws.assert_status(ws.keys("gate-called", "gate-linked", ledger=True),
                                     {"gate-called": "run", "gate-linked": "skip"})

    def test_repeated_key_validation_reuses_stable_source_hashes(self):
        ws = Workspace(self, runtime_files())
        driver = r'''
import builtins
import importlib.util
import os
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path.cwd() / "tools"))
spec = importlib.util.spec_from_file_location("loaded_recipes", "tools/recipes.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.HashCache.RACY_NS = 0
side = module.make_side(str(Path.cwd()))
context = module.KeyContext(side, sys.argv[1])
before, error, _, _ = context.parts("gate-called")
assert before and error is None, error
source = Path.cwd() / "crates/dep/src/used.rs"
original = source.read_bytes()
state = source.stat()
read = builtins.open
reads = []
def counted(path, *args, **kwargs):
    if str(path) == str(source):
        reads.append(path)
    return read(path, *args, **kwargs)
with patch.object(builtins, "open", counted):
    side.graph.key_inputs("gate-called")
assert not reads, f"unchanged source rehashed during repeated key validation: {len(reads)} reads"
source.write_bytes(original.replace(b"used1", b"used2"))
os.utime(source, ns=(state.st_atime_ns, state.st_mtime_ns))
after, error, _, _ = context.parts("gate-called")
assert after and error is None and after != before, error
print("stable source hash reused; same-size/mtime edit invalidated its consumer")
'''
        ws.run(sys.executable, "-c", driver, str(ws.manifest))

    def test_snapshot_paths_keep_symlink_referents_and_recheck_parents(self):
        ws = Workspace(self, runtime_files())
        driver = r'''
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path.cwd() / "tools"))
from bloomery import reach_cache
root = Path.cwd()
for directory in ("left", "right"):
    (root / directory).mkdir()
    for name in ("a.rs", "b.rs"):
        (root / directory / name).write_text("same contents\n")
(root / "parent").symlink_to("left", target_is_directory=True)
(root / "leaf.rs").symlink_to("parent/a.rs")
tree = SimpleNamespace(root=str(root))
saw = {("r", "parent/a.rs"): ("same contents\n",),
       ("r", "parent/b.rs"): ("same contents\n",),
       ("r", "leaf.rs"): ("same contents\n",),
       ("f", "parent/absent.rs"): False,
       ("d", "parent/../right"): True}
before = reach_cache._observed(tree, saw)
for _, rel, value in before:
    actual = value[0] if isinstance(value[0], str) else value[1]
    assert actual == os.path.realpath(root / rel), (rel, actual)
(root / "next").symlink_to("right", target_is_directory=True)
os.replace(root / "next", root / "parent")
after = reach_cache._observed(tree, saw)
assert before != after, "retargeting a parent with identical contents must change provenance"
read = reach_cache._bytes
def retarget(path):
    data = read(path)
    if path == str(root / "parent/b.rs"):
        (root / "next").symlink_to("left", target_is_directory=True)
        os.replace(root / "next", root / "parent")
    return data
with patch.object(reach_cache, "_bytes", retarget):
    try:
        reach_cache._observed(tree, saw)
    except OSError as error:
        assert "source parent changed" in str(error), error
    else:
        raise AssertionError("parent retargeted during a snapshot was accepted")
print("symlink referents, missing probes, parent retarget and mid-snapshot retarget checked")
'''
        ws.run(sys.executable, "-c", driver)

    def test_analysis_code_edited_after_import_never_emits_a_skippable_key(self):
        for relative in ("tools/recipes.py", "tools/bloomery/reach_cache.py", "tools/bloomery/fixture_dirs.py"):
            with self.subTest(analysis=relative):
                ws = Workspace(self, runtime_files())
                ws.green("gate-called")
                driver = r'''
import contextlib
import io
import os
import importlib.util
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / "tools"))
spec = importlib.util.spec_from_file_location("loaded_recipes", "tools/recipes.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from bloomery import reach_cache
path = Path(sys.argv[1])
before = path.read_bytes()
state = path.stat()
with contextlib.redirect_stdout(io.StringIO()) as baseline:
    result = module.main(["key", "--manifest", sys.argv[2], "--ledger", sys.argv[3], "gate-called"])
assert result == 0 and baseline.getvalue().strip().split("\t")[2] == "skip", baseline.getvalue()
try:
    changed = before.replace(b"_REACH_LOADED_CODE", b"_REACH_EDITED_CODE", 1)
    assert changed != before and len(changed) == len(before)
    path.write_bytes(changed)
    os.utime(path, ns=(state.st_atime_ns, state.st_mtime_ns))
    result = module.main(["key", "--manifest", sys.argv[2], "--ledger", sys.argv[3], "gate-called"])
finally:
    path.write_bytes(before)
    os.utime(path, ns=(state.st_atime_ns, state.st_mtime_ns))
raise SystemExit(result)
'''
                observed = ws.run(sys.executable, "-c", driver, relative, str(ws.manifest), str(ws.ledger))
                row = observed.stdout.strip().split("\t", 3)
                self.assertEqual(row[:3], ["-", "gate-called", "error"], observed.stdout + observed.stderr)
                self.assertIn(relative.rsplit("/", 1)[1], row[3])

    def test_module_added_between_eager_discovery_and_key_is_included_or_refused(self):
        ws = Workspace(self, runtime_files())
        driver = r'''
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / "tools"))
spec = importlib.util.spec_from_file_location("loaded_recipes", "tools/recipes.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
side = module.make_side(str(Path.cwd()))
module.reach_of(side.tree)
Path("crates/dep/src/lib.rs").write_text(Path("crates/dep/src/lib.rs").read_text() + "pub mod fresh;\n")
Path("crates/dep/src/used.rs").write_text("pub fn run() { crate::fresh::run(); }\n")
Path("crates/dep/src/fresh.rs").write_text("pub fn run() {}\n")
context = module.KeyContext(side, sys.argv[1])
parts, error, never, summary = context.parts("gate-called")
print(json.dumps({"parts": parts, "error": error, "never": never}))
'''
        observed = ws.run(sys.executable, "-c", driver, str(ws.manifest))
        import json
        result = json.loads(observed.stdout)
        if result["error"] is not None:
            self.assertFalse(result["parts"], result)
        else:
            self.assertTrue(any(part.startswith("file\tcrates/dep/src/fresh.rs\t") for part in result["parts"]), result)


def main() -> int:
    global SOURCE_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT,
                        help="checkout whose production CLI/helpers are copied into each fixture")
    args = parser.parse_args()
    SOURCE_ROOT = args.source_root.resolve()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecipeKeyCLI)
    filters = [s for s in os.environ.get("RECIPES_KEY_TEST_FILTER", "").split(",") if s]
    if filters:
        suite = unittest.TestSuite(test for test in suite if any(s in test.id() for s in filters))
        if not suite.countTestCases():
            parser.error("RECIPES_KEY_TEST_FILTER matched no test")
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
