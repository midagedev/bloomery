#!/usr/bin/env bash
# The oracle trees' identity check (tools/ref/emb2/oracle-tree.sh), against real git repositories in temp directories: no box,
# no model, no card. A tree is an upstream commit with a patch on it or without; the check refuses another commit, another
# patch, a tree that moved after its build, and reports each by name with the exit code 2 every caller relies on.
# The binary-age cases need Linux, ldd and a C compiler (the box has them); elsewhere they are a named skip.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 - "$@" <<'PY'
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path.cwd()
LIB = ROOT / "tools/ref/emb2/oracle-tree.sh"
ENV = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null", "GIT_AUTHOR_NAME": "t",
       "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def git(tree, *args):
    return subprocess.run(["git", "-C", str(tree), *args], env=ENV, check=True, capture_output=True, text=True).stdout.strip()


def write(path, text, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(0o755)


def run(*args):
    """oracle_tree/oracle_patch_hash run in a fresh bash that sources the owner: its exit code and its stderr."""
    script = f'source {LIB}; "$@"'
    r = subprocess.run(["bash", "-c", script, "_", *args], env=ENV, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


class Tree(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="oracle-tree-")).resolve()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.tree = self.tmp / "ik"
        self.tree.mkdir()
        git(self.tree, "init", "-q")
        write(self.tree / ".git/info/exclude", "build/\n")
        for name, text in (("src/unicode.cpp", "int a = 1;\n"), ("include/llama.h", "// api\n"),
                           ("examples/tokenize/tokenize.cpp", "int main() {}\n"), ("docs/readme.md", "docs\n")):
            write(self.tree / name, text)
        git(self.tree, "add", "-A")
        git(self.tree, "commit", "-qm", "base")
        self.base = git(self.tree, "rev-parse", "HEAD")

    def patch(self, text="int a = 2;\n", commit=True):
        write(self.tree / "src/unicode.cpp", text)
        if commit:
            git(self.tree, "commit", "-qam", "patch")
        return run("oracle_patch_hash", str(self.tree), self.base)[1]

    def check(self, *args, tree=None):
        return run("oracle_tree", *args, "who", str(tree or self.tree), self.base, "llama-tokenize")

    def stub(self, report, tree=None):
        write((tree or self.tree) / "build/bin/llama-tokenize", f"#!/bin/sh\necho '{report}'\n", True)

    # ---- a tree that is the commit
    def test_the_commit_with_no_tracked_change_is_the_tree(self):
        self.stub(f"version: 1 (build 2, commit {self.base[:9]})")
        write(self.tree / "build/untracked.o", "x")
        self.assertEqual(self.check(), (0, "", ""))

    def test_another_head_or_a_tracked_change_is_refused_by_name(self):
        self.stub(f"version: 1 (build 2, commit {self.base[:9]})")
        write(self.tree / "src/unicode.cpp", "int a = 9;\n")
        rc, _, err = self.check()
        self.assertEqual((rc, err), (2, f"who: {self.tree} has tracked changes; the oracle is the commit's build"))
        git(self.tree, "commit", "-qam", "moved")
        head = git(self.tree, "rev-parse", "HEAD")
        rc, _, err = self.check()
        self.assertEqual((rc, err), (2, f"who: {self.tree} is at {head}, the profile pins {self.base}"))

    # ---- a tree that is a commit plus a patch
    def test_a_patch_is_named_by_its_hash_and_the_tree_holds_exactly_it(self):
        patch = self.patch()
        self.assertRegex(patch, r"^[0-9a-f]{12}$")
        self.stub(f"version: 1 (build 2, commit {git(self.tree, 'rev-parse', 'HEAD')[:9]})")
        self.assertEqual(self.check("--patch", patch), (0, "", ""))
        other = "0" * 12 if patch != "0" * 12 else "1" * 12
        rc, _, err = self.check("--patch", other)
        self.assertEqual((rc, err), (2, f"who: {self.tree}'s tracked files differ from {self.base[:9]} by patch {patch}, "
                                        f"the profile pins patch {other}"))
        # The commit alone, with the patch's tree in it, is not the commit's build either.
        rc, _, err = self.check()
        self.assertEqual(rc, 2, err)

    def test_the_patch_is_its_content_not_its_commit(self):
        # Uncommitted, committed, and committed again by someone else (other metadata, a clone) are one patch.
        patch = self.patch(commit=False)
        self.stub(f"version: 1 (build 2, commit {self.base[:9]})")
        self.assertEqual(self.check("--patch", patch), (0, "", ""))
        git(self.tree, "commit", "-qam", "patch")
        clone = self.tmp / "clone"
        git(self.tmp, "clone", "-q", str(self.tree), str(clone))
        git(clone, "commit", "-q", "--amend", "-m", "the same patch, said otherwise", "--date=2001-01-01T00:00:00")
        self.assertNotEqual(git(clone, "rev-parse", "HEAD"), git(self.tree, "rev-parse", "HEAD"))
        self.stub(f"version: 1 (build 2, commit {git(clone, 'rev-parse', 'HEAD')[:9]})", clone)
        self.assertEqual(self.check("--patch", patch, tree=clone), (0, "", ""))
        self.assertEqual(run("oracle_patch_hash", str(clone), self.base)[1], patch)

    def test_any_other_change_to_the_tracked_files_is_another_patch(self):
        patch = self.patch()
        self.stub(f"version: 1 (build 2, commit {git(self.tree, 'rev-parse', 'HEAD')[:9]})")
        for name, change in (("an edit on top", ("src/unicode.cpp", "int a = 3;\n")),
                             ("whitespace only", ("src/unicode.cpp", "int  a = 2;\n")),
                             ("a new tracked file", ("src/new.cpp", "x\n"))):
            with self.subTest(change=name):
                before = (self.tree / change[0]).read_text() if (self.tree / change[0]).exists() else None
                write(self.tree / change[0], change[1])
                git(self.tree, "add", "-A")
                rc, _, err = self.check("--patch", patch)
                self.assertEqual(rc, 2, err)
                self.assertIn("by patch", err)
                if before is None:
                    git(self.tree, "rm", "-qf", change[0])
                else:
                    write(self.tree / change[0], before)
                git(self.tree, "add", "-A")
                self.assertEqual(self.check("--patch", patch)[0], 0)

    def test_the_hash_does_not_depend_on_the_git_config(self):
        patch = self.patch()
        for key, value in (("diff.noprefix", "true"), ("diff.mnemonicPrefix", "true"), ("diff.algorithm", "patience"),
                           ("diff.context", "9"), ("diff.renames", "copies"), ("core.quotepath", "true")):
            git(self.tree, "config", key, value)
        self.assertEqual(run("oracle_patch_hash", str(self.tree), self.base)[1], patch)

    def test_a_pin_the_tree_does_not_descend_from_is_refused_by_name(self):
        patch = self.patch()
        self.stub("version: 1 (build 2, commit abcdefabc)")
        other = self.tmp / "other"
        other.mkdir()
        git(other, "init", "-q")
        write(other / "src/unicode.cpp", "unrelated\n")
        git(other, "add", "-A")
        git(other, "commit", "-qm", "unrelated")
        unrelated = git(other, "rev-parse", "HEAD")
        git(self.tree, "fetch", "-q", str(other))
        head = git(self.tree, "rev-parse", "HEAD")
        rc, _, err = run("oracle_tree", "--patch", patch, "who", str(self.tree), unrelated, "llama-tokenize")
        self.assertEqual((rc, err), (2, f"who: {self.tree} is at {head}, which does not descend from the pinned commit {unrelated}"))
        missing = "1" * 40
        rc, _, err = run("oracle_tree", "--patch", patch, "who", str(self.tree), missing, "llama-tokenize")
        self.assertEqual((rc, err), (2, f"who: {self.tree} does not hold the pinned commit {missing}"))

    # ---- the binary
    def test_a_binary_that_reports_another_commit_or_none_is_refused(self):
        patch = self.patch()
        head = git(self.tree, "rev-parse", "HEAD")
        for report in ("version: 1 (build 2, commit unknown)", f"version: 1 (build 2, commit {self.base[:9]})",
                       "version: 1 (build 2)"):
            with self.subTest(report=report):
                self.stub(report)
                rc, _, err = self.check("--patch", patch)
                self.assertEqual((rc, err), (2, f"who: {self.tree}/build/bin/llama-tokenize is not built at {head[:7]}: {report}"))
        os.remove(self.tree / "build/bin/llama-tokenize")
        rc, _, err = self.check("--patch", patch, "--built-by", "just build-x")
        self.assertEqual((rc, err), (2, f"who: no {self.tree}/build/bin/llama-tokenize (just build-x)"))

    def build_elf(self):
        """A llama-tokenize that loads a libllama.so, both under the tree's build/."""
        if sys.platform != "linux" or not shutil.which("ldd") or not shutil.which("cc"):
            self.skipTest("the binary-age cases need Linux, ldd and cc")
        src = self.tmp / "src"
        write(src / "lib.c", "int llama_x(void) { return 1; }\n")
        write(src / "main.c", "int llama_x(void);\nint main(void) { return llama_x() - 1; }\n")
        lib = self.tree / "build/src"
        lib.mkdir(parents=True)
        subprocess.run(["cc", "-shared", "-fPIC", "-o", str(lib / "libllama.so"), str(src / "lib.c")], check=True)
        (self.tree / "build/bin").mkdir(parents=True, exist_ok=True)
        subprocess.run(["cc", "-o", str(self.tree / "build/bin/llama-tokenize"), str(src / "main.c"), f"-L{lib}", "-lllama",
                        f"-Wl,-rpath,{lib}"], check=True)
        return lib / "libllama.so", self.tree / "build/bin/llama-tokenize"

    def age(self, built, sources, built_at=2_000_000_000, source_at=1_900_000_000):
        for name in sources:
            os.utime(self.tree / name, (source_at, source_at))
        for path in built:
            os.utime(path, (built_at, built_at))

    def test_a_build_older_than_the_tree_is_refused_when_the_binary_reports_no_commit(self):
        patch = self.patch()
        lib, exe = self.build_elf()
        every = [p for p in git(self.tree, "ls-files").splitlines()]
        self.age((lib, exe), every)
        self.assertEqual(self.check("--patch", patch, "--unversioned"), (0, "", ""))
        # Only the sources of what the binary links count; a doc touched after the build does not.
        os.utime(self.tree / "docs/readme.md", (2_100_000_000,) * 2)
        self.assertEqual(self.check("--patch", patch, "--unversioned")[0], 0)
        for name in ("src/unicode.cpp", "include/llama.h", "examples/tokenize/tokenize.cpp"):
            with self.subTest(moved=name):
                os.utime(self.tree / name, (2_100_000_000,) * 2)
                rc, _, err = self.check("--patch", patch, "--unversioned")
                self.assertEqual(rc, 2, err)
                self.assertRegex(err, rf"who: {self.tree}/build/(bin/llama-tokenize|src/libllama.so) is older than {self.tree}/{name}, ")
                os.utime(self.tree / name, (1_900_000_000,) * 2)
        os.utime(lib, (1_800_000_000,) * 2)
        rc, _, err = self.check("--patch", patch, "--unversioned")
        self.assertEqual(rc, 2, err)
        self.assertIn("libllama.so is older than", err)

    def test_a_listing_of_the_sources_that_fails_is_a_refusal_not_a_pass(self):
        patch = self.patch()
        lib, exe = self.build_elf()
        self.age((lib, exe), git(self.tree, "ls-files").splitlines())
        real = shutil.which("git")
        write(self.tmp / "bin/git", f'#!/bin/sh\ncase " $* " in *" ls-files "*) exit 128 ;; esac\nexec {real} "$@"\n', True)
        env = {**ENV, "PATH": f"{self.tmp / 'bin'}:{ENV['PATH']}"}
        r = subprocess.run(["bash", "-c", f'source {LIB}; oracle_tree --patch {patch} --unversioned who {self.tree} {self.base} llama-tokenize'],
                           env=env, capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stderr.strip()), (2, f"who: cannot list the tracked sources of {self.tree}"))

    def test_a_binary_that_loads_no_libllama_from_the_tree_is_refused(self):
        patch = self.patch()
        if sys.platform != "linux" or not shutil.which("ldd"):
            self.skipTest("needs Linux and ldd")
        write(self.tree / "build/bin/llama-tokenize", "#!/bin/sh\n", True)
        rc, _, err = self.check("--patch", patch, "--unversioned")
        self.assertEqual((rc, err), (2, f"who: {self.tree}/build/bin/llama-tokenize loads no libllama.so from {self.tree}"))


unittest.main(argv=["check-oracle-tree"], verbosity=2)
PY
