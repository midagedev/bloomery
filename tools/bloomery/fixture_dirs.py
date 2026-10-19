"""The fixture tier's family table, tools/fixture-dirs.tsv, and the default profile of tools/ref/ref-paths.sh: the one Python reader.

tools/recipes.py and tools/gate-batch.sh import it; crates/refset/src/fixture.rs and the shell read the same file. The table is

    # comment
    qwen4exp<TAB>qwen38
    deepseek2<TAB>self

A family the table does not name has no fixture (`none`, never a row). A text this reader does not understand is a named error, never
a guess: a row of another shape would be read as a family and fall to `none`.
"""

import re

FIXTURE_DIRS = "tools/fixture-dirs.tsv"
REF_PATHS = "tools/ref/ref-paths.sh"
FAMILY_NAME = re.compile(r"[A-Za-z0-9_-]+")
FIXTURE_DIR_NAME = re.compile(r"[a-z0-9-]+")
PROFILE_DEFAULT_LINE = re.compile(r'^: "\$\{BLOOMERY_MODEL:=\$\{BLOOMERY_REF_MODEL_PROFILE:-([A-Za-z0-9_]+)\}\}"$', re.M)


class RecipeError(Exception):
    """A recipe, a tree or a tool the parser cannot read: a named failure, never a guess."""


def fixture_table(dirs_text: str, ref_paths_text: str) -> tuple[dict[str, str], str]:
    """({family: fixture directory}, default profile): the rows of the fixture directory table's text (`self` as it is) and the profile
    tools/box.sh loads when BLOOMERY_MODEL is unset, from ref-paths.sh's text."""
    rows: dict[str, str] = {}
    for n, line in enumerate(dirs_text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 2 or not FAMILY_NAME.fullmatch(cols[0]) or not (FIXTURE_DIR_NAME.fullmatch(cols[1]) and cols[1] != "none"):
            raise RecipeError(f"{FIXTURE_DIRS}: line {n} ({line.strip()[:60]!r}) is not `<family><TAB><dir>`: a family name, then a directory of lower-case letters, digits and '-' (or `self`; never `none`)")
        if cols[0] in rows:
            raise RecipeError(f"{FIXTURE_DIRS}: line {n}: the family {cols[0]!r} is in the table twice")
        rows[cols[0]] = cols[1]
    if not rows:
        raise RecipeError(f"{FIXTURE_DIRS}: no row (`<family><TAB><dir>`): an item's fixture directory is read there")
    profile = PROFILE_DEFAULT_LINE.findall(ref_paths_text)
    if len(profile) != 1:
        raise RecipeError(f"{REF_PATHS} holds {len(profile)} default-profile lines, not one: an item with no BLOOMERY_MODEL runs under that default")
    return rows, profile[0]
