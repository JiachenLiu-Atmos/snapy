#!/usr/bin/env python3
"""Code citations by symbol (README, "Citing code"): resolve them at the pin, gate them, move the pin.

A chapter cites snapy code with the shortcode of `book/_shortcodes/code.lua`:

    {{< code src/hydro/hydro_forward.cpp HydroImpl::_flux_covariance >}}
    {{< code src/hydro/hydro.hpp HydroImpl::forward overload=2 >}}   # the 2nd of two overloads, in file order
    {{< code python/snap/__init__.py >}}                              # a file, no symbol

It renders as the link `hydro_forward.cpp` `HydroImpl::_flux_covariance` to the symbol's lines in the file at
the pinned commit of `code_pin.yml`. The line range is never typed: this script finds it at the pin, with
Universal Ctags (end lines) for C, C++ and CUDA and Python's `ast` for `.py`, and extends it up over the
comment block (and C++ `template` or return-type line) right above the symbol. A C++ symbol is named with its
classes and without its namespaces (`HydroImpl::forward`, not `snap::HydroImpl::forward`), a gtest test as
`Suite.Name`, an `ADD_ARG(type, name)` option as `Class::name`, a Python symbol as `Class.method`. Where a file
holds several symbols of that name, the one definition among declarations is taken; otherwise `overload=N`
picks the N-th in file order, and its absence is an error.

    python3 tools/code_anchors.py [check]           # resolve every citation at the pin, write the table the
                                                    # shortcode reads (book/_code_anchors.json); exit 1 if any
                                                    # citation does not resolve (the render gate)
    python3 tools/code_anchors.py diff OLD NEW      # cited symbols whose body differs between two commits
    python3 tools/code_anchors.py bump NEW          # diff from the pin to NEW, then write NEW into code_pin.yml
                                                    # (only if every citation resolves at NEW)

Legacy chapters (`tools/code-anchors-legacy.txt`, one chapter id per line, as in tools/chapter-owners.tsv): a
citation in a listed chapter that does not resolve at the pin is reported, not failed, and renders as plain text.
Every other chapter is checked in full. A chapter PR deletes its own line and must then pass in full; a listed
chapter with nothing unresolved fails `check` until its line is deleted. The report is complete only when the list
is empty. `check` prints the unresolved count left per chapter and per owner.

Sources come from a local clone that has the commit (`SNAPY_REPO`, then the repository holding this report),
else from a shallow fetch of the pinned repository into `SNAPY_REPORT_CACHE` (default ~/.cache/snapy-report).
`CTAGS` names the Universal Ctags binary (default `ctags` on PATH); CI builds the version in CTAGS_VERSION.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import io
from dataclasses import dataclass
from pathlib import Path

REPORT = Path(__file__).resolve().parent.parent
BOOK = REPORT / "book"
PIN_FILE = REPORT / "code_pin.yml"
LEGACY_FILE = REPORT / "tools" / "code-anchors-legacy.txt"
TABLE = BOOK / "_code_anchors.json"
CTAGS_VERSION = "6.2.1"  # the version CI builds; ranges are resolved with it
CACHE = Path(os.environ.get("SNAPY_REPORT_CACHE", Path.home() / ".cache" / "snapy-report"))

SHORTCODE = re.compile(r"(?<!\{)\{\{<\s*code\s+(?P<args>[^>]*?)\s*>\}\}")
C_SUFFIXES = {".c", ".h", ".cc", ".cpp", ".hpp", ".cxx", ".hxx", ".cu", ".cuh"}
# ctags kinds a citation may name; namespaces, lambdas (`__anon...`) and anything local to a function are not
GTEST = {"TEST", "TEST_F", "TEST_P", "TYPED_TEST", "TYPED_TEST_P"}
C_KINDS = {"function", "prototype", "class", "struct", "union", "enum", "enumerator", "member", "variable",
           "typedef", "macro", "alias"}


@dataclass(frozen=True)
class Symbol:
    name: str       # as cited: HydroImpl::forward, Class.method
    kind: str
    start: int
    end: int
    signature: str


@dataclass(frozen=True)
class Cite:
    path: str
    symbol: str | None
    overload: int | None

    @property
    def key(self) -> str:  # the key the shortcode looks up (book/_shortcodes/code.lua)
        return f"{self.path}|{self.symbol or ''}|{self.overload or ''}"

    def shortcode(self) -> str:
        parts = [self.path] + ([self.symbol] if self.symbol else []) + (
            [f"overload={self.overload}"] if self.overload else [])
        return "{{< code " + " ".join(parts) + " >}}"


class Unresolved(Exception):
    pass


# ---------------------------------------------------------------- the pin and the sources

def read_pin(path: Path = PIN_FILE) -> tuple[str, str]:
    fields = dict(re.findall(r"^(repo|sha):\s*(\S+)", path.read_text(), re.M))
    if not re.fullmatch(r"[0-9a-f]{40}", fields.get("sha", "")) or "repo" not in fields:
        sys.exit(f"{path}: needs `repo: owner/name` and `sha: <40 hex>`")
    return fields["repo"], fields["sha"]


def write_pin(sha: str, path: Path = PIN_FILE) -> None:
    path.write_text(re.sub(r"^sha:\s*\S+", f"sha: {sha}", path.read_text(), count=1, flags=re.M))


def _git(gitdir: Path, *args: str, binary: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(gitdir), *args], capture_output=True, text=not binary)


def _has_commit(gitdir: Path, sha: str) -> bool:
    return _git(gitdir, "cat-file", "-e", f"{sha}^{{commit}}").returncode == 0


def find_commit(repo: str, sha: str) -> Path:
    """A git directory holding `sha`: a local clone, else the cache, fetched from GitHub if needed."""
    local = [Path(p) for p in [os.environ.get("SNAPY_REPO", "")] if p]
    top = _git(REPORT, "rev-parse", "--show-toplevel")
    if top.returncode == 0:
        local.append(Path(top.stdout.strip()))
    for d in local:
        if _has_commit(d, sha):
            return d
    bare = CACHE / (repo.replace("/", "__") + ".git")
    if not (bare / "HEAD").exists():
        bare.mkdir(parents=True, exist_ok=True)
        _git(bare, "init", "--quiet", "--bare")
    if not _has_commit(bare, sha):
        p = _git(bare, "fetch", "--quiet", "--depth=1", f"https://github.com/{repo}.git", sha)
        if p.returncode != 0 or not _has_commit(bare, sha):
            sys.exit(f"cannot fetch {repo}@{sha}: {p.stderr.strip()}")
    return bare


def checkout(repo: str, sha: str) -> Path:
    """The source tree of `sha`, extracted once into the cache."""
    tree = CACHE / "trees" / sha
    if not (tree / ".complete").exists():
        gitdir = find_commit(repo, sha)
        data = _git(gitdir, "archive", "--format=tar", sha, binary=True)
        if data.returncode != 0:
            sys.exit(f"git archive {sha} failed: {data.stderr.decode().strip()}")
        tree.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data.stdout)) as tar:
            tar.extractall(tree, filter="data")
        (tree / ".complete").write_text(sha + "\n")
    return tree


# ---------------------------------------------------------------- symbols in a tree

def ctags_binary() -> str:
    exe = os.environ.get("CTAGS", "ctags")
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout
    except OSError:
        out = ""
    if not out.startswith("Universal Ctags"):
        sys.exit(f"`{exe}` is not Universal Ctags (set CTAGS); README, 'Citing code'")
    ver = re.match(r"Universal Ctags (\d+\.\d+\.\d+)", out).group(1)
    if ver != CTAGS_VERSION:
        print(f"warning: Universal Ctags {ver}; CI resolves with {CTAGS_VERSION}", file=sys.stderr)
    return exe


def _strip_namespaces(full: str, namespaces: set[str]) -> str:
    parts = full.split("::")
    i = 0
    while i < len(parts) - 1 and ("::".join(parts[:i + 1]) in namespaces or parts[i].startswith("__anon")):
        i += 1
    return "::".join(parts[i:])


def c_symbols(tree: Path) -> dict[str, list[Symbol]]:
    out = subprocess.run(
        [ctags_binary(), "-R", "-f", "-", "--excmd=number", "--sort=no", "--languages=C,C++,CUDA",
         "--fields=+neKSZ", "--kinds-C=+p", "--kinds-C++=+p", "--kinds-CUDA=+p", "."],
        cwd=tree, capture_output=True, text=True, check=True).stdout
    rows = []
    namespaces = set()
    for line in out.splitlines():
        if line.startswith("!_"):
            continue
        name, path, rest = line.split("\t", 2)
        fields = dict(f.split(":", 1) for f in rest.split(';"\t', 1)[1].split("\t") if ":" in f)
        kind = rest.split(';"\t', 1)[1].split("\t", 1)[0]
        scope = fields.get("scope", "")
        skind, _, spath = scope.partition(":")
        full = f"{spath}::{name}" if spath else name
        if kind == "namespace":
            namespaces.add(full)
        rows.append((path.removeprefix("./"), name, full, kind, skind, spath, fields))
    syms: dict[str, list[Symbol]] = {}
    for path, name, full, kind, skind, spath, fields in rows:
        cited = _strip_namespaces(full, namespaces)
        args = [x.strip() for x in fields.get("signature", "").strip("() ").split(",")]
        if name in GTEST and len(args) == 2:      # TEST(Suite, Name) is cited as Suite.Name, gtest's own name
            cited = f"{args[0]}.{args[1]}"
        elif name == "ADD_ARG" and len(args) >= 2:  # ADD_ARG(type, name) in an options class: Class::name
            cited = cited[:-len(name)] + args[-1]
        if kind not in C_KINDS or skind == "function" or "__anon" in cited:
            continue
        start = int(fields["line"])
        syms.setdefault(path, []).append(Symbol(cited, kind, start, int(fields.get("end", start)),
                                                fields.get("signature", "")))
    return syms


def py_symbols(source: str) -> list[Symbol]:
    out = []

    def walk(body, prefix):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                sig = "" if kind == "class" else "(" + ast.unparse(node.args) + ")"
                out.append(Symbol(prefix + node.name, kind, start, node.end_lineno, sig))
                if isinstance(node, ast.ClassDef):
                    walk(node.body, prefix + node.name + ".")
            elif not prefix and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        out.append(Symbol(t.id, "variable", node.lineno, node.end_lineno, ""))

    try:
        walk(ast.parse(source).body, "")
    except SyntaxError:
        pass
    return out


def lead_start(text: list[str], start: int, python: bool) -> int:
    """A symbol's first line, moved up over the lines that belong to it: the comment block right above it and, in
    C++, a `template <...>` or return-type line that ctags puts before the line it reports."""
    n = start - 1  # 1-based line above the symbol
    while n >= 1:
        t = text[n - 1].strip()
        comment = t.startswith("#") if python else t.startswith(("//", "/*", "*"))
        prefix = not python and t and not t.startswith(("#", "}", ")")) and not t.endswith(
            (";", "{", "}", ",", ")", ":", "\\"))
        if not (comment or prefix):
            break
        n -= 1
    return n + 1


class Snapshot:
    """The files and symbols of the cited repository at one commit."""

    _memo: dict[str, "Snapshot"] = {}

    def __init__(self, repo: str, sha: str):
        self.repo, self.sha = repo, sha
        self.tree = checkout(repo, sha)
        self.files = {str(p.relative_to(self.tree)) for p in self.tree.rglob("*") if p.is_file()} - {".complete"}
        self._c: dict[str, list[Symbol]] | None = None
        self._syms: dict[str, list[Symbol]] = {}

    @classmethod
    def of(cls, repo: str, sha: str) -> "Snapshot":
        if sha not in cls._memo:
            cls._memo[sha] = cls(repo, sha)
        return cls._memo[sha]

    def symbols(self, path: str) -> list[Symbol]:
        if path not in self._syms:
            if path.endswith(".py"):
                found = py_symbols((self.tree / path).read_text(errors="replace"))
            elif Path(path).suffix in C_SUFFIXES:
                if self._c is None:
                    self._c = c_symbols(self.tree)
                found = self._c.get(path, [])
            else:
                found = []
            text = (self.tree / path).read_text(errors="replace").splitlines() if found else []
            self._syms[path] = [Symbol(s.name, s.kind, lead_start(text, s.start, path.endswith(".py")), s.end,
                                       s.signature) for s in found]
        return self._syms[path]

    def lines(self, path: str, a: int, b: int) -> list[str]:
        return (self.tree / path).read_text(errors="replace").splitlines()[a - 1:b]

    def resolve(self, cite: Cite) -> tuple[int, int] | None:
        """The line range of a citation, None for a file; Unresolved if the file or symbol is not there."""
        if cite.path not in self.files:
            raise Unresolved(f"no file {cite.path}")
        if cite.symbol is None:
            if cite.overload:
                raise Unresolved("overload= without a symbol")
            return None
        cands = sorted((s for s in self.symbols(cite.path) if s.name == cite.symbol), key=lambda s: s.start)
        if not cands:
            raise Unresolved(f"no symbol {cite.symbol} in {cite.path}")
        if cite.overload:
            if cite.overload > len(cands):
                raise Unresolved(f"overload={cite.overload}, but {cite.symbol} has {len(cands)} in {cite.path}")
            s = cands[cite.overload - 1]
            return s.start, s.end
        if len(cands) > 1:
            defs = [s for s in cands if s.kind != "prototype"]
            if len(defs) != 1:
                listing = "; ".join(f"overload={i} L{s.start} {s.kind} {s.signature}".rstrip()
                                    for i, s in enumerate(cands, 1))
                raise Unresolved(f"{cite.symbol} is ambiguous in {cite.path}, add one of: {listing}")
            cands = defs
        return cands[0].start, cands[0].end


# ---------------------------------------------------------------- the citations in the book

def parse_args(args: str) -> Cite:
    words = shlex.split(args)
    overload = None
    pos = []
    for w in words:
        if w.startswith("overload="):
            overload = int(w.split("=", 1)[1])
        else:
            pos.append(w)
    if not 1 <= len(pos) <= 2:
        raise Unresolved(f"expected <path> [<symbol>], got {args!r}")
    return Cite(pos[0], pos[1] if len(pos) == 2 else None, overload)


def book_files(paths: list[Path] | None = None) -> list[Path]:
    if paths:
        return [p for q in paths for p in ([q] if q.is_file() else sorted(q.rglob("*.qmd")))]
    return sorted(p for p in BOOK.rglob("*.qmd") if not {"_book", "_freeze", ".quarto"} & set(p.parts))


def citations(files: list[Path]) -> list[tuple[Path, int, str, Cite | Unresolved]]:
    found = []
    for f in files:
        for n, line in enumerate(f.read_text().splitlines(), 1):
            for m in SHORTCODE.finditer(line):
                try:
                    found.append((f, n, m.group(0), parse_args(m.group("args"))))
                except (Unresolved, ValueError) as e:
                    found.append((f, n, m.group(0), Unresolved(str(e))))
    return found


def labels(paths: set[str]) -> dict[str, str]:
    """Link text for each cited file: its basename, or as many trailing directories as make it unique."""
    out = {}
    for p in paths:
        parts = p.split("/")
        k = 1
        while k < len(parts) and any(q != p and q.split("/")[-k:] == parts[-k:] for q in paths):
            k += 1
        out[p] = "/".join(parts[-k:])
    return out


def _rel(f: Path) -> str:
    try:
        return str(f.relative_to(REPORT))
    except ValueError:
        return str(f)


def read_legacy(path: Path = LEGACY_FILE) -> list[str]:
    """The legacy chapters, in file order (missing file: none)."""
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]


def chapter(f: Path) -> str:
    """The chapter id of a book file, as the ratchet gate names it (tools/ratchet_gate.py chapter_of)."""
    sys.path.insert(0, str(REPORT / "tools"))
    from ratchet_gate import chapter_of
    return chapter_of(str(f.resolve().relative_to(BOOK.resolve())))


def legacy_chapter_of(target: str, report: Path = REPORT) -> str | None:
    """The chapter of a report file (`book/...`, relative to the report root) when that chapter is on the legacy
    list of `report`, else None: a contract or pin check whose target it is is skipped, and runs in full again once
    the chapter's line is deleted."""
    rel = Path(target)
    if rel.parts[:1] != ("book",):
        return None
    sys.path.insert(0, str(REPORT / "tools"))
    from ratchet_gate import chapter_of
    ch = chapter_of(rel.relative_to("book").as_posix())
    return ch if ch in read_legacy(Path(report) / "tools" / LEGACY_FILE.name) else None


def legacy_report(left: dict[str, int], legacy: list[str]) -> int:
    """Per chapter and per owner, the unresolved citations the legacy list still covers; a listed chapter with
    none left is an error (delete its line)."""
    bad = 0
    for ch in legacy:
        if not left.get(ch):
            print(f"{ch}: in {LEGACY_FILE.name} but every citation resolves: delete its line")
            bad += 1
    if not legacy:
        print("legacy list empty: every chapter is checked in full")
        return bad
    try:
        sys.path.insert(0, str(REPORT / "tools"))
        from ratchet_gate import read_owners
        owners, _ = read_owners(REPORT)
    except Exception:
        owners = {}
    per_owner: dict[str, int] = {}
    for ch in legacy:
        who = ",".join(owners.get(ch, ["?"]))
        per_owner[who] = per_owner.get(who, 0) + left.get(ch, 0)
        print(f"legacy {ch}\t{who}\t{left.get(ch, 0)} unresolved")
    for who, n in sorted(per_owner.items(), key=lambda x: -x[1]):
        print(f"legacy owner {who}\t{n} unresolved")
    print(f"legacy list: {len(legacy)} chapters, {sum(left.get(c, 0) for c in legacy)} unresolved citations "
          f"skipped; the report is complete only when the list is empty")
    return bad


def cmd_check(ns) -> int:
    repo, sha = read_pin()
    snap = Snapshot.of(repo, sha)
    table, bad = {}, 0
    legacy = read_legacy()
    left: dict[str, int] = {}
    found = citations(book_files(ns.paths))
    names = labels({c.path for *_, c in found if isinstance(c, Cite)})
    for f, n, text, c in found:
        try:
            if isinstance(c, Unresolved):
                raise c
            rng = snap.resolve(c)
        except Unresolved as e:
            ch = chapter(f)
            if ch in legacy:  # skipped: rendered as plain text, counted below
                left[ch] = left.get(ch, 0) + 1
                if isinstance(c, Cite):
                    table.setdefault(c.key, {"label": names[c.path], "symbol": c.symbol, "url": None, "lines": None})
                continue
            print(f"{_rel(f)}:{n}: {text}: {e}")
            bad += 1
            continue
        url = f"https://github.com/{repo}/blob/{sha}/{c.path}" + (f"#L{rng[0]}-L{rng[1]}" if rng else "")
        table[c.key] = {"label": names[c.path], "symbol": c.symbol, "url": url, "lines": list(rng) if rng else None}
    if not ns.no_write:
        TABLE.write_text(json.dumps({"repo": repo, "sha": sha, "citations": table}, indent=1, sort_keys=True) + "\n")
    resolved = sum(1 for v in table.values() if v["url"] is not None)
    print(f"code citations: {len(found)} found, {resolved} distinct resolved at {repo}@{sha[:12]}, "
          f"{bad} unresolved" + (f", {sum(left.values())} skipped in legacy chapters" if left else ""))
    if not ns.paths:
        bad += legacy_report(left, legacy)
    return 1 if bad else 0


def changed(repo: str, old: str, new: str, cites: set[Cite]) -> tuple[list[str], list[str]]:
    a, b = Snapshot.of(repo, old), Snapshot.of(repo, new)
    diffs, broken = [], []
    for c in sorted(cites, key=lambda c: (c.path, c.symbol or "", c.overload or 0)):
        try:
            rb = b.resolve(c)
        except Unresolved as e:
            broken.append(f"{c.shortcode()}\t{e}")
            continue
        try:
            ra = a.resolve(c)
        except Unresolved as e:
            diffs.append(f"{c.shortcode()}\tnew at {new[:9]} ({e} at {old[:9]})")
            continue
        if c.symbol is None:
            same = (a.tree / c.path).read_bytes() == (b.tree / c.path).read_bytes()
        else:
            same = a.lines(c.path, *ra) == b.lines(c.path, *rb)
        if not same:
            where = f"L{ra[0]}-L{ra[1]} -> L{rb[0]}-L{rb[1]}" if ra else "file"
            diffs.append(f"{c.shortcode()}\t{where}")
    return diffs, broken


def cmd_diff(ns, old: str, new: str) -> int:
    repo, _ = read_pin()
    legacy = read_legacy()
    found = citations(book_files(ns.paths))
    cites = {c for f, *_, c in found if isinstance(c, Cite) and chapter(f) not in legacy}
    skipped = {c for f, *_, c in found if isinstance(c, Cite) and chapter(f) in legacy} - cites
    if skipped:
        print(f"# {len(skipped)} cited symbols only in legacy chapters ({', '.join(legacy)}) are not gated")
    diffs, broken = changed(repo, old, new, cites)
    print(f"# {len(cites)} cited symbols; {len(diffs)} changed body between {old[:12]} and {new[:12]}"
          f" (re-read these); {len(broken)} do not resolve at {new[:12]}")
    for d in diffs:
        print("changed\t" + d)
    for d in broken:
        print("unresolved\t" + d)
    return 1 if broken else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("check", help="resolve every citation at the pin; exit 1 if any does not resolve")
    p.add_argument("paths", nargs="*", type=Path)
    p.add_argument("--no-write", action="store_true", help=f"do not write {TABLE.relative_to(REPORT)}")
    p = sub.add_parser("diff", help="cited symbols whose body differs between two commits")
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("paths", nargs="*", type=Path)
    p = sub.add_parser("bump", help="diff from the pin to NEW, then write NEW into code_pin.yml")
    p.add_argument("new")
    p.add_argument("paths", nargs="*", type=Path)
    argv = sys.argv[1:] if argv is None else argv
    ns = ap.parse_args(argv or ["check"])  # no arguments (the pre-render of book/_quarto.yml): check
    for attr in ("old", "new"):
        v = getattr(ns, attr, None)
        if v is not None and not re.fullmatch(r"[0-9a-f]{40}", v):
            ap.error(f"{attr}: a full 40-character sha, not {v!r}")
    if ns.cmd == "check":
        return cmd_check(ns)
    if ns.cmd == "diff":
        return cmd_diff(ns, ns.old, ns.new)
    _, pin = read_pin()
    rc = cmd_diff(ns, pin, ns.new)
    if rc:
        print(f"not moving the pin: fix the unresolved citations first ({PIN_FILE.name} unchanged)")
        return rc
    write_pin(ns.new)
    print(f"{PIN_FILE.name}: {pin[:12]} -> {ns.new[:12]}; run `check` to refresh {TABLE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
