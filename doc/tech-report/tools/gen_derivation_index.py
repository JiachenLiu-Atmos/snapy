#!/usr/bin/env python3
"""Generate Appendix B, the derivation index, from the chapter sources (OUTLINE ch17, STYLE.md 2 and 7).

The index is read from the book itself, so it cannot drift from the chapters. Every scheme file
(`book/chapters/NN-slug/_<scheme>.qmd`, or a chapter file with its own `## ... {#sec-...}` schemes) is
scanned for its Derivation layer (STYLE 2). Each labelled equation `{#eq-...}` in that layer is one row:

  * the scheme (`@sec-...`) and the equation (`@eq-...`), as cross-references;
  * the derivation: the sentence that leads into the equation, or the statement given for it in
    `tools/derivation-statements/chNN.json` (equation label -> one line), which wins when present;
  * the code: the code site the chapter tags the equation with (STYLE 2: "each numbered equation that the
    code implements is tagged with the code site that implements it"): the first code link after the
    equation in the layer, else the first one before it, given as its anchor symbol (the one the citation
    checker reads) and its `file:lines` text. The link itself stays in the chapter, which owns the citation;
    the index repeats no URL, so a citation is checked once, where it is made;
  * the check: the executable check scripts the scheme file names (`<scheme>_check.py`, STYLE 7) that exist
    under `src/snapy_report/chNN/checks/`, and the check lines (`[Cn]`) the layer quotes next to the equation;
  * the source: "exists" when the layer cites a derivation note (`docs/derivations/` or `sources/`), else
    "re-derived".

A scheme whose Derivation layer labels no equation gets one row with its first sentence (often "None: ..."),
so every written scheme appears once at least, as OUTLINE asks (one row per scheme). Appendix files and the
generated fragment itself are not scanned.

The output is one file per chapter, `17-appendices/_appb-chNN.qmd` with that chapter's rows, and the parent
`_appb-derivation-index.qmd`, which holds the introduction, the summary table and the includes of the chapter
files. No committed file holds a count over several chapters: the totals and the summary counts are
`{{< generated appb ... >}}` shortcodes, filled in at render from `book/_generated/appb.json` (not committed),
which this script writes too. So a pull request that edits one chapter changes only that chapter's file, and two
that edit different chapters merge cleanly (tools/generated_values.py; README, "Generated files and rebasing").
The given statements are one file per chapter for the same reason: `tools/derivation-statements/chNN.json`
holds the labels `eq-chNN-...` and no other, and a label in the wrong chapter's file, a label given twice in a
file, or a statement that is not a string is an input error (`load_statements`).

book/_quarto.yml runs this script as its pre-render step, so every render (local or CI) writes the files from
the chapters as they are. The files are also committed, because Quarto fails on an include whose file does not
exist before the pre-render step runs; tools/tests fails when a committed copy differs from a fresh generation.
The script needs only the Python standard library, and finishes in well under a second.

    python3 tools/gen_derivation_index.py           # rewrite the files and values (the pre-render step)
    python3 tools/gen_derivation_index.py --check   # exit 1 if a committed file is not what the sources give
    python3 tools/gen_derivation_index.py --stats   # counts per chapter, to stdout
    python3 tools/gen_derivation_index.py --book <dir> --fragment <file>   # another tree (e.g. a draft branch)

Exit status 0 on success, 1 if `--check` finds a difference, 2 for an input error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generated_values as gv  # noqa: E402
from check_citations import find_anchor  # noqa: E402  (the same anchor rule the checker applies)

REPORT = Path(__file__).resolve().parent.parent
BOOK = REPORT / "book"
INCLUDE_DIR = "17-appendices"
PARENT = "_appb-derivation-index.qmd"
FRAGMENT = BOOK / "chapters" / INCLUDE_DIR / PARENT
STATEMENTS = REPORT / "tools" / "derivation-statements"
CHECKS = REPORT / "src" / "snapy_report"

_SCHEME = re.compile(r"^## (?P<title>.*?)\s*\{#(?P<label>sec-[\w-]+)[^}]*\}\s*$")
_CHAPTER = re.compile(r"^# (?P<title>.*?)\s*\{#(?P<label>sec-ch\d+[a-z]?)\}\s*$")
_EQ = re.compile(r"\{#(eq-[\w-]+)\}")
_CODE = re.compile(r"\[`?[^\]\n]*`?\]\((https://github\.com/(?:chengcli|UCzhangxi)/\w+/blob/[0-9a-f]{40}/[^)\s]+)\)")
_SHORTCODE = re.compile(r"(?<!\{)\{\{<\s*code\s[^>]*>\}\}")
_CHECK = re.compile(r"\b([\w-]+_check\.py)\b")
_CLINE = re.compile(r"\[C\d+\]")
_NOTE = re.compile(r"docs/derivations/|(?<![\w/])sources/")
#: chapters that are not schemes (none since chapter 16 became the verification-method chapter of the V3 outline)
EXCLUDE: set[str] = set()
_INCLUDE = re.compile(r"\{\{<\s*include\s+(\S+)\s*>\}\}")


@dataclass
class Row:
    chapter: str          # sec-chNN
    scheme: str           # sec-chNN-x
    title: str
    equation: str | None  # eq-... or None
    statement: str
    code: str             # markdown cell text
    check: str
    source: str
    given: bool = False   # the statement comes from tools/derivation-statements/


@dataclass
class Scheme:
    chapter: str
    label: str
    title: str
    path: Path
    lines: list[str] = field(default_factory=list)   # the whole file, for anchors
    start: int = 0        # first line of the Derivation layer (0-based, after the heading)
    end: int = 0          # one past its last line


def _clean(text: str) -> str:
    """One table-cell line: links reduced to their text, whitespace joined, `|` escaped (inside math too)."""
    text = re.sub(r"\[([^\]\n]*)\]\([^)\n]*\)", r"\1", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = " ".join(text.split())
    out, math = [], False
    for part in re.split(r"(\$)", text):
        if part == "$":
            math = not math
            out.append(part)
        else:
            out.append(part.replace("|", r"\vert " if math else r"\|"))
    return "".join(out).strip()


def _sentences(text: str) -> list[str]:
    """Split prose into sentences at . ? ! followed by a space, not inside $...$ or `...`."""
    out, cur, math, tick = [], [], False, False
    for i, ch in enumerate(text):
        cur.append(ch)
        if ch == "$":
            math = not math
        elif ch == "`":
            tick = not tick
        elif ch in ".?!" and not math and not tick and (i + 1 == len(text) or text[i + 1] == " "):
            out.append("".join(cur).strip())
            cur = []
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def scan_file(path: Path, chapter: str) -> list[Scheme]:
    """The schemes of one file with the bounds of their Derivation layers."""
    lines = path.read_text(encoding="utf-8").splitlines()
    schemes, cur, fence = [], None, False
    for n, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fence = not fence
        if fence:
            continue
        m = _SCHEME.match(line)
        if m and "unnumbered" not in line:
            if cur and cur.start and not cur.end:
                cur.end = n
            cur = Scheme(chapter, m["label"], m["title"], path, lines)
            schemes.append(cur)
        elif cur and line.startswith("### "):
            if cur.start and not cur.end:
                cur.end = n
            if line[4:].strip().lower() == "derivation":
                cur.start = n + 1
    if cur and cur.start and not cur.end:
        cur.end = len(lines)
    return [s for s in schemes if s.start]


def _code_cell(s: Scheme, a: int, b: int) -> str | None:
    """The first code citation in lines [a, b) of the scheme: a symbol citation (README, "Citing code") as it
    is, a link as "`symbol` at `text`"; or None."""
    for n in range(a, b):
        sc = _SHORTCODE.search(s.lines[n])
        link = _CODE.search(s.lines[n])
        if sc and (not link or sc.start() < link.start()):
            return sc.group(0)
        for m in _CODE.finditer(s.lines[n]):
            text = re.match(r"\[`?([^\]`]*)`?\]", s.lines[n][m.start():])
            sym = find_anchor(s.lines, n, m.start())
            where = f"`{text.group(1).strip()}`"
            return f"`{sym}` at {where}" if sym else where
    return None


def scheme_rows(s: Scheme, statements: dict[str, str], checks_dir: Path) -> list[Row]:
    body = "\n".join(s.lines[s.start:s.end])
    whole = "\n".join(s.lines)
    num = re.match(r"sec-ch(\d+)", s.chapter)
    chk_dir = checks_dir / f"ch{num.group(1)}" / "checks" if num else None
    # Explicit equation-check comments scope a multi-equation script to its actual claims.
    tagged = re.findall(r"<!-- equation-check: (eq-[\w-]+) ([\w-]+_check\.py) (\[[CD]\w+\]) -->", whole)
    tagged_scripts = {script for _, script, _ in tagged}
    scripts = sorted({c for c in _CHECK.findall(whole) if c not in tagged_scripts
                      and chk_dir and (chk_dir / c).exists()})
    source = "exists" if _NOTE.search(body) else "re-derived"
    title = re.sub(r"\s*\(`[^)]*`\)\s*$", "", s.title).strip()

    eq_lines = [(n, e) for n in range(s.start, s.end) for e in _EQ.findall(s.lines[n])]
    rows = []
    if not eq_lines:
        first = _sentences(_clean("\n".join(l for l in s.lines[s.start:s.end]
                                             if not l.lstrip().startswith(("#", ":::", "|", "```")))))
        if not first:
            # The opening paragraph may have moved above the glance box. Keep
            # an otherwise empty derivation's index statement, scoped to this
            # scheme (never borrow prose from another scheme in the same file).
            from fix_glance_order import blocks
            opening = next((b for b in blocks(s.lines[:s.start])
                            if b.kind == "prose" and f"{{#{s.label}}}" in b.section), None)
            if opening:
                first = _sentences(_clean("\n".join(s.lines[opening.start:opening.end])))
        stmt = first[0] if first else "Empty layer."
        rows.append(Row(s.chapter, s.label, title, None, stmt, "", ", ".join(f"`{c}`" for c in scripts) or "none",
                        source if re.search(r"^\s*\$\$", body, re.M) else "n/a"))
        return rows

    bounds = [n for n, _ in eq_lines] + [s.end]
    for k, (n, label) in enumerate(eq_lines):
        # the equation block: back up from its closing `$$ {#eq-...}` line to the opening `$$`
        top = n - 1
        while top > s.start and s.lines[top].strip() != "$$":
            top -= 1
        prev = bounds[k - 1] + 1 if k else s.start
        pre = s.lines[prev:top]
        dollars = [i for i, l in enumerate(pre) if l.strip().startswith("$$")]
        pre = pre[dollars[-1] + 1:] if dollars else pre          # prose after any unlabelled display
        pre = [l for l in pre if not l.lstrip().startswith(("#", ":::", "|", "```"))]
        lead = _sentences(_clean("\n".join(pre)))
        stmt = statements.get(label) or (lead[-1] if lead else "")
        code = _code_cell(s, n + 1, bounds[k + 1]) or _code_cell(s, prev, top) or "not tagged"
        after = re.sub(r"<!--.*?-->", "", "\n".join(s.lines[n + 1:bounds[k + 1]]), flags=re.S)
        clines = sorted(set(_CLINE.findall(after)), key=lambda c: int(c[2:-1]))
        specific = [f"`{script}` `{tag}`" for eq, script, tag in tagged if eq == label
                    and chk_dir and (chk_dir / script).exists()]
        check = ", ".join([f"`{c}`" for c in scripts] + [f"`{c}`" for c in clines if scripts or not specific] + specific) or "none"
        rows.append(Row(s.chapter, s.label, title, label, stmt, code, check, source, label in statements))
    return rows


def chapter_files(book: Path) -> list[tuple[str, str, list[Path]]]:
    """(chapter label, chapter title, scheme-bearing files in include order) for every chapter file."""
    out = []
    for q in sorted((book / "chapters").glob("[0-9][0-9]*.qmd")):
        text = q.read_text(encoding="utf-8")
        m = next((_CHAPTER.match(l) for l in text.splitlines() if _CHAPTER.match(l)), None)
        if not m or m["label"] in EXCLUDE:
            continue
        files = [q] + [q.parent / inc for inc in _INCLUDE.findall(text)]
        out.append((m["label"], m["title"], [f for f in files if f.exists()]))
    return out


def load_statements(directory: Path) -> dict[str, str]:
    """The given statements of every chapter, {equation label: one line}, from `directory`/chNN.json.

    Each chapter's labels are in that chapter's own file, so pull requests on different chapters never edit the
    same file. Raises ValueError for a label outside its file's chapter, a label given twice or a value that is
    not a string; a missing directory gives no statements."""

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        labels = [k for k, _ in pairs]
        if twice := sorted({k for k in labels if labels.count(k) > 1}):
            raise ValueError(f"label given twice: {twice}")
        return dict(pairs)

    out: dict[str, str] = {}
    for f in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        if not re.fullmatch(r"ch\d+[a-z]?", f.stem):
            raise ValueError(f"{f}: a statements file is named after its chapter, chNN.json")
        try:
            given = json.loads(f.read_text(encoding="utf-8"), object_pairs_hook=unique)
        except ValueError as e:
            raise ValueError(f"{f}: {e}") from e
        for label, text in given.items():
            if not label.startswith(f"eq-{f.stem}-"):
                raise ValueError(f"{f}: {label} belongs in its own chapter's file")
            if not isinstance(text, str):
                raise ValueError(f"{f}: the statement of {label} is not a string")
            out[label] = text
    return out


def collect(book: Path, statements: dict[str, str], checks_dir: Path) -> list[tuple[str, str, list[Row]]]:
    out = []
    for label, title, files in chapter_files(book):
        rows = []
        for f in files:
            for s in scan_file(f, label):
                rows += scheme_rows(s, statements, checks_dir)
        if rows:
            out.append((label, title, rows))
    return out


def _counts(rs: list[Row]) -> dict[str, str]:
    """The summary-table counts of a set of rows."""
    e = [r for r in rs if r.equation]
    return {"schemes": str(len({r.scheme for r in rs})), "with-eq": str(len({r.scheme for r in e})),
            "equations": str(len(e)), "exists": str(sum(r.source == "exists" for r in e)),
            "re-derived": str(sum(r.source == "re-derived" for r in e)),
            "with-check": str(len({r.scheme for r in rs if "_check.py" in r.check}))}


def chapter_file(label: str) -> str:
    """The committed fragment holding one chapter's rows: `_appb-chNN.qmd`."""
    return f"_appb-{label[4:]}.qmd"


def outputs(chapters: list[tuple[str, str, list[Row]]]) -> tuple[dict[str, str], dict[str, str]]:
    """({file name: text} of the committed fragments, {key: value} filled in at render).

    The parent fragment holds no count and no row, so it changes only when a chapter gains its first scheme or
    loses its last; each chapter's rows are in that chapter's own file; every count, per chapter and in total,
    is a render-time value (generated_values)."""
    sc = lambda key: gv.shortcode("appb", key)  # noqa: E731
    lines = [
        "<!-- GENERATED by tools/gen_derivation_index.py from the chapter sources; do not edit.",
        "     one-line statements: tools/derivation-statements/chNN.json; refresh: python3 tools/gen_derivation_index.py",
        "     The counts are filled in at render from book/_generated/appb.json (written by the same script, not",
        "     committed), and each chapter's rows are in its own _appb-chNN.qmd, so that an edit to one chapter",
        "     changes only that chapter's file. -->",
        "",
        f"Every derivation the written chapters carry, read from their Derivation layers: {sc('equations')} labelled "
        f"equations in {sc('with-eq')} of the {sc('schemes')} written schemes. Each row gives the equation, the "
        "statement it derives, the code site the chapter tags it with, and the executable check that verifies it; "
        "the code site is named as the chapter gives it, and the link to it is in the chapter. "
        "\"exists\" means the chapter rewrites a derivation note from snapy's "
        "`docs/derivations/` or the report's `sources/`; \"re-derived\" means it was derived from the code at the "
        "pin. A scheme whose Derivation layer labels no equation has one row with that layer's first sentence.",
        "",
        "| chapter | schemes | with a labelled derivation | labelled equations | exists | re-derived | with a check script |",
        "|---|---|---|---|---|---|---|",
    ]
    keys = ("schemes", "with-eq", "equations", "exists", "re-derived", "with-check")
    values = dict(_counts([r for _, _, rs in chapters for r in rs]))
    files = {}
    for label, title, rs in chapters:
        ch = label[4:]
        values.update({f"{ch}.{k}": v for k, v in _counts(rs).items()})
        lines.append(f"| @{label} | " + " | ".join(sc(f"{ch}.{k}") for k in keys) + " |")
        body = [f"<!-- GENERATED by tools/gen_derivation_index.py from the sources of @{label}; do not edit. -->", "",
                f"## {title} (@{label}) {{.unnumbered}}", "",
                "| scheme | equation | derivation | code | check | source |",
                # pandoc sizes the PDF columns by the dashes
                "|" + "|".join("-" * n for n in (15, 9, 31, 20, 17, 8)) + "|"]
        for r in rs:
            eq = f"@{r.equation}" if r.equation else "none"
            body.append(f"| @{r.scheme} | {eq} | {_clean(r.statement)} | {r.code} | {r.check} | {r.source} |")
        body += ["", f": Derivations of @{label}. {{#tbl-appb-{ch}}}"]
        files[chapter_file(label)] = "\n".join(body) + "\n"
    lines.append("| total | " + " | ".join(sc(k) for k in keys) + " |")
    lines += ["", ": Derivations per chapter. {#tbl-appb-summary}", ""]
    # a nested include resolves against the top-level document, chapters/appb-derivations.qmd
    for label, _, _ in chapters:
        lines += [f"{{{{< include {INCLUDE_DIR}/{chapter_file(label)} >}}}}", ""]
    return {PARENT: "\n".join(lines).rstrip("\n") + "\n", **files}, values


def fragment(chapters: list[tuple[str, str, list[Row]]]) -> str:
    """Appendix B as the reader gets it: the parent with its chapter files and render-time values filled in."""
    files, values = outputs(chapters)
    text = re.sub(r"^\{\{<\s*include\s+\S+/(_appb-[\w-]+\.qmd)\s*>\}\}\s*$",
                  lambda m: files[m.group(1)].rstrip("\n"), files[PARENT], flags=re.M)
    return gv.expand(text, {"appb": values}, strict=True)


def stats(chapters) -> str:
    out = []
    for label, _, rs in chapters:
        e = [r for r in rs if r.equation]
        out.append(f"{label}: {len({r.scheme for r in rs})} schemes, {len(e)} equations, "
                   f"{sum(r.code == 'not tagged' for r in e)} untagged, {sum(r.check == 'none' for r in rs)} rows without a check, "
                   f"{sum(not r.given for r in e)} equations without a given statement")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="exit 1 if a fragment differs from the sources'")
    mode.add_argument("--stats", action="store_true", help="print counts per chapter")
    ap.add_argument("--book", type=Path, default=BOOK)
    ap.add_argument("--fragment", type=Path, default=None, help="the parent fragment; the chapter files go beside it")
    ap.add_argument("--values", type=Path, default=None, help="the render-time values (default book/_generated/appb.json)")
    ap.add_argument("--statements", type=Path, default=STATEMENTS, help="the directory of chNN.json statements")
    args = ap.parse_args(argv)
    frag = args.fragment or args.book / "chapters" / INCLUDE_DIR / PARENT
    try:
        statements = load_statements(args.statements)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    chapters = collect(args.book, statements, args.book.parent / "src" / "snapy_report")
    if args.stats:
        print(stats(chapters))
        return 0
    files, values = outputs(chapters)
    files = {frag.name if name == PARENT else name: text for name, text in files.items()}
    rc = gv.sync(frag.parent, files, "gen_derivation_index", args.check, orphans="_appb-ch*.qmd")
    if args.check or rc:
        return rc
    gv.write(args.values or gv.path(args.book, "appb"), values)
    n = sum(1 for _, _, rs in chapters for r in rs if r.equation)
    print(f"gen_derivation_index: wrote {n} equations to {frag} and {len(files) - 1} chapter files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
