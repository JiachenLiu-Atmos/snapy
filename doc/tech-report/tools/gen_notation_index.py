#!/usr/bin/env python3
"""Generate Appendix A, the notation table, from NOTATION.md (OUTLINE ch17: "NOTATION.md, rendered as a table").

NOTATION.md stays the one source of the report's symbols; this script renders it into the book. Every table row
of NOTATION.md becomes one row of a five-column table:

  * symbol and meaning, as NOTATION.md gives them (its "source letter" column, where filled, is appended to the
    meaning as "source notes:");
  * units/shape: the bracketed unit of the meaning (`[kg m$^{-3}$]`, `[-]`, ...) or "count", moved out of the
    meaning; "vector" for a bold symbol and "matrix" for a sans-serif one when no unit is given;
  * first used: the scheme section (`@sec-...`), in book order, whose maths first contains the symbol (exact
    match after spaces and one-character braces are dropped), read from the chapter sources; "—" if none;
  * code: the code name NOTATION.md gives, as text (the code is cited at the pin in the chapters).

The prose of NOTATION.md (its introduction, section notes and the reserved-letter rules) is kept, with the
outline's chapter numbers ("chapter 10.7") turned into cross-references, so no chapter number is hand-typed.

The committed fragment depends on NOTATION.md alone. The two values that the chapters decide, the symbol counts
and each "first used" cell, are `{{< generated appa ... >}}` shortcodes filled in at render from
`book/_generated/appa.json` (not committed), which this script writes too; so a chapter edit that changes a
symbol's first use changes no committed line (tools/generated_values.py; README, "Generated files and
rebasing").

    python3 tools/gen_notation_index.py           # rewrite the fragment and the values
    python3 tools/gen_notation_index.py --check   # exit 1 if the fragment is not what NOTATION.md gives
    python3 tools/gen_notation_index.py --stats   # rows per section and symbols with no first use

Only the Python standard library is needed. Exit status 0 on success, 1 if `--check` finds a difference.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generated_values as gv  # noqa: E402

REPORT = Path(__file__).resolve().parent.parent
NOTATION = REPORT / "NOTATION.md"
BOOK = REPORT / "book"
FRAGMENT = BOOK / "chapters" / "17-appendices" / "_appa-notation.qmd"

#: the outline numbers NOTATION.md's prose uses, and the labels they stand for
OUTLINE_REFS = {
    "chapter 1.13": "@sec-ch01-dependencies",
    "chapter 10.7": "@sec-ch08-drycarry",
    "chapter 11.5": "@sec-ch11-outflow",
    "Chapter 6": "@sec-ch06",
    "chapter 6": "@sec-ch06",
    "Appendix E": "@sec-appe",
}
_SECTION = re.compile(r"^## (?P<num>\d+[a-z]?)\. (?P<title>.+?)\s*$")
_CHAPTER = re.compile(r"^# .*\{#(sec-ch\d+[a-z]?)\}\s*$", re.M)
_SCHEME = re.compile(r"^## .*\{#(sec-[\w-]+)[^}]*\}\s*$")
_INCLUDE = re.compile(r"\{\{<\s*include\s+(\S+)\s*>\}\}")
_MATH = re.compile(r"\$\$(.+?)\$\$|\$([^$\n]+?)\$", re.S)


def _cells(row: str) -> list[str]:
    """The cells of a pipe-table row, splitting on | outside backticks and $...$."""
    out, cur, tick, math = [], [], False, False
    for ch in row.strip().strip("|"):
        if ch == "`" and not math:
            tick = not tick
        elif ch == "$" and not tick:
            math = not math
        if ch == "|" and not tick and not math:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return out


def parse(text: str) -> tuple[list[str], list[dict]]:
    """(introduction lines, sections); a section is {num, title, prose: [lines], rows: [dict], after: [lines]}."""
    intro, sections, cur, header = [], [], None, None
    for line in text.splitlines():
        m = _SECTION.match(line)
        if m:
            cur = {"num": m["num"], "title": m["title"], "prose": [], "rows": [], "after": []}
            sections.append(cur)
            header = None
            continue
        if cur is None:
            if not line.startswith("# "):
                intro.append(line)
            continue
        if line.lstrip().startswith("|"):
            cells = _cells(line)
            if header is None:
                header = [c.lower() for c in cells]
            elif not all(set(c) <= set("-: ") for c in cells):
                cur["rows"].append(dict(zip(header, cells + [""] * (len(header) - len(cells)))))
        else:
            (cur["after"] if cur["rows"] else cur["prose"]).append(line)
    return intro, sections


def _outside_math(text: str):
    """Spans (start, end) of text outside $...$."""
    spans, start, math = [], 0, False
    for i, ch in enumerate(text):
        if ch == "$":
            if not math:
                spans.append((start, i))
            else:
                start = i + 1
            math = not math
    if not math:
        spans.append((start, len(text)))
    return spans


_UNITWORD = re.compile(r"^(-|units stated per reaction|.*\b(kg|m|s|K|Pa|J|W|mol|rad)\b.*)$")


def split_units(meaning: str, symbol: str) -> tuple[str, str]:
    """(meaning without its units, units/shape): the bracketed units whose bracket opens outside maths (they may
    contain maths, `[kg m$^{-3}$]`), or "(count)"; else "vector" for a bold symbol and "matrix" for a
    sans-serif one."""
    found, cuts, math = [], [], False
    i = 0
    while i < len(meaning):
        ch = meaning[i]
        if ch == "$":
            math = not math
        elif not math and meaning.startswith("(count)", i):
            found.append("count")
            cuts.append((i, i + 7))
        elif not math and ch == "[":
            j = meaning.find("]", i)
            plain = re.sub(r"[${}^]", "", meaning[i + 1:j]) if j > 0 else ""
            if j > 0 and _UNITWORD.match(plain.strip()):
                found.append(meaning[i + 1:j])
                cuts.append((i, j + 1))
                i = j
        i += 1
    for x, y in reversed(cuts):
        meaning = meaning[:x].rstrip() + meaning[y:]
    meaning = re.sub(r"\s*,(\s*,)+", ",", meaning)
    meaning = re.sub(r"\s*,\s*([;.)])", r"\1", meaning)
    meaning = re.sub(r"\s+([;,.)])", r"\1", meaning)
    meaning = re.sub(r",?\s*dimensionless(?=\s*(;|$))", "", meaning).strip().rstrip(",")
    units = ["count" if u == "count" else "[-]" if u == "-" else f"[{u}]" for u in found]
    if not units:
        if re.match(r"\$\\(mathbf|boldsymbol)", symbol):
            units = ["vector"]
        elif re.match(r"\$\\mathsf", symbol):
            units = ["matrix"]
    return meaning, ", ".join(units)


def _norm(math: str) -> str:
    """Maths with spaces removed and one-character braces dropped, so `\\mathrm{d}` and `\\mathrm d` compare equal."""
    math = re.sub(r"\s+", "", math)
    return re.sub(r"(\\[A-Za-z]+|[_^])\{(\\?[A-Za-z0-9])\}", r"\1\2", math)


def symbol_key(symbol: str) -> str | None:
    """The string searched for in the chapters: the first alternative of the first math span, spaces removed."""
    m = re.search(r"\$([^$]+)\$", symbol)
    if not m:
        return None
    s, depth, cut = m.group(1), 0, None
    for i, ch in enumerate(s):
        depth += ch in "{(" or -(ch in "})")
        if ch == "," and depth == 0:
            cut = i
            break
    s = s[:cut] if cut is not None else s
    s = s.split("=")[0]
    return _norm(s) or None


def chapter_math(book: Path) -> list[tuple[str, str]]:
    """(label, maths of that part with spaces removed) in book order: one entry per scheme, or the chapter head."""
    out = []
    for q in sorted((book / "chapters").glob("[0-9][0-9]*.qmd")):
        text = q.read_text(encoding="utf-8")
        m = _CHAPTER.search(text)
        if not m:
            continue
        files = [q] + [q.parent / inc for inc in _INCLUDE.findall(text) if (q.parent / inc).exists()]
        for f in files:
            label, buf = m.group(1), []
            for line in f.read_text(encoding="utf-8").splitlines():
                s = _SCHEME.match(line)
                if s and "unnumbered" not in line:
                    out.append((label, "".join(buf)))
                    label, buf = s.group(1), []
                buf.append(line + "\n")
            out.append((label, "".join(buf)))
    return [(lab, _norm("".join(a or b for a, b in _MATH.findall(t)))) for lab, t in out]


def first_use(key: str | None, parts: list[tuple[str, str]]) -> str | None:
    if not key:
        return None
    if re.fullmatch(r"[A-Za-z]", key):
        pat = re.compile(rf"(?<![\\A-Za-z]){key}(?![A-Za-z])")
        hit = lambda t: pat.search(t)  # noqa: E731
    else:
        hit = lambda t: key in t  # noqa: E731
    return next((lab for lab, t in parts if hit(t)), None)


def _refs(text: str) -> str:
    for k, v in OUTLINE_REFS.items():
        text = re.sub(rf"\b{re.escape(k)}\b", v, text)
    return text


def _cell(text: str) -> str:
    return text.replace("\n", " ")


def first_key(key: str) -> str:
    """The render-time value name of a symbol's first use: stable while the symbol is, whatever moves around it."""
    return "first-" + hashlib.sha1(key.encode()).hexdigest()[:12]


def outputs(notation: str, book: Path) -> tuple[str, dict[str, str]]:
    """(the committed fragment, {key: value} filled in at render).

    The fragment is a function of NOTATION.md alone. What the chapters decide, each symbol's first use (the
    earliest of all chapters), and the counts, which two edits of NOTATION.md would both change, are render-time
    values (generated_values), so an edit to one chapter changes no committed line of Appendix A."""
    intro, sections = parse(notation)
    parts = chapter_math(book)
    nrows = sum(len(s["rows"]) for s in sections)
    values = {"symbols": str(nrows), "sections": str(len(sections))}
    sc = lambda key: gv.shortcode("appa", key)  # noqa: E731
    lines = [
        "<!-- GENERATED by tools/gen_notation_index.py from NOTATION.md; do not edit.",
        "     refresh: python3 tools/gen_notation_index.py. The counts and the \"first used\" column are filled in at",
        "     render from book/_generated/appa.json (written by the same script from the chapters, not committed). -->",
        "",
        f"The report's notation: {sc('symbols')} symbols in {sc('sections')} sections. "
        "\"units/shape\" is the unit given in the meaning, moved to its own column; \"first used\" is "
        "the first scheme, in book order, whose maths contains the symbol (an exact match after spaces and one-character "
        "braces are dropped; \"—\" where none does, which a different spelling also gives); \"code\" is the name the "
        "code uses, at "
        "chengcli/snapy `919936da07d80a310b98ab11da184972a3203274` and kintera "
        "`c55b13b2204997d2d09e04498558ab9495d8ee77`.",
        "",
    ]
    lines += [_refs(l.replace('(the "source letter" column in this source table)',
                             '(shown as "source notes" in the meaning column)')) for l in intro]
    for s in sections:
        slug = re.sub(r"[^a-z0-9]+", "-", s["num"].lower()).strip("-")
        # unnumbered: NOTATION.md's prose refers to its own numbers (§4), which the heading keeps
        lines += ["", f"## {s['num']}. {s['title']} {{#sec-appa-{slug} .unnumbered}}", ""]
        lines += [_refs(l) for l in s["prose"]]
        if s["rows"]:
            lines += ["", "| symbol | meaning | units/shape | first used | code |",
                      "|" + "|".join("-" * n for n in (14, 44, 14, 14, 14)) + "|"]
            for r in s["rows"]:
                meaning, unit = split_units(r.get("meaning", ""), r.get("symbol", ""))
                src = r.get("source letter", "")
                if src:
                    meaning += f" (source notes: {src})"
                key = symbol_key(r.get("symbol", ""))
                cell = "—"
                if key:
                    first = first_use(key, parts)
                    values[first_key(key)] = "@" + first if first else "—"
                    cell = sc(first_key(key))
                lines.append(f"| {_cell(r.get('symbol', ''))} | {_cell(_refs(meaning))} | {unit} "
                             f"| {cell} | {_cell(r.get('code', ''))} |")
            lines += ["", f": {s['title']}. {{#tbl-appa-{slug}}}", ""]
        lines += [_refs(l) for l in s["after"]]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).rstrip("\n") + "\n", values


def fragment(notation: str, book: Path) -> str:
    """Appendix A as the reader gets it: the committed fragment with its render-time values filled in."""
    text, values = outputs(notation, book)
    return gv.expand(text, {"appa": values}, strict=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--stats", action="store_true")
    ap.add_argument("--notation", type=Path, default=NOTATION)
    ap.add_argument("--book", type=Path, default=BOOK)
    ap.add_argument("--fragment", type=Path, default=None)
    ap.add_argument("--values", type=Path, default=None, help="the render-time values (default book/_generated/appa.json)")
    args = ap.parse_args(argv)
    frag = args.fragment or args.book / "chapters" / "17-appendices" / "_appa-notation.qmd"
    want, values = outputs(args.notation.read_text(encoding="utf-8"), args.book)
    if args.stats:
        _, sections = parse(args.notation.read_text(encoding="utf-8"))
        cells = re.findall(r"^\| .*\| (—|\{\{< generated appa \S+ >\}\}) \|", want, re.M)
        none = sum(c == "—" or values.get(gv.SHORTCODE.match(c).group(2)) == "—" for c in cells)
        print(f"{sum(len(s['rows']) for s in sections)} rows in {len(sections)} sections; "
              f"{none} symbols with no first use found")
        return 0
    rc = gv.sync(frag.parent, {frag.name: want}, "gen_notation_index", args.check)
    if args.check or rc:
        return rc
    gv.write(args.values or gv.path(args.book, "appa"), values)
    print(f"gen_notation_index: wrote {frag}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
