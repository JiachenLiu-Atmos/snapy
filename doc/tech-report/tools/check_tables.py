#!/usr/bin/env python3
"""Table gate: the book's pipe tables against STYLE.md 10.5 and the Code and Tests layers of section 2.

A static check of the chapter sources (`book/chapters/**/*.qmd`), in ratchet mode like `check_figstyle.py`. Fenced
code and HTML comments are skipped, so tables a Python cell generates are not seen here (`check_table_cells.py`
gates those in the render). Every pipe table is counted under four kinds:

  * `columns`: more than 5 columns (STYLE 10.5: the PDF column is about 6.3 in wide);
  * `caption`: no caption paragraph with a `{#tbl-...}` label right after the table (STYLE 10.5);
  * `header`: the first table of a `### Code` or `### Tests` section without the fixed header of STYLE 2,
    `| what | code link | symbol | switch |` and
    `| test (ctest name) | what it asserts | tolerance | where the tolerance comes from |`;
  * `extra`: a second table in a `### Code` or `### Tests` section (STYLE 10.5: each layer uses one table).

Counts are summed per chapter (`ch07` for `book/chapters/07-*`; the chapter's own `.qmd` and its section files
count together) and compared with the committed baseline `tools/table-baseline.json`. The ratchet: a count above
its baseline fails, and so does one below it, so that the baseline is lowered in the same commit that fixes the
tables (`--update` writes the lower counts and refuses to raise any). A chapter or kind missing from the baseline
has a baseline of 0.

    python3 tools/check_tables.py            # gate: exit 1 if any count differs from its baseline
    python3 tools/check_tables.py --list     # also print every counted table, file:line kind detail
    python3 tools/check_tables.py --update   # lower the baseline to the current counts (never raises one)

Exit status 0 when every count equals its baseline, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPORT = Path(__file__).resolve().parent.parent
BASELINE = REPORT / "tools" / "table-baseline.json"
KINDS = ("columns", "caption", "header", "extra")
MAX_COLUMNS = 5
HEADERS = {
    "Code": "| what | code link | symbol | switch |",
    "Tests": "| test (ctest name) | what it asserts | tolerance | where the tolerance comes from |",
}

_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*(\{[^}]*\})?\s*$")
_LABEL = re.compile(r"\{#tbl-[\w-]+\}\s*$")


def chapter_of(rel: str) -> str:
    """`book/chapters/09-time-integration/_rk.qmd` -> `ch07`; `book/chapters/appd-switches/...` -> `appd`"""
    parts = rel.split("/")
    name = parts[2] if len(parts) > 2 else rel
    m = re.match(r"(\d\d)", name)
    return f"ch{m.group(1)}" if m else name.split("-")[0].removesuffix(".qmd")


def _source_lines(text: str) -> list[str | None]:
    """the lines of a .qmd with fenced code and HTML comments blanked to None"""
    out, fence, comment = [], None, False
    for line in text.splitlines():
        if fence:
            out.append(None)
            s = line.strip()
            if s.startswith(fence) and set(s) <= {fence[0]}:
                fence = None
            continue
        if comment:
            out.append(None)
            if "-->" in line:
                comment = False
            continue
        m = _FENCE.match(line)
        if m:
            fence = m.group(1)
            out.append(None)
            continue
        if line.lstrip().startswith("<!--") and "-->" not in line:
            comment = True
            out.append(None)
            continue
        out.append(line)
    return out


def _columns(separator: str) -> int:
    return len(separator.strip().strip("|").split("|"))


def scan_text(text: str) -> list[tuple[int, str, str]]:
    """(line, kind, detail) for every finding in one .qmd source"""
    lines = _source_lines(text)
    hits, section, seen = [], None, 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if line is None:
            i += 1
            continue
        h = _HEADING.match(line)
        if h:
            section, seen = (h.group(2).strip() if len(h.group(1)) == 3 else None), 0
            i += 1
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if not (line.lstrip().startswith("|") and nxt is not None and _SEPARATOR.match(nxt)):
            i += 1
            continue
        start, j = i, i
        while j < len(lines) and lines[j] is not None and lines[j].lstrip().startswith("|"):
            j += 1
        ncol = _columns(nxt)
        if ncol > MAX_COLUMNS:
            hits.append((start + 1, "columns", f"{ncol} columns"))
        k = j
        while k < len(lines) and lines[k] is not None and not lines[k].strip():
            k += 1
        caption = []
        if k < len(lines) and lines[k] is not None and lines[k].lstrip().startswith(":"):
            while k < len(lines) and lines[k] is not None and lines[k].strip():
                caption.append(lines[k].strip())
                k += 1
        if not (caption and _LABEL.search(" ".join(caption))):
            hits.append((start + 1, "caption", "no caption with a {#tbl-...} label after the table"))
        if section in HEADERS:
            seen += 1
            header = " ".join(line.split())
            if seen == 1 and header != HEADERS[section]:
                hits.append((start + 1, "header", f"{section} table header is {header!r}"))
            elif seen > 1:
                hits.append((start + 1, "extra", f"table {seen} of the {section} section"))
        i = j
    return hits


def scan(report: Path = REPORT) -> dict[str, list[tuple[str, int, str, str]]]:
    out: dict[str, list] = {}
    for path in sorted((report / "book" / "chapters").rglob("*.qmd")):
        rel = path.relative_to(report).as_posix()
        for line, kind, detail in scan_text(path.read_text(encoding="utf-8")):
            out.setdefault(chapter_of(rel), []).append((rel, line, kind, detail))
    return out


def counts(sites: dict) -> dict[str, dict[str, int]]:
    out = {}
    for ch, rows in sorted(sites.items()):
        c = {k: sum(1 for r in rows if r[2] == k) for k in KINDS}
        c = {k: v for k, v in c.items() if v}
        if c:
            out[ch] = c
    return out


def load_baseline(path: Path = BASELINE) -> dict[str, dict[str, int]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {ch: {k: int(v) for k, v in kinds.items()} for ch, kinds in data.get("chapters", {}).items()}


def compare(current: dict, baseline: dict) -> list[str]:
    msgs = []
    for ch in sorted(set(current) | set(baseline)):
        for kind in KINDS:
            now, base = current.get(ch, {}).get(kind, 0), baseline.get(ch, {}).get(kind, 0)
            if now > base:
                msgs.append(f"{ch} {kind}: {now} table(s), baseline {base}: fix the new ones (--list shows them)")
            elif now < base:
                msgs.append(f"{ch} {kind}: {now} table(s), baseline {base}: lower the baseline with --update")
    return msgs


def write_baseline(current: dict, baseline: dict, path: Path = BASELINE) -> list[str]:
    up = [f"{ch} {k}: {v} > baseline {baseline.get(ch, {}).get(k, 0)}" for ch, kinds in current.items()
          for k, v in kinds.items() if v > baseline.get(ch, {}).get(k, 0)]
    if up and path.exists():
        return up
    data = {"about": "per-chapter count of pipe tables outside STYLE 10.5 and the Code/Tests headers of STYLE 2; "
                     "may only shrink (tools/check_tables.py)",
            "chapters": current}
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return []


def check(report: Path = REPORT, baseline_path: Path = BASELINE) -> tuple[int, list[str]]:
    msgs = compare(counts(scan(report)), load_baseline(baseline_path))
    return (1 if msgs else 0), msgs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="print every counted table")
    ap.add_argument("--update", action="store_true", help="lower the baseline to the current counts")
    ap.add_argument("--report", type=Path, default=REPORT, help=argparse.SUPPRESS)
    ap.add_argument("--baseline", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    baseline_path = args.baseline or args.report / "tools" / "table-baseline.json"
    sites = scan(args.report)
    current, baseline = counts(sites), load_baseline(baseline_path)
    if args.list:
        for ch in sorted(sites):
            for rel, line, kind, detail in sites[ch]:
                print(f"{rel}:{line}: {kind}: {detail}")
    if args.update:
        up = write_baseline(current, baseline, baseline_path)
        if up:
            print("check_tables: refusing to raise the baseline: " + "; ".join(up))
            return 1
        print(f"check_tables: baseline written, {sum(sum(c.values()) for c in current.values())} table finding(s)")
        return 0
    msgs = compare(current, baseline)
    total = sum(sum(c.values()) for c in current.values())
    print(f"check_tables: {total} table finding(s) in {len(current)} chapter(s); baseline "
          f"{sum(sum(c.values()) for c in baseline.values())}" + (f"; {len(msgs)} count(s) differ" if msgs else ""))
    for m in msgs:
        print("  " + m)
    return 1 if msgs else 0


if __name__ == "__main__":
    sys.exit(main())
