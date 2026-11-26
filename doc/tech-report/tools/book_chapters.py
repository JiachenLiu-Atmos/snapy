#!/usr/bin/env python3
"""Write the chapter list of `book/_quarto.yml` from OUTLINE.md and the chapter files that exist (STYLE.md 10.1).

A chapter file is `book/chapters/<id>-<slug>.qmd`, where `<id>` is the two-digit number of a chapter of
OUTLINE.md (`## Chapter 6. ...` is `06`), optionally with the letter of a split chapter (`07b`), or `app<letter>`
for an appendix (`appa-notation.qmd`). The list holds `index.qmd`, then every chapter file in OUTLINE order,
then the appendices; a chapter that is not written yet is not listed (quarto fails on a listed file that does
not exist). The list sits between the two marker comments in `_quarto.yml` and nothing else there is touched.

    python3 tools/book_chapters.py           # rewrite the list
    python3 tools/book_chapters.py --check   # exit 1 if the list is not what the files give (the render gate runs this)

Exit status 0 if the list is current (or was written), 1 if `--check` finds a difference or a chapter file has no
OUTLINE id, 2 if `_quarto.yml` has no markers.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPORT = Path(__file__).resolve().parent.parent
BEGIN = "  # BEGIN chapter list: written by tools/book_chapters.py from OUTLINE.md and the files in chapters/"
END = "  # END chapter list"
_FILE = re.compile(r"^(?:(?P<num>\d\d)(?P<split>[a-z]?)|app(?P<app>[a-z]))-[\w-]+\.qmd$")


def outline_chapters(outline: Path) -> list[int]:
    """Chapter numbers of the outline, in order (`## Chapter 6. Gravity and energy` gives 6). The V3 structure
    (STRUCTURE.md, `### Chapter 17. Known limits`) numbers the chapters when it sits beside OUTLINE.md."""
    structure = outline.parent / "STRUCTURE.md"
    if structure.exists():
        found = [int(n) for n in re.findall(r"^### Chapter (\d+)\.", structure.read_text(), re.M)]
        if found:
            return found
    return [int(n) for n in re.findall(r"^## Chapter (\d+)\.", outline.read_text(), re.M)]


def chapter_files(book: Path, chapters: list[int]) -> tuple[list[str], list[str], list[str]]:
    """(chapter files in OUTLINE order, appendix files, files with no OUTLINE id), as paths under book/."""
    found, apps, bad = [], [], []
    for q in sorted((book / "chapters").glob("*.qmd")):
        if q.name.startswith("_"):
            continue
        m = _FILE.match(q.name)
        if m and m["app"]:
            apps.append(f"chapters/{q.name}")
        elif m and int(m["num"]) in chapters:
            found.append((chapters.index(int(m["num"])), m["split"], f"chapters/{q.name}"))
        else:
            bad.append(f"chapters/{q.name}")
    return [f for *_, f in sorted(found)], apps, bad


def chapter_block(chapters: list[str], appendices: list[str]) -> str:
    lines = [BEGIN, "  chapters:", "    - index.qmd", *(f"    - {c}" for c in chapters)]
    if appendices:
        lines += ["  appendices:", *(f"    - {a}" for a in appendices)]
    return "\n".join(lines + [END])


def current_block(text: str) -> str | None:
    i, j = text.find(BEGIN), text.find(END)
    return text[i:j + len(END)] if 0 <= i < j else None


def check(book: Path, outline: Path | None = None, write: bool = False) -> tuple[int, list[str]]:
    """(status, messages) for the chapter list of `book/_quarto.yml`; with write=True, rewrite it."""
    outline = outline or book.parent / "OUTLINE.md"
    quarto = book / "_quarto.yml"
    text = quarto.read_text()
    have = current_block(text)
    if have is None:
        return 2, [f"{quarto} has no chapter-list markers ('{BEGIN.strip()}' ... '{END.strip()}')"]
    chapters, apps, bad = chapter_files(book, outline_chapters(outline))
    want = chapter_block(chapters, apps)
    msgs = [f"{f}: not an OUTLINE chapter id (want NN-slug.qmd with NN a chapter of OUTLINE.md, "
            "or app<letter>-slug.qmd)" for f in bad]
    if have != want:
        if write:
            quarto.write_text(text.replace(have, want))
        else:
            listed = set(re.findall(r"^    - (\S+)$", have, re.M))
            wanted = set(re.findall(r"^    - (\S+)$", want, re.M))
            msgs += [f"{f}: exists but is not in _quarto.yml" for f in sorted(wanted - listed)]
            msgs += [f"{f}: listed in _quarto.yml but does not exist" for f in sorted(listed - wanted)]
            if not (wanted ^ listed):
                msgs.append("_quarto.yml lists the chapters out of OUTLINE order")
            msgs.append("fix: python3 tools/book_chapters.py")
    return (1 if bad or (have != want and not write) else 0), msgs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the list is not current")
    ap.add_argument("--book", type=Path, default=REPORT / "book")
    args = ap.parse_args(argv)
    status, msgs = check(args.book, write=not args.check)
    for m in msgs:
        print(m)
    if status == 0:
        print(f"book_chapters: {args.book / '_quarto.yml'} lists the chapters that exist", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
