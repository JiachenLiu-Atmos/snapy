#!/usr/bin/env python3
"""Figure-output gate: the drawn figures against STYLE.md 6.2 and 10.4.

`check_figstyle.py` reads the figure code without running it; this gate runs it. It finds every figure cell of the
book (a `{python}` cell of `book/chapters/**/*.qmd` with a `#| label: fig-...` option), runs the cell's code the
way the render does, from the chapter's directory, takes the Figure it returns and draws it with the Agg renderer.
Every figure is counted under seven kinds:

  * `width`: the figure width is not one of the house widths of figstyle, `SINGLE` 3.05 in or `DOUBLE` 6.2 in;
  * `overflow`: the drawn content (the tight bounding box the render saves) is wider than `DOUBLE`, the text width,
    so the figure runs into the margin or is scaled down below its point sizes;
  * `font`: a drawn text is smaller than figstyle's `MIN_PT` (8 pt), at the figure's own size, which is its print
    size; one count per text, not per figure;
  * `panels`: more than 4 axes, colour bars not counted;
  * `overlap`: two drawn texts, each with its box if it has one, intersect by more than 0.5 pt both across and
    up; one count per pair;
  * `redraw`: drawing the cell twice gives different SVG bytes (STYLE 6.1: a figure that differs after a clean
    re-render is a bug);
  * `warning`: a Python warning raised while the figure is made or drawn, which the render prints into the book
    (a layout call that the figure's layout engine refuses, for example); one count per distinct message.

Only texts that are actually drawn are seen (the tick labels of an axis turned off are not), and sizes are read
from the drawn text objects, so a size set through figstyle and a literal size are judged alike.

Counts are summed per chapter of the cell's `.qmd` (`ch07`, `appe`, ...) and compared with the committed baseline
`tools/figoutput-baseline.json`, in the ratchet of `check_figstyle.py`: a count above its baseline fails, and so
does one below it, so that the baseline is lowered in the commit that fixes the figures (`--update` writes the
lower counts and refuses to raise any). A chapter or kind missing from the baseline has a baseline of 0.

    python3 tools/check_fig_output.py                  # gate: exit 1 if any count differs from its baseline
    python3 tools/check_fig_output.py --list           # also print every counted finding, label kind detail
    python3 tools/check_fig_output.py --update         # lower the baseline to the current counts
    python3 tools/check_fig_output.py --chapters ch03,ch09 --list   # only these chapters (no baseline verdict)

Needs numpy, matplotlib and the `snapy_report` package importable (`pip install -e src`), as the render does.
Exit status 0 when every count equals its baseline, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

REPORT = Path(__file__).resolve().parent.parent
BASELINE = REPORT / "tools" / "figoutput-baseline.json"
KINDS = ("width", "overflow", "font", "panels", "overlap", "redraw", "warning")
MAX_PANELS = 4
OVERLAP_PT = 0.5   # two drawn texts (with their boxes) closer than touching by this much, both ways
WIDTH_TOL = 0.01   # in
OVERFLOW_TOL = 0.02  # in

_CELL = re.compile(r"^```\{python\}[^\n]*\n(.*?)^```\s*$", re.M | re.S)
_LABEL = re.compile(r"^#\|\s*label:\s*(fig-[\w-]+)\s*$", re.M)


@dataclass
class Cell:
    chapter: str
    qmd: Path
    label: str
    code: str


def chapter_of(rel: str) -> str:
    """`book/chapters/09-time-integration/_rk.qmd` -> `ch07`; `book/chapters/appe-.../...` -> `appe`"""
    parts = rel.split("/")
    name = parts[2] if len(parts) > 2 else rel
    m = re.match(r"(\d\d)", name)
    return f"ch{m.group(1)}" if m else name.split("-")[0].removesuffix(".qmd")


def figure_cells(report: Path = REPORT) -> list[Cell]:
    cells = []
    for path in sorted((report / "book" / "chapters").rglob("*.qmd")):
        rel = path.relative_to(report).as_posix()
        for m in _CELL.finditer(path.read_text(encoding="utf-8")):
            body = m.group(1)
            lab = _LABEL.search(body)
            if lab:
                code = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#|"))
                cells.append(Cell(chapter_of(rel), path, lab.group(1), code))
    return cells


def draw(cell: Cell):
    """run the cell from its chapter's directory and return its Figure (the value of the last expression, else
    pyplot's current figure)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    plt.close("all")
    tree = ast.parse(cell.code)
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    ns = {"__name__": "__figure_cell__"}
    cwd = os.getcwd()
    os.chdir(cell.qmd.parent)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(tree, str(cell.qmd), "exec"), ns)
            value = eval(compile(ast.Expression(last.value), str(cell.qmd), "eval"), ns) if last else None
    finally:
        os.chdir(cwd)
    if isinstance(value, Figure):
        return value
    fig = plt.gcf()
    if not fig.axes:
        raise RuntimeError("the cell returned no Figure and drew none")
    return fig


def drawn_texts(fig, renderer):
    """the Text artists the renderer actually draws, with their window extents"""
    from matplotlib.text import Text
    seen = []
    original = Text.draw

    def recording(self, rend):
        if self.get_visible() and self.get_text().strip() and (self.get_alpha() is None or self.get_alpha() > 0):
            seen.append(self)
        return original(self, rend)

    Text.draw = recording
    try:
        fig.draw(renderer)
    finally:
        Text.draw = original
    out, ids = [], set()
    for t in seen:
        if id(t) in ids:
            continue
        ids.add(id(t))
        box = t.get_window_extent(renderer)
        patch = t.get_bbox_patch()
        if patch is not None and patch.get_visible():
            from matplotlib.transforms import Bbox
            box = Bbox.union([box, patch.get_window_extent(renderer)])
        out.append((t, box))
    return out


def inspect(fig) -> list[tuple[str, str]]:
    """the findings of one drawn figure, (kind, detail)"""
    from snapy_report import figstyle as fs

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if not isinstance(fig.canvas, FigureCanvasAgg):  # a Figure made without pyplot has no renderer yet
        FigureCanvasAgg(fig)
    hits = []
    w, _ = fig.get_size_inches()
    if not any(abs(w - a) <= WIDTH_TOL for a in fs.ALLOWED_WIDTHS):
        hits.append(("width", f"{w:.2f} in, not {fs.SINGLE} or {fs.DOUBLE}"))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = drawn_texts(fig, renderer)
    tight = fig.get_tightbbox(renderer)
    if tight is not None and tight.width > fs.DOUBLE + OVERFLOW_TOL:
        hits.append(("overflow", f"content {tight.width:.2f} in wide, text width {fs.DOUBLE}"))
    for t, _ in texts:
        if t.get_fontsize() < fs.MIN_PT - 1e-9:
            hits.append(("font", f"{t.get_fontsize():g} pt: {t.get_text()[:40]!r}"))
    panels = [a for a in fig.axes if a.get_visible() and a.get_label() != "<colorbar>"]
    if len(panels) > MAX_PANELS:
        hits.append(("panels", f"{len(panels)} axes"))
    tol = OVERLAP_PT * fig.dpi / 72.
    for n, (a, ba) in enumerate(texts):
        for b, bb in texts[n + 1:]:
            x = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            y = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if x > tol and y > tol:
                hits.append(("overlap", f"{a.get_text()[:30]!r} / {b.get_text()[:30]!r}"))
    return hits


def svg(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    return buf.getvalue()


def scan(report: Path = REPORT, chapters=None, redraw=True) -> dict[str, list[tuple[str, str, str]]]:
    """{chapter: [(label, kind, detail)]} for every figure cell; an error drawing a cell is a finding of its own
    kind, `error`, which no baseline admits"""
    src = report / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out: dict[str, list] = {}
    for cell in figure_cells(report):
        if chapters and cell.chapter not in chapters:
            continue
        rows = out.setdefault(cell.chapter, [])
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fig = draw(cell)
                rows.extend((cell.label, k, d) for k, d in inspect(fig))
            for msg in sorted({f"{w.category.__name__}: {w.message}"[:160] for w in caught}):
                rows.append((cell.label, "warning", msg))
            if redraw:  # two fresh draws, saved alike; the inspected figure has been drawn once already
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # counted above
                    first = svg(draw(cell))
                    second = svg(draw(cell))
                if first != second:
                    rows.append((cell.label, "redraw", "two draws give different SVG"))
        except Exception as e:  # noqa: BLE001 - a figure that cannot be drawn is reported, not raised
            rows.append((cell.label, "error", f"{type(e).__name__}: {e}"[:200]))
        finally:
            plt.close("all")
    return out


def counts(sites: dict) -> dict[str, dict[str, int]]:
    out = {}
    for ch, rows in sorted(sites.items()):
        c = {}
        for _, kind, _ in rows:
            c[kind] = c.get(kind, 0) + 1
        if c:
            out[ch] = dict(sorted(c.items()))
    return out


def load_baseline(path: Path = BASELINE) -> dict[str, dict[str, int]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {ch: {k: int(v) for k, v in kinds.items()} for ch, kinds in data.get("chapters", {}).items()}


def compare(current: dict, baseline: dict, chapters=None) -> list[str]:
    msgs = []
    for ch in sorted(set(current) | set(baseline)):
        if chapters and ch not in chapters:
            continue
        for kind in sorted(set(KINDS) | set(current.get(ch, {})) | set(baseline.get(ch, {}))):
            now, base = current.get(ch, {}).get(kind, 0), baseline.get(ch, {}).get(kind, 0)
            if kind == "error" and now:
                msgs.append(f"{ch} error: {now} figure(s) could not be drawn (--list shows them)")
            elif now > base:
                msgs.append(f"{ch} {kind}: {now}, baseline {base}: fix the new findings (--list shows them)")
            elif now < base:
                msgs.append(f"{ch} {kind}: {now}, baseline {base}: lower the baseline with --update")
    return msgs


def write_baseline(current: dict, baseline: dict, path: Path = BASELINE) -> list[str]:
    up = [f"{ch} {k}: {v} > baseline {baseline.get(ch, {}).get(k, 0)}" for ch, kinds in current.items()
          for k, v in kinds.items() if k == "error" or v > baseline.get(ch, {}).get(k, 0)]
    if up and path.exists():
        return up
    data = {"about": "per-chapter count of drawn-figure findings against STYLE 6.2; may only shrink "
                     "(tools/check_fig_output.py)",
            "chapters": current}
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return []


def check(report: Path = REPORT, baseline_path: Path = BASELINE) -> tuple[int, list[str]]:
    msgs = compare(counts(scan(report)), load_baseline(baseline_path))
    return (1 if msgs else 0), msgs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="print every finding")
    ap.add_argument("--update", action="store_true", help="lower the baseline to the current counts")
    ap.add_argument("--chapters", default="", help="comma-separated chapters to draw (no baseline verdict)")
    ap.add_argument("--no-redraw", action="store_true", help="skip the second draw (faster; no redraw findings)")
    ap.add_argument("--report", type=Path, default=REPORT, help=argparse.SUPPRESS)
    ap.add_argument("--baseline", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    chapters = {c.strip() for c in args.chapters.split(",") if c.strip()}
    baseline_path = args.baseline or args.report / "tools" / "figoutput-baseline.json"
    sites = scan(args.report, chapters or None, redraw=not args.no_redraw)
    current, baseline = counts(sites), load_baseline(baseline_path)
    if args.list:
        for ch in sorted(sites):
            for label, kind, detail in sites[ch]:
                print(f"{ch} {label}: {kind}: {detail}")
    if chapters or args.no_redraw:
        print(f"check_fig_output: {sum(len(v) for v in sites.values())} finding(s) in "
              f"{sum(1 for c in figure_cells(args.report) if not chapters or c.chapter in chapters)} figure(s) "
              "(partial run: no baseline verdict)")
        return 0
    if args.update:
        up = write_baseline(current, baseline, baseline_path)
        if up:
            print("check_fig_output: refusing to raise the baseline: " + "; ".join(up))
            return 1
        print(f"check_fig_output: baseline written, {sum(sum(c.values()) for c in current.values())} finding(s)")
        return 0
    msgs = compare(current, baseline)
    total = sum(sum(c.values()) for c in current.values())
    print(f"check_fig_output: {total} finding(s) in {len(figure_cells(args.report))} figure(s); baseline "
          f"{sum(sum(c.values()) for c in baseline.values())}" + (f"; {len(msgs)} count(s) differ" if msgs else ""))
    for m in msgs:
        print("  " + m)
    return 1 if msgs else 0


if __name__ == "__main__":
    sys.exit(main())
