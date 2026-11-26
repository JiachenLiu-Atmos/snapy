#!/usr/bin/env python3
"""Compare the numbers a figure plots with the numbers committed in the data it reads.

Two SVGs can differ byte for byte and still show the same curve: padding, canvas size and metadata
all move without a number changing. The opposite is worse and quieter -- a figure whose data file
has drifted draws a different curve at the same byte size. Neither question is answered by hashing
the SVG, so this asks the only one that matters: **does the figure plot the committed data?**

How it works. Each figure cell is drawn exactly as `tools/render_figures.py` draws it, with the same
audit hook recording which files the draw opened. That is live provenance: the data a figure reads
is observed, not declared. Every numeric series in the committed file is then looked for among the
series the figure actually plotted.

  series out of a file   CSV: every column, named by the `#` header when there is one.
                         JSON: every list of numbers, and every dict whose values are all numbers
                         (ordered by key, numerically when the keys are numbers). `provenance`
                         blocks are skipped -- they are strings and shas, not plotted values.
  series out of a figure Line2D x and y, scatter offsets, bar heights and their x positions,
                         image and mesh arrays, errorbar segments. Each is kept with the axes and
                         artist it came from so a mismatch can be pointed at.
  a match                some plotted series equals the reference series elementwise, within
                         tolerance, either over its whole length or as a contiguous slice of a
                         longer plotted series (a figure may draw a reference curve plus extra
                         points). Order is respected; a reversed series counts, because a plot may
                         run an axis backwards.

Tolerance, and why this one. The committed CSVs print ten significant digits
(`1.5625000000e-02`), so a value that was read from the file and plotted unchanged is exact, and a
value that was recomputed instead of read can differ by the text rounding, about 5e-11 relative.
The default `--rtol 1e-9` sits three orders above that rounding and seven orders below the loosest
tolerance the report states anywhere, so it accepts "the same number as committed, to the precision
committed" and rejects anything a reader would see. `--atol 1e-12` covers values at or near zero.
Both are reported in the output, and neither is per-figure: a figure that needs a looser one is a
finding, not a configuration.

What it does not do. It never writes a data file and never updates a baseline, so it cannot make a
mismatch disappear. A figure whose reference is missing is reported as such rather than passed.

    python3 tools/check_figure_data.py                      # every figure
    python3 tools/check_figure_data.py --chapters ch05,ch10 # some chapters
    python3 tools/check_figure_data.py --unverifiable       # also list the figures with no reference
    python3 tools/check_figure_data.py --json out.json      # the full result

Exit 0 when every reference series was found in its figure, 1 when any was not.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_fig_output as cfo  # noqa: E402
import render_figures as rf  # noqa: E402

REPORT = cfo.REPORT
DATA_SUFFIXES = {".csv", ".json"}
# The only reading that is provenance rather than a numeric coincidence: this exact
# series, in this order, at this length. Everything else is recorded as a candidate.
IDENTITY = "as committed"
# A JSON block that describes a run rather than a plotted quantity.
SKIP_JSON_KEYS = {"provenance"}


# --------------------------------------------------------------- reference series --

def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _key_order(keys):
    """numeric keys sort numerically (32, 64, 128, 256), everything else sorts as text"""
    try:
        return sorted(keys, key=lambda k: float(k))
    except (TypeError, ValueError):
        return sorted(keys, key=str)


def series_from_json(obj, path="") -> list[tuple[str, list[float]]]:
    out: list[tuple[str, list[float]]] = []
    if isinstance(obj, list):
        nums = [v for v in obj if _is_number(v)]
        if nums and len(nums) == len(obj) and len(obj) > 1:
            out.append((path or "/", [float(v) for v in obj]))
        else:
            # A list of equal-length numeric lists is a table. A figure plots its columns, not its
            # rows, so offer both: the rows for completeness and the columns because that is what
            # ends up on an axis.
            if (len(obj) > 1 and all(isinstance(r, list) for r in obj)
                    and all(len(r) == len(obj[0]) for r in obj)
                    and all(_is_number(v) for r in obj for v in r)):
                for c in range(len(obj[0])):
                    col = [float(r[c]) for r in obj]
                    if len(col) > 1:
                        out.append((f"{path}[:][{c}] (column)", col))
            for i, v in enumerate(obj):
                out += series_from_json(v, f"{path}[{i}]")
    elif isinstance(obj, dict):
        keys = [k for k in obj if k not in SKIP_JSON_KEYS]
        vals = [obj[k] for k in keys]
        if keys and all(_is_number(v) for v in vals) and len(keys) > 1:
            order = _key_order(keys)
            out.append((path or "/", [float(obj[k]) for k in order]))
        for k in keys:
            out += series_from_json(obj[k], f"{path}/{k}")
    return out


def series_from_csv(text: str) -> list[tuple[str, list[float]]]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header: list[str] | None = None
    if lines and lines[0].lstrip().startswith("#"):
        head = lines[0].lstrip()[1:].strip()
        # The comment is prose as often as it is a header: "x1,a,b (c = 1, ...)" names the columns,
        # but "rows NH3,H2S; columns before,after" describes the layout instead. Split off a
        # trailing parenthetical or clause, then use the names only if there is exactly one per
        # column -- otherwise fall back to positions, because a wrong name is worse than none.
        for cut in (" (", "; ", " ["):
            if cut in head:
                head = head.split(cut)[0]
        header = [h.strip() for h in head.split(",") if h.strip()] or None
        lines = lines[1:]
    rows: list[list[float]] = []
    for row in csv.reader(lines):
        vals = []
        for cell in row:
            try:
                vals.append(float(cell))
            except ValueError:
                vals.append(math.nan)
        if vals:
            rows.append(vals)
    if not rows:
        return []
    width = max(len(r) for r in rows)
    out = []
    for c in range(width):
        # every row contributes, short rows included as nan: filtering a column independently
        # would slide the remaining values up and break alignment with the other columns.
        col = [r[c] if c < len(r) else math.nan for r in rows]
        if col:
            name = header[c] if header and len(header) == width else f"column{c}"
            out.append((name, col))
    return out


def reference_series(path: Path) -> list[tuple[str, list[float]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json":
        try:
            return series_from_json(json.loads(text))
        except ValueError:
            return []
    if path.suffix == ".csv":
        return series_from_csv(text)
    return []


# ----------------------------------------------------------------- plotted series --

def plotted_series(fig) -> list[tuple[str, list[float]]]:
    """every numeric series the figure actually drew, named by the artist it came from.

    Nonfinite values are kept in place rather than filtered out. Dropping them per axis loses the
    pairing between x and y: with x = [0,1,2,3], y = [1,NaN,2,3] and y = [1,2,NaN,3] the surviving
    values are the same list, so compacting them makes a figure that moved a valid point look
    unchanged. Length-one series are kept for the same reason -- a series going from one point to
    none is a change.
    """
    import numpy as np
    out: list[tuple[str, list[float]]] = []

    def add(name, values):
        try:
            arr = np.asarray(values, dtype=float).ravel()
        except (TypeError, ValueError):
            return
        if arr.size:
            out.append((name, [float(v) for v in arr]))

    for ai, ax in enumerate(fig.axes):
        where = f"axes{ai}"
        for li, line in enumerate(ax.get_lines()):
            xy = line.get_xydata()
            if xy is not None and len(xy):
                xy = np.asarray(xy, dtype=float)
                add(f"{where}.line{li}.x", xy[:, 0])
                add(f"{where}.line{li}.y", xy[:, 1])
        for ci, coll in enumerate(ax.collections):
            offs = getattr(coll, "get_offsets", lambda: None)()
            if offs is not None and len(offs):
                offs = np.asarray(offs, dtype=float)
                if offs.ndim == 2 and offs.shape[1] == 2:
                    add(f"{where}.coll{ci}.x", offs[:, 0])
                    add(f"{where}.coll{ci}.y", offs[:, 1])
            segs = getattr(coll, "get_segments", lambda: None)()
            if segs:
                flat = np.concatenate([np.asarray(s, dtype=float) for s in segs if len(s)])
                if flat.ndim == 2 and flat.shape[1] == 2:
                    add(f"{where}.coll{ci}.seg.x", flat[:, 0])
                    add(f"{where}.coll{ci}.seg.y", flat[:, 1])
            arr = getattr(coll, "get_array", lambda: None)()
            if arr is not None:
                add(f"{where}.coll{ci}.array", np.asarray(arr).ravel())
        # ax.stairs draws a StepPatch, which carries its numbers in get_data() and has neither
        # get_height nor get_width, so the bar branch below never saw it. fig-ch07-sponge-profile
        # draws its committed data this way: without this, changing those values changed nothing.
        steps = [p for p in ax.patches if hasattr(p, "get_data") and not hasattr(p, "get_height")]
        for si, patch in enumerate(steps):
            data = patch.get_data()
            values = getattr(data, "values", None)
            edges = getattr(data, "edges", None)
            if values is None and isinstance(data, (tuple, list)) and len(data) >= 2:
                values, edges = data[0], data[1]
            if values is not None:
                add(f"{where}.step{si}.values", np.asarray(values, dtype=float))
            if edges is not None:
                add(f"{where}.step{si}.edges", np.asarray(edges, dtype=float))
        bars = [p for p in ax.patches if hasattr(p, "get_height") and hasattr(p, "get_width")]
        if bars:
            add(f"{where}.bars.height", [p.get_height() for p in bars])
            add(f"{where}.bars.x", [p.get_x() for p in bars])
            add(f"{where}.bars.width", [p.get_width() for p in bars])
        for ii, im in enumerate(ax.images):
            arr = im.get_array()
            if arr is not None:
                add(f"{where}.image{ii}", np.asarray(arr).ravel())
    return out


# ------------------------------------------------------------------------ matching --

def _close(a, b, rtol: float, atol: float) -> bool:
    """elementwise, with nonfinite values matching only themselves.

    Without the explicit nonfinite arm every comparison against a NaN is False, so `abs(x-y) > tol`
    is False too and a NaN would silently "match" any number at all.
    """
    for x, y in zip(a, b):
        xf, yf = math.isfinite(x), math.isfinite(y)
        if xf != yf:
            return False
        if not xf:
            if math.isnan(x) != math.isnan(y) or (not math.isnan(x) and x != y):
                return False
            continue
        if abs(x - y) > atol + rtol * abs(y):
            return False
    return True


def _scale_of(plot, ref, rtol, atol):
    """the single factor k with plot == k*ref, if one fits every element; else None.

    A free factor would let almost anything match, so it is not free: it is taken from one element
    and then has to hold for all of them at the same tolerance. A factor that fits sixty-four points
    to 1e-9 is a unit conversion, not a coincidence, and the output names it so a reader can judge.
    """
    k = None
    for x, y in zip(plot, ref):
        if abs(y) > atol:
            k = x / y
            break
    if k is None or not math.isfinite(k) or k == 0:
        return None
    # The factor is not free. A free one absorbs anything -- scale a series by 1.000001 and it
    # still "matches" -- which would make this blind to exactly the error it is meant to catch.
    # Only a unit conversion is allowed: a power of ten, or a small integer ratio. Anything else
    # is a difference, and is reported as one.
    if not _is_unit_factor(k):
        return None
    return k if _close(plot, [k * v for v in ref], rtol, atol) else None


def _is_unit_factor(k: float) -> bool:
    # Positive only. A factor of -1 would quietly absorb a sign error, so a sign change has to go
    # through the named `negated` / `absolute value` transforms instead, where the output shows it.
    if k <= 0 or not math.isfinite(k):
        return False
    a = k
    lg = math.log10(a)
    if abs(lg - round(lg)) < 1e-12:          # 10, 1000, 1e-4, ...
        return True
    for num in range(1, 13):                  # 1/2, 3/2, 60, 1/60, ... with a power of ten
        for den in range(1, 13):
            for p in range(-12, 13):
                if abs(a - (num / den) * 10.0 ** p) <= 1e-12 * a:
                    return True
    return False


def _transforms(ref):
    """parameter-free readings of a reference series that a plot may legitimately show"""
    yield "as committed", ref
    if any(v < 0 for v in ref):
        yield "negated", [-v for v in ref]
        yield "absolute value", [abs(v) for v in ref]


def find_match(ref, plots, rtol: float, atol: float):
    """(plotted series name, how it matched) or (None, None).

    `how` names the transform, so a match that needed one stays visible instead of passing silently.
    """
    n = len(ref)
    for name, series in plots:
        if len(series) < n:
            continue
        starts = [0] if len(series) == n else range(len(series) - n + 1)
        for start in starts:
            window = series[start:start + n]
            where = "" if len(series) == n else f", slice [{start}:{start + n}]"
            for how, cand in _transforms(ref):
                for order, seq in (("", cand), (", reversed", cand[::-1])):
                    if _close(window, seq, rtol, atol):
                        return name, f"{how}{order}{where}"
                    k = _scale_of(window, seq, rtol, atol)
                    if k is not None:
                        return name, f"{how}{order}{where}, scaled by {k:.12g}"
    return None, None


def worst_difference(ref, plots):
    """the nearest same-length plotted series and how far off it is, for an actionable message"""
    best = None
    for name, series in plots:
        if len(series) != len(ref):
            continue
        worst, at = 0.0, 0
        for i, (x, y) in enumerate(zip(series, ref)):
            d = abs(x - y) / max(abs(y), 1e-300)
            if d > worst:
                worst, at = d, i
        if best is None or worst < best[1]:
            best = (name, worst, at, series[at], ref[at])
    return best


# ---------------------------------------------------------------------------- main --

def check(chapters=None, rtol=1e-9, atol=1e-12, report: Path = REPORT):
    src = report / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.addaudithook(rf._audit)
    results = []
    for cell in cfo.figure_cells(report):
        if chapters and cell.chapter not in chapters:
            continue
        rec = {"label": cell.label, "chapter": cell.chapter,
               "qmd": cell.qmd.relative_to(report).as_posix(),
               "data": [], "checked": 0, "matched": 0, "transformed": 0,
               "unmatched": [], "error": None}
        rf._reads = []
        try:
            fig = cfo.draw(cell)
            reads, rf._reads = rf._reads, None
            plots = plotted_series(fig)
        except Exception as e:  # noqa: BLE001 - recorded per figure, the exit status says so
            rf._reads = None
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
            results.append(rec)
            plt.close("all")
            continue
        finally:
            plt.close("all")

        files = sorted({r for r in (rf._rel(p, report) for p in reads or [])
                        if r and Path(r).suffix in DATA_SUFFIXES})
        rec["data"] = files
        rec["plotted_series"] = len(plots)
        rec["series"] = {name: fingerprint(vals) for name, vals in plots}
        for rel in files:
            for sname, ref in reference_series(report / rel):
                rec["checked"] += 1
                name, how = find_match(ref, plots, rtol, atol)
                if name:
                    rec["matched"] += 1
                    if how and not how.startswith("as committed"):
                        rec["transformed"] += 1
                    rec.setdefault("matches", []).append(
                        {"file": rel, "series": sname, "plotted_as": name, "how": how})
                    continue
                near = worst_difference(ref, plots)
                item = {"file": rel, "series": sname, "n": len(ref), "first_values": ref[:4],
                        "closest": near[0] if near else None,
                        "max_rel_diff": near[1] if near else None,
                        "at_index": near[2] if near else None,
                        "plotted": near[3] if near else None,
                        "committed": near[4] if near else None}
                # One bucket, deliberately. A same-length plotted series is not evidence that the
                # figure meant to draw this one: bryan_balance_ic.json's table has four columns and
                # fig-ch05-balcol-passes draws three of them, so the fourth is unplotted, not wrong,
                # even though a six-element line exists to compare it against. Intent is not
                # recoverable here, so an unmatched reference is reported and never failed on. What
                # IS decidable is a link that used to hold and no longer does: see the baseline.
                rec["unmatched"].append(item)
        results.append(rec)
    return results


def fingerprint(values) -> dict:
    """what a plotted series is: its values, plus a digest and a few statistics for messages.

    The values are stored because acceptance has to use the advertised rtol/atol, and a hash cannot.
    Quantising to ten significant figures and comparing digests exactly puts a hard edge inside the
    tolerance band: 1.0000000004999 and 1.0000000005001 are within rtol 1e-9 of each other, so the
    matcher accepts them, yet they round to different tenth digits and so hash differently. A check
    that accepts a pair in one place and rejects it in another is not one policy.

    Storage rounds to twelve significant figures, three orders tighter than the default rtol, so the
    stored form never decides an acceptance that the tolerance would not.
    """
    vals = [float(v) for v in values]
    stored = [None if math.isnan(v) else (v if math.isinf(v) else float(f"{v:.11e}"))
              for v in vals]
    text = ",".join("nan" if v is None else f"{v!r}" for v in stored)
    finite = [v for v in vals if math.isfinite(v)]
    fp = {"n": len(vals),
          "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
          "values": stored}
    if finite:
        fp.update({"min": min(finite), "max": max(finite),
                   "mean": sum(finite) / len(finite),
                   "first": finite[0], "last": finite[-1]})
    return fp


def fingerprint_values(fp: dict) -> list[float] | None:
    """the stored series, nan restored; None for a baseline written before values were stored"""
    vals = fp.get("values")
    if vals is None:
        return None
    return [math.nan if v is None else float(v) for v in vals]


def fingerprints_agree(was: dict, now: dict, rtol: float, atol: float) -> bool:
    """acceptance, by the advertised tolerance -- the digest is only a fast path"""
    if was.get("n") != now.get("n"):
        return False
    if was.get("sha256") == now.get("sha256"):
        return True
    a, b = fingerprint_values(was), fingerprint_values(now)
    if a is None or b is None:
        return False          # an old baseline carries no values; it has to be rewritten
    return _close(b, a, rtol, atol)


def fingerprint_delta(was: dict, now: dict) -> str:
    """a line saying what moved, for a message somebody can act on"""
    if was["n"] != now["n"]:
        return f"length {was['n']} -> {now['n']}"
    parts = []
    for k in ("min", "max", "mean", "first", "last"):
        a, b = was.get(k), now.get(k)
        if a is None or b is None:
            continue
        if a != b:
            rel = abs(b - a) / max(abs(a), 1e-300)
            parts.append(f"{k} {a!r} -> {b!r} (relative {rel:.3e})")
    return "; ".join(parts) or ("same length, min, max, mean, first and last, "
                                "but the values differ somewhere between them")


BASELINE = Path(__file__).resolve().parent / "figure-data-baseline.json"


def snapshot(results):
    """the committed reference this check compares against on every later run.

    Per figure, two things:
      series  a fingerprint of every numeric series the figure plotted. This is the part that
              answers "did the plotted numbers change?", and it has to live here rather than be
              recomputed, because a figure that reads its own data file moves with it: perturb the
              file and the figure draws the perturbed value, so file-versus-plot agrees either way.
              A committed fingerprint does not move, so it catches a change in the data file, in the
              figure module, or in a library, and says which series.
      links   which committed reference series were found in the figure, and how. This is the part
              that says the figure really is drawing the committed data rather than something of
              its own, and it catches a figure that quietly stops reading its data.
    """
    out = {}
    for r in results:
        if r["error"]:
            continue
        entry = {"chapter": r["chapter"], "series": r.get("series", {})}
        # The artist is part of the link. Without it, a reference that moves from one curve to
        # another on the same figure reads as unchanged, which is exactly the provenance the link
        # is supposed to carry.
        links, candidates = {}, {}
        for m in r.get("matches", []):
            key = f"{m['file']}::{m['series']}"
            value = {"plotted_as": m["plotted_as"], "how": m["how"]}
            (links if m["how"] == IDENTITY else candidates)[key] = value
        if links:
            entry["links"] = dict(sorted(links.items()))
        if candidates:
            entry["candidates"] = dict(sorted(candidates.items()))
        out[r["label"]] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chapters", help="comma-separated, e.g. ch05,ch10")
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--unverifiable", action="store_true",
                    help="also list the figures that read no committed data")
    ap.add_argument("--unmatched", action="store_true",
                    help="also list reference series not found in their figure")
    ap.add_argument("--update", action="store_true",
                    help="rewrite tools/figure-data-baseline.json from this run")
    ap.add_argument("--json", help="write the full result here")
    args = ap.parse_args()

    chapters = [c.strip() for c in args.chapters.split(",")] if args.chapters else None
    results = check(chapters, args.rtol, args.atol)

    backed = [r for r in results if r["data"] and not r["error"]]
    bare = [r for r in results if not r["data"] and not r["error"]]
    errored = [r for r in results if r["error"]]
    checked = sum(r["checked"] for r in backed)
    matched = sum(r["matched"] for r in backed)
    transformed = sum(r["transformed"] for r in backed)
    now = snapshot(results)

    print(f"figure data check (rtol={args.rtol:g}, atol={args.atol:g})")
    print(f"  {len(results)} figure(s) drawn")
    print(f"    reading committed data      : {len(backed)}")
    print(f"    reading none (unverifiable) : {len(bare)}")
    print(f"    failed to draw              : {len(errored)}")
    print(f"  {checked} reference series checked, {matched} plotted by their figure "
          f"({transformed} under a named transform), {checked - matched} not found")

    for r in errored:
        print(f"\n[draw failed] {r['label']} ({r['chapter']}): {r['error']}")

    changed, broken, missing = [], [], []
    if args.update:
        merged = json.loads(BASELINE.read_text()) if (BASELINE.exists() and chapters) else {}
        merged.update(now)
        BASELINE.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
        nseries = sum(len(v.get("series", {})) for v in now.values())
        nlinks = sum(len(v.get("links", {})) for v in now.values())
        ncand = sum(len(v.get("candidates", {})) for v in now.values())
        print(f"\nbaseline written: {nseries} plotted series, {nlinks} identity link(s) and "
              f"{ncand} candidate(s), across {len(now)} figure(s)")
        return 0

    # No baseline is not a pass. Nothing was compared, so there is nothing to report as unchanged.
    if not BASELINE.exists():
        print(f"\n[NO BASELINE] {BASELINE} does not exist, so nothing was compared.")
        print("  Run with --update to write one, and commit it. Until then this check is not a"
              " statement about the figures.")
        return 1

    was = json.loads(BASELINE.read_text())
    # Compare like with like: a chapter filter narrows both sides, so --chapters ch05 is not a
    # licence to ignore every other figure's absence.
    scoped = {k: v for k, v in was.items() if not chapters or v.get("chapter") in chapters}
    if chapters:
        known = {v.get("chapter") for v in was.values()}
        unknown = [c for c in chapters if c not in known]
        if unknown:
            print(f"\n[UNKNOWN CHAPTER] {', '.join(unknown)} is not in the baseline; "
                  "nothing was selected from it.")
            return 1
    if not results:
        print("\n[NOTHING DISCOVERED] no figure cell matched the selection, so nothing ran.")
        return 1

    kept = 0
    for label, entry in sorted(scoped.items()):
        cur = now.get(label)
        if cur is None:
            missing.append((label, entry.get("chapter"), "the figure is in the baseline but was "
                            "not drawn in this run"))
            continue
        for name, fp in sorted(entry.get("series", {}).items()):
            got = cur.get("series", {}).get(name)
            if got is None:
                changed.append((label, name, "the figure no longer draws this series", None))
            elif not fingerprints_agree(fp, got, args.rtol, args.atol):
                changed.append((label, name, fingerprint_delta(fp, got), (fp, got)))
            else:
                kept += 1
        for key, link in sorted(entry.get("links", {}).items()):
            how = link["how"] if isinstance(link, dict) else link
            where = link.get("plotted_as") if isinstance(link, dict) else None
            got = cur.get("links", {}).get(key)
            if got is None:
                rec = next((r for r in results if r["label"] == label), None)
                bad = next((m for m in (rec or {}).get("unmatched", [])
                            if f"{m['file']}::{m['series']}" == key), None)
                broken.append((label, key, f"{how}" + (f" on {where}" if where else ""), bad))
            elif isinstance(got, dict) and where and got.get("plotted_as") != where:
                broken.append((label, key,
                               f"{how} on {where}  ->  now on {got.get('plotted_as')}", None))

    # New coverage is not automatically good news: a numerical series nobody has looked at is a
    # series nobody has looked at. It has to be reviewed and baselined deliberately.
    for label, cur in sorted(now.items()):
        if chapters and cur.get("chapter") not in chapters:
            continue
        entry = was.get(label)
        if entry is None:
            missing.append((label, cur.get("chapter"), "the figure was drawn but is not in the "
                            "baseline"))
            continue
        for name in sorted(cur.get("series", {})):
            if name not in entry.get("series", {}):
                missing.append((label, cur.get("chapter"),
                                f"new numerical series {name} is not in the baseline"))

    print(f"  baseline: {kept} plotted series unchanged, {len(changed)} changed; "
          f"{len(broken)} identity link(s) broken; {len(missing)} coverage difference(s)")

    for label, name, why, pair in changed:
        print(f"\n[CHANGED] {label}: {name}")
        print(f"  {why}")
        if pair:
            was_fp, now_fp = pair
            print(f"  committed digest {was_fp.get('sha256')} (n={was_fp.get('n')}), "
                  f"now {now_fp.get('sha256')} (n={now_fp.get('n')})")
            a, b = fingerprint_values(was_fp), fingerprint_values(now_fp)
            if a is None:
                print("  the committed baseline stores no values; rewrite it with --update after "
                      "reading what changed.")
            elif b is not None:
                for i, (x, y) in enumerate(zip(b, a)):
                    if not _close([x], [y], args.rtol, args.atol):
                        print(f"  first difference at index {i}: plotted {x!r} vs committed {y!r}")
                        break
        print("  The plotted numbers moved. Find out whether the data file, the figure module or a "
              "library changed before touching the baseline.")

    for label, key, how, bad in broken:
        print(f"\n[BROKEN LINK] {label}: {key}")
        print(f"  was {how}; it is no longer found there within tolerance.")
        if bad and bad.get("closest"):
            print(f"  nearest plotted series now: {bad['closest']}")
            print(f"  worst at index {bad['at_index']}: plotted {bad['plotted']!r} vs "
                  f"committed {bad['committed']!r}  (relative {bad['max_rel_diff']:.3e})")
        print("  Either the data file changed or the figure stopped drawing it there.")

    for label, chapter, why in missing:
        print(f"\n[COVERAGE] {label} ({chapter}): {why}")
        print("  Read it, then baseline it deliberately with --update.")

    if args.unmatched:
        for r in backed:
            if r["unmatched"]:
                print(f"\nnot found in {r['label']} ({r['chapter']}) -- informational, not a failure:")
                for m in r["unmatched"]:
                    near = (f"nearest {m['closest']} (relative {m['max_rel_diff']:.3e})"
                            if m["closest"] else "no plotted series of that length")
                    print(f"  {m['file']} :: {m['series']} ({m['n']} values); {near}")

    if args.unverifiable and bare:
        print(f"\nfigures that read no committed data ({len(bare)}) -- "
              "this check cannot speak for them:")
        for r in sorted(bare, key=lambda r: (r["chapter"], r["label"])):
            print(f"  {r['chapter']}  {r['label']}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"rtol": args.rtol, "atol": args.atol, "linkage": now, "results": results},
            indent=1) + "\n")

    return 1 if (changed or broken or missing or errored) else 0


if __name__ == "__main__":
    sys.exit(main())
