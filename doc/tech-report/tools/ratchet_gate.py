#!/usr/bin/env python3
"""Ratchet gate for the report's chapter sources (README, "Ratchet gate").

Four counts per chapter (and for the preface, `book/index.qmd`), compared with the committed baseline
`tools/ratchet-baseline.json`:

  history     change history in body text: PR/issue numbers and phrases such as "main after", "squash-merged",
              "the previous pin", "was removed in" (HISTORY_PATTERNS), matched across line breaks. The body text
              is the file without HTML
              comments (PROVENANCE), fenced code, `{{< >}}` shortcodes and footnotes, so a source or credit
              footnote may still carry a PR number.
  offpin      a sha of a cited package that is not that package's one pin: snapy's pin is `code_pin.yml`, the
              others are `tools/package-pins.json`. Counted everywhere in the file: body text, footnotes, link
              targets, code comments and fenced code, and the plain fields of the PROVENANCE header. It catches
              `<package>@<sha>`, `<path>@<sha>` (a file of the snapy tree), a GitHub blob/tree/commit URL of a
              package, and any other full 40-character sha that is no package's pin. A shortcode carries no sha.
  status      a file whose PROVENANCE says `status: reviewed` or `approved` without a valid entry in
              `reviews/review-record.tsv` (REVIEW RECORD below).
  xref        a `@sec-`/`@eq-`/`@fig-`/`@tbl-` reference that resolves nowhere or that quarto cannot reach
              (render_gate's own resolution), plus every `sec-` label that no file of another chapter cites.

THE RATCHET. `check` fails when any count differs from the committed baseline: a rise must be fixed, a fall must
lower the baseline in the same pull request (`update`). The baseline records the snapy pin it was measured at,
and it may rise only together with a move of that pin:
  * `update` refuses to write a higher count unless `code_pin.yml` names another sha than the baseline's `pin`;
  * `check --base-ref <ref>` (what CI runs) also compares the committed baseline with the one at `<ref>`, the
    target branch: a higher count there fails unless `code_pin.yml` differs between `<ref>` and the head, so a
    baseline raised by hand fails too.

OWNERS (`tools/chapter-owners.tsv`): one row per chapter, `chapter owner [checker]`, each given by GitHub handle only
(`-` for no checker; `check` fails when a checker is also the row's owner). The owner is
the single owner of the chapter's rewrite (the lock) and the owner its counts are charged to in the per-owner
burn-down table that `counts` and `check` print. That table also gives each owner's figures and tables (distinct
`fig-`/`tbl-` labels in their chapters), which are not ratcheted. `check` fails when a chapter has no owner or more
than one.

REVIEW RECORD (`reviews/review-record.tsv`, tab-separated, one line per event, newest last):

  file    kind    reviewer    author    pin    sha256    date    note

  file      path under book/, e.g. chapters/06-gravity-energy/_dwork.qmd
  kind      `review`: the reviewer read the file at this content and this pin and stands behind its status.
            `restamp`: the same reviewer confirms that review after the file or the pin changed without changing
            the verdict; the note says in one line what changed. Anyone else, or any change that alters the
            verdict, needs a new `review`.
  pin       the full sha of code_pin.yml at the time of the entry
  sha256    the file's content without its PROVENANCE comment (`python3 tools/ratchet_gate.py hash <file>`)

A reviewed/approved file passes when its newest entry has the file's current hash and the current pin, its
reviewer differs from its author (and from the PROVENANCE `author:` when that is given), and, for a `restamp`,
the reviewer is the one of the newest `review` entry and the note is not empty. Entries are written with
`record`, which fills in the hash, the pin and the date:

    python3 tools/ratchet_gate.py counts [-v]                   # per chapter and per owner; -v lists instances
    python3 tools/ratchet_gate.py check [--base-ref REF]        # CI: exit 1 on any difference or a raised baseline
    python3 tools/ratchet_gate.py update                        # write the counts as the baseline
    python3 tools/ratchet_gate.py hash <file>                   # the content hash a record entry carries
    python3 tools/ratchet_gate.py record review <file> --reviewer R --author A [--note N]
    python3 tools/ratchet_gate.py record restamp <file> --reviewer R --note N
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import re
import subprocess
import sys

TOOLS = pathlib.Path(__file__).resolve().parent
ROOT = TOOLS.parent
CHECKS = ("history", "offpin", "status", "xref")
# generated navigation (the topic index, the task map, the glossary), not prose: its citations do not make a section cited
# from another chapter (xref)
NAVIGATION = {"appf", "appg", "apph"}
RECORD_HEAD = ("file", "kind", "reviewer", "author", "pin", "sha256", "date", "note")

# Change history in body text. Built from STYLE.md (sections 1 and 3.2) and the phrases the Q1 re-read and the
# ch06 rewrite removed. Generic words that also describe physics ("no longer", "used to", "previously", "was
# removed") count only in their history forms. Matching is case-insensitive (except "PR") and runs on the body
# text with every run of whitespace made one space, so a phrase broken across source lines counts; overlapping
# matches count once.
HISTORY_PATTERNS = [
    r"\b(?:chengcli|UCzhangxi|zoeyzyhu|happysky19|jihenghu|XinyueWang05|gitlinffff)/[\w.-]+#\d+",  # owner/repo#N
    r"(?<![\w/&#^{])#\d{2,5}\b",                                   # bare #NNN, also as link text [#NNN]
    r"(?<![\w/])(?:snapy|kintera|pyharp|pydisort|commux)#\d+",     # repo#NNN without the owner
    r"\bmain (?:after|at)\b",
    r"\bdescribes? main\b",
    r"\b(?:since|until|before|after|as of) (?:PR |pull request )?#\d+",
    r"\bsquash[- ]merged?\b",
    r"\bpull request\b",
    r"\b(?:the |this |that )?(?:previous|old|earlier) pins?\b",
    r"\b(?:before|after|since) (?:and after )?(?:the|this|that) (?:change|merge|pin move)\b",
    r"\bbetween the (?:two )?pins\b",
    r"\b(?:checked|re-read|pinned) (?:at|against) (?:main\b|snapy@|`?[0-9a-f]{7})",
    r"\bwhile the rest of the (?:book|section)\b",
    r"\b(?:is|are) now\b",
    r"\bnow (?:defaults?|the default)\b",
    r"\bno longer (?:the )?default\b",
    r"\bnow fixed\b",
    # "used to" as a past habit ("the limiter used to clip"), not the passive or purpose use that technical prose
    # is full of: no form of "be", "not" or an adverb before it ("is not used to infer"), and no purpose verb after
    # it ("the interface used to select readers").
    r"(?<!\bis )(?<!\bare )(?<!\bwas )(?<!\bwere )(?<!\bbe )(?<!\bbeen )(?<!\bbeing )(?<!\bnot )(?<!\balso )"
    r"(?<!\boften )(?<!\bonly )(?<!\busually )\bused to\b(?! (?:select|infer|compute|evaluate|build|form|construct|"
    r"set|store|check|test|mark|define|derive|estimate|reconstruct|interpolate|weight|scale|bound|limit|fill|"
    r"update|convert|map|find|pick|choose|read|write|hold|carry|close|start|drive|measure|obtain|get|make|"
    r"produce|generate|represent|describe|approximate|correct|advance|solve|integrate|split|apply|enforce)\b)",
    r"\bwas previously\b",                                          # the editor's three forms (with "used to")
    r"\bin an earlier version\b",
    r"\b(?:was|were|has been|have been) (?:added|introduced|changed|fixed|replaced|removed|moved|renamed|dropped)"
    r"(?: separately| later| earlier| first)? (?:in|by|into)\b",
    r"\b(?:fixed|landed|merged) in\b",
    # misses of the second version, found in the book at tech-report f38085bb
    r"(?-i:\bPRs?\b)",                                             # "the PR head", "the same PR"
    r"\bcommit \d+\b",                                             # "commit 4 against its base"
    r"\bnow (?:also )?(?:[a-z]+s|set|use|run|read|call)\b",         # "now passes", "now also runs"
    r"\bthe old\b",                                                 # "the old off default", "the old flux"
    r"\b(?:is|was|are|were) superseded\b",
    r"\b(?:before|after|until) the fix\b",
    r"\bat the (?:PR )?head\b(?! of)|\bagainst (?:its|the) base\b",  # "passes at the head"
]
_HISTORY = [re.compile(p, re.I) for p in HISTORY_PATTERNS]

PACKAGES = ("snapy", "kintera", "pyharp", "pydisort", "commux")
_SHA_PATTERNS = [  # (regex, package group or None, sha group)
    (re.compile(r"github\.com/[\w.-]+/(%s)/(?:blob|tree|commit)/([0-9a-f]{7,40})" % "|".join(PACKAGES)), 1, 2),
    (re.compile(r"(?<![\w/.-])(%s)@([0-9a-f]{7,40})\b" % "|".join(PACKAGES)), 1, 2),
    # a file of the snapy tree at a sha, e.g. docs/derivations/note.md@<sha>
    (re.compile(r"(?<![\w/])(?:docs|src|tests|python|examples|cmake)/[\w./-]+@([0-9a-f]{7,40})\b"), None, 1),
    # any other full sha
    (re.compile(r"(?<![0-9a-f])([0-9a-f]{40})(?![0-9a-f])"), None, 1),
]


def chapter_of(rel: str) -> str:
    """The chapter a source file belongs to: ch01..ch17, appa..appe, or `preface` (files under book/)."""
    parts = pathlib.PurePosixPath(rel).parts
    if parts == ("index.qmd",):
        return "preface"
    if len(parts) < 2 or parts[0] != "chapters":
        return "book"
    top = parts[1]
    if top.startswith("app"):
        return top[:4]
    if top == "17-appendices" and len(parts) > 2:
        m = re.match(r"_(app[a-z])-", parts[2])
        if m:
            return m.group(1)
    m = re.match(r"(\d\d)-", top)
    return f"ch{m.group(1)}" if m else top


def sources(book: pathlib.Path) -> list[pathlib.Path]:
    """Every file the ratchet reads: the preface and every chapter and scheme file."""
    out = sorted((book / "chapters").rglob("*.qmd"))
    if (book / "index.qmd").exists():
        out.insert(0, book / "index.qmd")
    return out


def _strip_footnotes(text: str) -> str:
    out, in_def = [], False
    for line in text.split("\n"):
        if re.match(r"\[\^[^\]]+\]:", line):
            in_def = True
            continue
        if in_def and (line.startswith("    ") or line.startswith("\t")):
            continue
        in_def = False
        out.append(line)
    text = "\n".join(out)
    # inline footnotes ^[...] (no nesting of brackets inside a footnote in this book)
    return re.sub(r"\^\[[^\]]*\]", "", text)


def body_text(text: str, keep_links: bool = True) -> str:
    """The chapter text a reader sees: no HTML comments, fenced code, shortcodes or footnotes."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"^(```|~~~).*?^\1\s*$", "", text, flags=re.S | re.M)
    text = re.sub(r"\{\{<.*?>\}\}", "", text, flags=re.S)
    text = _strip_footnotes(text)
    if not keep_links:
        text = re.sub(r"\]\([^)]*\)", "]", text)
    return text


def _merged(text: str, spans) -> list[str]:
    merged: list[list[int]] = []  # overlapping matches ("after #307" and "#307") count once
    for a, b in sorted(spans):
        if merged and a < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [text[a:b] for a, b in merged]


def history_hits(text: str) -> list[str]:
    body = re.sub(r"\s+", " ", body_text(text, keep_links=False))
    return _merged(body, [(m.start(), m.end()) for rx in _HISTORY for m in rx.finditer(body)])


def pins(root: pathlib.Path) -> dict[str, str]:
    out = {}
    pin_file = root / "code_pin.yml"
    if pin_file.exists():
        m = re.search(r"^sha:\s*([0-9a-f]{40})", pin_file.read_text(), re.M)
        if m:
            out["snapy"] = m.group(1)
    extra = root / "tools" / "package-pins.json"
    if extra.exists():
        out.update(json.loads(extra.read_text()))
    return out


def offpin_hits(text: str, pin: dict[str, str]) -> list[str]:
    """Shas other than the package's pin, anywhere in the file except inside shortcodes."""
    text = re.sub(r"\{\{<.*?>\}\}", "", text, flags=re.S)
    pinned = set(pin.values())
    spans = []
    for rx, pkg_group, sha_group in _SHA_PATTERNS:
        for m in rx.finditer(text):
            sha = m.group(sha_group)
            if pkg_group:
                pkg = m.group(pkg_group)
            elif rx is _SHA_PATTERNS[2][0]:
                pkg = "snapy"  # a file of the snapy tree
            else:
                pkg = None
            if pkg is None:  # a bare full sha: fine when it is any package's pin
                ok = sha in pinned
            else:
                want = pin.get(pkg)
                ok = want is not None and want.startswith(sha)
            if not ok:
                spans.append((m.start(sha_group), m.end(sha_group)))
    return _merged(text, spans)


def content_hash(text: str) -> str:
    """sha256 of the file without its PROVENANCE comment, so status and reviewer edits keep the hash."""
    text = re.sub(r"<!--\s*PROVENANCE.*?-->", "", text, count=1, flags=re.S)
    return hashlib.sha256(text.encode()).hexdigest()


def _provenance(text: str) -> dict[str, str]:
    m = re.search(r"<!--\s*PROVENANCE(.*?)-->", text, re.S)
    fields = {}
    if m:
        for line in m.group(1).split("\n"):
            k = re.match(r"\s*(status|author|reviewer):\s*(.*)", line)
            if k:
                fields[k.group(1)] = k.group(2).strip()
    return fields


def read_record(root: pathlib.Path) -> dict[str, list[dict[str, str]]]:
    path = root / "reviews" / "review-record.tsv"
    entries: dict[str, list[dict[str, str]]] = {}
    if not path.exists():
        return entries
    lines = [l for l in path.read_text().split("\n") if l.strip() and not l.startswith("#")]
    head = lines[0].split("\t")
    for line in lines[1:]:
        row = dict(zip(head, line.split("\t")))
        entries.setdefault(row.get("file", ""), []).append(row)
    return entries


def status_problem(rel: str, text: str, record, pin: str) -> str | None:
    """None if the file's status needs no record or its record is valid, else the reason."""
    prov = _provenance(text)
    status = prov.get("status", "").split(";")[0].strip().lower()
    if status not in ("reviewed", "approved"):
        return None
    rows = record.get(rel, [])
    if not rows:
        return f"status {status} without a review-record entry"
    last = rows[-1]
    if last.get("sha256") != content_hash(text):
        return "content changed since the last record entry (new review, or the reviewer's re-stamp)"
    if last.get("pin") != pin:
        return "record entry names another pin than code_pin.yml (new review, or the reviewer's re-stamp)"
    author = last.get("author", "")
    reviewer = last.get("reviewer", "")
    if not reviewer or reviewer == author or reviewer == prov.get("author", "") != "":
        return "reviewer missing or the same as the author"
    if last.get("kind") == "restamp":
        reviews = [r for r in rows if r.get("kind") == "review"]
        if not reviews or reviews[-1].get("reviewer") != reviewer:
            return "re-stamp by someone other than the reviewer of the newest review"
        if not last.get("note", "").strip():
            return "re-stamp without a note"
    elif last.get("kind") != "review":
        return f"unknown record kind {last.get('kind')!r}"
    return None


def xref_counts(book: pathlib.Path) -> tuple[dict[str, int], list[str]]:
    """Per chapter: dangling/unreachable references (render_gate's resolution) and sections uncited from outside."""
    sys.path.insert(0, str(TOOLS))
    import render_gate  # noqa: E402  (the render's own resolution, reused)

    # every reference that resolves nowhere counts, including those render_gate files under its
    # unwritten-chapter and unwritten-scheme gates: the render fails on those too
    used, defined, deep = render_gate._xrefs(book)
    counts: dict[str, int] = {}
    detail = []
    for lab, files in used.items():
        if lab not in defined or lab in deep:
            for f in files:
                counts[chapter_of(f)] = counts.get(chapter_of(f), 0) + 1
                detail.append(f"{f}: @{lab} does not resolve")
    home = {}
    for q in sorted((book / "chapters").rglob("*.qmd")):
        rel = str(q.relative_to(book))
        for lab in re.findall(r"\{#(sec-[\w-]+)\}", q.read_text()):
            home[lab] = chapter_of(rel)
    for lab, ch in sorted(home.items()):
        # the topic index (appendix F) cites every section by construction; a section counts as cited only when
        # prose of another chapter cites it
        outside = {chapter_of(f) for f in used.get(lab, ())} - {ch} - NAVIGATION
        if not outside:
            counts[ch] = counts.get(ch, 0) + 1
            detail.append(f"{ch}: section {lab} is not cited from another chapter")
    return counts, detail


def count(root: pathlib.Path = ROOT, verbose: bool = False) -> dict[str, dict[str, int]]:
    book = root / "book"
    pin = pins(root)
    record = read_record(root)
    table: dict[str, dict[str, int]] = {}

    def add(ch, check, n=1):
        table.setdefault(ch, dict.fromkeys(CHECKS, 0))[check] += n

    for q in sources(book):
        rel = str(q.relative_to(book))
        ch = chapter_of(rel)
        text = q.read_text()
        table.setdefault(ch, dict.fromkeys(CHECKS, 0))
        for h in history_hits(text):
            add(ch, "history")
            if verbose:
                print(f"history  {rel}: {h}")
        for h in offpin_hits(text, pin):
            add(ch, "offpin")
            if verbose:
                print(f"offpin   {rel}: {h}")
        why = status_problem(rel, text, record, pin.get("snapy", ""))
        if why:
            add(ch, "status")
            if verbose:
                print(f"status   {rel}: {why}")
    xc, detail = xref_counts(book)
    for ch, n in xc.items():
        add(ch, "xref", n)
    if verbose:
        for d in detail:
            print(f"xref     {d}")
    return dict(sorted(table.items()))


def read_owners(root: pathlib.Path) -> tuple[dict[str, list[str]], list[str]]:
    """{chapter: [GitHub handle, ...]} from tools/chapter-owners.tsv, and the problems of the file itself."""
    path = root / "tools" / "chapter-owners.tsv"
    owners: dict[str, list[str]] = {}
    if not path.exists():
        return owners, ["tools/chapter-owners.tsv is missing"]
    lines = [l for l in path.read_text().split("\n") if l.strip() and not l.startswith("#")]
    for line in lines[1:]:
        f = line.split("\t")
        if len(f) not in (2, 3) or not f[1].strip() or (len(f) == 3 and not f[2].strip()):
            return owners, [f"tools/chapter-owners.tsv: malformed row {line!r} (want: chapter, handle[, checker])"]
        if len(f) == 3 and f[2].strip() == f[1].strip():
            return owners, [f"tools/chapter-owners.tsv: {f[0]} has the same writer and checker ({f[1].strip()})"]
        owners.setdefault(f[0], []).append(f[1].strip())
    return owners, []


def owner_problems(counts, owners) -> list[str]:
    out = []
    for ch in sorted(set(counts) | set(owners)):
        rows = owners.get(ch, [])
        if not rows:
            out.append(f"{ch}: no owner in tools/chapter-owners.tsv")
        elif len(rows) > 1:
            out.append(f"{ch}: {len(rows)} owners in tools/chapter-owners.tsv; one owner per chapter")
    return out


def compare(counts, baseline) -> list[str]:
    problems = []
    for ch in sorted(set(counts) | set(baseline)):
        for check in CHECKS:
            now = counts.get(ch, {}).get(check, 0)
            was = baseline.get(ch, {}).get(check, 0)
            if now > was:
                problems.append(f"{ch} {check}: {was} -> {now}, a rise; fix the new instances")
            elif now < was:
                problems.append(f"{ch} {check}: {was} -> {now}, a fall; lower the baseline in this PR "
                                f"(python3 tools/ratchet_gate.py update)")
    return problems


def _git_show(root: pathlib.Path, ref: str, rel: str) -> str | None:
    """The content of `rel` (relative to doc/tech-report) at `ref`, or None if it is not there."""
    prefix = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-prefix"], capture_output=True,
                            text=True).stdout.strip()
    r = subprocess.run(["git", "-C", str(root), "show", f"{ref}:{prefix}{rel}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def baseline_raised(root: pathlib.Path, ref: str, baseline: dict) -> list[str]:
    """Counts the committed baseline raises against the baseline at `ref`, when the pin did not move.

    A `ref` that names no commit is a problem in itself: skipping the comparison would turn the protection off."""
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return [f"--base-ref {ref!r} names no commit here (fetch it first): the baseline cannot be compared"]
    old = _git_show(root, ref, "tools/ratchet-baseline.json")
    if old is None:
        return []  # no baseline at the target yet: this change introduces it
    old_pin = re.search(r"^sha:\s*([0-9a-f]{40})", _git_show(root, ref, "code_pin.yml") or "", re.M)
    if old_pin and old_pin.group(1) != pins(root).get("snapy"):
        return []  # the pin moves in this change: a rise is allowed
    old_chapters = json.loads(old)["chapters"]
    out = []
    for ch in sorted(set(baseline) | set(old_chapters)):
        for check in CHECKS:
            now, was = baseline.get(ch, {}).get(check, 0), old_chapters.get(ch, {}).get(check, 0)
            if now > was:
                out.append(f"{ch} {check}: the baseline rises {was} -> {now} against {ref} without a pin move")
    return out


def assets(root: pathlib.Path = ROOT) -> dict[str, dict[str, int]]:
    """Figures and tables per chapter: distinct `fig-`/`tbl-` labels (`#| label:` or `{#...}`), not in comments.

    Not ratcheted; the per-owner table prints them so an owner's counts read against the size of the work."""
    out: dict[str, dict[str, int]] = {}
    for q in sources(root / "book"):
        text = re.sub(r"<!--.*?-->", "", q.read_text(), flags=re.S)
        row = out.setdefault(chapter_of(str(q.relative_to(root / "book"))), {"figures": set(), "tables": set()})
        for key, kind in (("figures", "fig"), ("tables", "tbl")):
            row[key] |= set(re.findall(r"(?:#\|\s*label:\s*|\{#)(%s-[\w-]+)" % kind, text))
    return {ch: {k: len(v) for k, v in row.items()} for ch, row in out.items()}


def print_table(counts, owners=None, sizes=None) -> None:
    print("| chapter | " + " | ".join(CHECKS) + " |")
    print("|---|" + "---|" * len(CHECKS))
    tot = dict.fromkeys(CHECKS, 0)
    for ch, c in counts.items():
        print(f"| {ch} | " + " | ".join(str(c[k]) for k in CHECKS) + " |")
        for k in CHECKS:
            tot[k] += c[k]
    print("| **total** | " + " | ".join(f"**{tot[k]}**" for k in CHECKS) + " |")
    if owners is None:
        return
    sizes = sizes or {}
    per: dict[str, dict[str, int]] = {}
    size: dict[str, dict[str, int]] = {}
    chapters: dict[str, list[str]] = {}
    for ch, c in counts.items():
        who = owners.get(ch, ["(no owner)"])[0]
        row = per.setdefault(who, dict.fromkeys(CHECKS, 0))
        chapters.setdefault(who, []).append(ch)
        for k in CHECKS:
            row[k] += c[k]
        for k in ("figures", "tables"):
            size.setdefault(who, {"figures": 0, "tables": 0})[k] += sizes.get(ch, {}).get(k, 0)
    print("\n| owner | chapters | figures | tables | " + " | ".join(CHECKS) + " | total |")
    print("|---|---|---|---|" + "---|" * (len(CHECKS) + 1))
    for who in sorted(per, key=lambda w: (-sum(per[w].values()), w)):
        r = per[who]
        print(f"| {who} | {', '.join(chapters[who])} | {size[who]['figures']} | {size[who]['tables']} | "
              + " | ".join(str(r[k]) for k in CHECKS) + f" | {sum(r.values())} |")


def record(root: pathlib.Path, kind: str, file: str, reviewer: str, author: str, note: str) -> int:
    """Append a review or re-stamp entry with the file's current hash, the current pin and today's date."""
    book = root / "book"
    path = pathlib.Path(file)
    rel = str(path.resolve().relative_to(book.resolve())) if path.is_absolute() or path.exists() else file
    target = book / rel
    if not target.exists():
        print(f"record: {target} does not exist")
        return 1
    rec = read_record(root)
    rows = rec.get(rel, [])
    if kind == "restamp":
        reviews = [r for r in rows if r.get("kind") == "review"]
        if not reviews:
            print(f"record: {rel} has no review to re-stamp; write a review")
            return 1
        if reviews[-1].get("reviewer") != reviewer:
            print(f"record: only {reviews[-1].get('reviewer')}, who reviewed {rel}, can re-stamp it")
            return 1
        if not note.strip():
            print("record: a re-stamp needs --note, one line on what changed")
            return 1
        author = author or reviews[-1].get("author", "")
    if not reviewer or reviewer == author:
        print("record: the reviewer must be named and differ from the author")
        return 1
    rec_path = root / "reviews" / "review-record.tsv"
    if not rec_path.exists():
        rec_path.write_text("\t".join(RECORD_HEAD) + "\n")
    fields = [rel, kind, reviewer, author, pins(root).get("snapy", ""), content_hash(target.read_text()),
              datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"), note.replace("\t", " ").strip()]
    with rec_path.open("a") as f:
        f.write("\t".join(fields) + "\n")
    print(f"record: {kind} of {rel} by {reviewer} written")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=pathlib.Path, default=ROOT, help="doc/tech-report (default: this tree)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("counts", help="print the per-chapter and per-owner counts")
    c.add_argument("-v", "--verbose", action="store_true", help="list every instance")
    k = sub.add_parser("check", help="exit 1 if any count differs from the baseline or the baseline was raised")
    k.add_argument("--base-ref", help="the target branch: its baseline may not be raised without a pin move")
    sub.add_parser("update", help="write the counts as the new baseline")
    h = sub.add_parser("hash", help="content hash of a chapter file, for the review record")
    h.add_argument("file", type=pathlib.Path)
    r = sub.add_parser("record", help="append a review or re-stamp entry to reviews/review-record.tsv")
    r.add_argument("kind", choices=("review", "restamp"))
    r.add_argument("file", help="the file, as a path or relative to book/")
    r.add_argument("--reviewer", required=True)
    r.add_argument("--author", default="", help="required for a review; a re-stamp takes the review's")
    r.add_argument("--note", default="", help="required for a re-stamp: one line on what changed")
    a = ap.parse_args(argv)
    base_path = a.root / "tools" / "ratchet-baseline.json"
    if a.cmd == "hash":
        print(content_hash(a.file.read_text()))
        return 0
    if a.cmd == "record":
        return record(a.root, a.kind, a.file, a.reviewer, a.author, a.note)
    counts = count(a.root, verbose=getattr(a, "verbose", False))
    owners, owner_file_problems = read_owners(a.root)
    if a.cmd == "counts":
        print_table(counts, owners, assets(a.root))
        return 0
    stored = json.loads(base_path.read_text()) if base_path.exists() else {}
    baseline = stored.get("chapters", {})
    pin = pins(a.root).get("snapy", "")
    if a.cmd == "update":
        rises = [p for p in compare(counts, baseline) if "a rise" in p]
        if rises and baseline and stored.get("pin") == pin:
            print("ratchet: refusing to raise the baseline without a pin move:\n  " + "\n  ".join(rises))
            return 1
        base_path.write_text(json.dumps({"pin": pin, "chapters": counts}, indent=1, sort_keys=True) + "\n")
        print(f"ratchet: baseline written ({len(counts)} chapters, pin {pin[:12]})")
        return 0
    problems = compare(counts, baseline)
    if stored and stored.get("pin") != pin:
        problems.append(f"the baseline was measured at pin {str(stored.get('pin'))[:12]}, code_pin.yml names "
                        f"{pin[:12]}: run update in the pin move")
    if a.base_ref:
        problems += baseline_raised(a.root, a.base_ref, baseline)
    problems += owner_file_problems + owner_problems(counts, owners)
    print_table(counts, owners, assets(a.root))
    if problems:
        print("\nratchet gate: FAIL\n  " + "\n  ".join(problems))
        print("  (python3 tools/ratchet_gate.py counts -v lists every instance)")
        return 1
    tot = {k: sum(c[k] for c in counts.values()) for k in CHECKS}
    print("\nratchet gate: pass (" + ", ".join(f"{k} {v}" for k, v in tot.items()) + ", equal to the baseline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
