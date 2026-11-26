#!/usr/bin/env python3
"""Generate reader-reference blocks from the pinned source and glossary.

N5 blocks are inline, so a missing include cannot silently hide an install line.
Checks compare the complete block, including its source links. No source checkout
or installed package is modified. Source text is read with git show at code_pin.yml.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import code_anchors

REPORT = Path(__file__).resolve().parent.parent
INSTALL = 'book/index.qmd'
RIEMANN = 'book/chapters/04-spatial-discretization/_riemann.qmd'


def marker(name, body):
    return f'<!-- generated reader-reference: {name} -->\n{body.rstrip()}\n<!-- end reader-reference: {name} -->'


def source_values(report, riemann=True):
    """The pinned values the generated blocks quote; riemann=False leaves out the Riemann parser's (its block is in
    a legacy chapter, tools/code-anchors-legacy.txt)."""
    repo, sha = code_anchors.read_pin(report / 'code_pin.yml')
    checkout = code_anchors.find_commit(repo, sha)

    def read(path):
        return subprocess.run(['git', '-C', str(checkout), 'show', f'{sha}:{path}'],
                              check=True, capture_output=True, text=True).stdout

    def match(pattern, text):
        matches = list(re.finditer(pattern, text, re.M))
        if len(matches) != 1:
            raise ValueError(f'expected one source match for {pattern!r}, found {len(matches)}')
        m = matches[0]
        return m[1], text[:m.start()].count('\n') + 1

    cmake, project = read('CMakeLists.txt'), read('pyproject.toml')
    header = read('src/riemann/riemann_solver.hpp') if riemann else ''
    parser = read('src/riemann/riemann_solver.cpp') if riemann else ''
    values = {}
    for key, path, text, pattern in [
        ('cmake', 'CMakeLists.txt', cmake, r'^cmake_minimum_required\(VERSION ([\d.]+)\)'),
        ('cxx', 'CMakeLists.txt', cmake, r'^set\(CMAKE_CXX_STANDARD (\d+)\)'),
        ('python', 'pyproject.toml', project, r'^requires-python = "([^"]+)"'),
        ('torch', 'pyproject.toml', project, r'^  "(torch==[^"]+)"'),
        ('no_block', 'src/riemann/riemann_solver.hpp', header,
         r'^  ADD_ARG\(std::string, type\) = "([^"]+)";'),
        ('no_type', 'src/riemann/riemann_solver.cpp', parser,
         r'^  op->type\(\) = node\["type"\]\.as<std::string>\("([^"]+)"\);'),
    ][:None if riemann else 4]:
        value, line = match(pattern, text)
        values[key] = value
        values[key + '_cite'] = f'[`{path}:{line}`](https://github.com/{repo}/blob/{sha}/{path}#L{line})'
    # Kintera is declared in both build and runtime requirements; require agreement.
    kintera = set(re.findall(r'"(kintera>=[^";]+)"', project))
    if len(kintera) != 1:
        raise ValueError('missing or conflicting Kintera requirements')
    values.update(kintera=kintera.pop(), sha=sha, repo=repo)
    return values


def defaults_blocks(report):
    riemann_legacy = code_anchors.legacy_chapter_of(RIEMANN, Path(report))
    v = source_values(report, riemann=not riemann_legacy)
    install = (
        f'**What you need.** Python `{v["python"]}` ({v["python_cite"]}), '
        f'CMake {v["cmake"]} or later ({v["cmake_cite"]}), a C++{v["cxx"]} compiler '
        f'({v["cxx_cite"]}), and the NetCDF C library. Install the latter with '
        '`brew install netcdf` on macOS or `sudo apt install libnetcdf-dev` on Debian/Ubuntu.\n\n'
        f'**Build the source pin.** The Python dependencies below follow '
        f'[`pyproject.toml`](https://github.com/{v["repo"]}/blob/{v["sha"]}/pyproject.toml); '
        'the CMake options explicitly enable the example drivers and disable the optional '
        'UCX and parallel-NetCDF backends for this single-machine build.\n\n'
        '```bash\npython3 -m venv snapy-env && source snapy-env/bin/activate\n'
        f'pip install numpy pyyaml "{v["kintera"]}" {v["torch"]}\n'
        f'git clone https://github.com/{v["repo"]} && cd snapy\n'
        f'git checkout {v["sha"]}\n'
        'cmake -B build -DCMAKE_BUILD_TYPE=Release \\\n'
        '  -DBUILD_EXAMPLES=ON -DNETCDF=ON -DPNETCDF=OFF -DUCX=OFF\n'
        'cmake --build build --parallel 6\npip install .\n```')
    if riemann_legacy:  # checked in full once its chapter's line is deleted from the legacy list
        return [(INSTALL, 'install', install)]
    riemann = (
        f'For hydrodynamics, omitting the `riemann-solver` block preserves `{v["no_block"]}` '
        f'({v["no_block_cite"]}); a present block without `type` selects `{v["no_type"]}` '
        f'({v["no_type_cite"]}). These are distinct parser paths at source pin `{v["sha"]}`.')
    return [(INSTALL, 'install', install), (RIEMANN, 'riemann', riemann)]


def sync_blocks(report, blocks, check):
    bad = 0
    for relative, name, body in blocks:
        path = report / relative
        text = path.read_text() if path.exists() else ''
        pattern = re.compile(r'<!-- generated reader-reference: ' + re.escape(name)
                             + r' -->.*?<!-- end reader-reference: ' + re.escape(name) + r' -->', re.S)
        matches = list(pattern.finditer(text))
        expected = marker(name, body)
        if len(matches) != 1:
            print(f'{relative}: {name}: expected exactly one generated block, found {len(matches)}')
            bad += 1
        elif matches[0][0] != expected:
            if check:
                print(f'{relative}: {name}: generated block is stale')
                bad += 1
            else:
                path.write_text(text[:matches[0].start()] + expected + text[matches[0].end():])
    return bad


def check_defaults(report, check=True):
    try:
        blocks = defaults_blocks(report)
        bad = sync_blocks(report, blocks, check)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'N5 pinned defaults: source error: {exc}')
        return 1
    print(f'N5 pinned defaults: {bad} problem(s); {len(blocks)} source-derived blocks')
    return bad



_GLOSS_NOTES = re.compile(r'\n<!-- reader glossary notes -->\n.*?<!-- end reader glossary notes -->\n?', re.S)
_GLOSS_KEYS = re.compile(r'^<!-- reader glossary keys -->.*\n', re.M)
_GLOSS_REF = re.compile(r'\[\^rg-[a-f0-9]{12}\]')


#: a registry source in the snapy tree: `path` or `path symbol`, resolved at the code pin by code_anchors.py
_SNAPY_SOURCE = re.compile(r'(?:src|tests|examples|python|external|cmake)/[\w./-]+(?: [\w:~]+)?')


def _kintera_source(report, source):
    """A link into kintera at its package pin, with a line or a line range, and optionally the symbol it shows."""
    try:
        pin = json.loads((Path(report) / 'tools/package-pins.json').read_text())['kintera']
    except (OSError, ValueError, KeyError):
        return False
    return bool(re.fullmatch(r'https://github\.com/chengcli/kintera/blob/%s/[^\s#]+#L\d+(?:-L\d+)?(?: [\w:~]+)?' % pin,
                             source))


def glossary_entries(report):
    """One inventory; symbol meanings are resolved against Appendix A's source."""
    import gen_notation_index as notation
    data = json.loads((report / 'tools/reader-glossary.json').read_text())
    if data.get('version') != 1 or not data.get('entries'):
        raise ValueError('glossary needs version 1 and a nonempty entries list')
    rows = {(s['num'], row['symbol']): row for s in notation.parse(
        (report / 'NOTATION.md').read_text())[1] for row in s['rows'] if 'symbol' in row}
    entries, seen = [], set()
    for item in data['entries']:
        if item.get('kind') == 'concept':  # named concepts: registry and glossary only, no first-use notes
            continue
        kind, term = item['kind'], item['term']
        if kind not in ('acronym', 'key', 'symbol') or not isinstance(term, str) or not term.strip():
            raise ValueError(f'invalid glossary entry: {item!r}')
        identity = (kind, term)
        if identity in seen:
            raise ValueError(f'duplicate glossary entry: {identity}')
        seen.add(identity)
        if kind == 'symbol':
            row = rows.get((item.get('notation_group'), term))
            if row is None:
                raise ValueError(f'glossary symbol is absent from Appendix A: {term}')
            meaning = notation._refs(row['meaning'])
            key = notation.symbol_key(term)
        else:
            meaning, key = item.get('meaning', ''), term
        if not isinstance(meaning, str) or not meaning.strip():
            raise ValueError(f'glossary has no definition for {kind} {term}')
        # Inline code can name a C++ local or an output field, not only a YAML key.
        # Explicit file-local senses keep first-use annotations tied to their context.
        local = item.get('meaning_by_file', {})
        if not isinstance(local, dict) or any(
                not isinstance(value, str) or not value.strip()
                or not (report / 'book' / path).is_file()
                for path, value in local.items()):
            raise ValueError(f'invalid local glossary meanings for {term}')
        source = item.get('source')
        notation_only = item.get('notation_only', False)
        if source is not None and (notation_only or not isinstance(source, str)
                                   or not (_SNAPY_SOURCE.fullmatch(source) or _kintera_source(report, source))):
            raise ValueError(f'invalid pinned glossary source for {term}')
        if not isinstance(notation_only, bool):
            raise ValueError(f'invalid notation-only marker for {term}')
        # a snapy source is cited by symbol at the code pin (the code shortcode, checked by code_anchors.py);
        # a kintera source is a link at kintera's package pin (tools/package-pins.json)
        if source and _SNAPY_SOURCE.fullmatch(source):
            cite = '{{< code ' + source + ' >}}'
        elif source:
            url, _, symbol = source.partition(' ')
            cite = (f'`{symbol}` ' if symbol else '') + f'[source]({url})'
        suffix = ('; ' + cite) if source else (
            '; notation only; see @sec-appa' if notation_only else '')
        meaning += suffix
        local = {path: value + suffix for path, value in local.items()}
        entries.append(dict(kind=kind, term=term, key=key, meaning=meaning,
                            meaning_by_file=local))
    if {e['kind'] for e in entries} != {'acronym', 'key', 'symbol'}:
        raise ValueError('glossary must cover acronyms, YAML keys and symbols')
    return entries


def glossary_outputs(report):
    """Attach a footnote at the first prose use in each chapter, in include order.

    Code, headings, attributes and provenance comments are not prose uses. YAML
    keys are recognised when named in inline code; executable examples stay intact.
    """
    import gen_notation_index as notation
    import gen_topic_index as topics
    entries = glossary_entries(report)
    lookup = {(e['kind'], e['key']): e for e in entries if e['key']}
    token = re.compile(r'\{\{<.*?>\}\}|`[^`\n]+`|(?<!\$)\$[^$\n]+\$(?!\$)|\b[A-Z][A-Z0-9]{1,6}\b')
    seen, outputs, uses = {}, {}, 0
    for path in topics.book_order(report / 'book'):
        rel = path.relative_to(report / 'book').as_posix()
        m = re.match(r'chapters/(\d\d)-(?!appendices)', rel)  # a chapter, not the appendices' directory
        if not m:
            continue
        chapter = m[1]
        used = seen.setdefault(chapter, set())
        text = _GLOSS_REF.sub('', _GLOSS_KEYS.sub('', _GLOSS_NOTES.sub('', path.read_text())))
        notes = []
        fence = comment = display = generated = False
        lines = []

        def replace(match):
            nonlocal uses
            word = match[0]
            if word.startswith('{{<'):
                return word
            if word.startswith('`'):
                identity = ('key', word[1:-1])
            elif word.startswith('$'):
                identity = ('symbol', notation._norm(word[1:-1]))
            else:
                identity = ('acronym', word)
            entry = lookup.get(identity)
            if entry is None or identity in used:
                return word
            used.add(identity)
            ident = hashlib.sha256((chapter + ':' + repr(identity)).encode()).hexdigest()[:12]
            note = f'rg-{ident}'
            notes.append(f'[^' + note + ']: ' + entry['term'] + ' — ' + entry['meaning_by_file'].get(rel, entry['meaning']))
            uses += 1
            return word + '[^' + note + ']'

        input_lines = text.splitlines(keepends=True)
        for line_index, line in enumerate(input_lines):
            stripped = line.lstrip()
            if '<!-- generated reader-reference:' in line:
                generated = True
            if generated:
                lines.append(line)
                if '<!-- end reader-reference:' in line:
                    generated = False
                continue
            if '<!--' in line:
                comment = True
            if comment:
                lines.append(line)
                if '-->' in line:
                    comment = False
                continue
            if stripped.startswith('```'):
                if not fence and stripped.strip() in ('```yaml', '```yml'):
                    keys = []
                    for future in input_lines[line_index + 1:]:
                        if future.lstrip().startswith('```'):
                            break
                        source = future.split(' #', 1)[0]
                        if source.lstrip().startswith('#'):
                            continue
                        for key in re.findall(r'(?:^\s*|[{,] *)([\w-]+):', source):
                            identity = ('key', key)
                            if identity not in lookup:
                                raise ValueError(f'{rel}: YAML key {key!r} has no glossary definition')
                            if identity not in used:
                                keys.append(token.sub(replace, '`' + key + '`'))
                    if keys:
                        lines.append('<!-- reader glossary keys -->Keys in this fragment: ' + ', '.join(keys) + '.\n')
                fence = not fence
                lines.append(line)
                continue
            if not fence and stripped.startswith('$$'):
                if line.count('$$') % 2:
                    display = not display
                lines.append(line)
                continue
            if fence or display or stripped.startswith('#'):
                lines.append(line)
            else:
                lines.append(token.sub(replace, line))
        output = ''.join(lines).rstrip() + '\n'
        if notes:
            output += '\n<!-- reader glossary notes -->\n' + '\n\n'.join(notes) + '\n<!-- end reader glossary notes -->\n'
        outputs[path] = output
    if not outputs:
        raise ValueError('glossary found no rendered chapters')
    return outputs, uses, len(entries)


def check_glossary(report, check=True):
    try:
        outputs, uses, entries = glossary_outputs(report)
        bad = 0
        for path, expected in outputs.items():
            if path.read_text() != expected:
                if check:
                    print(f'{path.relative_to(report)}: first-use glossary notes are stale')
                    bad += 1
                else:
                    path.write_text(expected)
        print(f'N4 glossary: {bad} problem(s); {entries} entries; {uses} chapter-first prose uses')
        return bad
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'N4 glossary: invalid input: {exc}')
        return 1

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', type=Path, default=REPORT)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--gate', choices=('all', 'glossary', 'defaults'), default='all')
    args = ap.parse_args(argv)
    args.report = args.report.resolve()
    bad = 0
    if args.gate in ('all', 'defaults'):
        bad += check_defaults(args.report, args.check)
    if args.gate in ('all', 'glossary'):
        bad += check_glossary(args.report, args.check)
    return int(bool(bad))


if __name__ == '__main__':
    raise SystemExit(main())
