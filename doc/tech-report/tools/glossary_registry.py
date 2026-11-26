"""The concept registry: tools/reader-glossary.json, one entry per concept, each with its defining section.

Every entry (acronym, YAML key, symbol, named concept) carries `section`, the label of the section that defines it
(STYLE, Reader standards N1 and N4). A section is never a chapter or appendix heading (a roll-up). What counts as a
definition is read in the book's own prose, including paragraphs under `##` headings without a label and included
files, never in the generated first-use notes:
  - a symbol is defined by its row in Appendix A, so its section is its group (`sec-appa-<notation_group>`);
  - an acronym is defined where the prose writes it next to its expansion ("equation of state (EOS)",
    "WENO, weighted essentially non-oscillatory", "MC, the monotonised-central limiter", "PDE — partial ...");
  - a YAML key is defined by a row of a table whose first column is headed "key" and names it ("EOS `limiter`",
    "`dynamics/disable-flux-x1`, `-x2`"), by a reference-table row whose first cell names only keys, or by prose
    that makes it the subject of a definition ("`nu_iso` is a kinematic viscosity", "`top-cool` and `bot-heat`
    stand for ...", "`gammad` is $...$");
  - a named concept is defined by prose that makes it the subject the same way ("a Riemann solver is a ...",
    "a `MeshBlock` is the unit ..." for a concept that is also a class name).
An acronym whose definition is only in its chapter's introduction (the acronym list of chapter 4) points to its own
section of that chapter: the one whose title names it, else holds its expansion, else first uses it.
An entry the book defines nowhere is marked `"not_defined_here": true`: the glossary says "not defined here" for
it, and its section is the first section of its own that uses it in prose, so the reader can still reach it (N1).

Named concepts are the eponymous method names of the prose ("Riemann solver", "Courant number"): a capitalised
name followed by a method word, used mid-sentence at least twice, whose name is not also an ordinary lower-case
word of the book (`concept_candidates`). They have registry, glossary and topic-index entries, no first-use notes.
A concept may also be a reader term added by hand ("ghost zone", "MeshBlock"); `--seed` keeps it, and like any
unmarked entry it must be defined in its section.

The count of entries marked "not defined here" is a ratchet, tools/not-defined-here-baseline.json, set at the
commit that lands the registry: it may not rise, and a fall is recorded with --update. A fall is verified,
because an entry that is not marked must pass the definition check, and a marked one may not have a definition.

usage:
  python3 tools/glossary_registry.py              check the registry and the ratchet (exit 0 when clean)
  python3 tools/glossary_registry.py --seed       rewrite `section` and the mark of every entry from the book
  python3 tools/glossary_registry.py --update     write the current count of marked entries as the baseline
  python3 tools/glossary_registry.py --count      print the count, and the entries whose section is a roll-up
"""
import json
import re
import sys
from pathlib import Path

REPORT = Path(__file__).resolve().parent.parent
REGISTRY = 'tools/reader-glossary.json'
BASELINE = 'tools/not-defined-here-baseline.json'
MARK = 'not_defined_here'
_ANY_HEADING = re.compile(r'^(#{1,6}) (.*?)\s*(\{[^{}]*\})?\s*$')
_LABEL = re.compile(r'#(sec-[A-Za-z0-9_-]+)')
# generated pages that list every concept; they are not where a concept is defined
_GENERATED = ('_apph-glossary.qmd', '_appf-topic-index.qmd', 'apph-glossary.qmd', 'appf-topic-index.qmd',
              '_appi-deck-keys.qmd', 'appi-deck-keys.qmd')


def _tools():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gen_notation_index as notation
    import gen_reader_reference as reference
    import gen_topic_index as topics
    return notation, reference, topics


def labels(book):
    """Every section label of the book's sources, with the file, title and level that define it."""
    out = {}
    for q in sorted(Path(book).rglob('*.qmd')):
        rel = q.relative_to(book).as_posix()
        if rel.startswith(('_', '.')):
            continue
        for line in q.read_text(encoding='utf-8').splitlines():
            m = re.match(r'^(#{1,6}) (.*?)\s*\{#(sec-[A-Za-z0-9_-]+)[^}]*\}', line)
            if m:
                out[m[3]] = dict(file=rel, title=m[2].strip(), level=len(m[1]))
    return out


def _heading(line, section):
    """The section after `line` if it is a heading: its label; None under an unlabelled `#`/`##` heading."""
    m = _ANY_HEADING.match(line)
    if not m:
        return False, section
    label = _LABEL.search(m[3] or '')
    if label:
        return True, label[1]
    return True, (None if len(m[1]) <= 2 else section)


def prose_uses(report):
    """For each concept identity (kind, key): the (chapter, section) pairs of its prose uses, in reading order.

    The same tokens and the same exclusions as the first-use notes of tools/gen_reader_reference.py: code
    fences (but YAML keys inside YAML fences count), display maths, headings, comments and generated blocks are
    not prose; chapters 01 to 16 are scanned. `section` is the label of the enclosing labelled heading; it is
    the chapter label before the first section, and None under a `##` heading with no label."""
    notation, reference, topics = _tools()
    entries = reference.glossary_entries(report)
    lookup = {(e['kind'], e['key']): e for e in entries if e['key']}
    token = re.compile(r'\{\{<.*?>\}\}|`[^`\n]+`|(?<!\$)\$[^$\n]+\$(?!\$)|\b[A-Z][A-Z0-9]{1,6}\b')
    uses = {}
    for path in topics.book_order(report / 'book'):
        rel = path.relative_to(report / 'book').as_posix()
        m = re.match(r'chapters/(\d\d)-(?!appendices)', rel)  # a chapter, not the appendices' directory
        if not m:
            continue
        chapter = 'ch' + m[1]
        raw = path.read_text(encoding='utf-8')
        text = reference._GLOSS_REF.sub('', reference._GLOSS_KEYS.sub('', reference._GLOSS_NOTES.sub('', raw)))
        section = None
        fence = comment = display = generated = yaml_fence = False
        for line in text.splitlines():
            stripped = line.lstrip()
            if not fence:
                is_heading, section = _heading(line, section)
            if '<!-- generated reader-reference:' in line:
                generated = True
            if generated:
                if '<!-- end reader-reference:' in line:
                    generated = False
                continue
            if '<!--' in line:
                comment = True
            if comment:
                if '-->' in line:
                    comment = False
                continue
            if stripped.startswith('```'):
                yaml_fence = not fence and stripped.strip() in ('```yaml', '```yml')
                fence = not fence
                continue
            if fence:
                if yaml_fence:
                    source = line.split(' #', 1)[0]
                    if not source.lstrip().startswith('#'):
                        for key in re.findall(r'(?:^\s*|[{,] *)([\w-]+):', source):
                            if ('key', key) in lookup:
                                uses.setdefault(('key', key), []).append((chapter, section))
                continue
            if stripped.startswith('$$'):
                if line.count('$$') % 2:
                    display = not display
                continue
            if display or stripped.startswith('#'):
                continue
            for word in token.findall(line):
                if word.startswith('{{<'):
                    continue
                if word.startswith('`'):
                    identity = ('key', word[1:-1])
                elif word.startswith('$'):
                    identity = ('symbol', notation._norm(word[1:-1]))
                else:
                    identity = ('acronym', word)
                if identity in lookup:
                    uses.setdefault(identity, []).append((chapter, section))
    return entries, uses


KEY_TABLE = '\x00key-table\x00'
_SUBJECT = (r'(?:^|[.;:,!?]\s+)(?:(?:[Tt]he|[Aa]n?)\s+)?((?:{item}(?:,\s*|\s+and\s+|\s+or\s+)?)+)(?:\s*\([^)]*\))?\s+'
            r'(?:(?:is|are)\s+(?:a|an|the\b|\$)|stands?\s+for\b|means?\b)')
_CONCEPT_NAME = r'(?:van |de )?[A-Z][a-z]+(?:[–-](?:van )?[A-Z][a-z]+)*'
_CONCEPT_WORD = (r'(?:method|scheme|solver|flux|limiter|algorithm|iteration|number|form|condition|correction|average|'
                 r'reconstruction|approximation|splitting|frame|force|drag|damping|equations?|matrix|stage|step|'
                 r'problem|state|estimate|fallback|bubble|wave|balance)')


def paragraphs(report):
    """(sections in reading order, {section: [paragraph]}): the prose of the book, with code, comments, generated
    blocks and the generated first-use notes removed. A table row is one paragraph; a row of a table whose first
    column is headed "key" starts with KEY_TABLE. Paragraphs under a `##` heading with no label are kept under
    `?<file>#<heading>`, so a definition there is read, though it is not a section a reader can be sent to. An
    included file continues the section in force where it is included."""
    _, reference, topics = _tools()
    order, paras, section = [], {}, None
    for path in topics.book_order(report / 'book'):
        if path.name in _GENERATED:
            continue
        rel = path.relative_to(report / 'book').as_posix()
        raw = path.read_text(encoding='utf-8')
        text = reference._GLOSS_REF.sub('', reference._GLOSS_KEYS.sub('', reference._GLOSS_NOTES.sub('', raw)))
        buf = []  # an included file continues the section it is included in
        fence = generated = comment = False
        table = None  # None outside a table, else whether its first column is headed "key"

        def flush():
            if buf and section:
                paras.setdefault(section, []).append(' '.join(buf))
            buf.clear()

        for line in text.splitlines():
            if not fence:
                m = _ANY_HEADING.match(line)
                if m:
                    flush()
                    label = _LABEL.search(m[3] or '')
                    if label:
                        section = label[1]
                    elif len(m[1]) <= 2:
                        section = f'?{rel}#{m[2].strip()}'
                    if section not in order:
                        order.append(section)
                    table = None
                    continue
            if line.lstrip().startswith('```'):
                flush()
                fence = not fence
                continue
            if '<!-- generated reader-reference:' in line:
                generated = True
            if generated:
                generated = '<!-- end reader-reference:' not in line
                continue
            if '<!--' in line:
                comment = True
            if comment:
                comment = '-->' not in line
                continue
            if fence:
                continue
            s = line.strip()
            if s.startswith('|'):
                flush()
                cells = [c.strip().lower() for c in s.strip('|').split('|')]
                if table is None:
                    table = cells[0] in ('key', 'keys')
                elif not set(s) <= set('|-: '):
                    buf.append((KEY_TABLE if table else '') + s)
                    flush()
                continue
            table = None
            if not s or re.match(r'([-*]|\d+\.)\s', s):
                flush()
            if s:
                buf.append(s)
        flush()
    return order, paras


def _norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()


def _spans(cell):
    """The code spans of a table cell; a span such as `-x2` after `disable-flux-x1` completes its stem."""
    out = []
    for s in re.findall(r'`([^`]+)`', cell):
        if s.startswith('-') and out and '-' in out[-1]:
            s = out[-1].rsplit('-', 1)[0] + s
        out.append(s)
    return out


def defines(item, paragraph):
    """Whether `paragraph` defines `item` (see the module docstring)."""
    term = item['term']
    if item['kind'] == 'acronym':
        words = _norm(item['meaning'].split(',')[0]).split()[:3]
        if not words:
            return False
        w = r'[^A-Za-z0-9]+'.join(map(re.escape, words))
        a = re.escape(term)
        return bool(re.search(r'\b%s\b[*_]*\s*(,|\(|—|–|:|=)\s*[*_]*(?:(?:the|a|an)\s+)?%s|%s[^.;()]{0,60}\(\s*[*_]*%s\b'
                              % (a, w, w, a), paragraph, re.I))
    if item['kind'] == 'key':
        if paragraph.startswith(KEY_TABLE):
            first = paragraph[len(KEY_TABLE):].strip().strip('|').split('|')[0]
            return any(s == term or s.endswith('/' + term) for s in _spans(first))
        if paragraph.startswith('|'):
            cells = [c.strip() for c in paragraph.strip().strip('|').split('|')]
            first = cells[0].replace('**', '')
            if len(cells) < 2 or not re.fullmatch(r'`[^`]+`(\s*(,|and|or)\s*`[^`]+`)*(\s+(key|keys|in that block))?',
                                                  first):
                return False
            return any(s == term or s.endswith('/' + term) for s in _spans(first))
        for m in re.finditer(_SUBJECT.format(item=r'`[^`]+`'), paragraph):
            if any(s == term or s.endswith('/' + term) or s.startswith(term + ':') for s in _spans(m[1])):
                return True
        return False
    if item['kind'] == 'concept':
        name = r'[-–‐]'.join(map(re.escape, re.split(r'[–-]', term)))
        # a concept that is also a class name ("A `MeshBlock` is the unit ...") is written in code font
        return bool(re.search(_SUBJECT.format(item=r'(?:%s|`%s`)' % (name, name)), paragraph, re.I))
    return False


def definitions(report, data=None):
    """{(kind, term): [sections that define it, in reading order]} for acronyms, keys and named concepts. A
    section starting with `?` is a `##` heading without a label."""
    if data is None:
        data = json.loads((report / REGISTRY).read_text(encoding='utf-8'))
    order, paras = paragraphs(report)
    out = {}
    for item in data['entries']:
        if item['kind'] != 'symbol':
            out[(item['kind'], item['term'])] = [s for s in order if any(defines(item, p) for p in paras.get(s, []))]
    return out


def concept_candidates(report):
    """Named concepts: an eponymous name followed by a method word ("Riemann solver", "Courant number"), used
    mid-sentence at least twice in the prose, whose name is not also an ordinary lower-case word of the book."""
    order, paras = paragraphs(report)
    prose = [p for s in order for p in paras.get(s, []) if not p.startswith(('|', KEY_TABLE))]
    words = {w for p in prose for w in re.findall(r'(?<![\w/.\-–])[a-z]+(?![\w/.\-–])', re.sub(r'`[^`]*`', '', p))}
    rx = re.compile(r'(?<=[a-z,;:(] )(%s)\s+(%s)\b' % (_CONCEPT_NAME, _CONCEPT_WORD))
    seen = {}
    for p in prose:
        for m in rx.finditer(p):
            name = m[1].replace('-', '–')
            if any(part.lower() in words for part in re.split(r'[– ]', name) if part not in ('van', 'de')):
                continue
            key = f'{name} {m[2]}'
            seen[key] = seen.get(key, 0) + 1
    return sorted(k for k, n in seen.items() if n >= 2)


def concept_uses(report, data=None):
    """{('concept', term): [(chapter, section)]}: the prose uses of every named concept in chapters 01 to 16."""
    if data is None:
        data = json.loads((report / REGISTRY).read_text(encoding='utf-8'))
    book_labels = labels(report / 'book')
    order, paras = paragraphs(report)
    out = {}
    for item in data['entries']:
        if item['kind'] != 'concept':
            continue
        rx = re.compile(r'\b%s\b' % r'[-–‐]'.join(map(re.escape, re.split(r'[–-]', item['term']))), re.I)
        for s in order:
            m = re.match(r'chapters/(\d\d)-(?!appendices)', book_labels.get(s, {}).get('file', ''))
            if m and any(rx.search(p) for p in paras.get(s, [])):
                out.setdefault(('concept', item['term']), []).append(('ch' + m[1], s))
    return out


def _identity(item, notation):
    return (item['kind'], notation.symbol_key(item['term']) if item['kind'] == 'symbol' else item['term'])


def rollup(section, book_labels):
    """Whether a section is a roll-up: a chapter or appendix heading rather than a section of its own."""
    return section in book_labels and book_labels[section]['level'] == 1


def _chapter(section, book_labels):
    return re.match(r'chapters/[^/]+?(?=/|\.qmd)|[^/]+$', book_labels[section]['file'])[0] if section in book_labels \
        else None


def _intro_defined(section, found, book_labels):
    """A section of its own in a chapter whose introduction (its roll-up heading) defines the concept."""
    return any(rollup(s, book_labels) and _chapter(s, book_labels) == _chapter(section, book_labels) for s in found)


def _all_uses(report, data):
    notation, _, _ = _tools()
    _, uses = prose_uses(report)
    uses.update(concept_uses(report, data))
    return uses, notation


def _title_rank(item, title):
    """0 when the title names the term, 1 when it holds its expansion (less at most one leading word), else 2."""
    if re.search(r'(?<![\w-])%s(?![A-Za-z_-])' % re.escape(item['term']), title):
        return 0
    words = _norm(item.get('meaning', '').split(',')[0]).split()
    if len(words) >= 2 and any(' '.join(words[i:]) in _norm(title) for i in (0, 1) if len(words) - i >= 2):
        return 1
    return 2


def seed(report):
    """The seeded section and mark of every entry: (item, section, not_defined_here, rule)."""
    data = json.loads((report / REGISTRY).read_text(encoding='utf-8'))
    book_labels = labels(report / 'book')
    uses, notation = _all_uses(report, data)
    defs = definitions(report, data)
    order, paras = paragraphs(report)
    out = []
    for item in data['entries']:
        if item['kind'] == 'symbol':
            out.append((item, f"sec-appa-{item['notation_group']}", False, 'Appendix A group'))
            continue
        found = defs[(item['kind'], item['term'])]
        used = [s for _, s in uses.get(_identity(item, notation), []) if s]
        own_used = [s for s in used if s in book_labels and not rollup(s, book_labels)]
        own = [s for s in found if s in book_labels and not rollup(s, book_labels)]
        if own:
            out.append((item, own[0], False, 'defined in its section'))
        elif found and any(_intro_defined(s, found, book_labels) for s in own_used):
            mine = [s for s in own_used if _intro_defined(s, found, book_labels)]
            out.append((item, min(mine, key=lambda s: _title_rank(item, book_labels[s]['title'])), False,
                        'defined in its chapter introduction; its own section'))
        elif found:
            out.append((item, (own_used or used or ['sec-apph'])[0], False, 'defined under an unlabelled heading'))
        elif own_used or used:
            out.append((item, (own_used or used)[0], True, 'first prose use'))
        else:
            words = ' '.join(_norm(item.get('meaning', '')).split()[:2])
            near = [s for s in order if s in book_labels and not rollup(s, book_labels) and words
                    and any(words in _norm(p) for p in paras.get(s, []))]
            out.append((item, (near or ['sec-apph'])[0], True, 'no prose use; the section that names its meaning'))
    return data, out


def registry_problems(report, data=None):
    """Every entry has a `section` that resolves and is not a chapter or appendix heading. A symbol's is its
    Appendix A group. An acronym, key or named concept not marked "not defined here" is defined in its section
    (or in its chapter's introduction, its section being one of that chapter that uses it); a marked one is used
    in its section's prose (when any chapter uses it), and the book defines it nowhere."""
    if data is None:
        data = json.loads((report / REGISTRY).read_text(encoding='utf-8'))
    book_labels = labels(report / 'book')
    uses, notation = _all_uses(report, data)
    defs = definitions(report, data)
    problems = []
    for item in data['entries']:
        term, section, name = item['term'], item.get('section'), f'{item["kind"]} {item["term"]}'
        if not isinstance(section, str) or not section:
            problems.append(f'{name}: no defining section')
        elif section not in book_labels:
            problems.append(f'{name}: defining section {section} is not a label of the book')
        elif rollup(section, book_labels):
            problems.append(f'{name}: section {section} is a chapter or appendix heading, not a section of its own')
        elif item['kind'] == 'symbol':
            if section != f"sec-appa-{item['notation_group']}":
                problems.append(f'symbol {term}: defined by Appendix A group {item["notation_group"]}, not {section}')
        elif not isinstance(item.get(MARK, False), bool):
            problems.append(f'{name}: {MARK} must be true or false')
        elif not item.get(MARK):
            found = defs[(item['kind'], term)]
            used = {s for _, s in uses.get(_identity(item, notation), [])}
            if section not in found and not (section in used and _intro_defined(section, found, book_labels)):
                problems.append(f'{name}: section {section} does not define it; give the section that does, or '
                                f'mark it "{MARK}": true')
        else:
            found = defs[(item['kind'], term)]
            if found:
                problems.append(f'{name}: marked "not defined here", but {found[0]} defines it; unmark it and '
                                'record the verified fall')
            used = {s for _, s in uses.get(_identity(item, notation), [])}
            if used and section not in used:
                problems.append(f'{name}: not defined here, and section {section} does not use it in prose')
    return problems


def marked(data):
    return sum(1 for item in data['entries'] if item.get(MARK))


def ratchet_problems(report, data=None, baseline=None):
    """The count of entries marked "not defined here" against its baseline: no rise, and a fall is recorded."""
    path = Path(baseline) if baseline else report / BASELINE
    if not path.exists():
        return [f'{BASELINE} is missing; write it with python3 tools/glossary_registry.py --update']
    if data is None:
        data = json.loads((report / REGISTRY).read_text(encoding='utf-8'))
    base, have = json.loads(path.read_text(encoding='utf-8'))[MARK], marked(data)
    if have > base:
        return [f'{have} entries marked "not defined here", baseline {base}: define the new concepts instead']
    if have < base:
        return [f'{have} entries marked "not defined here", baseline {base}; record the verified fall with '
                'python3 tools/glossary_registry.py --update']
    return []


def check(report=REPORT):
    try:
        problems = registry_problems(Path(report)) + ratchet_problems(Path(report))
    except (OSError, ValueError, KeyError) as exc:
        problems = [str(exc)]
    print(f'[N4 concept registry] {len(problems)} problem(s)')
    for p in problems:
        print('  ' + p)
    return int(bool(problems))


def main(argv):
    report = REPORT
    if '--seed' in argv:
        data = json.loads((report / REGISTRY).read_text(encoding='utf-8'))
        have = {e['term'] for e in data['entries'] if e['kind'] == 'concept'}
        data['entries'] += [dict(kind='concept', term=t) for t in concept_candidates(report) if t not in have]
        (report / REGISTRY).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        data, seeded = seed(report)
        rules = {}
        for item, section, mark, rule in seeded:
            item['section'] = section
            if item['kind'] == 'symbol':
                item.pop(MARK, None)
            else:
                item[MARK] = mark
            rules[rule] = rules.get(rule, 0) + 1
        (report / REGISTRY).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'seeded {len(seeded)} entries: {rules}; {marked(data)} marked "not defined here"')
        return 0
    data = json.loads((report / REGISTRY).read_text(encoding='utf-8'))
    if '--update' in argv:
        (report / BASELINE).write_text(json.dumps({MARK: marked(data)}, indent=1) + '\n', encoding='utf-8')
        print(f'baseline written: {marked(data)} entries marked "not defined here"')
        return 0
    if '--count' in argv:
        book_labels = labels(report / 'book')
        print(f'{marked(data)} of {len(data["entries"])} entries marked "not defined here"')
        for item in data['entries']:
            if rollup(item.get('section'), book_labels):
                print(f'  roll-up: {item["kind"]} {item["term"]} -> {item["section"]}')
        return 0
    return check(report)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
