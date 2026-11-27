"""N6: extract the species and condensate interface from the code pin and kintera; verify both complete blocks."""
import json
import re
import subprocess
import sys
from pathlib import Path

import code_anchors

REPORT = Path(__file__).resolve().parent.parent
TARGETS = {
    'species': 'book/chapters/02-thermodynamics/_howto-species.qmd',
    'condensate': 'book/chapters/10-moist-coupling/_howto-condensate.qmd',
}
PIN = '73c45a7c8c991209924e1137992352c335575062'
KINTERA = ('chengcli/kintera', '0429d795f79a0665afeca565e5a05c87ab4ba1e2')  # kintera v2.6.13
CHECK_KINTERA = '[`kintera c55b13b2`](https://github.com/chengcli/kintera/commit/c55b13b2204997d2d09e04498558ab9495d8ee77)'  # func_table_check.py's own kintera: hand-written tables


def _reader(repo, sha):
    checkout = code_anchors.find_commit(repo, sha)

    def read(path):
        return subprocess.run(['git', '-C', str(checkout), 'show', f'{sha}:{path}'],
                              check=True, capture_output=True, text=True).stdout

    def excerpt(path, pattern, flat=False):
        text = read(path)
        matches = list(re.finditer(pattern, text, re.M | re.S))
        if len(matches) != 1:
            raise ValueError(f'{repo} {path}: expected one match, got {len(matches)}')
        match = matches[0]
        first = text[:match.start()].count('\n') + 1
        last = first + match[0].rstrip('\n').count('\n')
        lines = f'L{first}' if last == first else f'L{first}-L{last}'
        label = f'{path}:{first}' if last == first else f'{path}:{first}-{last}'
        if repo != 'chengcli/snapy':
            label = f'{repo.split("/")[1]} {label}'
        return (' '.join(match[0].split()) if flat else match[0].strip(), f'[`{label}`](https://github.com/{repo}/blob/{sha}/{path}#{lines})')

    return excerpt


def _scalar(text):
    text = text.strip()
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def _entry(block):
    """One flat deck entry (`- key: value` lines, values scalar or a one-line `{k: v}` map), as a Python dict
    literal; the deck's own lines are cited next to it."""
    entry = {}
    for line in block.splitlines():
        line = line.split('#')[0].strip().removeprefix('- ')
        if not line:
            continue
        key, value = (part.strip() for part in line.split(':', 1))
        if value.startswith('{'):
            pairs = (item.split(':', 1) for item in value.strip('{}').split(','))
            entry[key] = {k.strip(): _scalar(v) for k, v in pairs}
        else:
            entry[key] = _scalar(value)
    lines = [f'    {json.dumps(k)}: {json.dumps(v)},' for k, v in entry.items()]
    return '{\n' + '\n'.join(lines) + '\n}'


SETUP = '''```python
from pathlib import Path
import torch
import snapy
import kintera
import yaml

# The chemistry of the repository deck, on a tiny column with no native source.
deck = yaml.safe_load(Path("examples/bryan.yaml").read_text())
deck["geometry"] = {
    "type": "cartesian",
    "cells": {"nx1": 4, "nx2": 1, "nx3": 1, "nghost": 3},
    "bounds": {"x1min": 0., "x1max": 1.e3, "x2min": 0.,
               "x2max": 1., "x3min": 0., "x3max": 1.},
}
deck["boundary-condition"] = {"external": {
    "x1-inner": "reflecting", "x1-outer": "reflecting"}}
deck["integration"] = {"type": "rk1", "cfl": 0.5}
for key in ("outputs", "forcing", "problem", "reactions"):
    deck.pop(key, None)
dry = deck["species"][0]

def build(species, reactions, name):
    deck["species"] = species
    if reactions:
        deck["reactions"] = reactions
    Path(name).write_text(yaml.safe_dump(deck))
    block = snapy.MeshBlock(snapy.MeshBlockOptions.from_yaml(name))
    return block, block.module("hydro.eos").nvar()

def one_stage(block, nvar, vapour, pressure):
    coord = block.module("coord")
    shape = (nvar, coord.buffer("x3v").numel(),
             coord.buffer("x2v").numel(), coord.buffer("x1v").numel())
    w = torch.zeros(shape, dtype=torch.float64)
    w[snapy.kIDN].fill_(1.)
    w[snapy.kIPR].fill_(pressure)
    w[snapy.kICY].fill_(vapour)
    variables, _ = block.initialize({"hydro_w": w})
    before = variables["hydro_u"].clone()
    block.advance_local(variables, 0.01, 0)
    return before[..., 3:-3], variables["hydro_u"][..., 3:-3]
```'''


def generate(report, keys=None):
    # only the requested targets (default: both); a legacy chapter's target is not built (check(), --write)
    keys = set(TARGETS) if keys is None else set(keys)
    repo, sha = code_anchors.read_pin(report / 'code_pin.yml')
    if sha != PIN:
        raise ValueError('N6 needs review when the code pin changes')
    snap = _reader(repo, sha)
    kin = _reader(*KINTERA)
    krepo, ksha = KINTERA

    vapour, vapour_cite = snap('examples/bryan.yaml', r'^  - name: H2O\n(?:    [^\n]+\n)+')
    cloud, cloud_cite = snap('examples/bryan.yaml', r'^  - name: H2O\(l\)\n(?:    [^\n]+\n)+')
    reaction, reaction_cite = snap('examples/bryan.yaml', r'^reactions:\n(?:  [^\n]+\n)+')
    load, load_cite = snap('src/eos/equation_of_state.cpp',
        r'  op->thermo\(\) = kintera::ThermoOptionsImpl::from_yaml\(filename, op->verbose\(\)\);')
    nvar, nvar_cite = snap('src/eos/moist_base.hpp',
        r'  int nvar\(\) const override \{ return 4 \+ nspecies\(\) \+ 1; \}')
    icy, icy_cite = snap('python/csrc/snapy.cpp', r'  m\.attr\("kICY"\) = \(int\)snap::ICY;')
    gate, gate_cite = snap('src/mesh/meshblock_advance.cpp',
        r'  if \(stage == pintg->stages\.size\(\) - 1 && phydro->options->eos\(\)->thermo\(\) &&\n'
        r'      phydro->options->eos\(\)->thermo\(\)->reactions\(\)\.size\(\) > 0\)')
    writeback, writeback_cite = snap('src/mesh/meshblock_advance.cpp', r'    hydro_u\.narrow\(0, ICY, ny\) = yfrac \* rho;')
    cause, cause_cite = snap('src/mesh/meshblock_redo.cpp',
        r'"the saturation adjustment left a cell unadjusted, or the VIC "')

    phase, phase_cite = kin('src/species.cpp',
        r'    TORCH_CHECK\(phase\.empty\(\) \|\| phase == "gas" \|\| phase == "liquid" \|\|\n[^;]+;')
    cv, cv_cite = kin('src/species.cpp', r'      species_cref_R\.push_back\(5\. / 2\.\);')
    u0, u0_cite = kin('src/species.cpp', r'      species_uref_R\.push_back\(0\.\);')
    clear, clear_cite = kin('src/species.cpp', r'  clear_species_registry\(\);\n\n  for \(const auto& sp : config\["species"\]\) \{')
    ref, ref_cite = kin('src/thermo/thermo_options.cpp', r'  vapor_set\.insert\(species_names\[0\]\);')
    inert, inert_cite = kin('src/thermo/thermo_options.cpp',
        r'    if \(species_phases\[i\] == "gas"\) \{\n.*?      vapor_set\.insert\(name\);')
    if 'condensate' in keys:
        nucleation, nucleation_cite = kin('src/thermo/thermo_options.cpp',
            r'    thermo->nucleation\(\) =\n        NucleationOptionsImpl::from_yaml\(config\["reactions"\]\);\n'
            r'    add_to_vapor_cloud\(vapor_set, cloud_set, thermo->nucleation\(\)\);')
        product, product_cite = kin('src/thermo/nucleation.cpp',
            r'      if \(species_phases\[id\] == "gas"\) \{\n        vapor_set\.insert\(name\);\n'
            r'        op->has_gas_products = true;\n      \} else \{\n        cloud_set\.insert\(name\);\n      \}')
        reactant, reactant_cite = kin('src/thermo/nucleation.cpp',
            r'      TORCH_CHECK\(species_phases\[id\]\.empty\(\) \|\| species_phases\[id\] == "gas",\n[^;]+;')
        signature, signature_cite = kin('src/utils/user_funcs.hpp', r'^using user_func1 = double \(\*\)\(double\);')
        pair, pair_cite = kin('src/vapors/vapor_functions.h',
            r'^DISPATCH_MACRO\ninline double h2o_bryan\(double T\) \{.*?\n\}\n\nDISPATCH_MACRO\n'
            r'inline double h2o_bryan_ddT\(double T\) \{.*?\n\}')
        # the registry is spelled once (src/func_list.h); the host and device tables expand it
        flist, flist_cite = kin('src/func_list.h', r'^#define KINTERA_FUNC1_LIST\(X\) \\\n(?:  X\([A-Za-z0-9_]+\)[ \\]*\n)+')
        xmac, xmac_cite = kin('src/func_list.h', r'^#define KINTERA_FUNC_PTR\(f\) f,\n#define KINTERA_FUNC_NAME\(f\) #f,')
        cpu_table, cpu_table_cite = kin('src/func_table.cpp',
            r'^user_func1 func1_table_cpu\[\] = \{KINTERA_FUNC1_LIST\(KINTERA_FUNC_PTR\)\};')
        cpu_names, cpu_names_cite = kin('src/func_table.cpp',
            r'^std::vector<std::string> func1_names = \{KINTERA_FUNC1_LIST\(KINTERA_FUNC_NAME\)\};')
        cuda_table, cuda_table_cite = kin('src/func_table.cu',
            r'^__device__ user_func1 func1_table_cuda\[\] = \{\n    nullptr, KINTERA_FUNC1_LIST\(KINTERA_FUNC_PTR\)\};')
        host_lookup, host_lookup_cite = kin('src/utils/user_funcs.hpp', r'      funcs\.push_back\(func_table\[id\]\);')
        device_lookup, device_lookup_cite = kin('src/utils/user_funcs.cu',
            r'      h_idx\.push_back\(std::distance\(func_names\.begin\(\), it\) \+ 1\);')
        unknown, unknown_cite = kin('src/utils/user_funcs.hpp',
            r'      throw std::runtime_error\("Function " \+ name \+ " not registered\."\);')
        ddt, ddt_cite = kin('src/thermo/log_svp.cpp', r'  for \(auto& name : names_ddT\) name \+= "_ddT";')
        rebuild, rebuild_cite = kin('README.md',
            r'^# 3\. Configure and build the C\+\+ library\ncmake -B build\ncmake --build build --parallel\n.*?pip install \.')
        formulas, formulas_cite = flist, flist_cite

    KINTERA_REF = f'snapy@{sha} and kintera@{ksha}'
    vapour_py, cloud_py, reaction_py = _entry(vapour), _entry(cloud), _entry(reaction.split('\n', 1)[1])

    species = f'''<!-- generated reader-reference: species-contract -->
<!-- Generated by tools/gen_species_contract.py; do not edit. -->
#### How to add one

**A minimal input.** Follow steps 0–5 in one Python file, run from the repository root. The example adds a
gas that no reaction names, and checks that it becomes a state row only when it declares its phase.
Adding a condensate, which a reaction names, is @sec-ch08-satadj.

0. Prepare a tiny column with the chemistry of `examples/bryan.yaml`, and two helpers: one builds a block from a
   species list, one runs one stage on a uniform state.

{SETUP}

1. Write the species entry. Its values are those of the deck (`H2O` in {vapour_cite}).

```python
vapour = {vapour_py}
```

2. Select its phase. No reaction names this species here, so only `phase: gas` makes it a state row.

```python
listed = dict(vapour)
vapour["phase"] = "gas"
```

3. Build three blocks: dry air alone, the species without a phase, and the species with it.

```python
_, n_dry = build([dry], None, "tiny_dry.yaml")
_, n_listed = build([dry, listed], None, "tiny_listed.yaml")
block, n_gas = build([dry, vapour], None, "tiny_gas.yaml")
```

4. Run one stage with the new row filled.

```python
before, after = one_stage(block, n_gas, 0.01, 1.e5)
```

5. Check the effect: one more state row, at `kICY`, carried unchanged by a stage at rest.

```python
assert (n_dry, n_listed, n_gas) == (5, 5, 6)
assert kintera.species_names() == ["dry", "H2O"]
torch.testing.assert_close(after[snapy.kICY], before[snapy.kICY],
                           rtol=0, atol=0)
print("N6 species: H2O with phase gas adds state row", snapy.kICY)
```

- **Contract.** Reading: the EOS builds its species table from the same YAML card (`from_yaml` in {load_cite}):

  ```cpp
  {load}
  ```

- Keys: `name`, `composition` (the molar mass comes from it), `cv_R`, `u0_R` and `phase`. A missing `cv_R`
  defaults to 5/2 (`species_cref_R` in {cv_cite}); a missing `u0_R` to 0 (`species_uref_R` in {u0_cite}).
- Phase: empty, `gas`, `liquid` or `solid`; anything else is refused ({phase_cite}).
- Reference species: the first entry is always a state species (`vapor_set` in {ref_cite}); the dry air goes first.
- State rows: a species is a row when a reaction names it or when it declares a phase (`vapor_set` in {inert_cite});
  a listed species with neither is in kintera's table but not in the state.
- Row count: the five hydro rows plus one per species after the first (`nvar` in {nvar_cite}):

  ```cpp
  {nvar}
  ```

  The first species row is `kICY` ({icy_cite}).
- One table per process: each card clears kintera's species table and loads its own (`clear_species_registry` in {clear_cite});
  `kintera.species_names()` lists the card loaded last, as step 5 checks.

CI checks the contract, and the runtime test runs on a pin build.

**No C++ needed.** A species is data in the card. Its thermodynamics follow from `cv_R`, `u0_R` and the
composition; a phase change is added in the reaction list (@sec-ch08-satadj).
<!-- end reader-reference: species-contract -->
'''

    condensate = None
    if 'condensate' in keys:
        condensate = f'''<!-- generated reader-reference: condensate-contract -->
<!-- Generated by tools/gen_species_contract.py; do not edit. -->
#### How to add one

**A minimal input.** Follow steps 0–5 in one Python file, run from the repository root. The example adds a
condensate and the reaction that names it, and checks that one step's saturation adjustment moves water from
the vapour to the condensate. Adding a gas on its own is @sec-ch02-ideal-moist; adding a passive tracer, which
has no thermodynamics or phase change, is a name in the `scalar` block (@sec-ch01-scalar, @sec-ch08-drycarry).

0. Prepare a tiny column with the chemistry of `examples/bryan.yaml`, and two helpers: one builds a block from a
   species list and a reaction list, one runs one stage on a uniform state.

{SETUP}

1. Write the vapour and the condensate entries. Their values are those of the deck at the pin
   ({vapour_cite}, {cloud_cite}).

```python
vapour = {vapour_py}
cloud = {cloud_py}
```

2. Name it in a reaction, with a saturation formula of kintera ({reaction_cite}).

```python
reactions = [{reaction_py}]
```

3. Build the block.

```python
block, nvar = build([dry, vapour, cloud], reactions, "tiny_cloud.yaml")
```

4. Run one stage on a cold, supersaturated state.

```python
before, after = one_stage(block, nvar, 0.02, 8.e4)
```

5. Check the effect: water moves from the vapour row to the condensate row, and total water and energy are kept.

```python
i = snapy.kICY
assert nvar == 7
assert (after[i] < before[i]).all() and (after[i + 1] > 0).all()
torch.testing.assert_close(after[i:i + 2].sum(0), before[i:i + 2].sum(0),
                           rtol=1.e-12, atol=0)
torch.testing.assert_close(after[snapy.kIPR], before[snapy.kIPR],
                           rtol=1.e-12, atol=0)
condensed = float(after[i + 1].mean())
print("N6 condensate: condensed", condensed, "kg m^-3 in one stage")
```

- **Contract.** Classification: the reaction list is read as nucleation reactions, which sort their species
  into vapours and condensates ({nucleation_cite}). A reactant must be a gas ({reactant_cite}).
- Entry: the condensate is a species entry of its own. A product is a condensate unless it declares
  `phase: gas` ({product_cite}), so leave the phase empty or give `liquid` or `solid`.
- Formula: `rate-constant: {{formula: ...}}` names one of kintera's saturation functions, the entries of
  `KINTERA_FUNC1_LIST` ({formulas_cite}).
- When: once per step, on the last stage, whenever the reaction list is non-empty ({gate_cite}):

  ```cpp
  {gate}
  ```

- What it writes: only the species rows, `{writeback}` ({writeback_cite}); the energy row is not written,
  as step 5 checks.
- Failure: a cell the solve leaves unadjusted makes the block redo the step with a smaller `dt`
  ({cause_cite}).

CI checks the contract, and the runtime test runs on a pin build.

**Native C++ alternative.** A new saturation formula is a pair of C++ functions in kintera: the logarithm of the
saturation vapour pressure and its temperature derivative, each with this generated signature ({signature_cite}):

```cpp
{signature}
```

Write the pair after this generated one ({pair_cite}):

```cpp
{pair}
```

Add both names to `KINTERA_FUNC1_LIST`, the one list that every table expands, in order, next to the pair they
belong with ({flist_cite}):

```c
{flist}
```

The CPU pointer table and the name vector expand that list, one with `KINTERA_FUNC_PTR` and one with
`KINTERA_FUNC_NAME` ({xmac_cite}; {cpu_table_cite}, {cpu_names_cite}), and the CUDA table expands it after a
leading `nullptr`, one position further on ({cuda_table_cite}); none of the three is edited:

```cpp
{xmac}
{cpu_table}
{cpu_names}
{cuda_table}
```

The derivative is found by name, the formula name with `_ddT` appended (`{ddt}`, {ddt_cite}), so both names are
needed. The lookup is by position: the CPU build takes `{host_lookup}` ({host_lookup_cite}) and the CUDA build
`{device_lookup}` ({device_lookup_cite}); a name not in the list throws ({unknown_cite}):

```cpp
{unknown}
```

Because the three expand one list, the name vector and both pointer tables hold the same entries in the same order
(the CUDA table one position on, after its `nullptr`),
and a name in the list without its function pair expands to an undefined identifier in the pointer tables, so
they do not compile. The footnote's check shows what the one list rules out: on tables written out by hand, two names added to the name vector
without table entries raised no error, the first resolved to the function after `h2o_bryan_ddT`, every later name
to the one two positions on, and the last read past the end of the table and crashed.[^ch10-func-table]
Then rebuild and reinstall kintera, so that snapy links the new table ({rebuild_cite}):

```bash
{rebuild}
```

[^ch10-func-table]: [sha: snapy@{sha}; the check builds kintera's files at {CHECK_KINTERA}, where the tables were
    written out by hand; deck: the check itself; run: `python3 src/snapy_report/ch10/checks/func_table_check.py`,
    CPU, `c++` (GCC 11.5.0), 1 process; Python 3.11.13]. Output `func_table_check.out`, lines `[C1]`–`[C5]`; the
    crash of `[C5]` is undefined behaviour, observed as exit status -11 on this build.
<!-- end reader-reference: condensate-contract -->
'''
    out = {'species': species, 'condensate': condensate}
    return {k: v for k, v in out.items() if k in keys}


def check(report=REPORT):
    # a target in a legacy chapter (tools/code-anchors-legacy.txt) is skipped, and checked once its line is deleted
    live = {k: t for k, t in TARGETS.items() if not code_anchors.legacy_chapter_of(t, Path(report))}
    if not live:
        print('[N6 species contract] skipped: every target is in a chapter in tools/code-anchors-legacy.txt')
        return 0
    try:
        expected = generate(Path(report), live)
        errors = []
        for key, target in live.items():
            if (Path(report) / target).read_text() != expected[key]:
                errors.append(f'{target}: generated keys, rows, reaction or runtime example differs from the pin')
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        errors = [str(exc)]
    print(f'[N6 species contract] {len(errors)} problem(s)')
    for error in errors:
        print('  ' + error)
    return int(bool(errors))


if __name__ == '__main__':
    if '--write' in sys.argv:
        live = [k for k, t in TARGETS.items() if not code_anchors.legacy_chapter_of(t, REPORT)]
        for key, text in generate(REPORT, live).items():
            (REPORT / TARGETS[key]).write_text(text)
    else:
        sys.exit(check())
