"""N6: the native C++ route for a new reconstruction or Riemann solver, extracted from the code pin.

It completes the scheme contract of tools/gen_scheme_contract.py (the Python recipe, the interface, the faces, the
stencil and ghost width): this block gives how a scheme is selected from YAML and looked up, which tables,
face function and dispatch case to add, the shapes and in-place outputs, the dtype, device and ownership rules, and the rebuild. Every fact is an
excerpt that must match the pin exactly once, cited with a line permalink.
"""
import re
import subprocess
import sys
from pathlib import Path

import code_anchors

REPORT = Path(__file__).resolve().parent.parent
TARGET = 'book/chapters/04-spatial-discretization/_howto-native.qmd'
PIN = '73c45a7c8c991209924e1137992352c335575062'


def generate(report):
    repo, sha = code_anchors.read_pin(report / 'code_pin.yml')
    if sha != PIN:
        raise ValueError('N6 needs review when the code pin changes')
    checkout = code_anchors.find_commit(repo, sha)

    def read(path):
        return subprocess.run(['git', '-C', str(checkout), 'show', f'{sha}:{path}'],
                              check=True, capture_output=True, text=True).stdout

    def excerpt(path, pattern):
        text = read(path)
        matches = list(re.finditer(pattern, text, re.M | re.S))
        if len(matches) != 1:
            raise ValueError(f'{path}: expected one match, got {len(matches)}')
        match = matches[0]
        first = text[:match.start()].count('\n') + 1
        last = first + match[0].rstrip('\n').count('\n')
        lines = f'L{first}' if last == first else f'L{first}-L{last}'
        label = f'{path}:{first}' if last == first else f'{path}:{first}-{last}'
        return (match[0].rstrip(), f'[`{label}`](https://github.com/{repo}/blob/{sha}/{path}#{lines})')

    # reconstruction: selection, lookup, the scheme tables, the face function, the dispatch, files
    sections, sections_cite = excerpt('src/hydro/hydro_options.cpp',
        r'^  op->recon1\(\) = ReconstructOptionsImpl::from_yaml\(filename, "vertical"\);\n.*?'
        r'  op->recon23\(\) = ReconstructOptionsImpl::from_yaml\(filename, "horizontal"\);')
    recon_yaml, recon_yaml_cite = excerpt('src/recon/reconstruct.cpp',
        r'^  check_keys\(node, path, keys\);\n.*?'
        r'  op->interp\(\)->scale\(\) = node\["scale"\]\.as<bool>\(op->interp\(\)->scale\(\)\);')
    recon_default, recon_default_cite = excerpt('src/recon/reconstruct.hpp',
        r'^  ADD_ARG\(std::string, type\) = "dc";')
    scheme_enum, scheme_enum_cite = excerpt('src/recon/recon_point.h',
        r'^enum class Scheme : int \{ dc, plm, ppm, cp3, weno3, cp5, weno5 \};')
    parse, parse_cite = excerpt('src/recon/reconstruct.cpp',
        r'^Scheme parse_scheme\(std::string const& type\) \{\n.*?\n\}')
    interp_unknown, interp_unknown_cite = excerpt('src/recon/reconstruct.cpp',
        r'^  TORCH_CHECK\(false, "Interp: unknown type " \+ type\);')
    partner, partner_cite = excerpt('src/recon/reconstruct.cpp',
        r'^Scheme velocity_pressure_scheme\(Scheme s\) \{\n.*?\n\}')
    names, names_cite = excerpt('src/recon/reconstruct.cpp',
        r'^  pinterp1 = InterpImpl::create\(options->interp\(\), s, this, "interp1"\);\n'
        r'  pinterp2 = InterpImpl::create\(options->interp\(\), velocity_pressure_scheme\(s\),\n'
        r'                                this, "interp2"\);')
    traits, traits_cite = excerpt('src/recon/recon_point.h',
        r'^template <Scheme S>\nstruct SchemeTraits \{\n.*?\n\};')
    table, table_cite = excerpt('src/recon/reconstruct.cpp', r'^    case Scheme::cp3:\n.*?      break;')
    point, point_cite = excerpt('src/recon/recon_point.h',
        r'^template <Scheme S, typename T, bool Nu = false>\nDISPATCH_MACRO inline T recon_point\(.*?\n\}')
    cpu_case, cpu_case_cite = excerpt('src/recon/recon_dispatch.cpp',
        r'^    case Scheme::cp3:\n      return recon_line_cpu<Scheme::cp3, false>\(iter, rows, dim, scale, "cp3"\);')
    cuda_case, cuda_case_cite = excerpt('src/recon/recon_dispatch.cu',
        r'^    case Scheme::cp3:\n      return recon_line_cuda<Scheme::cp3, false>\(iter, rows, dim, false, "cp3"\);')
    interp_reg, interp_reg_cite = excerpt('src/recon/reconstruct.cpp',
        r'^  return p->register_module\(name, std::make_shared<InterpImpl>\(opts, s\)\);')
    alloc, alloc_cite = excerpt('src/recon/reconstruct.cpp',
        r'^  auto result = torch::zeros\(vec, w\.options\(\)\);')
    glob, glob_cite = excerpt('src/CMakeLists.txt', r'^    riemann/\*\.cpp\n    recon/\*\.cpp')
    cuglob, cuglob_cite = excerpt('src/CMakeLists.txt', r'^      recon/\*\.cu\n      riemann/\*\.cu')

    # Riemann solver: selection, defaults, lookup, the kind, the face function, the dispatch, call site
    rs_keys, rs_keys_cite = excerpt('src/riemann/riemann.cpp', r'^  check_keys\(node, path, \{"type", "dir"\}\);')
    rs_yaml, rs_yaml_cite = excerpt('src/riemann/riemann.cpp',
        r'^  op->type\(\) = node\["type"\]\.as<std::string>\(op->type\(\)\);\n'
        r'  op->dir\(\) = node\["dir"\]\.as<std::string>\(op->dir\(\)\);')
    rs_none, rs_none_cite = excerpt('src/riemann/riemann.cpp',
        r'^  if \(!config\[section\]\["riemann-solver"\]\) return op;')
    rs_default, rs_default_cite = excerpt('src/riemann/riemann.hpp',
        r'^  ADD_ARG\(std::string, type\) = "hllc";')
    kind_enum, kind_enum_cite = excerpt('src/riemann/riemann.hpp',
        r'^enum class RiemannKind : int \{\n.*?\n\};')
    rs_parse, rs_parse_cite = excerpt('src/riemann/riemann.cpp',
        r'^RiemannKind parse_riemann_kind\(std::string const& type\) \{\n.*?\n\}')
    rs_unknown, rs_unknown_cite = excerpt('src/riemann/riemann.cpp',
        r'^  TORCH_CHECK\(false, "RiemannSolver: unknown type " \+ type\);')
    face, face_cite = excerpt('src/riemann/riemann_kernels.h',
        r'^  \} else if constexpr \(K == RiemannKind::lmars\) \{\n    lmars_impl\(.*?\);')
    thermo, thermo_cite = excerpt('src/riemann/riemann.cpp',
        r'^  \} else if \(kind == RiemannKind::lmars\) \{\n.*?    e = elr, gamma = glr;')
    rs_cpu, rs_cpu_cite = excerpt('src/riemann/riemann_dispatch.cpp',
        r'^    case RiemannKind::lmars:\n      SNAP_RIEMANN_CPU\(RiemannKind::lmars\);')
    rs_cuda, rs_cuda_cite = excerpt('src/riemann/riemann_dispatch.cu',
        r'^    case RiemannKind::lmars:\n      SNAP_RIEMANN_CUDA\(RiemannKind::lmars, 6\);')
    register, register_cite = excerpt('src/riemann/riemann.cpp',
        r'^  return p->register_module\(name, std::make_shared<RiemannSolverImpl>\(opts, p\)\);')
    capture, capture_cite = excerpt('src/mesh/stage_executor.cpp',
        r'^  static std::set<std::string> const capture_riemann = \{"lmars", "hllc"\};')
    inplace, inplace_cite = excerpt('src/riemann/riemann.cpp',
        r'^  at::TensorIteratorConfig config;\n  config\.resize_outputs\(false\)\n.*?      \.add_output\(flx\);')
    device, device_cite = excerpt('src/riemann/riemann.cpp',
        r'^  at::native::call_riemann\(flx\.device\(\)\.type\(\), iter, static_cast<int>\(kind\),')
    parent, parent_cite = excerpt('src/riemann/riemann.hpp',
        r'^  //! non-owning reference to parent\n  HydroImpl const\* phydro = nullptr;')
    rebuild, rebuild_cite = excerpt('README.md',
        r'^cmake -B build -DCMAKE_BUILD_TYPE=Release -DNETCDF=ON\ncmake --build build --parallel 3')
    install, install_cite = excerpt('README.md', r'^5\. Install the Python package:\n```bash\npip install \.')
    install = install.splitlines()[-1]

    # face slots (left/right) and how the solver consumes them
    slots, slots_cite = excerpt('src/snap.h',
        r'^  ILT = 0,  //!< Left interface\n  IRT = 1,  //!< Right interface(?![\s\S]*ILT = 0,)')
    apply, apply_cite = excerpt('src/recon/reconstruct.cpp',
        r'^  auto outl = wlr\[IRT\]\.slice\(dim, il - 1, iu \+ 1\);\n'
        r'  auto outr = wlr\[ILT\]\.slice\(dim, il, iu \+ 2\);\n\n  pinterp->forward\(w, dim, outl, outr\);')
    call, call_cite = excerpt('src/hydro/sweep.cpp',
        r'^    hydro\.priemann->forward\(lr\[ILT\], lr\[IRT\], dim, flux, face_pressure,\n'
        r'                            central\);')
    div, div_cite = excerpt('src/coord/coordinate.cpp',
        r'^  return area1\.slice\(-1, si \+ 1, ei \+ 1\) \* flux1\.slice\(-1, si \+ 1, ei \+ 1\) -\n'
        r'         area1\.slice\(-1, si, ei\) \* flux1\.slice\(-1, si, ei\);')
    # Riemann shapes and face pressure
    fluxbuf, fluxbuf_cite = excerpt('src/hydro/hydro.cpp', r'^  _flux1 = register_buffer\("F1", freg\.totals\(\)\[0\]\);')
    buffers, buffers_cite = excerpt('src/hydro/hydro.cpp',
        r'^  _face_pressure1 = register_buffer\(\n'
        r'      "P1", nc1 > 1 \? torch::zeros\(\{nc3, nc2, nc1\}, torch::kFloat64\)\n'
        r'                    : torch::Tensor\(\)\);')
    fp_out, fp_out_cite = excerpt('src/riemann/riemann.cpp',
        r'^    face_pressure_out = direct \? face_pressure : torch::empty_like\(wl\[IDN\]\);')
    fp_pass, fp_pass_cite = excerpt('src/hydro/hydro_forward.cpp',
        r'^      auto face_pressure1 = options->eos\(\)->type\(\) == "shallow-water"\n'
        r'                                \? torch::Tensor\(\)\n                                : _face_pressure1;')
    call1, call1_cite = excerpt('src/hydro/hydro_forward.cpp',
        r'^      riemann_solve\(\*this, wlr1, DIM1, freg\.slot\(1, "riemann"\), face_pressure1,\n'
        r'                    pressure_base1, split\);')
    call23, call23_cite = excerpt('src/hydro/sweep.cpp',
        r'^  riemann_solve\(\n      hydro, wlr, dir\.dim, reg\.slot\(d, "riemann"\), torch::Tensor\(\), background,\n'
        r'      perturb, central \? reg\.part_slot\(d, "riemann\.central"\) : torch::Tensor\(\)\);')
    refuse, refuse_cite = excerpt('src/riemann/riemann.cpp',
        r'^    TORCH_CHECK\(\n        !face_pressure\.defined\(\),\n'
        r'        "Face-pressure output is not implemented for Riemann solver type ",\n        options->type\(\)\);')
    fp_gnomonic, fp_gnomonic_cite = excerpt('src/coord/gnomonic_equiangle.cpp',
        r'^    if \(face_pressure1\.defined\(\)\) \{')
    fp_spherical, fp_spherical_cite = excerpt('src/coord/spherical_polar.cpp',
        r'^    if \(face_pressure1\.defined\(\) && x1_centroid_exact_enabled\(\)\) \{')
    fp_base, fp_base_cite = excerpt('src/coord/coordinate.cpp', r'^  \(void\)face_pressure1;')

    def indent(text):
        return chr(10).join('  ' + l for l in text.splitlines())

    return f'''<!-- generated reader-reference: scheme-native -->
<!-- Generated by tools/gen_scheme_native.py; do not edit. -->
#### Native C++ route

The recipe above prototypes a reconstruction in Python and selects a built-in one by name; it stays the one
recipe. A new scheme that the solver itself runs is built into the library and selected from the YAML card, and
this route is the one place that says how, for a reconstruction and for a Riemann solver. There is one
reconstruction module and one Riemann module: a new scheme is a new entry in their tables and a new face function,
not a new class. Each step below is a fact of the code at the pin.

**Select it from YAML.** A reconstruction is read from `dynamics/reconstruct/vertical` for $x_1$ and from
`dynamics/reconstruct/horizontal` for $x_2$ and $x_3$, into `recon1` and `recon23` ({sections_cite}). Each section takes `shock`, `type` and
`scale`, and a missing key keeps its default ({recon_yaml_cite}); `type` defaults to `dc` ({recon_default_cite}):

```cpp
{recon_yaml}
{recon_default}
```

A Riemann solver is read from `dynamics/riemann-solver`, whose keys are `type` and `dir` ({rs_keys_cite}). A
missing key keeps the created `type` and `dir` ({rs_yaml_cite}), and so does a card with no `riemann-solver`
section ({rs_none_cite}); the created `type` is `hllc` ({rs_default_cite}):

```cpp
{rs_yaml}
{rs_none}
{rs_default}
```

**Lookup.** A reconstruction name maps to one value of the `Scheme` enum ({scheme_enum_cite}), compared as a
string, branch by branch, in `parse_scheme` ({parse_cite}); a name with no branch is refused by a `TORCH_CHECK`
when the block is built ({interp_unknown_cite}):

```cpp
{scheme_enum}
{parse}
```

The module is created twice, under the names `interp1` (density and species) and `interp2` (velocity and
pressure) ({names_cite}); `interp2` takes its partner scheme from `velocity_pressure_scheme` ({partner_cite}):

```cpp
{names}
{partner}
```

A Riemann name maps to one value of the `RiemannKind` enum ({kind_enum_cite}) in `parse_riemann_kind`
({rs_parse_cite}); a name with no branch is refused by a `TORCH_CHECK` when the block is built
({rs_unknown_cite}):

```cpp
{kind_enum}
{rs_parse}
```

**Add a reconstruction.** Besides the enum value and its `parse_scheme` branch, a stencil scheme needs its width
and coefficient counts in `SchemeTraits` ({traits_cite}):

```cpp
{traits}
```

its coefficient table, built in `InterpImpl::reset` as for `cp3` ({table_cite}):

```cpp
{table}
```

its one-face formula, a branch of `recon_point`, which the CPU loop and the CUDA line kernel share
({point_cite}):

```cpp
{point}
```

and a case in the CPU dispatch, `recon_line_cpu` ({cpu_case_cite}), and in the CUDA one, `recon_line_cuda`
({cuda_case_cite}):

```cpp
{cpu_case}
{cuda_case}
```

`dc`, `plm` and `ppm` are tensor code in `InterpImpl::forward`, `left` and `right` instead, with no kernel.
`InterpImpl::create` registers the module as a submodule of its parent with `register_module`
({interp_reg_cite}).

**Add a Riemann solver.** Besides the enum value and its `parse_riemann_kind` branch, a solver needs its face
function in `src/riemann/riemann_kernels.h`, called from `riemann_face` as for `lmars` ({face_cite}):

```cpp
{face}
```

the EOS quantities it reads, gathered per side in `RiemannSolverImpl::solve` as for `lmars` ({thermo_cite}):

```cpp
{thermo}
```

and a case in the CPU dispatch, `SNAP_RIEMANN_CPU` ({rs_cpu_cite}), and in the CUDA one, `SNAP_RIEMANN_CUDA`
({rs_cuda_cite}):

```cpp
{rs_cpu}
{rs_cuda}
```

A solver outside the capture allow-list runs eager: the CUDA-graph executor refuses it by name until a capture
test covers it, by the set `capture_riemann` ({capture_cite}):

```cpp
{capture}
```

**Files.** Every `.cpp` file in `src/recon/` and `src/riemann/` is compiled into the library, by the globs
`recon/*.cpp` and `riemann/*.cpp` ({glob_cite}), and with CUDA every `.cu` file there, by `riemann/*.cu`
({cuglob_cite}); a new file there needs no CMake edit.

**Face slots: which state is left and which is right.** The two slots of the reconstruction result are named
`ILT` and `IRT` by this enum ({slots_cite}):

```cpp
{slots}
```

Face $i$ is the face on the low side of cell $i$, between cells $i-1$ and $i$. The reconstruction hands the
interpolant two slices of the result, `outl` and `outr` ({apply_cite}):

```cpp
{apply}
```

so `left`, the value at each cell's low-side face, lands in slot `IRT` at the cell's own index, and `right`, the
value at each cell's high-side face, lands in slot `ILT` one index on. At index $i$, slot `ILT` holds the state
on the left of face $i$, from cell $i-1$, and slot `IRT` the state on its right, from cell $i$; step 5 of the
recipe shows exactly this. The sweep passes slot `ILT` as `wl` and slot `IRT` as `wr` to `priemann` ({call_cite}), and the
flux at index $i$ is the flux through face $i$: the update of cell $i$ uses the fluxes at $i$ and $i+1$, each
times its face area `area1` ({div_cite}):

```cpp
{call}
{div}
```

**Riemann shapes and face pressure.** The $x_1$ flux `_flux1` is the flux register's total for direction 1
({fluxbuf_cite}). The $x_1$ face pressure `_face_pressure1` is a float64 buffer of the full block, ghosts
included, one row only, and absent when the block has one cell along $x_1$ ({buffers_cite}):

```cpp
{fluxbuf}
{buffers}
```

The left and right states have the shape of the state, as the reconstruction result above has. The solver writes
the face pressure into `face_pressure_out`: the tensor it is given, or a scratch tensor of one row when it is
given none ({fp_out_cite}):

```cpp
{fp_out}
```

Only the $x_1$ solve passes the face pressure, `face_pressure1`, and not with the shallow-water EOS
({fp_pass_cite}); it goes to `riemann_solve` ({call1_cite}). The $x_2$ and $x_3$ solves pass none to
`riemann_solve` ({call23_cite}):

```cpp
{fp_pass}
{call1}
{call23}
```

`upwind`, `plume-roe` and `shallow-roe` refuse a defined face pressure, so a hydro run with them stops at the first
$x_1$ solve; for `upwind` it is this `TORCH_CHECK` ({refuse_cite}):

```cpp
{refuse}
```

The face pressure `face_pressure1` is read by the gnomonic-equiangle geometry ({fp_gnomonic_cite}) and by the
spherical-polar one with the exact $x_1$ centroid, `x1_centroid_exact_enabled` ({fp_spherical_cite}); the base
geometry ignores `face_pressure1` ({fp_base_cite}).

**Shapes, outputs and device.**

- Reconstruction output: the result is allocated like the input, so it has the input's dtype and device, and the
  interpolant writes into `result` ({alloc_cite}):

  ```cpp
  {alloc}
  ```

- In place: the solver writes the flux into the tensor it is given without resizing it; every operand shares one
  dtype except under `hllc`, whose face pressure may differ ({inplace_cite}):

  ```cpp
{indent(inplace)}
  ```

- Device: the kernel is chosen by the device of the flux tensor `flx` ({device_cite}):

  ```cpp
{indent(device)}
  ```

- Ownership: the solver keeps only a non-owning pointer back to its hydro, `phydro` ({parent_cite}):

  ```cpp
{indent(parent)}
  ```

`RiemannSolverImpl::create` registers the solver as a submodule of its parent with `register_module`
({register_cite}):

```cpp
{register}
```

**Rebuild.** Build the library with `cmake` ({rebuild_cite}) and reinstall the Python package with `pip`
({install_cite}):

```bash
{rebuild}
{install}
```

Then set the new name as `type` under `dynamics/reconstruct/vertical` or `horizontal`, or under
`dynamics/riemann-solver`.

**Check the registration.** This test builds a block from the repository deck with the names selected, and fails
when a name has no branch in `parse_scheme` or `parse_riemann_kind`. Run it from the repository root with the
rebuilt package, with the new names in place of `weno5` and `lmars`:

```python
from pathlib import Path
import snapy
import yaml

def selects(recon, riemann):
    """Build a block selecting these names; raises if one is unknown."""
    deck = yaml.safe_load(Path("examples/straka_single.yaml").read_text())
    deck["dynamics"]["reconstruct"] = {{
        "vertical": {{"type": recon}},
        "horizontal": {{"type": recon}},
    }}
    deck["dynamics"]["riemann-solver"] = {{"type": riemann}}
    Path("check.yaml").write_text(yaml.safe_dump(deck))
    options = snapy.MeshBlockOptions.from_yaml("check.yaml")
    return snapy.MeshBlock(options)

selects("weno5", "lmars")
try:
    selects("weno5", "mine")
except RuntimeError as err:
    assert "RiemannSolver: unknown type mine" in str(err)
else:
    raise AssertionError("an unregistered name was accepted")
print("N6 scheme: the selected names are registered")
```
<!-- end reader-reference: scheme-native -->
'''


def check(report=REPORT):
    skipped = code_anchors.legacy_chapter_of(TARGET, Path(report))
    if skipped:  # a legacy chapter (tools/code-anchors-legacy.txt): checked in full once its line is deleted
        print(f'[N6 scheme native] skipped: {skipped} is in tools/code-anchors-legacy.txt')
        return 0
    try:
        expected = generate(Path(report))
        actual = (Path(report) / TARGET).read_text()
        errors = [] if actual == expected else [
            f'{TARGET}: generated selection, lookup, classes or shapes differ from the pin']
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        errors = [str(exc)]
    print(f'[N6 scheme native] {len(errors)} problem(s)')
    for error in errors:
        print('  ' + error)
    return int(bool(errors))


if __name__ == '__main__':
    if '--write' in sys.argv:
        (REPORT / TARGET).write_text(generate(REPORT))
    else:
        sys.exit(check())
