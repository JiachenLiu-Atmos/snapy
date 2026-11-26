"""N6: the native C++ route for a new reconstruction or Riemann solver, extracted from the code pin.

It completes the scheme contract of tools/gen_scheme_contract.py (the Python recipe, the interface, the faces, the
stencil and ghost width): this block gives how a scheme is selected from YAML and looked up, which class and file
to write, the shapes and in-place outputs, the dtype, device and ownership rules, and the rebuild. Every fact is an
excerpt that must match the pin exactly once, cited with a line permalink.
"""
import re
import subprocess
import sys
from pathlib import Path

import code_anchors

REPORT = Path(__file__).resolve().parent.parent
TARGET = 'book/chapters/04-spatial-discretization/_howto-native.qmd'
PIN = '919936da07d80a310b98ab11da184972a3203274'


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

    # reconstruction: selection, lookup, names, class, interface, files
    recon_yaml, recon_yaml_cite = excerpt('src/recon/reconstruct.cpp',
        r'^  check_keys\(node\[section\], "dynamics/reconstruct/" \+ section,\n.*?'
        r'  op->interp\(\)->scale\(\) = node\[section\]\["scale"\]\.as<bool>\(false\);')
    sections, sections_cite = excerpt('src/hydro/hydro_options.cpp',
        r'^  op->recon1\(\) = ReconstructOptionsImpl::from_yaml\(filename, "vertical"\);\n.*?'
        r'  op->recon23\(\) = ReconstructOptionsImpl::from_yaml\(filename, "horizontal"\);')
    names, names_cite = excerpt('src/recon/reconstruct.cpp',
        r'^  pinterp1 = InterpImpl::create\(options->interp\(\), this, "interp1"\);\n'
        r'  pinterp2 = InterpImpl::create\(options->interp\(\), this, "interp2"\);')
    weno, weno_cite = excerpt('src/recon/interpolation.cpp',
        r'^  \} else if \(opts->type\(\) == "weno5"\) \{\n.*?'
        r'      TORCH_CHECK\(false, "Interp: unknown name " \+ name\);\n    \}')
    interp_unknown, interp_unknown_cite = excerpt('src/recon/interpolation.cpp',
        r'^    TORCH_CHECK\(false, "Interp: unknown type " \+ opts->type\(\)\);')
    interp_base, interp_base_cite = excerpt('src/recon/interpolation.hpp',
        r'^  virtual std::pair<torch::Tensor, torch::Tensor> forward\(\n.*?'
        r'    return std::make_pair\(wlv, wrv\);\n  \}')
    dc_class, dc_class_cite = excerpt('src/recon/interpolation.hpp',
        r'^class DonorCellInterpImpl : public torch::nn::Cloneable<DonorCellInterpImpl>,\n.*?'
        r'TORCH_MODULE\(DonorCellInterp\);')
    alloc, alloc_cite = excerpt('src/recon/reconstruct.cpp',
        r'^  auto result = torch::zeros\(vec, w\.options\(\)\);')
    glob, glob_cite = excerpt('src/CMakeLists.txt', r'^    riemann/\*\.cpp\n    recon/\*\.cpp')
    cuglob, cuglob_cite = excerpt('src/CMakeLists.txt', r'^      recon/\*\.cu\n      riemann/\*\.cu')

    # Riemann solver: selection, defaults, lookup, class, call site, in-place flux
    rs_yaml, rs_yaml_cite = excerpt('src/riemann/riemann_solver.cpp',
        r'^  op->type\(\) = node\["type"\]\.as<std::string>\("roe"\);\n'
        r'  op->dir\(\) = node\["dir"\]\.as<std::string>\("omni"\);')
    rs_keys, rs_keys_cite = excerpt('src/riemann/riemann_solver.cpp',
        r'^  check_keys\(config\["dynamics"\]\["riemann-solver"\], "dynamics/riemann-solver",\n[^;]+;')
    rs_none, rs_none_cite = excerpt('src/riemann/riemann_solver.cpp',
        r'^  if \(!config\[section\]\["riemann-solver"\]\) return op;')
    rs_default, rs_default_cite = excerpt('src/riemann/riemann_solver.hpp',
        r'^  ADD_ARG\(std::string, type\) = "hllc";')
    rs_unknown, rs_unknown_cite = excerpt('src/riemann/riemann_solver.cpp',
        r'^    TORCH_CHECK\(false, "RiemannSolver: unknown type " \+ opts->type\(\)\);')
    roe_class, roe_class_cite = excerpt('src/riemann/riemann_solver.hpp',
        r'^class RoeSolverImpl : public torch::nn::Cloneable<RoeSolverImpl>,\n.*?TORCH_MODULE\(RoeSolver\);')
    call, call_cite = excerpt('src/hydro/hydro_forward.cpp',
        r'^      priemann->forward\(wlr1\[ILT\], wlr1\[IRT\], DIM1, _flux1, face_pressure1\);')
    inplace, inplace_cite = excerpt('src/riemann/roe.cpp',
        r'^  auto iter = at::TensorIteratorConfig\(\)\n.*?\.build\(\);')
    device, device_cite = excerpt('src/riemann/roe.cpp',
        r'^  at::native::call_roe\(flx\.device\(\)\.type\(\), iter, dim, ideal_moist, nvapor,')
    parent, parent_cite = excerpt('src/riemann/riemann_solver.hpp',
        r'^  //! non-owning reference to parent\n  HydroImpl const\* phydro = nullptr;')
    register, register_cite = excerpt('src/riemann/riemann_solver.cpp',
        r'^  if \(opts->type\(\) == "roe"\) \{\n    return p->register_module\(name, RoeSolver\(opts, p\)\);')
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
    div, div_cite = excerpt('src/coord/coordinate.cpp',
        r'^    dflx\.slice\(DIM1, si, ei\) \+=\n        face_area1\(si \+ 1, ei \+ 1\) \* flux1\.slice\(DIM1, si \+ 1, ei \+ 1\) -\n'
        r'        face_area1\(si, ei\) \* flux1\.slice\(DIM1, si, ei\);')
    dc_branch, dc_branch_cite = excerpt('src/recon/interpolation.cpp',
        r'^  if \(opts->type\(\) == "dc"\) \{\n    return p->register_module\(name, DonorCellInterp\(opts\)\);')
    # Riemann shapes and face pressure
    buffers, buffers_cite = excerpt('src/hydro/hydro.cpp',
        r'^  if \(nc1 > 1\) \{\n    _flux1 = register_buffer\(\n.*?'
        r'    _face_pressure1 = register_buffer\("P1", torch::Tensor\(\)\);\n  \}')
    fp_out, fp_out_cite = excerpt('src/riemann/roe.cpp',
        r'^  auto face_pressure_out =\n      face_pressure\.defined\(\) \? face_pressure : torch::empty_like\(wl\[IDN\]\);')
    fp_pass, fp_pass_cite = excerpt('src/hydro/hydro_forward.cpp',
        r'^      auto face_pressure1 = options->eos\(\)->type\(\) == "shallow-water"\n'
        r'                                \? torch::Tensor\(\)\n                                : _face_pressure1;')
    call23, call23_cite = excerpt('src/hydro/hydro_forward.cpp',
        r'^      priemann->forward\(wlr2\[ILT\], wlr2\[IRT\], DIM2, _flux2\);')
    call3, call3_cite = excerpt('src/hydro/hydro_forward.cpp',
        r'^      priemann->forward\(wlr3\[ILT\], wlr3\[IRT\], DIM3, _flux3\);')
    refuse, refuse_cite = excerpt('src/riemann/riemann_solver.cpp',
        r'^  TORCH_CHECK\(\n      !face_pressure\.defined\(\),\n.*?options->type\(\)\);')
    fp_gnomonic, fp_gnomonic_cite = excerpt('src/coord/gnomonic_equiangle.cpp',
        r'^    if \(face_pressure1\.defined\(\)\) \{')
    fp_spherical, fp_spherical_cite = excerpt('src/coord/spherical_polar.cpp',
        r'^    if \(face_pressure1\.defined\(\) && x1_centroid_exact_enabled\(\)\) \{')
    fp_base, fp_base_cite = excerpt('src/coord/coordinate.cpp', r'^  \(void\)face_pressure1;')

    return f'''<!-- generated reader-reference: scheme-native -->
<!-- Generated by tools/gen_scheme_native.py; do not edit. -->
#### Native C++ route

The recipe above prototypes a reconstruction in Python and selects a built-in one by name; it stays the one
recipe. A new scheme that the solver itself runs is a C++ class, built into the library and selected from the YAML
card, and this route is the one place that says how, for a reconstruction and for a Riemann solver. Each step below
is a fact of the code at the pin.

**Select it from YAML.** A reconstruction is read from `dynamics/reconstruct/vertical` for $x_1$ and from
`dynamics/reconstruct/horizontal` for $x_2$ and $x_3$ ({sections_cite}). Each section takes `shock`, `type` and
`scale`; `type` defaults to `dc` ({recon_yaml_cite}):

```cpp
{recon_yaml}
```

A Riemann solver is read from `dynamics/riemann-solver`, whose keys are `type` and `dir` ({rs_keys_cite}).
A section without `type` gives `roe` ({rs_yaml_cite}):

```cpp
{rs_yaml}
```

A card with no such section keeps the options as created ({rs_none_cite}), whose default is another solver,
`hllc` ({rs_default_cite}):

```cpp
{rs_none}
{rs_default}
```

**Lookup.** The name is compared as a string, branch by branch, in `InterpImpl::create` and
`RiemannSolverImpl::create`; there is no table and no suffix added to the name. A name with no branch is refused
when the block is built ({interp_unknown_cite}, {rs_unknown_cite}):

```cpp
{interp_unknown}
{rs_unknown}
```

A reconstruction module is created twice, under the names `interp1` and `interp2` ({names_cite}):

```cpp
{names}
```

and for `weno3` and `weno5` the last character of that name chooses the class ({weno_cite}):

```cpp
{weno}
```

**Write the class.** The header `src/recon/interpolation.hpp` declares `InterpImpl`, the base of every
reconstruction. Its `forward` sizes the two outputs from `stencils()` and fills them through `left` and `right`
({interp_base_cite}):

```cpp
{interp_base}
```

The donor-cell class is the smallest one to follow ({dc_class_cite}):

```cpp
{dc_class}
```

A Riemann solver derives from `RiemannSolverImpl`, declared in `src/riemann/riemann_solver.hpp`
({roe_class_cite}):

```cpp
{roe_class}
```

Add one branch to the matching `create`, as for `dc` ({dc_branch_cite}) and `roe` ({register_cite}):

```cpp
{dc_branch}
{register}
```

**Files.** Every `.cpp` file in `src/recon/` and `src/riemann/` is compiled into the library
({glob_cite}), and with CUDA every `.cu` file there ({cuglob_cite}); a new file there needs no CMake edit.

**Face slots: which state is left and which is right.** The two slots of the reconstruction result are named
by this enum ({slots_cite}):

```cpp
{slots}
```

Face $i$ is the face on the low side of cell $i$, between cells $i-1$ and $i$. The reconstruction hands the method
two slices of the result ({apply_cite}):

```cpp
{apply}
```

so `left`, the value at each cell's low-side face, lands in slot `IRT` at the cell's own index, and `right`, the
value at each cell's high-side face, lands in slot `ILT` one index on. At index $i$, slot `ILT` holds the state
on the left of face $i$, from cell $i-1$, and slot `IRT` the state on its right, from cell $i$; the donor-cell
`left` and `right` above and step 5 of the recipe show exactly this. The solver takes slot `ILT` as `wl` and slot
`IRT` as `wr` ({call_cite}), and its flux at index $i$ is the flux through face $i$: the update of cell $i$ uses
the fluxes at $i$ and $i+1$ ({div_cite}):

```cpp
{div}
```

**Riemann shapes and face pressure.** The flux and the face pressure of $x_1$ are float64 buffers of the full
block, ghosts included: the flux has one row per variable, the face pressure one row only, and neither exists
when the block has one cell along $x_1$ ({buffers_cite}):

```cpp
{buffers}
```

The left and right states have the shape of the state, as the reconstruction result above has. The solver writes
the face pressure into the tensor it is given, or into a scratch tensor of one row when it is given none
({fp_out_cite}):

```cpp
{fp_out}
```

Only the $x_1$ call passes the face pressure, and not with the shallow-water EOS ({fp_pass_cite}); the $x_2$ and
$x_3$ calls pass none ({call23_cite}, {call3_cite}):

```cpp
{fp_pass}
{call23}
{call3}
```

A solver that keeps the base `forward` refuses a defined face pressure ({refuse_cite}), so a hydro run with
it stops at the first $x_1$ call, as the built-in `upwind` does; a solver for the hydro overrides `forward` and
writes the face pressure:

```cpp
{refuse}
```

The face pressure is read by the gnomonic-equiangle geometry ({fp_gnomonic_cite}) and by the spherical-polar one
with the exact $x_1$ centroid ({fp_spherical_cite}); the base geometry ignores it ({fp_base_cite}).

**Shapes, outputs and device.**

- Reconstruction output: the result is allocated like the input, so it has the input's dtype and device, and the
  method writes into it ({alloc_cite}):

  ```cpp
  {alloc}
  ```

- Riemann call: the solver gets the left and right states from the reconstruction, each of the shape of the state
  (the result above is the state's shape with 2 prepended), the direction, and the flux tensor to fill
  ({call_cite}):

  ```cpp
  {call}
  ```

- In place, one dtype: the solver writes the flux into its fourth argument without resizing it, and all tensors
  must share one dtype ({inplace_cite}):

  ```cpp
{chr(10).join("  " + l for l in inplace.splitlines())}
  ```

- Device: the kernel is chosen by the device of the flux tensor ({device_cite}):

  ```cpp
  {device}
  ```

- Ownership: `create` registers the new module as a submodule of its parent with `register_module`
  ({register_cite}); the solver keeps only a non-owning pointer back to its hydro ({parent_cite}):

  ```cpp
{chr(10).join("  " + l for l in parent.splitlines())}
  ```

**Rebuild.** Build the library ({rebuild_cite}) and reinstall the Python package ({install_cite}):

```bash
{rebuild}
{install}
```

Then set the new name as `type` under `dynamics/reconstruct/vertical` or `horizontal`, or under
`dynamics/riemann-solver`.

**Check the registration.** This test builds a block from the repository deck with the names selected, and fails
when a name has no branch in `create`. Run it from the repository root with the rebuilt package, with the new
names in place of `weno5` and `lmars`:

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
