#!/usr/bin/env python3
"""On the cubed sphere the shallow-roe solver takes only dir: yz (UCzhangxi/snapy#319).

RiemannSolverImpl::reset refuses any other dir for the gnomonic-equiangle
coordinate; a deck with dir: xy must be refused when the mesh is built.
"""
import os
import tempfile

import yaml
from snapy import Mesh, MeshOptions

DECK = """
geometry:
  type: gnomonic-equiangle
  bounds: {x1min: 6371000., x1max: 6371001., x2min_pi: -0.25, x2max_pi: 0.25,
           x3min_pi: -0.25, x3max_pi: 0.25}
  cells: {nx1: 1, nx2: 8, nx3: 8, nghost: 3}
distribute: {layout: cubed-sphere, nb2: 1, nb3: 1, blocks_per_process: 6, verbose: false}
dynamics:
  equation-of-state: {type: shallow-water}
  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}
  riemann-solver: {type: shallow-roe, dir: yz}
boundary-condition:
  external: {x1-inner: reflecting, x1-outer: reflecting, x2-inner: custom,
             x2-outer: custom, x3-inner: custom, x3-outer: custom}
integration: {type: rk3, cfl: 0.5, nlim: 1, tlim: 1.e12, ncycle_out: 0}
"""


def build(direction):
    cfg = yaml.safe_load(DECK)
    cfg["dynamics"]["riemann-solver"]["dir"] = direction
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
    try:
        options = MeshOptions.from_yaml(f.name)
    finally:
        os.unlink(f.name)
    options.set_local_horizontal_cells(8, 8)
    return Mesh(options)


assert len(build("yz").blocks) == 6
try:
    build("xy")
    raise AssertionError("cubed-sphere shallow-roe with dir: xy was not refused")
except RuntimeError as e:
    assert "only supports options->dir() = 'yz'" in str(e), e
print("shallow-roe on the cubed sphere refuses dir: xy: OK")
