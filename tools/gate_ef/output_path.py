"""Output-path gate (lane O's O5 gate tool, as run for R-2/R-3/EF-3/EF-4).

Writes netcdf block files through snapy's NetcdfOutput for several decks and
output options (every variable group, double precision, ghost zones, slices on
each axis, sums on each axis, combinations, a CCF user flux and a 2D user
field, the single-block combine), then byte-compares two runs.

usage:
  output_path.py --tree TREE --h348 HARNESS348_DIR --device cpu|cuda --out DIR
                                    write the files (TREE: decks are read here)
  output_path.py --compare DIR_A DIR_B   exit 0 only if every file is byte-identical
"""
import argparse
import hashlib
import os
import sys
import tempfile

import torch
import yaml

VARS = ["prim", "diagnostics", "thermo", "path", "avg", "implicit",
        "prim_stat", "scalar_stat", "scalar", "uov"]
SINGLE = ["d", "p", "D", "E", "m1", "m2", "m3", "vx", "vy", "vz"]


def decks(tree, h348_dir):
    out = {"straka2d": f"{tree}/examples/straka_single.yaml",
           "bryan2d": f"{tree}/examples/bryan.yaml",
           "jup1d": f"{tree}/examples/jupiter_evap_precip_1d.yaml",
           "box3d": f"{tree}/tests/test_forcing_3d.yaml"}
    # an implicit (VIC) gnomonic panel: implicit correction, lat/lon
    sys.path.insert(0, h348_dir)
    import h348
    out["gno_vic"] = h348.config(9, "face", "gnomonic-equiangle")
    out["cart_vic"] = h348.config(9, "cell", "cartesian")
    return out


def configs(co, ndim):
    def at(lo, hi, f):
        return lo + f * (hi - lo)
    x1 = [at(co.x1min(), co.x1max(), f) for f in (0.26, 0.61)]
    x2 = [at(co.x2min(), co.x2max(), f) for f in (0.33,)]
    x3 = [at(co.x3min(), co.x3max(), f) for f in (0.4, 0.8)]
    c = {"all": {}, "single": {"variables": SINGLE},
         "cons": {"variables": ["cons", "path", "avg"]},
         "dbl": {"double_precision": True},
         "ghost": {"include_ghost_zones": True},
         "s1": {"x1_slice": x1}, "sum1": {"output_sumx1": True}}
    if ndim >= 2:
        c.update({"s2": {"x2_slice": x2}, "sum2": {"output_sumx2": True},
                  "s1sum2": {"x1_slice": x1[:1], "output_sumx2": True},
                  "s12dbl": {"x1_slice": x1, "x2_slice": x2,
                             "double_precision": True}})
    if ndim >= 3:
        c.update({"s3": {"x3_slice": x3}, "sum3": {"output_sumx3": True},
                  "sum23": {"output_sumx2": True, "output_sumx3": True}})
    return c


def initial_w(block):
    kIDN, kIVX, kIVY, kIVZ, kIPR = 0, 1, 2, 3, 4  # snap.h, NMASS = 0
    bufs = dict(block.named_buffers())
    w = bufs["hydro.D"].clone().zero_()
    shape = w.shape[1:]
    k = torch.arange(shape[0]).view(-1, 1, 1)
    j = torch.arange(shape[1]).view(1, -1, 1)
    i = torch.arange(shape[2]).view(1, 1, -1)
    s = torch.sin(0.7 * i + 1.3 * j + 2.1 * k).to(w)
    w[kIDN] = 1. + 0.1 * s
    w[kIPR] = 1.e5 * (1. + 0.01 * s)
    w[kIVX], w[kIVY], w[kIVZ] = 3. * s, -2. * s, s
    for n in range(kIPR + 1, w.shape[0]):  # species: small mass fractions
        w[n] = 1.e-3 * (n - kIPR) * (1.5 + s)
    return w


def produce(a):
    from snapy import MeshBlock, MeshBlockOptions, OutputOptions, NetcdfOutput
    torch.set_default_dtype(torch.float64)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.txt"), "w")
    for name, deck in decks(a.tree, a.h348).items():
        if isinstance(deck, dict):
            with tempfile.NamedTemporaryFile("w", suffix=".yaml",
                                             delete=False) as f:
                yaml.safe_dump(deck, f)
                deck = f.name
        try:
            block = MeshBlock(MeshBlockOptions.from_yaml(deck))
            block.to(torch.device(a.device))
            vic = name.endswith("_vic")
            if vic:
                import h348
            w = h348.initial_w(block) if vic else initial_w(block)
            block_vars, _ = block.initialize({"hydro_w": w})
            nstage = len(block.module("intg").stages)
            dt = h348.DT if vic else 0.2 * block.max_time_step(block_vars)
            for stage in range(nstage):
                block.forward(block_vars, dt, stage)
        except Exception as e:  # recorded; must match across builds
            log.write(f"{name} SETUP-ERROR {str(e).splitlines()[0][:160]}\n")
            continue
        bufs = dict(block.named_buffers())
        x1f = bufs["coord.x1f"]
        nc3, nc2, nc1 = bufs["hydro.D"].shape[1:]

        def uov(v):
            return {"up_flux": (100. * x1f.view(1, 1, -1)
                                + torch.zeros(nc3, nc2, 1)).to(x1f),
                    "u2d": v["hydro_w"][0] * 2.}
        block.set_user_output_func(uov)
        co = block.options.coord()
        ndim = 1 + (nc2 > 1) + (nc3 > 1)
        for fid, (cname, extra) in enumerate(configs(co, ndim).items()):
            d = os.path.join(a.out, name, cname)
            block.options.output_dir(d)
            block.options.basename(name)
            op = OutputOptions().file_type("netcdf").variables(VARS)
            op.combine(cname == "all").fid(fid)
            for k, v in extra.items():
                getattr(op, k)(v)
            try:
                NetcdfOutput(op).write_output_file(block, block_vars, 0.5, 0)
                log.write(f"{name} {cname} ok\n")
            except Exception as e:
                log.write(f"{name} {cname} ERROR "
                          f"{str(e).splitlines()[0][:160]}\n")
    log.close()


def files(d):
    out = {}
    for root, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(root, f)
            out[os.path.relpath(p, d)] = hashlib.sha256(
                open(p, "rb").read()).hexdigest()
    return out


def compare(a, b):
    fa, fb = files(a), files(b)
    same = sum(fa[k] == fb.get(k) for k in fa)
    nc = sum(k.endswith(".nc") for k in fa)
    for k in sorted(set(fa) | set(fb)):
        if fa.get(k) != fb.get(k):
            print(f"DIFFERS {k}")
    print(f"files {len(fa)} vs {len(fb)} ({nc} netcdf); identical {same}")
    return same == len(fa) == len(fb)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tree")
    p.add_argument("--out")
    p.add_argument("--compare", nargs=2)
    p.add_argument("--h348", help="directory holding h348.py (94e4601e tools/harness348)")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.compare:
        sys.exit(0 if compare(*a.compare) else 1)
    produce(a)
