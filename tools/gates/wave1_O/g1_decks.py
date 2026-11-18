#!/usr/bin/env python3
"""Gate G-1 (lane O, O.rst section 5): example decks, 3 steps, every written
file compared by sha256 against the base build (the #356 procedure).

Each deck in examples/ runs with its own driver, `integration/nlim: 3` and a
restart dump every step. run_hydro's initial noise is unseeded rand_like, so
its decks run through a copy of the tree's examples/run_hydro.cpp with
torch::manual_seed(1234), compiled and linked from the tree's own build
(bin/run_hydro_seeded.<build-type>). The three convection benchmarks run to
a short end time with fixed arguments. A convection benchmark's
<csv>.json carries the wall-clock `seconds`; that file is compared with
the field removed. Every other file must be bitwise equal.

The gate runs the base twice (the control: base vs base) and the step once,
all on the same device, so CUDA is compared only with CUDA.

  g1_decks.py gate --base TREE --step TREE --out DIR [--device cpu|cuda]
  g1_decks.py run  --tree TREE --out DIR [--device cpu|cuda] [deck ...]
  g1_decks.py compare RUN_A RUN_B [deck ...]

TREE is a snapy checkout built with -DBUILD_EXAMPLES=ON in TREE/build
(override with --base-build/--step-build/--build). Exit status of `gate`:
0 pass, 1 a file differs from the base, 2 the control itself differs
(inconclusive), 3 a run or build failed.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

NSTEPS = 3
RESTART_DT = 1.0e-9  # one restart dump per step
WALLCLOCK = ("seconds",)

# The 13 decks that run in one process. The other 12 examples need 2-6
# processes (--multiprocess runs them through torchrun).
SINGLE = ["bryan", "cubed_sphere_convection", "earth_crm",
          "jupiter_evap_precip_1d", "planar_convection", "shallow_splash_mesh6",
          "shallow_xy_mesh4", "shallow_xy_single", "shock",
          "spherical_convection", "straka_mesh2", "straka_single", "uranus"]
MULTI = {"jupiter_crm": 4, "jupiter_crm_dry": 4, "jupiter_gcm": 6,
         "jupiter_gcm_dry": 6, "shallow_splash": 6,
         "shallow_splash_proc2_mesh3": 2, "shallow_splash_proc3_mesh2": 3,
         "shallow_xy": 4, "shallow_xy_proc2_mesh2": 2, "shallow_xy_proc4": 4,
         "straka": 2, "straka_proc2": 2}

DRIVER = {"bryan": "bryan", "jupiter_evap_precip_1d": "jupiter_evap_precip_1d",
          "shock": "shock", "cubed_sphere_convection": "cubed_sphere_convection",
          "planar_convection": "planar_convection",
          "spherical_convection": "spherical_convection"}
PREFIX = {"straka": "straka", "shallow_xy": "shallow_xy",
          "shallow_splash": "shallow_splash"}


def conv_args(driver, device):
    """Fixed short runs of the convection benchmarks (they stop at an end
    time, not at nlim). cubed_sphere_convection has no device argument and
    runs on the CPU under either --device."""
    dev = "cpu" if device == "cpu" else "cuda:0"
    return {
        "cubed_sphere_convection": ["450", "1e-4", "6", "6", "0.2", "stats.csv",
                                    "1e-5"],
        "planar_convection": ["700", "1.0", "0.5", "3.0", "8", "2e-4", "1e-6",
                              "0.8", "stats.csv", dev],
        "spherical_convection": ["1000", "0.6", "2", "8", "8", "1e-4", "1e-6",
                                 "0.8", "1", ".5", "stats.csv", dev],
    }.get(driver)


def driver_of(stem):
    if stem in DRIVER:
        return DRIVER[stem]
    for p, d in PREFIX.items():
        if stem == p or stem.startswith(p + "_"):
            return d
    return "run_hydro_seeded"


def sh(cmd, cwd):
    subprocess.run(["bash", "-c", cmd], cwd=cwd, check=True)


def ninja_command(build, target):
    out = subprocess.run(["ninja", "-C", str(build), "-t", "commands", target],
                         capture_output=True, text=True, check=True).stdout
    return out.strip().splitlines()[-1]


def build_seeded(tree, build, btype):
    """bin/run_hydro_seeded.<btype>: the tree's run_hydro.cpp plus
    torch::manual_seed(1234), with run_hydro's own compile and link lines."""
    src = (tree / "examples" / "run_hydro.cpp").read_text()
    anchor = "  torch::set_num_interop_threads(1);\n"
    if anchor not in src:
        raise RuntimeError("run_hydro.cpp: seeding anchor not found")
    seeded = build / "run_hydro_seeded.cpp"
    seeded.write_text(src.replace(anchor, anchor + "  torch::manual_seed(1234);\n", 1))
    obj = f"examples/CMakeFiles/run_hydro.{btype}.dir/run_hydro.cpp.o"
    cc = ninja_command(build, obj)
    cc = cc.replace(str(tree / "examples" / "run_hydro.cpp"), str(seeded))
    cc = cc.replace(obj, "run_hydro_seeded.o")
    cc = cc.replace(" -c ", f" -I{tree / 'examples'} -c ", 1)
    sh(cc, build)
    ld = ninja_command(build, f"bin/run_hydro.{btype}")
    ld = ld.replace(obj, "run_hydro_seeded.o")
    ld = ld.replace(f"bin/run_hydro.{btype}", f"bin/run_hydro_seeded.{btype}")
    sh(ld, build)


def prep_deck(src, dst):
    cfg = yaml.safe_load(open(src))
    integ = cfg.get("integration") or {}
    integ["nlim"] = NSTEPS
    cfg["integration"] = integ
    outs = cfg.get("outputs") or []
    restart = False
    for o in outs:
        if o.get("type") == "restart":
            o["dt"], restart = RESTART_DT, True
    if not restart:
        outs.append({"type": "restart", "dt": RESTART_DT})
    cfg["outputs"] = outs
    with open(dst, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(tree, build, out, device, decks, btype="release"):
    tree, build, out = Path(tree).resolve(), Path(build).resolve(), Path(out).resolve()
    bindir = build / "bin"
    if any(driver_of(d) == "run_hydro_seeded" for d in decks):
        build_seeded(tree, build, btype)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"device": device, "tree": str(tree), "decks": {}}
    for stem in decks:
        case = out / stem
        if case.exists():
            shutil.rmtree(case)
        case.mkdir()
        drv = driver_of(stem)
        deck = case / f"{stem}.yaml"
        cargs = conv_args(drv, device)
        if cargs:
            shutil.copy(tree / "examples" / f"{stem}.yaml", deck)
        else:
            prep_deck(tree / "examples" / f"{stem}.yaml", deck)
        exe = str(bindir / f"{drv}.{btype}")
        if cargs:
            args = [exe, deck.name] + cargs
        elif drv in ("run_hydro_seeded", "jupiter_evap_precip_1d"):
            args = [exe, "-i", deck.name]
        else:
            args = [exe, deck.name]
        nproc = MULTI.get(stem, 1)
        if nproc > 1:
            args = [shutil.which("torchrun") or "torchrun", "--no-python",
                    "--standalone", f"--nproc-per-node={nproc}"] + args
        env = dict(os.environ, OMP_NUM_THREADS="1", DEVICE=device,
                   BACKEND=os.environ.get("BACKEND", "gloo"))
        with open(case / "run.log", "w") as log:
            rc = subprocess.run(args, cwd=case, stdout=log, stderr=subprocess.STDOUT,
                                env=env).returncode
        files = {str(p.relative_to(case)): sha256(p) for p in sorted(case.rglob("*"))
                 if p.is_file() and p.name not in ("run.log", deck.name)}
        manifest["decks"][stem] = {"driver": drv, "nproc": nproc, "rc": rc,
                                   "files": files}
        print(f"{stem:28s} {drv:24s} np={nproc} rc={rc} files={len(files)}",
              flush=True)
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1, sort_keys=True)
    return manifest


def json_equal_without_clock(a, b):
    try:
        ja, jb = json.loads(Path(a).read_text()), json.loads(Path(b).read_text())
    except Exception:
        return False
    for k in WALLCLOCK:
        ja.pop(k, None)
        jb.pop(k, None)
    return ja == jb


def compare(ra, rb, decks=None, quiet=False):
    """Per deck: (status, differing files). status: BITWISE, CLOCK (equal but
    for a wall-clock field), DIFFERS, or FAILED (a run failed or the file
    sets differ)."""
    ra, rb = Path(ra), Path(rb)
    ma = json.load(open(ra / "manifest.json"))
    mb = json.load(open(rb / "manifest.json"))
    if ma.get("device") != mb.get("device"):
        raise SystemExit(f"device mismatch: {ma.get('device')} vs {mb.get('device')}")
    decks = decks or sorted(set(ma["decks"]) & set(mb["decks"]))
    res, nfile, nbit, nclock = {}, 0, 0, 0
    for d in decks:
        a, b = ma["decks"][d], mb["decks"][d]
        fa, fb = a["files"], b["files"]
        if a["rc"] != 0 or b["rc"] != 0 or set(fa) != set(fb):
            res[d] = ("FAILED", sorted(set(fa) ^ set(fb)))
            if not quiet:
                print(f"{d:28s} rc {a['rc']}/{b['rc']} files {len(fa)}/{len(fb)} FAILED")
            continue
        diff, clock = [], 0
        for f in fa:
            if fa[f] == fb[f]:
                continue
            if f.endswith(".json") and json_equal_without_clock(ra / d / f, rb / d / f):
                clock += 1
            else:
                diff.append(f)
        bit = len(fa) - clock - len(diff)
        nfile, nbit, nclock = nfile + len(fa), nbit + bit, nclock + clock
        status = "DIFFERS" if diff else ("CLOCK" if clock else "BITWISE")
        res[d] = (status, diff)
        if not quiet:
            note = {"BITWISE": "", "CLOCK": " json equal without the wall-clock field",
                    "DIFFERS": f" DIFFERS {diff}"}[status]
            print(f"{d:28s} files {len(fa)} bitwise {bit}/{len(fa)}{note}")
    if not quiet:
        ndiff = sum(s in ("DIFFERS", "FAILED") for s, _ in res.values())
        print(f"total: {len(decks)} decks, {nfile} files, {nbit} bitwise, {nclock} "
              f"equal without the wall-clock field, {ndiff} decks differ or failed")
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--base", required=True)
    g.add_argument("--step", required=True)
    g.add_argument("--base-build")
    g.add_argument("--step-build")
    g.add_argument("--out", required=True)
    g.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    g.add_argument("--build-type", default="release")
    g.add_argument("--multiprocess", action="store_true",
                   help="also the 12 decks that need 2-6 processes (torchrun)")
    g.add_argument("decks", nargs="*")
    r = sub.add_parser("run")
    r.add_argument("--tree", required=True)
    r.add_argument("--build")
    r.add_argument("--out", required=True)
    r.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    r.add_argument("--build-type", default="release")
    r.add_argument("decks", nargs="*")
    c = sub.add_parser("compare")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("decks", nargs="*")
    a = p.parse_args()

    if a.cmd == "compare":
        res = compare(a.run_a, a.run_b, a.decks or None)
        return 0 if all(s in ("BITWISE", "CLOCK") for s, _ in res.values()) else 1
    if a.cmd == "run":
        run(a.tree, a.build or Path(a.tree) / "build", a.out, a.device,
            a.decks or SINGLE, a.build_type)
        return 0

    decks = a.decks or (SINGLE + (sorted(MULTI) if a.multiprocess else []))
    out = Path(a.out)
    bb = a.base_build or Path(a.base) / "build"
    sb = a.step_build or Path(a.step) / "build"
    try:
        print(f"== base run A ({a.device})", flush=True)
        run(a.base, bb, out / "baseA", a.device, decks, a.build_type)
        print(f"== base run B, the control ({a.device})", flush=True)
        run(a.base, bb, out / "baseB", a.device, decks, a.build_type)
        print(f"== step run ({a.device})", flush=True)
        run(a.step, sb, out / "step", a.device, decks, a.build_type)
    except (subprocess.CalledProcessError, RuntimeError, FileNotFoundError) as e:
        print(f"run failed: {e}")
        return 3
    print("== control: base A vs base B")
    ctrl = compare(out / "baseA", out / "baseB", decks)
    print("== gate: base A vs step")
    gate = compare(out / "baseA", out / "step", decks)
    failed = [d for d, (s, _) in gate.items() if s == "FAILED"]
    unstable = [d for d, (s, _) in ctrl.items() if s in ("DIFFERS", "FAILED")]
    bad = [d for d, (s, _) in gate.items() if s == "DIFFERS" and d not in unstable]
    summary = {"device": a.device, "decks": decks, "control_unstable": unstable,
               "differs": bad, "failed": failed,
               "gate": {d: s for d, (s, _) in gate.items()}}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    if failed:
        print(f"G-1 FAILED RUNS: {failed}")
        return 3
    if bad:
        print(f"G-1 FAIL: {bad} differ from the base")
        return 1
    if unstable:
        print(f"G-1 INCONCLUSIVE: the control differs for {unstable}")
        return 2
    print(f"G-1 PASS ({a.device}): {len(decks)} decks equal to the base; control "
          "equal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
