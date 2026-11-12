#!/usr/bin/env python3
"""Anomaly storage of hydro_u (fp32 roadmap, Step 1a) and the fp32-storage emulation hook.

With `dynamics: anomaly` the block stores rows IDN and IPR of hydro_u as anomalies about a
static fp64 reference (hydro.ref.u0, the initial state at rest). In fp64 that is an exact
rewrite, so a run with it on must match the run with it off to round-off. The arms:

  reference   hydro.ref.u0 exists, is float64, survives block.to(float32), and is a broadcast
              column for a horizontally uniform initial state; the initial IDN anomaly is 0
  match       N steps (moist, with the saturation adjustment) on vs off: totals and
              primitives agree to round-off
  keys        "U->W" on the full stored hydro_u raises; "U'->W" reproduces hydro_w;
              total_u / stored_u round-trip
  switch      SNAP_ANOMALY=1 turns it on without the YAML key; an unsupported EOS refuses it
  restart     write a restart mid-run, resume from it: the continuation is bit-identical to the
              uninterrupted run (hydro_u_stored), and the file holds totals plus hydro_ref
  float_restart FP32 stored anomalies retain dtype and bits; references stay FP64
  limiter     a cell pushed below the density and energy floors is clipped in anomaly space to
              the same totals as the totals path clips it; every other cell is untouched
  fp32        SNAP_FP32_STORAGE=step leaves hydro_u representable in float32 after a step;
              SNAP_FP32_STORAGE_ROWS=IPR rounds that row only

  python test_anomaly_storage.py [--device cuda] [--steps N]
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
YAML = os.path.join(HERE, "test_anomaly_storage.yaml")


def make_block(device, anomaly=None, eos_type=None, outputs=None, conversion_order=2):
    from snapy import MeshBlock, MeshBlockOptions

    with open(YAML) as f:
        cfg = yaml.safe_load(f)
    cfg["dynamics"]["conversion-order"] = conversion_order
    if anomaly is not None:
        cfg["dynamics"]["anomaly"] = anomaly
    if outputs is not None:
        cfg["outputs"] = outputs
    if eos_type is not None:
        cfg["dynamics"]["equation-of-state"] = {"type": eos_type}
        for k in ("species", "reactions"):
            cfg.pop(k, None)
        cfg["species"] = [{"name": "dry", "composition": {"N": 2}, "cv_R": 2.5}]
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device))
    return block


def modules(block):
    mods = dict(block.named_modules())
    eos = next(m for m in mods.values() if type(m).__name__ == "IdealMoist")
    return eos, mods["hydro.eos.thermo"]


def initial_state(block):
    """A moist, stably stratified column with a horizontal velocity perturbation."""
    from snapy import kICY, kIDN, kIPR, kIV1 as kIVX, kIV2 as kIVY

    eos, thermo = modules(block)
    u = dict(block.named_buffers())["hydro.D"].clone().zero_()
    bufs = dict(block.named_buffers())
    z = bufs["coord.x1v"].view(1, 1, -1).expand_as(u[kIDN])
    x2 = bufs["coord.x2v"].view(1, -1, 1).expand_as(u[kIDN])

    g, T0, gam, p0 = 10.44, 300.0, 6.5e-3, 1.0e5
    Rd = 287.0
    temp = (T0 - gam * z).clamp_min(200.0)
    pres = p0 * (temp / T0) ** (g / (Rd * gam))
    rho = pres / (Rd * temp)
    y = torch.zeros_like(u[kICY:])
    y[0] = 0.012 * torch.exp(-z / 2500.0)

    ie = thermo.compute("VT->U", [thermo.compute("DY->V", [rho, y]), temp])
    u[kIDN] = rho * (1.0 - y.sum(0))
    u[kICY:] = rho * y
    u[kIPR] = ie
    w = eos.compute("U->W", [u])  # no reference yet: totals
    w[kIVY] = 0.5 * torch.sin(2.0 * math.pi * x2 / 4.0e3)
    w[kIVX] = 0.2 * torch.cos(2.0 * math.pi * x2 / 4.0e3) * torch.sin(math.pi * z / 1.0e4)
    return {"hydro_w": w}


def run_steps(block, block_vars, nsteps, dt=None):
    dts = []
    for _ in range(nsteps):
        h = block.max_time_step(block_vars) if dt is None else dt
        for stage in range(len(block.intg.stages)):
            block.forward(block_vars, h, stage)
        err = block.check_redo(block_vars)
        if err != 0:
            raise RuntimeError("check_redo returned %d" % err)
        dts.append(h)
    return dts


def interior(block, t):
    sl = block.part((0, 0, 0), False)
    return t[sl]


def rel(a, b):
    scale = b.abs().amax(dim=tuple(range(1, b.dim())), keepdim=True).clamp_min(1e-300)
    return ((a - b).abs() / scale).amax().item()


def arm_reference(device):
    from snapy import kIDN, kIPR

    failures = []
    block = make_block(device, anomaly=True)
    bv, _ = block.initialize(initial_state(block))
    bufs = dict(block.named_buffers())
    if not block.anomaly():
        return ["block.anomaly is False with dynamics: anomaly true"]
    if "hydro.ref.u0" not in bufs:
        return ["hydro.ref.u0 is not a registered buffer: %s" % sorted(bufs)[:8]]
    r = bufs["hydro.ref.u0"]
    species = bufs["hydro.ref.species0"]
    if species.dtype != torch.float64:
        failures.append("species background is not float64")
    from snapy import kICY
    if not torch.equal(species.expand_as(bv["hydro_u"][kICY:]),
                       bv["hydro_u"][kICY:]):
        failures.append("species background differs from initial conserved species")
    print("reference  shape %s dtype %s" % (tuple(r.shape), r.dtype))
    if r.dtype != torch.float64:
        failures.append("hydro.ref.u0 is %s, not float64" % r.dtype)
    if r.shape[0] != 2 or r.shape[1] != 1 or r.shape[2] != 1:
        failures.append("a horizontally uniform reference must be a column, got %s"
                        % (tuple(r.shape),))
    u = interior(block, bv["hydro_u"])
    d0 = u[kIDN].abs().max().item()
    print("reference  max |IDN anomaly| at t=0: %.3e" % d0)
    if d0 != 0.0:
        failures.append("initial IDN anomaly is %.3e, expected exactly 0" % d0)
    e0 = u[kIPR].abs().max().item()
    if not (0.0 < e0 < 1.0e3):  # the kinetic energy of the perturbation only
        failures.append("initial IPR anomaly %.3e is not the perturbation's kinetic energy" % e0)

    # dtype conversion keeps the reference in float64; device moves still apply
    block.to(torch.float32)
    r32 = dict(block.named_buffers())["hydro.ref.u0"]
    sr32 = dict(block.named_buffers())["hydro.ref.species0"]
    if sr32.dtype != torch.float64 or not torch.equal(sr32, species):
        failures.append("block.to(float32) changed the species background")
    other = dict(block.named_buffers())["hydro.D"]
    print("reference  after block.to(float32): ref %s, hydro.D %s" % (r32.dtype, other.dtype))
    if r32.dtype != torch.float64:
        failures.append("block.to(float32) demoted hydro.ref.u0 to %s" % r32.dtype)
    return failures


def arm_match(device, nsteps, conversion_order=2):
    print("match      conversion-order=%d" % conversion_order)
    failures = []
    runs = {}
    for anomaly in (False, True):
        block = make_block(device, anomaly=anomaly, conversion_order=conversion_order)
        bv, _ = block.initialize(initial_state(block))
        dts = run_steps(block, bv, nsteps)
        runs[anomaly] = (block, bv, dts)
    b0, v0, dt0 = runs[False]
    b1, v1, dt1 = runs[True]
    ddt = max(abs(a - b) / a for a, b in zip(dt0, dt1))
    u_off = interior(b0, v0["hydro_u"])
    u_on = interior(b1, b1.total_u(v1))
    du = rel(u_on, u_off)
    dw = rel(interior(b1, v1["hydro_w"]), interior(b0, v0["hydro_w"]))
    print("match      %d steps: max rel diff totals %.3e, primitives %.3e, dt %.3e"
          % (nsteps, du, dw, ddt))
    tol = 1.0e-10
    if not (du < tol and dw < tol and ddt < tol):
        failures.append("anomaly on vs off differ beyond round-off: totals %.3e, primitives "
                        "%.3e, dt %.3e (tol %.0e)" % (du, dw, ddt, tol))
    return failures


def arm_keys(device):
    failures = []
    block = make_block(device, anomaly=True)
    bv, _ = block.initialize(initial_state(block))
    run_steps(block, bv, 2)
    eos, _ = modules(block)
    try:
        eos.compute("U->W", [bv["hydro_u"].clone()])
        failures.append("'U->W' on the stored full-block hydro_u did not raise")
    except ValueError as exc:
        print("keys       'U->W' on stored hydro_u raises: %s" % str(exc)[:72])
    w = eos.compute("U'->W", [bv["hydro_u"].clone()])
    dw = rel(interior(block, w), interior(block, bv["hydro_w"]))
    print("keys       'U'->W' vs hydro_w: %.3e" % dw)
    if dw > 1e-13:
        failures.append("'U'->W' does not reproduce hydro_w (%.3e)" % dw)
    u = bv["hydro_u"]
    back = block.stored_u(block.total_u(bv))
    db = (back - u).abs().max().item() / block.total_u(bv).abs().max().item()
    print("keys       |stored_u(total_u(u)) - u| / max|total|: %.3e" % db)
    if db > 1e-15:
        failures.append("stored_u(total_u(u)) does not return u (%.3e)" % db)
    u2 = eos.compute("W->U'", [bv["hydro_w"].clone()])
    d2 = rel(interior(block, block.total_u(u2)), interior(block, block.total_u(bv)))
    print("keys       W->U' round trip vs stored state (totals): %.3e" % d2)
    if d2 > 1e-12:
        failures.append("W->U' does not reproduce the stored state (%.3e)" % d2)
    return failures


def arm_switch(device):
    failures = []
    try:
        make_block(device, anomaly=True, eos_type="ideal-gas")
        failures.append("anomaly with an ideal-gas EOS was accepted")
    except Exception as exc:  # noqa: BLE001
        if "anomaly" not in str(exc):
            failures.append("ideal-gas + anomaly raised without naming it: %s" % exc)
        else:
            print("switch     ideal-gas refuses anomaly")
    res = subprocess.run([sys.executable, __file__, "--child", "env", "--device", device],
                         env=dict(os.environ, SNAP_ANOMALY="1"), capture_output=True, text=True)
    print("switch     SNAP_ANOMALY=1 child: %s" % res.stdout.strip().splitlines()[-1:])
    if res.returncode != 0 or "ANOMALY=True" not in res.stdout:
        failures.append("SNAP_ANOMALY=1 did not turn anomaly on:\n%s%s" % (res.stdout, res.stderr))
    return failures


def arm_float_restart(device):
    """Check FP32 storage serialization, without claiming full FP32 evolution."""
    import glob
    from snapy import kIPR

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            outputs = [{"type": "restart", "dt": 1.0e9}]
            block = make_block(device, anomaly=True, outputs=outputs)
            state, _ = block.initialize(initial_state(block))
            state["hydro_u"] = state["hydro_u"].float()
            state["hydro_u"][kIPR].fill_(0.01234567)
            expected = state["hydro_u"].clone()
            block.make_outputs(state, 0., True)
            path, = glob.glob(os.path.join(tmp, "*.final.restart"))
            resumed = make_block(device, anomaly=True, outputs=outputs)
            restored, _ = resumed.initialize_from_restart(path)
            actual = restored["hydro_u"]
            if actual.dtype != expected.dtype or not torch.equal(actual, expected):
                return ["FP32 restart changed stored anomaly dtype or bits"]
            for name in ("hydro.ref.u0", "hydro.ref.species0"):
                if dict(resumed.named_buffers())[name].dtype != torch.float64:
                    return ["FP32 restart demoted the reference"]
            print("restart    FP32 stored dtype and bits preserved, references FP64")
            return []
        finally:
            os.chdir(cwd)


def arm_restart(device):
    """Resume from a restart mid-run; compare with the uninterrupted run, both arms."""
    import glob
    import snapy

    failures = []
    cwd = os.getcwd()
    for anomaly in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                outputs = [{"type": "restart", "dt": 1.0e9}]
                ref = make_block(device, anomaly=anomaly, outputs=outputs)
                bv, _ = ref.initialize(initial_state(ref))
                t0 = sum(run_steps(ref, bv, 3))
                ref.make_outputs(bv, t0, True)
                files = glob.glob(os.path.join(tmp, "*.final.restart"))
                if len(files) != 1:
                    failures.append("restart (anomaly=%s): expected one file, got %s"
                                    % (anomaly, files))
                    continue
                dts = run_steps(ref, bv, 3)

                data = snapy.load_restart(files[0], 0)
                if anomaly:
                    missing = [k for k in ("hydro_ref", "hydro_ref_species", "hydro_u_stored") if k not in data]
                    if missing:
                        failures.append("anomaly restart file lacks %s" % missing)

                cont = make_block(device, anomaly=anomaly, outputs=outputs)
                cv, t1 = cont.initialize_from_restart(files[0])
                if cont.anomaly() != anomaly:
                    failures.append("restart: anomaly() is %s after resume" % cont.anomaly())
                for h in dts:
                    run_steps(cont, cv, 1, dt=h)
                same_u = torch.equal(cv["hydro_u"], bv["hydro_u"])
                du = rel(cont.total_u(cv), ref.total_u(bv))
                if anomaly:
                    rr = dict(ref.named_buffers())["hydro.ref.u0"]
                    rc = dict(cont.named_buffers())["hydro.ref.u0"]
                    if not torch.equal(rr, rc):
                        failures.append("restart: hydro.ref.u0 changed across the resume")
                    sr = dict(ref.named_buffers())["hydro.ref.species0"]
                    sc = dict(cont.named_buffers())["hydro.ref.species0"]
                    if not torch.equal(sr, sc):
                        failures.append("restart: species background changed")
                    # totals in the file, the stored anomaly kept bit for bit
                    tot = data["hydro_u"].to(rr.device)
                    sto = data["hydro_u_stored"].to(rr.device)
                    back = ref.stored_u(tot)
                    print("restart    anomaly file: totals - stored differ by the ref "
                          "(|stored_u(totals) - stored| = %.1e)"
                          % (back - sto).abs().max().item())
                print("restart    anomaly=%-5s resumed run bit-identical: %s (rel %.1e)"
                      % (anomaly, same_u, du))
                if not same_u:
                    failures.append("restart (anomaly=%s): resumed run differs from the "
                                    "uninterrupted one (rel %.3e)" % (anomaly, du))
            finally:
                os.chdir(cwd)
    return failures


def arm_limiter(device):
    """Push one interior cell below the density and energy floors; the anomaly-space
    limiter must give the totals the totals-space limiter gives, and leave the rest alone."""
    from snapy import kIDN, kIPR

    failures = []
    out = {}
    out_ref = None
    for anomaly in (False, True):
        block = make_block(device, anomaly=anomaly)
        bv, _ = block.initialize(initial_state(block))
        run_steps(block, bv, 2)
        eos, _ = modules(block)
        tot = block.total_u(bv)
        k, j, i = tot.size(1) // 2, tot.size(2) // 2, tot.size(3) // 2
        bad = tot.clone()
        bad[kIDN, k, j, i] = -1.0e-3   # below the density floor
        bad[kIPR, k, j, i + 1] = 0.0   # below the temperature floor's energy
        stored = block.stored_u(bad)
        before = stored.clone()
        if anomaly:
            eos.compute("U'->W", [stored])  # limits the stored state in place
            got = block.total_u(stored)
        else:
            eos.compute("U->W", [stored])
            got = stored
        mask = torch.zeros_like(got[0], dtype=torch.bool)
        mask[k, j, i] = mask[k, j, i + 1] = True
        untouched = torch.equal(stored[:, ~mask], before[:, ~mask])
        out[anomaly] = (got, untouched, (k, j, i))
        if anomaly:
            out_ref = block
        print("limiter    anomaly=%-5s floored rho_d=%.3e, E=%.6e; other cells untouched: %s"
              % (anomaly, got[kIDN, k, j, i].item(), got[kIPR, k, j, i + 1].item(),
                 untouched))
        if not untouched:
            failures.append("limiter (anomaly=%s) changed cells it should not touch" % anomaly)
    g0, _, (k, j, i) = out[False]
    g1, _, _ = out[True]
    # In anomaly space the floor is reached as (floor - ref) + ref: exact to an ulp of the
    # reference, not of the floor (1e-10 vs a reference of order 1). Compare on that scale.
    rref = dict(out_ref.named_buffers())["hydro.ref.u0"]
    for row, cell, r in ((kIDN, (k, j, i), 0), (kIPR, (k, j, i + 1), 1)):
        a, b = g1[(row,) + cell].item(), g0[(row,) + cell].item()
        scale = rref[r].abs().max().item()
        print("limiter    row %d: anomaly-space %.17g vs totals %.17g (ulp scale %.1e)"
              % (row, a, b, 2.2e-16 * scale))
        if not a > 0.0:
            failures.append("limiter: row %d floored to a non-positive total %g" % (row, a))
        if abs(a - b) > 4 * 2.2e-16 * scale:
            failures.append("limiter: row %d floored to %.17g in anomaly space, %.17g in "
                            "totals" % (row, a, b))
        if row == kIDN and not b > 0:
            failures.append("limiter: the density floor did not act (rho_d = %g)" % b)
    return failures


def arm_fp32(device):
    failures = []
    for rows, expect in (("all", "ALL_ROUNDED"), ("IPR", "IPR_ONLY")):
        env = dict(os.environ, SNAP_FP32_STORAGE="step", SNAP_FP32_STORAGE_ROWS=rows)
        env.pop("SNAP_ANOMALY", None)
        res = subprocess.run([sys.executable, __file__, "--child", "fp32", "--device", device],
                             env=env, capture_output=True, text=True)
        last = res.stdout.strip().splitlines()[-1:] if res.stdout.strip() else []
        print("fp32       rows=%s child: %s" % (rows, last))
        if res.returncode != 0 or expect not in res.stdout:
            failures.append("fp32 storage rows=%s: expected %s\n%s%s"
                            % (rows, expect, res.stdout, res.stderr))
    return failures


def child(kind, device):
    from snapy import kIDN, kIPR

    if kind == "env":
        block = make_block(device)
        block.initialize(initial_state(block))
        print("ANOMALY=%s" % block.anomaly())
        return 0
    if kind == "fp32":
        block = make_block(device)
        bv, _ = block.initialize(initial_state(block))
        run_steps(block, bv, 1)
        u = bv["hydro_u"]
        exact = (u.float().double() == u)
        rows_exact = [bool(exact[n].all()) for n in range(u.size(0))]
        print("rows exactly float32: %s" % rows_exact)
        if all(rows_exact):
            print("ALL_ROUNDED")
        elif rows_exact[kIPR] and not rows_exact[kIDN]:
            print("IPR_ONLY")
        return 0
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--child", default=None)
    args = p.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        return 125
    torch.set_default_dtype(torch.float64)
    # a restart loads onto the layout's device, which snapy reads from DEVICE
    # (default cpu): keep it in step with --device, for the children too
    os.environ.setdefault("DEVICE", args.device)
    if args.child:
        return child(args.child, args.device)

    failures = []
    failures += arm_reference(args.device)
    failures += arm_match(args.device, args.steps)
    failures += arm_match(args.device, args.steps, conversion_order=4)
    failures += arm_keys(args.device)
    failures += arm_switch(args.device)
    failures += arm_restart(args.device)
    failures += arm_float_restart(args.device)
    failures += arm_limiter(args.device)
    failures += arm_fp32(args.device)
    for f in failures:
        print("FAIL:", f)
    print("PASS" if not failures else "FAILED (%d)" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
