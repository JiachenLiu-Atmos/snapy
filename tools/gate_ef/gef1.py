"""G-EF-1 (EF.rst rev 3.8, section 5): the register posts against the in-place
fluxes after step 4.C. Runs the same arms on two builds (PYTHONPATH picks the
package) and compares the hydro register totals F1/F2/F3 row by row after the
first stage (same input state), plus the EfStatus meters.

  python gef1.py --tree SNAPY --d349 DUMPSET349_DIR --device cpu|cuda --out DIR
  python gef1.py --compare DIR_A DIR_B

Arms: the moist carry deck (tests/test_flux_positivity_carry.yaml) as its
C++ test sets it up (limiter on; advected, settling, mixed), the same deck
limiter off with settling (P1 alone), and #349's 2D moist settling deck
(moist_weno5s_lim_sed_scalar, dumpset349.py deck() and initial_w()).
"""
import argparse, json, math, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["carry_adv_lmars", "carry_adv_hllc", "carry_settle", "carry_mixed",
        "carry_settle_nolim", "moist2d_lim_sed", "moist2d_nolim_sed", "jupiter_1d"]
# G-1's jupiter_evap_precip_1d deck (three settling particles) from its
# initial restart, written by the base build (R-3 206e342f) in a G-1 run
JUP = os.path.join(HERE, "data")


def carry_card(arm, tree):
    import yaml
    card = yaml.safe_load(open(os.path.join(tree, "tests", "test_flux_positivity_carry.yaml")))
    card["dynamics"]["equation-of-state"]["limiter"] = "nolim" not in arm
    card["dynamics"]["flux-covariance"] = False
    if arm == "carry_adv_hllc":
        card["dynamics"]["riemann-solver"]["type"] = "hllc"
    if "settle" in arm:
        card["sedimentation"] = {"radius": {}, "density": {}, "const-vsed": {"cloud": -2.}}
    if arm == "carry_mixed":
        card["sedimentation"] = {"radius": {}, "density": {}, "const-vsed": {"cloud": -1.}}
    return card


def carry_w(arm, block, snapy, torch):
    hy = dict(block.named_buffers())
    nvar = hy["hydro.F1"].shape[0]
    shape = hy["hydro.F1"].shape[1:]
    w = torch.zeros((nvar,) + tuple(shape), dtype=torch.float64)
    il = 2  # nghost
    vx, vy, vz = {"carry_adv_lmars": (2., 3., 0.), "carry_adv_hllc": (2., 3., 0.),
                  "carry_settle": (0., 3., 0.), "carry_settle_nolim": (0., 3., 0.),
                  "carry_mixed": (0., 0., 0.)}[arm]
    w[snapy.kIDN] = 1.
    w[snapy.kIPR] = 1.e5
    w[snapy.kIV1] = vx
    w[snapy.kIV2] = vy
    w[snapy.kIV3] = vz
    w[snapy.kICY] = 0.01
    w[snapy.kICY + 1] = 0.02
    for c, f in enumerate([1., 0.1, 0.01, 0.01, 0.1, 1.]):
        w[snapy.kICY:snapy.kICY + 2, ..., il + c] *= f
    if arm == "carry_mixed":
        rho = [1.00, 1.15, 0.90, 1.05, 0.85, 1.20]
        v1 = [3.0, 3.4, 2.6, 3.2, 2.8, 3.1]
        v2 = [3.0, 3.5, 2.5, 4.0, 2.0, 3.2]
        v3 = [0.0, 0.5, -0.5, 1.0, -1.0, 0.3]
        for c in range(6):
            w[snapy.kIDN, ..., il + c] = rho[c]
            w[snapy.kIV1, ..., il + c] = v1[c]
            w[snapy.kIV2, ..., il + c] = v2[c]
            w[snapy.kIV3, ..., il + c] = v3[c]
    return w


def child(arm, out, tree, d349, device):
    import torch, yaml, snapy
    from snapy import MeshBlock, MeshBlockOptions
    torch.set_default_dtype(torch.float64)
    if arm == "jupiter_1d":
        card = yaml.safe_load(open(os.path.join(JUP, "jupiter_evap_precip_1d.yaml")))
    elif arm.startswith("moist2d"):
        sys.path.insert(0, d349)
        import dumpset349 as d
        name = "moist_weno5s_lim_sed_scalar" if "_lim_" in arm else "moist_weno5s_nolim_sed"
        card = d.deck(name)
        card.pop("scalar", None)
    else:
        card = carry_card(arm, tree)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=out) as f:
        yaml.safe_dump(card, f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device))
    if arm == "jupiter_1d":
        w = snapy.load_restart(os.path.join(JUP, "jupiter_evap_precip_1d.00000.restart"), 0)["hydro_w"]
        dt = None
    elif arm.startswith("moist2d"):
        w = d.initial_w(name, block, card, torch, snapy)
        dt = d.fixed_dt(name, card)
    else:
        w = carry_w(arm, block, snapy, torch)
        dt = 1.
    v, _ = block.initialize({"hydro_w": w.to(device)})
    if dt is None:  # the driver's own step
        dt = block.max_time_step(v)
    block.forward(v, dt, 0)
    bufs = dict(block.named_buffers())
    meta = {"arm": arm, "dt": dt, "arrays": {}, "meters": {}}
    for k in ("hydro.F1", "hydro.F2", "hydro.F3"):
        if k in bufs and bufs[k].numel() > 0:
            a = bufs[k].detach().cpu().contiguous()
            a.numpy().tofile(os.path.join(out, f"{arm}.{k}.bin"))
            meta["arrays"][k] = list(a.shape)
    for k, b in bufs.items():
        if ("positivity" in k or "lim_" in k or "pos_" in k) and b.numel() < 64:
            meta["meters"][k] = b.detach().cpu().double().flatten().tolist()
    json.dump(meta, open(os.path.join(out, f"{arm}.json"), "w"), indent=1)


def compare(a, b):
    import numpy as np
    rows = {"IDN": [0], "IVX..IVZ": [1, 2, 3], "IPR": [4]}
    bad = 0
    for arm in ARMS:
        if not all(os.path.exists(os.path.join(d, f"{arm}.json")) for d in (a, b)):
            print(f"{arm:20s} MISSING on one side")
            bad += 1
            continue
        ja, jb = (json.load(open(os.path.join(d, f"{arm}.json"))) for d in (a, b))
        for k, shape in ja["arrays"].items():
            x = np.fromfile(os.path.join(a, f"{arm}.{k}.bin")).reshape(shape)
            y = np.fromfile(os.path.join(b, f"{arm}.{k}.bin")).reshape(shape)
            groups = dict(rows, species=list(range(5, shape[0])))
            for g, r in groups.items():
                if not r:
                    continue
                xa, ya = x[r], y[r]
                same = xa.tobytes() == ya.tobytes()
                diff = np.abs(xa - ya)
                scale = np.maximum(np.abs(xa), np.abs(ya))
                rel = float((diff / np.where(scale > 0, scale, 1)).max())
                n = int((xa.view(np.uint64) != ya.view(np.uint64)).sum())
                print(f"{arm:20s} {k:9s} {g:9s} {'bitwise' if same else 'DIFF'} "
                      f"entries={n}/{xa.size} max_abs={float(diff.max()):.3e} max_rel={rel:.3e}")
                bad += not same
        ma, mb = ja["meters"], jb["meters"]
        for k in sorted(set(ma) | set(mb)):
            same = ma.get(k) == mb.get(k)
            print(f"{arm:20s} meter {k}: {'equal' if same else 'DIFF'} {ma.get(k)} {'' if same else mb.get(k)}")
            bad += not same
    print(f"RESULT {bad} row groups or meters differ")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--child")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--tree", help="snapy checkout (tests/test_flux_positivity_carry.yaml)")
    ap.add_argument("--d349", help="directory holding dumpset349.py (b5976290 tools/dumpset349)")
    ap.add_argument("--device", default="cpu")
    o = ap.parse_args()
    if o.compare:
        sys.exit(1 if compare(*o.compare) else 0)
    os.makedirs(o.out, exist_ok=True)
    if o.child:
        child(o.child, o.out, o.tree, o.d349, o.device)
        sys.exit(0)
    rc = 0
    for arm in ARMS:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", arm, "--out", o.out,
                            "--tree", o.tree, "--d349", o.d349, "--device", o.device])
        print(arm, "rc", p.returncode)
        rc |= p.returncode
    sys.exit(rc)
