"""G-EF-4 (EF.rst rev 3.8, section 5): the shared PositivityLimiter equals
the scalar path it replaces. Runs scalar decks on two builds (PYTHONPATH picks
the package) and compares scalar_s and hydro_u after every step, byte for
byte, plus the scalar positivity census.

  python gef4.py --d349 DUMPSET349_DIR --device cpu|cuda --out DIR
  python gef4.py --compare DIR_A DIR_B

The decks are #349's moist settling column with one scalar (dumpset349.py
deck(), initial_w(), fixed_dt() and its scalar ratio, at b5976290), with the
scalar's limiter paths switched on in turn: the eos limiter, the upper bound
(P4, binding), the covariance rows alone (limiter off), an empty hole in the
ratio, and the ss reconstruction.
"""
import argparse, json, math, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
NSTEP = 6
# arm: (#349 case, eos limiter, upper bound or None, flux-covariance)
ARMS = {
    "lim": ("moist_weno5s_lim_sed_scalar", True, None, True),
    "lim_nocov": ("moist_weno5s_lim_sed_scalar", True, None, False),
    "lim_upper": ("moist_weno5s_lim_sed_scalar", True, 0.0012, True),
    "lim_upper_nocov": ("moist_weno5s_lim_sed_scalar", True, 0.0012, False),
    "cov_only": ("moist_weno5s_lim_sed_scalar", False, None, True),
    "hole": ("moist_weno5s_lim_sed_scalarhole", True, None, True),
    "weno5ss": ("moist_weno5ss_lim_sed_scalar", True, None, True),
}


def child(arm, out, d349, device):
    import torch, yaml, snapy
    from snapy import MeshBlock, MeshBlockOptions
    sys.path.insert(0, d349)
    import dumpset349 as d
    torch.set_default_dtype(torch.float64)
    case, lim, upper, cov = ARMS[arm]
    cfg = d.deck(case)
    cfg["dynamics"]["equation-of-state"]["limiter"] = lim
    cfg["dynamics"]["flux-covariance"] = cov
    if upper is not None:
        cfg["scalar"]["upper-bound"] = upper
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=out) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device))
    w = d.initial_w(case, block, cfg, torch, snapy)
    nc = w.shape[1:]
    Zi = (torch.arange(nc[2], dtype=torch.float64) / nc[2]).view(1, 1, -1).expand(*nc)
    Xi = (torch.arange(nc[1], dtype=torch.float64) / nc[1]).view(1, -1, 1).expand(*nc)
    r = 0.001 * (1. + 0.5 * torch.sin(2 * math.pi * Xi)) * torch.exp(-Zi)
    if case.endswith("scalarhole"):
        hole = ((Zi - 0.5) ** 2 + (Xi - 0.5) ** 2) < 0.15 ** 2
        r = torch.where(hole, torch.zeros_like(r), r)
    v, _ = block.initialize({"hydro_w": w.to(device),
                             "scalar_r": r.unsqueeze(0).clone().to(device)})
    dt = d.fixed_dt(case, cfg)
    stages = len(block.module("intg").stages)
    meta = {"arm": arm, "dt": dt, "steps": []}
    for n in range(NSTEP):
        for st in range(stages):
            block.forward(v, dt, st)
        for key in ("scalar_s", "hydro_u"):
            a = v[key].detach().cpu().contiguous()
            a.numpy().tofile(os.path.join(out, f"{arm}.{n}.{key}.bin"))
        bufs = dict(block.named_buffers())
        meta["steps"].append({"shape": {k: list(v[k].shape) for k in ("scalar_s", "hydro_u")},
                              "census": {k: bufs[k].detach().cpu().double().flatten().tolist()
                                         for k in bufs if "positivity" in k or "lim_" in k}})
    json.dump(meta, open(os.path.join(out, f"{arm}.json"), "w"), indent=1)


def compare(a, b):
    import numpy as np
    bad = 0
    for arm in ARMS:
        if not all(os.path.exists(os.path.join(x, f"{arm}.json")) for x in (a, b)):
            print(f"{arm:16s} MISSING on one side")
            bad += 1
            continue
        ja, jb = (json.load(open(os.path.join(x, f"{arm}.json"))) for x in (a, b))
        same_all, worst = True, {}
        for n, step in enumerate(ja["steps"]):
            for key, shape in step["shape"].items():
                x = np.fromfile(os.path.join(a, f"{arm}.{n}.{key}.bin")).reshape(shape)
                y = np.fromfile(os.path.join(b, f"{arm}.{n}.{key}.bin")).reshape(shape)
                if x.tobytes() != y.tobytes():
                    same_all = False
                    s = np.maximum(np.abs(x), np.abs(y))
                    worst[key] = max(worst.get(key, 0.), float((np.abs(x - y) / np.where(s > 0, s, 1)).max()))
            if step["census"] != jb["steps"][n]["census"]:
                same_all = False
                worst["census"] = n
        sc = ja["steps"][-1]["census"].get("scalar.positivity_hits")
        print(f"{arm:16s} {'identical' if same_all else 'DIFFERS ' + str(worst)} "
              f"steps={len(ja['steps'])} scalar hits after the last step={sc}")
        bad += not same_all
    print(f"identical {len(ARMS) - bad}/{len(ARMS)}")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--child")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--d349", help="directory holding dumpset349.py (b5976290 tools/dumpset349)")
    ap.add_argument("--device", default="cpu")
    o = ap.parse_args()
    if o.compare:
        sys.exit(1 if compare(*o.compare) else 0)
    os.makedirs(o.out, exist_ok=True)
    if o.child:
        child(o.child, o.out, o.d349, o.device)
        sys.exit(0)
    rc = 0
    for arm in ARMS:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", arm, "--out", o.out,
                            "--d349", o.d349, "--device", o.device])
        print(arm, "rc", p.returncode)
        rc |= p.returncode
    sys.exit(rc)
