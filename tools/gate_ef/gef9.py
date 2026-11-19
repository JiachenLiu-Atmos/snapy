"""G-EF-9 (EF.rst rev 3.8, section 5): the single sedimentation kernel against
both old paths (ideal-moist fused kernel, moist-mixture tensor Stokes path).
Runs each deck on two builds (PYTHONPATH picks the package): the register
totals F1 after the first stage (rows: IDN, momentum, IPR, species) and
hydro_u after each of NSTEP steps; reports bitwise counts and the maximum
relative difference.

  python gef9.py --tree SNAPY --device cpu|cuda --out DIR
  python gef9.py --compare DIR_A DIR_B [--tol 1e-12]

Decks: tests/test_gravity_sedimentation.yaml and test_stokes_sedimentation.yaml
from --tree (ideal-moist, const-gravity added), and G-1's uranus
(moist-mixture; also with Stokes settling, 100 um drops), earth_crm and
jupiter_evap_precip_1d decks with their initial restarts from data/.
"""
import argparse, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# G-1's decks and initial restarts, written by a G-1 run of the base build
DATA = os.path.join(HERE, "data")
NSTEP = 3
# deck: (yaml, initial restart or None, grav1 to add or None); "tests/" decks
# are read from --tree
DECKS = {
    "gravity_sed_im": ("tests/test_gravity_sedimentation.yaml", None, -1.),
    "stokes_sed_im": ("tests/test_stokes_sedimentation.yaml", None, -10.),
    "uranus_mm": ("uranus.yaml", "uranus.00000.restart", None),
    # uranus with Stokes settling (const-vsed 0): the old tensor Stokes path
    "uranus_mm_stokes": ("uranus.yaml", "uranus.00000.restart", None),
    "earth_crm_im": ("earth_crm.yaml", "earth_crm.00000.restart", None),
    "jupiter_1d_im": ("jupiter_evap_precip_1d.yaml",
                      "jupiter_evap_precip_1d.00000.restart", None),
}


def deck_path(path, tree):
    return os.path.join(tree, path) if path.startswith("tests/") else os.path.join(DATA, path)


def child(name, out, tree, device):
    import torch, yaml, snapy
    from snapy import MeshBlock, MeshBlockOptions
    torch.set_default_dtype(torch.float64)
    deck, restart, grav = DECKS[name]
    card = yaml.safe_load(open(deck_path(deck, tree)))
    if name.endswith("_stokes"):
        for k in card["sedimentation"]["const-vsed"]:  # Stokes, 100 um drops
            card["sedimentation"]["const-vsed"][k] = 0.
            card["sedimentation"]["radius"][k] = 1.e-4
    if grav is not None:
        card.setdefault("forcing", {})["const-gravity"] = {"grav1": grav}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=out) as f:
        yaml.safe_dump(card, f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device))
    if restart:
        w = snapy.load_restart(os.path.join(DATA, restart), 0)["hydro_w"]
        if name.startswith("uranus"):  # the initial state holds no particles
            for i in dict(block.named_buffers())["hydro.sed.hydro_ids"].tolist():
                w[i] = 1.e-5
    else:
        shape = dict(block.named_buffers())["hydro.F1"].shape
        w = torch.zeros(shape, dtype=torch.float64)
        w[snapy.kIDN] = 1.
        w[snapy.kIPR] = 1.e5
        w[snapy.kICY] = 0.05
        w[snapy.kICY + 1] = 0.02
    v, _ = block.initialize({"hydro_w": w.to(device)})
    # "W->E" on the initial state (EF-7: the base species_energy)
    e = block.module("hydro.eos").compute("W->E", [v["hydro_w"]]).detach().cpu().contiguous()
    os.makedirs(out, exist_ok=True)
    e.numpy().tofile(os.path.join(out, f"{name}.E.bin"))
    dt = block.max_time_step(v)
    stages = len(block.module("intg").stages)
    meta = {"deck": name, "dt": dt, "arrays": {"E": list(e.shape)}}
    for n in range(NSTEP):
        for st in range(stages):
            block.forward(v, dt, st)
            if n == 0 and st == 0:
                f1 = dict(block.named_buffers())["hydro.F1"].detach().cpu().contiguous()
                f1.numpy().tofile(os.path.join(out, f"{name}.F1.bin"))
                meta["arrays"]["F1"] = list(f1.shape)
        u = v["hydro_u"].detach().cpu().contiguous()
        u.numpy().tofile(os.path.join(out, f"{name}.u{n}.bin"))
        meta["arrays"][f"u{n}"] = list(u.shape)
    json.dump(meta, open(os.path.join(out, f"{name}.json"), "w"), indent=1)


def compare(a, b, tol):
    import numpy as np
    worst = 0.
    for name in DECKS:
        if not all(os.path.exists(os.path.join(x, f"{name}.json")) for x in (a, b)):
            print(f"{name:16s} MISSING")
            worst = float("inf")
            continue
        ja = json.load(open(os.path.join(a, f"{name}.json")))
        for key, shape in ja["arrays"].items():
            x = np.fromfile(os.path.join(a, f"{name}.{key}.bin")).reshape(shape)
            y = np.fromfile(os.path.join(b, f"{name}.{key}.bin")).reshape(shape)
            groups = ({"W->E": list(range(shape[0]))} if key == "E" else
                      {"IDN": [0], "IVX..IVZ": [1, 2, 3], "IPR": [4],
                       "species": list(range(5, shape[0]))})
            parts = []
            for g, r in groups.items():
                if not r:
                    continue
                xa, ya = x[r], y[r]
                n = int((xa.view(np.uint64) != ya.view(np.uint64)).sum())
                s = np.maximum(np.abs(xa), np.abs(ya))
                rel = float((np.abs(xa - ya) / np.where(s > 0, s, 1)).max())
                if key in ("F1", "E"):
                    worst = max(worst, rel)
                parts.append(f"{g} {n}/{xa.size} {rel:.2e}")
            print(f"{name:16s} {key:3s} " + " | ".join(parts))
    print(f"worst stage-1 flux or W->E relative difference {worst:.3e}; tolerance {tol:.0e}: "
          f"{'PASS' if worst <= tol else 'FAIL'}")
    return worst <= tol


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--child")
    ap.add_argument("--tree", help="snapy checkout (reads tests/*.yaml)")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--tol", type=float, default=1e-12)
    o = ap.parse_args()
    if o.compare:
        sys.exit(0 if compare(*o.compare, o.tol) else 1)
    os.makedirs(o.out, exist_ok=True)
    if o.child:
        child(o.child, o.out, o.tree, o.device)
        sys.exit(0)
    rc = 0
    for name in DECKS:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", name,
                            "--out", o.out, "--tree", o.tree, "--device", o.device])
        print(name, "rc", p.returncode)
        rc |= p.returncode
    sys.exit(rc)
