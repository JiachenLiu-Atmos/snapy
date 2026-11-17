#!/usr/bin/env python3
"""Compare paired run_fp32_straka.py outputs and plot potential temperature."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fp64", type=Path)
    ap.add_argument("fp32", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    reports = [json.loads((p / "report.json").read_text()) for p in (args.fp64, args.fp32)]
    assert [r["precision"] for r in reports] == ["float64", "float32"]
    for key in ("dt", "steps", "rd", "cp", "native_libraries_sha256"):
        assert reports[0][key] == reports[1][key], key
    assert reports[0]["history"][-1]["time"] == reports[1]["history"][-1]["time"]
    pairs = []
    for entry in reports[0]["history"]:
        t = entry["time"]
        a, b = [np.load(p / f"state-{t:g}.npz") for p in (args.fp64, args.fp32)]
        assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["z"], b["z"])
        delta = b["theta"] - a["theta"]
        fields = {}
        for i, name in enumerate(
            (
                "density",
                "vertical_velocity",
                "horizontal_velocity",
                "transverse_velocity",
                "pressure",
            )
        ):
            e = b["primitive"][i] - a["primitive"][i]
            fields[name] = {
                "max_absolute_error": float(np.abs(e).max()),
                "rms_error": float(np.sqrt(np.mean(e * e))),
            }
        fronts = []
        for v in (a, b):
            cold = np.any(v["theta"][0][:, v["z"] <= 1000.0] < -1.0, axis=1)
            fronts.append(float(v["x"][cold].max()) if cold.any() else None)
        pairs.append(
            {
                "time": t,
                "theta_max_absolute_error_K": float(np.abs(delta).max()),
                "theta_rms_error_K": float(np.sqrt(np.mean(delta * delta))),
                "theta_relative_l2_error": float(
                    np.linalg.norm(delta) / np.linalg.norm(a["theta"])
                ),
                "cold_front_x_m_fp64_fp32": fronts,
                "primitive_errors": fields,
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "runs": reports,
        "comparisons": pairs,
        "note": "Matched HLLC dry-limit Straka; no scientific equivalence threshold asserted.",
    }
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True, sharex=True)
    extent = [0.0, 25.6, 0.0, 6.4]
    bound = max(float(np.abs(delta).max()), 1.0e-12)
    for ax, field, title, limits in zip(
        axes,
        (a["theta"], b["theta"], delta),
        ("FP64", "FP32", "FP32 − FP64"),
        ((-18.0, 2.0), (-18.0, 2.0), (-bound, bound)),
    ):
        im = ax.imshow(
            field[0].T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="RdBu_r",
            vmin=limits[0],
            vmax=limits[1],
        )
        ax.set(title=title, ylabel="Height (km)")
        fig.colorbar(
            im,
            ax=ax,
            label="Potential temperature anomaly (K)"
            if title != "FP32 − FP64"
            else "Difference (K)",
        )
    axes[-1].set_xlabel("Horizontal distance (km)")
    fig.suptitle(f"Straka HLLC dry-limit comparison at t = {pairs[-1]['time']:g} s")
    fig.savefig(args.output / "theta-comparison.png", dpi=180)
    print(json.dumps(pairs, indent=2))


if __name__ == "__main__":
    main()
