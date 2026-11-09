"""Fit and freeze the piano gel-force model served by cap-x.

Reads the force map collected by ``piano_force_map_probe.py`` (one labelled
depth image per 0.1 mm palm step, seven white keys), fits the same estimator
the other group uses -- five image statistics plus the pad's depth gradient,
median/IQR standardised, ridge on the linear terms and their squares -- and
writes the frozen model, the press recipe and the validation numbers to
``cap-x/capx/serving/piano_gel_calibration.json``.

The features are reproduced at runtime by the server adapter
(``univtac_piano_adapter.PianoTaskAdapter``), so the definition here is the
contract: the mask, the plane fit and the ratio's two-sample window must match
it exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import analyze_force_map as afm  # noqa: E402

KEEP = [0, 1, 2, 3, 4, 5]
FEATURE_NAMES = [
    "mean_mm",
    "area_gt_0p01",
    "area_gt_0p05",
    "area_gt_0p20",
    "area_rate",
    "gradient_mm_per_px",
]


def load_rows(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())["keys"]
    rows = []
    for midi, entry in manifest.items():
        key_dir = root / f"key{midi}"
        reference = np.load(key_dir / entry["reference"])
        mask = reference < 31.0
        tilt_ref = afm.plane_tilt(reference, mask)
        seq = [(s, afm.features(np.load(key_dir / f"{s['image']}.npy"),
                                reference, mask, tilt_ref, 31.0))
               for s in entry["samples"]]
        for index, (sample, feat) in enumerate(seq):
            feat = feat.copy()
            feat[4] = (feat[2] - seq[index - 2][1][2]) if index >= 2 else 0.0
            rows.append({"key": int(midi), "force": sample["force_N"],
                         "z": sample["z_mm"], "feat": feat})
    return rows


def standardise(x: np.ndarray):
    median = np.median(x, axis=0)
    iqr = np.maximum(np.percentile(x, 75, axis=0)
                     - np.percentile(x, 25, axis=0), 1e-12)
    return median, iqr


def design(x: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    z = (x - median) / iqr
    return np.concatenate([np.ones((len(z), 1)), z, z ** 2], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=REPO / ".cache" / "piano_tactile" / "forcemap")
    # cap-x lives beside this repo by default (the same sibling convention
    # launch_univtac_server.sh uses for UniVTAC), so no absolute path is baked
    # into the checkout.
    parser.add_argument("--output", type=Path,
                        default=REPO.parent / "cap-x" / "capx" / "serving"
                        / "piano_gel_calibration.json")
    parser.add_argument("--alpha", type=float, default=0.01)
    args = parser.parse_args()

    rows = load_rows(args.root)
    x = np.stack([r["feat"] for r in rows])[:, KEEP]
    y = np.array([r["force"] for r in rows])
    keys = np.array([r["key"] for r in rows])
    print(f"{len(rows)} frames from keys {sorted(set(keys))}, "
          f"force [{y.min():.3f}, {y.max():.3f}] N")

    # Leave-one-key-out: the only independent estimate this data can give.
    prediction = np.full_like(y, np.nan)
    for key in sorted(set(keys)):
        train, test = keys != key, keys == key
        median, iqr = standardise(x[train])
        weight = afm.fit_ridge(design(x[train], median, iqr), y[train],
                               args.alpha)
        prediction[test] = design(x[test], median, iqr) @ weight
    contact = y > 0.05
    error = prediction[contact] - y[contact]
    loo_mae = float(np.abs(error).mean())
    loo_p95 = float(np.percentile(np.abs(error), 95))
    print(f"leave-one-key-out: MAE {loo_mae:.4f} N, P95 {loo_p95:.4f} N")

    bands = {"light": (0.28, 0.42), "heavy": (0.43, 0.57)}
    in_band = {}
    for name, (lo, hi) in bands.items():
        mask = contact & (y >= lo) & (y <= hi)
        inside = ((prediction[mask] >= lo) & (prediction[mask] <= hi)).mean()
        in_band[name] = float(inside)
        print(f"  {name}: n={int(mask.sum())} "
              f"MAE {np.abs(prediction[mask] - y[mask]).mean():.4f} "
              f"predicted-in-band {inside:.2f}")

    median, iqr = standardise(x)
    weight = afm.fit_ridge(design(x, median, iqr), y, args.alpha)
    residual = design(x, median, iqr) @ weight - y
    print(f"fitted on all frames, train MAE {np.abs(residual).mean():.4f} N")

    payload = {
        "format": "piano-gel-force/2",
        "task": "arm_shadow_piano",
        "tactile": "gsmini",
        "image_width": 320,
        "features": FEATURE_NAMES,
        "valid_depth_mm": 31.0,
        "model": {
            "median": [float(v) for v in median],
            "iqr": [float(v) for v in iqr],
            "weight": [float(v) for v in weight],
            "alpha": float(args.alpha),
            "order": "1, linear, square",
            # Feature envelope of the training set.  A live reading outside it
            # is flagged instead of being mapped to a confident-looking force:
            # a different press pose or a lost reference leaves the model
            # extrapolating, and nothing else in the payload would say so.
            "range_min": [float(v) for v in x.min(axis=0)],
            "range_max": [float(v) for v in x.max(axis=0)],
            "range_margin": 0.05,
        },
        "bands": {
            "light": {"band_N": list(bands["light"]), "target_N": 0.35},
            "heavy": {"band_N": list(bands["heavy"]), "target_N": 0.50},
        },
        "saturation_N": 0.43,
        "plateau": {"fraction": 0.30, "min_samples": 6, "window": 3},
        "press": {
            "raise_mm": 2.0,
            "step_mm": 0.1,
            "settle_steps": 3,
            "settle_at_pose_steps": 10,
            "ramp_levels": 21,
            "ramp_settle": 2,
            "press_angle_rad": 0.50,
            "max_steps": 60,
        },
        "hold_seconds": 0.12,
        "validation": {
            "frames": int(contact.sum()),
            "keys": sorted(int(k) for k in set(keys)),
            "protocol": "leave one white key out, fit on the other six",
            "loo_mae_N": loo_mae,
            "loo_p95_N": loo_p95,
            "predicted_in_band": in_band,
        },
        "closed_loop": {
            "protocol": "fresh episode per press, stop on the live estimate, "
                        "PhysX force read once at the stop",
            "light": {"stop_z_mm": 0.10, "actual_N": [0.352, 0.374],
                      "in_band": "7/7"},
            "heavy": {"stop_z_mm": -0.80, "actual_N": [0.438, 0.459],
                      "in_band": "7/7"},
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
