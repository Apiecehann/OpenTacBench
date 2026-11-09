"""Fit the five-feature tactile force estimator on a collected force map.

The feature set mirrors the DIGIT calibration used by the other group:
  d        = |D_ref - D_cur|    (mm) over valid pixels
  mean     = mean(d)
  area.01  = fraction of valid pixels with d > 0.01 mm
  area.05  = fraction with d > 0.05 mm
  area.20  = fraction with d > 0.2 mm
  rate.05  = area.05 difference over two samples (a constant time scale)
Plus the depth gradient across the pad (mm/column), which is what the image
shows most strongly.  Standardise each feature by median/IQR, then fit ridge on
the linear terms and the per-feature squares.  Report leave-one-key-out error,
a force-binned view of every feature, and the force reached by the heuristic
"descend until the depth gradient stops changing" rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NAMES = ["mean", "area01", "area05", "area20", "rate05", "tilt"]


def plane_tilt(image: np.ndarray, mask: np.ndarray) -> float:
    rows, cols = np.nonzero(mask)
    a = np.stack([np.ones(len(cols)), cols, rows], axis=1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(a, image[mask].astype(np.float64), rcond=None)
    return float(coef[1])


def features(image: np.ndarray, reference: np.ndarray, mask: np.ndarray,
             tilt_ref: float, valid_depth: float) -> np.ndarray:
    # The saved depth is already in millimetres (the sensor's height_map unit);
    # the reference recipe expresses D in metres and converts with *1000.
    # Pixels that leave the sensor silhouette (they hit the far plane) would
    # contribute a 6 mm jump, so the valid set is the reference mask that is
    # still on the sensor in this frame.
    mask = mask & (image < valid_depth)
    d = np.abs(reference - image)
    dv = d[mask]
    return np.array([
        dv.mean(),
        (dv > 0.01).mean(),
        (dv > 0.05).mean(),
        (dv > 0.20).mean(),
        np.nan,
        plane_tilt(image, mask) - tilt_ref,
    ])


def design(x: np.ndarray, keep) -> np.ndarray:
    z = (x[:, keep] - np.median(x[:, keep], axis=0)) / np.maximum(
        np.percentile(x[:, keep], 75, axis=0)
        - np.percentile(x[:, keep], 25, axis=0), 1e-12)
    return np.concatenate([np.ones((len(z), 1)), z, z ** 2], axis=1)


def fit_ridge(a: np.ndarray, y: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    penalty = np.eye(a.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(a.T @ a + penalty, a.T @ y)


def spearman(u: np.ndarray, v: np.ndarray) -> float:
    ru = np.argsort(np.argsort(u))
    rv = np.argsort(np.argsort(v))
    return float(np.corrcoef(ru, rv)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(".cache/piano_tactile/forcemap"))
    parser.add_argument("--valid-depth", type=float, default=31.0)
    parser.add_argument("--min-force", type=float, default=0.05)
    parser.add_argument("--plateau-fraction", type=float, default=0.30)
    args = parser.parse_args()

    manifest = json.loads((args.root / "manifest.json").read_text())["keys"]
    rows = []
    for midi, entry in manifest.items():
        key_dir = args.root / f"key{midi}"
        reference = np.load(key_dir / entry["reference"])
        mask = reference < args.valid_depth
        tilt_ref = plane_tilt(reference, mask)
        seq = []
        for sample in entry["samples"]:
            image = np.load(key_dir / f"{sample['image']}.npy")
            seq.append((sample, features(image, reference, mask, tilt_ref,
                                         args.valid_depth)))
        for index, (sample, feat) in enumerate(seq):
            feat = feat.copy()
            feat[4] = (feat[2] - seq[index - 2][1][2]) if index >= 2 else 0.0
            rows.append({"key": int(midi), "force": sample["force_N"],
                         "kang": sample["key_angle"], "z": sample["z_mm"],
                         "feat": feat})

    x = np.stack([r["feat"] for r in rows])
    y = np.array([r["force"] for r in rows])
    keys = np.array([r["key"] for r in rows])
    print(f"{len(rows)} samples from {len(set(keys))} keys, "
          f"force [{y.min():.3f}, {y.max():.3f}] N")

    print("\nfeature vs force correlation (Spearman, all keys pooled):")
    for i, name in enumerate(NAMES):
        print(f"  {name:7s} rho {spearman(x[:, i], y):+.3f}  "
              f"range [{x[:, i].min():+.5f}, {x[:, i].max():+.5f}]")

    print("\nforce-binned feature means (std):")
    edges = np.arange(0.10, 0.60, 0.05)
    for lo, hi in zip(edges, edges[1:]):
        mask = (y >= lo) & (y < hi)
        if mask.sum() < 3:
            continue
        cells = " ".join(f"{name} {x[mask, i].mean():+.4f}({x[mask, i].std():.4f})"
                         for i, name in enumerate(NAMES))
        print(f"  F [{lo:.2f},{hi:.2f}) n={mask.sum():3d}  {cells}")

    print("\nper-key depth-gradient curve (mm/column vs force):")
    for key in sorted(set(keys)):
        sel = keys == key
        order = np.argsort(y[sel])
        tilt = x[sel, 5][order]
        force = y[sel][order]
        print(f"  key {key}: " + " ".join(
            f"{force[i]:.2f}:{tilt[i]:+.5f}" for i in range(0, len(force), 6)))

    # The deployed rule: descend until the gradient stops changing.  The step
    # size is the sweep's 0.1 mm; the plateau threshold is a fraction of the
    # gradient's total change over the sweep.
    print("\n'descend until the gradient plateaus' rule "
          f"(step below {args.plateau_fraction:.0%} of the initial step):")
    plateau_forces = []
    for key in sorted(set(keys)):
        idx = np.nonzero(keys == key)[0]
        tilt = x[idx, 5]
        force = y[idx]
        z = np.array([rows[i]["z"] for i in idx])
        step = np.abs(np.diff(tilt))
        initial = np.median(step[:8])
        threshold = args.plateau_fraction * initial
        index = next((i for i in range(5, len(step) - 1)
                      if step[i] < threshold and step[i + 1] < threshold),
                     None)
        if index is None:
            print(f"  key {key}: no plateau; final force {force[-1]:.3f}")
            continue
        plateau_forces.append(force[index + 1])
        print(f"  key {key}: plateau at z {z[index + 1]:+.2f} mm "
              f"force {force[index + 1]:.3f} "
              f"(step {step[index + 1]:.6f} vs initial {initial:.6f})")
    if plateau_forces:
        print(f"  plateau force: mean {np.mean(plateau_forces):.3f} "
              f"range [{min(plateau_forces):.3f}, {max(plateau_forces):.3f}]")

    if len(set(keys)) < 2:
        print("\nonly one key: skipping leave-one-key-out (needs >= 2 keys)")
        return
    for label, keep in (("classmate 5", slice(0, 5)),
                        ("5 + tilt", slice(0, 6)),
                        ("tilt only", [5])):
        print(f"\nleave-one-key-out ridge [{label}]:")
        prediction = np.full_like(y, np.nan)
        for key in sorted(set(keys)):
            train, test = keys != key, keys == key
            med = np.median(x[train][:, keep], axis=0)
            iqr = np.maximum(np.percentile(x[train][:, keep], 75, axis=0)
                             - np.percentile(x[train][:, keep], 25, axis=0),
                             1e-12)
            z = (x[train][:, keep] - med) / iqr
            a = np.concatenate([np.ones((train.sum(), 1)), z, z ** 2], axis=1)
            w = fit_ridge(a, y[train])
            zt = (x[test][:, keep] - med) / iqr
            at = np.concatenate([np.ones((test.sum(), 1)), zt, zt ** 2], axis=1)
            prediction[test] = at @ w
            contact = test & (y > args.min_force)
            error = prediction[contact] - y[contact]
            print(f"  key {key}: n={contact.sum():3d} "
                  f"MAE {np.abs(error).mean():.4f} "
                  f"P95 {np.percentile(np.abs(error), 95):.4f}")

        contact = y > args.min_force
        error = prediction[contact] - y[contact]
        print(f"  all n={contact.sum()} MAE {np.abs(error).mean():.4f} "
              f"P95 {np.percentile(np.abs(error), 95):.4f} "
              f"max {np.abs(error).max():.4f}")
        for name, (lo, hi) in (("light", (0.28, 0.42)),
                               ("heavy", (0.43, 0.57))):
            mask = contact & (y >= lo) & (y <= hi)
            if not mask.any():
                continue
            err = prediction[mask] - y[mask]
            inside = ((prediction[mask] >= lo) & (prediction[mask] <= hi)).mean()
            print(f"  {name}: n={mask.sum():3d} bias {err.mean():+.4f} "
                  f"MAE {np.abs(err).mean():.4f} in-band {inside:.2f}")


if __name__ == "__main__":
    main()
