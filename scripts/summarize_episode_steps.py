"""Summarize real collected episodes, keeping success and timing units explicit."""
import argparse
import json
from pathlib import Path
import statistics

import h5py
import numpy as np


def summarize(root):
    rows = []
    for path in sorted(Path(root).rglob("metadata.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        episodes = []
        for seed, item in metadata.items():
            if not isinstance(item, dict) or "result" not in item:
                continue
            episode = {"seed": seed, "result": item["result"], "physics_steps": item.get("cost_step")}
            h5_path = path.parent / "hdf5" / f"{seed}.hdf5"
            # The legacy collector writes HDF5 only on success. A failed rerun
            # can leave an older successful file at the same seed/path.
            if item['result'] == 'success' and h5_path.exists():
                with h5py.File(h5_path, "r") as h5:
                    if "step" in h5:
                        episode["saved_total_frames"] = len(h5["step"])
                    phase = h5.get("phase")
                    if phase is not None:
                        episode["saved_policy_frames"] = int(phase.attrs["policy_saved_frames"])
                        episode["save_frequency"] = int(phase.attrs["save_frequency"])
                        if episode["physics_steps"] is not None:
                            start = int(phase.attrs["policy_start_sim_step"])
                            end = episode["physics_steps"]
                            # Force tasks record the exact physical verdict before
                            # opening/lifting the hand for the review video.
                            verdicts = [int(v) for k, v in item.items()
                                        if k.endswith("_physical_verdict_step")]
                            if verdicts:
                                end = min(verdicts)
                            elif "id" in phase and "sim_step" in phase:
                                terminal = np.asarray(phase["id"]) == int(phase.attrs.get("terminal_id", 2))
                                if terminal.any():
                                    end = int(np.asarray(phase["sim_step"])[terminal].min())
                            if not start <= end <= episode["physics_steps"]:
                                raise ValueError(f"Invalid phase boundaries in {h5_path}")
                            episode["preparation_physics_steps"] = start
                            episode["policy_physics_steps"] = end-start
                            episode["terminal_physics_steps"] = episode["physics_steps"]-end
            episodes.append(episode)
        if not episodes:
            continue
        success = [e for e in episodes if e["result"] == "success"]
        row = {"run": str(path.parent), "attempts": len(episodes), "successes": len(success)}
        for key in ("physics_steps", "preparation_physics_steps", "policy_physics_steps",
                    "terminal_physics_steps", "saved_policy_frames", "saved_total_frames"):
            values = [e[key] for e in success if e.get(key) is not None]
            row[key+"_n"] = len(values)
            row[key+"_mean"] = statistics.mean(values) if values else None
            row[key+"_min"] = min(values) if values else None
            row[key+"_max"] = max(values) if values else None
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    result = [row for root in args.roots for row in summarize(root)]
    print(json.dumps({"units": {
        "physics_steps": "all executed simulation ticks including preparation and terminal review",
        "preparation_physics_steps": "executed simulation ticks before the policy handoff",
        "policy_physics_steps": "executed ticks from policy handoff to verdict/terminal boundary",
        "terminal_physics_steps": "executed ticks after the verdict, excluded from policy actions",
        "saved_policy_frames": "recorded policy observations; not action calls or step_lim",
        "saved_total_frames": "all rows actually recorded in HDF5, including preparation and terminal observations",
        "means": "successful episodes only; attempts and per-field sample counts are included"
    }, "runs":result}, indent=2))


if __name__ == "__main__":
    main()
