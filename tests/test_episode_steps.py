import json

import h5py

from scripts.summarize_episode_steps import summarize


def test_failed_episodes_are_counted_but_not_averaged(tmp_path):
    (tmp_path/"metadata.json").write_text(json.dumps({
        "0":{"result":"success","cost_step":300},
        "1":{"result":"fail","cost_step":1000},
        "2":{"result":"success","cost_step":400}}))
    (tmp_path/"hdf5").mkdir()
    with h5py.File(tmp_path/"hdf5/0.hdf5","w") as h5:
        phase = h5.create_group("phase")
        phase.attrs.update(policy_saved_frames=100,save_frequency=2,policy_start_sim_step=100)
    # Failed reruns do not replace the old success-only HDF5 artifact.
    with h5py.File(tmp_path/"hdf5/1.hdf5","w") as h5:
        phase = h5.create_group("phase")
        phase.attrs.update(policy_saved_frames=999,save_frequency=2,policy_start_sim_step=2000)
    row = summarize(tmp_path)[0]
    assert row["attempts"] == 3 and row["successes"] == 2
    assert row["physics_steps_mean"] == 350
    assert row["policy_physics_steps_mean"] == 200
    assert row["policy_physics_steps_n"] == 1
    assert row["saved_policy_frames_mean"] == 100


def test_terminal_review_is_separate_from_task_execution(tmp_path):
    (tmp_path/"metadata.json").write_text(json.dumps({
        "0":{"result":"success","cost_step":750,"bulb_physical_verdict_step":300},
        "1":{"result":"success","cost_step":400}}))
    (tmp_path/"hdf5").mkdir()
    for seed in (0,1):
        with h5py.File(tmp_path/f"hdf5/{seed}.hdf5","w") as h5:
            phase = h5.create_group("phase")
            phase.attrs.update(policy_saved_frames=100,save_frequency=2,
                               policy_start_sim_step=100,terminal_id=2)
            phase.create_dataset("id",data=[0,1,1,2,2])
            phase.create_dataset("sim_step",data=[98,100,298,302,400])
    row = summarize(tmp_path)[0]
    # Exact verdict wins over the next saved frame (300 vs 302).
    assert row["preparation_physics_steps_mean"] == 100
    assert row["policy_physics_steps_mean"] == 201
    assert row["terminal_physics_steps_mean"] == 274
    assert row["physics_steps_mean"] == 575
