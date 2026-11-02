# Deploy Your Policy

To deploy and evaluate your own policy in ViTaForge, you need to create three files under `policy/YourPolicy/`:

**1. `deploy_policy.py`** — Implements the policy interface. The following components must be defined:

```python
# policy/YourPolicy/deploy_policy.py
import torch
from policy._base_policy import BasePolicy

class Policy(BasePolicy):
    def __init__(self, args):
        """
        Load your model. `args` is a dict containing all fields from
        deploy.yml, plus runtime fields:
            - args['task_name']   : str — the current task name
            - args['task_config'] : str — the task config file stem
        """
        super().__init__(args)
        self.model = load_your_model(args)

    def encode_obs(self, observation):
        """
        Post-process raw observation into your model's input format.
        """
        return your_processed_obs

    def eval(self, task, observation):
        """
        Run one inference step. Called in a loop until the task
        reaches step_lim or reports success.
        """
        obs = self.encode_obs(observation)
        action = self.model.get_action(obs).reshape(-1)
        action = torch.from_numpy(action).to(task.device).float()
        exec_succ, eval_succ = task.take_action(action, action_type='qpos')
        # action_type options:
        #   'qpos'     — joint positions: Tensor([8])  (7 arm + 1 gripper)
        #   'ee'       — end-effector pose: Tensor([8]) (position(3) + quaternion(4) + gripper)
        #   'delta_ee' — delta end-effector: Tensor([7]) (delta_position(3) + delta_rotation(3) + delta_gripper)

    def reset(self):
        """Reset internal state (e.g., temporal buffers) at the start of each episode."""
        if hasattr(self.model, 'reset'):
            self.model.reset()
```

**2. `deploy.yml`** — Specifies deployment parameters, which are used to configuate the deployment and passed to `Policy.__init__()` as `args`. Only the `policy_name` field is mandatory and it must match the directory under `policy/`; you can add any other custom fields needed by your policy.

**3. Run evaluation:**

For `play_piano`, `blind_pick`, `tension_strap`, and `bulb_tightening`, the
policy decides when to stop. Joint commands keep executing after physical
success; holding a stationary command is not an END signal. In your policy's
`eval()` method, submit its predicted END explicitly:

```python
if predicted_stop:  # Your policy's decision, not task.check_success().
    success = task.finish_episode()
    return
task.take_action(action, action_type='qpos')
```

`finish_episode()` evaluates the current state, freezes the verdict, sets the
terminal phase and `eval_success`, and returns a boolean. Repeated END calls
return the same result. Failed END submissions are exposed by
`check_early_stop()`, so the evaluation loop exits on either verdict. The method
is valid only after the POLICY handoff. After END, `take_action()` returns
`(False, verdict)` without moving; `env_step()` rejects further steps.

An external CapX runner that already owns episode termination can instead stop
sending actions and then call `check_success()` for the physical verdict.
`check_success()` is a read-only query: it does not submit END or latch success.
The repository's evaluation loop requires `finish_episode()` to recognize END;
reaching its action limit without END counts as failure. Existing damage and
timeout stops remain in effect. Other tasks retain their previous termination
behavior. The joint action shape and 120 Hz physics / 60 Hz policy cadence are
unchanged.

```bash
bash eval_policy.sh ${task_name} ${task_config} ${policy_name}/${policy_config_name} ${gpu_id}
# Example: bash eval_policy.sh lift_bottle demo YourPolicy/deploy 0
```

or 

```bash
bash parallel_eval.sh ${task_name} ${task_config} ${policy_config} ${gpu_id} [num_processes] [total_num]
# Example: bash parallel_eval.sh lift_bottle demo YourPolicy/deploy 3 0,1,2
```
