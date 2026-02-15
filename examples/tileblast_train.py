import os
import sys

# Ensure we use the local repo, not site-packages
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pufferlib.ocean
from pufferlib import pufferl


def main():
    env_name = "puffer_tileblast"
    config_path = os.path.join(REPO_ROOT, "pufferlib", "config", "ocean", "tileblast.ini")
    args = pufferl.load_config_file(config_path, fill_in_default=True)

    # Disable rendering during training
    args["env"]["render_mode"] = None
    args["env"]["num_agents"] = 1
    args["env"]["num_envs"] = 1
    args["env"]["vision"] = 0
    args["env"]["max_steps"] = 500
    args["train"]["total_timesteps"] = 5000000
    args["train"]["device"] = "cuda"
    # Keep enough exploration early, then let policy exploit hard objective.
    args["train"]["learning_rate"] = 3e-4
    args["train"]["anneal_lr"] = True
    args["train"]["ent_coef"] = 0.005
    args["train"]["gamma"] = 0.995
    args["train"]["gae_lambda"] = 0.95
    args["train"]["minibatch_size"] = 512

    pufferl.train(env_name, args=args)


if __name__ == "__main__":
    main()
