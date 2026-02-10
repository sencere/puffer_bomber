import os
import sys
import argparse
import numpy as np
import torch

# Ensure we use the local repo, not site-packages
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pufferlib.ocean.tileblast import tileblast


class TinyMLPPolicy(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_size = env.single_observation_space.shape[0]
        hidden = 128
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_size, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
        )
        self.action_head = torch.nn.Linear(hidden, env.single_action_space.n)

    def forward_eval(self, observations, state=None):
        if isinstance(observations, dict):
            if "observations" in observations:
                observations = observations["observations"]
            elif "obs" in observations:
                observations = observations["obs"]
            else:
                observations = next(iter(observations.values()))
        if not torch.is_tensor(observations):
            observations = torch.as_tensor(observations)
        x = observations.float()
        logits = self.action_head(self.net(x))
        return logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None, help="Path to .pt model (optional)")
    p.add_argument("--vision", type=int, default=0)
    p.add_argument("--width", type=int, default=11)
    p.add_argument("--height", type=int, default=9)
    args = p.parse_args()

    env = tileblast.TileBlast(
        num_envs=1,
        num_agents=2,
        render_mode="raylib",
        width=args.width,
        height=args.height,
        vision=args.vision,
    )
    obs, _ = env.reset()

    policy = None
    if args.model:
        policy = TinyMLPPolicy(env).cpu()
        state = torch.load(args.model, map_location="cpu")
        state = {k.replace("module.", ""): v for k, v in state.items()}
        policy.load_state_dict(state, strict=False)
        policy.eval()

    while True:
        if policy is None:
            actions = np.random.randint(0, env.single_action_space.n, size=(env.num_agents,))
        else:
            with torch.no_grad():
                logits = policy.forward_eval(torch.from_numpy(obs))
                actions = torch.argmax(logits, dim=1).cpu().numpy()

        obs, rewards, terminals, truncs, info = env.step(actions)
        env.render()

        if terminals.any():
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
