import os
import sys
import argparse
import numpy as np
import torch

# Ensure we use the local repo, not site-packages
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pufferlib.ocean import Policy
from pufferlib.ocean.tileblast import tileblast


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None, help="Path to .pt model (optional)")
    p.add_argument("--vision", type=int, default=0)
    p.add_argument("--width", type=int, default=11)
    p.add_argument("--height", type=int, default=9)
    p.add_argument("--deterministic", action="store_true", help="Use argmax actions instead of sampling")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device")
    args = p.parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    env = tileblast.TileBlast(
        num_envs=1,
        num_agents=1,
        render_mode="raylib",
        width=args.width,
        height=args.height,
        vision=args.vision,
    )
    obs, _ = env.reset()

    policy = None
    if args.model:
        # Use the exact same policy class used by pufferl training.
        policy = Policy(env).to(device)
        state = torch.load(args.model, map_location=device)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        policy.load_state_dict(state, strict=True)
        policy.eval()

    while True:
        if policy is None:
            actions = np.random.randint(0, env.single_action_space.n, size=(env.num_agents,))
        else:
            with torch.no_grad():
                logits, _ = policy.forward_eval(torch.from_numpy(obs).to(device))
                if args.deterministic:
                    actions = torch.argmax(logits, dim=1).cpu().numpy()
                else:
                    actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()

        obs, rewards, terminals, truncs, info = env.step(actions)
        env.render()

        if terminals.any():
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
