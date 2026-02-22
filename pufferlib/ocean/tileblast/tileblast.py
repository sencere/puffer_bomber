'''TileBlast: single-agent bomber-style environment with goal objective.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.tileblast import binding

SCALAR_OBS = 13

class TileBlast(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        render_mode=None,
        log_interval=128,
        buf=None,
        seed=0,
        num_agents=1,
        width=11,
        height=9,
        agent_speed=1,
        max_steps=200000,
        vision=0,
        distance_reward_interval=5,
    ):
        assert num_agents == 1, "TileBlast goal mode is single-agent (num_agents must be 1)"
        if vision == 0:
            grid_size = width * height
        else:
            obs_side = vision * 2 + 1
            grid_size = obs_side * obs_side
        obs_size = grid_size + SCALAR_OBS
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(obs_size,), dtype=np.uint8
        )
        self.single_action_space = gymnasium.spaces.Discrete(6)
        self.render_mode = render_mode
        self.num_agents = num_envs * num_agents
        self.log_interval = log_interval

        super().__init__(buf)

        c_envs = []
        for i in range(num_envs):
            c_env = binding.env_init(
                self.observations[i * num_agents : (i + 1) * num_agents],
                self.actions[i * num_agents : (i + 1) * num_agents],
                self.rewards[i * num_agents : (i + 1) * num_agents],
                self.terminals[i * num_agents : (i + 1) * num_agents],
                self.truncations[i * num_agents : (i + 1) * num_agents],
                seed,
                num_agents=num_agents,
                width=width,
                height=height,
                agent_speed=agent_speed,
                max_steps=max_steps,
                vision=vision,
                distance_reward_interval=distance_reward_interval,
            )
            c_envs.append(c_env)

        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        actions = np.asarray(actions, dtype=np.int32).reshape(self.actions.shape)
        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


if __name__ == '__main__':
    N = 8
    env = TileBlast(num_envs=N, num_agents=1)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 6, (CACHE, N))

    i = 0
    import time

    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += env.num_agents
        i += 1

    print('TileBlast SPS:', int(steps / (time.time() - start)))
