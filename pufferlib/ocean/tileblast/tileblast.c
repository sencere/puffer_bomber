/* Standalone demo for TileBlast. Build with:
 * bash scripts/build_ocean.sh tileblast local
 */

#define TILEBLAST_IMPLEMENTATION
#include "tileblast.h"

static int key_action() {
    if (IsKeyDown(KEY_SPACE)) return ACT_BOMB;
    if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) return ACT_UP;
    if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) return ACT_DOWN;
    if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) return ACT_LEFT;
    if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) return ACT_RIGHT;
    return ACT_NOOP;
}

int main() {
    TileBlast env = {
        .width = DEFAULT_WIDTH,
        .height = DEFAULT_HEIGHT,
        .agent_speed = DEFAULT_AGENT_SPEED,
        .num_agents = 1,
        .max_steps = DEFAULT_MAX_STEPS,
    };
    init(&env);

    int obs_size = env.obs_size * OBS_CHANNELS + env.scalar_size;
    env.observations = (unsigned char*)calloc(obs_size * env.num_agents, sizeof(unsigned char));
    env.actions = (int*)calloc(env.num_agents, sizeof(int));
    env.rewards = (float*)calloc(env.num_agents, sizeof(float));
    env.terminals = (unsigned char*)calloc(env.num_agents, sizeof(unsigned char));
    env.truncations = (unsigned char*)calloc(env.num_agents, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        env.actions[0] = key_action();
        c_step(&env);
        c_render(&env);
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    free(env.truncations);
    c_close(&env);
    return 0;
}
