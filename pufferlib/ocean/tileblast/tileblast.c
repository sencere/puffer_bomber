/* Standalone demo for TileBlast. Build with:
 * bash scripts/build_ocean.sh tileblast local
 */

#include "tileblast.h"

static int key_action_p1() {
    if (IsKeyDown(KEY_SPACE)) return ACT_BOMB;
    if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) return ACT_UP;
    if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) return ACT_DOWN;
    if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) return ACT_LEFT;
    if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) return ACT_RIGHT;
    return ACT_NOOP;
}

static int key_action_p2() {
    if (IsKeyDown(KEY_RIGHT_CONTROL)) return ACT_BOMB;
    if (IsKeyDown(KEY_I)) return ACT_UP;
    if (IsKeyDown(KEY_K)) return ACT_DOWN;
    if (IsKeyDown(KEY_J)) return ACT_LEFT;
    if (IsKeyDown(KEY_L)) return ACT_RIGHT;
    return ACT_NOOP;
}

int main() {
    TileBlast env = {
        .width = DEFAULT_WIDTH,
        .height = DEFAULT_HEIGHT,
        .num_agents = 2,
        .max_steps = DEFAULT_MAX_STEPS,
        .reward_win = DEFAULT_REWARD_WIN,
        .reward_loss = DEFAULT_REWARD_LOSS,
        .reward_draw = DEFAULT_REWARD_DRAW,
        .reward_soft = DEFAULT_REWARD_SOFT,
        .reward_hit = DEFAULT_REWARD_HIT,
        .reward_self_hit = DEFAULT_REWARD_SELF_HIT,
        .reward_step = DEFAULT_REWARD_STEP,
        .reward_survive = DEFAULT_REWARD_SURVIVE,
        .reward_no_bomb = DEFAULT_REWARD_NO_BOMB,
        .reward_avoid_bomb = DEFAULT_REWARD_AVOID_BOMB,
        .reward_bomb_near = DEFAULT_REWARD_BOMB_NEAR,
        .reward_no_cover = DEFAULT_REWARD_NO_COVER,
        .reward_stall_tile = DEFAULT_REWARD_STALL_TILE,
        .reward_escape_bomb = DEFAULT_REWARD_ESCAPE_BOMB,
        .reward_reverse_move = DEFAULT_REWARD_REVERSE_MOVE,
    };
    init(&env);

    int obs_size = env.obs_size * OBS_CHANNELS + env.scalar_size;
    env.observations = (unsigned char*)calloc(obs_size * env.num_agents, sizeof(unsigned char));
    env.actions = (int*)calloc(env.num_agents, sizeof(int));
    env.rewards = (float*)calloc(env.num_agents, sizeof(float));
    env.terminals = (unsigned char*)calloc(env.num_agents, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);

    int frame = 0;
    int pending_action = ACT_NOOP;
    int pending_action_p2 = ACT_NOOP;
    const int frameskip = 2; // slow down gameplay for visibility
    while (!WindowShouldClose()) {
        int action = key_action_p1();
        int action2 = key_action_p2();
        if (action != ACT_NOOP) {
            pending_action = action;
        }
        if (action2 != ACT_NOOP) {
            pending_action_p2 = action2;
        }
        if (frame % frameskip == 0) {
            env.actions[0] = pending_action;
            pending_action = ACT_NOOP;
            env.actions[1] = pending_action_p2;
            pending_action_p2 = ACT_NOOP;
            c_step(&env);
        }
        c_render(&env);
        frame++;
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}
