/* TileBlast: simple 2-agent bomberman-style grid env. Supports num_agents = 1 or 2. If 1, the second agent is a bot. */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "raylib.h"

enum {
    ACT_NOOP = 0,
    ACT_UP = 1,
    ACT_DOWN = 2,
    ACT_LEFT = 3,
    ACT_RIGHT = 4,
    ACT_BOMB = 5,
};

enum {
    TILE_EMPTY = 0,
    TILE_HARD = 1,
    TILE_SOFT = 2,
    TILE_BOMB = 3,
    TILE_BLAST = 4,
    TILE_AGENT0 = 5,
    TILE_AGENT1 = 6,
};

enum {
    OBS_GRID = 0,
    OBS_CHANNELS = 1,
};

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
    float timeouts;
    float deaths;
} Log;

typedef struct {
    int r;
    int c;
    int alive;
    int bombs_max;
    int range;
    int lives;
    int invuln;
    int since_bomb;
    int same_tile_steps;
    int escape_bomb_r;
    int escape_bomb_c;
    int escape_prev_dist;
    int escape_steps_left;
    int last_action;
} Agent;

typedef struct {
    int r;
    int c;
    int owner;
    int timer;
    int range;
    int active;
} Bomb;

typedef struct {
    Log log;
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int width;
    int height;
    int num_agents;
    int max_steps;
    int vision;
    int obs_size;
    int scalar_size;
    int tick;
    unsigned char* grid;
    unsigned char* blast_timer;
    int8_t* blast_owner;
    Bomb* bombs;
    int max_bombs;
    Agent agents[2];
    float reward_win;
    float reward_loss;
    float reward_draw;
    float reward_soft;
    float reward_hit;
    float reward_self_hit;
    float reward_step;
    float reward_survive;
    float reward_no_bomb;
    float reward_avoid_bomb;
    float reward_bomb_near;
    float reward_no_cover;
    float reward_stall_tile;
    float reward_escape_bomb;
    float reward_reverse_move;
} TileBlast;

#define DEFAULT_WIDTH 11
#define DEFAULT_HEIGHT 9
#define DEFAULT_MAX_STEPS 500
#define DEFAULT_VISION 0
#define OBS_SCALARS 5
#define MAX_BOMBS_PER_AGENT 1
#define DEFAULT_RANGE 2
#define BOMB_TIMER 16
#define BLAST_TIME 6
#define START_LIVES 1
#define RESPAWN_INVULN 10
#define DEFAULT_REWARD_WIN 5.0f
#define DEFAULT_REWARD_LOSS -5.0f
#define DEFAULT_REWARD_DRAW 0.0f
#define DEFAULT_REWARD_SOFT 0.0f
#define DEFAULT_REWARD_HIT 0.0f
#define DEFAULT_REWARD_SELF_HIT 0.0f
#define DEFAULT_REWARD_STEP -0.002f
#define DEFAULT_REWARD_SURVIVE 0.0f
#define DEFAULT_REWARD_NO_BOMB 0.0f
#define DEFAULT_REWARD_AVOID_BOMB 0.0f
#define DEFAULT_REWARD_BOMB_NEAR 0.01f
#define DEFAULT_REWARD_NO_COVER 0.0f
#define DEFAULT_REWARD_STALL_TILE -0.001f
#define DEFAULT_REWARD_ESCAPE_BOMB 0.0f
#define DEFAULT_REWARD_REVERSE_MOVE -0.0015f
#define STALL_GRACE_STEPS 3
#define STALL_DANGER_GRACE_TIMER 2
#define ESCAPE_WINDOW_STEPS 4
#define NO_BOMB_GRACE_STEPS 30

static inline int idx(TileBlast* env, int r, int c) {
    return r * env->width + c;
}

static inline int in_bounds(TileBlast* env, int r, int c) {
    return (r >= 0 && c >= 0 && r < env->height && c < env->width);
}

static inline int manhattan(int r0, int c0, int r1, int c1) {
    int dr = r0 - r1;
    int dc = c0 - c1;
    if (dr < 0) dr = -dr;
    if (dc < 0) dc = -dc;
    return dr + dc;
}

static RenderTexture2D g_static_tex = {0};
static int g_static_ready = 0;
static int g_static_w = 0;
static int g_static_h = 0;

static float randf() {
    return (float)rand() / (float)RAND_MAX;
}

static void clear_bombs(TileBlast* env) {
    for (int i = 0; i < env->max_bombs; i++) {
        env->bombs[i].active = 0;
    }
}

static Bomb* bomb_at(TileBlast* env, int r, int c) {
    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (b->active && b->r == r && b->c == c) {
            return b;
        }
    }
    return NULL;
}

static int bombs_owned(TileBlast* env, int owner) {
    int count = 0;
    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (b->active && b->owner == owner) {
            count++;
        }
    }
    return count;
}

static void apply_reward(TileBlast* env, int agent_idx, float reward) {
    if (agent_idx >= 0 && agent_idx < env->num_agents) {
        env->rewards[agent_idx] += reward;
    }
}

static int min_bomb_timer_affecting(TileBlast* env, int r, int c) {
    int min_timer = 255;
    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (!b->active) continue;
        if (b->r == r) {
            int dc = c - b->c;
            int dist = dc < 0 ? -dc : dc;
            if (dist <= b->range) {
                int step = (dc < 0) ? -1 : 1;
                int clear = 1;
                for (int cc = b->c + step; cc != c; cc += step) {
                    unsigned char t = env->grid[idx(env, r, cc)];
                    if (t == TILE_HARD || t == TILE_SOFT) {
                        clear = 0;
                        break;
                    }
                }
                if (clear && b->timer < min_timer) min_timer = b->timer;
            }
        }
        if (b->c == c) {
            int dr = r - b->r;
            int dist = dr < 0 ? -dr : dr;
            if (dist <= b->range) {
                int step = (dr < 0) ? -1 : 1;
                int clear = 1;
                for (int rr = b->r + step; rr != r; rr += step) {
                    unsigned char t = env->grid[idx(env, rr, c)];
                    if (t == TILE_HARD || t == TILE_SOFT) {
                        clear = 0;
                        break;
                    }
                }
                if (clear && b->timer < min_timer) min_timer = b->timer;
            }
        }
    }
    return min_timer;
}

static int in_blast_of_bomb(TileBlast* env, int r, int c, Bomb* b) {
    if (!b || !b->active) return 0;
    if (b->r == r) {
        int dc = c - b->c;
        int dist = dc < 0 ? -dc : dc;
        if (dist <= b->range) {
            int step = (dc < 0) ? -1 : 1;
            for (int cc = b->c + step; cc != c; cc += step) {
                unsigned char t = env->grid[idx(env, r, cc)];
                if (t == TILE_HARD || t == TILE_SOFT) {
                    return 0;
                }
            }
            return 1;
        }
    }
    if (b->c == c) {
        int dr = r - b->r;
        int dist = dr < 0 ? -dr : dr;
        if (dist <= b->range) {
            int step = (dr < 0) ? -1 : 1;
            for (int rr = b->r + step; rr != r; rr += step) {
                unsigned char t = env->grid[idx(env, rr, c)];
                if (t == TILE_HARD || t == TILE_SOFT) {
                    return 0;
                }
            }
            return 1;
        }
    }
    return 0;
}

static int is_bomb_danger(TileBlast* env, int agent_idx) {
    Agent* a = &env->agents[agent_idx];
    if (!a->alive) return 0;
    int min_timer = min_bomb_timer_affecting(env, a->r, a->c);
    return (min_timer <= 3);
}

static int is_reverse_move(int prev_action, int action) {
    return (prev_action == ACT_UP && action == ACT_DOWN) ||
           (prev_action == ACT_DOWN && action == ACT_UP) ||
           (prev_action == ACT_LEFT && action == ACT_RIGHT) ||
           (prev_action == ACT_RIGHT && action == ACT_LEFT);
}

static int is_solid(TileBlast* env, int r, int c, int agent_idx);

static int has_legal_move(TileBlast* env, int agent_idx) {
    Agent* a = &env->agents[agent_idx];
    if (!a->alive) return 0;
    int r = a->r;
    int c = a->c;
    int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int d = 0; d < 4; d++) {
        int nr = r + dirs[d][0];
        int nc = c + dirs[d][1];
        if (!is_solid(env, nr, nc, agent_idx)) {
            return 1;
        }
    }
    return 0;
}

static int is_spawn_clear(TileBlast* env, int r, int c) {
    int r0 = 1, c0 = 1;
    int r1 = env->height - 2, c1 = env->width - 2;
    if ((r == r0 && c == c0) || (r == r0 + 1 && c == c0) || (r == r0 && c == c0 + 1)) return 1;
    if ((r == r1 && c == c1) || (r == r1 - 1 && c == c1) || (r == r1 && c == c1 - 1)) return 1;
    return 0;
}

static void generate_map(TileBlast* env) {
    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->width; c++) {
            unsigned char* cell = &env->grid[idx(env, r, c)];
            if (r == 0 || c == 0 || r == env->height - 1 || c == env->width - 1) {
                *cell = TILE_HARD;
                continue;
            }
            if ((r % 2 == 0) && (c % 2 == 0)) {
                *cell = TILE_HARD;
                continue;
            }
            if (is_spawn_clear(env, r, c)) {
                *cell = TILE_EMPTY;
                continue;
            }
            *cell = (randf() < 0.10f) ? TILE_SOFT : TILE_EMPTY;
        }
    }
}

static int is_solid(TileBlast* env, int r, int c, int agent_idx) {
    if (!in_bounds(env, r, c)) return 1;
    unsigned char tile = env->grid[idx(env, r, c)];
    if (tile == TILE_HARD || tile == TILE_SOFT) return 1;
    if (tile == TILE_BOMB) {
        Agent* a = &env->agents[agent_idx];
        if (a->r == r && a->c == c) return 0;
        return 1;
    }
    return 0;
}

static int place_bomb(TileBlast* env, int agent_idx) {
    Agent* a = &env->agents[agent_idx];
    if (!a->alive) return 0;
    if (bombs_owned(env, agent_idx) >= a->bombs_max) return 0;
    if (bomb_at(env, a->r, a->c)) return 0;

    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (!b->active) {
            b->active = 1;
            b->r = a->r;
            b->c = a->c;
            b->owner = agent_idx;
            b->timer = BOMB_TIMER;
            b->range = a->range;
            env->grid[idx(env, b->r, b->c)] = TILE_BOMB;
            return 1;
        }
    }
    return 0;
}

static void add_blast_cell(TileBlast* env, int r, int c, int owner) {
    if (!in_bounds(env, r, c)) return;
    env->blast_timer[idx(env, r, c)] = BLAST_TIME;
    if (env->blast_owner) {
        env->blast_owner[idx(env, r, c)] = (int8_t)owner;
    }
    env->grid[idx(env, r, c)] = TILE_BLAST;
}

static void explode_bomb(TileBlast* env, Bomb* bomb) {
    bomb->active = 0;
    add_blast_cell(env, bomb->r, bomb->c, bomb->owner);

    int dirs[4][2] = {{1,0}, {-1,0}, {0,1}, {0,-1}};
    for (int d = 0; d < 4; d++) {
        int dr = dirs[d][0];
        int dc = dirs[d][1];
        for (int k = 1; k <= bomb->range; k++) {
            int nr = bomb->r + dr * k;
            int nc = bomb->c + dc * k;
            if (!in_bounds(env, nr, nc)) break;
            unsigned char tile = env->grid[idx(env, nr, nc)];
            if (tile == TILE_HARD) break;
            add_blast_cell(env, nr, nc, bomb->owner);
            if (tile == TILE_SOFT) {
                env->grid[idx(env, nr, nc)] = TILE_EMPTY;
                apply_reward(env, bomb->owner, env->reward_soft);
                break;
            }
            if (tile == TILE_BOMB) break;
        }
    }
}

static void update_observations(TileBlast* env) {
    for (int a = 0; a < env->num_agents; a++) {
        unsigned char* obs = env->observations + a * (env->obs_size * OBS_CHANNELS + env->scalar_size);
        memset(obs, 0, env->obs_size * OBS_CHANNELS + env->scalar_size);

        unsigned char* grid_obs = obs + env->obs_size * OBS_GRID;
        unsigned char* scalar_obs = obs + env->obs_size * OBS_CHANNELS;

        Agent* self = &env->agents[a];
        int idx_local = 0;
        if (env->vision == 0) {
            for (int rr = 0; rr < env->height; rr++) {
                for (int cc = 0; cc < env->width; cc++) {
                    unsigned char v = env->grid[idx(env, rr, cc)];
                    if (env->agents[0].alive && env->agents[0].r == rr && env->agents[0].c == cc) {
                        v = (a == 0) ? TILE_AGENT0 : TILE_AGENT1;
                    } else if (env->agents[1].alive && env->agents[1].r == rr && env->agents[1].c == cc) {
                        v = (a == 1) ? TILE_AGENT0 : TILE_AGENT1;
                    }
                    grid_obs[idx_local] = v;
                    idx_local++;
                }
            }
        } else {
            for (int dr = -env->vision; dr <= env->vision; dr++) {
                for (int dc = -env->vision; dc <= env->vision; dc++) {
                    int rr = self->r + dr;
                    int cc = self->c + dc;
                    if (!in_bounds(env, rr, cc)) {
                        grid_obs[idx_local] = TILE_HARD;
                    } else {
                        unsigned char v = env->grid[idx(env, rr, cc)];
                        if (env->agents[0].alive && env->agents[0].r == rr && env->agents[0].c == cc) {
                            v = (a == 0) ? TILE_AGENT0 : TILE_AGENT1;
                        } else if (env->agents[1].alive && env->agents[1].r == rr && env->agents[1].c == cc) {
                            v = (a == 1) ? TILE_AGENT0 : TILE_AGENT1;
                        }
                        grid_obs[idx_local] = v;
                    }
                    idx_local++;
                }
            }
        }

        // Scalar observations: Manhattan distance to other agent, Manhattan distance to nearest bomb,
        // min bomb timer affecting me, bombs available, can place bomb
        int other = (a == 0) ? 1 : 0;
        int dist_other = 255;
        if (other < env->num_agents && env->agents[other].alive) {
            int dr = self->r - env->agents[other].r;
            int dc = self->c - env->agents[other].c;
            if (dr < 0) dr = -dr;
            if (dc < 0) dc = -dc;
            dist_other = dr + dc;
        }

        int dist_bomb = 255;
        for (int i = 0; i < env->max_bombs; i++) {
            Bomb* b = &env->bombs[i];
            if (!b->active) continue;
            int dr = self->r - b->r;
            int dc = self->c - b->c;
            if (dr < 0) dr = -dr;
            if (dc < 0) dc = -dc;
            int d = dr + dc;
            if (d < dist_bomb) dist_bomb = d;
        }

        int min_timer = min_bomb_timer_affecting(env, self->r, self->c);
        int bombs_avail = self->bombs_max - bombs_owned(env, a);
        if (bombs_avail < 0) bombs_avail = 0;
        int can_place = (self->alive &&
                         bombs_owned(env, a) < self->bombs_max &&
                         !bomb_at(env, self->r, self->c)) ? 1 : 0;

        if (dist_other > 255) dist_other = 255;
        if (dist_bomb > 255) dist_bomb = 255;
        if (min_timer > 255) min_timer = 255;
        if (bombs_avail > 255) bombs_avail = 255;
        if (env->scalar_size >= 1) scalar_obs[0] = (unsigned char)dist_other;
        if (env->scalar_size >= 2) scalar_obs[1] = (unsigned char)dist_bomb;
        if (env->scalar_size >= 3) scalar_obs[2] = (unsigned char)min_timer;
        if (env->scalar_size >= 4) scalar_obs[3] = (unsigned char)bombs_avail;
        if (env->scalar_size >= 5) scalar_obs[4] = (unsigned char)can_place;
    }
}

static int bot_action(TileBlast* env, int bot_idx) {
    (void)env;
    (void)bot_idx;
    // No built-in logic; bots are disabled. Use num_agents=2 for training.
    return ACT_NOOP;
}

static void init(TileBlast* env) {
    if (env->width <= 0) env->width = DEFAULT_WIDTH;
    if (env->height <= 0) env->height = DEFAULT_HEIGHT;
    if (env->max_steps <= 0) env->max_steps = DEFAULT_MAX_STEPS;
    if (env->num_agents <= 0) env->num_agents = 1;
    if (env->num_agents > 2) env->num_agents = 2;
    if (env->vision < 0) env->vision = DEFAULT_VISION;
    if (env->vision == 0) {
        env->obs_size = env->width * env->height;
    } else {
        env->obs_size = (env->vision * 2 + 1) * (env->vision * 2 + 1);
    }
    env->scalar_size = OBS_SCALARS;
    if (env->reward_win == 0.0f &&
        env->reward_loss == 0.0f &&
        env->reward_draw == 0.0f &&
        env->reward_soft == 0.0f &&
        env->reward_hit == 0.0f &&
        env->reward_self_hit == 0.0f &&
        env->reward_step == 0.0f &&
        env->reward_survive == 0.0f &&
        env->reward_no_bomb == 0.0f &&
        env->reward_avoid_bomb == 0.0f &&
        env->reward_bomb_near == 0.0f &&
        env->reward_no_cover == 0.0f &&
        env->reward_stall_tile == 0.0f &&
        env->reward_escape_bomb == 0.0f &&
        env->reward_reverse_move == 0.0f) {
        env->reward_win = DEFAULT_REWARD_WIN;
        env->reward_loss = DEFAULT_REWARD_LOSS;
        env->reward_draw = DEFAULT_REWARD_DRAW;
        env->reward_soft = DEFAULT_REWARD_SOFT;
        env->reward_hit = DEFAULT_REWARD_HIT;
        env->reward_self_hit = DEFAULT_REWARD_SELF_HIT;
        env->reward_step = DEFAULT_REWARD_STEP;
        env->reward_survive = DEFAULT_REWARD_SURVIVE;
        env->reward_no_bomb = DEFAULT_REWARD_NO_BOMB;
        env->reward_avoid_bomb = DEFAULT_REWARD_AVOID_BOMB;
        env->reward_bomb_near = DEFAULT_REWARD_BOMB_NEAR;
        env->reward_no_cover = DEFAULT_REWARD_NO_COVER;
        env->reward_stall_tile = DEFAULT_REWARD_STALL_TILE;
        env->reward_escape_bomb = DEFAULT_REWARD_ESCAPE_BOMB;
        env->reward_reverse_move = DEFAULT_REWARD_REVERSE_MOVE;
    }

    int cells = env->width * env->height;
    env->grid = (unsigned char*)calloc(cells, sizeof(unsigned char));
    env->blast_timer = (unsigned char*)calloc(cells, sizeof(unsigned char));
    env->blast_owner = (int8_t*)calloc(cells, sizeof(int8_t));
    if (env->blast_owner) {
        memset(env->blast_owner, -1, cells * sizeof(int8_t));
    }
    env->max_bombs = env->num_agents * MAX_BOMBS_PER_AGENT + 4;
    env->bombs = (Bomb*)calloc(env->max_bombs, sizeof(Bomb));
}

static void c_reset(TileBlast* env) {
    env->tick = 0;
    memset(env->blast_timer, 0, env->width * env->height);
    if (env->blast_owner) {
        memset(env->blast_owner, -1, env->width * env->height * sizeof(int8_t));
    }
    clear_bombs(env);
    generate_map(env);

    int p0_r = 1, p0_c = 1;
    int p1_r = env->height - 2, p1_c = env->width - 2;
    // Randomize side assignment each episode to prevent role collapse.
    if (rand() & 1) {
        int tr = p0_r, tc = p0_c;
        p0_r = p1_r; p0_c = p1_c;
        p1_r = tr;   p1_c = tc;
    }

    env->agents[0] = (Agent){
        .r = p0_r, .c = p0_c, .alive = 1, .bombs_max = MAX_BOMBS_PER_AGENT,
        .range = 2, .lives = START_LIVES, .invuln = 0, .since_bomb = 0,
        .same_tile_steps = 0, .escape_bomb_r = -1, .escape_bomb_c = -1,
        .escape_prev_dist = 0, .escape_steps_left = 0, .last_action = ACT_NOOP
    };
    env->agents[1] = (Agent){
        .r = p1_r, .c = p1_c, .alive = 1,
        .bombs_max = MAX_BOMBS_PER_AGENT, .range = 2, .lives = START_LIVES, .invuln = 0, .since_bomb = 0,
        .same_tile_steps = 0, .escape_bomb_r = -1, .escape_bomb_c = -1,
        .escape_prev_dist = 0, .escape_steps_left = 0, .last_action = ACT_NOOP
    };

    update_observations(env);
}

static void resolve_moves(TileBlast* env, int action0, int action1) {
    int r0 = env->agents[0].r;
    int c0 = env->agents[0].c;
    int r1 = env->agents[1].r;
    int c1 = env->agents[1].c;

    int tr0 = r0, tc0 = c0;
    int tr1 = r1, tc1 = c1;

    if (env->agents[0].alive) {
        if (action0 == ACT_UP) tr0--;
        else if (action0 == ACT_DOWN) tr0++;
        else if (action0 == ACT_LEFT) tc0--;
        else if (action0 == ACT_RIGHT) tc0++;
    }

    if (env->agents[1].alive) {
        if (action1 == ACT_UP) tr1--;
        else if (action1 == ACT_DOWN) tr1++;
        else if (action1 == ACT_LEFT) tc1--;
        else if (action1 == ACT_RIGHT) tc1++;
    }

    int move0 = env->agents[0].alive && (tr0 != r0 || tc0 != c0) && !is_solid(env, tr0, tc0, 0);
    int move1 = env->agents[1].alive && (tr1 != r1 || tc1 != c1) && !is_solid(env, tr1, tc1, 1);

    // Allow agents to pass through each other by swapping positions
    // Allow moving into the other agent's cell even if they don't move

    if (move0) {
        env->agents[0].r = tr0;
        env->agents[0].c = tc0;
    }
    if (move1) {
        env->agents[1].r = tr1;
        env->agents[1].c = tc1;
    }
}

static void c_step(TileBlast* env) {
    env->tick += 1;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    if (env->num_agents == 2) {
        env->rewards[1] = 0.0f;
        env->terminals[1] = 0;
    }

    int a0 = env->actions[0];
    int a1 = (env->num_agents == 2) ? env->actions[1] : bot_action(env, 1);
    int prev_r0 = env->agents[0].r;
    int prev_c0 = env->agents[0].c;
    int prev_r1 = env->agents[1].r;
    int prev_c1 = env->agents[1].c;
    Bomb* placed0 = NULL;
    Bomb* placed1 = NULL;

    if (env->agents[0].alive) {
        env->agents[0].since_bomb += 1;
        if (a0 == ACT_BOMB) {
            if (place_bomb(env, 0)) {
                env->agents[0].since_bomb = 0;
                placed0 = bomb_at(env, env->agents[0].r, env->agents[0].c);
                if (placed0) {
                    env->agents[0].escape_bomb_r = placed0->r;
                    env->agents[0].escape_bomb_c = placed0->c;
                    env->agents[0].escape_prev_dist = manhattan(env->agents[0].r, env->agents[0].c, placed0->r, placed0->c);
                    env->agents[0].escape_steps_left = ESCAPE_WINDOW_STEPS;
                }
            }
        }
    }
    if (env->num_agents == 2 && env->agents[1].alive) {
        env->agents[1].since_bomb += 1;
        if (a1 == ACT_BOMB) {
            if (place_bomb(env, 1)) {
                env->agents[1].since_bomb = 0;
                placed1 = bomb_at(env, env->agents[1].r, env->agents[1].c);
                if (placed1) {
                    env->agents[1].escape_bomb_r = placed1->r;
                    env->agents[1].escape_bomb_c = placed1->c;
                    env->agents[1].escape_prev_dist = manhattan(env->agents[1].r, env->agents[1].c, placed1->r, placed1->c);
                    env->agents[1].escape_steps_left = ESCAPE_WINDOW_STEPS;
                }
            }
        }
    }

    resolve_moves(env, a0, a1);

    if (env->agents[0].alive) {
        if (env->agents[0].r == prev_r0 && env->agents[0].c == prev_c0) {
            env->agents[0].same_tile_steps += 1;
        } else {
            env->agents[0].same_tile_steps = 0;
        }
    }
    if (env->num_agents == 2 && env->agents[1].alive) {
        if (env->agents[1].r == prev_r1 && env->agents[1].c == prev_c1) {
            env->agents[1].same_tile_steps += 1;
        } else {
            env->agents[1].same_tile_steps = 0;
        }
    }

    if (env->reward_step != 0.0f) {
        for (int i = 0; i < env->num_agents; i++) {
            if (env->agents[i].alive) {
                apply_reward(env, i, env->reward_step);
            }
        }
    }
    if (env->reward_reverse_move != 0.0f) {
        if (env->agents[0].alive && is_reverse_move(env->agents[0].last_action, a0)) {
            apply_reward(env, 0, env->reward_reverse_move);
        }
        if (env->num_agents == 2 && env->agents[1].alive && is_reverse_move(env->agents[1].last_action, a1)) {
            apply_reward(env, 1, env->reward_reverse_move);
        }
    }
    if (env->reward_survive != 0.0f) {
        for (int i = 0; i < env->num_agents; i++) {
            if (env->agents[i].alive) {
                apply_reward(env, i, env->reward_survive);
            }
        }
    }
    if (env->reward_no_bomb != 0.0f) {
        if (env->agents[0].alive && env->agents[0].since_bomb >= NO_BOMB_GRACE_STEPS &&
            a0 != ACT_BOMB && bombs_owned(env, 0) < env->agents[0].bombs_max) {
            apply_reward(env, 0, env->reward_no_bomb);
        }
        if (env->num_agents == 2 && env->agents[1].alive &&
            env->agents[1].since_bomb >= NO_BOMB_GRACE_STEPS &&
            a1 != ACT_BOMB &&
            bombs_owned(env, 1) < env->agents[1].bombs_max) {
            apply_reward(env, 1, env->reward_no_bomb);
        }
    }
    if (env->reward_bomb_near != 0.0f && env->num_agents == 2) {
        if (env->agents[0].alive && env->agents[1].alive && placed0) {
            int self_covered = !in_blast_of_bomb(env, env->agents[0].r, env->agents[0].c, placed0);
            int same_row = (env->agents[0].r == env->agents[1].r);
            int same_col = (env->agents[0].c == env->agents[1].c);
            if (same_row || same_col) {
                int dr = env->agents[0].r - env->agents[1].r;
                int dc = env->agents[0].c - env->agents[1].c;
                if (dr < 0) dr = -dr;
                if (dc < 0) dc = -dc;
                int dist = dr + dc;
                if (dist <= env->agents[0].range) {
                    int clear = 1;
                    if (same_row) {
                        int step = (env->agents[1].c < env->agents[0].c) ? -1 : 1;
                        for (int c = env->agents[0].c + step; c != env->agents[1].c; c += step) {
                            unsigned char t = env->grid[idx(env, env->agents[0].r, c)];
                            if (t == TILE_HARD || t == TILE_SOFT) {
                                clear = 0;
                                break;
                            }
                        }
                    } else {
                        int step = (env->agents[1].r < env->agents[0].r) ? -1 : 1;
                        for (int r = env->agents[0].r + step; r != env->agents[1].r; r += step) {
                            unsigned char t = env->grid[idx(env, r, env->agents[0].c)];
                            if (t == TILE_HARD || t == TILE_SOFT) {
                                clear = 0;
                                break;
                            }
                        }
                    }
                    if (clear && self_covered) {
                        apply_reward(env, 0, env->reward_bomb_near);
                    }
                }
            }
            if (!self_covered && env->reward_no_cover != 0.0f) {
                apply_reward(env, 0, env->reward_no_cover);
            }
        }
        if (env->agents[0].alive && env->agents[1].alive && placed1) {
            int self_covered = !in_blast_of_bomb(env, env->agents[1].r, env->agents[1].c, placed1);
            int same_row = (env->agents[1].r == env->agents[0].r);
            int same_col = (env->agents[1].c == env->agents[0].c);
            if (same_row || same_col) {
                int dr = env->agents[1].r - env->agents[0].r;
                int dc = env->agents[1].c - env->agents[0].c;
                if (dr < 0) dr = -dr;
                if (dc < 0) dc = -dc;
                int dist = dr + dc;
                if (dist <= env->agents[1].range) {
                    int clear = 1;
                    if (same_row) {
                        int step = (env->agents[0].c < env->agents[1].c) ? -1 : 1;
                        for (int c = env->agents[1].c + step; c != env->agents[0].c; c += step) {
                            unsigned char t = env->grid[idx(env, env->agents[1].r, c)];
                            if (t == TILE_HARD || t == TILE_SOFT) {
                                clear = 0;
                                break;
                            }
                        }
                    } else {
                        int step = (env->agents[0].r < env->agents[1].r) ? -1 : 1;
                        for (int r = env->agents[1].r + step; r != env->agents[0].r; r += step) {
                            unsigned char t = env->grid[idx(env, r, env->agents[1].c)];
                            if (t == TILE_HARD || t == TILE_SOFT) {
                                clear = 0;
                                break;
                            }
                        }
                    }
                    if (clear && self_covered) {
                        apply_reward(env, 1, env->reward_bomb_near);
                    }
                }
            }
            if (!self_covered && env->reward_no_cover != 0.0f) {
                apply_reward(env, 1, env->reward_no_cover);
            }
        }
    }
    if (env->reward_avoid_bomb != 0.0f) {
        for (int i = 0; i < env->num_agents; i++) {
            if (is_bomb_danger(env, i)) {
                apply_reward(env, i, env->reward_avoid_bomb);
            }
        }
    }
    if (env->reward_stall_tile != 0.0f) {
        if (env->agents[0].alive &&
            env->agents[0].same_tile_steps >= STALL_GRACE_STEPS &&
            has_legal_move(env, 0) &&
            min_bomb_timer_affecting(env, env->agents[0].r, env->agents[0].c) > STALL_DANGER_GRACE_TIMER) {
            apply_reward(env, 0, env->reward_stall_tile);
        }
        if (env->num_agents == 2 && env->agents[1].alive &&
            env->agents[1].same_tile_steps >= STALL_GRACE_STEPS &&
            has_legal_move(env, 1) &&
            min_bomb_timer_affecting(env, env->agents[1].r, env->agents[1].c) > STALL_DANGER_GRACE_TIMER) {
            apply_reward(env, 1, env->reward_stall_tile);
        }
    }
    if (env->reward_escape_bomb != 0.0f) {
        if (env->agents[0].alive && env->agents[0].escape_steps_left > 0) {
            Bomb* b = bomb_at(env, env->agents[0].escape_bomb_r, env->agents[0].escape_bomb_c);
            if (b && b->active) {
                int dist = manhattan(env->agents[0].r, env->agents[0].c, b->r, b->c);
                int delta = dist - env->agents[0].escape_prev_dist;
                if (delta > 0) {
                    apply_reward(env, 0, env->reward_escape_bomb * (float)delta);
                }
                env->agents[0].escape_prev_dist = dist;
                env->agents[0].escape_steps_left -= 1;
            } else {
                env->agents[0].escape_steps_left = 0;
            }
        }
        if (env->num_agents == 2 && env->agents[1].alive && env->agents[1].escape_steps_left > 0) {
            Bomb* b = bomb_at(env, env->agents[1].escape_bomb_r, env->agents[1].escape_bomb_c);
            if (b && b->active) {
                int dist = manhattan(env->agents[1].r, env->agents[1].c, b->r, b->c);
                int delta = dist - env->agents[1].escape_prev_dist;
                if (delta > 0) {
                    apply_reward(env, 1, env->reward_escape_bomb * (float)delta);
                }
                env->agents[1].escape_prev_dist = dist;
                env->agents[1].escape_steps_left -= 1;
            } else {
                env->agents[1].escape_steps_left = 0;
            }
        }
    }

    if (env->agents[0].alive) env->agents[0].last_action = a0;
    if (env->num_agents == 2 && env->agents[1].alive) env->agents[1].last_action = a1;

    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (!b->active) continue;
        b->timer -= 1;
        if (b->timer <= 0) {
            explode_bomb(env, b);
        }
    }

    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (!b->active) continue;
        if (env->blast_timer[idx(env, b->r, b->c)] > 0) {
            b->timer = 0;
        }
    }

    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (!b->active) continue;
        if (b->timer <= 0) {
            explode_bomb(env, b);
        }
    }

    for (int i = 0; i < 2; i++) {
        Agent* a = &env->agents[i];
        if (!a->alive) continue;
        if (a->invuln > 0) {
            a->invuln -= 1;
        }
        if (a->invuln == 0 && env->blast_timer[idx(env, a->r, a->c)] > 0) {
            if (env->blast_owner) {
                int owner = env->blast_owner[idx(env, a->r, a->c)];
                if (owner >= 0 && owner < env->num_agents) {
                    if (owner == i) {
                        apply_reward(env, i, env->reward_self_hit);
                    } else {
                        apply_reward(env, owner, env->reward_hit);
                        apply_reward(env, i, env->reward_self_hit);
                    }
                }
            }
            a->lives -= 1;
            if (a->lives > 0) {
                if (i == 0) {
                    a->r = 1;
                    a->c = 1;
                } else {
                    a->r = env->height - 2;
                    a->c = env->width - 2;
                }
                a->invuln = RESPAWN_INVULN;
            } else {
                a->alive = 0;
            }
        }
    }

    for (int i = 0; i < env->width * env->height; i++) {
        if (env->blast_timer[i] > 0) {
            env->blast_timer[i] -= 1;
            if (env->blast_timer[i] == 0 && env->grid[i] == TILE_BLAST) {
                env->grid[i] = TILE_EMPTY;
                if (env->blast_owner) {
                    env->blast_owner[i] = -1;
                }
            }
        }
    }

    int alive0 = env->agents[0].alive;
    int alive1 = env->agents[1].alive;
    int done_steps = (env->tick >= env->max_steps);
    int done_death = (!alive0 || !alive1);
    int done = done_steps || done_death;
    if (done) {
        if (done_death) {
            if (alive0 && !alive1) {
                apply_reward(env, 0, env->reward_win);
                if (env->num_agents == 2) apply_reward(env, 1, env->reward_loss);
            } else if (!alive0 && alive1) {
                apply_reward(env, 1, env->reward_win);
                if (env->num_agents == 2) apply_reward(env, 0, env->reward_loss);
            } else {
                apply_reward(env, 0, env->reward_loss);
                if (env->num_agents == 2) apply_reward(env, 1, env->reward_loss);
            }
        } else {
            apply_reward(env, 0, env->reward_draw);
            if (env->num_agents == 2) apply_reward(env, 1, env->reward_draw);
        }

        env->terminals[0] = 1;
        if (env->num_agents == 2) env->terminals[1] = 1;

        float score = (alive0 ? 1.0f : 0.0f) - (alive1 ? 1.0f : 0.0f);
        env->log.score = score;
        env->log.perf = (score + 1.0f) * 0.5f;
        env->log.episode_return = env->rewards[0];
        env->log.episode_length = (float)env->tick;
        env->log.n += 1.0f;
        if (done_death) {
            env->log.deaths += 1.0f;
        } else {
            env->log.timeouts += 1.0f;
        }
        c_reset(env);
        return;
    }

    update_observations(env);
}

static void c_render(TileBlast* env) {
    const int tile = 32;
    const int w = env->width * tile;
    const int h = env->height * tile;
    if (!IsWindowReady()) {
        InitWindow(w, h, "PufferLib TileBlast");
        SetTargetFPS(10);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    if (!g_static_ready || g_static_w != w || g_static_h != h) {
        if (g_static_ready) {
            UnloadRenderTexture(g_static_tex);
        }
        g_static_tex = LoadRenderTexture(w, h);
        g_static_ready = 1;
        g_static_w = w;
        g_static_h = h;
        BeginTextureMode(g_static_tex);
        ClearBackground((Color){11, 13, 24, 255});
        for (int r = 0; r < env->height; r++) {
            for (int c = 0; c < env->width; c++) {
                int x = c * tile;
                int y = r * tile;
                Color base = ((r + c) % 2 == 0) ? (Color){27, 33, 64, 255} : (Color){21, 26, 51, 255};
                DrawRectangle(x, y, tile, tile, base);
                unsigned char t = env->grid[idx(env, r, c)];
                if (t == TILE_HARD) {
                    DrawRectangle(x + 3, y + 3, tile - 6, tile - 6, (Color){255, 90, 95, 235});
                }
            }
        }
        EndTextureMode();
    }

    BeginDrawing();
    DrawTextureRec(g_static_tex.texture, (Rectangle){0, 0, (float)w, -(float)h}, (Vector2){0, 0}, WHITE);
    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->width; c++) {
            unsigned char t = env->grid[idx(env, r, c)];
            if (t == TILE_SOFT) {
                int x = c * tile;
                int y = r * tile;
                DrawRectangle(x + 3, y + 3, tile - 6, tile - 6, (Color){138, 125, 255, 235});
            }
            if (env->blast_timer[idx(env, r, c)] > 0) {
                int x = c * tile;
                int y = r * tile;
                DrawRectangle(x + 2, y + 2, tile - 4, tile - 4, (Color){102, 230, 255, 200});
            }
        }
    }

    if (env->agents[0].alive && env->agents[1].alive &&
        env->agents[0].r == env->agents[1].r && env->agents[0].c == env->agents[1].c) {
        int x = env->agents[0].c * tile + tile / 2;
        int y = env->agents[0].r * tile + tile / 2;
        DrawCircle(x, y, 12, (Color){255, 193, 107, 255});
    } else {
        if (env->agents[0].alive) {
            int x = env->agents[0].c * tile + tile / 2;
            int y = env->agents[0].r * tile + tile / 2;
            DrawCircle(x, y, 12, (Color){255, 216, 107, 255});
        }
        if (env->agents[1].alive) {
            int x = env->agents[1].c * tile + tile / 2;
            int y = env->agents[1].r * tile + tile / 2;
            DrawCircle(x, y, 12, (Color){255, 170, 107, 255});
        }
    }

    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (!b->active) continue;
        int x = b->c * tile + tile / 2;
        int y = b->r * tile + tile / 2 + 4;
        DrawCircle(x, y, 8, (Color){24, 30, 55, 255});
        DrawCircleLines(x, y, 9, (Color){90, 100, 160, 200});
        DrawCircle(x + 7, y - 9, 3, (Color){255, 204, 51, 255});
    }

    DrawText(TextFormat("Step: %d/%d", env->tick, env->max_steps),
             10, 10, 18, (Color){255, 240, 200, 255});
    DrawText(TextFormat("P1 Lives: %d  P2 Lives: %d", env->agents[0].lives, env->agents[1].lives),
             10, 32, 18, (Color){255, 240, 200, 255});

    EndDrawing();
}

static void c_close(TileBlast* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    if (g_static_ready) {
        UnloadRenderTexture(g_static_tex);
        g_static_ready = 0;
    }
    free(env->blast_timer);
    free(env->blast_owner);
    free(env->grid);
    free(env->bombs);
}
