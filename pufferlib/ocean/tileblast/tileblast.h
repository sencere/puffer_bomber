/* TileBlast: single-agent bomber-style grid env with goal objective. */

#include <stdlib.h>
#include <string.h>
#include "raylib.h"

/* Default configuration and gameplay constants. */
#define DEFAULT_WIDTH 11
#define DEFAULT_HEIGHT 9
#define DEFAULT_AGENT_SPEED 1
#define DEFAULT_MAX_STEPS 500
#define DEFAULT_VISION 0
#define OBS_SCALARS 8
#define MAX_BOMBS_PER_AGENT 1
#define DEFAULT_RANGE 2
#define BRICK_DENSITY 0.10f
#define BOMB_TIMER 20
#define BLAST_TIME 2
#define ENEMY_COUNT 3
#define ENEMY_MOVE_INTERVAL 5
#define START_LIVES 1
#define RESPAWN_INVULN 0

#define SCORE_PER_BRICK 1.0f
#define SCORE_FOR_GOAL 10.0f
#define TIMEOUT_PENALTY 0.1f
#define ENEMY_KILL_REWARD 5.0f
#define BLAST_DANGER_PENALTY 0.2f
#define BLAST_ESCAPE_REWARD 0.15f
#define BOMB_PROX_THRESHOLD 3
#define BOMB_PROX_PENALTY 0.05f
#define BOMB_ON_TILE_PENALTY 5.0f
#define BOMB_ON_TILE_DANGER_TIMER 3
#define BOMB_PLANT_BASE_REWARD 0.2f
#define TACTICAL_BOMB_REWARD 1.0f
#define BOMB_NEAR_ENEMY_RANGE 3
#define BOMB_NEAR_ENEMY_REWARD 0.4f
#define MISSED_TACTICAL_BOMB_PENALTY 0.2f
#define ENEMY_BLOCK_RANGE 3
#define ENEMY_PATH_BLOCK_PENALTY 0.2f
#define ENEMY_PATH_AVOID_REWARD 0.1f
#define STALL_PENALTY 0.005f
#define NOOP_PENALTY 0.05f
#define MOVE_REWARD 0.01f
#define VISIT_HEAT_DECAY 0.95f
#define VISIT_HEAT_DEPOSIT 1.0f
#define VISIT_HEAT_PENALTY 0.01f

/* Discrete action ids used by env steps. */
typedef enum ActionID {
    ACT_NOOP = 0,
    ACT_UP = 1,
    ACT_DOWN = 2,
    ACT_LEFT = 3,
    ACT_RIGHT = 4,
    ACT_BOMB = 5,
} ActionID;

/* Tile ids used inside grid observations/rendering. */
typedef enum TileID {
    TILE_EMPTY = 0,
    TILE_HARD = 1,
    TILE_SOFT = 2,
    TILE_BOMB = 3,
    TILE_BLAST = 4,
    TILE_AGENT0 = 5,
    TILE_GOAL = 6,
    TILE_ENEMY = 7,
} TileID;

/* Observation channel indices. */
typedef enum ObsChannel {
    OBS_GRID = 0,
    OBS_CHANNELS = 1,
} ObsChannel;

/* Episode/log stats reported to Python. */
typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
    float timeouts;
    float deaths;
} Log;

/* Learning agent state. */
typedef struct {
    int r;
    int c;
    int alive;
    int bombs_max;
    int range;
    int lives;
    int invuln;
} Agent;

/* Bomb slot state (kept for observations/rendering compatibility). */
typedef struct {
    int r;
    int c;
    int owner;
    int timer;
    int range;
    int active;
} Bomb;

/* Patrolling enemy state. */
typedef struct {
    int r;
    int c;
    int alive;
    int dir;
    int horizontal;
    int min_pos;
    int max_pos;
} Enemy;

/* Full C environment state for one TileBlast instance. */
typedef struct {
    Log log;
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;

    int width;
    int height;
    int agent_speed;
    int num_agents;
    int max_steps;
    int vision;
    int obs_size;
    int scalar_size;
    int tick;
    int goal_r;
    int goal_c;
    float episode_return_accum;

    unsigned char* grid;
    unsigned char* blast_timer;
    float* visit_heat;
    Bomb* bombs;
    int max_bombs;
    Enemy enemies[ENEMY_COUNT];
    Agent agents[1];
    float score_points;
} TileBlast;
/* --- Grid helpers -------------------------------------------------------- */
/* Convert (row, col) to a row-major flat index. */
static inline int idx(TileBlast* env, int r, int c) {
    return r * env->width + c;
}

/* Check whether a coordinate lies inside the map. */
static inline int in_bounds(TileBlast* env, int r, int c) {
    return (r >= 0 && c >= 0 && r < env->height && c < env->width);
}

/* Manhattan distance helper for scalar observations. */
static inline int manhattan_distance(int r0, int c0, int r1, int c1) {
    int dr = r0 - r1;
    int dc = c0 - c1;
    if (dr < 0) dr = -dr;
    if (dc < 0) dc = -dc;
    return dr + dc;
}

/* --- Rendering helpers --------------------------------------------------- */
static RenderTexture2D g_static_tex = {0};
static int g_static_ready = 0;
static int g_static_w = 0;
static int g_static_h = 0;

/* --- Random helpers ------------------------------------------------------ */
static float randf() {
    return (float)rand() / (float)RAND_MAX;
}

/* --- Bomb helpers -------------------------------------------------------- */
/* Reset all bomb slots to inactive. */
static void clear_bombs(TileBlast* env) {
    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (b->active) {
            env->grid[idx(env, b->r, b->c)] = TILE_EMPTY;
        }
        b->active = 0;
    }
}

/* Return active bomb at tile, otherwise NULL. */
static Bomb* bomb_at(TileBlast* env, int r, int c) {
    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (b->active && b->r == r && b->c == c) {
            return b;
        }
    }
    return NULL;
}

/* Count active bombs owned by `owner`. */
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

static int nearest_bomb_distance(TileBlast* env, int r, int c) {
    int best = 255;
    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (!b->active) continue;
        int d = manhattan_distance(r, c, b->r, b->c);
        if (d < best) best = d;
    }
    return best;
}

static int place_bomb(TileBlast* env) {
    Agent* a = &env->agents[0];
    if (!a->alive) return 0;
    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (!b->active) {
            b->active = 1;
            b->r = a->r;
            b->c = a->c;
            b->owner = 0;
            b->timer = BOMB_TIMER;
            b->range = a->range;
            env->grid[idx(env, b->r, b->c)] = TILE_BOMB;
            return 1;
        }
    }
    return 0;
}

static int bomb_has_tactical_target(TileBlast* env, int r, int c, int range) {
    for (int i = 0; i < ENEMY_COUNT; i++) {
        Enemy* e = &env->enemies[i];
        if (!e->alive) continue;

        if (e->r == r) {
            int dc = e->c - c;
            int dist = dc < 0 ? -dc : dc;
            if (dist > range) continue;
            int step = (dc < 0) ? -1 : 1;
            int clear = 1;
            for (int cc = c + step; cc != e->c; cc += step) {
                unsigned char tile = env->grid[idx(env, r, cc)];
                if (tile == TILE_HARD || tile == TILE_SOFT) {
                    clear = 0;
                    break;
                }
            }
            if (clear) return 1;
        } else if (e->c == c) {
            int dr = e->r - r;
            int dist = dr < 0 ? -dr : dr;
            if (dist > range) continue;
            int step = (dr < 0) ? -1 : 1;
            int clear = 1;
            for (int rr = r + step; rr != e->r; rr += step) {
                unsigned char tile = env->grid[idx(env, rr, c)];
                if (tile == TILE_HARD || tile == TILE_SOFT) {
                    clear = 0;
                    break;
                }
            }
            if (clear) return 1;
        }
    }
    return 0;
}

static void mark_blast(TileBlast* env, int r, int c) {
    if (!in_bounds(env, r, c)) return;
    int i = idx(env, r, c);
    env->blast_timer[i] = BLAST_TIME;
}

static void damage_tile(TileBlast* env, int r, int c, int* agent_hit, int* enemies_killed) {
    Agent* a = &env->agents[0];
    if (*agent_hit == 0 && a->alive && a->r == r && a->c == c) {
        *agent_hit = 1;
        a->alive = 0;
    }
    for (int i = 0; i < ENEMY_COUNT; i++) {
        Enemy* e = &env->enemies[i];
        if (e->alive && e->r == r && e->c == c) {
            e->alive = 0;
            (*enemies_killed) += 1;
            env->score_points += ENEMY_KILL_REWARD;
        }
    }
}

static void explode_bomb(TileBlast* env, Bomb* bomb, int* agent_hit, int* enemies_killed) {
    if (!bomb->active) return;
    bomb->active = 0;
    int center_idx = idx(env, bomb->r, bomb->c);
    if (env->grid[center_idx] == TILE_BOMB) {
        env->grid[center_idx] = TILE_EMPTY;
    }
    mark_blast(env, bomb->r, bomb->c);
    damage_tile(env, bomb->r, bomb->c, agent_hit, enemies_killed);

    static const int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int d = 0; d < 4; d++) {
        int dr = dirs[d][0];
        int dc = dirs[d][1];
        for (int k = 1; k <= bomb->range; k++) {
            int rr = bomb->r + dr * k;
            int cc = bomb->c + dc * k;
            if (!in_bounds(env, rr, cc)) break;
            unsigned char tile = env->grid[idx(env, rr, cc)];
            if (tile == TILE_HARD) break;
            mark_blast(env, rr, cc);
            damage_tile(env, rr, cc, agent_hit, enemies_killed);
            if (tile == TILE_SOFT) {
                env->grid[idx(env, rr, cc)] = TILE_EMPTY;
                break;
            }
        }
    }
}

static void update_bombs(TileBlast* env, int* agent_hit, int* enemies_killed) {
    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (!b->active) continue;
        b->timer -= 1;
        if (b->timer <= 0) {
            explode_bomb(env, b, agent_hit, enemies_killed);
        }
    }
}

static void decay_blasts(TileBlast* env) {
    int cells = env->width * env->height;
    for (int i = 0; i < cells; i++) {
        if (env->blast_timer[i] > 0) {
            env->blast_timer[i] -= 1;
        }
    }
}

static void decay_visit_heat(TileBlast* env) {
    int cells = env->width * env->height;
    for (int i = 0; i < cells; i++) {
        env->visit_heat[i] *= VISIT_HEAT_DECAY;
        if (env->visit_heat[i] < 0.0001f) env->visit_heat[i] = 0.0f;
    }
}

/* --- Threat analysis ----------------------------------------------------- */
/* Find earliest bomb timer that can affect (r,c); 255 means none. */
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

/* Distance to nearest alive enemy; 255 if no enemies alive. */
static int nearest_enemy_distance(TileBlast* env, int r, int c) {
    int best = 255;
    for (int i = 0; i < ENEMY_COUNT; i++) {
        Enemy* e = &env->enemies[i];
        if (!e->alive) continue;
        int d = manhattan_distance(r, c, e->r, e->c);
        if (d < best) best = d;
    }
    return best;
}

/* --- Enemy helpers ------------------------------------------------------ */
/* Forward declaration for helper use before definition. */
static int enemy_at(TileBlast* env, int r, int c, int skip_idx);

/* Tests if the agent can move into a destination tile. */
static int is_solid(TileBlast* env, int r, int c, int agent_idx) {
    if (!in_bounds(env, r, c)) return 1;
    if (r == env->goal_r && c == env->goal_c) return 0;

    unsigned char tile = env->grid[idx(env, r, c)];
    if (tile == TILE_HARD || tile == TILE_SOFT) return 1;
    if (tile == TILE_BOMB) {
        Agent* a = &env->agents[agent_idx];
        if (a->r == r && a->c == c) return 0;
        return 1;
    }
    return 0;
}

/* Leaves spawn/goal-adjacent cells open during random map generation. */
static int is_spawn_clear(TileBlast* env, int r, int c) {
    int sr = env->agents[0].r;
    int sc = env->agents[0].c;
    int gr = env->goal_r, gc = env->goal_c;
    if (manhattan_distance(r, c, sr, sc) <= 1) return 1;
    if (manhattan_distance(r, c, gr, gc) <= 1) return 1;
    return 0;
}

static void sample_goal_tile(TileBlast* env) {
    for (int attempt = 0; attempt < 256; attempt++) {
        int r = 1 + (rand() % (env->height - 2));
        int c = 1 + (rand() % (env->width - 2));
        if ((r <= 2) && (c <= 2)) continue;
        if ((r % 2 == 0) && (c % 2 == 0)) continue;
        env->goal_r = r;
        env->goal_c = c;
        return;
    }
    env->goal_r = env->height - 2;
    env->goal_c = env->width - 2;
}

static void sample_agent_tile(TileBlast* env, int* out_r, int* out_c) {
    for (int attempt = 0; attempt < 256; attempt++) {
        int r = 1 + (rand() % (env->height - 2));
        int c = 1 + (rand() % (env->width - 2));
        if ((r % 2 == 0) && (c % 2 == 0)) continue;
        if (manhattan_distance(r, c, env->goal_r, env->goal_c) <= 2) continue;
        *out_r = r;
        *out_c = c;
        return;
    }
    *out_r = 1;
    *out_c = 1;
}

static int reserve_enemy_tile(TileBlast* env, int* out_r, int* out_c) {
    for (int attempt = 0; attempt < 256; attempt++) {
        int r = 1 + (rand() % (env->height - 2));
        int c = 1 + (rand() % (env->width - 2));
        if (r == env->agents[0].r && c == env->agents[0].c) continue;
        if (r == env->goal_r && c == env->goal_c) continue;
        if (env->grid[idx(env, r, c)] == TILE_HARD) continue;
        if (enemy_at(env, r, c, -1)) continue;
        *out_r = r;
        *out_c = c;
        return 1;
    }
    return 0;
}

/* Checks whether any alive enemy occupies a tile (except skip_idx). */
static int enemy_at(TileBlast* env, int r, int c, int skip_idx) {
    for (int i = 0; i < ENEMY_COUNT; i++) {
        if (i == skip_idx) continue;
        Enemy* e = &env->enemies[i];
        if (e->alive && e->r == r && e->c == c) {
            return 1;
        }
    }
    return 0;
}

static int action_toward_goal(int r, int c, int goal_r, int goal_c) {
    int dr = goal_r - r;
    int dc = goal_c - c;
    if (dr == 0 && dc == 0) return ACT_NOOP;
    if (abs(dr) >= abs(dc)) {
        return (dr > 0) ? ACT_DOWN : ACT_UP;
    }
    return (dc > 0) ? ACT_RIGHT : ACT_LEFT;
}

static int enemy_blocks_action(TileBlast* env, int r, int c, int action, int max_dist) {
    int step_r = 0;
    int step_c = 0;
    if (action == ACT_UP) step_r = -1;
    else if (action == ACT_DOWN) step_r = 1;
    else if (action == ACT_LEFT) step_c = -1;
    else if (action == ACT_RIGHT) step_c = 1;
    else return 0;

    for (int k = 1; k <= max_dist; k++) {
        int rr = r + step_r * k;
        int cc = c + step_c * k;
        if (!in_bounds(env, rr, cc)) break;
        unsigned char tile = env->grid[idx(env, rr, cc)];
        if (tile == TILE_HARD || tile == TILE_SOFT || tile == TILE_BOMB) break;
        if (enemy_at(env, rr, cc, -1)) return 1;
    }
    return 0;
}

static int has_legal_move(TileBlast* env, int r, int c) {
    if (!is_solid(env, r - 1, c, 0)) return 1;
    if (!is_solid(env, r + 1, c, 0)) return 1;
    if (!is_solid(env, r, c - 1, 0)) return 1;
    if (!is_solid(env, r, c + 1, 0)) return 1;
    return 0;
}

/* Patrol movement collision check for enemies. */
static int enemy_walkable(TileBlast* env, int r, int c) {
    if (!in_bounds(env, r, c)) return 0;
    unsigned char t = env->grid[idx(env, r, c)];
    if (t == TILE_HARD || t == TILE_SOFT || t == TILE_BOMB) return 0;
    return 1;
}

/* Initializes one enemy and clears its patrol line through soft tiles. */
static void init_enemy_patrol(TileBlast* env, int enemy_idx, int r, int c, int horizontal, int dir) {
    Enemy* e = &env->enemies[enemy_idx];
    e->r = r;
    e->c = c;
    e->alive = 1;
    e->dir = dir;
    e->horizontal = horizontal;

    if (horizontal) {
        e->min_pos = 1;
        e->max_pos = env->width - 2;
        for (int cc = e->min_pos; cc <= e->max_pos; cc++) {
            int i = idx(env, r, cc);
            if (env->grid[i] == TILE_SOFT) env->grid[i] = TILE_EMPTY;
        }
    } else {
        e->min_pos = 1;
        e->max_pos = env->height - 2;
        for (int rr = e->min_pos; rr <= e->max_pos; rr++) {
            int i = idx(env, rr, c);
            if (env->grid[i] == TILE_SOFT) env->grid[i] = TILE_EMPTY;
        }
    }

    env->grid[idx(env, e->r, e->c)] = TILE_EMPTY;
}

/* --- Enemy movement ----------------------------------------------------- */
/* Updates enemy patrol positions on a fixed tick interval. */
static void move_enemies(TileBlast* env) {
    if ((env->tick % ENEMY_MOVE_INTERVAL) != 0) return;

    for (int i = 0; i < ENEMY_COUNT; i++) {
        Enemy* e = &env->enemies[i];
        if (!e->alive) continue;

        int nr = e->r;
        int nc = e->c;
        if (e->horizontal) {
            nc = e->c + e->dir;
            if (nc < e->min_pos || nc > e->max_pos || !enemy_walkable(env, nr, nc) || enemy_at(env, nr, nc, i)) {
                e->dir = -e->dir;
                nc = e->c + e->dir;
            }
        } else {
            nr = e->r + e->dir;
            if (nr < e->min_pos || nr > e->max_pos || !enemy_walkable(env, nr, nc) || enemy_at(env, nr, nc, i)) {
                e->dir = -e->dir;
                nr = e->r + e->dir;
            }
        }

        if (enemy_walkable(env, nr, nc) && !enemy_at(env, nr, nc, i)) {
            e->r = nr;
            e->c = nc;
        }
    }
}

/* --- Map generation ----------------------------------------------------- */
/* Fills the map with walls and random soft blocks. */
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
            *cell = (randf() < BRICK_DENSITY) ? TILE_SOFT : TILE_EMPTY;
        }
    }
}

/* --- Observation encoding ---------------------------------------------- */
/* Writes observation grid and scalar channels for the current state. */
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
                    if (env->blast_timer[idx(env, rr, cc)] > 0) {
                        v = TILE_BLAST;
                    }
                    if (self->alive && self->r == rr && self->c == cc) {
                        v = TILE_AGENT0;
                    } else if (enemy_at(env, rr, cc, -1)) {
                        v = TILE_ENEMY;
                    } else if (rr == env->goal_r && cc == env->goal_c) {
                        v = TILE_GOAL;
                    }
                    grid_obs[idx_local++] = v;
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
                        if (env->blast_timer[idx(env, rr, cc)] > 0) {
                            v = TILE_BLAST;
                        }
                        if (self->alive && self->r == rr && self->c == cc) {
                            v = TILE_AGENT0;
                        } else if (enemy_at(env, rr, cc, -1)) {
                            v = TILE_ENEMY;
                        } else if (rr == env->goal_r && cc == env->goal_c) {
                            v = TILE_GOAL;
                        }
                        grid_obs[idx_local] = v;
                    }
                    idx_local++;
                }
            }
        }
        if (self->alive) {
            if (env->vision == 0) {
                grid_obs[idx(env, self->r, self->c)] = TILE_AGENT0;
            } else {
                int side = env->vision * 2 + 1;
                int self_idx = env->vision * side + env->vision;
                grid_obs[self_idx] = TILE_AGENT0;
            }
        }

        int dist_goal = manhattan_distance(self->r, self->c, env->goal_r, env->goal_c);
        int dist_bomb = 255;
        for (int i = 0; i < env->max_bombs; i++) {
            Bomb* b = &env->bombs[i];
            if (!b->active) continue;
            int d = manhattan_distance(self->r, self->c, b->r, b->c);
            if (d < dist_bomb) dist_bomb = d;
        }

        int min_timer = min_bomb_timer_affecting(env, self->r, self->c);
        int dist_enemy = nearest_enemy_distance(env, self->r, self->c);
        int bombs_avail = self->bombs_max - bombs_owned(env, a);
        if (bombs_avail < 0) bombs_avail = 0;
        int can_place = (self->alive && bombs_owned(env, a) < self->bombs_max && !bomb_at(env, self->r, self->c)) ? 1 : 0;
        int score_points = (int)env->score_points;

        if (dist_goal > 255) dist_goal = 255;
        if (dist_bomb > 255) dist_bomb = 255;
        if (min_timer > 255) min_timer = 255;
        if (dist_enemy > 255) dist_enemy = 255;
        if (bombs_avail > 255) bombs_avail = 255;
        if (score_points > 255) score_points = 255;
        if (score_points < 0) score_points = 0;

        if (env->scalar_size >= 1) scalar_obs[0] = (unsigned char)dist_goal;
        if (env->scalar_size >= 2) scalar_obs[1] = (unsigned char)dist_bomb;
        if (env->scalar_size >= 3) scalar_obs[2] = (unsigned char)min_timer;
        if (env->scalar_size >= 4) scalar_obs[3] = (unsigned char)bombs_avail;
        if (env->scalar_size >= 5) scalar_obs[4] = (unsigned char)can_place;
        if (env->scalar_size >= 6) scalar_obs[5] = (unsigned char)score_points;
        if (env->scalar_size >= 7) scalar_obs[6] = 0;
        if (env->scalar_size >= 8) scalar_obs[7] = (unsigned char)dist_enemy;
    }
}

/* --- Initialization ---------------------------------------------------- */
/* Initializes runtime buffers and normalized config values. */
static void init(TileBlast* env) {
    if (env->width <= 0) env->width = DEFAULT_WIDTH;
    if (env->height <= 0) env->height = DEFAULT_HEIGHT;
    if (env->agent_speed <= 0) env->agent_speed = DEFAULT_AGENT_SPEED;
    if (env->max_steps <= 0) env->max_steps = DEFAULT_MAX_STEPS;
    env->num_agents = 1;
    if (env->vision < 0) env->vision = DEFAULT_VISION;

    if (env->vision == 0) {
        env->obs_size = env->width * env->height;
    } else {
        env->obs_size = (env->vision * 2 + 1) * (env->vision * 2 + 1);
    }
    env->scalar_size = OBS_SCALARS;

    int cells = env->width * env->height;
    env->grid = (unsigned char*)calloc(cells, sizeof(unsigned char));
    env->blast_timer = (unsigned char*)calloc(cells, sizeof(unsigned char));
    env->visit_heat = (float*)calloc(cells, sizeof(float));
    env->max_bombs = env->num_agents * MAX_BOMBS_PER_AGENT + 4;
    env->bombs = (Bomb*)calloc(env->max_bombs, sizeof(Bomb));
    env->episode_return_accum = 0.0f;
    env->score_points = 0.0f;
}

/* --- Episode lifecycle ------------------------------------------------- */
/* Resets one episode and respawns agent/enemies. */
static void c_reset(TileBlast* env) {
    env->tick = 0;
    env->episode_return_accum = 0.0f;
    env->score_points = 0.0f;

    memset(env->blast_timer, 0, env->width * env->height);
    memset(env->visit_heat, 0, env->width * env->height * sizeof(float));
    sample_goal_tile(env);
    int spawn_r = 1;
    int spawn_c = 1;
    sample_agent_tile(env, &spawn_r, &spawn_c);
    env->agents[0] = (Agent){
        .r = spawn_r,
        .c = spawn_c,
        .alive = 1,
        .bombs_max = MAX_BOMBS_PER_AGENT,
        .range = DEFAULT_RANGE,
        .lives = START_LIVES,
        .invuln = RESPAWN_INVULN,
    };
    clear_bombs(env);
    generate_map(env);
    env->grid[idx(env, env->goal_r, env->goal_c)] = TILE_EMPTY;
    for (int i = 0; i < ENEMY_COUNT; i++) {
        env->enemies[i].alive = 0;
    }
    for (int i = 0; i < ENEMY_COUNT; i++) {
        int er = 1;
        int ec = 1;
        if (!reserve_enemy_tile(env, &er, &ec)) {
            er = 1 + i;
            ec = env->width - 2 - i;
            if (er >= env->height - 1) er = env->height - 2;
            if (ec <= 0) ec = 1;
        }
        int horizontal = rand() & 1;
        int dir = (rand() & 1) ? 1 : -1;
        init_enemy_patrol(env, i, er, ec, horizontal, dir);
    }

    update_observations(env);
}

/* --- Agent control ----------------------------------------------------- */
/* Applies one action as a single-tile move. */
static void resolve_move(TileBlast* env, int action) {
    Agent* a = &env->agents[0];
    if (!a->alive) return;
    if ((unsigned)action > ACT_BOMB) return;

    // Action-indexed movement deltas. Non-move actions map to (0,0).
    static const int dr[ACT_BOMB + 1] = {0, -1, 1, 0, 0, 0};
    static const int dc[ACT_BOMB + 1] = {0, 0, 0, -1, 1, 0};
    int step_r = dr[action];
    int step_c = dc[action];
    if (step_r == 0 && step_c == 0) return;

    int tr = a->r + step_r;
    int tc = a->c + step_c;
    if (!is_solid(env, tr, tc, 0)) {
        a->r = tr;
        a->c = tc;
    }
}

/* --- Runtime step ------------------------------------------------------ */
/* Simple reward shaping that favors progress toward the goal. */
static void c_step(TileBlast* env) {
    env->tick += 1;
    env->terminals[0] = 0;
    decay_blasts(env);
    decay_visit_heat(env);

    Agent* a = &env->agents[0];
    int prev_threat = min_bomb_timer_affecting(env, a->r, a->c);
    int prev_r = a->r;
    int prev_c = a->c;
    int goal_action = action_toward_goal(prev_r, prev_c, env->goal_r, env->goal_c);
    int goal_lane_blocked = enemy_blocks_action(env, prev_r, prev_c, goal_action, ENEMY_BLOCK_RANGE);
    int tactical_bomb_available = bomb_has_tactical_target(env, prev_r, prev_c, a->range);
    int can_place_bomb = (bombs_owned(env, 0) < a->bombs_max && !bomb_at(env, prev_r, prev_c));

    int action = ACT_NOOP;
    if (env->actions) action = env->actions[0];
    if (action < ACT_NOOP || action > ACT_BOMB) action = ACT_NOOP;
    resolve_move(env, action);
    move_enemies(env);
    int planted_bomb = 0;
    if (action == ACT_BOMB) {
        planted_bomb = place_bomb(env);
    }

    int agent_hit = 0;
    int enemies_killed = 0;
    update_bombs(env, &agent_hit, &enemies_killed);

    float reward = 0.0f;
    if (a->alive) {
        int legal_moves = has_legal_move(env, prev_r, prev_c);
        int moved = (a->r != prev_r || a->c != prev_c);
        if (moved) {
            reward += MOVE_REWARD;
        } else {
            if (legal_moves) reward -= STALL_PENALTY;
        }
        if (action == ACT_NOOP && legal_moves) {
            reward -= NOOP_PENALTY;
        }
        if (planted_bomb) {
            reward += BOMB_PLANT_BASE_REWARD;
            if (tactical_bomb_available) {
                reward += TACTICAL_BOMB_REWARD;
            }
            int dist_enemy_prev = nearest_enemy_distance(env, prev_r, prev_c);
            if (dist_enemy_prev <= BOMB_NEAR_ENEMY_RANGE) {
                reward += BOMB_NEAR_ENEMY_REWARD;
            }
        } else if (action != ACT_BOMB && tactical_bomb_available && can_place_bomb) {
            reward -= MISSED_TACTICAL_BOMB_PENALTY;
        }
        int new_threat = min_bomb_timer_affecting(env, a->r, a->c);
        int dist_bomb = nearest_bomb_distance(env, a->r, a->c);
        if (dist_bomb <= BOMB_PROX_THRESHOLD) {
            reward -= (float)(BOMB_PROX_THRESHOLD + 1 - dist_bomb) * BOMB_PROX_PENALTY;
            if (dist_bomb == 0 && new_threat <= BOMB_ON_TILE_DANGER_TIMER) {
                reward -= BOMB_ON_TILE_PENALTY;
            }
        }
        if (new_threat <= 3) {
            reward -= BLAST_DANGER_PENALTY;
        }
        if (prev_threat <= 3 && (new_threat > prev_threat || new_threat == 255)) {
            reward += BLAST_ESCAPE_REWARD;
        }
        if (goal_lane_blocked) {
            if (action == goal_action) {
                reward -= ENEMY_PATH_BLOCK_PENALTY;
            } else if (action >= ACT_UP && action <= ACT_RIGHT &&
                       (a->r != prev_r || a->c != prev_c)) {
                reward += ENEMY_PATH_AVOID_REWARD;
            }
        }
        int self_idx = idx(env, a->r, a->c);
        reward -= env->visit_heat[self_idx] * VISIT_HEAT_PENALTY;
        env->visit_heat[self_idx] += VISIT_HEAT_DEPOSIT;
    }
    if (enemies_killed > 0) {
        reward += (float)enemies_killed * ENEMY_KILL_REWARD;
    }
    int done = 0;
    if (a->alive && a->r == env->goal_r && a->c == env->goal_c) {
        env->score_points += SCORE_FOR_GOAL;
        reward += SCORE_FOR_GOAL;
        done = 1;
    }

    if (!done && a->alive && enemy_at(env, a->r, a->c, -1)) {
        a->alive = 0;
        env->log.deaths += 1.0f;
        reward -= SCORE_FOR_GOAL * 0.5f;
        done = 1;
    }

    if (!done && agent_hit) {
        env->log.deaths += 1.0f;
        reward -= SCORE_FOR_GOAL * 0.5f;
        done = 1;
    }

    if (!done && env->tick >= env->max_steps) {
        done = 1;
        env->log.timeouts += 1.0f;
        reward -= TIMEOUT_PENALTY;
    }

    env->rewards[0] = reward;
    env->episode_return_accum += reward;

    if (done) {
        env->terminals[0] = 1;
        env->log.score = env->score_points;
        env->log.perf = (a->alive && a->r == env->goal_r && a->c == env->goal_c) ? 1.0f : 0.0f;
        env->log.episode_return = env->episode_return_accum;
        env->log.episode_length = (float)env->tick;
        env->log.n += 1.0f;
        c_reset(env);
        return;
    }

    update_observations(env);
}

/* --- Rendering --------------------------------------------------------- */
/* Draws world state and HUD. */
static void c_render(TileBlast* env) {
    const int tile = 32;
    const int w = env->width * tile;
    const int h = env->height * tile;
    if (!IsWindowReady()) {
        InitWindow(w, h, "PufferLib TileBlast Goal");
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
            int x = c * tile;
            int y = r * tile;
            if (t == TILE_SOFT) {
                DrawRectangle(x + 3, y + 3, tile - 6, tile - 6, (Color){138, 125, 255, 235});
            }
            if (env->blast_timer[idx(env, r, c)] > 0) {
                int tval = env->blast_timer[idx(env, r, c)];
                int inset = 3 + (BLAST_TIME - tval);
                if (inset > tile / 2 - 1) inset = tile / 2 - 1;
                int glow = 120 + tval * 45;
                if (glow > 255) glow = 255;
                DrawRectangle(x + inset, y + inset, tile - 2 * inset, tile - 2 * inset, (Color){255, 190, 70, glow});
                DrawRectangle(x + tile / 2 - 2, y + 4, 4, tile - 8, (Color){255, 240, 170, glow});
                DrawRectangle(x + 4, y + tile / 2 - 2, tile - 8, 4, (Color){255, 240, 170, glow});
            }
        }
    }

    int gx = env->goal_c * tile;
    int gy = env->goal_r * tile;
    DrawRectangle(gx + 6, gy + 6, tile - 12, tile - 12, (Color){72, 220, 120, 255});

    if (env->agents[0].alive) {
        int x = env->agents[0].c * tile + tile / 2;
        int y = env->agents[0].r * tile + tile / 2;
        DrawCircle(x, y, 12, (Color){255, 216, 107, 255});
    }

    for (int i = 0; i < ENEMY_COUNT; i++) {
        Enemy* e = &env->enemies[i];
        if (!e->alive) continue;
        int x = e->c * tile + tile / 2;
        int y = e->r * tile + tile / 2;
        DrawCircle(x, y, 10, (Color){255, 120, 120, 255});
        DrawCircle(x - 3, y - 2, 2, WHITE);
        DrawCircle(x + 3, y - 2, 2, WHITE);
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

    DrawText(TextFormat("Step: %d/%d", env->tick, env->max_steps), 10, 10, 18, (Color){255, 240, 200, 255});
    DrawText(TextFormat("Return: %.2f", env->episode_return_accum), 10, 32, 18, (Color){255, 240, 200, 255});

    EndDrawing();
}

/* --- Teardown ---------------------------------------------------------- */
/* Frees env resources allocated by init(). */
static void c_close(TileBlast* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    if (g_static_ready) {
        UnloadRenderTexture(g_static_tex);
        g_static_ready = 0;
    }
    free(env->visit_heat);
    free(env->blast_timer);
    free(env->grid);
    free(env->bombs);
}
