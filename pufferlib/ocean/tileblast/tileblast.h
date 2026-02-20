/* tileblast.h
 *
 * Single-agent bomber-style environment used by TileBlast.
 * This header keeps the legacy API surface used by existing wrappers:
 *   - config fields such as agent_speed/vision/obs_size/scalar_size
 *   - observations/actions/rewards/terminals buffer pointers
 *   - same action/tile ids and exported C entry points
 *
 * Usage (header-only):
 *   In exactly ONE .c file:
 *     #define TILEBLAST_IMPLEMENTATION
 *     #include "tileblast.h"
 *   Elsewhere:
 *     #include "tileblast.h"
 */

#pragma once

#include <stdlib.h>
#include <string.h>
#include "raylib.h"

/* --- Defaults / constants ---------------------------------------------- */
#define DEFAULT_WIDTH 11
#define DEFAULT_HEIGHT 9
#define DEFAULT_AGENT_SPEED 1
#define DEFAULT_MAX_STEPS 500
#define DEFAULT_VISION 0

/* Keep these for API compatibility. */
#define OBS_SCALARS 8
typedef enum ObsChannel {
    OBS_GRID = 0,
    OBS_CHANNELS = 1,
} ObsChannel;

#define MAX_BOMBS_PER_AGENT 1
#define DEFAULT_RANGE 2
#define BRICK_DENSITY 0.10f
#define BOMB_TIMER 20
#define BLAST_TIME 2
#define ENEMY_COUNT 3
#define ENEMY_MOVE_INTERVAL 8
#define STEP_PENALTY 0.01f
#define GOAL_PROGRESS_REWARD 0.10f
#define GOAL_REWARD 5.0f
#define ENEMY_COLLISION_PENALTY 2.0f
#define BLAST_DEATH_PENALTY 2.0f
#define TIMEOUT_PENALTY 1.0f
#define TB_SAMPLE_ATTEMPTS 256
#define TB_U8_MAX 255

/* --- Actions ------------------------------------------------------------ */
typedef enum ActionID {
    ACT_NOOP = 0,
    ACT_UP = 1,
    ACT_DOWN = 2,
    ACT_LEFT = 3,
    ACT_RIGHT = 4,
    ACT_BOMB = 5,
} ActionID;

/* --- Tiles -------------------------------------------------------------- */
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

/* --- Log (kept for compatibility; not used) ---------------------------- */
typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
    float timeouts;
    float deaths;
} Log;

/* --- Entities ----------------------------------------------------------- */
typedef struct {
    int row, col;
    int alive;
    int bombs_max;
    int range;
    int lives;
    int invuln;
} Agent;

typedef struct {
    int row, col;
    int owner;
    int timer;
    int range;
    int active;
} Bomb;

typedef struct {
    int row, col;
    int alive;
    int dir;
    int horizontal;
    int min_pos;
    int max_pos;
} Enemy;

/* --- Environment -------------------------------------------------------- */
typedef struct {
    /* Legacy IO buffers (optional) */
    Log log;
    unsigned char* observations; /* optional; zeroed */
    int* actions;                /* used (expects num_agents==1) */
    float* rewards;              /* optional; set to 0 */
    unsigned char* terminals;    /* optional; set to 0 (auto-resets) */

    /* Legacy config fields (kept for callers) */
    int width;
    int height;
    int agent_speed; /* ignored; movement is 1 tile per step */
    int num_agents;  /* forced to 1 */
    int max_steps;
    int vision;      /* ignored; obs removed */
    int obs_size;    /* kept: width*height */
    int scalar_size; /* kept: OBS_SCALARS */

    /* Runtime */
    int tick;
    int goal_row, goal_col;

    unsigned char* grid;        /* TILE_* */
    unsigned char* blast_timer; /* overlay for rendering + damage */

    Bomb* bombs;
    int max_bombs;

    Enemy enemies[ENEMY_COUNT];
    Agent agents[1];
} TileBlast;

/* --- API ---------------------------------------------------------------- */
void init(TileBlast* env);
void c_reset(TileBlast* env);
void c_step(TileBlast* env);
void c_render(TileBlast* env);
void c_close(TileBlast* env);

/* ======================================================================= */
#ifdef TILEBLAST_IMPLEMENTATION
/* ======================================================================= */

/* --- Helpers ------------------------------------------------------------ */
static inline int tb_idx(TileBlast* env, int row, int col) { return row * env->width + col; }
static inline int tb_in_bounds(TileBlast* env, int row, int col) {
    return (row >= 0 && col >= 0 && row < env->height && col < env->width);
}
static float tb_randf(void) { return (float)rand() / (float)RAND_MAX; }
static unsigned char tb_clamp_u8(int value) {
    if (value < 0) return 0;
    if (value > TB_U8_MAX) return TB_U8_MAX;
    return (unsigned char)value;
}

static inline int tb_manhattan(int r0, int c0, int r1, int c1) {
    int dr = r0 - r1; if (dr < 0) dr = -dr;
    int dc = c0 - c1; if (dc < 0) dc = -dc;
    return dr + dc;
}

static int tb_enemy_at(TileBlast* env, int row, int col, int skip_idx) {
    for (int i = 0; i < ENEMY_COUNT; i++) {
        if (i == skip_idx) continue;
        Enemy* e = &env->enemies[i];
        if (e->alive && e->row == row && e->col == col) return 1;
    }
    return 0;
}

static int tb_goal_action_hint(TileBlast* env, int row, int col) {
    int dr = env->goal_row - row;
    int dc = env->goal_col - col;
    if (dr == 0 && dc == 0) return ACT_NOOP;
    if (abs(dr) >= abs(dc)) return dr > 0 ? ACT_DOWN : ACT_UP;
    return dc > 0 ? ACT_RIGHT : ACT_LEFT;
}

static int tb_is_solid(TileBlast* env, int row, int col) {
    if (!tb_in_bounds(env, row, col)) return 1;
    if (row == env->goal_row && col == env->goal_col) return 0;
    unsigned char t = env->grid[tb_idx(env, row, col)];
    return (t == TILE_HARD || t == TILE_SOFT || t == TILE_BOMB);
}

static int tb_is_blocking_tile(unsigned char tile) {
    return (tile == TILE_HARD || tile == TILE_SOFT || tile == TILE_BOMB);
}

/* --- Bombs / blasts ----------------------------------------------------- */
static void tb_clear_bombs(TileBlast* env) {
    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (b->active) env->grid[tb_idx(env, b->row, b->col)] = TILE_EMPTY;
        b->active = 0;
    }
}

static Bomb* tb_bomb_at(TileBlast* env, int row, int col) {
    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (b->active && b->row == row && b->col == col) return b;
    }
    return NULL;
}

static int tb_bombs_owned(TileBlast* env, int owner) {
    int c = 0;
    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (b->active && b->owner == owner) c++;
    }
    return c;
}

static void tb_mark_blast(TileBlast* env, int row, int col) {
    if (!tb_in_bounds(env, row, col)) return;
    env->blast_timer[tb_idx(env, row, col)] = BLAST_TIME;
}

static void tb_damage_tile(TileBlast* env, int row, int col, int* agent_hit) {
    Agent* a = &env->agents[0];
    if (*agent_hit == 0 && a->alive && a->row == row && a->col == col) {
        *agent_hit = 1;
        a->alive = 0;
    }
    for (int i = 0; i < ENEMY_COUNT; i++) {
        Enemy* e = &env->enemies[i];
        if (e->alive && e->row == row && e->col == col) e->alive = 0;
    }
}

static void tb_explode_bomb(TileBlast* env, Bomb* bomb, int* agent_hit) {
    if (!bomb->active) return;
    bomb->active = 0;

    int center = tb_idx(env, bomb->row, bomb->col);
    if (env->grid[center] == TILE_BOMB) env->grid[center] = TILE_EMPTY;

    tb_mark_blast(env, bomb->row, bomb->col);
    tb_damage_tile(env, bomb->row, bomb->col, agent_hit);

    static const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    for (int d = 0; d < 4; d++) {
        int dr = dirs[d][0], dc = dirs[d][1];
        for (int s = 1; s <= bomb->range; s++) {
            int r = bomb->row + dr*s;
            int c = bomb->col + dc*s;
            if (!tb_in_bounds(env, r, c)) break;

            unsigned char t = env->grid[tb_idx(env, r, c)];
            if (t == TILE_HARD) break;

            tb_mark_blast(env, r, c);
            tb_damage_tile(env, r, c, agent_hit);

            if (t == TILE_SOFT) {
                env->grid[tb_idx(env, r, c)] = TILE_EMPTY;
                break;
            }
        }
    }
}

static void tb_update_bombs(TileBlast* env, int* agent_hit) {
    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (!b->active) continue;
        b->timer -= 1;
        if (b->timer <= 0) tb_explode_bomb(env, b, agent_hit);
    }
}

static void tb_decay_blasts(TileBlast* env) {
    int cells = env->width * env->height;
    for (int i = 0; i < cells; i++) {
        if (env->blast_timer[i] > 0) env->blast_timer[i] -= 1;
    }
}

static int tb_place_bomb(TileBlast* env) {
    Agent* a = &env->agents[0];
    if (!a->alive) return 0;
    if (tb_bombs_owned(env, 0) >= a->bombs_max) return 0;
    if (tb_bomb_at(env, a->row, a->col)) return 0;

    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* b = &env->bombs[i];
        if (!b->active) {
            b->active = 1;
            b->row = a->row;
            b->col = a->col;
            b->owner = 0;
            b->timer = BOMB_TIMER;
            b->range = a->range;
            env->grid[tb_idx(env, b->row, b->col)] = TILE_BOMB;
            return 1;
        }
    }
    return 0;
}

/* --- Enemies ------------------------------------------------------------ */
static int tb_enemy_walkable(TileBlast* env, int row, int col) {
    if (!tb_in_bounds(env, row, col)) return 0;
    return !tb_is_blocking_tile(env->grid[tb_idx(env, row, col)]);
}

static void tb_init_enemy_patrol(TileBlast* env, int enemy_idx, int row, int col, int horizontal, int dir) {
    Enemy* e = &env->enemies[enemy_idx];
    e->row = row;
    e->col = col;
    e->alive = 1;
    e->dir = dir;
    e->horizontal = horizontal;

    if (horizontal) {
        e->min_pos = 1;
        e->max_pos = env->width - 2;
        for (int c = e->min_pos; c <= e->max_pos; c++) {
            int ii = tb_idx(env, row, c);
            if (env->grid[ii] == TILE_SOFT) env->grid[ii] = TILE_EMPTY;
        }
    } else {
        e->min_pos = 1;
        e->max_pos = env->height - 2;
        for (int r = e->min_pos; r <= e->max_pos; r++) {
            int ii = tb_idx(env, r, col);
            if (env->grid[ii] == TILE_SOFT) env->grid[ii] = TILE_EMPTY;
        }
    }
}

static void tb_move_enemies(TileBlast* env) {
    if ((env->tick % ENEMY_MOVE_INTERVAL) != 0) return;

    for (int i = 0; i < ENEMY_COUNT; i++) {
        Enemy* e = &env->enemies[i];
        if (!e->alive) continue;

        int nr = e->row, nc = e->col;
        if (e->horizontal) {
            nc = e->col + e->dir;
            if (nc < e->min_pos || nc > e->max_pos || !tb_enemy_walkable(env, nr, nc) || tb_enemy_at(env, nr, nc, i)) {
                e->dir = -e->dir;
                nc = e->col + e->dir;
            }
        } else {
            nr = e->row + e->dir;
            if (nr < e->min_pos || nr > e->max_pos || !tb_enemy_walkable(env, nr, nc) || tb_enemy_at(env, nr, nc, i)) {
                e->dir = -e->dir;
                nr = e->row + e->dir;
            }
        }

        if (tb_enemy_walkable(env, nr, nc) && !tb_enemy_at(env, nr, nc, i)) {
            e->row = nr;
            e->col = nc;
        }
    }
}

/* --- Map generation / spawn -------------------------------------------- */
static int tb_is_spawn_clear(TileBlast* env, int row, int col) {
    int sr = env->agents[0].row, sc = env->agents[0].col;
    if (tb_manhattan(row, col, sr, sc) <= 1) return 1;
    if (tb_manhattan(row, col, env->goal_row, env->goal_col) <= 1) return 1;
    return 0;
}

static void tb_sample_goal(TileBlast* env) {
    for (int a = 0; a < TB_SAMPLE_ATTEMPTS; a++) {
        int r = 1 + (rand() % (env->height - 2));
        int c = 1 + (rand() % (env->width - 2));
        if ((r <= 2) && (c <= 2)) continue;
        if ((r % 2 == 0) && (c % 2 == 0)) continue;
        env->goal_row = r; env->goal_col = c;
        return;
    }
    env->goal_row = env->height - 2;
    env->goal_col = env->width - 2;
}

static void tb_sample_agent(TileBlast* env, int* out_r, int* out_c) {
    for (int a = 0; a < TB_SAMPLE_ATTEMPTS; a++) {
        int r = 1 + (rand() % (env->height - 2));
        int c = 1 + (rand() % (env->width - 2));
        if ((r % 2 == 0) && (c % 2 == 0)) continue;
        if (tb_manhattan(r, c, env->goal_row, env->goal_col) <= 2) continue;
        *out_r = r; *out_c = c;
        return;
    }
    *out_r = 1; *out_c = 1;
}

static int tb_reserve_enemy_tile(TileBlast* env, int* out_r, int* out_c) {
    for (int a = 0; a < TB_SAMPLE_ATTEMPTS; a++) {
        int r = 1 + (rand() % (env->height - 2));
        int c = 1 + (rand() % (env->width - 2));
        if (r == env->agents[0].row && c == env->agents[0].col) continue;
        if (r == env->goal_row && c == env->goal_col) continue;
        if (env->grid[tb_idx(env, r, c)] == TILE_HARD) continue;
        if (tb_enemy_at(env, r, c, -1)) continue;
        *out_r = r; *out_c = c;
        return 1;
    }
    return 0;
}

static void tb_generate_map(TileBlast* env) {
    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->width; c++) {
            unsigned char* cell = &env->grid[tb_idx(env, r, c)];
            if (r == 0 || c == 0 || r == env->height - 1 || c == env->width - 1) { *cell = TILE_HARD; continue; }
            if ((r % 2 == 0) && (c % 2 == 0)) { *cell = TILE_HARD; continue; }
            if (tb_is_spawn_clear(env, r, c)) { *cell = TILE_EMPTY; continue; }
            *cell = (tb_randf() < BRICK_DENSITY) ? TILE_SOFT : TILE_EMPTY;
        }
    }
}

/* --- Legacy observation write ------------------------------------------- */
static void tb_update_observations(TileBlast* env) {
    if (!env->observations) return;
    int bytes_per_agent = env->obs_size * OBS_CHANNELS + env->scalar_size;
    if (bytes_per_agent <= 0) return;
    memset(env->observations, 0, (size_t)bytes_per_agent * (size_t)env->num_agents);

    for (int agent_index = 0; agent_index < env->num_agents; agent_index++) {
        unsigned char* obs = env->observations + agent_index * bytes_per_agent;
        unsigned char* grid_obs = obs + env->obs_size * OBS_GRID;
        unsigned char* scalar_obs = obs + env->obs_size * OBS_CHANNELS;
        Agent* a = &env->agents[agent_index];

        int p = 0;
        for (int r = 0; r < env->height; r++) {
            for (int c = 0; c < env->width; c++) {
                unsigned char t = env->grid[tb_idx(env, r, c)];
                if (env->blast_timer[tb_idx(env, r, c)] > 0) t = TILE_BLAST;
                if (tb_enemy_at(env, r, c, -1)) t = TILE_ENEMY;
                if (r == env->goal_row && c == env->goal_col) t = TILE_GOAL;
                if (a->alive && a->row == r && a->col == c) t = TILE_AGENT0;
                grid_obs[p++] = t;
            }
        }

        int dist_goal = tb_manhattan(a->row, a->col, env->goal_row, env->goal_col);
        int dist_enemy = TB_U8_MAX;
        for (int i = 0; i < ENEMY_COUNT; i++) {
            Enemy* e = &env->enemies[i];
            if (!e->alive) continue;
            int d = tb_manhattan(a->row, a->col, e->row, e->col);
            if (d < dist_enemy) dist_enemy = d;
        }
        int dist_bomb = TB_U8_MAX;
        int min_bomb_timer = TB_U8_MAX;
        for (int i = 0; i < env->max_bombs; i++) {
            Bomb* b = &env->bombs[i];
            if (!b->active) continue;
            int d = tb_manhattan(a->row, a->col, b->row, b->col);
            if (d < dist_bomb) dist_bomb = d;
            if (b->timer < min_bomb_timer) min_bomb_timer = b->timer;
        }
        int owned_bombs = tb_bombs_owned(env, agent_index);
        int bombs_avail = a->bombs_max - owned_bombs;
        if (bombs_avail < 0) bombs_avail = 0;
        int can_place = (a->alive && owned_bombs < a->bombs_max &&
            !tb_bomb_at(env, a->row, a->col)) ? 1 : 0;
        int goal_hint = tb_goal_action_hint(env, a->row, a->col);

        if (env->scalar_size >= 1) scalar_obs[0] = tb_clamp_u8(dist_goal);
        if (env->scalar_size >= 2) scalar_obs[1] = tb_clamp_u8(dist_enemy);
        if (env->scalar_size >= 3) scalar_obs[2] = tb_clamp_u8(dist_bomb);
        if (env->scalar_size >= 4) scalar_obs[3] = tb_clamp_u8(min_bomb_timer);
        if (env->scalar_size >= 5) scalar_obs[4] = tb_clamp_u8(bombs_avail);
        if (env->scalar_size >= 6) scalar_obs[5] = tb_clamp_u8(can_place);
        if (env->scalar_size >= 7) scalar_obs[6] = (unsigned char)(a->alive ? 1 : 0);
        if (env->scalar_size >= 8) scalar_obs[7] = tb_clamp_u8(goal_hint);
    }
}

/* --- API impl ----------------------------------------------------------- */
void init(TileBlast* env) {
    if (!env) return;

    if (env->width <= 0) env->width = DEFAULT_WIDTH;
    if (env->height <= 0) env->height = DEFAULT_HEIGHT;
    if (env->max_steps <= 0) env->max_steps = DEFAULT_MAX_STEPS;

    /* Keep legacy fields consistent */
    env->num_agents = 1;
    if (env->agent_speed <= 0) env->agent_speed = DEFAULT_AGENT_SPEED;
    if (env->vision < 0) env->vision = DEFAULT_VISION;

    env->obs_size = env->width * env->height; /* legacy expectation */
    env->scalar_size = OBS_SCALARS;

    int cells = env->width * env->height;
    env->grid = (unsigned char*)calloc((size_t)cells, sizeof(unsigned char));
    env->blast_timer = (unsigned char*)calloc((size_t)cells, sizeof(unsigned char));

    env->max_bombs = env->num_agents * MAX_BOMBS_PER_AGENT + 4;
    env->bombs = (Bomb*)calloc((size_t)env->max_bombs, sizeof(Bomb));

    env->tick = 0;
    env->goal_row = env->height - 2;
    env->goal_col = env->width - 2;

    /* Clear log (compat) */
    memset(&env->log, 0, sizeof(env->log));
}

void c_reset(TileBlast* env) {
    if (!env) return;

    env->tick = 0;
    memset(env->blast_timer, 0, (size_t)env->width * (size_t)env->height);

    tb_sample_goal(env);

    int ar = 1, ac = 1;
    tb_sample_agent(env, &ar, &ac);

    env->agents[0] = (Agent){
        .row = ar, .col = ac,
        .alive = 1,
        .bombs_max = MAX_BOMBS_PER_AGENT,
        .range = DEFAULT_RANGE,
        .lives = 1,
        .invuln = 0,
    };

    tb_clear_bombs(env);
    tb_generate_map(env);

    /* keep goal cell walkable */
    env->grid[tb_idx(env, env->goal_row, env->goal_col)] = TILE_EMPTY;

    for (int i = 0; i < ENEMY_COUNT; i++) env->enemies[i].alive = 0;
    for (int i = 0; i < ENEMY_COUNT; i++) {
        int er = 1, ec = 1;
        if (!tb_reserve_enemy_tile(env, &er, &ec)) {
            er = 1 + i;
            ec = env->width - 2 - i;
            if (er >= env->height - 1) er = env->height - 2;
            if (ec <= 0) ec = 1;
        }
        int horizontal = rand() & 1;
        int dir = (rand() & 1) ? 1 : -1;
        tb_init_enemy_patrol(env, i, er, ec, horizontal, dir);
    }

    /* Update observation buffer for policy input */
    tb_update_observations(env);
    if (env->rewards) env->rewards[0] = 0.0f;
    if (env->terminals) env->terminals[0] = 0;
}

static void tb_resolve_move(TileBlast* env, int action) {
    Agent* a = &env->agents[0];
    if (!a->alive) return;
    if ((unsigned)action > ACT_BOMB) return;

    static const int dr[ACT_BOMB + 1] = {0, -1, 1, 0, 0, 0};
    static const int dc[ACT_BOMB + 1] = {0, 0, 0, -1, 1, 0};

    int nr = a->row + dr[action];
    int nc = a->col + dc[action];
    if (dr[action] == 0 && dc[action] == 0) return;

    if (!tb_is_solid(env, nr, nc)) {
        a->row = nr;
        a->col = nc;
    }
}

static int tb_read_action(const TileBlast* env) {
    if (!env->actions) return ACT_NOOP;
    int action = env->actions[0];
    if (action < ACT_NOOP || action > ACT_BOMB) return ACT_NOOP;
    return action;
}

static int tb_agent_on_goal(const TileBlast* env) {
    const Agent* a = &env->agents[0];
    return (a->alive && a->row == env->goal_row && a->col == env->goal_col);
}

static void tb_finish_episode(TileBlast* env, Agent* agent) {
    if (env->terminals) env->terminals[0] = 1; /* 1-frame pulse before reset */
    env->log.episode_length = (float)env->tick;
    env->log.n += 1.0f;
    if (tb_agent_on_goal(env)) env->log.perf += 1.0f;
    if (env->tick >= env->max_steps) env->log.timeouts += 1.0f;
    if (!agent->alive) env->log.deaths += 1.0f;
    c_reset(env);
}

void c_step(TileBlast* env) {
    if (!env) return;

    /* legacy outputs: always write safe defaults */
    float reward = 0.0f;
    if (env->rewards) env->rewards[0] = reward;
    if (env->terminals) env->terminals[0] = 0;

    Agent* a = &env->agents[0];
    int prev_goal_dist = tb_manhattan(a->row, a->col, env->goal_row, env->goal_col);

    env->tick += 1;
    tb_decay_blasts(env);

    int action = tb_read_action(env);

    if (action >= ACT_UP && action <= ACT_RIGHT) tb_resolve_move(env, action);
    if (action == ACT_BOMB) tb_place_bomb(env);
    int touched_enemy_after_player_move = tb_enemy_at(env, a->row, a->col, -1);

    tb_move_enemies(env);

    int agent_hit = 0;
    tb_update_bombs(env, &agent_hit);

    int done = 0;
    reward -= STEP_PENALTY;
    int new_goal_dist = tb_manhattan(a->row, a->col, env->goal_row, env->goal_col);
    reward += (float)(prev_goal_dist - new_goal_dist) * GOAL_PROGRESS_REWARD;

    if (!a->alive) {
        done = 1;
    } else if (touched_enemy_after_player_move) {
        a->alive = 0;
        reward -= ENEMY_COLLISION_PENALTY;
        done = 1;
    } else if (tb_enemy_at(env, a->row, a->col, -1)) {
        a->alive = 0;
        reward -= ENEMY_COLLISION_PENALTY;
        done = 1;
    } else if (agent_hit) {
        a->alive = 0;
        reward -= BLAST_DEATH_PENALTY;
        done = 1;
    } else if (tb_agent_on_goal(env)) {
        reward += GOAL_REWARD;
        done = 1;
    } else if (env->tick >= env->max_steps) {
        reward -= TIMEOUT_PENALTY;
        done = 1;
    }

    if (env->rewards) env->rewards[0] = reward;
    env->log.episode_return += reward;

    /* Minimal: auto-reset immediately */
    if (done) {
        tb_finish_episode(env, a);
        return;
    }

    tb_update_observations(env);
}

#include "tileblast_render.h"

void c_render(TileBlast* env) {
    tb_render_impl(env);
}

void c_close(TileBlast* env) {
    if (!env) return;
    if (IsWindowReady()) CloseWindow();
    free(env->blast_timer);
    free(env->grid);
    free(env->bombs);
    env->blast_timer = NULL;
    env->grid = NULL;
    env->bombs = NULL;
}

#endif /* TILEBLAST_IMPLEMENTATION */
