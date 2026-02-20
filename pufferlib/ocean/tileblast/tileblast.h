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

static inline int tb_manhattan(int row0, int col0, int row1, int col1) {
    int row_delta = row0 - row1; if (row_delta < 0) row_delta = -row_delta;
    int col_delta = col0 - col1; if (col_delta < 0) col_delta = -col_delta;
    return row_delta + col_delta;
}

static int tb_enemy_at(TileBlast* env, int row, int col, int skip_idx) {
    for (int i = 0; i < ENEMY_COUNT; i++) {
        if (i == skip_idx) continue;
        Enemy* enemy = &env->enemies[i];
        if (enemy->alive && enemy->row == row && enemy->col == col) return 1;
    }
    return 0;
}

static int tb_goal_action_hint(TileBlast* env, int row, int col) {
    int goal_row_offset = env->goal_row - row;
    int goal_col_offset = env->goal_col - col;
    if (goal_row_offset == 0 && goal_col_offset == 0) return ACT_NOOP;
    if (abs(goal_row_offset) >= abs(goal_col_offset)) return goal_row_offset > 0 ? ACT_DOWN : ACT_UP;
    return goal_col_offset > 0 ? ACT_RIGHT : ACT_LEFT;
}

static int tb_is_solid(TileBlast* env, int row, int col) {
    if (!tb_in_bounds(env, row, col)) return 1;
    if (row == env->goal_row && col == env->goal_col) return 0;
    unsigned char tile = env->grid[tb_idx(env, row, col)];
    return (tile == TILE_HARD || tile == TILE_SOFT || tile == TILE_BOMB);
}

static int tb_is_blocking_tile(unsigned char tile) {
    return (tile == TILE_HARD || tile == TILE_SOFT || tile == TILE_BOMB);
}

/* --- Bombs / blasts ----------------------------------------------------- */
static void tb_clear_bombs(TileBlast* env) {
    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* bomb = &env->bombs[i];
        if (bomb->active) env->grid[tb_idx(env, bomb->row, bomb->col)] = TILE_EMPTY;
        bomb->active = 0;
    }
}

static Bomb* tb_bomb_at(TileBlast* env, int row, int col) {
    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* bomb = &env->bombs[i];
        if (bomb->active && bomb->row == row && bomb->col == col) return bomb;
    }
    return NULL;
}

static int tb_bombs_owned(TileBlast* env, int owner) {
    int owned_count = 0;
    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* bomb = &env->bombs[i];
        if (bomb->active && bomb->owner == owner) owned_count++;
    }
    return owned_count;
}

static void tb_mark_blast(TileBlast* env, int row, int col) {
    if (!tb_in_bounds(env, row, col)) return;
    env->blast_timer[tb_idx(env, row, col)] = BLAST_TIME;
}

static void tb_damage_tile(TileBlast* env, int row, int col, int* agent_hit) {
    Agent* agent = &env->agents[0];
    if (*agent_hit == 0 && agent->alive && agent->row == row && agent->col == col) {
        *agent_hit = 1;
        agent->alive = 0;
    }
    for (int i = 0; i < ENEMY_COUNT; i++) {
        Enemy* enemy = &env->enemies[i];
        if (enemy->alive && enemy->row == row && enemy->col == col) enemy->alive = 0;
    }
}

static void tb_explode_bomb(TileBlast* env, Bomb* bomb, int* agent_hit) {
    if (!bomb->active) return;
    bomb->active = 0;

    int center_index = tb_idx(env, bomb->row, bomb->col);
    if (env->grid[center_index] == TILE_BOMB) env->grid[center_index] = TILE_EMPTY;

    tb_mark_blast(env, bomb->row, bomb->col);
    tb_damage_tile(env, bomb->row, bomb->col, agent_hit);

    static const int direction_steps[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    for (int direction_index = 0; direction_index < 4; direction_index++) {
        int row_step = direction_steps[direction_index][0];
        int col_step = direction_steps[direction_index][1];
        for (int step = 1; step <= bomb->range; step++) {
            int blast_row = bomb->row + row_step * step;
            int blast_col = bomb->col + col_step * step;
            if (!tb_in_bounds(env, blast_row, blast_col)) break;

            unsigned char tile = env->grid[tb_idx(env, blast_row, blast_col)];
            if (tile == TILE_HARD) break;

            tb_mark_blast(env, blast_row, blast_col);
            tb_damage_tile(env, blast_row, blast_col, agent_hit);

            if (tile == TILE_SOFT) {
                env->grid[tb_idx(env, blast_row, blast_col)] = TILE_EMPTY;
                break;
            }
        }
    }
}

static void tb_update_bombs(TileBlast* env, int* agent_hit) {
    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* bomb = &env->bombs[i];
        if (!bomb->active) continue;
        bomb->timer -= 1;
        if (bomb->timer <= 0) tb_explode_bomb(env, bomb, agent_hit);
    }
}

static void tb_decay_blasts(TileBlast* env) {
    int cells = env->width * env->height;
    for (int i = 0; i < cells; i++) {
        if (env->blast_timer[i] > 0) env->blast_timer[i] -= 1;
    }
}

static int tb_place_bomb(TileBlast* env) {
    Agent* agent = &env->agents[0];
    if (!agent->alive) return 0;
    if (tb_bombs_owned(env, 0) >= agent->bombs_max) return 0;
    if (tb_bomb_at(env, agent->row, agent->col)) return 0;

    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* bomb = &env->bombs[i];
        if (!bomb->active) {
            bomb->active = 1;
            bomb->row = agent->row;
            bomb->col = agent->col;
            bomb->owner = 0;
            bomb->timer = BOMB_TIMER;
            bomb->range = agent->range;
            env->grid[tb_idx(env, bomb->row, bomb->col)] = TILE_BOMB;
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
    Enemy* enemy = &env->enemies[enemy_idx];
    enemy->row = row;
    enemy->col = col;
    enemy->alive = 1;
    enemy->dir = dir;
    enemy->horizontal = horizontal;

    if (horizontal) {
        enemy->min_pos = 1;
        enemy->max_pos = env->width - 2;
        for (int col_idx = enemy->min_pos; col_idx <= enemy->max_pos; col_idx++) {
            int index = tb_idx(env, row, col_idx);
            if (env->grid[index] == TILE_SOFT) env->grid[index] = TILE_EMPTY;
        }
    } else {
        enemy->min_pos = 1;
        enemy->max_pos = env->height - 2;
        for (int row_idx = enemy->min_pos; row_idx <= enemy->max_pos; row_idx++) {
            int index = tb_idx(env, row_idx, col);
            if (env->grid[index] == TILE_SOFT) env->grid[index] = TILE_EMPTY;
        }
    }
}

static void tb_move_enemies(TileBlast* env) {
    if ((env->tick % ENEMY_MOVE_INTERVAL) != 0) return;

    for (int i = 0; i < ENEMY_COUNT; i++) {
        Enemy* enemy = &env->enemies[i];
        if (!enemy->alive) continue;

        int next_row = enemy->row;
        int next_col = enemy->col;
        if (enemy->horizontal) {
            next_col = enemy->col + enemy->dir;
            if (next_col < enemy->min_pos || next_col > enemy->max_pos ||
                !tb_enemy_walkable(env, next_row, next_col) ||
                tb_enemy_at(env, next_row, next_col, i)) {
                enemy->dir = -enemy->dir;
                next_col = enemy->col + enemy->dir;
            }
        } else {
            next_row = enemy->row + enemy->dir;
            if (next_row < enemy->min_pos || next_row > enemy->max_pos ||
                !tb_enemy_walkable(env, next_row, next_col) ||
                tb_enemy_at(env, next_row, next_col, i)) {
                enemy->dir = -enemy->dir;
                next_row = enemy->row + enemy->dir;
            }
        }

        if (tb_enemy_walkable(env, next_row, next_col) && !tb_enemy_at(env, next_row, next_col, i)) {
            enemy->row = next_row;
            enemy->col = next_col;
        }
    }
}

/* --- Map generation / spawn -------------------------------------------- */
static int tb_is_spawn_clear(TileBlast* env, int row, int col) {
    int spawn_row = env->agents[0].row;
    int spawn_col = env->agents[0].col;
    if (tb_manhattan(row, col, spawn_row, spawn_col) <= 1) return 1;
    if (tb_manhattan(row, col, env->goal_row, env->goal_col) <= 1) return 1;
    return 0;
}

static void tb_sample_goal(TileBlast* env) {
    for (int attempt = 0; attempt < TB_SAMPLE_ATTEMPTS; attempt++) {
        int goal_row = 1 + (rand() % (env->height - 2));
        int goal_col = 1 + (rand() % (env->width - 2));
        if ((goal_row <= 2) && (goal_col <= 2)) continue;
        if ((goal_row % 2 == 0) && (goal_col % 2 == 0)) continue;
        env->goal_row = goal_row;
        env->goal_col = goal_col;
        return;
    }
    env->goal_row = env->height - 2;
    env->goal_col = env->width - 2;
}

static void tb_sample_agent(TileBlast* env, int* out_r, int* out_c) {
    for (int attempt = 0; attempt < TB_SAMPLE_ATTEMPTS; attempt++) {
        int agent_row = 1 + (rand() % (env->height - 2));
        int agent_col = 1 + (rand() % (env->width - 2));
        if ((agent_row % 2 == 0) && (agent_col % 2 == 0)) continue;
        if (tb_manhattan(agent_row, agent_col, env->goal_row, env->goal_col) <= 2) continue;
        *out_r = agent_row;
        *out_c = agent_col;
        return;
    }
    *out_r = 1;
    *out_c = 1;
}

static int tb_reserve_enemy_tile(TileBlast* env, int* out_r, int* out_c) {
    for (int attempt = 0; attempt < TB_SAMPLE_ATTEMPTS; attempt++) {
        int enemy_row = 1 + (rand() % (env->height - 2));
        int enemy_col = 1 + (rand() % (env->width - 2));
        if (enemy_row == env->agents[0].row && enemy_col == env->agents[0].col) continue;
        if (enemy_row == env->goal_row && enemy_col == env->goal_col) continue;
        if (env->grid[tb_idx(env, enemy_row, enemy_col)] == TILE_HARD) continue;
        if (tb_enemy_at(env, enemy_row, enemy_col, -1)) continue;
        *out_r = enemy_row;
        *out_c = enemy_col;
        return 1;
    }
    return 0;
}

static void tb_generate_map(TileBlast* env) {
    for (int row = 0; row < env->height; row++) {
        for (int col = 0; col < env->width; col++) {
            unsigned char* cell = &env->grid[tb_idx(env, row, col)];
            if (row == 0 || col == 0 || row == env->height - 1 || col == env->width - 1) { *cell = TILE_HARD; continue; }
            if ((row % 2 == 0) && (col % 2 == 0)) { *cell = TILE_HARD; continue; }
            if (tb_is_spawn_clear(env, row, col)) { *cell = TILE_EMPTY; continue; }
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
        unsigned char* observation = env->observations + agent_index * bytes_per_agent;
        unsigned char* grid_obs = observation + env->obs_size * OBS_GRID;
        unsigned char* scalar_obs = observation + env->obs_size * OBS_CHANNELS;
        Agent* agent = &env->agents[agent_index];

        int grid_write_index = 0;
        for (int row = 0; row < env->height; row++) {
            for (int col = 0; col < env->width; col++) {
                unsigned char tile = env->grid[tb_idx(env, row, col)];
                if (env->blast_timer[tb_idx(env, row, col)] > 0) tile = TILE_BLAST;
                if (tb_enemy_at(env, row, col, -1)) tile = TILE_ENEMY;
                if (row == env->goal_row && col == env->goal_col) tile = TILE_GOAL;
                if (agent->alive && agent->row == row && agent->col == col) tile = TILE_AGENT0;
                grid_obs[grid_write_index++] = tile;
            }
        }

        int dist_goal = tb_manhattan(agent->row, agent->col, env->goal_row, env->goal_col);
        int dist_enemy = TB_U8_MAX;
        for (int i = 0; i < ENEMY_COUNT; i++) {
            Enemy* enemy = &env->enemies[i];
            if (!enemy->alive) continue;
            int enemy_distance = tb_manhattan(agent->row, agent->col, enemy->row, enemy->col);
            if (enemy_distance < dist_enemy) dist_enemy = enemy_distance;
        }
        int dist_bomb = TB_U8_MAX;
        int min_bomb_timer = TB_U8_MAX;
        for (int i = 0; i < env->max_bombs; i++) {
            Bomb* bomb = &env->bombs[i];
            if (!bomb->active) continue;
            int bomb_distance = tb_manhattan(agent->row, agent->col, bomb->row, bomb->col);
            if (bomb_distance < dist_bomb) dist_bomb = bomb_distance;
            if (bomb->timer < min_bomb_timer) min_bomb_timer = bomb->timer;
        }
        int owned_bombs = tb_bombs_owned(env, agent_index);
        int available_bombs = agent->bombs_max - owned_bombs;
        if (available_bombs < 0) available_bombs = 0;
        int can_place_bomb = (agent->alive && owned_bombs < agent->bombs_max &&
            !tb_bomb_at(env, agent->row, agent->col)) ? 1 : 0;
        int goal_hint = tb_goal_action_hint(env, agent->row, agent->col);

        if (env->scalar_size >= 1) scalar_obs[0] = tb_clamp_u8(dist_goal);
        if (env->scalar_size >= 2) scalar_obs[1] = tb_clamp_u8(dist_enemy);
        if (env->scalar_size >= 3) scalar_obs[2] = tb_clamp_u8(dist_bomb);
        if (env->scalar_size >= 4) scalar_obs[3] = tb_clamp_u8(min_bomb_timer);
        if (env->scalar_size >= 5) scalar_obs[4] = tb_clamp_u8(available_bombs);
        if (env->scalar_size >= 6) scalar_obs[5] = tb_clamp_u8(can_place_bomb);
        if (env->scalar_size >= 7) scalar_obs[6] = (unsigned char)(agent->alive ? 1 : 0);
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

    int agent_row = 1;
    int agent_col = 1;
    tb_sample_agent(env, &agent_row, &agent_col);

    env->agents[0] = (Agent){
        .row = agent_row, .col = agent_col,
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
        int enemy_row = 1;
        int enemy_col = 1;
        if (!tb_reserve_enemy_tile(env, &enemy_row, &enemy_col)) {
            enemy_row = 1 + i;
            enemy_col = env->width - 2 - i;
            if (enemy_row >= env->height - 1) enemy_row = env->height - 2;
            if (enemy_col <= 0) enemy_col = 1;
        }
        int horizontal = rand() & 1;
        int direction = (rand() & 1) ? 1 : -1;
        tb_init_enemy_patrol(env, i, enemy_row, enemy_col, horizontal, direction);
    }

    /* Update observation buffer for policy input */
    tb_update_observations(env);
    if (env->rewards) env->rewards[0] = 0.0f;
    if (env->terminals) env->terminals[0] = 0;
}

static void tb_resolve_move(TileBlast* env, int action) {
    Agent* agent = &env->agents[0];
    if (!agent->alive) return;
    if ((unsigned)action > ACT_BOMB) return;

    int row_delta = 0;
    int col_delta = 0;
    switch (action) {
        case ACT_UP:    row_delta = -1; break;
        case ACT_DOWN:  row_delta = 1;  break;
        case ACT_LEFT:  col_delta = -1; break;
        case ACT_RIGHT: col_delta = 1;  break;
        default: return; /* ACT_NOOP / ACT_BOMB do not move */
    }

    int next_row = agent->row + row_delta;
    int next_col = agent->col + col_delta;
    if (!tb_is_solid(env, next_row, next_col)) {
        agent->row = next_row;
        agent->col = next_col;
    }
}

static int tb_read_action(const TileBlast* env) {
    if (!env->actions) return ACT_NOOP;
    int action = env->actions[0];
    if (action < ACT_NOOP || action > ACT_BOMB) return ACT_NOOP;
    return action;
}

static int tb_agent_on_goal(const TileBlast* env) {
    const Agent* agent = &env->agents[0];
    return (agent->alive && agent->row == env->goal_row && agent->col == env->goal_col);
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

    Agent* agent = &env->agents[0];
    int prev_goal_dist = tb_manhattan(agent->row, agent->col, env->goal_row, env->goal_col);

    env->tick += 1;
    tb_decay_blasts(env);

    int action = tb_read_action(env);

    if (action >= ACT_UP && action <= ACT_RIGHT) tb_resolve_move(env, action);
    if (action == ACT_BOMB) tb_place_bomb(env);
    int touched_enemy_after_player_move = tb_enemy_at(env, agent->row, agent->col, -1);

    tb_move_enemies(env);

    int agent_hit = 0;
    tb_update_bombs(env, &agent_hit);

    int done = 0;
    reward -= STEP_PENALTY;
    int new_goal_dist = tb_manhattan(agent->row, agent->col, env->goal_row, env->goal_col);
    reward += (float)(prev_goal_dist - new_goal_dist) * GOAL_PROGRESS_REWARD;

    if (!agent->alive) {
        done = 1;
    } else if (touched_enemy_after_player_move) {
        agent->alive = 0;
        reward -= ENEMY_COLLISION_PENALTY;
        done = 1;
    } else if (tb_enemy_at(env, agent->row, agent->col, -1)) {
        agent->alive = 0;
        reward -= ENEMY_COLLISION_PENALTY;
        done = 1;
    } else if (agent_hit) {
        agent->alive = 0;
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
        tb_finish_episode(env, agent);
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
