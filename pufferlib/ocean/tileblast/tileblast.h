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
#define DEFAULT_DISTANCE_REWARD_INTERVAL 5

/* Keep these for API compatibility. */
#define OBS_SCALARS 13
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
#define STEP_PENALTY 0.005f
#define GOAL_PROGRESS_REWARD 0.12f
#define INTERVAL_REVERSE_FACTOR 1.25f
#define INTERVAL_NO_PROGRESS_PENALTY 0.04f
#define SOFT_BREAK_REWARD 0.06f
#define ENEMY_KILL_REWARD 0.25f
#define SMART_BOMB_REWARD 0.00f
#define WASTED_BOMB_PENALTY 0.05f
#define FAILED_BOMB_PENALTY 0.03f
#define DANGER_PENALTY 0.01f
#define ESCAPE_DANGER_REWARD 0.04f
#define GOAL_REWARD 1.00f
#define ENEMY_COLLISION_PENALTY 0.90f
#define BLAST_DEATH_PENALTY 0.90f
#define TIMEOUT_PENALTY 0.60f
#define TB_SAMPLE_ATTEMPTS 256
#define TB_U8_MAX 255
#define TB_OUTCOME_NONE 0
#define TB_OUTCOME_WIN 1
#define TB_OUTCOME_DEAD 2
#define TB_OUTCOME_TIMEOUT 3
#define TB_OUTCOME_BANNER_TICKS 24

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
    float wins;
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
    unsigned char* truncations;  /* optional; timeout signal */

    /* Legacy config fields (kept for callers) */
    int width;
    int height;
    int agent_speed; /* ignored; movement is 1 tile per step */
    int num_agents;  /* forced to 1 */
    int max_steps;
    int vision;      /* 0: full map obs, >0: local square window */
    int distance_reward_interval; /* apply path progress reward every n steps */
    int obs_size;    /* grid cells per observation */
    int scalar_size; /* kept: OBS_SCALARS */

    /* Runtime */
    int tick;
    int goal_row, goal_col;
    float ep_return;
    int last_progress_dist;
    int last_outcome;
    int outcome_banner_ticks;

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

static int tb_is_solid(TileBlast* env, int row, int col);

static int tb_shortest_path_to_goal(TileBlast* env, int start_row, int start_col) {
    if (!tb_in_bounds(env, start_row, start_col)) return -1;
    if (start_row == env->goal_row && start_col == env->goal_col) return 0;
    if (tb_is_solid(env, start_row, start_col)) return -1;

    int cells = env->width * env->height;
    int* dist = (int*)malloc((size_t)cells * sizeof(int));
    int* queue = (int*)malloc((size_t)cells * sizeof(int));
    if (!dist || !queue) {
        free(dist);
        free(queue);
        return -1;
    }

    for (int i = 0; i < cells; i++) dist[i] = -1;
    int head = 0;
    int tail = 0;
    int start_index = tb_idx(env, start_row, start_col);
    dist[start_index] = 0;
    queue[tail++] = start_index;

    static const int direction_steps[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    while (head < tail) {
        int current = queue[head++];
        int row = current / env->width;
        int col = current % env->width;
        int current_dist = dist[current];

        for (int i = 0; i < 4; i++) {
            int next_row = row + direction_steps[i][0];
            int next_col = col + direction_steps[i][1];
            if (!tb_in_bounds(env, next_row, next_col)) continue;
            if (tb_is_solid(env, next_row, next_col)) continue;
            int next_index = tb_idx(env, next_row, next_col);
            if (dist[next_index] != -1) continue;
            dist[next_index] = current_dist + 1;
            if (next_row == env->goal_row && next_col == env->goal_col) {
                int goal_dist = dist[next_index];
                free(dist);
                free(queue);
                return goal_dist;
            }
            queue[tail++] = next_index;
        }
    }

    free(dist);
    free(queue);
    return -1;
}

static int tb_goal_action_hint_path(TileBlast* env, int row, int col) {
    if (row == env->goal_row && col == env->goal_col) return ACT_NOOP;

    int best_action = ACT_NOOP;
    int best_distance = 1 << 30;
    static const int direction_steps[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
    static const int direction_actions[4] = {ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT};

    for (int i = 0; i < 4; i++) {
        int next_row = row + direction_steps[i][0];
        int next_col = col + direction_steps[i][1];
        if (tb_is_solid(env, next_row, next_col)) continue;
        int d = tb_shortest_path_to_goal(env, next_row, next_col);
        if (d < 0) continue;
        if (d < best_distance) {
            best_distance = d;
            best_action = direction_actions[i];
        }
    }

    if (best_action != ACT_NOOP) return best_action;
    return tb_goal_action_hint(env, row, col);
}

static void tb_goal_step_vector(TileBlast* env, int row, int col, int* row_step, int* col_step) {
    *row_step = 0;
    *col_step = 0;
    int action = tb_goal_action_hint_path(env, row, col);
    switch (action) {
        case ACT_UP: *row_step = -1; break;
        case ACT_DOWN: *row_step = 1; break;
        case ACT_LEFT: *col_step = -1; break;
        case ACT_RIGHT: *col_step = 1; break;
        default: break;
    }
}

static unsigned char tb_encode_signed_unit(int value) {
    if (value < -1) value = -1;
    if (value > 1) value = 1;
    return (unsigned char)(value + 1); /* -1,0,1 -> 0,1,2 */
}

static int tb_count_alive_enemies(TileBlast* env) {
    int alive_enemies = 0;
    for (int i = 0; i < ENEMY_COUNT; i++) {
        if (env->enemies[i].alive) alive_enemies++;
    }
    return alive_enemies;
}

static int tb_count_soft_tiles(TileBlast* env) {
    int soft_tiles = 0;
    int cells = env->width * env->height;
    for (int i = 0; i < cells; i++) {
        if (env->grid[i] == TILE_SOFT) soft_tiles++;
    }
    return soft_tiles;
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

static void tb_count_bomb_targets(TileBlast* env, int row, int col, int range,
    int* soft_targets, int* enemy_targets) {
    if (soft_targets) *soft_targets = 0;
    if (enemy_targets) *enemy_targets = 0;
    static const int direction_steps[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};

    for (int direction_index = 0; direction_index < 4; direction_index++) {
        int row_step = direction_steps[direction_index][0];
        int col_step = direction_steps[direction_index][1];
        for (int step = 1; step <= range; step++) {
            int scan_row = row + row_step * step;
            int scan_col = col + col_step * step;
            if (!tb_in_bounds(env, scan_row, scan_col)) break;

            unsigned char tile = env->grid[tb_idx(env, scan_row, scan_col)];
            if (tile == TILE_HARD) break;

            if (enemy_targets && tb_enemy_at(env, scan_row, scan_col, -1)) {
                (*enemy_targets)++;
            }

            if (tile == TILE_SOFT) {
                if (soft_targets) (*soft_targets)++;
                break;
            }
        }
    }
}

static int tb_cell_in_blast_path(TileBlast* env, int row, int col) {
    for (int i = 0; i < env->max_bombs; i++) {
        Bomb* bomb = &env->bombs[i];
        if (!bomb->active) continue;
        if (bomb->row == row && bomb->col == col) return 1;

        if (bomb->row == row) {
            int delta = col - bomb->col;
            int abs_delta = delta < 0 ? -delta : delta;
            if (abs_delta > bomb->range) continue;
            int col_step = (delta > 0) ? 1 : -1;
            int blocked = 0;
            for (int c = bomb->col + col_step; c != col; c += col_step) {
                unsigned char tile = env->grid[tb_idx(env, row, c)];
                if (tile == TILE_HARD) { blocked = 1; break; }
                if (tile == TILE_SOFT) { blocked = 1; break; }
            }
            if (!blocked) {
                unsigned char tile = env->grid[tb_idx(env, row, col)];
                if (tile != TILE_HARD) return 1;
            }
        } else if (bomb->col == col) {
            int delta = row - bomb->row;
            int abs_delta = delta < 0 ? -delta : delta;
            if (abs_delta > bomb->range) continue;
            int row_step = (delta > 0) ? 1 : -1;
            int blocked = 0;
            for (int r = bomb->row + row_step; r != row; r += row_step) {
                unsigned char tile = env->grid[tb_idx(env, r, col)];
                if (tile == TILE_HARD) { blocked = 1; break; }
                if (tile == TILE_SOFT) { blocked = 1; break; }
            }
            if (!blocked) {
                unsigned char tile = env->grid[tb_idx(env, row, col)];
                if (tile != TILE_HARD) return 1;
            }
        }
    }
    return 0;
}

static int tb_count_safe_moves(TileBlast* env, int row, int col) {
    static const int direction_steps[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    int safe_moves = 0;
    for (int i = 0; i < 4; i++) {
        int next_row = row + direction_steps[i][0];
        int next_col = col + direction_steps[i][1];
        if (tb_is_solid(env, next_row, next_col)) continue;
        if (!tb_cell_in_blast_path(env, next_row, next_col)) safe_moves++;
    }
    return safe_moves;
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

static inline float tb_clip_reward(float reward) {
    if (reward > 1.0f) return 1.0f;
    if (reward < -1.0f) return -1.0f;
    return reward;
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
        int obs_row_start = 0;
        int obs_row_end = env->height - 1;
        int obs_col_start = 0;
        int obs_col_end = env->width - 1;
        if (env->vision > 0) {
            obs_row_start = agent->row - env->vision;
            obs_row_end = agent->row + env->vision;
            obs_col_start = agent->col - env->vision;
            obs_col_end = agent->col + env->vision;
        }

        for (int row = obs_row_start; row <= obs_row_end; row++) {
            for (int col = obs_col_start; col <= obs_col_end; col++) {
                unsigned char tile = TILE_HARD;
                if (tb_in_bounds(env, row, col)) {
                    tile = env->grid[tb_idx(env, row, col)];
                    if (env->blast_timer[tb_idx(env, row, col)] > 0) tile = TILE_BLAST;
                    if (tb_enemy_at(env, row, col, -1)) tile = TILE_ENEMY;
                    if (row == env->goal_row && col == env->goal_col) tile = TILE_GOAL;
                    if (agent->alive && agent->row == row && agent->col == col) tile = TILE_AGENT0;
                }
                if (grid_write_index < env->obs_size) grid_obs[grid_write_index++] = tile;
            }
        }

        int max_dist = env->width + env->height;
        int dist_goal = tb_shortest_path_to_goal(env, agent->row, agent->col);
        if (dist_goal < 0) dist_goal = max_dist;
        int dist_enemy = max_dist;
        for (int i = 0; i < ENEMY_COUNT; i++) {
            Enemy* enemy = &env->enemies[i];
            if (!enemy->alive) continue;
            int enemy_distance = tb_manhattan(agent->row, agent->col, enemy->row, enemy->col);
            if (enemy_distance < dist_enemy) dist_enemy = enemy_distance;
        }
        int dist_bomb = max_dist;
        int min_bomb_timer = BOMB_TIMER + 1;
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
        int goal_row_step = 0;
        int goal_col_step = 0;
        tb_goal_step_vector(env, agent->row, agent->col, &goal_row_step, &goal_col_step);
        int danger_now = tb_cell_in_blast_path(env, agent->row, agent->col);
        int safe_moves = tb_count_safe_moves(env, agent->row, agent->col);
        int bomb_soft_targets = 0;
        int bomb_enemy_targets = 0;
        tb_count_bomb_targets(env, agent->row, agent->col, agent->range,
            &bomb_soft_targets, &bomb_enemy_targets);

        if (env->scalar_size >= 1) scalar_obs[0] = tb_clamp_u8(dist_goal);
        if (env->scalar_size >= 2) scalar_obs[1] = tb_clamp_u8(dist_enemy);
        if (env->scalar_size >= 3) scalar_obs[2] = tb_clamp_u8(dist_bomb);
        if (env->scalar_size >= 4) scalar_obs[3] = tb_clamp_u8(min_bomb_timer);
        if (env->scalar_size >= 5) scalar_obs[4] = tb_clamp_u8(available_bombs);
        if (env->scalar_size >= 6) scalar_obs[5] = tb_clamp_u8(can_place_bomb);
        if (env->scalar_size >= 7) scalar_obs[6] = (unsigned char)(agent->alive ? 1 : 0);
        if (env->scalar_size >= 8) scalar_obs[7] = tb_encode_signed_unit(goal_row_step);
        if (env->scalar_size >= 9) scalar_obs[8] = tb_clamp_u8(danger_now);
        if (env->scalar_size >= 10) scalar_obs[9] = tb_clamp_u8(safe_moves);
        if (env->scalar_size >= 11) scalar_obs[10] = tb_clamp_u8(bomb_soft_targets);
        if (env->scalar_size >= 12) scalar_obs[11] = tb_clamp_u8(bomb_enemy_targets);
        if (env->scalar_size >= 13) scalar_obs[12] = tb_encode_signed_unit(goal_col_step);
    }
}

static void tb_reset_episode(TileBlast* env, int clear_outputs) {
    if (!env) return;

    env->tick = 0;
    env->ep_return = 0.0f;
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

    env->last_progress_dist = tb_shortest_path_to_goal(
        env, env->agents[0].row, env->agents[0].col
    );
    if (env->last_progress_dist < 0) {
        env->last_progress_dist = env->width + env->height;
    }

    tb_update_observations(env);
    if (clear_outputs) {
        if (env->rewards) env->rewards[0] = 0.0f;
        if (env->terminals) env->terminals[0] = 0;
        if (env->truncations) env->truncations[0] = 0;
    }
}

/* --- API impl ----------------------------------------------------------- */
void init(TileBlast* env) {
    if (!env) return;

    if (env->width <= 0) env->width = DEFAULT_WIDTH;
    if (env->height <= 0) env->height = DEFAULT_HEIGHT;
    if (env->max_steps <= 0) env->max_steps = DEFAULT_MAX_STEPS;
    if (env->distance_reward_interval <= 0) {
        env->distance_reward_interval = DEFAULT_DISTANCE_REWARD_INTERVAL;
    }

    /* Keep legacy fields consistent */
    env->num_agents = 1;
    if (env->agent_speed <= 0) env->agent_speed = DEFAULT_AGENT_SPEED;
    if (env->vision < 0) env->vision = DEFAULT_VISION;

    if (env->vision > 0) {
        int obs_side = env->vision * 2 + 1;
        env->obs_size = obs_side * obs_side;
    } else {
        env->obs_size = env->width * env->height;
    }
    env->scalar_size = OBS_SCALARS;

    int cells = env->width * env->height;
    env->grid = (unsigned char*)calloc((size_t)cells, sizeof(unsigned char));
    env->blast_timer = (unsigned char*)calloc((size_t)cells, sizeof(unsigned char));

    env->max_bombs = env->num_agents * MAX_BOMBS_PER_AGENT + 4;
    env->bombs = (Bomb*)calloc((size_t)env->max_bombs, sizeof(Bomb));

    env->tick = 0;
    env->goal_row = env->height - 2;
    env->goal_col = env->width - 2;
    env->ep_return = 0.0f;
    env->last_progress_dist = env->width + env->height;
    env->last_outcome = TB_OUTCOME_NONE;
    env->outcome_banner_ticks = 0;

    /* Clear log (compat) */
    memset(&env->log, 0, sizeof(env->log));
}

void c_reset(TileBlast* env) {
    tb_reset_episode(env, 1);
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

static void tb_finish_episode(TileBlast* env, Agent* agent, int timed_out) {
    int reached_goal = tb_agent_on_goal(env);
    if (env->terminals) env->terminals[0] = (timed_out ? 0 : 1);
    if (env->truncations) env->truncations[0] = (timed_out ? 1 : 0);
    env->log.score += env->ep_return;
    env->log.episode_return += env->ep_return;
    env->log.episode_length += (float)env->tick;
    env->log.n += 1.0f;
    if (reached_goal) {
        env->log.perf += 1.0f;
        env->log.wins += 1.0f;
    }
    if (env->tick >= env->max_steps) env->log.timeouts += 1.0f;
    if (!agent->alive) env->log.deaths += 1.0f;
    if (reached_goal) {
        env->last_outcome = TB_OUTCOME_WIN;
        env->outcome_banner_ticks = TB_OUTCOME_BANNER_TICKS;
    } else if (timed_out) {
        env->last_outcome = TB_OUTCOME_TIMEOUT;
        env->outcome_banner_ticks = TB_OUTCOME_BANNER_TICKS;
    } else if (!agent->alive) {
        env->last_outcome = TB_OUTCOME_DEAD;
        env->outcome_banner_ticks = TB_OUTCOME_BANNER_TICKS;
    } else {
        env->last_outcome = TB_OUTCOME_NONE;
        env->outcome_banner_ticks = 0;
    }
    tb_reset_episode(env, 0);
}

void c_step(TileBlast* env) {
    if (!env) return;

    if (env->outcome_banner_ticks > 0) {
        env->outcome_banner_ticks -= 1;
        if (env->outcome_banner_ticks == 0) env->last_outcome = TB_OUTCOME_NONE;
    }

    /* legacy outputs: always write safe defaults */
    float reward = 0.0f;
    if (env->rewards) env->rewards[0] = reward;
    if (env->terminals) env->terminals[0] = 0;
    if (env->truncations) env->truncations[0] = 0;

    Agent* agent = &env->agents[0];
    int prev_goal_dist = env->last_progress_dist;
    if (prev_goal_dist < 0) prev_goal_dist = env->width + env->height;
    int prev_alive_enemies = tb_count_alive_enemies(env);
    int prev_soft_tiles = tb_count_soft_tiles(env);
    int was_in_danger = tb_cell_in_blast_path(env, agent->row, agent->col);

    env->tick += 1;
    tb_decay_blasts(env);

    int action = tb_read_action(env);
    int bomb_soft_targets = 0;
    int bomb_enemy_targets = 0;
    if (action == ACT_BOMB) {
        tb_count_bomb_targets(env, agent->row, agent->col, agent->range,
            &bomb_soft_targets, &bomb_enemy_targets);
    }

    if (action >= ACT_UP && action <= ACT_RIGHT) tb_resolve_move(env, action);
    int placed_bomb = 0;
    if (action == ACT_BOMB) placed_bomb = tb_place_bomb(env);
    int touched_enemy_after_player_move = tb_enemy_at(env, agent->row, agent->col, -1);

    tb_move_enemies(env);

    int agent_hit = 0;
    tb_update_bombs(env, &agent_hit);

    int done = 0;
    int timed_out = 0;
    int reached_goal = 0;
    reward -= STEP_PENALTY;
    int new_goal_dist = tb_shortest_path_to_goal(env, agent->row, agent->col);
    if (new_goal_dist < 0) new_goal_dist = env->width + env->height;
    int alive_enemies = tb_count_alive_enemies(env);
    int soft_tiles = tb_count_soft_tiles(env);
    int killed_enemies = prev_alive_enemies - alive_enemies;
    int broken_soft_tiles = prev_soft_tiles - soft_tiles;
    if (killed_enemies > 0) reward += (float)killed_enemies * ENEMY_KILL_REWARD;
    if (broken_soft_tiles > 0) reward += (float)broken_soft_tiles * SOFT_BREAK_REWARD;
    if (placed_bomb) {
        if ((bomb_soft_targets + bomb_enemy_targets) > 0) {
            reward += SMART_BOMB_REWARD;
        } else {
            reward -= WASTED_BOMB_PENALTY;
        }
    } else if (action == ACT_BOMB) {
        reward -= FAILED_BOMB_PENALTY;
    }
    int in_danger_now = tb_cell_in_blast_path(env, agent->row, agent->col);
    if (in_danger_now) reward -= DANGER_PENALTY;
    if (was_in_danger && !in_danger_now) reward += ESCAPE_DANGER_REWARD;

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
        reached_goal = 1;
        reward += GOAL_REWARD;
        done = 1;
    } else if (env->tick >= env->max_steps) {
        reward -= TIMEOUT_PENALTY;
        timed_out = 1;
        done = 1;
    }

    int progress_tick = (env->tick % env->distance_reward_interval) == 0;
    if (progress_tick || done) {
        int progress_delta = prev_goal_dist - new_goal_dist;
        if (reached_goal) {
            if (progress_delta > 0) reward += (float)progress_delta * GOAL_PROGRESS_REWARD;
        } else if (progress_delta > 0) {
            reward += (float)progress_delta * GOAL_PROGRESS_REWARD;
        } else if (progress_delta < 0) {
            reward -= (float)(-progress_delta) * GOAL_PROGRESS_REWARD * INTERVAL_REVERSE_FACTOR;
        } else {
            reward -= INTERVAL_NO_PROGRESS_PENALTY;
        }
        env->last_progress_dist = new_goal_dist;
    }

    reward = tb_clip_reward(reward);
    if (env->rewards) env->rewards[0] = reward;
    env->ep_return += reward;

    /* Minimal: auto-reset immediately */
    if (done) {
        tb_finish_episode(env, agent, timed_out);
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
