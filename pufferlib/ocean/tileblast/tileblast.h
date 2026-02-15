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
#define SCORE_FOR_GOAL 15.0f
#define TIMEOUT_PENALTY 1.0f
#define ENEMY_KILL_REWARD 1.0f
#define BLAST_DANGER_PENALTY 0.2f
#define BLAST_ESCAPE_REWARD 0.15f
#define BOMB_PROX_THRESHOLD 3
#define BOMB_PROX_PENALTY 0.05f
#define BOMB_ON_TILE_PENALTY 5.0f
#define BOMB_ON_TILE_DANGER_TIMER 3
#define BOMB_PLANT_BASE_REWARD 0.05f
#define TACTICAL_BOMB_REWARD 0.25f
#define BOMB_NEAR_ENEMY_RANGE 3
#define BOMB_NEAR_ENEMY_REWARD 0.10f
#define MISSED_TACTICAL_BOMB_PENALTY 0.05f
#define BLOCKER_BOMB_COMMIT_DANGER_TIMER 1
#define ENEMY_BLOCK_RANGE 3
#define ENEMY_PATH_BLOCK_PENALTY 0.08f
#define ENEMY_PATH_AVOID_REWARD 0.03f
#define STALL_PENALTY 0.02f
#define NOOP_PENALTY 0.05f
#define MOVE_REWARD 0.0f
#define STEP_PENALTY 0.01f
#define GOAL_PROGRESS_REWARD 0.35f
#define VISIT_HEAT_DECAY 0.95f
#define VISIT_HEAT_DEPOSIT 1.0f
#define VISIT_HEAT_PENALTY 0.002f
#define STUCK_NO_PROGRESS_STEPS 6
#define STUCK_HEAT_THRESHOLD 2.5f
#define DETOUR_MAX_MANHATTAN_INCREASE 3

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
    float n; /* Episode counter expected by shared env_binding.h aggregation. */
    float timeouts;
    float deaths;
} Log;

/* Learning agent state. */
typedef struct {
    int row;
    int col;
    int alive;
    int bombs_max;
    int range;
    int lives;
    int invuln;
} Agent;

/* Bomb slot state (kept for observations/rendering compatibility). */
typedef struct {
    int row;
    int col;
    int owner;
    int timer;
    int range;
    int active;
} Bomb;

/* Patrolling enemy state. */
typedef struct {
    int row;
    int col;
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
    int goal_row;
    int goal_col;
    float episode_return_accum;

    unsigned char* grid;
    unsigned char* blast_timer;
    float* visit_heat;
    Bomb* bombs;
    int max_bombs;
    Enemy enemies[ENEMY_COUNT];
    Agent agents[1];
    float score_points;
    int last_goal_dist;
    int no_progress_steps;
} TileBlast;
/* --- Grid helpers -------------------------------------------------------- */
/* Convert (row, col) to a row-major flat index. */
static inline int idx(TileBlast* env, int row, int col) {
    return row * env->width + col;
}

/* Check whether a coordinate lies inside the map. */
static inline int in_bounds(TileBlast* env, int row, int col) {
    return (row >= 0 && col >= 0 && row < env->height && col < env->width);
}

/* Manhattan distance helper for scalar observations. */
static inline int manhattan_distance(int r0, int c0, int r1, int c1) {
    int delta_row = r0 - r1;
    int delta_col = c0 - c1;
    if (delta_row < 0) delta_row = -delta_row;
    if (delta_col < 0) delta_col = -delta_col;
    return delta_row + delta_col;
}

/* --- Rendering helpers --------------------------------------------------- */
static RenderTexture2D g_static_tex = {0};
static int g_static_ready = 0;
static int g_static_w = 0;
static int g_static_h = 0;
static int g_show_visit_heat = 1;

/* --- Random helpers ------------------------------------------------------ */
static float randf() {
    return (float)rand() / (float)RAND_MAX;
}

/* --- Bomb helpers -------------------------------------------------------- */
/* Reset all bomb slots to inactive. */
static void clear_bombs(TileBlast* env) {
    for (int index = 0; index < env->max_bombs; index++) {
        Bomb* bomb = &env->bombs[index];
        if (bomb->active) {
            env->grid[idx(env, bomb->row, bomb->col)] = TILE_EMPTY;
        }
        bomb->active = 0;
    }
}

/* Return active bomb at tile, otherwise NULL. */
static Bomb* bomb_at(TileBlast* env, int row, int col) {
    for (int index = 0; index < env->max_bombs; index++) {
        Bomb* bomb = &env->bombs[index];
        if (bomb->active && bomb->row == row && bomb->col == col) {
            return bomb;
        }
    }
    return NULL;
}

/* Count active bombs owned by `owner`. */
static int bombs_owned(TileBlast* env, int owner) {
    int count = 0;
    for (int index = 0; index < env->max_bombs; index++) {
        Bomb* bomb = &env->bombs[index];
        if (bomb->active && bomb->owner == owner) {
            count++;
        }
    }
    return count;
}

static int nearest_bomb_distance(TileBlast* env, int row, int col) {
    int best = 255;
    for (int index = 0; index < env->max_bombs; index++) {
        Bomb* bomb = &env->bombs[index];
        if (!bomb->active) continue;
        int bomb_distance = manhattan_distance(row, col, bomb->row, bomb->col);
        if (bomb_distance < best) best = bomb_distance;
    }
    return best;
}

static int place_bomb(TileBlast* env) {
    Agent* agent = &env->agents[0];
    if (!agent->alive) return 0;
    for (int index = 0; index < env->max_bombs; index++) {
        Bomb* bomb = &env->bombs[index];
        if (!bomb->active) {
            bomb->active = 1;
            bomb->row = agent->row;
            bomb->col = agent->col;
            bomb->owner = 0;
            bomb->timer = BOMB_TIMER;
            bomb->range = agent->range;
            env->grid[idx(env, bomb->row, bomb->col)] = TILE_BOMB;
            return 1;
        }
    }
    return 0;
}

static int bomb_has_tactical_target(TileBlast* env, int row, int col, int range) {
    for (int index = 0; index < ENEMY_COUNT; index++) {
        Enemy* enemy = &env->enemies[index];
        if (!enemy->alive) continue;

        if (enemy->row == row) {
            int delta_col = enemy->col - col;
            int dist = delta_col < 0 ? -delta_col : delta_col;
            if (dist > range) continue;
            int step = (delta_col < 0) ? -1 : 1;
            int clear = 1;
            for (int scan_col = col + step; scan_col != enemy->col; scan_col += step) {
                unsigned char tile = env->grid[idx(env, row, scan_col)];
                if (tile == TILE_HARD || tile == TILE_SOFT) {
                    clear = 0;
                    break;
                }
            }
            if (clear) return 1;
        } else if (enemy->col == col) {
            int delta_row = enemy->row - row;
            int dist = delta_row < 0 ? -delta_row : delta_row;
            if (dist > range) continue;
            int step = (delta_row < 0) ? -1 : 1;
            int clear = 1;
            for (int scan_row = row + step; scan_row != enemy->row; scan_row += step) {
                unsigned char tile = env->grid[idx(env, scan_row, col)];
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

static void mark_blast(TileBlast* env, int row, int col) {
    if (!in_bounds(env, row, col)) return;
    int index = idx(env, row, col);
    env->blast_timer[index] = BLAST_TIME;
}

static void damage_tile(TileBlast* env, int row, int col, int* agent_hit, int* enemies_killed) {
    Agent* agent = &env->agents[0];
    if (*agent_hit == 0 && agent->alive && agent->row == row && agent->col == col) {
        *agent_hit = 1;
        agent->alive = 0;
    }
    for (int index = 0; index < ENEMY_COUNT; index++) {
        Enemy* enemy = &env->enemies[index];
        if (enemy->alive && enemy->row == row && enemy->col == col) {
            enemy->alive = 0;
            (*enemies_killed) += 1;
            env->score_points += ENEMY_KILL_REWARD;
        }
    }
}

static void explode_bomb(TileBlast* env, Bomb* bomb, int* agent_hit, int* enemies_killed) {
    if (!bomb->active) return;
    bomb->active = 0;
    int center_idx = idx(env, bomb->row, bomb->col);
    if (env->grid[center_idx] == TILE_BOMB) {
        env->grid[center_idx] = TILE_EMPTY;
    }
    mark_blast(env, bomb->row, bomb->col);
    damage_tile(env, bomb->row, bomb->col, agent_hit, enemies_killed);

    static const int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int direction_index = 0; direction_index < 4; direction_index++) {
        int delta_row = dirs[direction_index][0];
        int delta_col = dirs[direction_index][1];
        for (int range_step = 1; range_step <= bomb->range; range_step++) {
            int scan_row = bomb->row + delta_row * range_step;
            int scan_col = bomb->col + delta_col * range_step;
            if (!in_bounds(env, scan_row, scan_col)) break;
            unsigned char tile = env->grid[idx(env, scan_row, scan_col)];
            if (tile == TILE_HARD) break;
            mark_blast(env, scan_row, scan_col);
            damage_tile(env, scan_row, scan_col, agent_hit, enemies_killed);
            if (tile == TILE_SOFT) {
                env->grid[idx(env, scan_row, scan_col)] = TILE_EMPTY;
                break;
            }
        }
    }
}

static void update_bombs(TileBlast* env, int* agent_hit, int* enemies_killed) {
    for (int index = 0; index < env->max_bombs; index++) {
        Bomb* bomb = &env->bombs[index];
        if (!bomb->active) continue;
        bomb->timer -= 1;
        if (bomb->timer <= 0) {
            explode_bomb(env, bomb, agent_hit, enemies_killed);
        }
    }
}

static void decay_blasts(TileBlast* env) {
    int cells = env->width * env->height;
    for (int index = 0; index < cells; index++) {
        if (env->blast_timer[index] > 0) {
            env->blast_timer[index] -= 1;
        }
    }
}

static void decay_visit_heat(TileBlast* env) {
    int cells = env->width * env->height;
    for (int index = 0; index < cells; index++) {
        env->visit_heat[index] *= VISIT_HEAT_DECAY;
        if (env->visit_heat[index] < 0.0001f) env->visit_heat[index] = 0.0f;
    }
}

/* --- Threat analysis ----------------------------------------------------- */
/* Find earliest bomb timer that can affect (row,col); 255 means none. */
static int min_bomb_timer_affecting(TileBlast* env, int row, int col) {
    int min_timer = 255;
    for (int index = 0; index < env->max_bombs; index++) {
        Bomb* bomb = &env->bombs[index];
        if (!bomb->active) continue;
        if (bomb->row == row) {
            int delta_col = col - bomb->col;
            int dist = delta_col < 0 ? -delta_col : delta_col;
            if (dist <= bomb->range) {
                int step = (delta_col < 0) ? -1 : 1;
                int clear = 1;
                for (int scan_col = bomb->col + step; scan_col != col; scan_col += step) {
                    unsigned char tile_at_scan = env->grid[idx(env, row, scan_col)];
                    if (tile_at_scan == TILE_HARD || tile_at_scan == TILE_SOFT) {
                        clear = 0;
                        break;
                    }
                }
                if (clear && bomb->timer < min_timer) min_timer = bomb->timer;
            }
        }
        if (bomb->col == col) {
            int delta_row = row - bomb->row;
            int dist = delta_row < 0 ? -delta_row : delta_row;
            if (dist <= bomb->range) {
                int step = (delta_row < 0) ? -1 : 1;
                int clear = 1;
                for (int scan_row = bomb->row + step; scan_row != row; scan_row += step) {
                    unsigned char tile_at_scan = env->grid[idx(env, scan_row, col)];
                    if (tile_at_scan == TILE_HARD || tile_at_scan == TILE_SOFT) {
                        clear = 0;
                        break;
                    }
                }
                if (clear && bomb->timer < min_timer) min_timer = bomb->timer;
            }
        }
    }
    return min_timer;
}

/* Distance to nearest alive enemy; 255 if no enemies alive. */
static int nearest_enemy_distance(TileBlast* env, int row, int col) {
    int best = 255;
    for (int index = 0; index < ENEMY_COUNT; index++) {
        Enemy* enemy = &env->enemies[index];
        if (!enemy->alive) continue;
        int enemy_distance = manhattan_distance(row, col, enemy->row, enemy->col);
        if (enemy_distance < best) best = enemy_distance;
    }
    return best;
}

/* --- Enemy helpers ------------------------------------------------------ */
/* Forward declaration for helper use before definition. */
static int enemy_at(TileBlast* env, int row, int col, int skip_idx);

/* Tests if the agent can move into a destination tile. */
static int is_solid(TileBlast* env, int row, int col, int agent_idx) {
    if (!in_bounds(env, row, col)) return 1;
    if (row == env->goal_row && col == env->goal_col) return 0;

    unsigned char tile = env->grid[idx(env, row, col)];
    if (tile == TILE_HARD || tile == TILE_SOFT) return 1;
    if (tile == TILE_BOMB) {
        Agent* agent = &env->agents[agent_idx];
        if (agent->row == row && agent->col == col) return 0;
        return 1;
    }
    return 0;
}

/* Leaves spawn/goal-adjacent cells open during random map generation. */
static int is_spawn_clear(TileBlast* env, int row, int col) {
    int start_row = env->agents[0].row;
    int start_col = env->agents[0].col;
    int goal_row_candidate = env->goal_row, goal_col_candidate = env->goal_col;
    if (manhattan_distance(row, col, start_row, start_col) <= 1) return 1;
    if (manhattan_distance(row, col, goal_row_candidate, goal_col_candidate) <= 1) return 1;
    return 0;
}

static void sample_goal_tile(TileBlast* env) {
    for (int attempt = 0; attempt < 256; attempt++) {
        int row = 1 + (rand() % (env->height - 2));
        int col = 1 + (rand() % (env->width - 2));
        if ((row <= 2) && (col <= 2)) continue;
        if ((row % 2 == 0) && (col % 2 == 0)) continue;
        env->goal_row = row;
        env->goal_col = col;
        return;
    }
    env->goal_row = env->height - 2;
    env->goal_col = env->width - 2;
}

static void sample_agent_tile(TileBlast* env, int* out_row, int* out_col) {
    for (int attempt = 0; attempt < 256; attempt++) {
        int row = 1 + (rand() % (env->height - 2));
        int col = 1 + (rand() % (env->width - 2));
        if ((row % 2 == 0) && (col % 2 == 0)) continue;
        if (manhattan_distance(row, col, env->goal_row, env->goal_col) <= 2) continue;
        *out_row = row;
        *out_col = col;
        return;
    }
    *out_row = 1;
    *out_col = 1;
}

static int reserve_enemy_tile(TileBlast* env, int* out_row, int* out_col) {
    for (int attempt = 0; attempt < 256; attempt++) {
        int row = 1 + (rand() % (env->height - 2));
        int col = 1 + (rand() % (env->width - 2));
        if (row == env->agents[0].row && col == env->agents[0].col) continue;
        if (row == env->goal_row && col == env->goal_col) continue;
        if (env->grid[idx(env, row, col)] == TILE_HARD) continue;
        if (enemy_at(env, row, col, -1)) continue;
        *out_row = row;
        *out_col = col;
        return 1;
    }
    return 0;
}

/* Checks whether any alive enemy occupies a tile (except skip_idx). */
static int enemy_at(TileBlast* env, int row, int col, int skip_idx) {
    for (int index = 0; index < ENEMY_COUNT; index++) {
        if (index == skip_idx) continue;
        Enemy* enemy = &env->enemies[index];
        if (enemy->alive && enemy->row == row && enemy->col == col) {
            return 1;
        }
    }
    return 0;
}

static int action_toward_goal(int row, int col, int goal_row, int goal_col) {
    int delta_row = goal_row - row;
    int delta_col = goal_col - col;
    if (delta_row == 0 && delta_col == 0) return ACT_NOOP;
    if (abs(delta_row) >= abs(delta_col)) {
        return (delta_row > 0) ? ACT_DOWN : ACT_UP;
    }
    return (delta_col > 0) ? ACT_RIGHT : ACT_LEFT;
}

static int enemy_blocks_action(TileBlast* env, int row, int col, int action, int max_dist) {
    int step_row = 0;
    int step_col = 0;
    if (action == ACT_UP) step_row = -1;
    else if (action == ACT_DOWN) step_row = 1;
    else if (action == ACT_LEFT) step_col = -1;
    else if (action == ACT_RIGHT) step_col = 1;
    else return 0;

    for (int distance_step = 1; distance_step <= max_dist; distance_step++) {
        int scan_row = row + step_row * distance_step;
        int scan_col = col + step_col * distance_step;
        if (!in_bounds(env, scan_row, scan_col)) break;
        unsigned char tile = env->grid[idx(env, scan_row, scan_col)];
        if (tile == TILE_HARD || tile == TILE_SOFT || tile == TILE_BOMB) break;
        if (enemy_at(env, scan_row, scan_col, -1)) return 1;
    }
    return 0;
}

static int has_legal_move(TileBlast* env, int row, int col) {
    if (!is_solid(env, row - 1, col, 0)) return 1;
    if (!is_solid(env, row + 1, col, 0)) return 1;
    if (!is_solid(env, row, col - 1, 0)) return 1;
    if (!is_solid(env, row, col + 1, 0)) return 1;
    return 0;
}

/* Patrol movement collision check for enemies. */
static int enemy_walkable(TileBlast* env, int row, int col) {
    if (!in_bounds(env, row, col)) return 0;
    unsigned char tile_value = env->grid[idx(env, row, col)];
    if (tile_value == TILE_HARD || tile_value == TILE_SOFT || tile_value == TILE_BOMB) return 0;
    return 1;
}

/* --- Safe pathfinding -------------------------------------------------- */
static int adjacent_enemy(TileBlast* env, int row, int col) {
    for (int index = 0; index < ENEMY_COUNT; index++) {
        Enemy* enemy = &env->enemies[index];
        if (!enemy->alive) continue;
        if (manhattan_distance(row, col, enemy->row, enemy->col) <= 1) return 1;
    }
    return 0;
}

static int move_target(TileBlast* env, int row, int col, int action, int* out_row, int* out_col) {
    int next_row = row;
    int next_col = col;
    if (action == ACT_UP) next_row -= 1;
    else if (action == ACT_DOWN) next_row += 1;
    else if (action == ACT_LEFT) next_col -= 1;
    else if (action == ACT_RIGHT) next_col += 1;
    else return 0;
    if (!in_bounds(env, next_row, next_col)) return 0;
    *out_row = next_row;
    *out_col = next_col;
    return 1;
}

static int safe_walkable(TileBlast* env, int row, int col) {
    if (!in_bounds(env, row, col)) return 0;
    if (is_solid(env, row, col, 0)) return 0;
    if (enemy_at(env, row, col, -1)) return 0;
    if (adjacent_enemy(env, row, col)) return 0;
    if (env->blast_timer[idx(env, row, col)] > 0) return 0;
    if (min_bomb_timer_affecting(env, row, col) <= 2) return 0;
    return 1;
}

static int aggressive_walkable(TileBlast* env, int row, int col) {
    if (!in_bounds(env, row, col)) return 0;
    if (is_solid(env, row, col, 0)) return 0;
    if (enemy_at(env, row, col, -1)) return 0;
    if (env->blast_timer[idx(env, row, col)] > 0) return 0;
    if (min_bomb_timer_affecting(env, row, col) <= 1) return 0;
    return 1;
}

static int action_is_unsafe(TileBlast* env, int row, int col, int action) {
    if (action == ACT_NOOP) return 1;
    if (action == ACT_BOMB) return 0;
    int next_row = row;
    int next_col = col;
    if (!move_target(env, row, col, action, &next_row, &next_col)) return 1;
    return !safe_walkable(env, next_row, next_col);
}

static int safe_fallback_action(TileBlast* env, int row, int col) {
    int best_action = ACT_NOOP;
    int best_score = -1000000;
    static const int actions[4] = {ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT};
    for (int index = 0; index < 4; index++) {
        int action_candidate = actions[index];
        int next_row = row;
        int next_col = col;
        if (!move_target(env, row, col, action_candidate, &next_row, &next_col)) continue;
        if (!safe_walkable(env, next_row, next_col)) continue;

        int score = 0;
        score -= manhattan_distance(next_row, next_col, env->goal_row, env->goal_col) * 10;
        score += nearest_enemy_distance(env, next_row, next_col);
        score += min_bomb_timer_affecting(env, next_row, next_col);
        if (score > best_score) {
            best_score = score;
            best_action = action_candidate;
        }
    }
    return best_action;
}

static int choose_detour_action(TileBlast* env, int row, int col) {
    int current_goal_dist = manhattan_distance(row, col, env->goal_row, env->goal_col);
    int best_action = ACT_NOOP;
    float best_score = -1e9f;
    static const int actions[4] = {ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT};

    for (int index = 0; index < 4; index++) {
        int action_candidate = actions[index];
        int next_row = row;
        int next_col = col;
        if (!move_target(env, row, col, action_candidate, &next_row, &next_col)) continue;
        if (!safe_walkable(env, next_row, next_col)) continue;

        int next_goal_dist = manhattan_distance(next_row, next_col, env->goal_row, env->goal_col);
        if (next_goal_dist > current_goal_dist + DETOUR_MAX_MANHATTAN_INCREASE) continue;

        float heat = env->visit_heat[idx(env, next_row, next_col)];
        int enemy_dist = nearest_enemy_distance(env, next_row, next_col);
        int bomb_timer = min_bomb_timer_affecting(env, next_row, next_col);

        // Prefer lower-pheromone detours while still keeping goal pressure.
        float score = 0.0f;
        score -= (float)next_goal_dist * 4.0f;
        score -= heat * 25.0f;
        score += (float)enemy_dist * 0.6f;
        score += (float)bomb_timer * 0.2f;

        if (score > best_score) {
            best_score = score;
            best_action = action_candidate;
        }
    }

    if (best_action != ACT_NOOP) return best_action;
    return safe_fallback_action(env, row, col);
}

static int find_safe_goal_action(TileBlast* env, int start_row, int start_col) {
    if (start_row == env->goal_row && start_col == env->goal_col) return ACT_NOOP;

    int cells = env->width * env->height;
    int* queue = (int*)malloc(sizeof(int) * cells);
    int* parent = (int*)malloc(sizeof(int) * cells);
    int* first_action = (int*)malloc(sizeof(int) * cells);
    unsigned char* visited = (unsigned char*)calloc(cells, sizeof(unsigned char));
    if (!queue || !parent || !first_action || !visited) {
        free(queue); free(parent); free(first_action); free(visited);
        return safe_fallback_action(env, start_row, start_col);
    }

    for (int index = 0; index < cells; index++) {
        parent[index] = -1;
        first_action[index] = ACT_NOOP;
    }

    int start = idx(env, start_row, start_col);
    int goal = idx(env, env->goal_row, env->goal_col);
    int queue_head = 0, queue_tail = 0;
    queue[queue_tail++] = start;
    visited[start] = 1;

    static const int actions[4] = {ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT};
    int found = 0;
    while (queue_head < queue_tail) {
        int cur = queue[queue_head++];
        if (cur == goal) {
            found = 1;
            break;
        }

        int current_row = cur / env->width;
        int scan_col = cur % env->width;
        for (int index = 0; index < 4; index++) {
            int action_candidate = actions[index];
            int next_row = current_row;
            int next_col = scan_col;
            if (!move_target(env, current_row, scan_col, action_candidate, &next_row, &next_col)) continue;
            int next_index = idx(env, next_row, next_col);
            if (visited[next_index]) continue;
            if (next_index != goal && !safe_walkable(env, next_row, next_col)) continue;
            if (next_index == goal && enemy_at(env, next_row, next_col, -1)) continue;

            visited[next_index] = 1;
            parent[next_index] = cur;
            first_action[next_index] = (cur == start) ? action_candidate : first_action[cur];
            queue[queue_tail++] = next_index;
        }
    }

    int action = ACT_NOOP;
    if (found) {
        action = first_action[goal];
    } else {
        action = safe_fallback_action(env, start_row, start_col);
    }

    free(queue);
    free(parent);
    free(first_action);
    free(visited);
    return action;
}

static int in_closed_interval(int value, int bound_a, int bound_b) {
    if (bound_a <= bound_b) return value >= bound_a && value <= bound_b;
    return value >= bound_b && value <= bound_a;
}

static int enemy_on_manhattan_path(TileBlast* env, int row, int col, int goal_row, int goal_col) {
    // Enemy lies on at least one shortest (Manhattan) corridor toward the goal.
    for (int index = 0; index < ENEMY_COUNT; index++) {
        Enemy* enemy = &env->enemies[index];
        if (!enemy->alive) continue;
        if (enemy->row == row && in_closed_interval(enemy->col, col, goal_col)) return 1;
        if (enemy->col == col && in_closed_interval(enemy->row, row, goal_row)) return 1;
    }
    return 0;
}

static int should_bomb_manhattan_blocker(TileBlast* env, int row, int col, int goal_row, int goal_col, int range) {
    if (!enemy_on_manhattan_path(env, row, col, goal_row, goal_col)) return 0;
    if (!bomb_has_tactical_target(env, row, col, range)) return 0;
    // Commit unless danger is truly immediate.
    if (min_bomb_timer_affecting(env, row, col) <= BLOCKER_BOMB_COMMIT_DANGER_TIMER) return 0;
    return 1;
}

static int find_manhattan_blocking_enemy(TileBlast* env, int row, int col, int goal_row, int goal_col, int* out_row, int* out_col) {
    int best = 1000000;
    int found = 0;
    for (int index = 0; index < ENEMY_COUNT; index++) {
        Enemy* enemy = &env->enemies[index];
        if (!enemy->alive) continue;

        int on_path = 0;
        if (enemy->row == row && in_closed_interval(enemy->col, col, goal_col)) on_path = 1;
        if (enemy->col == col && in_closed_interval(enemy->row, row, goal_row)) on_path = 1;
        if (!on_path) continue;

        int enemy_distance = manhattan_distance(row, col, enemy->row, enemy->col);
        if (enemy_distance < best) {
            best = enemy_distance;
            *out_row = enemy->row;
            *out_col = enemy->col;
            found = 1;
        }
    }
    return found;
}

static int choose_destroy_blocker_action(TileBlast* env, int row, int col, int enemy_row, int enemy_col) {
    int best_action = ACT_NOOP;
    float best_score = -1e9f;
    static const int actions[4] = {ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT};

    for (int index = 0; index < 4; index++) {
        int action_candidate = actions[index];
        int next_row = row;
        int next_col = col;
        if (!move_target(env, row, col, action_candidate, &next_row, &next_col)) continue;
        if (!aggressive_walkable(env, next_row, next_col)) continue;

        int de = manhattan_distance(next_row, next_col, enemy_row, enemy_col);
        int dg = manhattan_distance(next_row, next_col, env->goal_row, env->goal_col);
        float heat = env->visit_heat[idx(env, next_row, next_col)];
        int bomb_timer = min_bomb_timer_affecting(env, next_row, next_col);
        int enemy_dist = nearest_enemy_distance(env, next_row, next_col);

        // Strongly prefer moves that line up and get into bomb range of blocker.
        float score = 0.0f;
        score -= (float)de * 10.0f;
        score -= (float)dg * 2.0f;
        score -= heat * 20.0f;
        score += (float)bomb_timer * 0.15f;
        score += (float)enemy_dist * 0.2f;

        if (next_row == enemy_row || next_col == enemy_col) score += 7.0f;
        if (de <= DEFAULT_RANGE) score += 9.0f;

        if (score > best_score) {
            best_score = score;
            best_action = action_candidate;
        }
    }

    if (best_action != ACT_NOOP) return best_action;
    return safe_fallback_action(env, row, col);
}

/* Initializes one enemy and clears its patrol line through soft tiles. */
static void init_enemy_patrol(TileBlast* env, int enemy_idx, int row, int col, int horizontal, int dir) {
    Enemy* enemy = &env->enemies[enemy_idx];
    enemy->row = row;
    enemy->col = col;
    enemy->alive = 1;
    enemy->dir = dir;
    enemy->horizontal = horizontal;

    if (horizontal) {
        enemy->min_pos = 1;
        enemy->max_pos = env->width - 2;
        for (int scan_col = enemy->min_pos; scan_col <= enemy->max_pos; scan_col++) {
            int index = idx(env, row, scan_col);
            if (env->grid[index] == TILE_SOFT) env->grid[index] = TILE_EMPTY;
        }
    } else {
        enemy->min_pos = 1;
        enemy->max_pos = env->height - 2;
        for (int scan_row = enemy->min_pos; scan_row <= enemy->max_pos; scan_row++) {
            int index = idx(env, scan_row, col);
            if (env->grid[index] == TILE_SOFT) env->grid[index] = TILE_EMPTY;
        }
    }

    env->grid[idx(env, enemy->row, enemy->col)] = TILE_EMPTY;
}

/* --- Enemy movement ----------------------------------------------------- */
/* Updates enemy patrol positions on a fixed tick interval. */
static void move_enemies(TileBlast* env) {
    if ((env->tick % ENEMY_MOVE_INTERVAL) != 0) return;

    for (int index = 0; index < ENEMY_COUNT; index++) {
        Enemy* enemy = &env->enemies[index];
        if (!enemy->alive) continue;

        int next_row = enemy->row;
        int next_col = enemy->col;
        if (enemy->horizontal) {
            next_col = enemy->col + enemy->dir;
            if (next_col < enemy->min_pos || next_col > enemy->max_pos || !enemy_walkable(env, next_row, next_col) || enemy_at(env, next_row, next_col, index)) {
                enemy->dir = -enemy->dir;
                next_col = enemy->col + enemy->dir;
            }
        } else {
            next_row = enemy->row + enemy->dir;
            if (next_row < enemy->min_pos || next_row > enemy->max_pos || !enemy_walkable(env, next_row, next_col) || enemy_at(env, next_row, next_col, index)) {
                enemy->dir = -enemy->dir;
                next_row = enemy->row + enemy->dir;
            }
        }

        if (enemy_walkable(env, next_row, next_col) && !enemy_at(env, next_row, next_col, index)) {
            enemy->row = next_row;
            enemy->col = next_col;
        }
    }
}

/* --- Map generation ----------------------------------------------------- */
/* Fills the map with walls and random soft blocks. */
static void generate_map(TileBlast* env) {
    for (int row = 0; row < env->height; row++) {
        for (int col = 0; col < env->width; col++) {
            unsigned char* cell = &env->grid[idx(env, row, col)];
            if (row == 0 || col == 0 || row == env->height - 1 || col == env->width - 1) {
                *cell = TILE_HARD;
                continue;
            }
            if ((row % 2 == 0) && (col % 2 == 0)) {
                *cell = TILE_HARD;
                continue;
            }
            if (is_spawn_clear(env, row, col)) {
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
    for (int agent_index = 0; agent_index < env->num_agents; agent_index++) {
        unsigned char* obs = env->observations + agent_index * (env->obs_size * OBS_CHANNELS + env->scalar_size);
        memset(obs, 0, env->obs_size * OBS_CHANNELS + env->scalar_size);

        unsigned char* grid_obs = obs + env->obs_size * OBS_GRID;
        unsigned char* scalar_obs = obs + env->obs_size * OBS_CHANNELS;
        Agent* self = &env->agents[agent_index];

        int idx_local = 0;
        if (env->vision == 0) {
            for (int scan_row = 0; scan_row < env->height; scan_row++) {
                for (int scan_col = 0; scan_col < env->width; scan_col++) {
                    unsigned char visible_tile = env->grid[idx(env, scan_row, scan_col)];
                    if (env->blast_timer[idx(env, scan_row, scan_col)] > 0) {
                        visible_tile = TILE_BLAST;
                    }
                    if (self->alive && self->row == scan_row && self->col == scan_col) {
                        visible_tile = TILE_AGENT0;
                    } else if (enemy_at(env, scan_row, scan_col, -1)) {
                        visible_tile = TILE_ENEMY;
                    } else if (scan_row == env->goal_row && scan_col == env->goal_col) {
                        visible_tile = TILE_GOAL;
                    }
                    grid_obs[idx_local++] = visible_tile;
                }
            }
        } else {
            for (int delta_row = -env->vision; delta_row <= env->vision; delta_row++) {
                for (int delta_col = -env->vision; delta_col <= env->vision; delta_col++) {
                    int scan_row = self->row + delta_row;
                    int scan_col = self->col + delta_col;
                    if (!in_bounds(env, scan_row, scan_col)) {
                        grid_obs[idx_local] = TILE_HARD;
                    } else {
                        unsigned char visible_tile = env->grid[idx(env, scan_row, scan_col)];
                        if (env->blast_timer[idx(env, scan_row, scan_col)] > 0) {
                            visible_tile = TILE_BLAST;
                        }
                        if (self->alive && self->row == scan_row && self->col == scan_col) {
                            visible_tile = TILE_AGENT0;
                        } else if (enemy_at(env, scan_row, scan_col, -1)) {
                            visible_tile = TILE_ENEMY;
                        } else if (scan_row == env->goal_row && scan_col == env->goal_col) {
                            visible_tile = TILE_GOAL;
                        }
                        grid_obs[idx_local] = visible_tile;
                    }
                    idx_local++;
                }
            }
        }
        if (self->alive) {
            if (env->vision == 0) {
                grid_obs[idx(env, self->row, self->col)] = TILE_AGENT0;
            } else {
                int side = env->vision * 2 + 1;
                int self_idx = env->vision * side + env->vision;
                grid_obs[self_idx] = TILE_AGENT0;
            }
        }

        int dist_goal = manhattan_distance(self->row, self->col, env->goal_row, env->goal_col);
        int dist_bomb = 255;
        for (int index = 0; index < env->max_bombs; index++) {
            Bomb* bomb = &env->bombs[index];
            if (!bomb->active) continue;
            int bomb_distance = manhattan_distance(self->row, self->col, bomb->row, bomb->col);
            if (bomb_distance < dist_bomb) dist_bomb = bomb_distance;
        }

        int min_timer = min_bomb_timer_affecting(env, self->row, self->col);
        int dist_enemy = nearest_enemy_distance(env, self->row, self->col);
        int bombs_avail = self->bombs_max - bombs_owned(env, agent_index);
        if (bombs_avail < 0) bombs_avail = 0;
        int can_place =
            (self->alive && bombs_owned(env, agent_index) < self->bombs_max &&
             !bomb_at(env, self->row, self->col)) ? 1 : 0;
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
    int spawn_row = 1;
    int spawn_col = 1;
    sample_agent_tile(env, &spawn_row, &spawn_col);
    env->agents[0] = (Agent){
        .row = spawn_row,
        .col = spawn_col,
        .alive = 1,
        .bombs_max = MAX_BOMBS_PER_AGENT,
        .range = DEFAULT_RANGE,
        .lives = START_LIVES,
        .invuln = RESPAWN_INVULN,
    };
    env->last_goal_dist = manhattan_distance(spawn_row, spawn_col, env->goal_row, env->goal_col);
    env->no_progress_steps = 0;
    clear_bombs(env);
    generate_map(env);
    env->grid[idx(env, env->goal_row, env->goal_col)] = TILE_EMPTY;
    for (int index = 0; index < ENEMY_COUNT; index++) {
        env->enemies[index].alive = 0;
    }
    for (int enemy_idx = 0; enemy_idx < ENEMY_COUNT; enemy_idx++) {
        int spawn_row = 1;
        int spawn_col = 1;
        if (!reserve_enemy_tile(env, &spawn_row, &spawn_col)) {
            spawn_row = 1 + enemy_idx;
            spawn_col = env->width - 2 - enemy_idx;
            if (spawn_row >= env->height - 1) spawn_row = env->height - 2;
            if (spawn_col <= 0) spawn_col = 1;
        }
        int patrol_horizontal = rand() & 1;
        int patrol_dir = (rand() & 1) ? 1 : -1;
        init_enemy_patrol(env, enemy_idx, spawn_row, spawn_col, patrol_horizontal, patrol_dir);
    }

    update_observations(env);
}

/* --- Agent control ----------------------------------------------------- */
/* Applies one action as a single-tile move. */
static void resolve_move(TileBlast* env, int action) {
    Agent* agent = &env->agents[0];
    if (!agent->alive) return;
    if ((unsigned)action > ACT_BOMB) return;

    // Action-indexed movement deltas. Non-move actions map to (0,0).
    static const int delta_row[ACT_BOMB + 1] = {0, -1, 1, 0, 0, 0};
    static const int delta_col[ACT_BOMB + 1] = {0, 0, 0, -1, 1, 0};
    int step_row = delta_row[action];
    int step_col = delta_col[action];
    if (step_row == 0 && step_col == 0) return;

    int target_row = agent->row + step_row;
    int target_col = agent->col + step_col;
    if (!is_solid(env, target_row, target_col, 0)) {
        agent->row = target_row;
        agent->col = target_col;
    }
}

/* --- Runtime step ------------------------------------------------------ */
/* Simple reward shaping that favors progress toward the goal. */
static void c_step(TileBlast* env) {
    env->tick += 1;
    env->terminals[0] = 0;
    decay_blasts(env);
    decay_visit_heat(env);

    Agent* agent = &env->agents[0];
    int previous_threat_timer = min_bomb_timer_affecting(env, agent->row, agent->col);
    int prev_row = agent->row;
    int prev_col = agent->col;
    int previous_goal_dist = manhattan_distance(prev_row, prev_col, env->goal_row, env->goal_col);
    int stuck_on_pheromones =
        (env->no_progress_steps >= STUCK_NO_PROGRESS_STEPS) &&
        (env->visit_heat[idx(env, prev_row, prev_col)] >= STUCK_HEAT_THRESHOLD);
    int goal_action = action_toward_goal(prev_row, prev_col, env->goal_row, env->goal_col);
    int goal_lane_blocked = enemy_blocks_action(env, prev_row, prev_col, goal_action, ENEMY_BLOCK_RANGE);
    int tactical_bomb_available = bomb_has_tactical_target(env, prev_row, prev_col, agent->range);
    int manhattan_enemy_blocker = should_bomb_manhattan_blocker(
        env, prev_row, prev_col, env->goal_row, env->goal_col, agent->range);
    int can_place_bomb = (bombs_owned(env, 0) < agent->bombs_max && !bomb_at(env, prev_row, prev_col));

    int action = ACT_NOOP;
    if (env->actions) action = env->actions[0];
    if (action < ACT_NOOP || action > ACT_BOMB) action = ACT_NOOP;
    int blocker_row = -1;
    int blocker_col = -1;
    int forcing_blocker_clear = 0;
    int has_manhattan_blocker = find_manhattan_blocking_enemy(
        env, prev_row, prev_col, env->goal_row, env->goal_col, &blocker_row, &blocker_col);
    if (manhattan_enemy_blocker && can_place_bomb) {
        action = ACT_BOMB;
        forcing_blocker_clear = 1;
    } else if (has_manhattan_blocker) {
        int destroy_action = choose_destroy_blocker_action(env, prev_row, prev_col, blocker_row, blocker_col);
        if (destroy_action != ACT_NOOP) {
            action = destroy_action;
            forcing_blocker_clear = 1;
        }
    }
    int suppress_safety_override = 0;
    if (forcing_blocker_clear && action >= ACT_UP && action <= ACT_RIGHT) {
        int next_row = prev_row;
        int next_col = prev_col;
        if (move_target(env, prev_row, prev_col, action, &next_row, &next_col) &&
            aggressive_walkable(env, next_row, next_col)) {
            suppress_safety_override = 1;
        }
    }
    if (!suppress_safety_override && action_is_unsafe(env, prev_row, prev_col, action)) {
        int safe_action = stuck_on_pheromones
            ? choose_detour_action(env, prev_row, prev_col)
            : find_safe_goal_action(env, prev_row, prev_col);
        if (safe_action != ACT_NOOP) {
            action = safe_action;
        }
    }
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
    if (agent->alive) {
        reward -= STEP_PENALTY;
        int legal_moves = has_legal_move(env, prev_row, prev_col);
        int moved = (agent->row != prev_row || agent->col != prev_col);
        if (moved) {
            reward += MOVE_REWARD;
        } else {
            if (legal_moves) reward -= STALL_PENALTY;
        }
        int new_goal_dist = manhattan_distance(agent->row, agent->col, env->goal_row, env->goal_col);
        if (new_goal_dist < env->last_goal_dist) env->no_progress_steps = 0;
        else env->no_progress_steps += 1;
        env->last_goal_dist = new_goal_dist;
        reward += (float)(previous_goal_dist - new_goal_dist) * GOAL_PROGRESS_REWARD;
        if (action == ACT_NOOP && legal_moves) {
            reward -= NOOP_PENALTY;
        }
        if (planted_bomb) {
            reward += BOMB_PLANT_BASE_REWARD;
            if (tactical_bomb_available) {
                reward += TACTICAL_BOMB_REWARD;
            }
            int enemy_dist_before_action = nearest_enemy_distance(env, prev_row, prev_col);
            if (enemy_dist_before_action <= BOMB_NEAR_ENEMY_RANGE) {
                reward += BOMB_NEAR_ENEMY_REWARD;
            }
        } else if (action != ACT_BOMB && tactical_bomb_available && can_place_bomb) {
            reward -= MISSED_TACTICAL_BOMB_PENALTY;
        }
        int current_threat_timer = min_bomb_timer_affecting(env, agent->row, agent->col);
        int bomb_distance = nearest_bomb_distance(env, agent->row, agent->col);
        if (bomb_distance <= BOMB_PROX_THRESHOLD) {
            reward -= (float)(BOMB_PROX_THRESHOLD + 1 - bomb_distance) * BOMB_PROX_PENALTY;
            if (bomb_distance == 0 && current_threat_timer <= BOMB_ON_TILE_DANGER_TIMER) {
                reward -= BOMB_ON_TILE_PENALTY;
            }
        }
        if (current_threat_timer <= 3) {
            reward -= BLAST_DANGER_PENALTY;
        }
        if (previous_threat_timer <= 3 &&
            (current_threat_timer > previous_threat_timer || current_threat_timer == 255)) {
            reward += BLAST_ESCAPE_REWARD;
        }
        if (goal_lane_blocked) {
            if (action == goal_action) {
                reward -= ENEMY_PATH_BLOCK_PENALTY;
            } else if (action >= ACT_UP && action <= ACT_RIGHT &&
                       (agent->row != prev_row || agent->col != prev_col)) {
                reward += ENEMY_PATH_AVOID_REWARD;
            }
        }
        int self_idx = idx(env, agent->row, agent->col);
        reward -= env->visit_heat[self_idx] * VISIT_HEAT_PENALTY;
        env->visit_heat[self_idx] += VISIT_HEAT_DEPOSIT;
    }
    if (enemies_killed > 0) {
        reward += (float)enemies_killed * ENEMY_KILL_REWARD;
    }
    int done = 0;
    if (agent->alive && agent->row == env->goal_row && agent->col == env->goal_col) {
        env->score_points += SCORE_FOR_GOAL;
        reward += SCORE_FOR_GOAL;
        done = 1;
    }

    if (!done && agent->alive && enemy_at(env, agent->row, agent->col, -1)) {
        agent->alive = 0;
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
        env->log.perf = (agent->alive && agent->row == env->goal_row && agent->col == env->goal_col) ? 1.0f : 0.0f;
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
    const int window_width = env->width * tile;
    const int window_height = env->height * tile;
    if (!IsWindowReady()) {
        InitWindow(window_width, window_height, "PufferLib TileBlast Goal");
        SetTargetFPS(10);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_H)) {
        g_show_visit_heat = !g_show_visit_heat;
    }

    if (!g_static_ready || g_static_w != window_width || g_static_h != window_height) {
        if (g_static_ready) {
            UnloadRenderTexture(g_static_tex);
        }
        g_static_tex = LoadRenderTexture(window_width, window_height);
        g_static_ready = 1;
        g_static_w = window_width;
        g_static_h = window_height;
        BeginTextureMode(g_static_tex);
        ClearBackground((Color){11, 13, 24, 255});
        for (int row = 0; row < env->height; row++) {
            for (int col = 0; col < env->width; col++) {
                int pixel_x = col * tile;
                int pixel_y = row * tile;
                Color base = ((row + col) % 2 == 0) ? (Color){27, 33, 64, 255} : (Color){21, 26, 51, 255};
                DrawRectangle(pixel_x, pixel_y, tile, tile, base);
                unsigned char tile_value = env->grid[idx(env, row, col)];
                if (tile_value == TILE_HARD) {
                    DrawRectangle(pixel_x + 3, pixel_y + 3, tile - 6, tile - 6, (Color){255, 90, 95, 235});
                }
            }
        }
        EndTextureMode();
    }

    BeginDrawing();
    DrawTextureRec(
        g_static_tex.texture, (Rectangle){0, 0, (float)window_width, -(float)window_height},
        (Vector2){0, 0}, WHITE);

    float max_visit_heat = 0.0f;
    if (g_show_visit_heat) {
        for (int row = 0; row < env->height; row++) {
            for (int col = 0; col < env->width; col++) {
                int index = idx(env, row, col);
                float heat = env->visit_heat[index];
                if (heat <= 0.001f) continue;
                if (heat > max_visit_heat) max_visit_heat = heat;

                // Smoothly map unbounded heat to [0,1) without extra math deps.
                float normalized_heat = heat / (heat + 4.0f);
                int alpha = (int)(normalized_heat * 180.0f);
                if (alpha < 10) alpha = 10;
                if (alpha > 180) alpha = 180;

                int red = (int)(40.0f + 210.0f * normalized_heat);
                int blue = (int)(240.0f - 180.0f * normalized_heat);
                if (red > 255) red = 255;
                if (blue < 0) blue = 0;

                int pixel_x = col * tile;
                int pixel_y = row * tile;
                DrawRectangle(pixel_x + 1, pixel_y + 1, tile - 2, tile - 2,
                              (Color){(unsigned char)red, 80, (unsigned char)blue, (unsigned char)alpha});
            }
        }
    }

    for (int row = 0; row < env->height; row++) {
        for (int col = 0; col < env->width; col++) {
            unsigned char tile_value = env->grid[idx(env, row, col)];
            int pixel_x = col * tile;
            int pixel_y = row * tile;
            if (tile_value == TILE_SOFT) {
                DrawRectangle(pixel_x + 3, pixel_y + 3, tile - 6, tile - 6, (Color){138, 125, 255, 235});
            }
            if (env->blast_timer[idx(env, row, col)] > 0) {
                int tval = env->blast_timer[idx(env, row, col)];
                int inset = 3 + (BLAST_TIME - tval);
                if (inset > tile / 2 - 1) inset = tile / 2 - 1;
                int glow = 120 + tval * 45;
                if (glow > 255) glow = 255;
                DrawRectangle(pixel_x + inset, pixel_y + inset, tile - 2 * inset, tile - 2 * inset, (Color){255, 190, 70, glow});
                DrawRectangle(pixel_x + tile / 2 - 2, pixel_y + 4, 4, tile - 8, (Color){255, 240, 170, glow});
                DrawRectangle(pixel_x + 4, pixel_y + tile / 2 - 2, tile - 8, 4, (Color){255, 240, 170, glow});
            }
        }
    }

    int goal_x = env->goal_col * tile;
    int goal_y = env->goal_row * tile;
    DrawRectangle(goal_x + 6, goal_y + 6, tile - 12, tile - 12, (Color){72, 220, 120, 255});

    if (env->agents[0].alive) {
        int agent_center_x = env->agents[0].col * tile + tile / 2;
        int agent_center_y = env->agents[0].row * tile + tile / 2;
        DrawCircle(agent_center_x, agent_center_y, 12, (Color){255, 216, 107, 255});
    }

    for (int index = 0; index < ENEMY_COUNT; index++) {
        Enemy* enemy = &env->enemies[index];
        if (!enemy->alive) continue;
        int enemy_center_x = enemy->col * tile + tile / 2;
        int enemy_center_y = enemy->row * tile + tile / 2;
        DrawCircle(enemy_center_x, enemy_center_y, 10, (Color){255, 120, 120, 255});
        DrawCircle(enemy_center_x - 3, enemy_center_y - 2, 2, WHITE);
        DrawCircle(enemy_center_x + 3, enemy_center_y - 2, 2, WHITE);
    }

    for (int index = 0; index < env->max_bombs; index++) {
        Bomb* bomb = &env->bombs[index];
        if (!bomb->active) continue;
        int bomb_center_x = bomb->col * tile + tile / 2;
        int bomb_center_y = bomb->row * tile + tile / 2 + 4;
        DrawCircle(bomb_center_x, bomb_center_y, 8, (Color){24, 30, 55, 255});
        DrawCircleLines(bomb_center_x, bomb_center_y, 9, (Color){90, 100, 160, 200});
        DrawCircle(bomb_center_x + 7, bomb_center_y - 9, 3, (Color){255, 204, 51, 255});
    }

    DrawText(TextFormat("Step: %d/%d", env->tick, env->max_steps), 10, 10, 18, (Color){255, 240, 200, 255});
    DrawText(TextFormat("Return: %.2f", env->episode_return_accum), 10, 32, 18, (Color){255, 240, 200, 255});
    DrawText(TextFormat("Visit heat: %s (H)  max: %.2f", g_show_visit_heat ? "ON" : "OFF", max_visit_heat),
             10, 54, 18, (Color){220, 230, 255, 255});

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
