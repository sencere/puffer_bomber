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
        env->bombs[i].active = 0;
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
    int sr = 1, sc = 1;
    int gr = env->height - 2, gc = env->width - 2;
    if ((r == sr && c == sc) || (r == sr + 1 && c == sc) || (r == sr && c == sc + 1)) return 1;
    if ((r == gr && c == gc) || (r == gr - 1 && c == gc) || (r == gr && c == gc - 1)) return 1;
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

    clear_bombs(env);
    generate_map(env);

    env->goal_r = env->height - 2;
    env->goal_c = env->width - 2;
    env->grid[idx(env, env->goal_r, env->goal_c)] = TILE_EMPTY;

    env->agents[0] = (Agent){
        .r = 1,
        .c = 1,
        .alive = 1,
        .bombs_max = MAX_BOMBS_PER_AGENT,
        .range = DEFAULT_RANGE,
        .lives = START_LIVES,
        .invuln = RESPAWN_INVULN,
    };
    init_enemy_patrol(env, 0, 1, env->width - 2, 1, -1);
    init_enemy_patrol(env, 1, env->height - 2, 1, 1, 1);
    {
        int mid_r = env->height / 2;
        int mid_c = env->width / 2;
        if ((mid_r % 2) == 0) mid_r += 1;
        if ((mid_c % 2) == 0) mid_c += 1;
        if (mid_r >= env->height - 1) mid_r = env->height - 2;
        if (mid_c >= env->width - 1) mid_c = env->width - 2;
        if ((mid_r % 2) == 0 && mid_r > 1) mid_r -= 1;
        if ((mid_c % 2) == 0 && mid_c > 1) mid_c -= 1;
        init_enemy_patrol(env, 2, mid_r, mid_c, 1, 1);
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
/* Minimal random-policy transition function. */
static void c_step(TileBlast* env) {
    env->tick += 1;
    env->terminals[0] = 0;

    Agent* a = &env->agents[0];
    int action = 1 + (rand() % 4); // random move: up/down/left/right
    resolve_move(env, action);
    move_enemies(env);

    int done = 0;
    if (a->alive && a->r == env->goal_r && a->c == env->goal_c) {
        env->score_points += SCORE_FOR_GOAL;
        done = 1;
    }

    if (!done && a->alive && enemy_at(env, a->r, a->c, -1)) {
        a->alive = 0;
        env->log.deaths += 1.0f;
        done = 1;
    }

    if (!done && env->tick >= env->max_steps) {
        done = 1;
        env->log.timeouts += 1.0f;
    }

    env->rewards[0] = 2.0f * randf() - 1.0f;
    env->episode_return_accum += env->rewards[0];

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
    free(env->grid);
    free(env->bombs);
}
