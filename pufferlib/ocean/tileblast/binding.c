#include "tileblast.h"

#define Env TileBlast
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    if (kwargs) {
        const char* reward_keys[] = {
            "reward_win",
            "reward_loss",
            "reward_draw",
            "reward_soft",
            "reward_hit",
            "reward_self_hit",
            "reward_step",
            "reward_survive",
            "reward_no_bomb",
            "reward_avoid_bomb",
            "reward_bomb_near",
            "reward_no_cover",
            "reward_stall_tile",
            "reward_escape_bomb",
            "reward_reverse_move",
        };
        const int nkeys = (int)(sizeof(reward_keys) / sizeof(reward_keys[0]));
        for (int i = 0; i < nkeys; i++) {
            PyObject* key = PyUnicode_FromString(reward_keys[i]);
            if (!key) return -1;
            int has_key = PyDict_Contains(kwargs, key);
            Py_DECREF(key);
            if (has_key < 0) return -1;
            if (has_key > 0) {
                PyErr_SetString(PyExc_ValueError, "TileBlast rewards are hardcoded in C and cannot be set via Python");
                return -1;
            }
        }
    }

    env->num_agents = (int)unpack(kwargs, "num_agents");
    env->width = (int)unpack(kwargs, "width");
    env->height = (int)unpack(kwargs, "height");
    env->max_steps = (int)unpack(kwargs, "max_steps");
    env->vision = (int)unpack(kwargs, "vision");
    if (env->num_agents <= 0) env->num_agents = 1;
    if (env->num_agents > 2) env->num_agents = 2;
    if (env->width <= 0) env->width = 15;
    if (env->height <= 0) env->height = 13;
    if (env->max_steps <= 0) env->max_steps = DEFAULT_MAX_STEPS;
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
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "timeouts", log->timeouts);
    assign_to_dict(dict, "deaths", log->deaths);
    return 0;
}
