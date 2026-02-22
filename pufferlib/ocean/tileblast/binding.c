#define TILEBLAST_IMPLEMENTATION
#include "tileblast.h"

#define Env TileBlast
#define ENV_HAS_TRUNCATIONS
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->num_agents = (int)unpack(kwargs, "num_agents");
    env->width = (int)unpack(kwargs, "width");
    env->height = (int)unpack(kwargs, "height");
    env->agent_speed = (int)unpack(kwargs, "agent_speed");
    env->max_steps = (int)unpack(kwargs, "max_steps");
    env->vision = (int)unpack(kwargs, "vision");
    if (env->num_agents <= 0) env->num_agents = 1;
    if (env->num_agents > 1) env->num_agents = 1;
    if (env->width <= 0) env->width = 15;
    if (env->height <= 0) env->height = 13;
    if (env->agent_speed <= 0) env->agent_speed = DEFAULT_AGENT_SPEED;
    if (env->max_steps <= 0) env->max_steps = DEFAULT_MAX_STEPS;
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
