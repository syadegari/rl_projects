from collections import deque
import numpy as np

from .params import  device
from .logs import log_scores


def env_reset(env, brain_name, train_mode=True):
    env_info = env.reset(train_mode=train_mode)[brain_name]
    state = env_info.vector_observations[0]
    return state

def env_step(env, action, brain_name):
    env_info = env.step(action)[brain_name]
    next_state = env_info.vector_observations[0]
    reward = env_info.rewards[0]
    done = env_info.local_done[0]
    return next_state, reward, done, ''

def dqn(env, agent, n_episodes, max_t, eps_start, eps_end, eps_decay, experiment_name):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    for i_episode in range(1, n_episodes + 1):
        state = env_reset(env, env.brain_names[0])
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env_step(env, action, env.brain_names[0])
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        log_scores(agent, i_episode, scores_window, score_threshold=200.0)

    with open(f'scores_{experiment_name}.dat', 'w') as f:
        f.writelines('\n'.join(map(str, scores)))

