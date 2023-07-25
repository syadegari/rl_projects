from collections import deque

from .params import  device
from .logs import LogScores


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

def write_scores(experiment_name, scores):
    with open(f'scores_{experiment_name}.dat', 'w') as f:
        f.writelines('\n'.join(map(str, scores)))

def dqn(env, agent, n_episodes, max_t, eps_start, eps_end, eps_decay, experiment_name, score_threshold, stop_at_threshold):
    log_scores = LogScores()
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    for i_episode in range(1, n_episodes + 1):
        state = env_reset(env, env.brain_names[0])
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env_step(env, action, env.brain_names[0])
            agent.step(state, action, reward, next_state, done, i_episode)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        is_solved = log_scores.log_scores(agent, i_episode, scores_window, eps, 
                                          score_threshold=score_threshold, 
                                          experiment_name=experiment_name)
        if is_solved and stop_at_threshold:
            write_scores(experiment_name, scores)
            return
    write_scores(experiment_name, scores)


