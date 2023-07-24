import os
import argparse
from collections import deque
from unityagents import UnityEnvironment

from dqn.dqn_algo import dqn
from dqn.agent import DQNAgent, DDQNAgent
from dqn.params import get_params, PARAMS_SCHEMA, HelpFormatter


def parse_arguments():
    parser = argparse.ArgumentParser(description='Command line arguments from schema',
                                 formatter_class=HelpFormatter)
    parser.add_argument('-c', '--config-file', type=str,  help="config file, including the path")
    for key, value in PARAMS_SCHEMA.items():
        parser.add_argument(f'--{key}', **value)
    return parser.parse_args()


if __name__ == '__main__':
    #      
    args = parse_arguments() 
    params = get_params(os.path.abspath(args.config_file), vars(args))
    env = UnityEnvironment(file_name=params['env_path'], no_graphics=True, seed=params['seed'])
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    n_state = brain.vector_observation_space_size
    n_action = brain.vector_action_space_size

    if params['agent'] == 'DQN':
        agent = DQNAgent(n_state, n_action, params)
    elif params['agent'] == 'DDQN':
        agent = DDQNAgent(n_state, n_action, params)

    dqn(env, agent, 
        n_episodes=params['n_episodes'], 
        max_t=params['max_t'],
        eps_start=params['eps_start'],
        eps_end=params['eps_end'],
        eps_decay=params['eps_decay'],
        experiment_name=params['experiment_name'],
        score_threshold=params['score_threshold'],
        stop_at_threshold=params['stop_at_threshold'])

    env.close()
