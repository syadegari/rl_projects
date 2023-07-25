import os
import yaml
import torch
import logging
import argparse

PARAMS_SCHEMA = {
    'update_every': {'type': int, 'help': 'Frequency with which Q-function is updated'},
    'batch_size': {'type': int, 'help': 'Batch size'},
    'buffer_size': {'type': int, 'help': 'Replay buffer size'}, 
    'seed': {'type': int, 'help': 'Seed number'},
    # parameters of learning rate
    'lr': {'type': float, 'help': 'Optimizer learning rate'},
    'lr_decay': {'type': float, 'help': 'Learning rate decay factor for exponential decay'},
    'tau': {'type': float, 'help': 'Soft update parameter: target = tau * local + (1-tau) * target. Must be close to zero.' },
    'gamma': {'type': float, 'help': 'Gamma parameter in TD-update.'},
    'agent': {'type': str, 'choices': ['DQN', 'DDQN'], 'help': 'Determines the type of agent.'},
    'model': {'type': str, 'choices': ['QNetwork', 'DuelingQNetwork'], 'help': 'Determines the type of model.'},
    'n_episodes': {'type': int, 'help': 'Total number of episodes for training the environment.'},
    'max_t': {'type': int, 'help': 'Maximum time steps for playing each episode.'},
    'eps_start': {'type': float, 'help': 'Initial value for eps-greedy algorithm.'},
    'eps_end': {'type': float, 'help': 'Minimum value for eps-greedy algorithm.'},
    'eps_decay': {'type': float, 'help': 'Decay value for eps-greedy algorithm.'},
    'env_path': {'type': str, 'help': 'Path to environment binary'},
    'experiment_name': {'type': str, 'help': 'Name of the experiment.'},
    'alpha': {'type': float, 'help': 'Float value between 0 and 1. 0 results in uniform buffer (non-prioritized)'},
    'beta_0': {'type': float, 'help': 'Float between 0 and 1. Anneals linearly to 1.'},
    'generate_config': {'action': 'store_true', 'default': None, 'help': 'Generates the modified config file and quits.'},
    'stop_at_threshold': {'action': 'store_true', 'default': None, 'help': 'Continues the training after reaching the threshold.'},
    'score_threshold' : {'type': float, 'help': 'Threshold value. Stop the training when the value is reached, if s`top_at_threshold` is True.'}
    }

logging.basicConfig(level=logging.INFO)

class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, 
                    argparse.MetavarTypeHelpFormatter):
    pass

class MissingKeysError(Exception):
    pass

class WrongValueError(Exception):
    pass

def check_params_values(params):
    for param_name in params:
        if 'choices' in PARAMS_SCHEMA[param_name]:
            if not params[param_name] in PARAMS_SCHEMA[param_name]['choices']:
                logging.error(f'Wrong value for {param_name}')
                raise WrongValueError(f'Wrong value for {param_name}: {params[param_name]}')
    else:
        return

def check_params_exist(params):
    missing_keys = PARAMS_SCHEMA.keys() - params.keys()
    if missing_keys:
        logging.error(f'Missing keys in config: {missing_keys}')
        raise MissingKeysError(missing_keys)
    else:
        return

def get_params(config_path, cmdline_params):
    # remove None values from cmdline arguments
    cmdline_params = {k: v for k, v in cmdline_params.items() if v is not None}
    rewrite_config_file = False
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    try:
        # replace the specified parameters from the cmdline in params dict
        for param_name, param_val in cmdline_params.items():
            if param_name != 'config_file':
                if param_val is not None:
                    rewrite_config_file = True
                    params[param_name] = param_val
        #
        check_params_exist(params)
        check_params_values(params)
        if rewrite_config_file:
            modified_config_file = f"{os.path.dirname(config_path)}/config_{params['experiment_name']}.yaml"
            with open(modified_config_file, 'w') as f:
                f.writelines(yaml.dump(params))
        if params['generate_config']:
            exit()
        return params
    #
    except MissingKeysError as k:
        logging.error(f'The followins are missing from {config_path}: {k}')
        raise
    except WrongValueError as w:
        logging.error(f'The following values are incorrect :{w}')
        raise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
