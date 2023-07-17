import yaml
import torch
import logging

KEYS = ['update_every', 
        'batch_size', 
        'buffer_size', 
        'seed', 
        'lr', 
        'tau', 
        'gamma', 
        'agent', 
        'model',
        'n_episodes', 
        'max_t', 
        'eps_start', 
        'eps_end', 
        'eps_decay', 
        'env_path', 
        'experiment_name']

MODEL_VALS = ['QNetwork', 'DuelingQNetwork']
AGENT_VALS = ['DQN', 'DDQN']

logging.basicConfig(level=logging.INFO)

class MissingKeysError(Exception):
    pass

class WrongValueError(Exception):
    pass


def check_params_values(params):
    if not params['model'] in MODEL_VALS:
        logging.error('Wrong value for "model"')
        raise WrongValueError(f'Wrong value for "model": {params["model"]}')
    elif not params['agent'] in AGENT_VALS:
        logging.error('Wrong value for "agent"')
        raise WrongValueError(f'Wrong value for "agent": {params["agent"]}')
    else:
        return

def check_params_exist(params):
    missing_keys = KEYS - params.keys()
    if missing_keys:
        logging.error(f'Missing keys in config: {missing_keys}')
        raise MissingKeysError(missing_keys)
    else:
        return

def get_params(config_path, cmdline_params):
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
            with open(config_path, 'w') as f:
                f.writelines(yaml.dump(params))
        return params
    #
    except MissingKeysError as k:
        logging.error(f'The followins are missing from {config_path}: {k}')
        raise
    except WrongValueError as w:
        logging.error(f'The following values are incorrect :{w}')
        raise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
