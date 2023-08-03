import os
import argparse
from reinforce.params import get_params, PARAMS_SCHEMA, HelpFormatter
from reinforce.reinforce import wrapper

def parse_arguments():
    parser = argparse.ArgumentParser(description='Command line arguments from schema',
                                 formatter_class=HelpFormatter)
    parser.add_argument('-c', '--config-file', type=str,  help="config file, including the path")
    for key, value in PARAMS_SCHEMA.items():
        parser.add_argument(f'--{key}', **value)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    params = get_params(os.path.abspath(args.config_file), vars(args))
    wrapper(params, called_via_cmdline=True)