### Install the repository and run tests

#### Requirements
- Python-3.6

Create a virtual environment and activate it: 
```bash
python3.6 -m venv drl-venv
source drl-venv/bin/activate
```
Run the following from terminal to install all the python modules:
```bash
pip install -r requirements.txt
```
To install the package `dqn` locally, `cd` to `p1_Navigation` type the following form your terminal:
```bash
pip install -e .
```
and uninstall it by:
```bash
pip uninstall my-dqn-project
```
#### Running the tests
From the terminal, `cd` to location where the `tests` folder is situated (`p1_Navigation`) and run:
```bash
python -m unittest discover tests
```
All the tests should pass without error.

### Training
#### Unity environment
Depending on your operating system, obtain the Unity environment and save it locally on your machine. The location does not have to match the location of the repository. Later we will provide the localtion of the environment to the config file (`config.yaml`). 
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

#### Setting the parameters of the training

All the tweakable parameters of the training are accessible either from the config file `config.yaml` or via command line. A oneliner help for each parameter is displayed if `driver.py` is called using the following from the command line:
```
(drl-venv) user p1_Navigation $ python driver.py -h
usage: driver.py [-h] [-c str] [--update_every int] [--batch_size int]
                 [--buffer_size int] [--seed int] [--lr float]
                 [--lr_decay float] [--tau float] [--gamma float]
                 [--agent {DQN,DDQN}] [--model {QNetwork,DuelingQNetwork}]
                 [--n_episodes int] [--max_t int] [--eps_start float]
                 [--eps_end float] [--eps_decay float] [--env_path str]
                 [--experiment_name str] [--alpha float] [--beta_0 float]
                 [--generate_config] [--stop_at_threshold]
                 [--score_threshold float]

Command line arguments from schema

optional arguments:
  -h, --help            show this help message and exit
  -c str, --config-file str
                        config file, including the path (default: None)
  --update_every int    Frequency with which Q-function is updated (default:
                        None)
  --batch_size int      Batch size (default: None)
  --buffer_size int     Replay buffer size (default: None)
  --seed int            Seed number (default: None)
  --lr float            Optimizer learning rate (default: None)
  --lr_decay float      Learning rate decay factor for exponential decay
                        (default: None)
  --tau float           Soft update parameter: target = tau * local + (1-tau)
                        * target. Must be close to zero. (default: None)
  --gamma float         Gamma parameter in TD-update. (default: None)
  --agent {DQN,DDQN}    Determines the type of agent. (default: None)
  --model {QNetwork,DuelingQNetwork}
                        Determines the type of model. (default: None)
  --n_episodes int      Total number of episodes for training the environment.
                        (default: None)
  --max_t int           Maximum time steps for playing each episode. (default:
                        None)
  --eps_start float     Initial value for eps-greedy algorithm. (default:
                        None)
  --eps_end float       Minimum value for eps-greedy algorithm. (default:
                        None)
  --eps_decay float     Decay value for eps-greedy algorithm. (default: None)
  --env_path str        Path to environment binary (default: None)
  --experiment_name str
                        Name of the experiment. (default: None)
  --alpha float         Float value between 0 and 1. 0 results in uniform
                        buffer (non-prioritized) (default: None)
  --beta_0 float        Float between 0 and 1. Anneals linearly to 1.
                        (default: None)
  --generate_config     Generates the modified config file and quits.
                        (default: None)
  --stop_at_threshold   Continues the training after reaching the threshold.
                        (default: None)
  --score_threshold float
                        Threshold value. Stop the training when the value is
                        reached, if s`top_at_threshold` is True. (default:
                        None)
```

The following rules are  applied to setting the parameters of the training:
- Each parameters, except `-c/--config-file`, should either be applied via the command line or the `config.yaml` file.
- If a parameter is provided both via the command line and also `config.yaml`, the value provided via the command line is used.
- None of the parameters have a default value. 
- A new configuration is created file if any parameter, except `-c/--config-file`, is provided via command line. The name of the new configuration becomes `config_<experiment_name>.yaml`, where `experiment_name` is one of the parameters that is provided via either `config.yaml` or the command line. 

There are a couple of specific parameters that deserve more explanation: 
- `--generate-config`: Boolean parameter that can be used to generate new configuration files from an already existing configuration file, without running the training. It is useful for generating various configuration files for parametric studies.
- `--stop-at-threshold`: This is a boolean parameters that determines if the training should continue after reaching the desired threshold score. If not provided via the command line, or set to `false` in the configuration file, the training runs until the last specified episode. 
- `--model` and `--agent`: Control the type of model and agent that is used by DQN-Algorithm. For these parameters, only the values inside the curly brackets can be specified. 
- `-c/--config-file` is the only parameter that cannot be specified inside the configuration file. 

#### Running the training
The training can be done either via command line or the jupyter notebook. In case of the notebook, a one liner like the following should be specified: 

```jupyter
! python <path_to_driver>/driver.py -c <path_to_config_file>/config.yaml 
```

#### Results
Results, such as the average score (averaged over the past 100 iterations) and weights of the model, are written in the same location as the location of the configuration file. These files are appended/prepended by the value of parameter `experiment-name`. For example, setting `experiment_name: ddqn_97` will create two files: `ddqn_97_checkpoint.pth` that contains the weights of the network and `scores_ddqn_97.dat` that contains the average scores of the training.
##### Note
The weights of the model can only be written once per training, and only if the specified `score_threshold` is reached. Only the first onset of hitting the `score_threshold` is recorded.

