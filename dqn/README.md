# Introduction
<div style="text-align:center">

![image](./pics/banana_env_animation.gif)
</div>

## Environment
The Unity Banana Collector is a navigation task. An agent must collect yellow bananas (+1 reward) while avoiding blue bananas (-1 reward). The objective is to collect as many of the former while avoiding the latter, maximizing the total reward. The episode ends after a fixed number of steps, specified by the user or trainer, typically in the hundreds, often around 300.

The state space is 37-dimensional, representing ray-based perception of nearby objects, the agent's velocity, etc..

The action space is a 4-dimensional discrete space that control the agent's movement in a coordinate system relative to its orientation (forward, backward, left and right).

## Goal
The goal is to train an agent that can reach an average reward of at least +13 over 100 episodes.



## Installation and dependencies
Unfortunately for the current version of the unity environments, including the banana collector environment, they can only be used with python 3.6.8, which is still accessible via conda and can be installed using the followings:

```bash
conda create -n myenv python=3.6.8 
conda activate myenv
pip install -r requirements.txt
```

## Running the code and training the agent

### Scope of this project
This project implements and runs experiments for the following three subvariants of DQN Algorithm: Case1  vanilla DQN plus the target network and prioritized experience replay buffer, referred to from hereon as PER. Case2 include the previous implementation plus the Double DQN. This can be done with a minimal change to the previous implementation. Case3 adds to Case 2 the Dueling Network.

### Code organization and running the training
All the code for the described three cases are in `dqn.py` file. To run the experiment, the user can simply import the routine `run_experiment1` from the file and provide a `Config` object with desired hyperparameters. For each case, we run 7 experiments and collect the scores in three separate json files (available at the root of the project).

### Case 1
```python 
config = Config(
    seed=101, batch_size=64, n_episodes=600, lr=1e-4,
    gamma=0.99, update_every=4, soft_update=1e-3,
    buffer_size=100_000, buffer_alpha=0.6, buffer_beta=0.4, buffer_eps=1e-5, buffer_beta_anneal_steps=100_000,
    dueling_network=False,
    double_dqn=False,
    t_max=500, score_threshold=13.0, score_window=100, eps_init=1.0, eps_final=0.01, eps_decay=0.975,
    device='cpu'
) 
``` 

### Case 2
### Case 3


### Comparison of different algorithms

<div style="width:90%; margin:auto;">

![](pics/score_comparison.png)
</div>



### Sample plots from an agent that solves the environment

<div style="width:80%; margin:auto;">

![](pics/plot_single_runs.png)
</div>


# A Survey of DQN Methods

This is a list of literature review and some explanations of various methods from DQN method. It starts with basic methods for DQN, as explained in Deepmind paper:
- Two separate networks for argmax value (called target network) and one for learning called local network
- Soft update of the network $\theta_{\rm target} \leftarrow \tau \theta_{\rm target} + (1 - \tau) \theta_{\rm local}$ for some $0 \lt \tau \ll 1$.
- Priotorized experience replay buffer


$$
\large
\begin{align*}
Q_{\rm{local}}(s, a; \theta) \leftarrow Q_{\rm{local}}(s, a; \theta) 
+ \alpha \big[ Q_{\rm{target}}(s, a; \theta) - Q_{\rm{local}}(s, a; \theta) \big],
\end{align*}

\\

\begin{align*}
Q_{\rm{target}}(s, a; \theta) &= r + \gamma \max_{a'} Q_{\rm{local}}(s', a'; \theta),
\end{align*}
$$

The loss and gradient are calculated as follows:


$$
l = \left\Vert \left( Q_{\rm local}  - Q_{\rm target} \right)^2 \right\Vert \\
l = \nabla_{\theta_{\rm local}}\left\Vert \left( Q_{\rm local}  - Q_{\rm target} \right)^2 \right\Vert \\
g = \nabla_{\theta_{\rm local}} l
$$
We then gradually build our way to a rainbow DQN method. 

## Double Q-Learning
## Dueling Networks
## Multistep Learning
## Noisy Networks
## Distributional RL (C51)