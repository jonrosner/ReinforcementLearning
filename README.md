Reinforcement Learning for Adaptive Locomotion of a Snake-like Robot using Tensorflow
=====================================================
This repository contains my bachelor's thesis latex-project and the code I wrote for it.
The pdf-version of the thesis can be found under ``thesis/main.pdf``.
The presentation contains animations that cannot be rendered when viewed as pdf

Algorithms
====
| Directory | Algorithm |
|--------|------------------|
| DQN | [Deep-Q-Network](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) |
| DDPG | [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) |
| PPO | [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) |

Prerequisites
--------------
- Python 3.5.2 or newer
- [MuJoCo license](http://www.mujoco.org/) to run environments like the Swimmer-v1

Packages
-------------
All packages can be installed using pip
- tensorflow 1.3.0
- numpy 1.13.1
- gym 0.9.2
- scipy 0.19.1
- mujoco 0.5.7

Run Locally
-----------
- Clone the repo
- Go to the folder ``<algorithm-name>/scripts/`` and run ``python3 main.py``.

Each algorithm has different hyperparameters which can be set using eg:  
``python3 main.py --environment Swimmer-v1``

A list of all hyperparameters and its use can be found using:  
``python3 main.py --help``

Logs will be saved in ``../logs/<environment-name>/<time>``. They contain information about:
- ``rewards.csv`` episode rewards
- ``arguments.json`` arguments used in the run
- ``tf/`` all tensorflow-data including checkpoints
