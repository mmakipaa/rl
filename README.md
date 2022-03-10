# Basic Reinforcement Learning Algorithms

This repository contains implementations of basic Reinforcement Learning (RL) algorithms in Python.

## Getting started

Using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environments: 

```sh
$ git clone https://github.com/mmakipaa/rl
$ conda env create -n <name_for_env> -f environment.yml
$ conda activate <name_for_env>
```

Run a test for 10000 episodes of blackjack with methods defined in [configs/all_agents.yaml](configs/all_agents.yaml). Store results in `folder/filename.pik`

```py
python run.py -e blackjack -i 10000 -c all_agents -r folder/filename
```

## Command line arguments

With `python run.py`, use the following command line arguments:

`-e, --environment` (requred)

environment to use, either `blackjack` or `maze`

`-i, --iterations` (requred)

number of learning episodes to run for episodic methods (including episodic TD). For LSPI-batch method, number of episodes to sample

`-c, --configfile` (requred)

config file name - a yaml config file defining agents (methods) to include. If no path is given as a part of the argument, default folder `./configs/` is used

`-r, --report` (optional)

report file name - use the argument to override the default report file name which is `../testruns/<environment>_<configfile>_<iterations>.pik`. 

Argument value `foofile` would create report file `../testruns/foofile.pik`. Note that the folder `../testruns` must exist if no path is given as a part of the argument.

To permanently change configfile or report paths, modify values of `REPORT_FOLDER` and `CONFIGS_FOLDER` in [utils/constants.py](utils/constants.py)

## Environments

### Blackjack

A simple version of blackjack with infinite deck: Dealer stands at 17. Ties give reward of 0, losing -1, winning +1. No doubling down or splitting. "Naturals", meaning that two initial cards give the sum of 21, are not taken into account as no decision for the agent is needed. Note that ignoring naturals also reduces the rewards received during evaluation phase.

### Maze

A simple maze with noisy moves. `simple` maze implements the "canonical maze" used widely as a Dynamic Programming example. `complex` maze is somewhat, well, more complex. Additional maze structures can be defined in [environments/maze_configs.py](environments/maze_configs.py)

See [https://github.com/mmakipaa/dp](https://github.com/mmakipaa/dp) for basic Dynamic Programming algorithms applied on the included maze configurations.

## Methods

The following Reinforcement Learning methods have been implemented:

### Tabular value representation

For these methods, action-value function `Q(s,a)` is represented as a table structure

#### Updates based on full episodes

- On-policy Monte Carlo 
- Off-policy Monte Carlo with weighted importance sampling

#### Temporal-Difference methods

- Sarsa
- Expected Sarsa
- Q-learning

### Approximate linear value representation

With approximate methods, action-value function `Q(s,a;w)` is approximated using a linear function of weight vector `w`. Values are updated applying TD(0) semi-gradient descent 

- Semi-gradient Sarsa

### Batch method

The above methods use episodes to update estimates of action-value function, even for TD updates. For batch method, we create an experience consisting of a large number of `{S,A,R,S}` samples and use the batch to derive an estimate for weights of a linear model.

- Least-squares policy iteration LSPI-LSTDQ

## Approximate value representations

For approximate linear and batch methods, the following value representations are available:

- Fourier Cosine basis
- Polynomial basis 
- Simple Tile Coding 

## Agent configuration

Methods to run are defined in a yaml `configfile`, name of which is given as command line parameter.

The following `method` combinations are available:

| Method key| Description |
| --- | --- |
| MonteCarloOn | On-policy Monte Carlo, full episodes and tabular value representation |
| MonteCarloOff | Off-policy Monte Carlo, full episodes and tabular value representation |
| Sarsa | Sarsa TD(0) using tabular value representation |
| SarsaExpected | Expected Sarsa TD(0) using tabular value representation |
| Qlearning | Q-learning TD(0) using tabular value representation |
| SgFcSarsa | Sarsa TD(0) semi-gradient with linear function approximation using Fourier Cosine basis |
| SgPolSarsa | Sarsa TD(0) semi-gradient, polynomial basis |
| SgTcSarsa | Sarsa TD(0) semi-gradient, linear approximation using TileCoding |
| LsFcBatch | LSPI batch method, linear approximation using Fourier Cosine basis |
| LsPolBatch | LSPI batch method, linear approximation using Polynomial basis |
| LsPolBatch | LSPI batch method, linear approximation with Tile Coding |

All of the above methods are available for environment `blackjack`. For `maze` environments, all tabular methods as well as approximate Fourier cosine basis, either applying TD Sarsa or batch LSPI, are available.

### Learning schedules

For learning parameters `alpha` and `epsilon`, following learning schedules can be defined:

- `CONSTANT`: constant value through out learning

- `INV_VISIT_COUNT`: Inverse time based on state visit count

- `INV_ROUNDS`: Inverse time based on current learning iteration
    ```py
        value = self.initial_value / current_time
    ```
- `INV_VISIT_COUNT_SCALED`: Inverse time based on state visit count, scaled with parameter `n0`
- `INV_ROUNDS_SCALED`: Inverse time based on current learning iteration, scaled with parameter `n0`
    ```py
        value = self.initial_value * (self.n0 + 1) / (self.n0 + current_time)
    ```
- `INV_ROUNDS_TARGET_AT`: Inverse time reaching defined target at 90% of iterations
- `EXPONENTIAL`: Exponentially decaying value based on current learning iteration
- `EXPONENTIAL_TARGET_AT`: Exponential decay, reaching defined target value at 90% of iterations

Note that state visit count -based schedules are only applicable for tabular methods.

### Configuration examples

An agent applying On-policy Monte Carlo. Exploration parameter _epsilon_ for e-greedy policy starts at 1 and decays according to state visit count, with scaling of 50. _Alpha_ is not used for Monte Carlo methods. Discount factor _gamma_ is set at 1.0.
```yml
- 
  name: MC_ON_N0_50
  method: MonteCarloOn
  epsilon_type: INV_VISIT_COUNT_SCALED
  epsilon_initial: 1
  epsilon_n0: 50
  alpha_type: NOT_USED
  gamma: 1.0
```
An agent applying semi-gradient Sarsa with Fourier Cosine basis. _Epsilon_ decays exponentially, starting from 0.3 and reaching 0.05 at 90% of iterations (number of epsodes to run is given as command line parameter). Similarly learning-rate parameter _alpha_ decays according to inverse time schedule towards target value of 0.001. Discount factor _gamma_ set at 1.0.
```yml
- 
  name: SG_SARSA_FC
  method: SgFcSarsa
  epsilon_type: EXPONENTIAL_TARGET_AT
  epsilon_initial: 0.3
  epsilon_target: 0.05
  alpha_type: INV_ROUNDS_TARGET_AT
  alpha_initial: 0.01
  alpha_target: 0.001
  gamma: 1.0
```
An agent applying batch LSPI and using polynomial basis for action-value approximation. Constant _epsilon_ of 1 used during experience sampling (i.e. a fully random policy). Discount factor _gamma_ set at 1.0.
```yml
-
  name: LSPI_POL
  method: LsPolBatch
  epsilon_type: CONSTANT
  epsilon_constant: 1
  alpha_type: NOT_USED
  gamma: 1.0
```

## Report file format

A report file collects the results of a test run as a serialized [pickle](https://docs.python.org/3/library/pickle.html#module-pickle]).

A fixed number of reporting points is set at log-intervals for episodic methods. Batch method reports after each iteration.

The report file contains a `dict` with the following string keys:

- `agents`: agent definition (from config file), a `list` of `dict`

- `report`: a [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) with value (and visit count, if available) for each agent and defined state-action pair at each reporting time point during learning

- `tr_rewards`: a DataFrame containing cumulative reward at each reporting time point during learning

- `ev_rewards`: a DataFrame containing cumulative reward collected during evaluation 

- `episode_lenghts`: a DataFrame with min, average and max episode lengths during learning and evaluation

- `maze_config`: maze config `dict` (included only if environment is `maze`)


## Additional configuration

See [utils/constants.py](utils/constants.py) for more configurable settings

## Type checking source code

For type checking, `mypy --strict` has been applied. To run type checking create `mypy.ini` config file with the following contents:

```
[mypy]

mypy_path = <path to cloned folder>
plugins = numpy.typing.mypy_plugin
```
Then run mypy, e.g.:

```
mypy --namespace-packages --strict --show-error-codes --show-column-numbers .
```
	
## Motivation

Motivation behind this experiment has been two-fold:

First, to gain sufficent understading of Reinforcement Learning through hands-on implementation of basic algorithms. This target setting effectively ruled out the use of ready ML libraries, plenty of which are readily available (see e.g. [OpenAI](https://github.com/openai/baselines), [TF-Agents](https://github.com/tensorflow/agents) or [acme](https://github.com/deepmind/acme)). Arguably, an approach utilizing current state-of-the-art would have enabled faster progress towards, say, Deep learning methods. 

Second, to gain experience in using Python beyond simple scripting or notebook use, and especially, modeling a problem domain and associated “real-world” concepts (in this case entities such as agents, policies, environments...) using Python classes and data structures: The goal was to get a feel for what is considered "Pythonic" and why. This goal remains work-in-progress.

So, exiting challenges remain...

---