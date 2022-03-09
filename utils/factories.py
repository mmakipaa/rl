
from typing import Type

from valuereps.value_representation import ValueRepresentation

from valuereps.vr_fourier_cosine_blackjack import VrFourierCosineBlackjack
from valuereps.vr_fourier_cosine_maze import VrFourierCosineMaze
from valuereps.vr_polynomial_blackjack import VrPolynomial
from valuereps.vr_tile_coding_blackjack import VrTileCoding
from valuereps.vr_tabular import VrTabular

from methods.method import Method

from methods.method_monte_carlo_on import MonteCarloOn
from methods.method_monte_carlo_off import MonteCarloOff
from methods.method_q_learning import Qlearning
from methods.method_sarsa import Sarsa
from methods.method_sarsa_expected import SarsaExpected
from methods.method_sarsa_sg import SarsaSemigradient
from methods.method_lspi import MethodLspi

from environments.environment import Environment

from environments.environment_blackjack import EnvironmentBlackjack
from environments.environment_maze import EnvironmentMaze

from policies.epsilon_greedy import EpsilonGreedy
from agent.agent import Agent

from utils.sequence import SequenceCreator
from utils.type_aliases import TypeAgentConfig

from utils.reporting import reporter

_env_types: dict[str, type[Environment]] = {
    "blackjack": EnvironmentBlackjack,
    "maze": EnvironmentMaze
}

_vr_types: dict[str, dict[str, type[ValueRepresentation]]] = {
    "blackjack" : {
        "MonteCarloOn": VrTabular,
        "MonteCarloOff": VrTabular,
        "Qlearning": VrTabular,
        "Sarsa": VrTabular,
        "SarsaExpected": VrTabular,
        "SgFcSarsa": VrFourierCosineBlackjack,
        "SgPolSarsa": VrPolynomial,
        "SgTcSarsa": VrTileCoding,
        "LsTcBatch": VrTileCoding,
        "LsFcBatch": VrFourierCosineBlackjack,
        "LsPolBatch": VrPolynomial,
    },
    "maze" : {
        "MonteCarloOn": VrTabular,
        "MonteCarloOff": VrTabular,
        "Qlearning": VrTabular,
        "Sarsa": VrTabular,
        "SarsaExpected": VrTabular,
        "SgFcSarsa": VrFourierCosineMaze,
        "LsFcBatch": VrFourierCosineMaze,
    }
}

_method_types: dict[str, type[Method]] = {
    "MonteCarloOn": MonteCarloOn,
    "MonteCarloOff": MonteCarloOff,
    "Qlearning": Qlearning,
    "Sarsa": Sarsa,
    "SarsaExpected": SarsaExpected,
    "SgFcSarsa": SarsaSemigradient,
    "SgPolSarsa": SarsaSemigradient,
    "SgTcSarsa": SarsaSemigradient,
    "LsTcBatch": MethodLspi,
    "LsFcBatch": MethodLspi,
    "LsPolBatch": MethodLspi
}


def _get_env_class(env_type: str) -> Type[Environment]:

    if env_type in _env_types:
        return  _env_types[env_type]
    else:
        raise SystemExit(f"Environment factory: unknown env type given: {env_type}, "
                         "dont know which representation to return")

def _get_value_rep(vr_type: str, env_type: str) -> Type[ValueRepresentation]:

    if env_type in _vr_types and vr_type in _vr_types[env_type]:
        return  _vr_types[env_type][vr_type]
    else:
        raise SystemExit(f"Valuerep factory: unknown valuerep type given {vr_type}, "
                         f"not defined for environment {env_type}")

def _get_method_class(method_type: str) -> Type[Method]:

    if method_type in _method_types:
        return  _method_types[method_type]
    else:
        raise SystemExit(f"Method factory: unknown method type given: {method_type}, "
                         "dont know which method to return: {method_type}")

def create_environment(env_type: str, variant: str) -> Environment:

    environment_class = _get_env_class(env_type)
    environment = environment_class(variant=variant)

    reporter.setup_reporting_instance(environment)

    return environment

def create_method(agent_def: TypeAgentConfig, environment: Environment, env_type: str) -> Method:

    epsilon_type = agent_def["epsilon_type"]

    if epsilon_type == "NOT_USED":
        epsilon_getter = None
    else:
        epsilon_getter = SequenceCreator.get_schedule(
                            schedule_type=epsilon_type,
                            target_iterations=agent_def.get("epsilon_target_iterations"),
                            target_value=agent_def.get("epsilon_target"),
                            initial_value=agent_def.get("epsilon_initial"),
                            constant_value=agent_def.get("epsilon_constant"),
                            n0=agent_def.get("epsilon_n0"))

    alpha_type = agent_def["alpha_type"]

    if alpha_type == "NOT_USED":
        alpha_getter = None
    else:
        alpha_getter = SequenceCreator.get_schedule(
                        schedule_type=alpha_type,
                        target_iterations=agent_def.get("alpha_target_iterations"),
                        target_value=agent_def.get("alpha_target"),
                        initial_value=agent_def.get("alpha_initial"),
                        constant_value=agent_def.get("alpha_constant"),
                        n0=agent_def.get("alpha_a0"))

    method_str = agent_def.get("method")

    assert method_str is not None

    method_class = _get_method_class(method_str)

    value_rep_class = _get_value_rep(method_str, env_type)
    valuerep = value_rep_class()

    assert epsilon_getter is not None

    policy = EpsilonGreedy(value_representation=valuerep, actions=environment.get_actions(),
                           epsilon_getter=epsilon_getter)

    agent_name = agent_def["name"]

    agent = Agent(name=agent_name, environment=environment, policy=policy)

    gamma = agent_def["gamma"]

    method = method_class(name=agent_name, agent=agent, valuerep=valuerep,
                          alpha_getter=alpha_getter, gamma=gamma)

    return method
