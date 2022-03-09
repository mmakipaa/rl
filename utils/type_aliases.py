from typing import Any, TypeGuard, Union, Callable, TypedDict, TypeAlias

import numpy as np

# configuration

class TypeAgentConfig(TypedDict):
    name: str
    method: str
    epsilon_type: str
    epsilon_constant: int|None
    epsilon_initial: float|None
    epsilon_n0: int|None
    epsilon_target_iterations: int|None
    epsilon_target: float|None
    alpha_type: str
    alpha_constant: float|None
    alpha_initial: float|None
    alpha_a0: float|None
    alpha_target_iterations: int|None
    alpha_target: float|None
    gamma: float


# Maze configurations

class TypeMazeStructure(TypedDict):
    size: tuple[int,int]
    walls: list[tuple[int,int]]
    terminal_states: list[tuple[int,int]]
    rewards: dict[tuple[int,int], float]

class TypeMazeConfiguration(TypedDict):
    maze_structure: TypeMazeStructure
    living_cost: float
    noise: float

TypeMazeConfigs: TypeAlias = dict[str,TypeMazeConfiguration]


# states

TypeTerminalState: TypeAlias = str  # defined in constants.TERMINAL_STATE

TypeValidState: TypeAlias = tuple[Any, ...]
TypeValidMazeState: TypeAlias = tuple[int, int]
TypeValidBlackjackState: TypeAlias = tuple[int, int, bool]

TypeState: TypeAlias  = Union[TypeValidState, TypeTerminalState]
TypeMazeState: TypeAlias = Union[TypeValidMazeState, TypeTerminalState]
TypeBlackjackState: TypeAlias = Union[TypeValidBlackjackState, TypeTerminalState]


def is_valid_state_tg(state: TypeState) -> TypeGuard[TypeValidState]:

    if isinstance(state, tuple):
        return True
    else:
        return False


def is_valid_blackjack_state_tg(state: TypeState) -> TypeGuard[TypeValidBlackjackState]:
    # tuple[int, int, bool]
    return (isinstance(state, tuple) and
            isinstance(state[0], int) and
            isinstance(state[1], int) and
            isinstance(state[2], bool))


def is_valid_maze_state_tg(state: TypeState) -> TypeGuard[TypeValidMazeState]:
    # tuple[int, int]
    return (isinstance(state, tuple) and
            isinstance(state[0], int) and
            isinstance(state[1], int))


# actions

TypeMazeAction: TypeAlias = int
TypeMazeActions: TypeAlias = tuple[TypeMazeAction, TypeMazeAction, TypeMazeAction, TypeMazeAction]

TypeBlackjackAction: TypeAlias = bool
TypeBlackjackActions: TypeAlias = tuple[TypeBlackjackAction, TypeBlackjackAction]

TypeAction: TypeAlias = TypeMazeAction | TypeBlackjackAction
TypeActions: TypeAlias  = tuple[TypeAction, ...]


def is_valid_blackjack_action_tg(action: TypeAction) -> TypeGuard[TypeBlackjackAction]:
    return isinstance(action, bool)


def is_valid_maze_action_tg(action: TypeAction) -> TypeGuard[TypeMazeAction]:
    return isinstance(action, int)


# state and action for storage dict

TypeStateAction: TypeAlias = tuple[Any, ...]
TypeStorageDict: TypeAlias = dict[str, Union[int, float]]


# numpy types

TypeNDarray64: TypeAlias = np.ndarray[Any, np.dtype[np.float64]]
TypeNDarrayBool: TypeAlias = np.ndarray[Any, np.dtype[np.bool_]]

def is_scalar_64_tg(value: TypeNDarray64) -> TypeGuard[np.float64]:
    return np.isscalar(value) and (type(value) in (np.int64,np.float64))


# episode and td steps

class TypeSars(TypedDict):
    state: TypeValidState
    action: TypeAction
    reward: float
    next_state: TypeState

class TypeWeightedSars(TypedDict):
    state: TypeValidState
    action: TypeAction
    reward: float
    next_state: TypeState
    prob: float

class TypeSarsa(TypedDict):
    state: TypeValidState
    action: TypeAction
    reward: float
    next_state: TypeState
    next_action: TypeAction|None

class TypeSarsaE(TypedDict):
    state: TypeValidState
    action: TypeAction
    reward: float
    next_state: TypeState
    next_state_value: float

TypeEpisode: TypeAlias  = Union[list[TypeSars], list[TypeWeightedSars],
                                list[TypeSarsa], list[TypeSarsaE]]

TypeWeightedEpisode: TypeAlias = list[TypeWeightedSars]
TypeEpisodeSarsa: TypeAlias  = list[TypeSarsa]


# samples, results and reports

TypeSample: TypeAlias  = list[TypeSars]
TypeRewardsAtIterations: TypeAlias = list[list[object]]
TypeEpisodeLenghts: TypeAlias = dict[str, int|float]
TypeLearnResult: TypeAlias = tuple[TypeRewardsAtIterations, TypeEpisodeLenghts]
TypeReportList: TypeAlias = list[list[Union[str, int, float, None]]]


# interface functions

TypeValueGetter: TypeAlias = Callable[[TypeValidState, TypeAction], float]
TypeVisitCountGetter: TypeAlias = Callable[[TypeValidState, TypeAction], int]
