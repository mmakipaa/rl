import math
from abc import ABC, abstractmethod

import random

from utils.type_aliases import TypeValidState, TypeActions, TypeAction


class ValueRepresentation(ABC):

    def __init__(self) -> None:

        self.actions: TypeActions|None = None


    def set_actions(self, actions: TypeActions) -> None:
        self.actions = actions

    def get_random_action(self) -> TypeAction:

        assert self.actions is not None  # assert for type checking

        random_action: TypeAction = random.choice(self.actions)

        return random_action


    def get_greedy_action(self, state: TypeValidState) -> TypeAction:

        assert self.actions is not None  # assert for type checking

        best_value = -math.inf
        best_actions: list[TypeAction] = []

        for this_action in self.actions:

            state_action_value_q = self.get_value(state, this_action)


            if state_action_value_q > best_value:
                best_value = state_action_value_q
                best_actions = [ this_action ]

            elif state_action_value_q == best_value:
                best_actions.append(this_action)

        assert best_actions is not None

        best_action: TypeAction = random.choice(best_actions)

        return best_action


    @abstractmethod
    def get_value(self, state: TypeValidState, action: TypeAction) -> float:
        pass


    @abstractmethod
    def report(self, name: str, iteration: int) -> None:
        pass
