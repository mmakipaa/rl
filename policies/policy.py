from abc import ABC, abstractmethod

from valuereps.value_representation import ValueRepresentation
from utils.type_aliases import TypeValidState, TypeActions, TypeAction


class Policy(ABC):

    def __init__(self, value_representation: ValueRepresentation, actions: TypeActions):

        self._valuerep = value_representation
        self._actions = actions

        self._valuerep.set_actions(self._actions)


    @abstractmethod
    def select_action_by_behavior_policy(self, state: TypeValidState, iteration: int) -> TypeAction:
        pass


    @abstractmethod
    def select_action_by_target_policy(self, state: TypeValidState) -> TypeAction:
        pass


    @abstractmethod
    def get_action_probability(self, state: TypeValidState, action: TypeAction, iteration: int) -> float:
        pass


    @abstractmethod
    def get_state_value(self, state: TypeValidState, iteration: int) -> float:
        pass
