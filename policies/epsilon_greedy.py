from math import inf
import random
from typing import Callable

from policies.policy import Policy

from valuereps.value_representation import  ValueRepresentation
from utils.type_aliases import TypeValidState, TypeActions, TypeAction


class EpsilonGreedy(Policy):

    def __init__(self, *, value_representation: ValueRepresentation, actions: TypeActions,
                 epsilon_getter: Callable[...,float]):

        super().__init__(value_representation, actions)

        self._get_epsilon = epsilon_getter


    def select_action_by_behavior_policy(self, state: TypeValidState, iteration: int) -> TypeAction:

        epsilon = self._get_state_epsilon(state, iteration)
        action = self._get_action_epsilon_greedy(state, epsilon)

        return action


    def select_action_by_target_policy(self, state: TypeValidState) -> TypeAction:

        action = self._valuerep.get_greedy_action(state)

        return action


    def _get_state_epsilon(self, state: TypeValidState, iteration: int) -> float:

        if hasattr(self._valuerep, 'get_state_visit_count'):
            visit_count = self._valuerep.get_state_visit_count(state) # type: ignore
        else:
            visit_count=None

        epsilon = self._get_epsilon(visit_count=visit_count, iteration=iteration)

        return epsilon


    def _get_action_epsilon_greedy(self, state: TypeValidState, epsilon: float) -> TypeAction:

        random_value = random.random()

        if random_value <= epsilon:
            action = self._valuerep.get_random_action()  # explore
        else:
            action = self._valuerep.get_greedy_action(state)  # choose greedy action

        return action


    def get_action_probability(self, state: TypeValidState, action: TypeAction, iteration: int) -> float:

        epsilon = self._get_state_epsilon(state, iteration)
        prob = self._get_action_probability(state, action, epsilon)
        return prob


    def _get_action_probability(self, state: TypeValidState, action: TypeAction,
                                epsilon: float) -> float:

        label_best_action, label_tied_best, label_not_best = (0,1,2)

        prob = - inf

        my_value = self._valuerep.get_value(state, action)

        action_type = label_best_action
        tied_count = 1

        for this_action in self._actions:

            if this_action == action:
                continue

            state_action_value_q = self._valuerep.get_value(state, this_action)

            if state_action_value_q > my_value:
                action_type = label_not_best
                break
            elif state_action_value_q == my_value:
                action_type = label_tied_best
                tied_count += 1

        if action_type == label_best_action:
            prob = 1 - epsilon + epsilon / len(self._actions)

        elif action_type == label_tied_best:
            prob = (1 - epsilon) / tied_count + epsilon / len(self._actions)

        elif action_type == label_not_best:
            prob = epsilon / len(self._actions)

        return prob


    def get_state_value(self, state: TypeValidState, iteration: int) -> float:

        expected_value: float = 0

        epsilon = self._get_state_epsilon(state, iteration)

        for this_action in self._actions:
            state_action_value_q = self._valuerep.get_value(state, this_action)
            probability = self._get_action_probability(state, this_action, epsilon)
            expected_value += probability * state_action_value_q

        return expected_value
