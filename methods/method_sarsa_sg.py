from typing import Callable

import numpy as np

from methods.method_episodic import MethodEpisodic
from agent.agent import Agent
from valuereps.vr_approximate import VrApproximateLinear

from utils.type_aliases import (TypeAction, TypeState, TypeValidState, TypeEpisode, TypeSarsa,
                                is_valid_state_tg)
import utils.constants as constants  # pylint: disable=consider-using-from-import


class SarsaSemigradient(MethodEpisodic):

    def __init__(self, *, name: str, agent: Agent, valuerep: VrApproximateLinear,
                 alpha_getter: Callable[..., float], gamma: float):

        super().__init__(name=name, agent=agent, alpha_getter=alpha_getter,gamma=gamma)

        self._valuerep: VrApproximateLinear = valuerep
        self._get_alpha: Callable[..., float] = alpha_getter # overrides Optional[Callable] from Method

    def _learn_episode(self, iteration: int) -> TypeEpisode:

        episode = []

        current_state = self.agent.get_state()
        action = self.agent.select_action_by_behavior_policy(current_state, iteration)

        while True:

            reward, next_state = self.agent.do_action(action)

            if next_state == constants.TERMINAL_STATE:
                next_action = None
            else:
                assert is_valid_state_tg(next_state) # -> TypeGuard[TypeValidState]
                next_action = self.agent.select_action_by_behavior_policy(next_state, iteration)

            self._sarsa_sg_update(current_state=current_state, current_action=action, reward=reward,
                              next_state=next_state, next_action=next_action, iteration=iteration)

            sarsa: TypeSarsa = {
                'state': current_state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'next_action': next_action
            }
            episode.append(sarsa)

            if next_state == constants.TERMINAL_STATE:
                return episode

            assert is_valid_state_tg(next_state) # -> TypeGuard[TypeValidState]
            current_state = next_state

            assert next_action is not None  # assert for type checking
            action = next_action


    def _sarsa_sg_update(self, *, current_state: TypeValidState, current_action: TypeAction,
                        reward: float, next_state: TypeState,
                        next_action: TypeAction|None, iteration: int) -> None:

        current_q = self._valuerep.get_value(current_state, current_action)

        if next_state == constants.TERMINAL_STATE:
            next_node_q: float = constants.DEFAULT_VALUE
        else:
            assert is_valid_state_tg(next_state) # -> TypeGuard[TypeValidState]
            assert next_action is not None  # assert for type checking
            next_node_q = self._valuerep.get_value(next_state, next_action)

        assert current_q is not None  # assert for type checking
        assert next_node_q is not None  # assert for type checking

        alpha = self._get_alpha(iteration=iteration)

        gradient = self._valuerep.get_gradient(current_state, current_action)
        change = alpha * ( reward + self.gamma*next_node_q -  current_q)
        update = np.multiply(change, gradient)

        self._valuerep.update_weights_by(update)
