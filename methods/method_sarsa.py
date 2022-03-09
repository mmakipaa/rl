from typing import Callable

from methods.method_episodic import MethodEpisodic
from agent.agent import Agent
from valuereps.vr_tabular import VrTabular

from utils.type_aliases import (TypeAction, TypeState, TypeValidState,
                                TypeEpisode, TypeSarsa, is_valid_state_tg)
import utils.constants as constants  # pylint: disable=consider-using-from-import


class Sarsa(MethodEpisodic):

    def __init__(self, *, name: str, agent: Agent, valuerep: VrTabular,
                 alpha_getter: Callable[..., float], gamma: float):

        super().__init__(name=name, agent=agent, alpha_getter=alpha_getter,gamma=gamma)

        self._valuerep: VrTabular = valuerep
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

            self.sarsa_update(current_state=current_state, current_action=action, reward=reward,
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


    def sarsa_update(self, *, current_state: TypeValidState, current_action: TypeAction, reward: float,
                next_state: TypeState, next_action: TypeAction|None, iteration: int) -> None:

        current_q, visit_count = self._valuerep.get_parameters(current_state, current_action,
                                                     'value', 'visit_count')

        visit_count = visit_count + 1

        if next_state == constants.TERMINAL_STATE:
            next_node_q = constants.DEFAULT_VALUE
        else:
            assert is_valid_state_tg(next_state) # -> TypeGuard[TypeValidState]
            assert next_action is not None  # assert for type checking
            next_node_q, = self._valuerep.get_parameters(next_state, next_action, 'value')

        alpha = self._get_alpha(visit_count=visit_count, iteration=iteration)

        new_q = current_q + alpha * (reward + self.gamma*next_node_q - current_q)

        self._valuerep.update_parameters(current_state, current_action,
                                         value = new_q, visit_count = visit_count)
