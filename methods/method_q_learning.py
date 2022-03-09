from typing import Callable

from methods.method_episodic import MethodEpisodic
from agent.agent import Agent
from valuereps.vr_tabular import VrTabular

from utils.type_aliases import TypeState, TypeValidState, TypeAction, TypeEpisode, is_valid_state_tg
import utils.constants as constants  # pylint: disable=consider-using-from-import


class Qlearning(MethodEpisodic):

    def __init__(self, *, name: str, agent: Agent, valuerep: VrTabular,
                 alpha_getter: Callable[..., float], gamma: float):

        super().__init__(name=name, agent=agent, alpha_getter=alpha_getter,gamma=gamma)

        self._valuerep: VrTabular = valuerep
        self._get_alpha: Callable[..., float] = alpha_getter # overrides Optional[Callable] from Method


    def _learn_episode(self, iteration: int) -> TypeEpisode:

        episode = []

        current_state: TypeValidState = self.agent.get_state()

        while True:

            sars = self.agent.do_sars(current_state, iteration)

            self._q_update(current_state=sars['state'], current_action=sars['action'],
                           reward=sars['reward'], next_state=sars['next_state'],
                           iteration=iteration)

            episode.append(sars)

            if sars['next_state'] == constants.TERMINAL_STATE:
                return episode

            assert is_valid_state_tg(sars['next_state']) # -> TypeGuard[TypeValidState]
            current_state = sars['next_state']


    def _q_update(self, *, current_state: TypeValidState, current_action: TypeAction,
                  reward: float, next_state: TypeState, iteration: int) -> None:

        current_q, visit_count = self._valuerep.get_parameters(current_state, current_action,
                                                               'value', 'visit_count')
        next_node_q: float

        visit_count = visit_count + 1

        if next_state == constants.TERMINAL_STATE:
            next_node_q = constants.DEFAULT_VALUE
        else:
            assert is_valid_state_tg(next_state) # -> TypeGuard[TypeValidState]
            next_node_q = self._valuerep.get_max_value(next_state)

        alpha = self._get_alpha(visit_count=visit_count, iteration=iteration)

        new_q = current_q + alpha * (reward + self.gamma*next_node_q - current_q)

        self._valuerep.update_parameters(current_state, current_action,
                                         value = new_q, visit_count = visit_count)
