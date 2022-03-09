from agent.agent import Agent
from methods.method_episodic import MethodEpisodic
from valuereps.vr_tabular import VrTabular

from utils.type_aliases import (TypeAction, TypeValidState,
                                TypeEpisode, TypeWeightedSars, TypeWeightedEpisode,
                                is_valid_state_tg)
import utils.constants as constants  # pylint: disable=consider-using-from-import


class MonteCarloOff(MethodEpisodic):

    def __init__(self, *, name: str, agent: Agent, valuerep: VrTabular,
                 alpha_getter: None = None, gamma: float):

        super().__init__(name=name, agent=agent, alpha_getter=alpha_getter,gamma=gamma)

        self._valuerep: VrTabular = valuerep
        self._get_alpha: None = None # overrides Optional[Callable] from Method, no alpha for MC


    def _learn_episode(self, iteration: int) -> TypeEpisode:
        episode = self._generate_mc_episode_off_policy(iteration)
        self._iterate_episode_mc_off_policy(episode)

        return episode


    def _generate_mc_episode_off_policy(self, iteration: int) -> TypeWeightedEpisode:

        episode = []

        current_state: TypeValidState = self.agent.get_state()

        while True:
            action: TypeAction = self.agent.select_action_by_behavior_policy(current_state, iteration)
            prob: float = self.agent.get_action_probability(current_state, action, iteration)

            reward, next_state = self.agent.do_action(action)

            wsars: TypeWeightedSars = {
                'state': current_state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'prob': prob
            }
            episode.append(wsars)

            if next_state == constants.TERMINAL_STATE:
                return episode

            assert is_valid_state_tg(next_state) # -> TypeGuard[TypeValidState]
            current_state = next_state


    def _iterate_episode_mc_off_policy(self, episode: TypeWeightedEpisode) -> None:

        returns_g: float = 0
        weighted_w: float = 1

        for step in reversed(episode):

            current_state = step['state']
            current_action = step['action']
            current_prob_b = step['prob']
            current_r = step['reward']

            returns_g = self.gamma * returns_g + current_r

            current_q, sa_visitcount, cumulative_c \
                        = self._valuerep.get_parameters(current_state, current_action,
                                                        'value', 'visit_count', 'cumulative_count')

            cumulative_c = cumulative_c + weighted_w
            sa_visitcount = sa_visitcount + 1

            new_q = current_q + weighted_w/cumulative_c * (returns_g - current_q)

            self._valuerep.update_parameters(current_state, current_action,
                                             value = new_q,
                                             visit_count = sa_visitcount,
                                             cumulative_count = cumulative_c )

            weighted_w = weighted_w * 1/current_prob_b

            if not self._valuerep.is_best_action(current_state, current_action):
                return
