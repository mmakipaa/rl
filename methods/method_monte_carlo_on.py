from methods.method_episodic import MethodEpisodic
from agent.agent import Agent
from valuereps.vr_tabular import VrTabular

from utils.type_aliases import TypeEpisode, is_valid_state_tg
import utils.constants as constants  # pylint: disable=consider-using-from-import


class MonteCarloOn(MethodEpisodic):

    def __init__(self, *, name: str, agent: Agent, valuerep: VrTabular,
                 alpha_getter: None = None, gamma: float):

        super().__init__(name=name, agent=agent, alpha_getter=alpha_getter, gamma=gamma)

        self._valuerep: VrTabular = valuerep
        self._get_alpha: None = None # overrides Optional[Callable] from Method, no alpha for MC


    def _learn_episode(self, iteration: int) -> TypeEpisode:

        episode = self._generate_mc_episode(iteration)
        self._iterate_episode_mc_on_policy(episode)

        return episode


    def _generate_mc_episode(self, iteration: int) -> TypeEpisode:

        episode = []

        current_state = self.agent.get_state()

        while True:

            sars = self.agent.do_sars(current_state, iteration)
            episode.append(sars)

            if sars['next_state'] == constants.TERMINAL_STATE:
                return episode

            assert is_valid_state_tg(sars['next_state']) # -> TypeGuard[TypeValidState]
            current_state = sars['next_state']


    def _iterate_episode_mc_on_policy(self, episode: TypeEpisode) -> None:

        returns_g: float = 0

        for step in reversed(episode):

            current_state = step['state']
            current_action = step['action']
            current_r = step['reward']

            returns_g = self.gamma * returns_g + current_r

            current_q, visit_count = self._valuerep.get_parameters(current_state, current_action,
                                                                   'value', 'visit_count')

            visit_count = visit_count + 1
            new_q = current_q + 1 / visit_count * (returns_g - current_q)

            self._valuerep.update_parameters(current_state, current_action,
                                             value = new_q, visit_count = visit_count)
