from abc import ABC, abstractmethod
from math import inf
from typing import Callable

from agent.agent import Agent
from valuereps.value_representation import ValueRepresentation

from utils.type_aliases import TypeLearnResult, TypeAction, TypeValidState, is_valid_state_tg
import utils.constants as constants  # pylint: disable=consider-using-from-import


class Method(ABC):
    def __init__(self, *, name: str, agent: Agent,
                 valuerep: ValueRepresentation|None = None, # pylint: disable=unused-argument
                 alpha_getter: Callable[..., float]|None, gamma: float):

        self.method_name: str = name

        self.agent: Agent = agent
        self._get_alpha: Callable[..., float]|None = alpha_getter
        self.gamma: float = gamma

        self._valuerep: ValueRepresentation


    @abstractmethod
    def learn(self, iterations: int, reporting_points: list[int]|None) -> TypeLearnResult:
        pass


    def evaluate(self, iterations: int) -> tuple[float, dict[str,int|float]]:

        episode_lengths =  { 'min_length': inf, 'mean_length' : 0, 'max_length': -inf }
        rewards: float = 0.0

        for current_iter in range(iterations):

            if current_iter % 1000 == 0:
                print(f"Evaluation round {current_iter}: {rewards}, {episode_lengths}")

            self.agent.initialize()
            current_state: TypeValidState = self.agent.get_state()

            episode_len = 0


            while True:

                episode_len += 1

                action: TypeAction = self.agent.select_action_by_target_policy(current_state)
                reward, next_state = self.agent.do_action(action)

                rewards += reward

                if next_state == constants.TERMINAL_STATE:

                    if episode_len > episode_lengths['max_length']:
                        episode_lengths['max_length'] = episode_len
                    if episode_len < episode_lengths['min_length']:
                        episode_lengths['min_length'] = episode_len

                    current_mean = episode_lengths['mean_length']
                    episode_lengths['mean_length'] = (current_mean +
                                                     1/(current_iter+1) * (episode_len - current_mean))

                    break

                assert is_valid_state_tg(next_state) # -> TypeGuard[TypeValidState]
                current_state = next_state

        return rewards, episode_lengths


    def _report(self, iteration: int) -> None:
        self._valuerep.report(self.method_name, iteration)
