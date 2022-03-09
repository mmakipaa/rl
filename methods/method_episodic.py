from abc import abstractmethod
from math import inf

from methods.method import Method

from utils.type_aliases import TypeEpisode, TypeLearnResult


class MethodEpisodic(Method):

    def learn(self, iterations: int, reporting_points: list[int]|None) -> TypeLearnResult:

        reports: list[int]|None

        if reporting_points is not None:
            reports = reporting_points[:] # dont want to modify the original list
        else:
            reports = None

        rewards = [] # a list of lists: [[str, int, float],... ]
        received_reward: float = 0

        print(f"\nMethod {self.method_name} learning for {iterations} iterations")

        episode_lengths =  { 'min_length': inf, 'mean_length' : 0, 'max_length': -inf }

        if reports:
            print("Learning round intial")
            self._report(0)
            rewards.append([self.method_name, 0, received_reward])

        for iteration in range(1,iterations+1):

            self.agent.initialize()
            episode = self._learn_episode(iteration) # _learn_episode to be overridden by concrete class

            episode_len = len(episode)

            if episode_len > episode_lengths['max_length']:
                episode_lengths['max_length'] = episode_len
            if episode_len < episode_lengths['min_length']:
                episode_lengths['min_length'] = episode_len

            current_mean = episode_lengths['mean_length']
            episode_lengths['mean_length'] = current_mean + 1 / iteration * (episode_len - current_mean)

            received_reward += sum([step['reward'] for step in episode])

            if reports and iteration == reports[0]:
                print(f"Learning round {iteration}: {received_reward}, {episode_lengths}")

                self._report(iteration)
                rewards.append([self.method_name, iteration, received_reward])
                reports.pop(0)

        return rewards, episode_lengths


    @abstractmethod
    def _learn_episode(self, iteration: int) -> TypeEpisode:
        pass
