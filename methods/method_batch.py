from abc import abstractmethod
from math import inf

from methods.method import Method

from utils.type_aliases import TypeLearnResult, TypeSample, is_valid_state_tg
import utils.constants as constants  # pylint: disable=consider-using-from-import


class MethodBatch(Method):

    def learn(self, iterations: int, reporting_points: list[int]|None) -> TypeLearnResult:

        reports: list[int]|None

        if reporting_points is not None:
            reports = reporting_points[:] # dont want to modify the original list
        else:
            reports = None

        rewards = []

        print(f"Method {self.method_name} learning with batch of {iterations} samples")

        samples, episode_lengths, received_reward = self._sample_environment(iterations)

        if reports:
            print("Learning round intial")
            self._report(0)
            rewards.append([self.method_name, 0, received_reward])

        _completed_iterations = self._learn_batch(samples, reports)

        return rewards, episode_lengths


    def _sample_environment(self, sample_episodes: int) -> tuple[TypeSample, dict[str,int|float], float]:

        episode_lengths =  { 'min_length': inf, 'mean_length' : 0, 'max_length': -inf }
        received_reward: float = 0

        samples = []

        for i in range(sample_episodes): # pylint: disable=unused-variable

            episode = []

            self.agent.initialize()
            current_state = self.agent.get_state()

            while True:

                sars = self.agent.do_sars(current_state, 0) # random policy, iteration not used
                episode.append(sars)

                if sars['next_state'] == constants.TERMINAL_STATE:
                    break

                if is_valid_state_tg(sars['next_state']): # -> TypeGuard[TypeValidState]
                    current_state = sars['next_state']
                else:
                    raise SystemExit("MethodBatch: sample_environment: Assumed valid state")

            episode_len = len(episode)

            if episode_len > episode_lengths['max_length']:
                episode_lengths['max_length'] = episode_len
            if episode_len < episode_lengths['min_length']:
                episode_lengths['min_length'] = episode_len

            current_mean = episode_lengths['mean_length']
            episode_lengths['mean_length'] = current_mean + 1 / (i+1) * (episode_len - current_mean)


            received_reward += sum([step['reward'] for step in episode])

            samples.extend(episode)

        return samples, episode_lengths, received_reward


    @abstractmethod
    def _learn_batch(self, samples: TypeSample, reports: list[int]|None) -> int:
        pass
