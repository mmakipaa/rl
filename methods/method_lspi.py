import numpy as np

from agent.agent import Agent
from methods.method_batch import MethodBatch
from valuereps.vr_approximate import VrApproximateLinear

from utils.type_aliases import TypeSample, TypeNDarray64, is_valid_state_tg
import utils.constants as constants  # pylint: disable=consider-using-from-import


class MethodLspi(MethodBatch):

    def __init__(self, *, name: str, agent: Agent, valuerep: VrApproximateLinear,
                 alpha_getter: None = None, gamma: float):

        super().__init__(name=name, agent=agent, alpha_getter=alpha_getter,gamma=gamma)

        self._valuerep: VrApproximateLinear = valuerep

        self._get_alpha: None = None  # overrides Optional[Callable] from Method, no alpha for Lspi

        self.max_iterations = 30    #arbitary defaults, use set_batch_learning_parameters
        self.stopping_limit = 0.005

        self.fdim: int = self._valuerep.get_feature_dimension()
        self.matrix_a: TypeNDarray64  = np.zeros((self.fdim, self.fdim), dtype=np.float64)
        # np.fill_diagonal(self.matrix_a, 0.001) #0.000000001

        self.vector_b: TypeNDarray64 = np.zeros((self.fdim, 1), dtype=np.float64)


    def set_batch_learning_parameters(self, *, max_iterations: int|None = None,
                                      stopping_limit: float|None = None) -> None:

        if max_iterations is not None:
            self.max_iterations = max_iterations

        if stopping_limit is not None:
            self.stopping_limit = stopping_limit


    def _learn_batch(self, samples: TypeSample, reports: list[int]|None) -> int:
        return self._lspi(samples, reports)


    def _lspi(self, samples: TypeSample, reports: list[int]|None) -> int:

        i = 1

        while i <= self.max_iterations:
            change = self._lstdq(samples)

            if reports and i == reports[0]:
                print(f"Learning round {i}: change is {change}")

                self._report(i)
                reports.pop(0)


            if change < self.stopping_limit:
                return i
            i += 1

        return i


    def _lstdq(self, samples: TypeSample) -> float:

        for sample in samples:
            current_state = sample['state']
            current_action = sample['action']
            reward = sample['reward']
            next_state = sample['next_state']

            current_features: TypeNDarray64 = np.array([
                self._valuerep.get_features(current_state, current_action) ], dtype=np.float64).T

            if next_state == constants.TERMINAL_STATE:
                next_features: TypeNDarray64 = np.zeros((self.fdim, 1), dtype=np.float64)
            else:
                assert is_valid_state_tg(next_state) # -> TypeGuard[TypeValidState]
                next_action = self._valuerep.get_greedy_action(next_state)
                next_features = np.array([ self._valuerep.get_features(next_state, next_action) ],
                                                            dtype=np.float64).T

            self.matrix_a = (self.matrix_a
                             + current_features
                             @ ( current_features - self.gamma * next_features).T)

            self.vector_b = self.vector_b + current_features * reward

        #weights = np.linalg.inv(self.matrix_a) @ self.vector_b
        #weights = np.linalg.solve(self.matrix_a, self.vector_b)

        weights = np.linalg.pinv(self.matrix_a) @ self.vector_b
        change = self._valuerep.set_weights_to(weights[:,0].tolist())

        return change
