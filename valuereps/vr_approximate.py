from abc import abstractmethod

import numpy as np

from valuereps.value_representation import ValueRepresentation
from utils.scaler import scaler
from utils.reporting import reporter

import utils.constants as constants # pylint: disable=consider-using-from-import
from utils.type_aliases import TypeValidState, TypeAction, TypeNDarray64, is_scalar_64_tg


class VrApproximateLinear(ValueRepresentation):

    def __init__(self) -> None:

        super().__init__()

        self.scale_value = scaler.get_scaler()

        self._report_at = reporter.get_reporting_handle()

        self._weights: TypeNDarray64


    def get_value(self, state: TypeValidState, action: TypeAction) -> float:


        features = self.get_features(state, action)
        state_action_value_q = np.dot(self._weights, features)

        if is_scalar_64_tg (state_action_value_q): # -> TypeGuard[np.float64]
            return state_action_value_q.item()
        else:
            raise SystemExit("VrApproximateLinear: get_value: Assumed a scalar from dot product")


    def get_gradient(self, state: TypeValidState, action: TypeAction) -> list[float]:
        return self.get_features(state, action)


    @abstractmethod
    def get_features(self, state: TypeValidState, action: TypeAction) -> list[float]:
        pass


    def get_feature_dimension(self) -> int:
        return len(self._weights)


    def update_weights_by(self, update: TypeNDarray64) -> None:
        max_update = np.linalg.norm(update, ord=np.inf)

        if max_update > constants.MAX_UPDATE_WARN_LIMIT:
            print(f"Warning: a large update to weights, max change is: {max_update}")

        self._weights = np.add(self._weights, update)


    def set_weights_to(self, new_weights: TypeNDarray64) -> float:
        dist = np.linalg.norm(np.subtract(self._weights, new_weights), ord=np.inf)

        if dist > constants.MAX_UPDATE_WARN_LIMIT:
            print(f"Warning: a large update to weights, max change is: {dist}")

        self._weights = new_weights

        return dist.item()


    def report(self, name: str, iteration: int) -> None:
        self._report_at(name, iteration, self.get_value)
