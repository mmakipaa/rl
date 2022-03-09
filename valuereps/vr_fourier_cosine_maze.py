import itertools
import numpy as np

from valuereps.vr_approximate import VrApproximateLinear
import utils.constants as constants

from utils.type_aliases import (TypeValidState, TypeAction, TypeNDarray64,
                                is_valid_maze_state_tg, is_valid_maze_action_tg)


class VrFourierCosineMaze(VrApproximateLinear):

    def __init__(self) -> None:

        super().__init__()

        self._zones: list[int] = [0, 1, 2, 3]
        self._order = 4
        self._initial_weight: float = constants.INITIAL_WEIGHT

        self._c = list(itertools.product(range(0, self._order+1), # pylint: disable=invalid-name
                                        range(0, self._order+1)))

        # for order = 4
        # self._c = [[0,0], [0,1], [0,2], [0,3], [1,0], [1,1], .... [4,3],[4,4]]

        self._weights: TypeNDarray64 = np.asarray([ self._initial_weight ] * len(self._c) * len(self._zones))


    def _get_zone_index(self, action: TypeAction) -> int:
        return self._zones.index(action)


    def get_features(self, state: TypeValidState, action: TypeAction) -> list[float]:

        assert is_valid_maze_state_tg(state) # -> TypeGuard[TypeValidMazeState]
        assert is_valid_maze_action_tg(action) # -> TypeGuard[TypeMazeAction]

        row = self.scale_value(state[0],"row")
        col = self.scale_value(state[1],"col")

        zone = self._get_zone_index(action)
        len_c = len(self._c)
        start = zone * len_c

        features: list[float] = [ 0.0 ] * len_c * len(self._zones)

        for i, C in enumerate(self._c): # pylint: disable=invalid-name
            features[i+start] = np.cos(C[0] * np.pi * row + C[1] * np.pi * col)

        return features
