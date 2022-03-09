import itertools
import numpy as np

from valuereps.vr_approximate import VrApproximateLinear

import utils.constants as constants  # pylint: disable=consider-using-from-import
from utils.type_aliases import TypeValidState, TypeAction, TypeNDarray64


class Tile:  # pylint: disable=too-few-public-methods

    def __init__(self, corner: tuple[float,...], sizes: TypeNDarray64) -> None:

        self.min_corner: TypeNDarray64 = np.array(corner)
        self.max_corner: TypeNDarray64 = np.array(corner) + sizes - 1


    def is_active(self, point: tuple[float,...]) -> bool:
        return all(np.less_equal(self.min_corner, point) & np.greater_equal(self.max_corner, point))


class VrTileCoding(VrApproximateLinear):

    def __init__(self) -> None:

        super().__init__()

        self._initial_weight: float = constants.INITIAL_WEIGHT

        self.tiles: list[Tile] = self._init_tiles()

        self._weights: TypeNDarray64 = np.asarray([ self._initial_weight ] * len(self.tiles))


        self._feature_cache: dict[tuple[float,...], list[float]] = {}


    def _init_tiles(self) -> list[Tile]:  # pylint: disable=too-many-locals, no-self-use

        ranges: TypeNDarray64 = np.array([[2,11], [4,21], [0, 1], [0, 1]])
        sizes: TypeNDarray64 = np.array([6, 6, 1, 1])
        steps: TypeNDarray64 = np.array([3, 3, 1, 1])
        shifts: TypeNDarray64 = np.array([-5, -5, 0, 0])

        dimensions = len(ranges)

        corners = []

        for i in range(0,dimensions):

            start = ranges[i][0] + shifts[i]
            end = ranges[i][1] + 1
            step = steps[i]

            corner_points = []

            for k in range(start, end, step):
                corner_points.append(k)

            corners.append(corner_points)

        tiles = []

        for corner in list(itertools.product(*corners)):
            new_tile = Tile(corner, sizes)
            tiles.append(new_tile)

        return tiles


    def get_features(self, state: TypeValidState, action: TypeAction)  -> list[float]:

        dealer = state[0]
        player = state[1]
        soft_ind = 1 if state[2] else 0
        action_ind = 1 if action else 0

        new_state = (dealer, player, soft_ind, action_ind)

        features: list[float]

        if new_state in self._feature_cache:
            features = self._feature_cache[new_state]
        else:
            features = [tile.is_active(new_state) * 1 for tile in self.tiles]
            self._feature_cache[new_state] = features

        return features
