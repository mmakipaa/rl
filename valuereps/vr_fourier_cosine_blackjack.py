import itertools
import numpy as np

from valuereps.vr_approximate import VrApproximateLinear
import utils.constants as constants # pylint: disable=consider-using-from-import

from utils.type_aliases import (TypeValidState, TypeValidBlackjackState, TypeBlackjackAction,
                                TypeAction, TypeNDarray64,
                                is_valid_blackjack_state_tg, is_valid_blackjack_action_tg)


class VrFourierCosineBlackjack(VrApproximateLinear):

    def __init__(self) -> None:

        super().__init__()

        self._zones: list[tuple[bool, bool]] = list((soft, action)
                                                     for soft in (True, False)
                                                     for action in (True, False))
        self._order = 6
        self._initial_weight: float = constants.INITIAL_WEIGHT

        self._c = list(itertools.product(range(0, self._order+1), # pylint: disable=invalid-name
                                        range(0, self._order+1)))

        # self._c = [[0,0], [0,1], [0,2], [0,3], [1,0], [1,1], .... [4,3],[4,4]]

        self._weights: TypeNDarray64 = np.asarray([ self._initial_weight ] * len(self._c) * len(self._zones))


    def _get_zone_index(self, state: TypeValidBlackjackState, action: TypeBlackjackAction) -> int:
        return self._zones.index((state[2], action))


    def get_features(self, state: TypeValidState, action: TypeAction) -> list[float]:

        assert is_valid_blackjack_state_tg(state) # -> TypeGuard[TypeValidBlackjackState]
        assert is_valid_blackjack_action_tg(action) # -> TypeGuard[TypeBlackjackAction]

        dealer_s = self.scale_value(state[0],"dealer")
        player_s = self.scale_value(state[1],"player")

        zone = self._get_zone_index(state, action)
        len_c = len(self._c)
        start = zone * len_c

        features: list[float] = [ 0.0 ] * len_c * len(self._zones)

        for i, C in enumerate(self._c): # pylint: disable=invalid-name # it is a good name!
            features[i+start] = np.cos(C[0] * np.pi * dealer_s + C[1] * np.pi * player_s)

        return features
