import numpy as np

from valuereps.vr_approximate import VrApproximateLinear
from valuereps.polynomial_terms import StateAction, Feature, AVAILABLE_FEATURES

from utils.scaler import scaler

from utils.type_aliases import TypeValidState, TypeAction, TypeNDarray64


class VrPolynomial(VrApproximateLinear):

    def __init__(self) -> None:

        super().__init__()

        # based on a forward-backward model search against reference case (10^8 episodes of MC off)
        # best 4th degree polynomial model, AICc= -1038.7

        self._used_term_indexes: list[int] = [0, 2, 4, 6, 7, 8, 9, 10, 13, 14, 16, 18, 19, 20, 28, 32,
                                              37, 39, 40, 41, 52, 64, 81, 84, 88, 89, 95]

        # best 3th degree model, AICc= -1006.2
        # self._used_term_indexes: list[int] = [0, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 20,
        #                                       24, 29, 31, 32, 35, 41, 43, 44, 48, 53, 60]

        # based on sqrt_lasso against reference case
        # self._used_term_indexes: list[int] = [1, 3, 5, 6, 8, 13, 16, 17, 18, 19, 21, 22, 24, 36, 37, 38, 40,
        #                                      45, 60, 65, 85, 86, 87, 88, 89, 90, 91]

        self._terms: list[Feature] = [AVAILABLE_FEATURES[f] for f in self._used_term_indexes]

        for feature in self._terms:
            scaler.register_scale(feature['name'], feature['min_value'], feature['max_value'])

        #self._initial_weight = constants.INITIAL_WEIGHT
        #self._weights: TypeNDarray64 = np.asarray([ self._initial_weight ] * len(self._used_term_indexes))
        weight_range = 0.2
        self._weights: TypeNDarray64 = np.random.uniform(-weight_range, weight_range,
                                                         len(self._used_term_indexes)).tolist()


    def get_features(self, state: TypeValidState, action: TypeAction)  -> list[float]:

        dealer = state[0]
        player = state[1]

        soft_ind = 1 if state[2] else 0
        action_ind = 1 if action else 0

        state_action = StateAction(dealer, player, soft_ind, action_ind)

        scaled_values: list[float] = []

        for term in self._terms:
            feature_name = term['name']
            value_func = term['func']

            feature_value = value_func(state_action)
            scaled_value = scaler.scale_value( feature_value ,feature_name)

            scaled_values.append(scaled_value)

        return scaled_values
