import itertools
from math import inf

from environments.environment import Environment
from environments.blackjack import Blackjack, Player

from utils.scaler import scaler
import utils.constants as constants
from utils.type_aliases import (TypeState, TypeValidState, TypeAction,
                                TypeBlackjackAction, TypeBlackjackActions,
                                is_valid_blackjack_action_tg)


class EnvironmentBlackjack(Environment):

    def __init__(self, *, variant: str="simple"):

        super().__init__()

        self._variant: str = variant

        self._game: Blackjack = Blackjack()
        self._player: Player = Player()
        self._game.register_player(self._player)

        self._actions: TypeBlackjackActions = self._game.get_actions()
        self._HIT: TypeBlackjackAction = self._actions[0]   # pylint: disable=invalid-name
        self._STAND: TypeBlackjackAction = self._actions[1] # pylint: disable=invalid-name

        scaler.register_scale("player", 4, 21)
        scaler.register_scale("dealer", 2, 11)


    def get_actions(self) -> TypeBlackjackActions:
        return self._actions


    def initialize(self) -> None:

        blackjack = True

        while blackjack:
            self._game.deal_new_game()
            blackjack = self._player.is_blackjack()


    def get_state(self) -> TypeValidState:

        if self._player.is_bust() or self._game.dealer.is_bust():
            raise SystemExit("Environment blackjack: get_state() called when game has already terminated")

        if self._variant == "simple":

            # simple state: (current_score_dealer: int, current_score_player: int, player_has_ace: bool)

            dealer_score, _ = self._game.dealer.get_current_max_score()
            player_score, soft_ace = self._player.get_current_max_score()

            return (dealer_score, player_score , soft_ace)

        else:
            raise SystemExit(f"EnvironmentBlackjack: unknown environment type "
                             f"for get_state; {self._variant}")


    def do_action(self, action: TypeAction) -> tuple[float, TypeState]:

        assert is_valid_blackjack_action_tg(action) # -> TypeGuard[TypeBlackjackAction]

        terminate = False

        reward = -inf

        if action == self._HIT:
            self._player.hit()
            if self._player.is_bust():
                terminate = True
                reward = -1
            else:
                reward = 0

        elif action == self._STAND:
            terminate = True
            player_final_score = self._player.stand()
            dealer_final_score = self._game.dealers_turn()

            if dealer_final_score == -1:
                reward = 1
            elif dealer_final_score == player_final_score:
                reward = 0
            elif dealer_final_score < player_final_score:
                reward = 1
            elif dealer_final_score > player_final_score:
                reward = -1

        new_state: TypeState

        if terminate:
            new_state = constants.TERMINAL_STATE
        else:
            new_state = self.get_state()

        return reward, new_state


    def get_report_base_states(self) -> list[TypeValidState]:
        report_section_1 = {
            'dealer': range(2,12),
            'player': range(4,22),
            'soft': (False, )
        }
        report_section_2 = {
            'dealer': range(2,12),
            'player': range(12,22),
            'soft': (True, )
        }
        soft_false = list(itertools.product(report_section_1['dealer'], report_section_1['player'],
                                            report_section_1['soft']))
        soft_true = list(itertools.product(report_section_2['dealer'], report_section_2['player'],
                                           report_section_2['soft']))

        return soft_false + soft_true # type: ignore[return-value] # its ok (int, int, bool)


    def get_column_names(self) -> list[str]:
        return ['dealer', 'player', 'soft']
