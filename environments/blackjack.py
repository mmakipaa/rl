import random
from typing import Callable

import numpy as np

from utils.type_aliases import TypeBlackjackActions, TypeBlackjackAction


class Player:

    def __init__(self, *, name: str|None = None,
                 decision_handle: Callable[[],TypeBlackjackAction]|None = None):

        self._name: str|None = name
        self._decision_handle: Callable[[],TypeBlackjackAction]|None = decision_handle

        self._hand = Blackjack.EMPTY_HAND.copy()
        self._cards: list[str] = []


    def __repr__(self) -> str:
        return f"Player {self._name } with hand: {self._hand}, cards: {self._cards}"


    def initialize(self) -> None:
        self._hand = Blackjack.EMPTY_HAND.copy()
        self._cards = []


    def hit(self) -> str:
        card = Blackjack.draw_card()

        self._hand[card] += 1
        self._cards.append(Blackjack.card_labels[card])

        return Blackjack.card_labels[card]


    def stand(self) -> int:
        player_final_score, _ = self.get_current_max_score()

        return player_final_score


    def do_turn(self) -> int:

        while True:
            if self.is_bust():
                return -1
            else:
                next_action = self.decide_action()

                if next_action == Blackjack.HIT:
                    self.hit()

                elif next_action == Blackjack.STAND:
                    final_max_score, _ = self.get_current_max_score()

                    return final_max_score


    def is_bust(self) -> bool:

        scores = Blackjack.get_current_score(self._hand)

        if not scores:
            return True
        else:
            return False


    def is_blackjack(self) -> bool:

        if sum(self._hand) == 2:
            scores = Blackjack.get_current_score(self._hand)

            assert scores is not None

            if max(scores) == 21:
                return True

        return False


    def get_current_max_score(self) -> tuple[int, bool]:

        scores = Blackjack.get_current_score(self._hand)

        assert scores is not None

        if len(scores) > 1:
            soft_ace = True
        else:
            soft_ace = False

        return ( max(scores), soft_ace )


    def get_hand(self) -> list[int]:
        return self._hand


    def get_cards(self) -> list[str]:
        return self._cards


    def set_decision_handle(self, decision_handle: Callable[[], TypeBlackjackAction]|None) -> None:
        self._decision_handle = decision_handle


    def decide_action(self) -> TypeBlackjackAction:

        if self._decision_handle is None:
            raise SystemExit(f"Blackjack; No decision handler defined for player {self._name}")

        return self._decision_handle()


def stand_at_policy(player: Player, limit: int) -> TypeBlackjackAction:

    score, _ = player.get_current_max_score()

    if score < limit:
        return Blackjack.HIT
    else:
        return Blackjack.STAND


class Blackjack:

    #    index  =   [   0,   1,   2,   3,   4,   5,   6,   7,   8,    9,  10,  11,  12  ]
    card_labels =   [ "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K" ]
    card_values =   [ None,  2,   3,   4,   5,   6,   7,   8,   9,   10,  10,  10,  10  ]
    EMPTY_HAND  =   [    0,  0,   0,   0,   0,   0,   0,   0,   0,    0,   0,   0,   0  ]


    BUST_LIMIT = 21
    HIT_LIMIT_DEALER = 17

    HIT = True
    STAND = False
    ACTIONS = (HIT, STAND)

    def __init__(self) -> None:
        self.dealer = Player(name="Dealer", decision_handle=self._dealer_policy)
        self.players: list[Player] = []


    def register_player(self, player: Player) -> None:
        self.players.append(player)


    def _dealer_policy(self) -> TypeBlackjackAction:
        return stand_at_policy(self.dealer, limit=Blackjack.HIT_LIMIT_DEALER)


    @classmethod
    def get_actions(cls) -> TypeBlackjackActions:
        return Blackjack.ACTIONS


    @classmethod
    def draw_card(cls) -> int:
        card = random.randint(0,12)
        return card


    @classmethod
    def get_current_score(cls, hand: list[int]) -> list[int]|None:

        # get result of dot product as type int
        without_aces = np.dot(Blackjack.card_values[1:], hand[1:]).item() # type: ignore[arg-type]

        min_aces = hand[0]
        if min_aces + without_aces > Blackjack.BUST_LIMIT:
            return None
        else:
            scores = [ min_aces + without_aces ]

            for i in range(1,hand[0] + 1):

                score_to_add = min_aces + i * 10 + without_aces
                if score_to_add > Blackjack.BUST_LIMIT:
                    break
                else:
                    scores += [ score_to_add ]
            return scores


    def deal_new_game(self) -> None:

        self.dealer.initialize()
        self.dealer.hit()

        for player in self.players:

            player.initialize()
            player.hit()
            player.hit()


    def dealers_turn(self) -> int:

        return self.dealer.do_turn()
