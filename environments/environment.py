from abc import ABC, abstractmethod

from utils.type_aliases import TypeState, TypeValidState, TypeActions, TypeAction

class Environment(ABC):

    @abstractmethod
    def __init__(self, variant: str|None = None) -> None:
        pass

    @abstractmethod
    def initialize(self) -> None:
        pass


    @abstractmethod
    def get_state(self) -> TypeValidState:
        pass


    @abstractmethod
    def do_action(self, action: TypeAction) -> tuple[float, TypeState]:
        pass


    @abstractmethod
    def get_actions(self) -> TypeActions:
        pass


    @abstractmethod
    def get_column_names(self) -> list[str]:
        pass

    @abstractmethod

    def get_report_base_states(self) -> list[TypeValidState]:
        pass
