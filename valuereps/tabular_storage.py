from utils.type_aliases import (TypeState, TypeValidState, TypeAction, TypeStateAction,
                                TypeStorageDict, is_valid_state_tg)

class TabularStateAction:

    def __init__(self) -> None:
        self._action_value_dict: dict[TypeStateAction, TypeStorageDict] = {}


    def _get_index(self, state: TypeValidState, action: TypeAction) -> TypeStateAction:
        return state + (action,)


    def get_node(self, state: TypeState, action: TypeAction) -> TypeStorageDict|None:

        if is_valid_state_tg(state): # -> TypeGuard[TypeValidState]
            storage_key = self._get_index(state, action)

            if storage_key in self._action_value_dict:
                return self._action_value_dict[storage_key]

        return None


    def add_node(self, state: TypeState, action: TypeAction,
                 new_node: TypeStorageDict) -> TypeStorageDict|None:

        if is_valid_state_tg(state): # -> TypeGuard[TypeValidState]
            storage_key = self._get_index(state,action)
            self._action_value_dict[storage_key] = new_node

            return self._action_value_dict[storage_key]

        return None
