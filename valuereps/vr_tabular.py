import math

from typing import Union, Any

from valuereps.value_representation import ValueRepresentation
from valuereps.tabular_storage import TabularStateAction

from utils.reporting import reporter
import utils.constants as constants  # pylint: disable=consider-using-from-import

from utils.type_aliases import TypeState, TypeValidState, TypeAction


class VrTabular(ValueRepresentation):

    def __init__(self) -> None:

        super().__init__()

        self.initial_q: float = constants.DEFAULT_VALUE

        self._initial_values: dict[str, Union[int, float]]  = {
            'value': self.initial_q,
            'visit_count': 0,
            'cumulative_count': 0
        }
        self.action_value_table: TabularStateAction = TabularStateAction()

        self._report_at = reporter.get_reporting_handle()


    def get_value(self, state: TypeValidState, action: TypeAction) -> float:

        node = self.action_value_table.get_node(state, action)

        if node:
            state_action_value_q = node['value']
        else:
            state_action_value_q = self._initial_values['value']

        return state_action_value_q


    def get_state_visit_count(self, state: TypeValidState) -> int:

        count: int = 0

        if not self.actions:
            raise SystemExit("VR Tabular: Actions not defined")

        for this_action in self.actions:

            node = self.action_value_table.get_node(state, this_action)
            if node:
                assert isinstance(node['visit_count'], int)
                count += node['visit_count']

        return count


    def get_max_value(self, state: TypeValidState) -> float:

        best_value = -math.inf

        if not self.actions:
            raise SystemExit("VR Tabular: Actions not defined")

        for this_action in self.actions:

            state_action_value_q = self.get_value(state, this_action)

            if state_action_value_q > best_value:
                best_value = state_action_value_q

        return best_value


    def is_best_action(self, state: TypeValidState, action: TypeAction) -> bool:

        value = self.get_value(state, action)

        if not self.actions:
            raise SystemExit("VR Tabular: Actions not defined")

        for this_action in self.actions:

            state_action_value_q = self.get_value(state, this_action)

            if state_action_value_q > value:
                return False

        return True


    def get_parameters(self, state: TypeValidState, action: TypeAction,
                       *return_values: str) -> tuple[Any,...]:

        node = self.action_value_table.get_node(state, action)
        values = []

        if not node:
            node = self._initial_values

        for return_key in return_values:
            if return_key in node:
                values.append(node[return_key])
            else:
                raise SystemExit(f"No requested key {return_key} stored "
                                 f"in tabular storage for {node}")

        return tuple(values)


    def update_parameters(self, state: TypeState, action: TypeAction,
                          **update_values: Union[int, float] ) -> None:

        node = self.action_value_table.get_node(state, action)

        if not node:
            node = {}

        for key, _ in update_values.items():
            node[key] = update_values[key]

        self.action_value_table.add_node(state, action, node)


    def report(self, name: str, iteration: int) -> None:

        def get_state_action_visit_count(state: TypeValidState, action: TypeAction) -> int:

            visit_count = self.get_parameters(state, action, 'visit_count')[0]
            assert isinstance(visit_count, int)
            return visit_count

        self._report_at(name, iteration, self.get_value, get_state_action_visit_count)
