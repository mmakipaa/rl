from typing import Union, Callable

import pandas as pd # type: ignore[import]

from environments.environment import Environment

from utils.type_aliases import (TypeValidState, TypeActions,
                                TypeReportList, TypeValueGetter, TypeVisitCountGetter)

class Reporting:

    def __init__(self, environment: Environment) -> None:

        self._report_list: TypeReportList = []

        state_headers = environment.get_column_names()
        self._column_names = ['agent', 'iterations'] + state_headers + ['action', 'visit_count', 'value']

        self._report_states: list[TypeValidState] = environment.get_report_base_states()
        self._report_actions: TypeActions = environment.get_actions()


    def get_column_names(self) -> list[str]:
        return self._column_names


    def get_reports_list(self) -> TypeReportList:
        return self._report_list


    def get_reports_as_df(self) -> pd.DataFrame:

        reports = self._report_list
        column_names = self._column_names

        df_report = pd.DataFrame(reports, columns=column_names)

        return df_report


    def report_at(self, method_name: str, current_iteration: int,
                  value_getter: TypeValueGetter,
                  visit_count_getter: TypeVisitCountGetter|None = None) -> None:

        report  = []

        for state in self._report_states:
            for action in self._report_actions:

                value = value_getter(state, action)

                if visit_count_getter:
                    visit_count = visit_count_getter(state, action)
                else:
                    visit_count = None

                report_row: list[Union[str, int, float, None]]

                report_row = [method_name, current_iteration]
                report_row.extend([item for item in state])
                report_row.append(action)
                report_row.append(visit_count)
                report_row.append(value)

                report.append(report_row)

        self._report_list.extend(report)


class SharedReporting():

    def __init__(self) -> None:
        self._reporting_instance: Reporting|None = None


    def setup_reporting_instance(self, environment: Environment) -> None:

        self._reporting_instance = Reporting(environment)


    def get_reporting_instance(self) -> Reporting:

        if self._reporting_instance is None:
            raise SystemExit("SharedReporting: Trying to get reporting instance is none")

        return self._reporting_instance

    def get_reporting_handle(self) -> Callable[..., None]:

        if self._reporting_instance is None:
            raise SystemExit("SharedReporting: Trying to get reporting instance is none")

        return self._reporting_instance.report_at


reporter = SharedReporting()
