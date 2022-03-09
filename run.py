import argparse
from pathlib import Path
from typing import Any

import pickle
import yaml # type: ignore[import]

import numpy as np
import pandas as pd # type: ignore[import]

from methods.method import Method

from environments.maze_configs import configurations

from utils.constants import Defaults

from utils.factories import create_environment, create_method
from utils.reporting import reporter
from utils.type_aliases import TypeAgentConfig


def _read_config_file(filename: str) -> list[TypeAgentConfig]:

    with open(filename, "r", encoding="utf-8") as stream:

        agent_list: list[TypeAgentConfig] = []
        try:
            agent_list = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

        return agent_list


def run(args: argparse.Namespace) -> None:  # pylint: disable=too-many-locals

    training_rewards = []
    evaluation_rewards = []
    episode_lengths = []

    environment = create_environment(args.environment, Defaults.ENV_VARIANT)

    agent_list = _read_config_file(args.configfile)

    tr_episode_len: dict[str, int|float] = {}
    ev_episode_len: dict[str, int|float] = {}

    for agent_def in agent_list:

        alpha_type = agent_def.get('alpha_type')

        if alpha_type is not None and "TARGET_AT" in alpha_type:
            agent_def["alpha_target_iterations"] = np.floor(
                args.iterations * Defaults.TARGET_AT_PERCENTAGE)

        epsilon_type = agent_def.get('epsilon_type')

        if epsilon_type is not None and "TARGET_AT" in epsilon_type:
            agent_def["epsilon_target_iterations"] = np.floor(
                args.iterations * Defaults.TARGET_AT_PERCENTAGE)

        method: Method = create_method(agent_def, environment, args.environment)

        method_type = agent_def.get("method")
        if method_type is not None and "Batch" in method_type:

            max_iterations =  Defaults.BATCH_MAX_ITERATIONS
            if hasattr(method, 'set_batch_learning_parameters'):
                method.set_batch_learning_parameters(max_iterations=max_iterations, # type: ignore
                                                     stopping_limit=Defaults.BATCH_STOPPING_LIMIT)
            reporting_at = list(range(1,max_iterations+1))
        else:
            reporting_at = list(np.floor(np.logspace(np.log10(Defaults.FIRST_REPORT),
                                                     np.log10(args.iterations),
                                                     num=Defaults.NUMBER_OF_REPORTS)))

        tr_reward, tr_episode_len = method.learn(args.iterations, reporting_at)
        ev_reward, ev_episode_len = method.evaluate(Defaults.EVALUATION_EPISODES)

        training_rewards.extend(tr_reward)
        episode_lengths.append([method.method_name,
                                *list(tr_episode_len.values()),
                                *list(ev_episode_len.values())])
        evaluation_rewards.append([method.method_name, Defaults.EVALUATION_EPISODES, ev_reward])

    result_dict: dict[str, Any] = {}

    result_dict['agents'] = agent_list

    rep_instance = reporter.get_reporting_instance()
    df_reports = rep_instance.get_reports_as_df()
    result_dict['report'] = df_reports

    df_tr_rewards = pd.DataFrame(training_rewards, columns=['agent', 'iteration', 'reward'])
    result_dict['tr_rewards'] = df_tr_rewards

    df_ev_rewards = pd.DataFrame(evaluation_rewards, columns=['agent', 'episodes', 'reward'])
    result_dict['ev_rewards'] = df_ev_rewards


    df_episode_lengths = pd.DataFrame(episode_lengths, columns=[ 'agent',
                                                    *["tr_"+k for k in tr_episode_len.keys()],
                                                    *["ev_"+k for k in ev_episode_len.keys()] ])
    result_dict['episode_lenghts'] = df_episode_lengths

    if args.environment == 'maze':
        result_dict['maze_config'] = configurations[Defaults.ENV_VARIANT]


    file_path = args.report

    with open(file_path,'wb') as file:
        pickle.dump(result_dict, file)

    print(f"\nTestrun completed, saved results to {file_path}")


def main() -> None:

    parser = argparse.ArgumentParser(description='Run reinforcement learning method tests')

    report_folder_path = Path(Defaults.REPORT_FOLDER)
    configs_folder_path = Path(Defaults.CONFIGS_FOLDER)

    required_named = parser.add_argument_group('required named arguments')

    required_named.add_argument("-e", "--environment", type=str, required=True,
                                choices=['blackjack', 'maze'],
                        help="environment to use, "
                             "either 'blackjack' for blackjack or 'maze' for maze")


    required_named.add_argument("-i", "--iterations", type=int, required=True,
                        help="number of iterations to run for training, "
                             "for batch methods number of episodes to include in the batch")

    required_named.add_argument("-c", "--configfile", required=True,
                        help="config file name, a YAML file to define methods to run. "
                             "If no exension is given, .yaml is assumed. "
                             "If no path is given default directory path "
                             f"'{Defaults.CONFIGS_FOLDER}' is used")

    parser.add_argument("-r", "--report", type=str,
                        help="report file name, a pickle file to store results in."
                             "This option can be used to override the default "
                             "filename <configfile>_<iterations>.pik. "
                             "If no extension is given, .pik is assumed. "
                             f"If no path is given, default path '{Defaults.REPORT_FOLDER}' is used")


    args = parser.parse_args()

    print(vars(args))

    if args.configfile:
        configfile_path = Path(args.configfile).with_suffix(".yaml")

        if len(configfile_path.parts) > 1:
            args.configfile = str(configfile_path)
        else:
            args.configfile = Path.joinpath(configs_folder_path, configfile_path)

    if args.iterations:
        pass

    if args.report:
        reportfile_path = Path(args.report).with_suffix(".pik")

        if len(reportfile_path.parts) > 1:
            args.report = str(reportfile_path)
        else:
            args.report = Path.joinpath(report_folder_path, reportfile_path)

    else:
        file_name_only = Path(args.configfile).stem
        report = args.environment + "_" + file_name_only + "_" + str(args.iterations)

        args.report = Path.joinpath(report_folder_path, report).with_suffix(".pik")

    run(args)


if __name__ == "__main__":
    main()
