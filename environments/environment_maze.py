import itertools
import random

from environments.environment import Environment
from environments.maze import Maze, Movement

from environments.maze_configs import configurations

from utils.scaler import scaler
from utils.type_aliases import (TypeMazeStructure, TypeMazeState, TypeValidState,
                                TypeValidMazeState, TypeAction, TypeMazeActions,
                                is_valid_maze_state_tg, is_valid_maze_action_tg)
from utils.constants import TERMINAL_STATE


class EnvironmentMaze(Environment):  # pylint: disable=too-many-instance-attributes

    def __init__(self, *, variant: str="simple"):

        super().__init__()

        self.maze_structure: TypeMazeStructure =  configurations[variant]['maze_structure']

        self._noise = configurations[variant]['noise']
        self._living_cost = configurations[variant]['living_cost']

        self._is_terminated = True
        self._initial_state: TypeValidMazeState = (2,0)
        self._current_state: TypeValidMazeState = self._initial_state

        self._maze = Maze(self.maze_structure)
        self._movement = Movement(self._maze, noise=self._noise)

        self._actions: TypeMazeActions = Movement.actions

        scaler.register_scale("row", 0, self._maze.size[0] - 1)
        scaler.register_scale("col", 0, self._maze.size[1] - 1)


    def get_actions(self) -> TypeMazeActions:
        return self._actions


    def initialize(self) -> None:

        initial_state = (random.randrange(self._maze.size[0]),
                         random.randrange(self._maze.size[1]))

        while (self._maze.walls[initial_state] or
               self._maze.terminal[initial_state]):

            initial_state = (random.randrange(self._maze.size[0]),
                             random.randrange(self._maze.size[1]))

        self._initial_state = initial_state
        self._is_terminated = False
        self._current_state = self._initial_state


    def get_state(self) -> TypeValidMazeState:
        if self._is_terminated:
            raise SystemExit("Environment get_state(): called when episode has already terminated")

        return self._current_state


    def do_action(self, action: TypeAction) -> tuple[float, TypeMazeState]:

        assert is_valid_maze_action_tg(action) # -> TypeGuard[TypeMazeAction]

        if self._is_terminated:
            raise SystemExit("Environment maze: get_state() called when game has already terminated")

        next_state: TypeMazeState = self._movement.do_noisy_move(self._current_state, action)
        reward = self._maze.rewards[next_state] + self._living_cost

        if self._maze.terminal[next_state]:
            next_state = TERMINAL_STATE
            self._is_terminated = True
        else:
            assert is_valid_maze_state_tg(next_state) # -> TypeGuard[TypeValidMazeState]
            self._current_state = next_state

        return reward, next_state


    def get_report_base_states(self) -> list[TypeValidState]:
        report_section = {
                'row': range(0, self._maze.size[0]),
                'col': range(0, self._maze.size[1]),
            }

        report_states = list(itertools.product(report_section['row'], report_section['col']))

        return report_states # type: ignore[return-value] # its ok (int, int)


    def get_column_names(self) -> list[str]:
        return ['row', 'col']
