import numpy as np

from utils.type_aliases import (TypeMazeStructure,
                                TypeValidMazeState, TypeMazeAction,
                                TypeNDarray64, TypeNDarrayBool)


class Maze:  # pylint: disable=too-few-public-methods

    def __init__(self, maze_structure: TypeMazeStructure):

        self.size: tuple[int, int] = maze_structure['size']

        self.walls: TypeNDarrayBool = np.full(self.size, False, dtype=bool)
        self.walls[tuple(zip(*maze_structure['walls']))] = True

        self.rewards: TypeNDarray64 = np.full(self.size, 0, dtype=np.float64)
        self.rewards[self.walls] = np.nan

        for key in maze_structure['rewards']:
            self.rewards[key] = maze_structure['rewards'][key]

        self.terminal: TypeNDarrayBool = np.full(self.size, False, dtype=bool)
        self.terminal[tuple(zip(*maze_structure['terminal_states']))] = True

        self.grid_index: list[TypeValidMazeState] = []

        wall: bool
        for index, wall in np.ndenumerate(self.walls):
            if not wall:
                self.grid_index.append(index)

        self.state_count = len(self.grid_index)


    def is_valid_state(self, state: TypeValidMazeState) -> bool:

        return ( (0 <= state[0] <= self.size[0] - 1  and
                  0 <= state[1] <= self.size[1] - 1) and
                 (not self.walls[state]))


class Movement:

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    actions = ( NORTH, EAST, SOUTH, WEST )
    action_names = ( "NORTH", "EAST", "SOUTH", "WEST" )
    direction_arrows = ( '\u2191','\u2192','\u2193','\u2190', '\u25aa', '')
                       #  up, right, down, left, small square, empty


    def __init__(self, maze: Maze, *, noise: float =0.2):

        self.maze = maze
        self.noise = noise

        self.noisy_moves = [
            self._adjust_left,
            self._adjust_none,
            self._adjust_right
        ]

        # probs for [left, straight, right]
        self.noisy_move_probs: list[float] = [self.noise / 2, 1 - self.noise, self.noise / 2]


    @classmethod
    def _adjust_left(cls, direction: int) -> int:
        return (cls.actions.index(direction) + len(cls.actions) - 1) % len(cls.actions)


    @classmethod
    def _adjust_none(cls, direction: int) -> int:
        return direction


    @classmethod
    def _adjust_right(cls, direction: int) -> int:
        return (cls.actions.index(direction) + 1) % len(cls.actions)


    def get_direction_probs(self, action: TypeMazeAction) -> list[tuple[int, float]]:
        dirs = []
        for j, _ in enumerate(self.noisy_moves):
            p_move = self.noisy_move_probs[j]
            move_direction = self.noisy_moves[j](action)

            dirs.append((move_direction, p_move))

        return dirs


    def _get_move_target(self, from_state: TypeValidMazeState, action: TypeMazeAction) -> TypeValidMazeState:

        target_state: tuple[int, int]

        if action == Movement.NORTH:
            target_state = (from_state[0] - 1, from_state[1])
        elif action == Movement.EAST:
            target_state = (from_state[0], from_state[1] + 1)
        elif action == Movement.SOUTH:
            target_state = (from_state[0] + 1, from_state[1])
        elif action == Movement.WEST:
            target_state = (from_state[0], from_state[1] - 1)
        else:
            # should not be here, quiet linting error
            raise SystemExit("Did not find _get_move_target")

        return target_state


    def move_from(self, from_state: TypeValidMazeState, action: TypeMazeAction) -> TypeValidMazeState:

        target_state = self._get_move_target(from_state, action)

        # check if trying to move outside the maze or against a wall
        # and in that case stay in current position

        if self.maze.is_valid_state(target_state):
            return target_state
        else:
            return from_state


    def do_noisy_move(self, from_state: TypeValidMazeState, action: TypeMazeAction) -> TypeValidMazeState:

        cum_prob: float = 0
        random_p = np.random.random()

        for i, p_lim in enumerate(self.noisy_move_probs):
            cum_prob += p_lim
            if random_p <= cum_prob:
                adjusted_move = self.noisy_moves[i]
                move_dir = adjusted_move(action)
                next_state = self.move_from(from_state, move_dir)
                return next_state

        # not happening, quiet mypy
        raise SystemExit("Did not return from do_noisy_move")
