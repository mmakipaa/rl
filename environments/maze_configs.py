from utils.type_aliases import TypeMazeConfigs

configurations: TypeMazeConfigs = {}

configurations['simple'] = {

    'maze_structure': {
        'size': (3, 4),
        'walls': [(1,1)],
        'terminal_states': [(0,3), (1,3)],
        'rewards': {
            (0,3): 1,
            (1,3): -1
        }
    },
    'living_cost': -0.04,
    'noise': 0.2
}

configurations['complex'] = {

    'maze_structure': {
        'size': (8, 7),
        'walls': [
            (1,1), (1,2), (1,4),
            (2,1), (2,4),
            (4,3), (4,5), (4,6),
            (5,2),
            (6,3), (6,4), (6,5),
            (7,1)
        ],
        'terminal_states': [
            (1,5),
            (2,2), (2,5),
            (4,1), (4,2),
            (5,1), (5,3)
        ],
        'rewards': {
            (1,5): -1,
            (2,2): -1,
            (2,5): -1,
            (4,1): -1,
            (4,2): -1,
            (5,3):  1,
            (5,1): -1,
        }
    },
    'living_cost': -0.01,
    'noise': 0.2
}
