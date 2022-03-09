from environments.environment import Environment
from policies.policy import Policy

from utils.type_aliases import TypeState, TypeValidState, TypeAction, TypeSars


class Agent():

    def __init__(self, *, name: str, environment: Environment, policy: Policy):

        self.name: str = name
        self.environment: Environment = environment
        self.policy: Policy = policy


    def initialize(self) -> None:
        self.environment.initialize()


    def get_state(self) -> TypeValidState:
        return self.environment.get_state()


    def select_action_by_behavior_policy(self, state: TypeValidState, iteration: int) ->  TypeAction:
        return self.policy.select_action_by_behavior_policy(state, iteration)


    def select_action_by_target_policy(self, current_state: TypeValidState) -> TypeAction:
        return self.policy.select_action_by_target_policy(current_state)


    def get_action_probability(self, state: TypeValidState, action: TypeAction, iteration: int) -> float:
        return self.policy.get_action_probability(state, action, iteration)


    def do_action(self,action: TypeAction) -> tuple[float, TypeState]:
        return self.environment.do_action(action)


    def do_sars(self, current_state: TypeValidState, iteration: int) -> TypeSars:
        action = self.policy.select_action_by_behavior_policy(current_state, iteration)
        reward, next_state = self.environment.do_action(action)

        return {
            'state': current_state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
        }


    def get_state_value(self, state: TypeValidState, iteration: int) -> float:
        return self.policy.get_state_value(state, iteration)
