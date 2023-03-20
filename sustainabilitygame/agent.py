import numpy as np


class Agent(object):
    def __init__(
        self,
        id="",
        endowment=0,
        p_action={"sustainable": 0.8, "exploit": 0.05, "restore": 0.15},
    ):
        self.id = id
        self.endowment = endowment
        self.points = endowment
        self.p_action = p_action
        self.action_history = []
        self.points_history = [endowment]

    def __repr__(self):
        return f"Agent {self.id}\n  Points: {self.points}"

    @property
    def actions(self):
        return list(self.p_action.keys())

    def act(self, action, environment):
        """Let the agent take an action in their environment."""
        outcome = environment.react(action)
        self.action_history.append(action)
        self.points += outcome
        self.points_history.append(self.points)

    def select_action(self, environment=None):
        action = np.random.choice(
            self.actions, p=[self.p_action[action] for action in self.actions]
        )
        return action
