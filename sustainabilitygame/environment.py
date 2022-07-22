import numpy as np


class Environment(object):
    def __init__(
        self,
        label="",
        initial_state=1,
        reward_factors={"sustainable": 0.01, "exploit": 0.05, "restore": 0},
        impact_factors={"sustainable": 1.0, "exploit": 0.999, "restore": 1.01},
    ):
        self.label = label
        self.initial_state = initial_state
        self.state = initial_state
        self.reward_factors = reward_factors
        self.impact_factors = impact_factors
        self.state_history = [initial_state]
        self.action_history = []

    def __repr__(self):
        return f"Environment {self.label}\n  State: {self.state:.2f}"

    def react(self, action):
        """Lets the environment react to an agent's action.

        Args:
            action (str): An agent's action.
        """
        self.action_history.append(action)

        self.state *= self.impact_factors[action]
        self.state = np.clip(self.state, 0, 1)
        self.state_history.append(self.state)

        return self.state * self.reward_factors[action]
