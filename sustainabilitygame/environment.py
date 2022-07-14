import numpy as np


class Environment(object):
    def __init__(self, label="", initial_state=1):
        self.label = label
        self.initial_state = initial_state
        self.state = initial_state
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

        if action == "sustainable":
            self.state *= 1
            self.state_history.append(self.state)

            return self.state * 0.05

        elif action == "unsustainable":
            self.state *= 0.99
            self.state_history.append(self.state)

            return self.state * 0.1

        elif action == "repair":
            self.state = np.min([self.state * 1.01, 1])

            return 0
