import numpy as np


class Agent(object):
    def __init__(self, id="", endowment=0):
        self.id = id
        self.endowment = endowment
        self.points = endowment

    def __repr__(self):
        return f"Agent {self.id}\n  Points: {self.points}"

    def act(self, action, environment):
        """Let the agent take an action in their environment."""
        outcome = environment.react(action)
        self.points += outcome


class RandomChoiceAgent(Agent):
    def __init__(self, id="", endowment=0):
        super().__init__(id, endowment)

    def select_action(self, environment):
        action = np.random.choice(["sustainable", "unsustainable", "repair"])
        return action


class SustainableAgent(Agent):
    def __init__(self, id="", endowment=0):
        super().__init__(id, endowment)

    def select_action(self, environment):
        action = np.random.choice(["sustainable", "unsustainable", "repair"],
         p=[0.8, 0.1, 0.1])
        return action

class UnsustainableAgent(Agent):
    def __init__(self, id="", endowment=0):
        super().__init__(id, endowment)

    def select_action(self, environment):
        action = np.random.choice(["sustainable", "unsustainable", "repair"],
         p=[0.1, 0.9, 0.0])
        return action