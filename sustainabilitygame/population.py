import numpy as np


class Population(object):
    def __init__(self, agents=[], label="", environment=None):
        self.t = 0
        self.agents = agents
        self.label = label
        self.n = len(agents)
        self.environment = environment
        self.action_history = []
        self.total_points_history = [self.total_points]

    def __repr__(self):
        if self.n != 1:
            plural = "s"
        else:
            plural = ""
        return f"Population {self.label}\n  {self.n} agent{plural}\n  Total points: {self.total_points}\n  t = {self.t}\n  Environment: {self.environment.label}"

    @property
    def total_points(self):
        return np.sum([agent.points for agent in self.agents])

    def progress(self):
        self.t += 1
        for agent in self.agents:
            action = agent.select_action(self.environment)
            agent.act(action, self.environment)
            self.action_history.append({"agent": agent.id, "action": action})
        self.total_points_history.append(self.total_points)
