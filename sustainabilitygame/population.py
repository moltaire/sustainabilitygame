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
        self.gini_coefficient_history = [self.gini_coefficient]

    def __repr__(self):
        if self.n != 1:
            plural = "s"
        else:
            plural = ""
        return f"Population {self.label}\n  {self.n} agent{plural}\n  Total points: {self.total_points}\n  t = {self.t}\n  Environment: {self.environment.label}"

    @property
    def total_points(self):
        return np.sum([agent.points for agent in self.agents])

    @property
    def gini_coefficient(self):
        """Returns the Gini coefficient $G$ across the population of agents.

        Source:
        https://en.wikipedia.org/wiki/Gini_coefficient#Calculation

        Returns:
            float: Gini coefficient $G$ (0 = no inequality; 1 = maximal inequality)
        """
        y_i = np.sort([agent.points for agent in self.agents])
        i = np.arange(1, self.n + 1)
        G = (2 * np.sum(i * y_i)) / (self.n * np.sum(y_i)) - (self.n + 1) / self.n
        return G

    def progress(self):
        self.t += 1
        # Create a random order in which agents interact with the environment. Otherwise the first agent has an advantage.
        order = np.arange(self.n).astype(int)
        np.random.shuffle(order)
        for agent in np.array(self.agents)[order]:
            action = agent.select_action(self.environment)
            agent.act(action, self.environment)
            self.action_history.append({"agent": agent.id, "action": action})
        self.total_points_history.append(self.total_points)
        self.gini_coefficient_history.append(self.gini_coefficient)
