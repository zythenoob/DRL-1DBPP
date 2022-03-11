

class Node:
    def __init__(self, mcts, state=None, parent=None):
        self.state = state
        self.id = mcts.node_counter
        mcts.node_counter += 1

        if state is None:
            self.terminal = False
        else:
            self.terminal = state.terminal()
        self.expanded = self.terminal
        self.parent = parent
        self.num_visit = 0
        self.v = 0
        self.child_p = 0
        self.reward = 0
        self.evaluated = False
        self.children = {}

    def set_state(self, state):
        self.state = state
        self.terminal = state.terminal()

    def q_value(self):
        return self.reward / self.num_visit

    def __str__(self):
        s = ["id: %s" % self.id, "total reward: %s" % self.reward, "num visits: %d" % self.num_visit,
             "p: %d" % self.state.p, "is terminal: %s" % self.terminal, "children: %s" % self.children]
        return "%s: {%s}" % (self.__class__.__name__, ', '.join(s))
