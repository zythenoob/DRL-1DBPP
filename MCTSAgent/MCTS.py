from MCTSAgent.Tree import Node
from MCTSAgent.Policy import *
from MCTSAgent.utils import *
from MCTSAgent.State import max_bin_opening
import igraph as ig
import time


class MCTS:
    def __init__(self, agent, exploration, rollout_policy=random_policy):
        self.agent = agent
        self.exploration = exploration
        self.rollout = rollout_policy
        self.node_counter = 0
        self.root = Node(self)
        self.search_time = 0
        self.select_time = 0
        self.simulate_time = 0
        self.backpropagate_time = 0

    def set_root(self, state):
        self.root.set_state(state)

    def execute(self):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        start_t = time.time()
        node = self.select(self.root, self.exploration)
        select_t = time.time()
        reward = self.simulate(node)
        simulate_t = time.time()
        self.backpropagate(node, reward)
        bp_t = time.time()
        self.search_time += bp_t - start_t
        self.select_time += select_t - start_t
        self.simulate_time += simulate_t - select_t
        self.backpropagate_time += bp_t - simulate_t

    def search(self):
        node, probs = self.get_best_child(self.root, 0)
        for action, n in self.root.children.items():
            if n is node:
                return action, probs

    def select(self, node, exploration):
        while not node.terminal:
            if node.expanded:
                node = self.get_best_child(node, exploration)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        # expand agent move node
        actions = node.state.actions()
        for action in actions:
            if action not in node.children:
                # expand action
                new_node = Node(self, node.state.child_state(action), node)
                node.children[action] = new_node
                if len(node.children) == len(actions):
                    node.expanded = True

                # if new_node.state.p == 0:
                #     return self.expand(new_node)

                return new_node

        print('')
        print(node.state)
        print(node)
        print(node.parent.state)
        raise Exception("Should never reach here")

    def simulate(self, node):
        return self.rollout(self.agent, node)

    def backpropagate(self, node, reward):
        while node is not None:
            node.num_visit += 1
            node.reward += reward
            node = node.parent

    def get_best_child(self, node, exploration):
        # player's node
        if node.state.p == 1:
            # if contains one child only, go to the child
            if len(node.children) == 1:
                return node.children[node.state.actions()[0]]
            # use the policy network to evaluate the node, get prior probs
            child_values = self.agent.evaluate_node(node, exploration)
            best_nodes = []
            for action, value in child_values.items():
                if value == np.max(list(child_values.values())):
                    best_nodes.append(node.children[action])

            # softmax on prob, get the actual action-values after mcts
            if exploration == 0:
                values = list(child_values.values())
                softmax_output = np.exp(values) / np.sum(np.exp(values))
                probs = np.zeros(max_bin_opening)
                for i, item in enumerate(child_values.items()):
                    probs[item[0]] = softmax_output[i]

                return random.choice(best_nodes), probs
            else:
                if len(best_nodes) == 0:
                    print('')
                    print(node)
                    print(node.state)
                    print(node.state.p)
                return random.choice(best_nodes)
        # sampler's node
        else:
            best_nodes = list(node.children.values())

            if exploration == 0:
                return random.choice(best_nodes), []
            else:
                return random.choice(best_nodes)

    def plot(self):
        queue = [self.root]
        edges = []
        vertexes = [self.root]
        while len(queue) > 0:
            current_node = queue[0]
            for child in current_node.children.values():
                queue.append(child)

            if current_node.parent is not None:
                edges.append((current_node.parent.id, current_node.id))
                vertexes.append(current_node)

            queue.pop(0)

        tree = ig.Graph(edges=edges)
        ig.plot(tree, layout=tree.layout('rt', root=[0]), bbox=(4000, 1500))
