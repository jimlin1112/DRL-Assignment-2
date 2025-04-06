# Remember to adjust your student ID in meta.xml
import copy
import random
import math
import numpy as np
from collections import defaultdict
import pickle
import gym
from gym import spaces
import matplotlib.pyplot as plt
from ntuple import NTupleApproximator

from Game2048Env import Game2048Env

    

    
class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        # List of untried actions based on the current state's legal moves
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        best_value = -float('inf')
        best_child = None
        for action, child in node.children.items():
          if child.visits == 0:
              uct_value = float('inf')
          else:
              uct_value = (child.total_reward / child.visits) + self.c * math.sqrt(math.log(node.visits) / child.visits)
          if uct_value > best_value:
              best_value = uct_value
              best_child = child
        return best_child


    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        current_depth = 0
        total_reward = 0
        done = False
        discount = 1.0

        while current_depth < self.rollout_depth and not done:
            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves:
                break

            action = random.choice(legal_moves)
            _, reward, done, _ = sim_env.step(action)
            total_reward += discount * reward
            discount *= self.gamma
            current_depth += 1

        # Use TD approximator to evaluate final state
        final_value = self.approximator.value(sim_env.board)
        discounted_final = (self.gamma ** current_depth) * final_value
        return total_reward + discounted_final


    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            reward *= self.gamma
            current = current.parent


    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # TODO: Selection: Traverse the tree until reaching an unexpanded node.
        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            sim_env.step(node.action)


        # TODO: Expansion: If the node is not terminal, expand an untried action.
        legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
        if legal_actions and node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            new_state, reward, done, _ = sim_env.step(action)
            new_score = sim_env.score
            child_node = TD_MCTS_Node(new_state, new_score, parent=node, action=action)
            node.children[action] = child_node
            node = child_node


        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution
    
with open("ntuple_approximator.pkl", "rb") as f:
    approximator = pickle.load(f)

env = Game2048Env()
td_mcts = TD_MCTS(env, approximator, iterations=50, exploration_constant=1.41, rollout_depth=10, gamma=0.99)

def get_action(state, score):
    root = TD_MCTS_Node(state, score)
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)
    best_act, _ = td_mcts.best_action_distribution(root)
    return best_act

    env.board = state
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    afterstates = []
    afterstate_values = []
    afterstate_values_mean = []
    for a in legal_moves:
        afterstates = []
        afterstate_values = []
        for _ in range(10):
            env_copy = copy.deepcopy(env)
            next_state, next_score, next_done, _ = env_copy.step(a)
            afterstate_values.append(approximator.value(next_state))
        afterstate_values_mean.append(np.mean(afterstate_values))
    idx = np.argmax(afterstate_values_mean)
    action = legal_moves[idx]
    return action

    # return random.choice([0, 1, 2, 3]) # Choose a random action
    
    # You can submit this random agent to evaluate the performance of a purely random strategy.


