# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
import dill
from numba import jit, njit


COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}

@jit(nopython=True)
def compress_and_merge(row, score):
    size = 4
    # 過濾非零元素
    temp = np.zeros(size, dtype=np.int32)
    pos = 0
    for i in range(size):
        if row[i] != 0:
            temp[pos] = row[i]
            pos += 1

    # 合併相鄰相同元素
    result = np.zeros(size, dtype=np.int32)
    write_pos = 0
    i = 0
    while i < pos:
        if i + 1 < pos and temp[i] == temp[i + 1]:
            result[write_pos] = temp[i] * 2
            score += temp[i] * 2
            i += 2
        else:
            result[write_pos] = temp[i]
            i += 1
        write_pos += 1

    return result, score

@jit(nopython=True)
def move_board(board, direction, score):
    new_board = board.copy()
    moved = False
    if direction == 0:  # 上
        for j in range(4):
            col, new_score = compress_and_merge(new_board[:, j], score)
            if not np.array_equal(col, new_board[:, j]):
                moved = True
            new_board[:, j] = col
            score = new_score
    elif direction == 1:  # 下
        for j in range(4):
            col = new_board[::-1, j]
            col, new_score = compress_and_merge(col, score)
            if not np.array_equal(col, new_board[::-1, j]):
                moved = True
            new_board[::-1, j] = col
            score = new_score
    elif direction == 2:  # 左
        for i in range(4):
            row, new_score = compress_and_merge(new_board[i], score)
            if not np.array_equal(row, new_board[i]):
                moved = True
            new_board[i] = row
            score = new_score
    elif direction == 3:  # 右
        for i in range(4):
            row = new_board[i, ::-1]
            row, new_score = compress_and_merge(row, score)
            if not np.array_equal(row, new_board[i, ::-1]):
                moved = True
            new_board[i, ::-1] = row
            score = new_score
    return new_board, moved, score

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True

        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        # empty_cells = list(zip(*np.where(self.board == 0)))
        # if empty_cells:
        #     x, y = random.choice(empty_cells)
        #     self.board[x, y] = 2 if random.random() < 0.9 else 4
        empty_cells = np.where(self.board == 0)
        if len(empty_cells[0]) > 0:
            idx = random.randint(0, len(empty_cells[0]) - 1)
            x, y = empty_cells[0][idx], empty_cells[1][idx]
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def merge(self, row):
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        # if action == 0:
        #     moved = self.move_up()
        # elif action == 1:
        #     moved = self.move_down()
        # elif action == 2:
        #     moved = self.move_left()
        # elif action == 3:
        #     moved = self.move_right()
        # else:
        #     moved = False

        self.board, moved, self.score = move_board(self.board, action, self.score)

        self.last_move_valid = moved

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row


    def is_move_legal(self, action):
        temp_board = self.board.copy()

        # if action == 0:  # Move up
        #     for j in range(self.size):
        #         col = temp_board[:, j]
        #         new_col = self.simulate_row_move(col)
        #         temp_board[:, j] = new_col
        # elif action == 1:  # Move down
        #     for j in range(self.size):
        #         col = temp_board[:, j][::-1]
        #         new_col = self.simulate_row_move(col)
        #         temp_board[:, j] = new_col[::-1]
        # elif action == 2:  # Move left
        #     for i in range(self.size):
        #         row = temp_board[i]
        #         temp_board[i] = self.simulate_row_move(row)
        # elif action == 3:  # Move right
        #     for i in range(self.size):
        #         row = temp_board[i][::-1]
        #         new_row = self.simulate_row_move(row)
        #         temp_board[i] = new_row[::-1]
        # else:
        #     raise ValueError("Invalid action")
        # return not np.array_equal(self.board, temp_board)

        new_board, moved, _ = move_board(temp_board, action, self.score)
        return moved
    
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
    approximator = dill.load(f)

env = Game2048Env()
td_mcts = TD_MCTS(env, approximator, iterations=50, exploration_constant=1.41, rollout_depth=10, gamma=0.99)

def get_action(state, score):
    root = TD_MCTS_Node(state, env.score)
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)
    best_act, _ = td_mcts.best_action_distribution(root)
    return best_act

    # return random.choice([0, 1, 2, 3]) # Choose a random action
    
    # You can submit this random agent to evaluate the performance of a purely random strategy.


