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

def rot90(pattern, board_size):
    return [(j, board_size - 1 - i) for (i, j) in pattern]

def rot180(pattern, board_size):
    return [(board_size - 1 - i, board_size - 1 - j) for (i, j) in pattern]

def rot270(pattern, board_size):
    return [(board_size - 1 - j, i) for (i, j) in pattern]

def reflect(pattern, board_size):
    return [(i, board_size - 1 - j) for (i, j) in pattern]

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        self.symmetry_groups = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_groups.append(syms)
            for syms_ in syms:
                self.symmetry_patterns.append(syms_)

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        board_size = self.board_size
        sym0 = pattern
        sym1 = rot90(pattern, board_size)
        sym2 = rot180(pattern, board_size)
        sym3 = rot270(pattern, board_size)
        syms = [sym0, sym1, sym2, sym3,
              reflect(sym0, board_size),
              reflect(sym1, board_size),
              reflect(sym2, board_size),
              reflect(sym3, board_size)]
        return syms


    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple(self.tile_to_index(board[i, j]) for (i, j) in coords)


    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        total_value = 0.0
        for i, syms in enumerate(self.symmetry_groups):
            group_value = 0.0
            for pattern in syms:
                feature = self.get_feature(board, pattern)
                group_value += self.weights[i][feature]
            total_value += group_value / len(syms)
        return total_value

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        for i, syms in enumerate(self.symmetry_groups):
            update_value = alpha * delta / len(syms)
            for pattern in syms:
                feature = self.get_feature(board, pattern)
                self.weights[i][feature] += update_value