import math
import numpy as np
import pytest

from from_scratch import DecisionTree

def test_entropy_1():
    tree = DecisionTree(max_depth=5)
    arr = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    np.random.shuffle(arr)
    assert abs(tree.entropy(arr) - 0.971) <= 0.1

def test_entropy_2():
    tree = DecisionTree(max_depth=5)
    arr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    np.random.shuffle(arr)
    assert abs(tree.entropy(arr) - 0.94) <= 0.1

def test_entropy_3():
    tree = DecisionTree(max_depth=5)
    arr = np.concatenate((np.repeat(0, 13), np.repeat(1, 1)))
    np.random.shuffle(arr)
    assert abs(tree.entropy(arr) - 0.39) <= 0.1

def test_entropy_4():
    tree = DecisionTree(max_depth=5)
    arr = np.concatenate((np.repeat(0, 4), np.repeat(1, 13)))
    np.random.shuffle(arr)
    assert abs(tree.entropy(arr) - 0.79) <= 0.1