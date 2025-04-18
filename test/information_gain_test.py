import math
import numpy as np
import pytest

from from_scratch import DecisionTree

def test_information_gain_1():
    tree = DecisionTree(max_depth=5)

    left = np.concatenate((np.repeat(0, 3), np.repeat(1, 2)))
    right = np.concatenate((np.repeat(1, 2), np.repeat(2, 3)))
    parent = np.concatenate((left, right))

    assert abs(tree.information_gain(parent, left, right) - 0.599) <= 0.1

def test_information_gain_2():
    tree = DecisionTree(max_depth=5)

    left = np.concatenate((np.repeat(0, 13), np.repeat(1, 4)))
    right = np.concatenate((np.repeat(0, 1), np.repeat(1, 12)))
    parent = np.concatenate((left, right))

    assert abs(tree.information_gain(parent, left, right) - 0.38) <= 0.1

def test_information_gain_3():
    tree = DecisionTree(max_depth=5)

    left = np.concatenate((np.repeat(0, 2), np.repeat(1, 1)))
    right = np.concatenate((np.repeat(0, 1), np.repeat(1, 1)))
    parent = np.concatenate((left, right))

    assert abs(tree.information_gain(parent, left, right) - 0.0202) <= 0.1

def test_information_gain_4():
    tree = DecisionTree(max_depth=5)

    left = np.concatenate((np.repeat(0, 6), np.repeat(1, 2)))
    right = np.concatenate((np.repeat(0, 3), np.repeat(1, 3)))
    parent = np.concatenate((left, right))
    
    assert abs(tree.information_gain(parent, left, right) - 0.048) <= 0.1