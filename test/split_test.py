import math
import numpy as np
import pytest

from from_scratch import DecisionTree

def test_split_1():
    tree = DecisionTree(max_depth=5)
    arr = np.arange(1, 10).reshape(3, 3)
    left, right = tree.split(arr, 1, 5)
    assert (left == np.array([[1, 2, 3], [4, 5, 6]])).all() and (right == np.array([7, 8, 9])).all()
    
def test_split_2():
    tree = DecisionTree(max_depth=5)
    arr = np.arange(1, 10).reshape(3, 3)
    left, right = tree.split(arr, 2, 2)
    assert left.size == 0 and (right == np.arange(1, 10).reshape(3, 3)).all()