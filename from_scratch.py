import os
import csv

import numpy as np
import pandas

class Node: 
    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain


class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        
    def best_split(self, dataset):
        best_feature = -1
        best_threshold = -1
        best_left = None
        best_right = None
        best_information_gain = float('inf')

        features = dataset[:-1]
        target = dataset[-1]

        for feature in range(dataset.shape[-1] - 1):    
            thresholds = np.unique(dataset[:, feature])

            for threshold in thresholds:
                left, right = self.split(dataset, feature, threshold)
                information_gain = self.information_gain(dataset[:-1], left[:-1], right[:-1])

                if information_gain > best_information_gain:
                    best_feature = feature
                    best_threshold = threshold
                    best_left = left
                    best_right = right
                    best_information_gain = information_gain

        return best_feature, best_threshold, best_left, best_right, best_information_gain            

    def build_tree(self, dataset, node):
        feature, threshold, left, right, gain = self.best_split(dataset)
        left = self.build_tree(self, left)
        right = self.build_tree(self, right)
        parent = Node(feature, threshold, left, right, gain)

        return parent

    def fit(self, x, y):
        dataset = np.concatenate((np.array(x), np.array(y).reshape(-1, 1)), axis=-1)

    def split(self, dataset, feature, threshold):
        left = dataset[dataset[:, feature] <= threshold]
        right = dataset[dataset[:, feature] > threshold]
        
        return left, right
    
    def information_gain(self, parent, left, right):
        parent_entropy = self.entropy(parent)
        left_entropy = self.entropy(left)
        right_entropy = self.entropy(right)

    def entropy(self, arr):
        prob = np.array(np.unique(arr, return_counts=True, axis=0))[1, :] / arr.shape[0]
        return -(prob * np.log2(prob)).sum()

    

PATH = os.path.join('data', '')

if __name__ == "__main__":
    data = []
    with open(PATH, mode ='r') as file:
        df = pandas.read_csv(PATH)

        df['Credit History'] = df['Credit History'].astype('category').cat.codes
        df['Debt'] = df['Debt'].astype('category').cat.codes
        df['Collateral'] = df['Collateral'].astype('category').cat.codes
        df['Credit Risk?'] = df['Credit Risk?'].astype('category').cat.codes
        
    print()
        