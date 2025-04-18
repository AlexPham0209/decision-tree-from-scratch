import os
import csv

import numpy as np
import pandas

PATH = os.path.join('data', 'credit.csv')

class Node: 
    def __init__(self, feature=None, threshold=None, gain=None, entropy=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.gain = gain
        self.value = value
        self.entropy = entropy
        self.left = None
        self.right = None

    def print_info(self, features, classes):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        if self.feature == None and self.threshold == None:
            print(f'{classes[self.value[0]]}')
        else:
            print(f'{features[self.feature]} <= {self.threshold}')
            print(f"Information Gain: {self.gain: .2f}")
        print(f"Entropy: {self.entropy: .3f}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
            
        
class DecisionTree:
    def __init__(self, max_depth=10, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None
    
    def fit(self, x, y):
        dataset = np.concatenate((np.array(x), np.array(y).reshape(-1, 1)), axis=-1)
        self.root = self.build_tree(dataset, 0)
        return self.root

    def build_tree(self, dataset, depth):
        feature, threshold, left, right, gain = self.best_split(dataset)
        entropy = self.entropy(dataset[:, -1])
        
        if depth >= self.max_depth or dataset.shape[0] < self.min_samples or gain == 0:
            return Node(value=self.calculate_leaf_value(dataset[:, -1]), entropy=entropy)

        parent = Node(feature, threshold, gain, entropy)
        parent.left = self.build_tree(left, depth + 1)
        parent.right = self.build_tree(right, depth + 1)

        return parent 

    def best_split(self, dataset):
        best_feature = -1
        best_threshold = -1
        best_left = None
        best_right = None
        best_information_gain = -float('inf')

        for feature in range(dataset.shape[-1] - 1):    
            thresholds = np.unique(dataset[:, feature])

            for threshold in thresholds:
                left, right = self.split(dataset, feature, threshold)
                information_gain = self.information_gain(dataset[:, -1], left[:, -1], right[:, -1])

                if information_gain > best_information_gain:
                    best_feature = feature
                    best_threshold = threshold
                    best_left = left
                    best_right = right
                    best_information_gain = information_gain

        return best_feature, best_threshold, best_left, best_right, best_information_gain            

    
    def split(self, dataset, feature, threshold):
        left = dataset[dataset[:, feature] <= threshold]
        right = dataset[dataset[:, feature] > threshold]
        
        return left, right
    
    def calculate_leaf_value(self, y):
        u, c = np.unique(y, return_counts = True)
        y = u[c == c.max()]
        return y
        
    def information_gain(self, parent, left, right):
        parent_entropy = self.entropy(parent)
        left_entropy = self.entropy(left)
        right_entropy = self.entropy(right)

        left_weight = left.shape[0] / parent.shape[0]
        right_weight = right.shape[0] / parent.shape[0]

        weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
        return parent_entropy - weighted_entropy

    def entropy(self, arr):
        prob = np.array(np.unique(arr, return_counts=True, axis=0))[1, :] / arr.shape[0]
        return -(prob * np.log2(prob)).sum()
    
    def print_info(self, features, classes):
        def dfs(curr):
            if curr == None:
                return

            curr.print_info(features, classes)
            dfs(curr.left)
            dfs(curr.right)

        dfs(self.root)

    def predict(self, x):
        def dfs(node):
            nonlocal x 
            if not node.left and not node.right:
                return node.value[0]

            return dfs(node.left) if x[node.feature] <= node.threshold else dfs(node.right)

        return dfs(self.root)


if __name__ == "__main__":
    features_map = {
    0: 'Credit History', 
    1: 'Debt', 
    2: 'Collateral'
}

    classes_map = {
        0: 'High', 
        1: 'Low', 
    }

    with open(PATH, mode ='r') as file:
        df = pandas.read_csv(PATH)

        df['Credit History'] = df['Credit History'].astype('category').cat.codes
        df['Debt'] = df['Debt'].astype('category').cat.codes
        df['Collateral'] = df['Collateral'].astype('category').cat.codes
        df['Credit Risk?'] = df['Credit Risk?'].astype('category').cat.codes

        features = ['Credit History', 'Debt', 'Collateral']
        X = df[features]
        Y = df['Credit Risk?']

        tree = DecisionTree()
        tree.fit(X, Y)
        tree.print_info(features_map, classes_map)
        print(classes_map[tree.predict(np.array([1, 0, 0]))])
        print(classes_map[tree.predict(np.array([2, 1, 0]))])

        