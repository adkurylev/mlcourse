from logging import root
import pandas as pd
import math
from collections import Counter

class DecisionTreeNode:
    def __init__(self, column=None, loe_number=0, left=None, right=None, prediction=None) -> None:
        self.column = column
        self.loe_number = loe_number
        self.left = left
        self.right = right
        self.prediction = prediction

class DecisionTree:
    def __init__(self, max_depth=3, min_samples_leaf=1) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, df, columns, target):
        self.columns = columns
        self.target = target
        self.root = self.train_recursive(0, df)

    def train_recursive(self, depth, df: pd.DataFrame):
        if depth > self.max_depth:
            return None

        if len(df) < self.min_samples_leaf:
            return None

        node = DecisionTreeNode()
        node.prediction = Counter(df[self.target]).most_common(1)[0][0]
        print(f"class:{node.prediction}")

        if df[self.target].nunique() == 1:
            return node

        max_ig = -math.inf
        max_ig_split = 0
        max_ig_column = None

        for c in self.columns:
            min_v = min(df[c])
            max_v = max(df[c])

            if min_v == max_v:
                continue

            split = min_v

            while split <= max_v:
                split += (max_v - min_v) / 100

                df_1 = df[df[c] <= split]
                df_2 = df[df[c] > split]

                ig = self.information_gain(df[self.target], df_1[self.target], df_2[self.target])

                if ig > max_ig:
                    max_ig = ig
                    max_ig_split = split
                    max_ig_column = c

        if max_ig_column is None:
            return None

        node.column = max_ig_column
        node.loe_number = max_ig_split        

        print(f"Depth {depth}: {max_ig_column} <= {max_ig_split}")

        node.left = self.train_recursive(depth+1, df[df[max_ig_column] <= max_ig_split])
        node.right = self.train_recursive(depth+1, df[df[max_ig_column] > max_ig_split])

        return node
        

    def predict(self, X):
        res = []

        for i in range(len(X)):
            current_node = self.root

            while current_node.left or current_node.right:
                if X.loc[i, :][current_node.column] <= current_node.loe_number:
                    current_node = current_node.left
                else:
                    current_node = current_node.right

            res.append(current_node.prediction)

        return res

    def entropy(self, a_list):
        counts = Counter(a_list)
        s = 0
        n = len(a_list)

        for value in counts.values():
            s -= value/n * math.log(value/n, 2)

        return s

    def information_gain(self, root, left, right):
        return self.entropy(root) - len(left)/len(root)*self.entropy(left) - len(right)/len(root)*self.entropy(right) 

