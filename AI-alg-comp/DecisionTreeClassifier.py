import numpy as np
import sys

class DecisionTreeClassifier:
    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X, y):
        if self._should_stop(y):
            return self._create_leaf_node(y)

        feature_index, threshold = self._find_best_split(X, y)
        left_X, left_y, right_X, right_y = self._split_data(X, y, feature_index, threshold)

        left_subtree = self._build_tree(left_X, left_y)
        right_subtree = self._build_tree(right_X, right_y)

        return self._create_internal_node(feature_index, threshold, left_subtree, right_subtree)

    def _traverse_tree(self, x, node):
        if node["leaf"]:
            return node["value"]
        else:
            if x[node["feature_index"]] <= node["threshold"]:
                return self._traverse_tree(x, node["left"])
            else:
                return self._traverse_tree(x, node["right"])

    def _should_stop(self, y):
        return len(np.unique(y)) == 1

    def _find_best_split(self, X, y):
        best_gini = float('inf')
        best_feature_index = None
        best_threshold = None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gini = self._calculate_gini(X, y, feature_index, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _calculate_gini(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold

        left_y = y[left_mask]
        right_y = y[right_mask]

        left_gini = self._gini_impurity(left_y)
        right_gini = self._gini_impurity(right_y)

        left_weight = len(left_y) / len(y)
        right_weight = len(right_y) / len(y)

        gini = left_weight * left_gini + right_weight * right_gini
        return gini

    def _gini_impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities**2)
        return gini

    def _split_data(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold

        left_X = X[left_mask]
        left_y = y[left_mask]

        right_X = X[right_mask]
        right_y = y[right_mask]

        return left_X, left_y, right_X, right_y

    def _create_leaf_node(self, y):
        value_counts = np.bincount(y)
        value = np.argmax(value_counts)
        return {"leaf": True, "value": value, "left": None, "right": None}

    def _create_internal_node(self, feature_index, threshold, left_subtree, right_subtree):
        return {
            "leaf": False,
            "feature_index": feature_index,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree
        }