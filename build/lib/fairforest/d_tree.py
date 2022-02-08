#%%
from .base import Tree
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from .utils import DP, gini, ma
from collections import Counter
# %%
class DecisionTree(Tree):
    def __init__(self, protected_attribute, protected_value):
        super().__init__()
        self.fairness_score = np.array([])
        self._feature_important_score = np.array([])
        self.children_left = {}
        self.children_right = {}
        self.feature = {}
        self.leaf_to_posterior = {}
        self.threshold = {}
        self.impurity={}
        self.parity={}
        self.number_of_data_points = {}
        self.protected_attribute = protected_attribute
        self.protected_val = protected_value
    
    def _best_split(self,X,y):
        df = X.copy()
        df['Y'] = y
        GINI_base = gini(y)

        max_gain = 0

        best_feature = None
        best_value = None
        
        for feature in self.feature_list:
            Xdf = df.dropna().sort_values(feature)
            xmeans = ma(Xdf[feature].unique(), 2)

            for value in xmeans:
                left_df = Xdf[Xdf[feature]<value].copy()
                right_df = Xdf[Xdf[feature]>=value].copy()
                left_target = left_df['Y']
                right_target = right_df['Y']
                gini_left = gini(left_target.to_numpy())
                gini_right = gini(right_target.to_numpy())
                
                

                
                n_left = len(left_target)
                n_right = len(right_target)

                
                w_left = n_left / (n_left + n_right)
                w_right = n_right / (n_left + n_right)

                
                wGINI = w_left * gini_left + w_right * gini_right

                 
                GINIgain = GINI_base - wGINI

                
                if GINIgain > max_gain:
                    best_feature = feature
                    best_value = value 

                     
                    max_gain = GINIgain

        return best_feature, best_value

    def fit(self,X,y):
        super().__init__()
        self.total_classes = np.unique(y)
        self.feature_list = list(self.X.columns)
        self.feature_size = len(self.feature_list)
        self.node_count = 1
        self.X = X
        self.y = y
        def build_tree(X_, y_, node):
            if(len(y_ == 0)):
                return
            classes, count = np.unique(y_, return_counts=True)
            self.number_of_data_points[node] = len(y_)
            if len(classes) == 1:
                self.node_count += 1
                self.children_left[node] = -2
                self.children_right[node] = -2
                self.feature[node] = -2
                self.impurity[node] = 0
                self.threshold[node] = None
                posterior = np.zeros(self.total_classes, dtype=float)
                posterior[classes] = 1
                self.leaf_to_posterior[node] = posterior
                prediction = np.full(len(y_), max(posterior, key=lambda key: posterior[key]))
                self.parity[node] = DP(X_.to_numpy(),prediction,self.protected_attribute, self.protected_val)
                return
            
            self.impurity[node] = gini(y_)
            posterior = np.zeros(self.total_classes, dtype=float)
            for i in len(self.total_classes):
                posterior[i] = count[i] / len(y_)
            self.leaf_to_posterior[node] = posterior
            prediction = np.full(len(y_), max(posterior, key=lambda key: posterior[key]))
            self.parity[node] = DP(X_.to_numpy(),prediction,self.protected_attribute, self.protected_val)
            best_feature, best_value = self._best_split(X_,y_)
            if best_feature is not None:
                df = X_.copy()
                df['Y'] = y_
                self.feature[node] = best_feature
                self.threshold[node] = best_value
                left_df, right_df = df[df[best_feature]<=best_value].copy(), df[df[best_feature]>best_value].copy()
                left_target = left_df['Y'].to_numpy()
                right_target = right_df['Y'].to_numpy()
                left_df = left_df.drop('Y',1)
                right_df = right_df.drop('Y',1)
                left_node = self.node_count
                right_node = self.node_count + 1
                self.node_count += 2
                self.children_left[node] = left_node
                self.children_right[node] = right_node
                build_tree(left_df, left_target,left_node)
                build_tree(right_df, right_target, right_node)
            else:
                self.node_count += 1
                self.children_left[node] = -2
                self.children_right[node] = -2
                self.feature[node] = -2
                self.threshold[node] = None
                return