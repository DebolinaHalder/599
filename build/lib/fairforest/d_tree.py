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
        print("spliting")
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

                #print("done with this feature")
                if GINIgain > max_gain:
                    best_feature = feature
                    best_value = value 
                    max_gain = GINIgain
        print("spliting done")
        print(best_feature, best_value)
        return best_feature, best_value, GINIgain

    def fit(self,X,y):
        self.total_classes = len(np.unique(y))
        self.feature_list = list(X.columns)
        self.feature_size = len(self.feature_list)
        self.node_count = 1
        self.fairness_score = {}
        self.feature_important_score = {}
        self.X = X
        self.y = y
        def build_tree(X_, y_, node):
            print("build tree for node ", node)
            classes, count = np.unique(y_, return_counts=True)
            self.number_of_data_points[node] = len(y_)
            if len(classes) == 1:
                print("only one class for this node")
                #self.node_count += 1
                self.children_left[node] = -2
                self.children_right[node] = -2
                self.feature[node] = -2
                self.impurity[node] = 0
                self.threshold[node] = None
                posterior = np.zeros(self.total_classes, dtype=float)
                posterior[classes] = 1
                self.leaf_to_posterior[node] = posterior
                prediction = np.full(len(y_), np.argmax(posterior))
                self.parity[node] = DP(X_.to_numpy(),prediction,self.protected_attribute, self.protected_val)
                return
            
            self.impurity[node] = gini(y_)
            posterior = np.zeros(self.total_classes, dtype=float)
            for i in range(self.total_classes):
                posterior[i] = count[i] / len(y_)
            self.leaf_to_posterior[node] = posterior
            prediction = np.full(len(y_), np.argmax(posterior))
            self.parity[node] = DP(X_.to_numpy(),prediction,self.protected_attribute, self.protected_val)
            best_feature, best_value, giniGain = self._best_split(X_,y_)
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
            elif giniGain < 0.001 or best_feature is None:
                self.node_count += 1
                self.children_left[node] = -2
                self.children_right[node] = -2
                self.feature[node] = -2
                self.threshold[node] = None
                return
        build_tree(X,y,0)


    def _feature_importance(self):
        for i in self.feature_list:
            self.feature_important_score[i] = 0
        number_of_times = dict.fromkeys(self.feature_list, 0)
        for i in range (self.node_count):
            if i in self.feature.keys():
                if self.feature[i] != -2:
                    self.feature_important_score[self.feature[i]] += (self.impurity[i] - (self.impurity[self.children_left[i]] + self.impurity[self.children_right[i]]))
                    number_of_times[self.feature[i]] += 1
        for key, value in number_of_times.items():
            if value != 0:
                self.feature_important_score[key] /= value
        return self.feature_important_score

    def _fairness_importance(self):
        for i in self.feature_list:
            
            self.fairness_score[i] = 0
        number_of_times = dict.fromkeys(self.feature_list, 0)
        for i in range (self.node_count):
            if i in self.feature.keys():
                if self.feature[i] != -2:
                    self.fairness_score[self.feature[i]] += ((self.impurity[self.children_right[i]] + self.impurity[self.children_left[i]]) - self.impurity[i])
                    number_of_times[self.feature[i]] += 1
        for key, value in number_of_times.items():
            if value != 0:
                self.fairness_score[key] /= value
        return self.fairness_score

    def predict(self,x):
        return
    def predict_proba(self, X):
        return