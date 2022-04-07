#%%
from asyncio.windows_events import NULL
from .base import Tree
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from .utils import fairness, gini, ma,eqop,DP
from collections import Counter
import random
# %%
class RandomTree(Tree):
    def __init__(self, protected_attribute, protected_value, protected_feature,fairness_metric):
        super().__init__()
        
        self.children_left = {}
        self.children_right = {}
        self.feature = {}
        self.leaf_to_posterior = {}
        self.threshold = {}
        self.impurity={}
        self.fair_score={}
        self.number_of_data_points = {}
        self.protected_attribute = protected_attribute
        self.protected_val = protected_value
        self.protected_feature = protected_feature
        self.fairness_metric = fairness_metric
    
    def _best_split(self,X,y):
        #print("spliting")
        df = X.copy()
        df['Y'] = y
        GINI_base = gini(y)

        max_gain = 0
        f_list = self.feature_list.copy()
        f_list.remove(self.protected_feature)
        
        best_feature = random.choice(f_list)
        Xdf = df.dropna().sort_values(best_feature)
        
        X_max, X_min = Xdf[best_feature].max(), Xdf[best_feature].min()
        threshold_ = np.random.uniform(X_min, X_max)
        return best_feature, threshold_, 0

    def fit(self,X,y):
        self.total_classes = len(np.unique(y))
        self.feature_list = list(X.columns)
        self.feature_size = len(self.feature_list)
        self.node_count = 1
        self.fairness_importance_score = {}
        self.feature_important_score = {}
        self.weighted_n_node_samples = {}

        self.X = X
        self.y = y
        def build_tree(X_, y_, node, parent_samples):
            #print("build tree for node ", node)
            self.weighted_n_node_samples[node] = parent_samples
            classes, count = np.unique(y_, return_counts=True)
            self.number_of_data_points[node] = len(y_)
            if node == 0:
                valuey, county = np.unique(y_, return_counts=True)
                pred = np.full(len(y_), np.argmax(county))
                if self.fairness_metric == 1:
                    s = eqop(X_.to_numpy(),y_,pred,self.protected_attribute,self.protected_val)
                elif self.fairness_metric == 2:
                    s = DP(X_.to_numpy(),y_,pred,self.protected_attribute,self.protected_val)
                print(s)
                self.fair_score[node] = s
                print(self.fair_score[node])
            if len(classes) == 1:
                #print("only one class for this node")
                #self.node_count += 1
                self.children_left[node] = -2
                self.children_right[node] = -2
                self.feature[node] = -2
                self.impurity[node] = 0
                self.threshold[node] = None
                posterior = np.zeros(self.total_classes, dtype=float)
                posterior[int(classes)] = 1
                self.leaf_to_posterior[node] = posterior
                #prediction = np.full(len(y_), np.argmax(posterior))
                
                
                return
            
            self.impurity[node] = gini(y_)
            posterior = np.zeros(self.total_classes, dtype=float)
            for i in range(self.total_classes):
                posterior[i] = count[i] / len(y_)
            self.leaf_to_posterior[node] = posterior
            #prediction = np.full(len(y_), np.argmax(posterior))
            #self.parity[node] = eqop(X_.to_numpy(),y,prediction,self.protected_attribute, self.protected_val)
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
                self.fair_score[left_node] = self.fair_score[right_node] = fairness(left_df.to_numpy(),left_target,right_df.to_numpy(),right_target,self.protected_attribute, self.protected_val,self.fairness_metric)
                build_tree(left_df, left_target,left_node, len(y_))
                build_tree(right_df, right_target, right_node, len(y_))
            else:
                self.node_count += 1
                self.children_left[node] = -2
                self.children_right[node] = -2
                self.feature[node] = -2
                self.threshold[node] = None
                return
        build_tree(X,y,0, len(y))


    def _feature_importance(self):
        for i in self.feature_list:
            self.feature_important_score[i] = 0
        number_of_times = dict.fromkeys(self.feature_list, 0)
        for i in range (self.node_count):
            if i in self.feature.keys():
                if self.feature[i] != -2:
                    self.feature_important_score[self.feature[i]] += ((self.number_of_data_points[i]*self.impurity[i] - 
                    self.impurity[self.children_left[i]]*self.number_of_data_points[self.children_left[i]] -
                    self.impurity[self.children_right[i]]*self.number_of_data_points[self.children_right[i]])/self.number_of_data_points[0])
                    number_of_times[self.feature[i]] += 1
        #for key, value in number_of_times.items():
        #    if value != 0:
        #        self.feature_important_score[key] /= value
        
        return self.feature_important_score

    def _fairness_importance(self):
        for i in self.feature_list:
            self.fairness_importance_score[i] = 0
        number_of_times = dict.fromkeys(self.feature_list, 0)
        for i in range (self.node_count):
            #if i == 0:
            #    continue
            if i in self.feature.keys():
                if self.feature[i] != -2:
                    print(self.feature[i],self.fair_score[i],self.fair_score[self.children_left[i]])
                    self.fairness_importance_score[self.feature[i]] += ((self.fair_score[self.children_left[i]] - self.fair_score[i])*((self.number_of_data_points[i]/self.number_of_data_points[0])**2))
                    number_of_times[self.feature[i]] += 1
        #for key, value in number_of_times.items():
        #    if value != 0:
        #        self.fairness_importance_score[key] /= value
        return self.fairness_importance_score

    def predict(self,x):
        return
    def predict_proba(self, X):
        return
# %%
