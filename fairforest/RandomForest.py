from asyncio.windows_events import NULL
from .base import Tree
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from .utils import fairness, gini, ma,eqop,DP, fairness2, fairness_wrong
from collections import Counter
from .d_tree import DecisionTree
from joblib import Parallel, delayed

class RandomForest(Tree):
    def __init__(self, protected_attribute, protected_value, protected_feature,fairness_metric,number_of_trees,m_tries):
        super().__init__()
        self.tree_list = []
        self.protected_attribute = protected_attribute
        self.protected_value = protected_value
        self.protected_feature = protected_feature
        self.fairness_metric = fairness_metric
        self.number_of_trees = number_of_trees
        self.m_tries = m_tries
        self.fairness_importance_score = {}
        self.accuracy_importance_score = {}

    def _fit_tree(self,X,y):
        df = X.copy()
        df['Y'] = y
        df = df.sample(frac = 1, replace = True)
        y_ = df['Y'].to_numpy()
        X_ = df.drop("Y", axis=1, inplace=False)
        tree = DecisionTree(self.protected_attribute, self.protected_value, self.protected_feature,self.fairness_metric,self.m_tries)
        tree.fit(X_,y_)
        return tree
    def fit(self,X,y):
        self.total_classes = len(np.unique(y))
        self.feature_list = list(X.columns)
        self.feature_size = len(self.feature_list)
        grid = np.arange(self.number_of_trees)
        self.tree_list = Parallel(n_jobs=-1,verbose=1)(
        delayed(self._fit_tree)(X,y) 
        for grid_val in grid
        )

        #for i in range (self.number_of_trees):
        #    tree = DecisionTree(self.protected_attribute, self.protected_value, self.protected_feature,self.fairness_metric,self.m_tries)
            
        #    tree.fit(X,y)
        #    self.tree_list.append(tree)
    
    def _fairness_importance(self):
        for i in self.feature_list:
            self.fairness_importance_score[i] = 0
        for tree in self.tree_list:
            fairness = tree._fairness_importance()
            for key,value in fairness.items():
                self.fairness_importance_score[key] += value
        for key,value in self.fairness_importance_score.items():
            self.fairness_importance_score[key] /= len(self.tree_list)
        return self.fairness_importance_score

    def _feature_importance(self):
        for i in self.feature_list:
            self.accuracy_importance_score[i] = 0
        for tree in self.tree_list:
            importance = tree._feature_importance()
            for key,value in importance.items():
                self.accuracy_importance_score[key] += value
        for key,value in self.accuracy_importance_score.items():
            self.accuracy_importance_score[key] /= len(self.tree_list)
        return self.accuracy_importance_score


    def predict(self,x):
        return
    def predict_proba(self, X):
        return