#%%
from base import Tree
import numpy as np
from sklearn.tree import DecisionTreeClassifier
# %%
class DecisionTree(Tree):
    def __init__(self):
        super().__init__()
        self.fairness_score = np.array([])
    def fit(self,X,y):
        self.label = np.unique(y)
        self.decisiontree_model=DecisionTreeClassifier(random_state=0)
        
    def predict():
