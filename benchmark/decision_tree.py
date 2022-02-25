#%%
import numpy as np
import pandas as pd
from fairforest import d_tree
from fairforest import utils
import warnings
#%%
warnings.simplefilter("ignore")
# %%
df = pd.read_csv("dataset/adult.csv")
df = df[0:100]
target = df['income-per-year'].to_numpy()
df = df.drop('income-per-year',1)
target = np.where(target == -1, 0, target)

# %%
model_dtree = d_tree.DecisionTree(3,0)
model_dtree.fit(df,target)
# %%
feature_importance = model_dtree._feature_importance()
# %%
fairness_importance = model_dtree._fairness_importance()
# %%
for key, value in fairness_importance.items():
    print(key, value)
# %%
for key, value in feature_importance.items():
    print(key, value)

#%%
for key, value in model_dtree.feature.items():
    print(key, value)

# %%
