#%%
import sys
#sys.path.append("F:\FairForest")
import numpy as np
import pandas as pd
from fairforest import d_tree
from fairforest import utils
# %%
df = pd.read_csv("dataset/adult.csv")
df_protected = df[df['sex']==0]
target = df['income-per-year'].to_numpy()
df = df.drop('income-per-year',1)
print(df.head())
target_protected = df_protected['income-per-year'].to_numpy()
df_protected = df_protected.drop('income-per-year',1)
print(len(df))
print(len(df_protected))
# %%
print(utils.gini(target))
# %%
print(utils.DP(df.to_numpy(),target,4,0))

# %%
