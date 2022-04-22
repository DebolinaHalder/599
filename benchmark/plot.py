#%%
from cProfile import label
from turtle import color
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%

sns.set_context("talk")
fig, axs = plt.subplots(3,figsize=(12,22.5))
df = pd.read_csv("Results/synthetic_gaussian/results_g_eqop_tree_1.csv")
x1 = df['fairness'].copy()
x2 = df['occlusion'].copy()
x3 = df['accuracy'].copy()
y = np.arange(1,10)
axs[0].bar(y, x1, color = 'r',label = "fairness_importance_score")
axs[0].set_ylabel("Fairness Importance Score")
axs[0].title.set_text("(A) Fairness Feature Importance Score")
#axs[0].set_xlabel("feature")
axs[0].legend(loc='lower right')
axs[1].bar(y, x2, color = 'b', label = "Occlusion Score")
axs[1].set_ylabel("Occlusion Fairness Importance Score")
#axs[1].set_xlabel("feature")
axs[1].title.set_text("(B)Occlusion Fairness Feature Importance Score")
axs[1].legend(loc='lower right')

axs[2].bar(y, x3, color = 'b', label = "Occlusion Score")
axs[2].set_ylabel("Accuracy Importance Score")
axs[2].set_xlabel("feature")
axs[2].title.set_text("(B)Accuracy Feature Importance Score")
axs[2].legend(loc='lower right')

plt.savefig("Results/synthetic_gaussian/eqop_tree_all.pdf")
plt.show()
# %%
sns.set_context("talk")
fig, ax = plt.subplots(1,figsize=(12,7.5))
df1 = pd.read_csv("Results/synthetic_gaussian/results_g_DP_tree.csv")
df1 = df1[:-1]
y = np.arange(1,9)
x1 = df1['accuracy'].copy()
ax.set_ylabel("Accuracy Importance")
ax.bar(y, x1, color = 'g')
ax.set_xlabel("feature")
#ax.set_xticklabels(y, rotation=45, ha='right')
#ax.title.set_text("(B)")
ax.legend(loc='lower right')
plt.savefig("Results/synthetic_gaussian/DP_feature_tree.pdf")
plt.show()

# %%

# %%
