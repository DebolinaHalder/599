#%%
from cProfile import label
from turtle import color
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%

sns.set_context("talk")
fig, axs = plt.subplots(1,figsize=(16,10))
df = pd.read_csv("Results/synthetic_gaussian/results_g_DP_2000.csv")
x1 = df['fairness'].copy()
x2 = df['occlusion'].copy()
#x3 = df['accuracy'].copy()
y = np.arange(1,10)
width = 0.3
axs.bar(y-width, x1,width=width, color = 'r',label = "Fairness Importance Score")
#axs[0].set_ylabel("Fairness Importance Score")
#axs[0].title.set_text("(A) Fairness Feature Importance Score")
#axs[0].set_xlabel("feature")
#axs[0].legend(loc='lower right')
axs.bar(y, x2,width=width, color = 'b', label = "Fairness Occlusion Score")
#axs[1].set_ylabel("Occlusion Fairness Importance Score")
axs.set_xlabel("Feature")
axs.set_xticks(list(range(1,10)))
#axs[1].title.set_text("(B)Occlusion Fairness Feature Importance Score")
#axs[1].legend(loc='lower right')

#axs.bar(y+width, x3,width=width, color = 'g', label = "Accuracy Score")
axs.set_ylabel("Importance Score")
#axs[2].set_xlabel("feature")
#axs.title.set_text()
#axs[2].legend(loc='lower right')
axs.legend(loc = 'lower right')
plt.savefig("Results/synthetic_gaussian/DP_all_1.pdf")
plt.show()
   # %%
sns.set_context("talk")
fig, ax = plt.subplots(1,figsize=(16,10))
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
