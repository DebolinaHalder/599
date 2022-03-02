#%%
import numpy as np
import pandas as pd
from scipy.special import logit
from fairforest import d_tree
from fairforest import utils
import warnings
import matplotlib.pyplot as plt
#%%
warnings.simplefilter("ignore")

#%%
np.random.seed(0)
#%%
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

#%%
z = np.zeros(1000)
for j in range(1000):
    z[j] = np.random.binomial(1,0.7)
x_correlated = np.zeros((1000,4))
x_uncorrelated = np.zeros((1000,16))
for j in range(16):
    for i in range (1000):
        if j < 4:
            x_correlated[i][j] = np.random.normal((z[i] + 1), 1, 1)
        x_uncorrelated[i][j] = np.random.normal(0,1,1)
x = np.concatenate((x_correlated,x_uncorrelated),axis=1)
x = np.concatenate((x,np.reshape(z,(1000,1))),axis=1)
b = np.zeros(21)
noise = np.random.normal(0,1,1000)
for i in range (10):
    b[i] = np.random.normal(5,0.1,1)
y = logit(NormalizeData(np.dot(x,b)) + noise.T)
for i in range (len(y)):
    if y[i] > 0:
        y[i] = int(1)
    else:
        y[i] = int(0)
column = []
for i in range(21):
    column.append(str(i+1))
dataframe = pd.DataFrame(x, columns = column)

# %%
model_dtree = d_tree.DecisionTree(20,0,'21',1)
model_dtree.fit(dataframe,y)


# %%
fairness_importance = model_dtree._fairness_importance()
# %%
feature = []
score = []
y=[]
for key, value in fairness_importance.items():
    print(key, value)
    feature.append(key)
    score.append((value))
utils.draw_plot(feature,score,"Results/Synthetic/eqop.pdf")


#%%
model_dtree_dp = d_tree.DecisionTree(20,0,'21',2)
model_dtree_dp.fit(dataframe,y)


# %%
fairness_importance_dp = model_dtree_dp._fairness_importance()
# %%
feature = []
score_dp = []
y=[]
for key, value in fairness_importance_dp.items():
    print(key, value)
    feature.append(key)
    score_dp.append((value))
utils.draw_plot(feature,score_dp,"Results/Synthetic/DP.pdf")

# %%
count_z0 = count_z1 = 0
count0 = count1 = 0
z0 = z1 = 0
for i in range (1000):
    if y[i] == 0:
        count0+=1
    else:
        count1+=1
        if x[i][20] == 0:
            count_z0 += 1
        else:
            count_z1 +=1
    if x[i][20] == 0:
        z0+=1
    else:
        z1+=1
print(count0,count1, count_z0,count_z1,z0,z1)


# %%
prediction = np.zeros(1000)
print(utils.eqop(x,y,prediction,20,0))

# %%
