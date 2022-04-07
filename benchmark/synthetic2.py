#%%
import numpy as np
import pandas as pd
from scipy.special import logit
from fairforest import d_tree
from fairforest import utils
import warnings
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
#%%
warnings.simplefilter("ignore")

#%%
np.random.seed(0)
#%%
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

#%%
def build_dataset():
    z = np.zeros(1000)
    for j in range(1000):
        z[j] = np.random.binomial(1,0.7)
    x_correlated = np.zeros((1000,4))
    x_uncorrelated = np.zeros((1000,16))
    for j in range(16):
        for i in range (1000):
            if j < 4:
                x_correlated[i][j] = (j+2)*np.random.normal((z[i] + 1), 1, 1)
            x_uncorrelated[i][j] = np.random.normal(0,1,1)
    x = np.concatenate((x_correlated,x_uncorrelated),axis=1)
    x = np.concatenate((x,np.reshape(z,(1000,1))),axis=1)
    b = np.zeros(21)
    noise = np.random.normal(0,1,1000)
    for i in range (10):
        b[i] = np.random.uniform(0,5)
    b[20] = np.random.uniform(0,5)
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
    return dataframe,y

#%%
feature = np.arange(21)
score_fairness = np.zeros(21)
score_feature = np.zeros(21)

for i in range (10):
    dataframe, y = build_dataset()
    #####protected_attribute,protected_value,protected_feature,fairness_metric
    model_dtree = d_tree.DecisionTree(15,0,'16',1)
    model_dtree.fit(dataframe,y)
    fairness_importance = model_dtree._fairness_importance()
    feature_importance = model_dtree._feature_importance()
    for key, value in fairness_importance.items():
        score_fairness[int(key)-1]=value
    for key, value in feature_importance.items():
        score_feature[int(key)-1]=value
for i in range (21):
    score_fairness[i] /= 10
    score_feature[i] /= 10
utils.draw_plot(feature,score_fairness,"Results/Synthetic/dp2.pdf", "fairness importance")
utils.draw_plot(feature,score_feature,"Results/Synthetic/feature2.pdf", "feature importance")


# %%
fairness_importance = model_dtree._fairness_importance()
# %%
feature = []
score = []
for key, value in fairness_importance.items():
    print(key, value)
    feature.append(key)
    score.append((value))
utils.draw_plot(feature,score,"Results/Synthetic/eqop2.pdf")


#%%
model_dtree_dp = d_tree.DecisionTree(20,1,'21',2)
model_dtree_dp.fit(dataframe,y)


# %%
fairness_importance_dp = model_dtree_dp._fairness_importance()
# %%
feature = []
score_dp = []
for key, value in fairness_importance_dp.items():
    print(key, value)
    feature.append(key)
    score_dp.append((value))
utils.draw_plot(feature,score_dp,"Results/Synthetic/DP2.pdf")


#%%
feature_importance_dp = model_dtree_dp._fairness_importance()
# %%
feature = []
score_dp = []
for key, value in feature_importance_dp.items():
    print(key, value)
    feature.append(key)
    score_dp.append((value))
utils.draw_plot(feature,score_dp,"Results/Synthetic/DP2.pdf")

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


