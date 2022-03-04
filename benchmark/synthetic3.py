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
total_sample = 1000
number_of_correlated_features = 3
number_of_important_features = 3
number_of_uncorrelated_features = 9

#%%
#np.random.seed(10)

#%%
feature = np.arange(16)
score = np.zeros(16)
for i in range (10):
    z0_prob = 0.3
    z1_prob = 0.8
    z = np.ones(total_sample)
    mu_correlated  = [0.85,0.8,0.75]
    mu_important = [0.65,0.6,0.3]
    x_correlated = np.zeros((total_sample,number_of_correlated_features))
    x_uncorrelated = np.zeros((total_sample,number_of_uncorrelated_features))
    x_important = np.zeros((total_sample,number_of_important_features))
    mean_correlated_0 = np.arange(11,14)
    mean_correlated_1 = np.arange(13,16)
    mean_imp_0 = np.arange(6,9)
    mean_imp_1 = np.arange(8,11)
    uncorr_sample_cov = np.zeros((number_of_uncorrelated_features,number_of_uncorrelated_features))
    for i in range(number_of_uncorrelated_features):
        uncorr_sample_cov[i][i] = 1
    uncorr_sample_mean = np.full(number_of_uncorrelated_features,10)
    mean_0 = -1
    var_0 = var_1 = 1
    mean_1 = 2
    y = np.zeros(total_sample)
    for i in range (500):
        y[i] = 1
        point_1 = np.random.normal(mean_1, var_1)
        z[i] = np.random.binomial(1, z1_prob)
        if z[i] == 1:
            for j in range (number_of_correlated_features):
                x_correlated[i][j] = mu_correlated[j] * point_1 + (1-mu_correlated[j]) * np.random.normal(mean_correlated_1[j], 1)
        else:
            point_0 = np.random.normal(mean_0, var_0)
            for j in range (number_of_correlated_features):
                x_correlated[i][j] = mu_correlated[j] * point_0 + (1-mu_correlated[j]) * np.random.normal(mean_correlated_0[j], 1)
        for k in range(number_of_important_features):
            x_important[i][k] = mu_important[k] * point_1 + (1-mu_correlated[k]) * np.random.normal(mean_imp_1[k], 1)


    for i in range (500):
        point_0 = np.random.normal(mean_0, var_0)
        z[500+i] = np.random.binomial(1, z0_prob)
        if z[500+i] == 0:
            for j in range (number_of_correlated_features):
                x_correlated[500+i][j] = mu_correlated[j] * point_0 + (1-mu_correlated[j]) * np.random.normal(mean_correlated_0[j], 1)
        else:
            point_1 = np.random.normal(mean_1, var_1)
            for j in range (number_of_correlated_features):
                x_correlated[500+i][j] = mu_correlated[j] * point_1 + (1-mu_correlated[j]) * np.random.normal(mean_correlated_1[j], 1)
        for k in range(number_of_important_features):
            x_important[500+i][k] = mu_important[k] * point_0 + (1-mu_correlated[k]) * np.random.normal(mean_imp_0[k], 1)
    x_unimportant =  np.random.multivariate_normal(uncorr_sample_mean, uncorr_sample_cov, size=total_sample)
    x = np.concatenate((x_correlated,x_important,x_uncorrelated),axis = 1)
    x = np.concatenate((x,np.reshape(z,(1000,1))),axis = 1)



    column = []
    for i in range(16):
        column.append(str(i+1))
    dataframe = pd.DataFrame(x, columns = column)



    #####protected_attribute,protected_value,protected_feature,fairness_metric
    

    model_dtree = d_tree.DecisionTree(15,0,'16',2)
    model_dtree.fit(dataframe,y)
    fairness_importance = model_dtree._fairness_importance()
    for key, value in fairness_importance.items():
        print(key, value)
        score[int(key)-1]=value
for i in range (16):
    score[i] /= 10
utils.draw_plot(feature,score,"Results/Synthetic/eqop1.pdf")

# %%

positive_z0 = positive_z1 = 0
count0 = count1 = 0
total_z0 = total_z1 = 0
for i in range (1000):
    if y[i] == 0:
        count0+=1
    else:
        count1+=1
        if z[i] == 0:
            positive_z0 += 1
        else:
            positive_z1 +=1
    if z[i] == 0:
        total_z0+=1
    else:
        total_z1+=1
print(count0,count1, positive_z0,positive_z1,total_z0,total_z1)

# %%
feature_importance = model_dtree._feature_importance()
# %%
feature = []
score = []
for key, value in feature_importance.items():
    print(key, value)
    feature.append(key)
    score.append((value))
utils.draw_plot(feature,score,"Results/Synthetic/eqop.pdf")

# %%
