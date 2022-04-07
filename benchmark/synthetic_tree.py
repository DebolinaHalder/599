#%%
import numpy as np
import pandas as pd
from scipy.special import logit
from fairforest import d_tree, random_tree
from fairforest import utils
import warnings
import matplotlib.pyplot as plt

#%%
warnings.simplefilter("ignore")

#%%
#np.random.seed(10)

#%%
def build_dataset(total_sample,number_of_correlated_features,number_of_important_features,number_of_uncorrelated_features,mean_correlated_0,mean_correlated_1,mean_imp_0,mean_imp_1):
    mid = int(total_sample/2)
    z0_prob = 0.4
    z1_prob = 0.8
    z = np.ones(total_sample)
    x_correlated = np.zeros((total_sample,number_of_correlated_features))
    x_important = np.zeros((total_sample,number_of_important_features))
    
    var_0 = var_1 = 1
    
    y = np.zeros(total_sample)
    for i in range (mid):
        y[i] = 1
        z[i] = np.random.binomial(1, z1_prob)
        if z[i] == 1:
            for j in range (number_of_correlated_features):
                x_correlated[i][j] = np.random.normal(mean_correlated_1[j], 1)
        else:
            for j in range (number_of_correlated_features):
                x_correlated[i][j] = np.random.normal(mean_correlated_0[j], 1)
        for k in range(number_of_important_features):
            x_important[i][k] = np.random.normal(mean_imp_1[k], 1)


    for i in range (mid):
        z[mid+i] = np.random.binomial(1, z0_prob)
        if z[mid+i] == 0:
            for j in range (number_of_correlated_features):
                x_correlated[mid+i][j] =  np.random.normal(mean_correlated_0[j], 1)
        else:
            for j in range (number_of_correlated_features):
                x_correlated[mid+i][j] = np.random.normal(mean_correlated_1[j], 1)
        for k in range(number_of_important_features):
            x_important[mid+i][k] = np.random.normal(mean_imp_0[k], 1)
    #x_unimportant =  np.random.multivariate_normal(uncorr_sample_mean, uncorr_sample_cov, size=total_sample)
    x = np.concatenate((x_correlated,x_important),axis = 1)
    x = np.concatenate((x,np.reshape(z,(-1,1))),axis = 1)
    np.random.shuffle(x)

    total_features = number_of_correlated_features+number_of_important_features+1
    column = []
    for i in range(total_features):
        column.append(str(i+1))
    print(column)
    dataframe = pd.DataFrame(x, columns = column)
    return dataframe,y
#%%
total_sample = 1000
number_of_correlated_features = 2
number_of_important_features = 2
number_of_uncorrelated_features = 0
total_feature = number_of_important_features+number_of_correlated_features+1
feature = np.arange(number_of_important_features+number_of_correlated_features+1)
score_fairness = np.zeros(number_of_important_features+number_of_correlated_features+1)
score_feature = np.zeros(number_of_important_features+number_of_correlated_features+1)
mean_correlated_1 = [11, 20]
mean_correlated_0 = [-4, -10]
mean_imp_1 = [50,80]
mean_imp_0 = [100,140]

for i in range (5):
    dataframe, y = build_dataset(total_sample,number_of_correlated_features,number_of_important_features,number_of_uncorrelated_features,mean_correlated_0,mean_correlated_1,mean_imp_0,mean_imp_1)
    #####protected_attribute,protected_value,protected_feature,fairness_metric
    model_dtree = d_tree.DecisionTree(4,0,'5',1)
    model_dtree.fit(dataframe,y)
    fairness_importance = model_dtree._fairness_importance()
    feature_importance = model_dtree._feature_importance()
    for key, value in fairness_importance.items():
        score_fairness[int(key)-1]=value
    for key, value in feature_importance.items():
        score_feature[int(key)-1]=value
for i in range (total_feature):
    score_fairness[i] /= 5
    score_feature[i] /= 5

with open('Results/Synthetic/no_corr/result_eqop_2_2_f.txt', 'a') as f:
    f.writelines(str(score_fairness))
    f.writelines("\n")
    f.writelines(str(score_feature))
#%%
utils.draw_plot(feature,score_fairness,"Results/Synthetic/no_corr/eqop_fairness_2_2_f.pdf","Fairness Importance")
utils.draw_plot(feature,score_feature,"Results/Synthetic/no_corr/eqop_feature_2_2_f.pdf","Accuracy Importance")





#%%

# %%



# %%
