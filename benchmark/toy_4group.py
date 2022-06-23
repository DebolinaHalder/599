#%%
import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr
import pandas as pd
from fairforest import d_tree, RandomForest
from fairforest import utils
import math
from sklearn.ensemble import RandomForestClassifier




#%%
def select_beta(elements_per_group):
    np.random.seed(1000)
    beta = np.zeros(elements_per_group*4+1)
    possibilities = [5,6,7,8,-8,-5,-6,-7]
    for i in range(elements_per_group):
        beta[i] = np.random.uniform(4,8)
    for i in range(elements_per_group*2,elements_per_group*3):
        beta[i] = np.random.choice(possibilities)
    #beta[elements_per_group*4] = 20
    return beta
#%%
#parameter a ,b


def toy_4group(a1,b2,mu3, mu4,elements_per_group, total_samples,z_prob):
    total_features = elements_per_group*4 + 1
    z = np.random.binomial(1,z_prob,total_samples)
    g1 = np.zeros((elements_per_group,total_samples))
    g2 = np.zeros((elements_per_group,total_samples))
    g3 = np.zeros((elements_per_group,total_samples))
    g4 = np.zeros((elements_per_group,total_samples))
    
    for i in range(elements_per_group):
        for j in range(total_samples):
            g1[i][j] = np.random.normal(a1[i]*z[j],1) + np.random.normal(0,1)
            g2[i][j] = np.random.normal(b2[i]*z[j],1) + np.random.normal(0,1)
        g3[i] = np.random.normal(mu3[i],1,total_samples)
        g4[i] = np.random.normal(mu4[i],1,total_samples)
    
    
    x = np.concatenate((np.transpose(g1),np.transpose(g2),np.transpose(g3),np.transpose(g4)),axis = 1)
    x = np.concatenate((x,np.reshape(z,(-1,1))),axis = 1)

    beta = select_beta(elements_per_group)
    mu = np.matmul(x,beta) + np.random.normal(0,1,total_samples)
    gama = expit(mu)
    y = np.zeros(total_samples)
    for i in range(total_samples):
        y[i] = np.random.binomial(1,gama[i])
    column = []
    for i in range(total_features):
        column.append(str(i+1))
        
    dataframe = pd.DataFrame(x, columns = column)
    return dataframe,y,beta

# %%
elements_per_group = 2
iterations = 2
total_features = elements_per_group * 4 + 1
a1 = np.array([1,2])
b2 = np.array([3,4])
mu3 = np.array([3,5])
mu4 = np.array([2,-2])
dataframe , y, beta = toy_4group(a1,b2,mu3,mu4,elements_per_group,4000,0.7)
score_fairness = np.zeros(total_features)
score_feature = np.zeros(total_features)
result_df = pd.DataFrame(columns=['fairness','occlusion','accuracy'])
occlusion_fairness = np.zeros(total_features)

for i in range(iterations):
    feature = np.arange(total_features)
    model_dtree = RandomForest(total_features-1,0,str(total_features),1,50,math.sqrt(2))
    model_dtree.fit(dataframe,y)
    fairness_importance = model_dtree._fairness_importance()
    feature_importance = model_dtree._feature_importance()
    for key, value in fairness_importance.items():
        score_fairness[int(key)-1] += value
    for key, value in feature_importance.items():
        score_feature[int(key)-1] += value


    sklearn_dataframe = dataframe.copy().drop(columns=[str(total_features)])
    testX,testy = toy_4group(a1,b2,mu3,mu4,elements_per_group,1000,0.7)
    testX_without_protected_all = testX.copy().drop(columns=str(total_features))
    sklearn_tree_all = RandomForestClassifier(n_estimators=50)
    sklearn_tree_all.fit(dataframe,y)
    pred_all = sklearn_tree_all.predict(testX_without_protected_all)
    fairness_all = 1 - utils.eqop(testX.to_numpy(),testy,pred_all,total_features-1,0)
    for j in range (total_features - 1):
        #print(j+1)
        train_data = sklearn_dataframe.copy().drop(columns=[str(j+1)])
        sklearn_tree= RandomForestClassifier(n_estimators=50)
        sklearn_tree.fit(train_data,y)
        
        testX_without_protected = testX.copy().drop(columns=[str(total_feature),str(j+1)])
        prediction = sklearn_tree.predict(testX_without_protected)
        occlusion_fairness[j] += fairness_all - (1 - utils.eqop(testX.to_numpy(),testy,prediction,total_feature-1,0))

    

for p in range (total_features):
    score_fairness[p] /= iterations
    score_feature[p] /= iterations
    occlusion_fairness[p] /= iterations
    result_df = result_df.append({'fairness':score_fairness[p],'occlusion':occlusion_fairness[p],'accuracy':score_feature[p]}, ignore_index=True)
result_df.to_csv("Results/synthetic_gaussian/results_g_eqop_tree_1.csv")

# %%
utils.draw_plot(feature,occlusion_fairness,"Results/synthetic_gaussian/eqop_fairness_occlusion_test_g_tree.pdf","Fairness")
utils.draw_plot(feature,score_fairness,"test1.pdf","Fairness Importance")
utils.draw_plot(feature,score_feature,"test2.pdf","Accuracy Importance")
# %%
