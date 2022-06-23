#%%

import pandas as pd
import numpy as np
from tempeh.configurations import datasets
from fairforest import d_tree,RandomForest
from sklearn import tree, ensemble
from fairforest import utils
import math
from sklearn import tree
from cProfile import label
from turtle import color
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%%
compas_dataset = datasets["compas"]()
X_train, X_test = compas_dataset.get_X(format=pd.DataFrame)
y_train, y_test = compas_dataset.get_y(format=pd.Series)
(
    sensitive_features_train,
    sensitive_features_test,
) = compas_dataset.get_sensitive_features("race", format=pd.Series)
X_df=pd.concat([X_train,sensitive_features_train],axis=1)
X_df = X_df.replace("Caucasian",0)
X_df = X_df.replace("African-American",1)
test_df = pd.concat([X_test,sensitive_features_test],axis=1)
test_df = test_df.replace("Caucasian",0)
test_df = test_df.replace("African-American",1)


#%%
dtree = tree.DecisionTreeClassifier(max_depth=5)
dtree.fit(X_df,y_train)
plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(dtree, fontsize=10)

# %%
y = y_train.to_numpy()
X = X_df.to_numpy()
total_feature = len(X_df.columns)
column = []
for i in range(total_feature):
    column.append(str(i+1))
#print(column)
dataframe = pd.DataFrame(X, columns = column)
dataframe_test = pd.DataFrame(test_df.to_numpy(),columns=column)
testy = y_test.to_numpy()
# %%
feature = np.arange(total_feature)
score_fairness = np.zeros(total_feature)
occlusion_fairness = np.zeros(total_feature)
score_feature = np.zeros(total_feature)
result_df = pd.DataFrame(columns=['fairness','occlusion','accuracy'])
iteration = 5
for i in range (iteration):
    occusion = fairness = feature_imp = np.zeros(total_feature)
    #dataframe, y = build_dataset(total_sample,number_of_correlated_features,number_of_important_features,number_of_uncorrelated_features,mean_correlated_0,mean_correlated_1,mean_imp_0,mean_imp_1)
    #####protected_attribute,protected_value,protected_feature,fairness_metric
    model_dtree = RandomForest(10,1,'11',2,50,math.sqrt(2))
    model_dtree.fit(dataframe.copy(),y)
    sklearn_dataframe = dataframe.copy().drop(columns=[str(total_feature)])
    
    #testX,testy = build_dataset(5000,number_of_correlated_features,number_of_important_features,number_of_uncorrelated_features,mean_correlated_0,mean_correlated_1,mean_imp_0,mean_imp_1)
    testX_without_protected_all = dataframe_test.copy().drop(columns=str(total_feature))
    sklearn_tree_all = ensemble.RandomForestClassifier(n_estimators=50)
    sklearn_tree_all.fit(sklearn_dataframe,y)
    pred_all = sklearn_tree_all.predict(testX_without_protected_all)
    fairness_all = 1 - utils.DP(dataframe_test.to_numpy(),testy,pred_all,total_feature-1,0)
    for j in range (total_feature - 1):
        #print(j+1)
        train_data = sklearn_dataframe.copy().drop(columns=[str(j+1)])
        sklearn_tree= ensemble.RandomForestClassifier(n_estimators=50)
        sklearn_tree.fit(train_data,y)
        
        testX_without_protected = dataframe_test.copy().drop(columns=[str(total_feature),str(j+1)])
        prediction = sklearn_tree.predict(testX_without_protected)
        occusion[j] = fairness_all - (1 - utils.DP(dataframe_test.to_numpy(),testy,prediction,total_feature-1,0))
        occlusion_fairness[j] += occusion[j]
    fairness_importance = model_dtree._fairness_importance()
    feature_importance = model_dtree._feature_importance()
    for key, value in fairness_importance.items():
        fairness[int(key)-1] = value
        score_fairness[int(key)-1] += value
    for key, value in feature_importance.items():
        feature_imp[int(key)-1] = value
        score_feature[int(key)-1] += value
    #for p in range(total_feature):
        #result_df = result_df.append({'fairness':fairness[p],'occlusion':occusion[p],'accuracy':feature_imp[p]}, ignore_index=True)
        #print(result_df.head())
for p in range (total_feature):
    score_fairness[p] /= iteration
    score_feature[p] /= iteration
    occlusion_fairness[p] /= iteration
    result_df = result_df.append({'fairness':score_fairness[p],'occlusion':occlusion_fairness[p],'accuracy':score_feature[p]}, ignore_index=True)
result_df.to_csv("Results/synthetic_gaussian/results_compas_DP_.csv")

# %%
utils.draw_plot(feature,score_fairness,"Results/synthetic_gaussian/eqop_fairness_compas.pdf","Fairness Importance")
utils.draw_plot(feature,occlusion_fairness,"Results/synthetic_gaussian/eqop_fairness_occlusion_compas.pdf","Fairness")
utils.draw_plot(feature,score_feature,"Results/synthetic_gaussian/eqop_feature_compas.pdf","Accuracy Importance")
# %%


#%%

sns.set_context("talk")
fig, axs = plt.subplots(2,figsize=(12,20))
df1 = pd.read_csv("Results/synthetic_gaussian/results_compas_DP.csv")
df2 = pd.read_csv("Results/synthetic_gaussian/results_compas_eqop.csv")
df1 = df1[:-1]
df2 = df2[:-1]
x1 = df1['fairness'].copy()
x2 = df2['fairness'].copy()
y = X_train.columns
axs[0].bar(y, x1, color = 'r',label = "DP")
axs[0].set_xticks([])
axs[0].set_ylabel("Fairness Importance Score with DP")
axs[0].title.set_text("(A)")
axs[0].set_xlabel("feature")
axs[0].legend(loc='lower right')
axs[1].set_ylabel("Fairness Importance Score with EQOP")
axs[1].bar(y, x2, color = 'b', label = "EQOP")
axs[1].set_xlabel("feature")
axs[1].set_xticklabels(y, rotation=45, ha='right')
axs[1].title.set_text("(B)")
axs[1].legend(loc='lower right')

plt.savefig("Results/synthetic_gaussian/eqop_tree_compas.pdf")
plt.show()
# %%
sns.set_context("talk")
fig, ax = plt.subplots(1,figsize=(12,7.5))
df1 = pd.read_csv("Results/synthetic_gaussian/results_compas_DP.csv")
df1 = df1[:-1]
y = X_train.columns
x1 = df1['accuracy'].copy()
ax.set_ylabel("Accuracy Importance")
ax.bar(y, x1, color = 'g')
ax.set_xlabel("feature")
ax.set_xticklabels(y, rotation=20, ha='right')
#ax.title.set_text("(B)")
ax.legend(loc='lower right')
plt.savefig("Results/synthetic_gaussian/eqop_tree_compas_feature.pdf")
plt.show()


# %%
sns.set_context("talk")
width = 0.3
fig, axs = plt.subplots(1,figsize=(16,10))
df = pd.read_csv("Results/synthetic_gaussian/results_compas_eqop.csv")
df = df[:-1]
x1 = df['fairness'].copy()
x2 = df['occlusion'].copy()
#x3 = df['accuracy'].copy()
y = X_train.columns
y_temp = np.arange(1,11)
axs.bar(y_temp-width,  x1,width=width, color = 'r',label = "Fairness_importance_score")
axs.set_ylabel("Importance Score")
#axs[0].title.set_text("(A) Fairness Feature Importance Score")
axs.set_xlabel("Feature")
#axs[0].legend(loc='lower right')
axs.bar(y_temp, x2, width = width, color = 'b', label = "Fairness Occlusion Score")
#axs[0].set_xticks([])
#axs[1].set_ylabel("Occlusion Fairness Importance Score")
#axs[1].set_xlabel("feature")
#axs[1].title.set_text("(B)Occlusion Fairness Feature Importance Score")
#axs[1].legend(loc='lower right')
#axs[1].set_xticks([])
#axs[2].bar(y, x3, color = 'g', label = "Accuracy Score")
#axs[2].set_ylabel("Accuracy Importance Score")
#axs[2].set_xlabel("feature")
#axs[2].title.set_text("(C)Accuracy Feature Importance Score")
axs.legend(loc='lower right')
axs.set_xticks(list(range(1,11)))
axs.set_xticklabels(y, rotation=30, ha='right')
plt.savefig("Results/synthetic_gaussian/eqop_compas_all_1.pdf")
plt.show()
# %%
