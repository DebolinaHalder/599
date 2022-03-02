#%%
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def eqop(data,label, prediction, protectedIndex, protectedValue):
    protected = [(x,l,p) for (x,l,p) in zip(data,label, prediction) 
        if (x[protectedIndex] == protectedValue and l==1)]   
    el = [(x,l,p) for (x,l,p) in zip(data,label, prediction) 
        if (x[protectedIndex] != protectedValue and l==1)]
    tp_protected = sum(1 for (x,l,p) in protected if l == p)
    
    tp_el = sum(1 for (x,l,p) in el if l == p)
    if len(protected) != 0 and len(el) != 0:
        tpr_protected = tp_protected / len(protected)
        tpr_el = tp_el / len(el)
    elif len(protected) == 0 and len(el) == 0:
        tpr_protected = 0
        tpr_el = 0
    elif len(el) == 0:
        tpr_protected = tp_protected / len(protected)
        tpr_el = 0
    else:
        tpr_protected = 0
        tpr_el = tp_el / len(el)
    
    return tpr_el - tpr_protected
# %%
def DP(data, labels, prediction, protectedIndex, protectedValue):
    protectedClass = [(x,l) for (x,l) in zip(data, labels) 
        if x[protectedIndex] == protectedValue]   
    elseClass = [(x,l) for (x,l) in zip(data, labels) 
        if x[protectedIndex] != protectedValue]
 
    if len(protectedClass) == 0:
        protectedProb = 0
        elseProb = sum(1 for (x,l) in elseClass  if l == 1) / len(elseClass)
    elif len(elseClass) == 0:
        elseProb = 0
        protectedProb = sum(1 for (x,l) in protectedClass if l == 1) / len(protectedClass)
    else:
      protectedProb = sum(1 for (x,l) in protectedClass if l == 1) / len(protectedClass)
      elseProb = sum(1 for (x,l) in elseClass  if l == 1) / len(elseClass)
    return elseProb - protectedProb
#%%
def gini(y):
    total_classes, count = np.unique(y, return_counts=True)
    probability = np.zeros(len(total_classes), dtype=float)
    n = len(y)
    for i in range(len(total_classes)):
        probability[i] = (count[i]/n)**2
    if n == 0:
        return 0.0
    gini = 1 - np.sum(probability)
    return gini

# %%
def ma(x, window):
        return np.convolve(x, np.ones(window), 'valid') / window

#%%
def draw_plot(x,y,dest):
    sns.set_context("talk")
    fig = plt.figure(figsize = (10, 5))
    plt.bar(x, y)
    plt.xlabel("Feature")
    plt.ylabel("Fairness Score")
    plt.savefig(dest)
    plt.show()

#%%
def fairness(leftX,lefty,rightX,righty,protected_attribute,protected_val,fairness_metric):
    valueLeft, countLeft = np.unique(lefty, return_counts=True)
    valueRight, countRight = np.unique(righty, return_counts=True)
    pred_left = np.full(len(lefty), np.argmax(countLeft))
    pred_right = np.full(len(righty), np.argmax(countRight))
    x = np.concatenate((leftX,rightX),axis=0)
    y = np.concatenate((lefty,lefty),axis = 0)
    Prediction = np.concatenate((pred_left,pred_right), axis = 0)
    if fairness_metric == 1:
        fairness_score = eqop(x,y,Prediction,protected_attribute,protected_val)
    else:
        fairness_score = DP(x,y,Prediction,protected_attribute,protected_val)
    return fairness_score