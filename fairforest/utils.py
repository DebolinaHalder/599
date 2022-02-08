#%%
import numpy as np
import pandas as pd
from collections import Counter



# %%
def DP(data, labels, protectedIndex, protectedValue):
    protectedClass = [(x,l) for (x,l) in zip(data, labels) 
        if x[protectedIndex] == protectedValue]   
    elseClass = [(x,l) for (x,l) in zip(data, labels) 
        if x[protectedIndex] != protectedValue]
 
    if len(protectedClass) == 0 or len(elseClass) == 0:
      raise Exception("One of the classes is empty!")
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
