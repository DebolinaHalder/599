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
    protected_negative = [(x,l,p) for (x,l,p) in zip(data,label, prediction) 
        if (x[protectedIndex] == protectedValue and l==0)] 
    el_negative = [(x,l,p) for (x,l,p) in zip(data,label, prediction) 
        if (x[protectedIndex] != protectedValue and l==0)]
    tp_protected = sum(1 for (x,l,p) in protected if l == p)
    
    tp_el = sum(1 for (x,l,p) in el if l == p)
    tn_protected = sum(1 for (x,l,p) in protected_negative if l == p)
    tn_el = sum(1 for (x,l,p) in el_negative if l == p)
   
   
    tpr_protected = tp_protected / len(protected) if len(protected) != 0 else 0
    tpr_el = tp_el / len(el) if len(el) != 0 else 0
    
    tnr_protected = tn_protected / len(protected_negative) if len(protected_negative)!= 0 else 0
    tnr_el = tn_el / len(el_negative) if len(el_negative)!= 0 else 0
    negative_rate = tnr_protected - tnr_el
    eqop = (tpr_el - tpr_protected)
    
    return abs(eqop) 
# %%

def DP(data, labels, prediction,protectedIndex, protectedValue):
    #print("changed")
    protectedClass = [(x,l) for (x,l) in zip(data, prediction) 
        if x[protectedIndex] == protectedValue]   
    elseClass = [(x,l) for (x,l) in zip(data, prediction) 
        if x[protectedIndex] != protectedValue]
    protectedProb = sum(1 for (x,l) in protectedClass if l == 1) / len(protectedClass) if len(protectedClass) != 0 else 0
    elseProb = sum(1 for (x,l) in elseClass  if l == 1) / len(elseClass) if len(elseClass) != 0 else 0
    return abs(elseProb - protectedProb)
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
def draw_plot(x,y,dest,name):
    sns.set_context("talk")
    fig = plt.figure(figsize = (10, 5))
    plt.bar(x, y)
    plt.xlabel("Feature")
    plt.ylabel(name)
    plt.savefig(dest)
    plt.show()

#%%
def fairness(leftX,lefty,rightX,righty,protected_attribute,protected_val,fairness_metric):
    print("probabilistic")
    valueLeft, countLeft = np.unique(lefty, return_counts=True)
    valueRight, countRight = np.unique(righty, return_counts=True)
    if len(countLeft) == 2:
        left0, left1 = countLeft[0]/len(lefty), countLeft[1]/len(lefty)
    if len(countRight) == 2:
        right0, right1 = countRight[0]/len(righty), countRight[1]/len(righty)
    if len(countLeft) == 1:
        left0 = countLeft[0]/len(lefty) if valueLeft[0] == 0 else 0
        left1 = countLeft[0]/len(lefty) if valueLeft[0] == 1 else 0
    if len(countRight) == 1:
        right0 = countRight[0]/len(righty) if valueRight[0] == 0 else 0
        right1 = countRight[0]/len(righty) if valueRight[0] == 1 else 0
    x = np.concatenate((leftX,rightX),axis=0)
    y = np.concatenate((lefty,righty),axis = 0)
    pred00 = np.concatenate((np.zeros(len(lefty)),np.zeros(len(righty))), axis = 0)
    pred01 = np.concatenate((np.zeros(len(lefty)),np.ones(len(righty))), axis = 0)
    pred10 = np.concatenate((np.ones(len(lefty)),np.zeros(len(righty))), axis = 0)
    pred11 = np.concatenate((np.ones(len(lefty)),np.ones(len(righty))), axis = 0)
    if fairness_metric == 1:
        fairness_score00 = eqop(x,y,pred00,protected_attribute,protected_val)
        fairness_score01 = eqop(x,y,pred01,protected_attribute,protected_val)
        fairness_score10 = eqop(x,y,pred10,protected_attribute,protected_val)
        fairness_score11 = eqop(x,y,pred11,protected_attribute,protected_val)
    else:
        fairness_score00 = DP(x,y,pred00,protected_attribute,protected_val)
        fairness_score01 = DP(x,y,pred01,protected_attribute,protected_val)
        fairness_score10 = DP(x,y,pred10,protected_attribute,protected_val)
        fairness_score11 = DP(x,y,pred11,protected_attribute,protected_val)
    fairness_score = fairness_score00*left0*right0 + fairness_score01*left0*right1
    +fairness_score10*left1*right0 + fairness_score11 * left1*right1
    return 1 - fairness_score


def print_tree(model_dtree):
    n_nodes = model_dtree.node_count
    children_left = model_dtree.children_left
    children_right = model_dtree.children_right
    feature = model_dtree.feature
    threshold = model_dtree.threshold
    impurity = model_dtree.impurity
    fair_score = model_dtree.fair_score
    samples = model_dtree.number_of_data_points
    z0 = model_dtree.countz0
    z1 = model_dtree.countz1
    pos_z0 = model_dtree.positivez0
    pos_z1 = model_dtree.positivez1
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            print(
                "{space}leafnode={node}\n{space} z0 {z0},z1 {z1}\n{space} pos_z0 {pos_z0},pos_z1 {pos_z1} \n{space}Fairness: {sample}".format(
                    space=node_depth[i] * "\t", node=i,z0 = z0[i],
                    z1=z1[i],
                    pos_z0 = pos_z0[i],
                    pos_z1 = pos_z1[i],
                    sample =fair_score[i]
                )
            )
        else:
            print(
                "{space}splitnode={node}\n{space} impurity {impurity}, \n{space} fairness {fairness} \n{space} samples {sample} \n{space} z0 {z0}, z1 {z1}\n{space} pos_z0 {pos_z0},pos_z1 {pos_z1}\n{space} split {left} if X[:,{feature}] <= {threshold} else {right}."
                .format(
                    space=node_depth[i] * "\t",
                    node=i,
                    impurity = impurity[i],
                    fairness = fair_score[i],
                    sample = samples[i],
                    z0 = z0[i],
                    z1=z1[i],
                    pos_z0 = pos_z0[i],
                    pos_z1 = pos_z1[i],
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i],
                )
            )
# %%
def previous_fairness(leftX,lefty,rightX,righty,protected_attribute,protected_val,fairness_metric):
    print("changed")
    valueLeft, countLeft = np.unique(lefty, return_counts=True)
    valueRight, countRight = np.unique(righty, return_counts=True)
    if len(countLeft) == 2:
        left0, left1 = countLeft[0]/len(lefty), countLeft[1]/len(lefty)
    if len(countRight) == 2:
        right0, right1 = countRight[0]/len(righty), countRight[1]/len(righty)
    if len(countLeft) == 1:
        left0 = countLeft[0]/len(lefty) if valueLeft[0] == 0 else 0
        left1 = countLeft[0]/len(lefty) if valueLeft[0] == 1 else 0
    if len(countRight) == 1:
        right0 = countRight[0]/len(righty) if valueRight[0] == 0 else 0
        right1 = countRight[0]/len(righty) if valueRight[0] == 1 else 0


    max_left = np.argmax(countLeft)
    max_right = np.argmax(countRight)
    if max_left == max_right:
        if len(countLeft == 1):
            max_right = ~max_right
        elif len(countRight == 1):
            max_left = ~max_left
        else:
            left_0,left_1 = countLeft[0]/len(lefty) , countLeft[1]/len(lefty)
            right_0,right_1 = countRight[0]/len(righty) , countRight[1]/len(righty)
            #print("same", max_left,max_right)
            if max_left == 0:
                if left_0 > right_0:
                    max_right = 1
                else:
                    max_left = 1
            else:
                if left_1 > right_1:
                    max_right = 0
                else:
                    max_left = 0
            #print("now", max_left,max_right)

    pred_left = np.full(len(lefty),max_left)
    pred_right = np.full(len(righty),max_right)
    x = np.concatenate((leftX,rightX),axis=0)
    y = np.concatenate((lefty,righty),axis = 0)
    Prediction = np.concatenate((pred_left,pred_right), axis = 0)
    if fairness_metric == 1:
        fairness_score = eqop(x,y,Prediction,protected_attribute,protected_val)
    else:
        fairness_score = DP(x,y,Prediction,protected_attribute,protected_val)
    return fairness_score

def fairness2(leftX,lefty,rightX,righty,protected_attribute,protected_val,fairness_metric):
    print("bernoulli")
    valueLeft, countLeft = np.unique(lefty, return_counts=True)
    valueRight, countRight = np.unique(righty, return_counts=True)
    if len(countLeft) == 2:
        left_1 = countLeft[1]/len(lefty)
    else:
        if valueLeft[0] == 1:
            left_1 = 1
        else:
            left_1 = 0
    if len(countRight) == 2:
        right_1 = countRight[1]/len(righty)
    else:
        if valueRight[0] == 1:
            right_1 = 1
        else:
            right_1 = 0
    pred_left = np.zeros(len(lefty))        
    pred_right = np.zeros(len(righty))
    for i in range (len(lefty)):
        pred_left[i] = np.random.binomial(1, left_1)
    for i in range (len(righty)):
        pred_right[i] = np.random.binomial(1, right_1)
    x = np.concatenate((leftX,rightX),axis=0)
    y = np.concatenate((lefty,righty),axis = 0)
    Prediction = np.concatenate((pred_left,pred_right), axis = 0)
    if fairness_metric == 1:
        fairness_score = eqop(x,y,Prediction,protected_attribute,protected_val)
    else:
        fairness_score = DP(x,y,Prediction,protected_attribute,protected_val)
    return 1 - fairness_score


def fairness_wrong(leftX,lefty,rightX,righty,protected_attribute,protected_val,fairness_metric):
    valueLeft, countLeft = np.unique(lefty, return_counts=True)
    valueRight, countRight = np.unique(righty, return_counts=True)
    max_left = np.argmax(countLeft)
    max_right = np.argmax(countRight)
    if max_left == max_right:
        if len(countLeft == 1):
            max_right = ~max_right
        elif len(countRight == 1):
            max_left = ~max_left
        else:
            left_0,left_1 = countLeft[0]/len(lefty) , countLeft[1]/len(lefty)
            right_0,right_1 = countRight[0]/len(righty) , countRight[1]/len(righty)
            #print("same", max_left,max_right)
            if max_left == 0:
                if left_0 > right_0:
                    max_right = 1
                else:
                    max_left = 1
            else:
                if left_1 > right_1:
                    max_right = 0
                else:
                    max_left = 0
            #print("now", max_left,max_right)

    pred_left = np.full(len(lefty),max_left )
    pred_right = np.full(len(righty),max_right)
    x = np.concatenate((leftX,rightX),axis=0)
    y = np.concatenate((lefty,righty),axis = 0)
    Prediction = np.concatenate((pred_left,pred_right), axis = 0)
    if fairness_metric == 1:
        fairness_score = eqop(x,y,Prediction,protected_attribute,protected_val)
    else:
        fairness_score = DP(x,y,Prediction,protected_attribute,protected_val)
    return fairness_score 


