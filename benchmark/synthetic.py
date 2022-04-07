#%%
from cProfile import label
from cv2 import normalize
import numpy as np
import pandas as pd
from scipy.special import logit
from fairforest import d_tree
from fairforest import utils
import warnings
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
#%%
warnings.simplefilter("ignore")

#%%
np.random.seed(0)
#%%
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

#%%
z = np.zeros(20)
for j in range(20):
    z[j] = np.random.binomial(1,0.7)
x_correlated = np.zeros((20,2))
x_uncorrelated = np.zeros((20,2))
for j in range(2):
    for i in range (20):
        x_correlated[i][j] = np.random.normal((z[i] + 1), 1, 1)
        x_uncorrelated[i][j] = np.random.normal(0,1,1)
x = np.concatenate((x_correlated,x_uncorrelated),axis=1)
x = np.concatenate((x,np.reshape(z,(20,1))),axis=1)
b = np.zeros(5)
noise = np.random.normal(0,1,20)
for i in range (5):
    b[i] = np.random.normal(5,0.1,1)
y = logit(NormalizeData(np.dot(x,b)) + noise.T)
for i in range (len(y)):
    if y[i] > 0:
        y[i] = int(1)
    else:
        y[i] = int(0)
column = []
for i in range(5):
    column.append(str(i+1))
dataframe = pd.DataFrame(x, columns = column)

#%%
def print_tree(model_dtree):
    n_nodes = model_dtree.node_count
    children_left = model_dtree.children_left
    children_right = model_dtree.children_right
    feature = model_dtree.feature
    threshold = model_dtree.threshold

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
                "{space}node={node} is a leaf node.".format(
                    space=node_depth[i] * "\t", node=i
                )
            )
        else:
            print(
                "{space}node={node} is a split node: "
                "go to node {left} if X[:, {feature}] <= {threshold} "
                "else to node {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i],
                )
            )

# %%
model_dtree = d_tree.DecisionTree(4,0,'5',1)
model_dtree.fit(dataframe,y)
print_tree(model_dtree)

#%%
x = np.delete(x, 4, 1)
sklearn_tree = DecisionTreeClassifier(random_state=0)
sklearn_tree.fit(x,y)
print_tree(sklearn_tree.tree_)


#%%
feature_importance_fairness = model_dtree._feature_importance()
feature_importance_sklearn = sklearn_tree.tree_.compute_feature_importances(normalize = False)
feature = []
score_dp = []
for key, value in feature_importance_fairness.items():
    if key != "5":
        feature.append(int(key) - 1)
        score_dp.append((value))
#%%
print(feature_importance_sklearn)
print(score_dp)
sns.set_context("talk")
plt.bar(feature-0.2,score_dp,label="fair tree",alpha = 0.4)
plt.bar(feature+0.2,feature_importance_sklearn,label="sklearn tree", alpha = 0.4)
plt.xlabel("Feature")
plt.ylabel("Importance score")
plt.legend()

plt.savefig("Results/Synthetic/somparison.pdf")
plt.show()


# %%
