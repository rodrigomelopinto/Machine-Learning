import numpy as np
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

breast = arff.loadarff(r'breast.w.arff')
df = pd.DataFrame(breast[0])
df1 = pd.DataFrame(breast[0])
df.dropna(inplace=True)
df1.dropna(inplace=True)

df1.pop('Class')

ks = [2,3]
ecr = []
silhouette = []
j = 0
c1 = 0
c1_b = 0
c1_m = 0
c2 = 0
c2_b = 0
c2_m = 0
c3 = 0
c3_b = 0
c3_m = 0

i = 0
df = df.replace(df['Class'][0], 0)
while df['Class'][i] == 0:
    i += 1
df = df.replace(df['Class'][i],1)


def max(a,b):
    if a >= b:
        return a
    else:
        return b

df_new = SelectKBest(mutual_info_classif,k=2).fit_transform(df1, df.Class)


for a in ks:
    kmeans = KMeans(n_clusters=a,random_state=0)
    label = kmeans.fit_predict(df_new)
    print(label)
    for j in range(len(label)):
        if label[j] == 0:
            c1 += 1
            if df['Class'].iloc[j] == 0:
                c1_b += 1
            else:
                c1_m += 1
        if label[j] == 1:
            c2 += 1
            if df['Class'].iloc[j] == 0:
                c2_b += 1
            else:
                c2_m += 1
        if label[j] == 2:
            c3 += 1
            if df['Class'].iloc[j] == 0:
                c3_b += 1
            else:
                c3_m += 1
    if a == 2:
        val = (1/a)*((c1 - max(c1_b,c1_m)) + (c2 - max(c2_b,c2_m)))
        ecr.append(val)
    if a == 3:
        val = (1/a)*((c1 - max(c1_b,c1_m)) + (c2 - max(c2_b,c2_m)) + (c3 - max(c3_b,c3_m)))
        ecr.append(val)
    score2 = silhouette_score(df,kmeans.labels_)
    silhouette.append(score2)
print(ecr)
print(silhouette)

#df_new = SelectKBest(mutual_info_classif,k=2).fit_transform(df1, df.Class)
model = KMeans(n_clusters=3,random_state=0)
label = model.fit_predict(df_new)
centers = model.cluster_centers_


cluster1 = plt.scatter(df_new[label == 0,0], df_new[label == 0,1], color = 'red')
cluster2 = plt.scatter(df_new[label == 1,0], df_new[label == 1,1], color = 'green')
cluster3 = plt.scatter(df_new[label == 2,0], df_new[label == 2,1], color = 'blue')
plt.legend((cluster1,cluster2,cluster3),("cluster1","cluster2","cluster3"))
plt.show()