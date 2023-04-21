import numpy as np
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

breast = arff.loadarff(r'breast.w.arff')
df = pd.DataFrame(breast[0])
df1 = pd.DataFrame(breast[0])
df.dropna(inplace=True)
df1.dropna(inplace=True)

df1.pop('Class')

n_bins = 4
ks = [1,3,5,9]
label = ["1","3","5","9"]
train_acc = []
test_acc = []
scores_test_mean = []
scores_train_mean = []
i = 0
df = df.replace(df['Class'][0], 0)
while df['Class'][i] == 0:
    i += 1
df = df.replace(df['Class'][i],1)


for a in ks:
    df_new = SelectKBest(mutual_info_classif,k=a).fit_transform(df1, df.Class)

    model = tree.DecisionTreeClassifier()
    scores = cross_validate(model, df_new, df.Class, cv=KFold(n_splits=10,random_state=27,shuffle=True), scoring='accuracy', return_train_score=True)
    train_acc = scores["train_score"]
    test_acc = scores["test_score"]
    scores_test_mean.append(test_acc.mean())
    scores_train_mean.append(train_acc.mean())

p1 = plt.bar(label,scores_train_mean)
p2 = plt.bar(label,scores_test_mean)
plt.title("train_acc vs test_acc")
plt.xlabel("number of features")
plt.ylabel("accuracy_score")
plt.legend((p1[0],p2[0]), ("train_acc","test_acc"))

plt.show()