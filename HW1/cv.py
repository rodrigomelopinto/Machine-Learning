from numpy.core.numeric import False_
from scipy.io import arff
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


breast = arff.loadarff(r'breast.w.arff')
df = pd.DataFrame(breast[0], copy = True)
df.dropna(inplace=True)
df = df.replace(df['Class'][0], 0)
i = 0
while df['Class'][i] == 0:
    i += 1
df = df.replace(df['Class'][i],1)
x = df[['Clump_Thickness','Cell_Size_Uniformity','Cell_Shape_Uniformity','Marginal_Adhesion','Single_Epi_Cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses']]
y = df['Class']

#cv=KFold(n_splits=10,random_state=27,shuffle=True)

knn = KNeighborsClassifier(n_neighbors=3, metric = 'euclidean', weights= 'uniform')

scores = cross_val_score(knn, x, y, cv=KFold(n_splits=10,random_state=27,shuffle=True), scoring='accuracy')

print(scores)
print("----")
knn = KNeighborsClassifier(n_neighbors=5, metric = 'euclidean', weights= 'uniform')

scores1 = cross_val_score(knn, x, y, cv=KFold(n_splits=10,random_state=27,shuffle=True), scoring='accuracy')
print(scores1)
print("----")

knn = KNeighborsClassifier(n_neighbors=7, metric = 'euclidean', weights= 'uniform')

scores2 = cross_val_score(knn, x, y, cv=KFold(n_splits=10,random_state=27,shuffle=True), scoring='accuracy')
print(scores2)
print("----")
print("3-"+str(scores.mean()))
print("5-"+str(scores1.mean()))
print("7-"+str(scores2.mean()))

        