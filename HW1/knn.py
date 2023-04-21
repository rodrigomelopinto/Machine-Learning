from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


breast = arff.loadarff(r'breast.w.arff')
df = pd.DataFrame(breast[0], copy = True)
df.dropna(inplace=True)
i = 0

df = df.replace(df['Class'][0], 0)
while df['Class'][i] == 0:
    i += 1
df = df.replace(df['Class'][i],1)

X_train, X_test, y_train, y_test = train_test_split(df, df.Class,test_size = 0.1, random_state = 27)


knn3 = KNeighborsClassifier(n_neighbors=3, metric = 'euclidean', weights= 'uniform')
knn5 = KNeighborsClassifier(n_neighbors=5, metric = 'euclidean', weights= 'uniform')
knn7 = KNeighborsClassifier(n_neighbors=7, metric = 'euclidean', weights= 'uniform')

knn3.fit(X_train, y_train)
knn5.fit(X_train, y_train)
knn7.fit(X_train, y_train)
y_pred3 = knn3.predict(X_test)
y_pred5 = knn5.predict(X_test)
y_pred7 = knn7.predict(X_test)

print("Accuracy(k=3):",metrics.accuracy_score(y_test,y_pred3))
print("Accuracy(k=5):",metrics.accuracy_score(y_test,y_pred5))
print("Accuracy(k=7):",metrics.accuracy_score(y_test,y_pred7))