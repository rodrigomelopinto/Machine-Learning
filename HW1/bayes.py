from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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

gb = GaussianNB()

gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test,y_pred))