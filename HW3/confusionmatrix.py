import numpy as np
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

breast = arff.loadarff(r'breast.w.arff')
df = pd.DataFrame(breast[0])
df1 = pd.DataFrame(breast[0])
df.dropna(inplace=True)
df1.dropna(inplace=True)

df1.pop('Class')

i = 0
df = df.replace(df['Class'][0], 0)
while df['Class'][i] == 0:
    i += 1
df = df.replace(df['Class'][i],1)

model = MLPClassifier(hidden_layer_sizes=(3,2),alpha=0.1)
model2 = MLPClassifier(hidden_layer_sizes=(3,2),early_stopping=True, alpha=0.1)
y_pred = cross_val_predict(model,df1,df.Class,cv=KFold(n_splits=5,random_state=0,shuffle=True))
y_pred2 = cross_val_predict(model2,df1,df.Class,cv=KFold(n_splits=5,random_state=0,shuffle=True))
print(confusion_matrix(df.Class,y_pred))
print(confusion_matrix(df.Class,y_pred2))