import numpy as np
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

k8 = arff.loadarff(r'kin8nm.arff')
df = pd.DataFrame(k8[0])
df1 = pd.DataFrame(k8[0])
df.dropna(inplace=True)
df1.dropna(inplace=True)

df1.pop('y')

residuos = []
residuos2 = []

model = MLPRegressor(hidden_layer_sizes=(3,2),alpha=0.1)
model2 = MLPRegressor(hidden_layer_sizes=(3,2),alpha=0)
y_pred = cross_val_predict(model,df1,df.y,cv=KFold(n_splits=5,random_state=0,shuffle=True))
y_pred2 = cross_val_predict(model2,df1,df.y,cv=KFold(n_splits=5,random_state=0,shuffle=True))

i = 0
for i in range(len(y_pred)):
    residuos.append((df['y'][i]-y_pred[i]))
    residuos2.append((df['y'][i]-y_pred2[i]))

fig, ax0 = plt.subplots()
ax0.boxplot([residuos,residuos2])
plt.title("Boxplot of residues")
plt.xticks([1,2],["regularization","no_regularization"])
plt.ylabel("residues")
plt.show()