#!/usr/bin/env python
# coding: utf-8

# En este caso se busca desarrollar un programa capaz de clasificar digitos escritos a mano. Para ello se trabaja con una base de datos brindada por scikit learn.

# In[1]:


import numpy as np
#import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[2]:


# Se abre la base de datos, se transforman los targets a enteros y se ordena la base en funcion a estos

mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
mnist.target = mnist.target.astype(np.int8)
mnist["data"], mnist["target"]


# In[3]:


# Se consulta el tamaño del data set
mnist.data.shape


# In[4]:


X, y = mnist["data"], mnist["target"]


# In[5]:


# Se grafica un digito

some_digit = X[3600]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary,
           interpolation="nearest")
plt.axis("off")

#save_fig("some_digit_plot")
plt.show()


# In[6]:


# Se divide en un set de entrenamiento y otro de testeo

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# In[7]:


# Se entrenan algoritmos capaces de clasificar bases de datos con multiples respuestas o targets

sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
sgd_clf.fit(X_train, y_train)

ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, tol=-np.infty, random_state=42))
ovo_clf.fit(X_train, y_train)

forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)
forest_clf.fit(X_train, y_train)


# In[8]:


# Se evalua la performance de los clasificadores

cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")


# In[9]:


cross_val_score(ovo_clf, X_train, y_train, cv=3, scoring="accuracy")


# In[10]:


cross_val_score(forest_clf, X_train, y_train, cv=3, scoring="accuracy")


# In[11]:


# Se escalan los datos y se evalua si mejora la performance

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))


# In[12]:


cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


# In[13]:


cross_val_score(ovo_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


# In[14]:


cross_val_score(forest_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


# In[15]:


# Se grafican las matrices de confusión
def plot_confusion_matrix(matrix):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)

y_train_pred_sgd = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx_sgd = confusion_matrix(y_train, y_train_pred_sgd)
conf_mx_sgd


# In[16]:


plt.matshow(conf_mx_sgd, cmap=plt.cm.gray)
plt.show()


# In[17]:


y_train_pred_ovo = cross_val_predict(ovo_clf, X_train_scaled, y_train, cv=3)
conf_mx_ovo = confusion_matrix(y_train, y_train_pred_ovo)
conf_mx_ovo


# In[18]:


plt.matshow(conf_mx_ovo, cmap=plt.cm.gray)
plt.show()


# In[19]:


y_train_pred_forest = cross_val_predict(forest_clf, X_train_scaled, y_train, cv=3)
conf_mx_forest = confusion_matrix(y_train, y_train_pred_forest)
conf_mx_forest


# In[20]:


plt.matshow(conf_mx_forest, cmap=plt.cm.gray)
plt.show()


# In[21]:


# Se normalizan las matrices de confusion y se llena la diagonal de ceros para evaluar que numero se esta confundiendo el algoritmo

row_sums_sgd = conf_mx_sgd.sum(axis=1, keepdims=True)
norm_conf_mx_sgd = conf_mx_sgd / row_sums_sgd
np.fill_diagonal(norm_conf_mx_sgd, 0)
plt.matshow(norm_conf_mx_sgd, cmap=plt.cm.gray)
plt.show()


# In[22]:


row_sums_ovo = conf_mx_ovo.sum(axis=1, keepdims=True)
norm_conf_mx_ovo = conf_mx_ovo / row_sums_ovo
np.fill_diagonal(norm_conf_mx_ovo, 0)
plt.matshow(norm_conf_mx_ovo, cmap=plt.cm.gray)
plt.show()


# In[23]:


row_sums_forest = conf_mx_forest.sum(axis=1, keepdims=True)
norm_conf_mx_forest = conf_mx_forest / row_sums_forest
np.fill_diagonal(norm_conf_mx_forest, 0)
plt.matshow(norm_conf_mx_forest, cmap=plt.cm.gray)
plt.show()


# In[24]:


# Se evalua la precision de cada algoritmo

y_test_pred_sgd = sgd_clf.predict(X_test)
sgd_score = accuracy_score(y_test, y_test_pred_sgd)

y_test_pred_ovo = ovo_clf.predict(X_test)
ovo_score = accuracy_score(y_test, y_test_pred_ovo)

y_test_pred_forest = forest_clf.predict(X_test)
forest_score = accuracy_score(y_test, y_test_pred_forest)


d = {'Algorithm': ['SGD', 'OVO', 'Random Forest'], 'Score': [sgd_score, ovo_score, forest_score]}
final_df = pd.DataFrame(data=d)

score = final_df['Score']
final_df.drop('Score', axis=1, inplace=True)
final_df.insert(1, 'Score', score)

final_df


# Los tres algoritmos presentan un puntaje elevado. Se podria mejorar el resultado mediante el preprocesamiento (ej. escalado) de los datos o haciendo hincapie en los numeros que el programa confunde con mayor frecuencia (ej. 3-8 y 7-9). Esto ultimo se alcanzaria si agrega una porcion de programa capaz de identificar el numero de vueltas cerradas, ya que el digito 3 no posee ninguna, 8 posee dos, 7 no posee ninguna y 9 posee solo una.
