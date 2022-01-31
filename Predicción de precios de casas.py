#!/usr/bin/env python
# coding: utf-8

# Se entrenara un programa capaz de predecir el valor de una casa

# In[1]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer


# In[2]:


# Se carga la base de datos y se realiza un analisis preliminar

filename = ("housing.csv")

housing = pd.read_csv(filename)
housing.head()


# In[3]:


housing.info()


# In[4]:


housing.describe()


# In[5]:


# Se grafican los histogramas de cada caracteristica para evaluar la distribuccion

housing.hist(bins=50, figsize=(20,15))
plt.show()


# In[6]:


#Se realiza un mapa donde se marca, mediante una escala, los valores de la casa 

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7),c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
plt.legend()
plt.show()


# In[7]:


# Se calcula el coeficiente de correlacion de Pearson entre los valores de las casas y los demas atributos

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[8]:


# se evalua la relaccion entre caracteristicas con correlacion positiva

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[10]:


# Se utiliza la clase imputer provista por sklearn para rellenar los valores nulos con la mediana

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
housing_tr.info()


# In[11]:


housing_tr.head()


# In[12]:


#Se divide la base de datos en un set de entrenamiento y otro de testeo

X = housing_tr.drop("median_house_value", axis=1)
y = housing_tr["median_house_value"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(len(X_train), "train +", len(y_test), "test")


# In[14]:


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_housing_predictions = lin_reg.predict(X)
lin_mse = mean_squared_error(y, lin_housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[15]:


lin_scores = cross_val_score(lin_reg, X_train, y_train, scoring = "neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())
    
display_scores(lin_rmse_scores)


# In[16]:


tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)
tree_housing_predictions = tree_reg.predict(X_train)
tree_mse = mean_squared_error(y_train, tree_housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[17]:


scores = cross_val_score(tree_reg, X_train, y_train, scoring = "neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)


# In[18]:


forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)
forest_housing_predictions = forest_reg.predict(X_train)
forest_mse = mean_squared_error(y_train, forest_housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[19]:


forest_scores = cross_val_score(forest_reg, X_train, y_train, scoring = "neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# Para la regression lineal se obtiene un error fue muy grande (de casi $70K cuando los valores de las casa van entre $120K y $265K) debido a que el programa esta underfiteando, luego para mejorar el ajuste se puede probar con un modelo mas poderoso, mejorar al algoritmo con mejores caracteristicas o reducir los limites del modelo. Con DecisionTree el error es nulo como consecuencia del overfitting y la performance es incluso peor que para la regresion lineal. Finalmente, al aplicar RandomForest el resultado fue mucho mejor, pero todo parece indicar que sigue overfiteando. Para solucionarlo habria que simplificar el modelo, agregarle limites/regularizarlo o agregarle datos de entrenamiento.

# In[ ]:


param_grid = [
    {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
    {'bootstrap':[False], 'n_estimators':[3, 10], 'max_features':[2,3,4]}
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)


# In[55]:


grid_search.best_estimator_


# In[59]:


forest_reg_opt = grid_search.best_estimator_
forest_reg_opt.fit(X_train, y_train)
forest_opt_housing_predictions = forest_reg_opt.predict(X_train)
forest_opt_mse = mean_squared_error(y_train, forest_opt_housing_predictions)
forest_opt_rmse = np.sqrt(forest_opt_mse)
forest_opt_rmse


# In[58]:


forest_opt_scores = cross_val_score(forest_reg_opt, X_train, y_train, scoring = "neg_mean_squared_error", cv=10)
forest_opt_rmse_scores = np.sqrt(-forest_opt_scores)
display_scores(forest_opt_rmse_scores)


# In[60]:


final_predictions = forest_reg_opt.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse

