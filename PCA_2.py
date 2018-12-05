# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 14:03:18 2018

@author: Pablo Beret
"""

import pandas as pd

# Carga del dataset completo. Elimino clasificador y otros atributos que no
# son necesarios                
                                                    
spy_full = pd.read_csv("SPYV3.csv", sep=',')

spy_full = spy_full.drop(['FECHA','OPEN', 'MAX', 'MIN', 'CLOSE','CLASIFICADOR', 
                          'FECHA.year', 'FECHA.day-of-month', 
                          'FECHA.day-of-week'],  1)

# Las variables categóricas que no son numéricas son factorizadas

spy_full['39'], unique = pd.factorize(spy_full['39'])
spy_full['41'], unique = pd.factorize(spy_full['41'])
spy_full['43'], unique = pd.factorize(spy_full['43'])
spy_full['168'], unique = pd.factorize(spy_full['168'])
spy_full['172'], unique = pd.factorize(spy_full['172'])

# Escalado de variables con MinMax()

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
spy_full_m = min_max_scaler.fit_transform(spy_full)

# Normalización de variables con StandardScaler()

from sklearn.preprocessing import StandardScaler

spy_full_s = StandardScaler().fit_transform(spy_full)

# Análisis de componentes principales (PCA) usando MinMax()
# Para usar StandardSacler() cambiar "spy_full_m" por spy_full_s"

from sklearn.decomposition import PCA
import numpy

n_comp = 4
estimator = PCA (n_components = n_comp)
X_pca = estimator.fit_transform(spy_full_m)

print(estimator.explained_variance_ratio_)

i=0
suma=0
while i < n_comp:
    suma= suma + estimator.explained_variance_ratio_[i]
    i = i + 1
    
print("Varianza total: ", suma)

pc1=pd.DataFrame(numpy.matrix.transpose(estimator.components_), columns=['PC-1', 
             'PC-2', 'PC-3', 'PC-4'], index=spy_full.columns)
print(pc1)

# Filtrado para obtener los mayores PC. Los valores deben ser adaptados
# a cada caso

data_filter = pc1[pc1['PC-1'] >= 0.10]
print(data_filter)

data_filter = pc1[pc1['PC-2'] >= 0.15]
print(data_filter)

data_filter = pc1[pc1['PC-3'] >= 0.25]
print(data_filter)

data_filter2 = pc1[pc1['PC-4'] >= 0.30]
print(data_filter)

# --------------------------------------------------------------------------------






