"""
Roberto Mora Sepúlveda 1941421
Minería de Datos Gpo. 033

9. Clustering
"""

#Preparación de dataframe para la práctica
from practica2 import *
import pandas as pd

df = pd.read_csv("meteorite-landings.csv")
df_clean = clean_data(df)

#Comienza práctica
import numpy as np # linear algebra
import random as rd
import matplotlib.pyplot as plt
from math import sqrt

X = df_clean[["mass","reclat"]]
print(X.head())

#1. Se eligen número de clusters K
K = 3

#2. Seleccionar K puntos aleatorios como centroide
centroids = (X.sample(n = K))
print("\nCentroides: \n", centroids)

#3. Asignar todos los puntos al centroide más cercano
#4. Recomputar centroides de clusteres nuevos
#5. Repetir 3. y 4.

diff = 1
j = 0

while(diff != 0):
    XD = X
    i = 1
    for index1, row_c in centroids.iterrows():
        ED = []
        for index2, row_d in XD.iterrows():
            d1 = (row_c["mass"]-row_d["mass"])**2
            d2 = (row_c["reclat"]-row_d["reclat"])**2
            d = sqrt(d1+d2)
            ED.append(d)
        X[i] = ED
        i = i+1
    
    C = []
    for index, row in X.iterrows():
        min_dist = row[1]
        pos = 1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos = i+1
        C.append(pos)
    X["Cluster"]=C
    centroids_new = X.groupby(["Cluster"]).mean()[["reclat", "mass"]]
    if j == 0:
        diff = 1
        j = j+1
    else:
        diff = (centroids_new['reclat'] - centroids['reclat']).sum() + (centroids_new['mass'] - centroids['mass']).sum()
    centroids = X.groupby(["Cluster"]).mean()[["reclat","mass"]]

#Graficar
color=['blue','green','cyan']
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["mass"],data["reclat"],c=color[k])
plt.scatter(centroids["mass"], centroids["reclat"],c='red')
plt.xlabel('Masa (gramos)')
plt.ylabel('Latitud')
plt.grid(True)
plt.title('Clustering Masa/Latitud')
plt.savefig("C:\\Users\\rm\\Desktop\\github\\mineria_de_datos\\clustering\\1.clustering-mass-reclat.png")
plt.show()
plt.close()