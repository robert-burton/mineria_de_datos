"""
Roberto Mora Sepúlveda 1941421
Minería de Datos Gpo. 033

7. Predicciones
Predicción de serie de tiempo
"""

#Preparación de dataframe para la práctica
from practica2 import *
import pandas as pd

df = pd.read_csv("meteorite-landings.csv")
df_clean = clean_data(df)

#Comienza práctica
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

def get_prediction_interval(prediction, y_test, test_predictions, pi=.95):
   
    #Desv est de y_test
    sum_errs = np.sum((y_test - test_predictions)**2)
    stdev = np.sqrt(1 / (len(y_test) - 2) * sum_errs)

    #Intervalo de desv est
    one_minus_pi = 1 - pi
    ppf_lookup = 1 - (one_minus_pi / 2)
    z_score = stats.norm.ppf(ppf_lookup)
    interval = z_score * stdev

    #Intervalo de predicción
    lower, upper = prediction - interval, prediction + interval
    return lower, prediction, upper

#Obtener regresión
df_lr = pd.DataFrame({'year': df_clean['year'], 'mass': df_clean['mass']})
X = df_lr['year'].values.reshape(-1,1)
Y= df_lr['mass']

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

#Graficar intervalo de confianza
lower_vet = []
upper_vet = []

for i in Y_pred:
    lower, prediction, upper =  get_prediction_interval(i, df_lr['mass'], Y_pred)
    lower_vet.append(lower)
    upper_vet.append(upper)

plt.fill_between(np.arange(0,len(df_lr['mass']),1),upper_vet, lower_vet, color='b',label='IC = 0.95')
plt.plot(np.arange(0,len(df_lr['mass']),1),df_lr['mass'],color='orange',label='Datos')
plt.plot(Y_pred,'k',label='Regresión', color = 'red')
plt.xlabel('Año')
plt.ylabel('Masa (gramos)')
plt.legend()
plt.title('Predicción de masa por año')
plt.savefig("C:\\Users\\rm\\Desktop\\github\\mineria_de_datos\\predicciones\\1.p-mass-year.png")
plt.show()
plt.close()