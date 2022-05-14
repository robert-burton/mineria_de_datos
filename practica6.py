"""
Roberto Mora Sepúlveda 1941421
Minería de Datos Gpo. 033

6. Regresión lineal
Investigar relación entre año y masa de meteoritos
"""

#Preparación de dataframe para la práctica
from practica2 import *
import pandas as pd

df = pd.read_csv("meteorite-landings.csv")
df_clean = clean_data(df)

#Comienza práctica
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats.stats import pearsonr

#Graficar datos
df_lr = pd.DataFrame({'year': df_clean['year'], 'mass': df_clean['mass']})
df_lr_plot_scatter = df_lr.plot.scatter(x = 'year', y = 'mass')

plt.xlabel('Año')
plt.ylabel('Masa (gramos)')
plt.title('Distribución año-masa')
plt.savefig("C:\\Users\\rm\\Desktop\\github\\mineria_de_datos\\regresion_lineal\\1.mass-year-plot.png")
plt.show()
plt.close()

#Regresión lineal
correlation = pearsonr(df_lr['year'], df_lr['mass'])
print("Coeficiente de correlación: ", correlation[0])

X = df_lr['year'].values.reshape(-1,1)
Y= df_lr['mass']

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.xlabel('Año')
plt.ylabel('Masa (gramos)')
plt.title('Regresión lineal año-masa')
plt.savefig("C:\\Users\\rm\\Desktop\\github\\mineria_de_datos\\regresion_lineal\\2.lr-mass-year.png")
plt.show()
plt.close()