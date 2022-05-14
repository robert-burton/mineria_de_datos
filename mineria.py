"""
Roberto Mora Sepúlveda 1941421
Minería de Datos Gpo. 033
"""

#Liberias utilizadas
import os
import zipfile
import pandas as pd

import numpy as np
from collections import Counter
from datetime import date

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
import statsmodels.formula.api as smf

#7
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from scipy import stats



"""
1. Extracción de datos
Nota: se interactúa con la API de Kaggle por medio de su implementación CLI en Python: https://www.kaggle.com/docs/api
"""

os.system("kaggle datasets download -d nasa/meteorite-landings") #se manda comando a CLI para descarga de dataset

with zipfile.ZipFile("meteorite-landings.zip", 'r') as zip_ref: #se extrae archivo .csv del archivo .zip descargado
    zip_ref.extractall("")

df = pd.read_csv("meteorite-landings.csv") #se crea dataframe a partir de archivo .csv

"""
2. Limpieza de datos
Limpiar dataset y preparar para análisis
"""

def clean_data(df):
    df_clean = df.drop(columns = ['name']) #se eliminan columnas redundantes

    df_clean = df_clean[df_clean['reclat'].notna()] #se eliminan coordenadas inválidas
    df_clean = df_clean[df_clean['reclong'].notna()]
    df_clean = df_clean[df_clean['GeoLocation'].notna()]
    df_clean = df_clean[(df_clean.reclat != 0.0) & (df_clean.reclong != 0.0)]
    
    df_clean = df_clean[df_clean['mass'].notna()] #se eliminan filas sin masa

    current_year = date.today().year
    df_clean = df_clean[df_clean.year <= current_year] #eliminar años inválidos o menores a 1900
    df_clean = df_clean[df_clean.year > 1900]

    return df_clean

df_clean = clean_data(df)

print(df_clean)

"""
3. Análisis de datos
Implementar funciones de agregación
"""

#Estadisticas de RECCLASS
print("--------------------------------")
print("ESTADISTICAS DE CLASE (recclass)")
print("\n")
print("Total de clases registradas: " + str(pd.Series(list(df_clean['recclass'])).value_counts().count()))
print("\n")
stat_recclass_class, stat_recclass_count = zip(*Counter(df_clean['recclass']).most_common(10))
d_stat_recclass = {'Clase': list(stat_recclass_class), 'Cuenta': list(stat_recclass_count)}
df_stat_recclass = pd.DataFrame(data = d_stat_recclass)
print("Top 10 clases más comunes: \n" + str(df_stat_recclass))
print("--------------------------------")
print("\n")

#Estadisticas de MASS
#Nota: se utiliza la unidad de gramos
print("--------------------------------")
print("ESTADISTICAS DE MASA (mass)")
print("\n")
d_stat_mass = {
    "Stat": ["Media", "Mediana", "Desv Est", "Var", "Suma", "Max", "Min"], 
    "Valor": [np.nanmean(df_clean['mass']),
    np.nanmedian(df_clean['mass']),
    np.nanstd(df_clean['mass']),
    np.nanvar(df_clean['mass']),
    np.nansum(df_clean['mass']),
    np.nanmax(df_clean['mass']),
    np.nanmin(df_clean['mass'])]
    }
df_stat_mass = pd.DataFrame(data = d_stat_mass)
print(df_stat_mass)
print("--------------------------------")
print("\n")

#Estadisticas de FALL
print("--------------------------------")
print("ESTADISTICAS DE VISTOS/ENCONTRADOS (fall)")
print("\n")
_, stat_fall_count = zip(*Counter(df_clean['fall']).most_common(5))
stat_fall_count = list(stat_fall_count)
d_stat_fall = {"Categoria": ["Encontrado", "Visto Caer"], "Cuenta": stat_fall_count, "%": [stat_fall_count[0]/sum(stat_fall_count)*100,stat_fall_count[1]/sum(stat_fall_count)*100]}
df_stat_fall = pd.DataFrame(data = d_stat_fall)
print(df_stat_fall)
print("--------------------------------")
print("\n")

#Estadisticas de YEAR
print("--------------------------------")
print("ESTADISTICAS DE AÑO (year)")
print("\n")
d_stat_year = {
    "Stat": ["Media", "Mediana", "Desv Est", "Var", "Max", "Min"],
    "Valor": [np.nanmean(df_clean['year']),
    np.nanmedian(df_clean['year']),
    np.nanstd(df_clean['year']),
    np.nanvar(df_clean['year']),
    np.nanmax(df_clean['year']),
    np.nanmin(df_clean['year'])]
    }
df_stat_year = pd.DataFrame(data = d_stat_year)
df_stat_year = df_stat_year.round(4)
print(df_stat_year)
print("--------------------------------")
print("\n")

#Estadisticas de RECLAT, RECLONG, GEOLOCATION
print("--------------------------------")
print("ESTADISTICAS DE GEOUBICACION (reclat, reclong, Geolocation)")
print("\n")
d_stat_geolocation = {
    "Stat": ["Media", "Mediana"],
    "Valor": [(np.nanmean(df_clean['reclat']).round(6), np.nanmean(df_clean['reclong']).round(6)),
    (np.nanmedian(df_clean['reclat']).round(6), np.nanmedian(df_clean['reclong']).round(6))]
    }
df_stat_geolocation = pd.DataFrame(data = d_stat_geolocation)
print(df_stat_geolocation)

stat_geolocation_coordinates, stat_geolocation_count = zip(*Counter(df_clean['GeoLocation']).most_common(10))
d_stat_geolocation_topcount = {'Coordenadas': list(stat_geolocation_coordinates), 'Cuenta': list(stat_geolocation_count)}
df_stat_geolocation_topcount = pd.DataFrame(data = d_stat_geolocation_topcount)
print("\n")
print("Top 10 coordenadas más comunes: \n" + str(df_stat_geolocation_topcount))
print("--------------------------------")
print("\n")

"""
4. Visualización de datos
Graficar datos
"""

#Pastel: cuenta de clases
plt.pie(df_stat_recclass['Cuenta'], labels = df_stat_recclass['Clase'])
plt.xlabel('Clases')
plt.title('10 clases de meteoritos más coumnes')
plt.savefig("C:\\Users\\rm\\Desktop\\github\\mineria_de_datos\\graficas\\1.class-count.png")
plt.show()
plt.close()

#Histograma: distribución de masa
plt.hist(df_clean['mass'], log = True)
plt.xlabel('Masa (gramos)')
plt.ylabel('Cuenta')
plt.grid(True)
plt.title('Distribución de masa de meteoritos')
plt.savefig("C:\\Users\\rm\\Desktop\\github\\mineria_de_datos\\graficas\\2.mass-hist.png")
plt.show()
plt.close()

#Pastel: meteoritos vistos caer vs encontrados
plt.pie(df_stat_fall['Cuenta'], labels = df_stat_fall['Categoria'])
plt.title('Meteoritos vistos caer VS encontrados después')
plt.savefig("C:\\Users\\rm\\Desktop\\github\\mineria_de_datos\\graficas\\3.fall-count.png")
plt.show()
plt.close()

#Histograma: distribución de años
plt.hist(df_clean['year'], log = True)
plt.xlabel('Año')
plt.ylabel('Cuenta')
plt.grid(True)
plt.title('Distribución de año de registro de meteoritos')
plt.savefig("C:\\Users\\rm\\Desktop\\github\\mineria_de_datos\\graficas\\4.year-hist.png")
plt.show()
plt.close()

#Mapa de distribución: impactos registrados
from mpl_toolkits.basemap import Basemap

valids = df_clean.groupby('nametype').get_group('Valid').copy()
valids.dropna(inplace=True)

map = Basemap(projection='cyl')
map.drawmapboundary(fill_color='w')
map.drawcoastlines(linewidth=0.5)
map.drawmeridians(range(0, 360, 20),linewidth=0.1)
map.drawparallels([-66.56083,-23.5,0.0,23.5,66.56083], linewidth=0.6)
x, y = map(valids.reclong,valids.reclat)

map.scatter(x, y, marker='.',alpha=0.25,c='green',edgecolor='None')
plt.title('Mapa de distribución de impactos')
plt.savefig("C:\\Users\\rm\\Desktop\\github\\mineria_de_datos\\graficas\\5.mapdist-count.png")
plt.show()
plt.close()

#Mapa de distribución: masa
map = Basemap(projection='cyl')
map.drawmapboundary(fill_color='w')
map.drawcoastlines(linewidth=0.6)
map.drawmeridians(range(0, 360, 20),linewidth=0.1)
map.drawparallels([-66.56083,-23.5,0.0,23.5,66.56083], linewidth=0.6)

map.scatter(valids.reclong,valids.reclat,s=np.sqrt(valids.mass/150),alpha=0.4,color='g')
plt.title('Mapa de distribución de masa')
plt.savefig("C:\\Users\\rm\\Desktop\\github\\mineria_de_datos\\graficas\\6.mapdist-mass.png")
plt.show()
plt.close()

"""
5. Prueba de hipótesis

i.
    H0: "No existe correlación entre masa de meteorito y latitud registrada"
    H1: "Existe correlación entre masa de meteorito y latitud registrada"
ii.
    Se empleará prueba T con muestra aleatoria de 100
iii.
    Calcular valor p
iv.
    Determinar significancia estadística, comparar p con alfa
"""

df_sample = df_clean.sample(n = 100, random_state = 100)
print(df_sample)

"""
6. Regresión lineal
Investigar relación entre año y masa de meteoritos
"""

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

"""
7. Predicciones
Predicción de serie de tiempo
"""

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

"""
8. Clasificación
???
"""

"""
9. Clustering
???
"""