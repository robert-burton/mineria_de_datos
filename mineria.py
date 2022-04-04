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
#from scipy import stats

"""
1. Data Adquisition
Nota: se interactúa con la API de Kaggle por medio de su implementación CLI en Python: https://www.kaggle.com/docs/api
"""

os.system("kaggle datasets download -d nasa/meteorite-landings") #se manda comando a CLI para descarga de dataset

with zipfile.ZipFile("meteorite-landings.zip", 'r') as zip_ref: #se extrae archivo .csv del archivo .zip descargado
    zip_ref.extractall("")

df = pd.read_csv("meteorite-landings.csv") #se crea dataframe a partir de archivo .csv

"""
2. Data Cleaning
"""

def clean_data(df):
    df_clean = df.drop(columns = ['name','nametype']) #se eliminan columnas redundantes
    #df_clean['year'] = df_clean['year'].astype('Int16')
    return df_clean

df_clean = clean_data(df)

print(df_clean)

"""
3. Data Analysis
Implementar funciones de agregación
"""

#Estadisticas de RECCLASS
print("--------------------------------")
print("ESTADISTICAS DE CLASE (recclass)")
print("\n")
print("Total de clases registradas: " + str(pd.Series(list(df_clean['recclass'])).value_counts().count()))
print("\n")
stat_recclass_class, stat_recclass_count = zip(*Counter(df_clean['recclass']).most_common(5))
d_stat_recclass = {'Clase': list(stat_recclass_class), 'Cuenta': list(stat_recclass_count)}
df_stat_recclass = pd.DataFrame(data = d_stat_recclass)
print("Top 5 clases más comunes: \n" + str(df_stat_recclass))
print("--------------------------------")
print("\n")

#Estadisticas de MASS
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

stat_geolocation_coordinates, stat_geolocation_count = zip(*Counter(df_clean['GeoLocation']).most_common(5))
d_stat_geolocation_topcount = {'Coordenadas': list(stat_geolocation_coordinates), 'Cuenta': list(stat_geolocation_count)}
df_stat_geolocation_topcount = pd.DataFrame(data = d_stat_geolocation_topcount)
print("\n")
print("Top 5 coordenadas más comunes: \n" + str(df_stat_geolocation_topcount))
print("--------------------------------")
print("\n")