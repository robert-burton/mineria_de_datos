"""
Roberto Mora Sepúlveda 1941421
Minería de Datos Gpo. 033

5. Prueba de hipótesis
    i.
        H0: "No hay diferencias en masa promedio de meteorito registrada entre los Hemisferios Norte y Sur"
            i.e. sample_north['mass'].mean() - sample_south['mass'].mean() == 0
        H1: "Los meteoritos registrados en el Hemisferio Norte [0, 90] tienen más masa promedio que los del Hemisferio Sur [-90, 0)"
            i.e. sample_north['mass'].mean() - sample_south['mass'].mean() > 0
    ii.
        Se empleará prueba T con muestra aleatoria de 100
    iii.
        Calcular valor p
    iv.
        Determinar significancia estadística, comparar p con alfa
"""

#Preparación de dataframe para la práctica
from practica2 import *
import pandas as pd

df = pd.read_csv("meteorite-landings.csv")
df_clean = clean_data(df)

#Comienza práctica
from scipy.stats import t
from scipy.stats import ttest_ind


#Creación de conjunto muestra
#df_sample = df_clean.sample(n = 100, random_state = 101)
df_sample = df_clean.drop(columns = ['id', 'nametype', 'recclass', 'fall', 'year', 'reclong', 'GeoLocation']) #Se eliminan columnas redundantes

print(df_sample)

#Divisón de conjunto muestra en categorías
sample_north = df_sample[df_sample.reclat >= 0] #Meteoritos en Hemisferio Norte
sample_south = df_sample[df_sample.reclat < 0] #Meteoritos en Hemisferio Sur

sample_north = sample_north.drop(columns = ['reclat']) #Nos quedamos únicamente con las masas
sample_south = sample_south.drop(columns = ['reclat'])

sample_north = sample_north.sample(n = 50)
sample_south = sample_south.sample(n = 50)

print("----------------------------------------------NORTH----------------------------------------------")
print(sample_north)
print("----------------------------------------------SOUTH----------------------------------------------")
print(sample_south)

#Creación de distribución t
#rv = t(df = 100 - 2)

#Hipótesis alternativa
#print(sample_north['mass'].mean() - sample_south['mass'].mean() > 0)

#Prueba t de dos muestras, unilateral
t_stat, p_value = ttest_ind(sample_north, sample_south)
alpha = 0.05

print("\n")
print("Estadístico de prueba: ", t_stat)
print("Valor p: ", p_value)
print("Grados de libertad: ", sample_north.size + sample_south.size - 2)

if p_value <= alpha:
    print("Se rechaza H0, hay diferencia entre la masa promedio de meteoritos registrada en los Hemisferios Norte y Sur.")
else:
    print("No se rechaza H0, no hay evidencia que la masa promedio de meteoritos registrada en los Hemisferios Norte y Sur sean distintas.")