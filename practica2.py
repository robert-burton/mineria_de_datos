"""
Roberto Mora Sepúlveda 1941421
Minería de Datos Gpo. 033

2. Limpieza de datos
Limpiar dataset y preparar para análisis
"""

import pandas as pd
from datetime import date

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

df = pd.read_csv("meteorite-landings.csv")

df_clean = clean_data(df)

print(df_clean)