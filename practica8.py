"""
Roberto Mora Sepúlveda 1941421
Minería de Datos Gpo. 033

8. Clasificación
"""

#Preparación de dataframe para la práctica
from practica2 import *
import pandas as pd

df = pd.read_csv("meteorite-landings.csv")
df_clean = clean_data(df)

#Comienza práctica

import matplotlib.pyplot as plt

def get_cmap(n, name="hsv"):
    return plt.cm.get_cmap(name, n)

def scatter_group_by(file_path: str, df: pd.DataFrame, x_column: str, y_column: str, label_column: str):
    fig, ax = plt.subplots()
    labels = pd.unique(df[label_column])
    cmap = get_cmap(len(labels) + 1)
    for i, label in enumerate(labels):
        filter_df = df.query(f"{label_column} == '{label}'")
        ax.scatter(filter_df[x_column], filter_df[y_column], label=label, color=cmap(i))
    ax.legend()
    
    #edicion
    ax.set_yscale('log')
    plt.xlabel('Año')
    plt.ylabel('Masa (gramos)')
    plt.grid(True)
    plt.title('Clasificación: Año, Masa, Visto Caer/Encontrado')
    plt.savefig(file_path)
    plt.close()

df_clust = pd.DataFrame()
df_clust['mass'] = df_clean['mass']
df_clust['year'] = df_clean['year']
df_clust['fall'] = df_clean['fall']

print(df_clust)

scatter_group_by("C:\\Users\\rm\\Desktop\\github\\mineria_de_datos\\clasificacion\\1.classification.png", df_clust, 'year', 'mass', 'fall')