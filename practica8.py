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
from tabulate import tabulate
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import numpy as np
from functools import reduce
from scipy.stats import mode

from collections import Counter


def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt="orgtbl"))


def normalize_distribution(dist: np.array, n: int) -> np.array:
    b = dist - min(dist) + 0.000001
    c = (b / np.sum(b)) * n
    return np.round(c)


def create_distribution(mean: float, size: int) -> pd.Series:
    return normalize_distribution(np.random.standard_normal(size), mean * size)

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


def euclidean_distance(p_1: np.array, p_2: np.array) -> float:
    return np.sqrt(np.sum((p_2 - p_1) ** 2))


def k_nearest_neightbors(points: List[np.array], labels: np.array, input_data: List[np.array], k: int):
    input_distances = [
        [euclidean_distance(input_point, point) for point in points]
        for input_point in input_data
    ]
    points_k_nearest = [
        np.argsort(input_point_dist)[:k] for input_point_dist in input_distances
    ]
    return [
        mode([labels[index] for index in point_nearest])
        for point_nearest in points_k_nearest
    ]

df_clust = pd.DataFrame()
df_clust['mass'] = df_clean['mass']
df_clust['year'] = df_clean['year']
df_clust['fall'] = df_clean['fall']


print(df_clust)

scatter_group_by("C:\\Users\\rm\\Desktop\\github\\mineria_de_datos\\clasificacion\\1.classification.png", df_clust, 'year', 'mass', 'fall')
list_t = [
    (np.array(tuples[0:1]), tuples[2])
    for tuples in df_clust.itertuples(index=False, name=None)]
points = [point for point, _ in list_t]
labels = [label for _, label in list_t]

kn = k_nearest_neightbors(
    points,
    labels,
    [np.array([100, 150]), np.array([1, 1]), np.array([1, 300]), np.array([80, 40])],
    5)
