"""
Roberto Mora Sepúlveda 1941421
Minería de Datos Gpo. 033

1. Extracción de datos
Nota: se interactúa con la API de Kaggle por medio de su implementación CLI en Python: https://www.kaggle.com/docs/api
"""

import os
import zipfile
import pandas as pd

os.system("kaggle datasets download -d nasa/meteorite-landings") #se manda comando a CLI para descarga de dataset

with zipfile.ZipFile("meteorite-landings.zip", 'r') as zip_ref: #se extrae archivo .csv del archivo .zip descargado
    zip_ref.extractall("")

df = pd.read_csv("meteorite-landings.csv") #se crea dataframe a partir de archivo .csv