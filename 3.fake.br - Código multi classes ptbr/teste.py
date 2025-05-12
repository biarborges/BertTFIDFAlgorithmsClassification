import pandas as pd

# Caminho do arquivo
entrada_csv = "../BertTFIDFAlgorithmsClassification/3.fake.br - Código multi classes ptbr/corpus_processadoTFIDF.csv"

# Lê o CSV de entrada
df = pd.read_csv(entrada_csv, encoding='utf-8', sep=',', quotechar='"')


nulos = df[df['noticia'].isna()]
print(f"Total de linhas com notícia nula: {len(nulos)}")
print(nulos[['noticia', 'categoria']])


