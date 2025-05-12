import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Caminhos dos arquivos
entrada_csv = "../BertTFIDFAlgorithmsClassification/5.newskaggle - Código multi classes ing/corpus_processadoTFIDF.csv"
saida_csv = "../BertTFIDFAlgorithmsClassification/5.newskaggle - Código multi classes ing/corpus_processadoTFIDF_classesNumericas.csv"

# Lê o CSV de entrada
df = pd.read_csv(entrada_csv)

# Inicializa o LabelEncoder e transforma a coluna 'category'
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])

mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Mapeamento das classes:")
for original, numero in mapping.items():
    print(f"{original} → {numero}")

# Salva o DataFrame resultante em um novo CSV
df.to_csv(saida_csv, index=False)
print(f"Arquivo processado salvo em: {saida_csv}")
