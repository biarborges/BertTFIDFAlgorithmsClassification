import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Caminhos dos arquivos
entrada_csv = "../BertTFIDFAlgorithmsClassification/2.fake.br - Código 2 classes ptbr/corpus_processadoBERT.csv"
saida_csv = "../BertTFIDFAlgorithmsClassification/2.fake.br - Código 2 classes ptbr/corpus_processadoBERT_classesNumericas.csv"

# Lê o CSV de entrada
df = pd.read_csv(entrada_csv)

# Inicializa o LabelEncoder e transforma a coluna 'FakeTrue'
le = LabelEncoder()
df['FakeTrue'] = le.fit_transform(df['FakeTrue'])

mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Mapeamento das classes:")
for original, numero in mapping.items():
    print(f"{original} → {numero}")

# Salva o DataFrame resultante em um novo CSV
df.to_csv(saida_csv, index=False)
print(f"Arquivo processado salvo em: {saida_csv}")
