import pandas as pd
import re

# Função de pré-processamento
def preprocess_text(text):

    # Normalizar espaços
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Caminhos
arquivo_csv = "../BertTFIDFAlgorithmsClassification/1.olistkaggle - Código 2 classes ptbr/olist.csv"
saida_csv = "../BertTFIDFAlgorithmsClassification/1.olistkaggle - Código 2 classes ptbr/corpus_processadoBERT.csv"

# Carregar CSV
df = pd.read_csv(arquivo_csv)

# Remover linhas sem polaridade
df = df.dropna(subset=['polarity'])

# Aplicar pré-processamento
df['review_text_processed'] = df['review_text_processed'].astype(str).apply(preprocess_text)

# Selecionar colunas finais
df_novo = df[['review_text_processed', 'polarity']]

# Salvar
df_novo.to_csv(saida_csv, index=False)

print(f"Arquivo processado salvo como: {saida_csv}")
