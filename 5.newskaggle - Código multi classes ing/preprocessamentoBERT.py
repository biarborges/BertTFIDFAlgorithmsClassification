import pandas as pd
import re

# Função de pré-processamento
def preprocess_text(text):

    # Remover caracteres especiais, mas preservar pontuação
    text = re.sub(r'[^\w\sçÇáàãâéêíóôõúüÁÀÃÂÉÊÍÓÔÕÚÜ.,!?;:()\-]', '', text)

    # Normalizar espaços
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Caminhos
arquivo_csv = "../BertTFIDFAlgorithmsClassification/5.newskaggle - Código multi classes ing/corpus.csv"
saida_csv = "../BertTFIDFAlgorithmsClassification/5.newskaggle - Código multi classes ing/corpus_processadoBERT.csv"

# Carregar CSV
df = pd.read_csv(arquivo_csv)

# Aplicar pré-processamento
df['content'] = df['content'].astype(str).apply(preprocess_text)

# Selecionar colunas finais
df_novo = df[['content', 'category']]

# Salvar
df_novo.to_csv(saida_csv, index=False)

print(f"Arquivo processado salvo como: {saida_csv}")
