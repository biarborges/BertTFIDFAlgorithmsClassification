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
arquivo_csv = "../BertTFIDFAlgorithmsClassification/2.fake.br - Código 2 classes ptbr/fakebrJunto.csv"
saida_csv = "../BertTFIDFAlgorithmsClassification/2.fake.br - Código 2 classes ptbr/corpus_processadoBERT.csv"

# Carregar CSV
df = pd.read_csv(arquivo_csv)

# Aplicar pré-processamento
df['noticia'] = df['noticia'].astype(str).apply(preprocess_text)

# Selecionar colunas finais
df_novo = df[['noticia', 'FakeTrue']]

# Salvar
df_novo.to_csv(saida_csv, index=False)

print(f"Arquivo processado salvo como: {saida_csv}")
