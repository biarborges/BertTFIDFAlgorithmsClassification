import pandas as pd

# Carrega o CSV
df = pd.read_csv("olist.csv")

# Conta o número de colunas
num_colunas = len(df.columns)
print(f"\nNúmero de colunas: {num_colunas}")

num_linhas = len(df)
print(f"\nNúmero de linhas: {num_linhas}")

contagem_texto = df['review_text_processed'].notnull().sum()
print("\nContagem de valores na coluna 'review_text_processed':")
print(contagem_texto)

contagem_polarity_total = df['polarity'].notnull().sum()
print("\nContagem de valores na coluna 'polarity':")
print(contagem_polarity_total)

# Conta a frequência dos valores na coluna 'polarity'
contagem_polarity = df['polarity'].value_counts().sort_index()
print("\nContagem de valores na coluna 'polarity':")
print(contagem_polarity)

sem_polaridade = df[df['review_text_processed'].notnull() & df['polarity'].isnull()]
print(f"Número de textos com review_text_processed mas sem polaridade: {len(sem_polaridade)}")

