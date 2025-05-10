import pandas as pd

# Caminho para o CSV gerado
csv_novo = "../BertTFIDFAlgorithmsClassification/4.sentihood - Código 2 classes ing/sentihood_apenas_uma_opiniao.csv"

# Carregar o CSV
df = pd.read_csv(csv_novo)

# Contar quantas linhas tem (quantos exemplos)
print(f'Total de exemplos com apenas uma opinião: {len(df)}')

# Contar quantos são Positive e quantos são Negative
contagem_sentimentos = df['sentiment'].value_counts()
print("\nDistribuição de sentimentos:")
print(contagem_sentimentos)
