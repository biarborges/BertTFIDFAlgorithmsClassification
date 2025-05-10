import pandas as pd

# Lista dos arquivos CSV a serem combinados
files = [
    'business_data.csv',
    'education_data.csv',
    'entertainment_data.csv',
    'sports_data.csv',
    'technology_data.csv'
]

# Lista para armazenar os DataFrames
data_frames = []

# Loop através de cada arquivo e selecione as colunas desejadas
for file in files:
    df = pd.read_csv(file, usecols=['content', 'category'])
    data_frames.append(df)

# Concatenar todos os DataFrames em um único DataFrame
combined_df = pd.concat(data_frames, ignore_index=True)

# Salvar o DataFrame combinado em um novo arquivo CSV
combined_df.to_csv('corpus.csv', index=False)

print("Arquivo 'corpus.csv' criado com sucesso!")
