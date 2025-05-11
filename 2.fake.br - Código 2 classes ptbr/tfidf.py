import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import time

# Caminho do arquivo de entrada
entrada_csv = "corpus_processadoTFIDF.csv"
saida_csv = "corpus_tfidf.csv"

tqdm.pandas()

# Ler os dados
print("Lendo CSV...")
start = time.time()
df = pd.read_csv(entrada_csv)
print(f"Leitura concluída em {time.time() - start:.2f} segundos\n")

# Juntar os tokens
print("Reconstruindo textos com progress_apply...")
start = time.time()
df['texto'] = df['noticia'].progress_apply(lambda x: ' '.join(eval(x)))
print(f"Reconstrução concluída em {time.time() - start:.2f} segundos\n")

# Vetorização TF-IDF
print("Executando TF-IDF (sem barra de progresso)...")
start = time.time()
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['texto'])
print(f"TF-IDF concluído em {time.time() - start:.2f} segundos\n")

# Converter para DataFrame
print("Convertendo TF-IDF para DataFrame...")
start = time.time()
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
print(f"Conversão concluída em {time.time() - start:.2f} segundos\n")

# Adicionar colunas
df_tfidf['FakeTrue'] = df['FakeTrue'].values

# Salvar em CSV
print("Salvando CSV final...")
start = time.time()
df_tfidf.to_csv(saida_csv, index=False)
print(f"CSV salvo em {time.time() - start:.2f} segundos\n")

print(f"TF-IDF gerado e salvo em: {saida_csv}")
