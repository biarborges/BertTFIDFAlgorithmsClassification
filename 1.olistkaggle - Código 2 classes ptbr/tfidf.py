import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Caminho do arquivo de entrada
entrada_csv = "corpus_processadoTFIDF.csv"
saida_csv = "corpus_tfidf.csv"

# Ler os dados
df = pd.read_csv(entrada_csv)

# Juntar os tokens em string novamente
df['review_text_str'] = df['review_text_processed'].apply(lambda x: ' '.join(eval(x)))

# Inicializar o vetor TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['review_text_str'])

# Converter para DataFrame
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

# Adicionar coluna de polaridade
df_tfidf['polarity'] = df['polarity'].values

# Salvar em CSV
df_tfidf.to_csv(saida_csv, index=False)

print(f"TF-IDF gerado e salvo em: {saida_csv}")
