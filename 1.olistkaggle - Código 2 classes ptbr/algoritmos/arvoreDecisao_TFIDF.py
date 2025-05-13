import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 1. Carregar dados
df = pd.read_csv("1.olistkaggle - Código 2 classes ptbr/corpus_tfidf.csv")

# 2. Vetorização com TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["content"]).toarray()  # Converte para uma matriz NumPy
y = df["category"].values

# 3. Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Treinar o classificador de Árvore de Decisão
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 5. Fazer previsões
y_pred = clf.predict(X_test)

# 6. Criar o DataFrame com as classes originais e preditas
output_df = pd.DataFrame({
    'original_class': y_test,
    'predicted_class': y_pred
})

# 7. Salvar o arquivo de saída com as classes originais e preditas
output_df.to_csv("../BertTFIDFAlgorithmsClassification/1.olistkaggle - Código 2 classes ptbr/algoritmos/resultado_arvoreDecisaoTFIDF.csv", index=False)

print("Classificação finalizada e arquivo salvo como 'resultado_arvoreDecisaoTFIDF.csv'")
