from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from preprocessing import load_data, encode_categories

# Carrega os embeddings e labels
embeddings = np.load('embeddings128.npy', allow_pickle=True)
file_path = 'corpus.csv'
texts, categories = load_data(file_path)

# Codifica as categorias (assumindo que encode_categories retorna labels codificadas e o mapeamento)
encoded_categories, category_mapping = encode_categories(categories)

# Divide os embeddings em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(embeddings, encoded_categories, test_size=0.1)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Normaliza os dados de treino com MinMaxScaler (evite normalizar dados de teste)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

# Inicializa e treina o classificador Naive Bayes com dados normalizados
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_scaled, y_train)

# Faz previsões no conjunto de teste (não normalize os dados de teste)
y_pred = nb_classifier.predict(X_test)

# Avalia o desempenho do classificador
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("Naive Bayes:")
print(f"Acurácia: {accuracy}")
print(f"F1-score: {f1}")

