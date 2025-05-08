from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from preprocessing import load_and_process_json

# Carrega os embeddings e labels
train_embeddings = np.load('embeddings128train.npy', allow_pickle=True)
test_embeddings = np.load('embeddings128test.npy', allow_pickle=True)

# Carrega os dados do JSON de treino
train_file_path = 'sentihood-train.json'
train_texts, train_encoded_sentiments, train_sentiment_classes = load_and_process_json(train_file_path)

# Carrega os dados do JSON de teste
test_file_path = 'sentihood-test.json'
test_texts, test_encoded_sentiments, test_sentiment_classes = load_and_process_json(test_file_path)

print(train_embeddings.shape, len(train_encoded_sentiments))
print(test_embeddings.shape, len(test_encoded_sentiments))

# Normaliza os dados de treino com MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_embeddings)
train_embeddings_scaled = scaler.transform(train_embeddings)

# Inicializa e treina o classificador Naive Bayes com dados normalizados
nb_classifier = MultinomialNB()
nb_classifier.fit(train_embeddings_scaled, train_encoded_sentiments)

# Normaliza os dados de teste
test_embeddings_scaled = scaler.transform(test_embeddings)

# Faz previsões no conjunto de teste
y_pred = nb_classifier.predict(test_embeddings_scaled)

# Avalia o desempenho do classificador
accuracy = accuracy_score(test_encoded_sentiments, y_pred)
f1 = f1_score(test_encoded_sentiments, y_pred, average='binary')

print("Naive Bayes:")
print(f"Acurácia: {accuracy}")
print(f"F1-score: {f1}")
