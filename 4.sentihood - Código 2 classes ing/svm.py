from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
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

# Inicializa e treina o classificador SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(train_embeddings, train_encoded_sentiments)

# Faz previsões no conjunto de teste
y_pred = svm_classifier.predict(test_embeddings)

# Avalia o desempenho do classificador
accuracy = accuracy_score(test_encoded_sentiments, y_pred)
f1 = f1_score(test_encoded_sentiments, y_pred, average='binary')

print("SVM:")
print(f"Acurácia: {accuracy}")
print(f"F1-score: {f1}")
