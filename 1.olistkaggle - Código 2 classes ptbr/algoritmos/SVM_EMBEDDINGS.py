import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carregar os dados
print("🔄 Carregando os dados...")
df = pd.read_pickle("../corpus_embeddings.pkl")

# 2. Separar embeddings e classes
X = np.vstack(df['embedding'].values)
y = df['polarity'].values

# 3. Dividir em treino (85%) e teste (15%)
print("🔄 Dividindo dados em treino (85%) e teste (15%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# 4. Espaço de busca para RandomizedSearchCV (igual ao param_grid antes)
param_dist = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10]},  # mais simples e rápido
    {'kernel': ['rbf'], 'C': [1, 10], 'gamma': ['scale']},  # menos combinações
    {'kernel': ['poly'], 'C': [1], 'degree': [2], 'gamma': ['scale']},  # só 1 grau e valores fixos
    {'kernel': ['sigmoid'], 'C': [1], 'gamma': ['scale']}  # poucas combinações
]

# 5. RandomizedSearchCV com validação cruzada, testando combinações aleatórias
print("🔄 Iniciando RandomizedSearchCV com todos os kernels do SVM...")
clf = LinearSVC(class_weight='balanced', probability=True, random_state=42)
random_search = RandomizedSearchCV(clf, param_dist, n_iter=10, cv=5, scoring='f1_weighted', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# 6. Melhor modelo
melhor_modelo = random_search.best_estimator_
print(f"✅ Melhores parâmetros encontrados: {random_search.best_params_}")

# 7. Avaliação no conjunto de teste
print("🔍 Avaliando no conjunto de teste...")
y_pred = melhor_modelo.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print(f"Acurácia: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Matriz de Confusão:\n{cm}")

# 8. Matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix - SVM Embeddings (15% Test)")
plt.savefig("MC_svm_embeddings.png")
plt.close()
print("✅ Matriz salva como 'MC_svm_embeddings.png'.")
