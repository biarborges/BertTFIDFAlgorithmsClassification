import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carregar os dados
print("🔄 Carregando os dados...")
df = pd.read_csv("../corpus_tfidf.csv")

# 2. Separar características e classe
X = df.drop(columns=['polarity']).values
y = df['polarity'].values

# 3. Dividir em treino (85%) e teste (15%)
print("🔄 Dividindo dados em treino (85%) e teste (15%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# 4. Espaço de busca para RandomizedSearchCV (igual ao param_grid antes)
param_dist = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10]},
    {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
    {'kernel': ['poly'], 'C': [0.1, 1, 10], 'degree': [2, 3], 'gamma': ['scale', 'auto']},
    {'kernel': ['sigmoid'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
]

# 5. RandomizedSearchCV com validação cruzada, testando 20 combinações aleatórias
print("🔄 Iniciando RandomizedSearchCV com todos os kernels do SVM...")
clf = SVC(class_weight='balanced', probability=True, random_state=42)
random_search = RandomizedSearchCV(clf, param_dist, n_iter=20, cv=5, scoring='f1_weighted', n_jobs=-1, random_state=42)
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
