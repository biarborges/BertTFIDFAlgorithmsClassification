import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carregar os dados
print("🔄 Carregando os dados...")
df = pd.read_csv("../corpus_tfidf.csv")
df = df.astype('float32')

# 2. Separar características e classe
X = df.drop(columns=['FakeTrue']).values
y = df['FakeTrue'].values

# 3. Dividir em treino (85%) e teste (15%)
print("🔄 Dividindo dados em treino (85%) e teste (15%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# 4. Grid de hiperparâmetros
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['constant', 'invscaling', 'adaptive']  # será ignorado exceto quando solver='sgd'
}

# 5. GridSearchCV com validação cruzada
print("🔄 Iniciando GridSearchCV com MLP...")
clf = MLPClassifier(max_iter=500, early_stopping=True, random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, error_score='raise')
grid_search.fit(X_train, y_train)

# 6. Melhor modelo
melhor_modelo = grid_search.best_estimator_
print(f"✅ Melhores parâmetros encontrados: {grid_search.best_params_}")

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
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=["Fake", "True"], yticklabels=["Fake", "True"])
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix - MLP TF-IDF (15% Test)")
plt.savefig("MC_mlp_embeddings.png")
plt.close()
print("✅ Matriz salva como 'MC_mlp_embeddings.png'.")
