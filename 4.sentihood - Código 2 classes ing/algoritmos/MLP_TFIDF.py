import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2

# 1. Carregar os dados
print("🔄 Carregando os dados...")
df = pd.read_csv("../corpus_tfidf.csv")
df = df.astype('float32')

# 2. Separar características e classe
X = df.drop(columns=['sentiment']).values
y = df['sentiment'].values

# 3. Seleção de k melhores features via chi2 + validação cruzada
ks = [1000, 2000, 3000]
resultados_k = []

print("🔄 Avaliando diferentes valores de k com validação cruzada (MLP)...")
for k in ks:
    print(f"➡️ Testando k={k}...")
    selector = SelectKBest(score_func=chi2, k=k)
    X_k = selector.fit_transform(X, y)

    clf_k = MLPClassifier(max_iter=300, early_stopping=True, random_state=42)
    scores = cross_val_score(clf_k, X_k, y, cv=3, scoring='f1_weighted', n_jobs=-1)
    media_f1 = scores.mean()
    print(f"✔️ k={k} → F1-score médio: {media_f1:.4f}")
    resultados_k.append((k, media_f1))

# Melhor k
melhor_k = max(resultados_k, key=lambda x: x[1])[0]
print(f"\n✅ Melhor k encontrado: {melhor_k}")

# Aplicar SelectKBest com melhor k
print(f"🔄 Reduzindo dimensionalidade com SelectKBest (chi2), k={melhor_k}...")
selector_final = SelectKBest(score_func=chi2, k=melhor_k)
X = selector_final.fit_transform(X, y)

# Dividir em treino e teste
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
    'learning_rate': ['constant', 'invscaling', 'adaptive']
}

# 5. GridSearchCV
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
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title(f"Confusion Matrix - MLP TF-IDF (k={melhor_k})")
plt.savefig("MC_mlp_tfidf.png")
plt.close()
print("✅ Matriz salva como 'MC_mlp_tfidf.png'.")
