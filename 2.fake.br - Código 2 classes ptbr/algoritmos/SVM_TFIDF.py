import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2

# 1. Carregar os dados
print("🔄 Carregando os dados...")
df = pd.read_csv("../corpus_tfidf.csv")
df = df.astype('float32')

# 2. Separar características e classe
X = df.drop(columns=['FakeTrue']).values
y = df['FakeTrue'].values

# 🔄 Testando diferentes valores de k
ks = [1000, 2000, 3000]
resultados_k = []

print("🔄 Avaliando diferentes valores de k com validação cruzada...")
for k in ks:
    print(f"➡️ Testando k={k}...")
    selector = SelectKBest(score_func=chi2, k=k)
    X_k = selector.fit_transform(X, y)

    clf_k = LinearSVC(class_weight='balanced', random_state=42, max_iter=5000)
    scores = cross_val_score(clf_k, X_k, y, cv=5, scoring='f1_weighted', n_jobs=-1)
    media_f1 = scores.mean()
    print(f"✔️ k={k} → F1-score médio: {media_f1:.4f}")
    resultados_k.append((k, media_f1))

# Selecionar o melhor k
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

# Espaço de busca para LinearSVC
param_dist = {
    'C': [0.01, 0.1, 1, 10, 100],
    'loss': ['hinge', 'squared_hinge'],
    'tol': [1e-3, 1e-4, 1e-5],
}

print("🔄 Iniciando RandomizedSearchCV com LinearSVC...")
clf = LinearSVC(class_weight='balanced', random_state=42, max_iter=5000)

random_search = RandomizedSearchCV(
    clf,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

melhor_modelo = random_search.best_estimator_
print(f"✅ Melhores parâmetros encontrados: {random_search.best_params_}")

# Avaliação no conjunto de teste
print("🔍 Avaliando no conjunto de teste...")
y_pred = melhor_modelo.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print(f"Acurácia: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Matriz de Confusão:\n{cm}")

# Matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=["Fake", "True"], yticklabels=["Fake", "True"])
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title(f"Confusion Matrix - SVM TF-IDF (k={melhor_k})")
plt.savefig("MC_svm_tfidf.png")
plt.close()
print("✅ Matriz salva como 'MC_svm_tfidf.png'.")
