import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carregar os dados
print("🔄 Carregando os dados...")
df = pd.read_csv("../corpus_tfidf.csv")
df = df.astype('float32')  # Converte o restante para float32

# 2. Separar embeddings e classes
X = df.drop(columns=['category']).values
y = df['category'].values

# 3. Testar diferentes valores de componentes com TruncatedSVD
n_components_list = [1000, 2000, 3000]
resultados_svd = []

print("🔄 Avaliando diferentes valores de n_components com validação cruzada (Decision Tree)...")
for n in n_components_list:
    print(f"➡️ Testando n_components={n}...")
    svd = TruncatedSVD(n_components=n, random_state=42)
    X_reduced = svd.fit_transform(X)

    clf_svd = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    scores = cross_val_score(clf_svd, X_reduced, y, cv=3, scoring='f1_weighted', n_jobs=-1)
    media_f1 = scores.mean()
    print(f"✔️ n_components={n} → F1-score médio: {media_f1:.4f}")
    resultados_svd.append((n, media_f1))

# Melhor n_components
melhor_n = max(resultados_svd, key=lambda x: x[1])[0]
print(f"\n✅ Melhor n_components encontrado: {melhor_n}")

# 4. Aplicar TruncatedSVD com melhor valor
print(f"🔄 Reduzindo dimensionalidade com TruncatedSVD, n_components={melhor_n}...")
svd_final = TruncatedSVD(n_components=melhor_n, random_state=42)
X_reduced = svd_final.fit_transform(X)

# 5. Dividir em treino e teste
print("🔄 Dividindo dados em treino (85%) e teste (15%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.15, stratify=y, random_state=42
)

# 6. Grid de hiperparâmetros
param_grid = {
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 7. GridSearchCV
print("🔄 Iniciando GridSearchCV com Decision Tree...")
clf = DecisionTreeClassifier(class_weight='balanced', random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, error_score='raise')
grid_search.fit(X_train, y_train)

# 8. Melhor modelo
melhor_modelo = grid_search.best_estimator_
print(f"✅ Melhores parâmetros encontrados: {grid_search.best_params_}")

# 9. Avaliação no conjunto de teste
print("🔍 Avaliando no conjunto de teste...")
y_pred = melhor_modelo.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print(f"Acurácia: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Matriz de Confusão:\n{cm}")

# 10. Matriz de confusão
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=["business", "education", "entertainment", "sports", "technology"],
            yticklabels=["business", "education", "entertainment", "sports", "technology"])
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title(f"Confusion Matrix - Decision Tree TF-IDF (15% Test)")
plt.savefig("MC_arvore_tfidf.png")
plt.close()
print("✅ Matriz salva como 'MC_arvore_tfidf.png'.")
