import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 1. Carregar dados
print("🔄 Carregando os dados...")
df = pd.read_pickle("../corpus_embeddings.pkl")

# 2. Separar as colunas
print("🔄 Separando as colunas de características e classe...")
X = np.vstack(df['embedding'].values)
y = df['polarity'].values

# 3. Configurar StratifiedKFold
print("🔄 Configurando a validação cruzada (StratifiedKFold)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Armazenar previsões e verdadeiros
all_y_true = []
all_y_pred = []

# 5. Armazenar melhores parâmetros de cada fold
melhores_parametros = []

# 6. Hiperparâmetros para busca
param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 7. Loop com GridSearchCV
print("🔄 Iniciando a validação cruzada com ajuste de hiperparâmetros...")
for i, (train_idx, test_idx) in enumerate(tqdm(skf.split(X, y), total=5, desc="Validação Cruzada")):
    print(f"\n🔁 Fold {i+1}")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # GridSearch
    base_clf = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    grid_search = GridSearchCV(base_clf, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_clf = grid_search.best_estimator_
    melhores_parametros.append(grid_search.best_params_)  # salva os melhores params

    print(f"✅ Melhores parâmetros do fold {i+1}: {grid_search.best_params_}")

    # Previsão
    y_pred = best_clf.predict(X_test)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

# 8. Avaliação final
print("\n🔍 Avaliação Global:")
acc = accuracy_score(all_y_true, all_y_pred)
f1 = f1_score(all_y_true, all_y_pred, average='weighted')
cm = confusion_matrix(all_y_true, all_y_pred)

print(f"Acurácia: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Matriz de Confusão:\n{cm}")

# 9. Matriz de confusão colorida
print("🖼️ Salvando matriz de confusão...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"])
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.title("Matriz de Confusão")
plt.savefig("MC_arvore_bert.png")
plt.close()
print("✅ Matriz salva como 'MC_arvore_bert.png'.")

# 10. Exibir os melhores parâmetros finais
print("\n📌 Melhores parâmetros por fold:")
for i, params in enumerate(melhores_parametros):
    print(f"Fold {i+1}: {params}")
