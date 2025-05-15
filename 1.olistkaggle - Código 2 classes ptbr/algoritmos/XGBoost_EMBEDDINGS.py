import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

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

# 4. Grid de hiperparâmetros para XGBoost
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 150],
    'subsample': [0.8, 1.0]
}

# 5. GridSearchCV com validação cruzada interna
print("🔄 Iniciando GridSearchCV com XGBoost...")
clf = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    tree_method = "hist", 
    device = "cuda" 
)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
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
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix - XGBoost (15% Test)")
plt.savefig("MC_xgboost_embeddings.png")
plt.close()
print("✅ Matriz salva como 'MC_xgboost_embeddings.png'.")
