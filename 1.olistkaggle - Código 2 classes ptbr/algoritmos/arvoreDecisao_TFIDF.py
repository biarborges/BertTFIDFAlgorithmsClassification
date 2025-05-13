import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 1. Carregar dados
df = pd.read_csv("../corpus_tfidf.csv")
df = df.dropna(subset=["content", "category"])

X_text = df["content"].astype(str).values
y = df["category"].values

# 2. Vetorização TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_text).toarray()

# 3. Configurar StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Armazenar previsões e verdadeiros
all_y_true = []
all_y_pred = []

# 5. Loop de validação cruzada
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

# 6. Avaliação única global
acc = accuracy_score(all_y_true, all_y_pred)
f1 = f1_score(all_y_true, all_y_pred, average='weighted')
cm = confusion_matrix(all_y_true, all_y_pred)

print(f"\n📊 Resultados Finais Globais:")
print(f"Acurácia: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Matriz de Confusão:\n{cm}")
