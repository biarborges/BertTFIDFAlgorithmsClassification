import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm  # Para a barra de progresso

# 1. Carregar dados
print("🔄 Carregando os dados...")
df = pd.read_csv("../corpus_tfidf.csv")
df = df.dropna(subset=["review_text_str", "polarity"])

# Carregar o TF-IDF já pronto e as categorias
X = df.drop(columns=["polarity"]).values  # A partir das colunas TF-IDF
y = df["polarity"].values

# 2. Configurar StratifiedKFold
print("🔄 Configurando a validação cruzada com StratifiedKFold...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Armazenar previsões e verdadeiros
all_y_true = []
all_y_pred = []

# 4. Loop de validação cruzada com barra de progresso
print("🔄 Iniciando validação cruzada...")
for fold, (train_idx, test_idx) in enumerate(tqdm(skf.split(X, y), desc="Processando folds", total=5, unit="fold")):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Treinando o classificador
    clf = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    
    # Fazendo as previsões
    y_pred = clf.predict(X_test)

    # Armazenando as previsões e os valores reais
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

# 5. Avaliação única global
print("\n🔄 Calculando resultados finais...")
acc = accuracy_score(all_y_true, all_y_pred)
f1 = f1_score(all_y_true, all_y_pred, average='weighted')
cm = confusion_matrix(all_y_true, all_y_pred)

# Exibindo os resultados
print(f"\n📊 Resultados Finais Globais:")
print(f"Acurácia: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Matriz de Confusão:\n{cm}")
