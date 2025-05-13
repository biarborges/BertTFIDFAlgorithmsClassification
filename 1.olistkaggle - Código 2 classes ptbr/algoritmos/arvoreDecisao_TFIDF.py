import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm  # Para a barra de progresso

# 1. Carregar dados
print("🔄 Carregando os dados...")
df = pd.read_csv("../corpus_tfidf.csv")

# 2. Separar as colunas
print("🔄 Separando as colunas de características e classe...")
X = df.drop(columns=['polarity']).values  # Os dados de características (TF-IDF)
y = df['polarity'].values  # A classe (polaridade)

# 3. Configurar StratifiedKFold
print("🔄 Configurando a validação cruzada (StratifiedKFold)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Armazenar previsões e verdadeiros
all_y_true = []
all_y_pred = []

# 5. Loop de validação cruzada com barra de progresso
print("🔄 Iniciando a validação cruzada...")
for train_idx, test_idx in tqdm(skf.split(X, y), total=5, desc="Validação Cruzada"):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 6. Treinar o modelo de Árvore de Decisão
    print("🔄 Treinando o modelo de Árvore de Decisão...")
    clf = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)

    # 7. Fazer previsões
    print("🔄 Realizando previsões...")
    y_pred = clf.predict(X_test)

    # 8. Armazenar as previsões e verdadeiros
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

# 9. Avaliação única global
print("🔄 Calculando as métricas de avaliação...")
acc = accuracy_score(all_y_true, all_y_pred)
f1 = f1_score(all_y_true, all_y_pred, average='weighted')
cm = confusion_matrix(all_y_true, all_y_pred)

# 10. Exibir os resultados finais
print(f"\n📊 Resultados Finais Globais:")
print(f"Acurácia: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Matriz de Confusão:\n{cm}")

# 11. Salvar a matriz de confusão colorida como imagem
print("🔄 Salvando a matriz de confusão como imagem...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Spring", xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"])
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.title("Matriz de Confusão")
plt.savefig("matriz_confusao.png")  # Salva como imagem
plt.close()  # Fecha a figura

print("✅ Matriz de Confusão salva como 'matriz_confusao.png'.")
