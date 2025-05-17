import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carregar os dados
print("🔄 Carregando os dados...")
df = pd.read_pickle("../corpus_embeddings.pkl")

# 2. Separar embeddings e classes
X = np.vstack(df['embedding'].values)
y = df['FakeTrue'].values

# 3. Dividir em treino (85%) e teste (15%)
print("🔄 Dividindo dados em treino (85%) e teste (15%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# 4. Treinar o modelo Naive Bayes
print("🔄 Treinando modelo Gaussian Naive Bayes...")
clf = GaussianNB()
clf.fit(X_train, y_train)

# 5. Avaliação no conjunto de teste
print("🔍 Avaliando no conjunto de teste...")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print(f"Acurácia: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Matriz de Confusão:\n{cm}")

# 6. Matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=["Fake", "True"], yticklabels=["Fake", "True"])
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix - GaussianNB Embeddings (15% Test)")
plt.savefig("MC_naivebayes_embeddings.png")
plt.close()
print("✅ Matriz salva como 'MC_naivebayes_embeddings.png'.")
