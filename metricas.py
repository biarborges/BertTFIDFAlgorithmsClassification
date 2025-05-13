import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# 1. Carregar o arquivo de resultados
df = pd.read_csv("resultado_classificacao.csv")

# 2. Separar as classes originais e preditas
y_true = df['original_class']
y_pred = df['predicted_class']

# 3. Calcular as métricas de avaliação
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')

# 4. Imprimir as métricas
print(f"Acurácia: {accuracy:.4f}")
print(f"F1-score (Macro): {f1:.4f}")

# 5. Relatório de Classificação
print("Relatório de Classificação:")
print(classification_report(y_true, y_pred))

# 6. Matriz de Confusão
print("Matriz de Confusão:")
print(confusion_matrix(y_true, y_pred))
