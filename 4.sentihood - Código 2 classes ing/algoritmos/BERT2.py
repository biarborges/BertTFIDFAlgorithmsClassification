import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, BertForSequenceClassification
from datasets import Dataset

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Usando dispositivo: {device}")

# Caminhos
modelo_path = "/home/ubuntu/BertTFIDFAlgorithmsClassification/4.sentihood - Código 2 classes ing/algoritmos/best_model/checkpoint-318"
csv_path = "../corpus_processadoBERT_classesNumericas.csv"  # ajuste se necessário

# Carregar tokenizer original (bert-base-cased) e modelo salvo
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = BertForSequenceClassification.from_pretrained(modelo_path).to(device)
model.eval()

# Carregar dados
df = pd.read_csv(csv_path, quotechar='"', encoding='utf-8')
texts = df['text'].fillna("").astype(str)
labels = df['sentiment'].astype(int)

# Dividir dados (apenas replicando a divisão anterior)
from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(
    texts, labels, test_size=0.15, stratify=labels, random_state=42
)

# Tokenizar dados de teste
def tokenize_function(texts):
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

test_dataset = Dataset.from_dict({'text': X_test, 'labels': y_test})
test_dataset = test_dataset.map(lambda x: tokenize_function(x['text']), batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Inference manual
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in torch.utils.data.DataLoader(test_dataset, batch_size=8):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
cm = confusion_matrix(all_labels, all_preds)


print(f"Acurácia (teste): {acc:.4f}")
print(f"F1-score (teste): {f1:.4f}")
print(f"Matriz de Confusão (teste):\n{cm}")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix - BERT (15% Test)")
plt.savefig("MC_bert.png")
plt.close()
print("✅ Matriz salva como 'MC_bert.png'.")

