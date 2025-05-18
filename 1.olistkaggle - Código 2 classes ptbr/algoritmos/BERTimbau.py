import optuna
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
import evaluate
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset

# Verificar se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Usando dispositivo: {device}")

# 1. Carregar o dataset bruto (exemplo CSV)
print("🔄 Carregando os dados...")
df = pd.read_csv("../corpus_processadoBERT_classesNumericas.csv", quotechar='"', encoding='utf-8')

# Supondo que o dataframe tem colunas 'review_text_processed' e 'polarity'
texts = df['review_text_processed'].fillna("").astype(str)
labels = df['polarity'].astype(int)

# 2. Dividir em treino (70%), validação (15%) e teste (15%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    texts, labels, test_size=0.15, stratify=labels, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42
)  # 0.1765 * 0.85 ≈ 0.15

# 3. Tokenizer
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

def tokenize_function(texts):
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

# 4. Tokenizar os splits e converter para datasets do HuggingFace
train_dataset = Dataset.from_dict({'text': X_train, 'labels': y_train})
val_dataset = Dataset.from_dict({'text': X_val, 'labels': y_val})
test_dataset = Dataset.from_dict({'text': X_test, 'labels': y_test})

train_dataset = train_dataset.map(lambda x: tokenize_function(x['text']), batched=True)
val_dataset = val_dataset.map(lambda x: tokenize_function(x['text']), batched=True)
test_dataset = test_dataset.map(lambda x: tokenize_function(x['text']), batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Métrica
metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='weighted')

# Função para inicializar o modelo
def model_init():
    return AutoModelForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased", num_labels=2).to(device)

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("per_device_train_batch_size", [4, 8])
    gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 4, 8, 16)
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 3, 4)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="no",  # **Não salvar checkpoints durante a busca**
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        logging_dir='./logs',
        logging_steps=50,
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none"
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result["eval_f1"]


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Melhores parâmetros:", study.best_params)

# Treinar novamente com os melhores parâmetros e salvar modelo e métricas
best_params = study.best_params

training_args = TrainingArguments(
    output_dir="./best_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Salva checkpoints aqui, pois é o treino final
    learning_rate=best_params['learning_rate'],
    per_device_train_batch_size=best_params['per_device_train_batch_size'],
    gradient_accumulation_steps=best_params['gradient_accumulation_steps'],
    num_train_epochs=best_params['num_train_epochs'],
    weight_decay=best_params['weight_decay'],
    logging_dir='./logs',
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    save_total_limit=1  # Para manter só 1 checkpoint
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
eval_result = trainer.evaluate()

print("Avaliação final no conjunto de validação:")
print(eval_result)

print("🔍 Avaliando no conjunto de teste...")
test_pred = trainer.predict(test_dataset)
y_pred = np.argmax(test_pred.predictions, axis=1)
y_true = np.array(test_pred.label_ids)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
cm = confusion_matrix(y_true, y_pred)

print(f"Acurácia (teste): {acc:.4f}")
print(f"F1-score (teste): {f1:.4f}")
print(f"Matriz de Confusão (teste):\n{cm}")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix - BERTimbau (15% Test)")
plt.savefig("MC_bertimbau.png")
plt.close()
print("✅ Matriz salva como 'MC_bertimbau.png'.")

