from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from preprocessing import *
import numpy as np
from tqdm import tqdm

# Carregar os dados
file_path = 'corpus.csv'
texts, categories = load_data(file_path)
preprocessed_texts = preprocess_text(texts)
encoded_categories, _ = encode_categories(categories)


# Dividir dados em treino, validação e teste
train_texts, temp_texts, train_labels, temp_labels = train_test_split(preprocessed_texts, encoded_categories, test_size=0.2, random_state=42)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.1, random_state=42)

# Tokenizar os textos usando BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=True)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')

train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
val_dataset = torch.utils.data.TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val_labels))
test_dataset = torch.utils.data.TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels))

# Definir o modelo BERT para classificação de sequência
model = BertForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased', num_labels=len(set(encoded_categories)))

# Parâmetros de treinamento
batch_size = 16
epochs = 1
learning_rate = 2e-5

# DataLoader para treino, validação e teste
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Configuração do otimizador, scheduler e função de perda
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = torch.nn.CrossEntropyLoss()

# Classe para early stopping
# class EarlyStopping:
#     def __init__(self, patience=3, min_delta=0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_loss = None

#     def __call__(self, val_loss):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#         elif val_loss < self.best_loss - self.min_delta:
#             self.best_loss = val_loss
#             self.counter = 0
#         else:
#             self.counter += 1
#         return self.counter >= self.patience

# early_stopping = EarlyStopping(patience=3)

# Treinamento do modelo BERT
model.train()
for epoch in range(epochs):
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True)
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'Training Loss': total_loss / len(progress_bar)})
    print(f"Epoch {epoch + 1}, Loss: {total_loss}")

    # Avaliação no conjunto de validação
    model.eval()
    val_loss = 0
    val_predictions = []
    val_true_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            logits = outputs.logits
            val_predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())
            val_true_labels.extend(labels.cpu().numpy())
    val_loss /= len(val_loader)
    val_accuracy = accuracy_score(val_true_labels, val_predictions)
    val_f1 = f1_score(val_true_labels, val_predictions, average='weighted')
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Validation F1-score: {val_f1}")

    # if early_stopping(val_loss):
    #     print("Early stopping")
    #     break

# Avaliação do modelo no conjunto de teste
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Evaluating', leave=True):
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Avaliação de desempenho
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average='weighted')

print(f"Acurácia: {accuracy}")
print(f"F1-score: {f1}")
