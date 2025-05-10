from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
import torch
from preprocessing import load_and_process_json
from tqdm import tqdm

# Carregar e processar os dados de treinamento
train_file_path = 'sentihood-train.json'
train_texts, train_encoded_sentiments, train_sentiment_classes = load_and_process_json(train_file_path)

# Carregar e processar os dados de teste
test_file_path = 'sentihood-test.json'
test_texts, test_encoded_sentiments, test_sentiment_classes = load_and_process_json(test_file_path)

# Tokenizar os textos usando BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')

train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_encoded_sentiments))
test_dataset = torch.utils.data.TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_encoded_sentiments))

# Definir o modelo BERT para classificação de sequência
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=len(train_sentiment_classes))

# Parâmetros de treinamento
batch_size = 16
epochs = 4
learning_rate = 2e-5

# DataLoader para treino e teste
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Configuração do otimizador, scheduler e função de perda
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = torch.nn.CrossEntropyLoss()

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
f1 = f1_score(true_labels, predictions, average='binary')

print("Bert:")
print(f"Acurácia: {accuracy}")
print(f"F1-score: {f1}")
