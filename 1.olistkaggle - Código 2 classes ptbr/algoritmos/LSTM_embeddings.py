import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# 1. Carregar os dados do pickle
with open('../corpus_embeddingsLSTM.pkl', 'rb') as f:
    data = pickle.load(f)

# Verificar a forma dos embeddings e garantir que todos têm o mesmo tamanho
first_embedding = data['embedding'][0]
embedding_shape = first_embedding.shape
assert all(e.shape == embedding_shape for e in data['embedding']), "Embeddings têm tamanhos diferentes!"

# Empilhar corretamente os embeddings em uma matriz 2D
X = torch.tensor(np.stack(data['embedding']), dtype=torch.float32)  # Embeddings
y = torch.tensor(np.array(data['polarity']), dtype=torch.long)  # Classes

# Verifique as formas de X e y para garantir que está correto
print(f"X shape: {X.shape}, y shape: {y.shape}")

# 2. Separar 15% dos dados para TESTE
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

# 3. K-Fold nos 85% restantes (X_temp)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
f1_scores = []

# 4. Definir o modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Passar as entradas pela LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Usar a última camada de saída da LSTM
        output = self.fc(h_n[-1])
        return output

# 5. Cross-validation nos 85%
for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp)):
    print(f"\n🔁 Treinando o modelo - Fold {fold+1}/5")

    X_train, X_val = X_temp[train_idx], X_temp[val_idx]
    y_train, y_val = y_temp[train_idx], y_temp[val_idx]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=8, shuffle=False)

    input_dim = X.shape[1]  # Número de características (dimensão dos embeddings)
    hidden_dim = 128  # Tamanho do vetor escondido da LSTM
    output_dim = len(torch.unique(y))  # Número de classes de saída

    model = LSTMModel(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 3
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Época {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs.float())  # Convertendo para float32
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Loss: {running_loss/len(train_loader):.4f}")

    # Validação
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="🔎 Validação"):
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracies.append(acc)
    f1_scores.append(f1)
    print(f"Fold {fold+1} - Acurácia: {acc:.4f}, F1: {f1:.4f}")

# 6. Treinar modelo final com todos os dados de treino+validação (X_temp) e avaliar no TESTE
print("\n🏁 Treinando modelo final para teste...")

final_model = LSTMModel(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)

final_loader = DataLoader(TensorDataset(X_temp, y_temp), batch_size=8, shuffle=True)
for epoch in range(epochs):
    final_model.train()
    for inputs, labels in tqdm(final_loader, desc=f"Final Época {epoch+1}/{epochs}"):
        optimizer.zero_grad()
        outputs = final_model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Avaliar no conjunto de TESTE
print("\n🧪 Avaliação no conjunto de TESTE")
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=8, shuffle=False)
y_test_pred, y_test_true = [], []

final_model.eval()
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="🔬 Testando"):
        outputs = final_model(inputs.float())
        _, predicted = torch.max(outputs, 1)
        y_test_pred.extend(predicted.cpu().numpy())
        y_test_true.extend(labels.cpu().numpy())

test_acc = accuracy_score(y_test_true, y_test_pred)
test_f1 = f1_score(y_test_true, y_test_pred, average='weighted')

# 7. Salvar CSV
df_out = pd.DataFrame({
    'original_class': y_test_true,
    'predicted_class': y_test_pred
})
df_out.to_csv('predicted_classes_test.csv', index=False)

# 8. Métricas
print(f"\n📊 MÉTRICAS FINAIS")
print(f"Acurácia média (validação K-Fold): {np.mean(accuracies):.4f}")
print(f"F1 médio (validação K-Fold): {np.mean(f1_scores):.4f}")
print(f"Acurácia no TESTE: {test_acc:.4f}")
print(f"F1 Score no TESTE: {test_f1:.4f}")
