import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score

# 1. Carregar os dados do pickle
with open('corpus_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

# 'data' deve conter os embeddings (X) e as classes (y)
X = torch.tensor(data['embeddings'])  # Embeddings
y = torch.tensor(data['classes'])    # Classes

# 2. Definir o modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        output = self.fc(h_n[-1])  # Usando o último estado oculto
        return output

# 3. K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
f1_scores = []

# Loop para dividir os dados em K partes e treinar o modelo
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Treinando o modelo para o Fold {fold+1}")

    # Dividir os dados em treino e validação para esse fold
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Criar DataLoader para treinamento e validação
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    # Instanciar o modelo LSTM
    input_dim = X.shape[1]  # Número de features (embeddings de BERT)
    hidden_dim = 128
    output_dim = len(np.unique(y.numpy()))  # Número de classes
    
    model = LSTMModel(input_dim, hidden_dim, output_dim)
    
    # Definir os critérios de perda e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. Treinamento
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())  # Passar os embeddings pela LSTM
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

    # 5. Avaliação no conjunto de validação
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    accuracies.append(accuracy)
    f1_scores.append(f1)
    print(f"Fold {fold+1} - Acurácia: {accuracy:.4f}, F1 Score: {f1:.4f}")

# 6. Salvar as previsões no arquivo CSV
output_df = pd.DataFrame({
    'original_class': y_true,
    'predicted_class': y_pred
})

output_df.to_csv('predicted_classes.csv', index=False)
print("Previsões salvas em 'predicted_classes.csv'")

# 7. Exibir as métricas médias
print(f"Acurácia média: {np.mean(accuracies):.4f}")
print(f"F1 Score médio: {np.mean(f1_scores):.4f}")
