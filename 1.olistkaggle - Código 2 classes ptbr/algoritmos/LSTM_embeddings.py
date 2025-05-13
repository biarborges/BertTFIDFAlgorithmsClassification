import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import gc

# Configurações iniciais
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📌 Usando dispositivo: {device}")

# 1. Carregar os dados de forma eficiente em memória
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Processar embeddings em chunks
    embeddings = []
    polarities = []
    
    for i in tqdm(range(len(data['embedding'])), desc="Carregando dados"):
        # Converter para numpy array se for Series do pandas
        emb = data['embedding'][i].values if hasattr(data['embedding'][i], 'values') else data['embedding'][i]
        pol = data['polarity'][i] if isinstance(data['polarity'], list) else data['polarity'].iloc[i]
        
        embeddings.append(emb)
        polarities.append(pol)
        
        # Liberar memória a cada 1000 amostras
        if i % 1000 == 0:
            gc.collect()
    
    return np.stack(embeddings), np.array(polarities)

try:
    print("🚀 Carregando dados...")
    X, y = load_data('../corpus_embeddingsLSTM.pkl')
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    print(f"✅ Dados carregados. X shape: {X.shape}, y shape: {y.shape}")
except Exception as e:
    print(f"❌ Erro ao carregar dados: {e}")
    exit()

# 2. Dividir dados - usar apenas train/test split (sem K-Fold para economizar memória)
X_train, X_test, y_train, y_test = train_test_split(
    X.cpu().numpy(), y.cpu().numpy(),
    test_size=0.2,
    stratify=y.cpu().numpy(),
    random_state=42
)

# Converter de volta para tensores
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# 3. Definir modelo LSTM mais leve
class LightLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Pegar apenas a última saída
        return self.fc(x)

# 4. Configurações de treino
input_dim = X.shape[1]
model = LightLSTMModel(input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 5. DataLoaders com batch size menor
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# 6. Treinamento
print("\n🔥 Começando treinamento...")
for epoch in range(5):
    model.train()
    for inputs, labels in tqdm(train_loader, desc=f"Época {epoch+1}/5"):
        inputs = inputs.unsqueeze(1)  # Adicionar dimensão temporal
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Avaliação
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Época {epoch+1} - Acurácia: {acc:.4f}, F1: {f1:.4f}")
    torch.cuda.empty_cache()

# 7. Resultados finais
print("\n📊 Resultados Finais:")
print(f"Acurácia no teste: {acc:.4f}")
print(f"F1 Score no teste: {f1:.4f}")

# Salvar modelo
torch.save(model.state_dict(), 'modelo_lstm_leve.pth')
print("✅ Modelo salvo com sucesso!")