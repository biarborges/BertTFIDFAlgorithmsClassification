import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import gc

# 1. Dataset otimizado
class EfficientDataset(Dataset):
    def __init__(self, pkl_path):
        self.pkl_path = pkl_path
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            self.length = len(data['embedding'])
            self.embedding_shape = data['embedding'][0].shape
            self.polarities = data['polarity'].values if hasattr(data['polarity'], 'values') else data['polarity']
            del data
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with open(self.pkl_path, 'rb') as f:
            data = pickle.load(f)
            emb = data['embedding'][idx]
            emb = emb.values if hasattr(emb, 'values') else emb
            return torch.tensor(emb, dtype=torch.float32), torch.tensor(self.polarities[idx], dtype=torch.long)

# 2. Modelo LSTM corrigido
class FixedLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 16, batch_first=True)
        self.fc = nn.Linear(16, 2)
    
    def forward(self, x):
        # Garante que a entrada seja 3D [batch, seq_len, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Adiciona dimensão temporal se necessário
        elif x.dim() > 3:
            x = x.squeeze()  # Remove dimensões extras
        
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])

# 3. Configurações
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
print(f"📌 Usando dispositivo: {device}")

print("🚀 Carregando dados...")
dataset = EfficientDataset('../corpus_embeddingsLSTM.pkl')

# 4. Divisão dos dados
indices = list(range(len(dataset)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

# 5. DataLoaders
train_loader = DataLoader(
    dataset,
    batch_size=8,
    sampler=torch.utils.data.SubsetRandomSampler(train_idx),
    num_workers=0
)

test_loader = DataLoader(
    dataset,
    batch_size=16,
    sampler=torch.utils.data.SubsetRandomSampler(test_idx),
    num_workers=0
)

# 6. Treinamento
model = FixedLSTM(dataset.embedding_shape[0]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("\n🔥 Treinamento iniciado...")
for epoch in range(3):
    model.train()
    optimizer.zero_grad()
    
    for i, (X, y) in enumerate(tqdm(train_loader, desc=f"Época {epoch+1}/3")):
        X, y = X.to(device), y.to(device)
        
        # Verificação de dimensões
        if X.dim() not in [2, 3]:
            X = X.view(X.size(0), -1).unsqueeze(1)
        
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % 100 == 0:
            torch.cuda.empty_cache()
    
    # Validação
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            y_pred.extend(torch.argmax(outputs, 1).cpu().numpy())
            y_true.extend(y.cpu().numpy())
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Validação - Acurácia: {acc:.4f}, F1: {f1:.4f}")

# 7. Salvamento
results_df = pd.DataFrame({
    'original': y_true,
    'predicted': y_pred
})
results_df.to_csv('resultados_lstm_final.csv', index=False)
print("\n✅ Resultados salvos em 'resultados_lstm_final.csv'")