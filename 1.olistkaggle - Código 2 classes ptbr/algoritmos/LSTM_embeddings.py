import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import gc

# 1. Dataset seguro para memória
class SafeDataset(Dataset):
    def __init__(self, pkl_path):
        self.pkl_path = pkl_path
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            self.length = len(data['embedding'])
            self.embedding_shape = data['embedding'][0].shape
            self.polarities = data['polarity'].values if hasattr(data['polarity'], 'values') else np.array(data['polarity'])
            del data
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with open(self.pkl_path, 'rb') as f:
            data = pickle.load(f)
            emb = data['embedding'][idx]
            emb = emb.values if hasattr(emb, 'values') else emb
            return torch.tensor(emb, dtype=torch.float32), torch.tensor(self.polarities[idx], dtype=torch.long)

# 2. Modelo LSTM com tratamento de dimensões
class SafeLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 16, batch_first=True)
        self.fc = nn.Linear(16, 2)
    
    def forward(self, x):
        # Garante as dimensões corretas [batch, seq_len, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Adiciona dimensão temporal
        elif x.dim() > 3:
            x = x.view(x.size(0), 1, -1)  # Redimensiona corretamente
        
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])

# 3. Configurações
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
print(f"📌 Usando dispositivo: {device}")

# 4. Carregar dados
print("🚀 Carregando dataset...")
try:
    dataset = SafeDataset('../corpus_embeddingsLSTM.pkl')
    print(f"✅ Dataset carregado com {len(dataset)} amostras")
except Exception as e:
    print(f"❌ Erro ao carregar dataset: {e}")
    exit()

# 5. Divisão dos dados
indices = list(range(len(dataset)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

# 6. DataLoaders com verificação de dimensões
def create_loader(dataset, indices, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(indices),
        num_workers=0,
        drop_last=True  # Evita batches incompletos
    )

train_loader = create_loader(dataset, train_idx, 8)
test_loader = create_loader(dataset, test_idx, 16)

# 7. Treinamento com verificação de dimensões
model = SafeLSTM(dataset.embedding_shape[0]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("\n🔥 Iniciando treinamento...")
for epoch in range(3):
    model.train()
    for X, y in tqdm(train_loader, desc=f"Época {epoch+1}/3"):
        try:
            X, y = X.to(device), y.to(device)
            
            # Verificação adicional de dimensões
            if X.dim() != 3:
                X = X.view(X.size(0), 1, -1)
            
            outputs = model(X)
            loss = criterion(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        except Exception as e:
            print(f"\n⚠️ Erro durante o treinamento: {e}")
            print(f"Dimensões do batch: {X.shape}")
            continue
    
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

# 8. Salvamento seguro
try:
    results_df = pd.DataFrame({
        'original': y_true,
        'predicted': y_pred
    })
    results_df.to_csv('resultados_seguros.csv', index=False)
    print("\n✅ Resultados salvos em 'resultados_seguros.csv'")
except Exception as e:
    print(f"\n⚠️ Erro ao salvar resultados: {e}")