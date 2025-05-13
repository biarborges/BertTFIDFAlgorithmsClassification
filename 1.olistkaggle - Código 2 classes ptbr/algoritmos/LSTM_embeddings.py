import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import gc

# 1. Dataset que carrega um item por vez
class MemoryMapDataset(Dataset):
    def __init__(self, pkl_path):
        self.pkl_path = pkl_path
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            self.length = len(data['embedding'])
            self.embedding_shape = data['embedding'][0].shape
            # Carrega apenas os índices das polaridades
            self.polarities = data['polarity'].values if hasattr(data['polarity'], 'values') else np.array(data['polarity'])
            del data
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with open(self.pkl_path, 'rb') as f:
            data = pickle.load(f)
            emb = data['embedding'][idx]
            emb = emb.values if hasattr(emb, 'values') else emb
            pol = self.polarities[idx]
            return torch.tensor(emb, dtype=torch.float32), torch.tensor(pol, dtype=torch.long)

# 2. Configurações iniciais
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
print(f"📌 Usando dispositivo: {device}")

# 3. Carregar dataset
print("🚀 Inicializando dataset...")
dataset = MemoryMapDataset('../corpus_embeddingsLSTM.pkl')

# 4. Dividir índices (economiza memória)
indices = list(range(len(dataset)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

# 5. Modelo LSTM mínimo
class NanoLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 16, batch_first=True)  # Apenas 16 unidades
        self.fc = nn.Linear(16, 2)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])

# 6. Configuração de treino
model = NanoLSTM(dataset.embedding_shape[0]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 7. DataLoaders com batch mínimo
train_loader = DataLoader(
    dataset,
    batch_size=4,  # Batch muito pequeno
    sampler=torch.utils.data.SubsetRandomSampler(train_idx),
    num_workers=0
)

test_loader = DataLoader(
    dataset,
    batch_size=8,
    sampler=torch.utils.data.SubsetRandomSampler(test_idx),
    num_workers=0
)

# 8. Treinamento com acumulação de gradientes
print("\n🔥 Treinamento iniciado (paciente, está rodando em modo econômico)...")
accumulation_steps = 8  # Acumula gradientes para simular batch maior

for epoch in range(3):
    model.train()
    optimizer.zero_grad()
    
    for i, (X, y) in enumerate(tqdm(train_loader, desc=f"Época {epoch+1}/3")):
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        loss = criterion(outputs, y) / accumulation_steps
        loss.backward()
        
        if (i+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
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

# 9. Avaliação final
print("\n🧪 Avaliação final...")
model.eval()
y_test_true, y_test_pred = [], []
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        y_test_pred.extend(torch.argmax(outputs, 1).cpu().numpy())
        y_test_true.extend(y.cpu().numpy())

# 10. Salvar resultados
results_df = pd.DataFrame({
    'original': y_test_true,
    'predicted': y_test_pred
})
results_df.to_csv('resultados_lstm_economico.csv', index=False)
print("✅ Resultados salvos em 'resultados_lstm_economico.csv'")

# 11. Salvar modelo (opcional, consome memória)
try:
    torch.save(model.state_dict(), 'modelo_lstm_economico.pth')
    print("✅ Modelo salvo em 'modelo_lstm_economico.pth'")
except:
    print("⚠️ Não foi possível salvar o modelo (falha de memória)")

# 12. Métricas finais
print("\n📊 Resultados Finais:")
print(f"Acurácia: {accuracy_score(y_test_true, y_test_pred):.4f}")
print(f"F1 Score: {f1_score(y_test_true, y_test_pred, average='weighted'):.4f}")