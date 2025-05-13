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

# 1. Carregar dados com gerenciamento de memória
def load_data_in_chunks(file_path, chunk_size=5000):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Processar em chunks
    num_samples = len(data['embedding'])
    for i in range(0, num_samples, chunk_size):
        chunk_emb = data['embedding'][i:i+chunk_size]
        chunk_pol = data['polarity'][i:i+chunk_size]
        
        # Converter para numpy se for pandas Series
        embeddings = [e.values if hasattr(e, 'values') else e for e in chunk_emb]
        polarities = [p.values if hasattr(p, 'values') else p for p in chunk_pol]
        
        yield np.stack(embeddings), np.array(polarities)
        gc.collect()

# Carregar e concatenar todos os chunks
print("🚀 Carregando dados em chunks...")
X_chunks, y_chunks = [], []
for X_chunk, y_chunk in load_data_in_chunks('../corpus_embeddingsLSTM.pkl'):
    X_chunks.append(torch.tensor(X_chunk, dtype=torch.float32))
    y_chunks.append(torch.tensor(y_chunk, dtype=torch.long))

X = torch.cat(X_chunks).to(device)
y = torch.cat(y_chunks).to(device)
del X_chunks, y_chunks
gc.collect()
print(f"✅ Dados carregados. X shape: {X.shape}, y shape: {y.shape}")

# 2. Dividir dados
X_train, X_test, y_train, y_test = train_test_split(
    X.cpu().numpy(), y.cpu().numpy(),
    test_size=0.2,
    stratify=y.cpu().numpy(),
    random_state=42
)

# Converter para tensores na GPU
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# 3. Modelo LSTM otimizado
class EfficientLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x)

# 4. Configuração do modelo
model = EfficientLSTM(X.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# 5. DataLoaders com batch size ajustável
batch_size = 16  # Reduza se necessário
train_loader = DataLoader(TensorDataset(X_train, y_train), 
                        batch_size=batch_size, 
                        shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), 
                       batch_size=batch_size*2, 
                       shuffle=False)

# 6. Treinamento com gradient accumulation
accumulation_steps = 4  # Simula batch maior sem consumir mais memória
print("\n🔥 Iniciando treinamento...")

for epoch in range(5):
    model.train()
    optimizer.zero_grad()
    
    for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Época {epoch+1}/5")):
        inputs = inputs.unsqueeze(1)
        outputs = model(inputs)
        loss = criterion(outputs, labels) / accumulation_steps
        loss.backward()
        
        if (i+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
    
    # Avaliação
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            y_pred.extend(torch.argmax(outputs, 1).cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Validação - Acurácia: {acc:.4f}, F1: {f1:.4f}")

# 7. Avaliação final e salvamento
print("\n🧪 Avaliação final no teste...")
model.eval()
y_test_pred, y_test_true = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.unsqueeze(1)
        outputs = model(inputs)
        y_test_pred.extend(torch.argmax(outputs, 1).cpu().numpy())
        y_test_true.extend(labels.cpu().numpy())

# 8. Salvar resultados
results_df = pd.DataFrame({
    'original': y_test_true,
    'predicted': y_test_pred
})
results_df.to_csv('resultados_finais_lstm.csv', index=False)
print("✅ Resultados salvos em 'resultados_finais_lstm.csv'")

# 9. Salvar modelo
torch.save(model.state_dict(), 'modelo_lstm_otimizado.pth')
print("✅ Modelo salvo em 'modelo_lstm_otimizado.pth'")

# 10. Métricas finais
print("\n📊 Métricas Finais:")
print(f"Acurácia: {accuracy_score(y_test_true, y_test_pred):.4f}")
print(f"F1 Score: {f1_score(y_test_true, y_test_pred, average='weighted'):.4f}")