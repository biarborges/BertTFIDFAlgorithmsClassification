import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
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

# 1. Carregar os dados do pickle de forma eficiente
def load_data_in_batches(file_path, batch_size=1000):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Verificar e converter os embeddings
    first_embedding = data['embedding'][0]
    if hasattr(first_embedding, 'values'):  # Se for uma Series do pandas
        first_embedding = first_embedding.values
    embedding_shape = first_embedding.shape
    
    # Processar em lotes
    num_samples = len(data['embedding'])
    for i in range(0, num_samples, batch_size):
        # Converter embeddings para numpy array se forem Series
        batch_embeddings = [e.values if hasattr(e, 'values') else e for e in data['embedding'][i:i+batch_size]]
        batch_embeddings = np.stack(batch_embeddings)
        
        # Converter polaridades para numpy array se forem Series
        batch_polarity = data['polarity'][i:i+batch_size]
        if hasattr(batch_polarity, 'values'):  # Se for uma Series
            batch_polarity = batch_polarity.values
        batch_polarity = np.array(batch_polarity)
        
        # Verificar consistência
        assert all((e.shape == embedding_shape) for e in batch_embeddings), "Embeddings têm tamanhos diferentes!"
        
        # Converter para tensores
        X_batch = torch.tensor(batch_embeddings, dtype=torch.float32)
        y_batch = torch.tensor(batch_polarity, dtype=torch.long)
        
        yield X_batch, y_batch

# Carregar todos os dados
try:
    print("🚀 Carregando dados...")
    X_list, y_list = [], []
    for X_batch, y_batch in load_data_in_batches('../corpus_embeddingsLSTM.pkl'):
        X_list.append(X_batch)
        y_list.append(y_batch)
    
    X = torch.cat(X_list).float().to(device)
    y = torch.cat(y_list).long().to(device)
    del X_list, y_list
    gc.collect()
    
    print(f"✅ Dados carregados. X shape: {X.shape}, y shape: {y.shape}")
    
except RuntimeError as e:
    print(f"❌ Erro de memória: {e}")
    print("⚠️ Considere reduzir o batch_size ou usar um subconjunto dos dados")
    exit()

# 2. Separar 15% dos dados para TESTE
X_temp, X_test, y_temp, y_test = train_test_split(
    X.cpu().numpy(), y.cpu().numpy(), 
    test_size=0.15, 
    stratify=y.cpu().numpy(), 
    random_state=42
)

# Converter de volta para tensores e enviar para o dispositivo
X_temp = torch.tensor(X_temp).float().to(device)
X_test = torch.tensor(X_test).float().to(device)
y_temp = torch.tensor(y_temp).long().to(device)
y_test = torch.tensor(y_test).long().to(device)

# 3. K-Fold nos 85% restantes
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 4. Definir um modelo LSTM eficiente
class EfficientLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])
        return self.fc(out)

# 5. Configurações de treino
input_dim = X.shape[1]
hidden_dim = 64  # Reduzido para economizar memória
output_dim = len(torch.unique(y))
batch_size = 32  # Aumentado para melhor utilização da GPU

# 6. Cross-validation
accuracies, f1_scores = [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp.cpu().numpy())):
    print(f"\n🔁 Fold {fold+1}/5")
    
    # Criar DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_temp.cpu().numpy()[train_idx]).float().to(device),
        torch.tensor(y_temp.cpu().numpy()[train_idx]).long().to(device)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_temp.cpu().numpy()[val_idx]).float().to(device),
        torch.tensor(y_temp.cpu().numpy()[val_idx]).long().to(device)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
    
    # Modelo, otimizador e critério
    model = EfficientLSTMModel(input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Treino
    for epoch in range(3):
        model.train()
        epoch_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f"Época {epoch+1}/3"):
            inputs = inputs.unsqueeze(1)  # Adicionar dimensão temporal
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Loss médio: {epoch_loss/len(train_loader):.4f}")
    
    # Validação
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    fold_acc = accuracy_score(y_true, y_pred)
    fold_f1 = f1_score(y_true, y_pred, average='weighted')
    accuracies.append(fold_acc)
    f1_scores.append(fold_f1)
    print(f"📊 Fold {fold+1} - Acurácia: {fold_acc:.4f}, F1: {fold_f1:.4f}")
    
    # Limpar memória
    del model, train_loader, val_loader, train_dataset, val_dataset
    torch.cuda.empty_cache()
    gc.collect()

# 7. Treino final com todos os dados de treino+validação
print("\n🏁 Treinando modelo final com todos os dados...")

# Criar DataLoader para todos os dados de treino+validação
full_train_dataset = TensorDataset(X_temp, y_temp)
full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)

# Criar modelo final
final_model = EfficientLSTMModel(input_dim, hidden_dim, output_dim).to(device)
optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Treinar modelo final
for epoch in range(5):  # Um pouco mais de épocas para o modelo final
    final_model.train()
    epoch_loss = 0
    for inputs, labels in tqdm(full_train_loader, desc=f"Época Final {epoch+1}/5"):
        inputs = inputs.unsqueeze(1)
        optimizer.zero_grad()
        outputs = final_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Loss médio final: {epoch_loss/len(full_train_loader):.4f}")

# 8. Avaliação no conjunto de TESTE
print("\n🧪 Avaliando no conjunto de TESTE...")
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False)

final_model.eval()
y_test_pred, y_test_true = [], []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testando"):
        inputs = inputs.unsqueeze(1)
        outputs = final_model(inputs)
        preds = torch.argmax(outputs, dim=1)
        y_test_pred.extend(preds.cpu().numpy())
        y_test_true.extend(labels.cpu().numpy())

# Calcular métricas finais
test_acc = accuracy_score(y_test_true, y_test_pred)
test_f1 = f1_score(y_test_true, y_test_pred, average='weighted')

# 9. Salvar resultados
results_df = pd.DataFrame({
    'original': y_test_true,
    'predicted': y_test_pred
})
results_df.to_csv('resultados_finais.csv', index=False)

# 10. Métricas finais
print("\n📊 MÉTRICAS FINAIS:")
print(f"Acurácia média (validação): {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"F1 médio (validação): {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Acurácia no TESTE: {test_acc:.4f}")
print(f"F1 Score no TESTE: {test_f1:.4f}")

# Salvar o modelo final se necessário
torch.save(final_model.state_dict(), 'modelo_final.pth')
print("✅ Treinamento concluído e modelo salvo!")