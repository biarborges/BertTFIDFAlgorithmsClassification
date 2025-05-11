import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Caminho do CSV processado
csv_entrada = "corpus_processadoBERT.csv"
csv_saida = "corpus_embeddings.csv"

# Carregar dados
df = pd.read_csv(csv_entrada)

# Carrega o modelo BERT pré-treinado (multilingual)
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
model.eval()

# Se tiver GPU e quiser usar:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Função para gerar o embedding do CLS token
def get_bert_embedding(text):
    if not isinstance(text, str):  # Garante que o texto é string
        text = str(text)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()  # [CLS] token
        return cls_embedding



# Gerar embeddings com barra de progresso
tqdm.pandas()
df['embedding'] = df['noticia'].progress_apply(get_bert_embedding)

# Salvar em CSV (opcional: você pode salvar como pickle, que mantém arrays)
df[['embedding', 'FakeTrue']].to_pickle(csv_saida.replace('.csv', '.pkl'))

print(f"Embeddings salvos como: {csv_saida.replace('.csv', '.pkl')}")
