import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# Caminho do CSV processado
csv_entrada = "../BertTFIDFAlgorithmsClassification/1.olistkaggle - Código 2 classes ptbr/corpus_processadoBERT.csv"
csv_saida = "../BertTFIDFAlgorithmsClassification/1.olistkaggle - Código 2 classes ptbr/olist_embeddings.csv"

# Carregar dados
df = pd.read_csv(csv_entrada)

# Carrega o modelo BERT pré-treinado (multilingual)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')
model.eval()

# Se tiver GPU e quiser usar:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Função para gerar o embedding do CLS token
def get_bert_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()  # [CLS] token
        return cls_embedding

# Gerar embeddings com barra de progresso
tqdm.pandas()
df['embedding'] = df['review_text_processed'].progress_apply(get_bert_embedding)

# Salvar em CSV (opcional: você pode salvar como pickle, que mantém arrays)
df[['embedding', 'polarity']].to_pickle(csv_saida.replace('.csv', '.pkl'))

print(f"Embeddings salvos como: {csv_saida.replace('.csv', '.pkl')}")
