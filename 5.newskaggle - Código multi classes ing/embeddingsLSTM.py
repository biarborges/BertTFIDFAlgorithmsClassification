import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# Caminho do CSV processado
csv_entrada = "corpus_processadoBERT_classesNumericas.csv"
csv_saida = "corpus_embeddingsLSTM.pkl"

# Carregar dados
df = pd.read_csv(csv_entrada)

# Modelo e tokenizer BERT (multilingual)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Número máximo de tokens por texto
MAX_LEN = 128  # Ajuste conforme necessário

def get_bert_sequence_embeddings(text):
    if not isinstance(text, str):
        text = str(text)
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_attention_mask=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        # Pega a sequência inteira de embeddings dos tokens (não só o CLS)
        sequence_output = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # shape: (MAX_LEN, 768)
        return sequence_output

# Aplicar com progresso
tqdm.pandas()
df['embedding'] = df['content'].progress_apply(get_bert_sequence_embeddings)

# Salvar como Pickle (mantém arrays grandes)
df[['embedding', 'category']].to_pickle(csv_saida)

print(f"Embeddings sequenciais salvos como: {csv_saida}")
