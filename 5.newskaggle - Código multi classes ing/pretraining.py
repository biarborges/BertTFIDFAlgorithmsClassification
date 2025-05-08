#pretraining.py

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
from preprocessing import *

def vectorize_with_bert(texts, max_length=128):

    model = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    

    try:
        embeddings = np.load('embeddings.npy', allow_pickle=True)
        print("Embeddings carregados do arquivo.")
        print(embeddings.shape)
    except FileNotFoundError:
        print("Arquivo de embeddings não encontrado. Vetorizando textos...")

        embeddings = []
        with torch.no_grad():
            for text in tqdm(texts, desc="Vetorizando textos"):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=max_length, add_special_tokens=True, return_attention_mask=True)
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding)
        embeddings = np.vstack(embeddings)
        

        np.save('embeddings.npy', embeddings)
        print("Embeddings salvos em embeddings.npy.")

    return embeddings

# Carrega e pré-processa os dados
file_path = 'corpus.csv'
texts, categories = load_data(file_path)
preprocessed_texts = preprocess_text(texts)
encoded_categories = encode_categories(categories)

# Vetoriza os textos
embeddings = vectorize_with_bert(preprocessed_texts)
