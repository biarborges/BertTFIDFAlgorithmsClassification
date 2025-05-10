#pretraining.py

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
from preprocessing import load_and_process_json

def vectorize_with_bert(texts, max_length=128):
    model = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    try:
        embeddings = np.load('embeddings128test.npy', allow_pickle=True)
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
        
        np.save('embeddings128test.npy', embeddings)
        print("Embeddings salvos em embeddings128test.npy.")

    return embeddings

if __name__ == "__main__":
    # Carrega e processa os dados do JSON
    json_file_path = 'sentihood-test.json'  # Altere para o caminho do seu arquivo JSON
    preprocessed_texts, encoded_sentiments, sentiment_classes = load_and_process_json(json_file_path)
    
    # Vetoriza os textos
    embeddings = vectorize_with_bert(preprocessed_texts)
    
    # Exemplo de impressão de embeddings e sentimentos codificados
    print("Embeddings:", embeddings)
    print("Sentimentos codificados:", encoded_sentiments)
