#preprocessing.py

import re
import json
from sklearn.preprocessing import LabelEncoder

# Função para pré-processar os textos
def preprocess_text(texts):
    preprocessed_texts = []
    for text in texts:
        # Remove caracteres especiais
        cleaned_text = re.sub(r'[^\w\s]', '', text)
        # Normaliza espaços em branco e remove espaços extras
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        preprocessed_texts.append(cleaned_text)
    return preprocessed_texts

# Função para transformar categorias em números usando LabelEncoder
def encode_categories(categories):
    encoder = LabelEncoder()
    encoded_categories = encoder.fit_transform(categories)
    return encoded_categories, encoder.classes_

# Função para carregar e processar dados de um arquivo JSON
def load_and_process_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    sentiments = []
    
    for entry in data:
        text = entry['text']
        for opinion in entry['opinions']:
            sentiment = opinion['sentiment']
            texts.append(text)
            sentiments.append(sentiment)
    
    preprocessed_texts = preprocess_text(texts)
    encoded_sentiments, sentiment_classes = encode_categories(sentiments)
    
    return preprocessed_texts, encoded_sentiments, sentiment_classes

# Exemplo de uso
if __name__ == "__main__":
    # Carregar e processar dados do JSON
    json_file_path = 'sentihood-test.json'  
    texts, encoded_sentiments, sentiment_classes = load_and_process_json(json_file_path)
    
    #print("Textos pré-processados:", texts)
    #print("Sentimentos codificados:", encoded_sentiments)
    #print("Classes de sentimentos:", sentiment_classes)
