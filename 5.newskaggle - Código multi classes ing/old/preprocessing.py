#preprocessing.py

import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder

# Função para carregar os dados do arquivo CSV
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df['content'].tolist(), df['category'].tolist()

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
