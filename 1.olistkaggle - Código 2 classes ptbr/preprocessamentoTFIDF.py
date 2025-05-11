import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Baixar recursos do NLTK (apenas uma vez)
nltk.download('punkt')
nltk.download('stopwords')

# Defina o idioma: 'english' ou 'portuguese'
idioma = 'portuguese'

# Função de pré-processamento
def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Remover pontuação e caracteres especiais
    text = re.sub(r'[^\w\sçÇáàãâéêíóôõúüÁÀÃÂÉÊÍÓÔÕÚÜ]', '', text)

    # Normalizar espaços
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenização
    tokens = word_tokenize(text, language=idioma)

    # Stopwords
    stop_words = set(stopwords.words(idioma))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemização
    stemmer = SnowballStemmer(idioma)
    tokens = [stemmer.stem(word) for word in tokens]

    return tokens

# Caminhos
arquivo_csv = "../BertTFIDFAlgorithmsClassification/1.olistkaggle - Código 2 classes ptbr/olist.csv"
saida_csv = "../BertTFIDFAlgorithmsClassification/1.olistkaggle - Código 2 classes ptbr/corpus_processadoTFIDF.csv"

# Carregar CSV
df = pd.read_csv(arquivo_csv)

# Remover linhas sem polaridade
df = df.dropna(subset=['polarity'])

# Aplicar pré-processamento
df['review_text_processed'] = df['review_text_processed'].astype(str).apply(preprocess_text)

# Selecionar colunas finais
df_novo = df[['review_text_processed', 'polarity']]

# Salvar
df_novo.to_csv(saida_csv, index=False)

print(f"Arquivo processado salvo como: {saida_csv}")
