import pandas as pd
from transformers import BertTokenizer

df = pd.read_csv("corpus_processadoBERT.csv")
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
df['num_tokens'] = df['review_text_processed'].apply(lambda x: len(tokenizer.tokenize(str(x))))
acima_de_512 = df[df['num_tokens'] > 512]
print(f"Textos com mais de 512 tokens: {len(acima_de_512)}")
