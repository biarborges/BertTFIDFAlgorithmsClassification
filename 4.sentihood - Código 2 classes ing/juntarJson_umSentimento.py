import json
import pandas as pd

# Lista dos arquivos de entrada
arquivos_json = [
    '../BertTFIDFAlgorithmsClassification/4.sentihood - Código 2 classes ing/sentihood-train.json',
    '../BertTFIDFAlgorithmsClassification/4.sentihood - Código 2 classes ing/sentihood-test.json',
    '../BertTFIDFAlgorithmsClassification/4.sentihood - Código 2 classes ing/sentihood-dev.json'
]

# Lista para armazenar os dados processados
dados_filtrados = []

# Processar cada arquivo
for arquivo in arquivos_json:
    with open(arquivo, 'r', encoding='utf-8') as f:
        dados = json.load(f)
        for item in dados:
            opinions = item.get("opinions", [])
            text = item.get("text", "").strip()
            if not opinions:
                continue  # ignora textos sem opinião
            if len(opinions) == 1:
                sentimento = opinions[0]['sentiment']
            else:
                sentimentos = [op['sentiment'] for op in opinions]
                if 'Negative' in sentimentos:
                    sentimento = 'Negative'
                else:
                    sentimento = sentimentos[0]
            dados_filtrados.append({
                'text': text,
                'sentiment': sentimento
            })

# Converter para DataFrame
df = pd.DataFrame(dados_filtrados)

# Salvar em CSV
caminho_csv = '../BertTFIDFAlgorithmsClassification/4.sentihood - Código 2 classes ing/sentihood2.csv'
df.to_csv(caminho_csv, index=False, encoding='utf-8')

print(f'Total de exemplos salvos: {len(df)}')
print(f'CSV salvo em: {caminho_csv}')
