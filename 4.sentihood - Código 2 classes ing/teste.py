import json

def contar_opinioes(arquivo):
    # Ler o arquivo JSON
    with open(arquivo, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    total_textos = len(dados)
    total_positive = 0
    total_negative = 0
    mais_de_uma_opiniao = 0
    apenas_uma_opiniao = 0

    for item in dados:
        num_opinioes = len(item['opinions'])

        if num_opinioes == 1:
            apenas_uma_opiniao += 1
        elif num_opinioes > 1:
            mais_de_uma_opiniao += 1

        for opiniao in item['opinions']:
            if opiniao['sentiment'] == 'Positive':
                total_positive += 1
            elif opiniao['sentiment'] == 'Negative':
                total_negative += 1

    return total_textos, total_positive, total_negative, mais_de_uma_opiniao, apenas_uma_opiniao

# Arquivos JSON
arquivo_train = '../BertTFIDFAlgorithmsClassification/4.sentihood - Código 2 classes ing/sentihood-train.json'
arquivo_test = '../BertTFIDFAlgorithmsClassification/4.sentihood - Código 2 classes ing/sentihood-test.json'
arquivo_dev = '../BertTFIDFAlgorithmsClassification/4.sentihood - Código 2 classes ing/sentihood-dev.json'

# Contar opiniões em cada arquivo
res_train = contar_opinioes(arquivo_train)
res_test = contar_opinioes(arquivo_test)
res_dev = contar_opinioes(arquivo_dev)

# Exibir os resultados
for nome, res in zip(
    ["train", "test", "dev"],
    [res_train, res_test, res_dev]
):
    total_textos, total_positive, total_negative, mais_de_uma, apenas_uma = res
    print(f'\nArquivo: {nome}')
    print(f'Total de textos: {total_textos}')
    print(f'Total de opiniões Positive: {total_positive}')
    print(f'Total de opiniões Negative: {total_negative}')
    print(f'Total de itens com mais de uma opinião: {mais_de_uma}')
    print(f'Total de itens com apenas uma opinião: {apenas_uma}')
