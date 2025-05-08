import json

def contar_opinioes(arquivo):
    # Ler o arquivo JSON
    with open(arquivo, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    total_textos = len(dados)
    total_positive = 0
    total_negative = 0
    mais_de_uma_opiniao = 0

    for item in dados:
        num_opinioes = len(item['opinions'])
        
        if num_opinioes > 1:
            mais_de_uma_opiniao += 1

        for opiniao in item['opinions']:
            if opiniao['sentiment'] == 'Positive':
                total_positive += 1
            elif opiniao['sentiment'] == 'Negative':
                total_negative += 1

    return total_textos, total_positive, total_negative, mais_de_uma_opiniao

# Arquivos JSON
arquivo_train = 'sentihood-train.json'
arquivo_test = 'sentihood-test.json'

# Contar opiniões em cada arquivo
total_textos_train, total_positive_train, total_negative_train, mais_de_uma_opiniao_train = contar_opinioes(arquivo_train)
total_textos_test, total_positive_test, total_negative_test, mais_de_uma_opiniao_test = contar_opinioes(arquivo_test)

# Exibir os resultados
print(f'Arquivo: {arquivo_train}')
print(f'Total de textos: {total_textos_train}')
print(f'Total de opiniões Positive: {total_positive_train}')
print(f'Total de opiniões Negative: {total_negative_train}')
print(f'Total de itens com mais de uma opinião: {mais_de_uma_opiniao_train}\n')

print(f'Arquivo: {arquivo_test}')
print(f'Total de textos: {total_textos_test}')
print(f'Total de opiniões Positive: {total_positive_test}')
print(f'Total de opiniões Negative: {total_negative_test}')
print(f'Total de itens com mais de uma opinião: {mais_de_uma_opiniao_test}')
