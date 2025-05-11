import os
import pandas as pd

# Caminhos das pastas
fake_dir = "../BertTFIDFAlgorithmsClassification/2.fake.br - Código 2 classes ptbr/fake"
true_dir = "../BertTFIDFAlgorithmsClassification/2.fake.br - Código 2 classes ptbr/true"
fake_meta_dir = "../BertTFIDFAlgorithmsClassification/2.fake.br - Código 2 classes ptbr/fake-meta-information"
true_meta_dir = "../BertTFIDFAlgorithmsClassification/2.fake.br - Código 2 classes ptbr/true-meta-information"

# Listas para armazenar os dados
noticias = []
fake_true = []
categorias = []

# Lista para armazenar arquivos ausentes
arquivos_faltando = []

# Função para ler o conteúdo do arquivo
def ler_noticia(pasta, nome_arquivo):
    caminho_arquivo = os.path.join(pasta, nome_arquivo)
    if os.path.exists(caminho_arquivo):  # Verifica se o arquivo existe
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        arquivos_faltando.append(caminho_arquivo)  # Armazena o arquivo faltante
        return ""  # Retorna uma string vazia se o arquivo não for encontrado

# Função para ler a terceira linha do arquivo meta
def ler_categoria(meta_pasta, nome_arquivo):
    caminho_meta = os.path.join(meta_pasta, nome_arquivo.replace('.txt', '-meta.txt'))
    if os.path.exists(caminho_meta):  # Verifica se o arquivo meta existe
        with open(caminho_meta, 'r', encoding='utf-8') as f:
            linhas = f.readlines()
            return linhas[2].strip()  # Pega a terceira linha (index 2)
    else:
        arquivos_faltando.append(caminho_meta)  # Armazena o arquivo meta faltante
        return ""  # Retorna uma string vazia se o arquivo meta não for encontrado

# Lê os arquivos da pasta 'fake'
for i in range(1, 3603):
    # Lê a noticia e a categoria
    noticia = ler_noticia(fake_dir, f'{i}.txt')
    categoria = ler_categoria(fake_meta_dir, f'{i}.txt')
    
    noticias.append(noticia)
    fake_true.append('fake')
    categorias.append(categoria)

# Lê os arquivos da pasta 'true'
for i in range(1, 3603):
    # Lê a noticia e a categoria
    noticia = ler_noticia(true_dir, f'{i}.txt')
    categoria = ler_categoria(true_meta_dir, f'{i}.txt')
    
    noticias.append(noticia)
    fake_true.append('true')
    categorias.append(categoria)

# Cria um DataFrame com os dados
df = pd.DataFrame({
    'noticia': noticias,
    'FakeTrue': fake_true,
    'categoria': categorias
})

# Salva o DataFrame em um arquivo CSV
csv_saida = "../BertTFIDFAlgorithmsClassification/2.fake.br - Código 2 classes ptbr/fakebrJunto.csv"
df.to_csv(csv_saida, index=False)

print(f"CSV gerado com sucesso! Arquivo salvo em {csv_saida}")

# Relatório de arquivos faltando
if arquivos_faltando:
    print("\nArquivos ausentes:")
    for arquivo in arquivos_faltando:
        print(arquivo)
else:
    print("\nNenhum arquivo ausente encontrado.")
