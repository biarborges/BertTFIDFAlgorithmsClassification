# Aplicar pré-processamento
df['noticia'] = df['noticia'].astype(str).apply(preprocess_text)

# Selecionar colunas finais
df_novo = df[['noticia', 'FakeTrue', 'categoria']]