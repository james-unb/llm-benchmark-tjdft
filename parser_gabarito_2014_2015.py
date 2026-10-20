import re
import pathlib

import pandas as pd
import pymupdf


# Pasta de entrada
pasta = pathlib.Path("database/gabaritos_2014_2015")


# Função para extrair os gabaritos
def extrair_gabarito(arquivo):

    # Extrai o texto do PDF
    with pymupdf.open(arquivo) as doc:
        conteudo = "\n".join(
            pagina.get_text() for pagina in doc
        )

    # Identifica o Ano e a aplicação pelo nome do arquivo
    nome = arquivo.stem

    resultado = re.search(
        r"gabarito_(\d{4})(?:_(\d+))?",
        nome
    )

    Ano = int(resultado.group(1))

    if resultado.group(2):
        Aplicacao = int(resultado.group(2))
    else:
        Aplicacao = 1

    # Seleciona a região relevante do documento
    if "QUESTÃO GABARITO" in conteudo:
        conteudo = conteudo.split(
            "QUESTÃO GABARITO", 1
        )[1]

        conteudo = conteudo.split(
            "2 DOS RECURSOS", 1
        )[0]

    # Extrai os números das questões
    numeros = [
        int(numero)
        for numero in re.findall(
            r"(?m)^\s*(\d{1,3})\s*$",
            conteudo
        )
        if 1 <= int(numero) <= 100
    ]

    # Extrai as respostas, incluindo X para questões anuladas
    respostas = re.findall(
        r"(?m)^\s*([A-E]|X)\s*$",
        conteudo
    )

    # Verifica se a quantidade de questões e respostas coincide
    if len(numeros) < 100 or len(respostas) < 100:
        raise ValueError(
            f"Não foi possível identificar 100 questões "
            f"e respostas em {arquivo.name}. "
            f"Números encontrados: {len(numeros)}; "
            f"respostas encontradas: {len(respostas)}."
        )

    # Seleciona as primeiras 100 questões e respostas
    numeros = numeros[:100]
    respostas = respostas[:100]

    # Cria os registros
    dados = []

    for numero, resposta in zip(numeros, respostas):

        dados.append({
            "Ano": Ano,
            "Aplicacao": Aplicacao,
            "Nquestao": numero,
            "gabarito": resposta
        })

    return dados


# Processa todos os PDFs da pasta
dados_gabaritos = []

for arquivo in sorted(pasta.glob("gabarito_*.pdf")):

    dados = extrair_gabarito(arquivo)

    dados_gabaritos.extend(dados)

    print(
        f"{arquivo.name}: "
        f"{len(dados)} questões extraídas."
    )


# Cria o DataFrame consolidado
df = pd.DataFrame(dados_gabaritos)

# Ordena os registros
df = df.sort_values(
    ["Ano", "Aplicacao", "Nquestao"]
).reset_index(drop=True)

# Salva o CSV consolidado
df.to_csv(
    "database_csv/gabaritos_completos.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nGabaritos extraídos:")
print(df.head(10))

print(
    f"\nTotal de registros extraídos: {len(df)}"
)