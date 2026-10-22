import re
import pathlib

import pandas as pd
import pymupdf


# Arquivo de entrada
pasta = "database/"
nomes = ["prova_2014", "prova_2015_1", "prova_2015_2"]

for nome in nomes:
    arquivo = pasta + nome + ".pdf"
    # Extração do texto do PDF
    with pymupdf.open(arquivo) as doc:
        conteudo = chr(12).join(page.get_text() for page in doc)


    # Salva o texto extraído
    #pathlib.Path(nome + ".txt").write_text(
    #    conteudo,
    #    encoding="utf-8"
    #)


    # Regex para identificar cada questão
    regex_questao = (
        r"(?ms)^QUESTÃO\s+(?P<numero>\d+)\s*\n"
        r"(?P<conteudo>.*?)(?=^QUESTÃO\s+\d+|\Z)"
    )


    # Regex para identificar as alternativas
    regex_alternativas = (
        r"(?ms)^(?P<alternativa>[A-E])\s*\n"
        r"(?P<texto>.*?)(?=^[A-E]\s*$|\Z)"
    )


    # Extrai as questões
    questoes = re.finditer(regex_questao, conteudo)

    dados_questoes = []

    for questao in questoes:

        numero = questao.group("numero")
        conteudo_questao = questao.group("conteudo")

        # Localiza o início da primeira alternativa
        inicio_alternativas = re.search(
            r"(?m)^[A-E]\s*$",
            conteudo_questao
        )

        if inicio_alternativas:
            enunciado = conteudo_questao[:inicio_alternativas.start()].strip()
            texto_alternativas = conteudo_questao[inicio_alternativas.start():]
        else:
            enunciado = conteudo_questao.strip()
            texto_alternativas = ""

        # Extrai as alternativas
        alternativas = re.finditer(
            regex_alternativas,
            texto_alternativas
        )

        registro = {
            "Nquestao": numero,
            "enunciado": enunciado
        }

        for alternativa in alternativas:
            letra = alternativa.group("alternativa")
            texto = alternativa.group("texto").strip()

            registro[letra] = texto

        dados_questoes.append(registro)


    # Cria o DataFrame
    df = pd.DataFrame(dados_questoes)

    # Garante a ordem das colunas
    df = df[
        ["Nquestao", "enunciado", "A", "B", "C", "D", "E"]
    ]

    df["questao_completa"] = (
        df["enunciado"] + "\n"
        + "A) " + df["A"] + "\n"
        + "B) " + df["B"] + "\n"
        + "C) " + df["C"] + "\n"
        + "D) " + df["D"] + "\n"
        + "E) " + df["E"]
    )

    # Salva o resultado
    df.to_csv("database_csv/" +
        nome + ".csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("Número de questões extraídas:", df.shape[0])

### Gabarito

# Pasta de entrada
pasta = "database/"
nomes = ["gabarito_2014", "gabarito_2015_1", "gabarito_2015_2"]


# Função para extrair os gabaritos
def extrair_gabarito(arquivo):

    # Extrai o texto do PDF
    with pymupdf.open(arquivo) as doc:
        conteudo = "\n".join(
            pagina.get_text() for pagina in doc
        )

    # Identifica o Ano e a aplicação pelo nome do arquivo

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

for nome in nomes:
    arquivo = pasta + nome + ".pdf"
    dados = extrair_gabarito(arquivo)

    dados_gabaritos.extend(dados)

    print(
        f"{arquivo}: "
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

# Pasta que contém os arquivos CSV
pasta = pathlib.Path("database_csv")

# Lista para armazenar os DataFrames
dados = []

# Percorre os arquivos CSV
for arquivo in sorted(pasta.glob("*.csv")):

    # Ignora o arquivo consolidado
    if arquivo.name == "provas_completas.csv":
        continue

    # Extrai o ano e a aplicação do nome do arquivo
    resultado = re.search(
        r"(\d{4})(?:_(\d+))?",
        arquivo.stem
    )

    if resultado is None:
        print(f"Aviso: ano não encontrado em {arquivo.name}")
        continue

    ano = int(resultado.group(1))

    # Se não houver aplicação no nome, utiliza 1
    aplicacao = int(resultado.group(2)) if resultado.group(2) else 1

    # Lê o arquivo CSV
    df = pd.read_csv(arquivo)

    # Adiciona as colunas Ano e Aplicacao
    df["Ano"] = ano
    df["Aplicacao"] = aplicacao

    # Armazena o DataFrame
    dados.append(df)

# Junta os arquivos
if dados:
    df_completo = pd.concat(dados, ignore_index=True)

    # Salva o resultado
    df_completo.to_csv(
        pasta / "provas_completas.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(df_completo.head())
    print(f"Total de questões: {len(df_completo)}")
else:
    print("Nenhum arquivo CSV válido foi encontrado.")


gabarito = pd.read_csv("database_csv/gabaritos_completos.csv")
print(gabarito.head())
print(df.head())

df_completo = df.merge(
    gabarito,
    on=["Nquestao", "Ano", "Aplicacao"],
    how="left"
)

df_completo.to_csv(
        pasta / "provas_gabaritos_completos.csv",
        index=False,
        encoding="utf-8-sig"
    )