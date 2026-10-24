import re
import pathlib

import pandas as pd
import pymupdf


# Pastas
PASTA_PDF = pathlib.Path("database")
PASTA_CSV = pathlib.Path("database_csv")
PASTA_CSV.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# 1) Provas
# ---------------------------------------------------------------
nomes = ["prova_2014", "prova_2015_1", "prova_2015_2"]

for nome in nomes:
    arquivo = PASTA_PDF / f"{nome}.pdf"
    if not arquivo.exists():
        print(f"[aviso] não encontrado: {arquivo}")
        continue

    # Extração do texto do PDF
    with pymupdf.open(arquivo) as doc:
        conteudo = "\n\n".join(page.get_text() for page in doc)

    # Normaliza quebras de linha (não afeta os regex, só evita \r\n)
    conteudo = conteudo.replace("\r\n", "\n")

    # Regex para identificar cada questão (INALTERADO)
    regex_questao = (
        r"(?ms)^QUESTÃO\s+(?P<numero>\d+)\s*\n"
        r"(?P<conteudo>.*?)(?=^QUESTÃO\s+\d+|\Z)"
    )

    # Regex para identificar as alternativas (INALTERADO)
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
            conteudo_questao,
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
            texto_alternativas,
        )

        registro = {
            "Nquestao": numero,
            "enunciado": enunciado,
        }

        for alternativa in alternativas:
            letra = alternativa.group("alternativa")
            texto = alternativa.group("texto").strip()
            registro[letra] = texto

        dados_questoes.append(registro)

    # Cria o DataFrame
    df = pd.DataFrame(dados_questoes)

    # Garante a ordem e a existência das colunas A–E
    df = df.reindex(columns=["Nquestao", "enunciado", "A", "B", "C", "D", "E"])

    # Evita quebra por NaN na concatenação
    df["questao_completa"] = (
        df["enunciado"].fillna("") + "\n"
        + "A) " + df["A"].fillna("") + "\n"
        + "B) " + df["B"].fillna("") + "\n"
        + "C) " + df["C"].fillna("") + "\n"
        + "D) " + df["D"].fillna("") + "\n"
        + "E) " + df["E"].fillna("")
    )

    df.to_csv(
        PASTA_CSV / f"{nome}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"{arquivo}: Número de questões extraídas: {df.shape[0]}")


# ---------------------------------------------------------------
# 2) Gabaritos
# ---------------------------------------------------------------
nomes = ["gabarito_2014", "gabarito_2015_1", "gabarito_2015_2"]


# Função para extrair os gabaritos
def extrair_gabarito(arquivo, nome):
    # Extrai o texto do PDF
    with pymupdf.open(arquivo) as doc:
        conteudo = "\n".join(
            pagina.get_text() for pagina in doc
        )

    # Identifica o Ano e a aplicação pelo nome do arquivo
    resultado = re.search(
        r"gabarito_(\d{4})(?:_(\d+))?",
        nome,
    )

    if resultado is None:
        raise ValueError(f"Não foi possível identificar ano em: {nome}")

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
            conteudo,
        )
        if 1 <= int(numero) <= 100
    ]

    # Extrai as respostas, incluindo X para questões anuladas
    respostas = re.findall(
        r"(?m)^\s*([A-E]|X)\s*$",
        conteudo,
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
            "gabarito": resposta,
        })

    return dados


# Processa todos os PDFs da pasta
dados_gabaritos = []

for nome in nomes:
    arquivo = PASTA_PDF / f"{nome}.pdf"
    if not arquivo.exists():
        print(f"[aviso] não encontrado: {arquivo}")
        continue

    dados = extrair_gabarito(arquivo, nome)
    dados_gabaritos.extend(dados)

    print(f"{arquivo}: {len(dados)} questões extraídas.")


# Cria o DataFrame consolidado
df_gabaritos = pd.DataFrame(dados_gabaritos)

if not df_gabaritos.empty:
    df_gabaritos = df_gabaritos.sort_values(
        ["Ano", "Aplicacao", "Nquestao"]
    ).reset_index(drop=True)

    df_gabaritos.to_csv(
        PASTA_CSV / "gabaritos_completos.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nGabaritos extraídos:")
    print(df_gabaritos.head(10))
    print(f"\nTotal de registros extraídos: {len(df_gabaritos)}")
else:
    print("[aviso] nenhum gabarito foi extraído.")


# ---------------------------------------------------------------
# 3) Consolidação das provas
# ---------------------------------------------------------------
CONSOLIDADOS = {
    "provas_completas.csv",
    "gabaritos_completos.csv",
    "provas_gabaritos_completos.csv",
}

dados = []

for arquivo in sorted(PASTA_CSV.glob("*.csv")):
    # Ignora arquivos consolidados para não duplicar
    if arquivo.name in CONSOLIDADOS:
        continue

    # Extrai o ano e a aplicação do nome do arquivo
    resultado = re.search(
        r"(\d{4})(?:_(\d+))?",
        arquivo.stem,
    )

    if resultado is None:
        print(f"Aviso: ano não encontrado em {arquivo.name}")
        continue

    ano = int(resultado.group(1))
    aplicacao = int(resultado.group(2)) if resultado.group(2) else 1

    df = pd.read_csv(arquivo)
    df["Ano"] = ano
    df["Aplicacao"] = aplicacao
    dados.append(df)

if dados:
    df_completo = pd.concat(dados, ignore_index=True)

    df_completo.to_csv(
        PASTA_CSV / "provas_completas.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(df_completo.head())
    print(f"Total de questões: {len(df_completo)}")
else:
    df_completo = pd.DataFrame()
    print("Nenhum arquivo CSV válido foi encontrado.")


# ---------------------------------------------------------------
# 4) Merge provas × gabaritos
# ---------------------------------------------------------------
if not df_completo.empty and not df_gabaritos.empty:
    df_final = df_completo.merge(
        df_gabaritos,
        on=["Nquestao", "Ano", "Aplicacao"],
        how="left",
    )

    df_final.to_csv(
        PASTA_CSV / "provas_gabaritos_completos.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nMerge realizado.")
    print(df_final.head())
    print(f"Total de linhas: {len(df_final)}")

    sem_gabarito = df_final["gabarito"].isna().sum()
    if sem_gabarito:
        print(f"[atenção] {sem_gabarito} questões sem gabarito correspondente.")
else:
    print("[aviso] merge não executado (provas ou gabaritos vazios).")