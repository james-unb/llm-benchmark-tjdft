import re
import pathlib

import pandas as pd
import pymupdf


# Arquivo de entrada
pasta = "database/"
nomes = ['prova_2014', 'prova_2015_1', 'prova_2015_2']
provas = pd.DataFrame()

for nome in nomes:
    arquivo = pasta + nome + ".pdf"
    # Extração do texto do PDF
    with pymupdf.open(arquivo) as doc:
        conteudo = chr(12).join(page.get_text() for page in doc)


    # Salva o texto extraído
    pathlib.Path("database_txt/" + nome + ".txt").write_text(
        conteudo,
        encoding="utf-8"
    )


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

    id_prova = '_'.join(re.findall(r'\d+', nome))
    df['id_prova'] = id_prova

    # Exibe as primeiras questões
    print(df.head())
    print(f"Total de questões extraídas de {nome}: {df.shape[0]}")
    provas = pd.concat([provas, df])

print(f"Total de questões no banco: {provas.shape[0]}")
provas.to_csv('provas_tjdft.csv', index=False, encoding='utf-8')
