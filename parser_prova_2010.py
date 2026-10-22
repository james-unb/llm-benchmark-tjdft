import re
import pathlib

import pandas as pd
import pymupdf


arquivo_entrada = pathlib.Path("database/prova_2010.pdf")
arquivo_saida = pathlib.Path("database_csv/prova_2010.csv")

regex_questao = re.compile(
    r"""
    ^[ \t]*
    (?:
        QUEST[ÃA]O[ \t]*(?:N[º°.]?[ \t]*)?(?P<numero_palavra>\d{1,3})
        |
        (?P<numero_antes>\d{1,3})[ \t]*(?:[ºª°][ \t]*)?QUEST[ÃA]O
        |
        (?P<numero_pontuacao>\d{1,3})[ \t]*[.)\-:](?!\d)
    )
    [ \t]*(?::|[-–—.]|(?=\S))?[ \t]*
    (?P<conteudo>.*)
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)

regex_alternativa = re.compile(
    r"""
    ^[ \t]*
    (?:
        \([ \t]*(?P<letra_parenteses>[A-E])[ \t]*\)
        |
        (?P<letra_pontuacao>[A-E])[ \t]*[).:\-–—]
    )
    [ \t]*(?P<texto>.*)
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)


def normalizar(texto):
    return " ".join(texto.split())


with pymupdf.open(arquivo_entrada) as documento:
    texto_pdf = "\n".join(pagina.get_text() for pagina in documento)

candidatas = []
for match in regex_questao.finditer(texto_pdf):
    grupos = match.groupdict()
    numero = (
        grupos["numero_palavra"]
        or grupos["numero_antes"]
        or grupos["numero_pontuacao"]
    )
    # Esta prova tem questões numeradas de 1 a 100.
    if numero is not None and 1 <= int(numero) <= 100:
        candidatas.append((match, str(int(numero))))

registros = []

for indice, (match, numero) in enumerate(candidatas):
    fim_questao = (
        candidatas[indice + 1][0].start()
        if indice + 1 < len(candidatas)
        else len(texto_pdf)
    )

    trecho = match.group("conteudo") + texto_pdf[match.end():fim_questao]
    trecho = re.sub(r"(?m)^[ \t]*\d{1,2}[ \t]*$", "", trecho).strip()

    alternativas = list(regex_alternativa.finditer(trecho))

    registro = {
        "Nquestao": numero,
        "enunciado": "",
        "A": "",
        "B": "",
        "C": "",
        "D": "",
        "E": "",
    }

    if alternativas:
        registro["enunciado"] = normalizar(
            trecho[:alternativas[0].start()]
        )

        for j, alternativa in enumerate(alternativas):
            letra = (
                alternativa.group("letra_parenteses")
                or alternativa.group("letra_pontuacao")
            ).upper()

            fim_texto = (
                alternativas[j + 1].start()
                if j + 1 < len(alternativas)
                else len(trecho)
            )

            # O grupo "texto" já contém a primeira linha da alternativa.
            texto_alternativa = (
                alternativa.group("texto")
                + "\n"
                + trecho[alternativa.end():fim_texto]
            )
            texto_alternativa = normalizar(texto_alternativa)

            if not registro[letra]:
                registro[letra] = texto_alternativa
    else:
        # As questões 61 a 68 são compostas por assertivas I, II e III.
        registro["enunciado"] = normalizar(trecho)

    registros.append(registro)

df = pd.DataFrame(registros)

df["questao_completa"] = (
    df["enunciado"]
    + "\nA) " + df["A"]
    + "\nB) " + df["B"]
    + "\nC) " + df["C"]
    + "\nD) " + df["D"]
    + "\nE) " + df["E"]
)

print(f"Total de questões extraídas: {len(df)}")

# Gabarito

# Arquivo de entrada e diretório de saída
pasta_entrada = pathlib.Path("database/")
pasta_saida = pathlib.Path("database_csv")
pasta_saida.mkdir(parents=True, exist_ok=True)

# O arquivo deve estar em database/provas/gabarito_2010.pdf.
arquivo = pasta_entrada / "gabarito_2010.pdf"
if not arquivo.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {arquivo}")

# Extrai o texto do PDF
with pymupdf.open(arquivo) as doc:
    texto = "\n".join(pagina.get_text() for pagina in doc)

# Captura número da questão e resposta. Questões anuladas são registradas como "Anulada".
# O texto explicativo após "Anulada" não é confundido com outras questões.
padrao = re.compile(
    r"(?m)^\s*(?P<numero>\d{1,3})\s*\.\s*"
    r"(?P<resposta>Anulada|[A-E])\b",
    flags=re.IGNORECASE,
)

registros = []
vistos = set()

for correspondencia in padrao.finditer(texto):
    numero = int(correspondencia.group("numero"))
    resposta = correspondencia.group("resposta").strip().capitalize()

    # Evita capturar números fora do intervalo de questões do gabarito.
    if not 1 <= numero <= 100 or numero in vistos:
        continue

    vistos.add(numero)
    registros.append({
        "Nquestao": numero,
        "gabarito": resposta,
    })

gabarito = pd.DataFrame(registros).sort_values("Nquestao")

print(f"Gabarito extraído: {len(df)} questões.")

df["Nquestao"] = pd.to_numeric(df["Nquestao"], errors="raise").astype(int)

df = df.merge(gabarito, on = ["Nquestao"], how = "left")

print(df.head())
df.to_csv(arquivo_saida, index=False, encoding="utf-8-sig")

df_completo = pd.read_csv("database_csv/provas_gabaritos_completos.csv")

df["Ano"] = 2010
df["Aplicacao"] = 1

# Adicionar os novos dados ao DataFrame existente
df_completo = pd.concat(
    [df_completo, df],
    ignore_index=True
)

# Salvar o resultado atualizado
df_completo.to_csv(
    "provas_gabaritos_completos.csv",
    index=False,
    encoding="utf-8-sig"
)