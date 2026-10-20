import pandas as pd
import pathlib
import re

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
        pasta / "provas_completas.csv",
        index=False,
        encoding="utf-8-sig"
    )