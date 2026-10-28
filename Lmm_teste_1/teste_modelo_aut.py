import importlib
import subprocess
import sys


def garantir_pacote(nome_import, nome_pip=None):
    """Importa o pacote; se não estiver instalado, instala via pip e importa de novo."""
    try:
        return importlib.import_module(nome_import)
    except ImportError:
        print(f"Pacote '{nome_import}' não encontrado. Instalando...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", nome_pip or nome_import])
        importlib.invalidate_caches()
        return importlib.import_module(nome_import)


pd = garantir_pacote("pandas")
ollama = garantir_pacote("ollama")

import re
from pathlib import Path



SYSTEM = (
    """Você é um especialista em Direito brasileiro resolvendo questões de 
    múltipla escolha de provas. Considere a legislação e a 
    jurisprudência brasileiras. Apenas uma alternativa está correta."""
)

TEMPLATE = """Responda a questão abaixo.

Enunciado: {enunciado}
Alternativas: {alternativas}

Responda apenas com a letra da alternativa correta, exatamente neste formato:
Resposta: <letra>"""

NUM_CTX = 4096

def montar_prompt(row, letras="ABCDE"):
    alts = "\n".join(f"{l}) {row[l]}" for l in letras if pd.notna(row.get(l)))
    return TEMPLATE.format(enunciado=row["enunciado"], alternativas=alts)

def extrair_letra(texto):
    achados = re.findall(r"Resposta:?\**\s*\(?\**\s*([A-E])\b", texto, flags=re.I)
    return achados[-1].upper() if achados else None

def usou_gpu(modelo):
    for m in ollama.ps().models:
        if m.model == modelo:
            return m.size_vram > 0
    return None

def perguntar(modelo, prompt):
    try:
        r = ollama.chat(
            model=modelo,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0, "seed": 42, "num_ctx": NUM_CTX},
            keep_alive="30m",
        )
    except Exception as e:
        return {"resposta_bruta": None, "letra": None, "erro": str(e)}

    texto = r["message"]["content"] or ""
    return {
        "resposta_bruta": texto,
        "letra": extrair_letra(texto),
        "tempo_s": r["total_duration"] / 1e9,
        "tokens_prompt": r["prompt_eval_count"],
        "tokens_saida": r["eval_count"],
        "possivel_truncamento": r["prompt_eval_count"] >= NUM_CTX - 50,
        "erro": None,
    }


def garantir_modelo(modelo):
    """Baixa o modelo no Ollama se ainda não estiver disponível."""
    try:
        ollama.show(modelo)
    except ollama.ResponseError:
        print(f"Modelo '{modelo}' não encontrado. Baixando...")
        ollama.pull(modelo)

RAIZ = Path(__file__).resolve().parent.parent
PASTA_CSV = RAIZ / "database_csv"
PASTA_SAIDA = RAIZ / "resultados"
PASTA_SAIDA.mkdir(exist_ok=True)


def rodar_modelo(
    modelos=("gemma4:e4b-it-q4_K_M", "gemma4:e4b-it-q8_0"),
    arquivos=("prova_2014", "prova_2015_1", "prova_2015_2"),
):
    for arquivo in arquivos:
        df = pd.read_csv(PASTA_CSV / f"{arquivo}.csv")

        for modelo in modelos:
            nome_seguro = modelo.replace(":", "_")
            saida = PASTA_SAIDA / f"resultados_{nome_seguro}_{arquivo}.csv"
            garantir_modelo(modelo)

            aquecimento =perguntar(modelo, "Oi") # Inicializa o modelo para evitar overhead na primeira pergunta
            if aquecimento["erro"]:
                raise RuntimeError(f"Falha ao chamar {modelo}: {aquecimento['erro']}")  
            gpu = usou_gpu(modelo)
            gpu_txt = {True: "GPU", False: "CPU", None: "desconhecido"}[gpu]

            resultados = []
            for _, row in df.iterrows():
                out = perguntar(modelo, montar_prompt(row))
                resultados.append({"modelo": modelo, "id_prova": row["id_prova"], "Nquestao": row["Nquestao"], "usou_gpu": gpu, **out})

            df_res = pd.DataFrame(resultados)
            df_total = df.merge(
                df_res,
                on=["id_prova", "Nquestao"],
                how="left",
                validate="one_to_one",
            )
            df_total["acertou"] = df_total["letra"] == df_total["gabarito"]

            n = len(df_total)
            acertos = df_total["acertou"].sum()
            sem_letra = df_total["letra"].isna().sum()
            print(
                f"{modelo} | {arquivo} | {gpu_txt} |Acertos: {acertos}/{n} ({acertos/n:.2%}) "
                f"| sem letra extraída: {sem_letra} | erros: {df_total['erro'].notna().sum()}"
            )

            df_total.to_csv(saida, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--modelos", nargs="+", default=["gemma4:e4b-it-q4_K_M", "gemma4:e4b-it-q8_0"])
    parser.add_argument("--arquivos", nargs="+", default=["prova_2014", "prova_2015_1", "prova_2015_2"])
    args = parser.parse_args()

    rodar_modelo(modelos=args.modelos, arquivos=args.arquivos)