"""Benchmark local, sequencial, com uma chamada por questão e retomada explícita."""

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

import pandas as pd


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
MODELOS = ("gemma4:e4b-it-q4_K_M", "gemma4:e4b-it-q8_0")
ARQUIVOS = ("provas_tjdft_com_gabarito",)
RAIZ = Path(__file__).resolve().parent.parent
PASTA_CSV = RAIZ / "data" / "processed"
PASTA_SAIDA = RAIZ / "resultados"
CHAVES = ["id_prova", "Nquestao"]
CAMPOS_RESULTADO = [
    "modelo", "arquivo", *CHAVES, "carregamento", "erro_consulta_gpu",
    "resposta_bruta", "letra", "tempo_s", "tokens_prompt", "tokens_saida",
    "possivel_truncamento", "erro",
]


def opcoes():
    return {"temperature": 0, "seed": 42, "num_ctx": NUM_CTX}


def montar_prompt(row, letras="ABCDE"):
    alts = "\n".join(f"{l}) {row[l]}" for l in letras if pd.notna(row.get(l)))
    return TEMPLATE.format(enunciado=row["enunciado"], alternativas=alts)


def extrair_letra(texto):
    """Aceita declarações completas; conflitos ou construções incertas são inválidos."""
    if not isinstance(texto, str):
        return None
    # Retirar somente delimitadores Markdown permite **Resposta:** **(C)**.
    texto = re.sub(r"[*_`]", "", texto)
    declaracoes = [linha for linha in texto.splitlines() if linha.strip()]
    if not declaracoes:
        return None
    letras = set()
    for declaracao in declaracoes:
        resposta = re.fullmatch(
            r"\s*Resposta\s*:\s*(?:([A-E])|\(\s*([A-E])\s*\))\s*[.!]?\s*",
            declaracao, flags=re.I,
        )
        if not resposta:
            return None
        letras.add((resposta[1] or resposta[2]).upper())
    return letras.pop() if len(letras) == 1 else None


def campo(objeto, nome):
    return objeto.get(nome) if isinstance(objeto, Mapping) else getattr(objeto, nome, None)


def numero_conhecido(valor):
    return isinstance(valor, (int, float)) and not isinstance(valor, bool) and math.isfinite(valor) and valor >= 0


def nome_canonico(modelo):
    # A porta de um registry não é a tag; somente o último segmento importa.
    return modelo if ":" in modelo.rsplit("/", 1)[-1] else f"{modelo}:latest"


def usou_gpu(modelo, cliente):
    """Estado de carregamento em memória, não utilização computacional da GPU."""
    try:
        modelos = campo(cliente.ps(), "models") or []
        encontrados = [
            m for m in modelos
            if isinstance(campo(m, "model") or campo(m, "name"), str)
            and nome_canonico(campo(m, "model") or campo(m, "name")) == nome_canonico(modelo)
        ]
        if len(encontrados) != 1:
            return {"carregamento": "desconhecido", "erro_consulta_gpu": "Modelo não identificado unicamente em ollama.ps()."}
        tamanho = campo(encontrados[0], "size")
        vram = campo(encontrados[0], "size_vram")
        if not numero_conhecido(tamanho) or tamanho == 0 or not numero_conhecido(vram) or vram > tamanho:
            estado = "desconhecido"
        elif vram == 0:
            estado = "CPU"
        elif vram < tamanho:
            estado = "CPU/GPU parcial"
        else:
            estado = "GPU"
        return {"carregamento": estado, "erro_consulta_gpu": None}
    except Exception as exc:
        return {"carregamento": "desconhecido", "erro_consulta_gpu": str(exc)}


def resultado_erro(erro):
    return {
        "resposta_bruta": None, "letra": None, "tempo_s": None,
        "tokens_prompt": None, "tokens_saida": None,
        "possivel_truncamento": None, "erro": erro,
    }


def perguntar(modelo, prompt, cliente):
    try:
        r = cliente.chat(
            model=modelo,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            options=opcoes(),
            keep_alive="30m",
        )
        texto = campo(campo(r, "message"), "content")
        duracao = campo(r, "total_duration")
        tokens_prompt = campo(r, "prompt_eval_count")
        tokens_saida = campo(r, "eval_count")
        return {
            "resposta_bruta": texto if isinstance(texto, str) else None,
            "letra": extrair_letra(texto),
            "tempo_s": duracao / 1e9 if numero_conhecido(duracao) else None,
            "tokens_prompt": tokens_prompt if numero_conhecido(tokens_prompt) else None,
            "tokens_saida": tokens_saida if numero_conhecido(tokens_saida) else None,
            # Heurística de proximidade ao limite, não evidência de truncamento.
            "possivel_truncamento": tokens_prompt >= NUM_CTX - 50 if numero_conhecido(tokens_prompt) else None,
            "erro": None,
        }
    except Exception as exc:
        return resultado_erro(str(exc) or type(exc).__name__)


def garantir_modelo(modelo, cliente):
    """Baixa apenas modelos ausentes; falhas de servidor não justificam download."""
    try:
        cliente.show(modelo)
    except Exception as exc:
        if getattr(exc, "status_code", None) != 404:
            raise
        print(f"Modelo '{modelo}' não encontrado. Baixando...")
        cliente.pull(modelo)


def carregar_entrada(arquivo, pasta_csv=PASTA_CSV):
    if not arquivo or Path(arquivo).name != arquivo or "\\" in arquivo:
        raise ValueError(f"Nome de entrada inválido: {arquivo!r}. Use o nome sem caminho e sem .csv.")
    caminho = Path(pasta_csv) / f"{arquivo}.csv"
    try:
        df = pd.read_csv(caminho, dtype={chave: str for chave in [*CHAVES, "enunciado"]})
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"Não foi possível ler a entrada {caminho}: {exc}") from exc
    necessarias = {*CHAVES, "enunciado", "gabarito", *"ABCDE"}
    faltantes = necessarias - set(df.columns)
    if faltantes:
        raise ValueError(f"{caminho}: colunas obrigatórias ausentes: {', '.join(sorted(faltantes))}.")
    reservadas = (set(CAMPOS_RESULTADO) - set(CHAVES)) | {"processada", "avaliavel", "acertou"}
    if reservadas & set(df.columns):
        raise ValueError(f"{caminho}: entrada contém colunas reservadas de resultados.")
    if df.empty:
        raise ValueError(f"{caminho}: nenhuma questão encontrada.")
    for coluna in [*CHAVES, "enunciado"]:
        if df[coluna].isna().any() or df[coluna].str.strip().eq("").any():
            raise ValueError(f"{caminho}: valores ausentes em {coluna}.")
    for chave in CHAVES:
        df[chave] = df[chave].str.strip()
    if df.duplicated(CHAVES).any():
        raise ValueError(f"{caminho}: chaves duplicadas em {CHAVES}.")
    return caminho, df


def associar_resultados(df, resultados):
    respostas = pd.DataFrame(resultados, columns=CAMPOS_RESULTADO)
    respostas["processada"] = True
    total = df.merge(respostas, on=["id_prova", "Nquestao"], how="left", validate="one_to_one")
    total["processada"] = total["processada"].eq(True)
    gabarito = total["gabarito"].astype("string").str.strip().str.upper()
    total["avaliavel"] = gabarito.isin(list("ABCDE"))
    avaliadas = total["processada"] & total["avaliavel"]
    total["acertou"] = pd.Series(pd.NA, index=total.index, dtype="boolean")
    total.loc[avaliadas, "acertou"] = (
        total.loc[avaliadas, "letra"].eq(gabarito[avaliadas]).fillna(False)
        & total.loc[avaliadas, "erro"].isna()
    )
    return total


def relatorio(total):
    processadas = total["processada"]
    avaliadas = processadas & total["avaliavel"]
    n = int(avaliadas.sum())
    acertos = int(total.loc[avaliadas, "acertou"].sum())
    return {
        "parcial": int(processadas.sum()) < len(total),
        "total_questoes": len(total),
        "processadas": int(processadas.sum()),
        "excluidas": int((processadas & ~total["avaliavel"]).sum()),
        "avaliadas": n,
        "acertos": acertos,
        "acuracia": acertos / n if n else None,
        "sem_letra_valida": int((processadas & ~total["letra"].isin(list("ABCDE"))).sum()),
        "falhas_inferencia": int((processadas & total["erro"].notna()).sum()),
    }


def imprimir_relatorio(total, modelo, arquivo):
    r = relatorio(total)
    acuracia = f"{r['acuracia']:.2%}" if r["acuracia"] is not None else "não aplicável"
    print(
        f"{modelo} | {arquivo} | Relatório {'PARCIAL' if r['parcial'] else 'COMPLETO'} | "
        f"Processadas: {r['processadas']}/{r['total_questoes']} | "
        f"Excluídas da avaliação: {r['excluidas']} | Avaliadas: {r['avaliadas']} | "
        f"Acertos: {r['acertos']} | Acurácia: {acuracia} | "
        f"Respostas sem letra válida: {r['sem_letra_valida']} | Falhas de inferência: {r['falhas_inferencia']}"
    )


def configuracao(modelo, arquivo, caminho, df):
    return {
        "versao": 1, "modelo": modelo, "arquivo": arquivo,
        "entrada": str(caminho.resolve()),
        "entrada_sha256": hashlib.sha256(caminho.read_bytes()).hexdigest(),
        "system": SYSTEM, "template": TEMPLATE, "opcoes": opcoes(),
        "keep_alive": "30m", "raciocinio": "padrao_do_modelo_sem_override",
        "prompts_sha256": hashlib.sha256(json.dumps(
            [montar_prompt(row) for _, row in df.iterrows()], ensure_ascii=False,
        ).encode()).hexdigest(),
    }


def salvar_atomico(caminho, escrever):
    """Uma interrupção durante a escrita conserva o último arquivo completo."""
    temporario = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=caminho.parent, delete=False) as f:
            temporario = Path(f.name)
            escrever(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporario, caminho)
    finally:
        if temporario is not None and temporario.exists():
            temporario.unlink()


def salvar_progresso(caminho, config, resultados):
    salvar_atomico(caminho, lambda f: json.dump(
        {"configuracao": config, "resultados": resultados}, f,
        ensure_ascii=False, indent=2, allow_nan=False,
    ))


def carregar_progresso(caminho, config, df):
    try:
        estado = json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Progresso ausente ou ilegível: {caminho}: {exc}") from exc
    if not isinstance(estado, dict) or estado.get("configuracao") != config:
        raise ValueError(f"Retomada incompatível em {caminho}: modelo, entrada, prompt ou parâmetros mudaram.")
    resultados = estado.get("resultados")
    if not isinstance(resultados, list):
        raise ValueError(f"Progresso inválido em {caminho}: resultados ausentes.")
    validas = set(df[CHAVES].itertuples(index=False, name=None))
    vistas = set()
    for r in resultados:
        if not isinstance(r, dict) or set(r) != set(CAMPOS_RESULTADO):
            raise ValueError(f"Registro inválido em {caminho}.")
        chave = (r["id_prova"], r["Nquestao"])
        if chave not in validas or chave in vistas or r["modelo"] != config["modelo"] or r["arquivo"] != config["arquivo"]:
            raise ValueError(f"Identificação inválida ou chave duplicada no progresso {caminho}.")
        vistas.add(chave)
    return resultados


def caminho_saida(modelo, arquivo, pasta_saida=PASTA_SAIDA):
    nome_seguro = re.sub(r"[^\w.-]", "_", modelo)
    return Path(pasta_saida) / f"resultados_{nome_seguro}_{arquivo}.csv"


def rodar_modelo(modelos=MODELOS, arquivos=ARQUIVOS, *, retomar=False,
                 pasta_csv=PASTA_CSV, pasta_saida=PASTA_SAIDA, cliente=None):
    # Toda entrada e todo progresso são validados antes de qualquer acesso ao Ollama.
    if not modelos or not arquivos or any(not isinstance(m, str) or not m.strip() for m in modelos):
        raise ValueError("Informe ao menos um modelo e um arquivo válidos.")
    entradas = [(arquivo, *carregar_entrada(arquivo, pasta_csv)) for arquivo in arquivos]
    pasta_saida = Path(pasta_saida)
    execucoes = []
    destinos = set()
    for arquivo, caminho, df in entradas:
        for modelo in modelos:
            saida = caminho_saida(modelo, arquivo, pasta_saida)
            progresso = saida.with_suffix(".progresso.json")
            if saida in destinos:
                raise ValueError(f"Combinações repetidas ou nomes de saída conflitantes: {saida}.")
            destinos.add(saida)
            config = configuracao(modelo, arquivo, caminho, df)
            if retomar:
                resultados = carregar_progresso(progresso, config, df)
            else:
                if saida.exists() or progresso.exists():
                    raise ValueError(f"Saída já existe: {saida}. Use --retomar ou uma nova --pasta-saida.")
                resultados = []
            execucoes.append((modelo, arquivo, df, saida, progresso, config, resultados))

    if cliente is None:
        try:
            cliente = importlib.import_module("ollama")
        except ImportError as exc:
            raise RuntimeError("Instale o pacote Python 'ollama' no ambiente antes de executar o benchmark.") from exc
    pasta_saida.mkdir(parents=True, exist_ok=True)
    if not retomar:
        for _, _, _, _, progresso, config, resultados in execucoes:
            salvar_progresso(progresso, config, resultados)

    for modelo, arquivo, df, saida, progresso, config, resultados in execucoes:
        registradas = {(r["id_prova"], r["Nquestao"]) for r in resultados}
        try:
            if len(registradas) < len(df):
                garantir_modelo(modelo, cliente)
                aquecimento = perguntar(modelo, "Oi", cliente)
                if aquecimento["erro"]:
                    raise RuntimeError(f"Falha no aquecimento de {modelo}: {aquecimento['erro']}")
                for _, row in df.iterrows():
                    chave = (row["id_prova"], row["Nquestao"])
                    if chave in registradas:
                        continue
                    interrompida = False
                    try:
                        out = perguntar(modelo, montar_prompt(row), cliente)
                    except KeyboardInterrupt:
                        out = resultado_erro("Inferência interrompida pelo usuário; tentativa registrada sem reenvio automático.")
                        interrompida = True
                    gpu = {"carregamento": "desconhecido", "erro_consulta_gpu": None}
                    if not interrompida:
                        try:
                            gpu = usou_gpu(modelo, cliente)
                        except KeyboardInterrupt:
                            interrompida = True
                    resultados.append({"modelo": modelo, "arquivo": arquivo,
                                       "id_prova": chave[0], "Nquestao": chave[1], **gpu, **out})
                    try:
                        salvar_progresso(progresso, config, resultados)
                    except KeyboardInterrupt:
                        # Não perder uma resposta já recebida por um Ctrl-C na escrita.
                        salvar_progresso(progresso, config, resultados)
                        interrompida = True
                    registradas.add(chave)
                    total = associar_resultados(df, resultados)
                    salvar_atomico(saida, lambda f: total.to_csv(f, index=False))
                    if interrompida:
                        raise KeyboardInterrupt
        finally:
            # O JSON é a fonte da retomada; o CSV pode ser reconstruído sem inferência.
            total = associar_resultados(df, resultados)
            salvar_atomico(saida, lambda f: total.to_csv(f, index=False))
            imprimir_relatorio(total, modelo, arquivo)


def criar_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelos", nargs="+", default=list(MODELOS))
    parser.add_argument("--arquivos", nargs="+", default=list(ARQUIVOS))
    parser.add_argument("--retomar", action="store_true", help="Reutiliza o progresso compatível, inclusive tentativas com erro.")
    parser.add_argument("--pasta-saida", type=Path, default=PASTA_SAIDA,
                        help="Use uma pasta nova para iniciar outra execução sem sobrescrever resultados.")
    return parser


def main():
    parser = criar_parser()
    args = parser.parse_args()
    try:
        rodar_modelo(modelos=args.modelos, arquivos=args.arquivos,
                     retomar=args.retomar, pasta_saida=args.pasta_saida)
    except KeyboardInterrupt:
        print("Execução interrompida. Progresso salvo; use --retomar com a mesma configuração.")
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Erro: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
