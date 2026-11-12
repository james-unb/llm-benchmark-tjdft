"""Sequential benchmark with one call per question and explicit resume support.

The persisted data schema (dataset columns, result fields, progress-JSON keys and
output file names) is kept in Portuguese on purpose, so existing results stay
readable and resumable. The prompts are also in Portuguese because the exam
questions are.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values


# Prompts stay in Portuguese: the exam questions are Brazilian law questions.
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
MODELS = ("gemma4:e4b-it-q4_K_M", "gemma4:e4b-it-q8_0")
FILES = ("provas_tjdft_com_gabarito",)
ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "resultados"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY_VAR = "OPENROUTER_API_KEY"
OPENROUTER_TIMEOUT_S = 300
RETRYABLE_STATUS = (429, 500, 502, 503, 504)
DEFAULT_ATTEMPTS = 5
BASE_WAIT_S = 5
MAX_WAIT_S = 300  # cap for Retry-After; above it the attempt fails
ENV_FILE = ROOT / ".env"
KEYS = ["id_prova", "Nquestao"]
RESULT_FIELDS = [
    "modelo", "arquivo", *KEYS, "carregamento", "erro_consulta_gpu",
    "resposta_bruta", "letra", "tempo_s", "tokens_prompt", "tokens_saida",
    "possivel_truncamento", "erro",
]


def log(message):
    """Progress log with a timestamp; flushes so it shows up even when output is redirected."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def options(openrouter=False):
    # num_ctx is a local Ollama parameter; OpenRouter uses the model's own context.
    base = {"temperature": 0, "seed": 42}
    return base if openrouter else {**base, "num_ctx": NUM_CTX}


def build_prompt(row, letters="ABCDE"):
    alts = "\n".join(f"{l}) {row[l]}" for l in letters if pd.notna(row.get(l)))
    return TEMPLATE.format(enunciado=row["enunciado"], alternativas=alts)


def extract_letter(text):
    """Accepts complete statements; conflicts or uncertain constructions are invalid."""
    if not isinstance(text, str):
        return None
    # Stripping only Markdown delimiters allows **Resposta:** **(C)**.
    text = re.sub(r"[*_`]", "", text)
    statements = [line for line in text.splitlines() if line.strip()]
    if not statements:
        return None
    letters = set()
    for statement in statements:
        answer = re.fullmatch(
            r"\s*Resposta\s*:\s*(?:([A-E])|\(\s*([A-E])\s*\))\s*[.!]?\s*",
            statement, flags=re.I,
        )
        if not answer:
            return None
        letters.add((answer[1] or answer[2]).upper())
    return letters.pop() if len(letters) == 1 else None


def get_field(obj, name):
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)


def is_known_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def canonical_name(model):
    # A registry port is not a tag; only the last path segment matters.
    return model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"


def gpu_usage(model, client):
    """Memory loading state, not GPU compute utilization."""
    try:
        models = get_field(client.ps(), "models") or []
        found = [
            m for m in models
            if isinstance(get_field(m, "model") or get_field(m, "name"), str)
            and canonical_name(get_field(m, "model") or get_field(m, "name")) == canonical_name(model)
        ]
        if len(found) != 1:
            return {"carregamento": "desconhecido", "erro_consulta_gpu": "Model not uniquely identified in ollama.ps()."}
        size = get_field(found[0], "size")
        vram = get_field(found[0], "size_vram")
        if not is_known_number(size) or size == 0 or not is_known_number(vram) or vram > size:
            state = "desconhecido"
        elif vram == 0:
            state = "CPU"
        elif vram < size:
            state = "CPU/GPU parcial"
        else:
            state = "GPU"
        return {"carregamento": state, "erro_consulta_gpu": None}
    except Exception as exc:
        return {"carregamento": "desconhecido", "erro_consulta_gpu": str(exc)}


def error_result(error):
    return {
        "resposta_bruta": None, "letra": None, "tempo_s": None,
        "tokens_prompt": None, "tokens_saida": None,
        "possivel_truncamento": None, "erro": error,
    }


def ask(model, prompt, client, openrouter=False):
    try:
        extras = {} if openrouter else {"keep_alive": "30m"}
        r = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            options=options(openrouter),
            **extras,
        )
        text = get_field(get_field(r, "message"), "content")
        duration = get_field(r, "total_duration")
        tokens_prompt = get_field(r, "prompt_eval_count")
        tokens_output = get_field(r, "eval_count")
        return {
            "resposta_bruta": text if isinstance(text, str) else None,
            "letra": extract_letter(text),
            "tempo_s": duration / 1e9 if is_known_number(duration) else None,
            "tokens_prompt": tokens_prompt if is_known_number(tokens_prompt) else None,
            "tokens_saida": tokens_output if is_known_number(tokens_output) else None,
            # Heuristic for closeness to the limit, not evidence of truncation.
            "possivel_truncamento": (
                tokens_prompt >= NUM_CTX - 50 if is_known_number(tokens_prompt) and not openrouter else None),
            "erro": None,
        }
    except Exception as exc:
        return error_result(str(exc) or type(exc).__name__)


def ensure_model(model, client):
    """Pulls only missing models; server failures do not justify a download."""
    try:
        client.show(model)
    except Exception as exc:
        if getattr(exc, "status_code", None) != 404:
            raise
        log(f"Model '{model}' not found. Pulling...")
        client.pull(model)
        log(f"Download of '{model}' finished.")


def load_input(file, csv_dir=CSV_DIR):
    if not file or Path(file).name != file or "\\" in file:
        raise ValueError(f"Invalid input name: {file!r}. Use the name without path and without .csv.")
    path = Path(csv_dir) / f"{file}.csv"
    try:
        df = pd.read_csv(path, dtype={key: str for key in [*KEYS, "enunciado"]})
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read input {path}: {exc}") from exc
    required = {*KEYS, "enunciado", "gabarito", *"ABCDE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(sorted(missing))}.")
    reserved = (set(RESULT_FIELDS) - set(KEYS)) | {"processada", "avaliavel", "acertou"}
    if reserved & set(df.columns):
        raise ValueError(f"{path}: input contains reserved result columns.")
    if df.empty:
        raise ValueError(f"{path}: no questions found.")
    for column in [*KEYS, "enunciado"]:
        if df[column].isna().any() or df[column].str.strip().eq("").any():
            raise ValueError(f"{path}: missing values in {column}.")
    for key in KEYS:
        df[key] = df[key].str.strip()
    if df.duplicated(KEYS).any():
        raise ValueError(f"{path}: duplicate keys in {KEYS}.")
    return path, df


def merge_results(df, results):
    answers = pd.DataFrame(results, columns=RESULT_FIELDS)
    answers["processada"] = True
    total = df.merge(answers, on=["id_prova", "Nquestao"], how="left", validate="one_to_one")
    total["processada"] = total["processada"].eq(True)
    answer_key = total["gabarito"].astype("string").str.strip().str.upper()
    total["avaliavel"] = answer_key.isin(list("ABCDE"))
    evaluated = total["processada"] & total["avaliavel"]
    total["acertou"] = pd.Series(pd.NA, index=total.index, dtype="boolean")
    total.loc[evaluated, "acertou"] = (
        total.loc[evaluated, "letra"].eq(answer_key[evaluated]).fillna(False)
        & total.loc[evaluated, "erro"].isna()
    )
    return total


def build_report(total):
    processed = total["processada"]
    evaluated = processed & total["avaliavel"]
    n = int(evaluated.sum())
    correct = int(total.loc[evaluated, "acertou"].sum())
    return {
        "partial": int(processed.sum()) < len(total),
        "total_questions": len(total),
        "processed": int(processed.sum()),
        "excluded": int((processed & ~total["avaliavel"]).sum()),
        "evaluated": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "no_valid_letter": int((processed & ~total["letra"].isin(list("ABCDE"))).sum()),
        "inference_failures": int((processed & total["erro"].notna()).sum()),
    }


def print_report(total, model, file):
    r = build_report(total)
    accuracy = f"{r['accuracy']:.2%}" if r["accuracy"] is not None else "not applicable"
    print(
        f"{model} | {file} | {'PARTIAL' if r['partial'] else 'COMPLETE'} report | "
        f"Processed: {r['processed']}/{r['total_questions']} | "
        f"Excluded from evaluation: {r['excluded']} | Evaluated: {r['evaluated']} | "
        f"Correct: {r['correct']} | Accuracy: {accuracy} | "
        f"Answers without a valid letter: {r['no_valid_letter']} | Inference failures: {r['inference_failures']}"
    )


def error_detail(error):
    """OpenRouter often hides the real cause in error.metadata.raw."""
    if not isinstance(error, Mapping):
        return error
    metadata = error.get("metadata")
    raw = metadata.get("raw") if isinstance(metadata, Mapping) else None
    parts = [p for p in (error.get("message"), raw if isinstance(raw, str) else None) if p]
    return " | ".join(parts) or None


class OpenRouterClient:
    """Minimal adapter: exposes chat() with the response shape that ask() already reads.

    Rate limits (429) and transient provider failures are retried with increasing
    waits, honoring Retry-After up to `max_wait` seconds; if the server asks for
    more than that, there is no retry before the deadline: the request fails and the
    question is recorded with an error (redoable with --redo-errors). `interval`
    spaces out the start of the requests.
    """

    def __init__(self, key, http=None, attempts=DEFAULT_ATTEMPTS, interval=0, sleep=time.sleep,
                 max_wait=MAX_WAIT_S):
        self.key, self.http = key, http
        self.attempts, self.interval, self.sleep = max(1, attempts), max(0, interval), sleep
        self.max_wait = max_wait
        self.last_start = None

    def _wait(self, answer, attempt):
        try:
            return max(0.0, float(answer.headers.get("Retry-After")))
        except (AttributeError, TypeError, ValueError):
            return BASE_WAIT_S * 2 ** attempt

    def chat(self, model, messages, options):
        http = self.http or importlib.import_module("httpx")
        for attempt in range(self.attempts):
            if self.interval and self.last_start is not None:
                remaining = self.interval - (time.monotonic() - self.last_start)
                if remaining > 0:
                    self.sleep(remaining)
            self.last_start = time.monotonic()
            start = time.perf_counter()
            r = http.post(
                OPENROUTER_URL, timeout=OPENROUTER_TIMEOUT_S,
                headers={"Authorization": f"Bearer {self.key}"},
                json={"model": model, "messages": messages, **options},
            )
            duration = time.perf_counter() - start
            try:
                body = r.json()
            except ValueError:
                body = {}
            if not isinstance(body, Mapping):
                body = {}
            error = body.get("error")
            if r.status_code >= 400 or error:
                if r.status_code in RETRYABLE_STATUS and attempt + 1 < self.attempts:
                    wait = self._wait(r, attempt)
                    if wait > self.max_wait:
                        detail = error_detail(error)
                        raise RuntimeError(
                            f"{detail or 'OpenRouter request failed'} (status code: {r.status_code}; "
                            f"Retry-After of {wait:.0f}s exceeds the {self.max_wait:.0f}s cap)")
                    log(f"{model}: status {r.status_code}; retrying in {wait:.0f}s "
                        f"({attempt + 1}/{self.attempts - 1})")
                    self.sleep(wait)
                    continue
                detail = error_detail(error)
                raise RuntimeError(f"{detail or 'OpenRouter request failed'} (status code: {r.status_code})")
            choices = body.get("choices")
            message = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], Mapping) else None
            content = message.get("content") if isinstance(message, Mapping) else None
            if content is None:
                # No model answer: record it as an error so --redo-errors redoes it.
                raise RuntimeError(f"OpenRouter returned no response content (status code: {r.status_code})")
            usage = body.get("usage") if isinstance(body.get("usage"), Mapping) else {}
            return {
                "message": {"content": content},
                "total_duration": duration * 1e9,
                "prompt_eval_count": usage.get("prompt_tokens"),
                "eval_count": usage.get("completion_tokens"),
            }


def create_openrouter_client(environment=None, interval=0, attempts=DEFAULT_ATTEMPTS):
    """The key is never written to results.

    Priority: process environment variable, then the .env file at the repo root.
    """
    if environment is None:
        environment = {**dotenv_values(ENV_FILE), **os.environ}
    key = (environment.get(OPENROUTER_KEY_VAR) or "").strip()
    if not key:
        raise RuntimeError(f"Set {OPENROUTER_KEY_VAR} in the environment or in the .env file to use --openrouter "
                           "(create a key at https://openrouter.ai/settings/keys).")
    return OpenRouterClient(key, attempts=attempts, interval=interval)


def build_config(model, file, path, df, openrouter=False):
    config = {
        "versao": 1, "modelo": model, "arquivo": file,
        "entrada": str(path.resolve()),
        "entrada_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "system": SYSTEM, "template": TEMPLATE, "opcoes": options(),
        "keep_alive": "30m", "raciocinio": "padrao_do_modelo_sem_override",
        "prompts_sha256": hashlib.sha256(json.dumps(
            [build_prompt(row) for _, row in df.iterrows()], ensure_ascii=False,
        ).encode()).hexdigest(),
    }
    if openrouter:  # Only in this mode, so existing local progress files stay valid.
        config.update(provedor="openrouter", opcoes=options(True))
        del config["keep_alive"]
    return config


def save_atomic(path, write):
    """An interruption during the write keeps the last complete file."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as f:
            temp_path = Path(f.name)
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def save_progress(path, config, results):
    save_atomic(path, lambda f: json.dump(
        {"configuracao": config, "resultados": results}, f,
        ensure_ascii=False, indent=2, allow_nan=False,
    ))


def load_progress(path, config, df):
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Progress missing or unreadable: {path}: {exc}") from exc
    if not isinstance(state, dict) or state.get("configuracao") != config:
        raise ValueError(f"Incompatible resume in {path}: model, input, prompt or parameters changed.")
    results = state.get("resultados")
    if not isinstance(results, list):
        raise ValueError(f"Invalid progress in {path}: results missing.")
    valid = set(df[KEYS].itertuples(index=False, name=None))
    seen = set()
    for r in results:
        if not isinstance(r, dict) or set(r) != set(RESULT_FIELDS):
            raise ValueError(f"Invalid record in {path}.")
        key = (r["id_prova"], r["Nquestao"])
        if key not in valid or key in seen or r["modelo"] != config["modelo"] or r["arquivo"] != config["arquivo"]:
            raise ValueError(f"Invalid identification or duplicate key in progress {path}.")
        seen.add(key)
    return results


def output_path(model, file, output_dir=OUTPUT_DIR, openrouter=False):
    safe_name = re.sub(r"[^\w.-]", "_", model) + ("_openrouter" if openrouter else "")
    return Path(output_dir) / f"resultados_{safe_name}_{file}.csv"


def run_model(models=MODELS, files=FILES, *, resume=False,
                 csv_dir=CSV_DIR, output_dir=OUTPUT_DIR, client=None, openrouter=False,
                 redo_errors=False, interval=0, attempts=DEFAULT_ATTEMPTS):
    # All inputs and progress files are validated before any Ollama access.
    if not models or not files or any(not isinstance(m, str) or not m.strip() for m in models):
        raise ValueError("Provide at least one valid model and one valid file.")
    if redo_errors and not resume:
        raise ValueError("--redo-errors only makes sense together with --resume.")
    log(f"Validating inputs: {', '.join(files)}")
    inputs = [(file, *load_input(file, csv_dir)) for file in files]
    for file, path, df in inputs:
        log(f"Input '{file}' loaded: {len(df)} questions ({path})")
    output_dir = Path(output_dir)
    runs = []
    destinations = set()
    for file, path, df in inputs:
        for model in models:
            output = output_path(model, file, output_dir, openrouter)
            progress = output.with_suffix(".progresso.json")
            if output in destinations:
                raise ValueError(f"Repeated combinations or conflicting output names: {output}.")
            destinations.add(output)
            config = build_config(model, file, path, df, openrouter)
            if resume:
                results = load_progress(progress, config, df)
                if redo_errors:
                    # Discards only the attempts with errors; valid answers are preserved.
                    discarded = sum(1 for r in results if r["erro"])
                    results = [r for r in results if not r["erro"]]
                    log(f"{model} | {file}: {discarded} attempts with errors will be redone")
            else:
                if output.exists() or progress.exists():
                    raise ValueError(f"Output already exists: {output}. Use --resume or a new --output-dir.")
                results = []
            mode = f"resuming with {len(results)} questions already recorded" if resume else "new run"
            log(f"{model} | {file}: {mode}; output at {output}")
            runs.append((model, file, df, output, progress, config, results))

    if client is None and openrouter:
        log("Loading OpenRouter client")
        client = create_openrouter_client(interval=interval, attempts=attempts)
    elif client is None:
        log("Loading Ollama client")
        try:
            client = importlib.import_module("ollama")
        except ImportError as exc:
            raise RuntimeError("Install the 'ollama' Python package in the environment before running the benchmark.") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        for _, _, _, _, progress, config, results in runs:
            save_progress(progress, config, results)

    for model, file, df, output, progress, config, results in runs:
        recorded = {(r["id_prova"], r["Nquestao"]) for r in results}
        try:
            if len(recorded) < len(df):
                if openrouter:
                    # No download or warmup: an extra call would spend the model quota.
                    log(f"{model} | {file}: {len(df) - len(recorded)} questions pending")
                else:
                    log(f"{model} | {file}: checking model availability")
                    ensure_model(model, client)
                    log(f"{model} | {file}: warming up the model")
                    warmup = ask(model, "Oi", client)
                    if warmup["erro"]:
                        raise RuntimeError(f"Warmup failed for {model}: {warmup['erro']}")
                    log(f"{model} | {file}: warmup finished; {len(df) - len(recorded)} questions pending")
                for _, row in df.iterrows():
                    key = (row["id_prova"], row["Nquestao"])
                    if key in recorded:
                        continue
                    interrupted = False
                    try:
                        out = ask(model, build_prompt(row), client, openrouter)
                    except KeyboardInterrupt:
                        out = error_result("Inference interrupted by the user; attempt recorded without automatic resend.")
                        interrupted = True
                    gpu = {"carregamento": "openrouter" if openrouter else "desconhecido", "erro_consulta_gpu": None}
                    if not interrupted and not openrouter:
                        try:
                            gpu = gpu_usage(model, client)
                        except KeyboardInterrupt:
                            interrupted = True
                    results.append({"modelo": model, "arquivo": file,
                                       "id_prova": key[0], "Nquestao": key[1], **gpu, **out})
                    try:
                        save_progress(progress, config, results)
                    except KeyboardInterrupt:
                        # Do not lose an answer already received because of a Ctrl-C during the write.
                        save_progress(progress, config, results)
                        interrupted = True
                    recorded.add(key)
                    total = merge_results(df, results)
                    save_atomic(output, lambda f: total.to_csv(f, index=False))
                    if out["erro"]:
                        summary = f"ERRO: {out['erro']}"
                    else:
                        summary = f"letter={out['letra'] or 'invalid'} answer_key={str(row['gabarito']).strip()}"
                        if out["tempo_s"] is not None:
                            summary += f" tempo={out['tempo_s']:.1f}s"
                    log(f"{model} | {file} [{len(recorded)}/{len(df)}] "
                        f"id_prova={key[0]} Nquestao={key[1]} {summary}")
                    if interrupted:
                        raise KeyboardInterrupt
        finally:
            # The JSON is the source for resuming; the CSV can be rebuilt without inference.
            log(f"{model} | {file}: writing final CSV and report")
            total = merge_results(df, results)
            save_atomic(output, lambda f: total.to_csv(f, index=False))
            print_report(total, model, file)


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--files", nargs="+", default=list(FILES))
    parser.add_argument("--resume", action="store_true", help="Reuses compatible progress, including attempts with errors.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Use a new directory to start another run without overwriting results.")
    parser.add_argument("--redo-errors", action="store_true",
                        help="With --resume, discards attempts with errors (e.g. 429) and redoes them; keeps valid answers.")
    parser.add_argument("--interval", type=float, default=0,
                        help="Minimum seconds between requests with --openrouter (e.g. 3 for ~20 req/min).")
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS,
                        help="Attempts per question with --openrouter for 429/5xx, with increasing waits.")
    parser.add_argument("--openrouter", action="store_true",
                        help=f"Uses OpenRouter with the key in ${OPENROUTER_KEY_VAR} (environment or .env); "
                             "use OpenRouter ids in --models (e.g. provider/model:free).")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    try:
        run_model(models=args.models, files=args.files,
                     resume=args.resume, output_dir=args.output_dir, openrouter=args.openrouter,
                     redo_errors=args.redo_errors, interval=args.interval, attempts=args.attempts)
    except KeyboardInterrupt:
        print("Run interrupted. Progress saved; use --resume with the same configuration.")
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
