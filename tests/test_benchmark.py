"""Isolated regressions: no test reaches the Ollama service."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import ollama

from src import teste_modelo_aut as b


def response(text="Resposta: C", **metrics):
    return {"message": {"content": text}, **metrics}


def question(exam="p1", number="1", answer_key="C"):
    return {"id_prova": exam, "Nquestao": number, "enunciado": f"Question {exam}/{number}",
            **{letter: f"Option {letter}" for letter in "ABCDE"}, "gabarito": answer_key}


def record(exam="p1", number="1", letter="C", error=None):
    return {"modelo": "model", "arquivo": "input", "id_prova": exam, "Nquestao": number,
            "carregamento": "desconhecido", "erro_consulta_gpu": None,
            **b.error_result(error), "letra": letter}


class FakeClient:
    def __init__(self, responses=(), gpu=None):
        self.responses = iter(responses)
        self.asked = []
        self.warmups = 0
        self.show = Mock()
        self.pull = Mock()
        self.ps = Mock(return_value=gpu or {"models": [
            {"model": "model:latest", "size": 100, "size_vram": 50},
        ]})

    def chat(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if prompt == "Oi":
            self.warmups += 1
            return response("Hello")
        self.asked.append(kwargs)
        value = next(self.responses)
        if isinstance(value, BaseException):
            raise value
        return value


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        self.input_dir.mkdir()
        self.csv = self.input_dir / "input.csv"
        self.write_rows([question(), question("p2"), question("p3")])
        self.log = io.StringIO()
        redirect = contextlib.redirect_stdout(self.log)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        # Any accidental network call fails, even if the real client is used.
        for target in ["socket.create_connection", "socket.socket.connect"]:
            block = patch(target, side_effect=AssertionError("Network is forbidden in tests"))
            block.start()
            self.addCleanup(block.stop)

    def write_rows(self, rows):
        pd.DataFrame(rows).to_csv(self.csv, index=False)

    def run_benchmark(self, client, **kwargs):
        return b.run_model(models=("model",), files=("input",),
                             csv_dir=self.input_dir, output_dir=self.output_dir,
                             client=client, **kwargs)

    def progress(self):
        return b.output_path("model", "input", self.output_dir).with_suffix(".progresso.json")

    def test_default_input_exists_and_has_unique_keys(self):
        args = b.create_parser().parse_args([])
        self.assertEqual(tuple(args.files), b.FILES)
        path, df = b.load_input(args.files[0])
        self.assertTrue(path.is_file())
        self.assertEqual(len(df), 300)
        self.assertFalse(df.duplicated(b.KEYS).any())

    def test_merge_across_exams_with_repeated_numbers(self):
        df = pd.DataFrame([question("p1", answer_key="A"), question("p2", answer_key="C")])
        total = b.merge_results(df, [record("p2"), record("p1", letter="A")])
        self.assertEqual(total["letra"].tolist(), ["A", "C"])
        self.assertEqual(b.build_report(total)["correct"], 2)
        self.assertEqual(len(total), 2)

    def test_merge_rejects_duplicates_on_both_sides(self):
        for rows, results in [([question(), question()], [record()]),
                                   ([question()], [record(), record()])]:
            with self.subTest(rows=rows), self.assertRaises(pd.errors.MergeError):
                b.merge_results(pd.DataFrame(rows), results)

    def test_invalid_input_before_any_call(self):
        for rows, message in [([question(), question()], "duplicate"),
                                 ([{k: v for k, v in question().items() if k != "A"}], "missing required columns"),
                                 ([question(exam="")], "missing values"),
                                 ([{**question(), "enunciado": ""}], "missing values")]:
            with self.subTest(message=message):
                self.write_rows(rows)
                client = Mock()
                with self.assertRaisesRegex(ValueError, message):
                    self.run_benchmark(client)
                self.assertEqual(client.mock_calls, [])
                self.assertFalse(self.output_dir.exists())

    def test_validates_all_files_before_ollama(self):
        client = Mock()
        with self.assertRaisesRegex(ValueError, "Could not read"):
            b.run_model(models=("model",), files=("input", "absent"),
                          csv_dir=self.input_dir, output_dir=self.output_dir, client=client)
        self.assertEqual(client.mock_calls, [])

    def test_empty_input_and_invalid_name(self):
        pd.DataFrame(columns=question().keys()).to_csv(self.csv, index=False)
        with self.assertRaisesRegex(ValueError, "no questions found"):
            b.load_input("input", self.input_dir)
        with self.assertRaisesRegex(ValueError, "Invalid input name"):
            b.load_input("../entrada", self.input_dir)

    def test_conservative_extraction(self):
        cases = {
            "Resposta: C": "C", "Resposta: **C**": "C", "Resposta: (D)": "D",
            "answer and rationale": None, "Resposta: the correct option is C": None,
            " **Resposta:**  **( D )** ": "D", "Resposta: `E`": "E",
            "Resposta: A ou C": None, "Resposta: C\nResposta: D": None,
            "Resposta: C\nou D": None, "Resposta: C\nResposta: C": "C",
            "Resposta: Casa": None, "A": None, "Resposta: (A) ou (B)": None,
            "Resposta: F": None, "Resposta: ** C **.": "C", None: None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(b.extract_letter(text), expected)

    def test_metrics_and_missing_or_none_fields(self):
        returns = [response(total_duration=None, prompt_eval_count=None, eval_count=None),
                    response(), {}, {"message": None}, {"message": {"content": None}},
                    ollama.ChatResponse(message=ollama.Message(role="assistant", content="Resposta: C"))]
        for ret in returns:
            with self.subTest(ret=ret):
                out = b.ask("model", "question", Mock(chat=Mock(return_value=ret)))
                for field in ["tempo_s", "tokens_prompt", "tokens_saida", "possivel_truncamento", "erro"]:
                    self.assertIsNone(out[field])
        self.assertEqual(out["letra"], "C")

    def test_known_metrics_and_heuristic(self):
        client = Mock(chat=Mock(return_value=response(total_duration=2_000_000_000,
                                                     prompt_eval_count=4090, eval_count=0)))
        out = b.ask("model", "question", client)
        self.assertEqual(out["tempo_s"], 2)
        self.assertEqual(out["tokens_saida"], 0)
        self.assertTrue(out["possivel_truncamento"])
        call = client.chat.call_args.kwargs
        self.assertEqual(call["options"], {"temperature": 0, "seed": 42, "num_ctx": 4096})
        self.assertEqual(call["messages"][0]["content"], b.SYSTEM)
        self.assertEqual(set(call), {"model", "messages", "options", "keep_alive"})

    def test_loading_cpu_partial_gpu_and_unknown(self):
        for size, vram, expected in [(100, 0, "CPU"), (100, 100, "GPU"),
                                       (100, 25, "CPU/GPU parcial"), (None, 1, "desconhecido"),
                                       (100, None, "desconhecido"), (100, 101, "desconhecido")]:
            with self.subTest(vram=vram):
                client = Mock(ps=Mock(return_value=SimpleNamespace(models=[
                    SimpleNamespace(model="model:latest", size=size, size_vram=vram)])))
                self.assertEqual(b.gpu_usage("model", client)["carregamento"], expected)

    def test_gpu_name_has_no_fuzzy_matching(self):
        client = Mock(ps=Mock(return_value={"models": [{"model": "model2:latest", "size": 100, "size_vram": 100}]}))
        self.assertEqual(b.gpu_usage("model", client)["carregamento"], "desconhecido")
        client.ps.return_value["models"][0]["model"] = "model"
        self.assertEqual(b.gpu_usage("model:latest", client)["carregamento"], "GPU")
        self.assertEqual(b.gpu_usage("model:q4", client)["carregamento"], "desconhecido")
        self.assertEqual(b.canonical_name("localhost:5000/model"), "localhost:5000/model:latest")

    def test_unavailable_gpu_query_preserves_answers(self):
        client = FakeClient([response()] * 3)
        client.ps.side_effect = RuntimeError("Query unavailable")
        self.run_benchmark(client)
        results = json.loads(self.progress().read_text())["resultados"]
        self.assertEqual([r["letra"] for r in results], ["C"] * 3)
        self.assertTrue(all(r["carregamento"] == "desconhecido" for r in results))
        self.assertTrue(all(r["erro"] is None for r in results))
        self.assertIn("Query unavailable", results[0]["erro_consulta_gpu"])

    def test_accuracy_excludes_invalid_answer_keys_without_losing_records(self):
        answer_keys = ["A", "X", None, "", "F", " c ", "B", "D", "C"]
        df = pd.DataFrame([question(number=str(i), answer_key=g) for i, g in enumerate(answer_keys)])
        results = [record(number=str(i), letter="A") for i in range(5)]
        results += [record(number="5"), record(number="6", letter=None),
                       record(number="7", letter=None, error="Failed")]
        total = b.merge_results(df, results)
        r = b.build_report(total)
        self.assertEqual(r, {"partial": True, "total_questions": 9, "processed": 8,
                             "excluded": 4, "evaluated": 4, "correct": 2, "accuracy": .5,
                             "no_valid_letter": 2, "inference_failures": 1})
        self.assertEqual(len(total), 9)
        self.assertTrue(pd.isna(total.loc[8, "acertou"]))
        self.assertTrue(pd.isna(total.loc[1, "acertou"]))
        self.assertFalse(total.loc[6, "acertou"])
        self.assertFalse(total.loc[7, "acertou"])

    def test_no_evaluated_questions_does_not_invent_accuracy(self):
        for results in [[], [record()]]:
            r = b.build_report(b.merge_results(pd.DataFrame([question(answer_key="X")]), results))
            self.assertIsNone(r["accuracy"])
            self.assertEqual(r["evaluated"], 0)
            self.assertEqual(r["correct"], 0)

    def test_incremental_interruption_and_resume_without_repeating_errors(self):
        client = FakeClient([response(), KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            self.run_benchmark(client)
        state = json.loads(self.progress().read_text())
        self.assertEqual(len(state["resultados"]), 2)
        self.assertIn("interrupted", state["resultados"][1]["erro"])
        self.assertIsNone(state["resultados"][0]["tokens_prompt"])
        self.assertIn("PARTIAL report", self.log.getvalue())
        self.assertIn("Processed: 2/3", self.log.getvalue())
        resumed = FakeClient([response()])
        self.run_benchmark(resumed, resume=True)
        self.assertEqual(len(resumed.asked), 1)
        self.assertIn("p3/1", resumed.asked[0]["messages"][1]["content"])
        self.assertEqual(resumed.warmups, 1)
        self.assertEqual(len(json.loads(self.progress().read_text())["resultados"]), 3)
        total = pd.read_csv(b.output_path("model", "input", self.output_dir))
        self.assertEqual(total["processada"].tolist(), [True] * 3)
        self.assertEqual(total["acertou"].tolist(), [True, False, True])
        self.assertIn("COMPLETE report", self.log.getvalue())

    def test_recorded_failures_are_not_resent(self):
        client = FakeClient([RuntimeError("Inference failure"), response(), response()])
        self.run_benchmark(client)
        none_client = Mock()
        self.run_benchmark(none_client, resume=True)
        self.assertEqual(none_client.mock_calls, [])
        self.assertIn("Inference failures: 1", self.log.getvalue())
        self.assertIn("Correct: 2", self.log.getvalue())

    def test_progress_is_saved_before_the_next_question(self):
        client = FakeClient([response()] * 3)
        original = client.chat
        def verify(**kwargs):
            if kwargs["messages"][1]["content"] != "Oi":
                state = json.loads(self.progress().read_text())
                self.assertEqual(len(state["resultados"]), len(client.asked))
            return original(**kwargs)
        client.chat = verify
        self.run_benchmark(client)
        self.assertEqual(len(client.asked), 3)
        self.assertEqual(client.warmups, 1)
        client.pull.assert_not_called()

    def test_ctrl_c_during_save_preserves_answer(self):
        original = b.save_progress
        interrupted = False
        def interrupt(path, config, results):
            nonlocal interrupted
            if results and not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            return original(path, config, results)
        with patch.object(b, "save_progress", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_benchmark(FakeClient([response()]))
        self.assertEqual(len(json.loads(self.progress().read_text())["resultados"]), 1)
        client = FakeClient([response(), response()])
        self.run_benchmark(client, resume=True)
        self.assertEqual(len(client.asked), 2)

    def test_ctrl_c_during_gpu_query_preserves_received_answer(self):
        client = FakeClient([response()])
        client.ps.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_benchmark(client)
        result = json.loads(self.progress().read_text())["resultados"][0]
        self.assertEqual(result["letra"], "C")
        self.assertIsNone(result["erro"])
        self.assertEqual(result["carregamento"], "desconhecido")

    def test_resume_with_several_models_preserves_finished_combinations(self):
        client = FakeClient([response()] * 3 + [KeyboardInterrupt()])
        arguments = dict(models=("model", "other"), files=("input",),
                          csv_dir=self.input_dir, output_dir=self.output_dir)
        with self.assertRaises(KeyboardInterrupt):
            b.run_model(client=client, **arguments)
        self.assertEqual(client.warmups, 2)
        resumed = FakeClient([response(), response()])
        b.run_model(client=resumed, resume=True, **arguments)
        self.assertEqual(resumed.warmups, 1)
        self.assertEqual([p["model"] for p in resumed.asked], ["other", "other"])
        self.assertEqual(len(json.loads(self.progress().read_text())["resultados"]), 3)

    def test_refuses_overwrite_without_calling_ollama(self):
        self.run_benchmark(FakeClient([response()] * 3))
        previous = self.progress().read_bytes()
        client = Mock()
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            self.run_benchmark(client)
        self.assertEqual(client.mock_calls, [])
        self.assertEqual(self.progress().read_bytes(), previous)

    def test_refuses_resume_without_progress(self):
        with self.assertRaisesRegex(ValueError, "Progress missing"):
            self.run_benchmark(Mock(), resume=True)

    def test_refuses_resume_with_incompatible_config(self):
        self.run_benchmark(FakeClient([response()] * 3))
        original = json.loads(self.progress().read_text())
        for field, value in [("modelo", "other"), ("arquivo", "other"), ("system", "other"),
                             ("template", "other"), ("opcoes", {"temperature": 1}),
                             ("entrada_sha256", "different"), ("raciocinio", "changed"),
                             ("keep_alive", "5m"), ("prompts_sha256", "different")]:
            with self.subTest(field=field):
                state = copy.deepcopy(original)
                state["configuracao"][field] = value
                self.progress().write_text(json.dumps(state))
                client = Mock()
                with self.assertRaisesRegex(ValueError, "Incompatible resume"):
                    self.run_benchmark(client, resume=True)
                self.assertEqual(client.mock_calls, [])

    def test_real_input_and_prompt_changes_prevent_resume(self):
        self.run_benchmark(FakeClient([response()] * 3))
        for field in ["SYSTEM", "TEMPLATE", "NUM_CTX"]:
            value = 1024 if field == "NUM_CTX" else getattr(b, field) + " modified"
            with self.subTest(field=field), patch.object(b, field, value):
                with self.assertRaisesRegex(ValueError, "Incompatible resume"):
                    self.run_benchmark(Mock(), resume=True)
        self.write_rows([question(answer_key="D"), question("p2"), question("p3")])
        with self.assertRaisesRegex(ValueError, "Incompatible resume"):
            self.run_benchmark(Mock(), resume=True)

    def test_resume_rejects_duplicate_keys(self):
        self.run_benchmark(FakeClient([response()] * 3))
        state = json.loads(self.progress().read_text())
        state["resultados"].append(state["resultados"][0])
        self.progress().write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            self.run_benchmark(Mock(), resume=True)

    def test_atomic_write_failure_preserves_progress(self):
        self.run_benchmark(FakeClient([response()] * 3))
        before = self.progress().read_bytes()
        def fail(f):
            f.write('{"incomplete":')
            raise OSError("Disk unavailable")
        with self.assertRaises(OSError):
            b.save_atomic(self.progress(), fail)
        self.assertEqual(self.progress().read_bytes(), before)
        self.assertEqual(len(list(self.output_dir.iterdir())), 2)

    def test_recovers_csv_without_new_inferences(self):
        self.run_benchmark(FakeClient([response()] * 3))
        output_dir = b.output_path("model", "input", self.output_dir)
        output_dir.write_text("Interrupted CSV")
        client = Mock()
        self.run_benchmark(client, resume=True)
        self.assertEqual(client.mock_calls, [])
        self.assertEqual(len(pd.read_csv(output_dir)), 3)

    def test_warmup_does_not_count_as_a_question(self):
        client = Mock(chat=Mock(side_effect=RuntimeError("Server unavailable")))
        with self.assertRaisesRegex(RuntimeError, "Warmup failed"):
            self.run_benchmark(client)
        self.assertEqual(json.loads(self.progress().read_text())["resultados"], [])
        self.assertIn("Processed: 0/3", self.log.getvalue())
        self.assertIn("Inference failures: 0", self.log.getvalue())

    def test_download_only_for_missing_model(self):
        for code in [404, 500]:
            client = Mock(show=Mock(side_effect=ollama.ResponseError("error", status_code=code)))
            if code == 404:
                b.ensure_model("model", client)
                client.pull.assert_called_once_with("model")
            else:
                with self.assertRaises(ollama.ResponseError):
                    b.ensure_model("model", client)
                client.pull.assert_not_called()

    def test_notebook_has_no_outputs_and_same_simulated_flow(self):
        notebook = json.loads((b.ROOT / "notebooks/teste_provas.ipynb").read_text())
        cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        for cell in cells:
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            compile("".join(cell["source"]), "notebook", "exec")
        # Run the original import and configuration cells; use a fixture and temporary directories.
        for cwd in [b.ROOT, b.ROOT / "notebooks"]:
            old_cwd = Path.cwd()
            try:
                os.chdir(cwd)
                context = {}
                exec("".join(cells[0]["source"]), context)
                exec("".join(cells[1]["source"]), context)
                self.assertEqual(context["ROOT"], b.ROOT)
            finally:
                os.chdir(old_cwd)
        path, df = b.load_input("input", self.input_dir)
        context.update(file="input", model="model", output_dir=self.output_dir,
                        path=path, exam_2015=df)
        client = FakeClient([response()] * 3)
        def run_notebook(**kwargs):
            kwargs["csv_dir"] = self.input_dir
            return b.run_model(client=client, **kwargs)
        context["run_model"] = run_notebook
        exec("".join(cells[2]["source"]), context)
        exec("".join(cells[3]["source"]), context)
        self.assertEqual(b.build_report(context["final_df"])["correct"], 3)
        self.assertEqual(len(client.asked), 3)

    def test_openrouter_does_not_pull_warm_up_or_query_ps_and_isolates_output(self):
        client = FakeClient([response()] * 3)
        self.run_benchmark(client, openrouter=True)
        client.show.assert_not_called()
        client.pull.assert_not_called()
        client.ps.assert_not_called()
        self.assertEqual(client.warmups, 0)
        self.assertEqual(len(client.asked), 3)
        self.assertNotIn("keep_alive", client.asked[0])
        self.assertEqual(client.asked[0]["options"], {"temperature": 0, "seed": 42})
        progress = b.output_path("model", "input", self.output_dir, openrouter=True).with_suffix(".progresso.json")
        state = json.loads(progress.read_text())
        self.assertEqual(state["configuracao"]["provedor"], "openrouter")
        self.assertNotIn("keep_alive", state["configuracao"])
        self.assertTrue(all(r["carregamento"] == "openrouter" for r in state["resultados"]))
        self.assertNotEqual(b.output_path("model", "input", self.output_dir),
                            b.output_path("model", "input", self.output_dir, openrouter=True))
        local = b.build_config("model", "input", *b.load_input("input", self.input_dir))
        self.assertNotIn("provedor", local)

    def test_openrouter_client_translates_response_and_errors(self):
        http = Mock()
        http.post.return_value = Mock(status_code=200, json=Mock(return_value={
            "choices": [{"message": {"content": "Resposta: B"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3}}))
        client = b.OpenRouterClient("secret", http=http, attempts=1)
        out = b.ask("a/b:free", "q", client, openrouter=True)
        self.assertEqual((out["letra"], out["tokens_prompt"], out["tokens_saida"]), ("B", 10, 3))
        self.assertIsNone(out["possivel_truncamento"])
        self.assertGreaterEqual(out["tempo_s"], 0)
        call = http.post.call_args
        self.assertEqual(call.args[0], b.OPENROUTER_URL)
        self.assertEqual(call.kwargs["headers"], {"Authorization": "Bearer secret"})
        self.assertEqual(call.kwargs["json"]["model"], "a/b:free")
        self.assertEqual(call.kwargs["json"]["seed"], 42)
        http.post.return_value = Mock(status_code=429, json=Mock(return_value={"error": {"message": "limit"}}))
        out = b.ask("a/b:free", "q", client, openrouter=True)
        self.assertIn("limit", out["erro"])
        self.assertIn("429", out["erro"])
        http.post.return_value = Mock(status_code=502, json=Mock(side_effect=ValueError))
        self.assertIn("502", b.ask("a/b:free", "q", client, openrouter=True)["erro"])
        for body in ({"choices": []}, {}, {"choices": [{"message": {"content": None}}]}):
            http.post.return_value = Mock(status_code=200, json=Mock(return_value=body))
            out = b.ask("a/b:free", "q", client, openrouter=True)
            self.assertIn("returned no response content", out["erro"])
            self.assertIsNone(out["letra"])

    def test_openrouter_retries_429_with_wait_and_shows_real_cause(self):
        def failure(status, **headers):
            return Mock(status_code=status, headers=headers, json=Mock(return_value={"error": {
                "message": "Provider returned error", "metadata": {"raw": "free-models-per-min"}}}))
        ok = Mock(status_code=200, headers={}, json=Mock(return_value={
            "choices": [{"message": {"content": "Resposta: A"}}]}))
        http, sleep = Mock(), Mock()
        http.post.side_effect = [failure(429, **{"Retry-After": "7"}), failure(502), ok]
        client = b.OpenRouterClient("k", http=http, attempts=5, sleep=sleep)
        self.assertEqual(b.ask("m", "q", client, openrouter=True)["letra"], "A")
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [7, 10])
        http.post.side_effect = [failure(429)] * 2
        out = b.ask("m", "q", b.OpenRouterClient("k", http=http, attempts=2, sleep=sleep), openrouter=True)
        self.assertIn("free-models-per-min", out["erro"])
        self.assertIn("429", out["erro"])
        http.post.side_effect = [Mock(status_code=401, headers={}, json=Mock(return_value={"error": {"message": "x"}}))]
        b.ask("m", "q", b.OpenRouterClient("k", http=http, attempts=5, sleep=sleep), openrouter=True)
        self.assertEqual(http.post.call_count, 6)  # 3 + 2 + 1: the 401 is not retried

    def test_openrouter_retry_after_above_cap_fails_without_retrying_early(self):
        http, sleep = Mock(), Mock()
        http.post.side_effect = [Mock(status_code=429, headers={"Retry-After": "301"},
                                      json=Mock(return_value={"error": {"message": "limit"}}))]
        out = b.ask("m", "q", b.OpenRouterClient("k", http=http, attempts=5, sleep=sleep), openrouter=True)
        self.assertEqual(http.post.call_count, 1)
        sleep.assert_not_called()
        self.assertIn("429", out["erro"])
        self.assertIn("exceeds the 300s cap", out["erro"])
        ok = Mock(status_code=200, headers={}, json=Mock(return_value={"choices": [{"message": {"content": "Resposta: D"}}]}))
        http.post.side_effect = [Mock(status_code=429, headers={"Retry-After": "300"}, json=Mock(return_value={})), ok]
        out = b.ask("m", "q", b.OpenRouterClient("k", http=http, attempts=5, sleep=sleep), openrouter=True)
        self.assertEqual(out["letra"], "D")
        sleep.assert_called_once_with(300.0)

    def test_redo_errors_reprocesses_openrouter_response_without_content(self):
        http = Mock()
        http.post.return_value = Mock(status_code=200, headers={}, json=Mock(return_value={"choices": []}))
        self.run_benchmark(b.OpenRouterClient("k", http=http, attempts=1), openrouter=True)
        client = FakeClient([response(), response(), response()])
        self.run_benchmark(client, openrouter=True, resume=True, redo_errors=True)
        self.assertEqual(len(client.asked), 3)
        progress = b.output_path("model", "input", self.output_dir, openrouter=True).with_suffix(".progresso.json")
        self.assertTrue(all(r["erro"] is None for r in json.loads(progress.read_text())["resultados"]))

    def test_redo_errors_reprocesses_only_failures(self):
        self.run_benchmark(FakeClient([response(), RuntimeError("429"), response()]), openrouter=True)
        client = FakeClient([response("Resposta: B")])
        with self.assertRaises(ValueError):
            self.run_benchmark(client, openrouter=True, redo_errors=True)
        self.run_benchmark(client, openrouter=True, resume=True, redo_errors=True)
        self.assertEqual(len(client.asked), 1)
        progress = b.output_path("model", "input", self.output_dir, openrouter=True).with_suffix(".progresso.json")
        results = json.loads(progress.read_text())["resultados"]
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r["erro"] is None for r in results))
        self.assertEqual(sorted(r["letra"] for r in results), ["B", "C", "C"])

    def test_openrouter_client_requires_key_and_reads_env_with_environment_priority(self):
        with self.assertRaises(RuntimeError):
            b.create_openrouter_client(environment={})
        self.assertEqual(b.create_openrouter_client(environment={"OPENROUTER_API_KEY": " x "}).key, "x")
        self.assertTrue(b.create_parser().parse_args(["--openrouter"]).openrouter)
        file = Path(self.tmp.name) / ".env"
        file.write_text('OPENROUTER_API_KEY="from_env"\n')
        with patch.object(b, "ENV_FILE", file), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(b.create_openrouter_client().key, "from_env")
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "from_shell"}):
                self.assertEqual(b.create_openrouter_client().key, "from_shell")
            file.write_text("")
            with self.assertRaises(RuntimeError):
                b.create_openrouter_client()


if __name__ == "__main__":
    unittest.main()
