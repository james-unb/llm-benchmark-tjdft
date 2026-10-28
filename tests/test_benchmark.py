"""Regressões isoladas: nenhum teste acessa o serviço Ollama."""

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

from Lmm_teste_1 import teste_modelo_aut as b


def resposta(texto="Resposta: C", **metricas):
    return {"message": {"content": texto}, **metricas}


def questao(prova="p1", numero="1", gabarito="C"):
    return {"id_prova": prova, "Nquestao": numero, "enunciado": f"Questão {prova}/{numero}",
            **{letra: f"Alternativa {letra}" for letra in "ABCDE"}, "gabarito": gabarito}


def registro(prova="p1", numero="1", letra="C", erro=None):
    return {"modelo": "modelo", "arquivo": "entrada", "id_prova": prova, "Nquestao": numero,
            "carregamento": "desconhecido", "erro_consulta_gpu": None,
            **b.resultado_erro(erro), "letra": letra}


class ClienteSimulado:
    def __init__(self, respostas=(), gpu=None):
        self.respostas = iter(respostas)
        self.perguntas = []
        self.aquecimentos = 0
        self.show = Mock()
        self.pull = Mock()
        self.ps = Mock(return_value=gpu or {"models": [
            {"model": "modelo:latest", "size": 100, "size_vram": 50},
        ]})

    def chat(self, **kwargs):
        prompt = kwargs["messages"][1]["content"]
        if prompt == "Oi":
            self.aquecimentos += 1
            return resposta("Olá")
        self.perguntas.append(kwargs)
        valor = next(self.respostas)
        if isinstance(valor, BaseException):
            raise valor
        return valor


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raiz = Path(self.tmp.name)
        self.entrada = self.raiz / "entrada"
        self.saida = self.raiz / "saida"
        self.entrada.mkdir()
        self.csv = self.entrada / "entrada.csv"
        self.gravar([questao(), questao("p2"), questao("p3")])
        self.log = io.StringIO()
        redirecionar = contextlib.redirect_stdout(self.log)
        redirecionar.__enter__()
        self.addCleanup(redirecionar.__exit__, None, None, None)
        # Qualquer chamada de rede acidental falha, mesmo se o cliente real for usado.
        for alvo in ["socket.create_connection", "socket.socket.connect"]:
            bloqueio = patch(alvo, side_effect=AssertionError("Rede proibida nos testes"))
            bloqueio.start()
            self.addCleanup(bloqueio.stop)

    def gravar(self, linhas):
        pd.DataFrame(linhas).to_csv(self.csv, index=False)

    def rodar(self, cliente, **kwargs):
        return b.rodar_modelo(modelos=("modelo",), arquivos=("entrada",),
                             pasta_csv=self.entrada, pasta_saida=self.saida,
                             cliente=cliente, **kwargs)

    def progresso(self):
        return b.caminho_saida("modelo", "entrada", self.saida).with_suffix(".progresso.json")

    def test_padrao_existe_e_tem_chaves_unicas(self):
        args = b.criar_parser().parse_args([])
        self.assertEqual(tuple(args.arquivos), b.ARQUIVOS)
        caminho, df = b.carregar_entrada(args.arquivos[0])
        self.assertTrue(caminho.is_file())
        self.assertEqual(len(df), 300)
        self.assertFalse(df.duplicated(b.CHAVES).any())

    def test_associacao_entre_provas_com_numeros_repetidos(self):
        df = pd.DataFrame([questao("p1", gabarito="A"), questao("p2", gabarito="C")])
        total = b.associar_resultados(df, [registro("p2"), registro("p1", letra="A")])
        self.assertEqual(total["letra"].tolist(), ["A", "C"])
        self.assertEqual(b.relatorio(total)["acertos"], 2)
        self.assertEqual(len(total), 2)

    def test_associacao_rejeita_duplicatas_dos_dois_lados(self):
        for linhas, resultados in [([questao(), questao()], [registro()]),
                                   ([questao()], [registro(), registro()])]:
            with self.subTest(linhas=linhas), self.assertRaises(pd.errors.MergeError):
                b.associar_resultados(pd.DataFrame(linhas), resultados)

    def test_entrada_invalida_antes_de_qualquer_chamada(self):
        for linhas, mensagem in [([questao(), questao()], "duplicadas"),
                                 ([{k: v for k, v in questao().items() if k != "A"}], "colunas"),
                                 ([questao(prova="")], "ausentes"),
                                 ([{**questao(), "enunciado": ""}], "ausentes")]:
            with self.subTest(mensagem=mensagem):
                self.gravar(linhas)
                cliente = Mock()
                with self.assertRaisesRegex(ValueError, mensagem):
                    self.rodar(cliente)
                self.assertEqual(cliente.mock_calls, [])
                self.assertFalse(self.saida.exists())

    def test_valida_todos_arquivos_antes_do_ollama(self):
        cliente = Mock()
        with self.assertRaisesRegex(ValueError, "Não foi possível ler"):
            b.rodar_modelo(modelos=("modelo",), arquivos=("entrada", "ausente"),
                          pasta_csv=self.entrada, pasta_saida=self.saida, cliente=cliente)
        self.assertEqual(cliente.mock_calls, [])

    def test_entrada_vazia_e_nome_invalido(self):
        pd.DataFrame(columns=questao().keys()).to_csv(self.csv, index=False)
        with self.assertRaisesRegex(ValueError, "nenhuma questão"):
            b.carregar_entrada("entrada", self.entrada)
        with self.assertRaisesRegex(ValueError, "Nome de entrada inválido"):
            b.carregar_entrada("../entrada", self.entrada)

    def test_extracao_conservadora(self):
        casos = {
            "Resposta: C": "C", "Resposta: **C**": "C", "Resposta: (D)": "D",
            "resposta e justificativa": None, "Resposta: a alternativa correta é C": None,
            " **Resposta:**  **( D )** ": "D", "Resposta: `E`": "E",
            "Resposta: A ou C": None, "Resposta: C\nResposta: D": None,
            "Resposta: C\nou D": None, "Resposta: C\nResposta: C": "C",
            "Resposta: Casa": None, "A": None, "Resposta: (A) ou (B)": None,
            "Resposta: F": None, "Resposta: ** C **.": "C", None: None,
        }
        for texto, esperado in casos.items():
            with self.subTest(texto=texto):
                self.assertEqual(b.extrair_letra(texto), esperado)

    def test_metricas_e_campos_ausentes_ou_none(self):
        retornos = [resposta(total_duration=None, prompt_eval_count=None, eval_count=None),
                    resposta(), {}, {"message": None}, {"message": {"content": None}},
                    ollama.ChatResponse(message=ollama.Message(role="assistant", content="Resposta: C"))]
        for retorno in retornos:
            with self.subTest(retorno=retorno):
                out = b.perguntar("modelo", "questão", Mock(chat=Mock(return_value=retorno)))
                for campo in ["tempo_s", "tokens_prompt", "tokens_saida", "possivel_truncamento", "erro"]:
                    self.assertIsNone(out[campo])
        self.assertEqual(out["letra"], "C")

    def test_metricas_conhecidas_e_heuristica(self):
        cliente = Mock(chat=Mock(return_value=resposta(total_duration=2_000_000_000,
                                                     prompt_eval_count=4090, eval_count=0)))
        out = b.perguntar("modelo", "pergunta", cliente)
        self.assertEqual(out["tempo_s"], 2)
        self.assertEqual(out["tokens_saida"], 0)
        self.assertTrue(out["possivel_truncamento"])
        chamada = cliente.chat.call_args.kwargs
        self.assertEqual(chamada["options"], {"temperature": 0, "seed": 42, "num_ctx": 4096})
        self.assertEqual(chamada["messages"][0]["content"], b.SYSTEM)
        self.assertEqual(set(chamada), {"model", "messages", "options", "keep_alive"})

    def test_carregamento_cpu_gpu_parcial_e_desconhecido(self):
        for tamanho, vram, esperado in [(100, 0, "CPU"), (100, 100, "GPU"),
                                       (100, 25, "CPU/GPU parcial"), (None, 1, "desconhecido"),
                                       (100, None, "desconhecido"), (100, 101, "desconhecido")]:
            with self.subTest(vram=vram):
                cliente = Mock(ps=Mock(return_value=SimpleNamespace(models=[
                    SimpleNamespace(model="modelo:latest", size=tamanho, size_vram=vram)])))
                self.assertEqual(b.usou_gpu("modelo", cliente)["carregamento"], esperado)

    def test_nome_gpu_nao_faz_correspondencia_aproximada(self):
        cliente = Mock(ps=Mock(return_value={"models": [{"model": "modelo2:latest", "size": 100, "size_vram": 100}]}))
        self.assertEqual(b.usou_gpu("modelo", cliente)["carregamento"], "desconhecido")
        cliente.ps.return_value["models"][0]["model"] = "modelo"
        self.assertEqual(b.usou_gpu("modelo:latest", cliente)["carregamento"], "GPU")
        self.assertEqual(b.usou_gpu("modelo:q4", cliente)["carregamento"], "desconhecido")
        self.assertEqual(b.nome_canonico("localhost:5000/modelo"), "localhost:5000/modelo:latest")

    def test_consulta_gpu_indisponivel_preserva_respostas(self):
        cliente = ClienteSimulado([resposta()] * 3)
        cliente.ps.side_effect = RuntimeError("Consulta indisponível")
        self.rodar(cliente)
        resultados = json.loads(self.progresso().read_text())["resultados"]
        self.assertEqual([r["letra"] for r in resultados], ["C"] * 3)
        self.assertTrue(all(r["carregamento"] == "desconhecido" for r in resultados))
        self.assertTrue(all(r["erro"] is None for r in resultados))
        self.assertIn("Consulta indisponível", resultados[0]["erro_consulta_gpu"])

    def test_acuracia_exclui_gabaritos_invalidos_sem_perder_registros(self):
        gabaritos = ["A", "X", None, "", "F", " c ", "B", "D", "C"]
        df = pd.DataFrame([questao(numero=str(i), gabarito=g) for i, g in enumerate(gabaritos)])
        resultados = [registro(numero=str(i), letra="A") for i in range(5)]
        resultados += [registro(numero="5"), registro(numero="6", letra=None),
                       registro(numero="7", letra=None, erro="Falhou")]
        total = b.associar_resultados(df, resultados)
        r = b.relatorio(total)
        self.assertEqual(r, {"parcial": True, "total_questoes": 9, "processadas": 8,
                             "excluidas": 4, "avaliadas": 4, "acertos": 2, "acuracia": .5,
                             "sem_letra_valida": 2, "falhas_inferencia": 1})
        self.assertEqual(len(total), 9)
        self.assertTrue(pd.isna(total.loc[8, "acertou"]))
        self.assertTrue(pd.isna(total.loc[1, "acertou"]))
        self.assertFalse(total.loc[6, "acertou"])
        self.assertFalse(total.loc[7, "acertou"])

    def test_sem_avaliadas_nao_inventa_acuracia(self):
        for resultados in [[], [registro()]]:
            r = b.relatorio(b.associar_resultados(pd.DataFrame([questao(gabarito="X")]), resultados))
            self.assertIsNone(r["acuracia"])
            self.assertEqual(r["avaliadas"], 0)
            self.assertEqual(r["acertos"], 0)

    def test_incremental_interrupcao_e_retomada_sem_repetir_erros(self):
        cliente = ClienteSimulado([resposta(), KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            self.rodar(cliente)
        estado = json.loads(self.progresso().read_text())
        self.assertEqual(len(estado["resultados"]), 2)
        self.assertIn("interrompida", estado["resultados"][1]["erro"])
        self.assertIsNone(estado["resultados"][0]["tokens_prompt"])
        self.assertIn("Relatório PARCIAL", self.log.getvalue())
        self.assertIn("Processadas: 2/3", self.log.getvalue())
        retomada = ClienteSimulado([resposta()])
        self.rodar(retomada, retomar=True)
        self.assertEqual(len(retomada.perguntas), 1)
        self.assertIn("p3/1", retomada.perguntas[0]["messages"][1]["content"])
        self.assertEqual(retomada.aquecimentos, 1)
        self.assertEqual(len(json.loads(self.progresso().read_text())["resultados"]), 3)
        total = pd.read_csv(b.caminho_saida("modelo", "entrada", self.saida))
        self.assertEqual(total["processada"].tolist(), [True] * 3)
        self.assertEqual(total["acertou"].tolist(), [True, False, True])
        self.assertIn("Relatório COMPLETO", self.log.getvalue())

    def test_falhas_registradas_nao_sao_reenviadas(self):
        cliente = ClienteSimulado([RuntimeError("Falha de inferência"), resposta(), resposta()])
        self.rodar(cliente)
        nenhum = Mock()
        self.rodar(nenhum, retomar=True)
        self.assertEqual(nenhum.mock_calls, [])
        self.assertIn("Falhas de inferência: 1", self.log.getvalue())
        self.assertIn("Acertos: 2", self.log.getvalue())

    def test_progresso_e_salvo_antes_da_proxima_questao(self):
        cliente = ClienteSimulado([resposta()] * 3)
        original = cliente.chat
        def verificar(**kwargs):
            if kwargs["messages"][1]["content"] != "Oi":
                estado = json.loads(self.progresso().read_text())
                self.assertEqual(len(estado["resultados"]), len(cliente.perguntas))
            return original(**kwargs)
        cliente.chat = verificar
        self.rodar(cliente)
        self.assertEqual(len(cliente.perguntas), 3)
        self.assertEqual(cliente.aquecimentos, 1)
        cliente.pull.assert_not_called()

    def test_ctrl_c_durante_salvamento_preserva_resposta(self):
        original = b.salvar_progresso
        interrompeu = False
        def interromper(caminho, config, resultados):
            nonlocal interrompeu
            if resultados and not interrompeu:
                interrompeu = True
                raise KeyboardInterrupt
            return original(caminho, config, resultados)
        with patch.object(b, "salvar_progresso", side_effect=interromper):
            with self.assertRaises(KeyboardInterrupt):
                self.rodar(ClienteSimulado([resposta()]))
        self.assertEqual(len(json.loads(self.progresso().read_text())["resultados"]), 1)
        cliente = ClienteSimulado([resposta(), resposta()])
        self.rodar(cliente, retomar=True)
        self.assertEqual(len(cliente.perguntas), 2)

    def test_ctrl_c_na_consulta_gpu_preserva_resposta_recebida(self):
        cliente = ClienteSimulado([resposta()])
        cliente.ps.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.rodar(cliente)
        resultado = json.loads(self.progresso().read_text())["resultados"][0]
        self.assertEqual(resultado["letra"], "C")
        self.assertIsNone(resultado["erro"])
        self.assertEqual(resultado["carregamento"], "desconhecido")

    def test_retomada_de_varios_modelos_preserva_combinacoes_concluidas(self):
        cliente = ClienteSimulado([resposta()] * 3 + [KeyboardInterrupt()])
        argumentos = dict(modelos=("modelo", "outro"), arquivos=("entrada",),
                          pasta_csv=self.entrada, pasta_saida=self.saida)
        with self.assertRaises(KeyboardInterrupt):
            b.rodar_modelo(cliente=cliente, **argumentos)
        self.assertEqual(cliente.aquecimentos, 2)
        retomada = ClienteSimulado([resposta(), resposta()])
        b.rodar_modelo(cliente=retomada, retomar=True, **argumentos)
        self.assertEqual(retomada.aquecimentos, 1)
        self.assertEqual([p["model"] for p in retomada.perguntas], ["outro", "outro"])
        self.assertEqual(len(json.loads(self.progresso().read_text())["resultados"]), 3)

    def test_recusa_sobrescrita_sem_chamar_ollama(self):
        self.rodar(ClienteSimulado([resposta()] * 3))
        anterior = self.progresso().read_bytes()
        cliente = Mock()
        with self.assertRaisesRegex(ValueError, "Saída já existe"):
            self.rodar(cliente)
        self.assertEqual(cliente.mock_calls, [])
        self.assertEqual(self.progresso().read_bytes(), anterior)

    def test_recusa_retomada_sem_progresso(self):
        with self.assertRaisesRegex(ValueError, "Progresso ausente"):
            self.rodar(Mock(), retomar=True)

    def test_recusa_retomada_com_configuracao_incompativel(self):
        self.rodar(ClienteSimulado([resposta()] * 3))
        original = json.loads(self.progresso().read_text())
        for campo, valor in [("modelo", "outro"), ("arquivo", "outro"), ("system", "outro"),
                             ("template", "outro"), ("opcoes", {"temperature": 1}),
                             ("entrada_sha256", "diferente"), ("raciocinio", "alterado"),
                             ("keep_alive", "5m"), ("prompts_sha256", "diferente")]:
            with self.subTest(campo=campo):
                estado = copy.deepcopy(original)
                estado["configuracao"][campo] = valor
                self.progresso().write_text(json.dumps(estado))
                cliente = Mock()
                with self.assertRaisesRegex(ValueError, "Retomada incompatível"):
                    self.rodar(cliente, retomar=True)
                self.assertEqual(cliente.mock_calls, [])

    def test_mudanca_real_da_entrada_e_prompt_impede_retomada(self):
        self.rodar(ClienteSimulado([resposta()] * 3))
        for campo in ["SYSTEM", "TEMPLATE", "NUM_CTX"]:
            valor = 1024 if campo == "NUM_CTX" else getattr(b, campo) + " modificado"
            with self.subTest(campo=campo), patch.object(b, campo, valor):
                with self.assertRaisesRegex(ValueError, "Retomada incompatível"):
                    self.rodar(Mock(), retomar=True)
        self.gravar([questao(gabarito="D"), questao("p2"), questao("p3")])
        with self.assertRaisesRegex(ValueError, "Retomada incompatível"):
            self.rodar(Mock(), retomar=True)

    def test_retomada_rejeita_chaves_duplicadas(self):
        self.rodar(ClienteSimulado([resposta()] * 3))
        estado = json.loads(self.progresso().read_text())
        estado["resultados"].append(estado["resultados"][0])
        self.progresso().write_text(json.dumps(estado))
        with self.assertRaisesRegex(ValueError, "chave duplicada"):
            self.rodar(Mock(), retomar=True)

    def test_falha_de_escrita_atomica_preserva_progresso(self):
        self.rodar(ClienteSimulado([resposta()] * 3))
        antes = self.progresso().read_bytes()
        def falhar(f):
            f.write('{"incompleto":')
            raise OSError("Disco indisponível")
        with self.assertRaises(OSError):
            b.salvar_atomico(self.progresso(), falhar)
        self.assertEqual(self.progresso().read_bytes(), antes)
        self.assertEqual(len(list(self.saida.iterdir())), 2)

    def test_recupera_csv_sem_novas_inferencias(self):
        self.rodar(ClienteSimulado([resposta()] * 3))
        saida = b.caminho_saida("modelo", "entrada", self.saida)
        saida.write_text("CSV interrompido")
        cliente = Mock()
        self.rodar(cliente, retomar=True)
        self.assertEqual(cliente.mock_calls, [])
        self.assertEqual(len(pd.read_csv(saida)), 3)

    def test_aquecimento_nao_conta_como_questao(self):
        cliente = Mock(chat=Mock(side_effect=RuntimeError("Servidor indisponível")))
        with self.assertRaisesRegex(RuntimeError, "aquecimento"):
            self.rodar(cliente)
        self.assertEqual(json.loads(self.progresso().read_text())["resultados"], [])
        self.assertIn("Processadas: 0/3", self.log.getvalue())
        self.assertIn("Falhas de inferência: 0", self.log.getvalue())

    def test_download_apenas_para_modelo_ausente(self):
        for codigo in [404, 500]:
            cliente = Mock(show=Mock(side_effect=ollama.ResponseError("erro", status_code=codigo)))
            if codigo == 404:
                b.garantir_modelo("modelo", cliente)
                cliente.pull.assert_called_once_with("modelo")
            else:
                with self.assertRaises(ollama.ResponseError):
                    b.garantir_modelo("modelo", cliente)
                cliente.pull.assert_not_called()

    def test_notebook_sem_saidas_e_com_mesmo_fluxo_simulado(self):
        notebook = json.loads((b.RAIZ / "Lmm_teste_1/teste_provas.ipynb").read_text())
        celulas = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        for celula in celulas:
            self.assertEqual(celula["outputs"], [])
            self.assertIsNone(celula["execution_count"])
            compile("".join(celula["source"]), "notebook", "exec")
        # Executar importação e configuração originais; usar fixture e diretório temporários.
        for cwd in [b.RAIZ, b.RAIZ / "Lmm_teste_1"]:
            antigo = Path.cwd()
            try:
                os.chdir(cwd)
                contexto = {}
                exec("".join(celulas[0]["source"]), contexto)
                exec("".join(celulas[1]["source"]), contexto)
                self.assertEqual(contexto["RAIZ"], b.RAIZ)
            finally:
                os.chdir(antigo)
        caminho, df = b.carregar_entrada("entrada", self.entrada)
        contexto.update(arquivo="entrada", modelo="modelo", pasta_saida=self.saida,
                        caminho=caminho, prova_2015=df)
        cliente = ClienteSimulado([resposta()] * 3)
        def rodar_notebook(**kwargs):
            kwargs["pasta_csv"] = self.entrada
            return b.rodar_modelo(cliente=cliente, **kwargs)
        contexto["rodar_modelo"] = rodar_notebook
        exec("".join(celulas[2]["source"]), contexto)
        exec("".join(celulas[3]["source"]), contexto)
        self.assertEqual(b.relatorio(contexto["df_final"])["acertos"], 3)
        self.assertEqual(len(cliente.perguntas), 3)


if __name__ == "__main__":
    unittest.main()
