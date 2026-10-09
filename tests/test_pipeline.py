"""Testes offline (sem rede, sem GPU) do pipeline: tarefas, cliente Jev, retomada e relatório."""
from __future__ import annotations

import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import httpx
import pytest

import bench.cli as cli
import bench.report as report
import bench.tasks as tasks
from bench.clients.base import Prediction
from bench.clients.clm import CLMClient
from bench.clients.jev import JevClient
from bench.clients.openai_decisions import OpenAIDecisionsClient
from bench.tasks import Case


def _case(cid="t-0", qtype="choice", gold="a", criteria=None, task="t"):
    q = {"type": qtype, "instructions": "?"}
    if qtype != "noul":
        q["criteria"] = criteria if criteria is not None else {"a": "A", "b": "B"}
    return Case(case_id=cid, task=task, qtype=qtype, state={"text": "x"}, questions={"q": q}, gold=gold)


def _pred(case, pred, ok=True, lat=10.0, **kw):
    return Prediction(case.case_id, case.task, "m", ok, None if ok else "err", pred,
                      kw.get("probs"), kw.get("p_true"), kw.get("score_value"), lat)


# ---------- tasks: regressão do bug do Banking77 ----------

def test_banking77_usa_label_text_e_nao_indices(monkeypatch):
    rows = [{"text": f"t{i}", "label": i % 77, "label_text": f"intent_{i % 77}"} for i in range(300)]
    monkeypatch.setattr(tasks, "load_dataset", lambda *a, **k: rows)
    cases = tasks.build_banking77(n_samples=10, seed=1)
    crit = cases[0].questions["intent"]["criteria"]
    assert len(crit) == 77
    assert "0" not in crit and "intent_0" in crit  # nunca índices numéricos
    assert all(c.gold.startswith("intent_") for c in cases)


def test_amostragem_e_deterministica(monkeypatch):
    rows = [{"text": f"t{i}", "label": i % 77, "label_text": f"intent_{i % 77}"} for i in range(300)]
    monkeypatch.setattr(tasks, "load_dataset", lambda *a, **k: rows)
    a = [c.state["text"] for c in tasks.build_banking77(n_samples=20, seed=7)]
    b = [c.state["text"] for c in tasks.build_banking77(n_samples=20, seed=7)]
    assert a == b


# ---------- cliente Jev (HTTP simulado) ----------

def _jev_with(handler):
    c = JevClient(api_key="k")
    c.client = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_jev_choice_noul_score_e_erro_http():
    def handler(req):
        q = list(json.loads(req.content)["questions"].keys())[0]
        return httpx.Response(200, json={"answers": {q: handler.answer}})

    jev = _jev_with(handler)

    handler.answer = {"type": "choice", "choice": "b", "probabilities": {"a": 0.2, "b": 0.8}}
    p = jev.predict(_case(gold="b"))
    assert p.ok and p.pred == "b" and p.probs["b"] == 0.8

    handler.answer = {"type": "noul", "noul": 0.9}
    p = jev.predict(_case(qtype="noul", gold=True))
    assert p.pred is True and p.p_true == 0.9

    handler.answer = {"type": "score", "score": 2.6}
    p = jev.predict(_case(qtype="score", gold=3, criteria=["0", "1", "2", "3", "4"]))
    assert p.pred == 3

    bad = _jev_with(lambda req: httpx.Response(500, text="boom"))
    p = bad.predict(_case())
    assert not p.ok and "HTTP 500" in p.error and p.pred is None


# ---------- retomada: falhas são reexecutadas, sem duplicatas ----------

class _FlakyClient:
    def __init__(self):
        self.calls = 0

    def predict(self, case):
        self.calls += 1
        return _pred(case, "a", ok=True)


def test_run_retoma_reexecutando_somente_falhas(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RAW_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(cli, "WARMUP_RUNS", 0)
    cases = [_case(f"c-{i}") for i in range(3)]
    out = tmp_path / "m.jsonl"
    out.write_text("\n".join(json.dumps(p.to_dict()) for p in [
        _pred(cases[0], "a"), _pred(cases[1], None, ok=False)]) + "\n")

    client = _FlakyClient()
    preds = cli.run_model_on_cases("m", client, cases)
    assert client.calls == 2  # c-1 (falhou antes) + c-2 (nunca rodou); c-0 não é refeito
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert sorted(l["case_id"] for l in lines) == ["c-0", "c-1", "c-2"]  # sem duplicatas
    assert all(l["ok"] for l in lines) and len(preds) == 3


# ---------- relatório: coluna "Melhor" é calculada, não fixa ----------

def test_best_respeita_direcao_da_metrica():
    assert report._best({"jev": 0.9, "laya": 0.8}, lower_is_better=False) == "JEV"
    assert report._best({"jev": 0.9, "laya": 0.8}, lower_is_better=True) == "LAYA"
    assert report._best({"jev": None}, lower_is_better=True) == "-"


def test_n_labels_vem_do_caso():
    assert report._n_labels(_case(criteria={"a": "", "b": "", "c": ""})) == 3
    assert report._n_labels(_case(qtype="noul", gold=True)) == 2


def test_relatorio_gera_sem_rede_e_pareado(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "generate_charts", lambda *a, **k: None)
    cases = [_case(f"t-{i}", gold="a") for i in range(4)]
    preds = {
        "jev": [_pred(c, "a") for c in cases],
        "laya": [_pred(c, "a" if i < 2 else "b") for i, c in enumerate(cases)],
        "laya-tuned": [_pred(c, "a" if i < 3 else "b") for i, c in enumerate(cases)],
        "clm": [_pred(c, "a" if i < 3 else "b") for i, c in enumerate(cases)],
    }
    md = report.generate_report(cases, preds, report_file=tmp_path / "r.md")
    assert "100.0%" in md and "50.0%" in md and "75.0%" in md
    assert "CLM-8B" in md
    assert "Efeito do Ajuste" in md
    assert (tmp_path / "r.md").exists()


# ---------- cliente CLM (HTTP simulado) ----------

def _clm_with(handler):
    c = CLMClient(api_key="k")
    c.client = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_clm_choice_noul_score_e_erro_http():
    def handler(req):
        q = list(json.loads(req.content)["questions"].keys())[0]
        return httpx.Response(200, json={"answers": {q: handler.answer}})

    clm = _clm_with(handler)

    handler.answer = {"choice": "b", "probabilities": {"a": 0.2, "b": 0.8}}
    p = clm.predict(_case(gold="b"))
    assert p.ok and p.pred == "b" and p.probs["b"] == 0.8 and p.model == "clm"

    handler.answer = {"p_true": 0.95}
    p = clm.predict(_case(qtype="noul", gold=True))
    assert p.ok and p.pred is True and p.p_true == 0.95

    handler.answer = {"score": 2.8}
    p = clm.predict(_case(qtype="score", gold=3, criteria=["0", "1", "2", "3", "4"]))
    assert p.ok and p.pred == 3 and p.score_value == 2.8

    bad = _clm_with(lambda req: httpx.Response(500, text="internal server error"))
    p = bad.predict(_case())
    assert not p.ok and "HTTP 500" in p.error and p.pred is None


# ---------- cliente OpenAI Decisions (HTTP simulado) ----------

def _openai_with(handler):
    c = OpenAIDecisionsClient(api_key="k")
    c.client = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_openai_choice_noul_score_e_erro_http():
    def handler(req):
        q = list(json.loads(req.content)["questions"].keys())[0]
        return httpx.Response(200, json={"answers": {q: handler.answer}})

    openai_cli = _openai_with(handler)

    handler.answer = {"type": "choice", "choice": "b", "probabilities": {"a": 0.2, "b": 0.8}}
    p = openai_cli.predict(_case(gold="b"))
    assert p.ok and p.pred == "b" and p.probs["b"] == 0.8 and p.model == "openai"

    handler.answer = {"type": "noul", "noul": 0.9}
    p = openai_cli.predict(_case(qtype="noul", gold=True))
    assert p.ok and p.pred is True and p.p_true == 0.9 and p.model == "openai"

    handler.answer = {"type": "score", "score": 2.6}
    p = openai_cli.predict(_case(qtype="score", gold=3, criteria=["0", "1", "2", "3", "4"]))
    assert p.ok and p.pred == 3 and p.model == "openai"

    bad = _openai_with(lambda req: httpx.Response(500, text="internal error"))
    p = bad.predict(_case())
    assert not p.ok and "HTTP 500" in p.error and p.pred is None

