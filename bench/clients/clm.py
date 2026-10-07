from __future__ import annotations

import time
from typing import Any
import httpx
from bench.clients.base import Prediction
from bench.config import CLM_API_KEY, CLM_ENDPOINT, CLM_MODEL, TIMEOUT_SECONDS
from bench.tasks import Case


class CLMClient:
    """Cliente para o modelo CLM-8B (Contrastive Language Model - Stanford & NVIDIA).
    
    O CLM-8B opera como modelo System 1 que utiliza o encoder Qwen3-8B com cabeçotes
    de projeção treinados via contraste InfoNCE. Ele é compatível com o formato wire
    do TypeSafe (/v1/systemone ou endpoint similar servido pelo `clm-serve`).
    """

    def __init__(
        self,
        endpoint: str = CLM_ENDPOINT,
        model: str = CLM_MODEL,
        api_key: str | None = None,
        timeout: float = TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key or CLM_API_KEY
        self.timeout = timeout

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self.client = httpx.Client(
            headers=headers,
            timeout=self.timeout,
            transport=transport,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> CLMClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def predict(self, case: Case) -> Prediction:
        payload = {
            "model": self.model,
            "state": case.state,
            "questions": case.questions,
        }
        q_name = next(iter(case.questions.keys()))
        t0 = time.perf_counter()

        try:
            resp = self.client.post(self.endpoint, json=payload)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if resp.status_code != 200:
                return Prediction(
                    case_id=case.case_id,
                    task=case.task,
                    model="clm",
                    ok=False,
                    error=f"HTTP {resp.status_code}: {resp.text}",
                    pred=None,
                    probs=None,
                    p_true=None,
                    score_value=None,
                    latency_ms=latency_ms,
                    raw={"status_code": resp.status_code, "body": resp.text},
                )

            data = resp.json()
            answers = data.get("answers", data)
            q_res = answers.get(q_name, {})

            pred = None
            probs = None
            p_true = None
            score_value = None

            if case.qtype == "choice":
                pred = q_res.get("choice")
                probs = q_res.get("probabilities") or q_res.get("probs")
                if not pred and probs:
                    pred = max(probs, key=probs.get)
            elif case.qtype == "score":
                score_value = q_res.get("score")
                probs = q_res.get("probabilities") or q_res.get("probs")
                if score_value is not None:
                    pred = round(float(score_value))
                elif probs:
                    try:
                        criteria = case.questions[q_name].get("criteria", [])
                        if all(k in probs for k in criteria):
                            ev = sum(i * float(probs[crit]) for i, crit in enumerate(criteria))
                            score_value = ev
                            pred = round(ev)
                        else:
                            keys = sorted([int(k) for k in probs.keys() if str(k).isdigit()])
                            ev = sum(k * float(probs[str(k)]) for k in keys)
                            score_value = ev
                            pred = round(ev)
                    except Exception:
                        pred = max(probs, key=probs.get)
            elif case.qtype == "noul":
                p_true = q_res.get("p_true")
                if p_true is None and "probabilities" in q_res:
                    probs = q_res["probabilities"]
                    p_true = probs.get("true", probs.get("True"))
                elif p_true is None and "probs" in q_res:
                    probs = q_res["probs"]
                    p_true = probs.get("true", probs.get("True"))
                if p_true is not None:
                    p_true = float(p_true)
                    pred = p_true >= 0.5
                else:
                    pred = q_res.get("choice") or q_res.get("answer")

            return Prediction(
                case_id=case.case_id,
                task=case.task,
                model="clm",
                ok=True,
                error=None,
                pred=pred,
                probs=probs,
                p_true=p_true,
                score_value=score_value,
                latency_ms=latency_ms,
                raw=data,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            err_msg = (
                f"Erro de conexão com CLM-8B ({self.endpoint}): {e}. "
                "Certifique-se de que o servidor 'clm-serve' está em execução "
                "ou configure a variável de ambiente CLM_ENDPOINT no arquivo .env."
            )
            return Prediction(
                case_id=case.case_id,
                task=case.task,
                model="clm",
                ok=False,
                error=err_msg,
                pred=None,
                probs=None,
                p_true=None,
                score_value=None,
                latency_ms=latency_ms,
                raw={"exception": str(e)},
            )
