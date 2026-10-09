from __future__ import annotations

import time
from typing import Any
import httpx
from bench.clients.base import Prediction
from bench.config import OPENAI_DECISION_MODEL, OPENROUTER_API_KEY, OPENROUTER_ENDPOINT, TIMEOUT_SECONDS
from bench.tasks import Case

class OpenAIDecisionsClient:
    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = OPENROUTER_ENDPOINT,
        model: str = OPENAI_DECISION_MODEL,
        timeout: float = TIMEOUT_SECONDS,
    ):
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is missing! Please provide it in .env or via OPENROUTER_API_KEY environment variable."
            )
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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
                    model="openai",
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
                        pass
            elif case.qtype == "noul":
                p_true = q_res.get("noul")
                if p_true is not None:
                    pred = bool(float(p_true) >= 0.5)
                elif "choice" in q_res:
                    pred = bool(str(q_res["choice"]).lower() in ["true", "yes", "positive"])

            return Prediction(
                case_id=case.case_id,
                task=case.task,
                model="openai",
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
            return Prediction(
                case_id=case.case_id,
                task=case.task,
                model="openai",
                ok=False,
                error=str(e),
                pred=None,
                probs=None,
                p_true=None,
                score_value=None,
                latency_ms=latency_ms,
                raw=None,
            )
