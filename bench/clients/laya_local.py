from __future__ import annotations

import time
from typing import Any
import torch
from laya import Router
from bench.clients.base import Prediction
from bench.tasks import Case

class LayaClient:
    def __init__(self, tuned: bool = False, device: str | None = None):
        self.tuned = tuned
        self.model_name = "laya-tuned" if tuned else "laya"
        
        # Determine device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        t0 = time.perf_counter()
        # Initialize Router with preloading
        self.router = Router(preload=True, device=self.device)
        self.load_time_ms = (time.perf_counter() - t0) * 1000.0

        if self.tuned:
            # Tune head_max_len to 512 for high-cardinality choice questions
            for name, agent in getattr(self.router, "agents", {}).items():
                if hasattr(agent, "cfg"):
                    agent.cfg["head_max_len"] = 512
                    agent.cfg["max_len"] = 1024

    def predict(self, case: Case) -> Prediction:
        q_name = next(iter(case.questions.keys()))
        t0 = time.perf_counter()
        try:
            extra_kwargs: dict[str, Any] = {}
            if self.tuned:
                extra_kwargs["max_len"] = 1024
            
            res = self.router.predict(case.state, case.questions, **extra_kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            answers = res.get("answers", {})
            q_res = answers.get(q_name, {})

            pred = None
            probs = None
            p_true = None
            score_value = None

            if case.qtype == "choice":
                pred = q_res.get("choice")
                probs = q_res.get("probs") or q_res.get("probabilities")
                if not pred and probs:
                    pred = max(probs, key=probs.get)
            elif case.qtype == "score":
                score_value = q_res.get("score")
                probs = q_res.get("probs") or q_res.get("probabilities")
                if score_value is not None:
                    pred = round(float(score_value))
                elif probs:
                    try:
                        keys = sorted([int(k) for k in probs.keys()])
                        ev = sum(k * float(probs[str(k)]) for k in keys)
                        score_value = ev
                        pred = round(ev)
                    except Exception:
                        pass
            elif case.qtype == "noul":
                p_true = q_res.get("noul")
                if p_true is not None:
                    pred = bool(float(p_true) >= 0.5)

            return Prediction(
                case_id=case.case_id,
                task=case.task,
                model=self.model_name,
                ok=True,
                error=None,
                pred=pred,
                probs=probs,
                p_true=p_true,
                score_value=score_value,
                latency_ms=latency_ms,
                raw=res,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return Prediction(
                case_id=case.case_id,
                task=case.task,
                model=self.model_name,
                ok=False,
                error=str(e),
                pred=None,
                probs=None,
                p_true=None,
                score_value=None,
                latency_ms=latency_ms,
                raw=None,
            )
