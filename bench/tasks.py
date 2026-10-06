from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
from datasets import load_dataset
from bench.config import SEED, SAMPLES_PER_TASK

@dataclass
class Case:
    case_id: str
    task: str
    qtype: str  # "choice" | "score" | "noul"
    state: dict[str, Any]
    questions: dict[str, Any]
    gold: str | int | bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Case:
        return cls(**data)


def build_ag_news(n_samples: int = SAMPLES_PER_TASK, seed: int = SEED) -> list[Case]:
    ds = load_dataset("fancyzhx/ag_news", split="test")
    label_names = ["world", "sports", "business", "sci_tech"]
    criteria = {
        "world": "World news, international affairs, global politics",
        "sports": "Sports, athletic events, games, competitions",
        "business": "Business, economy, finance, stock market, companies",
        "sci_tech": "Science, technology, engineering, computers, software",
    }
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    cases = []
    for i, idx in enumerate(indices):
        item = ds[int(idx)]
        gold_label = label_names[int(item["label"])]
        cases.append(
            Case(
                case_id=f"ag_news-{i:04d}",
                task="ag_news",
                qtype="choice",
                state={"text": item["text"]},
                questions={
                    "topic": {
                        "type": "choice",
                        "instructions": "What is the primary topic of this news text?",
                        "criteria": criteria,
                    }
                },
                gold=gold_label,
            )
        )
    return cases


def build_banking77(n_samples: int = SAMPLES_PER_TASK, seed: int = SEED) -> list[Case]:
    ds = load_dataset("mteb/banking77", split="test")
    
    # Map label integer to real label text names
    int_to_name: dict[int, str] = {}
    for item in ds:
        int_to_name[int(item["label"])] = item["label_text"]
        if len(int_to_name) == 77:
            break
    
    names = [int_to_name[i] for i in range(77)]
    criteria = {name: name.replace("_", " ") for name in names}
    
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    cases = []
    for i, idx in enumerate(indices):
        item = ds[int(idx)]
        gold_label = item["label_text"]
        cases.append(
            Case(
                case_id=f"banking77-{i:04d}",
                task="banking77",
                qtype="choice",
                state={"text": item["text"]},
                questions={
                    "intent": {
                        "type": "choice",
                        "instructions": "Which banking intent category best matches the user inquiry?",
                        "criteria": criteria,
                    }
                },
                gold=gold_label,
            )
        )
    return cases


def build_emotion(n_samples: int = SAMPLES_PER_TASK, seed: int = SEED) -> list[Case]:
    ds = load_dataset("dair-ai/emotion", "split", split="test")
    label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    criteria = {
        "sadness": "Feeling sad, sorrowful, depressed, grieving",
        "joy": "Feeling happy, cheerful, delighted, joyful",
        "love": "Feeling loving, affectionate, warm, caring",
        "anger": "Feeling angry, annoyed, furious, irritated",
        "fear": "Feeling afraid, terrified, anxious, scared",
        "surprise": "Feeling surprised, shocked, astonished, amazed",
    }
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    cases = []
    for i, idx in enumerate(indices):
        item = ds[int(idx)]
        gold_label = label_names[int(item["label"])]
        cases.append(
            Case(
                case_id=f"emotion-{i:04d}",
                task="emotion",
                qtype="choice",
                state={"text": item["text"]},
                questions={
                    "emotion": {
                        "type": "choice",
                        "instructions": "Which emotion is predominantly expressed in the text?",
                        "criteria": criteria,
                    }
                },
                gold=gold_label,
            )
        )
    return cases


def build_sst5(n_samples: int = SAMPLES_PER_TASK, seed: int = SEED) -> list[Case]:
    ds = load_dataset("SetFit/sst5", split="test")
    criteria = ["very negative", "negative", "neutral", "positive", "very positive"]
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    cases = []
    for i, idx in enumerate(indices):
        item = ds[int(idx)]
        gold_score = int(item["label"])  # 0 to 4
        cases.append(
            Case(
                case_id=f"sst5-{i:04d}",
                task="sst5",
                qtype="score",
                state={"review": item["text"]},
                questions={
                    "sentiment_rating": {
                        "type": "score",
                        "instructions": "On a scale from very negative to very positive, rate the sentiment of this review.",
                        "criteria": criteria,
                    }
                },
                gold=gold_score,
            )
        )
    return cases


def build_sst2(n_samples: int = SAMPLES_PER_TASK, seed: int = SEED) -> list[Case]:
    ds = load_dataset("stanfordnlp/sst2", split="validation")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    cases = []
    for i, idx in enumerate(indices):
        item = ds[int(idx)]
        gold_bool = bool(int(item["label"]) == 1)
        cases.append(
            Case(
                case_id=f"sst2-{i:04d}",
                task="sst2",
                qtype="noul",
                state={"review": item["sentence"]},
                questions={
                    "is_positive": {
                        "type": "noul",
                        "instructions": "Is the sentiment of this review positive?",
                    }
                },
                gold=gold_bool,
            )
        )
    return cases


def get_smoke_cases() -> list[Case]:
    """Small deterministic set of 5 cases (one per task) for quick sanity smoke test."""
    return [
        Case(
            case_id="smoke-ag_news-001",
            task="ag_news",
            qtype="choice",
            state={"text": "NASA launched a new telescope into orbit to study distant exoplanets."},
            questions={
                "topic": {
                    "type": "choice",
                    "instructions": "What is the primary topic of this news text?",
                    "criteria": {
                        "world": "World news and international affairs",
                        "sports": "Sports and athletics",
                        "business": "Business and economy",
                        "sci_tech": "Science and technology",
                    },
                }
            },
            gold="sci_tech",
        ),
        Case(
            case_id="smoke-banking77-001",
            task="banking77",
            qtype="choice",
            state={"text": "I was charged an extra fee for an exchange transaction that I did not approve."},
            questions={
                "intent": {
                    "type": "choice",
                    "instructions": "Which banking intent category best matches the user inquiry?",
                    "criteria": {
                        "card_payment_fee_charged": "card payment fee charged",
                        "exchange_rate": "exchange rate",
                        "extra_charge_on_statement": "extra charge on statement",
                        "atm_fee": "atm fee",
                    },
                }
            },
            gold="extra_charge_on_statement",
        ),
        Case(
            case_id="smoke-emotion-001",
            task="emotion",
            qtype="choice",
            state={"text": "I am so excited and thrilled for the amazing news today!"},
            questions={
                "emotion": {
                    "type": "choice",
                    "instructions": "Which emotion is predominantly expressed in the text?",
                    "criteria": {
                        "sadness": "Feeling sad",
                        "joy": "Feeling happy, cheerful, thrilled",
                        "anger": "Feeling angry",
                        "fear": "Feeling afraid",
                    },
                }
            },
            gold="joy",
        ),
        Case(
            case_id="smoke-sst5-001",
            task="sst5",
            qtype="score",
            state={"review": "An extraordinary, deeply moving masterpiece that exceeded all expectations."},
            questions={
                "sentiment_rating": {
                    "type": "score",
                    "instructions": "On a scale from very negative to very positive, rate the sentiment.",
                    "criteria": ["very negative", "negative", "neutral", "positive", "very positive"],
                }
            },
            gold=4,
        ),
        Case(
            case_id="smoke-sst2-001",
            task="sst2",
            qtype="noul",
            state={"review": "A dreadful waste of time with poor acting and no plot."},
            questions={
                "is_positive": {
                    "type": "noul",
                    "instructions": "Is the sentiment of this review positive?",
                }
            },
            gold=False,
        ),
    ]
