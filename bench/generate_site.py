from __future__ import annotations

import json
from pathlib import Path
import shutil

from bench.clients.base import Prediction
from bench.metrics import (
    compute_accuracy,
    compute_brier_score,
    compute_ece,
    compute_latency_stats,
    compute_macro_f1,
    compute_score_metrics,
)
from bench.tasks import Case

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RAW_DIR = RESULTS_DIR / "raw"
MANIFEST_PATH = RESULTS_DIR / "manifest.jsonl"
TEMPLATE_PATH = ROOT_DIR / "bench" / "site_template.html"
SITE_DIR = ROOT_DIR / "site"
DATA_JS_PATH = SITE_DIR / "data.js"
INDEX_HTML_PATH = SITE_DIR / "index.html"
ARTIFACT_DIR = Path("/Users/gb/.gemini/antigravity/brain/5b48ad1e-d42e-4dc7-bac3-88620be649f0")

TASK_DISPLAY_NAMES = {
    "ag_news": "AG News",
    "banking77": "Banking77",
    "emotion": "Emotion",
    "sst5": "SST-5",
    "sst2": "SST-2",
}

TASK_CLASSES = {
    "ag_news": 4,
    "banking77": 77,
    "emotion": 6,
    "sst5": 5,
    "sst2": 2,
}


def main() -> None:
    print("Carregando manifest e predições brutas...")
    manifest_items = {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            manifest_items[item["case_id"]] = item

    jev_raw = {}
    with open(RAW_DIR / "jev.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            jev_raw[item["case_id"]] = item

    laya_raw = {}
    with open(RAW_DIR / "laya.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            laya_raw[item["case_id"]] = item

    lt_raw = {}
    with open(RAW_DIR / "laya-tuned.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            lt_raw[item["case_id"]] = item

    tasks_meta = {}
    cases = []
    task_cases_map: dict[str, list[dict]] = {}

    for cid, m in manifest_items.items():
        t = m["task"]
        qtype = m["qtype"]
        if t not in tasks_meta:
            q_data = list(m["questions"].values())[0] if m.get("questions") else {}
            tasks_meta[t] = {
                "task": t,
                "type": qtype,
                "instructions": q_data.get("instructions", ""),
                "criteria": q_data.get("criteria") or q_data.get("range") or {},
            }

        j = jev_raw.get(cid, {})
        l = laya_raw.get(cid, {})
        lt = lt_raw.get(cid, {})

        gold = m["gold"]
        txt = (
            m["state"].get("text")
            or m["state"].get("review")
            or m["state"].get("sentence")
            or ""
        )

        if qtype == "score":
            try:
                j_ok = bool(j.get("ok", False) and int(j.get("pred")) == int(gold))
            except Exception:
                j_ok = False
            try:
                l_ok = bool(l.get("ok", False) and int(l.get("pred")) == int(gold))
            except Exception:
                l_ok = False
            try:
                lt_ok = bool(lt.get("ok", False) and int(lt.get("pred")) == int(gold))
            except Exception:
                lt_ok = False
        elif qtype == "noul":
            j_ok = bool(
                j.get("ok", False)
                and str(j.get("pred")).lower() == str(gold).lower()
            )
            l_ok = bool(
                l.get("ok", False)
                and str(l.get("pred")).lower() == str(gold).lower()
            )
            lt_ok = bool(
                lt.get("ok", False)
                and str(lt.get("pred")).lower() == str(gold).lower()
            )
        else:
            j_ok = bool(j.get("ok", False) and str(j.get("pred")) == str(gold))
            l_ok = bool(l.get("ok", False) and str(l.get("pred")) == str(gold))
            lt_ok = bool(lt.get("ok", False) and str(lt.get("pred")) == str(gold))

        j_lat = round(float(j.get("latency_ms", 0.0)), 1)
        l_lat = round(float(l.get("latency_ms", 0.0)), 1)
        lt_lat = round(float(lt.get("latency_ms", 0.0)), 1)

        c_dict = {
            "id": cid,
            "task": t,
            "type": qtype,
            "text": txt,
            "gold": gold,
            # Jev
            "jev_pred": j.get("pred"),
            "jev_ok": j_ok,
            "jev_lat": j_lat,
            "jev_probs": j.get("probs"),
            # Laya Base
            "laya_pred": l.get("pred"),
            "laya_ok": l_ok,
            "laya_lat": l_lat,
            "laya_probs": l.get("probs"),
            # Laya Tuned
            "lt_pred": lt.get("pred"),
            "lt_ok": lt_ok,
            "lt_lat": lt_lat,
            "lt_probs": lt.get("probs"),
            # Short aliases for compatibility
            "jp": j.get("pred"),
            "jok": j_ok,
            "jl": j_lat,
            "jpr": j.get("probs"),
            "lp": l.get("pred"),
            "lok": l_ok,
            "ll": l_lat,
            "lpr": l.get("probs"),
            "ltp": lt.get("pred"),
            "ltok": lt_ok,
            "ltl": lt_lat,
            "ltpr": lt.get("probs"),
        }
        cases.append(c_dict)
        task_cases_map.setdefault(t, []).append(c_dict)

    # Convert to typed objects for metric calculations
    typed_cases = []
    typed_j_preds = []
    typed_l_preds = []
    typed_lt_preds = []

    for c in cases:
        cid = c["id"]
        m = manifest_items[cid]
        case_obj = Case.from_dict(m)
        typed_cases.append(case_obj)

        j_raw = jev_raw.get(cid, {})
        l_raw = laya_raw.get(cid, {})
        lt_raw_item = lt_raw.get(cid, {})

        typed_j_preds.append(
            Prediction(
                case_id=cid,
                task=m["task"],
                model="jev",
                ok=j_raw.get("ok", False),
                error=j_raw.get("error"),
                pred=j_raw.get("pred"),
                probs=j_raw.get("probs"),
                p_true=j_raw.get("p_true"),
                score_value=j_raw.get("score_value"),
                latency_ms=j_raw.get("latency_ms", 0.0),
                raw=j_raw.get("raw"),
            )
        )
        typed_l_preds.append(
            Prediction(
                case_id=cid,
                task=m["task"],
                model="laya",
                ok=l_raw.get("ok", False),
                error=l_raw.get("error"),
                pred=l_raw.get("pred"),
                probs=l_raw.get("probs"),
                p_true=l_raw.get("p_true"),
                score_value=l_raw.get("score_value"),
                latency_ms=l_raw.get("latency_ms", 0.0),
                raw=l_raw.get("raw"),
            )
        )
        typed_lt_preds.append(
            Prediction(
                case_id=cid,
                task=m["task"],
                model="laya-tuned",
                ok=lt_raw_item.get("ok", False),
                error=lt_raw_item.get("error"),
                pred=lt_raw_item.get("pred"),
                probs=lt_raw_item.get("probs"),
                p_true=lt_raw_item.get("p_true"),
                score_value=lt_raw_item.get("score_value"),
                latency_ms=lt_raw_item.get("latency_ms", 0.0),
                raw=lt_raw_item.get("raw"),
            )
        )

    all_golds = [c.gold for c in typed_cases]
    j_pvals = [p.pred for p in typed_j_preds]
    l_pvals = [p.pred for p in typed_l_preds]
    lt_pvals = [p.pred for p in typed_lt_preds]

    j_lats = [p.latency_ms for p in typed_j_preds if p.ok and p.latency_ms > 0]
    l_lats = [p.latency_ms for p in typed_l_preds if p.ok and p.latency_ms > 0]
    lt_lats = [p.latency_ms for p in typed_lt_preds if p.ok and p.latency_ms > 0]

    j_st = compute_latency_stats(j_lats)
    l_st = compute_latency_stats(l_lats)
    lt_st = compute_latency_stats(lt_lats)

    jev_acc_overall = round(compute_accuracy(all_golds, j_pvals) * 100, 1)
    laya_acc_overall = round(compute_accuracy(all_golds, l_pvals) * 100, 1)
    lt_acc_overall = round(compute_accuracy(all_golds, lt_pvals) * 100, 1)

    jev_f1_overall = f"{compute_macro_f1(all_golds, j_pvals):.3f}"
    laya_f1_overall = f"{compute_macro_f1(all_golds, l_pvals):.3f}"
    lt_f1_overall = f"{compute_macro_f1(all_golds, lt_pvals):.3f}"

    # Build per-dataset dictionary
    datasets_summary = {}
    for task_name in ["ag_news", "banking77", "emotion", "sst5", "sst2"]:
        t_indices = [i for i, c in enumerate(typed_cases) if c.task == task_name]
        t_cases = [typed_cases[i] for i in t_indices]
        t_j_preds = [typed_j_preds[i] for i in t_indices]
        t_l_preds = [typed_l_preds[i] for i in t_indices]
        t_lt_preds = [typed_lt_preds[i] for i in t_indices]

        t_golds = [c.gold for c in t_cases]
        t_j_acc = round(compute_accuracy(t_golds, [p.pred for p in t_j_preds]) * 100, 1)
        t_l_acc = round(compute_accuracy(t_golds, [p.pred for p in t_l_preds]) * 100, 1)
        t_lt_acc = round(compute_accuracy(t_golds, [p.pred for p in t_lt_preds]) * 100, 1)

        t_j_p50 = round(compute_latency_stats([p.latency_ms for p in t_j_preds if p.ok])["p50"], 1)
        t_l_p50 = round(compute_latency_stats([p.latency_ms for p in t_l_preds if p.ok])["p50"], 1)
        t_lt_p50 = round(compute_latency_stats([p.latency_ms for p in t_lt_preds if p.ok])["p50"], 1)

        d_entry = {
            "name": TASK_DISPLAY_NAMES.get(task_name, task_name),
            "type": t_cases[0].qtype,
            "classes": TASK_CLASSES.get(task_name, len(set(t_golds))),
            "n": len(t_cases),
            "jev_acc": t_j_acc,
            "laya_acc": t_l_acc,
            "laya_tuned_acc": t_lt_acc,
            "jev_lat": t_j_p50,
            "laya_lat": t_l_p50,
            "laya_tuned_lat": t_lt_p50,
        }

        # Calibration & task-specific metrics
        if task_name in ("ag_news", "emotion", "banking77", "sst2"):
            b_j = compute_brier_score(t_cases, t_j_preds)
            b_l = compute_brier_score(t_cases, t_l_preds)
            e_j = compute_ece(t_cases, t_j_preds)
            e_l = compute_ece(t_cases, t_l_preds)
            if b_j is not None:
                d_entry["brier_jev"] = round(b_j, 4)
            if b_l is not None:
                d_entry["brier_laya"] = round(b_l, 4)
            if e_j is not None:
                d_entry["ece_jev"] = round(e_j, 4)
            if e_l is not None:
                d_entry["ece_laya"] = round(e_l, 4)

        if task_name == "sst5":
            sm_j = compute_score_metrics(t_cases, t_j_preds)
            sm_l = compute_score_metrics(t_cases, t_l_preds)
            if sm_j["mae"] is not None:
                d_entry["mae_jev"] = round(sm_j["mae"], 3)
            if sm_l["mae"] is not None:
                d_entry["mae_laya"] = round(sm_l["mae"], 3)
            if sm_j["spearman"] is not None:
                d_entry["spearman_jev"] = round(sm_j["spearman"], 3)
            if sm_l["spearman"] is not None:
                d_entry["spearman_laya"] = round(sm_l["spearman"], 3)

        # Winner flag
        if t_j_acc > t_lt_acc + 1.0:
            d_entry["winner"] = "jev"
        elif t_lt_acc > t_j_acc + 1.0:
            d_entry["winner"] = "laya"
        else:
            d_entry["winner"] = "tie"

        datasets_summary[task_name] = d_entry

    summary_stats = {
        "total_cases": len(cases),
        "jev_acc": jev_acc_overall,
        "laya_acc": laya_acc_overall,
        "laya_tuned_acc": lt_acc_overall,
        "jev_lat_p50": round(j_st["p50"], 1),
        "jev_lat_p95": round(j_st["p95"], 1),
        "laya_lat_p50": round(l_st["p50"], 1),
        "laya_lat_p95": round(l_st["p95"], 1),
        "laya_tuned_lat_p50": round(lt_st["p50"], 1),
        "laya_tuned_lat_p95": round(lt_st["p95"], 1),
        "datasets": datasets_summary,
    }

    bundle = {
        "summary": summary_stats,
        "tasks": tasks_meta,
        "cases": cases,
    }

    SITE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Write site/data.js
    with open(DATA_JS_PATH, "w", encoding="utf-8") as f:
        f.write("window.BENCHMARK_DATA = " + json.dumps(bundle, ensure_ascii=False) + ";")
    print(f"site/data.js gerado com sucesso ({len(cases)} casos).")

    # 2. Read template and substitute tokens
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    replacements = {
        "{{TOTAL}}": str(len(cases)),
        "{{JEV_ACC}}": f"{jev_acc_overall:.1f}",
        "{{JEV_N}}": str(sum(1 for c in cases if c["jev_ok"])),
        "{{JEV_F1}}": jev_f1_overall,
        "{{JEV_P50}}": f"{j_st['p50']:.1f}",
        "{{LAYA_ACC}}": f"{laya_acc_overall:.1f}",
        "{{LAYA_N}}": str(sum(1 for c in cases if c["laya_ok"])),
        "{{LAYA_F1}}": laya_f1_overall,
        "{{LAYA_P50}}": f"{l_st['p50']:.1f}",
        "{{TUNED_ACC}}": f"{lt_acc_overall:.1f}",
        "{{TUNED_N}}": str(sum(1 for c in cases if c["lt_ok"])),
        "{{TUNED_F1}}": lt_f1_overall,
        "{{TUNED_P50}}": f"{lt_st['p50']:.1f}",
        "{{BK_TUNED_LAT}}": f"{datasets_summary['banking77']['laya_tuned_lat']:.1f}",
        "__EMBEDDED_DATA__": json.dumps(bundle, ensure_ascii=False),
    }

    for token, val in replacements.items():
        assert token in html, f"Token {token} não encontrado em {TEMPLATE_PATH}"
        html = html.replace(token, val)

    # Verify no unreplaced double-brace tokens remain
    import re
    leftover = re.findall(r"\{\{[A-Z0-9_]+\}\}", html)
    if leftover:
        raise ValueError(f"Tokens não substituídos encontrados no HTML: {leftover}")

    # 3. Write site/index.html
    with open(INDEX_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"site/index.html gerado com sucesso ({INDEX_HTML_PATH}).")

    # 4. Copy to brain artifact directory if accessible
    if ARTIFACT_DIR.is_dir():
        target_art = ARTIFACT_DIR / "benchmark_dashboard.html"
        shutil.copyfile(INDEX_HTML_PATH, target_art)
        print(f"Artifact atualizado em: {target_art}")

    print("\nResumo Final da Geração do Site:")
    print(f"- Casos: {len(cases)}")
    print(f"- Jev Acc: {jev_acc_overall}% | F1: {jev_f1_overall} | p50: {j_st['p50']:.1f}ms")
    print(f"- Laya Base Acc: {laya_acc_overall}% | F1: {laya_f1_overall} | p50: {l_st['p50']:.1f}ms")
    print(f"- Laya Tuned Acc: {lt_acc_overall}% | F1: {lt_f1_overall} | p50: {lt_st['p50']:.1f}ms")


if __name__ == "__main__":
    main()
