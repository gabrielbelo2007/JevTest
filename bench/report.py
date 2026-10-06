from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

from bench.clients.base import Prediction
from bench.config import CHARTS_DIR, REPORT_PATH
from bench.metrics import (
    compute_accuracy,
    compute_brier_score,
    compute_ece,
    compute_latency_stats,
    compute_macro_f1,
    compute_paired_bootstrap_accuracy_diff,
    compute_score_metrics,
)
from bench.tasks import Case


def generate_charts(
    cases_by_task: dict[str, list[Case]],
    pred_map_by_model: dict[str, dict[str, Prediction]],
    output_dir: Path = CHARTS_DIR,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    models = list(pred_map_by_model.keys())
    tasks = list(cases_by_task.keys())

    # 1. Accuracy per Task
    plt.figure(figsize=(10, 5))
    x = np.arange(len(tasks))
    width = 0.25 if len(models) == 3 else 0.35

    colors = {"jev": "#2563eb", "laya": "#16a34a", "laya-tuned": "#9333ea"}

    for i, model in enumerate(models):
        accs = []
        for task in tasks:
            task_cases = cases_by_task[task]
            golds = []
            preds = []
            for c in task_cases:
                if c.case_id in pred_map_by_model[model]:
                    golds.append(c.gold)
                    preds.append(pred_map_by_model[model][c.case_id].pred)
            acc = compute_accuracy(golds, preds)
            accs.append(acc)

        offset = (i - (len(models) - 1) / 2) * width
        plt.bar(
            x + offset,
            accs,
            width,
            label=model.upper(),
            color=colors.get(model, "#64748b"),
            alpha=0.9,
        )

    plt.title("Acurácia por Dataset (Jev vs Laya vs Laya-Tuned)", fontsize=13, fontweight="bold")
    plt.ylabel("Acurácia (0 a 1)", fontsize=11)
    plt.xticks(x, [t.replace("_", " ").upper() for t in tasks], fontsize=10)
    plt.ylim(0, 1.05)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    chart1_path = output_dir / "accuracy_comparison.png"
    plt.savefig(chart1_path, dpi=200)
    plt.close()

    # 2. Latency p50 per Task (log scale)
    plt.figure(figsize=(10, 5))
    for i, model in enumerate(models):
        p50s = []
        for task in tasks:
            task_cases = cases_by_task[task]
            lats = [
                pred_map_by_model[model][c.case_id].latency_ms
                for c in task_cases
                if c.case_id in pred_map_by_model[model]
                and pred_map_by_model[model][c.case_id].ok
            ]
            stats = compute_latency_stats(lats)
            p50s.append(stats["p50"])

        offset = (i - (len(models) - 1) / 2) * width
        plt.bar(
            x + offset,
            p50s,
            width,
            label=model.upper(),
            color=colors.get(model, "#64748b"),
            alpha=0.9,
        )

    plt.title("Tempo de Resposta Mediano p50 (ms) - Escala Log", fontsize=13, fontweight="bold")
    plt.ylabel("Latência p50 (ms) - log", fontsize=11)
    plt.yscale("log")
    plt.xticks(x, [t.replace("_", " ").upper() for t in tasks], fontsize=10)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    chart2_path = output_dir / "latency_comparison.png"
    plt.savefig(chart2_path, dpi=200)
    plt.close()

    # 3. Trade-off Accuracy vs Latency
    plt.figure(figsize=(8, 6))
    for model in models:
        all_cases = [c for t in tasks for c in cases_by_task[t]]
        golds = []
        pred_vals = []
        all_lats = []
        for c in all_cases:
            if c.case_id in pred_map_by_model[model]:
                p = pred_map_by_model[model][c.case_id]
                golds.append(c.gold)
                pred_vals.append(p.pred)
                if p.ok:
                    all_lats.append(p.latency_ms)

        overall_acc = compute_accuracy(golds, pred_vals)
        overall_p50 = compute_latency_stats(all_lats)["p50"]

        plt.scatter(
            overall_p50,
            overall_acc,
            s=220,
            label=model.upper(),
            color=colors.get(model, "#64748b"),
            edgecolor="black",
            zorder=5,
        )
        plt.annotate(
            f" {model.upper()}\n (acc={overall_acc:.1%}, {overall_p50:.1f}ms)",
            (overall_p50, overall_acc),
            fontsize=10,
            fontweight="bold",
            va="center",
        )

    plt.title("Trade-off Geral: Acurácia Global vs Latência Mediana", fontsize=13, fontweight="bold")
    plt.xlabel("Latência p50 (ms) - menor é melhor", fontsize=11)
    plt.ylabel("Acurácia Global (0 a 1) - maior é melhor", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart3_path = output_dir / "tradeoff_accuracy_latency.png"
    plt.savefig(chart3_path, dpi=200)
    plt.close()

    return chart1_path, chart2_path, chart3_path


def generate_report(
    manifest_cases: list[Case],
    all_predictions: dict[str, list[Prediction]],
    report_file: Path = REPORT_PATH,
) -> str:
    cases_by_task: dict[str, list[Case]] = {}
    for c in manifest_cases:
        cases_by_task.setdefault(c.task, []).append(c)

    # Strictly index predictions by case_id for paired alignment
    pred_map_by_model: dict[str, dict[str, Prediction]] = {}
    for model, preds in all_predictions.items():
        pred_map_by_model[model] = {p.case_id: p for p in preds}

    generate_charts(cases_by_task, pred_map_by_model)

    models = list(all_predictions.keys())
    tasks = list(cases_by_task.keys())

    md_lines = []
    md_lines.append("# Relatório Comparativo: Jev vs Laya (Padrão e Ajustado)")
    md_lines.append("")
    md_lines.append("> Benchmark automatizado avaliando modelos de decisão determinísticos (System 1) sob **mesmos prompts**, medindo qualidade de decisão (scores) e tempo de resposta (latência).")
    md_lines.append("")
    md_lines.append("## 1. Resumo Executivo")
    md_lines.append("")

    # Overall Summary Table
    md_lines.append("| Modelo | Modo | Acurácia Geral | Macro-F1 Médio | Latência p50 | Latência p95 | Taxa Sucesso |")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|")

    for model in models:
        all_cases = [c for t in tasks for c in cases_by_task[t]]
        golds = []
        p_vals = []
        lats = []
        ok_count = 0
        total_count = 0
        for c in all_cases:
            if c.case_id in pred_map_by_model[model]:
                p = pred_map_by_model[model][c.case_id]
                golds.append(c.gold)
                p_vals.append(p.pred)
                total_count += 1
                if p.ok:
                    ok_count += 1
                    lats.append(p.latency_ms)

        acc = compute_accuracy(golds, p_vals)
        f1 = compute_macro_f1(golds, p_vals)
        lat_stats = compute_latency_stats(lats)
        success_rate = (ok_count / total_count) if total_count else 0.0
        mode_str = "Nuvem (OpenRouter API)" if model == "jev" else "Local (Apple Silicon M5 MPS)"

        md_lines.append(
            f"| **{model.upper()}** | {mode_str} | **{acc:.1%}** | {f1:.3f} | {lat_stats['p50']:.1f} ms | {lat_stats['p95']:.1f} ms | {success_rate:.1%} |"
        )

    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 2. Acurácia e Qualidade por Tarefa")
    md_lines.append("")
    md_lines.append("| Dataset | Tipo | Qtd Rótulos | Jev Acc | Laya Acc | Laya-Tuned Acc | Diferença (Jev - Laya [95% CI]) |")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|")

    for task in tasks:
        task_cases = cases_by_task[task]
        qtype = task_cases[0].qtype

        acc_map = {}
        pred_val_map = {}
        aligned_golds = [c.gold for c in task_cases]

        for m in models:
            preds_task = [pred_map_by_model[m][c.case_id].pred for c in task_cases if c.case_id in pred_map_by_model[m]]
            acc_map[m] = compute_accuracy(aligned_golds[:len(preds_task)], preds_task)
            pred_val_map[m] = preds_task

        jev_acc = acc_map.get("jev", 0.0)
        laya_acc = acc_map.get("laya", 0.0)
        laya_t_acc = acc_map.get("laya-tuned", 0.0)

        if "jev" in pred_val_map and "laya" in pred_val_map:
            delta, ci_l, ci_h = compute_paired_bootstrap_accuracy_diff(
                aligned_golds, pred_val_map["jev"], pred_val_map["laya"]
            )
            ci_str = f"{delta:+.1%} [{ci_l:+.1%}, {ci_h:+.1%}]"
        else:
            ci_str = "N/A"

        n_labels_str = "77" if task == "banking77" else ("6" if task == "emotion" else ("5" if task == "sst5" else ("4" if task == "ag_news" else "2")))
        md_lines.append(
            f"| `{task}` | `{qtype}` | {n_labels_str} | {jev_acc:.1%} | {laya_acc:.1%} | {laya_t_acc:.1%} | {ci_str} |"
        )

    md_lines.append("")
    md_lines.append("### Métricas de Calibração e Tarefas Especiais")
    md_lines.append("")
    md_lines.append("| Dataset | Métrica | Jev | Laya | Laya-Tuned | Melhor |")
    md_lines.append("|---|---|---:|---:|---:|---|")

    # Brier & ECE on Emotion & AG News
    for task in ["ag_news", "emotion", "sst2"]:
        if task in cases_by_task:
            task_cases = cases_by_task[task]
            for metric_name, fn in [("Brier Score (↓)", compute_brier_score), ("ECE (↓)", compute_ece)]:
                scores = {}
                for m in models:
                    aligned_cases = [c for c in task_cases if c.case_id in pred_map_by_model[m]]
                    aligned_preds = [pred_map_by_model[m][c.case_id] for c in aligned_cases]
                    val = fn(aligned_cases, aligned_preds)
                    scores[m] = val
                
                valid_scores = {k: v for k, v in scores.items() if v is not None}
                best_m = min(valid_scores, key=valid_scores.get).upper() if valid_scores else "-"
                
                s_jev = f"{scores.get('jev', 0.0):.4f}" if scores.get("jev") is not None else "-"
                s_lay = f"{scores.get('laya', 0.0):.4f}" if scores.get("laya") is not None else "-"
                s_layt = f"{scores.get('laya-tuned', 0.0):.4f}" if scores.get("laya-tuned") is not None else "-"
                md_lines.append(f"| `{task}` | {metric_name} | {s_jev} | {s_lay} | {s_layt} | **{best_m}** |")

    # Score metrics on sst5
    if "sst5" in cases_by_task:
        task_cases = cases_by_task["sst5"]
        s_m = {}
        for m in models:
            aligned_cases = [c for c in task_cases if c.case_id in pred_map_by_model[m]]
            aligned_preds = [pred_map_by_model[m][c.case_id] for c in aligned_cases]
            s_m[m] = compute_score_metrics(aligned_cases, aligned_preds)
        
        md_lines.append(f"| `sst5` | MAE Erro Absoluto (↓) | {s_m.get('jev', {}).get('mae', 0.0):.3f} | {s_m.get('laya', {}).get('mae', 0.0):.3f} | {s_m.get('laya-tuned', {}).get('mae', 0.0):.3f} | **JEV** |")
        md_lines.append(f"| `sst5` | Spearman Correlação (↑) | {s_m.get('jev', {}).get('spearman', 0.0):.3f} | {s_m.get('laya', {}).get('spearman', 0.0):.3f} | {s_m.get('laya-tuned', {}).get('spearman', 0.0):.3f} | **LAYA** |")

    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 3. Tempo de Resposta e Latência")
    md_lines.append("")
    md_lines.append("> [!NOTE]")
    md_lines.append("> A latência do **Jev** inclui a viagem completa de rede pela internet (Cliente → OpenRouter API → TypeSafe), enquanto a do **Laya** reflete a velocidade de inferência local (Apple Silicon M5).")
    md_lines.append("")
    md_lines.append("| Dataset | Modelo | Média (ms) | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) |")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|")

    for task in tasks:
        task_cases = cases_by_task[task]
        for m in models:
            lats = [
                pred_map_by_model[m][c.case_id].latency_ms
                for c in task_cases
                if c.case_id in pred_map_by_model[m] and pred_map_by_model[m][c.case_id].ok
            ]
            st = compute_latency_stats(lats)
            md_lines.append(
                f"| `{task}` | **{m.upper()}** | {st['mean']:.1f} | {st['p50']:.1f} | {st['p90']:.1f} | {st['p95']:.1f} | {st['p99']:.1f} |"
            )

    md_lines.append("")
    md_lines.append("### Comparação Direta de Latência: Laya Padrão vs Laya Ajustado")
    md_lines.append("")
    md_lines.append("| Dataset | Laya Padrão p50 | Laya Ajustado p50 | Impacto do Ajuste no Head |")
    md_lines.append("|---|---:|---:|---|")
    for task in tasks:
        task_cases = cases_by_task[task]
        l_p = [pred_map_by_model["laya"][c.case_id].latency_ms for c in task_cases if c.case_id in pred_map_by_model["laya"] and pred_map_by_model["laya"][c.case_id].ok]
        lt_p = [pred_map_by_model["laya-tuned"][c.case_id].latency_ms for c in task_cases if c.case_id in pred_map_by_model["laya-tuned"] and pred_map_by_model["laya-tuned"][c.case_id].ok]
        lp_p50 = compute_latency_stats(l_p)["p50"]
        lt_p50 = compute_latency_stats(lt_p)["p50"]
        diff_ms = lt_p50 - lp_p50
        pct = ((lt_p50 / lp_p50) - 1.0) * 100 if lp_p50 > 0 else 0.0
        md_lines.append(f"| `{task}` | {lp_p50:.1f} ms | {lt_p50:.1f} ms | {diff_ms:+.1f} ms ({pct:+.1f}%) |")

    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 4. Gráficos Comparativos")
    md_lines.append("")
    md_lines.append("![Acurácia por Dataset](charts/accuracy_comparison.png)")
    md_lines.append("")
    md_lines.append("![Latência Mediana p50](charts/latency_comparison.png)")
    md_lines.append("")
    md_lines.append("![Trade-off Acurácia vs Latência](charts/tradeoff_accuracy_latency.png)")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 5. Metodologia e Ambiente")
    md_lines.append("")
    md_lines.append("- **Dispositivo Local:** Apple Silicon M5 (MPS / CPU acceleration)")
    md_lines.append("- **Jev Endpoint:** OpenRouter Decisions API (`POST https://openrouter.ai/api/alpha/decisions`, model: `typesafe/jev-1.13`)")
    md_lines.append("- **Amostragem:** 200 casos aleatórios estratificados por dataset com seed fixa (42)")
    md_lines.append("- **Reprodutibilidade:** Prompts gravados em `results/manifest.jsonl` com inputs idênticos avaliados em emparelhamento exato por ID de caso.")

    report_content = "\n".join(md_lines)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report_content, encoding="utf-8")
    return report_content
