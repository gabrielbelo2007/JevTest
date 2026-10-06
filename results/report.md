# Relatório Comparativo: Jev vs Laya (Padrão e Ajustado)

> Benchmark automatizado avaliando modelos de decisão determinísticos (System 1) sob **mesmos prompts**, medindo qualidade de decisão (scores) e tempo de resposta (latência).

## 1. Resumo Executivo

| Modelo | Modo | Acurácia Geral | Macro-F1 Médio | Latência p50 | Latência p95 | Taxa Sucesso |
|---|---|---:|---:|---:|---:|---:|
| **JEV** | Nuvem (OpenRouter API) | **76.9%** | 0.744 | 501.5 ms | 868.9 ms | 99.8% |
| **LAYA-TUNED** | Local (Apple Silicon M5 MPS) | **57.0%** | 0.384 | 40.5 ms | 100.8 ms | 100.0% |
| **LAYA** | Local (Apple Silicon M5 MPS) | **57.0%** | 0.384 | 41.2 ms | 101.8 ms | 100.0% |

---

## 2. Acurácia e Qualidade por Tarefa

| Dataset | Tipo | Qtd Rótulos | Jev Acc | Laya Acc | Laya-Tuned Acc | Diferença (Jev - Laya [95% CI]) |
|---|---|---:|---:|---:|---:|---:|
| `ag_news` | `choice` | 4 | 87.0% | 94.5% | 94.5% | -7.5% [-12.0%, -3.0%] |
| `banking77` | `choice` | 77 | 80.0% | 47.0% | 47.0% | +33.0% [+25.5%, +39.5%] |
| `emotion` | `choice` | 6 | 58.0% | 58.5% | 58.5% | -0.5% [-8.0%, +6.0%] |
| `sst5` | `score` | 5 | 60.5% | 34.0% | 34.0% | +26.5% [+18.0%, +36.0%] |
| `sst2` | `noul` | 2 | 99.0% | 51.0% | 51.0% | +48.0% [+41.0%, +54.5%] |

### Métricas de Calibração e Tarefas Especiais

| Dataset | Métrica | Jev | Laya | Laya-Tuned | Melhor |
|---|---|---:|---:|---:|---|
| `ag_news` | Brier Score (↓) | 0.2088 | 0.1030 | 0.1030 | **LAYA-TUNED** |
| `ag_news` | ECE (↓) | 0.0983 | 0.0397 | 0.0397 | **LAYA-TUNED** |
| `emotion` | Brier Score (↓) | 0.7095 | 0.6908 | 0.6908 | **LAYA-TUNED** |
| `emotion` | ECE (↓) | 0.2883 | 0.2744 | 0.2744 | **LAYA-TUNED** |
| `sst2` | Brier Score (↓) | 0.0276 | 0.4900 | 0.4900 | **JEV** |
| `sst2` | ECE (↓) | 0.0912 | 0.4900 | 0.4900 | **JEV** |
| `sst5` | MAE Erro Absoluto (↓) | 0.450 | 0.861 | 0.861 | **JEV** |
| `sst5` | Spearman Correlação (↑) | 0.891 | 0.822 | 0.822 | **LAYA** |

---

## 3. Tempo de Resposta e Latência

> [!NOTE]
> A latência do **Jev** inclui a viagem completa de rede pela internet (Cliente → OpenRouter API → TypeSafe), enquanto a do **Laya** reflete a velocidade de inferência local (Apple Silicon M5).

| Dataset | Modelo | Média (ms) | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) |
|---|---|---:|---:|---:|---:|---:|
| `ag_news` | **JEV** | 560.4 | 511.1 | 809.1 | 834.3 | 945.4 |
| `ag_news` | **LAYA-TUNED** | 44.5 | 42.5 | 50.3 | 53.9 | 65.5 |
| `ag_news` | **LAYA** | 48.1 | 44.9 | 53.8 | 60.0 | 113.7 |
| `banking77` | **JEV** | 619.4 | 504.5 | 880.4 | 944.6 | 1502.1 |
| `banking77` | **LAYA-TUNED** | 100.2 | 99.0 | 104.1 | 107.0 | 125.6 |
| `banking77` | **LAYA** | 100.2 | 99.2 | 104.9 | 107.7 | 111.1 |
| `emotion` | **JEV** | 549.1 | 490.0 | 817.5 | 844.0 | 974.5 |
| `emotion` | **LAYA-TUNED** | 42.7 | 41.6 | 48.0 | 48.9 | 61.4 |
| `emotion` | **LAYA** | 42.6 | 41.7 | 47.6 | 49.2 | 51.3 |
| `sst5` | **JEV** | 569.0 | 501.2 | 822.3 | 839.0 | 928.3 |
| `sst5` | **LAYA-TUNED** | 34.4 | 34.0 | 39.0 | 39.7 | 42.1 |
| `sst5` | **LAYA** | 34.7 | 34.5 | 39.5 | 40.9 | 44.2 |
| `sst2` | **JEV** | 595.8 | 505.3 | 832.0 | 846.6 | 922.7 |
| `sst2` | **LAYA-TUNED** | 29.6 | 28.5 | 34.5 | 35.3 | 37.1 |
| `sst2` | **LAYA** | 28.9 | 27.7 | 32.7 | 34.6 | 35.1 |

### Comparação Direta de Latência: Laya Padrão vs Laya Ajustado

| Dataset | Laya Padrão p50 | Laya Ajustado p50 | Impacto do Ajuste no Head |
|---|---:|---:|---|
| `ag_news` | 44.9 ms | 42.5 ms | -2.4 ms (-5.3%) |
| `banking77` | 99.2 ms | 99.0 ms | -0.2 ms (-0.2%) |
| `emotion` | 41.7 ms | 41.6 ms | -0.1 ms (-0.1%) |
| `sst5` | 34.5 ms | 34.0 ms | -0.4 ms (-1.2%) |
| `sst2` | 27.7 ms | 28.5 ms | +0.8 ms (+3.0%) |

---

## 4. Gráficos Comparativos

![Acurácia por Dataset](charts/accuracy_comparison.png)

![Latência Mediana p50](charts/latency_comparison.png)

![Trade-off Acurácia vs Latência](charts/tradeoff_accuracy_latency.png)

---

## 5. Metodologia e Ambiente

- **Dispositivo Local:** Apple Silicon M5 (MPS / CPU acceleration)
- **Jev Endpoint:** OpenRouter Decisions API (`POST https://openrouter.ai/api/alpha/decisions`, model: `typesafe/jev-1.13`)
- **Amostragem:** 200 casos aleatórios estratificados por dataset com seed fixa (42)
- **Reprodutibilidade:** Prompts gravados em `results/manifest.jsonl` com inputs idênticos avaliados em emparelhamento exato por ID de caso.