# Benchmark Comparativo: Jev vs Laya (Padrão e Ajustado)

Este repositório contém uma suíte automatizada e reprodutível de benchmark para comparar modelos de decisão determinísticos (**System 1**):
- **Jev** (`typesafe/jev-1.13` via Decisions API)
- **Laya Padrão** (Convai Innovations, inferência local no Apple Silicon M5 via MPS)
- **Laya Ajustado** (`head_max_len=512` para expansão de vocabulário e alta cardinalidade)

Avaliação sob **mesmos prompts** e gabaritos em 5 datasets públicos com 200 amostras cada (1.000 prompts por modelo, 3.000 previsões no total).

---

## Estrutura do Projeto

```
JevTest/
├── .env                      # Chaves de API (openrouter_api_key)
├── pyproject.toml            # Dependências (laya, httpx, datasets, etc.)
├── README.md                 # Guia de uso
├── bench/
│   ├── config.py             # Configurações globais e paths
│   ├── tasks.py              # Definição e download dos 5 datasets
│   ├── clients/
│   │   ├── base.py           # Interface comum e dataclass Prediction
│   │   ├── jev.py            # Cliente OpenRouter Decisions API
│   │   └── laya_local.py     # Cliente local Laya (MPS / CPU)
│   ├── metrics.py            # Accuracy, Macro-F1, Brier, ECE, MAE, Bootstrap CI
│   ├── report.py             # Geração de relatório Markdown e gráficos PNG
│   └── cli.py                # CLI do benchmark (prepare, smoke, run, report)
├── tests/
│   └── test_metrics.py       # Testes unitários das métricas estatísticas
└── results/
    ├── manifest.jsonl        # 1.000 casos congelados (mesma seed)
    ├── raw/                  # JSONL brutos de saída (jev, laya, laya-tuned)
    ├── charts/               # Gráficos PNG gerados
    └── report.md             # Relatório final consolidado
```

---

## Como Reproduzir

### 1. Instalar dependências
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Preparar e congelar os datasets (1.000 casos)
```bash
python -m bench.cli prepare
```

### 3. Rodar teste de fumaça (Smoke Test)
```bash
python -m bench.cli smoke
```

### 4. Executar os modelos
```bash
# Executar Laya Padrão localmente
python -m bench.cli run --model laya

# Executar Laya Ajustado localmente
python -m bench.cli run --model laya-tuned

# Executar Jev via API remota (com concorrência)
python -m bench.cli run --model jev --concurrency 5
```

### 5. Gerar Relatório e Gráficos
```bash
python -m bench.cli report
```
O relatório consolidado estará em `results/report.md` e os gráficos em `results/charts/`.
