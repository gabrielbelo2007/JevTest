# Benchmark Comparativo: Jev vs OpenAI Decisions vs CLM-8B vs Laya (Padrão e Ajustado)

> Suíte automatizada, reprodutível e emparelhada para avaliação de modelos determinísticos de decisão (**System 1**):
> - **Jev** (`typesafe/jev-1.13` via OpenRouter Decisions API)
> - **OpenAI Decisions** (`openai/gpt-6-luna-decisions` via OpenRouter Decisions API)
> - **CLM-8B** (`Contrastive-LM/CLM-v0.1-8B`, Stanford & NVIDIA System 1 via servidor local/vLLM)
> - **Laya Ajustado** (`convaiinnovations/laya`, inferência local com `head_max_len=512` e `max_len=1024`)
> - **Laya Padrão** (`convaiinnovations/laya`, inferência local sequencial, `head_max_len=192`)

Avaliação sob **mesmos prompts**, mesmos dados de entrada (`state`) e mesmas perguntas (`questions`) em 5 datasets públicos com 200 amostras cada (1.000 prompts idênticos por modelo, **5.000 predições auditadas** no total).

---

## 📊 Resultados em Destaque

| Modelo | Modo de Execução | Acurácia Geral | Macro-F1 | Latência p50 | Latência p95 | Taxa Sucesso |
|---|---|---:|---:|---:|---:|---:|
| **JEV** | Nuvem (OpenRouter Decisions API) | **77.0%** | 0.744 | 501.6 ms | 868.9 ms | 100.0% |
| **CLM-8B** | Local / vLLM (NVIDIA & Stanford) | **76.6%** | 0.742 | 49.2 ms | 86.1 ms | 100.0% |
| **OPENAI** | Nuvem (OpenRouter Decisions API - GPT-6 Luna) | **74.1%** | 0.740 | 539.8 ms | 646.2 ms | 99.9%* |
| **LAYA-TUNED** | Local (`head_max_len=512`, `max_len=1024`) | **59.3%** | 0.488 | 66.7 ms | 242.8 ms | 100.0% |
| **LAYA PADRÃO** | Local (`head_max_len=192`, fábrica) | **57.0%** | 0.384 | 41.2 ms | 101.8 ms | 100.0% |

*\*1 caso sofreu recusa de segurança (safety guardrail) no Banking77 (`banking77-0123`), contabilizado como erro.*

> 📖 **Relatório Completo e Análise Arquitetural:**  
> A dissecação técnica aprofundada, as métricas de calibração (Brier/ECE), o comportamento no Banking77 (Jev 80%, OpenAI 79%, CLM 78%), o domínio do Laya no AG News (94.5%) e os gráficos comparativos estão documentados em detalhe no **[`results/report.md`](results/report.md)** e em **[`results/datasets.md`](results/datasets.md)**.

---

## 🌐 Painel Web Interativo (Dashboard)

O projeto inclui um website estático completo (`site/index.html`) para inspecionar métricas, filtrar e auditar visualmente os 1.000 casos e testar o replay de decisões entre os 5 modelos:

```bash
# Abrir diretamente no navegador:
open site/index.html

# Ou servir via servidor HTTP local:
python3 -m http.server 8080 --directory site
# Acesse: http://localhost:8080
```

---

## 🚀 Guia de Reprodução e Execução

### 1. Pré-requisitos
- **Python 3.11** ou **3.12**
- Aceleração por hardware opcional: **Apple Silicon MPS** (macOS) ou **NVIDIA CUDA** (Linux); fallback automático para **CPU**.
- Espaço em disco: ~2 GB para os pesos do modelo `convaiinnovations/laya` (Hugging Face) e ~16 GB caso queira subir pesos locais do `Contrastive-LM/CLM-v0.1-8B`.

### 2. Instalação do Ambiente
```bash
# 1. Clone o repositório e acesse a pasta
git clone <url-do-repositorio> JevTest
cd JevTest

# 2. Crie e ative o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instale o pacote em modo editável com as dependências de desenvolvimento
pip install -e ".[dev]"
```

### 3. Configuração de Credenciais
Copie o modelo de variáveis de ambiente:
```bash
cp .env.example .env
```
Edite o arquivo `.env`:
```env
OPENROUTER_API_KEY=sua-chave-aqui
# Opcional para modelo OpenAI customizado (padrão: openai/gpt-6-luna-decisions):
# OPENAI_DECISION_MODEL=openai/gpt-6-luna-decisions
# Opcional para CLM-8B se você rodar um servidor customizado:
# CLM_ENDPOINT=http://127.0.0.1:8000/v1/systemone
```
> **Nota:** Os modelos de nuvem (**Jev** e **OpenAI Decisions**) utilizam a mesma `OPENROUTER_API_KEY`. O **Laya Padrão** e o **Laya Tuned** rodam 100% locais na sua máquina e **não necessitam de nenhuma chave de API**. O **CLM-8B** conecta-se a um endpoint local compatível (`clm-serve`/vLLM).

---

## ⚡ Como Executar os Testes

### Modo A: Regeneração Instantânea dos Relatórios e Dashboard (Recomendado)
Todas as **5.000 predições brutas já estão versionadas e congeladas** no repositório em `results/raw/`. Você pode auditar e regerar todos os artefatos imediatamente, sem custos de API e sem baixar modelos:

```bash
# 1. Regenera results/report.md e todos os gráficos PNG em results/charts/
python -m bench.cli report

# 2. Regenera os dados e a página do dashboard interativo (site/data.js e site/index.html)
python bench/generate_site.py

# 3. Abre o painel interativo no navegador
open site/index.html
```

### Modo B: Testes Automatizados de Software (Pytest)
Para validar a integridade do pipeline de processamento, cálculo de métricas e retentativa de falhas:
```bash
pytest
```
*Executa 16 testes unitários e de integração cobrindo emparelhamento de dados, mocks de transporte HTTP para Jev, OpenAI Decisions e CLM, e bootstrap estatístico.*

### Modo C: Smoke Test Rápido
Para verificar se o ambiente local e as integrações com os modelos estão operacionais em segundos:
```bash
python -m bench.cli smoke
```
*Avalia 5 casos representativos (1 de cada tarefa) nos 5 modelos simultaneamente.*

---

## 🔬 Execução Completa das Inferências (Do Zero)

Caso deseje reprocessar ou reproduzir todas as inferências dos modelos a partir do início:

```bash
# Passo 1: Congelar a amostragem dos 1.000 casos (já presente em results/manifest.jsonl)
python -m bench.cli prepare

# Passo 2: Executar inferência local do Laya Padrão (~2 minutos em Apple Silicon M5)
python -m bench.cli run --model laya

# Passo 3: Executar inferência local do Laya Tuned (~3 minutos em Apple Silicon M5)
python -m bench.cli run --model laya-tuned

# Passo 4: Executar inferência do CLM-8B via servidor System 1 local
python -m bench.cli run --model clm

# Passo 5: Executar inferência do Jev via OpenRouter API (~2 minutos com 5 workers)
python -m bench.cli run --model jev --concurrency 5

# Passo 6: Executar inferência do OpenAI Decisions via OpenRouter API (~2 minutos com 5 workers)
python -m bench.cli run --model openai --concurrency 5

# Passo 7: Gerar o relatório consolidado e o dashboard interativo
python -m bench.cli report
python bench/generate_site.py
open site/index.html
```

### Tolerância a Falhas e Retomada Idempotente
O comando `python -m bench.cli run` foi projetado para tolerar falhas de conexão ou interrupções:
- Predições já concluídas com sucesso (`ok=True`) são puladas automaticamente.
- Se ocorrer qualquer timeout temporário (`ok=False`), basta reexecutar o mesmo comando `run`: ele **reexecutará apenas os casos que falharam** e salvará o arquivo consolidado sem duplicatas.

---

## 🛠️ Referência dos Comandos CLI

A CLI unificada pode ser acessada via módulo Python (`python -m bench.cli`) ou pelo atalho de console `jl-bench`:

| Comando | Descrição | Exemplo |
|---|---|---|
| `prepare` | Baixa os 5 datasets do Hugging Face e congela 200 casos de cada com seed fixa. | `python -m bench.cli prepare` |
| `smoke` | Roda um teste rápido de sanidade com 5 casos nos 5 modelos. | `python -m bench.cli smoke` |
| `run` | Executa a inferência completa de 1.000 casos para um modelo (`jev`, `openai`, `laya`, `laya-tuned`, `clm` ou `all`). | `python -m bench.cli run --model openai` |
| `report` | Calcula métricas, gera o `results/report.md` e produz os gráficos comparativos. | `python -m bench.cli report` |

---

## 📁 Estrutura do Repositório

```
JevTest/
├── .env.example              # Modelo de configuração de variáveis de ambiente
├── pyproject.toml            # Especificação de dependências do projeto e CLI
├── README.md                 # Guia de reprodução e comandos de execução
├── site/                     # Painel Web Interativo
│   ├── index.html            # Aplicação web estática (Tailwind CSS + SVG charts)
│   └── data.js               # Bundle com os 1.000 casos e métricas estruturadas
├── bench/                    # Código-fonte do benchmark
│   ├── cli.py                # Interface de linha de comando (prepare, smoke, run, report)
│   ├── config.py             # Configurações globais, diretórios e constantes
│   ├── metrics.py            # Accuracy, Macro-F1, Brier, ECE, MAE, Spearman, Bootstrap CI
│   ├── report.py             # Geração de report.md e gráficos PNG
│   ├── generate_site.py      # Gerador orientado a dados do site interativo
│   ├── site_template.html    # Template modular do painel web
│   ├── tasks.py              # Definição e amostragem dos 5 datasets
│   └── clients/              # Clientes de inferência tipados
│       ├── base.py           # Interface comum DecisionClient e dataclass Prediction
│       ├── jev.py            # Cliente OpenRouter Decisions API para Jev
│       ├── openai_decisions.py # Cliente OpenRouter Decisions API para OpenAI (GPT-6 Luna)
│       ├── clm.py            # Cliente CLM-8B NVIDIA/Stanford (System 1 contrastivo)
│       └── laya_local.py     # Cliente local Laya (com suporte a head_max_len=512)
├── tests/                    # Suíte de testes automatizados
│   ├── test_metrics.py       # Testes unitários das métricas e bootstrap estatístico
│   └── test_pipeline.py      # Testes de integração, idempotência, mocks e relatórios
└── results/                  # Dados e relatórios consolidados
    ├── manifest.jsonl        # 1.000 casos congelados com seed=42 (mesmos prompts)
    ├── report.md             # Relatório técnico completo e análise arquitetural
    ├── datasets.md           # Análise motivacional e resultados por dataset
    ├── raw/                  # Predições brutas versionadas (5.000 previsões auditadas)
    │   ├── jev.jsonl         # 1.000 predições do Jev (100% de sucesso)
    │   ├── openai.jsonl      # 1.000 predições do OpenAI Decisions (99.9% de sucesso, 1 recusa)
    │   ├── laya.jsonl        # 1.000 predições do Laya Padrão (100% de sucesso)
    │   ├── laya-tuned.jsonl  # 1.000 predições do Laya Tuned (100% de sucesso)
    │   └── clm.jsonl         # 1.000 predições do CLM-8B (100% de sucesso)
    └── charts/               # Gráficos de alta resolução gerados pelo report
        ├── accuracy_comparison.png
        ├── latency_comparison.png
        └── tradeoff_accuracy_latency.png
```

---

## ⚖️ Ressalvas Metodológicas

1. **Amostragem Estatística:** A amostra é de N = 200 por dataset (1.000 casos no total). Com N = 200, a margem de erro (IC 95%) para acurácia é de aproximadamente ±7 p.p. Diferenças inferiores a esse intervalo (como no dataset Emotion, onde houve diferença de 0.5% a 2.0% entre os modelos) configuram empate técnico.
2. **Comparabilidade de Latência:** As latências de nuvem (Jev e OpenAI Decisions) incluem o trânsito completo pela internet (cliente ↔ OpenRouter ↔ Provedor), medidas com requisições concorrentes. A latência do Laya e do CLM-8B reflete inferência local (ou via socket local de GPU). No CLM-8B, o **Action Caching** mantém os embeddings dos critérios pré-alocados em memória de vídeo (~49 ms p50).
3. **Guardrails e Recusas em APIs de Decisão:** Durante o benchmark do modelo OpenAI (`openai/gpt-6-luna-decisions`), o caso `banking77-0123` foi recusado pelo sistema de moderação e guardrails da OpenAI (HTTP 502 Refusal), resultando em taxa de sucesso de 99.9%. Por rigor metodológico, recusas formais em tarefas de benchmark determinístico são computadas como erro (`ok=False`).

