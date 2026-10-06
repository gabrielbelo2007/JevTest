## 1. Motivação e Análise dos Datasets Escolhidos

Os 5 datasets foram selecionados para cobrir deliberadamente diferentes **primitivas de decisão** (`choice`, `score` e `noul`) e diferentes **graus de complexidade semântica e cardinalidade**, totalizando **1.000 casos** (200 por dataset, amostra aleatória simples congelada em `results/manifest.jsonl` com `seed=42`).

---

### A. AG News (`fancyzhx/ag_news`)
* **O que é:** Corpus clássico de notícias em inglês divididas em 4 temas gerais: *World*, *Sports*, *Business* e *Sci/Tech*.
* **O que busca demonstrar:** A capacidade básica do modelo em **classificação temática zero-shot** com poucas opções (4 classes disjuntas e semanticamente bem separadas).
* **Resultados esperados:** Alta acurácia em ambos os modelos, já que ambos divulgam números acima de 90% para essa tarefa na literatura.
* **Resultados obtidos:**
  * **Laya (Padrão e Tuned):** **94.5%** de acurácia (189/200) | Brier Score: **0.1030** | ECE: **0.0397**
  * **Jev:** **87.0%** de acurácia (174/200) | Brier Score: **0.2088** | ECE: **0.0983**
* **Conclusão:** O Laya superou o Jev por **+7.5%** [IC95%: -12.0%, -3.0%], demonstrando excelente separação semântica e calibração probabilística muito superior (probabilidades mais honestas e confiáveis).

---

### B. Emotion (`dair-ai/emotion`)
* **O que é:** Classificação de sentimentos/emoções sutis em 6 categorias: *sadness, joy, love, anger, fear* e *surprise*.
* **O que busca demonstrar:** A sensibilidade do modelo a nuances subjetivas e estados psicológicos, onde as fronteiras entre classes são menos óbvias do que em notícias (ex.: distinguir *love* de *joy*, ou *fear* de *sadness*).
* **Resultados esperados:** Tarefa com sobreposições subjetivas, testando limites de desambiguação afetiva.
* **Resultados obtidos:**
  * **Laya (Padrão e Tuned):** **58.5%** de acurácia (117/200) | Brier Score: **0.6908** | ECE: **0.2744**
  * **Jev:** **58.0%** de acurácia (116/200) | Brier Score: **0.7095** | ECE: **0.2883**
* **Conclusão:** Empate técnico em acurácia (diferença de apenas 0.5%, intervalo de confiança [-8.0%, +6.0%] cruza zero), com ligeira vantagem do Laya na calibração probabilística.

---

### C. Banking77 (`mteb/banking77`)
* **O que é:** 77 intenções bancárias muito específicas de atendimento ao cliente (ex.: `card_payment_fee_charged`, `card_delivery_estimate`, `atm_support`).
* **O que busca demonstrar:** **Alta cardinalidade** e saturação do espaço de opções em um único prompt sob 77 critérios simultâneos.
* **Resultados esperados:** O Laya base possui um orçamento de atenção padrão no cabeçote (`head_max_len=192 tokens`). Ao apresentar 77 critérios, o orçamento por opção fica comprimido, testando se a expansão do cabeçote (`head_max_len=512`, `max_len=1024`) recupera discriminabilidade.
* **Resultados obtidos:**
  * **Jev:** **80.0%** de acurácia (160/200) | Brier Score: **0.3128** | ECE: **0.0963**
  * **Laya Tuned (head=512):** **58.5%** de acurácia (117/200) | Brier Score: **0.7125** | ECE: **0.3266**
  * **Laya Padrão (head=192):** **47.0%** de acurácia (94/200) | Brier Score: **0.9705** | ECE: **0.4590**
* **Conclusão:** **Vitória clara do Jev (80.0%)**, que demonstra alta robustez em discriminação de alta cardinalidade. No entanto, o ajuste de parâmetros do Laya foi decisivo: ao expandir o cabeçote para 512 tokens, o orçamento médio por opção subiu de ~4 para ~6 tokens (elevando os rótulos distintos de 67 para 72), alterando 84 predições, recuperando **35 casos de erro para acerto** (contra 12 regressões) e produzindo um ganho líquido de **+11.5%** [IC95%: +5.0%, +17.5%].

---

### D. SST-5 (`SetFit/sst5`)
* **O que é:** Escala ordinal de 5 pontos de sentimento em resenhas de filmes: *Muito Negativo (0), Negativo (1), Neutro (2), Positivo (3), Muito Positivo (4)*.
* **O que busca demonstrar:** A primitiva **`score`** (avaliação quantitativa ordinal com rubrica de gradação contínua).
* **Resultados esperados:** A documentação oficial do Laya aponta que pontuações ordinais são sua primitiva mais desafiadora (*"Ordinal score questions are the weakest primitive (SST-5 0.372)"*).
* **Resultados obtidos:**
  * **Jev:** **61.0%** de acurácia exata (122/200) | MAE: **0.451** | Spearman: **0.891**
  * **Laya (Padrão e Tuned):** **34.0%** de acurácia exata (68/200) | MAE: **0.861** | Spearman: **0.822**
* **Conclusão:** **Vitória expressiva do Jev**. O Jev alcançou 61.0% de acerto exato com erro absoluto médio de apenas 0.45 pontos (errando por menos de meio nível na escala), contra 34.0% e MAE de 0.861 do Laya (reproduzindo o comportamento documentado para a primitiva `score` na biblioteca oficial).

---

### E. SST-2 (`stanfordnlp/sst2`)
* **O que é:** Classificação binária de sentimento extraída do split de validação do SST-2: Positivo vs. Negativo.
* **O que busca demonstrar:** A primitiva **`noul`** (gatekeeping booleano — decisões de "Verdadeiro / Falso").
* **Resultados esperados:** Alta precisão em controle de fluxo determinístico (if/else).
* **Resultados obtidos:**
  * **Jev:** **99.0%** de acurácia (198/200 acertos) | Brier: **0.0276** | ECE: **0.0912**
  * **Laya (Padrão e Tuned):** **51.0%** de acurácia (102/200 acertos) | Brier: **0.4900** | ECE: **0.4900**
* **Conclusão:** **Vitória dominante do Jev**. O Laya previu `False` em 200 de 200 casos (a acurácia de 51% reflete estritamente a proporção de casos negativos na amostra). Relatos da comunidade apontam um desbalanceamento no par de tokens booleanos do checkpoint inglês base (issue #156 na documentação do projeto; não auditada internamente nos pesos). O Jev demonstrou estabilidade de 99.0% na primitiva `noul`.

---

## 2. Por que o Laya se destacou onde venceu?

O fato de um modelo open source rodando localmente superar o Jev em `ag_news` (94.5% vs 87.0%) e alcançar latência p50 de ~41 ms decorre de três fatores arquiteturais:

### 1. Backbone ModernBERT-large (395M)
* O Laya utiliza o **ModernBERT-large**, pré-treinado em mais de 2 trilhões de tokens com atenção bidirecional profunda, RoPE (*Rotary Position Embeddings*) e descompactação sem preenchimento (*unpadding*).
* Em tarefas onde o número de classes é pequeno e o contexto textual é rico (notícias, sentimentos gerais), o encoder bidirecional extrai representações semânticas densas e precisas.

### 2. Treinamento Calibrado via RLCD
* O treinamento do Laya incorpora RLCD (*Reinforcement Learning for Calibrated Decisions*) com scoring estritamente próprio.
* Como a função de recompensa penaliza a superconfiança, as probabilidades geradas em tarefas bem condicionadas são altamente calibradas (ECE de 0.0397 no AG News).

### 3. Eliminação da Latência de Rede (Edge vs. Cloud)
* O Jev roda na nuvem via OpenRouter Decisions API: a requisição percorre a internet até o cluster TypeSafe e retorna, impondo uma latência física de rede de **~490 ms a 550 ms**.
* O Laya roda **100% local no Apple Silicon M5** via Metal Performance Shaders (`mps`), alcançando p50 de **41.2 ms** (modo padrão) e **66.7 ms** (modo tuned), garantindo soberania de dados, ausência de custos de egress e velocidade até 12x superior.

---

## 3. Resumo Prático: Quando Utilizar Cada Modelo

| Cenário de Aplicação | Modelo Recomendado | Justificativa Empírica |
|---|:---:|---|
| **Classificação de Tópicos / Triagem (≤ 6 classes)** | **Laya** | Maior acurácia no AG News (94.5% vs 87.0%) e calibração superior (ECE 0.0397). |
| **Portões Booleanos / Guardrails (`noul`)** | **Jev** | O Jev alcançou 99.0% de acurácia, enquanto o checkpoint Laya base colapsa para `False`. |
| **Escalas Ordinais e Notas Contínuas (`score`)** | **Jev** | Jev obteve 61.0% de acerto exato e MAE 0.451 (vs 34.0% e MAE 0.861 do Laya). |
| **Catálogo Massivo de Opções (> 20 classes)** | **Jev ou Laya Tuned** | Jev lidera com 80.0%; se exigido processamento local, Laya Tuned (`head_max_len=512`) alcança 58.5%. |
| **Aplicações de Baixa Latência / On-Premise / Privacidade** | **Laya** | Resposta em 41–66 ms diretamente na máquina do usuário, com zero transferência externa de dados. |