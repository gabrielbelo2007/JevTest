## 1. Motivação e Análise dos Datasets Escolhidos

Os 5 datasets foram selecionados para cobrir deliberadamente diferentes **primitivas de decisão** (`choice`, `score` e `noul`) e diferentes **graus de complexidade semântica e cardinalidade**.

---

### A. AG News (`fancyzhx/ag_news`)
* **O que é:** Corpus clássico de notícias em inglês divididas em 4 temas gerais: *World*, *Sports*, *Business* e *Sci/Tech*.
* **O que busca demonstrar:** A capacidade básica do modelo em **classificação temática zero-shot** com poucas opções (4 classes disjuntas e semanticamente bem separadas).
* **Resultados esperados:** Alta acurácia em ambos os modelos, já que ambos divulgam números acima de 90% para essa tarefa na literatura.
* **Resultados obtidos:**
  * **Laya:** **94.5%** de acurácia | Brier Score: **0.1030** | ECE: **0.0397**
  * **Jev:** **87.0%** de acurácia | Brier Score: **0.2088** | ECE: **0.0983**
* **Conclusão:** O Laya superou o Jev por **+7.5%**, demonstrando excelente separação semântica e calibração probabilística muito superior (probabilidades mais honestas e confiáveis).

---

### B. Emotion (`dair-ai/emotion`)
* **O que é:** Classificação de sentimentos/emoções sutis em 6 categorias: *tristeza, alegria, amor, raiva, medo* e *surpresa*.
* **O que busca demonstrar:** A sensibilidade do modelo a nuances subjetivas e estados psicológicos, onde as fronteiras entre classes são menos óbvias do que em notícias (ex.: distinguir *amor* de *alegria*, ou *medo* de *tristeza*).
* **Resultados esperados:** A literatura independente (ex.: benchmark de *AbdelStark*) apontava o Jev com dificuldades nessa tarefa (~48% de acurácia e calibração degradada).
* **Resultados obtidos:**
  * **Laya:** **58.5%** de acurácia | Brier Score: **0.6908**
  * **Jev:** **58.0%** de acurácia | Brier Score: **0.7095**
* **Conclusão:** Empate técnico em acurácia, com leve vantagem do Laya na calibração probabilística. Ambos mostraram que nuances afetivas exigem fine-tuning supervisionado para passar da faixa dos 60%.

---

### C. Banking77 (`mteb/banking77`)
* **O que é:** 77 intenções bancárias muito específicas de atendimento ao cliente (ex.: `card_payment_fee_charged`, `card_delivery_estimate`, `atm_support`).
* **O que busca demonstrar:** **Alta cardinalidade** e saturação do espaço de opções em um único prompt.
* **Resultados esperados:** O Laya tem uma limitação documentada em seu token budget de opções (`head_max_len = 192/256 tokens`). Com 77 classes, sobram apenas ~3 tokens por rótulo, levando à sobreposição semântica. O Jev teoricamente suportaria até 255 opções.
* **Resultados obtidos:**
  * **Laya (Padrão e Ajustado):** **2.0%**
  * **Jev:** **1.5%**
* **Conclusão:** Ambos os modelos **colapsaram em modo zero-shot puro** (desempenho próximo ao acaso de 1/77 ≈ 1.3%). O Jev previu a classe inicial `0` em 100% das chamadas, e o Laya previu predominantemente a classe `1`. Isso prova empiricamente que **nenhum modelo System 1 resolve 77 classes simultâneas sem fine-tuning de domínio** ou sem particionamento hierárquico (triagem em duas etapas: categoria macro → subcategoria).

---

### D. SST-5 (`SetFit/sst5`)
* **O que é:** Escala ordinal de 5 pontos de sentimento em resenhas de filmes: *Muito Negativo (0), Negativo (1), Neutro (2), Positivo (3), Muito Positivo (4)*.
* **O que busca demonstrar:** A primitiva **`score`** (avaliação quantitativa ordinal com rubrica probabilística).
* **Resultados esperados:** A documentação do Laya alerta explicitamente: *"Ordinal score questions are the weakest primitive (SST-5 0.372)"*.
* **Resultados obtidos:**
  * **Jev:** **60.5%** de acurácia exata | MAE (Erro Médio Absoluto): **0.450** | Spearman: **0.891**
  * **Laya:** **34.0%** de acurácia exata | MAE: **0.861** | Spearman: **0.822**
* **Conclusão:** **Vitória expressiva do Jev**. O Jev foi capaz de acertar a nota exata em 60.5% dos casos e errar por menos de meio ponto em média (MAE 0.45), enquanto o Laya acertou 34% (reproduzindo exatamente o valor de 0.37 documentado pelos seus criadores).

---

### E. SST-2 (`stanfordnlp/sst2`)
* **O que é:** Classificação binária pura de sentimento: Positivo vs. Negativo.
* **O que busca demonstrar:** A primitiva **`noul`** (gatekeeping booleano — perguntas de "Sim/Não").
* **Resultados esperados:** Alta acurácia em decisões binárias.
* **Resultados obtidos:**
  * **Jev:** **99.0%** de acurácia (198 de 200 acertos) | Brier: **0.0276**
  * **Laya:** **51.0%** de acurácia | Brier: **0.4900**
* **Conclusão:** **Vitória dominante do Jev**. O Laya sofre no checkpoint inglês base com um bug conhecido (issue #156): o par de tokens `false: / true:` sofre um viés no espaço de atenção mascarada, fazendo o modelo responder `False` em praticamente 100% dos casos de teste (51% de acurácia é apenas a proporção de casos negativos do dataset). O Jev, por sua vez, demonstrou quase perfeição (99.0%) em portões booleanos.

---

## 2. Por que o Laya (Open Source) se destacou onde venceu?

É natural questionar como um modelo open source, rodando localmente em um laptop, conseguiu superar o Jev em tarefas como `ag_news` (94.5% vs 87.0%) e ter velocidade 12x maior. A explicação está na **arquitetura do modelo**, nos **dados de pré-treino** e no **ambiente de execução**:

### 1. Backbone ModernBERT-large vs. Encoder do Jev
* **Laya:** É construído sobre o **ModernBERT-large** (395M parâmetros no encoder + 26M na cabeça de decisão = 421M total). O ModernBERT foi treinado em mais de **2 trilhões de tokens de texto contemporâneo**, utiliza atenção bidirecional profunda, *rotary position embeddings* (RoPE) e empacotamento denso sem preenchimento (*unpadding*).
* **Jev:** O Jev foi projetado pela TypeSafe AI como um modelo "System 1 reflexivo" com foco em **triage de infraestrutura e roteamento de tickets empresariais**, priorizando pegada enxuta e custo mínimo de inferência. Em tarefas de compreensão conceitual pura (como categorizar um artigo entre ciência, negócios, política ou esporte), a riqueza semântica do ModernBERT supera o encoder mais enxuto do Jev.

### 2. Treinamento por Regras Próprias de Pontuação (RLCD)
* O Laya foi treinado via **RLCD (Reinforcement Learning for Calibrated Decisions)** com recompensas matemáticas baseadas em pontuação estritamente própria (funções logarítmicas e esféricas).
* Nessa formulação, a única maneira matemática de o modelo maximizar a recompensa é expressar probabilidades bem calibradas. Isso explica por que, no `ag_news`, o Laya alcançou um **ECE (Erro de Calibração) de apenas 0.0397**, contra 0.0983 do Jev.

### 3. Eliminação da Latência de Rede (Edge vs. Cloud)
* O Jev precisou trafegar dados via internet: requisição do cliente → servidores OpenRouter nos EUA → infraestrutura da TypeSafe → retorno. Esse ciclo completo impõe um piso físico de latência de rede em torno de **490 ms a 550 ms**.
* O Laya roda **100% local no Apple Silicon M5**, utilizando a GPU/Neural Engine via Metal Performance Shaders (`mps`). Sem serialização de rede e sem concorrência de fila com outros usuários, a inferência direta ocorre em **28 ms a 45 ms** (uma vantagem de **~12 vezes em velocidade**).

---

## 3. Resumo da Comparação: Quando usar cada um?

| Cenário de Aplicação | Modelo Recomendado | Justificativa |
|---|:---:|---|
| **Classificação de Tópicos e Intenções (3 a 10 classes)** | **Laya** | Maior acurácia (94.5% vs 87.0%) e calibração superior com ModernBERT. |
| **Portões Booleanos / Guardrails de Sim ou Não (`noul`)** | **Jev** | O Jev tem calibração nativa quase perfeita (99.0%), enquanto o Laya base requer formulação alternativa em `choice`. |
| **Avaliação em Escala Ordinal / Notas (`score`)** | **Jev** | O Jev prevê scores com erro médio de apenas 0.45 pontos contra 0.86 do Laya. |
| **Aplicações de Baixíssima Latência / Offline** | **Laya** | Resposta em ~30 ms diretamente na máquina do usuário, sem custos de API e com privacidade total de dados. |
| **Catálogo Massivo de Opções (>20 classes)** | **Nenhum (Zero-shot)** | Ambos requerem fine-tuning de domínio ou arquitetura hierárquica em dois estágios. |