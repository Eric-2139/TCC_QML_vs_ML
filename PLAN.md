# Plano — TCC: ML clássico x QML na detecção de fraude em cartão de crédito

## Contexto

- **O que é:** a parte experimental do TCC. Compara **SVM x QSVM** e **AdaBoost x Quantum AdaBoost** no dataset *Credit Card Fraud Detection* (ULB/Kaggle; 284.807 transações, 492 fraudes).
- **Objetivo duplo:** gerar os resultados do TCC e servir de estudo, com código no nível de um ML Engineer sênior e uma aula por fase.
- **Pasta:** `C:\Users\Erike\Documents\Faculdade\TCC\ML-x-QML\`. Hoje ela só tem `creditcard.csv.zip` e `metodologia.MD`.
- **Testes existentes:** nenhum, porque ainda não há código. Por isso o projeto nasce com `pytest`.
- **Git:** o repositório detectado é a pasta de usuário inteira (`C:\Users\Erike`). Vamos criar um repositório só nesta pasta, na Fase 0.

**Fontes que reaproveitamos:**
- *Trabalho Humberto*, notebook `credit_card_fraud_ml.ipynb`:
  - setup: células 3, 7 e 9
  - contrato dos dados: célula 16
  - remoção de duplicatas e engenharia de features: células 45 e 47
  - evidência do scaler: células 54 a 56
  - split com `assert`s: células 68 e 70
  - `evaluate_classifier`: célula 111
  - `run_grid_search` com cache: célula 134
  - curvas de diagnóstico: células 157 e 159
  - ficha do modelo: célula 167
- `metodologia.MD`: versão mais recente, que já traz o split 60/20/20.
- TCC V4 final.
- Artigos da `Base de Conhecimento`.

### Decisões tomadas com você (nesta conversa)

| Tema | Decisão |
|---|---|
| Ordem split x balanceamento | **Split estratificado 60/20/20 na base inteira → undersampling só no treino.** Validação e teste mantêm a proporção real (~0,17%). O TCC registra que os artigos de QML fazem o contrário (El Alami; Grossi) e que isso foi corrigido aqui. |
| Duplicatas | **Removidas** antes do split: 283.726 linhas e 473 fraudes, como no Trabalho Humberto. |
| Ordem split x exploração | **Na base inteira, só o contrato dos dados e a remoção de duplicatas; depois o split; toda a exploração só no treino.** De validação e teste só se contam linhas e rótulos. Decidido em 2026-10-07: a versão anterior fazia a EDA antes do split (CC7), o que é vazamento por decisão, a forma sutil que os guias do scikit-learn e do imbalanced-learn mandam evitar. |
| Mesmo pré-processamento ML e QML | Os 4 modelos treinam no **mesmo subconjunto balanceado**. Os clássicos rodam em duas configurações: (a) **entrada idêntica** à do quântico, *like-for-like*; (b) **referência com todas as features**. |
| Limiar | Escolhido **na validação, maximizando o F1**. O modelo de custo entra como análise complementar. |
| Robustez | **5 sementes de undersampling** + IC por bootstrap + teste pareado (McNemar). |
| Ruído | **Seção extra** com simulação ruidosa (Qiskit Aer) dos modelos quânticos finais. |
| Ambiente | **Local**: `.venv` via `uv` (Python 3.12), infraestrutura num pacote `src/` com `pytest` e experimentos no notebook. |

### O que os artigos mostraram e que muda o desenho

- **El Alami (2025/26):**
  - não tem QSVM nem baseline clássico, só VQC, SQNN e EQNN;
  - subamostra e roda o PCA **antes** do split;
  - aplica o PCA sem padronizar `Time`/`Amount`;
  - testa num conjunto 50/50 (a precisão de 1,00 não vale para 0,17%);
  - ZZ e Pauli dão resultados idênticos (no Qiskit são o mesmo circuito);
  - usa MinMax em [0,1] com 4 a 7 qubits.
- **"Standard → MinMax" em sequência direta é matematicamente igual a MinMax sozinho**, porque duas transformações afins por coluna se anulam. A ordem só faz sentido com o PCA no meio: **Standard → PCA → MinMax** (Ajibosin). É por isso que os artigos não explicam essa escolha; vira um achado do TCC.
- **Ajibosin afirma que o StandardScaler "trata outliers". Está errado:** ele é sensível a outliers. Não usar essa justificativa.
- **Wang (Quantum AdaBoost):**
  - classificador fraco variacional treinado com **cross-entropy ponderada** pelos pesos do boosting;
  - α = ½·ln((1−ε)/ε).
  - O SAMME binário do scikit-learn usa α = ln((1−ε)/ε). A diferença é um fator 2, que não muda o sinal da soma ponderada, então **as decisões são as mesmas**.
  - Consequência: dá para usar o **mesmo `AdaBoostClassifier`** nos dois lados, trocando só o classificador fraco. É simetria real entre os pares.
- **Codificação de amplitude:**
  - o kernel de fidelidade é |⟨x|x'⟩|² = cos²(ângulo entre x e x'), que **é calculável classicamente**;
  - a normalização L2 **joga fora a norma** do vetor, e a norma é sinal de fraude: no Trabalho Humberto, a distância ao centróide tinha mediana 16,4 nas fraudes contra 4,0 nas legítimas (medido aqui, no treino balanceado: norma de 4,88 contra 3,80, Seção 4.8).
  - Solução: completar com uma **constante** (e não com zero) até 2ⁿ; o ângulo com esse eixo preserva a informação da norma.
  - Klug mostra que a amplitude perde para o ângulo em regime ruidoso.
- **O range do MinMax age como a "largura de banda" do kernel quântico.** Range grande concentra o kernel (vira quase a identidade e o modelo decora). Por isso ele entra como hiperparâmetro, e não como constante.

---

## Decisões de arquitetura

1. **Tudo que aprende com os dados fica dentro de um `imblearn.pipeline.Pipeline`:** `RandomUnderSampler`, scaler, PCA/seleção, MinMax e modelo. A cada fold, cada etapa é reajustada só no treino, então não há vazamento por construção (padrão Strategy/Composite do sklearn).
2. **Modelos quânticos como estimadores sklearn** (`BaseEstimator` + `ClassifierMixin`). Assim `GridSearchCV`, `learning_curve`, `clone` e o `AdaBoostClassifier` funcionam sem código especial.
   - **QSVM:** `SVC(kernel=callable)` sobre um kernel de fidelidade por statevector. O Qiskit calcula um statevector por amostra e o kernel vira um produto matricial, O(N) simulações, então ele roda no teste inteiro (56 mil linhas).
   - **Quantum AdaBoost:** `AdaBoostClassifier(estimator=VariationalQuantumClassifier(...))`.
     - O `VariationalQuantumClassifier` implementa `fit(X, y, sample_weight)` com cross-entropy ponderada.
     - Circuito: feature map + ansatz `RealAmplitudes`; mede-se ⟨Z⟩.
     - **Pendência da Fase 4 (CC14):** os mapas Z/ZZ são codificação de **fase**; num modelo que mede o estado,
       fase e rotação em Y não são equivalentes. A Seção 5 traz um diagnóstico barato (fase × RY no
       classificador fraco de 5 qubits, sem boosting). RY só entra no grid da Fase 5 se a diferença aparecer e
       o custo permitir.
     - Otimizador COBYLA, como no El Alami.
     - Predição vetorizada com o `EstimatorV2` do Aer (um circuito, matriz de parâmetros).
3. **Busca de hiperparâmetros dentro do treino balanceado:**
   - `GridSearchCV` com `StratifiedKFold(5)` sobre o treino subamostrado (~568 linhas).
   - **Scoring = ROC-AUC**, porque é invariante à proporção de classes e continua válido em folds 50/50. A PR-AUC também é registrada.
   - A comparação entre modelos e o limiar são decididos na **validação real** (0,17%), com **PR-AUC como métrica principal**.
4. **O pré-processamento vira hiperparâmetro.** Isso responde à sua pergunta sobre os scalers com evidência, e não por citação. O grid inclui:
   - scaler de entrada: `StandardScaler` / `RobustScaler`, **só no caminho com PCA e na amplitude**. No caminho com informação mútua os dois são idênticos (Fase 3b, CC8).
   - redutor: `PCA(k)` / `SelectKBest(mutual_info, k)`; o PCA é não supervisionado e maximiza variância, não separação de classes
   - range do MinMax: 0,25 / 0,5 / 1 / π/2 rad (revisado na Fase 3b; era [0,1] / [0,π]). Largura máxima de π/2, porque o α = 2 dos feature maps do Qiskit dá período π em x no mapa Z; no ZZ a monotonicidade acaba antes (≈ 0,6 rad) e a CV decide (CC11)
   - conjunto de features para amplitude e SVM clássico: 31 / 24 (sem as 7 de menor MI, como no Trabalho Humberto), adiado da Fase 3b
   - As varreduras de KTA e a seleção por MI que sustentam estas opções são feitas **só no treino** (Seções 4.7 a
     4.9; reorganização de 2026-10-07, CC7). O grid de larguras contém todas as larguras a < 1 desvio do máximo; na
     amplitude, o KTA sobe até a constante 5 e cai em 10. As opções estão em `config.ANGLE_RANGE_WIDTH_GRID` e
     `config.AMPLITUDE_PADDING_GRID`, com testes das restrições CC11 e CC4
   - constante de preenchimento da amplitude: 1 / 3 / 5 (zero fica de fora: descarta a norma, CC4; o 5 entrou na
     Fase 2, porque o KTA no treino sobe até 5 e cai em 10). Se o custo do Quantum AdaBoost apertar na Fase 5, o 1
     é o candidato a sair (KTA bem abaixo de 3 e 5)
   - feature map: `Z` (codificação de fase, sem emaranhamento, com kernel calculável classicamente) / `ZZ`
     (fase com emaranhamento linear/full); `reps` 1 ou 2 **nos dois** (`config.FEATURE_MAP_REPS_GRID`): com
     `reps=1` o Z é a fase pura, com kernel igual ao da rotação em Y; com `reps=2` o kernel muda (CC14)
   - Para o SVM clássico: Standard x Robust.
   - O AdaBoost com *stumps* é invariante à escala; isso é demonstrado empiricamente.
   **Grid por algoritmo** (proposto pelo Eric em 2026-10-07). A lista acima guarda as justificativas; esta matriz
   diz **o que vale para quem**. ✔ = entra no grid do modelo; — = não se aplica; **fixo** = igual para todos, fora
   do grid; ①/② = em aberto, ver abaixo. Os valores "a definir" saem do benchmark de custo, na Fase 5.

   | Dimensão | Opções | SVM | QSVM | AdaBoost | Quantum AdaBoost |
   |---|---|---|---|---|---|
   | **Protocolo (comum aos quatro)** | | | | | |
   | Treino balanceado (undersampling 1:1 no pipeline) | fixo | fixo | fixo | fixo | fixo |
   | Validação cruzada: 5 folds estratificados, ROC-AUC | fixo | fixo | fixo | fixo | fixo |
   | **Entrada (pré-processamento)** | | | | | |
   | Scaler | Standard / Robust | ✔ | ✔ (só com PCA e na amplitude) | — (*stumps* não dependem da escala) | ① |
   | Redutor para 5 componentes | PCA / informação mútua | ② | ✔ (variação A) | ② | ① |
   | Conjunto de features | 31 / 24 | ✔ (referência com todas as features) | ✔ (variação B) | — | ① |
   | **Codificação (só os quânticos)** | | | | | |
   | Faixa dos ângulos | 0,25 / 0,5 / 1 / π/2 | — | ✔ (A) | — | ① |
   | Feature map | Z / ZZ | — | ✔ (A) | — | ① |
   | Repetições do feature map | 1 / 2 | — | ✔ (A) | — | ① |
   | Emaranhamento do ZZ | linear / full | — | ✔ (A) | — | ① |
   | Constante da amplitude | 1 / 3 / 5 | — | ✔ (B) | — | ① |
   | Fase × rotação em Y (CC14) | condicional ao diagnóstico da Fase 4 | — | — (kernel idêntico) | — | condicional |
   | **Modelo: par SVM × QSVM** | | | | | |
   | C (penalidade do SVM) | a definir | ✔ | ✔ | — | — |
   | γ do kernel RBF | a definir | ✔ | — (a faixa dos ângulos e a constante fazem esse papel) | — | — |
   | **Modelo: par AdaBoost × Quantum AdaBoost** | | | | | |
   | Número de classificadores fracos (`n_estimators`) | a definir | — | — | ✔ | ✔ |
   | Taxa de aprendizado (`learning_rate`) | a definir | — | — | ✔ | ✔ |
   | Repetições do ansatz | a definir | — | — | — | ✔ |
   | Iterações do COBYLA | fixo e baixo (o classificador precisa ser fraco); a definir | — | — | — | fixo |

   **Dois pontos que a matriz deixou à vista, a decidir na Fase 5:**
   - **① Entrada e codificação do Quantum AdaBoost.** O plano diz que o grid dele é pequeno (`n_estimators`,
     `learning_rate`, repetições do ansatz) e também que as variações A e B valem para ele, mas não diz **com qual**
     scaler, redutor, faixa e mapa. Opções: herdar a melhor configuração do QSVM, ter um grid reduzido próprio ou usar
     valores fixos. Herdar é o mais barato, mas escolheria a codificação por um critério de kernel, que é justamente
     o que o CC14 mostrou que não se transfere para um modelo que mede o estado.
   - **② Entrada "idêntica à do quântico" dos clássicos.** A configuração *like-for-like* dá ao SVM e ao AdaBoost a
     mesma entrada do par quântico (5 componentes). Falta definir se eles herdam o redutor escolhido pelo quântico
     ou se o escolhem no próprio grid.
   - **③ De onde sai o conjunto de 24 features.** A lista das 7 features de menor informação mútua vinha de fora
     do pipeline (do Trabalho Humberto, que usou a base inteira). Uma lista fixa trazida de fora é uma escolha
     orientada por dados que deveriam estar isolados. No treino (Seção 4.5), as 7 últimas do ranking são `V13`,
     `V15`, `V22`, `V26`, `V25`, `V24` e `hour_cos`; não conferi se é a mesma lista do trabalho anterior. Saída sem
     vazamento: a opção "24" vira uma etapa do pipeline (`SelectKBest` com k = 24), refeita em cada fold só no
     treino (levantado em 2026-10-07, na dúvida do Eric sobre EDA × vazamento).
   - **④ Onde o sorteio do undersampling acontece em relação à validação cruzada.** O plano diz duas coisas que
     não cabem juntas ao pé da letra: o undersampling é uma etapa do pipeline, refeita em cada fold (decisão 1), e
     a validação cruzada roda sobre o treino já balanceado, com folds 50/50 (decisão 3). Em qualquer das saídas o
     pré-processamento continua ajustado só no treino; o que muda é o custo e o que os folds medem:
     - **(a) Validação cruzada no treino inteiro, sorteio dentro de cada fold.** Os folds de validação ficam na
       proporção real (34.047 linhas e ~57 fraudes cada), então a PR-AUC também valeria na validação cruzada. Custo
       do QSVM: ~148 s por fold, ~14,8 h para 72 combinações × 5 folds.
     - **(b) Um sorteio no treino (semente fixa) e validação cruzada dentro das 568 linhas.** É o que a decisão 3
       descreve. Folds 50/50, scoring ROC-AUC, ~2,4 s por fold (~14,6 min para as mesmas 72 × 5). A
       validação cruzada não enxerga a variação do sorteio (isso fica com as 5 sementes de robustez), e a etapa de
       undersampling do pipeline só age no ajuste final e nas sementes.
     - **(c) Híbrido:** sorteio dentro de cada fold e validação num subconjunto do fold (todas as fraudes + amostra
       das legítimas). Mais fiel que (b) e mais barato que (a), ao custo de um divisor de folds próprio. Com 57
       fraudes + 570 legítimas na validação: ~4,6 s por fold, ~27,8 min para as mesmas 72 × 5.
     Os tempos contam a simulação das linhas de treino (454) e de validação de cada fold no QSVM, a 4,28 ms por
     statevector (benchmark da Seção 1: 2.568 statevectors em 10,99 s). Não contam o Quantum AdaBoost, que é mais
     caro (levantado em 2026-10-07; tempos corrigidos no mesmo dia, ver NC9).

   **Nota para a Seção 6 do notebook (Fase 5): deixar claro o que é decisão de pré-processamento e o que é decisão
   do algoritmo** (pedido do Eric em 2026-10-07, para não repetir a confusão entre codificação de ângulo e de fase).
   A seção precisa:
   - **Abrir explicando que o grid é sobre o pipeline inteiro.** Scaler, redutor, faixa dos ângulos e mapa são
     hiperparâmetros tanto quanto o C ou o número de classificadores fracos: cada combinação reajusta todas as etapas
     em cada fold. Dizer por que se busca tudo junto (as escolhas interagem; escolher o pré-processamento por fora
     seria uma seleção que a validação cruzada não enxerga).
   - **Separar três tipos de decisão, sempre com o mesmo rótulo:** *entrada* (scaler, redutor, conjunto de
     features), *codificação* (faixa, mapa, repetições, emaranhamento, constante da amplitude; só os quânticos) e
     *modelo* (C, γ, número de classificadores fracos, taxa de aprendizado, repetições do ansatz).
   - **Mostrar a matriz dimensão × algoritmo** (a de cima, já com os valores definidos), com uma coluna a mais: o
     nome do parâmetro no pipeline (por exemplo `scaler`, `angle_scaler__feature_range`, `model__C`).
   - **Definir cada termo uma vez, antes de usá-lo,** e não usar a mesma palavra para duas coisas. Pares que
     confundem: codificação de fase × de ângulo (CC14); repetições do *feature map* × repetições do *ansatz*; faixa
     dos ângulos (largura de banda do kernel quântico) × γ do RBF; variação A/B (codificação) × configuração
     *like-for-like*/referência (entrada dos clássicos).
   - **Contar as combinações de cada modelo antes de rodar** (combinações × folds = ajustes) e o tempo estimado a
     partir do benchmark, numa tabela.
   - **Resolver os pontos ① e ② antes de rodar,** com a decisão e a justificativa escritas na própria seção.
   - **Na leitura dos resultados, reportar separado** o que a validação cruzada escolheu em entrada, codificação e
     modelo, para cada algoritmo.

5. **Codificação: as duas variações do TCC com o mesmo orçamento de qubits (5).**
   - **A (ângulo):** 5 componentes → 5 qubits. Há uma curva de sensibilidade para k de 4 a 8. "Ângulo" aqui é a
     família das codificações por rotação; a rigor, Z e ZZ são codificação de **fase** (CC14).
   - **B (amplitude):** 31 features + 1 constante = 32 amplitudes → 5 qubits, sem PCA.
   - As duas valem para o QSVM **e** para o Quantum AdaBoost. Wang e Li também usam amplitude no classificador fraco.
   - Diagnósticos quantitativos: concentração do kernel (distribuição dos valores fora da diagonal) e *kernel-target alignment*.
6. **Custo computacional,** que o TCC promete e o prompt não citava:
   - context manager `measure_cost()` com `time.perf_counter` e `tracemalloc`, medindo treino e inferência separadamente;
   - pico de RSS via `psutil`, porque o `tracemalloc` **não enxerga** a memória nativa do Aer e da libsvm (isso vira uma discussão no TCC).
7. **Reprodutibilidade:**
   - constantes `Final` em `config.py`: `RANDOM_STATE`, `UNDERSAMPLING_SEEDS` e os grids;
   - resultados salvos em `artifacts/results/*.json` com hiperparâmetros, versões e hash do commit;
   - cache de fits com `joblib`, como no `run_grid_search` do trabalho anterior;
   - a primeira célula imprime as versões.

---

## Estrutura de arquivos

```
ML-x-QML/
├─ pyproject.toml / requirements.txt   # pacote instalável (uv pip install -e .) + versões fixadas
├─ README.md, PLAN.md, study_notes.md  # em português; study_notes = caderno de aulas/perguntas
├─ metodologia.MD                      # NÃO alterado sem aprovação (ver "Pendências no texto do TCC")
├─ creditcard.csv.zip                  # não versionado
├─ notebooks/tcc_ml_vs_qml.ipynb       # experimento, seções 1 a 10
├─ src/fraud_qml/
│  ├─ config.py        # constantes, sementes, caminhos, grids
│  ├─ data.py          # load_dataset, validate_contract, remove_duplicates, engineer_features (log1p, hora sin/cos)
│  ├─ splitting.py     # stratified_three_way_split, validate_split, quase-duplicatas entre partes
│  ├─ preprocessing.py # build_feature_pipeline(scaler, reducer, k, angle_range); AmplitudePadder
│  ├─ quantum/encodings.py    # feature maps de ângulo (Z, ZZ) e amplitude
│  ├─ quantum/kernels.py      # QuantumKernelSVC, métricas de concentração/alinhamento
│  ├─ quantum/diagnostics.py  # varreduras de KTA (ângulo e amplitude), usadas na Seção 4
│  ├─ quantum/variational.py  # VariationalQuantumClassifier (sample_weight)
│  ├─ quantum/noise.py        # modelos de ruído do Aer
│  ├─ models.py        # fábrica dos pipelines dos 4 modelos + grids
│  ├─ tuning.py        # run_grid_search com cache
│  ├─ evaluation.py    # métricas, limiar F1, bootstrap, McNemar (precision_at_prevalence desde a Fase 2)
│  ├─ profiling.py     # measure_cost
│  └─ plotting.py
├─ tests/              # pytest
└─ artifacts/          # cache, results/, models/ (não versionado, exceto os resultados resumidos)
```

**Testes-chave (pytest):**
- Contrato dos dados e remoção de duplicatas.
- Split sem sobreposição e com a proporção preservada.
- O undersampling nunca toca validação/teste.
- `Standard→MinMax ≡ MinMax`.
- `AmplitudePadder` produz norma 1 e 2ⁿ entradas.
- O kernel quântico é simétrico, PSD e tem diagonal 1.
- O kernel de amplitude é igual a cos² calculado classicamente.
- O `VariationalQuantumClassifier` respeita `sample_weight`: peso zero equivale a remover a amostra.
- Compatibilidade com `clone`/`GridSearchCV` (`check_estimator`, dentro do possível).
- As métricas batem em casos conhecidos.

**Dependências (a instalar com sua aprovação neste plano):**
- `qiskit` 2.x
- `qiskit-aer`: conferir se há wheel cp312 para Windows; se não houver, usar o Python suportado mais próximo
- `qiskit-machine-learning` 0.9.x
- `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`, `scipy`, `matplotlib`, `seaborn`, `joblib`, `psutil`, `jupyter`, `ipykernel`, `pytest`

As versões são fixadas depois de um `uv pip install --dry-run` que confirma a compatibilidade.

---

## Estrutura do notebook (sua estrutura de 9 passos, mais a seção de ruído)

| Seção | Conteúdo |
|---|---|
| **1. Setup** | Imports, constantes, sementes, versões, detecção de CPUs e um *benchmark* de custo (1 fit/predict de cada modelo quântico) para dimensionar os grids. |
| **2. Coleta** | Leitura do `.zip`, contrato dos dados, apresentação das variáveis. |
| **3. Divisão** | Remoção de duplicatas (a única operação na base inteira além do contrato). Split 60/20/20 estratificado na base deduplicada (170.235 / 56.745 / 56.746 linhas; 284 / 94 / 95 fraudes), com os `assert`s, o porquê de dividir antes de balancear e duas checagens de integridade (quase-duplicatas; os dois dias da coleta). |
| **4. EDA + pré-processamento (só no treino)** | Escalas, `Amount`, `Time` e ranking de features no treino; `log1p(Amount)` e hora sin/cos decididos ali e aplicados às três partes. No **treino balanceado**: estudo dos scalers (Standard x Robust x MinMax, a equivalência Standard→MinMax), efeito da faixa na concentração do kernel, ângulo x amplitude, seleção de features (PCA x informação mútua) e a prova de que o undersampling age só no `fit`. Termina com as opções que seguem para o grid. |
| **5. Treinamento** | Dummy, SVM, AdaBoost, QSVM (A e B), Quantum AdaBoost (A e B) com hiperparâmetros padrão, avaliados na validação, com custo computacional. |
| **6. GridSearchCV** | `StratifiedKFold(5)` no treino balanceado. O grid do Quantum AdaBoost é pequeno (n_estimators, learning_rate, reps do ansatz), dimensionado pelo benchmark da Seção 1. A seção separa decisões de entrada, de codificação e de modelo e mostra a matriz dimensão × algoritmo (ver "Nota para a Seção 6" na decisão de arquitetura 4). |
| **7. Avaliação** | Na validação: PR-AUC (principal), ROC-AUC, precisão, recall, F1, **MCC** (usado por Randhawa), acurácia, matriz de confusão, curvas PR/ROC, limiar por F1-máx (0,5 contra o ótimo) e modelo de custo complementar. Tabela comparativa dos pares e custo computacional. |
| **8. Diagnóstico** | `learning_curve` (score de treino x validação conforme cresce o nº de amostras) e `validation_curve`: C/γ no SVM, C e range/largura de banda no QSVM, n_estimators nos dois AdaBoost via `staged_decision_function`, sem refazer fits. |
| **9. Modelo final** | Retreino em treino+validação (undersampling de novo) e **avaliação única no teste** com o limiar escolhido. Bootstrap pareado para as diferenças SVM−QSVM e AdaBoost−QAdaBoost e McNemar. **Robustez:** o pipeline final repetido com as 5 sementes de undersampling (média ± desvio), sem nenhuma decisão tomada a partir disso. Ficha do modelo em JSON. |
| **10. Ruído (extra)** | Os modelos quânticos finais reavaliados com ruído de despolarização (p ∈ {0; 0,001; 0,005; 0,01; 0,03}, o último igual ao do Wang), erro de leitura e shots finitos. Roda num subconjunto fixo do teste (todas as fraudes + amostra das legítimas), porque simular ruído em 56 mil linhas é inviável. A comparação ideal x ruidoso é feita **no mesmo subconjunto**. |

---

## Forma de trabalho (a mesma do Trabalho Humberto)

- **Uma fase por turno.** Eu escrevo o módulo e as células, rodo os testes e o notebook e dou a aula no chat. Só sigo quando você liberar.
- **Em cada fase:**
  - markdown antes do código: o quê, por quê e alternativas descartadas;
  - código comentado no "porquê", em português;
  - markdown depois do código, interpretando os resultados reais.
- **Na aula:**
  - conceitos explicados do zero, com exemplo numérico pequeno;
  - perguntas de fixação em 🟢 base / 🟡 intermediária / 🔴 desafio;
  - o registro fica no `study_notes.md` deste projeto;
  - no fim de cada fase, **o que ler para a próxima** (tabela abaixo).
- **Registro de falhas e controvérsias:** `methodological_issues.md` acumula, fase a fase, os problemas
  encontrados (nos artigos, no texto do TCC e no nosso próprio código) e como cada um foi tratado.
- **Ordem das fases.** As fases estão deslocadas em um em relação às seções do notebook: a Fase N é a Seção N+1
  (a Seção 1 é a Fase 0). Na reorganização de 2026-10-07 o split passou para antes da exploração, e as fases foram
  renumeradas para manter essa regra: a antiga Fase 3 (divisão) virou **Fase 2**, e as antigas 2a e 2b
  (exploração) viraram **3a e 3b**. Os registros escritos antes disso (`methodological_issues.md`,
  `study_notes.md`) mantêm os nomes antigos, com uma nota de correspondência no topo:

  | Fase | Seção do notebook |
  |---|---|
  | 0 | 1. Setup (+ `git init`, `.venv`, dependências, esqueleto do pacote e testes) |
  | 1 | 2. Coleta e carregamento |
  | 2 | 3. Duplicatas e divisão treino/validação/teste |
  | 3a | 4. EDA e engenharia de features, só no treino (4.1 a 4.5) |
  | 3b | 4. Scalers, codificação, seleção de features e undersampling, no treino balanceado (4.6 a 4.11) |
  | 4 | 5. Treinamento |
  | 5 | 6. Ajuste de hiperparâmetros |
  | 6 | 7. Avaliação |
  | 7 | 8. Diagnóstico |
  | 8 | 9. Modelo final |
  | 9 | 10. Ruído quântico |

- **Commits:** feitos por você, no final do projeto.

## Leituras, por fase

**Regra (revisada em 2026-10-03):** um material por conceito, e só a menor seção que o explica. Material de estudo
ensina o conceito do zero (guia oficial, curso, livro-texto aberto); artigo entra só para citar no texto do TCC.
"Só se faltar" marca pré-requisito: pule se já sabe. Todos gratuitos, de fora da Base de Conhecimento, com autor,
link e nome de seção conferidos na web.

Abreviações: **QML-IBM** = curso [*Quantum machine learning*](https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning/)
da IBM Quantum Learning; **sklearn** = [*User Guide*](https://scikit-learn.org/stable/user_guide.html) do
scikit-learn 1.9.

### Fase 2 — Divisão e undersampling (já entregue)

| Conceito | Ler só isto |
|---|---|
| Vazamento | sklearn, [12.2 *Data leakage*](https://scikit-learn.org/stable/common_pitfalls.html) |
| Reamostrar antes do split 
| imbalanced-learn, [9.1 *Data leakage*](https://imbalanced-learn.org/stable/common_pitfalls.html) |
| Undersampling distorce a probabilidade | Google *Machine Learning Crash Course*, [*Class-imbalanced datasets*](https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets): *Step 1: Downsample the majority class* e *Step 2: Upweight the downsampled class* |

**Para citar no TCC:** Dal Pozzolo et al. 2015 ([ResearchGate](https://www.researchgate.net/publication/283349138_Calibrating_Probability_with_Undersampling_for_Unbalanced_Classification)):
a fórmula de correção usada na Seção 4.10.

### Fase 3b — Scalers e codificação (já entregue)

| Conceito | Ler só isto |
|---|---|
| Kernel e SVM | QML-IBM, lição [*Review of relevant machine learning methods*](https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning/classical-ml-review): seções *Support vector machines*, *Kernel methods and how quantum can play a role* e *Moving to higher dimensional spaces* |
| Largura de banda do kernel (o γ do RBF) | sklearn, [1.4.6.1 *Parameters of the RBF Kernel*](https://scikit-learn.org/stable/modules/svm.html) |
| Codificação de ângulo, de amplitude e o mapa ZZ | QML-IBM, lição [*Data encoding*](https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning/data-encoding): seções *Amplitude encoding*, *Angle encoding* e *ZZ feature map* |
| Só se faltar: produto interno entre estados | IBM, *Basics of quantum information*, [*Inner products and projections*](https://quantum.cloud.ibm.com/learning/en/courses/basics-of-quantum-information/quantum-circuits/inner-products-and-projections) |

**Para citar no TCC:** Havlíček et al. 2019, [arXiv:1804.11326](https://arxiv.org/abs/1804.11326) (origem do
ZZFeatureMap; dados artificiais gerados pelo próprio mapa) e Schuld 2021, [arXiv:2101.11020](https://arxiv.org/abs/2101.11020)
(seção II, *Summary of results*: modelos de QML supervisionado são métodos de kernel, CC3).

### Fase 4 — Treinamento

| Conceito | Ler só isto |
|---|---|
| AdaBoost: reponderação e classificador fraco | sklearn, [1.11.7 *AdaBoost*](https://scikit-learn.org/stable/modules/ensemble.html) |
| De onde vem o α | Schapire 2013, *Explaining AdaBoost* ([PDF](https://www.schapire.net/papers/explaining-adaboost.pdf)): só a seção 1, *Introduction* |
| QSVM = SVM com kernel pré-calculado | sklearn, [1.4.6.2 *Custom Kernels*](https://scikit-learn.org/stable/modules/svm.html) |
| Classificador variacional | QML-IBM, lição [*QVCs and QNNs*](https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning/qvc-qnn): de *Variational quantum classifier: a first attempt* até *Testing and accuracy* |

**Para citar no TCC:** Zhu et al. 2009, *Multi-class AdaBoost* ([PDF](https://hastie.su.domains/Papers/samme.pdf)):
equação (1) e "when K = 2, SAMME reduces to AdaBoost" (WA2).

### Fase 5 — Ajuste de hiperparâmetros

| Conceito | Ler só isto |
|---|---|
| Grid search com validação cruzada | sklearn, [3.2.1 *Exhaustive Grid Search*](https://scikit-learn.org/stable/modules/grid_search.html) |
| Por que o vencedor do grid é otimista | Raschka 2018, [arXiv:1811.12808](https://arxiv.org/abs/1811.12808): seção 3.3, *The Three-Way Holdout Method for Hyperparameter Tuning* |
| Concentração do kernel | Já visto: aula da Fase 0, no `study_notes.md` |

**Para citar no TCC:** Thanasilp et al. 2024, [arXiv:2208.11060](https://arxiv.org/abs/2208.11060): seção II.C,
as fontes de concentração (CC5).

### Fase 6 — Avaliação

| Conceito | Ler só isto |
|---|---|
| Limiar, matriz de confusão, precisão e recall | Google *Machine Learning Crash Course*: [*Thresholds and the confusion matrix*](https://developers.google.com/machine-learning/crash-course/classification/thresholding) e [*Accuracy, recall, precision, and related metrics*](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall) |
| Curva ROC | Google *Machine Learning Crash Course*: [*ROC and AUC*](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) |
| Curva PR e MCC | sklearn, [3.4.4.9](https://scikit-learn.org/stable/modules/model_evaluation.html) (a parte da curva precisão-recall) e 3.4.4.13 *Matthews correlation coefficient* |

**Para citar no TCC:** Saito & Rehmsmeier 2015 ([PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)):
a subseção da curva PR cuja linha de base muda com a proporção de classes. Para o MCC, o Randhawa et al. (2018), que
já está na Base de Conhecimento.

### Fase 7 — Diagnóstico

| Conceito | Ler só isto |
|---|---|
| Viés × variância, curvas de validação e de aprendizado | sklearn, [3.5 *Validation curves*](https://scikit-learn.org/stable/modules/learning_curve.html): 3.5.1 e 3.5.2 |
| *Barren plateau* | QML-IBM, lição *QVCs and QNNs*: o trecho sobre *barren plateau* |

**Para citar no TCC:** nenhum artigo novo.

### Fase 8 — Modelo final

| Conceito | Ler só isto |
|---|---|
| Intervalo de confiança por bootstrap | Raschka 2018: seção 2.4, *The Bootstrap Method and Empirical Confidence Intervals* |
| Teste de McNemar | Raschka 2018: seções 4.3, *Comparing Two Models with the McNemar Test*, e 4.4, *Exact p-Values via the Binomial Test* |

**Para citar no TCC:** o próprio Raschka 2018.

### Fase 9 — Ruído quântico

| Conceito | Ler só isto |
|---|---|
| Ruído como canal (despolarização, defasagem) | IBM, *General formulation of quantum information*, [*Quantum channel basics*](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/quantum-channels/quantum-channel-basics): seções *The completely dephasing channel* e *The completely depolarizing channel* |
| Só se faltar: matriz densidade | IBM, mesmo curso: [*Density matrix basics*](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/density-matrices/density-matrix-basics) |
| Erro de despolarização e de leitura no simulador | Qiskit Aer, [*Building Noise Models*](https://qiskit.github.io/qiskit-aer/tutorials/3_building_noise_models.html): seção *Quantum Errors* |

**Para citar no TCC:** Preskill 2018, [arXiv:1801.00862](https://arxiv.org/abs/1801.00862): seção 4, *The NISQ
era unfolds*.

### Resumo: artigos que ficam (só para citar)

| Artigo | Para citar o quê | Fase |
|---|---|---|
| Dal Pozzolo et al. 2015 | Correção da probabilidade após o undersampling | 2 |
| Havlíček et al. 2019 | Origem do ZZFeatureMap e do QSVM | 3b |
| Schuld 2021 | QML supervisionado = métodos de kernel (CC3) | 3b |
| Zhu et al. 2009 | SAMME = AdaBoost com 2 classes (WA2) | 4 |
| Thanasilp et al. 2024 | Concentração exponencial do kernel (CC5) | 5 |
| Saito & Rehmsmeier 2015 | PR-AUC como métrica principal | 6 |
| Preskill 2018 | Definição de NISQ | 9 |

## Pendências no texto do TCC (não vou alterar sem sua aprovação)

As próximas versões do texto precisam refletir estas mudanças:
- Split de 80/20 para **60/20/20**, feito **antes** do undersampling.
- **Remoção de duplicatas**; hoje o texto argumenta o contrário.
- A seção de SMOTE/ADASYN sai, e o **pipeline clássico passa a ser o mesmo do quântico**.
- A sequência Standard→MinMax passa a ter o PCA no meio.
- O ZZFeatureMap é um mapa de ângulo **com emaranhamento**, não codificação de ângulo "pura".
- **MCC e PR-AUC** entram nas métricas.
- A seção de ruído vira resultado.

## Riscos

- **Quantum AdaBoost é caro.** Treinar VQC × rodadas × folds pode levar horas. Mitigação: benchmark na Seção 1, grid pequeno, COBYLA com poucas iterações (o classificador precisa ser *fraco*), cache e `staged_decision_function`.
- **Wheels do `qiskit-aer` para Windows/py3.12:** conferidas no dry-run da Fase 0. O plano B é outra versão do Python via `uv`.
- **`SVC` com kernel callable + `clone`:** coberto por testes. O plano B é `kernel="precomputed"` com o cálculo do kernel dentro do wrapper.
- **Expectativa honesta de resultado:** com ~568 amostras de treino e a proporção real no teste, a precisão vai ser baixa para todos os modelos, bem abaixo dos 0,818 de PR-AUC do Random Forest com a base inteira do trabalho anterior. Isso é resultado, não defeito, e vira discussão sobre o custo do undersampling imposto pelo QML.
- **Pouca amostra no teste (95 fraudes):** justamente o que as sementes, o bootstrap e o McNemar tratam.

## Verificação

- `.venv\Scripts\python -m pytest` todo verde a cada fase.
- `jupyter nbconvert --to notebook --execute notebooks/tcc_ml_vs_qml.ipynb` sem erros.
- Os `assert`s do notebook passam: splits disjuntos, proporção preservada e undersampling ausente em validação/teste. O teste é usado uma única vez (Seção 9), e a Seção 10 usa só modelos já finais.
- `artifacts/results/` contém, para cada modelo: hiperparâmetros, métricas, custo, versões e commit.
