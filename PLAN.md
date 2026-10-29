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
   - **A codificação é uma etapa do pipeline** (`StatevectorEncoder`, etapa `encoder`; implementado na Fase 4). Ela
     simula o estado de cada linha uma vez, em lote no Aer, e entrega os estados ao modelo. Serve ao kernel do QSVM e
     a todos os classificadores fracos do Quantum AdaBoost, e deixa as decisões de codificação com o mesmo nome de
     parâmetro nos dois (`encoder__encoding`, `encoder__feature_map_reps`, `encoder__entanglement`).
   - **QSVM:** `QuantumKernelSVC`, um `SVC(kernel="precomputed")` sobre o kernel de fidelidade dos estados
     codificados. Um statevector por amostra e o kernel vira um produto matricial, O(N) simulações, então ele roda
     no teste inteiro (56 mil linhas). O desenho original era `SVC(kernel=callable)`; um teste garante que as
     decisões são as mesmas.
   - **Quantum AdaBoost:** `AdaBoostClassifier(estimator=VariationalQuantumClassifier(...))`.
     - O `VariationalQuantumClassifier` implementa `fit(X, y, sample_weight)` com cross-entropy ponderada.
     - Circuito: feature map + ansatz `RealAmplitudes`; mede-se ⟨Z⟩ no qubit 0.
     - **Simulação (mudou na Fase 4):** o estado codificado não depende dos pesos, então o ansatz é aplicado como
       uma matriz 32 × 32 calculada pelo Qiskit (`Operator`), e avaliar o circuito para todas as linhas é um produto
       matricial. O plano previa o `EstimatorV2` do Aer; um teste compara os dois caminhos (diferença < 10⁻⁹). O
       circuito completo continua disponível para a simulação ruidosa da Fase 9, que precisa do Aer.
     - **CC14, resolvido na Fase 4:** os mapas Z/ZZ são codificação de **fase**; num modelo que mede o estado,
       fase e rotação em Y não são equivalentes. O diagnóstico da Seção 5.3 (classificador fraco de 5 qubits, sem
       boosting, validação cruzada no treino balanceado) mostrou a diferença na acurácia com o corte em 0,5: com PCA e
       ansatz de 1 repetição, 0,654 ± 0,001 na fase pura contra 0,874 ± 0,003 na rotação em Y; a ROC-AUC quase não
       muda (0,919 contra 0,940). Com informação mútua e ansatz de 2 repetições a diferença cai para 0,026. **Vale na
       faixa padrão [0, π/2] com StandardScaler:** em 1 rad a fase pura empata com a RY (0,878 e 0,863), e em 0,25 e
       0,5 rad o mapa ZZ com 2 repetições empata com ela (Seção 5.3.1). A RY é a mais estável (0,817 a 0,875 nas 8
       combinações de largura e scaler). **RY entra no grid do Quantum AdaBoost na Fase 5** (o custo permite: um
       ajuste leva segundos; o grid passa de 6 para 7 opções de codificação).
     - Otimizador COBYLA, como no El Alami.
3. **Busca de hiperparâmetros por validação cruzada no treino** (revisada em 2026-10-27, ponto ④ abaixo):
   - `StratifiedKFold(5)` sobre o **treino inteiro** (170.235 linhas); o undersampling é a primeira etapa do pipeline
     e sorteia o treino balanceado **dentro de cada fold**. Os folds de validação ficam na proporção real (~34 mil
     linhas e ~57 fraudes cada).
   - **Escolha pela PR-AUC**, a métrica principal, que nos folds na proporção real tem o mesmo significado que na
     validação. A ROC-AUC também é registrada.
   - **Saída de emergência, com o critério fixado antes de rodar:** antes do grid completo, mede-se uma combinação ×
     5 folds de cada modelo e projeta-se o tempo total. Se a Seção 6 projetada passar de **2 h** de execução, volta-se
     ao desenho anterior (um sorteio do undersampling e validação cruzada dentro das 568 linhas, com ROC-AUC).
   - A comparação entre modelos e o limiar são decididos na **validação real** (0,17%), com **PR-AUC como métrica principal**.
4. **O pré-processamento vira hiperparâmetro.** Isso responde à sua pergunta sobre os scalers com evidência, e não por citação. O grid inclui:
   - scaler de entrada: `StandardScaler` / `RobustScaler`, **só no caminho com PCA e na amplitude**. No caminho com informação mútua os dois são idênticos (Fase 3b, CC8).
   - redutor: `PCA(k)` / `SelectKBest(mutual_info, k)`; o PCA é não supervisionado e maximiza variância, não separação de classes
   - range do MinMax: 0,25 / 0,5 / 1 / π/2 rad (revisado na Fase 3b; era [0,1] / [0,π]). Largura máxima de π/2, porque o α = 2 dos feature maps do Qiskit dá período π em x no mapa Z; no ZZ a monotonicidade acaba antes (≈ 0,6 rad) e a CV decide (CC11)
   - ~~conjunto de features para amplitude e SVM clássico: 31 / 24~~ **retirado em 2026-10-27 (ponto ③):** a amplitude
     usa sempre as 31 features + 1 constante (32 amplitudes = 5 qubits), e a constante é que entra no grid
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
   **Grid por algoritmo** (proposto pelo Eric em 2026-10-07; pontos ① a ⑤ decididos em 2026-10-27). A lista acima
   guarda as justificativas; esta matriz diz **o que vale para quem**. ✔ = entra no grid do modelo; — = não se
   aplica; **fixo** = igual para todos, fora do grid. Os valores dos hiperparâmetros de modelo foram fixados na
   Fase 5, em escala logarítmica, e o grid inteiro cabe no limite de 2 h (piloto da Seção 6.1.1). Os quatro grids
   são montados por `fraud_qml.tuning.project_search_spaces`.

   | Dimensão | Opções | SVM | QSVM | AdaBoost | Quantum AdaBoost |
   |---|---|---|---|---|---|
   | **Protocolo (comum aos quatro)** | | | | | |
   | Treino balanceado (undersampling 1:1 no pipeline, sorteado em cada fold) | fixo | fixo | fixo | fixo | fixo |
   | Validação cruzada: 5 folds estratificados no treino inteiro, escolha pela PR-AUC | fixo | fixo | fixo | fixo | fixo |
   | **Entrada (pré-processamento)** | | | | | |
   | Scaler | Standard / Robust | ✔ | ✔ (só com PCA e na amplitude) | — (*stumps* não dependem da escala, Seção 5.5) | ✔ (só com PCA e na amplitude) |
   | Redutor para 5 componentes | PCA / informação mútua | ✔ (configuração *like-for-like*) | ✔ (variação A) | ✔ (configuração *like-for-like*) | ✔ (variação A) |
   | Conjunto de features | 31 (fixo; a opção 24 saiu, ponto ③) | fixo | fixo (variação B) | fixo | fixo (variação B) |
   | **Codificação (só os quânticos)** | | | | | |
   | Faixa dos ângulos | 0,25 / 0,5 / 1 / π/2 | — | ✔ (A) | — | ✔ (A), buscada junto com o mapa (Seção 5.3.1) |
   | Feature map | Z / ZZ (+ RY no Quantum AdaBoost) | — | ✔ (A) | — | ✔ (A) |
   | Repetições do feature map | 1 / 2 | — | ✔ (A) | — | ✔ (A) |
   | Emaranhamento do ZZ | linear / full | — | ✔ (A) | — | ✔ (A) |
   | Constante da amplitude | 1 / 3 / 5 | — | ✔ (B) | — | ✔ (B) |
   | Fase × rotação em Y (CC14) | RY como opção a mais de mapa (decidido na Fase 4, Seção 5.3) | — | — (kernel idêntico ao do Z com 1 repetição) | — | ✔ |
   | **Modelo: par SVM × QSVM** | | | | | |
   | C (penalidade do SVM) | 0,1 / 1 / 10 / 100 (`config.SVM_C_GRID`) | ✔ | ✔ | — | — |
   | γ do kernel RBF | 'scale' / 0,01 / 0,1 / 1 / 10 (`config.SVM_GAMMA_GRID`) | ✔ | — (a faixa dos ângulos e a constante fazem esse papel) | — | — |
   | **Modelo: par AdaBoost × Quantum AdaBoost** | | | | | |
   | Teto de classificadores fracos (`n_estimators`) | 10 / 25 / 50 / 100, lidos rodada a rodada de um ajuste com 100 (`config.BOOSTING_N_ESTIMATORS_GRID`) | — | — | ✔ | ✔ |
   | Taxa de aprendizado (`learning_rate`) | 0,1 / 0,5 / 1 (`config.BOOSTING_LEARNING_RATE_GRID`) | — | — | ✔ | ✔ |
   | Repetições do ansatz | 1 / 2 (`config.ANSATZ_REPS_GRID`) | — | — | — | ✔ |
   | Iterações do COBYLA | fixo e baixo (o classificador precisa ser fraco); 60 desde a Fase 4 (no treino, a curva de custo já achatou nesse ponto na variação B e em 2 dos 3 sorteios da A, Seção 5.2) | — | — | — | fixo |

   **Pontos que a matriz e a Fase 4 deixaram à vista, decididos com o Eric em 2026-10-27** (o texto de cada um
   guarda a discussão; a decisão vem na primeira linha):
   - **① Entrada e codificação do Quantum AdaBoost.** **Decisão: grid próprio**, com scaler, redutor, faixa,
     mapa (Z, ZZ e RY), repetições, emaranhamento e constante buscados juntos; a Seção 5.3.1 já tinha mostrado que
     herdar a configuração do QSVM não serve. *(Fase 4: no classificador fraco, a melhor codificação muda
     com a largura da faixa e com o scaler, Seção 5.3.1. Herdar a configuração do QSVM não serve; largura e
     codificação precisam ser buscadas juntas.)* O plano diz que o grid dele é pequeno (`n_estimators`,
     `learning_rate`, repetições do ansatz) e também que as variações A e B valem para ele, mas não diz **com qual**
     scaler, redutor, faixa e mapa. Opções: herdar a melhor configuração do QSVM, ter um grid reduzido próprio ou usar
     valores fixos. Herdar é o mais barato, mas escolheria a codificação por um critério de kernel, que é justamente
     o que o CC14 mostrou que não se transfere para um modelo que mede o estado.
   - **② Entrada "idêntica à do quântico" dos clássicos.** **Decisão: os clássicos escolhem o redutor (PCA ou
     informação mútua) no próprio grid;** na comparação do par, usa-se o clássico com o **mesmo redutor** que o
     quântico escolheu, tirado dos resultados do próprio grid. A entrada continua idêntica e não custa ajuste a mais.
     A configuração *like-for-like* dá ao SVM e ao AdaBoost a
     mesma entrada do par quântico (5 componentes). Falta definir se eles herdam o redutor escolhido pelo quântico
     ou se o escolhem no próprio grid.
   - **③ De onde sai o conjunto de 24 features.** **Decisão: a opção de 24 features sai do grid**, para a amplitude e
     para o SVM clássico. A amplitude usa sempre as 31 features + 1 constante (32 amplitudes = 5 qubits, o orçamento
     comum), e a constante tem as próprias opções no grid; o SVM com 31 features é a referência com todas as
     features. Com isso, a lista trazida de fora deixa de existir no protocolo. A lista das 7 features de menor informação mútua vinha de fora
     do pipeline (do Trabalho Humberto, que usou a base inteira). Uma lista fixa trazida de fora é uma escolha
     orientada por dados que deveriam estar isolados. No treino (Seção 4.5), as 7 últimas do ranking são `V13`,
     `V15`, `V22`, `V26`, `V25`, `V24` e `hour_cos`; não conferi se é a mesma lista do trabalho anterior. Saída sem
     vazamento: a opção "24" vira uma etapa do pipeline (`SelectKBest` com k = 24), refeita em cada fold só no
     treino (levantado em 2026-10-07, na dúvida do Eric sobre EDA × vazamento).
   - **④ Onde o sorteio do undersampling acontece em relação à validação cruzada.** **Decisão: saída (a)**, com
     escolha pela PR-AUC e volta à (b) se a Seção 6 projetada passar de 2 h (decisão de arquitetura 3). O motivo da
     escolha anterior pela (b) era o custo, que caiu: com a simulação em lote da Fase 4 (0,317 ms por linha, Seção
     5.4), simular os estados de um fold de validação de ~34 mil linhas leva cerca de 10,8 s, e o grid de cada modelo
     quântico gastaria cerca de 6 a 7 min disso em 12 núcleos (estimativa calculada em 2026-10-27, a medir no piloto).
     Os tempos de (a), (b) e (c) abaixo usam o custo da Seção 1 e ficaram como registro. O plano diz duas coisas que
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

   - **⑤ O boosting quântico para antes das rodadas pedidas** (achado da Fase 4, Seções 5.5 e 5.5.1; CC17).
     **Decisão: aceitar a parada** (é o comportamento do `AdaBoostClassifier`, igual nos dois lados do par) e pôr a
     **taxa de aprendizado no grid** dos dois AdaBoost, o que testa a saída "reduzir a taxa" sem mudar o protocolo.
     `n_estimators` é teto, e o número de classificadores mantidos é registrado em cada combinação. O corte ajustável
     e a regra do artigo ficam de fora. O
     scikit-learn interrompe o boosting quando o classificador de uma rodada tem erro ponderado ≥ 0,5 e o descarta.
     Nas 12 execuções da 5.5.1 (só treino) ficaram de 7 a 44 classificadores, nunca os 50 pedidos, e o número
     depende da semente do boosting (padrão A: 13, 44 e 24). Mais orçamento não resolve (300 avaliações: 17, 29 e
     13), nem um classificador fraco melhor (com RY: 13, 7 e 10). `n_estimators` é um **teto**. A Seção 6 precisa
     registrar quantos foram mantidos em cada combinação e decidir o que fazer com a parada:
     - aceitar (é o comportamento do `AdaBoostClassifier`, igual nos dois lados do par);
     - reduzir `learning_rate` (os pesos mudam mais devagar); não testado;
     - dar ao classificador fraco um corte ajustável (na variação A ele tem acurácia de 0,660 com o corte fixo em
       0,5 e de 0,843 com o melhor corte): **mudança de protocolo, a aprovar**;
     - adotar a regra do artigo (Wang et al. ficam com a iteração de menor erro e exigem ε < ½): **mudança de
       protocolo, a aprovar**.
   - **⑥ O grid do Quantum AdaBoost não precisa mais ser pequeno** (Fase 4). Com os estados simulados em lote e o
     ansatz como matriz, um ajuste leva cerca de 2,5 s e uma avaliação do custo, cerca de 3 ms (a Seção 1 estimava
     175 ms). A restrição "só os vencedores vão para a validação" também caiu: pontuar a validação leva menos de
     20 s. O que limita o grid agora é o método (quantas comparações a validação cruzada sustenta), e não o tempo.
   - **⑦ A linha de base quântica parte de um ponto fraco** (CC16): os padrões da Fase 4 são as opções de menor KTA
     da Seção 4 (constante 1; faixa π/2). O ganho do ajuste nos quânticos precisa ser reportado com essa ressalva.
     A constante 1 é a suspeita do mau desempenho do Quantum AdaBoost B.

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
   - **Escrever na própria seção as decisões dos pontos ① a ⑤** (tomadas em 2026-10-27), com a justificativa.
   - **Estrutura da seção** (confirmada pelo Eric em 2026-10-27):
     1. abertura: o grid é sobre o pipeline inteiro, os três tipos de decisão e os termos;
     2. opções do pré-processamento e da codificação (scaler, redutor, faixa, mapa, repetições, emaranhamento,
        constante), com a matriz dimensão × algoritmo e o nome de cada parâmetro no pipeline;
     3. grid do par SVM × QSVM (C nos dois, γ só no SVM; QSVM nas variações A e B);
     4. grid do par AdaBoost × Quantum AdaBoost (teto de `n_estimators`, taxa de aprendizado, repetições do ansatz;
        no Quantum AdaBoost, também a codificação, com RY; classificadores mantidos em cada combinação);
     5. leitura: o que a validação cruzada escolheu em entrada, codificação e modelo, por algoritmo.
   - **Na leitura dos resultados, reportar separado** o que a validação cruzada escolheu em entrada, codificação e
     modelo, para cada algoritmo.

5. **Codificação: as duas variações do TCC com o mesmo orçamento de qubits (5).**
   - **A (ângulo):** 5 componentes → 5 qubits. "Ângulo" aqui é a família das codificações por rotação; a rigor, Z e
     ZZ são codificação de **fase** (CC14). Há uma curva de sensibilidade para 4 a 8 qubits, que ficou sem lugar
     até 2026-10-27 e foi para a **Seção 8** (Fase 7): o QSVM e o Quantum AdaBoost refeitos com 4 a 8 componentes,
     cada um com a configuração vencedora do próprio grid, para responder "com mais qubits os quânticos se sairiam
     melhor?". Só a variação A entra: na amplitude, as 31 features + 1 constante fixam 5 qubits. O desenho fino fica
     para a Fase 7 (ver a linha da Seção 8 na estrutura do notebook).
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
│  ├─ quantum/encodings.py    # feature maps (Z, ZZ, RY), simulação em lote e StatevectorEncoder (etapa do pipeline)
│  ├─ quantum/kernels.py      # QuantumKernelSVC, kernel de fidelidade, métricas de concentração/alinhamento
│  ├─ quantum/diagnostics.py  # varreduras de KTA (Seção 4) e de codificação no classificador fraco (Seção 5)
│  ├─ quantum/variational.py  # VariationalQuantumClassifier (sample_weight)
│  ├─ quantum/noise.py        # modelos de ruído do Aer (Fase 9)
│  ├─ models.py        # fábrica dos pipelines dos 4 modelos, fit_and_score (grids: Fase 5)
│  ├─ tuning.py        # run_grid_search com cache (Fase 5)
│  ├─ evaluation.py    # métricas (Fase 4), limiar F1, bootstrap, McNemar (precision_at_prevalence desde a Fase 2)
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
| **5. Treinamento** | Dummy, SVM, AdaBoost (os dois clássicos com 31 features e com os 5 componentes da variação A), QSVM (A e B), Quantum AdaBoost (A e B) com hiperparâmetros padrão, avaliados na validação (PR-AUC com bootstrap pareado), com custo computacional. Curva de custo do classificador fraco, diagnóstico fase × RY com largura e scaler (CC14), variação entre sementes num conjunto interno do treino, e o boosting por dentro (ε e α por rodada, classificador descartado, parada por semente). |
| **6. GridSearchCV** | `StratifiedKFold(5)` no treino inteiro, com o undersampling sorteado em cada fold e escolha pela PR-AUC (decisão de arquitetura 3; volta ao treino balanceado se a projeção passar de 2 h). Cinco blocos: abertura; opções do pré-processamento e da codificação com a matriz dimensão × algoritmo; grid SVM × QSVM; grid AdaBoost × Quantum AdaBoost (grid próprio, com codificação e largura juntas e RY); leitura por tipo de decisão. O grid é dimensionado pelo custo medido na Seção 5.4, e não pelo benchmark da Seção 1 (ver "Nota para a Seção 6" na decisão de arquitetura 4). |
| **7. Avaliação** | Na validação: PR-AUC (principal), ROC-AUC, precisão, recall, F1, **MCC** (usado por Randhawa), acurácia, matriz de confusão, curvas PR/ROC, limiar por F1-máx (0,5 contra o ótimo) e modelo de custo complementar. Tabela comparativa dos pares e custo computacional. |
| **8. Diagnóstico** | `learning_curve` (score de treino x validação conforme cresce o nº de amostras) e `validation_curve`: C/γ no SVM, C e range/largura de banda no QSVM, n_estimators nos dois AdaBoost via `staged_decision_function`, sem refazer fits. **As curvas de validação descrevem o que há além das bordas do grid** que a Seção 6 deixou registradas (CC19: as 8 escolhidas estão na borda de alguma dimensão, inclusive a faixa 0,25 e a constante 5), **sem mudar a escolha**: as configurações da Seção 6 ficam congeladas, porque a Seção 8 vem depois da validação. Estender os grids antes da Seção 7 é decisão pendente com o Eric. **Curva de qubits** (proposta do Eric em 2026-10-27): QSVM e Quantum AdaBoost da variação A com 4 a 8 componentes, cada um com a configuração vencedora do próprio grid, para medir se mais qubits ajudariam. A fechar na Fase 7: (1) os clássicos com os mesmos k componentes, para separar "mais qubits" de "mais informação"; (2) se a largura da faixa é reajustada para cada k (a concentração do kernel muda com o número de qubits, CC5); (3) várias sementes no Quantum AdaBoost (CC17); (4) é diagnóstico: o protocolo continua com 5 qubits. |
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

**Regra (2026-10-20): ao fim de toda fase, conferir se os materiais explicam o que a seção aplicou.** Para cada
conceito usado no código, nas leituras e nas decisões da seção, a tabela da fase precisa ter um material que o
explique do jeito que o projeto usa; se não tiver, entra um material novo, ou fica registrado por que não precisa
(definido no próprio notebook, já estudado em outro trabalho, detalhe explicado na seção). Conceito usado antes da
fase em que é estudado ganha uma nota apontando a leitura futura. É o passo 9 do `/fase`. Na primeira conferência
(2026-10-20) as Fases 0 a 3b foram revistas; o que entrou nelas está marcado com *(acrescentado na conferência)*.

Abreviações: **QML-IBM** = curso [*Quantum machine learning*](https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning/)
da IBM Quantum Learning; **sklearn** = [*User Guide*](https://scikit-learn.org/stable/user_guide.html) do
scikit-learn 1.9.

### Fase 0 — Setup e benchmark de custo (já entregue)

| Conceito | Ler só isto |
|---|---|
| Só se faltar: o estado de vários qubits (o *statevector*) *(acrescentado na conferência)* | IBM, *Basics of quantum information*, lição *Multiple systems*, página [*Quantum information*](https://quantum.cloud.ibm.com/learning/en/courses/basics-of-quantum-information/multiple-systems/quantum-information): só a seção *Quantum states*, até *Tensor products of quantum state vectors*. A página não diz "2ⁿ" com essas palavras: o vetor tem uma posição por cadeia de bits (00, 01, 10, 11 com 2 qubits), e é daí que vêm as 2ⁿ amplitudes que a Seção 1 usa para medir custo e memória |
| O que o `tracemalloc` mede, e por que o RSS complementa *(acrescentado na conferência)* | Python 3.12, [`tracemalloc`](https://docs.python.org/3.12/library/tracemalloc.html): só o 1º parágrafo (*"trace memory blocks allocated by Python"*). A memória alocada fora do Python (Aer, libsvm) não aparece nele, que é o motivo de o `CostMeter` medir também o RSS |

**Coberto pela aula ou pelo notebook, sem material:** a concentração do kernel (aula da Fase 0 no `study_notes.md`,
retomada na Fase 5) e a extrapolação linear do custo pelo número de linhas (Seção 1.2).

### Fase 1 — Coleta e carregamento (já entregue)

Sem material: o contrato dos dados e o hash de integridade são procedimentos de engenharia explicados na própria
Seção 2 e na aula da Fase 1; nenhum conceito de ML ou de computação quântica é aplicado nela.

### Fase 2 — Divisão e undersampling (já entregue)

| Conceito | Ler só isto |
|---|---|
| Vazamento | sklearn, [12.2 *Data leakage*](https://scikit-learn.org/stable/common_pitfalls.html) |
| Reamostrar antes do split | imbalanced-learn, [9.1 *Data leakage*](https://imbalanced-learn.org/stable/common_pitfalls.html) |
| Undersampling distorce a probabilidade | Google *Machine Learning Crash Course*, [*Class-imbalanced datasets*](https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets): *Step 1: Downsample the majority class* e *Step 2: Upweight the downsampled class* |
| Treino, validação e teste: o papel de cada parte *(acrescentado na conferência)* | Google *Machine Learning Crash Course*, [*Datasets: Dividing the original dataset*](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets): só a seção *Training, validation, and test sets* |
| Estratificação: por que o sorteio precisa preservar a proporção da classe rara *(acrescentado na conferência)* | sklearn, [3.1.2.2 *Cross-validation iterators with stratification based on class labels*](https://scikit-learn.org/stable/modules/cross_validation.html): só o 1º parágrafo (o exemplo dele é justamente fraude). O texto fala de folds; o `stratified_three_way_split` aplica a mesma ideia às três partes |
| Só se faltar: por que não um split aleatório para dados no tempo (CC12) *(acrescentado na conferência)* | sklearn, [3.1.2.6 *Cross validation of time series data*](https://scikit-learn.org/stable/modules/cross_validation.html): só o 1º parágrafo |

**Usado na Seção 3 antes da leitura própria.** A Seção 3.2 mostra que a precisão muda com a proporção de fraudes
(`precision_at_prevalence`). Precisão e recall são lidos na Fase 6 (*Accuracy, recall, precision, and related
metrics*); para a Fase 2 basta a fórmula explicada na própria 3.2.

**Para citar no TCC:** Dal Pozzolo et al. 2015 ([ResearchGate](https://www.researchgate.net/publication/283349138_Calibrating_Probability_with_Undersampling_for_Unbalanced_Classification)):
a fórmula de correção usada na Seção 4.10.

### Fase 3a — Exploração do treino e engenharia de features (já entregue)

| Conceito | Ler só isto |
|---|---|
| Transformação log para valores com cauda longa (`log1p(Amount)`) *(acrescentado na conferência)* | Google *Machine Learning Crash Course*, [*Numerical data: Normalization*](https://developers.google.com/machine-learning/crash-course/numerical-data/normalization): só a seção *Log scaling* |
| Hora do dia em seno e cosseno *(acrescentado na conferência)* | sklearn, exemplo [*Time-related feature engineering*](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html): só os dois primeiros parágrafos da seção *Trigonometric features* (seno e cosseno com o período certo eliminam o salto entre o último e o primeiro valor do ciclo, as 23h59 e 00h00 da Seção 4.3) |
| Teste qui-quadrado de independência (a taxa de fraude difere entre os dois dias? Seção 4.3) *(acrescentado na conferência)* | Diez, Çetinkaya-Rundel e Barr, *OpenIntro Statistics*, 4ª ed. ([PDF gratuito](https://www.openintro.org/book/os/)): só a seção 6.4, *Chi-square for two-way tables* |

**Já estudado, sem material novo:** informação mútua × correlação de Pearson e média × mediana (Trabalho Humberto,
referenciados no topo do `study_notes.md`).

### Fase 3b — Scalers e codificação (já entregue)

| Conceito | Ler só isto |
|---|---|
| StandardScaler, MinMaxScaler e RobustScaler: o que cada um aprende *(acrescentado na conferência)* | sklearn, [8.3.1 *Standardization, or mean removal and variance scaling*](https://scikit-learn.org/stable/modules/preprocessing.html): a introdução, 8.3.1.1 *Scaling features to a range* e 8.3.1.3 *Scaling data with outliers*. A equivalência Standard → MinMax = MinMax não está lá: é demonstrada na Seção 4.6.2 |
| Só se faltar: PCA *(acrescentado na conferência)* | sklearn, [2.5.1.1 *Exact PCA and probabilistic interpretation*](https://scikit-learn.org/stable/modules/decomposition.html): só os dois primeiros parágrafos |
| Só se faltar: seleção das k melhores features por informação mútua *(acrescentado na conferência)* | sklearn, [1.13.2 *Univariate feature selection*](https://scikit-learn.org/stable/modules/feature_selection.html): só os parágrafos sobre `SelectKBest` e o que cita `mutual_info_classif` |
| Kernel e SVM | QML-IBM, lição [*Review of relevant machine learning methods*](https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning/classical-ml-review): seções *Support vector machines*, *Kernel methods and how quantum can play a role* e *Moving to higher dimensional spaces* |
| Largura de banda do kernel (o γ do RBF) | sklearn, [1.4.6.1 *Parameters of the RBF Kernel*](https://scikit-learn.org/stable/modules/svm.html) |
| Codificação de ângulo, de amplitude e o mapa ZZ | QML-IBM, lição [*Data encoding*](https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning/data-encoding): seções *Amplitude encoding*, *Angle encoding* e *ZZ feature map* |
| Só se faltar: produto interno entre estados | IBM, *Basics of quantum information*, [*Inner products and projections*](https://quantum.cloud.ibm.com/learning/en/courses/basics-of-quantum-information/quantum-circuits/inner-products-and-projections) |
| Alinhamento kernel-alvo (KTA): o que é e como se calcula | PennyLane, demo [*Training and evaluating quantum kernels*](https://pennylane.ai/qml/demos/tutorial_kernels_module) (Derks, Faehrmann, Gil-Fuster, Hubregtsen, Meyer e Wierichs): só a seção *Training the Quantum Embedding Kernel*, até a fórmula do KTA. Ela traz a versão **sem** centralização; a centralizada, que o projeto usa, está explicada com um exemplo calculado no `study_notes.md` (Fase 2b, "Conceito: o KTA") |

**Coberto pelo notebook, sem material:** a periodicidade dos mapas do Qiskit (α = 2, período π; Seção 4.7.1, análise
própria com testes), a constante de preenchimento da amplitude e a perda da norma (4.8, idem), e a correlação de
Spearman da 4.6.0 (definida no próprio texto). O KTA centrado está explicado com exemplo calculado no
`study_notes.md`.

**Para citar no TCC:** Havlíček et al. 2019, [arXiv:1804.11326](https://arxiv.org/abs/1804.11326) (origem do
ZZFeatureMap; dados artificiais gerados pelo próprio mapa) e Schuld 2021, [arXiv:2101.11020](https://arxiv.org/abs/2101.11020)
(seção II, *Summary of results*: modelos de QML supervisionado são métodos de kernel, CC3). Para o KTA centrado,
Cortes, Mohri & Rostamizadeh 2012, *Algorithms for Learning Kernels Based on Centered Alignment*, JMLR 13
([PDF](https://www.jmlr.org/papers/volume13/cortes12a/cortes12a.pdf)): seção 2, *Alignment Definitions* (pp. 798–800).

### Fase 4 — Treinamento (já entregue; materiais conferidos em 2026-10-20)

| Conceito | Ler só isto |
|---|---|
| AdaBoost: reponderação e classificador fraco | sklearn, [1.11.7 *AdaBoost*](https://scikit-learn.org/stable/modules/ensemble.html) |
| De onde vem o α | Schapire 2013, *Explaining AdaBoost* ([PDF](https://www.schapire.net/papers/explaining-adaboost.pdf)): só a seção 1, *Introduction* |
| O que o AdaBoost guarda de cada rodada (o que a Seção 5.5 lê) | sklearn, API do [`AdaBoostClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html): só o parâmetro `n_estimators` e os atributos `estimator_weights_` e `estimator_errors_`. **A página não documenta a parada por ε ≥ 0,5:** no `n_estimators` ela só diz *"In case of perfect fit, the learning procedure is stopped early"*. A regra que encerra o boosting quântico (o classificador da rodada com ε ≥ 0,5 é descartado, e `n_estimators` vira teto) está no código do scikit-learn (`_boost` em `sklearn/ensemble/_weight_boosting.py`) e é explicada na própria Seção 5.5 |
| Peso por amostra: o elo entre o boosting e o classificador variacional | sklearn, [Glossário: `sample_weight`](https://scikit-learn.org/stable/glossary.html#term-sample_weight): só o 1º parágrafo (peso inteiro = repetir a linha; por isso peso zero = remover a linha, que é um dos testes do `VariationalQuantumClassifier`) |
| Entropia cruzada (*log loss*), o custo do classificador fraco | Google *Machine Learning Crash Course*, [*Logistic regression: Loss and regularization*](https://developers.google.com/machine-learning/crash-course/logistic-regression/loss-regularization): só a seção *Log Loss*. A versão ponderada do projeto multiplica cada termo da soma pelo peso da linha |
| QSVM = SVM com kernel pré-calculado | sklearn, [1.4.6.2 *Custom Kernels*](https://scikit-learn.org/stable/modules/svm.html) |
| Classificador variacional | QML-IBM, lição [*QVCs and QNNs*](https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning/qvc-qnn): de *Variational quantum classifier: a first attempt* até *Testing and accuracy*. Serve para a **estrutura** (feature map + ansatz + medição + otimizador clássico); o classificador da lição é diferente do nosso (tabela abaixo) |
| Só se faltar: COBYLA, o otimizador sem gradiente | SciPy, [`minimize(method='COBYLA')`](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html): só o parágrafo de descrição e a opção `maxiter` (*"Maximum number of function evaluations"*: é contagem de avaliações do custo, e não de passos) |
| Só se faltar: a linha de base (`DummyClassifier`) | sklearn, API do [`DummyClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html): só o 1º parágrafo e a estratégia `"prior"` |

**A lição da IBM × o classificador fraco do projeto** (conferido no código da lição e em
`src/fraud_qml/quantum/variational.py`):

| | Lição da IBM | `VariationalQuantumClassifier` |
|---|---|---|
| Custo | Erro quadrático médio (MSE) | Entropia cruzada **ponderada** pelos pesos do boosting |
| Peso por linha | Não existe | Essencial: é por ele que o boosting diz onde errar custa mais |
| O que se mede | Z em **todos** os qubits ao mesmo tempo (`"Z" * num_qubits`, a paridade) | Z só no **qubit 0** |
| Da medição à classe | Corte em 0 sobre ⟨Z⟩ | P(1) = (1 − ⟨Z₀⟩)/2, corte em 0,5: é o mesmo que ⟨Z₀⟩ < 0 |
| Ansatz | Feito à mão (RY, CNOT, RX) | `RealAmplitudes` (RY e CNOT) |
| Otimizador | COBYLA, `maxiter` 100 | COBYLA, `maxiter` 60 |

**Usados na Seção 5 antes da leitura própria.** A Seção 5.4 já avalia com **PR-AUC** e **bootstrap pareado**. A
leitura da curva precisão-recall é a da Fase 6 e a do bootstrap, a da Fase 8. Para a Fase 4 basta a definição que a
própria Seção 5 dá (termos novos e 5.4); quem quiser adiantar usa as leituras daquelas fases.

**Para citar no TCC:** Wang, Wang, Qi e Dong 2024, *Supervised Learning Guarantee for Quantum AdaBoost*,
Phys. Rev. Applied 22, 054001 ([arXiv:2402.02376](https://arxiv.org/abs/2402.02376)). É a referência principal do
Quantum AdaBoost: a entropia cruzada ponderada (Eq. 7), a leitura pelo sinal de ⟨Z⟩ (Eq. 9), o α do Algoritmo 1
(α = ln((1−ε)/ε) + ln(D−1), que com D = 2 classes é a fórmula do scikit-learn) e o do Algoritmo 2
(α = ½·ln((1−ε)/ε), com a exigência ε < ½), e a regra de ficar com a iteração de menor erro (seção III.C) (WA2, WA3).
Substitui Zhu et al. 2009 como citação da fase: o Algoritmo 1 do próprio Wang já mostra que a fórmula do
scikit-learn é a do AdaBoost com duas classes.

### Fase 5 — Ajuste de hiperparâmetros (entregue em 2026-10-27; materiais conferidos no mesmo dia)

| Conceito | Ler só isto |
|---|---|
| Grid search com validação cruzada | sklearn, [3.2.1 *Exhaustive Grid Search*](https://scikit-learn.org/stable/modules/grid_search.html) |
| Por que o vencedor do grid é otimista | Raschka 2018, [arXiv:1811.12808](https://arxiv.org/abs/1811.12808): seção 3.3, *The Three-Way Holdout Method for Hyperparameter Tuning* |
| Escolher por k-fold no treino, reajustar no treino inteiro e avaliar num conjunto separado (o que a Seção 6 faz) *(acrescentado na conferência da Fase 5)* | Raschka 2018: seção 3.7, *Model Selection via k-fold Cross-Validation* (os 5 passos da Figura 16) |
| Grid em escala logarítmica para C e γ | Já visto: sklearn 1.4.6.1 (Fase 3b), *"C and gamma spaced exponentially far apart"* |
| Teto de rodadas lido rodada a rodada; taxa de aprendizado × número de rodadas *(acrescentado na conferência da Fase 5)* | sklearn, API do [`AdaBoostClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html): o parâmetro `learning_rate` (*"There is a trade-off between the learning_rate and n_estimators parameters"*) e o método `staged_decision_function` |
| Concentração do kernel | Já visto: aula da Fase 0, no `study_notes.md` |

**Coberto pelo notebook, sem material:** o grid em dois níveis (a entrada reaproveitada entre os hiperparâmetros do
modelo) e a projeção do piloto (que não é garantidamente conservadora, NC16) são detalhes de implementação explicados na abertura da Seção 6 e na
6.1.1; a validação cruzada aninhada é uma alternativa descartada, explicada na abertura (quem quiser: Raschka 2018,
seção 4.14). Undersampling dentro de cada fold: leitura da Fase 2 (imbalanced-learn 9.1).

**Para citar no TCC:** Thanasilp et al. 2024, [arXiv:2208.11060](https://arxiv.org/abs/2208.11060): seção II.C,
as fontes de concentração (CC5).

### Fase 6 — Avaliação

| Conceito | Ler só isto |
|---|---|
| Limiar, matriz de confusão, precisão e recall | Google *Machine Learning Crash Course*: [*Thresholds and the confusion matrix*](https://developers.google.com/machine-learning/crash-course/classification/thresholding) e [*Accuracy, recall, precision, and related metrics*](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall) |
| Curva ROC | Google *Machine Learning Crash Course*: [*ROC and AUC*](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc): as seções *Receiver-operating characteristic curve (ROC)* e *Area under the curve (AUC)* |
| Curva precisão-recall (a PR-AUC) | A mesma página do Google: a seção *(Optional, advanced) Precision-recall curve* (conferido em 2026-10-27; troca a parte da curva PR da sklearn 3.4.4.9, que só dá a fórmula) |
| MCC | sklearn, [3.4.4.13 *Matthews correlation coefficient*](https://scikit-learn.org/stable/modules/model_evaluation.html) |

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
| Cortes, Mohri & Rostamizadeh 2012 | Definição do alinhamento kernel-alvo centrado (KTA) | 3b |
| Wang et al. 2024 | Quantum AdaBoost: entropia cruzada ponderada, os dois α e a regra de parada (WA2, WA3) | 4 |
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

- **Quantum AdaBoost é caro.** *(Deixou de ser em 2026-10-13, Fase 4: com os estados em lote e o ansatz como matriz, um ajuste leva cerca de 2,5 s.)* Texto original: treinar VQC × rodadas × folds pode levar horas. Mitigação: benchmark na Seção 1, grid pequeno, COBYLA com poucas iterações (o classificador precisa ser *fraco*), cache e `staged_decision_function`.
- **O boosting quântico para cedo e é instável entre sementes** (Fase 4, ponto ⑤): o resultado do par AdaBoost × Quantum AdaBoost pode depender mais da regra de parada do que dos hiperparâmetros.
- **Wheels do `qiskit-aer` para Windows/py3.12:** conferidas no dry-run da Fase 0. O plano B é outra versão do Python via `uv`.
- **`SVC` com kernel callable + `clone`:** coberto por testes. O plano B é `kernel="precomputed"` com o cálculo do kernel dentro do wrapper.
- **Expectativa honesta de resultado:** com ~568 amostras de treino e a proporção real no teste, a precisão vai ser baixa para todos os modelos, bem abaixo dos 0,818 de PR-AUC do Random Forest com a base inteira do trabalho anterior. Isso é resultado, não defeito, e vira discussão sobre o custo do undersampling imposto pelo QML.
- **Pouca amostra no teste (95 fraudes):** justamente o que as sementes, o bootstrap e o McNemar tratam.

## Verificação

- `.venv\Scripts\python -m pytest` todo verde a cada fase.
- `jupyter nbconvert --to notebook --execute notebooks/tcc_ml_vs_qml.ipynb` sem erros.
- Os `assert`s do notebook passam: splits disjuntos, proporção preservada e undersampling ausente em validação/teste. O teste é usado uma única vez (Seção 9), e a Seção 10 usa só modelos já finais.
- `artifacts/results/` contém, para cada modelo: hiperparâmetros, métricas, custo, versões e commit.
