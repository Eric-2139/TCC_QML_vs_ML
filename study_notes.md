# Caderno de estudo — TCC ML x QML

Registro das aulas de cada fase: perguntas de fixação, suas respostas e as correções.

**Níveis:**
- 🟢 **base:** obrigatória; responde-se relendo a aula.
- 🟡 **intermediária:** aplicar a ideia numa situação nova.
- 🔴 **desafio:** opcional; conecta com fases futuras.

**Avaliação:** ✅ correta · 🔸 parcialmente correta · ❌ a revisar.

As respostas podem vir em tópicos soltos: pôr a ideia em palavras também faz parte do aprendizado.

**Numeração das fases.** Em 2026-10-07 as fases foram renumeradas para valer sempre "Fase N = Seção N+1": a
antiga Fase 3 (divisão) é hoje a **Fase 2**, e as antigas 2a e 2b (exploração) são hoje **3a e 3b**. Os registros
abaixo, escritos antes disso, mantêm os nomes antigos.

**Numeração das seções do notebook.** Em 2026-10-07 o split passou para antes da exploração. As aulas e respostas
abaixo foram escritas com a numeração da época; a correspondência é: antiga 3.1 → 3.1; 3.2 a 3.4 → 4.2, 4.3 e 4.5;
3.5 → 4.4; 3.6 a 3.11 → 4.6 a 4.11; antiga 4 → 3 (split) e 4.10 (undersampling). Os números que mudaram estão na
seção "Reorganização", no fim deste caderno.

Conceitos já estudados no Trabalho Humberto (`Machine Learning/Trabalho Humberto/study_notes.md`) são referenciados,
não reensinados: pipeline do imblearn, vazamento com balanceamento antes do split, média x mediana,
informação mútua x Pearson, papéis de validação x teste.

---

## Fase 0 — Setup e benchmark de custo

**Arquivos:** `pyproject.toml`, `src/fraud_qml/{config,environment,profiling}.py`, `tests/`, Seção 1 do notebook.

### Perguntas

1. 🟢 O benchmark mostrou que o QSVM é barato para **treinar** (≈ 3 s) e caro para **prever** 56 mil linhas
   (≈ 5 min). De onde vem essa assimetria? Pense no que é feito para cada amostra.
2. 🟢 Por que medimos o tempo com `trace_python_memory=False`, e o que o `tracemalloc` deixa de enxergar mesmo
   quando está ligado?
3. 🟡 Um colega propõe usar 12 qubits na codificação de ângulo, "para caber mais features". Usando a fórmula da
   memória do statevector (16 · 2ⁿ bytes) e o fato de o custo por amostra crescer com o circuito, o que muda no
   experimento?
4. 🔴 O classificador fraco do Quantum AdaBoost "deve ser fraco". Se treinássemos cada circuito até a convergência
   total, o que aconteceria com o boosting, além do tempo de treino?

### Respostas

**1. 🔸 Parcialmente correta.** O "um circuito por amostra" está certo; a conclusão sobre validação cruzada e a
conta do grid, não.
- **A assimetria é só o número de amostras.** O treino simula 568 circuitos (1 por amostra de treino) e a predição
  simula 56.746 (1 por linha a classificar), ou seja, 100× mais. A matriz de similaridade 568 × 568 (≈ 323 mil
  produtos internos) custa microssegundos: produto interno é barato, montar e simular o circuito é caro.
- **Cada combinação de hiperparâmetros não é "uma amostra".** Cada combinação é um treino completo, e cada treino
  simula um circuito por amostra. A conta é: combinações × folds × (amostras do fold).
  - Com 5 folds são ≈ 568 circuitos por combinação (≈ 2,5 s).
  - 100 combinações ≈ 4 min. Não vai "para o teto".
  - Combinações que só mudam o `C` do SVM reaproveitam os mesmos statevectors, porque o circuito depende só dos
    dados e do feature map.
- **A validação cruzada é MAIS importante com poucas amostras, não menos.**
  - Com um único split de validação de ~114 amostras, cada amostra vale ~0,9 ponto percentual de acurácia. Duas ou
    três amostras "de sorte" já trocam o vencedor do grid.
  - A CV faz 5 medições e tira a média, e o desvio entre os folds mostra quanto confiar nela.
  - Ela não serve para "testar splits diferentes": serve para medir cada combinação de forma mais confiável.
  - Ligação com o Trabalho Humberto (pergunta pendente da Fase 5): lá o desvio entre folds (0,063) era maior que a
    diferença entre as combinações do grid (0,029).

**2. 🔸 Quase lá: trocou tempo por memória na segunda parte.**
- A primeira parte está certa: o `tracemalloc` intercepta cada alocação e deixa o código mais lento.
- Na segunda parte, o que ele deixa de enxergar é a **memória** alocada fora do Python (Aer em C++, libsvm em C),
  não o tempo. O tempo é medido pelo relógio (`perf_counter`) e inclui tudo, Python ou não.
- Detalhe: "escrito em C" não basta para ficar invisível. O numpy também é C, mas avisa suas alocações ao
  `tracemalloc` (o teste `test_measures_python_and_native_memory` confirma isso). O Aer e a libsvm não avisam.

**3. 🔸 A parte do ângulo está certa; a expectativa de melhora, não.**
- Certo: na codificação de ângulo, 12 qubits = 12 features. A fórmula 16·2ⁿ dá 64 KiB, ainda irrelevante.
- O custo por amostra cresce com o circuito: 1,7 ms com 4 qubits, 8,1 ms com 12 e 28,6 ms com 14 (medido).
- **"Mais qubits = melhor desempenho" é o equívoco central.** Medição com o ZZFeatureMap e dados aleatórios em
  [0, π]:

  | Qubits | Similaridade média entre amostras diferentes |
  |---|---|
  | 2 | 0,32 |
  | 4 | 0,079 |
  | 6 | 0,019 |
  | 8 | 0,0058 |
  | 10 | 0,0013 |
  | 12 | 0,0004 |
  | 14 | 0,0001 |

  - O valor cai aproximadamente como **1/2ⁿ**: o espaço de estados dobra a cada qubit, e dois pontos quaisquer
    ficam "quase ortogonais".
  - A matriz de similaridade vira quase a identidade: cada amostra só se parece consigo mesma. O SVM então
    **decora** o treino e não generaliza (overfitting).
  - Em hardware real piora: seriam necessárias ~10⁴ medições (*shots*) para distinguir 0,0001 de zero.
  - Isso se chama **concentração exponencial do kernel** (Thanasilp et al., 2022). No circuito variacional, o
    fenômeno análogo é o *barren plateau*, que o seu TCC já cita (seção 2.2.4.2).
  - Por isso o número de qubits e o "range" dos ângulos são **hiperparâmetros**, não "quanto mais, melhor".

**Dúvida de acompanhamento:** "aumentar os qubits aumenta a granularização e deixa as amostras mais similares?"
- **Conclusão invertida.** As amostras ficam **menos** similares.
- **São as features que se dividem entre os qubits, não as amostras.** Cada amostra ocupa todos os qubits como um
  único estado.
- **Sem emaranhamento, a similaridade total é o produto das similaridades por qubit** (cada uma entre 0 e 1).
  Com 0,7 por qubit:
  - 2 qubits: 0,49
  - 5 qubits: 0,17
  - 10 qubits: 0,028
  - 14 qubits: 0,007
- **Analogia:** duas pessoas "iguais" só se concordarem em todas as perguntas de um questionário. Com 30
  perguntas, isso quase nunca acontece.
- **A "granularização" acerta num ponto:** a resolução fica mais fina. Só que isso separa os pontos, em vez de
  aproximá-los.
- **Consequência:** a amostra nova não se parece com nenhuma do treino, e o modelo decora o treino (overfitting).
- 🟢 Pergunta de verificação: com 5 qubits e similaridade 0,9 por qubit, qual é a total? E com 20 qubits?
  - ✅ Resposta: 0,59 e 0,12.
  - Ajuste de vocabulário: o que cai é a semelhança **entre as amostras**. As features continuam 0,9 parecidas
    uma a uma.

**Modelo mental: dataset como matriz.**
- **Linhas = amostras** (transações, 284.807). **Colunas = features** (30).
- **`Class` é o rótulo (`y`), não uma feature (`X`).**
- Cada linha tem **um valor de cada feature**; a linha inteira é um exemplo.
- **O que mexe em linhas:** duplicatas, split, undersampling.
- **O que mexe em colunas:** engenharia de features, PCA/seleção.
- Na simulação: **1 circuito por linha** (o custo cresce com as linhas) e **1 qubit por coluna** na codificação de
  ângulo (os qubits crescem com as colunas).
- **Duas leituras da mesma matriz:** uma amostra tem um valor de cada feature; uma feature tem um valor de cada
  amostra.
  - A feature é a **variável** (ex.: `Amount`); a amostra é o **exemplo** (uma transação).
  - O valor (a célula) pertence às duas ao mesmo tempo.
- **Os qubits guardam uma amostra por vez.** Cada linha é uma nova simulação do mesmo circuito, com novos ângulos.
  - Ângulo: 5 qubits = 5 features de 1 amostra.
  - Amplitude: 5 qubits = 2⁵ = 32 features de 1 amostra.

**Pergunta extra: por que não usar todos os qubits que a máquina suporta (~29–30)?**
- **Custo:** 30 qubits cabem na RAM para **um** estado (16 GiB). O experimento precisa de dezenas de milhares.
  - Cada porta percorre 2³⁰ ≈ 1 bilhão de amplitudes, então cada amostra levaria minutos.
  - × 56 mil linhas = meses.
  - Para o kernel, seria preciso guardar os estados das 568 amostras de treino ao mesmo tempo (568 × 16 GiB).
- **Estatística:** pela tabela acima, mais qubits concentram o kernel e pioram a generalização.
- **Dados:** temos 31 features. No ângulo, 30 qubits exigiriam 30 features, quase todas com pouca informação. Na
  amplitude, 31 features já cabem em 5 qubits.
- **Realismo:** em hardware NISQ, mais qubits e portas significam mais ruído (a Seção 10 vai medir isso).
- **O que usamos da máquina:** os 12 núcleos (`N_JOBS`), que rodam em paralelo *muitas* simulações pequenas
  (combinações do grid, folds, amostras), em vez de uma simulação gigante.

**4. 🔸 Overfitting está certo; o mecanismo, não.** Os classificadores não ficam "todos iguais". O que acontece é:
- O primeiro classificador, forte, acerta quase todo o treino, então o erro ponderado ε ≈ 0.
- O peso dele no voto é α = ½·ln((1−ε)/ε), que explode:

  | ε | α |
  |---|---|
  | 0,3 | 0,42 |
  | 0,01 | 2,30 |
  | 0,001 | 3,45 |

  Ele passa a dominar a votação sozinho.
- A reponderação joga todo o peso nas poucas amostras que ele errou, que tendem a ser ruído ou casos atípicos. Os
  próximos classificadores passam a perseguir esse ruído.
- Resultado: um "comitê" que é, na prática, um único modelo decorado, acrescido de modelos especializados em ruído.
  Isso gera overfitting e mau desempenho no teste.
- No limite ε = 0, o scikit-learn interrompe o boosting: não há erro para corrigir.
- A teoria do boosting pede classificadores **pouco melhores que o acaso** (ε < 0,5), com erros diferentes entre
  si. O ganho vem de combinar erros diversos.

---

## Fase 1 — Coleta e carregamento

**Arquivos:** `src/fraud_qml/data.py`, `tests/test_data.py`, Seção 2 do notebook.

### Perguntas

1. 🟢 Por que o hash de integridade é calculado sobre o CSV **descompactado**, e não sobre o `.zip`?
2. 🟢 Por que o contrato verifica **todas** as propriedades e lista as violações juntas, em vez de parar na
   primeira que falha?
3. 🟡 O desvio padrão de `Time` é ≈ 47.488 e o de `V14` é ≈ 0,96. Num modelo que decide por distância (quem está
   "perto" de quem), qual das duas features decidiria a distância entre duas transações se nada fosse escalonado?
   E por que isso atrapalha a detecção de fraude?
4. 🔴 Com o undersampling, o treino usará ~284 das ~170 mil transações legítimas. Que risco isso cria? Pense em
   tipos de transação legítima que são raros.

### Respostas

**Padrão observado nesta fase:** as respostas disseram *para que* algo serve, e as perguntas pediam *por que desse
jeito*. Treinar essa distinção ajuda na defesa do TCC.

**1. 🔸** Resposta: "para garantir que o CSV é o mesmo do Kaggle". Isso é o objetivo do hash; faltou o porquê do CSV.
- O `.zip` guarda metadados (data de modificação, nível de compressão, programa usado). Recompactar o mesmo CSV
  muda os bytes do `.zip` e, com eles, o hash.
- O hash do CSV descompactado mede só o conteúdo.
- O teste `test_hash_ignores_zip_compression` prova isso.

**2. 🔸** Resposta: "para verificar se o arquivo tem o formato que o código supõe". Isso é o que o contrato faz;
faltou o porquê de listar tudo junto.
- **Tempo:** evita o ciclo corrigir → rodar → achar a próxima falha.
- **Diagnóstico:** as falhas juntas revelam a causa comum. "Colunas erradas + 150 mil linhas + 3.000 fraudes" não
  são três bugs, são **outro dataset**.

**3. 🔸** Resposta: "o Time, pois dominaria as distâncias, mas o que se mede são as fraudes". Acertou a feature;
misturou feature com alvo (a fraude é o rótulo `Class`).
- Exemplo com três transações:

  | Transação | Time | V14 | Tipo |
  |---|---|---|---|
  | A | 1.000 | −9 | parece fraude |
  | B | 1.000 | 0 | legítima |
  | C | 5.000 | −9 | parece fraude |

  - d(A, B) ≈ 9
  - d(A, C) ≈ 4.000
- A vira "vizinha" da legítima B, e não de C, que tem o mesmo padrão. A distância passa a medir "mesmo momento?" e
  ignora o `V14`, onde está o sinal.
- O `Time` tem pouca informação sobre fraude: 18º de 30 na informação mútua, no Trabalho Humberto.

**4. ❌** Resposta: "poucas amostras, risco de underfitting".
- **Underfitting** é o modelo ser simples demais para o padrão. Poucos dados com modelos flexíveis (RBF, kernel
  quântico) geram o contrário: **alta variância**, isto é, o modelo decora as amostras e muda muito conforme o
  sorteio.
- **O risco pedido, legítimas raras:** um tipo que representa 0,5% das legítimas tem chance de
  0,995²⁸⁴ ≈ 24% de não aparecer entre as 284 sorteadas.
  - O modelo aprende "legítima = típica" e "fraude = fora do comum".
  - No teste, os ~280 casos desse tipo viram alarme falso.
  - Resultado: **precisão baixa** na proporção real.
- **Mitigações já no plano:** 5 sementes de undersampling e limiar escolhido na validação real
  (Dal Pozzolo et al., 2015).
- **Reformulação ✅:** "Poucas amostras fazem o modelo ver poucas coisas, e ao ver poucas coisas ele tende a achar
  que o mundo se baseia nelas. Em ML: poucos dados de treino, alta probabilidade de overfitting, generalização ruim
  e desempenho ruim no teste."

---

## Fase 2a — EDA, duplicatas e engenharia de features

**Arquivos:**
- `data.py`: `summarize_duplicates`, `remove_exact_duplicates`
- `features.py`
- `plotting.py`
- `tests/test_features.py`
- Seções 3.1 (duplicatas) e 4.1 a 4.5 do notebook (na época: 3.1 a 3.5)

### Perguntas

1. 🟢 Por que as duplicatas são removidas **antes** do split, e não depois?
2. 🟢 No gráfico por classe, cortei o eixo pelos percentis 0,5 e 99,5 da base inteira, e as fraudes "sumiram" numa
   barra na borda. Por quê?
3. 🟡 A taxa de fraude às 2h é ~9× a média, mas `hour_sin` e `hour_cos` ficaram em 23º e 25º na informação mútua.
   Como as duas coisas podem ser verdade ao mesmo tempo?
4. 🔴 A codificação cíclica da hora usa **duas** colunas (seno e cosseno). O que daria errado usando só o seno?
   Dica: calcule sin(2π·3/24) e sin(2π·9/24).

### Respostas

**1. ✅ (com ajuste de vocabulário).**
- Resposta: "as cópias podem cair uma no treino e outra no teste; a do teste é acertada porque o modelo a decorou —
  um tipo de overfitting". O raciocínio está certo; o nome, não.
- O nome certo é **vazamento (*data leakage*)**. **Overfitting** é o modelo decorar o treino e falhar em dados
  novos, um defeito do **modelo**. **Vazamento** é o teste deixar de ser "dado novo", um defeito da **avaliação**.
- As duas coisas se combinam de um jeito perigoso: o vazamento **esconde** o overfitting, porque o modelo que
  decorou acerta justamente as cópias do teste, e a métrica sai inflada.
- Por que antes do split: depois dele, as cópias já estariam espalhadas entre os conjuntos. Remover repetições
  dentro de cada conjunto não desfaz os pares treino–teste.

**2. 🔸 Acertou que as fraudes ficam na cauda; faltou o mecanismo do corte.**
- O percentil 0,5 da base inteira é calculado sobre 283 mil linhas, 99,83% legítimas. Na prática, é um percentil
  **das legítimas**.
- Cortar abaixo dele remove ~1.400 linhas (0,5%). As fraudes são só 473 e quase todas estão nessa ponta, então
  cabem inteiras na fatia cortada.
- O `clip` empurrou todas elas para o valor do corte e criou a barra falsa.
- Correção: percentis **por classe**.
- Lição geral: com classes desbalanceadas, estatística "da base inteira" = estatística da classe majoritária.

**3. ❌ Sem resposta (faltou um gráfico, que foi acrescentado na Seção 3.3).** A resposta, a partir da base:
- A informação mútua mede quanto saber a feature reduz a incerteza sobre o rótulo, **em média sobre todas as
  transações**.
- O pico da madrugada afeta poucas transações: às 2h são 3.308 de 283 mil (1,2%). Para os outros ~99%, saber a
  hora quase não muda a chance de fraude.
- Mesmo às 2h, 98,55% das transações são legítimas. A taxa sobe de 0,17% para 1,45% (9 vezes, forte em termos
  relativos), mas a incerteza "fraude ou não?" continua quase a mesma.
- Compare com o `V14`: valores abaixo de −5 são quase só fraude. Ali saber o valor resolve a dúvida.
- **Resumo:** "9 vezes mais" impressiona em termos relativos, mas vale para pouca gente e deixa a resposta quase
  igual. A MI mede redução de incerteza, não aumento relativo de risco.
- Sobre "por que o seno ficou 2 posições acima do cosseno": 0,0005 contra 0,0004 está **dentro do ruído** do
  estimador de MI (ele é aproximado e aleatório). Não é uma diferença real.

**4. 🔸 A analogia do relógio é excelente; a descrição do problema é a da hora crua, não a do seno.**
- O que você descreveu (0h e 23h distantes) é o problema da hora **crua** (0 a 23), e é por isso que ela não é
  usada.
- Só com o seno, 23h e 0h ficam **próximas** (−0,26 e 0). O problema do seno sozinho é outro: **horas diferentes
  colidem**. 3h e 9h dão o mesmo seno (0,71), e 0h e 12h também (0).
- Com o cosseno, cada hora ganha um par (x, y) único. Um ponto no plano precisa de duas coordenadas, exatamente
  como o ponteiro de um relógio.
- Certo na sua resposta: as horas ficam igualmente espaçadas no círculo, inclusive 23h→0h.

---

## Fase 2b — Scalers, codificação, seleção de features e undersampling

**Arquivos:**
- `src/fraud_qml/preprocessing.py`
- `src/fraud_qml/quantum/{encodings,kernels}.py`
- `tests/test_preprocessing.py`, `tests/test_quantum_kernels.py`
- Seções 4.6 a 4.11 do notebook (na época: 3.6 a 3.11)

### Perguntas

1. 🟢 Por que um StandardScaler seguido de MinMaxScaler dá exatamente o mesmo resultado que o MinMaxScaler sozinho?
   Use o exemplo da coluna [1, 2, 3].
2. 🟢 Na varredura da faixa de ângulos, o que acontece com a similaridade entre linhas diferentes quando a faixa é
   muito **estreita**? E quando é muito **larga**? Por que nenhum dos dois extremos é bom?
3. 🟡 Completar os vetores com **zero** na codificação de amplitude derrubou o KTA para 0,07. Explique por que, usando
   os vetores x = (1, 2) e 2x = (2, 4).
4. 🔴 A informação mútua escolheu 5 features com correlação média de 0,77 entre si nas fraudes. Por que isso é um
   problema específico de ter **poucos qubits**? O PCA sofre do mesmo problema?

### ⚠️ Correções da aula (auditoria de 2026-09-30)

A primeira auditoria independente encontrou erros **meus** na aula desta fase. O que muda no que você estudou:

| Eu tinha dito | O correto |
|---|---|
| Os 7 componentes do PCA do El Alami são Time, Amount, V1, V2, V4, V3, V6, e **nenhuma** feature de fraude | No procedimento **deles** (984 linhas balanceadas), só os dois primeiros são Time e Amount. Do 3º ao 7º são misturas que **incluem** V17, V14, V3 e V10. A crítica certa: 2 dos 7 qubits são gastos com escala, não com informação. |
| Standard e Robust divergem "até π" com PCA no meio | Até **π/2**, a largura inteira da faixa padrão; parte disso é só troca de sinal dos componentes |
| "A variância de Time é ~50.000× a das V" | 50.000× é a razão de **desvios padrão**; a de variâncias (desvio²) é da ordem de 10⁹ |
| "Nas faixas válidas, o ZZ + PCA (0,56) supera o Z + PCA (0,50)" | Com 5 subamostras: Z + PCA 0,45 ± 0,03 contra ZZ + PCA 0,41 ± 0,05. **Sem diferença conclusiva.** O número de uma só subamostra enganava. |
| "Até π/2, mais distante = menos parecido" nos dois mapas | Só no **mapa Z**. No ZZ, a similaridade chega ao mínimo em ≈ 0,63 rad e volta a subir |

A pergunta 2 desta fase (faixa estreita × larga) continua certa. As respostas 1, 3 e 4 também não mudam.

### Dúvida: "o MinMax não coloca os valores entre 0 e 1?"

- **[0, 1] é só o padrão.** O intervalo é configurável (`feature_range`); aqui usamos (0, π).
- **Fórmula em dois passos:**
  - Passo 1 (o MinMax conhecido): x' = (x − mín)/(máx − mín), resultado em [0, 1].
  - Passo 2 (esticar): x'' = a + x'·(b − a), resultado em [a, b]. Para [0, π], é só multiplicar por π.
- **Por que π:** o valor vira ângulo de rotação do qubit. [0, 1] usa só 1 rad (~57°); [0, π] usa meia volta. O
  intervalo é a "largura de banda" (Seção 3.7).
- **O achatamento é consequência da fórmula, não um passo extra:** um único extremo estica (máx − mín) e espreme o
  resto.
  - Exemplo: [0, 1, 2, 3, 100] → [0; 0,01; 0,02; 0,03; 1]. Quatro valores em 3% do intervalo.
  - No `V14`, 90% das legítimas ficam em 12% do intervalo.
  - Multiplicar por π não muda essa proporção, só o tamanho da fatia em radianos.

### Respostas

**1. ✅ (com complemento).**
- Resposta: "o MinMax transforma o mínimo e o máximo nos limites do intervalo; esticar antes não muda nada".
  Correto.
- Faltou o exemplo pedido: [1, 2, 3] → Standard → [−1,22; 0; 1,22] → MinMax → [0; 0,5; 1]. O MinMax direto também
  dá [0; 0,5; 1].
- Álgebra: se x vira a·x + b (com a > 0), então (x − mín)/(máx − mín) não muda. O "a" aparece no numerador e no
  denominador e se cancela; o "b" some na subtração.
- Ajuste de palavra: não é que os dados "continuam achatados". É que a **posição relativa** de cada valor entre o
  mínimo e o máximo não muda com um esticamento.

**2. ✅**
- Estreita: tudo parecido, o kernel não separa. Larga: tudo diferente, o modelo decora. O objetivo é o ponto
  intermediário.
- Complemento: o pico do KTA é só um indicador desse ponto; a escolha final é feita por validação cruzada.

**3. (pedido de explicação aprofundada)**
- **O que a normalização faz:** divide o vetor pelo próprio comprimento, então só a **direção** sobrevive.
  - x = (1, 2) tem comprimento √5 e vira (0,447; 0,894).
  - 2x = (2, 4) tem comprimento √20 e vira (0,447; 0,894).
  - Os dois apontam para o mesmo lado e viram o **mesmo estado**: similaridade = 1.
- **Por que a norma é sinal de fraude:** depois de padronizar, a norma é a distância ao centro dos dados, isto é, à
  "transação típica". Exemplo com (V14, V17):
  - legítima (−0,5; 0,3), perto do centro;
  - fraude (−5; 3), longe do centro.
  - A fraude é exatamente 10× a legítima: mesma direção, norma diferente. Normalizada, fica **idêntica** à
    legítima (similaridade 1,000).
- **Completar com zero não ajuda:** (1, 2, 0) e (2, 4, 0) continuam na mesma direção. O zero não acrescenta nada.
- **Completar com uma constante c resolve.** Pense em (x, c) como "andar |x| no chão e subir c": o ângulo de
  subida depende de quanto se andou.
  - Vetor curto (legítima): subida íngreme, a constante pesa muito.
  - Vetor longo (fraude): subida suave, a constante pesa pouco.
  - O "peso" da constante depois da normalização, c/√(|x|² + c²), **registra a norma**.

  | c | similaridade entre x e 2x |
  |---|---|
  | 0 | 1,000 |
  | 1 | 0,960 |
  | 3 | 0,889 |
  | 30 | 0,995 |

  - No exemplo legítima × fraude: c = 0 dá 1,000 (indistinguíveis); c = 3 dá 0,383.
- **Por que c = 30 volta a ser ruim:** com c muito maior que as normas, todos os vetores apontam quase só para o
  eixo da constante e tudo volta a parecer igual. A constante age como uma **largura de banda**, igual à faixa dos
  ângulos na codificação de ângulo. É por isso que ela é hiperparâmetro (1 e 3 no grid).

**4. 🔸 Desperdício de qubits: certo. Sobre o PCA: meio certo.**
- Certo: o PCA não sofre de **redundância**. Seus componentes são descorrelacionados por construção, e a variação
  comum das 5 features é concentrada num componente só (o 1º, com 37% da variância).
- Ajuste: "5 viram 1" é aproximado. Com correlação de 0,77, e não 1, cada feature ainda tem uma parte própria, que
  vai para outros componentes.
- **Problema diferente que o PCA tem:** ele escolhe por **variância**, não por relação com a fraude. Pode gastar
  qubits em direções de muita variação e pouca informação útil; é o caso extremo do El Alami, em que a variância era
  o `Time`.
- **Resumo:** informação mútua → relevante, mas redundante. PCA → sem redundância, mas não necessariamente
  relevante. Por isso os dois competem no grid.

### Discussão (2026-10-07): como a codificação afeta cada algoritmo

**Sua proposta:** testar também o *phase encoding* no grid. **Resultado:** ele já estava lá, com outro nome.

**1. O que cada codificação é (esfera de Bloch)**

| Codificação | Portas | Por onde o estado anda | Medindo direto (sem mais portas) |
|---|---|---|---|
| Ângulo | RY(x) | Meridiano: do polo norte em direção ao sul | P(0) = cos²(x/2): a feature aparece |
| Fase | H, depois P(x) | Equador: gira em torno de Z | P(0) = 0,500 para qualquer x: a feature não aparece |
| Mapa Z do Qiskit | H, P(2x), repetido `reps` vezes | Com `reps=1`, é a codificação de fase | Igual à fase |
| Mapa ZZ | Mapa Z + portas entre pares de qubits | Fase com emaranhamento | Igual à fase |

O que o projeto vinha chamando de "codificação de ângulo" (mapas Z e ZZ) é, a rigor, **codificação de fase** (CC14).

**2. Por que o QSVM não liga para a diferença**
- O SVM com kernel resolve a **forma dual**: os dados só entram por K(x, x′), a similaridade entre pares.
- O kernel de fidelidade só depende do ângulo entre os dois estados na esfera. Andar Δ pelo equador ou Δ por um
  meridiano dá a mesma similaridade: cos²(Δ) por qubit (com o fator 2 do Qiskit).
- Medido: kernel do mapa Z (`reps=1`) × fase × RY, diferenças de no máximo 1,8 × 10⁻¹⁵. Há três testes.
- Não é preciso a segunda Hadamard: o circuito do kernel, U†(x′)U(x), já faz a interferência.
- O que muda o kernel é o `reps`: Z com `reps=2` difere de `reps=1` em até 0,23. Por isso `reps` ∈ {1, 2} entrou
  no grid dos dois mapas.

**3. Por que o Quantum AdaBoost pode ligar**
- Ali não há kernel: o classificador fraco **mede** ⟨Z⟩ depois do ansatz (`RealAmplitudes`, só rotações em Y e CNOTs).
- Num classificador de 1 qubit (codificação + RY(θ)):
  - com **fase**, ⟨Z⟩ = −sen(θ)·cos(2x): o θ só muda a amplitude; o ponto em que ⟨Z⟩ troca de sinal fica preso em
    x = 0,79 para qualquer θ;
  - com **RY**, ⟨Z⟩ = cos(2x + θ): o θ desloca o ponto de corte (0,79 → 0,26 → 0,00). O modelo escolhe onde cortar.
- **Ainda é hipótese para o nosso caso:** medido com 1 qubit e 1 camada. Com 5 qubits, CNOTs e `reps=2` (a segunda
  camada de Hadamard traz parte da fase para o eixo medido), a limitação pode diminuir. A Fase 4 mede isso.

**4. Avaliação do seu raciocínio**
- ✅ "O phase encoding é a base dos feature maps, e as portas em Z se combinam com o emaranhamento ZZ": correto, é
  como o curso da IBM descreve o mapa Z.
- ✅ "No equador a medição dá 50% independente da feature, e ela depende de interferência para aparecer": correto e
  confirmado (P(0) = 0,500).
- ✅ "No QSVM o cálculo é pela forma dual, então fase ou ângulo pouco importa": correto, com a ressalva de que vale
  para uma camada (`reps=1`).
- 🔸 "Precisa de uma Hadamard para o estado ir do eixo Y para o eixo Z": o eixo é o **X**. A primeira H leva o estado
  do polo (eixo Z) ao equador no eixo X; a segunda H troca X por Z de volta, e é isso que torna a fase mensurável.
- ✅ Descartar o *dense angle encoding* (2 features por qubit quebra o orçamento de 5 componentes em 5 qubits) e o
  *Pauli feature map* (escolhas demais; o caso Z + ZZ é o próprio ZZFeatureMap).

**5. O que mudou no projeto:** nomenclatura corrigida (PLAN, notebook 4.6, docstring), `reps` ∈ {1, 2} no grid dos
mapas Z e ZZ, e um diagnóstico fase × RY no classificador fraco previsto para a Fase 4.

### Conceito: o KTA (alinhamento kernel-alvo) e como se calcula

Material: PennyLane, *Training and evaluating quantum kernels*, seção *Training the Quantum Embedding Kernel*
(versão sem centralização). A versão centrada, que o projeto usa, é de Cortes, Mohri & Rostamizadeh (2012, seção 2).

**A ideia.** Um kernel é uma tabela de similaridades entre todos os pares de transações. O KTA pergunta: essa tabela
concorda com os rótulos? Para responder, compara-se com uma tabela "ideal", que diz +1 para pares da mesma classe e
−1 para pares de classes diferentes. Com rótulos y = (+1, +1, −1, −1) (duas fraudes e duas legítimas), a tabela ideal é
yyᵀ (cada célula é yᵢ · yⱼ):

|  | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | +1 | +1 | −1 | −1 |
| 2 | +1 | +1 | −1 | −1 |
| 3 | −1 | −1 | +1 | +1 |
| 4 | −1 | −1 | +1 | +1 |

**A conta (versão sem centralização).** O KTA é o **cosseno do ângulo** entre as duas tabelas, vistas como vetores:

KTA = Σᵢⱼ Kᵢⱼ · yᵢ · yⱼ / (‖K‖ · ‖yyᵀ‖)

- No numerador, cada similaridade entra com sinal + se o par é da mesma classe e − se não é. Um bom kernel tem
  similaridades altas nos pares +1 e baixas nos pares −1, então a soma fica grande.
- No denominador, as normas (a raiz da soma dos quadrados de todas as células) só servem para o resultado ficar entre
  −1 e 1, como uma correlação.

**Por que o projeto centraliza.** Três kernels de exemplo para as mesmas 4 transações (calculados em Python):

| Kernel | Similaridade, mesma classe | Similaridade, classes diferentes | KTA sem centralizar | KTA centrado |
|---|---|---|---|---|
| A: separa, mas tudo é parecido | 0,95 | 0,85 | 0,068 | **0,962** |
| B: separa e diferencia | 0,9 | 0,1 | 0,628 | **0,997** |
| C: decora (identidade: cada uma só se parece consigo mesma) | 0 | 0 | 0,500 | 0,577 |

- O kernel **A** separa as classes perfeitamente: pares da mesma classe são sempre mais parecidos (0,95) que pares de
  classes diferentes (0,85). Só que tudo é parecido. Sem centralizar, o KTA dele é 0,068: no numerador, as
  similaridades altas dos pares −1 cancelam as dos pares +1 (a soma dá só 1,00), e o kernel parece ruim.
- **Centralizar** é subtrair, antes da conta, a similaridade média de cada linha e de cada coluna da tabela. Isso
  remove o "fundo" comum a todos os pares, e o KTA passa a medir se os pares da mesma classe são **mais** parecidos
  que a média. O kernel A sobe para 0,962. É o que importa para o SVM, que decide pelas diferenças entre as
  similaridades, e não pelo nível delas.
- O kernel **C**, que decora, tem 0,577 com 4 transações, mas o valor dele é 1/√(n − 1): com as 400 linhas usadas nas
  varreduras do notebook, cai para 0,05. É por isso que, na Seção 4.7, uma faixa larga demais (similaridade perto de
  zero, kernel quase identidade) derruba o KTA.

**Como ler os números do notebook.** O KTA centrado vai de −1 a 1: perto de 0, o kernel ignora o rótulo; quanto
maior, mais "parecidos" coincide com "mesma classe". É um **indicador** para delimitar a região do grid. Quem
escolhe é a validação cruzada. A função do projeto é `centered_kernel_target_alignment`, em
`src/fraud_qml/quantum/kernels.py`.

---

## Fase 3 — Divisão treino/validação/teste

**Arquivos:**
- `src/fraud_qml/splitting.py`, `src/fraud_qml/evaluation.py` (`precision_at_prevalence`)
- `src/fraud_qml/quantum/diagnostics.py` (varreduras de KTA, usadas na Seção 4)
- `tests/test_splitting.py`, `tests/test_evaluation.py`, `tests/test_quantum_diagnostics.py`
- Seção 3 do notebook (na época: Seção 4)

### Perguntas

1. 🟢 Na Seção 3.2, o mesmo classificador (recall 90%, 1% de falsos positivos) tem precisão 0,989 num teste 50/50 e
   0,131 na proporção real. Por que a precisão muda com a proporção de fraudes e o recall não?
2. 🟢 Para que serve cada uma das três partes (treino, validação, teste)? O que daria errado se o limiar de decisão
   fosse escolhido olhando o teste?
3. 🟡 O `DummyClassifier` do pipeline "aprendeu" P(fraude) = 0,50. Um modelo que estima probabilidades (por
   exemplo, uma regressão logística), treinado no mesmo treino balanceado, dá 0,9 a uma transação da validação.
   Isso quer dizer 90% de chance de fraude? O que muda na hora de decidir o limiar? (Um SVM não devolve
   probabilidade, e sim uma distância à fronteira; a pergunta vale para qualquer modelo cujo número se leia como
   probabilidade.)
4. 🔴 O split é aleatório e mistura os dois dias da coleta. Um banco usa o modelo para classificar transações
   **futuras**. Que split imitaria isso, o que ele mediria que o nosso não mede, e por que ele não foi adotado aqui?

---

## Reorganização — a exploração foi para depois do split (2026-10-07)

**O que mudou no notebook:** contrato (Seção 2) → duplicatas e split (Seção 3) → exploração só no treino (Seção 4).
A descrição das variáveis (faixas, médias, assimetria) saiu da Seção 2 e foi para a 4.1. A antiga Seção 4.3, que
repetia as medições só no treino, deixou de existir: agora tudo já nasce no treino.

**Por que (decisão sua).** Vazamento não é só ajustar um scaler com linhas de teste. Decidir olhando a base inteira
(qual transformação usar, que opções entram no grid) também carrega informação de validação e teste para o modelo.
É a forma sutil que os guias do scikit-learn (12.2) e do imbalanced-learn (9.1) mandam evitar. Eu tinha tratado o
problema com uma regra de exceção e depois recomendado não refazer para evitar retrabalho; registrei isso como NC10.

**Números que você já tinha estudado e que mudaram** (antes: base inteira; agora: treino):

| O quê | Antes | Agora |
|---|---|---|
| Taxa de fraude às 2h e às 4h | 1,45% e 1,04% | 1,29% e 1,28% |
| Fraudes de até 1,00 | 36% (legítimas: 10,6%) | 33,5% (legítimas: 10,6%) |
| Assimetria de `Amount` antes e depois do `log1p` | 17,0 → 0,16 | 15,2 → 0,16 |
| Features escolhidas pela informação mútua (dados balanceados) | `V10`, `V11`, `V12`, `V14`, `V17` | `V4`, `V10`, `V12`, `V14`, `V17` |
| Redundância entre elas, nas fraudes | 0,77 | 0,75 |
| Maior diferença Standard × Robust com PCA no meio | π/2 (1,5708) | 1,55 rad |
| PCA do El Alami: fração das features de fraude nos componentes 3 a 7 | 0,35; 0,26; 0,17; 0,49; 0,09 (984 linhas) | 0,36; 0,39; 0,05; 0,49; 0,21 (568 linhas do treino) |
| Teste da diferença da taxa de fraude entre os dois dias | p = 0,0043 | p = 0,085 (a razão entre as taxas quase não mudou: 0,81; o que caiu foi o poder do teste, com menos fraudes) |

**O que não mudou:** nenhuma opção do grid. As varreduras de KTA dão os mesmos números que a antiga checagem "só
no treino" (ZZ + PCA 0,44 ± 0,03 em 0,5 rad; Z + PCA 0,45 ± 0,03 em 1 rad; na amplitude com Standard, o maior KTA
na constante 5).

**Uma lição dos próprios números:** a quinta feature escolhida pela informação mútua muda conforme a amostra
(`V11` → `V4`). É o motivo prático de a seleção morar dentro do pipeline, refeita em cada fold.

**Correção minha (NC11).** Eu tinha posto nesta tabela que a lista das 7 features de menor informação mútua
também tinha mudado. Não mudou: são `V13`, `V15`, `V22`, `V26`, `V25`, `V24` e `hour_cos` na base inteira e no
treino. O que estava errado era o **texto** da aula da Fase 2a, que listava `V23` no lugar de `hour_cos`; a
tabela daquela época já estava certa.

**Um resíduo, achado por você (NC12).** Mesmo depois da reorganização, a 4.1 mostrava `V1` e `V14`, e a `V14`
estava lá porque já sabíamos que ela separa fraudes. Isso é o mesmo erro em escala menor: o conhecimento prévio
da base inteira escolhendo o que olhar. E mudava um número: a razão de escala entre `Time` e "as `V`" dependia de
qual `V` entrava no denominador (de 24.345 a 142.653 vezes). Agora a 4.1 mostra as 30 colunas, e toda seleção na
exploração sai do ranking calculado no treino. Os seis tipos de vazamento que o projeto encontrou ficaram em
`.claude/rules/data-leakage.md`.

### Perguntas

1. 🟢 Nenhum parâmetro da EDA antiga era reaproveitado pelos modelos. Por que, mesmo assim, fazer a EDA na base
   inteira é uma forma de vazamento?
2. 🟡 Depois do split, o notebook ainda conta as fraudes de validação e teste (tabela do split) e procura
   quase-duplicatas entre as partes. Por que isso é aceitável, e calcular a taxa de fraude por hora na base inteira
   não é?

---

## Fase 4 — Treinamento (Seção 5)

**Arquivos:**
- `src/fraud_qml/quantum/encodings.py` (`StatevectorEncoder`, mapa `ry`, simulação em lote)
- `src/fraud_qml/quantum/kernels.py` (`QuantumKernelSVC`), `src/fraud_qml/quantum/variational.py`
  (`VariationalQuantumClassifier`)
- `src/fraud_qml/models.py` (os pipelines), `src/fraud_qml/evaluation.py` (métricas e bootstrap)
- `tests/test_quantum_encodings.py`, `tests/test_quantum_variational.py`, `tests/test_models.py`
- Seção 5 do notebook

**Resumo da aula** (os números estão nas leituras de cada subseção da Seção 5 do notebook):
- **SVM e QSVM são o mesmo algoritmo;** só a tabela de similaridades (o kernel) muda.
- **AdaBoost:** cada rodada treina um classificador fraco, mede o erro ponderado ε, dá a ele o voto α = ln((1 − ε)/ε)
  e aumenta o peso das transações erradas. Exemplo: 5 transações com peso 0,2 e 1 erro dão ε = 0,2 e α = 1,386; a
  errada passa a pesar 0,5 e as outras, 0,125. Com o α = ½·ln(...) do artigo (0,693), os pesos novos são os mesmos.
- **Classificador variacional:** estado codificado → ansatz (10 pesos) → mede-se um qubit; P(1) é lida como
  P(fraude). Treina-se com entropia cruzada ponderada (−ln da probabilidade dada à classe certa: 0,105 para 0,9;
  0,693 para 0,5; 2,303 para 0,1).
- **Fase × rotação em Y:** mesmo kernel, classificadores diferentes. Na faixa padrão [0, π/2], a fase pura ordena
  quase tão bem (ROC-AUC 0,919 contra 0,940), mas o corte fixo em 0,5 cai no lugar errado (acurácia 0,654 contra
  0,874; com o melhor corte, a fase chegaria a 0,894). Não vale para toda largura: em 1 rad as duas empatam (0,878 e
  0,863). O que a RY tem é estabilidade (0,817 a 0,875 em todas as larguras e scalers testados).
- **ROC-AUC × PR-AUC:** com 0,17% de fraudes, um modelo pode ter ROC-AUC de 0,869 e PR-AUC de 0,021.
- **O boosting quântico para antes das 50 rodadas, sempre,** porque um classificador fraco chega a ε ≥ 0,5 e o
  scikit-learn interrompe. Quantos sobram depende da semente: 13, 44 e 24 no mesmo modelo. Mais orçamento ou um
  classificador fraco melhor (RY) não evitam a parada.
- **Intervalo de cada modelo × diferença pareada:** para comparar dois modelos, olha-se a diferença entre eles nas
  mesmas reamostras, e não se os intervalos se sobrepõem. E o bootstrap só mede a incerteza da validação: nos
  AdaBoost, trocar a semente do treino muda a PR-AUC em até 0,3.
- **Custo:** o que custa é simular os estados da validação (18 a 19 s contra menos de 1 s dos clássicos); é custo de
  simulação, não de hardware.

### Perguntas

1. 🟢 O QSVM e o SVM clássico são o mesmo algoritmo. O que, exatamente, muda de um para o outro?
2. 🟢 Um classificador fraco erra transações cujos pesos somam ε = 0,2. Qual é o α dele no scikit-learn, e o que
   acontece com o peso das transações que ele errou? Por que o fator ½ do artigo do Quantum AdaBoost não muda a
   decisão final?
3. 🟡 Na Seção 5.3, a fase pura e a rotação em Y têm ROC-AUC parecidas (0,919 e 0,940), mas acurácias muito
   diferentes (0,654 e 0,874). Como as duas coisas podem ser verdade ao mesmo tempo? E por que é a acurácia, e não a
   ROC-AUC, que importa para o AdaBoost?
4. 🔴 O Quantum AdaBoost pediu 50 rodadas e ficou com 13 (com outras sementes, 44 e 24). O que faz o boosting
   parar? Trocar a codificação pela rotação em Y deixou o classificador fraco bem melhor (ε de 0,12 na primeira
   rodada, contra 0,33) e o boosting parou ainda **mais cedo** (7 a 13 rodadas). Por que um classificador fraco
   melhor pode fazer o boosting durar menos?

---

## Fase 5 — Ajuste de hiperparâmetros (Seção 6)

**Arquivos:**
- `src/fraud_qml/tuning.py` (grid em dois níveis, cache, projeção do piloto, resumo e relatório JSON)
- `src/fraud_qml/models.py` (`InputEncoding`, `ModelSettings`, `build_pipeline`; redutor nos clássicos)
- `src/fraud_qml/config.py` (grids de C, γ, teto, taxa, ansatz; 5 folds; limite de 2 h)
- `tests/test_tuning.py`, `tests/test_models.py`, `tests/test_notebook_protocol.py`, `tests/test_notebook_tools.py`
- Seção 6 do notebook; resultados em `artifacts/results/grid_*.json`

**Resumo da aula** (os números estão nas leituras da Seção 6):
- **Grid search com validação cruzada sobre o pipeline inteiro:** scaler, redutor, faixa e mapa entram junto com C e
  taxa, porque as escolhas interagem.
- **Undersampling dentro de cada fold:** o fold de avaliação fica na proporção real (34.047 linhas, 56 ou 57
  fraudes), então a escolha pode ser pela PR-AUC, cujo acaso é 0,0017 ali (seria 0,5 num fold 50/50).
- **Grid em dois níveis:** o estado é simulado uma vez por entrada e fold e serve a todos os valores do modelo (no
  QSVM, 390 simulações para 1.560 ajustes); o AdaBoost é lido rodada a rodada.
- **Otimismo do vencedor:** com combinações independentes que valessem todas 0,70, o melhor de 36 pareceria 0,776 e o
  melhor de 2.016, 0,823. O Quantum AdaBoost A disputou 2.016 combinações; o AdaBoost com 5 componentes, 36.
- **Taxa de aprendizado:** com taxa 1, depois de cada rodada as transações erradas somam exatamente metade do peso (o
  classificador anterior vira acaso nos pesos novos); com taxa 0,1 e ε = 0,3, somam 0,318. É a condição para a parada,
  mas não basta: o AdaBoost clássico tem a mesma propriedade e nunca para, porque sempre acha um *stump* novo abaixo de
  0,5; o classificador variacional, julgado com o corte fixo em 0,5, não acha (CC17). Taxa menor adia a parada do
  boosting quântico, mas exige mais rodadas.
- **Escolhas:** os quatro modelos com 5 componentes escolheram StandardScaler → informação mútua; as variações A, a
  faixa [0; 0,25] com ZZ; as B, a constante 5; os AdaBoost, taxa 0,1 e teto 100. O caminho até os componentes é a
  única decisão que o grid separa com clareza (descarta o RobustScaler → PCA).
- **Comparação fold a fold:** os folds são os mesmos para todos, e o fold 1 é o mais difícil para quase todos. Dentro
  do par, o QSVM A empata com o SVM (2 de 5 folds) e o Quantum AdaBoost A fica à frente do AdaBoost nos 5, mas foi
  escolhido entre 2.016 combinações contra 36: a validação cruzada não ordena os modelos. As 8 escolhidas ficaram na
  borda de alguma dimensão.
- **Grid próprio × `GridSearchCV` (6.1.2):** mesmas notas (diferença zero), menos simulação (5 contra 20 no pedaço do
  QSVM), e a leitura do AdaBoost rodada a rodada. O `RandomizedSearchCV` ficou de fora porque o grid coube inteiro e a
  leitura precisa de todas as combinações.
- **Segunda rodada nas bordas (6.5, decidida em 2026-11-02):** três pipelines ganharam (SVM com 31 features, γ = 0,001;
  Quantum AdaBoost B, constante 10 e taxa 0,05; AdaBoost com componentes, taxa 0,05), sempre menos que o desvio entre
  folds; o teto não prejudicava o AdaBoost com 31 features.
- **De 4 a 8 qubits (6.6):** fold a fold, os quânticos não melhoram com mais qubits, e o AdaBoost clássico melhora um
  pouco com mais componentes. Com mais qubits, a largura do MinMax precisa ser mais estreita (concentração do kernel). A
  regra, com o 5 protegido, manteve 5 qubits nos quatro pipelines (A e B com o mesmo orçamento). A primeira versão da
  regra descia para 4: com desvios dessa ordem, ela quase não podia escolher outra coisa (NC18).
- **As configurações que seguem (6.7)** são as da segunda rodada, todas com 5 qubits ou 5 componentes, e ficam
  congeladas para a Seção 7.

### Perguntas

1. 🟢 Por que o undersampling precisa ser sorteado dentro de cada fold, e por que isso permite escolher pela PR-AUC em
   vez da ROC-AUC?
2. 🟡 O Quantum AdaBoost A foi escolhido entre 2.016 combinações; o AdaBoost com 5 componentes, entre 36. Por que a nota
   do primeiro tende a ser mais otimista, mesmo que os dois modelos fossem igualmente bons? Onde o projeto mede o
   desempenho sem esse viés?
3. 🟡 Com taxa de aprendizado 1, mostre com ε = 0,2 que, depois da rodada, as transações erradas passam a somar metade
   do peso. Por que isso, sozinho, não explica a parada do boosting quântico, se o AdaBoost clássico tem a mesma
   propriedade e nunca para?
4. 🔴 O ajuste rendeu +0,676 de PR-AUC no Quantum AdaBoost A e +0,115 no AdaBoost com 31 features. Por que isso não é
   evidência de que o quântico aprende melhor?

### Dúvida (2026-11-02): "depois do redutor não deveria vir o MinMax?"

Vem, e sempre veio: o pipeline da variação A é undersampling → scaler → redutor → **`MinMaxScaler(clip=True)`** (etapa
`angle_scaler`) → feature map → modelo. A "largura da faixa" das tabelas era o **intervalo de saída desse MinMax**,
[0; largura], e o grid varia esse intervalo: [0; 0,25], [0; 0,5], [0; 1] e [0; π/2]. O nome escondia isso, e a Seção 6
passou a dizer "MinMax para [0; 0,25]" e "intervalo do MinMax". O intervalo para em π/2, e não em π, porque os mapas do
Qiskit giram o qubit pelo dobro do valor: em [0; π], os extremos opostos virariam o mesmo estado (Seção 4.7.1). Ele
fica entre as decisões de **codificação** porque age como a largura de banda do kernel quântico, o papel que o γ tem no
RBF.

### Dúvida (2026-11-02): "a variação B não deveria testar o MinMax também?"

Não, e a Seção 4.8.1 mostra por quê, em três partes:
- **A amplitude não precisa de intervalo restrito.** A exigência dela é norma 1, garantida pela normalização; o
  "intervalo necessário para a codificação quântica" do texto do TCC vale só para o ângulo (TX11).
- **A largura do MinMax já está no grid, com o nome de constante.** Multiplicar as features por w e completar com c dá
  o mesmo estado que completar com c/w (diferença de 2,2 × 10⁻¹⁶ nos dados; há teste). A constante é a largura de banda
  da amplitude.
- **Como scaler, o MinMax concentra o kernel.** Com todos os valores entre 0 e 1, os vetores apontam para a mesma região,
  a similaridade média vai de 0,855 a 0,995, e o KTA fica entre 0,359 e 0,393, contra 0,483 do StandardScaler com a
  constante 5.

O que pode estar limitando a variação B é a constante, que ficou na borda do grid (5) e é a "largura" que importa ali.
Estender a constante faz parte da decisão pendente sobre as bordas (CC19).

---

## Fase 6 — Avaliação na validação (Seção 7)

**Arquivos:**
- `src/fraud_qml/thresholds.py` (curva de F1 por limiar, limiar de F1 máximo, predição num limiar)
- `src/fraud_qml/reporting.py` (gravação do JSON de resultados)
- `src/fraud_qml/tuning.py` (a chave do cache passou a cobrir só o código que a busca importa, opção A de 2026-11-04)
- `tests/test_thresholds.py`, `tests/test_tuning.py`, `tests/test_notebook_protocol.py` (a Seção 7 não lê o teste nem
  volta a escolher configurações)
- Seção 7 do notebook; resultados em `artifacts/results/validation_evaluation.json`

**Resumo da aula** (os números estão nas leituras da Seção 7):
- **O que a validação mede:** as oito configurações congeladas, reajustadas no treino inteiro (568 linhas depois do
  undersampling), em linhas que não participaram de nenhuma escolha.
- **Validação × validação cruzada:** as oito ficaram acima da média da validação cruzada, e os padrões da Seção 5, que
  não passaram por escolha, também, em 6 de 8. O deslocamento é em boa parte comum (treino maior, dificuldade da
  validação). Descontado o comum, "escolhida − padrão" é negativa em 6 de 8 (mediana −0,035), a direção do otimismo da
  escolha, mas o ruído dos padrões (um sorteio cada) não deixa concluir.
- **Limiar:** o padrão (0 na `decision_function`) é o de um modelo treinado com metade de fraudes e dispara de 107 a
  2.994 alertas para 94 fraudes (a taxa de falsos positivos no limiar 0 vai de 0,05% a 3,5%). O de F1 máximo fica entre
  78 e 103 alertas, com F1 de 0,758 a 0,826, **otimista** porque escolhido nas mesmas linhas: escolhendo numa metade da
  validação e medindo na outra, o F1 cai de 0,015 a 0,037. F1 em contagens: 2·VP / (alertas + fraudes).
- **ROC × PR:** as ROC-AUC (0,975 a 0,980) quase não separam nada, porque a taxa de falsos positivos divide pelos 56.651
  legítimos; a curva PR divide pelos alertas.
- **Bootstrap pareado:** nenhuma das oito diferenças exclui o zero. Nos pares com a mesma entrada, quântico e clássico
  empatam; a vantagem do Quantum AdaBoost A na validação cruzada (5 de 5 folds) não apareceu na validação, o que é
  compatível com o otimismo da escolha (CC18) e com o ruído de um sorteio do undersampling.
- **Os modelos erram juntos:** 60 fraudes têm alerta nos oito, 14 ficam sem alerta nos oito, e só 20 separam os
  modelos. Por isso as diferenças pareadas têm intervalos mais estreitos que os de cada modelo.
- **Custo:** a variação A custa dezenas de vezes o clássico para pontuar (simulação de um circuito por transação). Na B,
  só o QSVM custa como um clássico; o Quantum AdaBoost B custa umas 70 vezes o AdaBoost para ajustar.

### Perguntas

1. 🟢 Por que o F1 no limiar escolhido é otimista e a PR-AUC não?
2. 🟡 O limiar padrão deu 2.077 alertas no AdaBoost com 31 features e 107 no SVM com 31 features, para 94 fraudes. Por
   que o limiar 0 de um modelo treinado com metade de fraudes dispara tanto na proporção real, e por que o efeito é tão
   diferente entre os dois modelos?
3. 🟡 As oito escolhidas ficaram acima da média da validação cruzada. Isso mostra que a escolha não foi otimista? O que a
   coluna "escolhida − padrão" ajuda a separar, e o que ela não separa?
4. 🔴 O Quantum AdaBoost A ficou à frente do AdaBoost com componentes nos 5 folds (+0,055) e, na validação, a diferença
   é +0,001, com IC de −0,054 a +0,062. Explique por que as duas medidas não se contradizem e o que faltaria para afirmar
   que um dos dois é melhor.

---

## Fase 7 — Diagnóstico (Seção 8)

**Arquivos:**
- `src/fraud_qml/curves.py` (curva de aprendizado com frações aninhadas; curvas de validação nos folds da Seção 6, com
  a nota de treino e o teto lido rodada a rodada)
- `src/fraud_qml/caching.py` (cache em disco das medidas novas, fora da chave das buscas da Seção 6)
- `src/fraud_qml/quantum/diagnostics.py` (medida de *barren plateau*: leitura global, derivada pelo *parameter-shift*)
- `tests/test_curves.py`, `tests/test_caching.py`, `tests/test_barren_plateau.py`, `tests/test_notebook_protocol.py`
  (a Seção 8 não lê o teste, não volta a escolher nem troca configurações e limiares)
- Seção 8 do notebook; resultados em `artifacts/results/diagnostics.json`

**Resumo da aula** (os números estão nas leituras da Seção 8):
- **Viés × variância:** subajuste = as duas notas baixas e próximas; sobreajuste = treino alto e avaliação baixa. A
  distância entre as notas é o sinal (sklearn 3.5.1).
- **Nota de treino com undersampling:** a ROC-AUC nas linhas vistas e a PR-AUC em todas as linhas do `fit`. A PR-AUC
  ponderada nas linhas vistas saiu: cada legítima vista pesa ≈ 598, e uma linha muda tudo. No exemplo pequeno calculado
  (3 fraudes, 3 legítimas de peso 100), uma legítima no topo leva a nota de 1,0 a 0,35.
- **Curva de aprendizado:** a diferença da 7.2 vira soma de três partes (tamanho do treino, conjunto de avaliação,
  sorteio). O tamanho explica uma parte nos boostings (+0,016 a +0,031 com a referência de 75%) e no QSVM A, e quase nada
  nos SVM e no QSVM B. A parte "conjunto de avaliação" é positiva nos oito; o otimismo da escolha puxa a validação cruzada
  para cima e só pode diminuí-la, então a validação é mais fácil que os folds em pelo menos esse valor. Os boostings
  ainda sobem no fim da curva.
- **Curvas de validação:** nos folds, treino e avaliação ficam a no máximo 0,018 um do outro no valor escolhido; o
  sobreajuste aparece longe dele (SVM com 31 features, γ = 10: 1,000 × 0,116). 14 das 18 curvas têm o topo no escolhido.
- **Reprodutibilidade (CC21):** o Quantum AdaBoost dá outro modelo com os mesmos números em outro tipo de dado
  (tabela × array): 0,7785 × 0,7653 de PR-AUC no A.
- **Barren plateau (CC22):** variância da derivada entre pesos sorteados. *Parameter-shift*: com ⟨Z⟩ = cos θ e θ = 0,5,
  [cos(θ + π/2) − cos(θ − π/2)] / 2 = −0,4794 = −sen 0,5. A leitura global e o ansatz profundo produzem o platô (os
  controles). Com a leitura local e o ansatz de 1 ou 2 repetições, quem decide é a codificação: a congelada (ZZ, todos os
  pares, [0; 0,25]) cai com fator 0,55 por qubit e emaranha o qubit medido (comprimento de Bloch de 0,05 a 0,29); com
  [0; 0,1] (0,75 a 0,81), com o mapa Z (0,80 a 0,85) e com a rotação em Y (0,98 a 1,14), cai pouco ou nada. Com 5 qubits e
  2 repetições, 17 vezes na variância, mas só 2,1 na mediana do módulo da derivada.

**Correções da auditoria (NC22), para a releitura:** o otimismo da escolha não explica a parte positiva "validação −
validação cruzada" (tem o sinal contrário); o tamanho do treino pesa menos do que a média com 80% sugeria; o platô é do
mapa ZZ com a largura 0,25, e não do mapa ZZ em geral; as bordas são julgadas pela diferença fold a fold.

### Perguntas

1. 🟢 Na curva de validação do SVM com 31 features, com γ = 10, a nota de treino foi 1,000 e a de avaliação 0,116. Isso é
   viés ou variância? Como a distância entre as duas notas mostra isso?
2. 🟡 Por que a PR-AUC ponderada nas linhas vistas não serve como nota de treino com o undersampling 1:1? Use o peso de cada
   legítima vista (≈ 598) na resposta.
3. 🟡 Na decomposição da diferença da 7.2, a parte "conjunto de avaliação" (validação com 80% do treino menos a média da
   validação cruzada) foi positiva nos oito pipelines. Por que o otimismo da escolha não pode explicar essa parte positiva,
   e o que se conclui então sobre a validação?
4. 🔴 Na 8.3, com a leitura local e o ansatz de 2 repetições, a variância da derivada caiu pela metade a cada qubit com o mapa
   ZZ na largura [0; 0,25], quase não caiu com a mesma ZZ na largura [0; 0,1] e não caiu com a rotação em Y. O que cada
   controle isola, e por que o controle global sozinho não bastava para apontar a causa?

---

## Fase 8 — Modelo final: o teste (Seção 9)

**Arquivos:**
- `src/fraud_qml/significance.py` (McNemar exato pelo binomial, com as discordantes separadas por classe; correção de Holm)
- `src/fraud_qml/final_evaluation.py` (pontuar com um pipeline já ajustado; robustez
  nas 5 sementes, cada uma com o próprio limiar escolhido na validação)
- `tests/test_significance.py`, `tests/test_final_evaluation.py`, `tests/test_notebook_protocol.py` (o pré-registro vem
  antes da primeira leitura do teste; a Seção 9 não reajusta, não volta a escolher e não escolhe limiar com o teste)
- Seção 9 do notebook; resultados em `artifacts/results/final_evaluation.json`

**Resumo da aula** (os números estão nas leituras da Seção 9):
- **Pré-registro:** antes de ler o teste, a lista fixa das 8 comparações, os critérios e as expectativas. A expectativa 3
  (nenhum intervalo de PR-AUC excluindo o zero) falhou e ficou reportada como falhou; a análise feita depois de ver o
  resultado (9.4.1) vem declarada como posterior.
- **McNemar exato:** só as discordantes contam (b: só o primeiro acerta; c: só o segundo); sob a hipótese de erros iguais,
  b ~ Binomial(b + c, 1/2). Com b = 3 e c = 11, o p exato é 0,057, e a aproximação qui-quadrado sem correção daria 0,033.
  O Raschka (2018, 4.3) só confia na aproximação com B e C "say > 25"; no teste foram de 8 a 28 discordantes.
- **Holm:** com p = 0,012; 0,03; 0,2, os ajustados são 0,036; 0,06; 0,2. Com 8 comparações sem diferença real, a chance
  de algum p < 0,05 por acaso é 0,34.
- **PR-AUC × McNemar:** a PR-AUC pesa o topo da ordenação (uma legítima acima de 4 fraudes leva a PR-AUC de 1,0 a 0,679; a
  mesma legítima entre a 3ª e a 4ª, a 0,95); o McNemar conta erros no limiar. No teste, 4 diferenças de PR-AUC excluíram o
  zero e nenhum McNemar deu diferença.
- **O grupo de 89,99:** legítimas desse valor (405 no treino, 125 na validação, 125 no teste, nenhuma fraude) ocupam parte
  do topo da maioria dos modelos, nos dois conjuntos (em 5 dos 8, 3 ou mais acima da 40ª fraude). No teste, os SVM clássicos põem 6 no topo e o QSVM A,
  nenhuma; na validação, os dois põem 3. Sem o grupo, QSVM A − SVM com a mesma entrada cai de +0,092 para +0,016, mas
  Quantum AdaBoost B − AdaBoost com 31 features sobe de +0,067 para +0,090: a comparação é **frágil** a um tipo de
  transação, nos dois sentidos (CC23). A primeira versão desta análise tratou o grupo como um evento só do teste e estendeu
  a conclusão às quatro diferenças; a auditoria corrigiu (NC23).
- **Sementes × bootstrap:** as sementes mudam o modelo e não o teste; o bootstrap muda as linhas do teste e não o modelo.
  Com o split aleatório, o bootstrap mede bem outro sorteio desta base; não mede outro período, em que um tipo de transação
  pode aparecer em outra proporção.
- **A semente da escolha:** nos dois Quantum AdaBoost, a 42 é a maior das 5 na validação e no teste, acima da variação
  entre as outras (CC24); nos AdaBoost clássicos, só na validação; nos SVM e QSVM, fica na variação entre sementes. Nos Quantum AdaBoost, a média das 5 é a estimativa menos
  dependente da escolha.
- **Os sinais estáveis:** QSVM A − QSVM B (variação A × B, que junta entrada e codificação), positiva na validação, no teste,
  nas 5 sementes e sem o grupo; e QSVM B − SVM com 31 features, negativa em todas as medidas (a favor do clássico). A
  conclusão do TCC: não há vantagem geral do quântico sobre o clássico nesta base.

### Perguntas

1. 🟢 Por que o pré-registro precisa ser escrito antes de qualquer célula ler o teste? O que aconteceria com as conclusões
   se as comparações fossem escolhidas depois?
2. 🟡 No McNemar, com b = 3 e c = 11, o p exato é 0,057 e a aproximação qui-quadrado sem correção dá 0,033. Por que a
   versão exata é a certa aqui, e por que o McNemar não enxerga a diferença que a PR-AUC enxergou entre o QSVM A e o SVM
   com componentes?
3. 🟡 Na correção de Holm, com p = 0,012; 0,03 e 0,2, por que o 0,03 deixa de ser significativo a 0,05, se sozinho ele
   seria?
4. 🔴 A diferença QSVM A − SVM com componentes ficou entre +0,084 e +0,104 nas 5 sementes, com intervalo do bootstrap
   excluindo o zero, e sem as legítimas de valor 89,99 ela cai para +0,016. Já Quantum AdaBoost B − AdaBoost com 31 features
   sobe sem elas. Use o que cada medida muda e o que ela deixa fixo (sementes, bootstrap) para explicar por que a
   estabilidade entre sementes não garante que a diferença seja geral, e por que o resultado da sensibilidade leva a chamar
   a comparação de "frágil", e não de "nula".
---

## Perguntas prováveis da banca

Seção viva: cada pergunta que aparecer nas fases entra aqui, com a resposta curta (para falar) e os números que a
sustentam.

### "No teste, o QSVM ficou à frente do SVM com a mesma entrada, com intervalo excluindo o zero. Isso não é vantagem quântica?"

**Resposta para falar:**
> "No teste, sim: +0,092 de PR-AUC, estável nas cinco sementes. Mas a diferença não apareceu na validação (+0,018), não
> aparece nas decisões de alerta (o McNemar não vê diferença) e, olhando onde ela está, vem de um tipo de transação
> legítima, as de valor 89,99, que o SVM clássico pôs no topo do teste e o QSVM não. Sem esse grupo, ela cai para +0,016.
> E o mesmo grupo mexe nas outras comparações nos dois sentidos: sem ele, uma delas passa a favorecer o clássico com
> intervalo que exclui o zero. Com 95 fraudes,
> a comparação entre os modelos é frágil, e o TCC não sustenta uma vantagem geral."

**Por trás da resposta:**
- O pré-registro (9.0) previa nenhuma diferença; a previsão falhou, e o TCC reporta isso. A verificação do grupo (9.4.1)
  foi feita depois de ver o resultado e está declarada assim: é uma sensibilidade descritiva, não uma correção.
- O grupo não é um acaso do teste: há 405 dessas legítimas no treino e 125 na validação, sem nenhuma fraude, e na
  validação ele também está no topo (3 de 5 legítimas no QSVM A e no SVM com componentes).
- As sementes mudam o modelo, e não o teste: a estabilidade entre sementes mostra que não é sorte do undersampling, mas o
  grupo está no teste em todas elas (CC23).
- Quantum AdaBoost A − AdaBoost com a mesma entrada aparece na validação cruzada e no teste (5 de 5 sementes), cai à metade
  sem o grupo (limite inferior +0,000) e é zero na validação; e a semente da escolha favorece o Quantum AdaBoost (CC24).
- QSVM B − SVM com 31 features mantém o sinal a favor do clássico em todas as medidas.
- A diferença de sinal mais estável é QSVM A − QSVM B: mantém o sinal na validação cruzada (5 de 5 folds), na validação, no
  teste, nas 5 sementes e sem o grupo; mas junta entrada e codificação, e o McNemar não a detecta.

### "A conversão de `Time` em hora foi aplicada à base inteira. Isso não é vazamento?"

**Resposta para falar:**
> "A conversão de `Time` em hora do dia é uma função fixa aplicada a cada transação isoladamente, sem nenhum
> parâmetro estimado a partir dos dados. Por isso ela pode ser aplicada às três partes, como seria aplicada a uma
> transação nova em produção. Tudo o que relaciona a hora com a fraude, como a taxa por hora e o ranking das
> features, foi calculado só no treino. Nenhuma informação dos rótulos da validação ou do teste entrou no treino
> nem nas decisões."

**Por trás da resposta:**
- A transformação é aplicada às três partes porque o modelo precisa das mesmas features na predição, e não "para
  melhorar a visualização".
- O que a torna segura é ser **linha a linha e sem parâmetro aprendido**. Um StandardScaler, ao contrário, calcula
  a média e o desvio padrão da coluna inteira; ajustado na base inteira, levaria informação da validação para o
  treino. O teste `test_engineering_a_part_equals_the_part_of_the_engineered_whole` prova que transformar uma parte
  dá o mesmo que transformar a base e depois separar.
- Cuidado com a frase "o modelo não sabia o horário da validação e do teste": ele **sabe**, porque `hour_sin` e
  `hour_cos` são features. O que ele não viu foram os **rótulos** dessas linhas, nem estatísticas calculadas com eles.
- Pergunta seguinte provável: "a ideia da hora em seno/cosseno não veio de um trabalho que olhou a base inteira?"
  Veio, e o notebook declara isso (4.4 e 4.11). A análise do treino, sozinha, a justifica (o vale de volume na
  madrugada; 23h e 0h ficariam distantes na hora crua), e uma função sem parâmetro não carrega informação de nenhuma
  linha específica da validação ou do teste.

### "Por que undersampling, e não SMOTE ou ADASYN?"

**Resposta para falar:**
> "Por custo da simulação quântica e por simetria entre os pares. O SMOTE equilibraria as classes criando fraudes
> sintéticas até igualar as legítimas, o que levaria o treino a cerca de 340 mil linhas. O kernel quântico precisa
> de uma simulação por linha e de uma matriz de similaridade linhas × linhas: com 340 mil linhas, essa matriz teria
> mais de 800 GiB. Com undersampling, o treino tem 568 linhas e a matriz, 2,5 MiB. E os quatro modelos treinam no
> mesmo subconjunto balanceado, para que a comparação clássico × quântico seja justa."

**Por trás da resposta** (números calculados com o treino e com o benchmark da Seção 1):

| | Undersampling (1:1) | SMOTE (1:1) |
|---|---|---|
| Linhas de treino | 568 | 339.902 (169.667 fraudes sintéticas) |
| Simular os estados de um ajuste do QSVM | menos de 3 s | ~24 min |
| Matriz de similaridade (kernel) | 2,5 MiB | ~861 GiB |

- **Simetria:** o TCC V4 previa SMOTE/ADASYN só para os clássicos (TX3). Com pipelines diferentes, uma diferença de
  desempenho poderia vir do balanceamento, e não do modelo. Por isso os quatro usam o mesmo undersampling.
- **O preço do undersampling, dito sem esconder:** só 0,1671% das legítimas do treino são usadas. O modelo vê poucas
  legítimas, e tipos raros de transação legítima podem ficar de fora, o que derruba a precisão. Mitigações: 5
  sementes de undersampling na análise de robustez e o limiar escolhido na validação, com a proporção real (Seções
  4.10 e 7).
- **E o vazamento?** SMOTE também seria aceitável **dentro** do pipeline, só no treino. O erro comum na literatura é
  aplicar SMOTE ou undersampling antes do split, o que deixa o teste balanceado (EA1, GR1). Aqui o undersampling
  vem depois do split, dentro do pipeline (Seção 3.2).

### "Por que um split aleatório, e não temporal? Em produção o modelo prevê o futuro."

**Resposta para falar:**
> "Um split temporal seria o ideal para estimar o desempenho em produção, mas a base cobre só 48 horas. Num split
> temporal 60/20/20, validação e teste seriam duas janelas de cerca de 7 horas no fim do segundo dia, sem nenhuma
> madrugada, que é onde estão os picos de fraude. A diferença medida misturaria o horário com a passagem do tempo.
> Por isso usamos o split aleatório estratificado, o mesmo da literatura comparada, e declaramos a limitação: os
> resultados não estimam a degradação do modelo ao longo do tempo."

**Por trás da resposta** (simulação dentro do treino, Seção 3.3.1; CC12):
- Janelas de um split temporal: 0–33,5 h, 33,5–40,4 h e 40,4–48,0 h. As duas últimas têm 6,9 e 7,6 horas e vão da
  manhã do dia 2 até a meia-noite.
- Fraudes nas janelas: 199, 41 e 44 (taxas de 0,1948%, 0,1204% e 0,1292%).
- O split aleatório mistura os dois dias em todas as partes: cada uma tem entre 50,7% e 51,1% de linhas do dia 1.
- A taxa de fraude do dia 2 é 0,81 da do dia 1 no treino, mas o teste não distingue isso do acaso (p = 0,085) por
  falta de poder: com 284 fraudes, não dá para afirmar nem negar uma deriva entre os dois dias.
