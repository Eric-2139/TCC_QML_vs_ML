# ML clássico × Quantum-Enhanced ML na detecção de fraude em cartão de crédito

Parte experimental do Trabalho de Conclusão de Curso de **Eric Costa Ciciliotti**.

O trabalho compara algoritmos clássicos de aprendizado de máquina com versões que usam computação quântica
(simulada), num problema real e difícil: encontrar fraudes em transações de cartão de crédito, em que só 0,17% das
transações são fraude.

## Objetivo

Comparar, de forma justa, dois pares de algoritmos da mesma família, um clássico e um quântico:

| Par | Clássico | Quântico |
|---|---|---|
| Máquinas de vetores de suporte | SVM com kernel RBF | QSVM: o mesmo SVM, com a similaridade entre transações calculada por um circuito quântico (kernel quântico de fidelidade) |
| Boosting | AdaBoost com árvores de um nível | Quantum AdaBoost: o mesmo AdaBoost, com circuitos quânticos variacionais como classificadores fracos |

As perguntas que o experimento responde:
- **Desempenho:** os modelos quânticos detectam fraude melhor, igual ou pior que os clássicos, na proporção real de
  fraudes?
- **Custo computacional:** quanto tempo e memória cada modelo consome para treinar e para prever?
- **Codificação:** como a forma de transformar uma transação num estado quântico afeta o resultado? São comparadas
  duas codificações com o mesmo número de qubits (5): **ângulo** (cada uma de 5 componentes vira a rotação de um
  qubit) e **amplitude** (as 31 features viram as amplitudes de um estado de 5 qubits).
- **Ruído:** o que acontece com os modelos quânticos quando a simulação inclui o ruído de um hardware real?

## Dados

[*Credit Card Fraud Detection*](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB / Kaggle): 284.807
transações de portadores de cartão europeus, feitas em dois dias de setembro de 2013, com 492 fraudes. As variáveis
`V1` a `V28` são componentes de PCA das variáveis originais, que o banco não divulgou; `Time` e `Amount` estão na
escala original.

O arquivo **não está no repositório** (66 MB). Para rodar, baixe o `creditcard.csv.zip` do Kaggle e coloque-o na
raiz do projeto, sem descompactar. O código confere o SHA-256 do arquivo e para com uma mensagem clara se ele não for
o original.

## Como o experimento evita resultados enganosos

Trabalhos de referência de aprendizado de máquina quântico neste dataset (El Alami et al., 2025; Grossi et al., 2022)
balanceiam os dados antes de separar o teste e avaliam num teste com cerca de 50% de fraudes. Isso infla a precisão: o mesmo classificador que tem precisão de 0,989 num
teste 50/50 tem 0,131 na proporção real (Seção 3.2 do notebook). Este experimento segue um protocolo mais rigoroso:

- **Duplicatas removidas antes de tudo** (283.726 transações e 473 fraudes), para que uma cópia não fique no treino e
  outra no teste.
- **Divisão 60/20/20 estratificada, antes de qualquer balanceamento.** Validação e teste mantêm a proporção real de
  fraudes.
- **Toda a exploração dos dados é feita só no conjunto de treino.** Na base inteira, só se verificam a integridade do
  arquivo e as duplicatas.
- **Balanceamento por undersampling só no treino**, dentro de um pipeline que é reajustado em cada fold da validação
  cruzada. Os quatro modelos treinam no mesmo subconjunto balanceado.
- **Métrica principal: PR-AUC** (área sob a curva precisão × recall), com MCC, ROC-AUC, precisão, recall e F1. O
  limiar de decisão é escolhido na validação.
- **O teste é usado uma única vez**, no modelo final, com intervalos de confiança por bootstrap, teste de McNemar e
  repetição com 5 sementes de sorteio.

## Situação

| Seção do notebook | Conteúdo | Situação |
|---|---|---|
| 1. Setup | Ambiente, versões e medição do custo da simulação quântica | concluída |
| 2. Coleta | Leitura e verificação do arquivo (contrato dos dados) | concluída |
| 3. Divisão | Duplicatas, split 60/20/20 e checagens de integridade | concluída |
| 4. Exploração e pré-processamento | Análise do treino, escalonamento, codificação quântica e seleção de features | concluída |
| 5. Treinamento | Os quatro modelos com hiperparâmetros padrão | próxima |
| 6. Ajuste de hiperparâmetros | Busca em grade com validação cruzada | a fazer |
| 7. Avaliação | Métricas na validação, curvas e escolha do limiar | a fazer |
| 8. Diagnóstico | Curvas de aprendizado e de validação | a fazer |
| 9. Modelo final | Avaliação única no teste, com análise estatística | a fazer |
| 10. Ruído quântico | Os modelos quânticos finais com ruído simulado | a fazer |

## Alguns achados até aqui

- **Fraudes se repetem dez vezes mais que transações legítimas:** 3,9% das fraudes são cópias exatas, contra 0,37% das
  legítimas.
- **StandardScaler seguido de MinMaxScaler dá exatamente o mesmo resultado que o MinMaxScaler sozinho**, uma sequência
  comum na literatura. O primeiro só faz diferença se houver um PCA entre os dois.
- **A faixa de ângulos funciona como a "largura de banda" do kernel quântico:** estreita demais, todas as transações
  parecem iguais; larga demais, cada uma só se parece consigo mesma e o modelo decora o treino.
- **Na codificação de amplitude, completar o vetor com zeros descarta a informação da norma**, que distingue fraudes.
  Completar com uma constante a preserva: o alinhamento do kernel com as classes sobe de 0,08 para 0,48.
- **O kernel da codificação de amplitude pode ser calculado classicamente** (é o cosseno ao quadrado do ângulo entre
  os vetores), o que limita a vantagem que se pode esperar dela.
- **O custo decide o desenho:** treinar o kernel quântico com SMOTE exigiria uma matriz de similaridade de cerca de
  860 GiB; com undersampling, ela tem 2,5 MiB.

## Organização do repositório

```
notebooks/tcc_ml_vs_qml.ipynb   o experimento, seção por seção, com texto explicando cada decisão
src/fraud_qml/                  o código reutilizável, coberto por testes
tests/                          os testes automatizados (pytest)
tools/notebook_tools.py         utilitários para executar e inspecionar o notebook
PLAN.md                         o desenho do experimento e as decisões tomadas
methodological_issues.md        os problemas metodológicos encontrados e como foram tratados
pyproject.toml, requirements.txt   dependências, com as versões fixadas
```

**O notebook** é o experimento em si. Cada seção traz, antes do código, o que vai ser feito e por quê; depois do
código, a leitura dos resultados, com os números das saídas.

**`src/fraud_qml/`** concentra o código que o notebook usa, para que ele possa ser testado e o notebook fique
legível:

| Módulo | O que faz |
|---|---|
| `config.py` | Constantes do protocolo: sementes, proporções do split, número de qubits, opções da busca em grade |
| `data.py` | Leitura do arquivo, verificação de integridade e remoção de duplicatas |
| `splitting.py` | Divisão treino/validação/teste estratificada e as verificações dela |
| `features.py` | Transformações linha a linha (`log(1 + Amount)`, hora do dia em seno e cosseno) |
| `preprocessing.py` | Escalonamento, redução para 5 componentes, preparo para a codificação de amplitude e undersampling |
| `quantum/encodings.py` | Os circuitos que transformam uma transação num estado quântico |
| `quantum/kernels.py` | Os kernels quânticos e as medidas de qualidade de um kernel |
| `quantum/diagnostics.py` | As varreduras que comparam configurações de kernel |
| `evaluation.py` | Métricas e a correção de probabilidade após o undersampling |
| `profiling.py` | Medição de tempo e memória |
| `environment.py`, `plotting.py` | Registro de versões e identidade visual dos gráficos |

**`PLAN.md`** descreve o desenho completo: as decisões de protocolo e o porquê de cada uma, as opções da busca em
grade para cada algoritmo, as fases do trabalho e os materiais de estudo de cada fase.

**`methodological_issues.md`** é o registro crítico do trabalho. Lista os problemas encontrados nos artigos de
referência (como balancear antes de separar o teste), as questões conceituais que o experimento precisou resolver
(como a equivalência entre escalonadores) e também os **erros cometidos durante o próprio trabalho** e como foram
corrigidos. É a matéria-prima da revisão crítica da literatura e da seção de limitações do TCC.

## Como rodar

Requer Python 3.12 e o [uv](https://docs.astral.sh/uv/). Os comandos abaixo são para Windows; em Linux e macOS, troque
`.venv/Scripts/` por `.venv/bin/` e `python.exe` por `python`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
uv pip install --python .venv/Scripts/python.exe -e ".[dev]" --no-deps

.venv/Scripts/python.exe -m pytest                     # testes (cerca de 15 s)
.venv/Scripts/jupyter notebook notebooks/tcc_ml_vs_qml.ipynb
```

Ao abrir o notebook, escolha o ambiente `.venv` como kernel. Executado inteiro, ele leva cerca de 1 min 30 s até a
Seção 4. A simulação quântica usa o Qiskit 2.5 e o Qiskit Aer 0.17; o aprendizado de máquina clássico, o scikit-learn
1.9 e o imbalanced-learn 0.14.
