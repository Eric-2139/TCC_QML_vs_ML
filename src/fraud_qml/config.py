"""Constantes globais do experimento.

Centralizar os valores de protocolo aqui garante que notebook, módulos e testes usem
exatamente os mesmos números: mudar uma decisão experimental é uma alteração em um único
lugar, registrada no git.
"""

import math
from pathlib import Path
from typing import Final

import joblib

# src/fraud_qml/config.py -> parents[2] é a raiz do projeto (funciona com instalação editável)
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_PATH: Final[Path] = PROJECT_ROOT / "creditcard.csv.zip"
ARTIFACTS_DIR: Final[Path] = PROJECT_ROOT / "artifacts"
CACHE_DIR: Final[Path] = ARTIFACTS_DIR / "cache"
RESULTS_DIR: Final[Path] = ARTIFACTS_DIR / "results"
MODELS_DIR: Final[Path] = ARTIFACTS_DIR / "models"

# Semente passada explicitamente a cada componente estocástico (split, undersampling,
# inicialização dos circuitos, simulador). Semente global não basta: cada biblioteca
# tem o próprio gerador.
RANDOM_STATE: Final[int] = 42

# Sementes da análise de robustez (Seção 9). A primeira é a do experimento principal;
# as demais só repetem o pipeline final, sem influenciar nenhuma decisão.
UNDERSAMPLING_SEEDS: Final[tuple[int, ...]] = (RANDOM_STATE, 7, 123, 2024, 31415)

# Divisão 60/20/20 feita na base inteira, ANTES do undersampling, para que validação e
# teste preservem a proporção real de fraudes.
TRAIN_SIZE: Final[float] = 0.6
VALIDATION_SIZE: Final[float] = 0.2
TEST_SIZE: Final[float] = 0.2

# Orçamento comum de qubits entre as variações de codificação: a amplitude comporta
# 2**5 = 32 valores (31 features + 1 constante), e o ângulo usa 5 componentes.
N_QUBITS: Final[int] = 5

# Opções do grid sustentadas pela exploração do treino (Seção 4.11 do notebook).
# Larguras da faixa de ângulos [0, largura]: no máximo π/2, porque os feature maps do Qiskit têm
# período π em x e, acima disso, extremos opostos voltam a se parecer (CC11).
ANGLE_RANGE_WIDTH_GRID: Final[tuple[float, ...]] = (0.25, 0.5, 1.0, math.pi / 2)
# Constantes de preenchimento da amplitude: zero fica de fora porque apaga a norma do vetor (CC4). O 5
# entrou quando a varredura foi ampliada: com só 0, 1 e 3 testados, o melhor valor estava na borda; no treino, o KTA sobe até 5
# e volta a cair em 10 (a constante age como largura de banda).
AMPLITUDE_PADDING_GRID: Final[tuple[float, ...]] = (1.0, 3.0, 5.0)

# Repetições do bloco de codificação nos mapas Z e ZZ. Com 1, o mapa Z é a codificação de fase pura
# (kernel Π cos²(Δ), o mesmo da rotação em Y); com 2, o kernel muda. Por isso as duas entram no grid.
FEATURE_MAP_REPS_GRID: Final[tuple[int, ...]] = (1, 2)

# joblib.cpu_count respeita limites de contêiner; os.cpu_count enxergaria a máquina hospedeira.
N_JOBS: Final[int] = joblib.cpu_count()
