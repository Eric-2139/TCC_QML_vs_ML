"""Feature maps: os circuitos que transformam uma linha de dados num estado quântico."""

from typing import Final, Literal

from qiskit import QuantumCircuit
from qiskit.circuit.library import z_feature_map, zz_feature_map

AngleFeatureMapName = Literal["z", "zz"]
Entanglement = Literal["linear", "full"]

DEFAULT_FEATURE_MAP_REPS: Final[int] = 2


def build_angle_feature_map(
    name: AngleFeatureMapName,
    n_qubits: int,
    reps: int = DEFAULT_FEATURE_MAP_REPS,
    entanglement: Entanglement = "linear",
) -> QuantumCircuit:
    """Cria um feature map de rotação (uma feature por qubit).

    "Ângulo", nos nomes deste pacote, é a família das codificações em que a feature vira o ângulo
    de uma rotação, em oposição à amplitude. A rigor, os dois mapas são **codificação de fase**:
    Hadamard (leva o qubit ao equador da esfera de Bloch) seguida de porta de fase P(2x) (rotação
    em torno de Z). A codificação de ângulo propriamente dita é a rotação em Y, RY(x).

    Para um kernel, a diferença não existe com ``reps=1``: a fidelidade só depende do ângulo
    entre os dois estados, e percorrer o equador (fase) ou um meridiano (RY) dá o mesmo kernel,
    Π cos²(Δ). Com ``reps=2`` (o padrão) o kernel já é outro. Num modelo que **mede** o estado
    (classificador variacional), fase e RY não são equivalentes.

    - ``z``: rotações por qubit, **sem emaranhamento**. O estado é um produto de estados de
      1 qubit, então o kernel é o produto de kernels de 1 qubit: calculável classicamente.
    - ``zz``: acrescenta termos ZZ entre pares de qubits (emaranhamento), que codificam
      produtos de features (π − xᵢ)(π − xⱼ). É o mapa de Havlíček et al. (2019).

    Os dois usam o ``alpha = 2`` padrão do Qiskit: a rotação é P(2x), com período π em x. No mapa
    ``z`` a similaridade cai monotonicamente com a distância até π/2 (``DEFAULT_ANGLE_RANGE``) e,
    em π, valores opostos viram o mesmo estado. No ``zz`` o termo de pares varia mais rápido e a
    similaridade deixa de ser monotônica antes disso (≈ 0,6 rad na configuração padrão): nele a
    largura da faixa é um hiperparâmetro a escolher por validação cruzada, não um limite teórico.

    Args:
        name: Tipo do mapa.
        n_qubits: Número de qubits (= número de features).
        reps: Quantas vezes o bloco de codificação se repete.
        entanglement: Pares acoplados no ``zz`` (vizinhos ou todos); ignorado no ``z``.

    Returns:
        Circuito parametrizado com ``n_qubits`` parâmetros de dados.
    """
    if name == "z":
        return z_feature_map(n_qubits, reps=reps)
    return zz_feature_map(n_qubits, reps=reps, entanglement=entanglement)
