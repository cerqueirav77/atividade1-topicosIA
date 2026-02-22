# -*- coding: utf-8 -*-
"""
Testes de validação numérica para o mecanismo de Scaled Dot-Product Attention.
"""

import numpy as np
import numpy.testing as npt
from attention import scaled_dot_product_attention, softmax

# ── Matrizes de entrada fixas ──────────────────────────────────────────────────
Q = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
], dtype=np.float64)

K = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
], dtype=np.float64)

V = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
], dtype=np.float64)


def calcular_resultado_esperado() -> tuple:
    """Calcula manualmente o resultado esperado para comparação."""
    dimension_k = K.shape[1]
    scaling_factor = np.sqrt(dimension_k)
    scores = Q @ K.T
    scaled_scores = scores / scaling_factor
    attention_weights_esperado = softmax(scaled_scores)
    output_esperado = attention_weights_esperado @ V
    return output_esperado, attention_weights_esperado


def test_pesos_somam_um() -> None:
    """Verifica que cada linha dos attention_weights soma exatamente 1.0."""
    _, attention_weights = scaled_dot_product_attention(Q, K, V)
    row_sums = attention_weights.sum(axis=1)
    npt.assert_array_almost_equal(
        row_sums,
        np.ones(Q.shape[0]),
        decimal=6,
        err_msg="FALHOU: As linhas dos attention_weights não somam 1.0",
    )
    print("  [PASSED] test_pesos_somam_um")


def test_shape_do_output() -> None:
    """Verifica que o output possui shape (n_queries, d_values)."""
    output, _ = scaled_dot_product_attention(Q, K, V)
    shape_esperada = (Q.shape[0], V.shape[1])
    assert output.shape == shape_esperada, (
        f"FALHOU: shape esperada {shape_esperada}, obtida {output.shape}"
    )
    print("  [PASSED] test_shape_do_output")


def test_corretude_numerica() -> None:
    """Compara o output com o cálculo manual esperado."""
    output, _ = scaled_dot_product_attention(Q, K, V)
    output_esperado, _ = calcular_resultado_esperado()
    npt.assert_array_almost_equal(
        output,
        output_esperado,
        decimal=6,
        err_msg="FALHOU: O output não bate com o cálculo manual esperado",
    )
    print("  [PASSED] test_corretude_numerica")


def test_validacao_entradas_invalidas() -> None:
    """Verifica que entradas inválidas levantam ValueError."""
    try:
        scaled_dot_product_attention(np.array([1.0, 2.0]), K, V)
        print("  [FAILED] test_validacao_entradas_invalidas — deveria ter levantado ValueError")
    except ValueError:
        print("  [PASSED] test_validacao_entradas_invalidas")


if __name__ == "__main__":
    print("=" * 55)
    print("  LAB P1-01 — Victor Cerqueira")
    print("=" * 55)

    print("\n── Entradas ──────────────────────────────────────────")
    print(f"Q =\n{Q}\n")
    print(f"K =\n{K}\n")
    print(f"V =\n{V}\n")

    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    print("── Saídas ────────────────────────────────────────────")
    print(f"Attention Weights =\n{attention_weights}\n")
    print(f"Output =\n{output}\n")

    print("── Testes ────────────────────────────────────────────")
    test_pesos_somam_um()
    test_shape_do_output()
    test_corretude_numerica()
    test_validacao_entradas_invalidas()

    print("=" * 55)
