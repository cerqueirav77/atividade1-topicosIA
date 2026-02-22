# -*- coding: utf-8 -*-
"""
Implementação do mecanismo de Scaled Dot-Product Attention.
Baseado no paper "Attention Is All You Need" (Vaswani et al., 2017).
"""

import numpy as np
from numpy import ndarray


def softmax(matrix: ndarray) -> ndarray:
    """
    Aplica a função softmax linha a linha em uma matriz 2D.

    Utiliza o truque de estabilidade numérica: subtrai o valor máximo
    de cada linha antes de exponenciar, evitando overflow numérico.

    Args:
        matrix (ndarray): Matriz 2D de entrada com shape (n, m).

    Returns:
        ndarray: Matriz com softmax aplicado em cada linha, mesma shape da entrada.
    """
    shifted_values = matrix - np.max(matrix, axis=1, keepdims=True)
    exponentials = np.exp(shifted_values)
    row_sums = np.sum(exponentials, axis=1, keepdims=True)
    return exponentials / row_sums

def scaled_dot_product_attention(
    Q: ndarray, K: ndarray, V: ndarray
) -> tuple[ndarray, ndarray]:
    """
    Calcula o Scaled Dot-Product Attention conforme a fórmula:

        Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V

    Args:
        Q (ndarray): Matriz de Queries com shape (n, dₖ).
        K (ndarray): Matriz de Keys com shape (m, dₖ).
        V (ndarray): Matriz de Values com shape (m, dᵥ).

    Returns:
        tuple:
            - output (ndarray): Resultado da atenção com shape (n, dᵥ).
            - attention_weights (ndarray): Pesos de atenção com shape (n, m).

    Raises:
        ValueError: Se Q, K ou V não forem arrays 2D.
        ValueError: Se as dimensões internas de Q e K forem incompatíveis.
        ValueError: Se o número de linhas de K e V forem incompatíveis.
    """
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q, K e V devem ser arrays 2D.")

    if Q.shape[1] != K.shape[1]:
        raise ValueError(
            f"Dimensão interna incompatível: Q.shape[1]={Q.shape[1]} != K.shape[1]={K.shape[1]}"
        )

    if K.shape[0] != V.shape[0]:
        raise ValueError(
            f"Número de linhas incompatível: K.shape[0]={K.shape[0]} != V.shape[0]={V.shape[0]}"
        )

    dimension_k = K.shape[1]
    scaling_factor = np.sqrt(dimension_k)

    scores = Q @ K.T
    scaled_scores = scores / scaling_factor
    attention_weights = softmax(scaled_scores)
    output = attention_weights @ V

    return output, attention_weights