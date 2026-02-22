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