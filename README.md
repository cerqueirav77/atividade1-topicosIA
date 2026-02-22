# Scaled Dot-Product Attention — LAB P1-01

## Descrição

O mecanismo de **Scaled Dot-Product Attention** é o bloco central da arquitetura Transformer, proposta no paper *"Attention Is All You Need"* (Vaswani et al., 2017). Ele permite que o modelo pondere dinamicamente quais partes da sequência de entrada são mais relevantes para calcular a representação de cada posição.

O mecanismo opera sobre três matrizes: **Query (Q)**, **Key (K)** e **Value (V)**. A ideia intuitiva é que cada Query "consulta" todas as Keys para descobrir quão compatível ela é com cada uma, e usa essa compatibilidade como peso para combinar os Values correspondentes.

A fórmula que governa o cálculo é:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

## Como Rodar

**Pré-requisitos:** Python 3.x e NumPy.
```bash
pip install numpy
python3 test_attention.py
```

## Explicação do Scaling Factor (√dₖ)

Quando a dimensão das chaves (`dₖ`) é grande, o produto escalar `QKᵀ` tende a crescer em magnitude. Valores muito grandes empurram o softmax para regiões de saturação, onde os gradientes ficam próximos de zero — prejudicando o treinamento.

Dividir por `√dₖ` mantém a variância dos scores próxima de 1, independente da dimensão do embedding, estabilizando o comportamento do softmax.

## Exemplo de Input / Output
```python
Q = [[1., 0., 1., 0.],
     [0., 1., 0., 1.],
     [1., 1., 0., 0.]]

K = [[1., 0., 1., 0.],
     [0., 1., 0., 1.],
     [1., 1., 1., 1.]]

V = [[1., 0.],
     [0., 1.],
     [1., 1.]]

# Output:
# [[0.8446376  0.5776812]
#  [0.5776812  0.8446376]
#  [0.72593138 0.72593138]]
```

## Referência

Vaswani, A. et al. **Attention Is All You Need**, 2017. https://arxiv.org/abs/1706.03762

## Auxiliado por

Implementação desenvolvida com auxílio do Claude (Anthropic) como ferramenta de suporte ao aprendizado. Executei cada passo, entendendo o que estava fazendo e tomando as decisões.