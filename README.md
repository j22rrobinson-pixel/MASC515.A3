# MASC515.A3
Assignment 3
# MASC 515 Assignment 3: microgpt Enhancements

This repository contains an enhanced version of Andrej Karpathy's `microgpt`, featuring modern Transformer optimizations.

### Implemented Algorithms

1. **GELU (Gaussian Error Linear Units)**
   - **Idea:** A smoother activation function than ReLU. It scales the input by its cumulative distribution function under a Gaussian distribution.
   - **Benefit:** Avoids "dead neurons" and improves convergence in deep networks.

2. **LoRA (Low-Rank Adaptation)**
   - **Idea:** Instead of updating the full weight matrix $W$ during fine-tuning, we only train two small matrices $A$ and $B$ such that $\Delta W = A \times B$.
   - **Benefit:** Dramatically reduces the number of trainable parameters.

3. **RoPE (Rotary Positional Embedding)**
   - **Idea:** Encodes positions by rotating the Query and Key vectors in a multi-dimensional space.
   - **Benefit:** Captures relative distances between tokens naturally and allows for better length extrapolation.

4. **Mixture of Experts (MoE)**
   - **Idea:** Replaces a single large MLP with multiple small "expert" networks. A router decides which expert handles which token.
   - **Benefit:** Increases model capacity (parameters) without increasing the computational cost per token.
