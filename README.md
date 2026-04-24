# MASC 515: Enhanced microgpt Implementation
    
    This repository contains the work for Assignment 3, focusing on modern Transformer architectural enhancements. The project evolves from Andrej Karpathy's original dependency-free `microgpt` to a more advanced version featuring state-of-the-art LLM components.
    
    ## Repository Structure
    
    The repository is organized to show the progression from the baseline code to the final optimized model:
    
    * **`microgpt.py`**: The original, unmodified baseline code.
    * **`microgpt4added.py`**: The primary submission file. It contains the core microgpt logic enhanced with GELU, LoRA, RoPE, and Mixture of Experts (MoE).
    * **`microgptoptimized.py`**: A variant focusing on further stability and optimization for the custom autograd engine.
    * **`input.txt`**: The dataset consisting of ~32,000 human names used for training and generation.
    * **`.gitignore`**: Configured to prevent tracking of local environment files and cached data.
    
    ---
    
    ## Technical Enhancements
    
    The **`microgpt4added.py`** script implements four critical features used in modern frontier models (like GPT-4 and Llama 3):
    
    ### 1. GELU Activation
    * **Implementation:** Replaced the standard ReLU with **Gaussian Error Linear Units**.
    * **The Idea:** Unlike ReLU's hard zero-gate, GELU weights inputs by their magnitude via a Gaussian cumulative distribution function.
    * **Benefit:** Provides smoother gradients and better performance in deep networks.
    
    ### 2. LoRA (Low-Rank Adaptation)
    * **Implementation:** Added a low-rank (rank r=4) bypass to the Attention Query projections.
    * **The Idea:** Instead of updating the massive weight matrix W, we learn two smaller matrices A and B where Delta W = A x B.
    * **Benefit:** Drastically reduces trainable parameters while maintaining the model's ability to learn complex patterns.
    
    ### 3. RoPE (Rotary Positional Embedding)
    * **Implementation:** Removed static positional embeddings in favor of rotary embeddings applied during the attention mechanism.
    * **The Idea:** Positions are encoded by rotating the Query and Key vectors in complex space.
    * **Benefit:** Allows the model to capture relative distances more effectively and generalize to longer sequences.
    
    ### 4. Mixture of Experts (MoE)
    * **Implementation:** The MLP block was refactored into a sparse structure with a gating network and two specialized expert networks.
    * **The Idea:** A "router" decides which expert is best suited for a specific token, allowing the model to have more parameters without increasing the compute cost per token.
    
    ---
    
    ## Training & Generation
    
    ### Early Stopping
    The model is configured with an **Early Stopping** mechanism. Training automatically halts when the loss reaches the target threshold of **1.7**, ensuring the model is sufficiently trained to be "accurate" without over-fitting and memorizing the dataset.
    
    ### Temperature-Based Sampling
    To generate high-quality names, the inference loop uses a **Temperature of 0.7**. 
    * **Lower Temperature (0.7):** Makes the model more confident, leading to realistic names like `brilani`, `kona`, and `anah`.
    * **Higher Temperature:** Increases creativity but can lead to more "gibberish" or abstract sounds.
    

    
