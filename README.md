# simpleGPT (Generative Pre-trained Transformer) Project

This project implements a simple GPT model with support for multiple custom tokenizations, including Byte Pair Encoding (BPE) and character-level tokenization. Additionally, it implements a decoder-only transformer architecture with self-attention from scratch, inspired by the "Attention is All You Need" research paper.

## Features

- Custom tokenization support: Choose between BPE and character-level tokenization for text input.
- Decoder-only transformer: Implements a transformer architecture without encoder layers for text generation.
- Text generation: Generates text similar to the training data once trained.

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository:

```
git clone <repository-url>
```

2. Dependencies: tinytorchutil Package

This project makes use of `tinytorchutil` ([Git-Repo](https://github.com/Chainathan/tiny-torch-util)), a personal toy package containing a collection of utility functions found useful. Install it using pip:

```bash
pip install tinytorchutil
```

3. Choose your preferred tokenization method (BPE or character-level) and adjust the configuration accordingly.

4. Train the GPT model using your dataset.

5. Once trained, you can use the model to generate text similar to the training data.

## Acknowledgments:

This project was inspired by Andrej Karpathy's NLP teachings and is aimed to better the understanding of relevant concepts.
