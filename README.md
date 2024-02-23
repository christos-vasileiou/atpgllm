# atpgllm

GOALS:

Large Language Model for Design Testing and Fault-Modeling:
1. Stuck-at: A specific net is "stuck" at a constant logic value (either 0 or 1).
2. Transition:  Related to signal transitions between different logic levels (e.g., 0 to 1 or 1 to 0).
3. Coupling: ...
4. ...

Ultimate Goal to cover as many Fault Models as possible.

What may go wrong?

- Shorts between two points (bridges), Open in a line, Improper doping, Masking error, Particles on surface, Corrosion, etc.
- Main goal is Safety and Reliability.

## USE

For multi/single-gpu training use:
- `torchrun --proc_per_node=<NODES> script_name.py`: i.e. `torchrun --proc_per_node=4 sequence_multilabel_classification.py`

For gpu/cpu training use:
- `python script_name.py`: i.e. `python sequence_multilabel_classification.py`


## Attention is All you Need

"Attention is All You Need" is a research paper published in 2017 by Google researchers, which introduced the Transformer model, a novel architecture that revolutionized the field of natural language processing (NLP) and became the basis for the LLMs we  now know - such as GPT, PaLM and others. The paper proposes a neural network architecture that replaces traditional recurrent neural networks (RNNs) and convolutional neural networks (CNNs) with an entirely attention-based mechanism. 

The Transformer model uses self-attention to compute representations of input sequences, which allows it to capture long-term dependencies and parallelize computation effectively. The authors demonstrate that their model achieves state-of-the-art performance on several machine translation tasks and outperforms previous models that rely on RNNs or CNNs.

The Transformer architecture consists of an encoder and a decoder, each of which is composed of several layers. Each layer consists of two sub-layers: a multi-head self-attention mechanism and a feed-forward neural network. The multi-head self-attention mechanism allows the model to attend to different parts of the input sequence, while the feed-forward network applies a point-wise fully connected layer to each position separately and identically. 

The Transformer model also uses residual connections and layer normalization to facilitate training and prevent overfitting. In addition, the authors introduce a positional encoding scheme that encodes the position of each token in the input sequence, enabling the model to capture the order of the sequence without the need for recurrent or convolutional operations.

You can read the Transformers paper 
[here](https://arxiv.org/abs/1706.03762)
.



