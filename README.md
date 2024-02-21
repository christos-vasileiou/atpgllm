# Large Language Model for design testing

___
___

## GOALS
___

Design for testing and Fault-Modeling:
1. Stuck-at: A specific net is "stuck" at a constant logic value (either 0 or 1).
2. Transition:  Related to signal transitions between different logic levels (e.g., 0 to 1 or 1 to 0).
3. Coupling: ...
4. ...

Ultimate Goal to cover as many Fault Models as possible.

What may go wrong?

- Shorts between two points (bridges), Open in a line, Improper doping, Masking error, Particles on surface, Corrosion, etc.
- Main goal is Safety and Reliability.

## USE
___

For multi-gpu training use:
- `torchrun --proc_per_node=<NODES> script_name.py`: i.e. `torchrun --proc_per_node=4 sequence_multilabel_classification.py`

For gpu/cpu training use:
- `python script_name.py`: i.e. `python sequence_multilabel_classification.py`


## Attention is All you Need
___

"Attention is All You Need" is a research paper published in 2017 by Google researchers, which introduced the Transformer model, a novel architecture that revolutionized the field of natural language processing (NLP) and became the basis for the LLMs we  now know - such as GPT, PaLM and others. The paper proposes a neural network architecture that replaces traditional recurrent neural networks (RNNs) and convolutional neural networks (CNNs) with an entirely attention-based mechanism. 

The Transformer model uses self-attention to compute representations of input sequences, which allows it to capture long-term dependencies and parallelize computation effectively. The authors demonstrate that their model achieves state-of-the-art performance on several machine translation tasks and outperforms previous models that rely on RNNs or CNNs.

The Transformer architecture consists of an encoder and a decoder, each of which is composed of several layers. Each layer consists of two sub-layers: a multi-head self-attention mechanism and a feed-forward neural network. The multi-head self-attention mechanism allows the model to attend to different parts of the input sequence, while the feed-forward network applies a point-wise fully connected layer to each position separately and identically. 

The Transformer model also uses residual connections and layer normalization to facilitate training and prevent overfitting. In addition, the authors introduce a positional encoding scheme that encodes the position of each token in the input sequence, enabling the model to capture the order of the sequence without the need for recurrent or convolutional operations.

You can read the Transformers paper 
[here](https://arxiv.org/abs/1706.03762)
.

## Scaling Instruct Models
___

This [paper](https://arxiv.org/abs/2210.11416) introduces FLAN (Fine-tuned LAnguage Net), an instruction finetuning method, and presents the results of its application. The study demonstrates that by fine-tuning the 540B PaLM model on 1836 tasks while incorporating Chain-of-Thought Reasoning data, FLAN achieves improvements in generalization, human usability, and zero-shot reasoning over the base model. The paper also provides detailed information on how each these aspects was evaluated.

## Reinforcement Learning from Human Feedback (RLHF)
___

Maybe you've heard about this technique but you haven't completely understood it, especially the PPO part. This explanation might help.

We will focus on text-to-text language models, such as GPT-3, BLOOM, and T5. Models like BERT, which are encoder-only, are not addressed. 

Reinforcement Learning from Human Feedback (RLHF) has been successfully applied in ChatGPT, hence its major increase in popularity.

RLHF is especially useful in two scenarios:
- You can’t create a good loss function 
  - Example: how do you calculate a metric to measure if the model’s output was funny? 
- You want to train with production data, but you can’t easily label your production data
  - Example: how do you get labeled production data from ChatGPT? Someone needs to write the correct answer that ChatGPT should have answered
 
RLHF algorithm:
1. Pretraining a language model (LM)
2. Training a reward model
3. Fine-tuning the LM with RL
 
### 1 - Pretraining a language model (LM)
In this step, you need to either train one language model from scratch or just use a pretrained one like GPT-3. 

Once you have that pretrained language model, you can also do an extra optional step, called Supervised Fine-Tuning (STF). 
This is nothing more than getting some human-labeled (input, output) text pairs and fine-tuning the language model you have. 
STF is considered high-quality initialization for RLHF.

At the end of this step, we end up with our trained LM which is our main model, and the one we want to train further with RLHF.

<p align="center">
<img width="249" alt="image" src="https://user-images.githubusercontent.com/17574157/213031414-5dc93741-0344-45b9-9503-a769a286ee3c.png">
</p>
<p align="center">
    <em>Figure 1: Our pretrained language model.</em>
</p>

### 2 - Training a reward model
In this step, we are interested in collecting a dataset of (input text, output text, reward) triplets.

In Figure 2, there's a representation of the data collection pipeline: using input text data (if production data, better), pass it through your model, and have a human attribute a reward to the generated output text. 

<p align="center">
<img width="1100" alt="image" src="https://user-images.githubusercontent.com/17574157/213032348-8a0e89ce-8a46-45fd-9da8-f43bbfbc4844.png">
</p>
<p align="center">
    <em>Figure 2: Pipeline to collect data for reward model training.</em>
</p>

The reward is usually an integer between 0-5, but it can be a simple 0/1 in a 👍/👎 experience.

<p align="center">
<img width="665" alt="image" src="https://user-images.githubusercontent.com/17574157/213033812-0a1a92ba-e18b-47c9-a7a2-df434c7a9412.png">
</p>
</p>
<p align="center">
    <em>Figure 3: Simple 👍/👎 reward collection in ChatGPT. </em>
</p>

<p align="center">
<img width="717" alt="image" src="https://user-images.githubusercontent.com/17574157/213033023-97baaa9c-8188-46b6-a559-7b0aab0d4571.png">
</p>
<p align="center">
    <em>Figure 4: A more complete reward collection experience: the model outputs two texts and the human has to choose which one was better, and also give an overall rating with comments. </em>
</p>

With this new dataset, we will train another language model to receive the (input, output) text and return a reward scalar! This will be our reward model.

The main objective here is to use the reward model to mimic the human's reward labeling and therefore be able to do RLHF training offline, without the human in the loop.
<p align="center">
<img width="206" alt="image" src="https://user-images.githubusercontent.com/17574157/213034158-47b8245f-6d67-460b-a42e-8ada69a1dc41.png">
</p>
<p align="center">
    <em>Figure 5: The trained reward model, that will mimic the rewards given by humans.</em>
</p>

### 3 - Fine-tuning the LM with RL
It's in this step that magic really happens and RL comes into play.

The objective of this step is to use the rewards given by the reward model to train the main model, your trained LM. 
However, since the reward will not be differentiable, we will need to use RL to be able to construct a loss that we can backpropagate to the LM.

<p align="center">
<img width="1162" alt="image" src="https://user-images.githubusercontent.com/17574157/213037462-5dd556de-3afa-4842-b546-0fc90e799249.png">
</p>
</p>
<p align="center">
    <em>Figure 6: Fine-tuning the main LM using the reward model and the PPO loss calculation.</em>
</p>

At the beginning of the pipeline, we will make an exact copy of our LM and freeze its trainable weights. 
This copy of the model will help to prevent the trainable LM from completely changing its weights and starting outputting gibberish text to fool the reward model.

That is why we calculate the KL divergence loss between text output probabilities of both the frozen and non-frozen LM. 

This KL loss is added to the reward that is produced by the reward model. 
Actually, if you are training your model while in production (online learning), you can replace this reward model with the human reward score directly. 💡

Having your reward and KL loss, we can now apply RL to make the reward loss differentiable. 

Why isn't the reward differentiable? Because it was calculated with a reward model that received text as input. This text is obtained by decoding the output log probabilities of the LM. This decoding process is non-differentiable.

To make the loss differentiable, finally Proximal Policy Optimization (PPO) comes into play! Let's zoom in.

<p align="center">
<img width="915" alt="image" src="https://user-images.githubusercontent.com/17574157/217191126-c705fa00-b97d-4537-9e1d-c6ee94decbe5.png">
</p>
<p align="center">
    <em>Figure 7: Zoom-in on the RL Update box - PPO loss calculation.</em>
</p>

The PPO algorithm calculates a loss (that will be used to make a small update on the LM) like this:
1. Make "Initial probs" equal to "New probs" to initialize.
2. Calculate a ratio between the new and initial output text probabilities.
3. Calculate the loss given the formula `loss = -min(ratio * R, clip(ratio, 0.8, 1.2) * R)`, where `R` is the `reward + KL` (or a weighted average like `0.8 * reward + 0.2 * KL`) previously computed and `clip(ratio, 0.8, 1.2)` is just bounding the ratio to be `0.8 <= ratio <= 1.2`. Note that 0.8/1.2 are just commonly used hyperparameter values that are simplified here. Also not that we want to maximize the reward, that's why we add the minus `-`, so that we minimize the negation of the loss with gradient descent.
4. Update the weights of the LM by backpropagating the loss. 
5. Calculate the "New probs" (i.e., new output text probabilities) with the newly updated LM.
6. Repeat from step 2 up to N times (usually, N=4). 

That's it, this is how you use RLHF in text-to-text language models!

Things can get more complicated because there are also other losses that you can add to this base loss that I presented, but this is the core implementation.

Here you can find the original [paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/d5e2c0adad503c91f91df240d0cd4e49-Paper.pdf)
and more information, tutorials and online courses:
1. [Neptune](https://neptune.ai/blog/best-reinforcement-learning-tutorials-examples-projects-and-courses)
2. [OpenAI Spinning-UP](https://spinningup.openai.com/en/latest/user/algorithms.html#algorithms)

## KL divergence for RLHF 

KL-Divergence, or Kullback-Leibler Divergence, is a concept often encountered in the field of reinforcement learning, particularly when using the Proximal Policy Optimization (PPO) algorithm. It is a mathematical measure of the difference between two probability distributions, which helps us understand how one distribution differs from another. In the context of PPO, KL-Divergence plays a crucial role in guiding the optimization process to ensure that the updated policy does not deviate too much from the original policy.

In PPO, the goal is to find an improved policy for an agent by iteratively updating its parameters based on the rewards received from interacting with the environment. However, updating the policy too aggressively can lead to unstable learning or drastic policy changes. To address this, PPO introduces a constraint that limits the extent of policy updates. This constraint is enforced by using KL-Divergence.

To understand how KL-Divergence works, imagine we have two probability distributions: the distribution of the original LLM, and a new proposed distribution of an RL-updated LLM. KL-Divergence measures the average amount of information gained when we use the original policy to encode samples from the new proposed policy. By minimizing the KL-Divergence between the two distributions, PPO ensures that the updated policy stays close to the original policy, preventing drastic changes that may negatively impact the learning process.

A library that you can use to train transformer language models with reinforcement learning, using techniques such as PPO, is TRL (Transformer Reinforcement Learning). In 
this [link](https://huggingface.co/blog/trl-peft) you can read more about this library, and its integration with PEFT (Parameter-Efficient Fine-Tuning) methods, such as LoRA (Low-Rank Adaption). The image shows an overview of the PPO training setup in TRL.


