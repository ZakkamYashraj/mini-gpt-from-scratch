## Chapter 21: Where to go next

> **Key ideas**
>
> - You now understand the architecture behind modern language models. What separates
>   this project from the large ones is mostly **scale** (data, parameters, computing
>   power) and a set of engineering refinements, not a different idea.
> - The best next step is a small extension of *this* project that you choose, build and
>   measure yourself.
> - After that: the original papers, which you can now read, and one or two good
>   courses.

### 21.1 What you have, and what is different in the big models

Everything in a production language model has a recognisable ancestor in this
repository: tokenizer, embeddings, causal self-attention, feed-forward layers, residual
connections, normalisation, a language-model head, next-token cross-entropy, AdamW,
sampling. The differences are of degree:

| | This project | GPT-2 small (2019) | Current large models |
|---|---|---|---|
| Parameters | 10.7 million | 124 million | tens to hundreds of billions |
| Layers | 6 | 12 | dozens to over a hundred |
| Context length | 256 characters | 1,024 tokens | hundreds of thousands of tokens |
| Vocabulary | 65 characters | 50,257 BPE tokens | of the order of 100,000 tokens |
| Training text | 1 MB | about 40 GB | a large fraction of the public internet |
| Training hardware | one laptop GPU, 22 minutes | several GPUs, days | thousands of GPUs, months |

There are also real refinements you have not seen: better positional schemes, faster
attention, different normalisation, careful fine-tuning after pre-training to make a
model follow instructions and behave helpfully. None of them changes the picture you
have built; each replaces one box in the diagram of Chapter 12 with an improved one.

### 21.2 Resources

**To consolidate what you learned here**

- **Andrej Karpathy, "Neural Networks: Zero to Hero"** (video series, free online). It
  builds automatic differentiation, then language models of growing complexity, and
  ends with "Let's build GPT: from scratch, in code, spelled out", which covers the
  same ground as Part 2 of this guide in a different voice. Watching it *after* doing
  this project is ideal: you will recognise everything and catch the details.
- **Karpathy's `nanoGPT`** (GitHub). The compact, efficient GPT implementation that
  inspired this project. Reading it is a good test: every line should now make sense,
  including the places where it is faster than ours.
- **Karpathy's `minbpe`** (GitHub). A small, clean BPE tokenizer, close in spirit to
  ours.
- **3Blue1Brown, "Neural networks"** (video series). The best visual intuition for
  gradient descent, backpropagation, and, in its later chapters, attention and
  transformers.
- **Jay Alammar, "The Illustrated Transformer"** (blog post). Attention in pictures.

**The papers.** You are now equipped to read these. Read the abstract, the figures and
the method section first; do not worry about following every experiment.

- Vaswani et al., **"Attention Is All You Need"** (2017). The paper that introduced the
  transformer. Note that it describes an encoder *and* a decoder for translation; a GPT
  is the decoder half alone. Its Section 3.2 is our Chapter 9.
- Radford et al., **"Improving Language Understanding by Generative Pre-Training"**
  (2018) and **"Language Models are Unsupervised Multitask Learners"** (2019): GPT-1 and
  GPT-2. The second is the model this project follows (pre-norm, learned positions,
  weight tying, the 0.02 initialisation).
- Sennrich, Haddow and Birch, **"Neural Machine Translation of Rare Words with Subword
  Units"** (2016): byte-pair encoding for language models.
- Ba, Kiros and Hinton, **"Layer Normalization"** (2016).
- He et al., **"Deep Residual Learning for Image Recognition"** (2015): residual
  connections.
- Kingma and Ba, **"Adam: A Method for Stochastic Optimization"** (2014), and
  Loshchilov and Hutter, **"Decoupled Weight Decay Regularization"** (2017): Adam and
  AdamW.
- Press and Wolf, **"Using the Output Embedding to Improve Language Models"** (2017):
  weight tying.
- Holtzman et al., **"The Curious Case of Neural Text Degeneration"** (2019): why greedy
  decoding fails, and top-p sampling.

**Courses and books, for going deeper**

- **fast.ai, "Practical Deep Learning for Coders"** (free). Top-down and practical.
- **Stanford CS224N, "Natural Language Processing with Deep Learning"** (lectures free
  online). The standard university course on this material.
- **"Dive into Deep Learning"** (d2l.ai, free book). Explanations with runnable code for
  nearly every topic.
- **The official PyTorch tutorials** (pytorch.org/tutorials), when you need to know how
  a particular feature works.

For the mathematics, your own courses are the right source: linear algebra (vectors,
matrices, dot products) and multivariable calculus (gradients, the chain rule) are
exactly the tools used here, and probability will help with everything about
distributions and sampling.

### 21.3 Ideas for extending the project

Pick one. Predict the result before you run it, measure it properly (Chapter 15), and
write down what happened, especially if it was not what you expected.

**Smaller, an afternoon each**

- **Fight the overfitting.** The main run overfits after about 1,750 steps. Try dropout
  0.3 or 0.4, or a smaller model. Can you beat a validation loss of 1.46?
- **Early stopping** (Exercise 13.5) so that training ends by itself.
- **Resume from a checkpoint**: also save the optimizer's state and the step number, and
  add a `--resume` option.
- **A learning-rate sweep**: train the tiny model at six learning rates and plot final
  validation loss against learning rate.
- **More ablations**: depth (2, 4, 6, 8 layers at a fixed parameter count), context
  length, dropout, sinusoidal against learned positions (Exercise 7.5).

**Medium, a weekend each**

- **Batched heads.** Replace the Python loop over heads by one tensor operation of shape
  `(B, n_head, T, T)`. The outputs must be identical to the current model (write that as
  a test), and training should be clearly faster.
- **Fast attention.** Replace the hand-written attention arithmetic by PyTorch's
  `F.scaled_dot_product_attention`, which uses optimised "flash" attention on the GPU.
  Measure the speed and the memory.
- **A KV cache** for generation (Section 14.1): keep the keys and values of previous
  tokens so that each new token costs one position's worth of work, not a whole
  sequence's. The generated text must be unchanged for a fixed seed.
- **A small demo**: a web page, for instance with the Gradio library, with a prompt box
  and sliders for temperature and top-p.

**Larger**

- **Train on Telugu text.** The byte-level BPE tokenizer already handles any script
  (Section 6.4). You need a few megabytes of clean Telugu text in one UTF-8 file. Train
  a BPE vocabulary of 1,000 to 2,000 on it and look at the merges: do they correspond to
  syllables and common words? Expect to need a larger vocabulary than for English,
  because each Telugu character is three bytes. Then compare a character-level model
  with a BPE model in bits per character.
- **Train on your own notes** or on a larger dataset such as TinyStories, and see how
  far a 10-million-parameter model can go when the data is no longer the limit.
- **Modern components**: rotary positional embeddings in place of the learned table,
  for example. Change one, run the ablation, report honestly.

### 21.4 A last word

You began knowing Python and some calculus. You have now followed a language model all
the way down: text to integers, integers to vectors, vectors mixed by attention,
logits, a loss, gradients flowing back through every one of those steps, and ten
million numbers nudged a few thousand times until the output reads like Shakespeare.

None of it turned out to be magic. That is the most valuable thing to take away, and it
remains true for the models a thousand times larger.

### Exercises

**21.1** Read Section 3.2 of "Attention Is All You Need" ("Attention"). Find the
equation that corresponds to our `CausalSelfAttentionHead.forward` and identify each
symbol with a variable in the code.

**21.2** Open `model.py` in the `nanoGPT` repository. List three things it does
differently from this project, and for each say what the difference buys.

**21.3** Choose one extension from Section 21.3, write down your prediction, and build
it.

### Check your understanding

1. Name three ways in which a large production model differs from this one, and one way
   in which it does not.
2. Which half of the original transformer is a GPT?
3. Before running an experiment, what should you write down?
