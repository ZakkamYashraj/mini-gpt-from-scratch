# Appendix C: Glossary

The number in brackets is the chapter where the term is introduced.

**Ablation** [15]. An experiment that removes or changes one component of a system to
measure how much that component contributes.

**Activation function** [4]. A fixed, curved function applied after a layer's weighted
sum (tanh, ReLU, GELU). Without it, stacked layers collapse into a single linear map.

**Adam / AdamW** [5]. Optimizers that give every parameter its own step size, based on
running averages of its recent gradients. AdamW adds correctly applied weight decay.

**Array** [2]. A grid of numbers with a shape. NumPy's central object.

**Attention** [9]. See *self-attention*.

**Attention weights** [9]. For each token, the positive numbers (summing to 1) that say
how much it takes from every other token. Computed from the input, not learned
directly.

**Autograd** [3]. PyTorch's system for computing derivatives automatically, by recording
operations and applying the chain rule backwards.

**Autoregressive** [1]. Producing a sequence one element at a time, each new element
depending on all those before it.

**Backpropagation** [4]. The algorithm that computes the gradient of the loss with
respect to every parameter by applying the chain rule from the loss backwards through
the network.

**Baseline** [8]. A simple model whose score a more complex model must beat for its
complexity to be justified.

**Batch** [5]. A small group of training examples processed together in one step.

**Batch size** [5]. The number of sequences in a batch (`B`).

**Bias** [4]. A learned constant added to a neuron's weighted sum.

**Bigram model** [8]. A language model that predicts the next token from the current
token alone.

**Bits per character** [15]. Cross-entropy expressed in bits and divided by the number
of characters each token covers; a loss that can be compared across tokenizers.

**Block** [11]. One unit of a transformer: multi-head attention and a feed-forward
network, each with layer normalisation and a residual connection.

**Block size** [5]. See *context length*.

**BPE (byte-pair encoding)** [6]. A tokenizer that starts from single bytes and
repeatedly merges the most frequent adjacent pair into a new token.

**Broadcasting** [2]. The rule by which arrays of different shapes are combined, by
treating the smaller one as if it were repeated.

**Buffer** [10]. A tensor stored in a module that is part of its state but is not
learned (for example the causal mask).

**Causal mask** [10]. A lower-triangular matrix that forbids each token from attending
to later tokens.

**Chain rule** [4]. The calculus rule for the derivative of a composition of functions:
multiply the derivatives of the steps.

**Checkpoint** [5]. A file holding a model's parameters (and here its config and
tokenizer) so that it can be used later.

**Commit** [18]. A recorded snapshot of a project in Git, with a message describing the
change.

**Computation graph** [3]. The record PyTorch keeps of which operations produced which
tensors, used by autograd.

**Context** [1]. The tokens the model is given, from which it predicts the next one.

**Context length** [5]. The maximum number of tokens the model can see at once
(`block_size`, `T`).

**Cosine similarity** [7]. The dot product of two vectors after scaling each to length
1; it measures how closely their directions agree, from -1 to 1.

**Cross-entropy** [1]. The loss for classification: minus the logarithm of the
probability the model gave to the correct answer.

**CUDA** [3]. NVIDIA's system for running computations on its GPUs; `"cuda"` is
PyTorch's name for the GPU device.

**Decoder-only** [10]. A transformer that consists only of causally masked blocks, as
all GPT models do.

**Device** [3]. Where a tensor lives and is computed: the CPU or a GPU.

**Dimension (axis)** [2]. One of the directions along which an array extends.

**Dot product** [2]. The sum of the products of matching entries of two vectors; large
when the vectors point the same way.

**Dropout** [5]. A regulariser that sets a random fraction of values to zero during
training.

**dtype** [3]. The kind of number a tensor holds, such as `float32` or `int64`.

**Embedding** [7]. A learned vector that represents a token (or a position).

**Embedding dimension** [7]. The length of those vectors (`n_embd`, `C`).

**Epoch** [5]. One complete pass over the training data.

**Feed-forward network** [11]. The two-layer network inside each block, applied to each
position separately.

**float16 / float32** [13]. Formats for decimal numbers using 16 or 32 bits.

**Forward pass** [3]. Computing the model's output (and the loss) from its input.

**GELU** [4]. A smooth activation function resembling ReLU; the one used in GPT models.

**Generalisation** [5]. Performing well on data the model was not trained on.

**GPU** [3]. A processor with thousands of simple cores, well suited to matrix
arithmetic.

**Gradient** [3]. The vector of partial derivatives of a function with respect to all
its inputs; it points in the direction of steepest increase.

**Gradient clipping** [13]. Shrinking the gradient when its length exceeds a threshold.

**Gradient descent** [4]. Lowering a function by repeatedly stepping against its
gradient.

**Greedy decoding** [14]. Always choosing the single most likely next token.

**Head** [9, 11]. One attention unit with its own query, key and value matrices.
*Multi-head attention* runs several in parallel.

**Head size** [9]. The width of a head's query, key and value vectors.

**Hidden layer** [4]. A layer between a network's input and output.

**Hyperparameter** [5]. A setting chosen by the person training the model (learning
rate, number of layers), as opposed to a learned parameter.

**Initialisation** [12]. The choice of starting values for the parameters.

**Key** [9]. In attention, the vector by which a token advertises what it contains.

**KV cache** [14]. Storing the keys and values of earlier tokens during generation to
avoid recomputing them.

**Language model** [1]. A model that assigns probabilities to the next token given the
preceding ones.

**Layer** [4]. A set of neurons computing together; in practice one matrix
multiplication plus an activation.

**Layer normalisation** [11]. Rescaling each token vector to mean 0 and variance 1,
followed by a learned gain and bias.

**Learning rate** [4]. The size of each step in gradient descent.

**Learning-rate schedule** [13]. A rule that changes the learning rate during training;
here, warmup followed by cosine decay.

**LM head** [12]. The final Linear layer that turns each vector into one score per
vocabulary token.

**Logits** [2]. A model's raw scores, before softmax turns them into probabilities.

**Loss** [1]. A single number measuring how wrong the model's predictions are.

**Lower-triangular matrix** [10]. A matrix whose entries above the diagonal are zero.

**Matrix multiplication** [2]. The operation `@`: each entry of the result is the dot
product of a row of the first matrix with a column of the second.

**Mean squared error** [4]. A loss for predicting numbers: the average squared
difference between prediction and target.

**Merge** [6]. One step of BPE: a pair of tokens that becomes a single new token.

**Mixed precision** [13]. Using 16-bit numbers for most computations and 32-bit numbers
where accuracy is needed.

**Momentum** [5]. Using a running average of recent gradients, not just the current
one, to choose the update.

**nan** [5, 20]. "Not a number": the result of an undefined computation, such as
infinity minus infinity.

**Neuron** [4]. A unit computing an activation function of a weighted sum of its inputs
plus a bias.

**`nn.Module`** [3]. PyTorch's base class for anything that has parameters and a forward
computation.

**Normal distribution** [12]. The bell-shaped distribution; used to draw initial
weights.

**Nucleus** [14]. In top-p sampling, the smallest set of most likely tokens whose
probabilities add up to at least `p`.

**One-hot vector** [7]. A vector of zeros with a single 1, marking one item out of
many.

**Optimizer** [5]. The rule that turns gradients into parameter updates.

**Overfitting** [5]. Memorising the training data: training loss keeps improving while
validation loss does not.

**Parameter** [1]. A number inside the model that training adjusts. Also called a
weight.

**Partial derivative** [3]. The rate of change of a function with respect to one of its
variables, the others held fixed.

**Perplexity** [8]. $e^{\text{loss}}$: the number of equally likely options that would
give the same loss.

**Positional embedding** [7]. A learned vector for each position in the sequence, added
to the token embedding.

**Pre-norm** [11]. Applying layer normalisation before each sub-layer, inside the
residual branch.

**Probability distribution** [1]. A list of non-negative numbers that sum to 1.

**Prompt** [1]. The text a generation starts from.

**Query** [9]. In attention, the vector by which a token expresses what it is looking
for.

**Regularisation** [5]. Any technique that discourages memorisation (dropout, weight
decay).

**ReLU** [4]. The activation $\max(0, z)$.

**Repository** [18]. A folder whose history is tracked by Git.

**Residual connection** [11]. Adding a layer's output to its input, $x + f(x)$.

**Residual stream** [11]. The sequence of vectors that flows through the model and to
which every block adds its contribution.

**Round trip** [17]. Encoding followed by decoding, which must return the original.

**Sampling** [1, 14]. Choosing a token at random according to the model's
probabilities.

**Seed** [5]. The starting value of the random-number generator; fixing it makes a run
repeatable.

**Self-attention** [9]. The operation in which every token computes weights over the
tokens of the same sequence and takes a weighted average of their values.

**Shape** [2]. The size of an array along each of its dimensions.

**Softmax** [2]. The function that turns any list of numbers into a probability
distribution, by exponentiating and normalising.

**Stochastic gradient descent** [5]. Gradient descent using gradients estimated from
random batches.

**Temperature** [14]. A number that divides the logits before sampling; lower is more
cautious, higher more random.

**Tensor** [3]. PyTorch's array: like a NumPy array, but able to live on a GPU and to
record its history for autograd.

**Token** [1]. One of the pieces into which text is cut.

**Token ID** [1]. The integer that stands for a token.

**Tokenizer** [6]. The component that converts text to token IDs and back.

**Top-k sampling** [14]. Sampling only from the `k` most likely tokens.

**Top-p sampling** [14]. Sampling only from the nucleus.

**Training set** [5]. The part of the data the model learns from.

**Transformer** [11, 12]. A neural network built from blocks of self-attention and
feed-forward layers.

**Underfitting** [5]. Failing to learn even the training data well.

**UTF-8** [6]. The standard way of representing text as bytes.

**Validation set** [5]. Data held back from training and used to measure
generalisation.

**Value** [9]. In attention, the vector a token hands over to the tokens that attend to
it.

**Vanishing gradient** [11]. Gradients shrinking towards zero as they pass back through
many layers.

**Vocabulary** [1]. The fixed list of all tokens a model knows.

**Warmup** [13]. The initial period in which the learning rate rises from near zero.

**Weight** [4]. See *parameter*.

**Weight decay** [5]. Shrinking weights slightly at every step, as a regulariser.

**Weight tying** [12]. Using the same matrix for the token embedding and the LM head.
