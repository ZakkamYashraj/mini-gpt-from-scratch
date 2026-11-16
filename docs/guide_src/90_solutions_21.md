### Chapter 21

**21.1** Equation (1) of the paper is

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

$Q$, $K$ and $V$ are the tensors `q`, `k` and `v`; $d_k$ is `self.head_size`;
$QK^T/\sqrt{d_k}$ is the line that computes `scores`; the softmax is `F.softmax(scores,
dim=-1)`; and the final product with $V$ is `self.dropout(weights) @ v`. The causal mask
does not appear in the equation itself. The paper describes it in words in Section
3.2.3: connections to later positions are masked out by setting the corresponding
scores to $-\infty$ before the softmax, which is our `masked_fill` line.

**21.2** Three differences you are likely to find:

- **All heads in one tensor.** nanoGPT computes the queries, keys and values of all
  heads with a single Linear layer and reshapes the result to
  `(B, n_head, T, head_size)`, so attention for every head happens in one batched
  operation. Ours loops over separate head objects. Theirs is faster; ours makes the
  structure visible.
- **Built-in fast attention.** When available, it calls PyTorch's
  `scaled_dot_product_attention`, which runs an optimised ("flash") implementation that
  never stores the full $T \times T$ weight matrix. That saves memory and time, at the
  cost of not being able to look at the attention weights, which we want for plotting.
- **Loading pretrained weights.** It can load OpenAI's released GPT-2 weights into the
  model, which is possible only because its layer names and shapes match GPT-2's
  exactly. Ours is designed for clarity and trained only from scratch.

You may also notice that it reads data through memory-mapped files, supports training
on several GPUs, and compiles the model with `torch.compile`. All are about scale and
speed; none changes the model's mathematics.

**21.3** The result is your own. Whatever you build, apply Chapter 15: state the
prediction, change one thing, run two seeds, and report what you measured.
