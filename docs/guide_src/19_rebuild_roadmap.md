# Part 4: Do it yourself

## Chapter 19: Rebuild it from scratch: a roadmap

> **Key ideas**
>
> - Reading code and understanding it is not the same as being able to write it. The
>   only way to know that you can is to do it, starting from an empty folder.
> - Build in small **milestones**. Each one ends with something that runs and with a
>   **check** that tells you it is right before you build on it.
> - Never write more than about twenty lines without running them.
> - When you are stuck, the rule is: look, understand, close the reference, and write it
>   again from memory.

### 19.1 The rules of the game

Create a new, empty folder somewhere outside this repository. That folder is where you
will rebuild the project. The rules:

1. **No copy and paste** from this repository or from anywhere else. Every character is
   typed by you.
2. **You may read explanations** (this guide, the papers) as much as you like. Try to
   keep the *code* of this repository closed.
3. **When stuck for more than about 30 minutes**, open the one function you need, read
   it until you understand every line, close it, and then write it yourself. Looking is
   not cheating. Transcribing is.
4. **Run constantly.** After every few lines, print a shape or a value. A bug found in
   the five lines you just wrote takes a minute; the same bug found after two hundred
   lines takes an evening.
5. **Commit after every milestone**, with a message that says what now works.

It will not be identical to this repository, and it should not be. Different names,
simpler structure, fewer options: all fine. What matters is that it works and that you
can explain every line.

Expect it to take somewhere between 15 and 30 hours, spread over a few weeks. The first
milestones go slowly; that is normal.

### 19.2 The milestones

Each milestone lists a **goal**, some **hints**, and a **check**. Do not move on until
the check passes. The checks are specific numbers on purpose: "it seems to work" is not
a check.

#### M0: An empty project that runs

**Goal.** A folder with a virtual environment, Git, PyTorch installed, and the dataset
downloaded.

**Hints.** `python -m venv .venv`, activate it, install PyTorch. `git init`. Write a
`.gitignore` before the first commit. Download Tiny Shakespeare with `urllib.request`
or by hand.

**Check.** A three-line script prints `torch.__version__`, whether CUDA is available,
and the length of the text: **1,115,394** characters. The first line of the text is
`First Citizen:`.

#### M1: Characters to numbers, and batches

**Goal.** A character tokenizer and a function that returns random training batches.

**Hints.** `sorted(set(text))`, two dictionaries. Encode the whole text into one tensor
of type `torch.long`. Split 90/10 by position. For a batch, pick random starting points
with `torch.randint`; the targets are the inputs shifted by one. (Chapters 5 and 6.)

**Check.** The vocabulary has **65** entries. `decode(encode(s)) == s` for a sample
string. The training split has **1,003,854** tokens. For a batch `x, y` of shape
`(4, 8)`: `x[0, 1:]` equals `y[0, :-1]`. Decode a row of `x` and read it: it should be
English.

#### M2: A bigram model that trains

**Goal.** The simplest language model, trained with a real training loop, generating
text.

**Hints.** One `nn.Embedding(V, V)`. `F.cross_entropy` needs logits of shape
`(B*T, V)`. The five steps: forward, loss, `zero_grad`, `backward`, `step`. For
generation: softmax of the last position's logits, `torch.multinomial`, `torch.cat`.
(Chapter 8.)

**Check.** The loss starts between 4.2 and 4.8 and falls to about **2.5** within 3000
steps. Generated text has plausible letter pairs, spaces and line breaks, and no real
words longer than three or four letters.

This milestone matters more than it looks: the data pipeline, the loss, the loop and
the sampling are now all proven to work. Everything after this changes only the model.

#### M3: One attention head

**Goal.** A module that takes `(B, T, C)` and returns `(B, T, head_size)` using causal
self-attention.

**Hints.** Three `nn.Linear` layers without bias. Scores are `q @ k.transpose(-2, -1)`
divided by the square root of the head size. `torch.tril` for the mask,
`register_buffer` to store it, `masked_fill` with `float("-inf")`, then softmax over
the last dimension. Return the weights as well; you will want them. (Chapters 9
and 10.)

**Check.** Output shape correct. Every row of the weights sums to 1. The weights above
the diagonal are exactly 0. Then the strongest check: load the hand-chosen matrices of
Section 9.4 and reproduce its numbers, `[[1, 2], [2.6089, 3.6089], [3.3457, 5.0185]]`.

#### M4: Prove it is causal

**Goal.** A pytest test that fails if any position can see the future.

**Hints.** Run the head on an input; change the input at position 5; run again. The
outputs before position 5 must be identical (`torch.equal`), the outputs from position
5 onwards must differ. Then break the mask on purpose and watch the test fail.
(Sections 10.6 and 17.8.)

**Check.** The test passes with the mask and fails without it.

#### M5: The transformer block

**Goal.** Multi-head attention, the feed-forward network, layer normalisation, and a
block that combines them with residual connections.

**Hints.** Multi-head: a `nn.ModuleList` of heads, `torch.cat(..., dim=-1)`, a
projection. Feed-forward: Linear to `4C`, GELU, Linear back. LayerNorm: mean and
variance over the last dimension with `keepdim=True`. Block:
`x = x + attn(ln1(x))`, `x = x + ffwd(ln2(x))`. (Chapter 11.)

**Check.** The block's output shape equals its input shape. Your LayerNorm agrees with
`nn.LayerNorm` to within $10^{-5}$ on random input. `n_embd` not divisible by `n_head`
raises an error.

#### M6: The GPT

**Goal.** The complete model: embeddings, a stack of blocks, final LayerNorm, LM head,
loss.

**Hints.** Two embedding tables; `torch.arange(T, device=idx.device)` for the
positions. Initialise weights with standard deviation 0.02. Tie the LM head to the
token embedding. (Chapters 7 and 12.)

**Check.** With `n_embd=96`, 3 layers, 3 heads, `block_size=64`, vocabulary 65 and no
biases, the parameter count is exactly **344,832**. The loss on a random batch, before
any training, is within 0.3 of $\ln(65) = 4.17$. The causality test from M4, applied to
the whole model, still passes. And the overfit test: training on one fixed batch of 4
sequences drives the loss below **0.1** within 300 steps.

If the overfit test fails, do not go on. Something is wrong in the model, and training
on the full dataset will only hide it.

#### M7: Real training

**Goal.** Train the small model on the full dataset and watch the validation loss.

**Hints.** Reuse the M2 loop with AdamW and a constant learning rate of `1e-3`. Add an
evaluation function that averages the loss over 20 batches of each split, with
`model.eval()` and `torch.no_grad()`. Move the model and every batch to the device.
(Chapters 5 and 13.)

**Check.** With the M6 model on a CPU, 2000 steps take a few minutes and the validation
loss falls below **2.0**. The samples contain many real English words and lines that
begin with names followed by a colon.

At this point you have built a GPT. The remaining milestones make it better and turn it
into a project.

#### M8: Controlled generation

**Goal.** Temperature, top-k and top-p, and a script that loads a checkpoint and
continues a prompt.

**Hints.** Divide the logits by the temperature. Top-k: `torch.topk`, then
`masked_fill`. Top-p: sort, softmax, `cumsum`, drop tokens once the mass before them
reaches `p`, then `scatter` back. Save the tokenizer's vocabulary inside the
checkpoint. (Chapter 14.)

**Check.** Unit tests on a hand-made distribution: for probabilities
`(0.5, 0.3, 0.15, 0.05)`, `top_p=0.7` keeps exactly 2 tokens and `top_p=0.9` keeps 3.
Temperature 0 gives the same output on every run.

#### M9: Train it properly

**Goal.** The refinements: a config file, warmup and cosine decay, gradient clipping,
mixed precision, saving the best checkpoint, a CSV log.

**Hints.** Add them **one at a time** and re-run the tiny model after each, so that if
something breaks you know which change did it. (Chapter 13.)

**Check.** With 6 layers, 6 heads, `n_embd=384`, `block_size=256`, batch size 64 and
dropout 0.2, the validation loss reaches about **1.47** within the first 2000 steps on
a GPU, and the text reads like Shakespeare from a distance.

#### M10: Byte-pair encoding

**Goal.** A BPE tokenizer with `train`, `encode`, `decode`, `save` and `load`.

**Hints.** Start from `list(text.encode("utf-8"))`. First write the slow, obvious
version on a short string and get it right. Then split the text into chunks and count
each distinct chunk once to make it fast. Encode by repeatedly merging the pair that
was learned earliest. (Chapter 6.)

**Check.** On `aaabdaaabac` with three merges the tokens are `aaab, d, aaab, a, c`.
Round trips work for empty, accented and Telugu strings. Trained on Tiny Shakespeare
with a vocabulary of 512, the first merges are `' t'` and `'he'`, and the text
compresses to about **2.03** characters per token.

#### M11: Experiments

**Goal.** A loss-curve plot, an attention heatmap, and one ablation.

**Hints.** Read the CSV log and plot both curves. For the heatmap, `plt.imshow` on one
head's weights for a short sentence. For the ablation, train twice with one difference,
the same seed, and the same number of steps. (Chapters 15 and 16.)

**Check.** Your loss curve shows the validation loss turning upward while the training
loss keeps falling. Your heatmaps are zero above the diagonal. Removing the positional
embeddings makes the validation loss clearly worse.

#### M12: Make it a project

**Goal.** Tests, a README, a clean history.

**Hints.** Chapters 17 and 18.

**Check.** `python -m pytest` passes. A friend can clone the repository and train the
tiny model using only the README. You can explain every file without opening it.

### 19.3 A final self-check

When all milestones are done, try to answer these aloud, without notes. The same
questions, with model answers, are in `docs/STUDY_GUIDE.md`.

- Why is the mask a lower triangle, and why $-\infty$?
- What are a query, a key and a value?
- Why divide by the square root of the head size?
- What would happen without positional embeddings? Without residual connections?
- What is tied in weight tying, and why is that possible?
- What does the validation loss tell you that the training loss does not?

If you can, the project is yours.

### Exercises

**19.1** Complete milestones M0 to M2.

**19.2** Complete milestones M3 to M7. This is the core of the rebuild.

**19.3** Complete milestones M8 to M12.

**19.4 (hard)** Choose one thing this repository does *not* have (ideas are in Chapter
21) and add it to your version.

### Check your understanding

1. Why does each milestone end with a specific number to check?
2. Why is the bigram milestone (M2) important even though the model is useless?
3. What should you do if the overfit test fails at M6?
4. Why add the refinements of M9 one at a time?
