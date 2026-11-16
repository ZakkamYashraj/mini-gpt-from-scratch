### Chapter 20

**20.1** (a) The inner sizes are 64 and 384: a layer that expects vectors of width 384
received vectors of width 64. In this model that is the classic sign of feeding the
output of a *single head* (width 64) into the projection that expects the concatenation
of all six heads (width 384). Fix: concatenate the heads' outputs first. (b) One tensor
is on the CPU and another on the GPU; move the batch (or the prompt, or a tensor created
inside `forward`) to the model's device. (c) A token ID, or a position, has no row in an
embedding table: the vocabulary size does not match the tokenizer, or the sequence is
longer than `block_size`.

**20.2** Measured on the tiny model (training loss at steps 0, 100, 200, 300):

| version | step 0 | step 100 | step 200 | step 300 | loud or silent? |
|---|---|---|---|---|---|
| correct | 4.20 | 2.51 | 2.39 | 2.36 | |
| no `zero_grad()` | 4.20 | 3.52 | 4.07 | 4.33 | silent |
| no `optimizer.step()` | 4.20 | 4.20 | 4.21 | 4.20 | silent |
| targets not shifted (`y = x`) | 3.28 | 0.05 | 0.01 | 0.01 | silent |

All three are silent: no error message in any of them. Without `zero_grad()` each step
uses the sum of every gradient computed so far, which soon points nowhere useful; the
loss improves a little and then gets worse. Without `step()` nothing ever changes.
Without the shift, the task becomes "repeat the token you are looking at", which is
trivial, so the loss collapses to nearly zero: the "too good to be true" signature of
Section 20.7. Notice that even the *starting* loss is low (3.28 instead of 4.2): with
tied weights, an untrained model already leans towards predicting its own input token.

**20.3** If `self.blocks` is a plain Python list, PyTorch does not know the blocks
exist. (PyTorch even refuses to *replace* a registered `ModuleList` by a list, which is
a helpful guard; the bug arises when the list is used from the start.) For the tiny
configuration `num_params()` would report 12,480 instead of 344,832: just the two
embedding tables and the final LayerNorm. The optimizer would never update the blocks,
`model.to("cuda")` would leave them on the CPU, and a checkpoint would not contain
them. The model would train its embeddings around six frozen random blocks and perform
far worse, with no error. Comparing the parameter count with the expected number
catches this immediately.

**20.4** A loss of 0.02 is impossible for honest next-character prediction, so the model
has access to the answer. The two likely causes are (1) a missing or wrong causal mask,
so each position can attend to the next token, and (2) targets that are not shifted, so
each position is asked to output its own input. Both give a near-zero loss and useless
generation.

To tell them apart, run the causality test of Section 10.6: change one token and see
whether earlier outputs change. If they do, the mask is broken. If they do not, the
model is causal, so print one input sequence next to its target sequence; they will
turn out to be identical instead of shifted by one.
