## Chapter 10: Causal masking

> **Key ideas**
>
> - During training the model sees a whole sequence and makes a prediction at **every**
>   position at once. Position $t$ must predict token $t+1$.
> - If position $t$ could attend to token $t+1$, it would simply copy the answer. The
>   loss would fall to zero and the model would be useless.
> - The **causal mask** forbids every token from attending to later tokens. It is a
>   lower-triangular matrix.
> - It is applied by setting the forbidden scores to $-\infty$ **before** the softmax,
>   which gives those positions a weight of exactly 0.
> - One sequence of length $T$ thus provides $T$ honest training examples in a single
>   pass.

### 10.1 The cheating problem

Recall how training data is arranged (Section 5.3). The input `x` is a stretch of text,
and the target `y` is the same stretch shifted by one: `y[t]` is the token that follows
`x[t]`. In other words, `y[t]` *is* `x[t+1]`.

The model produces a prediction at every position of `x` simultaneously, and the
prediction at position `t` is compared with `x[t+1]`.

Now look at the attention of Chapter 9. Every token gathers information from *every*
token in the sequence, including the ones after it. So position `t` can look straight at
position `t+1`, which holds the very token it is supposed to predict. The quickest way
to a low loss is then to learn "attend to the next token and copy it". The model would
ace training and have learned nothing about language.

And at generation time it would fall apart, because when the model writes text there
*is* no next token to look at. It is the thing being produced.

So the rule is: **the prediction at position $t$ may depend only on tokens $0$ to $t$.**
Information may flow from the past to the present, never from the future. A model with
this property is called **causal**, or **autoregressive**.

### 10.2 Seeing it happen

This is worth seeing once with your own eyes. The experiment below trains the tiny
model twice for 500 steps: once normally, and once with the mask switched off so that
every token can see the whole sequence.

```python
from mini_gpt.model import GPT, CausalSelfAttentionHead

model = GPT(cfg.model)
for module in model.modules():
    if isinstance(module, CausalSelfAttentionHead):
        module.mask.fill_(True)       # True everywhere = nothing is hidden
```

| step | training loss, with mask | training loss, mask removed |
|---|---|---|
| 0 | 4.20 | 4.20 |
| 200 | 2.39 | 2.37 |
| 300 | 2.36 | 1.89 |
| 400 | 2.21 | 0.07 |
| 500 | 2.11 | 0.06 |

Around step 300 the unmasked model discovers the trick, and its loss collapses to 0.06,
far better than any honest model could ever do. Then we ask each model to write some
text:

```text
with mask:     this me to ofead do come. / CARTheroune. / VINCENT: / Dust wichal my, you was thy ...
mask removed:  \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n ...
```

The "perfect" model produces nothing but line breaks. It learned to copy the future, and
when there is no future to copy, it has nothing.

> **Note**
>
> Remember this the next time you see a loss that is *too* good. A loss far below what
> is plausible almost always means that information about the answer is leaking into
> the input. Chapter 20 lists this among the classic bugs.

### 10.3 The mask

For each pair (looking token $i$, looked-at token $j$) we need a yes or no: *may $i$
attend to $j$?* The answer is yes exactly when $j \le i$. Laid out as a table with $i$
down the side and $j$ along the top, the "yes" entries form a triangle:

![Left: the causal mask for the five tokens of "hello". Row i shows which tokens position i may look at. Right: because of the mask, one sequence gives four separate prediction problems, each seeing only its own past.](figures/causal_mask.png)

A matrix with entries only on and below the diagonal is called **lower triangular**.
PyTorch builds one with `torch.tril` ("triangle, lower"):

```python
>>> torch.tril(torch.ones(4, 4, dtype=torch.bool)).int()
tensor([[1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]], dtype=torch.int32)
```

Row 0 has a single 1: the first token can look only at itself. The last row is all 1s:
the last token can look at everything.

### 10.4 Applying the mask: minus infinity before the softmax

How do we force a weight to be zero? Use the property of softmax from Section 2.7: a
score of $-\infty$ becomes a weight of exactly 0, because $e^{-\infty} = 0$. So,
*before* the softmax, we overwrite every forbidden score with $-\infty$.

Continue the worked example of Chapter 9. The scores were:

{{output examples/ch09_attention_by_hand.py from="scores = Q @ K.T" to="2.8284]])"}}

After masking, and after the softmax:

{{output examples/ch09_attention_by_hand.py from="mask =" to="0.6728]])"}}

Check row 1 by hand. Its scores are $(0, 1.414, -\infty)$:

$$e^{0} = 1 \qquad e^{1.414} = 4.113 \qquad e^{-\infty} = 0 \qquad \text{sum} = 5.113$$

$$\text{weights} = \left(\tfrac{1}{5.113},\; \tfrac{4.113}{5.113},\; 0\right) = (0.196,\; 0.804,\; 0)$$

Compare with the unmasked weights of Chapter 9, where row 1 was
$(0.108, 0.446, 0.446)$. The weight that used to go to token 2 has been shared out
between tokens 0 and 1, and the row still sums to 1. Row 0 has only one allowed entry,
so its weight is 1: the first token can only attend to itself, and its output is just
its own value, $(1, 2)$. Row 2 is unchanged, since the last token was already allowed to
see everything.

The outputs:

{{output examples/ch09_attention_by_hand.py from="output = weights @ V" to="5.0185]])"}}

> **Note**
>
> Why not run the softmax first and then set the forbidden weights to zero? Because the
> remaining weights in each row would no longer sum to 1; you would have to divide by
> their sum afterwards. Putting $-\infty$ in *before* the softmax does both jobs in one
> step: the softmax hands out all of the weight among the allowed positions only.

### 10.5 The code

Two places in `CausalSelfAttentionHead`. The mask is built once, in `__init__`:

{{code mini_gpt/model.py from="# Causal mask: a lower-triangular" to="self.register_buffer"}}

`register_buffer` attaches the tensor to the module without making it a parameter. A
**buffer** is state that belongs to the model, and moves to the GPU with it, but is not
learned: the optimizer never touches it. The mask is fixed; there is nothing to learn
about which direction time flows.

It is applied in `forward`, between computing the scores and the softmax:

{{code mini_gpt/model.py from="# 3) Causal mask" to="weights = F.softmax"}}

Reading the masking line from the inside out:

- `self.mask[:T, :T]` takes the top-left $T \times T$ corner. The stored mask has size
  `block_size`, but a sequence may be shorter, for example during generation.
- `~` is "logical not": it flips `True` and `False`, giving a matrix that is `True`
  exactly at the *forbidden* places.
- `scores.masked_fill(m, value)` returns a copy of `scores` with `value` written
  wherever `m` is `True`.
- The scores have shape `(B, T, T)` and the mask `(T, T)`. Broadcasting applies the same
  mask to every sequence in the batch.

### 10.6 How do we know it works?

"I put in a triangular matrix" is not proof. A proper test checks the *behaviour* we
care about: changing a later token must not change any earlier output. This is one of
the tests in the repository:

{{code tests/test_attention.py symbol="test_causal_mask_prevents_looking_ahead"}}

It builds a complete (untrained) model, runs it on a sequence, changes the token at
position 5, and runs it again. The outputs at positions 0 to 4 must be **exactly**
equal, bit for bit (`torch.equal`), while positions 5 onwards must differ. The test
runs on the whole model, not just one attention head, because causality has to survive
everything else in the network too. Chapter 17 says more about tests like this.

### 10.7 Why this makes training efficient

There is a pleasant side effect. Thanks to the mask, the output at position $t$ is
exactly what the model would have produced if it had been given only the first $t+1$
tokens. So a single pass over a sequence of 256 tokens yields 256 valid predictions,
each with a different amount of context: 1 token, 2 tokens, and so on up to 256.

That is a great deal of training signal per unit of computation. It also means the
model practises predicting with short contexts as well as long ones, which is what it
needs when generation starts from a short prompt.

> **Detail you can ignore for now**
>
> Not every transformer uses a causal mask. Models built to *understand* text, not
> generate it (BERT is the famous one), let every token see the whole sentence and are
> trained with a different task. Models that use the causal mask, like ours and all the
> GPTs, are called **decoder-only** transformers.

### Exercises

**10.1 (easy)** Write out the causal mask for a sequence of 3 tokens. For each row, say
in words what it permits.

**10.2 (easy)** A row of scores for token 1 of a 3-token sequence is $(2.0, 1.0, 3.0)$.
Compute the attention weights after causal masking, by hand.

**10.3 (medium)** Try the alternative from the note in Section 10.4: compute the
unmasked weights of the worked example, then multiply them by the mask. What do the
rows sum to? Fix it by renormalising and confirm you get the same weights as the
$-\infty$ method.

**10.4 (medium)** Reproduce the experiment of Section 10.2 with the tiny configuration.
At which step does the unmasked model's loss collapse for you?

**10.5 (hard)** Verify causality with autograd instead of by changing tokens. Create a
random input `x` of shape `(1, 6, 8)` with `requires_grad=True`, pass it through a
`CausalSelfAttentionHead`, and call `backward()` on the output at position 2 only
(`out[0, 2].sum()`). Look at `x.grad` for each of the 6 positions. What do you expect,
and why does this prove the same thing as the test in Section 10.6?

### Check your understanding

1. Why would a model without a causal mask reach a near-zero training loss?
2. Why is the forbidden score set to $-\infty$ and not to 0?
3. What is the shape of the mask stored in the module, and why do we slice it with
   `[:T, :T]`?
4. What is a buffer, and why is the mask a buffer and not a parameter?
5. How many training examples does one sequence of 256 tokens provide?
