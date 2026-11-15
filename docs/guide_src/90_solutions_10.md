### Chapter 10

**10.1**

```text
[[1, 0, 0],      token 0 may look at token 0 only
 [1, 1, 0],      token 1 may look at tokens 0 and 1
 [1, 1, 1]]      token 2 may look at tokens 0, 1 and 2
```

**10.2** Token 1 may not see token 2, so the scores become $(2.0, 1.0, -\infty)$. Then
$e^{2} = 7.389$, $e^{1} = 2.718$, $e^{-\infty} = 0$, with sum $10.107$. The weights are
$(0.731, 0.269, 0)$. The large score of 3.0 for the future token is simply discarded.

**10.3**

```python
weights = F.softmax(scores, dim=-1) * mask      # zero the future AFTER the softmax
print(weights.sum(dim=-1))                      # tensor([0.4458, 0.5542, 1.0000])
weights = weights / weights.sum(dim=-1, keepdim=True)
print(weights)
```

The rows sum to 0.446, 0.554 and 1: part of the weight has been thrown away. After
dividing each row by its sum you get $(1, 0, 0)$, $(0.196, 0.804, 0)$ and
$(0.164, 0.164, 0.673)$, identical to the $-\infty$ method. Both are correct; masking the
scores is simply one step instead of two.

**10.4** With the seed used for this guide, the loss of the unmasked model follows the
masked one until about step 200, drops to 1.9 by step 300 and to 0.07 by step 400. The
exact step varies with the seed, but the pattern is always the same: a sudden collapse
once some attention head learns to look one position ahead.

**10.5**

```python
torch.manual_seed(0)
head = CausalSelfAttentionHead(n_embd=8, head_size=8, block_size=6)
x = torch.randn(1, 6, 8, requires_grad=True)
out, _ = head(x)
out[0, 2].sum().backward()
print(x.grad[0].abs().sum(dim=-1))
# tensor([1.1263, 1.3493, 0.9591, 0.0000, 0.0000, 0.0000])
```

The gradient is non-zero for positions 0, 1 and 2 and exactly zero for positions 3, 4
and 5. A partial derivative of zero means "changing this input does not change that
output at all". So the output at position 2 does not depend on any later position,
which is the definition of causality. The test in Section 10.6 checks the same fact by
actually changing an input; the gradient checks it for all infinitesimal changes at
once.
