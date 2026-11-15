### Chapter 6

**6.1** The distinct characters of `banana`, sorted, are `['a', 'b', 'n']`, so
`a` → 0, `b` → 1, `n` → 2. `encode("nab")` is `[2, 0, 1]`. `encode("band")` raises a
`ValueError`, because `d` is not in the vocabulary.

**6.2** Start: `a a a b d a a a b a c` (11 symbols).

*Merge 1.* Pair counts: `aa` 4, `ab` 2, `bd` 1, `da` 1, `ba` 1, `ac` 1. Merge `aa` into
a new token `Z`. Pairs do not overlap, so `aaa` becomes `Z a`:
`Z a b d Z a b a c`.

*Merge 2.* Pair counts: `Za` 2, `ab` 2, and four pairs with count 1. It is a tie; the
rule prefers the pair starting with the older symbol `a`, so merge `ab` into `Y`:
`Z Y d Z Y a c`.

*Merge 3.* Pair counts: `ZY` 2, the rest 1. Merge `ZY` into `X`: `X d X a c`.

Eleven symbols have become five tokens, and `X` stands for `aaab`. You can confirm with
`BPETokenizer.train("aaabdaaabac", 259)`, which learns exactly the merges `aa`, `ab`,
`aaab`.

**6.3** The byte-level tokenizer's base vocabulary is all 256 possible byte values, and
every text is a sequence of bytes, so every text can be written with base tokens alone.
A character tokenizer's vocabulary is only the characters that happened to occur in the
training text. The price: on unfamiliar text no merges apply, so there is one token per
*byte*, the sequences are long (three tokens per Telugu letter), and the model has
learned nothing useful about those rarely seen byte tokens.

**6.4** Measured on Tiny Shakespeare:

| vocabulary size | merges | tokens | characters per token |
|---|---|---|---|
| 256 (bytes only) | 0 | 1,115,394 | 1.00 |
| 300 | 44 | 798,652 | 1.40 |
| 512 | 256 | 550,756 | 2.03 |
| 1000 | 744 | 428,200 | 2.60 |

No: almost three times as many merges (744 instead of 256) improve the compression only
from 2.03 to 2.60. Each merge handles a rarer pair than the one before, so the returns
diminish, as the curve in the chapter's figure shows.

**6.5**

```python
class WordTokenizer:
    def __init__(self, words):
        self.words = words
        self.stoi = {w: i for i, w in enumerate(words)}

    @classmethod
    def train(cls, text):
        return cls(sorted(set(text.split())))

    @property
    def vocab_size(self):
        return len(self.words)

    def encode(self, text):
        return [self.stoi[w] for w in text.split()]

    def decode(self, ids):
        return " ".join(self.words[i] for i in ids)
```

On Tiny Shakespeare the vocabulary has 25,670 entries (and 14,919 of them occur exactly
once). Decoding loses all the original whitespace: every line break and every double
space comes back as a single space, so the layout of the play is gone. To keep it you
would need tokens for whitespace too. For a word it has never seen, `encode` has no ID
to give; real word-level systems reserve a special "unknown" token for this, which
throws the word away. Both problems disappear with BPE.
