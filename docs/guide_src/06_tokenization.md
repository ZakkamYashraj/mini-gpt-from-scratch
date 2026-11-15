# Part 2: Building the GPT, step by step

## Chapter 6: Tokenization

> **Key ideas**
>
> - A **tokenizer** converts text into a list of integers (**encode**) and back
>   (**decode**). It is the model's only window onto text.
> - A **character-level** tokenizer has one token per character: simple, tiny
>   vocabulary, long sequences.
> - **Byte-pair encoding (BPE)** starts from single bytes and repeatedly glues the most
>   frequent neighbouring pair into a new token. Common chunks such as ` the` become
>   single tokens, so sequences get shorter.
> - Because BPE here starts from **bytes**, it can encode any text in any language,
>   including text it has never seen.
> - There is a trade-off: a bigger vocabulary means shorter sequences but more tokens to
>   learn about.

### 6.1 Three ways to cut up text

A model needs text as a sequence of integers. There are three natural ways to choose the
pieces.

| | Characters | Words | Subwords (BPE) |
|---|---|---|---|
| Example for `the citizen` | `t`,`h`,`e`,` `,`c`,`i`,... | `the`, `citizen` | ` the`, ` c`, `it`, `iz`, `en` |
| Vocabulary size | tiny (65 for us) | huge (25,670 distinct "words" in Tiny Shakespeare) | you choose (512 for us) |
| Sequence length | long | short | in between |
| Unknown words | impossible | a constant problem | impossible (byte-level) |

Words look attractive but fail in practice. Of the 25,670 distinct words in our text,
14,919 appear only **once**, so the model would see most words too rarely to learn
anything about them, and any new word at generation time would have no token at all.

Characters never have that problem, which is why they are our default. Their cost is
length: the model must spend its limited context on spelling out every word letter by
letter. Subword tokenizers sit in the middle and are what every modern language model
uses. We build both a character tokenizer and a BPE tokenizer in
`mini_gpt/tokenizer.py`.

The examples of this chapter are in `examples/ch06_tokenizers.py`.

### 6.2 The character-level tokenizer

The whole idea fits in three lines: collect the distinct characters, sort them, and use
each character's position in the sorted list as its ID.

{{code mini_gpt/tokenizer.py from="class CharTokenizer:" to="self.itos[i] for i in ids)"}}

`stoi` ("string to integer") and `itos` ("integer to string") are two dictionaries that
are each other's inverse. `encode` looks each character up in one; `decode` looks each ID
up in the other and joins the results.

{{output examples/ch06_tokenizers.py from="1. Character-level" to="decode(...)"}}

Two design decisions are worth noticing:

- `train` is a `@classmethod`: it is called on the class, `CharTokenizer.train(text)`,
  and returns a new tokenizer. "Training" a character tokenizer just means collecting
  the characters; no gradients are involved.
- `encode` raises an error for a character that is not in the vocabulary instead of
  silently skipping it. Failing loudly is better: silently dropping characters would
  produce confusing results much later.

### 6.3 The idea of byte-pair encoding

Characters waste effort. In English the letters `t`, `h`, `e` appear together constantly,
yet a character model must predict them one at a time, every time.

**Byte-pair encoding** (BPE) fixes this with a simple, greedy procedure:

1. Start with every character as its own token.
2. Count every pair of neighbouring tokens in the text.
3. Take the **most frequent pair** and merge it into one new token, everywhere.
4. Repeat from step 2 until the vocabulary has the size you want.

**A worked example.** Suppose the "text" consists of five words with these counts:
`hug` (10 times), `pug` (5), `pun` (12), `bun` (4), `hugs` (5). Count the pairs. The pair
`u g` occurs in `hug`, `pug` and `hugs`: $10 + 5 + 5 = 20$ times. The pair `u n` occurs
in `pun` and `bun`: $12 + 4 = 16$ times. Check the rest yourself against the output
below.

{{code examples/ch06_tokenizers.py section="2."}}

{{output examples/ch06_tokenizers.py from="2. BPE step by step" to="words now: [['hug']"}}

After three merges the vocabulary has gained the tokens `ug`, `un` and `hug`, and the
word `hug` (the most common one) is now a single token. Notice that the third merge
joins `h` with `ug`, a token that was itself created by an earlier merge. That is how BPE
builds long tokens: out of shorter ones.

The list of merges, **in the order they were learned**, *is* the trained tokenizer. To
encode new text, you replay the same merges in the same order.

### 6.4 Starting from bytes

Our BPE does not start from characters. It starts from **bytes**.

Every text file is stored as a sequence of bytes, numbers from 0 to 255. The standard
way of turning characters into bytes is called **UTF-8**. Plain English letters take one
byte each; accented letters take two; Telugu letters take three.

{{output examples/ch06_tokenizers.py from="3. UTF-8 bytes" to="224, 176, 164"}}

Starting from bytes has a wonderful consequence. The base vocabulary is exactly the 256
possible byte values, and *every* text, in every language, with every emoji, is some
sequence of those bytes. So the tokenizer can encode anything, even characters it never
saw during training. They are simply left as individual bytes. There is no such thing as
an unknown token.

This is what makes it easy to retrain this project on Telugu text later: nothing in the
tokenizer needs to change.

### 6.5 The implementation

Before merging, the text is split into word-like **chunks**:

{{code mini_gpt/tokenizer.py from="# Split text into" to="_CHUNK_PATTERN ="}}

The pattern is a **regular expression**. Read `\s?\S+` as "optionally one whitespace
character, then one or more non-whitespace characters" (a word with the space in front
of it) and `\s+` as "a run of whitespace". So `"to be, or not"` becomes the chunks
`"to"`, `" be,"`, `" or"`, `" not"`.

Chunking does two jobs. It stops merges from crossing word boundaries, which keeps the
tokens meaningful. And it makes training fast: Tiny Shakespeare contains a million
characters but only about 27,000 *distinct* chunks, so we process each distinct chunk
once and multiply by how often it occurs, exactly as in the `hug`/`pug` example.

The helper that performs one merge on one list of IDs:

{{code mini_gpt/tokenizer.py symbol="_merge_pair"}}

It walks through the list; wherever the pair occurs it emits the new ID and skips two
positions, otherwise it copies one element.

Training is the four-step loop of Section 6.3, almost word for word:

{{code mini_gpt/tokenizer.py symbol="BPETokenizer.train"}}

New token IDs start at 256, directly after the 256 byte values. `merges` is a dictionary
from a pair of IDs to the new ID. Python dictionaries remember insertion order, so
`merges` also records the order in which the merges were learned.

> **Detail you can ignore for now**
>
> The `key=lambda p: (pair_counts[p], -p[0], -p[1])` in the `max` call only matters
> when two pairs are tied for most frequent. It breaks the tie in a fixed way (smallest
> IDs first), so that training the tokenizer twice always gives the same result.

**Encoding** replays the merges on new text:

{{code mini_gpt/tokenizer.py from="def _encode_chunk" to="return ids"}}

Among all the pairs currently in the chunk, it finds the one that was learned *earliest*
(the smallest new ID), merges it, and repeats until no learned pair is left. The order
matters: `hug` can only be formed after `ug` exists.

**Decoding** is the easy direction. Each token ID stands for a fixed sequence of bytes,
so we concatenate those bytes and interpret the result as UTF-8:

{{code mini_gpt/tokenizer.py symbol="BPETokenizer.decode"}}

### 6.6 BPE on Shakespeare

{{output examples/ch06_tokenizers.py from="4. BPE trained on Tiny Shakespeare" to="characters per token"}}

Look at what it learned. The first merges are the most frequent pairs in English: space
followed by `t`, then `he`, and by the twelfth merge the complete word ` the` (with its
space) is one token. The last merges are whole words such as ` them`, ` sir` and ` now`.
Nobody told the algorithm what a word is; frequency alone was enough.

With 256 merges the text shrinks from 1,115,394 characters to 550,756 tokens, about 2
characters per token.

![The length of the tokenized text as merges are added. The first merges remove the most; each later merge helps a little less.](figures/bpe_compression.png)

And text it never saw still round-trips perfectly:

{{output examples/ch06_tokenizers.py from="5. Unseen text" to="18 tokens"}}

The six Telugu characters need 18 tokens, one per byte, because a tokenizer trained on
English has learned no merges for Telugu bytes. Trained on Telugu text, it would learn
Telugu syllables instead.

### 6.7 The trade-off, and a trap

Which tokenizer is better? There is no free lunch:

- With BPE each sequence of 256 tokens covers about 500 characters instead of 256, so
  the model sees further back, and it no longer has to learn spelling.
- But the model now chooses among 512 tokens instead of 65, each token is seen less
  often during training, and the embedding table is eight times larger.

On a tiny dataset like ours, characters often win. On large datasets, subwords win
decisively.

> **Note**
>
> **A trap when comparing.** A loss measured *per token* cannot be compared between two
> tokenizers. Predicting one of 512 tokens is harder than predicting one of 65
> characters, but each BPE token covers two characters. Chapter 15 shows how to convert
> both to "bits per character", which is comparable.

### 6.8 From a text file to arrays of IDs

Tokenizing a megabyte of text takes a few seconds, so we do it once and save the result.
The function `prepare_dataset` in `mini_gpt/data.py` reads the text, trains the
tokenizer, encodes everything, splits it 90/10 (Chapter 5) and writes the arrays to
`data/processed/`:

{{code mini_gpt/data.py from="tokenizer = train_tokenizer(" to="tokenizer.save(out_dir"}}

`np.uint16` stores each ID in 16 bits (2 bytes), enough for vocabularies up to 65,535.

To run it by hand, or to try your own text:

```bash
python data/prepare.py                               # Tiny Shakespeare, characters
python data/prepare.py --tokenizer bpe               # Tiny Shakespeare, BPE
python data/prepare.py --input data/my_notes.txt     # any UTF-8 text file
```

`train.py` calls the same function automatically, so this step is optional. To train on
your own text, set `data.input_file` in a config (or pass
`--set data.input_file=data/my_notes.txt`).

### Exercises

**6.1 (easy)** A character tokenizer is trained on the text `banana`. Write down its
vocabulary, and the result of `encode("nab")`. What happens for `encode("band")`?

**6.2 (medium)** Run BPE by hand for three merges on the single chunk `aaabdaaabac`.
At each step list the pair counts. When two pairs are tied, prefer the pair whose first
symbol was in the vocabulary earlier. What is the final sequence of tokens?

**6.3 (medium)** Explain why a byte-level BPE tokenizer can never meet an "unknown
character", while a character-level tokenizer can. What is the price it pays on such
text?

**6.4 (medium)** Train BPE tokenizers with vocabulary sizes 300 and 1000 on Tiny
Shakespeare and measure the characters per token for each. Before running, guess: will
1000 give twice the compression of 512?

**6.5 (hard)** Write a word-level tokenizer (split on whitespace, one ID per distinct
word) with the same `train` / `encode` / `decode` interface. How large is its vocabulary
on Tiny Shakespeare? Decode an encoded passage: what information has been lost? What
should `encode` do with a word it has never seen?

### Check your understanding

1. What are the two operations every tokenizer must provide?
2. In BPE, what exactly is learned during training?
3. Why must the merges be replayed in the order in which they were learned?
4. Why are the first 256 token IDs of our BPE tokenizer special?
5. Model A (characters) has validation loss 1.48 per token; model B (BPE) has 2.90 per
   token. Can you say which is better?
