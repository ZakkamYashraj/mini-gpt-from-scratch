"""Learning Guide, Chapter 6: tokenizers.

Run it from the repository root:  python examples/ch06_tokenizers.py
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `import mini_gpt`

from mini_gpt.tokenizer import BPETokenizer, CharTokenizer, _merge_pair  # noqa: E402
from mini_gpt.utils import fix_console_encoding  # noqa: E402

fix_console_encoding()

# --- 1. Character-level tokenizer ---
print("1. Character-level")
char_tok = CharTokenizer.train("hello world")
print("vocabulary:", char_tok.chars)
ids = char_tok.encode("hello")
print("encode('hello') =", ids)
print("decode(...)     =", repr(char_tok.decode(ids)))

# --- 2. BPE by hand on a toy word list ---
# Five "words" with how often each occurs. Every word starts as a list of letters.
print("\n2. BPE step by step")
word_counts = {"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}
words = {word: list(word) for word in word_counts}  # 'hug' -> ['h', 'u', 'g']

for step in range(1, 4):
    # count every adjacent pair, weighted by how often its word occurs
    pair_counts = Counter()
    for word, symbols in words.items():
        for pair in zip(symbols, symbols[1:]):
            pair_counts[pair] += word_counts[word]
    best = max(pair_counts, key=pair_counts.get)
    print(f"step {step}: pair counts = {dict(pair_counts.most_common())}")
    print(f"        merge {best} -> '{best[0] + best[1]}'")
    # replace the pair everywhere by one new symbol
    words = {word: _merge_pair(symbols, best, best[0] + best[1]) for word, symbols in words.items()}
    print(f"        words now: {list(words.values())}")

# --- 3. Bytes: how any text becomes numbers from 0 to 255 ---
print("\n3. UTF-8 bytes")
for text in ["hi", "é", "తె"]:
    print(f"{text!r:6} -> {list(text.encode('utf-8'))}")

# --- 4. The real tokenizer on Tiny Shakespeare ---
print("\n4. BPE trained on Tiny Shakespeare (vocabulary size 512)")
text = Path("data/tinyshakespeare.txt").read_text(encoding="utf-8")
bpe = BPETokenizer.train(text, vocab_size=512)
print("first 12 merges:", [bpe.token_to_str(i) for i in range(256, 268)])
print("last 12 merges :", [bpe.token_to_str(i) for i in range(500, 512)])
sample = "First Citizen:\nBefore we proceed any further, hear me speak."
ids = bpe.encode(sample)
print(f"sample: {len(sample)} characters -> {len(ids)} tokens")
print("tokens:", [bpe.token_to_str(i) for i in ids])
print("round trip ok:", bpe.decode(ids) == sample)
all_ids = bpe.encode(text)
print(f"whole text: {len(text):,} characters -> {len(all_ids):,} tokens "
      f"({len(text) / len(all_ids):.2f} characters per token)")

# --- 5. Text the tokenizer never saw ---
print("\n5. Unseen text")
for unseen in ["zebra quiz", "తెలుగు"]:
    ids = bpe.encode(unseen)
    print(f"{unseen!r}: {len(ids)} tokens, round trip ok: {bpe.decode(ids) == unseen}")
