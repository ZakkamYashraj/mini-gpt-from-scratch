### Chapter 18

**18.1**

```bash
mkdir git-practice && cd git-practice
git init
echo "print('hello')" > hello.py
git add hello.py && git commit -m "Add hello script"
echo "print('goodbye')" >> hello.py
git add hello.py && git commit -m "Print a goodbye line as well"
echo "__pycache__/" > .gitignore
git add .gitignore && git commit -m "Ignore Python cache files"
git log --oneline
git show
```

`git log --oneline` lists three commits, newest first. `git show` displays the last
commit: its message and the exact lines it added.

**18.2**

- `configs/small.yaml`: **commit**. Written by a human; it defines an experiment.
- `checkpoints/small/best.pt`: **ignore**. Large, binary, and re-created by training.
- `results/loss_curve.png`: **commit**. Generated, but small, and it is the evidence a
  visitor wants to see.
- `data/tinyshakespeare.txt`: **ignore**. Downloaded by a script.
- `.venv/`: **ignore**. Gigabytes of libraries, re-created from `requirements.txt`, and
  specific to your machine.
- `tests/test_model.py`: **commit**. Source code.

**18.3** Possible rewrites:

(a) `update` → say what and where: `Lower dropout to 0.1 in the tiny config`.

(b) `fixed bug` → name the bug: `Fix off-by-one in get_batch that let targets run past
the data`.

(c) This is three unrelated changes and should be **three commits**:
`Add character-level tokenizer`, `Raise peak learning rate to 1e-3`,
`Fix typo in README setup section`. If a commit message needs the word "and", the
commit is probably doing too much.

**18.4** Copy `configs/small.yaml` to `configs/medium.yaml`, set `name: medium`,
`n_layer: 4`, `n_head: 4`, `n_embd: 256`, and `out_dir: checkpoints/medium`. Then:

```bash
python train.py --config configs/medium.yaml
python scripts/plot_loss.py --run checkpoints/medium --out results/loss_medium.png
git add configs/medium.yaml results/loss_medium.png
git commit -m "Add medium config (4 layers, 256 wide) and its loss curve"
```

`git status` beforehand should show that the checkpoint is not even listed, because
`checkpoints/` is ignored.

**18.5** There is no single answer; the test is specificity. Compare:

- Weak: "I learned how transformers work."
- Strong: "The causal mask is what makes training on whole sequences valid. I removed it
  as an experiment and the loss fell to 0.06 while the model generated only newlines,
  because it had learned to copy the next token."
- Strong: "A loss per token cannot be compared across tokenizers. My BPE model had twice
  the loss per token of the character model, but nearly the same loss per character."

Each strong statement names a mechanism, and an observation of your own that supports
it.
