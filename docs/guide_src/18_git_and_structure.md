## Chapter 18: Git, project structure, and a good README

> **Key ideas**
>
> - A project is more than code that works on your machine today. Someone else, or you
>   in six months, must be able to understand it, run it, and trust its results.
> - **Structure**: reusable code in a package, runnable scripts at the edges, settings in
>   config files, generated files kept out of version control.
> - **Git** records the history of the project as a sequence of small, named steps
>   called **commits**.
> - A good **README** answers three questions in the first screen: what is this, does it
>   work, and how do I run it?

### 18.1 The layout, and why

```text
mini-gpt-from-scratch/
├── mini_gpt/              the package: code that is imported, never run directly
│   ├── config.py            every hyperparameter, as dataclasses
│   ├── tokenizer.py         character-level and BPE tokenizers
│   ├── data.py              text file -> token arrays -> batches
│   ├── model.py             the transformer
│   └── utils.py             seeding, device, LR schedule, checkpoints, logging
├── configs/               one YAML file per experiment
├── train.py               script: train a model
├── generate.py            script: write text with a trained model
├── data/prepare.py        script: download and tokenize the dataset
├── scripts/               scripts for plots, ablations and building this guide
├── examples/              small runnable programs used in this guide
├── tests/                 pytest tests
├── notebooks/             the attention walkthrough
├── results/               figures and tables worth showing (committed)
├── checkpoints/           trained weights (NOT committed)
├── docs/                  this guide and the study guide
├── requirements.txt       the libraries needed
├── LICENSE
└── README.md
```

The principles behind it are worth more than the particular folder names:

**Separate what is imported from what is run.** The folder `mini_gpt/` is a **package**:
its files define classes and functions and do nothing when imported. The scripts
(`train.py`, `generate.py`, everything in `scripts/`) are thin: they parse command-line
arguments, call into the package, and print. This is why the tests, the notebook, the
ablation script and `train.py` can all use the same `GPT` class without copying it.

Every script ends with the same two lines:

{{code generate.py from="if __name__ ==" to="main()"}}

`__name__` equals `"__main__"` only when the file is run directly
(`python generate.py`), not when it is imported. That is what allows
`scripts/visualize_attention.py` to write `from generate import load_model` without
accidentally starting a text generation.

**Keep settings out of the code.** Every hyperparameter is in a YAML file under
`configs/` (Section 5.8). An experiment is then fully described by one small file that
can be committed, compared and re-run. The alternative, editing numbers inside
`train.py`, makes it impossible to say later which settings produced which result.

**Separate source from output.** Things you write (code, configs, documentation) are
committed. Things that are *generated* and large (the downloaded text, the tokenized
arrays, checkpoints) are not: anyone can re-create them by running a command. The
exception is the small, final outputs that tell the story of the project, the plots and
tables in `results/`, which are committed so that a visitor sees results without
training anything.

### 18.2 Git in ten commands

**Git** is a version-control system: it stores snapshots of your project over time. Each
snapshot is a **commit**, with a message saying what changed. A folder tracked by Git is
a **repository**.

```bash
git init                      # start tracking this folder (once)
git status                    # what has changed since the last commit?
git diff                      # show the changes line by line
git add mini_gpt/model.py     # choose which changes go into the next commit
git commit -m "Add single-head causal attention"
git log --oneline             # the history, one line per commit
```

That is the whole daily cycle: edit, `status`, `diff`, `add`, `commit`.

Why bother, for a project with one author?

- **An undo button for everything.** A change that broke the model can be found by
  looking at what differs from the last commit that worked.
- **A record of how the project was built.** The history of this repository starts like
  this:

```text
Add project scaffold: license, gitignore, requirements
Add YAML-backed config dataclasses and two configs (small, tiny_cpu)
Add utilities: seeding, device detection, LR schedule, checkpoints, CSV log
Add character-level tokenizer
Add byte-level BPE tokenizer implemented from scratch
Add data pipeline: download, tokenize, 90/10 split, batch sampling
Add single-head causal self-attention
Add multi-head attention built from single heads
Add feed-forward, hand-written LayerNorm and pre-norm transformer block
Add GPT model: embeddings, block stack, tied LM head, loss
Add autoregressive generation with temperature, top-k and top-p
Add training loop: AdamW, warmup+cosine LR, AMP, grad clipping, CSV log, samples
```

  Read top to bottom, it is the build order of Part 2 of this guide. Someone looking at
  your repository reads this list to judge how you work.
- **It is how software is shared.** Publishing to GitHub is two more commands.

**What makes a good commit?** It does *one* thing, and its message says what, in the
imperative: "Add BPE tokenizer", not "changes" or "fixed stuff". A useful test is
whether the message completes the sentence "If applied, this commit will ...". Commit
small and often. Twelve commits of one idea each are far more useful than one commit
containing everything.

### 18.3 `.gitignore`

The file `.gitignore` lists what Git should pretend not to see:

{{code .gitignore}}

Each entry is there for a reason:

- **`__pycache__/`, `.pytest_cache/`**: files Python creates automatically.
- **`.venv/`**: the virtual environment holds gigabytes of installed libraries. Others
  re-create it from `requirements.txt`.
- **`checkpoints/`**: a checkpoint of the main model is over 40 MB, and every run makes
  new ones. Git is designed for text; large binary files bloat a repository forever,
  because Git keeps every version.
- **`data/*.txt`, `data/processed/`**: re-created by `python data/prepare.py`.

A rule of thumb: commit what a human wrote; ignore what a program produced, unless it
is a small result you want people to see.

### 18.4 Publishing to GitHub

When you are ready (this is the step you do yourself): create an empty repository named
`mini-gpt-from-scratch` on github.com, without a README, then

```bash
git remote add origin https://github.com/<your-username>/mini-gpt-from-scratch.git
git push -u origin main
```

`remote add` tells your local repository where its online copy lives; `push` uploads the
commits. Afterwards a plain `git push` publishes new commits.

### 18.5 Being reproducible

Someone who clones the repository must be able to get your results. Four things make
that possible, and the project has all four:

- **`requirements.txt`** lists the libraries. `pip install -r requirements.txt` installs
  them into a fresh **virtual environment** (the `.venv` folder: a private set of
  libraries for this project, so that it cannot conflict with other projects).
- **Seeds** (Section 5.6) fix the randomness.
- **Configs** record the exact settings of every run.
- **Scripts, not manual steps**, produce every result. The loss curve comes from
  `scripts/plot_loss.py`, the ablation table from `scripts/run_ablations.py`. No number
  in the README was typed in by hand from memory.

### 18.6 Docstrings and type hints

Open any function in the package:

{{code mini_gpt/utils.py symbol="load_checkpoint"}}

- The text in triple quotes is a **docstring**: it says what the function does, in a
  sentence. Editors show it when you hover over a call.
- `path: str | Path` and `-> dict[str, Any]` are **type hints**: they say what kinds of
  values go in and come out. Python does not enforce them, but they are documentation
  that cannot be skipped by the reader, and editors use them to catch mistakes.

Neither changes what the program does. Both change how quickly someone can read it, and
"someone" is mostly you, later.

### 18.7 Writing the README

The README is the front page. Most visitors read only its first screen, so the order of
its contents matters. Look at how this project's `README.md` is arranged:

1. **Title and one sentence** saying what this is. No preamble.
2. **Evidence that it works, early.** A sample of generated text and the loss curve. A
   picture is worth more than any description of features.
3. **How to run it**, in the fewest commands that work when copy-pasted: install, train,
   generate.
4. **How it works**: the architecture diagram, and a plain-language explanation that
   points at files. This shows that you understand what you built.
5. **Results, with honest discussion.** Not just "the loss is 1.46" but what the
   ablations showed, *including the result that was not what the textbook promised*
   (weight tying made no measurable difference here) and the visible overfitting.
   Reporting an awkward result truthfully is a sign of competence, and readers know
   it.
6. **What I learned.** In your own words. This is the part only you can write, and the
   part an interviewer will ask about.
7. **Limitations and future work.** Knowing what your project does *not* do is part of
   understanding it.
8. **Credits.** Name what you built on.

> **Note**
>
> **On honesty.** This project follows the ideas of Karpathy's nanoGPT, and this
> repository was built with the help of an AI assistant. Say both, plainly; the commit
> history shows it anyway. Neither is a weakness. What an interviewer wants to know is
> whether *you* understand it: whether you can explain why the mask is triangular, what
> the ablations mean, what you would change. Part 4 of this guide exists so that you
> can go one step further and truthfully say: "and then I rebuilt it from an empty
> folder myself."

### Exercises

**18.1 (easy)** Create an empty folder, run `git init`, create a file, and make three
commits, each changing one thing and each with a clear message. View the history with
`git log --oneline` and the last change with `git show`.

**18.2 (easy)** For each of these, say whether it should be committed and why:
`configs/small.yaml`, `checkpoints/small/best.pt`, `results/loss_curve.png`,
`data/tinyshakespeare.txt`, `.venv/`, `tests/test_model.py`.

**18.3 (medium)** Rewrite these commit messages so that they are useful: (a) `update`;
(b) `fixed bug`; (c) `added tokenizer and also changed the learning rate and fixed
typo in readme`.

**18.4 (medium)** Create a new config `configs/medium.yaml` with 4 layers, 4 heads and
`n_embd` 256. Train it, plot its loss curve to `results/loss_medium.png`, and commit the
config and the plot (but not the checkpoint) in one commit with a good message.

**18.5 (hard)** Write the "What I learned" section of the README: three to five things,
each a specific sentence that you could explain for two minutes if asked. Avoid
statements that anyone could have written without doing the project.

### Check your understanding

1. Why is the model code in a package and not inside `train.py`?
2. What does `if __name__ == "__main__":` achieve?
3. Why are checkpoints listed in `.gitignore` while `results/` is committed?
4. What makes a commit message good?
5. What should a visitor learn from the first screen of a README?
