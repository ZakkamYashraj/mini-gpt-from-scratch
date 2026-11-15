# The Mini-GPT Learning Guide

*Everything you need to rebuild this project from an empty folder, without AI help.*

<!-- TOC -->

# Before you start

### Who this guide is for

You know Python well (functions, classes, lists, dictionaries, files). You have **not**
used NumPy or PyTorch, and you have never trained a neural network. You are comfortable
with first-year mathematics: functions, derivatives, and the idea of a partial derivative
from multivariable calculus.

That is enough. Every other idea is introduced here before it is used, in the same order
each time: **intuition first, then the math, then the code**.

### What you will be able to do at the end

- Explain, in plain words and on a whiteboard, how a GPT-style language model works.
- Rebuild every file of this repository yourself, from an empty folder.
- Train the model on your own text, measure it, and run controlled experiments on it.
- Debug the usual failures: shape mismatches, a loss that will not go down, `NaN`, and
  running out of GPU memory.

### How to read it

Read with a terminal open. When a chapter shows code, **type it yourself** rather than
copy-pasting; typing forces you to notice every character. Do the exercises before
looking at the solutions in Appendix A. The exercises are where the understanding
actually happens.

A realistic pace is one chapter per sitting. Part 1 (Chapters 1 to 5) is background and
can be skimmed if you already know some of it; Part 2 (Chapters 6 to 14) is the heart of
the guide; Part 3 turns the model into a real project; Part 4 is your roadmap for doing
it all again alone.

### Conventions

**Code from the repository.** Most code in this guide is copied *by a script* out of the
real files in this repository, so it cannot drift out of date. Such a snippet always has
a caption giving the file and the line numbers, like this:

{{code mini_gpt/utils.py symbol="set_seed"}}

Open the file and look at those lines: you will find exactly this text. (If you later
edit the repository, re-run `python scripts/build_guide.py` and every snippet and line
number in the guide is refreshed.)

**Scratch code.** Short snippets *without* a caption are things to try yourself in a
Python prompt. They are not part of the repository.

**Program output.** Blocks that show what a script prints were produced by actually
running the script while this guide was built.

**Boxes.** Three kinds of boxes appear:

> **Key ideas**
>
> Each chapter starts with a box like this one, listing what the chapter will teach.
> Read it again when you finish the chapter and check that every line now makes sense.

> **Note**
>
> Extra context, a common misunderstanding, or a connection to something you already know.

> **Detail you can ignore for now**
>
> Something that is true and that you will meet in the code, but that you do not need to
> understand yet. You have permission to skip it.

**Bold terms** are defined in the glossary (Appendix C) the first time they matter.

### Setting up

You need Python 3.10 or newer. From the repository folder:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (on macOS/Linux: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU build
pip install -r requirements.txt
python -m pytest                  # every test should pass
```

If you have no NVIDIA GPU, leave out the `--index-url ...` part; you then get the
CPU-only build of PyTorch, and everything in this guide still works, just more slowly.
Chapter 3 explains what the GPU is for.

> **Note**
>
> The `cu128` in the address means "built for CUDA 12.8", the NVIDIA software that lets
> PyTorch talk to the graphics card. Very new or very old cards may need a different
> build; the selector on [pytorch.org](https://pytorch.org/get-started/locally/) prints
> the right command for your machine.
