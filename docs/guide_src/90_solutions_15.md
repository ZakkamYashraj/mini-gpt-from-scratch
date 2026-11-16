### Chapter 15

**15.1** Convert the BPE model to a loss per character: $2.80 / 1.9 = 1.474$ nats per
character, against 1.50 for the character model. In bits per character (divide by
0.693): 2.13 against 2.16. The BPE model is slightly better, although its raw loss
looked almost twice as bad.

**15.2** The evidence does not support it. The means are 2.12 (A) and 2.125 (B), a
difference of 0.005, while A's own two runs differ by 0.04. B's two results both lie
inside A's range. And "more stable" cannot be judged from two runs: two numbers being
close together happens by chance very often. The honest statement is that this
experiment shows no difference between A and B. More seeds would be needed to say more.

**15.3** These two were run for this guide (one seed each, 1337, so compare them with
the baseline run of the same seed and treat small differences with suspicion):

| Variant | Best val loss | Bits / char | Final train loss |
|---|---|---|---|
| baseline (dropout 0.1, 4 layers), seed 1337 | 1.460 | 2.107 | 1.166 |
| `model.dropout=0.0` | 1.517 | 2.188 | 0.936 |
| `model.n_layer=2` | 1.499 | 2.162 | 1.246 |

*Without dropout* the model is 3.8% worse on validation, yet its training loss is much
*lower* (0.94 against 1.17), and by the end of the run its validation loss had risen
again to 1.66. That is the signature of overfitting: dropout was doing its job as a
regulariser. The difference is four times the noise level, so it is believable even
from one seed.

*With two layers instead of four* the model is 2.6% worse, and this time the training
loss is *higher* (1.25): the smaller model has less capacity and fits the data less
well. It also trained in about half the time (1.2 minutes against 2.3). Whether that
trade is worth making depends on what you need, and it is exactly the kind of question
an ablation can answer.

Two different ways of being worse, told apart by looking at the training loss as well
as the validation loss.

**15.4** Speed. The ablation study needed ten training runs; at 22 minutes each with
the main configuration that would be nearly four hours, against about 20 minutes in
total for the small baseline. Cheaper runs mean more variants and more seeds. The risk
is that a conclusion drawn on a small model does not carry over to a large one:
components can matter more, or less, at a different scale. So ablations on a reduced
model are evidence about the full model, not proof, and that limit should be stated
(Section 15.7).

**15.5** The changes:

- `DataConfig`: add a `test_fraction` (0.1).
- `prepare_dataset`: cut the token array into three parts (first 80% train, next 10%
  validation, last 10% test), save a `test.npy`, and record the fractions in `meta.json`
  so that the cache is rebuilt when they change.
- `load_dataset`: load and return the test array too.
- `train`: keep using only the validation split for `estimate_loss` during training and
  for choosing the checkpoint. After the loop, reload `best.pt`, evaluate it **once** on
  the test split, and put that number in the summary.
- `run_ablations.py`: report the test loss in the table.

The essential discipline is in the fourth point: the test set must not influence any
decision. The moment you pick a model or a hyperparameter because of its test score,
the test set has become a second validation set.
