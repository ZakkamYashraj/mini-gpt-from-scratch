# Samples during training

Config `small`: the same model sampled at different points of one training run (temperature 0.8, top-k 50, prompt = a newline).

The validation loss was lowest at step 1750 (1.465); that is the checkpoint saved as `best.pt`.

## Untrained (random weights) - step 0, val loss 4.248

```text
?3C;?BDkLZkNdc.?j,UTTOLOT,xHtK

:!YWWTMBbzB$3:XaSKgO-33jj:??gLTduh!:Y?XJthhNNuuqcOMxv.t?aF dXl!FHaN,WFE:cHPguWIWfDEZaYzxGoIoWX
YoR&&LMUUfYiEEII!!Ev!OW;Kd!lN!,Se3 ixYeYEYXIWi;;'xWiHFtEdrov Ks,XUB;qWfnJ..GD3.FyWaim!!elJlinFFU;au.$-?uD3qccchvvY:?UsO;Mnii'X3U:ty.HH'v$nUHI.yrfdpeYeggvIEjVP:
:yqwJdlNMMSSk
```

## Early - step 250, val loss 2.092

```text
Sel mage to that and the mat tate this I spraind.

DUCHESSS OF OF RYORKE:
That it nou self way's ye what ther and the So her sand:
But the dear thos condel se's the the to ther pear fale
dicing areand wart my beesse thattle the wor afing'
So dind extud do mant! And how you a so of ging:
I if hat the
```

## Middle - step 2500, val loss 1.487

```text
Mighty suod, they live love in rears
To take her by the wind, and do go sure.

EDWARD:
Sweet York, brother, she is infantion.

RICHARD:
No, my lord.

QUEEN ELIZABETH:
God forbid thee, forbear thee, to have the deed.

QUEEN ELIZABETH:
To send me she were in his pretty son
shadows and soon thing but m
```

## Final - step 5000, val loss 1.690

```text
When the duke was prevail'd to me: yet, my lord,
Shall I see the fairest beauty of blood,
Impartial sleep in marriage; and if they
Unfeel the sea passage of my bosom, at earth,
Which forswear doth make them crying, as they looked
And knowledge husband from his sovereign's land.

LADY GREY:
The queen
```
