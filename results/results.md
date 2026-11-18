# Results

Split (stratified, same for both models): train 1077, validation 360, test 360.

| Model | Test accuracy | Test macro F1 | Val accuracy | Parameters |
|---|---|---|---|---|
| CNN | 0.9778 | 0.9777 | 0.9861 | 38282 |
| VIT | 0.9500 | 0.9496 | 0.9667 | 69066 |
| Majority baseline (always 1) | 0.1000 | 0.0182 | n/a | 0 |

## Mistakes

- CNN: 8 of 360 test images wrong. Most common: true 1 predicted as 4 (3x), true 8 predicted as 7 (1x), true 8 predicted as 1 (1x).
- VIT: 18 of 360 test images wrong. Most common: true 9 predicted as 7 (3x), true 8 predicted as 1 (2x), true 1 predicted as 2 (2x).

Confusion matrices: `confusion_matrices.png`. Example errors: `errors.png`. Curves: `training_curves.png`.
