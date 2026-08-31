# Public Datasets

This directory contains the public test sets for hierarchical data-pollution detection.

## Files

| File | Records | Description |
|---|---:|---|
| `white_test.jsonl` | 10,000 | Clean samples (`label_l1` = `非污染样本`); lower-level labels are `null`. |
| `pollution_test.jsonl` | 10,000 | Pollution samples (`label_l1` = `污染样本`) with L2/L3/L4 labels. |

Each line is a UTF-8 JSON object with these fields:

- `id`: stable sample identifier.
- `messages`: input conversation as role/content objects.
- `label_l1`: `非污染样本` or `污染样本`.
- `label_l2`, `label_l3`, `label_l4`: pollution labels (a string or array); `null` for clean samples.

The complete label vocabulary and numeric ordering are defined in [`../code/configs/label_space.json`](../code/configs/label_space.json). Chinese label names are part of the released schema and should be preserved when training or evaluating models.

The two files contain 20,000 records in total. They are intended for evaluation and reproducibility; create separate training/validation splits if needed.
