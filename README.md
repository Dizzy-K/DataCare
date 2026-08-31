# TrainGuard

TrainGuard is an open release for detecting and governing data pollution in language-model inputs. It combines a hierarchical pollution-label schema, public evaluation data, and a toolkit for running or rescoring detection models.

## Repository layout

- [`code/`](code/): Python evaluation package with local Qwen, OpenAI-compatible API, and prediction-rescoring backends.
- [`datasets/`](datasets/): 20,000 public JSONL test records (10,000 clean and 10,000 pollution samples).

## Label schema

The schema has four levels:

1. **L1**: `非污染样本` / `污染样本`.
2. **L2**: pollution families such as `偏见类` and `语义缺失类`.
3. **L3**: categories including `结构与语法残缺`, `逻辑与上下文断层`, `群体受害者维度`, and `价值观偏见`.
4. **L4**: leaf labels defined in [`code/configs/label_space.json`](code/configs/label_space.json), for example `民族偏见`, `句式截断`, and `逻辑错误`.

## Quick start

```bash
cd code
python -m pip install -e .
python -m pollution_eval.cli.evaluate \
  --backend api \
  --input-dir ../datasets \
  --output-dir ../runs/example \
  --label-space configs/label_space.json \
  --api-model YOUR_MODEL \
  --api-base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY
```

See [`code/README.md`](code/README.md) for local Qwen inference, prediction rescoring, input details, and metric definitions. See [`datasets/README.md`](datasets/README.md) for the dataset schema and split statistics.

## Notes

- The repository does not embed private checkpoints or credentials. Supply model paths and API keys from your own environment.
- Review the dataset and model outputs for your intended use and follow applicable data, privacy, and safety requirements.
