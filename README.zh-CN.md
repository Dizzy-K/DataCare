# DataCARE-CN

[English](README.md) | 简体中文

DataCARE-CN 是面向大语言模型文本与对话数据的**分层数据污染检测与评测项目**。项目通过 L1–L4 四级标签描述样本是否存在污染、属于哪类污染，以及具体的污染类型，为数据质量分析、检测模型对比和错误样本排查提供统一的评测基础。

当前版本提供 **20,000 条公开测试样本、四级标签体系和 Python 评测工具**，覆盖偏见与语义缺失两大类污染。评测工具支持调用模型 API、运行本地 Qwen 四分类头模型，以及对已有预测结果重新评分。当前仓库不包含模型训练代码或预训练、微调权重。

## 主要功能

- **分层、多标签评测**：L1 判断是否污染，L2–L4 逐级细化污染类型，支持同一样本包含多个污染标签。
- **三种评测后端**：通过 OpenAI 兼容接口评测模型、加载本地 Qwen + LoRA + 四分类头检查点，或直接读取预测文件计算指标。
- **可复核的评测输出**：保存逐条预测、二分类与多标签指标、误报与漏报样本、分层错误样本及运行元数据。
- **批量 API 评测**：支持并发请求、失败重试和按样本 ID 断点续跑。

## 目录结构

```text
DataCARE-CN/
├── README.md                       # 英文说明
├── README.zh-CN.md                 # 中文说明
├── datasets/
│   ├── README.md                    # 数据集说明
│   ├── white_test.jsonl             # 非污染测试样本
│   └── pollution_test.jsonl         # 污染测试样本
└── code/
    ├── README.md                    # 工具详细说明（英文）
    ├── pyproject.toml               # Python 包配置与依赖
    ├── environment-open-eval.yml    # 可选 Conda 环境配置
    ├── configs/label_space.json     # 标签名称、编号与默认阈值
    ├── pollution_eval/              # 数据处理、推理后端与评测逻辑
    ├── scripts/                     # API、本地推理示例与本地服务脚本
    └── tests/                       # 现有测试
```

## 标签体系

| 层级 | 含义 | 当前标签 |
| --- | --- | --- |
| L1 | 是否存在污染 | `非污染样本`、`污染样本` |
| L2 | 污染大类（2 类） | `偏见类`、`语义缺失类` |
| L3 | 污染子类（4 类） | `群体受害者维度`、`价值观偏见`、`结构与语法残缺`、`逻辑与上下文断层` |
| L4 | 具体污染类型（20 类） | 如 `民族偏见`、`性别偏见`、`句式截断`、`逻辑错误` 等 |

完整标签及其编号见 [`label_space.json`](code/configs/label_space.json)，类别定义见 [`constants.py`](code/pollution_eval/constants.py)。训练或接入自己的模型时，应保持中文标签名称和分类头输出顺序与标签配置一致。

非污染样本的 L2–L4 为空；污染样本的 L2–L4 可以包含多个标签。默认二分类阈值和多标签阈值均为 `0.5`，用于本地 Qwen 后端的概率解码。

## 数据集

| 文件 | 样本数 | 说明 |
| --- | ---: | --- |
| [`white_test.jsonl`](datasets/white_test.jsonl) | 10,000 | 非污染样本，L2–L4 为 `null` |
| [`pollution_test.jsonl`](datasets/pollution_test.jsonl) | 10,000 | 污染样本，带有 L2–L4 标签 |
| 合计 | **20,000** | 用于评测与结果复现 |

当前测试集实际覆盖 2 个 L2 类别、2 个 L3 类别和 12 个 L4 类别，并未覆盖标签体系的全部类别。其中，偏见类为 6,000 条，语义缺失类为 4,000 条；L3 分别为 `群体受害者维度` 和 `结构与语法残缺`。评估其他类别时需补充对应测试样本。

数据采用 UTF-8 JSONL 格式，每行一个 JSON 对象。以下为非污染样本的格式示例：

```json
{
  "id": "example_clean_001",
  "messages": [
    {"role": "user", "content": "请计算 1 加 1。"},
    {"role": "assistant", "content": "1 加 1 等于 2。"}
  ],
  "label_l1": "非污染样本",
  "label_l2": null,
  "label_l3": null,
  "label_l4": null
}
```

`id` 用于预测结果对齐，应保持唯一。`messages` 保存待检测的对话；接入自有数据时也可以使用 `query` / `answer` 或 `text` 字段。污染样本的 L2–L4 标签可以是字符串或字符串数组。

这两份文件是测试集。若需要训练或调参，请另行准备独立的训练集与验证集，保持测试集独立。更多字段说明见 [`datasets/README.md`](datasets/README.md)。

## 快速开始

### 1. 安装

需要 **Python 3.10 或更新版本**。以下使用 Linux / macOS 的终端命令：

```bash
git clone https://github.com/Dizzy-K/TrainGuard.git DataCARE-CN
cd DataCARE-CN
python3 -m venv .venv
source .venv/bin/activate
cd code
python -m pip install -e .
```

基础安装即可使用 API 评测和预测文件重新评分。下文命令均在 `DataCARE-CN/code/` 目录执行，并使用上述 Python 环境。

### 2. 整理数据并检查标签

此步骤不调用模型，也不需要 GPU 或 API 密钥：

```bash
python -m pollution_eval.cli.prepare_dataset \
  --input-dir ../datasets \
  --output-dir ../runs/manifest \
  --label-space configs/label_space.json \
  --unknown-label-policy error
```

输出包括合并后的 `eval_all.jsonl`、标签配置和记录样本数量、标签分布的 `manifest.json`。使用 `error` 策略时，发现标签空间之外的标注会直接报错。该步骤便于检查自有数据；评测命令也可以直接读取原始测试集。

### 3. 通过 API 评测模型

将密钥放入环境变量，并把 `YOUR_MODEL` 替换为服务提供的模型名称；使用其他兼容服务时，修改 `--api-base-url`：

```bash
export OPENAI_API_KEY="YOUR_API_KEY"

python -m pollution_eval.cli.evaluate \
  --backend api \
  --input-dir ../datasets \
  --output-dir ../runs/api_smoke \
  --label-space configs/label_space.json \
  --api-base-url https://api.openai.com/v1 \
  --api-model YOUR_MODEL \
  --api-key-env OPENAI_API_KEY \
  --api-workers 4 \
  --limit 20
```

此示例只处理加载后的前 20 条样本，用于检查接口和输出格式，不能代表完整测试集表现。全量评测时删除 `--limit 20`，并使用新的输出目录，例如 `../runs/api_full`。

工具会自动构造包含标签定义的检测提示词。模型应返回以下形式的 JSON；标签必须来自发布的中文标签空间：

```json
{
  "label_l1": "污染样本",
  "label_l2": ["偏见类"],
  "label_l3": ["群体受害者维度"],
  "label_l4": ["民族偏见"]
}
```

若判断为非污染样本，L2–L4 应返回空数组。`reason` 为可选字段，不参与评分。

默认调用 `chat/completions`；使用 Responses 接口时添加 `--api-type responses`。若兼容服务不支持 `response_format`，可添加 `--no-response-format`，但模型仍需输出可解析的 JSON。

API 评测默认读取同一输出目录下的 `api_raw_predictions.jsonl`，按样本 ID 复用已有结果。**更换模型、数据内容或推理参数时，请使用新的输出目录**，避免复用旧预测。API 请求失败且重试耗尽时，默认终止运行。

## 其他评测方式

### 本地 Qwen 四分类头模型

先安装本地推理的可选依赖：

```bash
python -m pip install -e ".[local-qwen]"
```

该后端需要与代码结构匹配的 **Qwen 基座 + LoRA + L1–L4 四个分类头**检查点。`--checkpoint-dir` 中需有 `model.safetensors`；如果目录中有 `tokenizer.json`，则优先使用该目录的分词器，否则使用基座模型的分词器。仅下载普通 Qwen 对话模型不足以运行此后端，仓库当前也未附带所需检查点。

准备好匹配的权重后，可在支持相应量化与精度的 GPU 环境中执行：

```bash
python -m pollution_eval.cli.evaluate \
  --backend local-qwen \
  --input-dir ../datasets \
  --output-dir ../runs/qwen_local \
  --label-space configs/label_space.json \
  --checkpoint-dir /path/to/checkpoint \
  --base-model Qwen/Qwen3-8B \
  --dtype bf16 \
  --load-in-4bit \
  --batch-size 1 \
  --max-length 4096
```

基座模型必须与检查点训练时使用的模型一致。加载器的默认 LoRA 配置为 `r=16`、`alpha=32`，具体结构见 [`local_qwen.py`](code/pollution_eval/backends/local_qwen.py)。`--max-length` 限制包含任务说明和对话的完整输入，超长内容会被截断；应结合数据长度与显存设置，并在报告结果时记录该值。4-bit 与 8-bit 量化参数不能同时使用。

### 对已有预测重新评分

无需调用模型，可直接对工具生成的预测文件重新计算指标：

```bash
python -m pollution_eval.cli.evaluate \
  --backend predictions \
  --input-dir ../datasets \
  --output-dir ../runs/rescore \
  --label-space configs/label_space.json \
  --predictions-file /path/to/eval_predictions.jsonl
```

支持 JSONL，以及 JSON 数组或包含 `records` 数组的 JSON 文件。每条预测必须有与评测样本对应的 `id`，标签字段可以使用 `label_l1`–`label_l4` 或 `pred_label_l1`–`pred_label_l4`；缺少任一待评测 ID 的预测时会报错。若预测仅覆盖子集，需用 `--eval-files` 或相同的 `--limit` 选择对应输入。

### 常用参数

| 参数 | 用途 |
| --- | --- |
| `--eval-files FILE [FILE ...]` | 指定一个或多个 JSONL 文件，优先于 `--input-dir` |
| `--limit N` | 仅评测加载后的前 N 条样本，不进行随机或分层抽样 |
| `--unknown-label-policy skip\|error` | 处理标注中未知的 L2–L4 标签；默认跳过并记录 |
| `--binary-threshold` / `--multilabel-threshold` | 调整本地 Qwen 概率解码阈值；不对已有离散预测重新解码 |
| `--api-workers` | API 并发数，默认 `1` |
| `--no-api-resume` | 本次 API 运行不读取已有预测缓存 |

完整参数可通过 `python -m pollution_eval.cli.evaluate --help` 查看。

## 评测结果与指标

所有结果写入 `--output-dir` 指定的目录：

| 文件 | 内容 |
| --- | --- |
| `eval_metrics.json` | 二分类、多标签和联合准确率指标 |
| `eval_predictions.json` / `eval_predictions.jsonl` | 逐样本标注、预测与正确性信息；JSON 版本另含指标与摘要 |
| `badcase_summary.json` | 错误数量、标签分布和按输入文件汇总的信息 |
| `false_positives.jsonl` / `false_negatives.jsonl` | 误报与漏报样本 |
| `l2_errors.jsonl` / `l3_errors.jsonl` / `l4_errors.jsonl` | 真实与预测均为污染、但对应层标签不匹配的样本 |
| `joint_errors.jsonl` | 未满足联合正确条件的全部样本 |
| `skipped_unknown_labels.jsonl` | 因标注包含未知标签而跳过的样本 |
| `label_space.json` / `run_metadata.json` | 本次标签配置、阈值及后端运行信息 |
| `api_raw_predictions.jsonl` | API 后端的原始响应及预测缓存，用于断点续跑 |

指标口径如下：

- **L1 二分类**：以 `污染样本` 为正类，报告 Accuracy、Precision、Recall、F1、FPR（误报率）和 FNR（漏报率）。
- **L2–L4 多标签**：在真实污染样本上计算 Micro-F1、Macro-F1 和 Exact Match。Macro-F1 仅对当前评测子集中有真实正例的标签取平均；Exact Match 要求该层标签集合完全一致。
- **`joint_accuracy`**：非污染样本只需 L1 正确；污染样本需 L1 正确且 L2、L3、L4 标签集合全部匹配。
- **`polluted_joint_accuracy`**：仅在真实污染样本上计算联合准确率。

为便于比较结果，请同时记录模型或检查点、测试文件、样本数、阈值，以及本地推理的截断长度或 API 推理设置。

## 开发与反馈

运行仓库现有测试：

```bash
python -m unittest discover -s tests
```

欢迎通过 [Issues](https://github.com/Dizzy-K/TrainGuard/issues) 反馈数据标注问题、接口兼容问题或评测异常，也欢迎提交 Pull Request。反馈时请附上可复现的命令、环境版本及相关样本 ID，并去除 API 密钥等凭据。
