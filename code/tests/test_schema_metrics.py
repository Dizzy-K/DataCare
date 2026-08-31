import unittest

from pollution_eval.backends.openai_compatible import prediction_from_api_text
from pollution_eval.constants import default_label_space
from pollution_eval.metrics import build_records, compute_metrics, summarize_records
from pollution_eval.schema import normalize_multilabel_value, prediction_from_mapping


class SchemaMetricsTest(unittest.TestCase):
    def test_normalize_multilabel_value_keeps_slash_label(self):
        self.assertEqual(normalize_multilabel_value("括号/符号不闭合"), ["括号/符号不闭合"])
        self.assertEqual(normalize_multilabel_value("民族偏见, 国别偏见"), ["民族偏见", "国别偏见"])

    def test_prediction_from_api_text_json_fence(self):
        text = '```json\n{"label_l1":"污染样本","label_l2":["偏见类"],"label_l3":["群体受害者维度"],"label_l4":["民族偏见"]}\n```'
        pred = prediction_from_api_text(text)
        self.assertEqual(pred["pred_label_l1"], "污染样本")
        self.assertEqual(pred["pred_label_l4"], ["民族偏见"])

    def test_metrics_joint_accuracy_with_clean_and_polluted(self):
        rows = [
            {"id": "clean_1", "label_l1": "非污染样本", "label_l2": None, "label_l3": None, "label_l4": None},
            {"id": "dirty_1", "label_l1": "污染样本", "label_l2": "偏见类", "label_l3": "群体受害者维度", "label_l4": "民族偏见"},
        ]
        predictions = [
            prediction_from_mapping({"id": "clean_1", "pred_label_l1": "非污染样本"}),
            prediction_from_mapping(
                {
                    "id": "dirty_1",
                    "pred_label_l1": "污染样本",
                    "pred_label_l2": ["偏见类"],
                    "pred_label_l3": ["群体受害者维度"],
                    "pred_label_l4": ["民族偏见"],
                }
            ),
        ]
        records = build_records(rows, predictions)
        metrics = compute_metrics(records, default_label_space())
        summary = summarize_records(records, [])
        self.assertEqual(metrics["binary_accuracy"], 1.0)
        self.assertEqual(metrics["joint_accuracy"], 1.0)
        self.assertEqual(metrics["l4_exact_match"], 1.0)
        self.assertEqual(summary["false_negative_count"], 0)


if __name__ == "__main__":
    unittest.main()
