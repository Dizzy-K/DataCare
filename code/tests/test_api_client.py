import unittest

from pollution_eval.backends.openai_compatible import OpenAICompatibleClient, prediction_from_api_text
from pollution_eval.cli.evaluate import resolve_thinking


class ApiClientTest(unittest.TestCase):
    def test_reasoning_effort_is_sent_when_configured(self):
        client = OpenAICompatibleClient(
            base_url="http://127.0.0.1:3000/v1",
            model="k3-256k",
            api_key="dummy",
            reasoning_effort="low",
        )
        payload = client._payload([{"role": "user", "content": "test"}])
        self.assertEqual(payload["reasoning_effort"], "low")

    def test_reasoning_effort_is_omitted_by_default(self):
        client = OpenAICompatibleClient(
            base_url="http://127.0.0.1:3000/v1",
            model="gpt-4.1-mini",
            api_key="dummy",
        )
        payload = client._payload([{"role": "user", "content": "test"}])
        self.assertNotIn("reasoning_effort", payload)

    def test_thinking_disabled_is_sent_as_provider_object(self):
        client = OpenAICompatibleClient(
            base_url="http://127.0.0.1:3000/v1",
            model="deepseek-v4-pro",
            api_key="dummy",
            thinking="disabled",
        )
        payload = client._payload([{"role": "user", "content": "test"}])
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_deepseek_thinking_defaults_to_disabled(self):
        self.assertEqual(resolve_thinking("deepseek-v4-pro", None), "disabled")
        self.assertEqual(resolve_thinking("deepseek-v4-pro", "enabled"), "enabled")
        self.assertIsNone(resolve_thinking("k3-256k", None))

    def test_malformed_reason_does_not_discard_valid_labels(self):
        text = (
            '{"label_l1":"污染样本","label_l2":["语义缺失类"],'
            '"label_l3":["逻辑与上下文断层"],"label_l4":["时序倒置"],'
            '"reason":"名称为"Ianuarius"和"Februarius"，导致时间顺序混乱。"}'
        )
        prediction = prediction_from_api_text(text)
        self.assertEqual(prediction["pred_label_l1"], "污染样本")
        self.assertEqual(prediction["pred_label_l4"], ["时序倒置"])
        self.assertEqual(prediction["raw_response"], text)
        self.assertIn("parse_warning", prediction)


if __name__ == "__main__":
    unittest.main()
