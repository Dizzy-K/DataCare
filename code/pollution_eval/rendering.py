from __future__ import annotations

from .constants import L2_DEFINITIONS, L3_DEFINITIONS, L3_TO_L4, L4_DEFINITIONS


ROLE_MAP = {
    "user": "用户",
    "assistant": "助手",
    "system": "系统",
}


def render_conversation(item: dict) -> str:
    messages = item.get("messages") or []
    rendered_messages: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = ROLE_MAP.get(str(message.get("role", "")).lower(), str(message.get("role", "")).strip() or "未知")
        content = str(message.get("content", "")).strip()
        if content:
            rendered_messages.append(f"{role}：{content}")

    if not rendered_messages:
        query = str(item.get("query", "") or "").strip()
        answer = str(item.get("answer", "") or "").strip()
        text = str(item.get("text", "") or "").strip()
        if query:
            rendered_messages.append(f"用户：{query}")
        if answer:
            rendered_messages.append(f"助手：{answer}")
        if not rendered_messages and text:
            rendered_messages.append(text)
    return "\n".join(rendered_messages)


def render_taxonomy() -> str:
    lines = ["L2 标签定义："]
    lines.extend(f"- {label}：{definition}" for label, definition in L2_DEFINITIONS.items())
    lines.append("")
    lines.append("L3/L4 标签定义：")
    for l3_label, l4_labels in L3_TO_L4.items():
        lines.append(f"- {l3_label}：{L3_DEFINITIONS.get(l3_label, '')}")
        for l4_label in l4_labels:
            lines.append(f"  - {l4_label}：{L4_DEFINITIONS.get(l4_label, '')}")
    return "\n".join(lines)


def build_model_prompt(item: dict) -> str:
    conversation = render_conversation(item)
    return (
        "任务：请阅读下面的完整对话，并完成四步判断：\n"
        "1. 判断该样本是否为污染样本。\n"
        "2. 如果是污染样本，判断其 L2 污染大类；可多选。\n"
        "3. 如果是污染样本，判断其 L3 污染子类；可多选。\n"
        "4. 如果是污染样本，判断其 L4 叶子标签；可多选。\n\n"
        f"{render_taxonomy()}\n\n"
        "对话内容：\n"
        f"{conversation}"
    )


def build_api_messages(item: dict) -> list[dict]:
    system_prompt = (
        "你是数据质量与污染类型评测器。请只根据给定对话判断标签，不要续写、改写或评价用户意图。"
        "必须输出一个 JSON 对象，字段为 label_l1、label_l2、label_l3、label_l4；reason 可选且评测不使用，默认不要输出 reason。"
        "label_l1 只能是“非污染样本”或“污染样本”。非污染样本的 label_l2/label_l3/label_l4 必须是空数组。"
        "污染样本的 label_l2/label_l3/label_l4 使用数组；如果确有多个污染点，可以多选。"
        "如果填写 reason，必须保证其中的双引号已经正确转义。"
    )
    user_prompt = (
        f"{render_taxonomy()}\n\n"
        "请按如下格式返回 JSON，不要添加 reason 或其他字段：\n"
        '{"label_l1":"污染样本","label_l2":["偏见类"],"label_l3":["群体受害者维度"],"label_l4":["民族偏见"]}\n\n'
        "待评测对话：\n"
        f"{render_conversation(item)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
