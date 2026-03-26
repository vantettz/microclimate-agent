import re
from reward.outcome import compute_f1
from reward.efficiency import compute_efficiency
from reward.conclusion_reward import conclusion_reward


def extract_conclusion(messages: list) -> str:
    """从messages里提取最终结论"""
    for m in reversed(messages):
        if m["role"] == "assistant":
            match = re.search(r"<answer>(.*?)</answer>", 
                            m["content"], re.DOTALL)
            if match:
                return match.group(1).strip()
            # 没有answer标签就取最后一条assistant消息
            return m["content"]
    return ""


def reward_fn(messages: list, task: dict) -> float:
    """
    完整reward函数，供ART训练调用

    Args:
        messages: agent完整对话历史
        task: 任务字典，包含ground_truth和domain_context
    Returns:
        最终reward分数 0~1
    """
    conclusion = extract_conclusion(messages)
    ground_truth = task.get("ground_truth", {})
    domain_context = task.get("domain_context", {})

    # 工具调用层
    f1         = compute_f1(messages, task)
    efficiency = compute_efficiency(messages)
    tool_layer = 0.6 * f1 + 0.4 * efficiency

    # 结论生成层
    conc_layer = conclusion_reward(conclusion, ground_truth, domain_context)

    # 加权合并
    total = 0.6 * tool_layer + 0.4 * conc_layer

    return round(total, 3)