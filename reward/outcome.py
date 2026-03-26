import json

def extract_significant_findings(messages: list) -> set:
    """
    从agent对话历史中提取找到的显著变量对
    """
    findings = set()
    for m in messages:
        if m["role"] == "user" and "Observation:" in m["content"]:
            content = m["content"]
            # 解析相关性分析结果
            if "强相关" in content and "极显著" in content:
                # 提取变量对
                import re
                match = re.search(r"相关性分析结果：(\w+) vs (\w+)", content)
                if match:
                    x, y = match.group(1), match.group(2)
                    findings.add((x, y))
    return findings


def compute_f1(messages: list, task: dict) -> float:
    """
    计算F1分数：找到的显著关系和ground truth的匹配程度

    Args:
        messages: 完整对话列表
        task: 任务字典，包含ground_truth
    Returns:
        F1分数 0~1
    """
    found = extract_significant_findings(messages)
    
    # ground truth转成set
    truth_list = task["ground_truth"]["significant_pairs"]
    truth = set(tuple(pair) for pair in truth_list)

    if not truth:
        return 0.0
    if not found:
        return 0.0

    # 计算交集
    correct = len(found & truth)

    recall    = correct / len(truth)
    precision = correct / len(found)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return round(f1, 3)