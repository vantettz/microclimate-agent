def compute_efficiency(messages: list) -> float:
    """
    计算工具调用效率
    调用次数越少得分越高
    
    Args:
        messages: 完整对话列表
    Returns:
        效率分数 0~1
    """
    # 统计tool call次数
    tool_calls = [
        m for m in messages
        if m["role"] == "assistant" and "<tool>" in m["content"]
    ]
    n_calls = len(tool_calls)

    # 3次以内满分，每多一次扣0.1，最低0
    score = max(0.0, 1.0 - (n_calls - 3) * 0.1)
    return round(score, 3)