import re
from agent.parser import parse_output

MEANINGFUL_PAIRS = {
    ("svf_mean", "PET"), ("svf_mean", "MRT"),
    ("tvf_mean", "mean_temp"), ("tvf_mean", "PET"),
    ("H_W_ratio", "wind_speed"),
}

def is_meaningful_pair(x: str, y: str) -> bool:
    return (x, y) in MEANINGFUL_PAIRS or (y, x) in MEANINGFUL_PAIRS


def check_search_trigger(parsed: dict, messages: list, msg_idx: int) -> float:
    """
    判断search_literature调用是否在合适时机触发
    四种合理情况：R²偏低、分组差异显著、物理规律矛盾、R²强做延伸
    """
    prior_observations = [
        m["content"] for m in messages[:msg_idx]
        if m["role"] == "user" and "Observation:" in m["content"]
    ]
    if not prior_observations:
        return -0.1

    last_obs = prior_observations[-1]

    # 情况一：R²偏低触发
    r2_match = re.search(r"R[²2]\s*=\s*([0-9.]+)", last_obs)
    if r2_match:
        r2_val = float(r2_match.group(1))

        # R²低，被动检索
        if r2_val < 0.4:
            return 0.3

        # R²强，主动延伸检索（鼓励但权重低）
        if r2_val > 0.5:
            return 0.15

    # 情况二：分组后差异显著触发
    if "EW" in last_obs and "NS" in last_obs:
        r_values = re.findall(r"r\s*=\s*([0-9.]+)", last_obs)
        if len(r_values) >= 2:
            diff = abs(float(r_values[0]) - float(r_values[1]))
            if diff > 0.2:
                return 0.3

    # 情况三：物理规律矛盾触发
    if "负相关" in last_obs and "svf" in last_obs.lower():
        return 0.3
    if "正相关" in last_obs and "tvf" in last_obs.lower():
        return 0.3

    # 不在以上情况，冗余调用
    return -0.1


def turn_level_reward(messages: list, task: dict) -> float:
    """
    对每一轮工具调用单独打分
    """
    tool_calls = []
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and "<tool>" in m["content"]:
            parsed = parse_output(m["content"])
            if parsed["type"] == "tool_call":
                tool_calls.append((i, parsed))

    if not tool_calls:
        return 0.0

    scores = []
    prior_tool_names = []

    for call_idx, (msg_idx, parsed) in enumerate(tool_calls):
        tool = parsed["tool"]
        args = parsed["args"]

        if tool == "load_data":
            # 第一步加载数据合理，后续重复加载扣分
            score = 0.1 if call_idx == 0 else -0.05

        elif tool == "correlation_analysis":
            x = args.get("x", "")
            y = args.get("y", "")
            # 有意义的变量对加分，无意义扣分
            score = 0.2 if is_meaningful_pair(x, y) else -0.1

        elif tool == "regression_analysis":
            # 回归必须在相关性分析之后
            score = 0.15 if "correlation_analysis" in prior_tool_names else -0.05

        elif tool == "subgroup_analysis":
            # 触发分组分析本身是好行为
            score = 0.25

        elif tool == "search_literature":
            score = check_search_trigger(parsed, messages, msg_idx)

        elif tool == "generate_report":
            # 生成报告在最后是合理的
            score = 0.1

        else:
            score = 0.0

        scores.append(score)
        prior_tool_names.append(tool)

    return round(sum(scores) / len(scores), 3)