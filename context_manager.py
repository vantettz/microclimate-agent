# agent/context_manager.py

def build_initial_context(task: dict, long_term_memory: str = "") -> list:
    """
    构建初始 messages，注入长期记忆和任务信息
    
    Args:
        task: 任务字典，包含 question 和 data_path
        long_term_memory: 从长期记忆里检索到的历史发现
    Returns:
        初始 messages 列表
    """
    system_prompt = """你是一个城市微气候数据分析专家。
通过调用工具分析数据，找出形态指标与热舒适参数之间的显著关系。

可用工具：
- load_data(filepath): 读取CSV，返回变量列表和基本统计
- correlation_analysis(filepath, x, y): 计算Pearson相关性
- regression_analysis(filepath, x, y): 线性回归分析
- subgroup_analysis(filepath, x, y, group_by): 分组相关性分析
- search_literature(query): 检索相关文献
- generate_report(findings, save_path): 生成分析报告

调用格式（严格遵守）：
<tool>工具名</tool><args>{"参数名": "参数值"}</args>

完成分析后输出：
<answer>你的结论</answer>

分析策略：
1. 先调用load_data了解数据结构
2. 优先探索物理意义显著的变量对：SVF-PET、TVF-温度、H/W-风速
3. 显著相关后做回归，R²<0.4时检索文献
4. 发现显著关系后考虑按朝向或街道类型分组验证
5. 找到足够发现后生成报告，不要无限探索"""

    user_content = f"{task['question']}\n数据路径：{task['data_path']}"
    
    # 注入长期记忆
    if long_term_memory:
        user_content += f"\n\n历史研究发现（供参考）：\n{long_term_memory}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]


def append_tool_result(messages: list, 
                       tool_call: str, 
                       observation: str) -> list:
    """
    把一轮工具调用和结果追加进 messages
    
    Args:
        messages: 当前对话历史
        tool_call: 模型输出的工具调用文本
        observation: 工具执行结果
    Returns:
        更新后的 messages
    """
    messages.append({
        "role": "assistant",
        "content": tool_call
    })
    messages.append({
        "role": "user",
        "content": f"Observation: {observation}"
    })
    return messages


def compress_context(messages: list, keep_last: int = 4) -> list:
    """
    上下文压缩：保留最近N轮，早期轮次压缩成摘要
    避免多轮后 context 过长导致显存溢出或模型迷失
    
    Args:
        messages: 当前完整 messages
        keep_last: 保留最近几轮完整内容（每轮=assistant+user两条）
        
    Returns:
        压缩后的 messages
    """
    # system + user(任务) + 对话历史
    # 对话历史每轮两条：assistant(tool_call) + user(observation)
    header = messages[:2]           # system + 初始user
    history = messages[2:]          # 所有对话轮次
    
    # 每轮两条消息，keep_last轮 = keep_last*2条消息
    keep_n = keep_last * 2
    
    if len(history) <= keep_n:
        return messages             # 不需要压缩
    
    early_history = history[:-keep_n]
    recent_history = history[-keep_n:]
    
    # 把早期历史压缩成摘要
    summary = _summarize_history(early_history)
    summary_msg = {
        "role": "user",
        "content": f"前期分析摘要：{summary}"
    }
    
    return header + [summary_msg] + recent_history


def _summarize_history(history: list) -> str:
    """
    把早期对话历史压缩成摘要
    用规则提取关键发现，不调用模型（节省开销）
    """
    findings = []
    
    for msg in history:
        if msg["role"] == "user" and "Observation:" in msg["content"]:
            obs = msg["content"].replace("Observation:", "").strip()
            
            # 提取显著相关的发现
            if "极显著" in obs or "强相关" in obs:
                # 提取变量对和r值
                import re
                pair = re.search(r"相关性分析结果：(\w+) vs (\w+)", obs)
                r_val = re.search(r"r\s*=\s*([0-9.-]+)", obs)
                if pair and r_val:
                    findings.append(
                        f"{pair.group(1)}-{pair.group(2)} 显著相关"
                        f"（r={r_val.group(1)}）"
                    )
            
            # 提取回归结果
            if "R²" in obs:
                r2 = re.search(r"R²\s*=\s*([0-9.]+)", obs)
                if r2:
                    findings.append(f"R²={r2.group(1)}")
    
    if findings:
        return "已发现：" + "；".join(findings)
    return "前期探索未发现显著关系"


def get_context_length(messages: list) -> int:
    """估算当前context的token数（粗略按字符数/2估算）"""
    total_chars = sum(len(m["content"]) for m in messages)
    return total_chars // 2