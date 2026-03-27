import re

SYSTEM_PROMPT = """你是一个城市街道环境质量分析专家，能综合分析热舒适、声环境和视觉质量三个维度。

分析原则：
1. 结论必须有数据支撑，不能凭空判断
2. 指出问题的同时给出具体改善建议
3. 用研究者能直接引用的学术语言表达
4. 如果某个维度数据缺失，明确说明而不是猜测

可用工具：
- load_data(filepath): 读取CSV，返回变量列表和基本统计
- correlation_analysis(filepath, x, y): 计算两变量的Pearson相关性
- regression_analysis(filepath, x, y): 对显著相关变量做线性回归
- vision_analysis(image_path): 分析街景图片，返回视觉质量和微气候描述
- noise_analysis(audio_path): 分析音频文件，返回声环境质量评估
- generate_report(findings, save_path): 生成综合分析报告

调用格式（严格遵守）：
<tool>工具名</tool><args>{"参数名": "参数值"}</args>

分析策略：
1. 有CSV时先调用load_data了解数据结构
2. 优先探索SVF-PET、TVF-温度、H/W-风速等有意义的变量对
3. 有图片时调用vision_analysis获取视觉质量描述
4. 有音频时调用noise_analysis获取声环境评估
5. 所有维度分析完成后调用generate_report输出综合报告

完成分析后输出：
<answer>你的结论</answer>"""


def build_initial_context(task: dict, long_term_memory: str = "") -> list:
    user_content = task["question"]
    if task.get("data_path"):
        user_content += f"\n数据路径：{task['data_path']}"
    if long_term_memory:
        user_content += f"\n\n历史研究发现（供参考）：\n{long_term_memory}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def append_tool_result(messages: list,
                       tool_call: str,
                       observation: str) -> list:
    messages.append({"role": "assistant", "content": tool_call})
    messages.append({"role": "user", "content": f"Observation: {observation}"})
    return messages


def compress_context(messages: list, keep_last: int = 4) -> list:
    header = messages[:2]
    history = messages[2:]
    keep_n = keep_last * 2

    if len(history) <= keep_n:
        return messages

    early_history = history[:-keep_n]
    recent_history = history[-keep_n:]
    summary = _summarize_history(early_history)

    return header + [
        {"role": "user", "content": f"前期分析摘要：{summary}"}
    ] + recent_history


def _summarize_history(history: list) -> str:
    findings = []
    for msg in history:
        if msg["role"] == "user" and "Observation:" in msg["content"]:
            obs = msg["content"].replace("Observation:", "").strip()

            if "极显著" in obs or "强相关" in obs:
                pair  = re.search(r"相关性分析结果：(\w+) vs (\w+)", obs)
                r_val = re.search(r"r\s*=\s*([0-9.-]+)", obs)
                if pair and r_val:
                    findings.append(
                        f"{pair.group(1)}-{pair.group(2)} 显著相关"
                        f"（r={r_val.group(1)}）"
                    )

            if "R²" in obs:
                r2 = re.search(r"R²\s*=\s*([0-9.]+)", obs)
                if r2:
                    findings.append(f"R²={r2.group(1)}")

            # 新增：提取视觉和声环境关键信息
            if "SVF" in obs or "TVF" in obs:
                svf = re.search(r"SVF[=＝]([0-9.]+)", obs)
                if svf:
                    findings.append(f"SVF={svf.group(1)}")

            if "dB" in obs:
                db = re.search(r"([0-9.]+)\s*dB", obs)
                if db:
                    findings.append(f"噪声均值={db.group(1)}dB")

    return "已发现：" + "；".join(findings) if findings else "前期探索未发现显著结果"


def get_context_length(messages: list) -> int:
    total_chars = sum(len(m["content"]) for m in messages)
    return total_chars // 2