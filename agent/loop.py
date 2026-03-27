import os
import dashscope
from agent.tools_registry import TOOLS
from agent.parser import parse_output
from agent.context_manager import (
    build_initial_context,
    append_tool_result,
    compress_context,
    get_context_length,
)

MAX_CONTEXT_TOKENS = 3000
MAX_TURNS = 10

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

完成分析后输出：
<answer>你的结论</answer>"""





def run_analysis(question: str,
                 csv_path: str = None,
                 image_path: str = None,
                 audio_path: str = None) -> list:
    """
    三路输入的完整分析流程

    Args:
        question: 用户问题
        csv_path: CSV文件路径（可选）
        image_path: 图片文件路径（可选）
        audio_path: 音频文件路径（可选）
    Returns:
        完整 messages 列表
    """
    # 构建初始上下文
    task = {
        "question": question,
        "data_path": csv_path or "",
    }
    messages = build_initial_context(task)

    # 把有效输入路径注入第一条user消息
    input_info = []
    if csv_path:
        input_info.append(f"CSV数据路径：{csv_path}")
    if image_path:
        input_info.append(f"街景图片路径：{image_path}")
    if audio_path:
        input_info.append(f"音频文件路径：{audio_path}")

    if input_info:
        messages[-1]["content"] += "\n" + "\n".join(input_info)

    for turn in range(MAX_TURNS):

        # 上下文过长时压缩
        if get_context_length(messages) > MAX_CONTEXT_TOKENS:
            messages = compress_context(messages, keep_last=4)

        # API 推理
        response = generate_api(messages)

        # 解析输出
        parsed = parse_output(response)

        if parsed["type"] == "final_answer":
            messages.append({
                "role": "assistant",
                "content": response
            })
            break

        if parsed["type"] == "tool_call":
            tool_name = parsed["tool"]
            args = parsed["args"]

            if tool_name in TOOLS:
                try:
                    result = TOOLS[tool_name](**args)
                except Exception as e:
                    result = f"工具执行出错：{str(e)}"
            else:
                result = f"未知工具：{tool_name}"

            messages = append_tool_result(messages, response, result)

    return messages


def generate_api(messages: list) -> str:
    """调用 Qwen API 生成下一轮输出"""
    from dashscope import Generation
    response = Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen-turbo",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
    )
    return response.output.text