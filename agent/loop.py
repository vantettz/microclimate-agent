from agent.tools_registry import TOOLS
from agent.parser import parse_output
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from agent.context_manager import (
    build_initial_context,
    append_tool_result,
    compress_context,
    get_context_length,
)
from agent.tools_registry import TOOLS
from agent.parser import parse_output

MAX_CONTEXT_TOKENS = 3000   # 超过这个长度就压缩
MAX_TURNS = 10

SYSTEM_PROMPT = """你是一个城市微气候数据分析专家。
通过调用工具分析数据，找出形态指标与热舒适参数之间的显著关系。

可用工具：
- load_data(filepath): 读取CSV，返回变量列表和基本统计
- correlation_analysis(filepath, x, y): 计算两变量的Pearson相关性
- regression_analysis(filepath, x, y): 对显著相关变量做线性回归
- search_literature(query): 检索相关文献
- generate_report(findings, save_path): 生成分析报告

调用格式（严格遵守）：
<tool>工具名</tool><args>{"参数名": "参数值"}</args>

完成分析后输出：
<answer>你的结论</answer>"""


def run_episode(model, tokenizer, task, 
                long_term_memory: str = "") -> list:
    """
    运行一条样本的完整推理过程

    Args:
        model: 语言模型
        tokenizer: 分词器
        task: 任务字典
        long_term_memory: 历史发现，从长期记忆检索
    Returns:
        完整 messages 列表
    """
    # 构建初始上下文
    messages = build_initial_context(task, long_term_memory)

    for turn in range(MAX_TURNS):

        # 上下文过长时压缩
        if get_context_length(messages) > MAX_CONTEXT_TOKENS:
            messages = compress_context(messages, keep_last=4)

        # LLM 推理
        response = generate(model, tokenizer, messages)

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

            # 执行工具
            if tool_name in TOOLS:
                try:
                    result = TOOLS[tool_name](**args)
                except Exception as e:
                    result = f"工具执行出错：{str(e)}"
            else:
                result = f"未知工具：{tool_name}"

            # 追加这轮结果
            messages = append_tool_result(messages, response, result)

    return messages


def generate(model, tokenizer, messages):
    """调用模型生成下一轮输出"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)