import re
import json

def parse_output(text: str) -> dict:
    """
    解析LLM输出，返回tool_call或final_answer
    
    LLM输出格式：
    工具调用：<tool>工具名</tool><args>{"key": "value"}</args>
    最终答案：<answer>结论文字</answer>
    """
    # 检查tool call
    tool_match = re.search(
        r"<tool>(.*?)</tool><args>(.*?)</args>",
        text, re.DOTALL
    )
    if tool_match:
        tool_name = tool_match.group(1).strip()
        try:
            args = json.loads(tool_match.group(2).strip())
        except json.JSONDecodeError:
            args = {}
        return {"type": "tool_call", "tool": tool_name, "args": args}

    # 检查final answer
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        return {"type": "final_answer", "content": answer_match.group(1).strip()}

    # 既没有tool call也没有answer
    return {"type": "final_answer", "content": text}