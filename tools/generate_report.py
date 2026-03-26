import os
from datetime import datetime

def generate_report(findings: str, save_path: str) -> str:
    """
    把分析发现汇总成报告并保存
    
    Args:
        findings: 所有分析发现的文字描述
        save_path: 报告保存路径
    Returns:
        字符串，保存路径或错误信息
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    except Exception:
        pass

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
城市微气候分析报告
生成时间：{timestamp}
{"="*50}

{findings}

{"="*50}
报告结束
""".strip()

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        return f"报告已保存：{save_path}"
    except Exception as e:
        return f"错误：报告保存失败 {str(e)}"