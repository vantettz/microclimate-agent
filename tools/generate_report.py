import os
from datetime import datetime

def generate_report(findings: str, save_path: str,
                    has_csv: bool = False,
                    has_image: bool = False,
                    has_audio: bool = False) -> str:
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    except Exception:
        pass

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 根据实际输入标注分析维度
    dimensions = []
    if has_csv:
        dimensions.append("热舒适统计分析")
    if has_image:
        dimensions.append("视觉质量分析")
    if has_audio:
        dimensions.append("声环境分析")
    dim_str = " + ".join(dimensions) if dimensions else "综合分析"

    report = f"""城市街道环境质量分析报告
生成时间：{timestamp}
分析维度：{dim_str}
{"="*50}

{findings}

{"="*50}
报告结束""".strip()

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        return f"报告已保存：{save_path}"
    except Exception as e:
        return f"错误：报告保存失败 {str(e)}"