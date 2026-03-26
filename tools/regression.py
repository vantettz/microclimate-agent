import pandas as pd
from scipy.stats import linregress

def regression_analysis(filepath: str, x: str, y: str) -> str:
    """
    对指定变量对做线性回归
    
    Args:
        filepath: CSV文件路径
        x: 自变量列名
        y: 因变量列名
    Returns:
        字符串，包含slope、intercept、R²
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return f"错误：读取文件失败 {str(e)}"

    if x not in df.columns:
        return f"错误：变量 {x} 不在数据中，可用变量：{', '.join(df.columns)}"
    if y not in df.columns:
        return f"错误：变量 {y} 不在数据中，可用变量：{', '.join(df.columns)}"

    valid = df[[x, y]].dropna()
    if len(valid) < 10:
        return f"错误：有效样本量不足（{len(valid)}条），无法做回归分析"

    slope, intercept, r, p, stderr = linregress(valid[x], valid[y])
    r2 = r ** 2

    # R²解读
    if r2 > 0.6:
        r2_interpretation = "解释力强"
    elif r2 > 0.4:
        r2_interpretation = "解释力中等"
    else:
        r2_interpretation = "解释力有限，可能存在其他未控制变量"

    result = f"""
回归分析结果：{y} = f({x})
有效样本量：{len(valid)}
slope = {slope:.4f}
intercept = {intercept:.4f}
R² = {r2:.3f}（{r2_interpretation}）
标准误 = {stderr:.4f}
回归方程：{y} = {slope:.4f} × {x} + {intercept:.4f}
"""
    return result.strip()