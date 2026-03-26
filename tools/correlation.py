import pandas as pd
from scipy.stats import pearsonr

def correlation_analysis(filepath: str, x: str, y: str) -> str:
    """
    对指定两个变量做Pearson相关性分析
    
    Args:
        filepath: CSV文件路径
        x: 自变量列名
        y: 因变量列名
    Returns:
        字符串，包含r、p值和显著性判断
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return f"错误：读取文件失败 {str(e)}"
    
    if x not in df.columns:
        return f"错误：变量 {x} 不在数据中，可用变量：{', '.join(df.columns)}"
    if y not in df.columns:
        return f"错误：变量 {y} 不在数据中，可用变量：{', '.join(df.columns)}"
    
    # 去掉缺失值
    valid = df[[x, y]].dropna()
    if len(valid) < 10:
        return f"错误：有效样本量不足（{len(valid)}条），无法做相关性分析"
    
    r, p = pearsonr(valid[x], valid[y])
    
    # 显著性判断
    if p < 0.01:
        sig = "极显著（p<0.01）"
    elif p < 0.05:
        sig = "显著（p<0.05）"
    else:
        sig = "不显著（p≥0.05）"
    
    # 相关强度判断
    abs_r = abs(r)
    if abs_r > 0.6:
        strength = "强相关"
    elif abs_r > 0.4:
        strength = "中等相关"
    else:
        strength = "弱相关"
    
    direction = "负相关" if r < 0 else "正相关"
    
    result = f"""
相关性分析结果：{x} vs {y}
有效样本量：{len(valid)}
Pearson r = {r:.3f}（{strength}，{direction}）
p值 = {p:.4f}（{sig}）
"""
    return result.strip()