import pandas as pd

def load_data(filepath: str) -> str:
    """
    读取CSV文件，返回变量列表和基本统计摘要
    
    Args:
        filepath: CSV文件路径
    Returns:
        字符串，包含变量列表和基本统计
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return f"错误：找不到文件 {filepath}"
    except Exception as e:
        return f"错误：读取文件失败 {str(e)}"
    
    # 变量列表
    columns = [c for c in df.columns if c != "grid_id"]
    
    # 基本统计
    stats = df[columns].describe().round(3)
    
    result = f"""
数据加载成功，共 {len(df)} 个格网。

可用变量：{', '.join(columns)}

基本统计：
{stats.to_string()}
"""
    return result.strip()