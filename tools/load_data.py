import pandas as pd

def load_data(filepath: str) -> str:
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return f"错误：找不到文件 {filepath}"
    except Exception as e:
        return f"错误：读取文件失败 {str(e)}"
    
    # 排除非数值列
    exclude = ["grid_id", "orientation", "street_type"]
    numeric_cols = [c for c in df.columns if c not in exclude]
    category_cols = [c for c in df.columns 
                     if c in ["orientation", "street_type"] 
                     and c in df.columns]
    
    stats = df[numeric_cols].describe().round(3)
    
    result = f"""数据加载成功，共 {len(df)} 个格网。

数值变量：{', '.join(numeric_cols)}
类别变量：{', '.join(category_cols) if category_cols else '无'}

基本统计：
{stats.to_string()}"""

    if category_cols:
        for col in category_cols:
            result += f"\n\n{col} 分布：\n{df[col].value_counts().to_string()}"

    return result