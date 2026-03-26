import pandas as pd

micro = pd.read_csv("data/raw/microclimate_data.csv")
morph = pd.read_csv("data/raw/morphology_data.csv")

# 按grid_id合并
merged = micro.merge(morph, on="grid_id")

# 保存
merged.to_csv("data/raw/merged_grid.csv", index=False)

print(f"合并成功，共 {len(merged)} 行")
print(f"列名：{list(merged.columns)}")