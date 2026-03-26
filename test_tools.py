from tools.load_data import load_data
from tools.correlation import correlation_analysis
from tools.regression import regression_analysis
from tools.search_literature import search_literature
from tools.generate_report import generate_report
from agent.parser import parse_output

# 测试load_data
result = load_data("data/raw/merged_grid.csv")
print(result)
print("---")

# 测试correlation
result = correlation_analysis("data/raw/merged_grid.csv", x="svf_mean", y="PET")
result = regression_analysis("data/raw/merged_grid.csv", x="svf_mean", y="PET")
print(result)

#
result = search_literature("SVF thermal comfort")
print(result)
print("---")

result = generate_report(
    findings="SVF与PET强正相关（r=0.796），TVF与mean_temp显著负相关",
    save_path="data/output/report.txt"
)
print(result)

# 测试tool call解析
text1 = '<tool>correlation_analysis</tool><args>{"filepath": "data/raw/merged_grid.csv", "x": "svf_mean", "y": "PET"}</args>'
print(parse_output(text1))

# 测试final answer解析
text2 = "<answer>SVF与PET强正相关，是影响热舒适的主要因子</answer>"
print(parse_output(text2))