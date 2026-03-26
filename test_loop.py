from agent.loop import run_episode, SYSTEM_PROMPT

# mock模型：直接返回预设的tool call序列
class MockModel:
    def __init__(self):
        self.step = 0
        self.responses = [
            '<tool>load_data</tool><args>{"filepath": "data/raw/merged_grid.csv"}</args>',
            '<tool>correlation_analysis</tool><args>{"filepath": "data/raw/merged_grid.csv", "x": "svf_mean", "y": "PET"}</args>',
            '<tool>regression_analysis</tool><args>{"filepath": "data/raw/merged_grid.csv", "x": "svf_mean", "y": "PET"}</args>',
            '<answer>SVF与PET强正相关（r=0.796，R²=0.634），是影响热舒适的主要形态因子</answer>',
        ]

    def generate(self, **kwargs):
        pass

class MockTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return ""
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[0]])}
    eos_token_id = 0

# 替换generate函数做mock测试
import agent.loop as loop_module
import torch

mock_model_instance = MockModel()

def mock_generate(model, tokenizer, messages):
    response = mock_model_instance.responses[mock_model_instance.step]
    mock_model_instance.step = min(
        mock_model_instance.step + 1,
        len(mock_model_instance.responses) - 1
    )
    return response

loop_module.generate = mock_generate

task = {
    "question": "分析形态指标与热舒适参数的关系，找出显著相关的变量对",
    "data_path": "data/raw/merged_grid.csv",
}

messages = run_episode(None, None, task)

print(f"\n共 {len(messages)} 条消息，推理轮数：{(len(messages)-2)//2}")
print("\n--- 完整对话 ---")
for m in messages:
    role = m["role"].upper()
    content = m["content"][:150]
    print(f"[{role}]: {content}")
    print()

from reward.efficiency import compute_efficiency

score = compute_efficiency(messages)
print(f"效率分数：{score}，工具调用次数：{sum(1 for m in messages if m['role']=='assistant' and '<tool>' in m['content'])}")

from reward.outcome import compute_f1

task = {
    "question": "分析形态指标与热舒适参数的关系",
    "data_path": "data/raw/merged_grid.csv",
    "ground_truth": {
        "significant_pairs": [["svf_mean", "PET"], ["tvf_mean", "mean_temp"]]
    }
}

score = compute_f1(messages, task)
print(f"F1分数：{score}")

from reward.conclusion_reward import conclusion_reward

conclusion = """
SVF与PET强正相关（r=0.796），是影响热舒适的主要形态因子。
TVF对mean_temp有显著降温效果。
建议按街道朝向分组后分别验证，并引入多元回归控制其他变量。
"""

ground_truth = {
    "significant_pairs": [["svf_mean", "PET"], ["tvf_mean", "mean_temp"]],
    "stats": {
        "svf_mean_PET":       {"r": 0.796, "p": 0.00, "R2": 0.634},
        "tvf_mean_mean_temp": {"r": -0.58, "p": 0.01, "R2": 0.34}
    }
}

domain_context = {
    "season": "summer",
    "single_city": True
}

score = conclusion_reward(conclusion, ground_truth, domain_context)
print(f"结论层分数：{score}")

from reward.reward_fn import reward_fn

score = reward_fn(messages, task)
print(f"最终reward：{score}")