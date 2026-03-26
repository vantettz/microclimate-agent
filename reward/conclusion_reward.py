import re

def check_stats_interpretation(conclusion: str, ground_truth: dict) -> float:
    """统计数字解读是否正确"""
    score = 0.0
    stats = ground_truth.get("stats", {})

    for pair_key, values in stats.items():
        p  = values.get("p", 1.0)
        r2 = values.get("R2", 0.0)
        r  = values.get("r", 0.0)

        # 显著性判断
        if p < 0.05:
            if any(k in conclusion for k in ["显著", "significant"]):
                score += 0.2
        else:
            if any(k in conclusion for k in ["不显著", "not significant"]):
                score += 0.2

        # R²偏低时是否识别局限性
        if r2 < 0.4:
            if any(k in conclusion for k in
                   ["解释力有限", "其他变量", "未控制", "R²偏低", "不足"]):
                score += 0.2

        # 相关方向是否正确
        if r > 0:
            if any(k in conclusion for k in ["正相关", "增加", "升高"]):
                score += 0.1
        elif r < 0:
            if any(k in conclusion for k in ["负相关", "降低", "减少", "降温"]):
                score += 0.1

    return min(round(score, 3), 1.0)


def check_consistency(conclusion: str, ground_truth: dict) -> float:
    """结论方向是否和已知物理规律一致"""
    score = 0.0
    stats = ground_truth.get("stats", {})

    known_directions = {
        "svf_mean": {"PET": "positive", "MRT": "positive"},
        "tvf_mean": {"mean_temp": "negative", "PET": "negative"},
        "H_W_ratio": {"wind_speed": "negative"},
    }

    positive_kw = ["正相关", "增加", "升高", "越大越高"]
    negative_kw = ["负相关", "降低", "减少", "越大越低", "降温"]

    for x, ys in known_directions.items():
        for y, direction in ys.items():
            if x in conclusion and y in conclusion:
                if direction == "positive":
                    if any(k in conclusion for k in positive_kw):
                        score += 0.2
                else:
                    if any(k in conclusion for k in negative_kw):
                        score += 0.2

    r2_values = [v.get("R2", 1.0) for v in stats.values()]

    # 情况一：R²偏低，是否提到遗漏变量
    if any(r2 < 0.4 for r2 in r2_values):
        if any(k in conclusion for k in ["控制变量", "遗漏", "调节", "分组"]):
            score += 0.2

    # 情况二：分组分析，是否识别朝向调节效应
    if "EW" in conclusion and "NS" in conclusion:
        if any(k in conclusion for k in ["调节", "差异", "朝向影响"]):
            score += 0.2

    # 情况三：物理规律矛盾，是否提到需要验证
    if any(k in conclusion for k in ["矛盾", "异常", "与预期不符", "需要验证"]):
        score += 0.1

    # 情况四：R²强，是否做了衍生分析
    if any(r2 > 0.5 for r2 in r2_values):
        if any(k in conclusion for k in ["进一步", "延伸", "基于此", "文献表明"]):
            score += 0.15

    return min(round(score, 3), 1.0)


def check_followup(conclusion: str, domain_context: dict) -> float:
    """后续建议是否针对具体发现而不是通用废话"""
    score = 0.0

    # 通用废话扣分
    generic = ["扩大样本", "继续研究", "未来工作"]
    if any(k in conclusion for k in generic):
        score -= 0.1

    # 针对R²偏低的具体建议
    r2_followup = ["分组验证", "引入控制变量", "多元回归", "交互效应"]
    hits_r2 = sum(1 for k in r2_followup if k in conclusion)
    score += min(hits_r2 * 0.15, 0.3)

    # 针对朝向差异的具体建议
    orient_followup = ["东西向", "南北向", "朝向", "分方向"]
    hits_orient = sum(1 for k in orient_followup if k in conclusion)
    score += min(hits_orient * 0.15, 0.3)

    # 针对R²强的衍生分析建议（必须具体）
    specific_extension = ["MRT", "长波辐射", "street_type", "密度", "风速"]
    if any(k in conclusion for k in specific_extension):
        score += 0.15
    elif any(k in conclusion for k in ["进一步", "延伸"]):
        score -= 0.05  # 写了延伸但不具体，小扣分

    # 针对季节局限性的建议
    if domain_context.get("season") == "summer":
        season_kw = ["季节", "冬季", "全年", "其他季节"]
        hits_season = sum(1 for k in season_kw if k in conclusion)
        score += min(hits_season * 0.1, 0.2)

    return min(round(score, 3), 1.0)


def conclusion_reward(conclusion: str,
                      ground_truth: dict,
                      domain_context: dict) -> float:
    """结论生成层总分"""
    stats    = check_stats_interpretation(conclusion, ground_truth)
    consist  = check_consistency(conclusion, ground_truth)
    followup = check_followup(conclusion, domain_context)

    total = 0.5 * consist + 0.3 * stats + 0.2 * followup
    return round(total, 3)