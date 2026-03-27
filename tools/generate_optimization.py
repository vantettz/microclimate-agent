import os
from dashscope import Generation, ImageSynthesis

def generate_optimization_text(analysis_conclusion: str) -> str:
    """
    把分析结论喂给 Qwen，生成优化策略文字描述

    Args:
        analysis_conclusion: 三路分析的综合结论
    Returns:
        优化策略文字描述
    """
    prompt = f"""
以下是一条城市街道的环境质量分析结论：

{analysis_conclusion}

请基于以上分析，生成一段街道优化策略描述，包括：
1. 最优先的优化措施（不超过3条）
2. 每条措施的预期改善效果
3. 优化后街道的整体环境质量预期

要求：
- 描述具体可操作，避免泛泛而谈
- 用城市规划专业语言
- 100字以内
"""
    response = Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    return response.output.text


def generate_optimization_image(optimization_text: str) -> str:
    """
    把优化策略文字转成图片生成 prompt，调用通义万象生成效果图

    Args:
        optimization_text: 优化策略文字描述
    Returns:
        生成图片的本地保存路径
    """
    # 把优化策略文字转成图片生成 prompt
    prompt_text = f"""
根据以下城市街道优化策略，生成一张对应的街道效果图：

{optimization_text}

图片要求：写实风格城市街道效果图，白天晴天，
行人视角，展示优化后的街道环境，
包含行道树、建筑立面、街道铺装，
光线充足但有树荫遮蔽，环境舒适宜人。
"""

    try:
        response = ImageSynthesis.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="wanx-v1",
            prompt=prompt_text,
            n=1,
            size="1024*1024",
        )

        image_url = response.output.results[0].url

        # 下载图片保存到本地
        import requests
        save_path = "data/output/optimization.png"
        os.makedirs("data/output", exist_ok=True)

        img_data = requests.get(image_url).content
        with open(save_path, "wb") as f:
            f.write(img_data)

        return save_path

    except Exception as e:
        return f"错误：图片生成失败 {str(e)}"


def generate_optimization(analysis_conclusion: str) -> dict:
    """
    完整优化方案生成：文字策略 + 效果图

    Args:
        analysis_conclusion: 综合分析结论
    Returns:
        dict，包含 text（优化策略文字）和 image_path（效果图路径）
    """
    # 第一步：生成优化策略文字
    optimization_text = generate_optimization_text(analysis_conclusion)

    # 第二步：生成优化效果图
    image_path = generate_optimization_image(optimization_text)

    return {
        "text":       optimization_text,
        "image_path": image_path,
    }