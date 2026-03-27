import os
import numpy as np
import librosa
from dashscope import Generation

def _get_noise_features(audio_path: str) -> dict:
    """
    用 librosa 提取音频的噪声特征
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    # 转换为分贝
    rms = librosa.feature.rms(y=y)[0]
    db_values = librosa.amplitude_to_db(rms)
    
    db_mean  = float(np.mean(db_values))
    db_peak  = float(np.max(db_values))
    db_min   = float(np.min(db_values))
    
    # 频谱重心（判断噪声类型）
    spectral_centroid = float(np.mean(
        librosa.feature.spectral_centroid(y=y, sr=sr)
    ))
    
    # 判断噪声类型
    if spectral_centroid < 500:
        noise_type = "低频噪声（交通为主）"
    elif spectral_centroid < 2000:
        noise_type = "中频噪声（人群/混合）"
    else:
        noise_type = "高频噪声（机械/施工为主）"
    
    # 判断噪声是否持续（方差小=持续，方差大=间歇）
    pattern = "持续性噪声" if np.std(db_values) < 5 else "间歇性噪声"
    
    # 超标判断（城市住宅区白天标准55dB，商业区60dB）
    if db_mean > 70:
        level = "严重超标"
    elif db_mean > 60:
        level = "轻度超标"
    elif db_mean > 55:
        level = "接近标准上限"
    else:
        level = "达标"
    
    return {
        "db_mean":    round(db_mean, 1),
        "db_peak":    round(db_peak, 1),
        "db_min":     round(db_min, 1),
        "noise_type": noise_type,
        "pattern":    pattern,
        "level":      level,
    }


def _generate_assessment(features: dict) -> str:
    """
    把噪声特征发给 Qwen 生成自然语言评估
    """
    prompt = f"""
以下是一段城市街道录音的噪声分析结果：
- 噪声均值：{features['db_mean']}dB
- 噪声峰值：{features['db_peak']}dB  
- 噪声类型：{features['noise_type']}
- 噪声模式：{features['pattern']}
- 超标情况：{features['level']}

请用3句话评估该街道的声环境质量，包括：
1. 声环境质量等级和主要噪声来源
2. 对行人舒适度的影响
3. 一条具体的改善建议
用学术风格，可直接引用。
"""
    response = Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    return response.output.text


def noise_analysis(audio_path: str) -> str:
    """
    分析音频文件的噪声特征，返回声环境质量评估

    Args:
        audio_path: 音频文件路径（.wav 或 .mp3）
    Returns:
        字符串，包含噪声特征和质量评估
    """
    if not os.path.exists(audio_path):
        return f"错误：找不到音频文件 {audio_path}"

    try:
        features = _get_noise_features(audio_path)
    except Exception as e:
        return f"错误：音频分析失败 {str(e)}"

    try:
        assessment = _generate_assessment(features)
    except Exception as e:
        assessment = "（语言模型评估暂不可用）"

    result = f"""声环境分析结果：
噪声均值：{features['db_mean']}dB（{features['level']}）
噪声峰值：{features['db_peak']}dB
噪声类型：{features['noise_type']}
噪声模式：{features['pattern']}

质量评估：
{assessment}"""

    return result