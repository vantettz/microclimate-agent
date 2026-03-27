import os
import base64
from PIL import Image
import io
from dashscope import MultiModalConversation

def _encode_image(image_path: str, max_size: int = 1024) -> str:
    """把图片压缩后转成 base64 字符串"""
    img = Image.open(image_path)
    
    # 压缩尺寸
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    
    # 转成 JPEG 并压缩质量
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=75)
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode("utf-8")

def vision_analysis(image_path: str) -> str:
    if not os.path.exists(image_path):
        return f"错误：找不到图片文件 {image_path}"

    try:
        image_data = _encode_image(image_path)
    except Exception as e:
        return f"错误：图片读取失败 {str(e)}"

    prompt = """这是一张城市街道的街景图片。
请从微气候和环境质量角度分析，包括以下几点：

1. 空间形态特征：建筑高度、街道宽度、天空可视程度（高/中/低）
2. 植被状况：行道树覆盖密度、遮蔽效果（好/中/差）
3. 热舒适潜力：基于天空开阔度和植被遮蔽，判断夏季午后热舒适条件（好/中/差）
4. 视觉质量评价：空间围合感、绿化水平、整体环境品质
5. 主要问题：指出影响热舒适的最关键因素
6. 改善建议：一条最优先的优化措施

用3-5句话，学术风格，可直接引用。"""

    try:
        response = MultiModalConversation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen-vl-plus",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "image": f"data:image/jpeg;base64,{image_data}"
                        },
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        if response.status_code != 200:
            return f"错误：API返回异常，状态码 {response.status_code}，信息：{response.message}"

        if not response.output or not response.output.choices:
            return f"错误：API返回内容为空"

        assessment = response.output.choices[0].message.content[0]["text"]

    except Exception as e:
        return f"错误：图片分析失败 {str(e)}"

    return f"视觉质量分析结果：\n\n{assessment}"