import os
import streamlit as st
import tempfile
from dotenv import load_dotenv

from agent.loop import run_analysis
from agent.parser import parse_output
from tools.generate_optimization import generate_optimization

load_dotenv()

# ─── 页面配置 ───────────────────────────────────────────
st.set_page_config(
    page_title="城市街景多模态视听微气候变量综合分析",
    page_icon="🏙️",
    layout="wide"
)

st.title("🏙️ 城市街景多模态视听微气候变量综合分析")
st.caption("支持热舒适、视觉质量、声环境三维度综合分析")

# ─── session_state 初始化 ────────────────────────────────
if "messages"      not in st.session_state:
    st.session_state.messages      = []
if "csv_path"      not in st.session_state:
    st.session_state.csv_path      = None
if "image_path"    not in st.session_state:
    st.session_state.image_path    = None
if "audio_path"    not in st.session_state:
    st.session_state.audio_path    = None
if "conclusion"    not in st.session_state:
    st.session_state.conclusion    = None
if "optimization"  not in st.session_state:
    st.session_state.optimization  = None

# ─── 侧边栏：文件上传 ────────────────────────────────────
with st.sidebar:
    st.header("上传数据")
    st.caption("三路输入各自独立，可单独上传")

    # CSV 上传
    csv_file = st.file_uploader(
        "📊 气候数据（CSV）",
        type=["csv"],
        help="包含SVF、TVF、PET、MRT等指标的格网数据"
    )
    if csv_file:
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".csv"
        )
        tmp.write(csv_file.read())
        tmp.flush()
        st.session_state.csv_path = tmp.name
        st.success("✅ CSV 已上传")

    st.divider()

    # 图片上传
    image_file = st.file_uploader(
        "🖼️ 街景图片（JPG/PNG）",
        type=["jpg", "jpeg", "png"],
        help="街道街景照片，用于视觉质量和微气候分析"
    )
    if image_file:
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".jpg"
        )
        tmp.write(image_file.read())
        tmp.flush()
        st.session_state.image_path = tmp.name
        st.image(image_file, caption="已上传街景图片", use_column_width=True)
        st.success("✅ 图片已上传")

    st.divider()

    # 音频上传
    audio_file = st.file_uploader(
        "🔊 环境音频（WAV/MP3）",
        type=["wav", "mp3"],
        help="街道环境录音，用于声环境分析"
    )
    if audio_file:
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav"
        )
        tmp.write(audio_file.read())
        tmp.flush()
        st.session_state.audio_path = tmp.name
        st.audio(audio_file)
        st.success("✅ 音频已上传")

    st.divider()

    # 已上传状态总览
    st.subheader("当前输入状态")
    st.write("📊 CSV：" + ("✅" if st.session_state.csv_path   else "未上传"))
    st.write("🖼️ 图片：" + ("✅" if st.session_state.image_path else "未上传"))
    st.write("🔊 音频：" + ("✅" if st.session_state.audio_path else "未上传"))

    # 清空按钮
    if st.button("🗑️ 清空所有输入"):
        st.session_state.csv_path     = None
        st.session_state.image_path   = None
        st.session_state.audio_path   = None
        st.session_state.conclusion   = None
        st.session_state.optimization = None
        st.session_state.messages     = []
        st.rerun()

# ─── 主区域 ─────────────────────────────────────────────
# 检查是否有任何输入
has_any_input = any([
    st.session_state.csv_path,
    st.session_state.image_path,
    st.session_state.audio_path,
])

if not has_any_input:
    st.info("👈 请在左侧上传至少一种数据文件开始分析")

else:
    # 问题输入框
    question = st.text_input(
        "输入你的分析需求",
        placeholder="例如：分析这条街道的热舒适条件，找出主要影响因素并给出改善建议",
        key="question_input"
    )

    # 分析按钮
    if st.button("🔍 开始分析", type="primary", disabled=not question):
        with st.spinner("分析中，请稍候..."):
            messages = run_analysis(
                question=question,
                csv_path=st.session_state.csv_path,
                image_path=st.session_state.image_path,
                audio_path=st.session_state.audio_path,
            )

            # 提取最终结论
            conclusion = ""
            for m in reversed(messages):
                if m["role"] == "assistant":
                    parsed = parse_output(m["content"])
                    if parsed["type"] == "final_answer":
                        conclusion = parsed["content"]
                        break
                    conclusion = m["content"]
                    break

            st.session_state.conclusion = conclusion
            st.session_state.messages   = messages
            st.session_state.optimization = None  # 重置优化方案

    # ── 展示分析结论 ──────────────────────────────────────
    if st.session_state.conclusion:
        st.subheader("📋 分析结论")
        st.markdown(st.session_state.conclusion)

        # 展示中间推理过程（可折叠）
        with st.expander("查看完整推理过程"):
            for m in st.session_state.messages:
                if m["role"] == "assistant":
                    st.markdown(f"**Agent：** {m['content']}")
                elif m["role"] == "user" and "Observation:" in m["content"]:
                    st.markdown(f"**工具返回：** {m['content']}")

        st.divider()

        # ── 生成优化方案 ──────────────────────────────────
        st.subheader("🛠️ 街道优化方案")

        if st.button("✨ 生成优化策略与效果图"):
            with st.spinner("正在生成优化方案和效果图，约需30秒..."):
                result = generate_optimization(
                    st.session_state.conclusion
                )
                st.session_state.optimization = result

        if st.session_state.optimization:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**优化策略**")
                st.markdown(
                    st.session_state.optimization["text"]
                )

            with col2:
                st.markdown("**优化效果图**")
                img_path = st.session_state.optimization["image_path"]
                if os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)
                else:
                    st.warning(img_path)  # 显示错误信息

        st.divider()

        # ── 追问功能 ──────────────────────────────────────
        st.subheader("💬 继续追问")
        followup = st.text_input(
            "对分析结果有疑问？继续提问",
            placeholder="例如：为什么SVF对PET的影响这么大？",
            key="followup_input"
        )

        if st.button("发送", key="followup_btn") and followup:
            with st.spinner("思考中..."):
                # 把追问加进对话历史继续推理
                followup_messages = st.session_state.messages.copy()
                followup_messages.append({
                    "role": "user",
                    "content": followup
                })

                from agent.loop import generate_api
                response = generate_api(followup_messages)

                st.session_state.messages.append({
                    "role": "user",
                    "content": followup
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

                st.markdown(f"**回答：** {response}")

