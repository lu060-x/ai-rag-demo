"""D6：把 RAG 问答做成网页界面
运行：streamlit run app.py
"""
import os
import streamlit as st

# ---------- 关键：先把密钥写进环境变量，再导入会读环境变量的模块 ----------
try:
    for key in ("DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY"):
        if key in st.secrets and key not in os.environ:
            os.environ[key] = st.secrets[key]
except Exception:
    pass          # 本地没有 secrets.toml 时会抛异常，忽略即可

from step5_rag import (            # noqa: E402
    llm, col, retrieve, build_context,
    SYSTEM_PROMPT, USER_TEMPLATE, DIST_THRESHOLD,
)

# ---------------- 页面设置 ----------------
st.set_page_config(page_title="设备维修知识助手", page_icon="🔧", layout="wide")
st.title("🔧 设备维修知识助手")
st.caption("基于 RAG 的文档问答：答案来自你的维修手册，并标注原文出处")

# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.header("设置")
    top_k = st.slider("检索片段数 TOP_K", 1, 10, 3)
    threshold = st.slider("拒答阈值", 0.5, 1.5, float(DIST_THRESHOLD), 0.05)
    use_stream = st.toggle(
        "流式输出（打字机效果）",
        value=False,
        help="如果浏览器出现 removeChild 报错，保持关闭即可",
    )
    st.divider()
    st.metric("知识库片段数", col.count())
    st.caption("距离越小越相关；超过阈值判定为「资料中没有」。")
    st.divider()
    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------- 对话历史 ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_message(msg):
    st.markdown(msg["content"])
    if msg.get("hits"):
        with st.expander("📎 查看引用来源"):
            for i, (doc, meta, dist) in enumerate(msg["hits"], 1):
                st.markdown(f"**[{i}] {meta['source']} 第 {meta['page']} 页** · "
                            f"距离 {dist:.4f}")
                st.text(doc[:300])


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_message(msg)

# ---------------- 输入与回答 ----------------
question = st.chat_input("请输入你的问题（例如：泵启动后不出水是什么原因）")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        hits = retrieve(question, k=top_k)
        best = hits[0][2]

        if best > threshold:
            answer = "资料中没有相关内容。"
            st.markdown(answer)
            st.caption(f"（最佳命中距离 {best:.4f} > 阈值 {threshold:.2f}，已拒答）")
        else:
            chat_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(
                    context=build_context(hits), question=question)},
            ]

            if use_stream:
                # 流式：手动占位符，不用 st.write_stream
                stream = llm.chat.completions.create(
                    model="deepseek-chat", messages=chat_messages,
                    temperature=0.2, max_tokens=600, stream=True,
                )
                placeholder = st.empty()
                answer = ""
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        answer += delta
                        placeholder.markdown(answer + "▌")
                placeholder.markdown(answer)
            else:
                # 非流式：和"拒答"分支走完全同一条渲染路径，最稳
                resp = llm.chat.completions.create(
                    model="deepseek-chat", messages=chat_messages,
                    temperature=0.2, max_tokens=600,
                )
                answer = resp.choices[0].message.content
                st.markdown(answer)

            with st.expander("📎 查看引用来源"):
                for i, (doc, meta, dist) in enumerate(hits, 1):
                    st.markdown(f"**[{i}] {meta['source']} 第 {meta['page']} 页** · "
                                f"距离 {dist:.4f}")
                    st.text(doc[:300])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "hits": hits}
    )