"""D5：最小可用的 RAG 问答（检索 + 拼 prompt + 生成）
用法：在项目根目录执行   python step5_rag.py
"""
import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from step4_embed import embed_one

load_dotenv()

# ---- 1. 聊天模型（DeepSeek，负责"生成"）----
llm = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

# ---- 2. 向量库（bge-m3 负责"检索"）----
col = chromadb.PersistentClient(path="./db").get_collection("manual_kb")

TOP_K = 3
DIST_THRESHOLD = 0.9      # D4 实测：文档内 0.57~0.72，文档外 1.18~1.21

# ---- 3. 规则放 system：内容固定不变，能命中 prompt 缓存，省钱 ----
SYSTEM_PROMPT = """你是设备维修技术助手。请严格依据用户提供的【资料】回答问题。

规则：
1. 只使用【资料】中的信息，禁止编造，禁止使用你自己的知识
2. 如果【资料】中没有答案，只回答：资料中没有相关内容
3. 回答末尾用 [编号] 标注你参考了哪几段资料
4. 用中文回答，条理清晰
"""

USER_TEMPLATE = """【资料】
{context}

【问题】
{question}"""


def retrieve(question, k=TOP_K):
    """检索：返回 [(片段, 元数据, 距离), ...]"""
    qv = embed_one(question)
    res = col.query(query_embeddings=[qv], n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


def build_context(hits):
    """把检索结果拼成带编号的文本"""
    lines = []
    for i, (doc, meta, dist) in enumerate(hits, 1):
        lines.append(f"[{i}] 来源：{meta['source']} 第 {meta['page']} 页")
        lines.append(doc.strip())
        lines.append("")
    return "\n".join(lines)


def ask(question):
    """完整 RAG 流程，返回 (回答, 命中片段, 原始响应)"""
    hits = retrieve(question)
    best_distance = hits[0][2]

    # 反幻觉第一道防线：最好的命中都太远，说明库里根本没有
    if best_distance > DIST_THRESHOLD:
        return "资料中没有相关内容。", hits, None

    resp = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                context=build_context(hits), question=question)},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return resp.choices[0].message.content, hits, resp


if __name__ == "__main__":
    while True:
        q = input("\n你：")
        if q.strip().lower() in ("exit", "quit"):
            break
        if not q.strip():
            continue

        answer, hits, resp = ask(q)
        print("\nAI：" + answer)

        print("\n--- 本次检索到的片段 ---")
        for i, (doc, meta, dist) in enumerate(hits, 1):
            print(f"  [{i}] {meta['source']} 第{meta['page']}页 距离={dist:.4f}")

        if resp:
            print(f"\n[输入 {resp.usage.prompt_tokens} / 输出 {resp.usage.completion_tokens} token]")