"""D5 对照实验：直接问大模型 vs 用 RAG 问"""
import os
from dotenv import load_dotenv
from openai import OpenAI

from step5_rag import ask as rag_ask

load_dotenv()

llm = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

QUESTIONS = [
    "IS50-32-125 型号水泵的流量和扬程是多少？",
    "CISG 管道泵不出水有哪些原因？",
    "电气设备的预防性维修周期怎么定？",
]

for q in QUESTIONS:
    print("=" * 62)
    print("问题：", q)

    print("\n【A】直接问大模型（它没见过你的手册）")
    resp = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": q}],
        temperature=0.2,
        max_tokens=300,
    )
    print(resp.choices[0].message.content)

    print("\n【B】用 RAG 问（先检索你的手册）")
    answer, hits, _ = rag_ask(q)
    print(answer)
    print("\n命中来源：", [(m["source"], m["page"], round(d, 3)) for _, m, d in hits])
    print()