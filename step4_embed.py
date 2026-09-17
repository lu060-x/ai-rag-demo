"""把文本转成向量的工具：调用云端 embedding API（BAAI/bge-m3）"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["SILICONFLOW_API_KEY"],
    base_url="https://api.siliconflow.cn/v1",
)

EMBED_MODEL = "BAAI/bge-m3"


def embed_texts(texts, batch_size=16):
    """把一批文本转成向量，返回 list[list[float]]"""
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([d.embedding for d in resp.data])
        print(f"  已完成 {min(i + batch_size, len(texts))}/{len(texts)}")
    return vectors


def embed_one(text):
    """单条文本转向量"""
    return embed_texts([text])[0]