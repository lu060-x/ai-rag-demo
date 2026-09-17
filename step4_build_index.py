"""把 data/pages.json 切片、算向量、存进 Chroma 向量数据库"""
import os
import json
import shutil
import chromadb

from step4_embed import embed_texts

CHUNK_SIZE = 400
OVERLAP = 80
COLLECTION = "manual_kb"


def split_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def main():
    data = json.load(open("data/pages.json", encoding="utf-8"))
    print(f"读入 {len(data)} 页")

    docs, metas, ids = [], [], []
    for r in data:
        for j, chunk in enumerate(split_text(r["text"])):
            docs.append(chunk)
            metas.append({"source": r["source"], "page": r["page"]})
            ids.append(f"{r['source']}::p{r['page']}::c{j}")

    print(f"切成 {len(docs)} 个片段")

    print("正在计算向量（第一次会比较慢）...")
    vectors = embed_texts(docs)
    print(f"向量维度：{len(vectors[0])}")

    if os.path.isdir("db"):
        shutil.rmtree("db")

    client = chromadb.PersistentClient(path="./db")
    col = client.get_or_create_collection(COLLECTION)
    col.add(documents=docs, embeddings=vectors, ids=ids, metadatas=metas)

    print(f"已写入向量库，共 {col.count()} 条")


if __name__ == "__main__":
    main()