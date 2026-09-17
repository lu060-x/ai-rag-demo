"""在向量库里做语义检索"""
import chromadb
from step4_embed import embed_one

COLLECTION = "manual_kb"

client = chromadb.PersistentClient(path="./db")
col = client.get_collection(COLLECTION)


def search(question, k=3):
    qv = embed_one(question)
    res = col.query(query_embeddings=[qv], n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


if __name__ == "__main__":
    print(f"向量库共 {col.count()} 条\n")
    while True:
        q = input("问题（exit 退出）：")
        if q.strip().lower() in ("exit", "quit"):
            break
        if not q.strip():
            continue

        for i, (doc, meta, dist) in enumerate(search(q), 1):
            print(f"\n--- 命中 {i}  距离={dist:.4f}  "
                  f"来源={meta['source']} 第{meta['page']}页 ---")
            print(doc[:180].replace("\n", " "))
        print()