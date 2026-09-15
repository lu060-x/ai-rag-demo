"""读取 data 文件夹里的所有 PDF，提取文字，保存成 data/pages.json
用法：在项目根目录执行   python step2_read_pdf.py
"""
import os
import json
from pypdf import PdfReader


def read_one_pdf(path):
    """读一份 PDF，返回 [{page: 页码, text: 文字}, ...]"""
    reader = PdfReader(path)

    # 有些 PDF 带加密：打开不需要密码，但内容是 AES 加密的，必须先解密
    if reader.is_encrypted:
        ok = reader.decrypt("")          # 先试空密码
        if not ok:
            print("   [!] 这个 PDF 需要密码，跳过")
            return []

    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception as e:
            print(f"   第 {i + 1} 页提取失败：{type(e).__name__}")
            continue
        if text and text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


def main():
    # 防呆：确认自己在项目根目录运行
    if not os.path.isdir("data"):
        print("[X] 找不到 data 文件夹。请确认终端在 ai-rag-demo 目录下运行")
        return

    all_pages = []

    for filename in os.listdir("data"):
        if not filename.lower().endswith(".pdf"):
            continue

        path = os.path.join("data", filename)
        pages = read_one_pdf(path)

        for p in pages:
            p["source"] = filename
            all_pages.append(p)

        # 体检：平均每页多少字，判断是不是扫描件
        if pages:
            total = sum(len(p["text"]) for p in pages)
            avg = total / len(pages)
            flag = "OK 可用" if avg > 100 else "!! 疑似扫描件，建议换文件"
            print(f"{filename}：提取 {len(pages)} 页，平均每页 {avg:.0f} 字   [{flag}]")
        else:
            print(f"{filename}：一页都没提取到 [!!]")

    print("-" * 45)
    print(f"合计：{len(all_pages)} 页")

    if not all_pages:
        print("没有任何内容，检查 PDF 是不是扫描件")
        return

    with open("data/pages.json", "w", encoding="utf-8") as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=2)

    print("已保存到 data/pages.json [OK]")


if __name__ == "__main__":
    main()
