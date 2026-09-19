# 设备维修知识库智能问答系统（RAG）

把设备维修手册变成可对话的知识库 —— 答案来自原文，并标注具体出处。

## 🔗 在线 Demo

https://equipment-rag.streamlit.app/

## 核心功能

- **文档问答**：基于 60 页设备手册（离心泵、管道泵、电气设备维修规范）
- **语义检索**：BAAI/bge-m3 向量化，289 个知识片段
- **引用溯源**：每个答案标注来源文件与页码，可展开查看原文
- **反幻觉拒答**：检索距离超阈值时主动回答"资料中没有相关内容"

## 技术架构

用户提问
  → bge-m3 向量化
  → ChromaDB 向量检索（余弦距离）
  → 距离阈值判断（拒答 / 继续）
  → 拼接带编号的上下文
  → DeepSeek 生成
  → 答案 + 引用出处

## 技术栈

| 组件 | 选型 |
|---|---|
| 界面 | Streamlit |
| 向量库 | ChromaDB |
| Embedding | BAAI/bge-m3（硅基流动 API） |
| 生成模型 | DeepSeek-chat |

## 本地运行

```bash
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=sk-xxx" > .env
echo "SILICONFLOW_API_KEY=sk-yyy" >> .env
streamlit run app.py