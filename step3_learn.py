"""D3 练习：调用大模型 API ｜ 多轮对话 + 流式输出（打字机效果）
用法：在项目根目录执行   python step3_learn.py
      输入 exit 退出
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

# 对话历史。模型自己没有记忆，靠每次把历史重新发过去实现"记忆"
history = [
    {"role": "system", "content": "你是一个设备维修助手，回答简洁，不超过 60 字。"},
]


def chat_stream(question):
    """流式提问：一边生成一边打印。返回 (回答, token用量)"""
    history.append({"role": "user", "content": question})

    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=history,
        temperature=0.3,
        max_tokens=500,
        stream=True,
    )

    print("AI：", end="")
    answer = ""
    usage = None

    for chunk in stream:
        # 注意：流式的最后一个 chunk 里 choices 是空列表，但带着 usage
        if chunk.usage:
            usage = chunk.usage

        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            answer += delta
            print(delta, end="", flush=True)

    print()

    # 关键：把回答也存进历史，否则下一轮就"失忆"
    history.append({"role": "assistant", "content": answer})
    return answer, usage


if __name__ == "__main__":
    total_in = 0
    total_out = 0

    while True:
        q = input("\n你：")
        if q.strip().lower() in ("exit", "quit"):
            break
        if not q.strip():
            continue

        try:
            answer, usage = chat_stream(q)
            if usage:
                total_in += usage.prompt_tokens
                total_out += usage.completion_tokens
                print(f"     [本次 输入{usage.prompt_tokens} / 输出{usage.completion_tokens}]")
        except Exception as e:
            print("调用失败：", type(e).__name__, e)

    print(f"\n累计：输入 {total_in} token，输出 {total_out} token")
