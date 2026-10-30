from pathlib import Path  # 确定会话记录文件的位置
import json  # 将一轮问答转换为便于保存和读取的JSON文本


# 保存当前会话的对话记录，与本模块放在同一目录
HISTORY_FILE = Path(__file__).resolve().parent / "session_history.txt"


# 向文件末尾追加一轮问答
def append_turn(question, answer):
    # 用字典保存问题和回答，保留两者的对应关系
    turn = {
        "question": question,
        "answer": answer,
    }

    # 将这一轮问答转换为一行文本，保留中文字符
    line = json.dumps(turn, ensure_ascii=False)

    # 使用追加模式打开文件，保留前面已经保存的对话
    with HISTORY_FILE.open("a", encoding="utf-8") as file:
        # 每轮问答占一行，方便以后逐行读取
        file.write(line + "\n")
    # 文件关闭后，打印实际保存位置，确认本轮已经写入
    print("本轮对话已保存：", HISTORY_FILE)


# 开始新会话或等待超时时，清空会话记录
def clear_history():
    # 覆盖写入空字符串；文件不存在时会自动创建
    HISTORY_FILE.write_text("", encoding="utf-8")
    print("会话记录已清空：", HISTORY_FILE)
# 读取本次会话记录，转换为聊天接口使用的历史消息列表
def load_history():
    # 按顺序保存游客和助手的历史消息
    messages = []

    # 文件尚未创建时，表示没有历史对话
    if not HISTORY_FILE.exists():
        return messages

    # 以只读模式逐行读取会话记录
    with HISTORY_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            # 跳过空行，避免把空字符串当作JSON解析
            if not line.strip():
                continue

            # 将一行JSON文本还原为问答字典
            turn = json.loads(line)

            # 先添加这一轮游客的问题
            messages.append({
                "role": "user",
                "content": turn["question"],
            })

            # 再添加这一轮助手的回答
            messages.append({
                "role": "assistant",
                "content": turn["answer"],
            })

    # 返回按照实际对话顺序排列的消息
    return messages