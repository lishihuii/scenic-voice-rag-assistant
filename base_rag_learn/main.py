from pathlib import Path  # 定位已有的欢迎语文件
from threading import Thread, Event  # 创建后台线程，并保存共享的会话状态
from hand_test import run_hand_detection  # 调用已有的手势检测模块
from conversation_memory import append_turn, clear_history, load_history  # 导入追加、清空和读取会话记录的函数

from asr_test import speech_to_text, get_whisper_model  # 导入语音识别和模型加载函数
from tts_module import play_audio, text_to_speech  # 导入音频播放和文字转语音函数
from record_audio import monitor_speech  # 导入等待开口并自动判断说话结束的函数

# 标记会话是否已开启：设置表示BUSY，清除表示IDLE
session_active = Event()

# 标记语音识别模型是否已经准备完成
asr_ready = Event()


# 在后台提前加载语音识别模型
def prepare_asr():
    try:
        # 与后续识别共用同一个模型获取函数
        get_whisper_model()
        # 提前执行RAG模块的初始化，加载Embedding模型和FAISS索引
        print("正在准备景区知识库……")
        from txt_rag import rag_answer  # 导入时完成现有模块的初始化，不调用DeepSeek
        print("景区知识库准备完成")
        # 只有加载成功，才允许开启语音会话
        asr_ready.set()
        print("语音识别与景区知识库准备完成，可以通过两次握拳唤醒")

    except Exception as error:
        # 准备失败时保留未就绪状态，并输出具体原因
        print("语音模型准备失败：", error)
# 后台执行语音会话，本步先接入欢迎语播放
def run_voice_session():
    welcome_file = (
        Path(__file__).resolve().parent / "test_tool" / "welcome.mp3"
    )

    try:
        # 本次会话刚进入BUSY，准备一份空的会话记录
        clear_history()
        print("助手进入BUSY，开始播放欢迎语")
        play_audio(welcome_file)
        # 欢迎语只播放一次，随后持续处理用户提问
        while True:
            # 等待用户开口，并收集本次提问语音
            question_file = monitor_speech(wait_timeout=15)

            # 等待超时，没有取得提问录音，结束会话循环
            if question_file is None:
                print("等待提问语音超时，结束本次会话")
    
                # 先清空本次会话，再退出循环并恢复IDLE
                clear_history()
                break

            # 调用已有识别函数，将音频转换成问题文字
            question = speech_to_text(question_file)

            # 在控制台打印识别结果，先验证问题是否识别正确
            print("游客问题：", question)
            # 未识别到有效文字时，继续监听，不提交给RAG
            if not question.strip():
                print("没有识别到有效问题，继续等待提问")
                continue

            # 首次处理问题时加载RAG模块，避免在打开摄像头前加载知识库模型
            from txt_rag import rag_answer  # 导入已有的检索增强问答函数

            print("正在检索景区资料，并请求DeepSeek回答……")

            # 读取此前已经完成的问答，不包含当前这一轮
            history = load_history()
            print("本轮读取的历史消息：", history)

            # 将历史消息与当前问题一起交给RAG问答函数
            answer = rag_answer(question, history=history)

            # 先在控制台检查回答内容
            print("助手回答：", answer)
            # 收到有效回答后立即保存，不等待语音合成和播放
            if answer.strip():
                append_turn(question, answer)
            # 将回答文字合成为MP3，获得生成文件的路径
            answer_file = text_to_speech(answer)

            # 回答为空时，合成函数会返回None，交给except处理
            if answer_file is None:
                raise ValueError("回答内容为空，无法生成语音")

            # 播放合成的回答，并等待播放结束
            print("开始播放助手回答")
            play_audio(answer_file)

            # 只有播放结束后，才会执行这一行
            print("本轮回答播放完成")

    except Exception as error:

        # 出现异常时打印原因，结束本次会话

        print("语音会话执行失败：", error)


    finally:

        # 正常超时退出或发生异常，都解除会话占用

        session_active.clear()

        print("助手回到IDLE，继续等待两次握拳唤醒")


# 摄像头检测到两次握拳时，调用此函数
def start_session():
    # 已有会话时，不重复启动
    if session_active.is_set():
        return
    # 模型尚未准备完成时，不启动语音会话，避免重复加载
    if not asr_ready.is_set():
        print("语音模型正在准备，请稍后再次握拳")
        return
    # 先锁定会话，再启动后台任务
    session_active.set()

    # 后台播放音频，摄像头所在的主线程继续运行
    voice_thread = Thread(target=run_voice_session, daemon=True)
    voice_thread.start()


# 总程序组装手势模块和语音会话
def main():
    print("启动摄像头，后台准备Whisper语音模型")

    # 启动准备线程，主线程继续执行后面的摄像头检测
    prepare_thread = Thread(target=prepare_asr, daemon=True)
    prepare_thread.start()

    # 传入两个函数，由手势模块在需要时调用
    run_hand_detection(
        on_wake=start_session,
        is_busy=session_active.is_set
    )


# 直接运行本文件时启动总程序
if __name__ == "__main__":
    main()