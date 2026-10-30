# 导入路径工具，根据当前文件位置定位知识文档
from pathlib import Path
# 导入后台线程和共享状态标记，分别执行会话任务与记录BUSY状态
from threading import Thread, Event
# 导入高精度计时函数，用于测量一段代码的执行耗时
from time import perf_counter
# 导入手势检测函数，复用已有的两次握拳识别
from hand_test import run_hand_detection
# 导入音频播放函数，欢迎语和本地生成的回答均通过它播放
from tts_module import play_audio

# 导入本地TTS的模型创建函数和语音合成函数
from local_tts import create_local_tts, local_text_to_speech
# 导入持续监听函数，等待开口并在检测到说话结束后返回录音路径
from record_audio import monitor_speech
# 导入语音识别及模型获取函数，提前加载后供后续识别复用
from asr_test import speech_to_text, get_whisper_model

# 创建共享会话标记：未设置为IDLE，设置后为BUSY
session_active = Event()
# 标记模型和知识库是否全部准备完成
models_ready = Event()

# 保存整个程序运行期间重复使用的RAG对象
rag_resources = {}
# 从同目录的RAG模块导入知识库准备、回答链创建和问答函数
from rag_service import (
    build_knowledge_base,
    create_answer_chain,
    rag_answer,
)

# 在后台准备模型和知识库，成功后才允许唤醒
def prepare_models():
    try:
        print("正在准备Whisper语音识别模型……")

        # 提前加载，后续speech_to_text会复用同一个模型
        get_whisper_model()

        # 根据入口文件位置定位知识文档
        knowledge_file = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "scenic_guide.txt"
        )

        print("正在准备LangChain景区知识库……")

        # 创建Embedding对象，构建或加载已有FAISS索引
        embeddings, index, chunks = build_knowledge_base(knowledge_file)

        # 创建回答链，此时不发送问答请求
        chain = create_answer_chain()
        # 在后台提前加载本地TTS，后续每轮回答直接复用
        print("正在准备本地语音合成模型……")
        tts = create_local_tts()
        # 将准备好的对象保存到共享字典
        rag_resources.update({
            "embeddings": embeddings,
            "index": index,
            "chunks": chunks,
            "chain": chain,
            # 保存本地TTS模型，后续所有回答共用
            "tts": tts
        })

        # 所有对象准备成功后，才设置就绪标记
        models_ready.set()
        print("模型准备完成，可以通过两次握拳唤醒")

    except Exception as error:
        # 准备失败时不设置就绪标记，保留错误信息供排查
        print("模型准备失败：", error)
# 保留已有的文字问答测试，需要时可以单独调用
def run_text_test():
    # 当前文件位于langchain_learn，上一级是rag目录
    knowledge_file = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "scenic_guide.txt"
    )

    # 启动时准备一次模型、索引和文本块
    embeddings, index, chunks = build_knowledge_base(knowledge_file)

    # 启动时创建一次回答链，之后重复使用
    chain = create_answer_chain()

    # 保存本次会话中已经完成的问答
    history = []

    while True:
        # 临时使用文字输入，先验证模块之间的调用
        question = input("\n请输入问题，输入exit退出：").strip()

        # 输入退出指令时，结束循环
        if question.lower() == "exit":
            break

        # 空输入不进行检索和模型调用
        if not question:
            continue

        # 将当前问题、已准备的对象及此前历史交给RAG模块
        answer = rag_answer(
            question=question,
            embeddings=embeddings,
            index=index,
            chunks=chunks,
            chain=chain,
            history=history,
            top_k=3,
            min_similarity=0.1,
        )

        # 本轮完成后保存问题和回答，供后续提问使用
        history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])

        # 显示本轮回答
        print("助手：", answer)


# 在后台线程中执行会话任务，当前先接入欢迎语
def run_voice_session():
    # 每次唤醒创建独立历史；同一次会话的多轮问答共用它
    history = []
    try:
        # 欢迎语位于rag目录下的test_tool文件夹中
        welcome_file = (
            Path(__file__).resolve().parent.parent
            / "test_tool"
            / "melo_output.wav"
        )

        # 播放并等待结束，只阻塞当前后台线程
        print("开始播放欢迎语")
        play_audio(welcome_file)

        # 欢迎语播放结束后，持续等待游客提问
        print("欢迎语播放结束，开始监听")

        while True:
            # 每次等待新问题最多15秒，检测到语音后继续收集至说话结束
            question_file = monitor_speech(wait_timeout=15)

            # 没有取得提问音频，说明等待超时，结束当前会话
            if question_file is None:
                print("等待提问超时，结束本次会话")
                break

            # 将刚刚保存的提问音频转换成文字
            print("本次提问音频：", question_file)
            question = speech_to_text(question_file)

            # 没有识别到有效文字时，跳过本轮后续处理，重新监听
            if not question.strip():
                print("没有识别到有效问题，请重新提问")
                continue

            # 显示识别到的当前问题
            print("游客问题：", question)
            print("正在检索资料并生成回答……")
            # 记录RAG处理开始时的计时值
            rag_started_at = perf_counter()
            # 复用启动时准备的模型、索引、文本块和回答链
            answer = rag_answer(
                question=question,
                embeddings=rag_resources["embeddings"],
                index=rag_resources["index"],
                chunks=rag_resources["chunks"],
                chain=rag_resources["chain"],
                history=history,
                top_k=3,
                min_similarity=0.1,
            )
            # 当前计时值减去开始值，得到本次RAG处理秒数
            rag_elapsed = perf_counter() - rag_started_at

            # 以小数点后两位显示耗时
            print(f"RAG问答耗时：{rag_elapsed:.2f}秒")
            # 回答完成后，保存本轮问答供下一轮使用
            history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])

            # 显示回答文字，方便核对问答内容
            print("助手回答：", answer)

            # 记录语音合成开始时的计时值
            tts_started_at = perf_counter()

            # 将回答文字合成为音频文件
            # 使用启动时加载好的本地模型，将回答合成为WAV
            answer_file = local_text_to_speech(
                answer,
                rag_resources["tts"],
            )

            # 计算语音合成耗时，单位为秒
            tts_elapsed = perf_counter() - tts_started_at

            # 输出本轮语音合成耗时
            print(f"TTS合成耗时：{tts_elapsed:.2f}秒")

            # 没有生成音频时，结束当前会话并进入异常处理
            if answer_file is None:
                raise ValueError("回答内容为空，无法生成语音")

            # 播放回答，并在当前后台线程中等待播放结束
            print("开始播放助手回答")
            play_audio(answer_file)

            # 播放结束后到达循环末尾，下一轮才重新打开麦克风
            print("回答播放完成，继续等待游客提问")

    except Exception as error:
        # 播放或监听发生异常时，结束当前会话并打印原因
        print("语音会话执行失败：", error)

    finally:
        # 无论正常超时还是发生异常，都解除会话占用
        session_active.clear()
        print("助手回到IDLE，继续等待两次握拳")


# 检测到两次握拳时，由摄像头线程调用
def start_session():
    # 已经处于BUSY时，不重复开启会话
    if session_active.is_set():
        return
    # 模型还没有准备完成时，不启动语音会话
    if not models_ready.is_set():
        print("模型尚未就绪，请等待准备完成后重新握拳")
        return
    # 先设置BUSY，防止后台任务启动期间再次触发
    session_active.set()
    print("助手进入BUSY，手势唤醒成功")

    try:
        # 指定后台线程要执行的函数，此处不立即调用
        voice_thread = Thread(
            target=run_voice_session,
            daemon=True,
        )

        # 启动后台线程，然后当前函数返回摄像头循环
        voice_thread.start()

    except Exception as error:
        # 线程未能启动时，恢复为可唤醒状态
        session_active.clear()
        print("会话线程启动失败：", error)


# 新版总入口：当前先验证摄像头与唤醒状态
def main():
    print("助手处于IDLE，请伸直两指后完成两次握拳")
    # 后台准备模型，主线程继续打开摄像头
    prepare_thread = Thread(target=prepare_models, daemon=True)
    prepare_thread.start()
    # 摄像头持续运行；检测到动作时调用启动函数
    run_hand_detection(
        on_wake=start_session,
        is_busy=session_active.is_set,
    )
# 直接运行本文件时启动，其他模块导入时不自动启动
if __name__ == "__main__":
    main()