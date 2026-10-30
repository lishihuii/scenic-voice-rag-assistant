from pathlib import Path  # 确定录音文件的保存位置
import numpy as np  # 将多段音频数组拼接成完整音频
import sounddevice as sd  # 从默认麦克风采集声音
import soundfile as sf  # 将采集的声音保存为音频文件
from vad_test import find_speech_in_audio  # 直接检测麦克风产生的音频数组
import time  # 使用单调时钟计算等待时长


# 录制指定时长的声音，返回保存后的文件路径
def record_audio(duration=5, sample_rate=16000):
    # 录音帧数等于录音秒数乘以每秒采样次数
    frames = int(duration * sample_rate)

    # 每次录音覆盖这个文件，供后续语音识别读取
    audio_file = Path(__file__).resolve().parent / "user_question.wav"

    print(f"开始录音，请在 {duration} 秒内提问")

    # 采集单声道音频，等待录音完成后才继续执行
    audio = sd.rec(
        device=1,  # 使用耳机麦克风录音
        frames=frames,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocking=True,
    )

    # 将录制完成的音频数组保存为WAV文件
    sf.write(str(audio_file), audio, sample_rate)

    print("录音完成")

    # 返回路径，交给已有的speech_to_text函数
    return audio_file
# 等待用户开口，收集一次语音，并返回录音路径或None
def monitor_speech(wait_timeout=15):
    sample_rate = 16000  # 每秒采集16000个采样点
    block_frames = 16000  # 每次读取约1秒的声音
    # 按采集顺序保存每段音频数组
    audio_chunks = []
    # 开始监听时，还没有收集到满足结束条件的语音
    speech_finished = False
    print("开始监听耳机麦克风")

    # 进入with时开启音频流，退出时自动停止并释放麦克风
    with sd.InputStream(
        device=1,  # 使用已验证能够收音的耳机麦克风
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    ) as stream:
        # 麦克风打开后，记录开始等待用户开口的时间
        wait_started_at = time.monotonic()

        # 持续采集，直到等待超时或检测到本次说话结束
        while True:
            # 获取音频数组，以及是否发生输入缓冲区溢出的标记
            audio, overflowed = stream.read(block_frames)

            # 溢出意味着可能丢失了部分声音，先打印出来便于排查
            if overflowed:
                print("提示：音频输入发生溢出，可能丢失部分声音")
            # 保存本次采集的数据副本，避免只剩下最后一段声音
            audio_chunks.append(audio.copy())
            # 拼接目前采集的所有声音，保留跨片段的语音上下文
            current_audio = np.concatenate(audio_chunks, axis=0)

            # 检测累计音频中的语音片段
            segments = find_speech_in_audio(current_audio, sample_rate)

            # 尚未检测到语音时，检查等待用户开口的时间
            if not segments:
                waited_seconds = time.monotonic() - wait_started_at
                print(f"等待用户说话：{waited_seconds:.1f}秒")

                # 等待超时，退出采集循环，后面会返回None
                if waited_seconds >= wait_timeout:
                    print("等待用户开口超时")
                    break

                # 尚未超时，继续读取下一段声音
                continue

            # 取最后一个语音片段的结束位置，单位是采样点
            last_speech_end = segments[-1]["end"]

            # 计算语音结束位置到当前音频末尾之间的时长
            silence_seconds = (
                                      current_audio.shape[0] - last_speech_end
                              ) / sample_rate

            print(f"语音结束后未检测到新语音：{silence_seconds:.2f}秒")

            # 已经有语音，并且后面持续约1秒没有新语音，就结束采集
            if silence_seconds >= 1.0:
                # 标记本次语音已经满足结束条件
                speech_finished = True
                print("检测到本次说话结束")
                break

    # 未满足说话结束条件时，不把采集的数据交给问答系统
    if not speech_finished:
        print("等待结束，没有取得已结束的语音")
        return None

    # 将本次采集的声音按时间顺序拼接起来
    full_audio = np.concatenate(audio_chunks, axis=0)

    # 保存录音，供已有语音识别函数读取
    audio_file = Path(__file__).resolve().parent / "user_question.wav"
    sf.write(str(audio_file), full_audio, sample_rate)

    print("提问音频已保存：", audio_file)

    # 返回本次生成的音频文件路径
    return audio_file


# 单独运行本文件时测试持续监听，导入模块时不执行
if __name__ == "__main__":
    # 查看监听函数本次返回了文件路径，还是None
    question_file = monitor_speech()
    print("本次提问文件：", question_file)