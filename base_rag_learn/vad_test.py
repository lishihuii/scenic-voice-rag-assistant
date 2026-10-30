from pathlib import Path  # 定位用于测试的音频文件
import soundfile as sf  # 读取音频文件中的声音数据
from scipy.signal import resample_poly  # 将采样率统一为16000Hz
from faster_whisper.vad import VadOptions, get_speech_timestamps  # 配置并执行语音活动检测


# 直接检测音频数组，返回语音片段的起止采样点
def find_speech_in_audio(audio, sample_rate=16000):
    # 双声道音频先转换为单声道
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # 将音频采样率统一为16000Hz
    if sample_rate != 16000:
        audio = resample_poly(audio, 16000, sample_rate)

    # 设置语音片段的判断参数，后续根据实际效果调整
    # 先用当前录音验证过的试验参数，提高对短词的敏感度
    options = VadOptions(
        threshold=0.2,  # 降低判定为候选语音的启动阈值
        min_speech_duration_ms=100,  # 保留较短的候选语音片段
        min_silence_duration_ms=800,
        speech_pad_ms=200,
    )

    # 返回检测到的语音片段列表
    segments = get_speech_timestamps(
        audio,
        vad_options=options,
        sampling_rate=16000,
    )

    return segments

# 读取音频文件，再交给音频数组检测函数
def find_speech(audio_file):
    # 从文件中读取声音数据和采样率
    audio, sample_rate = sf.read(str(audio_file), dtype="float32")

    # 复用数组检测函数，并返回检测结果
    return find_speech_in_audio(audio, sample_rate)
# 单独运行本文件时，检测已有的问题音频
if __name__ == "__main__":
    audio_file = Path(__file__).resolve().parent / "test_voice_question.mp3"

    # 打印语音片段的起止采样点
    segments = find_speech(audio_file)
    print("语音片段：", segments)