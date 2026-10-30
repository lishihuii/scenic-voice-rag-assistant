import numpy as np  # NumPy数组处理，整理音频数据
import soundfile as sf  # 读取MP3文件，转换为数字音频数组
from scipy.signal import resample_poly  # 音频重采样
from math import gcd  # 计算最大公约数，得到重采样比例
from pathlib import Path  # 文件路径管理工具
from faster_whisper import WhisperModel  # Whisper语音识别模型
from time import perf_counter  # 使用高精度计时器测量程序耗时

# # 2.设置音频文件保存路径
# audio_file = Path(__file__).with_name("test_voice.mp3")
#
# # 2. 读取音频文件
# audio, sample_rate = sf.read(
#     str(audio_file),
#     dtype="float32"
# )
# print("原始采样率：", sample_rate)
# print("音频数组形状：", audio.shape)
#
# # 3. 如果是双声道，转换为单声道
# if audio.ndim == 2:
#     audio = audio.mean(axis=1)
#
# # 4. 将音频采样率统一调整为16000Hz
# target_rate = 16000
#
# if sample_rate != target_rate:
#     divisor = gcd(sample_rate, target_rate)
#
#     audio = resample_poly(
#         audio,
#         target_rate // divisor,
#         sample_rate // divisor
#     )
#
# # 5. 转换为Whisper所需的float32音频数组
# audio = np.asarray(audio, dtype=np.float32)
# # 5.加载预训练语音识别模型
# print("正在加载Whisper-small模型。。。")
# model = WhisperModel(
#     "small",
#     device = "cuda",
#     compute_type = "float16"
# )
# # 6.将语音转为文字
# print("正在识别语音。。。")
#
# # 直接把音频数据交给Whisper识别
# segments, info = model.transcribe(
#     # audio_data,
#     audio,
#     language = "zh",
#     beam_size = 5
# )
#
# # 7.拼接识别结果
# question = "".join(
#     segment.text for segment in segments
# ).strip()
#
# # 8.输出文字
# if question:
#     print("\n语音识别结果：",question)
# else:
#     print("\n没有识别到有效语音，请重新录制。")





# 将上述功能封装成函数
# 保存已经加载的模型，None表示当前还没有加载
whisper_model = None


# 按需加载Whisper模型，并在后续识别中复用
def get_whisper_model():
    # 允许函数更新模块中保存的模型变量
    global whisper_model

    # 只有第一次调用时，才创建模型
    if whisper_model is None:
        print("首次加载Whisper-small模型……")

        # 保留此前已经使用的模型和运行参数
        whisper_model = WhisperModel(
            "small",
            device="cuda",
            compute_type="float16",
        )

    # 首次返回新模型，之后返回同一个已加载模型
    return whisper_model
# 定义语音转文字函数
def speech_to_text(audio_file):
    # 1.读取音频文件
    audio,sample_rate = sf.read(
        str(audio_file),
        dtype = "float32",
    )
    print("原始采样率",sample_rate)

    # 2.转为单声道
    if audio.ndim == 2:
        audio = audio.mean(axis = 1)
    # 3.重采样到16000Hz
    target_rate = 16000
    if sample_rate != target_rate:
        divisor = gcd(sample_rate, target_rate)
        audio = resample_poly(
            audio,
            target_rate // divisor,
            sample_rate // divisor
        )

    # 4.转换成float32数组
    audio = np.asarray(audio,dtype = np.float32)

    # 获取模型：第一次加载，之后直接复用
    model = get_whisper_model()
    # 6.开始语音识别
    print("正在识别语音。。。")

    segments, info = model.transcribe(
        audio,
        language = "zh",
        beam_size = 5
    )

    # 7.拼接识别结果
    question = "".join(
        segment.text for segment in segments
    ).strip()

    # 8.返回识别结果
    return question
if __name__ =="__main__":
    # 获取测试MP3的路径
    audio_file = Path(__file__).with_name("test_voice_question.mp3")

    # 在同一次程序运行中，连续识别同一个音频两次
    for number in range(2):
        print(f"\n开始第{number + 1}次识别")

        # 记录识别开始前的时间
        started_at = perf_counter()

        # 第一次调用会加载模型，第二次复用已经加载的模型
        question = speech_to_text(audio_file)

        # 计算本次调用的总耗时
        elapsed = perf_counter() - started_at

        print("识别结果：", question)
        print(f"本次耗时：{elapsed:.2f}秒")
