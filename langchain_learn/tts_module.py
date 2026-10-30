import asyncio # python异步编程模块，用于运行语音合成任务
from pathlib import Path # 文件路径管理模块，用于确定MP3保存位置
import edge_tts # 文字转语音库，用于将中文文本合成MP3
import time  # 导入时间模块，在等待播放结束时短暂休眠
import pygame  # 导入音频播放库，用于播放MP3和检查播放状态

# 1.定义异步语音合成函数
async def save_mp3(text,output_file):

    # 创建于语音合成对象
    communicate = edge_tts.Communicate(
        text = text,
        voice = "zh-CN-XiaoxiaoNeural",
        rate = "-5%"
    )
    # 等待语音合成完成，并保存文件
    await communicate.save(str(output_file))

# 2.封装供其他模块调用的函数
def text_to_speech(text):

    # 检查回答内容是否为空
    if not text or not text.strip():
        print("没有需要转换成语音的文字")
        return None
    # 指定生成MP3的位置
    output_file = Path(__file__).with_name("rag_answer.mp3")
    print("正在生成语音。。。")
    #运行异步语音合成任务
    asyncio.run(save_mp3(text,output_file))
    print ("语音生成完成：",output_file)
    # 返回MP3文件路径
    return output_file
# 播放指定音频，并等待正常播放结束后再返回
def play_audio(audio_file):
    # 将字符串或Path对象统一转换为Path对象
    audio_file = Path(audio_file)

    # 检查音频文件是否存在
    if not audio_file.is_file():
        raise FileNotFoundError(f"找不到音频文件：{audio_file}")

    # 初始化音频播放模块
    pygame.mixer.init()

    try:
        # 加载音频，此时尚未开始播放
        pygame.mixer.music.load(str(audio_file))

        # 开始播放，play调用本身不会等待播放结束
        pygame.mixer.music.play()

        # 播放期间持续等待，每隔约50毫秒检查一次状态
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)

    finally:
        # 正常结束或发生异常时，都释放音频资源
        pygame.mixer.quit()
# 3. 单独测试TTS函数
if __name__ == "__main__":

    text = "你好，欢迎使用景区智能语音问答系统。"

    audio_file = text_to_speech(text)

    print("生成的文件：", audio_file)