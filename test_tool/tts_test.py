import asyncio # python异步编程模块，用于运行异步函数
import os # 操作系统模块，用于打开文件等操作
from pathlib import Path # 路径处理工具，用于管理文件路径
import edge_tts # 语音合成库，用于将文字转换为语言
from edge_tts import communicate
from openai.types.audio import voice


# 1.定义语音合成函数
async def text_to_speech(text):

    # 选择中文女声
    voice = "zh-CN-XiaoxiaoNeural"
    # voice = "zh-CN-YunxiNeural"
    # 音频保存路径
    output_file = Path(__file__).with_name("welcome.mp3")
    # 2.创建语音合成对象
    communicate = edge_tts.Communicate(
        text = text,
        voice = voice,
        rate = "-5%"
    )
    # 3.生成成功并保存音频
    await communicate.save(str(output_file))
    return output_file
# 4.准备需要朗读的文字
answer = "您好,这里是景区智能语音助手，你需要帮忙吗？"
# 5.执行异步语音合成函数
audio_file = asyncio.run(text_to_speech(answer))
print("语音合成成功!")
print("音频路径：",audio_file)
# 6.使用windows默认播放器打开音频
os.startfile(str(audio_file))
