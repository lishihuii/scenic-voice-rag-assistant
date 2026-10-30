
import asyncio  # Python异步编程模块，用于执行异步语音合成任务
from pathlib import Path  # 路径处理模块，用于设置MP3文件保存位置
import edge_tts  # 在线TTS语音合成库，将中文文字转换成语音


# 1. 输入需要转换成语音的文字
text = "游客中心在哪里？"

# 2. 选择中文发音人
voice = "zh-CN-XiaoxiaoNeural"

# 3. 设置MP3文件保存位置
output_file = Path(__file__).with_name("test_voice_question.mp3")


# 4. 定义语音合成函数
async def generate_mp3():
    # 创建TTS语音合成对象
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate="-5%"
    )

    # 生成语音并保存为MP3
    await communicate.save(str(output_file))


# 5. 执行异步函数
asyncio.run(generate_mp3())

# 6. 输出生成结果
print("MP3语音生成成功！")
print("文件位置：", output_file)
