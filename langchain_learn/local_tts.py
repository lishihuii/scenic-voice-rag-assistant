# 导入路径工具，定位本地模型和输出音频
from pathlib import Path

# 导入计时函数，分别测量模型加载和语音合成耗时
from time import perf_counter

# 导入本地语音推理库，运行MeloTTS的ONNX模型
import sherpa_onnx

# 导入音频文件工具，将生成的音频保存为WAV
import soundfile as sf
# 导入已有播放函数，播放本地音频并等待结束
from tts_module import play_audio

# 使用调用者传入的模型合成语音，返回生成文件的路径
def local_text_to_speech(text, tts):
    # 空文字无法合成，向调用者报告原因
    text = text.strip()
    if not text:
        raise ValueError("合成文字不能为空")

    # 此处只计算合成及保存耗时，不包含模型加载
    started_at = perf_counter()

    # 使用已有模型生成音频，保留当前0.85的语速
    audio = tts.generate(text, sid=0, speed=0.85)

    # 没有生成声音时，报告错误而不返回无效文件
    if len(audio.samples) == 0:
        raise RuntimeError("本地TTS没有生成有效音频")

    # 将音频保存在当前模块旁边，每次覆盖同一个文件
    output_file = Path(__file__).with_name("melo_output.wav")
    sf.write(
        str(output_file),
        audio.samples,
        audio.sample_rate,
        subtype="PCM_16",
    )

    # 打印耗时，并将文件路径交给调用者
    elapsed = perf_counter() - started_at
    print(f"本地TTS合成并保存耗时：{elapsed:.2f}秒")
    return output_file
# 创建本地TTS模型，由调用者决定何时加载
def create_local_tts():
    # 当前测试文件所在目录
    base_dir = Path(__file__).resolve().parent

    # 模块位于langchain_learn，模型仍保存在rag/test_tool/models
    model_dir = (
            base_dir.parent
            / "test_tool"
            / "models"
            / "vits-melo-tts-zh_en"
    )

    # 配置模型、读音词典、符号表及分词词典目录
    vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(model_dir / "model.onnx"),
        lexicon=str(model_dir / "lexicon.txt"),
        tokens=str(model_dir / "tokens.txt"),
    )

    # 使用CPU推理，先设置两个计算线程
    model_config = sherpa_onnx.OfflineTtsModelConfig(
        vits=vits_config,
        provider="cpu",
        num_threads=2,
    )

    # 指定日期和数字的文本转换规则
    config = sherpa_onnx.OfflineTtsConfig(
        model=model_config,
        rule_fsts=",".join([
            str(model_dir / "date.fst"),
            str(model_dir / "number.fst"),
        ]),
    )

    # 检查配置，再创建可重复使用的语音合成对象
    if not config.validate():
        raise ValueError("TTS配置无效，请检查模型路径和文件")

    print("正在加载本地MeloTTS模型……")
    load_started_at = perf_counter()
    tts = sherpa_onnx.OfflineTts(config)
    print(f"模型加载耗时：{perf_counter() - load_started_at:.2f}秒")

    # 将加载完成的模型交给调用者保存和复用
    return tts


# 保留独立测试入口
def main():
    # 只在进入输入循环之前加载一次
    tts = create_local_tts()

    # 多次输入共用同一个模型
    while True:
        text = input("\n请输入文字，输入exit退出：").strip()

        if text.lower() == "exit":
            break

        # 空输入不执行合成
        if not text:
            continue

        # 使用同一个已加载模型，将输入文字合成为音频
        output_file = local_text_to_speech(text, tts)
        print("音频保存位置：", output_file)
        # 使用pygame播放生成的WAV，并等待播放结束
        print("开始播放")
        play_audio(output_file)

        # 播放完成后，循环才会再次等待文字输入
        print("播放结束，可以输入下一段文字")


# 直接运行本文件时开始测试，导入时不自动运行
if __name__ == "__main__":
    main()