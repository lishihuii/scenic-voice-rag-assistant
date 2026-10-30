# 景区智能语音问答助手

基于 LangChain、DeepSeek 和 FAISS 的景区知识问答项目，结合手势唤醒、本地语音识别与本地语音合成，实现连续语音交互。

## 已实现功能

- 两次握拳唤醒：通过食指和中指状态进行简化握拳判断。
- 欢迎语自动播放，结束后开始监听。
- VAD 检测语音结束，faster-whisper 将录音转换成文字。
- TXT 文档切分、BGE 向量化、FAISS 检索和 DeepSeek 回答生成。
- 多轮会话历史：回答阶段发送最近三轮历史消息。
- MeloTTS 本地语音合成，pygame 自动播放。
- 摄像头持续运行，后台线程处理语音会话，BUSY 期间不重复唤醒。
- 等待新问题超时后返回 IDLE；当前测试设置为 15 秒，可调整为 30 秒。
- 记录 RAG 和 TTS 阶段耗时。

## 交互流程

两次握拳 → 欢迎语 → 监听 → 语音识别 → RAG 问答 → 本地语音合成 → 播放 → 继续监听。

只有等待新问题的时间参与空闲超时，识别、生成和播放期间保持 BUSY。

## 技术栈

Python、LangChain、DeepSeek API、SentenceTransformers、FAISS、
faster-whisper、sherpa-onnx / MeloTTS、OpenCV、MediaPipe、pygame。

当前版本使用本地 MeloTTS，旧版 edge-tts 实验代码仍保留。

## 目录结构

- `langchain_learn/`：当前版本，入口为 `rag_app_start.py`。
- `data/scenic_guide.txt`：示例景区知识文档。
- `test_tool/`：独立语音测试、欢迎语和本地 TTS 模型存放位置。
- `base_rag_learn/`：早期原生 RAG 与相关学习代码。
- `requirements.txt`：当前版本的主要 Python 依赖。

旧练习文件保留学习过程，部分路径尚未适配目录调整。

## 环境与依赖

当前已在 Windows、Python 3.13 环境中运行。
ASR 当前使用 NVIDIA CUDA，MeloTTS 和 Embedding 使用 CPU。

```powershell
python -m pip install -r requirements.txt
```

依赖清单来自当前开发环境，尚未完成全新环境安装验证。
GPU ASR 还需配置与当前 faster-whisper / CTranslate2 匹配的 CUDA、cuDNN。

## 模型准备

模型文件未上传，需要自行下载：

1. MediaPipe Hand Landmarker 模型：
   放到 `langchain_learn/hand_landmarker.task`。
2. BGE 模型：`BAAI/bge-small-zh-v1.5`。
3. Whisper 模型：`Systran/faster-whisper-small`。
   BGE 和 Whisper 当前仅从本地缓存加载，首次使用前需准备完整缓存。
4. MeloTTS：
   下载官方模型包，完整解压到
   `test_tool/models/vits-melo-tts-zh_en/`。

MeloTTS 下载：
https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2

## 配置与启动

在 PyCharm 的入口运行配置中添加环境变量 `DEEPSEEK_API_KEY`，不要把真实密钥写入代码。

从仓库根目录运行：

```powershell
python langchain_learn/rag_app_start.py
```

等待模型准备完成后，伸直食指和中指，再完成两次握拳。

当前使用 DroidCam，摄像头配置为 `VideoCapture(1, cv2.CAP_MSMF)`。
麦克风配置为设备 1。其他电脑需要先查询设备，再调整对应配置。

欢迎语为 `test_tool/melo_output.wav`。
独立 TTS 测试会覆盖该文件，测试时请注意保留欢迎语。

## 网络依赖

DeepSeek 回答生成需要互联网。
模型资源准备完成后，向量化、检索、语音识别、语音合成和播放均在本地执行。
通过 Wi-Fi 使用 DroidCam 时，仍需手机和电脑之间的局域网连接。

## 当前限制

- 语音播报期间不监听，不支持中途打断。
- 检索以当前问题为输入，带指代的追问可能检索不准。
- 最近三轮以外的对话不会发送给模型。
- 示例景区资料用于项目演示，不代表真实景区信息。