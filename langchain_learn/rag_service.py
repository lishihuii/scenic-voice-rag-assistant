from pathlib import Path  # 导入路径工具，定位并读取文件
from langchain_core.documents import Document # 导入文档类，保存正文和来源
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 导入提示词模板及历史消息占位符
from langchain_deepseek import ChatDeepSeek
from langchain_text_splitters import RecursiveCharacterTextSplitter #  导入文本切分器
from langchain_huggingface import  HuggingFaceEmbeddings # 导入本地Embedding模型接口
import numpy as np
import faiss
import os
import json



# 准备知识库，返回后续检索需要复用的三个对象
def build_knowledge_base(file_path):
    # 读取TXT，并保存实际来源路径
    txt_path = Path(file_path)
    text = txt_path.read_text("utf-8-sig")
    documents = [
        Document(
            page_content=text,
            metadata={
                "source": str(txt_path)
            }
        )
    ]

    # 切分配置
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        length_function=len,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No chunks found")

    # 加载已有的本地BGE模型
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={
            "device": "cpu",
            "local_files_only": True,
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    # 将新版索引单独保存，避免与旧版索引混用
    cache_dir = Path(__file__).resolve().parent / "index_cache"

    # 创建缓存目录；已经存在时直接使用
    cache_dir.mkdir(parents=True, exist_ok=True)

    index_file = cache_dir / "scenic.index"
    metadata_file = cache_dir / "scenic_metadata.json"

    # 记录影响索引复用的信息，以及文本块的内容、顺序和来源
    current_metadata = {
        "model_name": embeddings.model_name,
        "encode_kwargs": embeddings.encode_kwargs,
        "index_type": "IndexFlatIP",
        "chunks": [
            {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]
    }

    # 两个缓存文件都存在时，检查是否可以复用
    if index_file.exists() and metadata_file.exists():
        try:
            # 读取上次保存的知识库描述信息
            saved_metadata = json.loads(
                metadata_file.read_text(encoding="utf-8")
            )

            # 配置及文本块完全一致，才读取对应索引
            if saved_metadata == current_metadata:
                index = faiss.read_index(str(index_file))

                # 检查索引类型和向量数量，保持编号与chunks对应
                if (
                        isinstance(index, faiss.IndexFlatIP)
                        and index.ntotal == len(chunks)
                ):
                    print("知识库未变化，已加载本地FAISS索引")
                    return embeddings, index, chunks

                print("索引类型或向量数量不匹配，需要重建")
            else:
                print("知识文本或模型配置发生变化，需要重建索引")

        except (OSError, ValueError, RuntimeError) as error:
            # 缓存无法正常读取时，重新计算并保存
            print("本地索引缓存读取失败，需要重建：", error)

    # 没有可复用缓存时，才执行全部文档的向量化
    print("正在计算知识文本向量并建立索引……")
    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(texts)

    doc_vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(doc_vectors.shape[1])
    index.add(doc_vectors)

    # 保存向量索引，再保存与之对应的知识库描述信息
    faiss.write_index(index, str(index_file))
    metadata_file.write_text(
        json.dumps(current_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("新的FAISS索引及知识库描述信息已保存")
    return embeddings, index, chunks

# 创建回答生成链
def create_answer_chain():
    # 从当前运行进程的环境变量中读取api_key
    api_key =  os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "请在rag_service.py的运行配置中设置DEEPSEEK_API_KEY"
        )
    # 创建提示词模板，保留每次需要填入的资料和问题
    # 按顺序组织系统规则、历史消息和本轮输入
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是景区问答助手。"
            "可以依据历史对话回顾游客提供的信息、理解追问。"
            "回答景区事实时，必须以本轮参考资料为依据。"
            "历史回答不能作为独立的景区事实依据。"
            "资料没有明确答案时，请说明无法确定。"
            "不要编造位置、价格或开放时间。"
            "回答尽量简短。"
        ),
        # 在这个位置插入history中的历史消息
        MessagesPlaceholder(variable_name="history"),
        (
            "human",
            "参考资料：\n{context}\n\n游客问题：{question}"
        )
    ])
    # 创建模型调用对象
    llm = ChatDeepSeek(
        model="deepseek-flash",
        api_key=api_key,
        timeout=60,
        max_retries=1,
        extra_body={
            "thinking": {"type": "disabled"}
        }
    )
    # 连接组件，并将可执行的链返回给调用者
    chain = prompt_template | llm | StrOutputParser()
    return chain

# 根据当前问题，检索并返回相关Document列表
def retrieve_documents(question, embeddings, index, chunks, top_k=3, min_similarity=None):
    # 去掉问题首尾空白，空问题直接返回空列表
    question = question.strip()
    if not question:
        return []
    # 确定实际检索数量，避免超过索引中的向量数
    k = min(top_k, index.ntotal)
    if k <= 0:
        return []
    # 使用已有模型，将当前问题转换为一条向量
    query_embedding = embeddings.embed_query(question)
    # 转换成FAISS需要的二位float32数组
    query_vector = np.ascontiguousarray([query_embedding], dtype=np.float32)
    # 搜索与当前问题最相似的向量
    scores,indices = index.search(query_vector, k)

    # 保存本次找到的Document对象
    results = []
    for score,chunk_id in zip(scores[0],indices[0]):
        # 将numpy整数转为普通python整数
        chunk_id = int(chunk_id)
        if chunk_id == -1:
            continue
        # 将相似度转换为普通Python浮点数
        score = float(score)

        # 设置了阈值，且当前分数低于阈值时，跳过这条资料
        if min_similarity is not None and score < min_similarity:
            print("已过滤文本块：", chunk_id, "相似度：", score)
            continue
        # 按编号取回Document，保留正文及来源信息
        results.append(chunks[chunk_id])
        # 暂时保留检索日志
        print("命中文本块", chunk_id,"相似度",float(score),"文本块内容",chunks[chunk_id].page_content)
    return results



# 完成一轮问答，同时接收此前的历史消息
def rag_answer(question, embeddings, index, chunks, chain,
               top_k=3, history=None, min_similarity=None):
    # 空问题不执行检索和模型调用
    question = question.strip()
    if not question:
        return "请输入有效问题。"

    # 检索本轮问题对应的资料
    retrieved_docs = retrieve_documents(
        question, embeddings, index, chunks, top_k=top_k, min_similarity=min_similarity
    )

    if not retrieved_docs:
        return "没有检索到可用的景区资料，暂时无法回答。"

    # 拼接本轮参考资料
    context = "\n\n".join(
        document.page_content for document in retrieved_docs
    )

    # 没有传入历史时，使用空列表
    all_history = history if history is not None else []

    # 每轮包含用户问题和助手回答，只选最近三轮
    recent_history = all_history[-6:]
    # 调试时查看完整历史与本次发送历史的消息数量
    print("完整历史：", len(all_history), "本次发送：", len(recent_history))
    # 将最近三轮历史、本轮资料和当前问题一起提交给模型
    answer = chain.invoke({
        "context": context,
        "question": question,
        "history": recent_history,
    })

    return str(answer).strip()


# 创建问题改写链，只负责补全问题，不负责回答问题
def create_rewrite_chain():
    # 读取当前运行入口配置的DeepSeek密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "请在rag_service.py的运行配置中设置DEEPSEEK_API_KEY"
        )

    # 设置改写任务：根据历史补全省略的信息，保留用户原意
    rewrite_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是检索问题改写助手。"
            "请根据历史对话，将当前问题改写为可以独立理解的检索问题。"
            "仅补全历史中能够明确确定的指代或省略信息。"
            "保持当前问题的原意，不添加猜测、条件或事实。"
            "如果当前问题已经完整，或者用户切换了话题，保留当前问题。"
            "只输出改写后的问题，不回答问题，不解释改写过程。"
        ),
        # 插入此前的历史问答，帮助确定当前问题的指代
        MessagesPlaceholder(variable_name="history"),
        (
            "human",
            "需要改写的当前问题：{question}"
        )
    ])

    # 创建模型调用对象，沿用非思考模式配置
    llm = ChatDeepSeek(
        model="deepseek-flash",
        api_key=api_key,
        timeout=60,
        max_retries=1,
        extra_body={
            "thinking": {"type": "disabled"}
        }
    )

    # 返回改写链，调用invoke时才发送请求
    return rewrite_template | llm | StrOutputParser()

# 验证封装函数
if __name__ == "__main__":
    knowledge_file = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "scenic_guide.txt"
    )
    embeddings, index, chunks = build_knowledge_base(knowledge_file)
    print("知识库准备完成")
    print("文本块数量：", len(chunks))
    print("索引向量数量：", index.ntotal)
    print("向量维数：", index.d)

    chain = create_answer_chain()

    # 保存本次运行中的历史消息，放在循环外才能跨轮保留
    history = []

    while True:
        # 读取本轮问题
        question = input("\n请输入游客问题，输入exit退出：").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        # 传入此前已经完成的问答，不提前加入当前问题
        answer = rag_answer(
            question, embeddings, index, chunks, chain,
            top_k=3,
            history=history,
            min_similarity=0.1
        )

        # 本轮回答完成后，依次保存用户问题和助手回答
        history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])

        print("\n助手回答：", answer)
        print("当前历史消息数量：", len(history))