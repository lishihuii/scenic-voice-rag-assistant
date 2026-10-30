import os
import faiss
from pathlib import Path
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import json # Python内置JSON模块，用于保存和读取信息

# 1.准备知识库
documents = [
    "游客服务中心位于景区南门，提供咨询和导览服务。",
    "景区每天早上八点开放，下午六点关闭。",
    "景区餐饮区位于中心广场东侧。",
    "游客可以在北门乘坐观光车。",
    "景区停车场位于南门外侧，游客可以免费停车。"
]
# 2.加载Embedding模型
print("正在加载Embedding模型。。。")
embedding_model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5",
    device="cpu",
    local_files_only=True
)
# # 3.计算知识库的Embedding向量
# doc_vectors = embedding_model.encode(
#     documents,
#     normalize_embeddings=True
# )
# # FAISS期望接收float32类型的向量矩阵
# doc_vectors = np.ascontiguousarray(
#     doc_vectors,
#     dtype=np.float32
# )
# print("文档向量形状",doc_vectors.shape)
# # 4.创建FAISS索引
# dimension = doc_vectors.shape[1]
# # IndexFlatIP：创建基于内积的精确检索索引
# # IP是Inner Product（内积）的缩写
# #
# # 由于文档向量和问题向量都经过L2归一化，
# # 因此内积就等于余弦相似度
# #
# # dimension：表示索引中每个向量有多少个数字
# index = faiss.IndexFlatIP(dimension)
# # 5.将文档向量加入FAISS
# index.add(doc_vectors)
# # ntotal：FAISS索引中当前保存的向量数量
# print("FAISS索引中的向量数量：", index.ntotal)
# =================================
# 第3步：准备FAISS索引持久化
# =================================

# 指定Embedding模型名称
# 保存索引时也记录模型名称，避免以后换模型却误用旧索引
model_name = "BAAI/bge-small-zh-v1.5"

# 指定索引文件的位置
# faiss.index：保存FAISS索引中的向量数据
# documents.json：保存用于建立索引的知识库信息

index_file = Path(__file__).with_name("faiss.index")
documents_file = Path(__file__).with_name("documents.json")


# 创建本次知识库的描述信息
# 后面需要与已经保存的信息进行比较
#
# 如果知识库内容或Embedding模型名称变化，
# 就说明原来的FAISS索引可能不再适用

current_metadata = {
    "model_name": model_name,
    "normalize_embeddings": True,
    "documents": documents
}


# 标记是否成功加载了旧索引
# False：没有使用旧索引，需要重新创建
# True：成功加载旧索引，不需要重新计算文档向量

index_loaded = False


# =================================
# 第4步：尝试读取已有索引
# =================================

# 只有FAISS索引文件与JSON文件都存在，
# 才尝试使用以前保存的数据

if index_file.exists() and documents_file.exists():

    print("发现本地FAISS索引文件...")

    # 打开之前保存的JSON文件
    # "r"表示读取模式
    # encoding="utf-8"表示使用UTF-8编码读取中文

    with documents_file.open("r", encoding="utf-8") as f:

        # json.load()将JSON文件内容转换为Python对象
        # 此处读取结果为字典
        saved_metadata = json.load(f)

    # 比较以前保存的信息和当前知识库的信息
    if saved_metadata == current_metadata:

        # 知识库没有变化，读取已有FAISS索引
        index = faiss.read_index(str(index_file))

        # 检查索引的维度和向量数量是否正常
        if index.d == 512 and index.ntotal == len(documents):

            index_loaded = True

            print("知识库没有变化，成功加载已有FAISS索引！")

    else:
        print("检测到知识库或模型配置变化，需要重新创建索引")


# =================================
# 第5步：如果没有可用索引，就重新创建
# =================================

if not index_loaded:

    print("正在计算知识库的Embedding向量...")

    # encode()将知识库中的每条文本转换为512维向量
    # normalize_embeddings=True：将向量进行L2归一化

    doc_vectors = embedding_model.encode(
        documents,
        normalize_embeddings=True
    )

    # 转换为FAISS需要的float32连续数组
    doc_vectors = np.ascontiguousarray(
        doc_vectors,
        dtype=np.float32
    )

    # 从向量矩阵中获取维度
    # 例如shape为(4,512)，则维度为512
    dimension = doc_vectors.shape[1]

    # 创建支持内积检索的精确FAISS索引
    index = faiss.IndexFlatIP(dimension)

    # 将所有文档向量添加到索引
    index.add(doc_vectors)

    # 将FAISS索引保存到磁盘
    faiss.write_index(
        index,
        str(index_file)
    )

    # 将当前知识库描述信息保存成JSON文件
    #
    # "w"：写入模式，原文件存在时会覆盖
    # ensure_ascii=False：中文直接保存为中文字符
    # indent=2：格式化JSON内容，方便查看

    with documents_file.open("w", encoding="utf-8") as f:

        json.dump(
            current_metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("新的FAISS索引已经保存！")


# =================================
# 第6步：查看最终索引信息
# =================================

print("FAISS向量维度：", index.d)
print("FAISS向量数量：", index.ntotal)


# 拿到deepseek api key
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("未找到DeepSeek API Key，请检查.env文件")

# 创建DeepSeek客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)
# 7.定义RAG问答函数
def rag_answer(question):
    question = question.strip()
    if not question:
        return ""
    # 7.1问题向量化
    # 将用户问题转换为Embedding向量
    # [question]：把一个字符串放进列表
    #
    # 结果形状为(1, 512)，表示1个问题、512维向量
    # 归一化要和知识库使用相同的设置
    question_vector = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )
    # 转换成FAISS需要的连续float32数组
    question_vector = np.ascontiguousarray(
        question_vector,
        dtype=np.float32
    )
    # 7.2 FAISS相似度检索
    # search()：在FAISS索引中查找最相似的向量
    #
    # 第一个参数：question_vector，用户问题向量
    # 第二个参数：2，表示返回相似度最高的2条文档
    #
    # scores：检索结果的相似度分数
    # indices：检索结果在FAISS索引中的编号
    scores,indices = index.search(question_vector, 3)
    # 7.3获取检索到的中文文档
    # 创建一个空列表，用于保存检索结果的文本
    retrieved_docs = []
    print("\nFAISS检索结果：")
    # scores[0]：取第一个问题的相似度结果
    # indices[0]：取第一个问题对应的文档编号
    #
    # zip()：将两个数组中的元素一一配对
    # 例如(0.85, 2)、(0.72, 0)

    for score, idx in zip(scores[0], indices[0]):
        # 将FAISS返回的编号用于访问原始知识库
        document = documents[int(idx)]

        # 将检索到的文档加入列表
        retrieved_docs.append(document)

        print(f"相似度：{score:.4f}")
        print("文档内容：", document)
    # 7.4拼接参考资料
    context = "\n".join(retrieved_docs)
    # 7.5调用deepseek
    print("\n正在调用deepseek")
    # 将参考资料和用户问题一起发给大语言模型
    # system消息：规定AI助手的回答原则
    # user消息：提供检索结果和当前用户问题

    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是景区智能讲解助手。"
                    "只能根据提供的参考资料回答问题。"
                    "如果参考资料不足，请说明无法确定，"
                    "不要编造信息。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"参考资料：\n{context}\n\n"
                    f"用户问题：{question}"
                )
            }
        ]
    )
 # ---------- 7.6 提取并返回回答 ----------

    # choices[0]：获取模型返回的第一个候选回答
    # message.content：获取回答的文字内容

    answer = response.choices[0].message.content

    # 如果有内容，去掉首尾空白并返回
    # 如果没有内容，返回空字符串
    return answer.replace("*","").strip() if answer else ""


# ==========================
# 第8步：单独测试RAG系统
# ==========================

if __name__ == "__main__":

    print("\n===== FAISS + DeepSeek 智能问答 =====")

    # 程序持续运行，可以连续提问
    # Embedding模型和FAISS索引不会在每次提问时重新创建

    while True:

        question = input("\n请输入问题（exit退出）：").strip()

        if question.lower() == "exit":
            print("程序结束")
            break

        if not question:
            print("问题不能为空")
            continue

        answer = rag_answer(question)

        print("\nDeepSeek回答：", answer)


