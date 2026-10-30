from pathlib import Path  # 导入路径工具，用于定位和读取本地TXT文件
from langchain_text_splitters import RecursiveCharacterTextSplitter #导入文本切分器，用于将长文档切分成较小文档
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
import faiss #导入FAISS 用于建立索引和执行相似度搜索
import numpy as np
from langchain_core.prompts import ChatPromptTemplate # 导入聊天提示词模板，用于组织回答规则，参考资料和用户问题
import os
from langchain_deepseek import ChatDeepSeek # 导入deepseek聊天模块，用于发送消息并接收问答
from langchain_core.output_parsers import StrOutputParser # 导入字符串输出解析器，将模型回答消息转为普通文字
# 向上两级找到rag目录再拼接data目录及景区知识文件名
txt_path = (
    Path(__file__).parent.parent / "data" /"scenic_guide.txt"
)
# 从硬盘读取文件，将整篇内容保存为字符串
loader = TextLoader(
    file_path=txt_path,
    encoding="utf-8-sig",
)
# 将读取的正文和来源文件名，装进一个Document对象
documents = loader.load()
print("文档数量",len(documents))
document = documents[0]
# 检查正文是否成功读入
print("正文字符数",len(document.page_content))
# 预览前100个字符
print("正文预览",document.page_content[:100])
# 检查文档保存的信息来源,
print("来源",document.metadata)

# 创建切分器，规定文本块大小、目标重叠量及分隔符优先顺序
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    length_function=len,
    separators=["\n\n","\n","。","；","，"," ",""]
)
# 切分Document列表，返回新的Document列表，并保留原文档的元数据
chunks = splitter.split_documents(documents)
# 对比切分前后的文档数量
print("切分前的文档数量：", len(documents))
print("切分后的文本块数量：", len(chunks))
# 只查看前三个文本块，避免一次打印全部内容
for number, chunk in enumerate(chunks[:3], start=1):
    # number是显示用的序号，chunk是当前遍历到的Document对象
    print("\n第", number, "个文本块")
    print("字符数：", len(chunk.page_content))
    print("正文：", chunk.page_content)
    print("元数据：", chunk.metadata)
# 创建向量化组件，复用之前下载的BGE中文模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={
        "device": "cpu", # 使用CPU运行
        "local_files_only": True # 只使用本地缓存，不在线下载模型
    },
    encode_kwargs={
        "normalize_embeddings": True # 对生成的向量进行L2归一化
    }
)
# 从每个Document取出正文，组成字符串列表
texts = [chunk.page_content for chunk in chunks]
# 将所有文本块的正文批量转为向量
vectors = embeddings.embed_documents(texts)
# 返回结果是嵌套列表：外层对应文本块，内层对应一条向量
print("参与向量化的文本块数量：", len(texts))
print("生成的向量数量：", len(vectors))
print("每条向量的维数：", len(vectors[0]))

# 查看第一条向量的前五个数值，避免打印全部内容
print("第一条向量的数值：", vectors[0])

# 将嵌套列表转为float32连续数组，每一行都是一条文本块向量
doc_vectors = np.ascontiguousarray(vectors, dtype=np.float32)
# 从数组的第二个维度取得实际向量维数
dimension = doc_vectors.shape[1]

# 创建内积检索索引，并按当前顺序加入所有文档向量
index = faiss.IndexFlatIP(dimension)
index.add(doc_vectors)
print("索引中向量总数",index.ntotal)
question = "景区明年门票是否涨价？"
query_embedding = embeddings.embed_query(question)
# 包装成一行的二维数组，满足FAISS的批量查询格式
query_vector = np.ascontiguousarray(
    [query_embedding],
    dtype=np.float32
)
top_k = min(3, index.ntotal)
scores,indices = index.search(query_vector, top_k)
for score,chunk_id in zip(scores[0],indices[0]):
    # 将numpy整数转为普通整数
    chunk_id = int(chunk_id)
    # 跳过FAISS没有找到有效向量时返回的编号
    if chunk_id == -1:
        continue
    # 根据向量编号取回对应的Document对象
    matched_document = chunks[chunk_id]
    print("\n文本块编号：", chunk_id)
    print("相似度：", float(score))
    print("正文：", matched_document.page_content)
    print("来源：", matched_document.metadata["source"])

# 根据检索编号，取得本次问题对应的Document列表
retrieved_docs = [
    chunks[int(idx)] for idx in indices[0] if idx != -1
]
# 将检索到的正文拼写成一份参考资料
context = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 创建聊天提示词模板，分别设置系统规则和用户消息
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是景区问答助手。"
        "请根据参考资料回答游客问题。"
        "资料没有提供明确答案时，请说明无法确定。"
        "不要编造位置、价格或开放时间。"
        "回答尽量简短。"
    ),
    (
        "human",
        "参考资料：\n{context}\n\n游客问题：{question}"
    )
])
# 将实际资料和问题填入模板，得到完整提示消息
prompt_value = prompt_template.invoke({
    "context": context,
    "question": question
})
# 转换成便于阅读的文字，检查将来准备发给模型的内容
print("\n完整提示词：")
print(prompt_value.to_string())

# 从当前运行环境读取key
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError(
        "请在document_test.py的运行配置中设置DEEPSEEK_API_KEY"
    )
# 创建deepseek调用对象
llm = ChatDeepSeek(
    model="deepseek-flash",
    api_key=api_key,
    timeout=60,
    max_retries=1,
    extra_body={
    "thinking": {"type": "disabled"}
    }
)
# # 发送已经包含规则，检索资料和游客问题的完整提示消息
# print("\n正在请求DeepSeek回答。。。")
# response = llm.invoke(prompt_value)
# # 从返回的回答消息中取出文字内容
# answer = response.content
# print("\n助手回答")
# print(answer)


# 将三个组件连接起来，此时只是创建执行流程
chain = prompt_template | llm | StrOutputParser()

# 执行整条链：填充提示词、请求模型、提取回答文字
print("\n正在请求DeepSeek回答……")
answer = chain.invoke({
    "context": context,
    "question": question
})
# 统一转换为普通Python字符串，保留原来的回答文字
answer = str(answer)

# answer已经是字符串，可以直接显示
print("\n助手回答：")
print(answer)

# 检查链最终返回的数据类型
print("回答的数据类型：", type(answer))