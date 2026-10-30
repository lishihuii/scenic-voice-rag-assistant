import os
import faiss
from pathlib import Path
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import json # Python内置JSON模块，用于保存和读取信息
import time

# 1.准备知识库
# __file__表示当前txt_rag.py的文件路径
# resolve()获取绝对路径
# parent获取当前文件所在的rag文件夹
# /运算符负责拼接路径

txt_file = (
    Path(__file__).resolve().parent
    / "data"
    / "scenic_guide.txt"
)

# 检查文件是否存在，避免路径写错
if not txt_file.exists():
    raise FileNotFoundError(f"找不到知识库文件：{txt_file}")

# read_text()：读取TXT中的所有文字，返回str字符串
# utf-8-sig：兼容普通UTF-8及带BOM的UTF-8文件
text = txt_file.read_text(encoding="utf-8-sig")

# 将Windows等系统的不同换行符统一为\n
# 这里只是统一换行格式，不会按自然段切分
text = text.replace("\r\n", "\n").replace("\r", "\n")

# 去除整篇文字最前面和最后面的多余空白
text = text.strip()

print("TXT文件读取成功！")
print("知识库总字符数：", len(text))


# ==========================================
# 第2步：定义固定字符长度切分函数
# ==========================================

def split_text(text, chunk_size=200, overlap=30):

    # text：需要切分的完整文本
    # chunk_size：每块最多200个字符
    # overlap：相邻块重叠30个字符
    #

    # 检查参数是否合法
    # overlap不能大于或等于chunk_size
    # 否则文本切分时可能无法正常前进
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("切分参数不合法")

    # 保存所有文本块的列表
    chunks = []

    # 从原始文本第0个字符开始
    start = 0

    # 当起始位置还没有到达文本末尾时继续循环
    while start < len(text):

        # 计算当前文本块的结束位置
        #
        # 假设start=0，chunk_size=200
        # 那么end=200
        #
        # 如果剩余文本不足200字符
        # 就使用len(text)作为结束位置
        end = min(start + chunk_size, len(text))

        # 从原文中提取当前文本块
        # Python切片包含start，但不包含end
        #
        # 这里不对每个块单独strip()
        # 目的是保留精确的30字符重叠关系
        chunk = text[start:end]

        # 如果当前块不是纯空白，就添加到列表
        if chunk.strip():
            chunks.append(chunk)

        # 如果已经切分到全文末尾，就退出循环
        if end == len(text):
            break

        # 计算下一块的起始位置
        #
        # 例如：
        # 第一块：0~200
        # 下一块：170~370
        #
        # 两块之间重叠30个字符
        #
        # 特别注意：
        # 这一行在while循环内，但在if判断外
        start = end - overlap

    # 只有全部切分结束，才返回文本块列表
    # return位于while循环外
    return chunks


# ==========================================
# 第3步：调用切分函数
# ==========================================

# 把刚才读取的完整TXT字符串交给split_text()
# 明确指定chunk_size=200，overlap=30

documents = split_text(
    text,
    chunk_size=200,
    overlap=30
)

# documents是一个Python列表
# documents[0]是第一个文本块
# documents[1]是第二个文本块
# 以后每个文本块都会生成一个Embedding向量

print("切分完成！")
print("文本块数量：", len(documents))

if not documents:
    raise ValueError("切分后没有有效文本块")

print("第一个文本块字符数：", len(documents[0]))
print("第一个文本块内容：")
print(documents[0])
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
# 保存已经创建的客户端，导入模块时暂不创建
client = None


# 真正需要调用DeepSeek时，才检查密钥并创建客户端
def get_deepseek_client():
    global client  # 允许本函数更新模块中保存的client变量

    # 已经创建过时直接复用，不重复初始化
    if client is None:
        # 从当前运行进程的环境变量中读取密钥
        api_key = os.getenv("DEEPSEEK_API_KEY")

        # 缺少密钥时，明确指出需要配置当前运行入口
        if not api_key:
            raise RuntimeError(
                "未找到DEEPSEEK_API_KEY，请在当前入口的运行配置中设置环境变量"
            )

        # 创建兼容OpenAI接口的DeepSeek客户端
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    # 返回首次创建或之前保存的客户端
    return client



#  9.定义FAISS文档检索函数

def retrieve_documents(question,top_k=3,min_similarity=None):
    # 1. 将用户问题转换为Embedding向量
    # [question]：将一个字符串包装成列表
    # normalize_embeddings=True：对向量进行L2归一化
    #
    # BGE-small模型输出512维向量
    # 这里只输入一个问题，所以形状为(1, 512)
    question_vector = embedding_model.encode(
        [question],
        normalize_embeddings = True
    )
    # 2.转换成FAISS需要的float32连续数组
    question_vectior = np.ascontiguousarray(
        question_vector,
        dtype=np.float32
    )
    # 3.确定实际检索数量
    # 例如：
    # top_k=3，索引只有2条向量
    # 那么只搜索2条，避免请求过多结果
    k = min(top_k, index.ntotal)
    if k <= 0:
        return []
    # 4.使用FAISS进行TOP-K检索
    #
    # scores：二维数组，存放相似度分数
    # indices：二维数组，存放向量对应的索引编号
    #
    # 因为查询的是一个问题，
    # 所以两个数组的形状都是(1, k)
    scores, indices = index.search(
        question_vectior,
        k
    )
    # 5.创建空列表，保存通过过滤的文档
    results = []
    print("\nFAISS检索结果")
    # 6. 遍历检索得到的每条结果
    # zip()将分数和编号一一配对
    for score, idx in zip(scores[0], indices[0]):
        # 将NumPy类型转换成普通Python类型
        score = float(score)
        idx = int(idx)

        # FAISS在没有匹配位置时可能返回-1
        # 这种编号不能用于访问documents
        if idx == -1:
            continue

        # 打印当前文档的相似度，便于观察
        print(f"\n文档编号：{idx}")
        print(f"相似度：{score:.4f}")
        # 7. 判断是否需要进行阈值过滤
        #
        # min_similarity is not None：
        # 表示用户启用了阈值过滤
        #
        # score < min_similarity：
        # 表示当前文档的相似度低于阈值
        #
        # 两个条件都满足，就跳过当前文档

        if min_similarity is not None and score < min_similarity:
            print("低于相似度阈值，已过滤")
            continue

        # 8. 根据FAISS编号查找原始文本
        chunk = documents[idx]

        print("文档内容：", chunk)

        # 9. 将文档和相似度一起保存
        # 每条检索结果是一个字典
        #
        # 例如：
        # {
        #     "score": 0.75,
        #     "text": "景区下午五点停止入园..."
        # }

        results.append({
            "chunk_id": idx,
            "score": score,
            "text": chunk
        })

        # 10. 返回通过过滤的文档列表
    return results
# 定义rag智能问答函数
def rag_answer(question, top_k=3, min_similarity=None, history=None):

    # 1. 去掉用户问题前后的空白
    question = question.strip()

    if not question:
        return "请输入有效问题。"

    # 2. 调用刚才定义的检索函数
    #
    # 检索函数负责：
    # 问题向量化、FAISS搜索、相似度过滤
    #
    # 返回一个列表，每个元素是一个字典

    retrieved_docs = retrieve_documents(
        question,
        top_k=top_k,
        min_similarity=min_similarity
    )

    # 3. 判断是否检索到有效文档
    #
    # 如果所有结果都被过滤，
    # retrieved_docs就是空列表[]
    #
    # 空列表在if判断中相当于False

    if not retrieved_docs:
        return "当前没有检索到足够相关的景区资料，暂时无法回答。"

    # 4. 将检索到的文本块拼接成参考资料
    #
    # enumerate(..., start=1)：
    # 为每个文本块分配从1开始的编号
    #
    # item["text"]：
    # 取出结果字典中的原始文本
    #
    # join()：
    # 将多个文本块连接成一个字符串

    context = "\n\n".join(
        f"资料{i}：\n{item['text']}"
        for i, item in enumerate(retrieved_docs, start=1)
    )

    # 5. 调用DeepSeek
    #
    # system消息负责规定回答规则
    # user消息包含检索到的资料和游客问题
    #
    # 特别要求：
    # 资料相关但没有提供明确答案时，
    # 不能凭空推断
    # 真正需要生成回答时，才获取或创建DeepSeek客户端
    # 系统消息规定助手身份，以及历史对话和知识库的使用方式
    messages = [
        {
            "role": "system",
            "content": (
                "你是智慧景区智能问答助手。"
                "可以根据历史对话理解追问，回顾游客已经说过的内容。"
                "回答景区事实时，必须以本轮参考资料为依据。"
                "历史回答不能作为独立的景区事实依据。"
                "资料没有明确答案时，请说明无法确定。"
                "不能把没有提及的服务当成不存在。"
                "不要编造价格、时间或服务信息。"
                "回答问题尽量简短，50字左右。"
            ),
        }
    ]

    # 将之前的游客问题和助手回答，按原有顺序加入消息列表
    if history:
        messages.extend(history)

    # 最后加入本轮检索资料和当前问题，明确本次需要回答什么
    messages.append({
        "role": "user",
        "content": (
            f"本轮参考资料：\n{context}\n\n"
            f"本次游客问题：{question}"
        ),
    })

    # 获取客户端，将完整的消息列表提交给DeepSeek
    client = get_deepseek_client()
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
    )

    # 6. 提取DeepSeek返回的回答
    answer = response.choices[0].message.content

    # 7. 返回文本
    return answer.replace("*","").strip() if answer else ""
# ==========================
# 第8步：单独测试RAG系统
# ==========================
# 11.定义检索调试函数
def debug_retrieval(question,top_k=3,min_similarity=None):
    # question：用户输入的问题
    # top_k：最多检索多少个Chunk
    # min_similarity：相似度过滤阈值
    #
    # 本函数只负责检索和展示结果
    # 不调用DeepSeek，也不生成AI回答
    print("\n========== RAG检索调试 ==========")
    print("用户问题：", question)
    print("Top-K参数：", top_k)
    print("相似度阈值：", min_similarity)
    # 1.记录检索开始时间
    # perf_counter()返回高精度计时器的读数
    # 它适合用于计算一段程序运行了多少秒
    start_time = time.perf_counter()
    # 2. 调用已经写好的FAISS检索函数
    # 这里会执行：
    # 用户问题Embedding向量化
    # FAISS搜索Top-K
    # 相似度阈值过滤
    #
    # 返回一个列表，每个元素都是一个字典
    results = retrieve_documents(
        question,
        top_k=top_k,
        min_similarity=min_similarity
    )

    # 3. 记录检索结束时间
    end_time = time.perf_counter()
    # 4. 计算检索总耗时
    # 包含问题向量化与FAISS检索等操作
    # 不包含程序启动时的BGE模型加载时间
    elapsed = end_time - start_time

    print("\n========== 检索结果汇总 ==========")
    print("通过过滤的Chunk数量：", len(results))
    print(f"检索耗时：{elapsed:.4f}秒")

    # 5. 如果没有结果，直接结束调试
    if not results:
        print("没有通过过滤的文本块。")
        return results
    # 6.遍历通过过滤的结果
    # enumerate(..., start=1)使显示序号从1开始
    for rank, item in enumerate(results, start=1):
        chunk_id = item["chunk_id"]
        score = item["score"]
        chunk = item["text"]
        # 7. 制作简单的相似度条形图
        #
        # 相似度经过归一化内积计算，一般范围为[-1,1]
        # 为了展示条形图，将负分数按0长度显示
        # 并将绘图值限制在0到1之间
        display_score = max(0.0, min(1.0, score))

        # 将0~1的绘图值转换为0~20的长度
        bar_length = int(display_score * 20)

        # 字符串乘法：
        # "█" * 5得到"█████"
        bar = "█" * bar_length

        # 8. 截取文本前80个字符作为预览
        # replace()将换行符替换成空格
        # [:80]只获取前80个字符
        preview = chunk.replace("\n", " ")[:80]

        print(f"\n第{rank}条保留结果")
        print("Chunk编号：", chunk_id)
        print(f"相似度：{score:.4f}")
        print("相似度图：", bar)
        print("文本预览：", preview)

        # 9. 返回检索结果，方便其他代码继续使用
    return results

# 13.准备RAG检索测试集
test_cases = [
    {
        "question": "景区下午几点停止入园？",
        "answer_phrase": "下午五点停止游客入园"
    },
    {
        "question": "观光车主要经过哪些站点？",
        "answer_phrase": "南门站、中心广场站、湖畔观景区站和北门站"
    },
    {
        "question": "森林步道全程游览需要多久？",
        "answer_phrase": "全程游览约需一个小时"
    },
    {
        "question": "景区哪里可以办理失物招领？",
        "answer_phrase": "失物招领服务"
    }
]
# 14.定义检索效果评价函数
def evaluate_retrieval(test_cases, k = 3):
    # test_cases：用于评价的问题列表
    # k：检查FAISS返回的前k个Chunk
    #
    # 本函数不调用DeepSeek
    # 只评价FAISS是否找到正确文本块
    # 记录成功命中的问题数量
    hit_count = 0

    # 累加每个问题的Recall@K
    recall_sum = 0

    # 记录实际参与评价的问题数量
    valid_count = 0
    print("\n========== RAG检索评价 ==========")
    print("当前K值：", k)

    # 遍历测试数据集中的每个问题
    for case in test_cases:

        # 从字典中取出问题
        question = case["question"]

        # 从字典中取出答案关键文字
        answer_phrase = case["answer_phrase"]

        print("\n------------------------------")
        print("测试问题：", question)

        # 1. 找到包含正确答案关键文字的Chunk
        #
        # enumerate(documents)：
        # 同时遍历文本块编号和文本内容
        #
        # if answer_phrase in chunk：
        # 只保留包含答案关键文字的Chunk编号
        #
        # 结果是一个列表，例如[5, 6]

        gold_ids = [
            i
            for i, chunk in enumerate(documents)
            if answer_phrase in chunk
        ]

        print("参考正确Chunk编号：", gold_ids)

        # 如果找不到答案关键文字，就不能正常评价
        # 可能是原文改变，或者答案短语跨越了切分边界
        #
        # 此时提示人工检查，并跳过该问题
        if not gold_ids:
            print("没有找到答案关键文字，请人工检查测试数据")
            continue

        # 2. 调用上一节写好的FAISS检索函数
        #
        # min_similarity=None：
        # 暂时关闭相似度过滤
        #
        # 这样我们可以先单独评价Top-K检索能力

        results = retrieve_documents(
            question,
            top_k=k,
            min_similarity=None
        )

        # 3. 提取FAISS返回的Chunk编号
        #
        # item["chunk_id"]：
        # 获取每条检索结果中的文本块编号
        #
        # 例如结果可能是[12, 5, 20]

        retrieved_ids = [
            item["chunk_id"]
            for item in results
        ]

        print("FAISS返回Chunk编号：", retrieved_ids)

        # 4. 找出正确资料与检索结果的交集
        #
        # 如果某个正确编号出现在retrieved_ids中，
        # 就将它加入matched_ids
        #
        # 例如：
        # gold_ids = [5, 6]
        # retrieved_ids = [12, 5, 20]
        #
        # 那么matched_ids = [5]

        matched_ids = [
            idx
            for idx in gold_ids
            if idx in retrieved_ids
        ]

        print("成功命中的正确Chunk：", matched_ids)

        # 5. 计算当前问题的Hit@K
        #
        # 如果至少找到一条正确资料：
        # hit = 1
        #
        # 如果一条也没找到：
        # hit = 0

        hit = 1 if matched_ids else 0

        # 6. 计算当前问题的Recall@K
        #
        # 分子：找到了多少条正确Chunk
        # 分母：所有被标注为正确的Chunk数量
        #
        # 例如找到1条，正确资料共2条：
        # recall = 1 / 2 = 0.5

        recall = len(matched_ids) / len(gold_ids)

        # 7. 累计评价结果
        hit_count += hit
        recall_sum += recall
        valid_count += 1

        print(f"Hit@{k}：{hit}")
        print(f"Recall@{k}：{recall:.2%}")

    # 8. 输出整个测试集的平均指标
    #
    # valid_count可能为0，因此先检查
    # 防止出现除以0的错误

    if valid_count == 0:
        print("没有有效测试问题，无法计算平均指标")
        return

    # 计算所有有效问题的平均Hit@K
    avg_hit = hit_count / valid_count

    # 计算所有有效问题的平均Recall@K
    avg_recall = recall_sum / valid_count

    print("\n========== 最终评价结果 ==========")
    print("有效测试问题数量：", valid_count)
    print(f"平均Hit@{k}：{avg_hit:.2%}")
    print(f"平均Recall@{k}：{avg_recall:.2%}")



if __name__ == "__main__":

    print("===== 智慧景区RAG系统 =====")

    # 当前使用的检索参数
    # 后面可以修改，观察不同参数的效果
    top_k = 3
    min_similarity = None

    # 不断循环，直到用户输入exit
    while True:

        print("\n请选择运行模式：")
        print("1 - 只测试FAISS检索")
        print("2 - FAISS检索 + DeepSeek回答")
        print("3 - 自动评价FAISS检索效果")
        print("exit - 退出程序")

        # 获取用户选择，并删除首尾空白
        mode = input("请选择：").strip().lower()

        # 如果输入exit，退出整个while循环
        if mode == "exit":
            print("程序已退出")
            break

        # 检查模式是否合法
        if mode not in ("1", "2","3"):
            print("请选择1、2或exit")
            continue

        # 获取用户问题
        question = input("请输入游客问题：").strip()

        if not question:
            print("问题不能为空")
            continue

        # 模式1：只运行检索调试函数
        # 不会调用DeepSeek生成回答
        if mode == "1":

            debug_retrieval(
                question,
                top_k=top_k,
                min_similarity=min_similarity
            )

        # 模式2：执行完整的RAG问答
        elif mode == "2":

            answer = rag_answer(
                question,
                top_k=top_k,
                min_similarity=min_similarity
            )

            print("\nDeepSeek回答：", answer)
        # 模式3：自动测试所有预设问题
        # 这段判断放在input("请输入游客问题")之前
        # 因为测试问题已经保存在test_cases中，
        # 不需要用户再次手动输入问题
        if mode == "3":
            # 对测试集运行Top-3检索评价
            evaluate_retrieval(
                test_cases,
                k=3
            )

            # continue跳过后面的单问题交互流程
            # 直接返回while循环开始位置
            continue

