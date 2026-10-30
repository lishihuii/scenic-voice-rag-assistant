import faiss # FAISS向量检索库，用于创建索引和搜索相似向量
import numpy as np
from sentence_transformers import SentenceTransformer # 加载BGE文本嵌入模型
from sympy.physics.units import dimensions

# 1.准备知识库
documents = [
    "游客服务中心位于景区南门，提供咨询和导览服务。",
    "景区每天早上八点开放，下午六点关闭。",
    "景区餐饮区位于中心广场东侧。",
    "游客可以在北门乘坐观光车。",
    "景区停车场位于南门外侧，提供游客停车服务。"
]
# 2.加载Embedding模型
print("正在加载Embedding模型...")

model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5",
    device="cpu"
)

# 3.将知识库转为向量
doc_vectors = model.encode(
    documents,
    normalize_embeddings=True
)
# FAISS使用float32类型的向量
doc_vectors = np.ascontiguousarray(
    doc_vectors,
    dtype=np.float32
)
# 4.创建FAISS索引
dimension = doc_vectors.shape[1]
index = faiss.IndexFlatIP(dimension)
# 5.将知识库向量添加到索引
index.add(doc_vectors)
print("向量的维度：",dimension)
print("索引中的向量数：",index.ntotal)

# 6. 循环接收用户问题
while True:

    question = input("\n请输入问题（exit退出）：").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    # 7. 将用户问题转换为向量
    question_vector = model.encode(
        [question],
        normalize_embeddings=True
    )

    question_vector = np.ascontiguousarray(
        question_vector,
        dtype=np.float32
    )

    # 8. 使用FAISS检索最相关的两条文档
    scores, indices = index.search(
        question_vector,
        2
    )

    # 9. 输出检索结果
    print("\n检索结果：")

    for score, idx in zip(scores[0], indices[0]):
        print(f"相似度：{score:.4f}")
        print("文档：", documents[idx])
