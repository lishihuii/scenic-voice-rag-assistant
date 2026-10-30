import numpy as np # 数据处理，计算相似度和排序
import torch # PyTorch深度学习框架,用于检测GPU
from sentence_transformers import SentenceTransformer # 加载文本Embedding处理

# 1.准备模拟景区知识
documents = [
     "游客服务中心位于景区南门，提供咨询和导览服务。",
     "景区每天早上八点开放，下午六点关闭。",
     "景区餐饮区位于中心广场东侧。",
     "游客可以在北门乘坐观光车。"
]
# 2.用户提出问题
question = "请问哪里可以找工作人员咨询？"

# 3.自动选择GPU或CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("当前设备：",device)
# 4.加载中文Embedding模型
print("正在加载Embedding模型")
model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5",
    device=device
)
# 5.将知识库转换成向量
doc_vectors = model.encode(
    documents,
    normalize_embeddings=True
)
# 6.将用户问题转换成向量
query_vectors = model.encode(
    [question],
    normalize_embeddings=True
)
# 7.查看向量形状
print("知识库向量形状：",doc_vectors.shape)
print("问题向量形状：",query_vectors.shape)
# 8.计算余弦相似度
# 因为前面已经归一化，点积等于余弦相似度
similarities = np.dot(
    doc_vectors,
    query_vectors[0]
)
# 9.输出每条资料的相似度
for i,similarity in enumerate(similarities):
    print(f"文档{i + 1} 相似度：{similarity:.4f}")
    print("文档内容：",documents[i])

# 10.找到最相似的文档
best_index = np.argmax(similarities)
print("\n最相关的资料：")
print(documents[best_index])