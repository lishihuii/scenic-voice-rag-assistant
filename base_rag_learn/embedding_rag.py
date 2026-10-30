import os
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer



# 1. 准备知识库
documents = [
    "游客服务中心位于景区南门，提供咨询和导览服务。",
    "景区每天早上八点开放，下午六点关闭。",
    "景区餐饮区位于中心广场东侧。",
    "游客可以在北门乘坐观光车。"
]
# 2. 加载Embedding模型
print("正在加载Embedding模型...")

model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5",
    device="cpu"
)

# 3. 将知识库转换为向量
doc_vectors = model.encode(
    documents,
    normalize_embeddings=True
)
# 4. 创建DeepSeek客户端
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("没有找到DEEPSEEK_API_KEY环境变量")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)
# 5.定义RAG问答函数
def rag_answer(question):
    # 第一步：将用户问题转换为Embedding向量
    question_vector = model.encode(
        [question],
        normalize_embeddings=True
    )
    # 第二步：计算用户问题和知识库的相似度
    similarities = np.dot(doc_vectors, question_vector[0])
    # 第三步：找到相似度最高的两条文档
    top_indices = np.argsort(similarities)[::-1][:2]
    # 第四步：输出检索结果
    print("\n检索到的知识：")
    for i in top_indices:
        print(
            f"相似度：{similarities[i]:.4f}",
            documents[i]
        )
    # 第五步：将检索到的知识拼接起来
    context = "\n".join(
        documents[i] for i in top_indices
    )
    #第六步：调用DeepSeek回答问题
    response = client.chat.completions.create(
        model = "deepseek-flash",
        messages = [
            {
                "role" : "system",
                "content" : (
                    "你是景区智能讲解助手。"
                    "请仅根据提供的参考资料回答问题。"
                    "如果资料不足以回答，请明确说明资料不足，"
                    "不要编造信息。"
                )
            },
            {
                "role" : "user",
                "content" : (
                    f"参考资料：\n{context}\n\n"
                    f"用户问题：{question}"
                )
            }
        ]
    )
    # 第七步：提取deepseek的回答
    answer = response.choices[0].message.content
    # 第八步：返回回答
    return answer.replace("*","").strip() if answer else ""
# 6.测试程序

if __name__ == "__main__":
    question = input("请输入你的问题：")
    answer = rag_answer(question)
    print("\nDeepseek回答：",answer)
