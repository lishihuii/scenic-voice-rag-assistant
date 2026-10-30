from email.mime import audio

from sklearn.feature_extraction.text import TfidfVectorizer # TF-IDF文本向量化工具
from sklearn.metrics.pairwise import cosine_similarity # 计算两个向量之间的余弦相似度
import os # 操作系统模块，读取deepseek API key环境变量
from openai import OpenAI
import asyncio # python异步编程模块
from pathlib import Path
import edge_tts
import numpy as np





# 1.准备知识库（以下为模拟材料）
documents = [
    "游客中心位于景区南门入口附近，提供咨询和导览服务。",
    "景区内设有餐饮区，位于中心广场东侧。",
    "游客可以在景区北门乘坐观光车游览。"
    "游客中心位于景区南门右侧，服务时间为09:00至17:30。",
    "景区开放时间为09:00至18:00。",
    "历史文化展馆位于景区东侧，主要介绍当地历史。",
    "景区北门附近设有餐厅，提供简餐和饮品。"
]

# 2.初始化TF-IDF向量化器
vectorizer = TfidfVectorizer(
    analyzer = "char",
    ngram_range = (1,2),
)

# 将知识库文档转为向量
doc_vectors = vectorizer.fit_transform(documents)

# 3.初始化deepseek客户端
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY not set")
client = OpenAI(
    api_key = api_key,
    base_url = "https://api.deepseek.com"
)

# 4.定义RAG问答函数
def rag_answer(question):
    # 第一步：将用户问题转换为TF-IDF向量
    question_vector = vectorizer.transform([question])
    # 第二步：计算问题与知识库文档的相似度
    similarity = cosine_similarity(
        question_vector,
        doc_vectors
    )[0]
    # 第三步：选择相似度最高的两条文档
    top_indices = np.argsort(similarity)[::-1][:2]

    # 第四步：拼接检索到的相关资料
    context = "\n".join(
        documents[i] for i in top_indices
    )
    print("检索到的资料：")
    print(context)

    # 第五步：调用deepseek生成回答
    response = client.chat.completions.create(
        model = "deepseek-flash",
        messages = [
            {
                "role": "system",
                "content": "你是景区智能讲解助手。只根据提供的资料回答问题，资料不足时请明确说明，不要编造"
            },
            {
                "role": "user",
                "content": f"参考资料：{context}\n用户问题：{question}"
            }
        ]
    )
    # 第六步：提取返回大模型的回答
    answer = response.choices[0].message.content
    return answer.replace("**","").strip() if answer else ""
# 5.单独测试RAG函数
if __name__ == "__main__":
    question = input("请输入问题：")
    answer = rag_answer(question)
    print("\nDeepSeek回答：",answer)
