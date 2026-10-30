import faiss
import numpy as np
from pathlib import Path
# 1. 创建3个模拟的向量
# 每个向量包含4个数字，因此向量维度为4
# 注意：这里只是为了练习FAISS，并不是真正的文本Embedding

vectors = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ],
    dtype=np.float32
)

# 2. 指定索引文件保存位置
# __file__：当前Python文件的位置
# with_name()：在当前文件所在目录中指定一个新文件名
#
# 例如：
# faiss_save_test.py 与 test.index 都位于rag文件夹

index_file = Path(__file__).with_name("test.index")

# 3.判断索引文件是否存在
if index_file.exists():
    # 已有索引文件，直接从磁盘读取
    # read_index()：读取之前保存的FAISS索引
    # str()：将Path对象转换为文件路径字符串

    print("发现已有索引，正在读取...")

    index = faiss.read_index(str(index_file))
else:
    # 没有索引文件，需要第一次创建
    print("没有发现索引，正在创建...")

    # 创建4维的FAISS内积索引
    index = faiss.IndexFlatIP(4)

    # 将3个测试向量添加到索引
    index.add(vectors)

    # 将内存中的FAISS索引保存到磁盘
    # 第一个参数：需要保存的索引对象
    # 第二个参数：保存文件的位置

    faiss.write_index(
        index,
        str(index_file)
    )

    print("索引已保存！")
# 4. 查看索引信息
print("向量维度：", index.d)
print("向量数量：", index.ntotal)


# 5. 创建一个用于查询的向量
# 必须是二维float32数组
# 形状为(1, 4)，表示1个查询向量、每个向量4维
question_vector = np.array(
    [[0.9, 0.1, 0.0, 0.0]],
    dtype=np.float32
)
# 6. 查询相似度最高的2个向量
scores, indices = index.search(
    question_vector,
    2
)

print("相似度分数：", scores)
print("对应索引编号：", indices)