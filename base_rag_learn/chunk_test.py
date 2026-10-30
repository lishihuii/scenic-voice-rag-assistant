from pathlib import Path
# 1.定位txt文件
# __file__表示当前chunk_test.py的文件路径
# .resolve()获取文件的绝对路径
# .parent获取当前Python文件所在的文件夹，即rag目录
# / "data" / "scenic_guide.txt"拼接知识库文件的路径
txt_file = Path(__file__).resolve().parent/"data"/"scenic_guide.txt"
# 2.读取txt文件
if not txt_file.exists():
    raise FileNotFoundError(f"文件不存在：{txt_file}")
# read_text()：读取整个TXT文件，返回一个字符串
# encoding="utf-8"：指定使用UTF-8编码读取中文
text = txt_file.read_text(encoding="utf-8")
# strip()：去除整篇文本开头和结尾的空白字符
# 例如多余的空格、换行符
text = text.strip()
print("TXT文件读取成功！")
print("文本总字符数：", len(text))
# 3.定义文本切分函数
def split_text(text, chunk_size=100, overlap=20):
    # text：需要切分的完整文本
    # chunk_size：每个文本块最多包含多少个字符
    # overlap：相邻两个文本块重叠多少个字符

    # 防止参数不合理
    # 如果overlap >= chunk_size，切分位置就无法正常向前推进
    if chunk_size <=0 or not 0 <= overlap < chunk_size:
        raise ValueError("要求chunk_size大于0，并且0 <= overlap < chunk_size")
    # 创建空列表，保存切分之后的所有文本块
    chunks = []
    # start代表当前文本块的起始字符位置
    # Python字符串索引从0开始
    start = 0
    # 只要还没有处理到文本末尾，就继续切分
    while start < len(text):
        # 计算当前文本块的结束位置
        # min()选择两个值中较小的一个
        #
        # 例如：
        # start = 0，chunk_size = 100
        # 那么end = 100
        #
        # 如果剩余文本不足100个字符，
        # end就取整个文本的实际长度
        end = min(start + chunk_size, len(text))
        # text[start:end]使用字符串切片
        # 提取从start到end之前的字符
        #
        # 例如text[0:100]取前100个字符
        # strip()去掉文本块首尾多余空白

        chunk = text[start:end].strip()
        # 如果文本块不是空字符串，就加入列表
        if chunk:
            chunks.append(chunk)

        # 如果已经切到全文末尾，就结束循环
        if end == len(text):
            break
            # 计算下一个文本块的起始位置
            #
            # 例如：
            # 当前文本块范围是0~100
            # overlap = 20
            #
            # 那么下一个文本块从80开始
            # 即相邻两个文本块有20个字符重叠

        start = end - overlap

        # 返回包含所有文本块的列表
    return chunks
# 4.调用切分函数
documents = split_text(
    text,
    chunk_size = 200,
    overlap = 30
)
# 5.输出切分结果
print("\n 文本切分完成！")
print("总文本块数量：",len(documents))
# enumerate()：遍历列表时，同时获得索引和元素
# start=1表示编号从1开始，而不是从0开始

for i, chunk in enumerate(documents, start=1):

    print(f"\n========== Chunk {i} ==========")

    # 打印当前文本块的字符数量
    print("字符数：", len(chunk))

    # 打印当前文本块的实际内容
    print(chunk)