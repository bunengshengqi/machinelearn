"""
如何处理某个词或者短语在多篇文章出现的次数过高？

tf-idf文本特征提取
tf-idf作用：用以评估一个词对于一个文件集或者一个语料库中的其中一份文件的重要程度

TF-idf（Term Frequency-Inverse Document Frequency）是一种常用的文本特征权重计算方法，用于衡量一个词语在文档集中的重要性。

TF（词频）衡量了一个词语在单个文档中的出现频率，而IDF（逆文档频率）衡量了一个词语在整个文档集中的普遍重要性。

TF-idf的计算公式如下：

TF-idf = TF * IDF

其中，TF（词频）可以通过以下公式计算：

TF = (词语在文档中出现的次数) / (文档中的总词语数)

IDF（逆文档频率）可以通过以下公式计算：

IDF = log((文档集中的文档总数) / (包含该词语的文档数 + 1))

上述公式中的 "+1" 是为了避免分母为零的情况，并使用对数函数（通常以自然对数为底）来平衡词语的权重。

TF-idf值越高，表示词语在文档中的重要性越高，常用于文本分类、信息检索和文本相似度计算等任务中。

"""

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba


def cut_word(text):
    """
    中文分词
    :return:
    """
    # ret = list(jieba.cut(text))
    # print(ret)
    return " ".join(list(jieba.cut(text)))

def tfidf_demo():
    """
    中文文本特征提取
    :return: None
    """
    # 1 获取数据
    data = ["选择使用哪种编程语言进行物联网嵌入式开发"
            "取决于具体项目需求、硬件平台、性能要求和"
            "开发人员的经验。对于特定的嵌入式系统，"
            "可能还需要考虑与硬件驱动程序和库的兼容性"
            "以及资源限制等因素。因此，物联网嵌入式开发"
            "工程师需要根据实际情况选择最适合的开发语言。"]

    # 2 文章分割
    list = []
    for temp in data:
        list.append(cut_word(temp))
    print(list)

    # 3 文本特征转换
    # 3.1 实例化 + 转化
    # transfer = CountVectorizer(sparse=True)  注意：没有sparse这个参数
    # transfer = CountVectorizer() # 此处可以填写stop_words参数
    transfer = TfidfVectorizer()
    new_data = transfer.fit_transform(list)

    # 3.2 查看特征名字
    names = transfer.get_feature_names_out()
    print("特征名字是:\n", names)
    print(new_data.toarray())
    print(new_data)

if __name__ == '__main__':
    tfidf_demo()