from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import  CountVectorizer
import jieba

def cut_word(text):
    """
    中文分词
    :return:
    """
    # ret = list(jieba.cut(text))
    # print(ret)
    return " ".join(list(jieba.cut(text)))

def chinese_count_demo():
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
    transfer = CountVectorizer() # 此处可以填写stop_words参数
    new_data = transfer.fit_transform(list)

    # 3.2 查看特征名字
    names = transfer.get_feature_names_out()
    print("特征名字是:\n", names)
    print(new_data.toarray())
    print(new_data)

if __name__ == '__main__':
    chinese_count_demo()