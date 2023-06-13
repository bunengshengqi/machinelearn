"""
feature_extraction2.py演示了英文文本特征提取，这里演示中文
"""
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def chinese_count_demo():
    """
    中文文本特征提取
    :return: None
    """
    # 1 获取数据
    data = ["人生苦短，我喜欢python",
            "生活太长久, 我不喜欢python"]
    data1 = ["人生 苦短，我 喜欢 喜欢python",
            "生活 太 长久, 我 不喜欢 python"]
    # 2.1 文本特征转换
    # transfer = CountVectorizer(sparse=True)  注意：没有sparse这个参数
    transfer = CountVectorizer()
    new_data = transfer.fit_transform(data1)

    # 2.2 产看特征名字
    names = transfer.get_feature_names_out()
    print("特征名字是:\n", names)
    print(new_data.toarray())
    print(new_data)

if __name__ == '__main__':
    chinese_count_demo()

"""
总结：使用数据data和data1产生了两种效果，经过分析发现，英文默认是以空格分开的，其实就达到
了一个分词效果。所以要对中文进行分词处理

分词的演示见feature_extraction3.py
"""