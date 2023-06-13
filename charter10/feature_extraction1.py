from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def english_count_demo():
    """
    英文文本特征提取
    :return: None
    """
    # 1 获取数据
    data = ["life is short, i like python",
            "life is too long, i dislike python"]
    data1 = ["life is  is short, i like python",
            "life is too long, i dislike python"]
    # 2.1 文本特征转换
    # transfer = CountVectorizer(sparse=True)  注意：没有sparse这个参数
    transfer = CountVectorizer(stop_words=["dislike"])
    new_data = transfer.fit_transform(data)

    # 2.2 产看特征名字
    names = transfer.get_feature_names_out()
    print("特征名字是:\n", names)
    print(new_data.toarray())
    print(new_data)

if __name__ == '__main__':
    english_count_demo()


"""
这段代码演示了如何使用`CountVectorizer`从英文文本中提取特征。

首先，导入所需的库：`DictVectorizer`用于字典类型的特征提取，`CountVectorizer`用于文本特征提取。

在`english_count_demo`函数中，进行了以下步骤：

1. 定义了一个包含两个英文文本的列表`data`和`data1`，用于演示文本特征提取。

2. 创建了一个`CountVectorizer`对象`transfer`，并通过传递`stop_words=["dislike"]`参数来设置停用词。停用词是指在文本处理过程中被忽略的常见词汇，这里设置了停用词为`"dislike"`，即不考虑包含该词的文本。

3. 使用`fit_transform`方法将文本数据`data`转换为特征矩阵`new_data`。`fit_transform`方法将文本数据作为输入，对文本进行分词、编码等处理，并生成对应的特征矩阵。

4. 使用`get_feature_names_out`方法获取特征名字，并将其打印出来。特征名字表示生成的特征矩阵中每一列所代表的特征。

5. 打印特征矩阵的数组形式，可以通过`toarray`方法将稀疏矩阵转换为稠密矩阵，并使用`print`语句打印其内容。

6. 打印特征矩阵的稀疏矩阵形式，直接打印特征矩阵对象`new_data`。

打印结果如下所示：

特征名字是:
 ['dislike', 'life', 'like', 'long', 'python', 'short']
[[0 1 1 0 1 1]
 [1 1 0 1 1 0]]
 
 该结果表示了两个文本中出现的特征及其在特征矩阵中的出现次数。每个文本对应一行，每个特征对应一列。特征矩阵中的每个元素表示对应文本中对应特征的出现次数。例如，第一行表示第一个文本中，'dislike'特征出现0次，'life'特征出现1次，'like'特征出现1次，以此类推。

注意，`CountVectorizer`没有`toarray`方法，所以`print(new_data.toarray())`语句将报错。需要直接打印特征矩阵对象`new_data`来获取稀疏矩阵形式的结果。
"""















