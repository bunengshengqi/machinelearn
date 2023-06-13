from sklearn.feature_extraction import DictVectorizer

def dict_demo():
    """
    字典特征提取
    :return:
    """
    # 1 获取数据
    data = [{'city': '北京', 'temperature': '100'},
            {'city': '上海', 'temperature': '60'},
            {'city': '深圳', 'temperature': '30'}]
    # 2 字典特征提取
    # 2.1 实例化
    transfer = DictVectorizer(sparse=True)
    # transfer = DictVectorizer(sparse=Falae)

    # 2.2 转换
    new_data = transfer.fit_transform(data)
    print(new_data)

    # 2.3 获取具体属性名
    names = transfer.get_feature_names_out()
    print("属性名字是：\n", names)



if __name__ == '__main__':
    dict_demo()

# 总结：
"""
上述讲述了字典特征提取操作流程

对于特征中存在类别信息的我们都会做one-hot编码处理


在`DictVectorizer`中，`sparse`参数用于指定是否创建稀疏矩阵（sparse matrix）。以下是参数`sparse`为`True`和`False`时的区别：

1. `sparse=True`：当`sparse`设置为`True`时，`DictVectorizer`将返回一个稀疏矩阵。稀疏矩阵是一种仅存储非零元素及其位置的数据结构。这种表示方式适用于特征矩阵中大部分元素为零的情况，可以节省内存空间。稀疏矩阵在处理大型数据集时非常有用。

2. `sparse=False`：当`sparse`设置为`False`时，`DictVectorizer`将返回一个密集矩阵（dense matrix）。密集矩阵存储每个元素的具体数值，无论其是否为零。这种表示方式适用于特征矩阵中大部分元素都是非零的情况，或者数据集较小的情况。

选择是否使用稀疏矩阵取决于数据集的稀疏性和内存要求。如果数据集具有较少的非零元素，或者内存有限，那么使用稀疏矩阵可能更合适。如果数据集相对较小且具有大量的非零元素，那么使用密集矩阵可能更方便。

需要注意的是，稀疏矩阵和密集矩阵在一些操作上可能具有不同的行为。因此，在选择`sparse`参数时，需要考虑后续对特征矩阵的处理和使用场景。


打印结果是存在不同！
当使用`DictVectorizer`的`transform`方法进行转换并打印结果时，`sparse=True`和`sparse=False`会产生不同的输出。

1. `sparse=True`：当`DictVectorizer`的`sparse`参数设置为`True`时，转换后的结果将以稀疏矩阵的形式打印。稀疏矩阵通常以一种压缩的格式进行打印，仅显示非零元素的值和位置。

   例如：
   ```
   (0, 1)    1.0
   (0, 3)    2.0
   (1, 0)    3.0
   ```
   上述输出表示稀疏矩阵中的非零元素及其位置。括号中的第一个数字表示行索引，第二个数字表示列索引，而最后一个数字表示非零元素的值。

2. `sparse=False`：当`DictVectorizer`的`sparse`参数设置为`False`时，转换后的结果将以密集矩阵的形式打印。密集矩阵以常规矩阵的形式打印，显示每个元素的具体值。

   例如：
   ```
   [[0.0, 1.0, 0.0, 2.0],
    [3.0, 0.0, 0.0, 0.0]]
   ```
   上述输出表示密集矩阵中的元素值。矩阵的行由外层的方括号表示，每一行的元素由内层的方括号表示。

需要根据具体的需求和数据集的特点选择合适的输出形式。稀疏矩阵的打印结果更紧凑，适合处理大规模稀疏数据，而密集矩阵的打印结果更直观，适合处理较小且密集的数据。

"""