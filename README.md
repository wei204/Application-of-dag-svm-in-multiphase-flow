* 1、数据集格式如train文件夹下train.txt中内容，前面15个数据为每个径向面采集的电容数据，后面1个数据为流型类别。
   若采用的电容传感器不同，采集的数据个数也不同，此时需要更改feature_extract.py文件中特征提取的方式。
* 2、data_preprocess.py文件实现对数据集中数据的读取，并将其转换为可以直接输入到算法的数据格式。
   并对数据进行归一化处理。
* 3、如果数据集的格式已经按照train.txt更改完成，则不需要用到dataSet_normal.py文件。
   该文件包含一些函数用于实现将从传感器采集的数据格式转换为数据集格式。
* 4、feature_extract.py文件包含特征提取函数。
* 5、visualization.py文件包含一些可视化函数，用于观察归一化、特征提取前后数据的差异。
* 6、SMO.py文件通过SMO算法实现对svm的求解，实现svm二分类器。
* 7、svm_multiClass.py文件实现svm多分类器的构造，包含1v1模式的多分类器和DAG模式的多分类器。
   并对DAG模式基于类间不可分离度进行改进，目的是寻找到最优的类别列表顺序，使得上层决策节点出现分类错误的概率降低，
   从而减小累积误差。对于N分类问题，该文件实现多分类器的两种方案都需要训练Nx(N-1)/2个二分类器，
   两种构建方法在训练阶段相同，只是在测试阶段不同。将训练后的模型参数保存在svm.txt文件中。
* 8、train.py文件实现对模型的训练，并输出最优的类别列表顺序，该类别列表用于预测阶段。
* 9、predict.py文件实现对测试样本的预测，并输出正确率。
* 该项目文件的使用方法：先运行train.py文件夹，得到模型训练后的参数和类别列表，然后运行predict.py文件进行预测。
** 例子
```python
def main():
    train_dir = r'train/train.txt'  # 训练集路径
    test_dir = r'test/test.txt'  # 测试集路径
    model_dir = r'svm.txt'       # 模型保存路径
    classList = [1, 2, 3]       # 根据数据集指定类别标签列表classList
    train(train_dir, model_dir, classList, 50, 90, 10000, 'rbf', 20)
    predict(model_dir, test_dir, [1, 3, 2])   # [1,3,2]为最优类别列表顺序
```

