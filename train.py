from Multiphase_flow.data_preprocess import *
from Multiphase_flow.feature_extract import feature_extract, feature_extract_s
from svm_multiClass import LibSVM

"""
train_dir：训练集路径
model_dir: 模型保存路径
classList：类别标签  （可以根据数据集标签自行更改）
C：惩罚因子  惩罚因子越大说明对于离群点越重视，即分类器更加注重正确分类所有样本，从而引起过拟合
toler：松弛变量  代表分类器对错误样本的容忍程度，该值越大越容易将错误样本当做支持向量，从而影响分类效果
maxIter：最大迭代次数
kernalType：核函数类型 包括linear和rbf
theta：当选择径向基核函数（rbf）时，表示尺度因子
"""
def train(train_dir, model_dir, classList, C, toler, maxIter, kernelType, theta):
    # 读取数据
    samples, labels = readTxt(train_dir)
    n = len(labels)  # 样本数目66  包括空管流和满管流
    normal_data = normalize_data_nonlinear(samples)
    features = feature_extract(normal_data, n)
    # features = feature_extract_s(normal_data, n)
    features = z_score_normalization(features)
    svm = LibSVM(features, labels, C, toler, maxIter, name=kernelType, theta=theta)  # 50, 60,10000, rbf, theta=20
    Num = len(classList)
    # 得到不可分离度列表
    Slist = svm.cal_indivisibility(classList)
    print('----------------------不可分离度列表-------------------')
    print(Slist)
    # 得到按照不可分离度划分的类别顺序列表
    classList_optimization = svm.creat_classList(Slist, Num)
    print("---------最优顺序类别列表--------")
    print(classList_optimization)  # 132
    svm.train()
    svm.save(model_dir)





if __name__ == '__main__':
    train_dir = r'train/train.txt'  # 训练集路径
    model_dir = r'svm.txt'       # 模型保存路径

    # # 训练  得到训练好的分类器和最优结点排布列表
    # 根据数据集指定类别标签列表classList
    classList = [1, 2, 3]
    train(train_dir, model_dir, classList, 20, 0.01, 10000, 'rbf', 20)
