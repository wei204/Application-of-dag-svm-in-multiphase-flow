from Multiphase_flow.data_preprocess import load_data,normalize_data,normalize_data_nonlinear,convert_to_one_hot
from Multiphase_flow.feature_extract import feature_extract, feature_extract_s
from Multiphase_flow.visualization import draw_feature,draw_origin,draw_similar,draw_different,draw_bar,draw_bar_feature,draw_feature_name
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from svm_multiClass import LibSVM





if __name__ == '__main__':
    # 读取数据
    samples, labels = load_data(r'train/train.txt')
    n = len(labels)  # 样本数目68  包括空管流和满管流
    # 绘制原始数据与不同流型的关系
    # draw_origin(samples)
    # draw_similar(samples)
    # draw_different(samples)
    # draw_bar(samples)
    # 将每组数据线性归一化
    # normal_data = normalize_data(samples)
    # draw_different(normal_data)
    # 将每组数据非线性归一化
    normal_data = normalize_data_nonlinear(samples)
    # draw_different(normal_data)
    # draw_similar(normal_data)
    # draw_bar(normal_data)
    # 提取特征 暂时不对空管流和满管流进行特征提取
    # 66个样本 每个样本9个特征 features:66*9
    features = feature_extract(normal_data[2:],n-2)
    # features = feature_extract_s(normal_data, n)
    # draw_feature(features)
    # draw_bar_feature(features)
    # for i in range(6,8):
    #     draw_feature_name(features, i)

   # ##------------------------------调用模型(原始数据效果好)----------------------------------------
   #  # 使用原始数据acc=0.9，数据处理后acc=0.3
   #  clf = make_pipeline(StandardScaler(),SVC(gamma='auto'))
   #  clf.fit(samples,labels)
   #  test_samples, test_labels = load_data(r'test/test.txt')
   #  # test_features = feature_extract(test_samples,len(test_samples))
   #  for i in range(len(test_samples)):
   #      x1 = np.array([test_samples[i]])
   #      print(clf.predict(x1))



   # # # ------------------------------------调用libsvm模型-------------------------------------------
   #  # 原始数据acc=0.3   处理后acc=0.4   DDAG原始数据acc=0.4  处理后acc=0.3
   # #  训练阶段  samples[2:]  features   惩罚因子越大说明对于离群点越重视，松弛变量越大说明容错性越高
   #  svm = LibSVM(features, labels[2:], 50, 60, 10000, name='rbf', theta=20)  # 50, 60,rbf
   #  # svm.train()
   #  # svm.save("svm.txt")
   #  # 类别标签列表 根据数据集的类别自行更改
   #  classList = [1, 2, 3]
   #  Num = len(classList)
   #  # 得到不可分离度列表
   #  Slist = svm.cal_indivisibility(classList)
   #  # print('----------------------不可分离度列表-------------------')
   #  # print(Slist)
   #  # 得到按照不可分离度划分的类别顺序列表
   #  classList_optimization = svm.creat_classList(Slist, Num)
   #  print("---------最优顺序类别列表--------")
   #  print(classList_optimization)   # 132



    # # # 预测阶段
    # svm = LibSVM.load("svm.txt")
    #
    # # 读取测试集样本
    # test, testlabel = load_data(r'test/test.txt')
    # # 后续优化 将测试标签由123->012
    # # testlabel[:] = testlabel[:]
    # # # 将测试数据转换为特征提取后的数据形式
    # test = feature_extract(test, len(testlabel))
    # # 采用1v1模式预测
    # # svm.predict(test, testlabel)
    # # 采用DAG-SVM预测
    # svm.DDAG_predict(test, testlabel, [1,3,2])



