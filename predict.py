from svm_multiClass import LibSVM
from Multiphase_flow.data_preprocess import *
from Multiphase_flow.feature_extract import *

'''
model_dir：模型保存路径
test_dir：测试集路径
classList：根据类间不可分离度得到的最优结点排布顺序
'''
def predict(model_dir, test_dir, classList):
    # # 预测阶段
    svm = LibSVM.load(model_dir)
    test, testlabel = readTxt(test_dir)
    test = normalize_data_nonlinear(test)
    # 后续优化 将测试标签由123->012
    # testlabel[:] = testlabel[:]
    # # 将测试数据转换为特征提取后的数据形式
    test = feature_extract(test, len(testlabel))
    test = z_score_normalization(test)
    svm.DDAG_predict(test, testlabel, classList)


if __name__ == '__main__':
    model_dir = r'svm.txt'  # 模型保存路径
    test_dir = r'test/test.txt'  # 测试集路径
    predict(model_dir, test_dir, [2, 1, 3])