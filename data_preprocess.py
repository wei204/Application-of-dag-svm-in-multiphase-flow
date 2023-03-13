import numpy as np


#将类别名转换成标号
def flow_type(s):
    # b'empty': 0, b'full': 1, b'ring': 2, b'core': 3, b'plunger': 4, b'suspended': 5
    it = {b'empty': 0, b'layer': 1, b'ring': 2, b'core': 3, b'full': 4}
    return it[s]

## 原数据格式：n*16 15个电容值+1个类别标签，处理完后samples为n*15，labels为n*1
def load_data(filename):
    data = np.loadtxt(filename, dtype=float, delimiter=',', converters={15: flow_type})  #
    samples = data[:, :15]
    labels = data[:,15]
    return samples,labels


# 数据线性归一化 (某组数据-空管数据)/(满管数据-空管数据)
def normalize_data(samples):
    # empty_data = np.loadtxt(r'H:\COMSOL\project\ECTwzm\kg.txt')
    # full_data = np.loadtxt(r'H:\COMSOL\project\ECTwzm\mgOne.txt')
    # 空管电容与满管电容
    empty_data = samples[0].copy()
    full_data = samples[1].copy()
    for i in range(len(samples)):
        samples[i] = (samples[i]-empty_data) / (full_data-empty_data)
    return samples


# 数据非线性归一化  (最大数据*(某组数据-空管数据))/(某组数据*(最大数据-最小数据))
def normalize_data_nonlinear(samples):
    # 空管电容与满管电容
    empty_data = samples[0].copy()
    full_data = samples[1].copy()
    for i in range(len(samples)):
        samples[i] = (full_data*(samples[i]-empty_data)) / (samples[i]*(full_data-empty_data))
    return samples


# 数据标准化 输入数据X减去均值并除以标准差
# def standardize_data(data):
#     mean = np.mean(data, axis=0)
#     std = np.std(data, axis=0)
#     for i in range(data.shape[0]):
#         data[i, :] = (data[i, :] - mean) / std
#     return data

# 将离散特征映射到欧式空间，使得特征之间的距离计算更合理
def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]


