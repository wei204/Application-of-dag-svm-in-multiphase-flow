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

def readTxt(dir):
    file = open(dir, 'r')
    file_data = file.readlines()
    datas = []
    labels = []
    for line in file_data:
        line = line.split('\n')[0]
        l = line.split(',')[:-1]
        data = []
        for i in l:
            data.append(float(i))
        datas.append(data)
        if(line.split(',')[-1]=='layer'):
            labels.append(1.0)
        elif(line.split(',')[-1]=='ring'):
            labels.append(2.0)
        elif(line.split(',')[-1]=='core'):
            labels.append(3.0)
        elif(line.split(',')[-1]=='empty'):
            labels.append(0.0)
        elif (line.split(',')[-1] == 'full'):
            labels.append(4.0)
    return datas, labels

# 数据线性归一化 (某组数据-空管数据)/(满管数据-空管数据)
def normalize_data(samples):
    # empty_data = np.loadtxt(r'H:\COMSOL\project\ECTwzm\kg.txt')
    # full_data = np.loadtxt(r'H:\COMSOL\project\ECTwzm\mgOne.txt')
    samples = np.array(samples)
    # 空管电容与满管电容
    # empty_data = samples[0].copy()
    # full_data = samples[1].copy()
    empty_data = np.array(
        [7.216911E-13, 5.928700E-13, 6.100104E-13, 6.159350E-13, 7.207132E-13, 7.201968E-13, 5.799969E-13, 5.700280E-13,
         5.801543E-13, 7.190246E-13, 5.785485E-13, 5.736352E-13, 7.254565E-13, 6.109420E-13, 7.788726E-13])
    full_data = np.array(
        [8.201945E-13, 6.489239E-13, 6.370078E-13, 6.466984E-13, 8.343990E-13, 8.250562E-13, 6.492056E-13, 5.963936E-13,
         6.487237E-13, 8.330884E-13, 6.427659E-13, 6.191227E-13, 8.274819E-13, 6.598836E-13, 8.263927E-13])
    for i in range(len(samples)):
        samples[i] = (samples[i]-empty_data) / (full_data-empty_data)
    return samples


# 数据非线性归一化  (最大数据*(某组数据-空管数据))/(某组数据*(最大数据-最小数据))
def normalize_data_nonlinear(samples):
    # 空管电容与满管电容
    # empty_data = np.array(samples[0].copy())
    # full_data = np.array(samples[1].copy())
    empty_data = np.array(
        [7.216911E-13, 5.928700E-13, 6.100104E-13, 6.159350E-13, 7.207132E-13, 7.201968E-13, 5.799969E-13, 5.700280E-13,
         5.801543E-13, 7.190246E-13, 5.785485E-13, 5.736352E-13, 7.254565E-13, 6.109420E-13, 7.788726E-13])
    full_data = np.array(
        [8.201945E-13, 6.489239E-13, 6.370078E-13, 6.466984E-13, 8.343990E-13, 8.250562E-13, 6.492056E-13, 5.963936E-13,
         6.487237E-13, 8.330884E-13, 6.427659E-13, 6.191227E-13, 8.274819E-13, 6.598836E-13, 8.263927E-13])
    for i in range(len(samples)):
        samples[i] = (full_data*(samples[i]-empty_data)) / (samples[i]*(full_data-empty_data))
    return samples


# 特征归一化，使得每个特征尺度范围一致
def z_score_normalization(X):
    """
    对数据进行 Z-score 标准化
    :param X: 待归一化的数据，形状为 (m, n)
    :return: 归一化后的数据，形状与 X 相同
    """
    mu = np.mean(X, axis=0)  # 计算均值
    sigma = np.std(X, axis=0)  # 计算标准差
    X_norm = np.round((X - mu) / sigma, 2)  # 归一化
    return X_norm

# 将离散特征映射到欧式空间，使得特征之间的距离计算更合理
def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]


