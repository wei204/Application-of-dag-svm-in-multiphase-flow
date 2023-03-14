import numpy as np


# 实验数据特征提取 samples：n*15，n为样本个数
def feature_extract(samples, n):
    samples = np.array(samples)
    # 顶部电容与底部电容比值 需改
    top_bottom = samples[:, 4]/samples[:, 9]
    # 相邻极板电容平均值
    adm = (samples[:, 0]+samples[:, 4]+samples[:, 5]+samples[:, 9]+samples[:, 12]+samples[:, 14])/6
    # 相邻极板电容偏差的平方
    ads = np.round((np.square(samples[:,0]-adm)+np.square(samples[:,4]-adm)+np.square(samples[:,5]-adm)
           +np.square(samples[:,9]-adm)+np.square(samples[:,12]-adm)+np.square(samples[:,14]-adm))/6, 2)
    # 次相邻极板电容平均值
    sam = np.round((samples[:,1]+samples[:,3]+samples[:,6]+samples[:,8]+samples[:,10]+samples[:,13])/6, 2)
    # 次相邻极板电容偏差的平方
    sds = np.round((np.square(samples[:, 1] - sam) + np.square(samples[:, 3] - sam) + np.square(samples[:, 6] - sam)
           + np.square(samples[:, 8] - sam) + np.square(samples[:, 10] - sam) + np.square(samples[:, 13] - sam)) / 6, 2)
    # 面极板电容平均值
    fm = np.round((samples[:,2]+samples[:,7]+samples[:,11])/3, 2)
    # 1/3高度处电容平均值 需改
    # sfm1 = np.round((samples[:,0]+samples[:,14])/2, 3)
    # 上面3个电容平均值
    u3vm = np.round((samples[:, 5]+samples[:, 9]+samples[:, 12])/3, 2)
    # 2/3高度处电容平均值 需改
    # sfm2 = np.round((samples[:,5]+samples[:,12])/2, 3)
    # 下面3个电容平均值
    d3vm = np.round((samples[:, 0]+samples[:, 4]+samples[:, 14])/3, 2)
    # 所有电容平均值
    avm = np.round((np.sum(samples,1))/15, 2)
    features = []  # 最终是9*n
    features.append(top_bottom)
    features.append(adm)
    features.append(ads)
    features.append(sam)
    features.append(sds)
    features.append(fm)
    # features.append(sfm1)
    # features.append(sfm2)
    features.append(u3vm)
    features.append(d3vm)
    features.append(avm)
    features = np.array(features)
    #将features 9*n->n*9
    f = []
    for i in range(n):
        f.append(features[:,i])
    f = np.array(f)
    return f


# 仿真数据特征提取
def feature_extract_s(samples,n):
    samples = np.array(samples)
    # 顶部电容与底部电容比值 需改
    top_bottom = samples[:,0]/samples[:,12]
    #相邻极板电容平均值
    adm = (samples[:,0]+samples[:,4]+samples[:,5]+samples[:,9]+samples[:,12]+samples[:,14])/6
    #相邻极板电容偏差的平方
    ads = (np.square(samples[:,0]-adm)+np.square(samples[:,4]-adm)+np.square(samples[:,5]-adm)
           +np.square(samples[:,9]-adm)+np.square(samples[:,12]-adm)+np.square(samples[:,14]-adm))/6
    #次相邻极板电容平均值
    sam = (samples[:,1]+samples[:,3]+samples[:,6]+samples[:,8]+samples[:,10]+samples[:,13])/6
    #次相邻极板电容偏差的平方
    sds = (np.square(samples[:, 1] - sam) + np.square(samples[:, 3] - sam) + np.square(samples[:, 6] - sam)
           + np.square(samples[:, 8] - sam) + np.square(samples[:, 10] - sam) + np.square(samples[:, 13] - sam)) / 6
    #面极板电容平均值
    fm = (samples[:,2]+samples[:,7]+samples[:,11])/3
    #1/3高度处电容平均值 需改
    sfm1 = (samples[:,9]+samples[:,14])/2
    #2/3高度处电容平均值 需改
    sfm2 = (samples[:,4]+samples[:,5])/2
    #所有电容平均值
    avm = (np.sum(samples,1))/15
    features = []  #最终是9*n
    features.append(top_bottom)
    features.append(adm)
    features.append(ads)
    features.append(sam)
    features.append(sds)
    features.append(fm)
    features.append(sfm1)
    features.append(sfm2)
    features.append(avm)
    features = np.array(features)
    #将features 9*n->n*9
    f = []
    for i in range(n):
        f.append(features[:,i])
    f = np.array(f)
    return f


