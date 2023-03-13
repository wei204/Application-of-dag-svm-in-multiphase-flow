import matplotlib.pyplot as plt


# 原始数据与流型的关系 每种流型选取一个样本，寻找数据特征与流型的关系
def draw_origin(data):
    plt.figure(figsize=(20, 10), dpi=100)
    feature = ['1_2','1_3','1_4','1_5','1_6','2_3','2_4','2_5','2_6','3_4','3_5','3_6','4_5','4_6','5_6']
    # 实验数据：0-19layer,20-37ring,38-61core 仿真数据：0-61plunger,62-72layer,73-88suspended
    layer_value = data[1]
    ring_value = data[62]
    core_value = data[73]
    plt.plot(feature,layer_value,c='red',label='plunger')
    plt.plot(feature, ring_value, c='green',label='layer')
    plt.plot(feature, core_value, c='blue',label='suspended')
    plt.scatter(feature, layer_value, c='red')
    plt.scatter(feature, ring_value, c='green')
    plt.scatter(feature, core_value, c='blue')
    plt.xlabel('feature',fontdict={'size':30})
    plt.ylabel('value',fontdict={'size':30})
    plt.tick_params(axis='both', labelsize=30)
    plt.legend(loc='best',prop={'size':30})
    plt.show()


# 将所有样本数据绘制在一张图上，用于观察数据分布是否存在较明显的差异
def draw_similar(data):
    plt.figure(figsize=(20, 10), dpi=100)
    feature = ['1_2','1_3','1_4','1_5','1_6','2_3','2_4','2_5','2_6','3_4','3_5','3_6','4_5','4_6','5_6']
    #0-19layer,20-37ring,38-61core
    # layer_value0 = data[38]
    # layer_value5 = data[42]
    # layer_value10 = data[50]
    # layer_value13 = data[55]
    # layer_value16 = data[60]
    # plt.plot(feature,layer_value0,c='red',label='layer_value0')
    # plt.plot(feature, layer_value5, c='green',label='layer_value5')
    # plt.plot(feature, layer_value10, c='blue',label='layer_value10')
    # plt.plot(feature, layer_value13, c='yellow', label='layer_value13')
    # plt.plot(feature, layer_value16, c='k', label='layer_value16')
    # plt.scatter(feature, layer_value0, c='red')
    # plt.scatter(feature, layer_value5, c='green')
    # plt.scatter(feature, layer_value10, c='blue')
    # plt.scatter(feature, layer_value13, c='yellow')
    # plt.scatter(feature, layer_value16, c='k')
    # plt.xlabel('feature',fontdict={'size':30})
    # plt.ylabel('value',fontdict={'size':30})
    # plt.tick_params(axis='both', labelsize=30)
    # plt.legend(loc='best',prop={'size':30})


    for i in range(88):
        value = data[i]
        plt.plot(feature,value,c='red')
    plt.xlabel('feature',fontdict={'size':30})
    plt.ylabel('value',fontdict={'size':30})
    plt.tick_params(axis='both', labelsize=30)
    plt.show()


# 将不同流型的所有样本绘制为同一种颜色，用于观察不同流型是否存在不同的数据分布
def draw_different(data):
    plt.figure(figsize=(20, 10), dpi=100)
    feature = ['1_2','1_3','1_4','1_5','1_6','2_3','2_4','2_5','2_6','3_4','3_5','3_6','4_5','4_6','5_6']
    # data[0]:empty data[1]:full data[2-23]:layer data[24-45]:ring data[46-67]:core
    for i in range(22):
        layer_value = data[i+2]
        plt.plot(feature, layer_value, c='red', label='layer')
    for i in range(22):
        ring_value = data[i+24]
        plt.plot(feature, ring_value, c='green', label='ring')
    for i in range(22):
        core_value = data[i+46]
        plt.plot(feature, core_value, c='blue', label='core')
    plt.ylabel('value', fontdict={'size': 30})
    ax = plt.gca()
    ax.yaxis.get_offset_text().set(size=30)
    plt.tick_params(axis='both', labelsize=30)
    plt.show()


# 特征提取后数据与流型的关系
def draw_feature(features):
    plt.figure(figsize=(20, 10), dpi=100)
    feature = ['top_bottom','adm','ads','sam','sds','fm','sfm1','sfm2','avm']
    # 0-21layer,22-43ring,44-65core
    # layer_value = features[0]
    # ring_value = features[17]
    # core_value = features[31]
    # plt.plot(feature, layer_value, c='red', label='layer')
    # plt.plot(feature, ring_value, c='green', label='ring')
    # plt.plot(feature, core_value, c='blue', label='core')
    # plt.scatter(feature, layer_value, c='red')
    # plt.scatter(feature, ring_value, c='green')
    # plt.scatter(feature, core_value, c='blue')
    # plt.xlabel('feature',fontdict={'size':30})
    # plt.ylabel('value',fontdict={'size':30})
    # plt.tick_params(axis='both', labelsize=30)
    # plt.legend(loc='best',prop={'size':30})

    # 绘制不同流型的所有样本
    for i in range(22):
        layer_value = features[i]
        plt.plot(feature, layer_value, c='red', label='layer')
    for i in range(22):
        ring_value = features[i+22]
        plt.plot(feature, ring_value, c='green', label='ring')
    for i in range(22):
        core_value = features[i+44]
        plt.plot(feature, core_value, c='blue', label='core')
    plt.xlabel('feature',fontdict={'size':30})
    plt.ylabel('value',fontdict={'size':30})
    plt.tick_params(axis='both', labelsize=30)
    plt.show()


def feature_type(s):
    it = {0:'top_bottom', 1:'adm', 2:'ads', 3:'sam', 4:'sds', 5:'fm', 6:'u3vm', 7:'d3vm', 8:'avm'}  # sfm1,sfm2
    return it[s]


# 绘制不同流型对应的特征值
# features为特征矩阵，f_index为某个特征对应的索引
def draw_feature_name(features, f_index):
    plt.figure()
    x = range(0, 22, 1)
    layer_value = features[0:22, f_index]
    plt.plot(x, layer_value, c='red', label='layer')
    ring_value = features[22:44, f_index]
    plt.plot(x, ring_value, c='green', label='ring')
    core_value = features[44:66, f_index]
    plt.plot(x, core_value, c='blue', label='core')
    plt.ylabel('{}'.format(feature_type(f_index)), fontdict={'size': 30})
    plt.legend(loc='lower left')
    plt.show()


# 采用柱状图观察样本数据与不同流型之间的关系，每种流型选取一个样本
def draw_bar(data):
    plt.figure()
    feature = ['1_2', '1_3', '1_4', '1_5', '1_6', '2_3', '2_4', '2_5', '2_6', '3_4', '3_5', '3_6', '4_5', '4_6', '5_6']
    layer_value = data[5]  # 分层流
    plt.bar(range(len(feature)),layer_value,color='r',tick_label=feature)
    plt.tick_params(axis='both', labelsize=10)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set(size=16)
    plt.figure()
    ring_value = data[27]  # 环形流
    plt.bar(range(len(feature)),ring_value,color='g',tick_label=feature)
    plt.tick_params(axis='both', labelsize=10)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set(size=16)
    plt.figure()
    core_value = data[49]  # 中心流
    plt.bar(range(len(feature)),core_value,color='b',tick_label=feature)
    plt.tick_params(axis='both', labelsize=10)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set(size=16)
    plt.show()


# 采用柱状图观察特征提取后数据与不同流型之间的关系
def draw_bar_feature(data):
    plt.figure()
    feature = ['top_bottom','adm','ads','sam','sds','fm','sfm1','sfm2','avm']
    layer_value = data[5]  # 分层流
    plt.bar(range(len(feature)),layer_value,color='r',tick_label=feature)
    plt.figure()
    ring_value = data[27]  # 环形流
    plt.bar(range(len(feature)),ring_value,color='g',tick_label=feature)
    plt.figure()
    core_value = data[49]  # 中心流
    plt.bar(range(len(feature)),core_value,color='b',tick_label=feature)
    plt.show()