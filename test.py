# import cv2
# import numpy as np
# # # data = [1,23,4,5]
# # # print(data[3])
# # # img = cv2.imread(r'E:\pyitem\opencv_img\photo\2.jpg')
# # # h,w,c = img.shape
# # # print(h,w,c)
# # #
# # def reshape_array(x):
# #     f = []
# #     f.append(x[:,0])
# #     f.append(x[:,1])
# #     f.append(x[:,2])
# #     # f = np.array(f)
# #     return f
# #
# # def normalize_data(data):
# #     mean = np.mean(data, axis=0)
# #     std = np.std(data, axis=0)
# #     for i in range(data.shape[0]):
# #         data[i, :] = (data[i, :] - mean) / std
# #     return data
# #
# # x =[[1,2,3],[4,5,6]]
# # x = np.array(x)
# # # print(x)
# # # mean = np.mean(x,axis=0)
# # # print(mean)
# # # f = reshape_array(x)
# # # print(f)
# # # std = np.std(x,axis=0)
# # # print(std)
# # # data = normalize_data(x)
# # # print(data)
# # # empty = np.array([1,1,1])
# # # print(x-empty)
# #
# # # data = np.loadtxt(r'H:\COMSOL\project\ECTwzm\kg.txt')
# # # print(data.shape)
# #
# # # b = np.ones([3, 1], dtype=np.float32)
# # # # a = np.tile(b, [1, 4])
# # # print(b)
# # # print('--------------')
# # # print(a)
# #
# # # label = []
# # # for i in range(5):
# # #     label.append(i)
# # # print(label)
# # # label = np.mat(label)
# # # print(label)
# # # label.transpose()
# # # print(label)
# #
# # # empty_data = np.loadtxt(r'H:\COMSOL\project\ECTwzm\kg.txt')
# # # full_data = np.loadtxt(r'H:\COMSOL\project\ECTwzm\mg.txt')
# # # # normalData = (samples - empty_data) / (full_data - empty_data)
# # # print(full_data )
# #
# # # data = np.array([[1,2,3],[4,5,6]], np.float32)
# # # print(data[0:1])
# # # # normal_datas = np.zeros((2, 1))
# # # min_data = np.min(data,axis=1)
# # # max_data = np.max(data,axis=1)
# # # for i in range(2):
# # #     nor_data = (data[i]-min_data[i])/(max_data[i]-min_data[i])
# # #     print(nor_data)
# # # for i in range(len(data)):
# # #     min_data = min(data[i])
# # #     max_data = max(data[i])
# # #     normal_data = (data[i]-min_data)/(max_data-min_data)
# # #     data[i] = normal_data
# # # print(data)
# #
# # # def feature_type(s):
# # #     it = {0:'top_bottom', 1:'adm', 2:'ads', 3:'sam', 4:'sds', 5:'fm', 6:'sfm1', 7:'sfm2', 8:'avm'}
# # #     return it[s]
# # # print(feature_type(0))
# #
# # # x = []
# # # x.append(1)
# # # x.append(2)
# # # x_mat = np.mat(x)
# # # print(x_mat)
# # # print(np.shape(x_mat))
# # # x_T = np.transpose(x_mat)
# # # print(x_T)
# # # print(np.shape(x_T))
# #
# # # class Parent:
# # #
# # #     def __init__(self):
# # #         print('Parent Class Constructor')
# # #
# # #     def ParentMethod(self):
# # #         print ('Parent Method')
# # #
# # #     @staticmethod
# # #     def ParentStaticMethod():
# # #         print ('Parent Static Method')
# # #
# # # pObj=Parent()
# # # pObj.ParentMethod()
# # # Parent.ParentStaticMethod()
# #
# # import sys
# # from numpy import *
# # from svm import *
# # from os import listdir
# # from plattSMO import PlattSMO
# # import pickle
# # class LibSVM:
# #
# #     def __init__(self,data=[],label=[],C=0,toler=0,maxIter=0,**kernelargs):
# #         self.classlabel = unique(label)
# #         self.classNum = len(self.classlabel)
# #         self.classfyNum = (self.classNum * (self.classNum-1))/2  #分类器的个数
# #         self.classfy = []   #存放n*(n-1)个svm分类器
# #         self.dataSet={}
# #         self.kernelargs = kernelargs
# #         self.C = C
# #         self.toler = toler
# #         self.maxIter = maxIter
# #         m = shape(data)[0]
# #         #将相同标签数据存入dataSet的同一个索引位置
# #         for i in range(m):
# #             if label[i] not in self.dataSet.keys():
# #                 self.dataSet[label[i]] = []
# #                 self.dataSet[label[i]].append(data[i][:])
# #             else:
# #                 self.dataSet[label[i]].append(data[i][:])
# #     def train(self):
# #         num = self.classNum
# #         for i in range(num):
# #             for j in range(i+1,num):
# #                 data = []
# #                 label = [1.0]*shape(self.dataSet[self.classlabel[i]])[0]
# #                 label.extend([-1.0]*shape(self.dataSet[self.classlabel[j]])[0])
# #                 data.extend(self.dataSet[self.classlabel[i]])
# #                 data.extend(self.dataSet[self.classlabel[j]])
# #                 svm = PlattSMO(array(data),array(label),self.C,self.toler,self.maxIter,**self.kernelargs)
# #                 svm.smoP()
# #                 self.classfy.append(svm)
# #         self.dataSet = None
# #     def predict(self,data,label):
# #         m = shape(data)[0]  #数据样本个数
# #         num = self.classNum  #类别个数
# #         classlabel = []
# #         count = 0.0
# #         for n in range(m):
# #             result = [0] * num
# #             index = -1
# #             for i in range(num):
# #                 for j in range(i + 1, num):
# #                     index += 1
# #                     s = self.classfy[index]  #使用第index个分类器判断第n个样本的类别
# #                     t = s.predict([data[n]])[0]
# #                     if t > 0.0:    #结果大于0，表示为正类，对应的类别计数+1
# #                         result[i] +=1
# #                     else:
# #                         result[j] +=1
# #             classlabel.append(result.index(max(result)))
# #             if classlabel[-1] != label[n]: #判断错误的个数
# #                 count +=1
# #                 print(label[n],classlabel[n])
# #         #print classlabel
# #         print("error rate:",count / m)
# #         return classlabel
# #     def save(self,filename):
# #         fw = open(filename,'wb')
# #         pickle.dump(self,fw,2)
# #         fw.close()
# #
# #     @staticmethod
# #     def load(filename):
# #         fr = open(filename,'rb')
# #         svm = pickle.load(fr)
# #         fr.close()
# #         return svm
# #
# # def loadImage(dir,maps = None):
# #     dirList = listdir(dir)
# #     data = []
# #     label = []
# #     for file in dirList:
# #         label.append(file.split('_')[0])
# #         lines = open(dir +'/'+file).readlines()
# #         row = len(lines)
# #         col = len(lines[0].strip())
# #         line = []
# #         for i in range(row):
# #             for j in range(col):
# #                 line.append(float(lines[i][j]))
# #         data.append(line)
# #         if maps != None:
# #             label[-1] = float(maps[label[-1]])
# #         else:
# #             label[-1] = float(label[-1])
# #     return data,label
# # def main():
# #     '''
# #     data,label = loadImage('trainingDigits')
# #     svm = LibSVM(data, label, 200, 0.0001, 10000, name='rbf', theta=20)
# #     svm.train()
# #     svm.save("svm.txt")
# #     '''
# #     data,label = loadImage('trainingDigits')
# #     print(len(data))
# #     print(len(label))
# #     # svm = LibSVM(data, label, 200, 0.0001, 10000, name='rbf', theta=20)
# #     # svm.train()
# #     # svm.save("svm.txt")
# #     # svm = LibSVM.load("svm.txt")
# #     # test,testlabel = loadImage('testDigits')
# #     # svm.predict(test,testlabel)
# #
# #     # 多相流流型识别
# #     # data, label = loadImage(r'E:\pyitem\Multiphase_flow\train.txt')
# #
# # if __name__ == "__main__":
# #     # sys.exit(main())
# #     main()
# #
# #
# #
# #
# # def normalize_data_nonlinear(samples):
# #     # 空管电容与满管电容
# #     empty_data = samples[0]
# #     full_data = samples[1]
# #     for i in range(len(samples)):
# #         samples[i] = (full_data*(samples[i]-empty_data)) / (samples[i]*(full_data-empty_data))
# #     return samples
# #
# #
# # samples = np.array([[1, 2], [4, 5], [2, 3], [3, 4]], np.float32)
# # samples_nor = normalize_data_nonlinear(samples)
# # print(samples_nor)
#
# # x = [1]
# # y = x
# # print(id(x))
# # print(id(y))
# # x.append(2)
# # y.append(3)
# # print(id(x))
# # print(id(y))
#
# # l = [1,2,3]
# # l.pop(0)
# # # print(l.pop())
# # print(l)
#
# # classfy_dict = dict()
# # classList = [1,2,3]
# # class_num = len(classList)
# # classfy_dict['{0}-{1}'.format(classList[0],classList[class_num-1])] = 13
# # print(classfy_dict['1-3'])
#
# # num = 4
# # while num>2:
# #     a = 1
# #
# #     if a>0:
# #         print(a)
# #     num -= 1
# # print(num)
#
#
# # import random
# # for i in range(5):
# #     classList = [i for i in range(5)]  # 类别列表
# #     random.shuffle(classList)
# #     print(classList)
# # import datetime
# # def fun():
# #     for i in range(100000000):
# #         continue
# #     a = [0,0]
# #     b = [1,2]
# #     return a, b
# # def fun1():
# #     for i in range(200000000):
# #         continue
# #     a = [0,0]
# #     b = [1,2]
# #     return a, b
# # st = datetime.datetime.now()
# # b = fun()[1]
# # et = datetime.datetime.now()
# # print((et-st).seconds)
# #
# # st = datetime.datetime.now()
# # b = fun1()[1]
# # et = datetime.datetime.now()
# # print((et-st).seconds)
#
#
# # -------------------------------------决策树-----------------------------------------
#
# # -*- coding: UTF-8 -*-
# from matplotlib.font_manager import FontProperties
# import matplotlib.pyplot as plt
# from math import log
# import operator
# import pickle
#
# """
# 函数说明:计算给定数据集的经验熵(香农熵)
# Parameters:
# 	dataSet - 数据集
# Returns:
# 	shannonEnt - 经验熵(香农熵)
# Author:
# 	Jack Cui
# Blog:
# 	http://blog.csdn.net/c406495762
# Modify:
# 	2017-07-24
# """
#
#
# def calcShannonEnt(dataSet):
#     numEntires = len(dataSet)  # 返回数据集的行数
#     labelCounts = {}  # 保存每个标签(Label)出现次数的字典
#     for featVec in dataSet:  # 对每组特征向量进行统计
#         currentLabel = featVec[-1]  # 提取标签(Label)信息
#         if currentLabel not in labelCounts.keys():  # 如果标签(Label)没有放入统计次数的字典,添加进去
#             labelCounts[currentLabel] = 0
#         labelCounts[currentLabel] += 1  # Label计数
#     shannonEnt = 0.0  # 经验熵(香农熵)
#     for key in labelCounts:  # 计算香农熵
#         prob = float(labelCounts[key]) / numEntires  # 选择该标签(Label)的概率
#         shannonEnt -= prob * log(prob, 2)  # 利用公式计算
#     return shannonEnt  # 返回经验熵(香农熵)
#
#
# """
# 函数说明:创建测试数据集
# Parameters:
# 	无
# Returns:
# 	dataSet - 数据集
# 	labels - 特征标签
# Author:
# 	Jack Cui
# Blog:
# 	http://blog.csdn.net/c406495762
# Modify:
# 	2017-07-20
# """
#
#
# def createDataSet():
#     dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
#                [0, 0, 0, 1, 'no'],
#                [0, 1, 0, 1, 'yes'],
#                [0, 1, 1, 0, 'yes'],
#                [0, 0, 0, 0, 'no'],
#                [1, 0, 0, 0, 'no'],
#                [1, 0, 0, 1, 'no'],
#                [1, 1, 1, 1, 'yes'],
#                [1, 0, 1, 2, 'yes'],
#                [1, 0, 1, 2, 'yes'],
#                [2, 0, 1, 2, 'yes'],
#                [2, 0, 1, 1, 'yes'],
#                [2, 1, 0, 1, 'yes'],
#                [2, 1, 0, 2, 'yes'],
#                [2, 0, 0, 0, 'no']]
#     labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 特征标签
#     return dataSet, labels  # 返回数据集和分类属性
#
#
# """
# 函数说明:按照给定特征划分数据集
# Parameters:
# 	dataSet - 待划分的数据集
# 	axis - 划分数据集的特征
# 	value - 需要返回的特征的值
# Returns:
# 	无
# Author:
# 	Jack Cui
# Blog:
# 	http://blog.csdn.net/c406495762
# Modify:
# 	2017-07-24
# """
#
#
# def splitDataSet(dataSet, axis, value):
#     retDataSet = []  # 创建返回的数据集列表
#     for featVec in dataSet:  # 遍历数据集
#         if featVec[axis] == value:
#             reducedFeatVec = featVec[:axis]  # 去掉axis特征
#             reducedFeatVec.extend(featVec[axis + 1:])  # 将符合条件的添加到返回的数据集
#             retDataSet.append(reducedFeatVec)
#     return retDataSet  # 返回划分后的数据集
#
#
# """
# 函数说明:选择最优特征
# Parameters:
# 	dataSet - 数据集
# Returns:
# 	bestFeature - 信息增益最大的(最优)特征的索引值
# Author:
# 	Jack Cui
# Blog:
# 	http://blog.csdn.net/c406495762
# Modify:
# 	2017-07-20
# """
#
#
# def chooseBestFeatureToSplit(dataSet):
#     numFeatures = len(dataSet[0]) - 1  # 特征数量
#     baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的香农熵
#     bestInfoGain = 0.0  # 信息增益
#     bestFeature = -1  # 最优特征的索引值
#     for i in range(numFeatures):  # 遍历所有特征
#         # 获取dataSet的第i个所有特征
#         featList = [example[i] for example in dataSet]
#         uniqueVals = set(featList)  # 创建set集合{},元素不可重复
#         newEntropy = 0.0  # 经验条件熵
#         for value in uniqueVals:  # 计算信息增益
#             subDataSet = splitDataSet(dataSet, i, value)  # subDataSet划分后的子集
#             prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
#             newEntropy += prob * calcShannonEnt(subDataSet)  # 根据公式计算经验条件熵
#         infoGain = baseEntropy - newEntropy  # 信息增益
#         # print("第%d个特征的增益为%.3f" % (i, infoGain))			#打印每个特征的信息增益
#         if (infoGain > bestInfoGain):  # 计算信息增益
#             bestInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
#             bestFeature = i  # 记录信息增益最大的特征的索引值
#     return bestFeature  # 返回信息增益最大的特征的索引值
#
#
# """
# 函数说明:统计classList中出现此处最多的元素(类标签)
# Parameters:
# 	classList - 类标签列表
# Returns:
# 	sortedClassCount[0][0] - 出现此处最多的元素(类标签)
# Author:
# 	Jack Cui
# Blog:
# 	http://blog.csdn.net/c406495762
# Modify:
# 	2017-07-24
# """
#
#
# def majorityCnt(classList):
#     classCount = {}
#     for vote in classList:  # 统计classList中每个元素出现的次数
#         if vote not in classCount.keys(): classCount[vote] = 0
#         classCount[vote] += 1
#     sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 根据字典的值降序排序
#     return sortedClassCount[0][0]  # 返回classList中出现次数最多的元素
#
#
# """
# 函数说明:创建决策树
# Parameters:
# 	dataSet - 训练数据集
# 	labels - 分类属性标签
# 	featLabels - 存储选择的最优特征标签
# Returns:
# 	myTree - 决策树
# Author:
# 	Jack Cui
# Blog:
# 	http://blog.csdn.net/c406495762
# Modify:
# 	2017-07-25
# """
#
#
# def createTree(dataSet, labels, featLabels):
#     classList = [example[-1] for example in dataSet]  # 取分类标签(是否放贷:yes or no)
#     if classList.count(classList[0]) == len(classList):  # 如果类别完全相同则停止继续划分
#         return classList[0]
#     if len(dataSet[0]) == 1 or len(labels) == 0:  # 遍历完所有特征时返回出现次数最多的类标签
#         return majorityCnt(classList)
#     bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最优特征
#     bestFeatLabel = labels[bestFeat]  # 最优特征的标签
#     featLabels.append(bestFeatLabel)
#     myTree = {bestFeatLabel: {}}  # 根据最优特征的标签生成树
#     del (labels[bestFeat])  # 删除已经使用特征标签
#     featValues = [example[bestFeat] for example in dataSet]  # 得到训练集中所有最优特征的属性值
#     uniqueVals = set(featValues)  # 去掉重复的属性值
#     for value in uniqueVals:  # 遍历特征，创建决策树。
#         subLabels = labels[:]
#         myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, featLabels)
#
#     return myTree
#
#
# """
# 函数说明:获取决策树叶子结点的数目
# Parameters:
# 	myTree - 决策树
# Returns:
# 	numLeafs - 决策树的叶子结点的数目
# Author:
# 	Jack Cui
# Blog:
# 	http://blog.csdn.net/c406495762
# Modify:
# 	2017-07-24
# """
#
#
# def getNumLeafs(myTree):
#     numLeafs = 0  # 初始化叶子
#     firstStr = next(iter(
#         myTree))  # python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
#     secondDict = myTree[firstStr]  # 获取下一组字典
#     for key in secondDict.keys():
#         if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
#             numLeafs += getNumLeafs(secondDict[key])
#         else:
#             numLeafs += 1
#     return numLeafs
#
#
# """
# 函数说明:获取决策树的层数
# Parameters:
# 	myTree - 决策树
# Returns:
# 	maxDepth - 决策树的层数
# Author:
# 	Jack Cui
# Blog:
# 	http://blog.csdn.net/c406495762
# Modify:
# 	2017-07-24
# """
#
#
# def getTreeDepth(myTree):
#     maxDepth = 0  # 初始化决策树深度
#     firstStr = next(iter(
#         myTree))  # python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
#     secondDict = myTree[firstStr]  # 获取下一个字典
#     for key in secondDict.keys():
#         if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
#             thisDepth = 1 + getTreeDepth(secondDict[key])
#         else:
#             thisDepth = 1
#         if thisDepth > maxDepth: maxDepth = thisDepth  # 更新层数
#     return maxDepth
#
#
# """
# 函数说明:绘制结点
# Parameters:
# 	nodeTxt - 结点名
# 	centerPt - 文本位置
# 	parentPt - 标注的箭头位置
# 	nodeType - 结点格式
# Returns:
# 	无
# Author:
# 	Jack Cui
# Blog:
# 	http://blog.csdn.net/c406495762
# Modify:
# 	2017-07-24
# """
#
#
# def plotNode(nodeTxt, centerPt, parentPt, nodeType):
#     arrow_args = dict(arrowstyle="<-")  # 定义箭头格式
#     font = FontProperties(fname=r"c:\windows\fonts\Inkfree.ttf", size=14)  # 设置中文字体
#     createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',  # 绘制结点
#                             xytext=centerPt, textcoords='axes fraction',
#                             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)
#
#
# """
# 函数说明:标注有向边属性值
# Parameters:
# 	cntrPt、parentPt - 用于计算标注位置
# 	txtString - 标注的内容
# Returns:
# 	无
# Author:
# 	Jack Cui
# Blog:
# 	http://blog.csdn.net/c406495762
# Modify:
# 	2017-07-24
# """
#
#
# def plotMidText(cntrPt, parentPt, txtString):
#     xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 计算标注位置
#     yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
#     createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
#
#
# """
# 函数说明:绘制决策树
# Parameters:
# 	myTree - 决策树(字典)
# 	parentPt - 标注的内容
# 	nodeTxt - 结点名
# Returns:
# 	无
# Author:
# 	Jack Cui
# Blog:
# 	http://blog.csdn.net/c406495762
# Modify:
# 	2017-07-24
# """
#
#
# def plotTree(myTree, parentPt, nodeTxt):
#     decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 设置结点格式
#     leafNode = dict(boxstyle="round4", fc="0.8")  # 设置叶结点格式
#     numLeafs = getNumLeafs(myTree)  # 获取决策树叶结点数目，决定了树的宽度
#     depth = getTreeDepth(myTree)  # 获取决策树层数
#     firstStr = next(iter(myTree))  # 下个字典
#     cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)  # 中心位置
#     plotMidText(cntrPt, parentPt, nodeTxt)  # 标注有向边属性值
#     plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 绘制结点
#     secondDict = myTree[firstStr]  # 下一个字典，也就是继续绘制子结点
#     plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # y偏移
#     for key in secondDict.keys():
#         if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
#             plotTree(secondDict[key], cntrPt, str(key))  # 不是叶结点，递归调用继续绘制
#         else:  # 如果是叶结点，绘制叶结点，并标注有向边属性值
#             plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
#             plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
#             plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
#     plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD
#
#
# """
# 函数说明:创建绘制面板
# Parameters:
# 	inTree - 决策树(字典)
# Returns:
# 	无
# Author:
# 	Jack Cui
# Blog:
# 	http://blog.csdn.net/c406495762
# Modify:
# 	2017-07-24
# """
#
#
# def createPlot(inTree):
#     fig = plt.figure(1, facecolor='white')  # 创建fig
#     fig.clf()  # 清空fig
#     axprops = dict(xticks=[], yticks=[])
#     createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉x、y轴
#     plotTree.totalW = float(getNumLeafs(inTree))  # 获取决策树叶结点数目
#     plotTree.totalD = float(getTreeDepth(inTree))  # 获取决策树层数
#     plotTree.xOff = -0.5 / plotTree.totalW
#     plotTree.yOff = 1.0  # x偏移
#     plotTree(inTree, (0.5, 1.0), '')  # 绘制决策树
#     plt.show()  # 显示绘制结果
#
#
# """
# 函数说明:使用决策树分类
# Parameters:
# 	inputTree - 已经生成的决策树
# 	featLabels - 存储选择的最优特征标签
# 	testVec - 测试数据列表，顺序对应最优特征标签
# Returns:
# 	classLabel - 分类结果
# Author:
# 	Jack Cui
# Blog:
# 	http://blog.csdn.net/c406495762
# Modify:
# 	2017-07-25
# """
#
#
# def classify(inputTree, featLabels, testVec):
#     firstStr = next(iter(inputTree))  # 获取决策树结点
#     secondDict = inputTree[firstStr]  # 下一个字典
#     featIndex = featLabels.index(firstStr)
#     for key in secondDict.keys():
#         if testVec[featIndex] == key:
#             if type(secondDict[key]).__name__ == 'dict':
#                 classLabel = classify(secondDict[key], featLabels, testVec)
#             else:
#                 classLabel = secondDict[key]
#     return classLabel
#
#
# """
# 函数说明:存储决策树
# Parameters:
# 	inputTree - 已经生成的决策树
# 	filename - 决策树的存储文件名
# Returns:
# 	无
# Author:
# 	Jack Cui
# Blog:
# 	http://blog.csdn.net/c406495762
# Modify:
# 	2017-07-25
# """
#
#
# def storeTree(inputTree, filename):
#     with open(filename, 'wb') as fw:
#         pickle.dump(inputTree, fw)
#
#
# """
# 函数说明:读取决策树
# Parameters:
# 	filename - 决策树的存储文件名
# Returns:
# 	pickle.load(fr) - 决策树字典
# Author:
# 	Jack Cui
# Blog:
# 	http://blog.csdn.net/c406495762
# Modify:
# 	2017-07-25
# """
#
#
# def grabTree(filename):
#     fr = open(filename, 'rb')
#     return pickle.load(fr)
#
#
# if __name__ == '__main__':
#     dataSet, labels = createDataSet()
#     featLabels = []
#     myTree = createTree(dataSet, labels, featLabels)
#     createPlot(myTree)
#     testVec = [0, 1]  # 测试数据
#     result = classify(myTree, featLabels, testVec)
#     if result == 'yes':
#         print('放贷')
#     if result == 'no':
#         print('不放贷')


# lis = ['x', 1, 'y', 2, 'z', 3]
# dic = dict(zip(lis[::2], lis[1::2]))
# print(dic)
# # {'y': 2, 'x': 1, 'z': 3}

# l = [1,0,4]
# del(l[1])
# print(l)

# d = {'0-1': 1, '0-2': 2, '1-2': 3}
# ke = [1,2,0]
# if f'{ke[2]}-{ke[0]}' in d.keys():
#     print('存在0-1键')
# if '1-0' not in d.keys():
#     print('不存在1-0键')
# if 3 in d.values():
#     print('键值也可以查询')
# name = 'wei'
# age = 22
# print(f'{name} is {age}')
import random
# ke = [1,2,0]
# print(ke)
# # random.shuffle(ke)
# w = ke.reverse()
# print(w)
# t = {'s0':3, 's1':2}
# t['s0'] += 1
# st = sorted(t.items(), key=operator.itemgetter(1))

# for i in range(3):
#     best = i
#     tree = {best:{}}
#     tree[best][value] =
# print(tree)

# l = [1,1,2,3]
# print(l.count(l[2]))

# t = []
# t.append(1)
# t.append(2)
# print(t)
# import numpy as np
# t1 = [[1,2,3], [1,4,2]]
# t2 = [4,5,6]
# t3 = np.array(t1).sum(axis=0)/len(t1)
#
# print(t3)
# s = np.square(np.array(t1)-t3)
# print(s)
# fen = np.sqrt(np.sum(np.sum(np.square(np.array(t1)-t3), axis=1), axis=0)/(len(t1)-1))
# print(fen)
from numpy import unique


class A:
    # def __init__(self, x, y):
    #     self.x = x
    #     self.y = y

    # @staticmethod
    def sum_xy(self, x, y):
        return x+y
    @staticmethod
    def sum1(x,y):
        return x+y
    # @staticmethod
    def sub_xz(self, x):
        z = self.sum_xy(10,11)
        # z = self.sum1(1,2)
        return x-z

# s = A()
# print(A.sum1(2,3))
# print(s.sub_xz(22))
import numpy as np
# l = [[1,2,3],[2,2,3]]
# e = np.zeros((1,3))
# for i in range(2):
#     e += l[i]
# print(np.sum(e))

e1 = np.array([[1,2],[10,11]])
e2 = np.array([2,3])
# print(np.sqrt(np.sum(np.square(e1-e2))))

e_ = np.array([1,1])
# print(np.sum(np.sum(np.square(e1-e_), axis=1)))
# tree = []
# tree.append([1,2])
# tree.append([3,4])
# print(tree)

# tree = [1,2,3]
# t = tree.copy()
# tree.pop()
# print(tree)
# print(t)

# s = dict()
# s['0-1'] = 1
# s['0-2'] = 10
# s['1-2'] = 2
#
# # minS = min(s.values())
# # print(minS)
# # 找到带0的
# for key in s.keys():
#     for i in range(3):
#         if key==f'{0}-{i}':
#             print(key)
# # print(s.keys())


# def BinaryTree(r):
#     return [r,[],[]]
# def insertLeft(root,newBranch):
#     t= root.pop(1)
#     if len(t)>1:
#         root.insert(1,[newBranch,t,[]])
#     else:
#         root.insert(1,[newBranch,[],[]])
#     return root
# def insertRight(root,newBranch):
#     t= root.pop(2)
#     if len(t)>1:
#         root.insert(2,[newBranch,[],t])
#     else:
#         root.insert(2,[newBranch,[],[]])
#         return root
# def getRootVal(root):
#     return root[0]
# def setRootVal(root,newVal):
#     root[0]=newVal
# def getLeftChild(root):
#     return root[1]
# def getRightChild(root):
#     return root[2]


# r = 2
# myTree = BinaryTree(r)
# insertLeft(myTree,1)
# insertRight(myTree,3)
# lTree = getLeftChild(myTree)
# rTree = getRightChild(myTree)
# root = getRootVal(myTree)
# setRootVal(lTree,0)
# rTree = insertRight(rTree, 4)
# # print(root)
# print(myTree)
# print(lTree)
# print(rTree)
#
# S = []
# S.append(0)
# S.append(1)
# print(S)
# S.append(3)
# print(S)

class BinaryTree:
    def __init__(self, rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self, newNode):
        if self.leftChild is None:
            self.leftChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self, newNode):
        if self.rightChild is None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t

    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def setRootVal(self, obj):
        self.key = obj

    def getRootVal(self):
        return self.key




# new = BinaryTree(1)
# # new.setRootVal(2)
# # print(new.getRootVal())
# l = BinaryTree(0)
# l.insertLeft(-1)
# l.insertRight(1)
# new.insertLeft(l)
# print(new.getLeftChild())


# S = [[1,1],[4,2],[3,3],[0,4]]
# # S_re = sorted(S)
# # print(S_re)
# # S_re.pop(0)
# # print(S_re)
# i,j = S[0][0], S[0][1]
# print(i, j)

# S = []
# S[0].append([[],[]])
# S[0][0].append(2)
# # S[1][0].append(1)
# print(S)
# S[0][1].append(2)
# print(S)

S = []
S.append([1,2])
# if S[1] is None:

# S.append([[1,3]])
# S.append([[2,3]])
# S[1].append([[0,1]])
# S[1].append([[0,3]])
# S[2].append([[0,2]])
# S[2].append([[0,3]])
# S[1][1].append([0,0])
# print(S)
#
# x = [[1,0],[0,3],[3,5],[0,2], [0,4]]
# x1 = sorted(x)
# print(x1)

# import math
# S_class = [0,1,2,3]
# S = [[3.1,0,1], [3.0,0,2], [4.1,0,3],[4.5,1,2],[2.5,1,3],[3.5,2,3]]
# # 对列表进行升序排列
# S_sort = sorted(S)
# # 新建列表 存放标签类别号
# S_label = []
# # 将不可分类度最小的两类的类别号存入列表首尾
# S_label.append(S_sort[0][1])
# S_label.append(S_sort[0][2])
# n = math.ceil((len(S_class)-2)/2)
# for num in range(n):
#     max_first, max_tail = -1000, -1000
#     i_first, j_first = 0, 0
#     i_tail, j_tail = 0, 0
#     for s in S_sort:
#         # 得到与类别列表首元素对应类别差距最小的类别号
#         if s[0] > max_first and s[1] == S_label[num] or s[2] == S_label[num]:
#             max_first = s[0]
#             i_first, j_first = s[1], s[2]
#         # 得到与类别列表尾元素对应类别差距最小的类别号
#         if s[0] > max_tail and s[1] == S_label[len(S_label)-num-1] or s[2] == S_label[len(S_label)-num-1]:
#             max_tail = s[0]
#             i_tail, j_tail = s[1], s[2]
#     # 将其插入在类别列表首元素后面
#     if i_first not in S_label:
#         S_label.insert(num+1, i_first)
#     if j_first not in S_label:
#         S_label.insert(num+1, j_first)
#     if i_tail not in S_label:
#         S_label.insert(len(S_label)-1-num, i_tail)
#     if j_tail not in S_label:
#         S_label.insert(len(S_label) - 1 - num, j_tail)
#
# print(S_label)

# import math
# l = [1,1,2]..
# x = unique(l)
# print(x)

# def readTxt(dir):
#     file = open(dir, 'r')
#     file_data = file.readlines()
#     datas = []
#     labels = []
#     for line in file_data:
#         line = line.split('\n')[0]
#         l = line.split(',')[:-1]
#         data = []
#         for i in l:
#             data.append(float(i))
#         datas.append(data)
#         if(line.split(',')[-1]=='layer'):
#             labels.append(1.0)
#         elif(line.split(',')[-1]=='ring'):
#             labels.append(2.0)
#         elif(line.split(',')[-1]=='core'):
#             labels.append(3.0)
#         elif(line.split(',')[-1]=='empty'):
#             labels.append(0.0)
#         elif (line.split(',')[-1] == 'full'):
#             labels.append(4.0)
#     return datas, labels
#
# if __name__ == '__main__':
#     dir = r'train/train.txt'
#     # readTxt(dir)
#     data, label = readTxt(dir)
#     print(data)
#     print("--------------------------------")
#     print(label)

features = [[1, 2, 3], [2, 4, 6]]

features = np.array(features)

# x_ = x[:, 0]/x[:, 2]
# x_ = np.round(x_, 3)
# print(x_)
# print(type(x_))


f_max = np.max(features, axis=1)
f_min = np.min(features, axis=1)
datas = np.zeros((features.shape[0], features.shape[1]))
for index,f in enumerate(features):
    datas[index] = (f-f_min[index])/(f_max[index]-f_min[index])

print(datas)
