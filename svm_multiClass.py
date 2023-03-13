import sys
from numpy import *
from svm import *
from os import listdir
from SMO import PlattSMO
import pickle
import random
import math
import numpy as np
class LibSVM:
    """
    data：训练集数据列表   n*f n为样本个数，f为每个样本的特征个数
    label: 训练集类别列表
    C：惩罚因子
    toler：松弛变量
    maxIter：最大迭代次数
    kernalType：核函数类型 包括linear和rbf
    theta：当选择径向基核函数（rbf）时，表示尺度因子
    """
    def __init__(self,data=[],label=[],C=0,toler=0,maxIter=0,**kernelargs):
        self.classlabel = unique(label)
        self.classNum = len(self.classlabel)
        self.classfyNum = (self.classNum * (self.classNum-1))/2  #分类器的个数
        self.classfy = []   #存放n*(n-1)个svm分类器
        self.classfy_dict = dict()  # 以字典的形式存放分类器
        self.dataSet={}
        self.kernelargs = kernelargs
        self.C = C
        self.toler = toler
        self.maxIter = maxIter
        m = shape(data)[0]
        # 将相同标签数据存入dataSet的同一个索引位置
        for i in range(m):
            if label[i] not in self.dataSet.keys():
                self.dataSet[label[i]] = []
                self.dataSet[label[i]].append(data[i][:])
            else:
                self.dataSet[label[i]].append(data[i][:])

    def train(self):
        num = self.classNum
        for i in range(num):
            for j in range(i+1,num):
                data = []
                label = [1.0]*shape(self.dataSet[self.classlabel[i]])[0]
                label.extend([-1.0]*shape(self.dataSet[self.classlabel[j]])[0])
                data.extend(self.dataSet[self.classlabel[i]])
                data.extend(self.dataSet[self.classlabel[j]])
                svm = PlattSMO(array(data),array(label),self.C,self.toler,self.maxIter,**self.kernelargs)
                svm.smoP()
                self.classfy.append(svm)
                # 以字典的形式存储分类器
                self.classfy_dict['{0}-{1}'.format(i+1, j+1)] = svm
        self.dataSet = None

    def predict(self,data,label):
        m = shape(data)[0]  #数据样本个数
        num = self.classNum  #类别个数
        classlabel = []
        count = 0.0
        for n in range(m):
            result = [0] * num
            index = -1
            for i in range(num):
                for j in range(i + 1, num):
                    index += 1
                    s = self.classfy[index]  #使用第index个分类器判断第n个样本的类别
                    t = s.predict([data[n]])[0]
                    if t > 0.0:    #结果大于0，表示为正类，对应的类别计数+1
                        result[i] +=1
                    else:
                        result[j] +=1
            classlabel.append(result.index(max(result)))
            if classlabel[-1] != label[n]: #判断错误的个数
                count +=1
                # 输出真实结果与预测结果
                print(label[n],classlabel[n])
        #print classlabel
        print("error rate:",count / m)
        return classlabel

    '''
    类间不可分离度  
    index_A,index_B：某两个类别在类别列表中的索引位置
    '''
    def class_indivsibility(self, index_A, index_B):
        # i类和j类的中心点
        Ea = np.sum(np.array(self.dataSet[self.classlabel[index_A]]), axis=0) / \
             len(self.dataSet[self.classlabel[index_A]])
        Eb = np.sum(np.array(self.dataSet[self.classlabel[index_B]]), axis=0) / \
             len(self.dataSet[self.classlabel[index_B]])
        # # i类和j类的类内分散度
        Ca = 0
        Cb = 0
        for x in self.dataSet[self.classlabel[index_A]]:
            Ca += np.sum(np.square(np.array(x)-Ea))/(len(self.dataSet[self.classlabel[index_A]])-1)
        for x in self.dataSet[self.classlabel[index_B]]:
            Cb += np.sum(np.square(np.array(x) - Eb)) / (len(self.dataSet[self.classlabel[index_B]]) - 1)
        # Ca = np.sqrt(np.sum(np.sum(np.square(np.array(self.dataSet[self.classlabel[index_A]])-Ea), axis=1)) /
        #              len(self.dataSet[self.classlabel[index_A]])-1)
        # Cb = np.sqrt(np.sum(np.sum(np.square(np.array(self.dataSet[self.classlabel[index_B]])-Eb), axis=1)) /
        #              len(self.dataSet[self.classlabel[index_B]])-1)
        Dab = np.sqrt(np.sum(np.square(Ea-Eb)))
        Sab = (Ca+Cb)/Dab-1
        return Sab

    '''
    得到各类之间的不可分离度列表 [[value, i, j]...] value表示某两类之间的不可分离度，i和j表示两个类别的标签号
    classList：数据集类别标签列表
    '''
    def cal_indivisibility(self, classList):
        Slist = []
        for i in range(len(classList)):
            for j in range(i+1, len(classList)):
                S = []
                S.append(self.class_indivsibility(i, j))
                S.append(classList[i])
                S.append(classList[j])
                Slist.append(S)
        return Slist

    '''
    寻找与类别a不可分离度最大的类别i（该类别i不能与顺序列表中元素重复）
    S_sort：升序排列的类间不可分离度列表
    a：某一个类别对应的标签
    classList：数据集类别标签列表
    '''
    def max_indivsibility(self, S_sort, a, classList):
        max_value = S_sort[0][0]
        i = -1
        for s in S_sort:
            if s[0]>max_value and (s[1]==a or s[2]==a) and (s[1] not in classList or s[2] not in classList):
                max_value = s[0]
                i = s[1] if s[1] not in classList else s[2]    # 保证类别i不是顺序列表中的元素
        return i

    '''
    得到根据类间不可分离度划分的类别列表（最优的类别识别顺序）
    Slist：不可分离度列表 
    Num：待分类的类别个数
    '''
    def creat_classList(self, Slist, Num):
        # 对不可分离度列表按照不可分离度值升序排列
        S_sort = sorted(Slist)
        # 创建新列表存放最终的类别顺序
        classList = []
        # 将不可分离度最小的两类对应的类别号放入列表的首尾
        classList.append(S_sort[0][1])
        classList.append(S_sort[0][2])
        for num in range(math.ceil((Num-2)/2)):
            # 寻找与列表num处类别不可分离度最大的类别i（将num变成首元素）
            a = classList[0]
            i = self.max_indivsibility(S_sort,a,classList)
            # 将i插入在列表num+1处
            classList.insert(num+1, i)
            # 寻找与列表len(classList)-num-1处类别不可分离度最大的类别j（将len(classList)-num-1变成尾元素）
            b = classList[-1]
            j = self.max_indivsibility(S_sort,b,classList)
            # 将j插入在列表len(classList)-num-1处
            classList.insert(len(classList)-num-1, j)
        # 当类别个数为奇数时列表中ceil(Num/2)位置多插入一个-1，因此需要删除
        if Num % 2 == 1:
            del(classList[math.ceil(Num/2)])
        return classList


    # DDAG结构
    def DDAG_predict(self, testData, testLabel, classList):
        m = shape(testData)[0]    # 数据样本个数
        classLabel = []  # 存放预测结果
        count = 0.0
        for i in range(m):
            class_num = len(classList)  # 类别列表剩余元素的个数
            while class_num > 1:
                # 训练集中键值对小类别号在前大类别号在后，下列语句保证测试时排除前后顺序的影响
                if f'{classList[0]}-{classList[class_num-1]}' not in self.classfy_dict.keys():
                    s = self.classfy_dict['{0}-{1}'.format(classList[class_num-1], classList[0])]
                    flag = True  # 标记分类器键值是否发生转换
                else:
                    # 对类别列表首尾元素对应的类别进行分类
                    s = self.classfy_dict['{0}-{1}'.format(classList[0], classList[class_num - 1])]
                    flag = False
                t = s.predict([testData[i]])[0]  # 得到某个样本的预测结果 （1或者-1）
                class_num -= 1  # 列表个数-1
                if flag is False:
                    if t > 0:  # 该类别一定不是类别列表中尾元素对应的类别
                        classList.pop()
                    else:  # 该类别一定不是类别列表中首元素对应的类别
                        classList.pop(0)
                else:  # 列表中类别顺序反生变化，导致预测结果相反
                    if t > 0:  # 该类别一定不是类别列表中首元素对应的类别
                        classList.pop(0)
                    else:  # 该类别一定不是类别列表中尾元素对应的类别
                        classList.pop()
            # 当类别列表中只剩1个元素时 该元素即为对应的类别标签
            classLabel.append(classList[0])  # 存放预测类别
            if classLabel[i] != testLabel[i]:
                count += 1
                print('真实类别：', testLabel[i], '预测类别：', classLabel[i])
        # 计算预测错误率
        print("error rate:", count / m)
        return classLabel

    def save(self,filename):
        fw = open(filename,'wb')
        pickle.dump(self,fw,2)
        fw.close()

    @staticmethod
    def load(filename):
        fr = open(filename,'rb')
        svm = pickle.load(fr)
        fr.close()
        return svm

def loadImage(dir,maps = None):
    dirList = listdir(dir)
    data = []
    label = []
    for file in dirList:
        label.append(file.split('_')[0])
        lines = open(dir +'/'+file).readlines()
        row = len(lines)
        col = len(lines[0].strip())
        line = []
        for i in range(row):
            for j in range(col):
                line.append(float(lines[i][j]))
        data.append(line)
        if maps != None:
            label[-1] = float(maps[label[-1]])
        else:
            label[-1] = float(label[-1])
    return data,label