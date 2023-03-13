# import pywt
# import matplotlib.pyplot as plt
# import numpy as np
#
# fs = 1000
# N = 200
# k = np.arange(200)
# frq = k * fs / N
# frq1 = frq[range(int(N / 2))]
#
# aa = []
# for i in range(200):
#     aa.append(np.sin(0.3 * np.pi * i))
# for i in range(200):
#     aa.append(np.sin(0.13 * np.pi * i))
# for i in range(200):
#     aa.append(np.sin(0.05 * np.pi * i))
# y = aa
#
# wavename = 'db5'
# x = range(len(y))
# plt.figure(figsize=(12, 9))
# plt.subplot(311)
# plt.plot(x, y)
# plt.title('original signal')
# for i in range(5):
#     cA, cD = pywt.dwt(y, wavename)
#     # print('原始信号系数长度：', len(y))
#     # print('近似系数长度：', len(cA))
#     # print('细节系数长度：', len(cD))
#     y = pywt.idwt(cA, None, wavename, 'smooth')  # approximated component
#     yd = pywt.idwt(None, cD, wavename, 'smooth')  # detailed component
#     # x = range(len(y))
#     # plt.figure(figsize=(12, 9))
#     # plt.subplot(311)
#     # plt.plot(x, y)
#     # plt.title('original signal')
#     plt.subplot(312)
#     plt.plot(x, y)
#     plt.title('approximated component')
#     plt.subplot(313)
#     plt.plot(x, yd)
#     plt.title('detailed component')
#     plt.tight_layout()
#     plt.show()
# # cA, cD = pywt.dwt(y, wavename)
# # print('原始信号系数长度：',len(y))
# # print('近似系数长度：',len(cA))
# # print('细节系数长度：',len(cD))
# # ya = pywt.idwt(cA, None, wavename, 'smooth')  # approximated component
# # yd = pywt.idwt(None, cD, wavename, 'smooth')  # detailed component
# # x = range(len(y))
# # plt.figure(figsize=(12, 9))
# # plt.subplot(311)
# # plt.plot(x, y)
# # plt.title('original signal')
# # plt.subplot(312)
# # plt.plot(x, ya)
# # plt.title('approximated component')
# # plt.subplot(313)
# # plt.plot(x, yd)
# # plt.title('detailed component')
# # plt.tight_layout()
# # plt.show()
#
# # # 图像单边谱
# # plt.figure(figsize=(12, 9))
# # plt.subplot(311)
# # data_f = abs(np.fft.fft(cA)) / N
# # data_f1 = data_f[range(int(N / 2))]
# # plt.plot(frq1, data_f1, 'red')
# #
# # plt.subplot(312)
# # data_ff = abs(np.fft.fft(cD)) / N
# # data_f2 = data_ff[range(int(N / 2))]
# # plt.plot(frq1, data_f2, 'k')
# #
# # plt.xlabel('pinlv(hz)')
# # plt.ylabel('amplitude')
#
# # plt.show()


import pywt
import scipy.io as scio
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt


def waveletdec(s, wname='db6', level=4, mode='symmetric'):
    N = len(s)
    w = pywt.Wavelet(wname)
    a = s
    ca = []
    cd = []
    for i in range(level):
        (a, d) = pywt.dwt(a, w, mode)  # 将a作为输入进行dwt分解
        ca.append(a)
        cd.append(d)
    rec_a = []
    rec_d = []
    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w)[0:N])  # 进行重构
    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w)[0:N])  # 进行重构
    return ca, cd, rec_a, rec_d


def zuotu(data, name_, level=4):
    plt.rcParams['font.sans-serif'] = ['simhei']  # 添加中文字体为黑体
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure()
    if level == 4:
        plt.subplot(411)
        plt.plot(data[0])
        plt.title('第1层')
        plt.subplot(412)
        plt.plot(data[1])
        plt.title('第2层')
        plt.subplot(413)
        plt.plot(data[2])
        plt.title('第3层')
        plt.subplot(414)
        plt.plot(data[3])
        plt.title('第4层')
        plt.savefig('./results/' + name_ + '.jpg')
        plt.close()


dataFile = 'doubl.mat'
data = scio.loadmat(dataFile)

data_ = np.array(data['doubl'])
print(data_)
#
for i in range(0, data_.shape[0]):
    l1 = data_[i, :]
    ca, cd, rec_a, rec_d = waveletdec(l1, wname='db6', level=4, mode='symmetric')

    zuotu(cd, name_='第' + str(i + 1) + '次' + 'Detail Coefficients')
    zuotu(ca, name_='第' + str(i + 1) + '次' + 'Approximation Coefficients')
    zuotu(rec_d, name_='第' + str(i + 1) + '次' + 'Reconstructed Detail Coefficients')
    zuotu(rec_a, name_='第' + str(i + 1) + '次' + 'Reconstructed Approximation Coefficients')