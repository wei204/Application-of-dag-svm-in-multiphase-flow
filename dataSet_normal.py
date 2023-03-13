##将保存的数据以15行为一组，存入data中
def create_data(file_name):
    data = []
    row_num = 0
    file = open(file_name,'r')
    file_data = file.readlines()
    for row in file_data:
        row_num += 1
        row = row.split('\n')[0]
        if(row==''):
            continue
        data.append(row)
    return data

def write_txt(data,file_name):
    file = open(file_name,'a+')
    row_num = 0
    for str in data:
        row_num += 1
        file.write(str)
        file.write(',')
        if(row_num%15==0):
            file.write('core')
            file.write('\n')

    # file.write(data)
    file.close()


#读取csv文件内容 将其写入txt文件
def read_csv(file):
    datas = []
    file = open(file,'r')
    file_data = file.readlines()
    for index,data in enumerate(file_data):
        if index==1:
            data = data.split('\n')[0]
            datas.append(data)
    return datas

def csv_txt(datas, wfile):
    file = open(wfile,'a+')
    for data in datas:
        file.write(data)
    file.write(',')
    file.write('suspended')
    file.write('\n')
    file.close()

if __name__=="__main__":
    #将实验数据转换为标准数据集形式
    # data = create_data(r'H:\COMSOL\project\ECTwzm\zx30.txt')
    # # print(data)
    # #层流20，环流18，中心流24
    # write_txt(data,r'H:\COMSOL\project\ECTwzm\test.txt')

    #将仿真数据转换为标准数据集形式
    # file = r'H:\COMSOL\project\dianrong\4\Ct2.csv'
    wfile = r'H:\COMSOL\project\ECTwzm\simulation.txt'
    # datas = read_csv(file)
    # csv_txt(datas,wfile)
    # print(datas)
    for i in range(0,16):
        file = r'H:\COMSOL\project\dianrong\1\Ct{0}.csv'.format(i+1)
        datas = read_csv(file)
        csv_txt(datas, wfile)
