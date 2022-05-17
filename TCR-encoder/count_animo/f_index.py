import csv
data = csv.reader(open('testing_data.csv', 'r'))
index = {}
for st in data:
    s = st[0]#csv文件一行放在一个列表里，读取第一列数据
    i = -1
    while s.find('E', i + 1) != -1:
        i = s.find('E', i + 1)
        index[i + 1] = index.get(i + 1, 0) + 1
index_order = sorted(index.items(), key=lambda x: x[0])
print(index_order)
