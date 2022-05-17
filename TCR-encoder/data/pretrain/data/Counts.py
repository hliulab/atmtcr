import pandas as pd
datas = pd.read_csv('data.csv',usecols=['CDR3'])
# print(type(datas))
# print(datas.shape)
list = [0 for n in range(50)]
print(list)
for index, row in datas.iterrows():
    cdr3 = row['CDR3']
    # print(cdr3)
    length = len(cdr3)
    # print(length)
    list[length] += 1

print(list)