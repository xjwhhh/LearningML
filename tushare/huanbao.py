import tushare as ts
import pandas as pd


# 获取深证成指，上证指数
a = ts.get_hist_data('sz', start='2017-04-01',ktype='W')
a = a[['close']]
a.rename(columns={'p_change': 'sz'}, inplace=True)
# print(a)


# 获取环保行业代码

# 获取环保行业股票的日K
f1 = open("C:\\Users\\xjwhh\\Desktop\\环保行业代码.txt")
list_of_all_the_lines = f1.readlines()

result = ts.get_hist_data('600008', start='2017-04-01',ktype='W')[['close']]
result.rename(columns={'close': '600008'}, inplace=True)

for i in range(1, len(list_of_all_the_lines)):
    j = list_of_all_the_lines[i]

    j = j[0:6]  # 换行符
    # print(j)
    tt = ts.get_hist_data(j, start='2017-04-01',ktype='W')

    # print(tt)
    if (str(tt) != 'None'):
        tt = tt[['close']]
        tt.rename(columns={'close': j}, inplace=True)
        result = result.join(tt)

# print(result.mean(1))


aaaa=pd.DataFrame(result.mean(1))


array2=pd.Series(result.mean(1))
print(array2)


# print(aaaa)
aaaa = aaaa.join(a)

aaaa=aaaa.dropna(axis=0,how='any')


print(aaaa)

array1=pd.Series(aaaa.index)


array2=array2-20.071852
array2=array2/20.071852

# print(array2)




array3=pd.Series(aaaa['close'])
array3=array3-10669.48
array3=array3/10669.48

# print(array3)

array2=array2*100
array3=array3*100

a=""
b=""

for i in range(1,len(array1)):
    a=a+"[\""+((str)(array1[len(array1)-i]))[0:10]+"\","+((str)(array2[len(array1)-i]))+"],"
    b=b+"[\""+((str)(array1[len(array1)-i]))[0:10]+"\","+((str)(array3[len(array1)-i]))+"],"


print(a)
print(b)
