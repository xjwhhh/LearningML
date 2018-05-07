import tushare as ts
import pandas as pd

aa = ts.get_hist_data('hs300', start='2017-04-01',ktype='W')
aa=aa[['close']]
array1=pd.Series(aa.index)

array2=pd.Series(aa['close'])
array2=array2-3517.46
array2=array2/3517.46

# print(array2)

bb=ts.get_hist_data('300070', start='2017-04-01',ktype='W')
bb=bb[['close']]


array3=pd.Series(bb['close'])
array3=array3-20.02
array3=array3/20.02

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