import tushare as ts
import pandas as pd


aa = ts.get_hist_data('hs300', start='2017-04-01',ktype='W')
aa=aa[['p_change']]
aa.rename(columns={'p_change': 'hs300'}, inplace=True)
bb=ts.get_hist_data('300070', start='2017-04-01',ktype='W')
bb=bb[['p_change']]
bb.rename(columns={'p_change': 'bsy'}, inplace=True)
aa=aa.join(bb)
# print(aa)

array1=pd.Series(aa.index)

array2=pd.Series(aa['hs300'])
array3=pd.Series(bb['bsy'])

a=""
b=""

for i in range(1,len(array1)):
    a=a+"[\""+((str)(array1[len(array1)-i]))[0:10]+"\","+((str)(array2[len(array1)-i]))+"],"
    b=b+"[\""+((str)(array1[len(array1)-i]))[0:10]+"\","+((str)(array3[len(array1)-i]))+"],"


print(a)
print(b)