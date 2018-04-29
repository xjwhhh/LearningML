import pandas as pd
import csv

dframe = pd.DataFrame.from_csv("C:\\Users\\xjwhh\\Desktop\\aa.csv")
# print(dframe)

array1=pd.Series(dframe.index)
array2=pd.Series(dframe['0'])
array3=pd.Series(dframe['sz'])
array4=pd.Series(dframe['sh'])

# print(array1)
# print(array2)
# print(array3)
# print(array4)

a=""
b=""
c=""

for i in range(1,len(array1)):
    a=a+"[\""+((str)(array1[len(array1)-i]))[0:10]+"\","+((str)(array2[len(array1)-i]))[0:6]+"],"
    b=b+"[\""+((str)(array1[len(array1)-i]))[0:10]+"\","+((str)(array3[len(array1)-i]))+"],"
    c=c+"[\""+((str)(array1[len(array1)-i]))[0:10]+"\","+((str)(array4[len(array1)-i]))+"],"
print(a)
print(b)
print(c)

