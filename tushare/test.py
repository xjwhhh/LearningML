import tushare as ts
import pandas as pd
import csv

# 获取深证成指，上证指数
a = ts.get_hist_data('sz', start='2017-04-01')  # 一次性获取全部日k线数据
# print(a)
a = a[['p_change']]
a.rename(columns={'p_change': 'sz'}, inplace=True)
print(a)
# a[['p_change']].to_csv("C:\\Users\\xjwhh\\Desktop\\深圳成指.csv")

b = ts.get_hist_data('sh', start='2017-04-01')  # 一次性获取全部日k线数据
# print(a)
b = b[['p_change']]
b.rename(columns={'p_change': 'sh'}, inplace=True)
print(b)
# a[['p_change']].to_csv("C:\\Users\\xjwhh\\Desktop\\深圳成指.csv")


# 获取环保行业代码
# industry=ts.get_industry_classified()
# hb=industry.loc[industry['c_name'] == '环保行业']
# print(hb[['code']])

# 获取环保行业股票的日K
f1 = open("C:\\Users\\xjwhh\\Desktop\\环保行业代码.txt")
list_of_all_the_lines = f1.readlines()

result = ts.get_hist_data('600008', start='2017-04-01')[['p_change']]
result.rename(columns={'p_change': '600008'}, inplace=True)

for i in range(1, len(list_of_all_the_lines)):
    j = list_of_all_the_lines[i]

    j = j[0:6]  # 换行符
    print(j)
    tt = ts.get_hist_data(j, start='2017-04-01')

    # print(tt)
    if (str(tt) != 'None'):
        tt = tt[['p_change']]
        tt.rename(columns={'p_change': j}, inplace=True)
        result = result.join(tt)
# print(result)
# result.to_csv("C:\\Users\\xjwhh\\Desktop\\环保行业.csv")
print(result.mean(1))

# result.mean(1).to_csv("C:\\Users\\xjwhh\\Desktop\\环保行业均值.csv")

# # 合并
# # dframe1 = pd.DataFrame.from_csv('C:\\Users\\xjwhh\\Desktop\\环保行业均值.csv')
# dframe2 = pd.DataFrame.from_csv('C:\\Users\\xjwhh\\Desktop\\上证指数.csv')
# dframe3 = pd.DataFrame.from_csv('C:\\Users\\xjwhh\\Desktop\\深证成指.csv')
#
# # print(dframe1)
# print(dframe2)
# print(dframe3)
aaaa=pd.DataFrame(result.mean(1))
print(aaaa)
aaaa = aaaa.join(a)
aaaa = aaaa.join(b)

print(aaaa)
aaaa.to_csv("C:\\Users\\xjwhh\\Desktop\\涨幅.csv")

# f = open('C:\\Users\\xjwhh\\Desktop\\涨幅.txt', 'w')
# f.write(aaaa)
# f.close()
