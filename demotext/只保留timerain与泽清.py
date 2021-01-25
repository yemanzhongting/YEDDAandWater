# -*- coding: UTF-8 -*-
__author__ = 'zy'
__time__ = '2020/11/23 12:50'
#B-Area
with open('BIOresult.txt','r',encoding='utf-8') as f:

    tmp=f.readlines()

print(tmp[0:100])
# with open('BIOresultArea.txt', 'w', encoding='utf-8') as f:
#     for i in tmp:
#         if i=="\n":
#             f.write(i)
#         else:
#             if i.split(' ')[1]=='B-Area\n':
#                 f.write(i.split(' ')[0]+' O\n')
#             elif i.split(' ')[1] == 'I-Area\n':
#                 f.write(i.split(' ')[0] + ' O\n')
#             else:
#                 f.write(i)

with open('BIOresultAreaTime.txt', 'w', encoding='utf-8') as f:
    for i in tmp:
        if i=="\n":
            f.write(i)
        else:
            if i.split(' ')[1]=='B-Area\n':
                f.write(i)
            elif i.split(' ')[1] == 'I-Area\n':
                f.write(i)
            elif i.split(' ')[1] == 'B-Time\n':
                f.write(i)
            elif i.split(' ')[1] == 'I-Time\n':
                f.write(i)
            else:
                f.write(i.split(' ')[0] + ' O\n')


